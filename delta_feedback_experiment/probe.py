"""Authoritative offline plus Jobe CUDA execution gate."""

from __future__ import annotations

import contextlib
import copy
import gc
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def cuda_gate() -> None:
    """Capture and replay every default delta mode at full screen geometry."""
    from torch._dynamo.utils import counters
    from transformer_experiments import checkpoints

    from .cuda_kernels import bespoke_route
    from .cuda_kernels import triton as route_triton
    from .fla_ops import pkda_norm_gate
    from .model import (
        DeltaModel,
        KVCache,
        _ClassifierShadow,
        _fixed_cce_z,
        _route_sources,
        batch_vocab_order,
        condition_config,
        iterate_fused,
        linear_cross_entropy_apply,
        multipass,
        multipass_loss,
    )
    from .optim import OptimizerPair, build_optimizers
    from .pkda import (
        PreconditionedKDA,
        _l2norm,
        _PackedControlSplit,
        pkda_cuda_available,
    )
    from .train import (
        CONTRACT,
        CudaEvalRunner,
        CudaGraphTrainer,
        automatic_checkpoint,
        build_schedule,
        clip_gradients,
        evaluate,
        execution_fields,
        parse_run_args,
        route_summary,
    )

    if linear_cross_entropy_apply is None:
        raise RuntimeError("Jobe gate requires cut-cross-entropy")
    if route_triton is None:
        raise RuntimeError("Jobe gate requires the bespoke Triton router")
    if not pkda_cuda_available():
        raise RuntimeError("Jobe gate requires the pinned FLA PKDA kernels")

    # The fixed-capacity router must match the semantic implementation in both
    # values and gradients. H=4/N=5 covers the largest screen bank; H=8/N=8
    # covers the corresponding six-cell flagship payload bank. ``n_sources``
    # counts the width-``dim`` null that the bespoke kernel reads in place.
    def route_parity(
        dim,
        heads,
        batch,
        length,
        n_sources,
        seed,
        route_max_bound=0.05,
        weight_max_bound=0.015,
    ):
        torch.manual_seed(seed)
        query = torch.randn(dim, device="cuda", dtype=torch.float32).requires_grad_()
        key = torch.randn(dim, device="cuda", dtype=torch.float32).requires_grad_()
        null = torch.randn(dim, device="cuda", dtype=torch.bfloat16).requires_grad_()
        sources = tuple(
            torch.randn(
                batch, length, dim, device="cuda", dtype=torch.bfloat16
            ).requires_grad_()
            for _ in range(n_sources - 1)
        )
        projected = (query * key).to(torch.bfloat16)
        routed, weights = bespoke_route(projected, null, 1e-6, heads, sources)
        routed.float().square().mean().backward()
        grads = (
            query.grad.clone(),
            key.grad.clone(),
            null.grad.clone(),
            *(source.grad.clone() for source in sources),
        )

        ref_query = query.detach().clone().requires_grad_()
        ref_key = key.detach().clone().requires_grad_()
        ref_null = null.detach().clone().requires_grad_()
        ref_sources = tuple(
            source.detach().clone().requires_grad_() for source in sources
        )
        ref_routed, ref_weights = _route_sources(
            ref_query,
            ref_key,
            1e-6,
            heads,
            ref_null.expand(batch, length, dim),
            *ref_sources,
        )
        ref_routed.float().square().mean().backward()
        ref_grads = (
            ref_query.grad,
            ref_key.grad,
            ref_null.grad,
            *(source.grad for source in ref_sources),
        )

        # The bespoke value mix accumulates in FP32 rather than reproducing
        # source-by-source BF16 rounding. Bound that intentional difference.
        route_rel = torch.linalg.vector_norm(
            (routed - ref_routed).float()
        ) / torch.linalg.vector_norm(ref_routed.float())
        weight_rel = torch.linalg.vector_norm(
            weights - ref_weights
        ) / torch.linalg.vector_norm(ref_weights)
        route_max = (routed - ref_routed).abs().max()
        weight_max = (weights - ref_weights).abs().max()
        if route_rel >= 0.01 or route_max >= route_max_bound:
            raise AssertionError(
                f"H={heads} bespoke router value drift: "
                f"rel={route_rel.item():.4g}, max={route_max.item():.4g}"
            )
        if weight_rel >= 0.01 or weight_max >= weight_max_bound:
            raise AssertionError(
                f"H={heads} bespoke router weight drift: "
                f"rel={weight_rel.item():.4g}, max={weight_max.item():.4g}"
            )
        for actual, expected in zip(grads, ref_grads, strict=True):
            if not torch.allclose(actual, expected, rtol=5e-2, atol=5e-3):
                raise AssertionError(f"H={heads} bespoke router gradient drift")

        # Banked sources: two sites read the same bank, their gradients go
        # into the sources' accumulators instead of coming back per site, and
        # the finished accumulator must equal the sum autograd would have
        # formed. The comparison is against the unbanked run's own two-site
        # sum, so only the accumulation order can differ.
        banked = len(sources) - 1 or 1
        accumulators = tuple(
            torch.zeros_like(source) for source in sources[:banked]
        )
        for source in sources:
            source.grad = None
        query.grad = key.grad = null.grad = None
        # A fresh projection: the first backward already freed the graph that
        # built the one above.
        banked_projected = (query * key).to(sources[0].dtype)
        first, _ = bespoke_route(
            banked_projected, null, 1e-6, heads, sources, accumulators
        )
        second, _ = bespoke_route(
            banked_projected, null, 1e-6, heads, sources, accumulators
        )
        (first.float().square().mean() + second.float().square().mean()).backward()
        for index, source in enumerate(sources):
            gathered = (
                accumulators[index] if index < banked else source.grad
            )
            expected = 2 * grads[3 + index].float()
            if index < banked and source.grad is not None:
                raise AssertionError(
                    f"H={heads} banked source {index} still returned a gradient"
                )
            if not torch.allclose(
                gathered.float(), expected, rtol=5e-2, atol=5e-3
            ):
                raise AssertionError(
                    f"H={heads} banked router source {index} gradient drift"
                )

    route_parity(48, 4, 3, 11, 5, 7)
    route_parity(
        1536,
        8,
        2,
        3,
        8,
        8,
        route_max_bound=0.10,
        weight_max_bound=0.025,
    )
    # Ragged token counts must not disturb the multi-token backward programs.
    route_parity(48, 4, 5, 7, 3, 11)

    # Compare the exact chunk operator against the literal recurrent equations
    # across a chunk boundary. Inputs use the production dtypes, and the loss
    # reaches every recurrence input plus the learned main-decay and
    # preconditioner-center parameters.
    torch.manual_seed(9)
    pkda_ref = PreconditionedKDA(32, num_heads=2, head_dim=16).train()
    pkda_cuda = copy.deepcopy(pkda_ref).cuda().train()
    shapes = {
        "q": (1, 65, 2, 16),
        "k": (1, 65, 2, 16),
        "v": (1, 65, 2, 16),
        "raw_decay": (1, 65, 2, 16),
        "beta": (1, 65, 2),
        "precond_decay": (1, 65, 2),
        "precond_beta": (1, 65, 2),
    }
    pkda_inputs = {}
    pkda_cuda_inputs = {}
    for name, shape in shapes.items():
        dtype = torch.float32 if name == "precond_decay" else torch.bfloat16
        value = torch.randn(shape, dtype=dtype)
        if name == "precond_decay":
            value = -value.abs()
        elif name in {"beta", "precond_beta"}:
            value = value.sigmoid()
        pkda_inputs[name] = value.requires_grad_()
        pkda_cuda_inputs[name] = value.detach().cuda().requires_grad_()
    pkda_expected, _, _ = pkda_ref._portable_recurrence(
        **pkda_inputs,
        state=None,
        a_state=None,
        output_final_state=False,
    )
    pkda_actual, _, _ = pkda_cuda._cuda_recurrence(
        **pkda_cuda_inputs,
        state=None,
        a_state=None,
        output_final_state=False,
    )
    pkda_cotangent = torch.randn_like(pkda_expected)
    (pkda_expected.float() * pkda_cotangent.float()).mean().backward()
    (pkda_actual.float() * pkda_cotangent.cuda().float()).mean().backward()

    def relative_error(actual, expected):
        return (
            torch.linalg.vector_norm((actual - expected).float())
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-8)
        ).item()

    pkda_value_rel = relative_error(pkda_actual.cpu(), pkda_expected)
    if pkda_value_rel >= 0.05:
        raise AssertionError(f"PKDA chunk value drift: {pkda_value_rel:.4f}")
    for name in shapes:
        grad_rel = relative_error(
            pkda_cuda_inputs[name].grad.cpu(), pkda_inputs[name].grad
        )
        if grad_rel >= 0.15:
            raise AssertionError(f"PKDA chunk {name} gradient drift: {grad_rel:.4f}")
    for name in ("A_log", "dt_bias", "log_precond_center"):
        expected = dict(pkda_ref.named_parameters())[name].grad
        actual = dict(pkda_cuda.named_parameters())[name].grad.cpu()
        grad_rel = relative_error(actual, expected)
        if grad_rel >= 0.15:
            raise AssertionError(f"PKDA chunk {name} gradient drift: {grad_rel:.4f}")
    pkda_norm_eps = pkda_cuda.norm_eps
    del pkda_ref, pkda_cuda, pkda_inputs, pkda_cuda_inputs
    del pkda_expected, pkda_actual, pkda_cotangent

    # Dense no-cache PKDA projects Q/K/V separately but executes their equal-
    # width short convolutions, the SiLU, and the per-head Q/K L2 normalization
    # in one FLA Triton kernel. Compare that fused path to the literal
    # three-convolution implementation plus the explicit normalization in
    # values and gradients.
    torch.manual_seed(12)
    conv_ref = PreconditionedKDA(32, num_heads=2, head_dim=16).cuda().train()
    conv_fused = copy.deepcopy(conv_ref).train()
    conv_x_ref = torch.randn(
        2, 65, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    conv_x_fused = conv_x_ref.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref_q, _ = conv_ref._causal_conv(
            conv_ref.q_proj(conv_x_ref), conv_ref.q_conv, None, False
        )
        ref_k, _ = conv_ref._causal_conv(
            conv_ref.k_proj(conv_x_ref), conv_ref.k_conv, None, False
        )
        ref_v, _ = conv_ref._causal_conv(
            conv_ref.v_proj(conv_x_ref), conv_ref.v_conv, None, False
        )
        reference_qkv = (
            _l2norm(ref_q.reshape(2, 65, 2, 16)),
            _l2norm(ref_k.reshape(2, 65, 2, 16)),
            ref_v.reshape(2, 65, 2, 16),
        )
        fused_qkv = conv_fused._project(conv_x_fused, None, False)[:3]
    conv_cotangents = tuple(torch.randn_like(value) for value in reference_qkv)
    torch.autograd.backward(reference_qkv, conv_cotangents)
    torch.autograd.backward(fused_qkv, conv_cotangents)
    conv_rel = max(
        relative_error(actual, expected)
        for actual, expected in zip(fused_qkv, reference_qkv, strict=True)
    )
    if conv_rel >= 0.01:
        raise AssertionError(f"fused QKV convolution value drift: {conv_rel:.4f}")
    conv_grad_names = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "q_conv.weight",
        "k_conv.weight",
        "v_conv.weight",
    )
    ref_parameters = dict(conv_ref.named_parameters())
    fused_parameters = dict(conv_fused.named_parameters())
    for name in conv_grad_names:
        grad_rel = relative_error(
            fused_parameters[name].grad, ref_parameters[name].grad
        )
        if grad_rel >= 0.05:
            raise AssertionError(
                f"fused QKV convolution {name} gradient drift: {grad_rel:.4f}"
            )
    conv_input_grad_rel = relative_error(conv_x_fused.grad, conv_x_ref.grad)
    if conv_input_grad_rel >= 0.05:
        raise AssertionError(
            f"fused QKV convolution input gradient drift: {conv_input_grad_rel:.4f}"
        )
    del conv_ref, conv_fused, conv_x_ref, conv_x_fused
    del reference_qkv, fused_qkv, conv_cotangents

    # FLA's fused RMSNorm/sigmoid gate must preserve the portable PKDA
    # epilogue in both values and gradients at production dtypes.
    torch.manual_seed(10)
    norm_output = torch.randn(
        2, 65, 2, 16, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    norm_gate = torch.randn_like(norm_output).requires_grad_()
    norm_weight = torch.randn(16, device="cuda", dtype=torch.float32).requires_grad_()
    norm_actual, _ = pkda_norm_gate(
        norm_output, norm_gate, norm_weight, pkda_norm_eps
    )
    norm_ref_output = norm_output.detach().clone().requires_grad_()
    norm_ref_gate = norm_gate.detach().clone().requires_grad_()
    norm_ref_weight = norm_weight.detach().clone().requires_grad_()
    norm_reference = norm_ref_output.float() * torch.rsqrt(
        norm_ref_output.float().square().mean(-1, keepdim=True) + pkda_norm_eps
    )
    norm_reference = norm_reference * norm_ref_weight
    norm_reference = norm_reference.to(norm_ref_output.dtype) * torch.sigmoid(
        norm_ref_gate
    )
    norm_cotangent = torch.randn_like(norm_actual)
    (norm_actual.float() * norm_cotangent.float()).mean().backward()
    (norm_reference.float() * norm_cotangent.float()).mean().backward()
    norm_gate_rel = relative_error(norm_actual, norm_reference)
    if norm_gate_rel >= 0.01:
        raise AssertionError(f"PKDA fused norm/gate value drift: {norm_gate_rel:.4f}")
    for name, actual, expected in (
        ("output", norm_output.grad, norm_ref_output.grad),
        ("gate", norm_gate.grad, norm_ref_gate.grad),
        ("weight", norm_weight.grad, norm_ref_weight.grad),
    ):
        grad_rel = relative_error(actual, expected)
        if grad_rel >= 0.05:
            raise AssertionError(
                f"PKDA fused norm/gate {name} gradient drift: {grad_rel:.4f}"
            )
    del norm_output, norm_gate, norm_weight, norm_actual
    del norm_ref_output, norm_ref_gate, norm_ref_weight, norm_reference, norm_cotangent

    # CCE-native z-loss must retain the full-logit scalar and gradient semantics
    # while never constructing the classifier-wide activation in the real path.
    torch.manual_seed(11)
    cce_e = torch.randn(41, 32, device="cuda", dtype=torch.bfloat16).requires_grad_()
    cce_c = torch.randn(127, 32, device="cuda", dtype=torch.float32).requires_grad_()
    cce_shadow = cce_c.detach().to(torch.bfloat16)
    cce_t = torch.randint(0, 127, (41,), device="cuda")
    cce_ce, cce_z = _fixed_cce_z(
        cce_e, _ClassifierShadow.apply(cce_c, cce_shadow, None), cce_t
    )
    (cce_ce + 1e-2 * cce_z).backward()
    cce_de, cce_dc = cce_e.grad.float().clone(), cce_c.grad.clone()
    ref_e = cce_e.detach().clone().requires_grad_()
    ref_c = cce_c.detach().clone().requires_grad_()
    ref_logits = (ref_e @ ref_c.to(ref_e.dtype).mT).float()
    ref_ce = torch.nn.functional.cross_entropy(ref_logits, cce_t)
    ref_z = ref_logits.logsumexp(-1).square().mean()
    (ref_ce + 1e-2 * ref_z).backward()
    if not torch.allclose(cce_ce, ref_ce, rtol=2e-3, atol=2e-3):
        raise AssertionError(f"CCE CE drift: {cce_ce.item()} versus {ref_ce.item()}")
    if not torch.allclose(cce_z, ref_z, rtol=2e-3, atol=2e-3):
        raise AssertionError(f"CCE z drift: {cce_z.item()} versus {ref_z.item()}")
    if not torch.allclose(cce_de, ref_e.grad.float(), rtol=3e-2, atol=3e-3):
        raise AssertionError("CCE-native z embedding gradient drift")
    if not torch.allclose(cce_dc, ref_c.grad, rtol=3e-2, atol=3e-3):
        raise AssertionError("CCE-native z classifier gradient drift")
    del cce_e, cce_c, cce_shadow, cce_t, cce_de, cce_dc, ref_e, ref_c, ref_logits
    del cce_ce, cce_z, ref_ce, ref_z

    # Under a vocabulary ordering the backward skips a tile before recomputing
    # its logits exactly when the late gradient filter would drop it. Compare
    # the computed-tile sets of both paths on logits spread enough that the
    # filter keeps some tiles and drops others.
    # A hot vocabulary region carries all the probability mass and every
    # target, and the ordering packs it into the leading tiles, as the
    # frequency ordering does in production; the cold tiles must be filtered.
    torch.manual_seed(13)
    flag_e = (torch.randn(1024, 64, device="cuda") * 3).to(torch.bfloat16)
    flag_e.requires_grad_()
    flag_c = torch.randn(4096, 64, device="cuda", dtype=torch.float32)
    flag_c[512:] *= 0.05
    flag_c.requires_grad_()
    flag_shadow = flag_c.detach().to(torch.bfloat16)
    flag_t = torch.randint(0, 512, (1024,), device="cuda")
    flag_order = torch.cat(
        (torch.randperm(512, device="cuda"), 512 + torch.randperm(3584, device="cuda"))
    ).to(torch.int32)
    tile_sets = {}
    for skip_early in (False, True):
        tile_flags = torch.full((8, 32), -1, dtype=torch.int32, device="cuda")
        flag_e.grad = None
        flag_c.grad = None
        flag_ce, flag_z = _fixed_cce_z(
            flag_e,
            _ClassifierShadow.apply(flag_c, flag_shadow, None),
            flag_t,
            flag_order,
            skip_early=skip_early,
            tile_flags=tile_flags,
        )
        (flag_ce + 1e-2 * flag_z).backward()
        tile_sets[skip_early] = tile_flags.clone()
    if (tile_sets[True] < 0).any() or (tile_sets[False] < 0).any():
        raise AssertionError("CCE tile flags were not written for every tile")
    if not torch.equal(tile_sets[False] == 1, tile_sets[True] == 1):
        raise AssertionError("CCE early skip and late filter compute different tiles")
    if not (tile_sets[True] == 0).any() or not (tile_sets[False] == 2).any():
        raise AssertionError("CCE tile identity check exercised no filtered tile")
    if (tile_sets[True] == 2).any():
        raise AssertionError("a tile reached the late filter despite the early skip")
    del flag_e, flag_c, flag_shadow, flag_t, flag_order, tile_sets, tile_flags
    del flag_ce, flag_z

    # The head's classifier gradient reaches the FP32 sink either once per
    # microbatch or once per flush window, with the window's contributions
    # accumulated in the BF16 buffer the kernel already lock-adds into. Both
    # paths are measured against the exact FP32 gradient of the same operands:
    # a window may not move the step's gradient off the per-microbatch path by
    # more than the BF16 rounding it replaces, and above all may not change its
    # magnitude, which is what a coherent accumulation bias would do.
    torch.manual_seed(19)
    accum_vocab, accum_dim, accum_rows, accum_micros = 4096, 64, 256, 8
    accum_c = (torch.randn(accum_vocab, accum_dim, device="cuda") * 0.5).to(
        torch.bfloat16
    )
    accum_batches = [
        (
            (torch.randn(accum_rows, accum_dim, device="cuda") * 2.0).to(
                torch.bfloat16
            ),
            torch.randint(0, accum_vocab, (accum_rows,), device="cuda"),
        )
        for _ in range(accum_micros)
    ]
    accum_reference = torch.zeros(accum_vocab, accum_dim, device="cuda")
    with torch.no_grad():
        for accum_e, accum_t in accum_batches:
            probabilities = torch.softmax(accum_e.float() @ accum_c.float().mT, dim=-1)
            probabilities[torch.arange(accum_rows, device="cuda"), accum_t] -= 1.0
            accum_reference += probabilities.mT @ accum_e.float() / accum_rows
    del probabilities

    def head_gradient(
        window: int | None,
        accum_c: torch.Tensor,
        accum_batches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """The classifier gradient of every microbatch, summed in the sink."""
        sink = torch.zeros(accum_vocab, accum_dim, device="cuda")
        buffer = None if window is None else torch.zeros_like(accum_c)
        # A fresh master per call: its accumulator node would otherwise outlive
        # the call and be reused by the next one's backward.
        master = accum_c.float().requires_grad_()
        for index, (accum_e, accum_t) in enumerate(accum_batches, start=1):
            embeddings = accum_e.clone().requires_grad_()
            if buffer is None:
                classifier = _ClassifierShadow.apply(master, accum_c, sink)
            else:
                classifier = accum_c
            loss, _ = _fixed_cce_z(
                embeddings,
                classifier,
                accum_t,
                batch_vocab_order(embeddings, accum_c),
                c_grad_accum=buffer,
            )
            loss.backward()
            if buffer is not None and index % window == 0:
                sink.add_(buffer)
                buffer.zero_()
        if buffer is not None:
            sink.add_(buffer)
        if master.grad is not None:
            raise AssertionError("the sunk classifier gradient reached autograd")
        return sink

    def head_error(
        gradient: torch.Tensor, accum_reference: torch.Tensor
    ) -> tuple[float, float]:
        difference = (gradient - accum_reference).norm() / accum_reference.norm()
        return difference.item(), (gradient.norm() / accum_reference.norm()).item()

    per_micro_rel, per_micro_ratio = head_error(
        head_gradient(None, accum_c, accum_batches), accum_reference
    )
    repeat_rel, repeat_ratio = head_error(
        head_gradient(None, accum_c, accum_batches), accum_reference
    )
    # The head's own run-to-run floor: the forward's LSE lock combines its
    # vocabulary tiles in whatever order they arrive.
    accum_floor = max(
        abs(repeat_rel - per_micro_rel), abs(repeat_ratio - per_micro_ratio), 1e-5
    )
    once_rel, once_ratio = head_error(
        head_gradient(1, accum_c, accum_batches), accum_reference
    )
    if (
        abs(once_rel - per_micro_rel) > 8 * accum_floor
        or abs(once_ratio - per_micro_ratio) > 8 * accum_floor
    ):
        raise AssertionError(
            "flushing the head buffer every microbatch is not the per-microbatch "
            f"path: rel {once_rel:.4e} versus {per_micro_rel:.4e}, norm ratio "
            f"{once_ratio:.6f} versus {per_micro_ratio:.6f}, floor {accum_floor:.1e}"
        )
    flush_rel, flush_ratio = head_error(
        head_gradient(4, accum_c, accum_batches), accum_reference
    )
    # A four-microbatch window rounds its sum in BF16, which on these operands
    # costs 2.4e-4 of relative error and 9.4e-5 of norm; the bands below are
    # four times that, against a run-to-run floor of 1e-7. (On the trained
    # screen head, whose rows accumulate far more mass, the same window costs
    # 1.9e-2 against 7.9e-3 per microbatch and 0.21% of norm against 0.08%.)
    # A window that stopped accumulating, or accumulated the wrong rows, misses
    # or misplaces whole microbatches and fails here by orders of magnitude.
    if abs(flush_ratio - per_micro_ratio) > 5e-4:
        raise AssertionError(
            "the head's flush window moved the classifier gradient norm: "
            f"{flush_ratio:.6f} versus {per_micro_ratio:.6f}"
        )
    if flush_rel > per_micro_rel + 1e-3:
        raise AssertionError(
            f"the head's flush window drifted: {flush_rel:.4e} versus "
            f"{per_micro_rel:.4e}"
        )
    del accum_c, accum_batches, accum_reference

    # The packed control projection exposes five semantic slices, but its
    # backward writes their gradients directly into one GEMM-ready buffer.
    control_splits = (128, 8, 8, 8, 128)
    control = torch.randn(
        3,
        65,
        sum(control_splits),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    control_ref = control.detach().clone().requires_grad_()
    cotangents = tuple(
        torch.randn(3, 65, size, device="cuda", dtype=torch.bfloat16)
        for size in control_splits
    )
    control_parts = _PackedControlSplit.apply(control, control_splits)
    reference_parts = control_ref.split(control_splits, dim=-1)
    torch.autograd.backward(control_parts, cotangents)
    torch.autograd.backward(reference_parts, cotangents)
    if not torch.equal(control.grad, control_ref.grad):
        raise AssertionError("packed-control gradient assembly drift")
    del control, control_ref, cotangents, control_parts, reference_parts

    # Exercise causal Flash attention and explicit GQA prefix-cache decoding.
    # BF16 changes with block partitioning, so the
    # invariant is close recurrence rather than bit identity.
    def decode_parity(
        condition: str,
        layers: int,
        *,
        bf16: bool = True,
        bound: float = 0.04,
        **extra,
    ) -> float:
        decode_cfg = condition_config(
            condition,
            vocab_size=1000,
            dim=128,
            layers=layers,
            heads=4,
            kv_heads=2,
            head_dim=32,
            intermediate=256,
            pkda_heads=2,
            pkda_head_dim=128,
            max_seq_len=16,
            **extra,
        )
        torch.manual_seed(0)
        decode_model = DeltaModel(decode_cfg).cuda().eval()
        decode_tokens = torch.randint(0, 1000, (2, 12), device="cuda")
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if bf16
            else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast:
            full = decode_model.forward_column(
                decode_model.embed_tokens(decode_tokens)
            ).h_top
            cache = KVCache(
                decode_cfg, 2, "cuda", torch.bfloat16 if bf16 else torch.float32
            )
            prefill = decode_model.forward_column(
                decode_model.embed_tokens(decode_tokens[:, :5]), cache=cache
            )
            pieces = [prefill.h_top]
            for column in range(5, decode_tokens.shape[1]):
                pieces.append(
                    decode_model.step(
                        decode_tokens[:, column : column + 1], None, cache
                    ).h_top
                )
            incremental = torch.cat(pieces, dim=1)
        relative = (
            torch.linalg.vector_norm((full - incremental).float())
            / torch.linalg.vector_norm(full.float())
        ).item()
        # This is an accumulated four-layer BF16 path comparison, not a
        # single-kernel tolerance. Fan-in-scaled hidden matrices make the tiny
        # model's branches representative rather than nearly inert; the
        # independently gated projection and recurrence components remain
        # below their tighter bounds above.
        if relative >= bound:
            raise AssertionError(
                f"{condition!r} cache parity drift: {relative:.4f}"
            )
        del decode_model, decode_tokens, full, cache, prefill, pieces, incremental
        return relative

    plain_decode_rel = decode_parity("", 4)
    hybrid_decode_rel = decode_parity("a", 4)
    # The loop decodes through one core cache track per iteration. BF16 drift
    # between the parallel and single-column paths grows with executed depth.
    # Check the track wiring in FP32 to separate cache semantics from that
    # accumulated rounding error.
    loop_decode_rel = decode_parity(
        "arfl",
        12,
        bf16=False,
        bound=0.01,
        loop_iterations=2,
        loop_max_iterations=2,
    )
    rope_loop_decode_rel = decode_parity(
        "rfl",
        12,
        bf16=False,
        bound=0.01,
        loop_iterations=2,
        loop_max_iterations=2,
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    args = parse_run_args(["cuda-probe", "--condition", "arf"])
    schedule = build_schedule(args)
    torch.set_float32_matmul_precision("high")

    def screen_model(condition: str) -> DeltaModel:
        torch.manual_seed(args.seed)
        return (
            DeltaModel(
                condition_config(
                    condition,
                    vocab_size=args.vocab_size,
                    dim=args.dim,
                    layers=args.layers,
                    heads=args.heads,
                    kv_heads=args.kv_heads,
                    head_dim=args.head_dim,
                    intermediate=args.intermediate,
                    max_seq_len=args.seq_len + 1,
                    loop_iterations=args.loop_iterations,
                    loop_max_iterations=args.loop_max_iterations,
                )
            )
            .cuda()
            .train()
        )

    # At one iteration the loop is its unlooped condition: the same
    # parameters, banks, and compiled blocks. One eager two-pass microbatch at
    # the screen geometry must give the same loss and, up to the CUDA path's
    # own nondeterminism, the same gradients. The head accumulates its BF16
    # gradient through locks, and that order reaches every parameter through
    # the backward: the floor is measured from repeated flat-column gradients here,
    # and the loop is held to it.
    parity_rows = torch.randint(
        0,
        args.vocab_size,
        (2, args.seq_len + 1),
        generator=torch.Generator().manual_seed(23),
    ).cuda()
    parity_prefix = torch.full((1, 2), 7, dtype=torch.long, device="cuda")
    parity: dict[str, tuple[float, torch.Tensor]] = {}
    for label, condition, iterations, banked in (
        ("arf", "arf", None, True),
        ("arf again", "arf", None, True),
        ("arf unbanked", "arf", None, False),
        ("arfl", "arfl", 1, True),
    ):
        parity_model = screen_model(condition)
        parity_model.bank_sources = banked
        parity_model.refresh_shadows()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            parity_outs = multipass(
                parity_model,
                parity_rows,
                2,
                prefix_lens=parity_prefix,
                iterations=iterations,
            )
            # Index rather than unpack: a lingering per-pass losses list would
            # keep the whole eager graph, parameters, and gradients alive.
            parity_loss = multipass_loss(parity_model, parity_rows, parity_outs).total
        parity_loss.backward()
        parity[label] = (
            parity_loss.item(),
            torch.cat(
                [
                    parameter.grad.detach().flatten().float()
                    for _, parameter in sorted(parity_model.named_parameters())
                    if parameter.grad is not None
                ]
            ),
        )
        del parity_model, parity_outs, parity_loss
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    (flat_loss, flat_grad), (loop_loss, loop_grad) = parity["arf"], parity["arfl"]
    if not math.isclose(flat_loss, loop_loss, rel_tol=1e-4, abs_tol=1e-4):
        raise AssertionError(
            f"arfl at r = 1 drifts from arf: loss {loop_loss} versus {flat_loss}"
        )
    if flat_grad.shape != loop_grad.shape:
        raise AssertionError("arfl at r = 1 has a different gradient surface than arf")
    grad_floor = relative_error(parity["arf again"][1], flat_grad)
    # Every routed source hands its readers one accumulator instead of one
    # gradient each. The sum is over the same terms in the same order, but a
    # banked term is added to the running sum unrounded while an unbanked one
    # is rounded to BF16 first, and a BF16 running sum over the seed's
    # forty-eight readers carries about a percent either way. So the two
    # accumulations differ by more than the model against itself does; the
    # bound is wide enough for that and far below anything a misrouted
    # accumulator would produce.
    unbanked_loss, unbanked_grad = parity["arf unbanked"]
    # Banking cannot reach the forward at all, so any difference here is the
    # head's own tile accumulation, which moves the loss by about 3e-4 between
    # two runs of the same model. Hold it to that spread with room.
    loss_floor = abs(parity["arf again"][0] - flat_loss)
    if abs(unbanked_loss - flat_loss) > 2 * loss_floor + 1e-3:
        raise AssertionError(
            f"the source bank moved the loss: {flat_loss} versus {unbanked_loss}, "
            f"against a same-model floor of {loss_floor}"
        )
    unbanked_rel = relative_error(flat_grad, unbanked_grad)
    if unbanked_rel > 4 * grad_floor:
        raise AssertionError(
            f"the source bank drifts from per-reader accumulation: gradient rel "
            f"{unbanked_rel:.4f} against a same-model floor of {grad_floor:.4f}"
        )
    loop_grad_rel = relative_error(loop_grad, flat_grad)
    if loop_grad_rel > 1.5 * grad_floor + 1e-3:
        raise AssertionError(
            f"arfl at r = 1 drifts from arf: gradient rel {loop_grad_rel:.4f} "
            f"against a same-model floor of {grad_floor:.4f}"
        )
    del parity, flat_grad, loop_grad, unbanked_grad, parity_rows, parity_prefix
    # Collect the stage's cyclic garbage before the next model exists, so the
    # capture peak below measures the trainer and not this stage's remains.
    gc.collect()
    torch.cuda.empty_cache()

    # The first stage of the loop's memory measurement: one eager one-pass
    # microbatch at the iteration cap, forward and backward, under the
    # trainer's own activation policy.
    torch.cuda.reset_peak_memory_stats()
    loop_model = screen_model("arfl")
    loop_model.refresh_shadows()
    loop_model.grad_checkpoint = automatic_checkpoint(
        loop_model, 1, args.loop_max_iterations, args, torch.device("cuda")
    )
    loop_rows = torch.randint(
        0,
        args.vocab_size,
        (args.micro_rows, args.seq_len + 1),
        generator=torch.Generator().manual_seed(29),
    ).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loop_outs = multipass(
            loop_model, loop_rows, 1, iterations=args.loop_max_iterations
        )
        loop_loss = multipass_loss(loop_model, loop_rows, loop_outs).total
    loop_loss.backward()
    torch.cuda.synchronize()
    if not math.isfinite(loop_loss.item()):
        raise AssertionError("nonfinite arfl loss at the iteration cap")
    loop_cap_peak = torch.cuda.max_memory_allocated() / 2**30
    del loop_model, loop_rows, loop_outs, loop_loss
    gc.collect()
    torch.cuda.empty_cache()
    stage_residual = torch.cuda.memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()

    class _ProbeValidation:
        def __init__(self):
            self.rows = torch.randint(
                0,
                args.vocab_size,
                (args.eval_rows, args.seq_len + 1),
                generator=torch.Generator().manual_seed(19),
            )

        def batch(self, first, count, device=None):
            rows = self.rows[first : first + count]
            return rows.to(device) if device is not None else rows

    probe_validation = _ProbeValidation()
    torch.manual_seed(args.seed)
    model = (
        DeltaModel(
            condition_config(
                "arf",
                vocab_size=args.vocab_size,
                dim=args.dim,
                layers=args.layers,
                heads=args.heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                intermediate=args.intermediate,
                max_seq_len=args.seq_len + 1,
            )
        )
        .cuda()
        .train()
    )
    optimizers = build_optimizers(
        model,
        lr_normuonh=args.lr_normuonh,
        lr_nadam=args.lr_nadam,
    )
    model.refresh_shadows()
    classifier_shadow = model._classifier_shadow
    if classifier_shadow is None or classifier_shadow.dtype != torch.bfloat16:
        raise AssertionError("CUDA BF16 classifier shadow was not prepared")
    # The head tiles the classifier by the batch's own mean-logit ordering,
    # computed before the forward; it must be a permutation of the vocabulary.
    probe_order = batch_vocab_order(
        torch.randn(8, args.dim, device="cuda", dtype=torch.bfloat16),
        classifier_shadow,
    )
    if probe_order.dtype != torch.int32 or not torch.equal(
        probe_order.sort().values,
        torch.arange(args.vocab_size, device="cuda", dtype=torch.int32),
    ):
        raise AssertionError(
            "the head's batch ordering is not a vocabulary permutation"
        )
    del probe_order
    if any("classifier_shadow" in name for name in model.state_dict()):
        raise AssertionError("derived classifier shadow entered the checkpoint state")
    classifier_shadow_ptr = classifier_shadow.data_ptr()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embedded = model.embed_tokens(
            torch.zeros(1, 1, dtype=torch.long, device="cuda")
        )
    if embedded.dtype != torch.bfloat16:
        raise AssertionError(f"CUDA residual seed is {embedded.dtype}, not bfloat16")
    # Do not carry this uncaptured autograd edge into the blocking capture stream.
    del embedded
    torch.cuda.synchronize()

    started = time.monotonic()
    runner = CudaGraphTrainer(model, optimizers, args, schedule)
    nadam = optimizers[1]
    if any(not group["foreach"] or group["capturable"] for group in nadam.param_groups):
        raise AssertionError("CUDA NAdam is not using the qualified foreach path")
    if not nadam.state or any(
        state["step"].item() != 0 or state["mu_product"].item() != 1
        for state in nadam.state.values()
    ):
        raise AssertionError("CUDA NAdam did not materialize fresh optimizer state")
    eval_runner = CudaEvalRunner(model, args, runner.pool)
    backend = execution_fields(model, runner, eval_runner)
    if backend["flash_sdpa"] != 1 or backend["cuda_graphs"] != 4:
        raise AssertionError(f"invalid production execution telemetry: {backend}")
    torch.cuda.synchronize()
    prepared = time.monotonic() - started
    capture_peak = torch.cuda.max_memory_allocated() / 2**30
    capture_reserved = torch.cuda.max_memory_reserved() / 2**30
    if capture_peak >= 22.0:
        raise AssertionError(
            f"capture peak leaves unsafe headroom: {capture_peak:.2f} GiB"
        )

    eval_scores = eval_runner.run(probe_validation)
    if any(not math.isfinite(value) for value in eval_scores.values()):
        raise AssertionError(f"nonfinite captured evaluation: {eval_scores}")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        eager_scores = evaluate(model, probe_validation, args, torch.device("cuda"))
    for key, value in eval_scores.items():
        if not math.isclose(value, eager_scores[key], rel_tol=2e-3, abs_tol=2e-3):
            raise AssertionError(
                f"captured evaluation drift in {key}: {value} versus "
                f"{eager_scores[key]}"
            )

    # Exercise the exact first-report sequence used by training after all
    # capture-time whole-block specializations have been prepared.  This
    # catches global-state/compiler-cache mismatches in monitors that the
    # captured train/eval bodies cannot expose by themselves.
    summary = route_summary(model, probe_validation, args, torch.device("cuda"))
    if not summary:
        raise AssertionError("CUDA route summary is empty for the arf condition")
    trace = iterate_fused(model, probe_validation.batch(0, 2, "cuda"), n_iters=2)
    if any(
        not math.isfinite(record[key])
        for record in trace
        for key in ("loss", "update_norm")
    ):
        raise AssertionError(f"nonfinite CUDA contraction trace: {trace}")

    compiled = counters["stats"]["unique_graphs"]
    generator = torch.Generator().manual_seed(0)
    records = []
    for index, (spec, state) in enumerate(runner.states.items()):
        rows = torch.randint(
            0,
            args.vocab_size,
            (args.micro_rows, args.seq_len + 1),
            generator=generator,
        )
        samples = []
        for replay_index in range(7):
            runner.zero_grad()
            runner.begin(spec, args.zloss if replay_index % 2 else 0.0)
            torch.cuda.synchronize()
            started = time.monotonic()
            runner.replay(state, rows, index + 1, index * args.micro_rows)
            torch.cuda.synchronize()
            samples.append(time.monotonic() - started)
        elapsed = statistics.median(samples[2:])
        # The captured body must reach the head accumulator and the eager
        # flush must carry it into the FP32 sink: every other quantity in the
        # step would still look finite if the classifier's contribution
        # quietly stopped arriving.
        head_sink = model.embed_tokens.grad_sink
        if runner.head_accum is None or head_sink is None:
            raise AssertionError("the CUDA head has no classifier accumulator")
        runner.prepare_optimizer(state)
        if runner.head_accum.any() or runner._head_pending:
            raise AssertionError("the head accumulator survived its flush")
        # The last replay is the only one this sink holds, and the token
        # lookup can reach at most its own micro_rows x seq_len ids; the
        # head's gradient reaches most of the vocabulary, so the width of the
        # sink is the head's contribution arriving.
        # Reductions straight to [V]; a bool copy of the sink would be 39 MB.
        touched = int(
            ((head_sink.amax(dim=1) > 0) | (head_sink.amin(dim=1) < 0)).sum()
        )
        if touched <= 2 * args.micro_rows * args.seq_len:
            raise AssertionError(
                f"the head's classifier gradient did not reach the sink: "
                f"{touched} of {args.vocab_size} rows"
            )
        grad_norm = clip_gradients(model.parameters())
        if not math.isfinite(grad_norm) or not math.isfinite(state.loss_sum.item()):
            active_names = {
                id(parameter): name for name, parameter in model.named_parameters()
            }
            nonfinite = [
                active_names.get(id(parameter), "<unnamed>")
                for parameter in state.active
                if not torch.isfinite(parameter.grad).all()
            ]
            raise AssertionError(
                f"nonfinite CUDA result in {spec}: loss={state.loss_sum.item()}, "
                f"grad_norm={grad_norm}, parameters={nonfinite[:12]}"
            )
        for optimizer in optimizers:
            optimizer.step()
        model.refresh_shadows()
        if model._classifier_shadow.data_ptr() != classifier_shadow_ptr:
            raise AssertionError(
                "classifier shadow address changed after optimizer step"
            )
        runner.zero_grad()
        records.append(
            f"k={spec.n_passes}/ckpt={int(spec.checkpoint)}:{elapsed * 1000:.1f}ms"
        )

    radius_errors = []
    for parameter, optimizer_state in optimizers[0].state.items():
        radius = optimizer_state["radius"]
        radius_errors.append((parameter.norm() - radius).abs() / radius)
    max_radius_error = torch.stack(radius_errors).max().item()
    if max_radius_error > 5e-6:
        raise AssertionError(f"NorMuonH Frobenius-radius drift: {max_radius_error:.3e}")

    # Freeze the full model, optimizer, and RNG surface into pinned host memory.
    # Serialization itself is covered by the root package's CPU atomic-write
    # test; this gate exercises the CUDA staging stream at production scale.
    staged = checkpoints.stage(CONTRACT, model, OptimizerPair(optimizers), args, step=3)
    staged_payload = staged.wait()
    if any(
        tensor.device.type != "cpu"
        for tensor in staged_payload[CONTRACT.state_key].values()
    ):
        raise AssertionError("staged checkpoint retained CUDA model tensors")
    del staged, staged_payload

    torch.cuda.synchronize()
    escaped = counters["stats"]["unique_graphs"] - compiled
    if escaped:
        raise AssertionError(f"{escaped} compiled graph(s) escaped preparation")
    print(
        "cuda gate | "
        f"prepare={prepared:.1f}s | allocated={capture_peak:.2f}GiB | "
        f"reserved={capture_reserved:.2f}GiB | "
        f"pkda_rel={pkda_value_rel:.4f} | "
        f"conv_rel={conv_rel:.4f} | "
        f"norm_gate_rel={norm_gate_rel:.4f} | "
        f"decode_rel={plain_decode_rel:.4f}/{hybrid_decode_rel:.4f}"
        f"/{loop_decode_rel:.4f}/{rope_loop_decode_rel:.4f} | "
        f"loop_grad_rel={loop_grad_rel:.4f}/floor={grad_floor:.4f} | "
        f"head_flush_rel={flush_rel:.2e}/micro={per_micro_rel:.2e} | "
        f"loop_cap_peak={loop_cap_peak:.2f}GiB | "
        f"stage_residual={stage_residual:.2f}GiB | "
        f"graphs={len(runner.states) + len(eval_runner.states)} | "
        + " | ".join(records)
    )


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", *argv],
        check=False,
    )
    if result.returncode:
        raise SystemExit(result.returncode)
    if torch.cuda.is_available():
        cuda_gate()
        gc.collect()
        torch.cuda.empty_cache()
        from .moe_probe import cuda_moe_gate

        cuda_moe_gate()
        gc.collect()
        torch.cuda.empty_cache()
        from .mtp_probe import cuda_mtp_gate

        cuda_mtp_gate()


if __name__ == "__main__":
    main()

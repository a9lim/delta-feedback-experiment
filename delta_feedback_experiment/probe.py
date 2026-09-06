"""Authoritative offline plus Jobe CUDA execution gate."""

from __future__ import annotations

import copy
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def cuda_gate() -> None:
    """Capture and replay every default DF mode at full screen geometry."""
    from torch._dynamo.utils import counters
    from transformer_experiments import checkpoints

    from .cuda_kernels import bespoke_route
    from .cuda_kernels import triton as route_triton
    from .model import (
        DFModel,
        KVCache,
        _ClassifierShadow,
        _fixed_cce_z,
        _route_sources,
        arm_config,
        batch_vocab_order,
        iterate_fused,
        linear_cross_entropy,
    )
    from .optim import OptimizerPair, build_optimizers
    from .pkda import (
        PreconditionedKDA,
        _l2norm,
        _PackedControlSplit,
        pkda_cuda_available,
        rms_norm_gated,
    )
    from .train import (
        CONTRACT,
        CudaEvalRunner,
        CudaGraphTrainer,
        build_parser,
        build_schedule,
        clip_gradients,
        evaluate,
        execution_fields,
        route_summary,
    )

    if linear_cross_entropy is None:
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
        present = torch.rand(n_sources, batch, length, device="cuda") > 0.2
        present[0] = True
        projected = (query * key).to(torch.bfloat16)
        routed, weights = bespoke_route(projected, present, null, 1e-6, heads, sources)
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
            present,
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
    norm_actual = rms_norm_gated(
        norm_output,
        norm_gate,
        norm_weight,
        None,
        "sigmoid",
        eps=pkda_norm_eps,
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

    # Exercise causal FlexAttention and explicit GQA prefix-cache decoding.
    # BF16 changes with block partitioning, so the
    # invariant is close recurrence rather than bit identity.
    def decode_parity(arm: str, layers: int) -> float:
        decode_cfg = arm_config(
            arm,
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
        )
        torch.manual_seed(0)
        decode_model = DFModel(decode_cfg).cuda().eval()
        decode_tokens = torch.randint(0, 1000, (2, 12), device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            full = decode_model.forward_column(
                decode_model.embed_tokens(decode_tokens)
            ).h_top
            cache = KVCache(decode_cfg, 2, "cuda", torch.bfloat16)
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
        if relative >= 0.04:
            raise AssertionError(f"{arm} cache parity drift: {relative:.4f}")
        del decode_model, decode_tokens, full, cache, prefill, pieces, incremental
        return relative

    vanilla_decode_rel = decode_parity("vanilla", 3)
    hybrid_decode_rel = decode_parity("base", 4)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    args = build_parser().parse_args(["cuda-probe", "--arm", "df"])
    schedule = build_schedule(args)
    torch.set_float32_matmul_precision("high")

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
        DFModel(
            arm_config(
                "df",
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
    if backend["flex"] != 1 or backend["cuda_graphs"] != 4:
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
        raise AssertionError("CUDA route summary is empty for the DF arm")
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
        runner.prepare_optimizer(state)
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
        f"decode_rel={vanilla_decode_rel:.4f}/{hybrid_decode_rel:.4f} | "
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


if __name__ == "__main__":
    main()

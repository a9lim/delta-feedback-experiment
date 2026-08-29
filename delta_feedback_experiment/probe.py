"""Authoritative offline plus Jobe CUDA execution gate."""

from __future__ import annotations

import math
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
        _fixed_cce_z,
        _route_sources,
        arm_config,
        flash_attn_func,
        linear_cross_entropy_apply,
    )
    from .optim import OptimizerPair, build_optimizers
    from .train import (
        CONTRACT,
        CudaEvalRunner,
        CudaGraphTrainer,
        build_parser,
        build_schedule,
        evaluate,
    )

    if flash_attn_func is None or linear_cross_entropy_apply is None:
        raise RuntimeError("Jobe gate requires flash-attn and cut-cross-entropy")
    if route_triton is None:
        raise RuntimeError("Jobe gate requires the bespoke Triton router")

    # The fixed-capacity router must match the semantic implementation in both
    # values and its source/query gradient. Use four routing heads and a masked
    # source to cover both MHDAR specialization and DF-soft prefix masking.
    torch.manual_seed(7)
    route_query = torch.randn(48, device="cuda", dtype=torch.float32).requires_grad_()
    route_key = torch.randn(48, device="cuda", dtype=torch.float32).requires_grad_()
    route_sources = tuple(
        torch.randn(3, 11, 48, device="cuda", dtype=torch.bfloat16).requires_grad_()
        for _ in range(6)
    )
    route_present = torch.rand(6, 3, 11, device="cuda") > 0.2
    route_present[0] = True
    projected = (route_query * route_key).to(torch.bfloat16)
    routed, route_weights = bespoke_route(
        projected, route_present, False, 1e-6, 4, route_sources
    )
    route_loss = routed.float().square().mean()
    route_loss.backward()
    route_grads = (
        route_query.grad.clone(),
        route_key.grad.clone(),
        *(source.grad.clone() for source in route_sources),
    )
    ref_query = route_query.detach().clone().requires_grad_()
    ref_key = route_key.detach().clone().requires_grad_()
    ref_sources = tuple(
        source.detach().clone().requires_grad_() for source in route_sources
    )
    ref_routed, ref_weights = _route_sources(
        ref_query, ref_key, 1e-6, route_present, 4, *ref_sources
    )
    ref_routed.float().square().mean().backward()
    ref_grads = (
        ref_query.grad,
        ref_key.grad,
        *(source.grad for source in ref_sources),
    )
    # The fused kernel accumulates the value mix in FP32 instead of reproducing
    # the reference's source-by-source BF16 rounding. Bound that intentional
    # numerical change by both relative norm and a loose elementwise ceiling.
    route_rel = torch.linalg.vector_norm(
        (routed - ref_routed).float()
    ) / torch.linalg.vector_norm(ref_routed.float())
    weight_rel = torch.linalg.vector_norm(
        route_weights - ref_weights
    ) / torch.linalg.vector_norm(ref_weights)
    if route_rel >= 0.01 or (routed - ref_routed).abs().max() >= 0.03:
        raise AssertionError("bespoke router value drift")
    if weight_rel >= 0.01 or (route_weights - ref_weights).abs().max() >= 0.015:
        raise AssertionError("bespoke router weight drift")
    for actual, expected in zip(route_grads, ref_grads, strict=True):
        if not torch.allclose(actual, expected, rtol=5e-2, atol=5e-3):
            raise AssertionError("bespoke router gradient drift")
    del route_query, route_key, route_sources, route_present, projected
    del routed, route_weights, route_loss, route_grads
    del ref_query, ref_key, ref_sources, ref_routed, ref_weights, ref_grads

    # CCE-native z-loss must retain the full-logit scalar and gradient semantics
    # while never constructing the classifier-wide activation in the real path.
    torch.manual_seed(11)
    cce_e = torch.randn(41, 32, device="cuda", dtype=torch.bfloat16).requires_grad_()
    cce_c = torch.randn(127, 32, device="cuda", dtype=torch.float32).requires_grad_()
    cce_t = torch.randint(0, 127, (41,), device="cuda")
    cce_ce, cce_z = _fixed_cce_z(cce_e, cce_c, cce_t)
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
    del cce_e, cce_c, cce_t, cce_de, cce_dc, ref_e, ref_c, ref_logits
    del cce_ce, cce_z, ref_ce, ref_z

    # Exercise both FlashAttention entry points: full causal attention and the
    # in-place native GQA KV cache. BF16 changes with block partitioning, so the
    # invariant is close recurrence rather than bit identity.
    decode_cfg = arm_config(
        "vanilla",
        vocab_size=1000,
        dim=128,
        layers=3,
        heads=4,
        kv_heads=2,
        head_dim=32,
        intermediate=256,
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
    decode_rel = (
        torch.linalg.vector_norm((full - incremental).float())
        / torch.linalg.vector_norm(full.float())
    ).item()
    if decode_rel >= 0.02:
        raise AssertionError(f"FlashAttention cache parity drift: {decode_rel:.4f}")
    del decode_model, decode_tokens, full, cache, prefill, pieces, incremental
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    args = build_parser().parse_args(["cuda-probe", "--arm", "df"])
    schedule = build_schedule(args)
    torch.set_float32_matmul_precision("high")
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
        lr_muon=args.lr_muon,
        wd_muon=args.wd_muon,
        lr_adam=args.lr_adam,
    )
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
    eval_runner = CudaEvalRunner(model, args, runner.pool)
    torch.cuda.synchronize()
    prepared = time.monotonic() - started
    capture_peak = torch.cuda.max_memory_allocated() / 2**30
    if capture_peak >= 22.0:
        raise AssertionError(
            f"capture peak leaves unsafe headroom: {capture_peak:.2f} GiB"
        )

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
        runner.begin(spec)
        started = time.monotonic()
        runner.replay(state, rows, index + 1, index * args.micro_rows)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        runner.prepare_optimizer(state)
        norms = torch._foreach_norm([parameter.grad for parameter in state.active])
        grad_norm = torch.stack(norms).norm().item()
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
        runner.zero_grad()
        records.append(
            f"k={spec.n_passes}/z={int(spec.want_z)}/ckpt={int(spec.checkpoint)}:"
            f"{elapsed * 1000:.1f}ms"
        )

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
        f"prepare={prepared:.1f}s | peak={capture_peak:.2f}GiB | "
        f"decode_rel={decode_rel:.4f} | "
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

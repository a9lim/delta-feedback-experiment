"""Bounded CUDA qualification for the sequential second-token objective."""

from __future__ import annotations

import gc
import math
import time

import torch
import torch.nn.functional as F


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1e-20)
    ).item()


def _gradient_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def _compare_gradients(actual, expected, *, label: str, bound: float) -> float:
    if actual.keys() != expected.keys():
        raise AssertionError(
            f"{label} changed the gradient surface: "
            f"missing={sorted(expected.keys() - actual.keys())}, "
            f"extra={sorted(actual.keys() - expected.keys())}"
        )
    difference = reference = 0.0
    for name, gradient in actual.items():
        if not torch.isfinite(gradient).all():
            raise AssertionError(f"{label} has a nonfinite gradient in {name}")
        difference += (gradient - expected[name]).square().sum().item()
        reference += expected[name].square().sum().item()
    relative = math.sqrt(difference / max(reference, 1e-40))
    if relative > bound:
        raise AssertionError(f"{label} gradient drift: {relative:.4f} > {bound}")
    # The large shared vocabulary matrix must not hide a missing auxiliary
    # projection or a severed path through the trunk in the aggregate norm.
    for prefix in ("mtp.", "blocks.", "embed_tokens."):
        names = [name for name in expected if name.startswith(prefix)]
        expected_part = torch.cat([expected[name].flatten() for name in names])
        actual_part = torch.cat([actual[name].flatten() for name in names])
        relative_part = _relative_error(actual_part, expected_part)
        if expected_part.norm() <= 0 or actual_part.norm() <= 0:
            raise AssertionError(f"{label} has no gradient through {prefix}")
        if relative_part > 2 * bound:
            raise AssertionError(
                f"{label} gradient drift through {prefix}: {relative_part:.4f}"
            )
    return relative


class _ValidationRows:
    def __init__(self, args):
        self.rows = torch.randint(
            args.vocab_size,
            (args.eval_rows, args.seq_len + 1),
            generator=torch.Generator().manual_seed(83),
        )

    def batch(self, first, count, device=None):
        rows = self.rows[first : first + count]
        return rows.to(device) if device is not None else rows


def _new_model(args):
    from .model import DeltaModel, condition_config
    from .train import model_fields

    torch.manual_seed(args.seed)
    model = DeltaModel(condition_config(args.condition, **model_fields(args)))
    model = model.cuda().train()
    model.refresh_shadows()
    return model


def _auxiliary_dense_parity(args) -> float:
    """CCE versus explicit logits, including gradients from only the MTP loss."""
    from .model import multipass, sequence_ce

    model = _new_model(args)
    rows = _ValidationRows(args).rows.cuda()
    prefix = torch.full((2, rows.shape[0]), 7, dtype=torch.long, device="cuda")
    z_coef = 0.017
    results = {}
    for dense in (False, True):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = multipass(model, rows, 3, prefix_lens=prefix)
            if dense:
                ce_terms, z_terms = [], []
                for out in outs:
                    hidden = model.mtp_hidden(out.h_top[:, :-1], rows[:, 1:-1])
                    # Materialize only this tiny head. Use the same BF16
                    # operands as CCE so the comparison isolates its loss and
                    # backward rather than changing the activation recipe.
                    logits = F.linear(
                        model.readout_input(hidden),
                        model.embed_tokens.weight.to(torch.bfloat16),
                    ).float()
                    ce_terms.append(
                        F.cross_entropy(logits.flatten(0, 1), rows[:, 2:].reshape(-1))
                    )
                    z_terms.append(logits.logsumexp(-1).square().mean())
                loss = ce_terms[0] + (ce_terms[1] + ce_terms[2]) / 2
                loss = loss + z_coef * (z_terms[0] + (z_terms[1] + z_terms[2]) / 2)
            else:
                # Call the real cropped sequence_ce path, retaining CE and
                # z-loss independently so neither head weighting can mask it.
                ce_terms, z_terms = [], []
                for out in outs:
                    hidden = model.mtp_hidden(out.h_top[:, :-1], rows[:, 1:-1])
                    ce, z = sequence_ce(model, hidden, rows[:, 2:])
                    ce_terms.append(ce)
                    z_terms.append(z)
                loss = ce_terms[0] + (ce_terms[1] + ce_terms[2]) / 2
                loss = loss + z_coef * (z_terms[0] + (z_terms[1] + z_terms[2]) / 2)
        loss.backward()
        results[dense] = (
            torch.stack([*ce_terms, *z_terms]).detach().cpu(),
            _gradient_snapshot(model),
        )
        del outs, hidden, loss, ce_terms, z_terms
        if dense:
            del logits
        else:
            del ce, z
    torch.testing.assert_close(results[False][0], results[True][0], rtol=3e-3, atol=3e-3)
    relative = _compare_gradients(
        results[False][1], results[True][1], label="MTP CUDA CCE/dense", bound=0.06
    )
    del model, rows, prefix, results
    gc.collect()
    torch.cuda.empty_cache()
    return relative


def _tiny_capture_parity(args) -> tuple[float, float]:
    from .model import combine_pass_losses, multipass, multipass_loss
    from .optim import build_optimizers
    from .train import CudaGraphTrainer, GraphSpec, build_schedule, micro_draws

    class _CheckpointProbeTrainer(CudaGraphTrainer):
        def _reachable_specs(self, schedule):
            # Screen arfm's natural three-pass mode is below the checkpoint
            # threshold. Qualify the auxiliary recomputation separately using
            # the same captured trainer body on a small model.
            return [*super()._reachable_specs(schedule), GraphSpec(2, True)]

    schedule = build_schedule(args)
    rows = _ValidationRows(args).rows.cuda()
    eager = _new_model(args)
    expected = {}
    for count in (1, 2, 3):
        eager.zero_grad(set_to_none=True)
        metrics = torch.zeros(4, device="cuda")
        for offset in range(args.batch_rows):
            micro = rows[offset : offset + 1]
            prefix, jitter = micro_draws(
                args, 37, offset, count, 1, args.dim, torch.device("cuda")
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs = multipass(eager, micro, count, prefix_lens=prefix, jitter=jitter)
                result = multipass_loss(
                    eager, micro, outs, z_coef=0.017, mtp_weight=args.mtp_weight
                )
            (result.total / args.batch_rows).backward()
            metrics.add_(torch.stack([
                result.total.detach(), result.ntp[0].detach(),
                combine_pass_losses(result.ntp).detach(),
                combine_pass_losses(result.mtp).detach(),
            ]) / args.batch_rows)
            del outs, result
        expected[count] = (metrics.cpu(), _gradient_snapshot(eager))
    del eager, metrics, micro, prefix, jitter
    gc.collect()
    torch.cuda.empty_cache()

    model = _new_model(args)
    optimizers = build_optimizers(
        model, lr_normuonh=args.lr_normuonh, lr_nadam=args.lr_nadam
    )
    runner = _CheckpointProbeTrainer(model, optimizers, args, schedule)
    if {spec.n_passes for spec in runner.states} != {1, 2, 3}:
        raise AssertionError("the tiny MTP schedule did not reach all pass counts")
    max_relative = checkpoint_relative = 0.0
    for spec in runner.states:
        runner.zero_grad()
        state = runner.begin(spec, z_coef=0.017)
        for offset in range(args.batch_rows):
            runner.replay(state, rows[offset : offset + 1], 37, offset)
        runner.prepare_optimizer(state)
        if runner.head_accum is None or runner.head_accum.any() or runner._head_pending:
            raise AssertionError(f"MTP head accumulation did not flush in {spec}")
        actual_metrics = torch.stack([
            state.loss_sum, state.pass1_sum, state.ntp_sum, state.mtp_sum,
        ]).cpu()
        expected_metrics, expected_gradient = expected[spec.n_passes]
        torch.testing.assert_close(
            actual_metrics, expected_metrics, rtol=3e-3, atol=3e-3, msg=str(spec)
        )
        relative = _compare_gradients(
            _gradient_snapshot(model), expected_gradient,
            label=f"MTP captured/eager {spec}", bound=0.06,
        )
        max_relative = max(max_relative, relative)
        if spec.checkpoint:
            checkpoint_relative = relative
    del runner, model, optimizers, state, rows, expected
    gc.collect()
    torch.cuda.empty_cache()
    return max_relative, checkpoint_relative


def _screen_capture(args) -> dict:
    from torch._dynamo.utils import counters

    from .optim import build_optimizers
    from .train import CudaEvalRunner, CudaGraphTrainer, build_schedule, evaluate

    torch.cuda.reset_peak_memory_stats()
    model = _new_model(args)
    optimizers = build_optimizers(
        model, lr_normuonh=args.lr_normuonh, lr_nadam=args.lr_nadam
    )
    validation = _ValidationRows(args)
    started = time.monotonic()
    runner = CudaGraphTrainer(model, optimizers, args, build_schedule(args))
    if {spec.n_passes for spec in runner.states} != {1, 2, 3}:
        raise AssertionError("the production MTP schedule did not reach all pass counts")
    eval_runner = CudaEvalRunner(model, args, runner.pool)
    torch.cuda.synchronize()
    prepared = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    if peak >= 22.0:
        raise AssertionError(f"MTP capture leaves insufficient headroom: {peak:.2f} GiB")

    scores = eval_runner.run(validation)
    required_metrics = {"val", "val_fused", "val_mtp", "val_mtp_fused"}
    if set(scores) != required_metrics:
        raise AssertionError(f"MTP captured evaluation has wrong metrics: {scores}")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        eager_scores = evaluate(model, validation, args, torch.device("cuda"))
    if set(eager_scores) != required_metrics:
        raise AssertionError(f"MTP eager evaluation has wrong metrics: {eager_scores}")
    for key, value in scores.items():
        if not math.isfinite(value) or not math.isclose(
            value, eager_scores[key], rel_tol=2e-3, abs_tol=2e-3
        ):
            raise AssertionError(
                f"MTP captured evaluation drift in {key}: {value} / {eager_scores[key]}"
            )

    compiled = counters["stats"]["unique_graphs"]
    rows = validation.rows[: args.micro_rows].cuda()
    records = []
    for spec in runner.states:
        # Repeat the same graph with both zero and nonzero z-loss. Only the
        # coefficient tensor changes; the compiled graph must stay fixed.
        for z_coef in (0.0, 0.017):
            runner.zero_grad()
            state = runner.begin(spec, z_coef=z_coef)
            runner.replay(state, rows, 37, 0)
            calls = 2 * spec.n_passes
            if runner._head_calls(state) != calls:
                raise AssertionError(f"MTP undercounted classifier calls in {spec}")
            if runner.head_accum is None:
                raise AssertionError("MTP has no classifier accumulation buffer")
            if calls > args.head_flush_every and (
                runner.head_accum.any() or runner._head_pending
            ):
                raise AssertionError("MTP six-call mode exceeded the BF16 flush window")
            runner.prepare_optimizer(state)
            if runner.head_accum.any() or runner._head_pending:
                raise AssertionError("MTP classifier gradient survived the final flush")
            for name, parameter in model.named_parameters():
                if parameter in state.active and (
                    parameter.grad is None or not torch.isfinite(parameter.grad).all()
                ):
                    raise AssertionError(f"nonfinite MTP gradient in {spec}: {name}")
            for name, parameter in (
                ("auxiliary projection", model.mtp.projection.weight),
                ("trunk projection", model.blocks[0].attn.q_proj.weight),
                ("shared vocabulary", model.embed_tokens.weight),
            ):
                if parameter.grad is None or not parameter.grad.any():
                    raise AssertionError(f"MTP did not train the {name} in {spec}")
                if parameter.grad.dtype != torch.float32:
                    raise AssertionError(f"MTP lost FP32 accumulation for the {name}")
            metrics = torch.stack([
                state.loss_sum, state.pass1_sum, state.ntp_sum, state.mtp_sum,
            ])
            if not torch.isfinite(metrics).all() or not (metrics > 0).all():
                raise AssertionError(f"nonfinite or empty MTP training metrics: {metrics}")
            if z_coef == 0:
                torch.testing.assert_close(
                    state.loss_sum, state.ntp_sum + args.mtp_weight * state.mtp_sum,
                    rtol=2e-5, atol=2e-5,
                )
        # The classifier reaches vocabulary rows that no token lookup touched.
        sink = model.embed_tokens.grad_sink
        touched = int(((sink.amax(1) > 0) | (sink.amin(1) < 0)).sum())
        if touched <= 2 * args.micro_rows * args.seq_len:
            raise AssertionError(f"MTP classifier gradient reached only {touched} rows")
        records.append(f"k={spec.n_passes}/ckpt={int(spec.checkpoint)}/heads={calls}")

    # A second validation run after all training modes protects their shared
    # graph pool and fixed-address inputs from cross-mode reuse errors.
    repeated = eval_runner.run(validation)
    for key in required_metrics:
        if not math.isclose(scores[key], repeated[key], rel_tol=2e-3, abs_tol=2e-3):
            raise AssertionError(f"MTP training replay changed unstepped validation {key}")
    torch.cuda.synchronize()
    escaped = counters["stats"]["unique_graphs"] - compiled
    if escaped:
        raise AssertionError(f"{escaped} MTP compiled graph(s) escaped preparation")
    result = {
        "prepare": prepared, "peak": peak, "reserved": reserved,
        "graphs": len(runner.states) + len(eval_runner.states), "modes": records,
    }
    del runner, eval_runner, model, optimizers, state, rows, sink, parameter, metrics
    gc.collect()
    torch.cuda.empty_cache()
    return result


def cuda_mtp_gate() -> None:
    """Qualify CCE, FP32 sinks, checkpointing, and screen MTP graph replay."""
    from .model import linear_cross_entropy_apply
    from .pkda import pkda_cuda_available
    from .train import parse_run_args

    if not torch.cuda.is_available() or linear_cross_entropy_apply is None:
        raise RuntimeError("MTP CUDA gate requires CUDA and cut-cross-entropy")
    if not pkda_cuda_available():
        raise RuntimeError("MTP CUDA gate requires the pinned FLA PKDA kernels")
    torch.set_float32_matmul_precision("high")
    common = [
        "cuda-mtp-probe", "--condition", "arfm", "--steps", "64",
        "--feedback-start", "0.5", "--three-pass", "0.5", "--micro-rows", "1",
    ]
    tiny = parse_run_args([
        *common, "--batch-rows", "2", "--eval-rows", "2", "--seq-len", "64",
        "--vocab-size", "257", "--dim", "64", "--layers", "4", "--heads", "4",
        "--kv-heads", "2", "--head-dim", "16", "--intermediate", "128",
        "--pkda-heads", "2", "--pkda-head-dim", "32",
    ])
    cce_relative = _auxiliary_dense_parity(tiny)
    print(f"cuda mtp | cropped auxiliary CCE/dense gradient rel={cce_relative:.4f}", flush=True)
    captured_relative, checkpoint_relative = _tiny_capture_parity(tiny)
    print(
        f"cuda mtp | captured/eager gradient rel={captured_relative:.4f} | "
        f"checkpoint rel={checkpoint_relative:.4f}", flush=True,
    )
    screen = parse_run_args([
        *common, "--batch-rows", "1", "--eval-rows", "1", "--seq-len", "4096",
    ])
    result = _screen_capture(screen)
    print(
        "cuda mtp gate | "
        f"prepare={result['prepare']:.1f}s | allocated={result['peak']:.2f}GiB | "
        f"reserved={result['reserved']:.2f}GiB | graphs={result['graphs']} | "
        + " | ".join(result["modes"]), flush=True,
    )

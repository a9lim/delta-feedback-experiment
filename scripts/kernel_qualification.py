"""Compare CUDA kernel revisions on fixed trained weights and addressed rows.

Run from the experiment checkout on an idle CUDA host. This does not resume a
run or write training snapshots. The baseline stores FP32 accumulated gradients
on the host; a candidate compares every parameter after the same full batches.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from delta_feedback_experiment import analysis
from delta_feedback_experiment.data import TokenData
from delta_feedback_experiment.model import DeltaModel
from delta_feedback_experiment.optim import build_optimizers
from delta_feedback_experiment.train import (
    CudaEvalRunner,
    CudaGraphTrainer,
    build_schedule,
    parse_run_args,
    read_checkpoint,
)


def revision(directory: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()


def gradient_comparison(current: dict, reference: dict) -> dict:
    totals = {"all": [0.0, 0.0, 0.0, 0.0], "trunk": [0.0, 0.0, 0.0, 0.0]}
    parameters = []
    if current.keys() != reference.keys():
        raise AssertionError("active gradient names differ from the baseline")
    for name, gradient in current.items():
        baseline = reference[name]
        if not torch.isfinite(gradient).all():
            raise AssertionError(f"nonfinite gradient: {name}")
        difference = gradient - baseline
        error = difference.square().sum().item()
        original = baseline.square().sum().item()
        actual = gradient.square().sum().item()
        dot = (gradient * baseline).sum().item()
        for scope in ("all", "trunk") if name != "embed_tokens.weight" else ("all",):
            for index, value in enumerate((error, original, actual, dot)):
                totals[scope][index] += value
        parameters.append(
            {
                "name": name,
                "relative_l2": (error / max(original, 1e-30)) ** 0.5,
                "reference_norm": original**0.5,
                "max_absolute": difference.abs().max().item(),
            }
        )
    result = {}
    for scope, (error, original, actual, dot) in totals.items():
        result[scope] = {
            "relative_l2": (error / max(original, 1e-30)) ** 0.5,
            "reference_norm": original**0.5,
            "candidate_norm": actual**0.5,
            "norm_ratio": (actual / max(original, 1e-30)) ** 0.5,
            "cosine": dot / max((original * actual) ** 0.5, 1e-30),
        }
    result["largest_parameter_errors"] = sorted(
        parameters, key=lambda item: item["relative_l2"], reverse=True
    )[:12]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--first-row", type=int, default=100000)
    parser.add_argument("--step", type=int, default=9000)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/delta/dclm-100b"))
    parser.add_argument("--passes", type=int, nargs="+", default=[1, 2, 3])
    options = parser.parse_args()
    if options.write_reference and options.reference.exists():
        raise FileExistsError(options.reference)
    if options.write_reference:
        options.reference.mkdir(parents=True)

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(1)
    args = parse_run_args(["kernel-qualification", "--condition", "arf"])
    payload = read_checkpoint(options.snapshot)
    saved = analysis.saved_args(payload)
    for field in (*analysis.GEOMETRY, "seq_len", "condition"):
        setattr(args, field, saved[field])
    model = DeltaModel(analysis.config_from_args(saved))
    model.load_state_dict(payload["state"])
    del payload
    gc.collect()
    model.cuda().train()
    optimizers = build_optimizers(
        model, lr_normuonh=args.lr_normuonh, lr_nadam=args.lr_nadam
    )
    data = TokenData.load(options.data_dir, "train", args.seq_len)
    root = Path(__file__).resolve().parents[1]
    identity = {
        "experiment": revision(root),
        "fla": revision(root.parent / "vendor/flash-linear-attention"),
        "cce": revision(root.parent / "vendor/ml-cross-entropy"),
        "snapshot": str(options.snapshot.resolve()),
        "first_row": options.first_row,
        "randomness_step": options.step,
        "micro_rows": args.micro_rows,
        "batch_rows": args.batch_rows,
        "seq_len": args.seq_len,
        "zloss": args.zloss,
        "head_flush_every": args.head_flush_every,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    print(json.dumps({"identity": identity}), flush=True)
    if options.write_reference:
        (options.reference / "identity.json").write_text(json.dumps(identity, indent=2))
    else:
        prior = json.loads((options.reference / "identity.json").read_text())
        for field in (
            "snapshot",
            "first_row",
            "randomness_step",
            "micro_rows",
            "batch_rows",
            "seq_len",
            "zloss",
            "head_flush_every",
        ):
            if prior[field] != identity[field]:
                raise ValueError(f"baseline mismatch: {field}")

    started = time.monotonic()
    runner = CudaGraphTrainer(model, optimizers, args, build_schedule(args))
    evaluation = CudaEvalRunner(model, args, runner.pool)
    print(
        json.dumps(
            {
                "prepare_seconds": time.monotonic() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "eval_graphs": len(evaluation.states)
                if hasattr(evaluation, "states")
                else 1,
            }
        ),
        flush=True,
    )

    for spec, state in runner.states.items():
        if spec.n_passes not in options.passes:
            continue
        runner.zero_grad()
        runner.begin(spec, args.zloss)
        rows = data.batch(options.first_row, args.micro_rows)
        samples = []
        for trial in range(11):
            torch.cuda.synchronize()
            started = time.perf_counter()
            runner.replay(state, rows, options.step, options.first_row)
            torch.cuda.synchronize()
            if trial >= 2:
                samples.append((time.perf_counter() - started) * 1000)
        runner.zero_grad()
        runner.begin(spec, args.zloss)
        torch.cuda.synchronize()
        started = time.perf_counter()
        runner.replay_batch(state, data, options.step, options.first_row)
        torch.cuda.synchronize()
        batch_ms = (time.perf_counter() - started) * 1000
        runner.prepare_optimizer(state)
        gradients = {
            name: parameter.grad.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        result = {
            "passes": spec.n_passes,
            "replay_median_ms": statistics.median(samples),
            "replay_samples_ms": samples,
            "batch_forward_backward_ms": batch_ms,
            "loss": state.loss_sum.item(),
            "pass1_loss": state.pass1_sum.item(),
        }
        destination = options.reference / f"k{spec.n_passes}.pt"
        if options.write_reference:
            torch.save({"result": result, "gradients": gradients}, destination)
        else:
            reference = torch.load(destination, map_location="cpu", weights_only=True)
            result["gradient_comparison"] = gradient_comparison(
                gradients, reference["gradients"]
            )
            result["baseline"] = reference["result"]
            del reference
        print(json.dumps(result), flush=True)
        del gradients
        gc.collect()


if __name__ == "__main__":
    main()

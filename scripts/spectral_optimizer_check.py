"""Sample exact spectral errors and optionally time screen optimizer buckets.

Accuracy uses synthetic post-NorMuon directions at screen matrix shapes,
including rotating rank-one gradients and anisotropic row history. Exact SVD
is diagnostic only. CUDA benchmarking updates one bounded bucket at a time,
then weights the timings by screen's matrix counts. It does not run a model,
measure end-to-end training, or establish hyperparameter transfer.

    python scripts/spectral_optimizer_check.py --output /tmp/spectral.json
    python scripts/spectral_optimizer_check.py --device cuda --benchmark \
        --output /tmp/spectral-cuda.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch

from delta_feedback_experiment.model import DeltaModel, condition_config
from delta_feedback_experiment.optim import (
    SPECTRAL_POWER_STEPS,
    NorMuonH,
    _spectral_norm,
    orthogonalize,
    split_parameters,
)
from delta_feedback_experiment.train import model_fields, parse_run_args

BUCKET_ELEMENTS = 32 * 1024 * 1024


def screen_buckets() -> Counter:
    args = parse_run_args(["spectral-check", "--scale", "screen"])
    with torch.device("meta"):
        model = DeltaModel(condition_config("fl", **model_fields(args)))
    counts = Counter()
    for name, parameters in split_parameters(model).items():
        if not name.startswith("normuonh"):
            continue
        for shape, count in Counter(tuple(p.shape) for p in parameters).items():
            size = max(1, BUCKET_ELEMENTS // math.prod(shape))
            full, remainder = divmod(count, size)
            if full:
                counts[(*shape, size)] += full
            if remainder:
                counts[(*shape, remainder)] += 1
    return counts


@torch.no_grad()
def accuracy(device: torch.device, shapes) -> list[dict]:
    records = []
    for rows, cols in shapes:
        for kind in ("random", "rank_one", "anisotropic_rows"):
            generator = torch.Generator().manual_seed(20260912)
            weight = torch.randn(rows, cols, generator=generator).to(device) / math.sqrt(cols)
            radius = weight.norm()
            momentum = torch.zeros_like(weight)
            row_moment = torch.zeros(rows, 1, device=device)
            if kind == "anisotropic_rows":
                row_moment = torch.logspace(-3, 1, rows, device=device)[:, None]
            vector = torch.zeros(1, cols, device=device)
            for step in range(3):
                if kind == "rank_one":
                    gradient = torch.outer(
                        torch.randn(rows, generator=generator),
                        torch.randn(cols, generator=generator),
                    ).to(device)
                else:
                    gradient = torch.randn(rows, cols, generator=generator).to(device)
                momentum = torch.lerp(momentum, gradient, 0.05)
                update = orthogonalize(torch.lerp(gradient, momentum, 0.95))
                row_moment = torch.lerp(
                    row_moment, update.square().mean(dim=-1, keepdim=True), 0.05
                )
                update = update / (row_moment.sqrt() + 1e-8)
                update = update / update.norm()
                tangent = update - (weight * update).sum() / weight.square().sum() * weight
                tangent = tangent / tangent.norm()
                estimate, vector = _spectral_norm(tangent[None], vector, 1e-8)
                exact = torch.linalg.matrix_norm(tangent, ord=2)
                lr = 0.006
                trial = weight - lr * math.sqrt(rows / cols) * tangent / estimate[0]
                new_weight = radius * trial / trial.norm()
                final = math.sqrt(cols / rows) * torch.linalg.matrix_norm(
                    new_weight - weight, ord=2
                ) / lr
                records.append({
                    "shape": [rows, cols], "gradient": kind, "step": step + 1,
                    "estimated_over_exact": (estimate[0] / exact).item(),
                    "final_rms_operator_over_lr": final.item(),
                })
                weight = new_weight
    return records


@torch.no_grad()
def benchmark(device: torch.device, buckets: Counter) -> dict:
    records = []
    for (rows, cols, batch), multiplicity in buckets.items():
        generator = torch.Generator().manual_seed(20260912)
        parameters = [
            torch.nn.Parameter(
                torch.randn(rows, cols, generator=generator).to(device) / math.sqrt(cols)
            ) for _ in range(batch)
        ]
        for parameter in parameters:
            parameter.grad = torch.randn(rows, cols, generator=generator).to(device)
        optimizer = NorMuonH(parameters, max_bucket_elements=BUCKET_ELEMENTS)
        for _ in range(4):
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(10):
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 / 10
        records.append({
            "shape": [rows, cols], "batch": batch, "multiplicity": multiplicity,
            "milliseconds": elapsed,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
            if device.type == "cuda" else None,
        })
        print(f"bucket {rows}x{cols} x{batch}: {elapsed:.3f} ms", flush=True)
        del optimizer, parameters, parameter
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "buckets": records,
        "screen_weighted_optimizer_ms": sum(
            record["milliseconds"] * record["multiplicity"] for record in records
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")  # production trainer policy
    buckets = screen_buckets()
    shapes = sorted({(rows, cols) for rows, cols, _ in buckets})
    result = {
        "device": str(device), "torch": torch.__version__,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "power_steps": SPECTRAL_POWER_STEPS,
        "scope": "synthetic screen matrix samples and weighted bucket timings",
        "accuracy": [] if args.benchmark_only else accuracy(device, shapes),
    }
    if args.benchmark or args.benchmark_only:
        result["benchmark"] = benchmark(device, buckets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()

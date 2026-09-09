"""Compare split and packed projection gradients on the screen CUDA geometry.

    python scripts/packed_gradient_check.py --compile --selective --output /tmp/packed.json

Synthetic operands; includes 128-call accumulation, no training-quality claim.
``--selective`` adds pointwise work and a compiled checkpoint path to detect
omitted or repeated gradient-sink writes during activation recomputation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from pathlib import Path

import torch

from delta_feedback_experiment.activation import checkpoint_context
from delta_feedback_experiment.cuda_kernels import sink_linear


def discrepancy(actual, expected):
    actual, expected = actual.float(), expected.float()
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "relative_l2": float(
            (actual - expected).norm() / expected.norm().clamp_min(1e-30)
        ),
        "norm_ratio": float(actual.norm() / expected.norm().clamp_min(1e-30)),
        "max_abs": float((actual - expected).abs().max()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual.flatten(), expected.flatten(), dim=0
            )
        ),
    }


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    gc.collect()
    previous = gc.isenabled()
    gc.disable()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            out = fn()
    finally:
        if previous:
            gc.enable()
    return graph, out


def benchmark_case(name, splits, args):
    torch.manual_seed(1809)
    m, d = 4096, 768
    total_rows = sum(splits)
    weights = tuple(
        torch.nn.Parameter(torch.randn(rows, d, device="cuda") / math.sqrt(d))
        for rows in splits
    )
    shadow = torch.cat([weight.detach() for weight in weights]).to(torch.bfloat16)
    x = torch.randn(m, d, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn(
        m, total_rows, device="cuda", dtype=torch.bfloat16
    ) / math.sqrt(total_rows)
    paths = {}
    modes = (
        ("split", "packed", "packed_selective")
        if args.selective
        else ("split", "packed")
    )
    for mode in modes:
        packed_mode = mode != "split"
        slab = (
            torch.zeros(total_rows, d, device="cuda", dtype=torch.float32)
            if packed_mode
            else None
        )
        sinks = (
            slab.split(splits)
            if packed_mode
            else tuple(torch.zeros_like(weight) for weight in weights)
        )
        activations = x.clone().requires_grad_()
        activations.grad = torch.zeros_like(activations)

        def projection(value, sinks=sinks, slab=slab):
            output = sink_linear(value, weights, sinks, shadow, packed_sink=slab)
            # All comparison arms do identical work. The nonlinear composition
            # supplies genuine recomputation beyond the MUST_SAVE matrix op.
            return output.sin() * torch.sigmoid(output) if args.selective else output

        if mode == "packed_selective":

            def forward(value, projection=projection):
                return torch.utils.checkpoint.checkpoint(
                    projection,
                    value,
                    use_reentrant=False,
                    preserve_rng_state=False,
                    context_fn=checkpoint_context,
                )
        else:
            forward = projection
        if args.compile:
            forward = torch.compile(
                forward,
                fullgraph=True,
                dynamic=False,
                mode="max-autotune-no-cudagraphs",
            )

        def step(forward=forward, activations=activations):
            activations.grad.zero_()
            output = forward(activations)
            output.backward(upstream)
            return output

        graph, output = capture(step)
        paths[mode] = {
            "sinks": sinks,
            "x": activations,
            "graph": graph,
            "output": output,
        }

    result = {
        "name": name,
        "m": m,
        "dim": d,
        "splits": list(splits),
        "compiled": args.compile,
        "selective_check": args.selective,
        "note": "Captured projection forward+dX+dW; selective checks add identical nonlinear work to every arm.",
    }
    for calls in (1, 128):
        for path in paths.values():
            for sink in path["sinks"]:
                sink.fill_(0.125)
            for _ in range(calls):
                path["graph"].replay()
        torch.cuda.synchronize()
        result[f"parity_{calls}_calls"] = {
            "output": discrepancy(paths["packed"]["output"], paths["split"]["output"]),
            "input_gradient": discrepancy(
                paths["packed"]["x"].grad, paths["split"]["x"].grad
            ),
            "weight_gradients": [
                discrepancy(a, b)
                for a, b in zip(
                    paths["packed"]["sinks"], paths["split"]["sinks"], strict=True
                )
            ],
        }
        if args.selective:
            checkpointed, raw = paths["packed_selective"], paths["packed"]
            check = {
                "output": discrepancy(checkpointed["output"], raw["output"]),
                "input_gradient": discrepancy(checkpointed["x"].grad, raw["x"].grad),
                # Remove initial contents so a missing small gradient cannot
                # hide against the prefilled sink's norm.
                "weight_gradients": [
                    discrepancy(a - 0.125, b - 0.125)
                    for a, b in zip(checkpointed["sinks"], raw["sinks"], strict=True)
                ],
            }
            values = [
                check["output"],
                check["input_gradient"],
                *check["weight_gradients"],
            ]
            check["passed"] = all(
                row["finite"] and row["relative_l2"] <= args.max_relative_error
                for row in values
            )
            result[f"selective_parity_{calls}_calls"] = check
    timings = {mode: [] for mode in paths}
    for round_index in range(args.rounds):
        # Alternate order to reduce clock/thermal attribution bias.
        timing_order = tuple(paths) if round_index % 2 == 0 else tuple(reversed(paths))
        for mode in timing_order:
            graph = paths[mode]["graph"]
            for _ in range(5):
                graph.replay()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(args.replays):
                graph.replay()
            end.record()
            end.synchronize()
            timings[mode].append(start.elapsed_time(end) / args.replays)
    result["milliseconds"] = {
        mode: {"samples": samples, "median": statistics.median(samples)}
        for mode, samples in timings.items()
    }
    result["speed_ratio"] = (
        result["milliseconds"]["split"]["median"]
        / result["milliseconds"]["packed"]["median"]
    )
    result["saved_ms"] = (
        result["milliseconds"]["split"]["median"]
        - result["milliseconds"]["packed"]["median"]
    )
    del paths
    torch.cuda.synchronize()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/packed-gradient-check.json")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--selective",
        action="store_true",
        help="also check selective checkpointing; implies --compile",
    )
    parser.add_argument(
        "--max-relative-error",
        type=float,
        default=0.01,
        help="selective versus raw numerical failure threshold",
    )
    args = parser.parse_args()
    args.compile = args.compile or args.selective
    torch.set_float32_matmul_precision("high")
    report = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "cases": [],
    }
    for name, splits in [
        ("pkda_qkv", (1280, 1280, 1280)),
        ("dense_qkv_gate", (1536, 768)),
    ]:
        row = benchmark_case(name, splits, args)
        report["cases"].append(row)
        print(json.dumps(row), flush=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        torch.cuda.empty_cache()
    if args.selective and any(
        not row[f"selective_parity_{calls}_calls"]["passed"]
        for row in report["cases"]
        for calls in (1, 128)
    ):
        raise SystemExit("Selective checkpoint gradient parity failed; see saved JSON.")


if __name__ == "__main__":
    main()

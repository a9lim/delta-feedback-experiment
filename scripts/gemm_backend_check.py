"""Compare Inductor GEMM backends on screen-sized projections and SwiGLU.

Run each backend in a fresh process, serially on an otherwise idle GPU::

    python scripts/gemm_backend_check.py --backends ATEN,TRITON --output-dir /tmp/gemm-base
    python scripts/gemm_backend_check.py --backends ATEN,TRITON,NVGEMM --output-dir /tmp/gemm-nv

JSON reports include both operator eligibility and generated-code selection.
An enabled NVGEMM backend with no selected kernels is not an NVGEMM speedup.
This is an engineering microbenchmark, not a training-quality comparison.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import torch.nn.functional as F


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def discrepancy(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual, expected = actual.float(), expected.float()
    error = actual - expected
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs": error.abs().max().item(),
        "relative_l2": (error.norm() / expected.norm().clamp_min(1e-20)).item(),
        "cosine": F.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item(),
    }


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x, weight)


def swiglu(x: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    gate, value = F.linear(x, up).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * value, down)


def operator_inventory(shapes: list[tuple[int, int, int]]) -> list[dict]:
    """Ask CUTLASS about the actual GPU and all three GEMM orientations."""
    import cutlass.operators as ops

    target_sm = "".join(map(str, torch.cuda.get_device_capability()))
    inventory = []
    for m, n, k in shapes:
        # Forward x @ w.T, input gradient dy @ w, weight gradient dy.T @ x.
        x = torch.empty(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
        dy = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        for name, a, b in (
            ("forward", x, w.T),
            ("input_grad", dy, w),
            ("weight_grad", dy.T, x),
        ):
            output = torch.empty(
                a.shape[0], b.shape[1], device="cuda", dtype=torch.bfloat16
            )
            arguments = ops.GemmArguments(a, b, output, accumulator_type=torch.float32)
            operators = ops.get_operators(arguments, target_sm=target_sm)
            inventory.append(
                {
                    "projection_shape_mnk": [m, n, k],
                    "operation": name,
                    "eligible_count": len(operators),
                    "operator_examples": [
                        operator.metadata.operator_name for operator in operators[:3]
                    ],
                }
            )
    return inventory


def benchmark_case(name: str, options: argparse.Namespace) -> dict:
    from torch._inductor import config
    from torch._inductor.select_algorithm import (
        add_feedback_saver,
        clear_feedback_savers,
    )
    from torch._inductor.utils import run_and_get_code

    torch.manual_seed(1729)
    m, d, h = options.rows * options.seq_len, options.dim, options.intermediate
    if name == "projection":
        shapes, fn = [(m, d), (2 * d, d)], linear
    else:
        shapes, fn = [(m, d), (2 * h, d), (d, h)], swiglu
    values = [
        (torch.randn(shape, device="cuda") / (math.sqrt(shape[-1]) if i else 1.0))
        .to(torch.bfloat16)
        .requires_grad_()
        for i, shape in enumerate(shapes)
    ]
    reference_values = [value.detach().float().requires_grad_() for value in values]
    reference = fn(*reference_values)
    upstream = torch.randn_like(reference).to(torch.bfloat16)
    reference.backward(upstream.float())
    reference_grads = [value.grad.detach().clone() for value in reference_values]
    reference = reference.detach()
    del reference_values

    tuning = []

    def feedback(timings, operation, inputs, choices, *_):
        measured = [
            {
                "name": choice.name,
                "type": type(choice).__name__,
                "milliseconds": float(timings[choice]),
            }
            for choice in choices
            if choice in timings and math.isfinite(timings[choice])
        ]
        tuning.append(
            {
                "operation": operation,
                "candidate_count": len(choices),
                "nvgemm_candidates": sum(
                    "NVUniversalGemm" in type(choice).__name__ for choice in choices
                ),
                "fastest_measured": min(measured, key=lambda row: row["milliseconds"])
                if measured
                else None,
            }
        )

    add_feedback_saver(feedback)
    compiled = torch.compile(
        fn, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs"
    )

    def step():
        for value in values:
            if value.grad is not None:
                value.grad.zero_()
        result = compiled(*values)
        result.backward(upstream)
        return result

    started = time.perf_counter()
    try:
        # Disable graph caching so selection evidence is emitted even on reruns.
        with config.patch(
            max_autotune_gemm_backends=options.backends, force_disable_caches=True
        ):
            actual, sources = run_and_get_code(step)
            torch.cuda.synchronize()
    finally:
        clear_feedback_savers()
    compile_seconds = time.perf_counter() - started
    source_paths = []
    for index, source in enumerate(sources):
        path = options.output_dir / f"{name}-{index}.py"
        path.write_text(source)
        source_paths.append(str(path.resolve()))
    drift = {
        "output": discrepancy(actual, reference),
        "gradients": [
            discrepancy(value.grad, expected)
            for value, expected in zip(values, reference_grads)
        ],
    }
    if not all(row["finite"] for row in [drift["output"], *drift["gradients"]]):
        raise RuntimeError(f"{name}: nonfinite output or gradient: {drift}")
    del actual
    # Drift is reported without imposing a tight equivalence threshold.
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    eager_samples = []
    for _ in range(options.repeats):
        started = time.perf_counter()
        for _ in range(options.iterations):
            step()
        torch.cuda.synchronize()
        eager_samples.append(
            (time.perf_counter() - started) * 1000 / options.iterations
        )

    graph = torch.cuda.CUDAGraph()
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.graph(graph):
            step()
    finally:
        if was_enabled:
            gc.enable()
    graph_samples = []
    for _ in range(options.repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(options.iterations):
            graph.replay()
        end.record()
        end.synchronize()
        graph_samples.append(start.elapsed_time(end) / options.iterations)
    report = {
        "case": name,
        "backends": options.backends,
        "shapes": shapes,
        "compile_seconds": compile_seconds,
        "drift_vs_fp32": drift,
        "fwd_bwd_host_ms": statistics.median(eager_samples),
        "fwd_bwd_host_samples_ms": eager_samples,
        "fwd_bwd_graph_ms": statistics.median(graph_samples),
        "fwd_bwd_graph_samples_ms": graph_samples,
        "nvgemm_selected": any("nv_universal_gemm" in source for source in sources),
        "selection_evidence_available": bool(sources),
        "autotuning": tuning,
        "generated_code": source_paths,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    if not sources:
        raise RuntimeError(
            "No generated code captured; rerun with TORCHINDUCTOR_FORCE_DISABLE_CACHES=1"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", choices=("ATEN,TRITON", "ATEN,TRITON,NVGEMM"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case", choices=("projection", "swiglu", "all"), default="all"
    )
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--intermediate", type=int, default=3328)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    options = parser.parse_args()
    if (
        min(
            options.rows,
            options.seq_len,
            options.dim,
            options.intermediate,
            options.iterations,
            options.repeats,
        )
        <= 0
    ):
        parser.error("geometry, iterations, and repeats must be positive")
    if not torch.cuda.is_available():
        parser.error("an idle CUDA device is required")
    options.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("highest")
    runtime = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "packages": {
            name: package_version(name)
            for name in (
                "triton",
                "nvidia-cutlass-operators",
                "nvidia-cutlass-dsl",
                "nvidia-cutlass-dsl-libs-cu13",
            )
        },
    }
    print(json.dumps({"runtime": runtime}), flush=True)
    if "NVGEMM" in options.backends:
        m, d, h = options.rows * options.seq_len, options.dim, options.intermediate
        inventory = operator_inventory([(m, 2 * d, d), (m, 2 * h, d), (m, d, h)])
        print(json.dumps({"operator_inventory": inventory}), flush=True)
        (options.output_dir / "inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n"
        )
    reports = []
    for name in ("projection", "swiglu") if options.case == "all" else (options.case,):
        torch.cuda.reset_peak_memory_stats()
        report = benchmark_case(name, options)
        reports.append(report)
        print(json.dumps(report), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    (options.output_dir / "report.json").write_text(
        json.dumps({"runtime": runtime, "cases": reports}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

"""Compare a small set of PKDA launch configurations on projected BF16 inputs.

    python scripts/pkda_bench.py --heads 8,12,16 --rows 2 --candidates baseline,state16
    python scripts/pkda_bench.py --heads 12,16 --rows 2,4 --candidates baseline,intra64,wy128
    python scripts/pkda_bench.py --heads 8 --rows 16 --candidates baseline,state16+intra64 --trace

Uses the production custom operator, its gate activation, and all ten input
gradients. Q/K arrive normalized just as they do after the fused convolution.
Both retaining and recomputing backward are available. Candidate launch changes
exist only in this process; no fork source or production configuration is edited.
Each candidate must agree with the current operator before its timing is saved.
Timings cover captured forward/backward; projections, convolution, parameter
gradient sinks, optimizer, and the rest of the model require the replay benchmark.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

CANDIDATES = {
    "baseline": "Current fork launch choices and autotuning.",
    "intra64": "Intra backward BK=64, BC=16, 4 warps, 2 stages; halves K partitions.",
    "wy128": "WY forward/recompute BK=BV=128, 4 warps, 2 stages; removes tile loops.",
    "state16": "Forward/backward state BV=16, 2 warps, 2 stages; increases CTA count.",
    "state32": "Forward/backward state BV=32, 2 warps, 2 stages; fixed narrow baseline.",
    "scan128": "Gate forward scan BS=128, 4 warps; one K tile per chunk/head.",
    "atk-scan128": "ATK inter-chunk forward scan BK=128, 4 warps; one sweep through the chunks.",
    "intra:BK=../BC=../DIAG=../W=../S=..": "One explicit intra backward launch; DIAG is the diagonal column group.",
    "wy:BK=../BV=../W=../S=..": "One explicit WY plus inter backward launch.",
}
INPUT_NAMES = (
    "q", "k", "v", "g", "g_atk", "beta_atk", "beta", "A_log", "dt_bias", "log_atk_scale",
)
# Existing FLA test_chunk tolerances; retunes should normally be much closer.
RMS_LIMITS = dict(zip(INPUT_NAMES, (0.007, 0.008, 0.007, 0.015, 0.015, 0.015, 0.015, 0.025, 0.02, 0.02)))


class FixedLaunch:
    """Retain Triton heuristics while replacing autotuning with one launch."""

    def __init__(self, kernel, **meta):
        import triton
        from triton.runtime.autotuner import Autotuner, Heuristics

        def strip_autotune(value):
            if isinstance(value, Heuristics):
                return triton.heuristics(value.values)(strip_autotune(value.fn))
            if isinstance(value, Autotuner):
                return strip_autotune(value.fn)
            return value

        self.kernel = strip_autotune(kernel)
        self.meta = meta

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            args = list(args)
            meta = self.meta.copy()
            # ATK passes its tile width positionally; the PKDA kernels name it.
            for index, name in enumerate(self.kernel.arg_names[:len(args)]):
                if name in meta:
                    args[index] = meta.pop(name)
            return self.kernel[grid](*args, **(kwargs | meta))
        return launch


@contextmanager
def candidate_launches(name):
    changes = []

    def replace(module_name, attribute, value):
        module = importlib.import_module(module_name)
        changes.append((module, attribute, getattr(module, attribute)))
        setattr(module, attribute, value)

    def fixed(module_name, attribute, **meta):
        module = importlib.import_module(module_name)
        replace(module_name, attribute, FixedLaunch(getattr(module, attribute), **meta))

    try:
        for part in name.split("+"):
            if part == "baseline":
                continue
            if part.startswith("intra:"):
                # intra:BK=64/BC=16/DIAG=8/W=4/S=2 - one explicit intra backward launch.
                spec = dict(item.split("=") for item in part.removeprefix("intra:").split("/"))
                module = "fla.ops.precond_kda.chunk_intra"
                replace(module, "BWD_INTRA_BK", int(spec["BK"]))
                fixed(module, "chunk_precond_kda_bwd_kernel_intra",
                      BC=int(spec["BC"]), DIAG_J=int(spec["DIAG"]),
                      num_warps=int(spec["W"]), num_stages=int(spec["S"]))
            elif part.startswith("wy:"):
                # wy:BK=64/BV=128/W=4/S=2 - one explicit WY+inter backward launch.
                spec = dict(item.split("=") for item in part.removeprefix("wy:").split("/"))
                fixed("fla.ops.precond_kda.chunk_bwd", "chunk_precond_kda_bwd_kernel_wy_dqkg",
                      BK=int(spec["BK"]), BV=int(spec["BV"]),
                      num_warps=int(spec["W"]), num_stages=int(spec["S"]))
            elif part == "intra64":
                module = "fla.ops.precond_kda.chunk_intra"
                replace(module, "BWD_INTRA_BK", 64)
                fixed(module, "chunk_precond_kda_bwd_kernel_intra", BC=16, num_warps=4, num_stages=2)
            elif part == "wy128":
                fixed("fla.ops.precond_kda.wy_fast", "recompute_w_u_fwd_kernel",
                      BK=128, BV=128, num_warps=4, num_stages=2)
            elif part in {"state16", "state32"}:
                for kernel in ("chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
                               "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64"):
                    fixed("fla.ops.common.chunk_delta_h", kernel,
                          BV=int(part.removeprefix("state")), num_warps=2, num_stages=2)
            elif part == "scan128":
                fixed("fla.ops.kda.gate", "kda_gate_chunk_cumsum_vector_kernel", BS=128, num_warps=4)
            elif part == "atk-scan128":
                fixed("fla.ops.atk.chunk_atk_fwd", "_forward_pass_chunks", BK=128, num_warps=4)
            else:
                raise ValueError(f"Unknown candidate {part!r}")
        yield
    finally:
        for module, attribute, original in reversed(changes):
            setattr(module, attribute, original)


def checkout(path):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    try:
        diff = subprocess.check_output(
            ["git", "-C", str(path), "diff", "--binary", "HEAD"], stderr=subprocess.DEVNULL
        )
        return {"head": git("rev-parse", "HEAD"), "status": git("status", "--short"),
                "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        # Benchmarking from a synchronized copy that carries no repository.
        return {"head": None, "status": None, "tracked_diff_sha256": None, "path": str(path)}


def make_inputs(rows, length, heads, seed):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    shape = (rows, length, heads, 128)
    scalar_shape = shape[:-1]

    def randn(size):
        return torch.randn(size, device="cuda", dtype=torch.float32)

    q = F.normalize(F.silu(randn(shape)), dim=-1).bfloat16()
    k = F.normalize(F.silu(randn(shape)), dim=-1).bfloat16()
    v = F.silu(randn(shape)).bfloat16()
    g = (randn(shape) * 0.2).bfloat16()
    beta = torch.sigmoid(randn(scalar_shape) * 0.2).bfloat16()
    beta_atk = torch.sigmoid(randn(scalar_shape) * 0.2).bfloat16()
    A_log = torch.empty(heads, device="cuda").uniform_(1, 16).log()
    dt_bias = torch.zeros(heads * 128, device="cuda")
    precond_A = torch.empty(heads, device="cuda").uniform_(1, 16)
    precond_dt = torch.empty(heads, device="cuda").uniform_(math.log(0.001), math.log(0.1)).exp()
    precond_bias = precond_dt + torch.log(-torch.expm1(-precond_dt))
    g_atk = -precond_A * F.softplus(randn(scalar_shape) * 0.2 + precond_bias)
    log_atk_scale = torch.full((heads,), -0.2, device="cuda")
    inputs = (q, k, v, g, g_atk, beta_atk, beta, A_log, dt_bias, log_atk_scale)
    return tuple(x.requires_grad_() for x in inputs), randn(shape).bfloat16()


def forward_backward(inputs, upstream, lean):
    import torch

    from delta_feedback_experiment.fla_ops import pkda_recurrence

    output, _saved = pkda_recurrence(*inputs, 128**-0.5, 1.5, 1e-6, lean)
    gradients = torch.autograd.grad(output, inputs, upstream)
    # Projected-activation gradients remain BF16. Learned recurrence vectors and
    # preconditioner gates retain FP32; autograd.grad never accumulates into BF16 leaves.
    for index in (4, 7, 8, 9):
        if gradients[index].dtype != torch.float32:
            raise AssertionError(f"{INPUT_NAMES[index]} gradient lost FP32")
    return (output, *gradients)


def compare(reference, actual):
    import torch

    errors = {}
    for name, ref, value in zip(("output", *INPUT_NAMES), reference, actual, strict=True):
        ref = ref.float()
        value = value.detach().float()
        if not torch.isfinite(value).all().item() or not torch.isfinite(ref).all().item():
            raise AssertionError(f"Nonfinite {name}")
        difference = value - ref
        maximum = difference.abs().max().item()
        ratio = (difference.square().mean().sqrt() / (ref.square().mean().sqrt() + 1e-8)).item()
        limit = 0.005 if name == "output" else RMS_LIMITS[name]
        errors[name] = {"max_abs": maximum, "relative_rms": ratio, "limit": limit}
        if maximum > 1e-6 and ratio >= limit:
            raise AssertionError(f"{name}: relative RMS {ratio:.6g} >= {limit}; max abs {maximum:.6g}")
    return errors


def accumulation_check(inputs, upstream, lean, reference):
    """Two independent backwards must accumulate through FP32 buffers."""
    import torch

    first = forward_backward(inputs, upstream, lean)
    sinks = [gradient.detach().float().clone() for gradient in first[1:]]
    del first
    second = forward_backward(inputs, upstream, lean)
    for sink, gradient in zip(sinks, second[1:], strict=True):
        sink.add_(gradient)
        assert sink.dtype == torch.float32
    result = compare(reference, (second[0], *(sink * 0.5 for sink in sinks)))
    return result


def captured_run(inputs, upstream, lean, warm, repeat, trace):
    import torch

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warm):
            forward_backward(inputs, upstream, lean)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    gc.collect()
    graph = torch.cuda.CUDAGraph()
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.graph(graph, stream=stream):
            outputs = forward_backward(inputs, upstream, lean)
    finally:
        if gc_enabled:
            gc.enable()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    result = {"median_ms": statistics.median(samples), "min_ms": min(samples), "samples_ms": samples}
    if trace:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            graph.replay()
            torch.cuda.synchronize()
        by_name = {}
        for event in prof.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                entry = by_name.setdefault(event.name, {"count": 0, "ms": 0.0})
                entry["count"] += 1
                entry["ms"] += event.device_time_total / 1000
        result["top_kernels"] = [
            {"name": name, **entry}
            for name, entry in sorted(by_name.items(), key=lambda item: -item[1]["ms"])[:20]
        ]
    return result, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heads", default="8,12,16", help="screen, bridge, flagship head counts")
    parser.add_argument("--rows", default="2", help="comma-separated row widths; representative 2,4; wide screen 16")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--candidates", default="baseline,state16", help="comma-separated candidates; combine using +")
    parser.add_argument("--saving", choices=("lean", "full", "both"), default="full")
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--describe", action="store_true", help="list candidates without importing CUDA dependencies")
    parser.add_argument("--out", type=Path, default=Path("logs/pkda-bench/latest.json"))
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(CANDIDATES, indent=2))
        return
    heads, rows = ([int(value) for value in text.split(",")] for text in (args.heads, args.rows))
    candidates = list(dict.fromkeys(["baseline", *args.candidates.split(",")]))
    for candidate in candidates:
        parts = [part for part in candidate.split("+") if not part.startswith(("intra:", "wy:"))]
        if any(part not in CANDIDATES for part in parts) or {"state16", "state32"} <= set(parts):
            parser.error(f"Invalid candidate {candidate!r}; see --describe")
    if min(*heads, *rows, args.length, args.warm, args.repeat) < 1 or args.length % 64:
        parser.error("heads, rows, warm, repeat must be positive; length must be a positive multiple of 64")

    import torch
    import triton
    if not torch.cuda.is_available():
        parser.error("This benchmark requires CUDA and the declared CUDA extras")
    torch.set_float32_matmul_precision("high")
    torch.cuda.init()
    root = Path(__file__).resolve().parents[1]
    result = {
        "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__, "triton": triton.__version__,
        "experiment": checkout(root), "fla": checkout(root.parent / "vendor/flash-linear-attention"),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": vars(args) | {"out": str(args.out)}, "results": [],
        "scope": "Synthetic projected inputs; production PKDA custom op, all ten gradients; no full-model speed claim.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    modes = (False, True) if args.saving == "both" else (args.saving == "lean",)
    for head_count in heads:
        for row_count in rows:
            inputs, upstream = make_inputs(row_count, args.length, head_count, args.seed)
            for lean in modes:
                reference = None
                baseline_ms = None
                for candidate in candidates:
                    label = f"H={head_count} rows={row_count} {'lean' if lean else 'full'} {candidate}"
                    print(f"Compiling/checking {label}", flush=True)
                    start = time.perf_counter()
                    with candidate_launches(candidate):
                        actual = forward_backward(inputs, upstream, lean)
                        if reference is None:
                            reference = tuple(value.detach().clone() for value in actual)
                        errors = compare(reference, actual)
                        del actual
                        accumulation = accumulation_check(inputs, upstream, lean, reference)
                        timing, captured = captured_run(inputs, upstream, lean, args.warm, args.repeat, args.trace)
                        capture_errors = compare(reference, captured)
                        del captured
                    baseline_ms = baseline_ms or timing["median_ms"]
                    entry = {
                        "heads": head_count, "rows": row_count, "length": args.length,
                        "saving": "lean" if lean else "full", "candidate": candidate,
                        **timing, "speedup": baseline_ms / timing["median_ms"],
                        "ms_per_row": timing["median_ms"] / row_count,
                        "errors": errors, "capture_errors": capture_errors,
                        "fp32_accumulation_errors": accumulation,
                        "wall_s": time.perf_counter() - start,
                    }
                    result["results"].append(entry)
                    args.out.write_text(json.dumps(result, indent=2) + "\n")
                    print(f"{label}: {timing['median_ms']:.3f} ms, {entry['speedup']:.3f}x baseline", flush=True)
                    gc.collect()
                del reference
            del inputs, upstream
            gc.collect()
            torch.cuda.empty_cache()
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

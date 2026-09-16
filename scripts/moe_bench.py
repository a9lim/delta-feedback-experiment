"""Small grouped-expert tile sweep at current model geometries, on an idle GPU.

    python scripts/moe_bench.py --rows 2 --routing balanced skewed
    python scripts/moe_bench.py --scale bridge flagship --rows 2 --candidates dw128 k128
    python scripts/moe_bench.py --rows 4 16 --candidates baseline k128 dw128
    python scripts/moe_bench.py --check

Times the complete routed expert forward/backward (all six GEMMs, SwiGLU,
and input-gradient combination) in a CUDA graph. Input/gradient FP8 copies,
weight copies, routing, and sink zeroing are prepared outside the timed region.
The FP32 weight-gradient sinks are persistent and reset before every replay.
Each candidate must match the current production tiles on output, input
gradient, and both weight gradients before it receives a timing. This checks
tile equivalence, not the FP8 recipe against a higher-precision oracle.

Tile overrides exist only in this process. FP8 DACT.BN stays fixed because
it defines the gradient quantization recipe; DX.BK must divide that width.
Default sweep: four configurations, two routing distributions, one width.
Results are incremental JSON under logs/moe-bench/; no training state changes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

SCALE_NAMES = ("screen", "bridge", "flagship")
GEOMETRY_KEYS = (
    "dim",
    "expert_intermediate",
    "num_routed_experts",
    "experts_per_token",
    "seq_len",
)
TILE_NAMES = (
    "TILE_GATE_UP",
    "TILE_DOWN",
    "TILE_DACT",
    "TILE_DX",
    "TILE_DW_GATE",
    "TILE_DW_DOWN",
    "TILE_GATE_UP_FP8",
    "TILE_DOWN_FP8",
    "TILE_DACT_FP8",
    "TILE_DX_FP8",
)
# Few hypotheses, not a Cartesian sweep. No change to scale granularity.
CANDIDATES = {
    "baseline": {},
    "k128": {
        "TILE_GATE_UP": (64, 128, 64, 4, 3),
        "TILE_DOWN": (64, 64, 64, 4, 3),
        "TILE_DACT": (64, 64, 64, 4, 3),
        "TILE_DX": (128, 64, 64, 4, 3),
        "TILE_GATE_UP_FP8": (64, 128, 128, 4, 3),
        "TILE_DOWN_FP8": (64, 64, 128, 4, 3),
        "TILE_DACT_FP8": (64, 64, 128, 4, 3),
        "TILE_DW_GATE": (64, 64, 128, 4, 3),
        "TILE_DW_DOWN": (64, 64, 128, 4, 3),
    },
    "dw128": {
        "TILE_DW_GATE": (64, 128, 128, 4, 3),
        "TILE_DW_DOWN": (64, 128, 128, 4, 3),
    },
    "wide128": {
        "TILE_GATE_UP": (128, 128, 32, 8, 3),
        "TILE_DOWN": (128, 128, 32, 8, 3),
        "TILE_DACT": (128, 64, 32, 4, 3),
        "TILE_DX": (128, 128, 32, 8, 3),
        "TILE_GATE_UP_FP8": (128, 128, 64, 8, 3),
        "TILE_DOWN_FP8": (128, 128, 64, 8, 3),
        "TILE_DACT_FP8": (128, 64, 64, 4, 3),
        "TILE_DX_FP8": (128, 128, 64, 8, 3),
        "TILE_DW_GATE": (64, 128, 64, 4, 3),
        "TILE_DW_DOWN": (64, 128, 64, 4, 3),
    },
}


def routes(torch, tokens, kind, experts, selected):
    """Stable expert-major ordering; no duplicated expert within a token.

    Skew sends 75% of tokens to the first selected experts; the rest spread evenly.
    The hot experts receive about four times their balanced load.
    """
    ids = (torch.arange(tokens)[:, None] + torch.arange(selected)) % experts
    if kind == "skewed":
        ids[torch.arange(tokens) % 4 != 0] = torch.arange(selected)
    assignments = ids.flatten().argsort(stable=True)
    inverse = assignments.argsort()
    counts = torch.bincount(ids.flatten(), minlength=experts)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    return assignments, inverse, offsets, counts


@contextmanager
def tiles(module, baseline, candidate):
    chosen = baseline | CANDIDATES[candidate]
    if chosen["TILE_DACT_FP8"][1] != baseline["TILE_DACT_FP8"][1]:
        raise ValueError("DACT.BN changes the FP8 recipe")
    if chosen["TILE_DACT_FP8"][1] % chosen["TILE_DX_FP8"][2]:
        raise ValueError("DX.BK must divide DACT.BN")
    previous = {key: getattr(module, key) for key in TILE_NAMES}
    try:
        for key, value in chosen.items():
            setattr(module, key, value)
        yield chosen
    finally:
        for key, value in previous.items():
            setattr(module, key, value)


def make_case(torch, kernels, geometry, tokens, routing, precision, seed):
    from delta_feedback_experiment.cuda_kernels import Fp8Weights, quantize_rows

    dim, width, experts, selected = geometry[:4]
    torch.manual_seed(seed)
    assignments, inverse, offsets, counts = routes(
        torch, tokens, routing, experts, selected
    )
    assignments, inverse, offsets = (t.cuda() for t in (assignments, inverse, offsets))
    x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
    # Current NorMuonH matrices initialize with fan-in standard deviation.
    gate = (
        torch.randn(experts, 2 * width, dim, device="cuda", dtype=torch.bfloat16)
        * dim**-0.5
    )
    down = (
        torch.randn(experts, dim, width, device="cuda", dtype=torch.bfloat16)
        * width**-0.5
    )
    # A token's incoming gradient is shared across its selected experts and
    # weighted by selected * normalized_route_weight / sqrt(selected + 1).
    incoming = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16) * 0.02
    weights = torch.rand(tokens, selected, device="cuda") + 0.25
    weights /= weights.sum(-1, keepdim=True)
    gradient = (
        incoming[assignments // selected].float()
        * ((selected / (selected + 1) ** 0.5) * weights.flatten()[assignments, None])
    ).to(torch.bfloat16)
    del incoming, weights
    gate_sink = torch.zeros_like(gate, dtype=torch.float32)
    down_sink = torch.zeros_like(down, dtype=torch.float32)
    gate_sinks, down_sinks = list(gate_sink.unbind()), list(down_sink.unbind())
    flags = [True] * experts
    sink_args = (gate_sinks, down_sinks, flags, flags, flags, flags)

    if precision == "fp8":
        # Exact production quantization, outside timing; no compiler autotune
        # for this setup-only operation.
        def copies(weight):
            q, scale = quantize_rows(weight)
            qt, st = quantize_rows(weight.transpose(-1, -2).contiguous())
            return Fp8Weights(q.contiguous(), scale, qt.contiguous(), st)

        gate_fp8, down_fp8 = copies(gate), copies(down)
        xq, xs = quantize_rows(x)
        gq, gs = quantize_rows(gradient)

        def run():
            output, gate_up = kernels._forward_fp8(
                xq,
                xs,
                gate_fp8.weight,
                gate_fp8.scale,
                down_fp8.weight,
                down_fp8.scale,
                assignments,
                offsets,
            )
            dx, fresh = kernels._backward_fp8(
                gradient,
                gq,
                gs,
                x,
                gate_up,
                gate_fp8.transposed,
                gate_fp8.transposed_scale,
                down_fp8.transposed,
                down_fp8.transposed_scale,
                assignments,
                inverse,
                offsets,
                *sink_args,
            )
            assert not fresh
            return output, dx
    else:

        def run():
            output, gate_up = kernels._forward(x, gate, down, assignments, offsets)
            dx, fresh = kernels._backward(
                gradient,
                x,
                gate_up,
                gate,
                down,
                assignments,
                inverse,
                offsets,
                *sink_args,
            )
            assert not fresh
            return output, dx

    def reset():
        gate_sink.zero_()
        down_sink.zero_()

    return run, reset, (gate_sink, down_sink), counts.tolist()


def compare(torch, actual, expected):
    """Chunked errors avoid large FP32 temporaries at the 16-row shape."""
    stats = {}
    for name, got, want in zip(
        ("output", "dx", "dgate", "ddown"), actual, expected, strict=True
    ):
        sums = torch.zeros(2, device="cuda", dtype=torch.float64)
        maxima = torch.zeros(2, device="cuda")
        finite = torch.ones((), device="cuda", dtype=torch.bool)
        for a, b in zip(
            got.flatten().split(1 << 20), want.flatten().split(1 << 20), strict=True
        ):
            a, b = a.float(), b.float()
            error = a - b
            sums += torch.stack((error.square().sum(), b.square().sum())).double()
            maxima = torch.maximum(
                maxima, torch.stack((error.abs().max(), b.abs().max()))
            )
            finite &= torch.isfinite(a).all()
        values, max_values = sums.tolist(), maxima.tolist()
        rel_l2 = (values[0] / max(values[1], 1e-30)) ** 0.5
        rel_max = max_values[0] / max(max_values[1], 1e-30)
        # Tile regrouping can change rounding before FP8 casts. Check both
        # energy and outliers; this is a tile-equivalence tolerance, not an
        # allowance for a different scale recipe.
        passed = bool(finite.item()) and rel_l2 <= 0.01 and rel_max <= 0.05
        stats[name] = {
            "relative_l2": rel_l2,
            "max_error_over_max_reference": rel_max,
            "max_abs_error": max_values[0],
            "passed": passed,
        }
    return stats


def measure(torch, run, reset, sinks, warm, repeat, reference):
    # Compile all Triton kernels before graph capture, on a side stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warm):
            reset()
            run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    reset()
    stream.wait_stream(torch.cuda.current_stream())
    # Match the training capture contract: collect cyclic garbage outside it.
    gc.collect()
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.graph(graph, stream=stream):
            output, dx = run()
    finally:
        if gc_enabled:
            gc.enable()
    torch.cuda.current_stream().wait_stream(stream)
    reset()
    graph.replay()
    torch.cuda.synchronize()
    actual = (output, dx, *sinks)
    errors = compare(torch, actual, reference if reference is not None else actual)
    if not all(value["passed"] for value in errors.values()):
        if reference is None:
            raise RuntimeError(
                f"production baseline produced non-finite values: {errors}"
            )
        return {"status": "numerical_mismatch", "errors": errors}, None
    saved = tuple(t.clone() for t in actual) if reference is None else None
    samples = []
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for _ in range(repeat):
        reset()  # Same sink addresses; zeroing excluded from timing.
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "status": "ok",
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "samples_ms": samples,
        "errors": errors,
    }, saved


def check_cpu(torch, kernels, baseline, geometries):
    for _, _, experts, selected, seq_len in geometries.values():
        for rows in (2, 4, 16):
            tokens = rows * seq_len
            for kind in ("balanced", "skewed"):
                a, inverse, offsets, counts = routes(
                    torch, tokens, kind, experts, selected
                )
                assert torch.equal(a[inverse], torch.arange(tokens * selected))
                assert (
                    int(offsets[-1]) == tokens * selected
                    and int(counts.max()) <= tokens
                )
    for name in CANDIDATES:
        with tiles(kernels, baseline, name) as chosen:
            for shape in chosen.values():
                assert len(shape) == 5 and all(v > 0 for v in shape)
                assert all(v & (v - 1) == 0 for v in shape[:4])
    # Run the registered fake implementations to check current call shapes.
    dim, width, experts, selected, _ = geometries["screen"]
    x = torch.empty(4, dim, dtype=torch.bfloat16)
    gate = torch.empty(experts, 2 * width, dim, dtype=torch.bfloat16)
    down = torch.empty(experts, dim, width, dtype=torch.bfloat16)
    a, inverse, offsets, _ = routes(torch, 4, "skewed", experts, selected)
    flags = [True] * experts
    output, preact = kernels._forward_fake(x, gate, down, a, offsets)
    dx, fresh = kernels._backward_fake(
        output,
        x,
        preact,
        gate,
        down,
        a,
        inverse,
        offsets,
        [],
        [],
        flags,
        flags,
        flags,
        flags,
    )
    assert output.shape == (12, dim) and dx.shape == x.shape and not fresh
    output, preact = kernels._forward_fp8_fake(
        x, x[:, :1], gate, gate[..., :1], down, down[..., :1], a, offsets
    )
    dx, fresh = kernels._backward_fp8_fake(
        output,
        output,
        output[:, :1],
        x,
        preact,
        gate.transpose(-1, -2),
        gate[..., :1],
        down.transpose(-1, -2),
        down[..., :1],
        a,
        inverse,
        offsets,
        [],
        [],
        flags,
        flags,
        flags,
        flags,
    )
    assert output.shape == (12, dim) and dx.shape == x.shape and not fresh
    print(
        "CPU checks passed: routing, tile constraints, BF16/FP8 call shapes. CUDA execution untested."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scale", nargs="+", choices=SCALE_NAMES, default=["screen"])
    parser.add_argument("--rows", nargs="+", type=int, choices=(1, 2, 4, 8, 16), default=[2])
    parser.add_argument(
        "--routing",
        nargs="+",
        choices=("balanced", "skewed"),
        default=["balanced", "skewed"],
    )
    parser.add_argument(
        "--candidates", nargs="+", choices=tuple(CANDIDATES), default=list(CANDIDATES)
    )
    parser.add_argument("--precision", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--out", type=Path, default=Path("logs/moe-bench/tiles.json"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="CPU-only routing, tile and fake-interface checks",
    )
    args = parser.parse_args()
    if args.warm < 1 or args.repeat < 1:
        parser.error("--warm and --repeat must be positive")

    import torch

    from delta_feedback_experiment import moe_kernels as kernels
    from delta_feedback_experiment.train import SCALES

    geometries = {
        name: tuple(SCALES[name][key] for key in GEOMETRY_KEYS) for name in SCALE_NAMES
    }

    baseline = {key: getattr(kernels, key) for key in TILE_NAMES}
    if args.check:
        check_cpu(torch, kernels, baseline, geometries)
        return
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required; use --check for portable validation")
    if args.precision == "fp8" and torch.cuda.get_device_capability() < (8, 9):
        parser.error("FP8 kernels require compute capability 8.9 or newer")
    candidates = list(dict.fromkeys(["baseline", *args.candidates]))
    source = Path(kernels.__file__)
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "capability": torch.cuda.get_device_capability(),
        "precision": args.precision,
        "geometry_presets": {
            name: dict(zip(GEOMETRY_KEYS, shape, strict=True))
            for name, shape in geometries.items()
        },
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "seed": args.seed,
        "cases": [],
        "timing_scope": "CUDA graph of six GEMMs plus SwiGLU/combine; excludes dispatch, external FP8 copies, sink zeroing",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for scale in args.scale:
        seq_len = geometries[scale][-1]
        for rows in args.rows:
            for routing in args.routing:
                run, reset, sinks, counts = make_case(
                    torch,
                    kernels,
                    geometries[scale],
                    rows * seq_len,
                    routing,
                    args.precision,
                    args.seed,
                )
                reference, base_ms = None, None
                for candidate in candidates:
                    print(
                        f"scale={scale} rows={rows} routing={routing} candidate={candidate}: compile/check/time",
                        flush=True,
                    )
                    started = time.monotonic()
                    with tiles(kernels, baseline, candidate) as chosen:
                        measured, saved = measure(
                            torch, run, reset, sinks, args.warm, args.repeat, reference
                        )
                    if candidate == "baseline":
                        reference, base_ms = saved, measured["median_ms"]
                    if measured["status"] == "ok":
                        measured["speedup"] = base_ms / measured["median_ms"]
                    entry = {
                        "scale": scale,
                        "rows": rows,
                        "tokens": rows * seq_len,
                        "routing": routing,
                        "expert_counts": counts,
                        "candidate": candidate,
                        "tiles": chosen,
                        "wall_seconds": time.monotonic() - started,
                        **measured,
                    }
                    result["cases"].append(entry)
                    args.out.write_text(json.dumps(result, indent=2) + "\n")
                    print(
                        json.dumps(
                            {
                                key: entry[key]
                                for key in (
                                    "scale",
                                    "rows",
                                    "routing",
                                    "candidate",
                                    "status",
                                    "median_ms",
                                    "speedup",
                                )
                                if key in entry
                            }
                        ),
                        flush=True,
                    )
                del run, reset, sinks, reference, saved
                gc.collect()
                torch.cuda.empty_cache()
    print(f"Results: {args.out}")


if __name__ == "__main__":
    main()

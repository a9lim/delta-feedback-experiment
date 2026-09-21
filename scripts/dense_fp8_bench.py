"""Small Hopper dense-site FP8 pilot, with no production recipe changes.

    python scripts/dense_fp8_bench.py --scales flagship,bridge --rows 2
    python scripts/dense_fp8_bench.py --scales screen --sites L3.qkv --rows 4
    python scripts/dense_fp8_bench.py --list-shapes

Default: one packed PKDA QKV site at each scale, flagship first. Compare the
current rowwise recipe with activation 1x128 / weight 128x128 and 1x128 / 1x128
scaling. Both forward and dX include activation/gradient quantization; weight
and transposed-weight refresh is separate because production pays once per
optimizer step. The BF16-to-FP32 dW accumulation is unchanged and timed alone
and together with the pair. Only quantizers compile, in default mode.

The block scale shapes match the installed PyTorch meta rules. Their storage
layouts follow PyTorch's ScaledBlas.cpp: 1x128 scales are column-major before
the RHS transpose; 128x128 scales are row-major. CUDA support remains a live
runtime check, reported per candidate. This is a pilot on synthetic inputs,
not a trained-model numerical qualification or a tuned quantization kernel.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import fields
from pathlib import Path

from cce_bench import elapsed, error_metrics, revision

RECIPES = ("rowwise", "block128", "block1")
SCALE_SOURCE = "https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ScaledBlas.cpp"


def shapes(scales: list[str], sites: list[str]) -> list[dict]:
    """Read actual packed geometries without allocating model storage."""
    import torch

    from delta_feedback_experiment.model import (
        DeltaModel,
        ModelConfig,
        condition_config,
    )
    from delta_feedback_experiment.train import SCALES

    keys = {field.name for field in fields(ModelConfig)}
    result = []
    for scale in scales:
        with torch.device("meta"):
            model = DeltaModel(
                condition_config(
                    "fv", **{k: v for k, v in SCALES[scale].items() if k in keys}
                )
            )
        specs = {spec.name: spec for spec in model.slab_specs() if spec.kind == "rows"}
        for site in sites:
            spec = specs[site]
            result.append(
                {
                    "scale": scale,
                    "site": site,
                    "dim": model.cfg.dim,
                    "n": sum(member.shape[0] for member in spec.members),
                    "k": spec.members[0].shape[1],
                    "members": [list(member.shape) for member in spec.members],
                }
            )
    return result


def block1(x):
    """Per-row groups of 128; emit column-major [rows, K/128] scales."""
    import torch

    rows, cols = x.shape
    blocks = x.reshape(rows, cols // 128, 128).float()
    scale = (blocks.abs().amax(-1) / 448.0).clamp_min(2.0**-64)
    quantized = (blocks / scale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized.reshape(rows, cols).contiguous(), scale.T.contiguous().T


def block128(x):
    """Independent 128x128 tiles; emit row-major [rows/128, K/128] scales."""
    import torch

    rows, cols = x.shape
    blocks = x.reshape(rows // 128, 128, cols // 128, 128).float()
    scale = (blocks.abs().amax((1, 3)) / 448.0).clamp_min(2.0**-64)
    quantized = (
        (blocks / scale[:, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    )
    return quantized.reshape(rows, cols).contiguous(), scale.contiguous()


def capture(fn, reset=lambda: None):
    import torch

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            reset()
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    reset()
    graph = torch.cuda.CUDAGraph()
    enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.graph(graph):
            output = fn()
    finally:
        if enabled:
            gc.enable()
    return graph, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", default="flagship,bridge,screen")
    parser.add_argument("--sites", default="L0.qkv")
    parser.add_argument("--recipes", default=",".join(RECIPES))
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--list-shapes", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path("logs/dense-fp8-bench/latest.json")
    )
    args = parser.parse_args()
    recipes = args.recipes.split(",")
    if set(recipes) - set(RECIPES):
        parser.error(f"unknown recipes: {set(recipes) - set(RECIPES)}")
    if min(args.rows, args.seq_len, args.repeat) < 1:
        parser.error("rows, sequence length and repeat must be positive")
    geometries = shapes(args.scales.split(","), args.sites.split(","))
    if args.list_shapes:
        print(json.dumps(geometries, indent=2))
        return

    import torch

    from delta_feedback_experiment.cuda_kernels import (
        dw_accum,
        fp8_linear,
        quantize_rows,
    )

    if not torch.cuda.is_available():
        parser.error("CUDA is required; --help and --list-shapes work without it")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    torch.set_grad_enabled(False)
    result = {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "args": {**vars(args), "out": str(args.out)},
        "revision": revision(Path(__file__).resolve().parents[1]),
        "scale_layout_source": SCALE_SOURCE,
        "scope": "One projection fwd+dX and unchanged BF16/FP32 dW; includes per-call input quantization and reports per-step two-layout weight refresh separately. Synthetic normalized inputs.",
        "results": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_text(json.dumps(result, indent=2) + "\n")

    def timing(fn, reset=lambda: None, *, expected_outputs=None, buffers=None):
        graph, output = capture(fn, reset)
        measurement = elapsed(graph.replay, reset, args.warm, args.repeat)
        parity = {}
        if expected_outputs:
            outputs = output if isinstance(output, tuple) else (output,)
            for actual, (name, expected) in zip(
                outputs, expected_outputs.items(), strict=True
            ):
                parity[name] = error_metrics(actual, expected)
        for name, (actual, expected) in (buffers or {}).items():
            parity[name] = error_metrics(actual, expected)
        measurement["capture_parity"] = parity
        measurement["capture_pass"] = all(
            error["finite"] and error["relative_l2"] < 0.005
            for error in parity.values()
        )
        del graph, output
        return measurement

    for geometry in geometries:
        m, n, k = args.rows * args.seq_len, geometry["n"], geometry["k"]
        if n % 128 or k % 128:
            raise ValueError(f"pilot requires dimensions divisible by 128: {geometry}")
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        weight = (torch.randn(n, k, device="cuda") / k**0.5).to(torch.bfloat16)
        dy = (torch.randn(m, n, device="cuda") / n**0.5).to(torch.bfloat16)
        sink = torch.zeros(n, k, device="cuda", dtype=torch.float32)
        ref_y, ref_dx = x @ weight.T, dy @ weight
        ref_dw = torch.mm(dy.T, x, out_dtype=torch.float32)
        dw_accum(dy, x, sink)
        dw_error = error_metrics(sink, ref_dw)
        dw_accum(dy, x, sink)
        accum_error = error_metrics(sink, 2 * ref_dw)
        if any(
            not error["finite"] or error["relative_l2"] > 0.01
            for error in (dw_error, accum_error)
        ):
            raise RuntimeError(
                f"FP32 dW accumulation failed: {dw_error}, {accum_error}"
            )

        def bf16_with_dw(x=x, weight=weight, dy=dy, sink=sink):
            y, dx = x @ weight.T, dy @ weight
            fence = dw_accum(dy, x, sink)
            return y, dx + fence.to(dx.dtype)

        base = {
            **geometry,
            "m": m,
            "recipe": "bf16",
            "status": "ok",
            "pair": timing(
                lambda x=x, weight=weight, dy=dy: (x @ weight.T, dy @ weight),
                expected_outputs={"forward": ref_y, "dx": ref_dx},
            ),
            "dw": timing(
                lambda dy=dy, x=x, sink=sink: dw_accum(dy, x, sink),
                sink.zero_,
                buffers={"dw_sink": (sink, ref_dw)},
            ),
            "dw_error": dw_error,
            "dw_accumulation_error": accum_error,
            "pair_plus_dw": timing(
                bf16_with_dw,
                sink.zero_,
                expected_outputs={"forward": ref_y, "dx": ref_dx},
                buffers={"dw_sink": (sink, ref_dw)},
            ),
        }
        if not all(base[key]["capture_pass"] for key in ("pair", "dw", "pair_plus_dw")):
            base["status"] = "capture_failed"
        result["results"].append(base)
        save()
        print(json.dumps(base), flush=True)
        if base["status"] != "ok":
            raise RuntimeError("BF16 capture output or FP32 sink parity failed")
        for recipe in recipes:
            entry = {**geometry, "m": m, "recipe": recipe}
            started = time.monotonic()
            try:
                quant_a = quantize_rows if recipe == "rowwise" else block1
                quant_w = (
                    quantize_rows
                    if recipe == "rowwise"
                    else block128
                    if recipe == "block128"
                    else block1
                )
                qw, sw = quant_w(weight)
                qt, st = quant_w(weight.T)
                # Production keeps both FP8 weight layouts contiguous, even
                # though quantizing a transposed BF16 view need not do so.
                qw, qt = qw.contiguous(), qt.contiguous()

                def linear(values, operand, scales, quantizer):
                    q, s = quantizer(values)
                    return torch._scaled_mm(
                        q,
                        operand.T,
                        scale_a=s,
                        scale_b=scales.T,
                        out_dtype=torch.bfloat16,
                    )

                # Discover unsupported recipes before paying compilation cost.
                y, dx = linear(x, qw, sw, quant_a), linear(dy, qt, st, quant_a)
                errors = {
                    "forward": error_metrics(y, ref_y),
                    "dx": error_metrics(dx, ref_dx),
                }
                entry["errors"] = errors
                entry["scale_layouts"] = {
                    "forward_weight": {
                        "shape": list(sw.shape),
                        "stride": list(sw.stride()),
                    },
                    "dx_weight": {"shape": list(st.shape), "stride": list(st.stride())},
                }
                entry["numerics_pass"] = all(
                    e["finite"] and e["relative_l2"] < 0.08 for e in errors.values()
                )
                if not entry["numerics_pass"]:
                    entry["status"] = "numerics_failed"
                else:
                    quant_a = torch.compile(quant_a, fullgraph=True, dynamic=False)

                    def refresh(
                        quant_w=quant_w, weight=weight, qw=qw, sw=sw, qt=qt, st=st
                    ):
                        q, s = quant_w(weight)
                        qw.copy_(q)
                        sw.copy_(s)
                        q, s = quant_w(weight.T)
                        qt.copy_(q)
                        st.copy_(s)

                    refresh = torch.compile(refresh, fullgraph=True, dynamic=False)

                    def pair(
                        linear=linear,
                        quant_a=quant_a,
                        x=x,
                        qw=qw,
                        sw=sw,
                        dy=dy,
                        qt=qt,
                        st=st,
                    ):
                        return linear(x, qw, sw, quant_a), linear(dy, qt, st, quant_a)

                    def with_dw(pair=pair, dy=dy, x=x, sink=sink):
                        y, dx = pair()
                        fence = dw_accum(dy, x, sink)
                        return y, dx + fence.to(dx.dtype)

                    if recipe == "rowwise":
                        # Check that the reference path really is the current
                        # fp8_linear implementation, including scale direction.
                        entry["production_reference"] = error_metrics(
                            y, fp8_linear(x, qw, sw)
                        )
                    refresh()
                    entry["weight_refresh"] = timing(
                        refresh,
                        buffers={
                            "weight": (qw, qw.clone()),
                            "weight_scale": (sw, sw.clone()),
                            "transposed": (qt, qt.clone()),
                            "transposed_scale": (st, st.clone()),
                        },
                    )
                    compiled_y, compiled_dx = pair()
                    entry["compiled_errors"] = {
                        "forward": error_metrics(compiled_y, ref_y),
                        "dx": error_metrics(compiled_dx, ref_dx),
                    }
                    if any(
                        not error["finite"] or error["relative_l2"] >= 0.08
                        for error in entry["compiled_errors"].values()
                    ):
                        raise RuntimeError("compiled quantizers failed numerical check")
                    expected = {"forward": compiled_y, "dx": compiled_dx}
                    entry["pair"] = timing(pair, expected_outputs=expected)
                    entry["pair_plus_dw"] = timing(
                        with_dw,
                        sink.zero_,
                        expected_outputs=expected,
                        buffers={"dw_sink": (sink, ref_dw)},
                    )
                    entry["capture_pass"] = all(
                        entry[key]["capture_pass"]
                        for key in ("weight_refresh", "pair", "pair_plus_dw")
                    )
                    entry["status"] = (
                        "ok" if entry["capture_pass"] else "capture_failed"
                    )
                    entry["numerics_pass"] &= entry["capture_pass"]
                    del compiled_y, compiled_dx, expected
                del qw, sw, qt, st, y, dx
            except Exception as exc:  # noqa: BLE001 -- independent pilot candidates may be unsupported.
                entry.update(
                    status="unsupported_or_error", error=f"{type(exc).__name__}: {exc}"
                )
                result["results"].append(entry)
                save()
                print(json.dumps(entry), flush=True)
                torch.cuda.synchronize()  # Stop on poisoned contexts; skip ordinary unsupported calls.
                continue
            entry["wall_seconds"] = time.monotonic() - started
            result["results"].append(entry)
            save()
            print(json.dumps(entry), flush=True)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

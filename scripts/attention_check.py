"""Current screen Flash/Flex attention comparison, including captured backward.

The default is one 4,096-token row, eight query heads, four KV heads, and
96 features per head. Flex variants are diagnostic comparisons; production
uses the imported causal_attention function with explicit native Flash SDPA.
"""

import argparse
import gc
import json
import statistics

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from delta_feedback_experiment import INDUCTOR_MODE
from delta_feedback_experiment.attention import causal_attention
from delta_feedback_experiment.train import _capture_without_gc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--replays", type=int, default=40)
    args = parser.parse_args()
    if min(args.length, args.batch, args.samples, args.replays) < 1:
        parser.error("length, batch, samples, and replays must be positive")
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    # Q/K RMS normalization follows the model; V retains the packed projection
    # stride rather than becoming an artificially contiguous allocation.
    packed = torch.randn(
        args.batch, args.length, 2304, device=device, dtype=torch.bfloat16
    )
    q, k, v, _gate = packed.split((768, 384, 384, 768), dim=-1)
    values = []
    for value, heads, normalized in ((q, 8, True), (k, 4, True), (v, 4, False)):
        value = value.view(args.batch, args.length, heads, 96).transpose(1, 2)
        if normalized:
            wide = value.float()
            value = (
                wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + 1e-6)
            ).to(torch.bfloat16)
        values.append(value.detach().requires_grad_())
    upstream = torch.randn_like(values[0])

    def causal_mask(batch, head, query, key):
        return query >= key

    mask = create_block_mask(
        causal_mask, None, None, args.length, args.length, device=device
    )
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(),
                "shape_q_k_v": [list(value.shape) for value in values],
                "stride_q_k_v": [list(value.stride()) for value in values],
                "scope": "attention forward plus dQ/dK/dV; fixed normalized BF16 inputs",
                "matmul_precision": torch.get_float32_matmul_precision(),
            }
        ),
        flush=True,
    )
    graphs, outputs, timings, failures = {}, {}, {}, {}
    baseline = None
    for mode in ("flex_hints", "native_flash", "flex_prescale"):
        try:
            if mode == "native_flash":
                attention = causal_attention
            else:
                options = {
                    "BACKEND": "TRITON",
                    "ROWS_GUARANTEED_SAFE": True,
                    "fwd_BLOCKS_ARE_CONTIGUOUS": True,
                }
                if mode == "flex_prescale":
                    options["PRESCALE_QK"] = True

                def attention(q, k, v, options=options):
                    return flex_attention(
                        q,
                        k,
                        v,
                        block_mask=mask,
                        enable_gqa=True,
                        kernel_options=options,
                    )

                attention = torch.compile(
                    attention, fullgraph=True, dynamic=False, mode=INDUCTOR_MODE
                )

            def step(attention=attention):
                output = attention(*values)
                gradients = torch.autograd.grad(output, values, upstream)
                return output, *gradients

            for _ in range(3):
                warm = step()
            torch.cuda.synchronize()
            del warm
            gc.collect()
            graph = torch.cuda.CUDAGraph()
            with _capture_without_gc(graph, None):
                captured = step()
            graph.replay()
            torch.cuda.synchronize()
            results = [result.detach().float().clone() for result in captured]
            assert all(torch.isfinite(result).all() for result in results)
            if mode == "flex_hints":
                baseline = results
            relative = (
                None
                if baseline is None
                else [
                    (
                        (actual - expected).norm() / expected.norm().clamp_min(1e-20)
                    ).item()
                    for actual, expected in zip(results, baseline, strict=True)
                ]
            )
            print(
                json.dumps(
                    {"mode": mode, "relative_l2_output_q_k_v_vs_flex": relative}
                ),
                flush=True,
            )
            # These tensors retain the graph outputs' storage for every replay.
            graphs[mode], outputs[mode], timings[mode] = graph, captured, []
        except Exception as exc:
            failures[mode] = f"{type(exc).__name__}: {exc}"
            print(json.dumps({"mode": mode, "error": failures[mode]}), flush=True)

    # Reverse order every sample to reduce thermal/order bias. Events measure
    # captured device work; graph dispatch and compilation are outside timing.
    for sample in range(args.samples):
        order = list(graphs) if sample % 2 == 0 else list(reversed(graphs))
        for mode in order:
            graph = graphs[mode]
            for _ in range(3):
                graph.replay()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(args.replays):
                graph.replay()
            end.record()
            end.synchronize()
            timings[mode].append(start.elapsed_time(end) / args.replays)
    reference_ms = (
        statistics.median(timings["flex_hints"]) if "flex_hints" in timings else None
    )
    for mode, samples in timings.items():
        median = statistics.median(samples)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "milliseconds": median,
                    "samples": samples,
                    "time_reduction_percent_vs_flex": (
                        100 * (1 - median / reference_ms) if reference_ms else None
                    ),
                }
            ),
            flush=True,
        )
    if "native_flash" in failures:
        raise SystemExit("production native Flash attention failed")


if __name__ == "__main__":
    main()

"""Screen-shape forward/backward CUDA-graph comparison of causal mask hints."""

import argparse
import json
import statistics

import torch
from torch.nn.attention.flex_attention import flex_attention

from delta_feedback_experiment import INDUCTOR_MODE
from delta_feedback_experiment.attention import causal_block_mask
from delta_feedback_experiment.train import _capture_without_gc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=1024)
    length = parser.parse_args().length
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    values = [
        torch.randn(4, length, heads, 96, device="cuda", dtype=torch.bfloat16)
        .transpose(1, 2)
        .requires_grad_()
        for heads in (8, 4, 4)
    ]
    upstream = torch.randn_like(values[0])
    mask = causal_block_mask(length, torch.device("cuda"))
    baseline = None
    for mode in ("plain", "hints", "prescale"):
        options = {"BACKEND": "TRITON"}
        if mode != "plain":
            options.update(ROWS_GUARANTEED_SAFE=True, fwd_BLOCKS_ARE_CONTIGUOUS=True)
        if mode == "prescale":
            options["PRESCALE_QK"] = True

        def attention(q, k, v, options=options):
            return flex_attention(
                q, k, v, block_mask=mask, enable_gqa=True, kernel_options=options
            )

        compiled = torch.compile(
            attention, fullgraph=True, dynamic=False, mode=INDUCTOR_MODE
        )

        def step(compiled=compiled):
            for value in values:
                if value.grad is not None:
                    value.grad.zero_()
            output = compiled(*values)
            output.backward(upstream)
            return output

        for _ in range(3):
            output = step()
        torch.cuda.synchronize()
        results = [output.detach().float(), *(v.grad.float().clone() for v in values)]
        if baseline is None:
            baseline = [result.clone() for result in results]
        relative = [
            (a - b).norm().item() / max(b.norm().item(), 1e-20)
            for a, b in zip(results, baseline)
        ]
        assert all(torch.isfinite(result).all() for result in results)
        # Release the warmup autograd nodes before capture switches streams.
        del output
        graph = torch.cuda.CUDAGraph()
        with _capture_without_gc(graph, None):
            step()
        samples = []
        for _ in range(9):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 100)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "length": length,
                    "matmul_precision": torch.get_float32_matmul_precision(),
                    "milliseconds": statistics.median(samples),
                    "samples": samples,
                    "relative_l2_output_q_k_v": relative,
                }
            ),
            flush=True,
        )
        del graph, results


if __name__ == "__main__":
    main()

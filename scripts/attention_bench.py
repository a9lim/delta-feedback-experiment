"""Compare forced attention backends in the production compiled wrapper.

    python scripts/attention_bench.py --scales screen,bridge,flagship --rows 2,4

Five warm graph timings per shape are enough to shortlist a backend. Both
the output and all input gradients are compared with eager flash attention
on identical BF16 inputs. This measures attention forward and backward,
not projections or a whole model step.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", default="screen,bridge,flagship")
    parser.add_argument("--rows", default="2,4")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--backends", default="flash,cudnn")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("logs/attention-bench/gh200.json"))
    opts = parser.parse_args()
    if opts.repeat < 1 or opts.seq_len < 1:
        parser.error("repeat and sequence length must be positive")

    import torch
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from delta_feedback_experiment import attention
    from delta_feedback_experiment.train import SCALES, _capture_without_gc

    backends = {"flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}
    result = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "cases": []}
    opts.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        opts.out.write_text(json.dumps(result, indent=2) + "\n")

    def error(actual, reference):
        value = float((actual.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12))
        if not math.isfinite(value):
            raise RuntimeError("nonfinite attention output or gradient")
        return value

    def body(inputs, upstream):
        for value in inputs:
            value.grad.zero_()
        attention.causal_attention(*inputs).backward(upstream)

    for scale in opts.scales.split(","):
        cfg = SCALES[scale]
        for rows in map(int, opts.rows.split(",")):
            if rows < 1:
                parser.error("rows must be positive")
            torch.manual_seed(1)
            # Production Q/K/V are transposed [B,T,H,D] projection views.
            inputs = tuple(
                torch.randn(rows, opts.seq_len, heads, 192, device="cuda", dtype=torch.bfloat16)
                .transpose(1, 2).detach().requires_grad_()
                for heads in (cfg["heads"], cfg["kv_heads"], cfg["kv_heads"])
            )
            upstream = torch.randn_like(inputs[0])
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                reference = F.scaled_dot_product_attention(*inputs, is_causal=True, enable_gqa=True)
                ref_grads = tuple(g.detach() for g in torch.autograd.grad(reference, inputs, upstream))
            reference = reference.detach()

            for backend in opts.backends.split(","):
                entry = {"scale": scale, "rows": rows, "seq_len": opts.seq_len, "backend": backend}
                attention.FUSED_BACKENDS = [backends[backend]]
                try:
                    for value in inputs:
                        value.grad = None
                    output = attention.causal_attention(*inputs)
                    output.backward(upstream)
                    errors = {"output": error(output, reference)}
                    errors.update({name: error(value.grad, ref) for name, value, ref in zip(
                        ("dq", "dk", "dv"), inputs, ref_grads, strict=True
                    )})
                    entry["relative_l2"] = errors
                    if max(errors.values()) > 0.02:
                        raise RuntimeError(f"relative L2 error exceeds 0.02: {errors}")

                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            body(inputs, upstream)
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with _capture_without_gc(graph, None):
                        body(inputs, upstream)
                    for _ in range(3):
                        graph.replay()
                    torch.cuda.synchronize()
                    capture_errors = {name: error(value.grad, ref) for name, value, ref in zip(
                        ("dq", "dk", "dv"), inputs, ref_grads, strict=True
                    )}
                    entry["capture_relative_l2"] = capture_errors
                    if max(capture_errors.values()) > 0.02:
                        raise RuntimeError(f"captured gradients exceed 0.02 relative L2: {capture_errors}")
                    times = []
                    for _ in range(opts.repeat):
                        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        times.append(start.elapsed_time(end))
                    entry.update(ms=statistics.median(times), samples_ms=times)
                    graph.reset()
                    del graph, output
                except (RuntimeError, NotImplementedError) as exc:
                    entry["error"] = str(exc)
                result["cases"].append(entry)
                save()
                print(json.dumps(entry), flush=True)
            del inputs, upstream, reference, ref_grads
            torch.cuda.empty_cache()
    print(f"wrote {opts.out}", flush=True)


if __name__ == "__main__":
    main()

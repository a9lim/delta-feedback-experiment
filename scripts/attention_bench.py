"""Compare forced attention backends and optional padding in compiled wrappers.

    python scripts/attention_bench.py --scales screen,bridge,flagship --rows 2,4
    python scripts/attention_bench.py --scales bridge --rows 2 --head-dim 256 --backends cudnn
    python scripts/attention_bench.py --scales bridge --rows 2 --heads 6 --kv-heads 3 --head-dim 256 --backends cudnn
    python scripts/attention_bench.py --scales bridge --rows 2 --head-dim 192 --pad-to 256 --backends cudnn

Five warm graph timings per shape are enough to shortlist a backend. Both
the output and all input gradients are compared with eager flash attention
on identical BF16 inputs at the original head width. Padding preserves that
width's attention scale and slices the output back to it. This measures
attention forward and backward, including padding when requested, not
projections or a whole model step. No production architecture is changed.
Head-count overrides apply to every selected scale; use one scale when
testing a particular proposed geometry.
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
    parser.add_argument("--head-dim", type=int, default=192, help="original Q/K/V head width")
    parser.add_argument("--heads", type=int, default=None, help="query heads, overriding every selected scale")
    parser.add_argument("--kv-heads", type=int, default=None, help="KV heads, overriding every selected scale")
    parser.add_argument("--pad-to", type=int, default=None,
                        help="zero-pad Q/K/V to this width, retaining the original attention scale")
    parser.add_argument("--backends", default="flash,cudnn")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("logs/attention-bench/gh200.json"))
    opts = parser.parse_args()
    if min(opts.repeat, opts.seq_len, opts.head_dim) < 1:
        parser.error("repeat, sequence length and head dimension must be positive")
    if opts.pad_to is not None and opts.pad_to < opts.head_dim:
        parser.error("--pad-to must be at least --head-dim")
    if any(count is not None and count < 1 for count in (opts.heads, opts.kv_heads)):
        parser.error("head counts must be positive")
    if opts.heads is not None and opts.kv_heads is not None and opts.heads % opts.kv_heads:
        parser.error("query heads must be divisible by KV heads")

    import torch
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from delta_feedback_experiment import attention
    from delta_feedback_experiment.train import SCALES, _capture_without_gc

    shapes = []
    for scale in opts.scales.split(","):
        cfg = SCALES[scale]
        heads = opts.heads if opts.heads is not None else cfg["heads"]
        kv_heads = opts.kv_heads if opts.kv_heads is not None else cfg["kv_heads"]
        if heads % kv_heads:
            parser.error(f"{scale}: query heads ({heads}) must be divisible by KV heads ({kv_heads})")
        shapes.append((scale, heads, kv_heads))

    attend = attention.causal_attention
    if opts.pad_to is not None:
        def padded_attention(query, key, value):
            padding = (0, opts.pad_to - opts.head_dim)
            with sdpa_kernel(attention.FUSED_BACKENDS, set_priority=len(attention.FUSED_BACKENDS) > 1):
                output = F.scaled_dot_product_attention(
                    F.pad(query, padding), F.pad(key, padding), F.pad(value, padding),
                    scale=opts.head_dim**-0.5, is_causal=True, enable_gqa=True,
                )
            return output[..., :opts.head_dim]

        attend = torch.compile(padded_attention, fullgraph=True, dynamic=False, mode=attention.INDUCTOR_MODE)

    backends = {"flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}
    result = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "head_dim": opts.head_dim, "pad_to": opts.pad_to,
              "heads_override": opts.heads, "kv_heads_override": opts.kv_heads, "cases": []}
    opts.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        opts.out.write_text(json.dumps(result, indent=2) + "\n")

    def error(actual, reference):
        value = float((actual.detach().float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12))
        if not math.isfinite(value):
            raise RuntimeError("nonfinite attention output or gradient")
        return value

    def body(inputs, upstream):
        for value in inputs:
            value.grad.zero_()
        attend(*inputs).backward(upstream)

    for scale, heads, kv_heads in shapes:
        for rows in map(int, opts.rows.split(",")):
            if rows < 1:
                parser.error("rows must be positive")
            torch.manual_seed(1)
            # Production Q/K/V are transposed [B,T,H,D] projection views.
            inputs = tuple(
                torch.randn(rows, opts.seq_len, count, opts.head_dim, device="cuda", dtype=torch.bfloat16)
                .transpose(1, 2).detach().requires_grad_()
                for count in (heads, kv_heads, kv_heads)
            )
            upstream = torch.randn_like(inputs[0])
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                reference = F.scaled_dot_product_attention(*inputs, is_causal=True, enable_gqa=True)
                ref_grads = tuple(g.detach() for g in torch.autograd.grad(reference, inputs, upstream))
            reference = reference.detach()

            for backend in opts.backends.split(","):
                entry = {"scale": scale, "rows": rows, "seq_len": opts.seq_len,
                         "heads": heads, "kv_heads": kv_heads,
                         "head_dim": opts.head_dim, "pad_to": opts.pad_to, "backend": backend}
                attention.FUSED_BACKENDS = [backends[backend]]
                try:
                    for value in inputs:
                        value.grad = None
                    output = attend(*inputs)
                    output.backward(upstream)
                    errors = {"output": error(output, reference)}
                    errors.update({name: error(value.grad, ref) for name, value, ref in zip(
                        ("dq", "dk", "dv"), inputs, ref_grads, strict=True
                    )})
                    entry["relative_l2"] = errors
                    if max(errors.values()) > 0.02:
                        raise RuntimeError(f"relative L2 error exceeds 0.02: {errors}")
                    # Release the default-stream autograd graph before warmup
                    # creates AccumulateGrad nodes on the capture-side stream.
                    del output

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
                    del graph
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

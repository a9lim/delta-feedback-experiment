"""Short fixed-config CCE sweep, including production FP8 activation quantization.

    python scripts/cce_bench.py --dim 1152 --rows 2 --out logs/cce-bench/bridge.json
    python scripts/cce_bench.py --dim 1536 --rows 4 --out logs/cce-bench/flagship.json
    python scripts/cce_bench.py --dim 768 --rows 2 --candidates bf16-base,fp8-v128

Rows count 4096-token CCE input rows, including any concatenated MTP head.
For the paired NTP/MTP head, use twice the training replay's row count;
this benchmark does not model the small token crop at the sequence boundary.
This measures the head and its vocabulary ordering, not a whole training step.
Classifier FP8 refresh is measured separately: production pays it once per
optimizer step, whereas activation quantization runs inside every FP8 call.
The graph timings exclude resetting the persistent FP32 classifier sink.
Every measured replay starts with a fresh zero sink; no live trainer is used.

Configurations are rebound in this process only. CCE_AUTOTUNE is forbidden:
its candidate resets are incompatible with persistent gradient accumulators.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import statistics
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Tile:
    b: int
    v: int
    d: int
    warps: int
    stages: int

    def config(self):
        import triton

        return triton.Config(
            {"BLOCK_B": self.b, "BLOCK_V": self.v, "BLOCK_D": self.d},
            num_warps=self.warps,
            num_stages=self.stages,
        )


@dataclass(frozen=True)
class Candidate:
    precision: str
    forward: Tile
    backward: Tile


BF16_BASE = Tile(128, 128, 32, 4, 4)
FP8_FORWARD = Tile(256, 64, 64, 8, 3)
FP8_BACKWARD = Tile(64, 64, 32, 4, 4)
CANDIDATES = {
    "bf16-base": Candidate("bf16", BF16_BASE, BF16_BASE),
    "bf16-d64": Candidate("bf16", Tile(128, 128, 64, 4, 3), Tile(128, 128, 64, 4, 3)),
    "bf16-v256": Candidate("bf16", Tile(128, 256, 64, 8, 3), Tile(128, 256, 64, 8, 3)),
    "fp8-base": Candidate("fp8", FP8_FORWARD, FP8_BACKWARD),
    "fp8-v128": Candidate("fp8", Tile(256, 128, 64, 8, 3), Tile(64, 128, 64, 8, 3)),
    "fp8-b128": Candidate("fp8", Tile(256, 128, 128, 8, 3), Tile(128, 128, 64, 8, 3)),
}


def select(candidate: Candidate) -> None:
    """Replace the already-imported kernels' outer config heuristics as well.

    Importing any cut_cross_entropy submodule executes its package __init__,
    which decorates both kernels. Patching config factories alone is too late.
    The inner kernel heuristics remain intact; tile metadata calls the patched
    factories dynamically. BF16 halves match and FP8 vocabulary tiles match.
    """
    ta = importlib.import_module("cut_cross_entropy.tl_autotune")
    fw = importlib.import_module("cut_cross_entropy.cce_lse_forward")
    bw = importlib.import_module("cut_cross_entropy.cce_backward")
    assert not ta._AUTOTUNE
    assert candidate.forward.v == candidate.backward.v
    if candidate.precision == "bf16":
        assert candidate.forward == candidate.backward
        ta._cce_best_config = candidate.forward.config
    else:
        ta._cce_best_config_fp8 = candidate.forward.config
        ta._cce_best_config_fp8_backward = candidate.backward.config
    fw._cce_lse_forward_kernel = ta.cce_forward_autotune()(
        fw._cce_lse_forward_kernel.fn
    )
    bw._cce_backward_kernel = ta.cce_backward_autotune()(bw._cce_backward_kernel.fn)


def error_metrics(actual, expected) -> dict:
    import torch

    a, b = actual.detach().float().flatten(), expected.detach().float().flatten()
    return {
        "relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-30)),
        "max_abs": float((a - b).abs().max()),
        "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        "finite": bool(torch.isfinite(a).all()),
    }


def compare(actual, expected) -> dict:
    return {
        name: error_metrics(a, b)
        for name, a, b in zip(
            ("loss", "input_grad", "classifier_grad"), actual, expected, strict=True
        )
    }


def elapsed(replay, reset, warm: int, repeat: int) -> dict:
    import torch

    for _ in range(warm):
        reset()
        replay()
    torch.cuda.synchronize()
    pairs = []
    for _ in range(repeat):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        reset()  # Outside the event interval: production zeroes once per step.
        start.record()
        replay()
        end.record()
        pairs.append((start, end))
    torch.cuda.synchronize()
    values = [start.elapsed_time(end) for start, end in pairs]
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "samples_ms": values,
    }


def revision(path: Path) -> dict:
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    return {
        "head": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--vocab", type=int, default=50304)
    parser.add_argument("--candidates", default="bf16-base,fp8-v128,bf16-d64")
    parser.add_argument("--check-tokens", type=int, default=128)
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--z-coef", type=float, default=1e-5)
    parser.add_argument("--out", type=Path, default=Path("logs/cce-bench/latest.json"))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        print(
            json.dumps(
                {name: asdict(value) for name, value in CANDIDATES.items()}, indent=2
            )
        )
        return
    names = args.candidates.split(",")
    if set(names) - CANDIDATES.keys():
        parser.error(f"unknown candidates: {set(names) - CANDIDATES.keys()}")
    if (
        min(
            args.rows,
            args.seq_len,
            args.dim,
            args.vocab,
            args.check_tokens,
            args.repeat,
        )
        < 1
    ):
        parser.error("dimensions, check tokens and repeat must be positive")
    if os.environ.get("CCE_AUTOTUNE", "0") != "0":
        parser.error(
            "unset CCE_AUTOTUNE; this benchmark uses disposable fixed-config sinks"
        )
    os.environ["CCE_AUTOTUNE"] = "0"

    import torch
    import torch.nn.functional as F
    from cut_cross_entropy.cce import CCEParams, linear_cross_entropy_apply
    from cut_cross_entropy.utils import CCEFp8Classifier, _handle_eps

    # Import the production compiler repair before any compiled quantizer runs.
    from delta_feedback_experiment import inductor  # noqa: F401
    from delta_feedback_experiment.cuda_kernels import Fp8Weights, quantize_weights
    from delta_feedback_experiment.model import (
        BASE_NORMAL_INIT_STD,
        MUP_BASE_DIM,
        batch_vocab_order,
    )

    if not torch.cuda.is_available():
        parser.error("CUDA is required; --list and --help work without it")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    tokens = args.rows * args.seq_len
    embeddings = torch.randn(tokens, args.dim, device=device)
    embeddings = (
        F.normalize(embeddings, dim=-1) * args.dim**0.5 * (MUP_BASE_DIM / args.dim)
    )
    embeddings = embeddings.to(torch.bfloat16).requires_grad_()
    classifier = (
        torch.randn(args.vocab, args.dim, device=device) * BASE_NORMAL_INIT_STD
    ).to(torch.bfloat16)
    targets = torch.randint(args.vocab, (tokens,), device=device)
    sink = torch.zeros_like(classifier, dtype=torch.float32)
    fp8_weights = Fp8Weights.allocate(tuple(classifier.shape), device)
    quantize_weights(fp8_weights, classifier)
    fp8_classifier = CCEFp8Classifier(
        fp8_weights.weight,
        fp8_weights.scale.squeeze(1),
        fp8_weights.transposed,
        fp8_weights.transposed_scale.squeeze(1),
    )

    def call(e, y, fp8, filter_eps):
        ordering = batch_vocab_order(e, classifier)
        # The low-level CCEParams path takes explicit filter flags. Passing
        # None alone does not disable the backward's Triton comparisons.
        filtering = filter_eps is not None
        params = CCEParams(
            targets=y,
            valids=None,
            softcap=None,
            reduction="none",
            filter_eps=filter_eps,
            shift=0,
            batch_shape=y.shape,
            accum_e_fp32=False,
            accum_c_fp32=False,
            filter_e_grad=filtering,
            filter_c_grad=filtering,
            vocab_parallel_options=None,
            return_lse=True,
            vocab_ordering=ordering,
            c_grad_accum=sink,
            skip_early=filtering,
            fp8_classifier=fp8_classifier if fp8 else None,
        )
        nll, lse = linear_cross_entropy_apply(e, classifier, None, params)
        loss = nll.mean() + args.z_coef * lse.square().mean()
        (grad,) = torch.autograd.grad(loss, e)
        return loss, grad

    def run(e, y, fp8, filter_eps):
        sink.zero_()
        loss, grad = call(e, y, fp8, filter_eps)
        return loss.detach().clone(), grad.detach().clone(), sink.clone()

    check_e = embeddings[: min(args.check_tokens, tokens)].detach().requires_grad_()
    check_y = targets[: check_e.shape[0]]
    # Dense no-filter oracle checks algebra independently of CCE's skip logic.
    ref_c = classifier.detach().requires_grad_()
    logits = (check_e @ ref_c.T).float()
    dense_loss = (
        F.cross_entropy(logits, check_y)
        + args.z_coef * logits.logsumexp(-1).square().mean()
    )
    dense_de, dense_dc = torch.autograd.grad(dense_loss, (check_e, ref_c))
    dense = (dense_loss.detach(), dense_de, dense_dc.float())
    del logits, ref_c

    select(CANDIDATES["bf16-base"])
    filter_eps = _handle_eps("auto", torch.bfloat16)
    production_reference = run(embeddings, targets, False, filter_eps)
    # Explicitly exercise accumulated, nonzero sink semantics outside timing.
    loss, de = call(embeddings, targets, False, filter_eps)
    accumulation_error = error_metrics(sink, production_reference[2] * 2)
    if not accumulation_error["finite"] or accumulation_error["relative_l2"] > 0.03:
        raise RuntimeError(f"FP32 sink accumulation failed: {accumulation_error}")
    del loss, de

    # Warm the production classifier quantizer once; refresh is an additive
    # per-optimizer-step cost, never incorrectly charged once per head call.
    refresh_stream = torch.cuda.Stream()
    refresh_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(refresh_stream):
        for _ in range(2):
            quantize_weights(fp8_weights, classifier)
    torch.cuda.current_stream().wait_stream(refresh_stream)
    torch.cuda.synchronize()
    refresh_graph = torch.cuda.CUDAGraph()
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.graph(refresh_graph):
            quantize_weights(fp8_weights, classifier)
    finally:
        if gc_enabled:
            gc.enable()
    refresh_time = elapsed(refresh_graph.replay, lambda: None, args.warm, args.repeat)
    root = Path(__file__).resolve().parents[1]
    result = {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "args": {**vars(args), "out": str(args.out)},
        "revisions": {
            "experiment": revision(root),
            "cce": revision(root.parent / "vendor/ml-cross-entropy"),
        },
        "tokens": tokens,
        "filter_eps": filter_eps,
        "classifier_refresh": refresh_time,
        "sink_accumulation": accumulation_error,
        "scope": "Captured CCE fwd+bwd, vocabulary ordering and per-call activation quantization; excludes per-step sink zero and classifier refresh. Initialized synthetic normalized readouts, not a trained checkpoint.",
        "results": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_text(json.dumps(result, indent=2) + "\n")

    save()
    for name in names:
        candidate = CANDIDATES[name]
        select(candidate)
        fp8 = candidate.precision == "fp8"
        entry = {"name": name, **asdict(candidate)}
        started = time.monotonic()
        try:
            dense_check = compare(run(check_e, check_y, fp8, None), dense)
            full = run(embeddings, targets, fp8, filter_eps)
            production_check = compare(full, production_reference)
            entry.update(
                dense_no_filter=dense_check, production_filter=production_check
            )
            # BF16 tile changes can alter which entire tiles the intentional
            # filter drops. Keep that difference visible; these broad bounds
            # reject gross faults, not establish training equivalence.
            limit = 0.15 if fp8 else 0.04
            checks = [*dense_check.values(), *production_check.values()]
            good = all(item["finite"] for item in checks)
            good &= dense_check["loss"]["relative_l2"] < 0.005
            good &= all(
                dense_check[key]["relative_l2"] < limit
                for key in ("input_grad", "classifier_grad")
            )
            good &= all(
                production_check[key]["relative_l2"] < limit
                for key in ("input_grad", "classifier_grad")
            )
            entry["numerics_pass"] = good
            if not good:
                entry["status"] = "numerics_failed"
            else:
                # Warm on a side stream before capture; no compilation or GC
                # may occur inside CUDA graph capture.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        sink.zero_()
                        call(embeddings, targets, fp8, filter_eps)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                gc_enabled = gc.isenabled()
                gc.disable()
                try:
                    with torch.cuda.graph(graph):
                        graph_loss, graph_grad = call(
                            embeddings, targets, fp8, filter_eps
                        )
                finally:
                    if gc_enabled:
                        gc.enable()
                entry["timing"] = elapsed(
                    graph.replay, sink.zero_, args.warm, args.repeat
                )
                # elapsed leaves exactly the last replay's contribution in
                # the sink. Compare the captured outputs with this same
                # candidate's ordinary call, not a different precision/tile.
                parity = compare((graph_loss, graph_grad, sink), full)
                entry["capture_parity"] = parity
                entry["capture_pass"] = (
                    all(metric["finite"] for metric in parity.values())
                    and parity["loss"]["relative_l2"] < 2e-5
                    and parity["input_grad"]["relative_l2"] < 0.02
                    and parity["classifier_grad"]["relative_l2"] < 1e-3
                )
                entry["status"] = "ok" if entry["capture_pass"] else "capture_failed"
                entry["numerics_pass"] &= entry["capture_pass"]
                del graph, graph_loss, graph_grad
            del full
        except Exception as exc:  # noqa: BLE001 -- report failed candidates, then check CUDA health.
            entry.update(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
        entry["wall_seconds"] = time.monotonic() - started
        result["results"].append(entry)
        save()
        print(json.dumps(entry), flush=True)
        if entry["status"] == "error":
            # Persist the complete chained traceback before checking context
            # health. Compile failures can be skipped; illegal accesses abort.
            torch.cuda.synchronize()

    eligible = [r for r in result["results"] if r["status"] == "ok"]
    result["ranked"] = [
        r["name"] for r in sorted(eligible, key=lambda r: r["timing"]["median_ms"])
    ]
    save()
    print(
        f"Saved {args.out}; fastest numerically eligible: {result['ranked']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

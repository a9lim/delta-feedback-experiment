"""Authoritative offline plus Jobe CUDA execution gate."""

from __future__ import annotations

import math
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def cuda_gate() -> None:
    """Capture and replay every default DF mode at full screen geometry."""
    from torch._dynamo.utils import counters

    from .model import (
        DFModel,
        arm_config,
        flash_attn_func,
        linear_cross_entropy_apply,
    )
    from .optim import build_optimizers
    from .train import CudaGraphTrainer, build_parser, build_schedule

    if flash_attn_func is None or linear_cross_entropy_apply is None:
        raise RuntimeError("Jobe gate requires flash-attn and cut-cross-entropy")

    args = build_parser().parse_args(["cuda-probe", "--arm", "df"])
    schedule = build_schedule(args)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    model = (
        DFModel(
            arm_config(
                "df",
                vocab_size=args.vocab_size,
                dim=args.dim,
                layers=args.layers,
                heads=args.heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                intermediate=args.intermediate,
                max_seq_len=args.seq_len + 1,
            )
        )
        .cuda()
        .train()
    )
    optimizers = build_optimizers(
        model,
        lr_muon=args.lr_muon,
        wd_muon=args.wd_muon,
        lr_adam=args.lr_adam,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embedded = model.embed_tokens(
            torch.zeros(1, 1, dtype=torch.long, device="cuda")
        )
    if embedded.dtype != torch.bfloat16:
        raise AssertionError(f"CUDA residual seed is {embedded.dtype}, not bfloat16")
    # Do not carry this uncaptured autograd edge into the blocking capture stream.
    del embedded
    torch.cuda.synchronize()

    started = time.monotonic()
    runner = CudaGraphTrainer(model, optimizers, args, schedule)
    torch.cuda.synchronize()
    prepared = time.monotonic() - started
    capture_peak = torch.cuda.max_memory_allocated() / 2**30
    if capture_peak >= 22.0:
        raise AssertionError(
            f"capture peak leaves unsafe headroom: {capture_peak:.2f} GiB"
        )

    compiled = counters["stats"]["unique_graphs"]
    generator = torch.Generator().manual_seed(0)
    records = []
    for index, (spec, state) in enumerate(runner.states.items()):
        rows = torch.randint(
            0,
            args.vocab_size,
            (args.micro_rows, args.seq_len + 1),
            generator=generator,
        )
        runner.begin(spec)
        started = time.monotonic()
        runner.replay(state, rows, index + 1, index * args.micro_rows)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        runner.prepare_optimizer(state)
        norms = torch._foreach_norm([parameter.grad for parameter in state.active])
        grad_norm = torch.stack(norms).norm().item()
        if not math.isfinite(grad_norm) or not math.isfinite(state.loss_sum.item()):
            raise AssertionError(f"nonfinite CUDA result in {spec}")
        for optimizer in optimizers:
            optimizer.step()
        runner.zero_grad()
        records.append(
            f"k={spec.n_passes}/z={int(spec.want_z)}/ckpt={int(spec.checkpoint)}:"
            f"{elapsed * 1000:.1f}ms"
        )

    torch.cuda.synchronize()
    escaped = counters["stats"]["unique_graphs"] - compiled
    if escaped:
        raise AssertionError(f"{escaped} compiled graph(s) escaped preparation")
    print(
        "cuda gate | "
        f"prepare={prepared:.1f}s | peak={capture_peak:.2f}GiB | "
        f"graphs={len(runner.states)} | " + " | ".join(records)
    )


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", *argv],
        check=False,
    )
    if result.returncode:
        raise SystemExit(result.returncode)
    if torch.cuda.is_available():
        cuda_gate()


if __name__ == "__main__":
    main()

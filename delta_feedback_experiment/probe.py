"""Fast CPU contracts or one small CUDA train/eval/decode execution smoke."""

from __future__ import annotations

import gc
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


@torch.compiler.set_stance("force_eager")
def cuda_probe() -> None:
    """Exercise CUDA kernels and graph replay without Inductor compilation.

    FLA, CCE, sparse experts, routing, and causal Flash SDPA use real CUDA
    kernels. Cached attention uses FlexAttention's eager reference. This is
    an execution smoke, not compiler or production-memory qualification.
    """
    from .model import DeltaModel, KVCache, condition_config
    from .optim import build_optimizers
    from .train import (
        CudaEvalRunner,
        CudaGraphTrainer,
        GraphSpec,
        build_schedule,
        clip_gradients,
        evaluate,
        model_fields,
        parse_run_args,
    )

    # Keep PKDA's supported 128-wide CUDA heads; shrink the residual stream,
    # vocabulary, rows, and sequence instead of invoking a production screen.
    args = parse_run_args(
        [
            "probe",
            "--condition",
            "fl",
            "--steps",
            "2",
            "--feedback-start",
            "0",
            "--three-pass",
            "0",
            "--batch-rows",
            "1",
            "--micro-rows",
            "1",
            "--eval-rows",
            "1",
            "--seq-len",
            "64",
            "--vocab-size",
            "257",
            "--dim",
            "128",
            "--layers",
            "12",
            "--heads",
            "4",
            "--kv-heads",
            "2",
            "--head-dim",
            "32",
            "--intermediate",
            "256",
            "--pkda-heads",
            "2",
            "--pkda-head-dim",
            "128",
            "--loop-iterations",
            "2",
            "--loop-max-iterations",
            "2",
        ]
    )
    cfg = condition_config(args.condition, **model_fields(args))

    class ProbeTrainer(CudaGraphTrainer):
        def _reachable_specs(self, schedule):
            return [GraphSpec(2, True, 2)]

    class Rows:
        def __init__(self):
            self.rows = torch.randint(
                0,
                args.vocab_size,
                (1, args.seq_len + 1),
                generator=torch.Generator().manual_seed(19),
            )

        def batch(self, first, count, device=None):
            rows = self.rows[first : first + count]
            return rows.to(device) if device is not None else rows

    torch.manual_seed(7)
    torch.set_float32_matmul_precision("high")
    gc.collect()
    started = time.monotonic()
    model = DeltaModel(cfg).cuda().train()
    optimizers = build_optimizers(
        model, lr_normuonh=args.lr_normuonh, lr_nadam=args.lr_nadam
    )
    print("cuda probe | preparing one tiny training graph", flush=True)
    runner = ProbeTrainer(model, optimizers, args, build_schedule(args))
    data = Rows()
    spec = next(iter(runner.states))
    initial = model.embed_tokens.weight.detach().clone()
    for step in (1, 2):
        runner.zero_grad()
        state = runner.begin(spec, args.zloss)
        runner.replay_batch(state, data, step, 0)
        runner.prepare_optimizer(state)
        assert math.isfinite(state.loss_sum.item())
        assert math.isfinite(clip_gradients(model.parameters()))
        assert runner.head_accum is not None and not runner.head_accum.any()
        for name in (
            "embed_tokens.weight",
            "fuse_value.weight",
            "payload_router.query",
            "mtp.projection.weight",
            "blocks.0.mlp.shared.down_proj.weight",
        ):
            gradient = dict(model.named_parameters())[name].grad
            assert gradient is not None and gradient.dtype == torch.float32, name
            assert gradient.abs().sum() > 0 and torch.isfinite(gradient).all(), name
        for optimizer in optimizers:
            optimizer.step()
        bias_before = model.blocks[0].mlp.expert_bias.clone()
        model.update_expert_bias(state.expert_counts)
        assert not torch.equal(bias_before, model.blocks[0].mlp.expert_bias)
        model.refresh_shadows()
    assert not torch.equal(initial, model.embed_tokens.weight)
    runner.zero_grad()

    print(
        "cuda probe | train replay passed; checking evaluation and decode", flush=True
    )
    evaluator = CudaEvalRunner(model, args, runner.pool)
    actual = evaluator.run(data)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = evaluate(model, data, args, torch.device("cuda"))
    assert actual.keys() == expected.keys()
    for key in actual:
        assert math.isfinite(actual[key])
        assert math.isclose(actual[key], expected[key], rel_tol=3e-3, abs_tol=3e-3), key

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        toks = data.rows[:, :4].cuda()
        embedded = model.embed_tokens(toks)
        cache = KVCache(cfg, batch=1, device="cuda", dtype=torch.bfloat16)
        prefill = model.forward_column(embedded[:, :3], cache=cache)
        decoded = model.step(toks[:, 3:], prefill.payload[:, -1:], cache)
        reference_rows = torch.cat(
            [embedded[:, :3], model.fuse(prefill.payload[:, -1:], embedded[:, 3:])],
            dim=1,
        )
        reference = model.forward_column(reference_rows).h_top[:, -1:]
        relative = (
            decoded.h_top.float() - reference.float()
        ).norm() / reference.float().norm().clamp_min(1e-6)
        assert torch.isfinite(decoded.h_top).all() and relative < 0.05, relative.item()
    torch.cuda.synchronize()
    print(
        f"cuda probe passed | {time.monotonic() - started:.1f}s | train_graphs=1 | eval_graphs=1 | decode_rel={relative.item():.4g}"
    )


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if torch.cuda.is_available():
        if argv:
            raise SystemExit(
                "CUDA probe takes no arguments; run pytest directly for test selection"
            )
        cuda_probe()
    else:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", *argv],
            check=False,
        )
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

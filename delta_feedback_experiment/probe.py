"""One small CUDA train/eval/decode smoke; portable contracts live in pytest."""

from __future__ import annotations

import argparse
import math
import time
import warnings


def cuda_probe() -> None:
    import torch

    from .model import DeltaModel, KVCache, condition_config
    from .optim import apply_schedule, build_optimizers
    from .train import (
        CudaEvalRunner,
        CudaGraphTrainer,
        GraphSpec,
        build_schedule,
        clip_gradients,
        model_fields,
        parse_run_args,
    )

    # Three four-layer cells exercise both mixers and a repeated core. PKDA
    # keeps its CUDA head width; sparse experts and token rows stay tiny.
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
            "16",
            "--vocab-size",
            "128",
            "--dim",
            "64",
            "--layers",
            "12",
            "--heads",
            "4",
            "--kv-heads",
            "2",
            "--head-dim",
            "16",
            "--expert-intermediate",
            "32",
            "--num-routed-experts",
            "3",
            "--experts-per-token",
            "1",
            "--pkda-heads",
            "1",
            "--pkda-head-dim",
            "128",
            "--loop-iterations",
            "2",
            "--loop-max-iterations",
            "2",
        ]
    )

    class ProbeTrainer(CudaGraphTrainer):
        def _reachable_specs(self, schedule):
            return [GraphSpec(2, True, 2)]

    class Rows:
        def __init__(self):
            self.rows = torch.randint(0, args.vocab_size, (1, args.seq_len + 1))

        def batch(self, first, count, device=None):
            return self.rows[first : first + count].to(device=device)

    torch.manual_seed(7)
    torch.set_float32_matmul_precision("high")
    started = time.monotonic()
    model = DeltaModel(condition_config(args.condition, **model_fields(args))).cuda()
    optimizers = build_optimizers(model)
    schedule = build_schedule(args)
    print("cuda probe | train graph", flush=True)
    runner = ProbeTrainer(model, optimizers, args, schedule)
    data = Rows()
    spec = next(iter(runner.states))
    initial = model.embed_tokens.weight.detach().clone()
    for step in (1, 2):
        apply_schedule(optimizers, schedule, step)
        runner.zero_grad()
        state = runner.begin(spec, args.zloss)
        runner.replay_batch(state, data, step, 0)
        runner.prepare_optimizer(state)
        assert math.isfinite(state.loss_sum.item())
        assert math.isfinite(clip_gradients(model.parameters()))
        assert runner.head_accum is not None and not runner.head_accum.any()
        parameters = dict(model.named_parameters())
        for name in (
            "embed_tokens.weight",
            "fuse_value.weight",
            "payload_router.query",
            "mtp.projection.weight",
            "blocks.4.attn.control_proj.weight",
            "blocks.0.mlp.shared.down_proj.weight",
        ):
            gradient = parameters[name].grad
            assert gradient is not None and gradient.dtype == torch.float32, name
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name
        routed = [
            expert.down_proj.weight.grad for expert in model.blocks[0].mlp.experts
        ]
        assert all(
            gradient is not None and gradient.dtype == torch.float32
            for gradient in routed
        )
        assert sum(gradient.abs().sum() for gradient in routed) > 0
        for optimizer in optimizers:
            optimizer.step()
        model.update_expert_bias(state.expert_counts)
        model.refresh_shadows()
    assert not torch.equal(initial, model.embed_tokens.weight)
    runner.zero_grad()

    print("cuda probe | eval graph and cached decode", flush=True)
    evaluator = CudaEvalRunner(model, args, runner.pool)
    assert all(math.isfinite(value) for value in evaluator.run(data).values())
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = data.rows[:, :4].cuda()
        cache = KVCache(model.cfg, batch=1, device="cuda", dtype=torch.bfloat16)
        prefill = model.forward_column(model.embed_tokens(tokens[:, :3]), cache=cache)
        decoded = model.step(tokens[:, 3:], prefill.payload[:, -1:], cache)
        assert cache.pos == 4 and decoded.h_top.shape == (1, 1, args.dim)
        assert torch.isfinite(decoded.h_top).all()
    torch.cuda.synchronize()
    print(f"cuda probe passed | {time.monotonic() - started:.1f}s")


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser("delta probe", description=__doc__).parse_args(argv)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("delta probe requires CUDA; run pytest for portable tests")
    # Exercise real CUDA kernels and graph replay without compiling a matrix
    # of Inductor specializations. Cached FlexAttention uses its reference.
    with torch.compiler.set_stance("force_eager"), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"flex_attention called without torch\.compile\(\)"
        )
        cuda_probe()


if __name__ == "__main__":
    main()

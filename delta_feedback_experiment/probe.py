"""One small CUDA train/eval/decode smoke; portable contracts live in pytest."""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from dataclasses import replace


def cuda_probe() -> None:
    import torch

    from . import distributed
    from .model import LOOP_MAX_ITERATIONS, DeltaModel, KVCache, condition_config
    from .optim import apply_schedule, build_optimizers
    from .sites import ParameterSites
    from .train import (
        CudaEvalRunner,
        CudaGraphTrainer,
        GraphSpec,
        build_schedule,
        model_fields,
        parse_run_args,
        pick_device,
    )

    topology = distributed.Topology.from_environment()
    # Three four-layer cells exercise both mixers; the loop repeats the whole
    # column. Both mixers keep their production head widths; experts and rows
    # stay tiny: one row per rank, so a multi-rank probe runs the same
    # collectives a training step does.
    args = parse_run_args(
        [
            "probe",
            "--condition",
            "fl",
            "--steps",
            "2",
            "--recurrence-start",
            "0",
            "--three-rate",
            "0",
            "--batch-rows",
            str(max(2, topology.world)),
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
        ]
    )

    class ProbeTrainer(CudaGraphTrainer):
        def __init__(self, model, optimizers, args, schedule, topology):
            super().__init__(model, optimizers, sites, args, schedule, topology)

        def _reachable_specs(self, schedule):
            return [GraphSpec(1, 1), GraphSpec(2, LOOP_MAX_ITERATIONS)]

        def _plan(self, spec, calibration, budget_bytes):
            # The tiny model fits raw; recompute a few blocks anyway so the
            # checkpoint wrappers run under capture, and let the looped graph
            # rebuild its recurrence intermediates while the flat one keeps them.
            plan = super()._plan(spec, calibration, budget_bytes)
            return replace(plan, checkpoint_blocks=3, lean=spec.iterations > 1)

    class Rows:
        def __init__(self):
            self.rows = torch.randint(0, args.vocab_size, (args.batch_rows, args.seq_len + 1))

        def batch(self, first, count, device=None):
            return self.rows[first : first + count].to(device=device)

    def agreed(*values: torch.Tensor) -> bool:
        """Whether every rank holds these exact bytes."""
        digest = torch.stack(
            [value.detach().view(torch.int32 if value.element_size() == 4 else torch.int16)
             .to(torch.int64).sum() for value in values]
        )
        low, high = -digest.clone(), digest.clone()
        distributed.all_reduce_(low, maximum=True)
        distributed.all_reduce_(high, maximum=True)
        return torch.equal(-low, high)

    torch.manual_seed(7)
    torch.set_float32_matmul_precision("high")
    started = time.monotonic()
    device = pick_device(None, topology)
    distributed.initialize(topology, device)
    model = DeltaModel(condition_config(args.condition, **model_fields(args))).to(device)
    sites = ParameterSites(model, topology)
    optimizers = build_optimizers(model, owned=sites.owned)
    schedule = build_schedule(args)
    if topology.main:
        print(f"cuda probe | train graph on {topology.world} rank(s)", flush=True)
    runner = ProbeTrainer(model, optimizers, args, schedule, topology)
    data = Rows()
    initial = model.embed_tokens.weight.detach().clone()
    parameters = dict(model.named_parameters())
    normuonh = optimizers[0]
    for spec in runner.states:
        for step in (1, 2):
            apply_schedule(optimizers, schedule, step)
            runner.zero_grad()
            state = runner.begin(spec, args.zloss)
            runner.replay_batch(state, data, step, 0)
            sites.reduce_gradients()
            runner.prepare_optimizer(state)
            assert math.isfinite(state.loss_sum.item())
            assert math.isfinite(sites.gradient_norm())
            gradients = sites.gradients()
            for name in (
                "embed_tokens.weight",
                "fuse_proj.weight",
                "payload_norm.weight",
                "payload_router.query",
                "mtp.block.attn.q_proj.weight",
                "blocks.3.attn.qkv_proj.weight",
                "blocks.3.attn.o_proj.weight",
                "attention_gates.0.weight",
                "blocks.4.attn.control_proj.weight",
                "blocks.0.mlp.shared.down_proj.weight",
            ):
                gradient = gradients[parameters[name]]
                assert gradient.dtype == torch.float32, name
                assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name
                if parameters[name] not in sites.sharded:
                    assert parameters[name].grad is gradient, name
            routed = [
                gradients[expert.down_proj.weight]
                for expert in model.blocks[0].mlp.experts
            ]
            assert sum(gradient.abs().sum() for gradient in routed) > 0
            # Every sharded matrix is its BF16 working copy; its owner's
            # optimizer holds the FP32 master and writes both.
            matrix = parameters["blocks.3.attn.o_proj.weight"]
            assert matrix.dtype == torch.bfloat16
            owned = matrix in sites.owned
            if owned:
                master = normuonh.master_of(matrix)
                assert master.dtype == torch.float32
                assert torch.equal(matrix, master.to(torch.bfloat16))
            with runner.pool_scope():
                for optimizer in optimizers:
                    optimizer.step()
                model.update_expert_bias(state.expert_counts)
            sites.gather_weights()
            model.refresh_shadows()
            if owned:
                assert torch.equal(
                    matrix, normuonh.master_of(matrix).to(torch.bfloat16)
                )
            # After the gather every rank holds the same replicated masters,
            # the same working copies, and the same expert biases.
            assert agreed(
                model.embed_tokens.weight, matrix, model.blocks[0].mlp.expert_bias
            )
    assert not torch.equal(initial, model.embed_tokens.weight)
    runner.zero_grad()

    if topology.main:
        print("cuda probe | eval graph and cached decode", flush=True)
    evaluator = CudaEvalRunner(model, args, runner.pool, topology)
    assert all(math.isfinite(value) for value in evaluator.run(data).values())
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = data.rows[:1, :4].to(device)
        cache = KVCache(model.cfg, batch=1, device=device, dtype=torch.bfloat16)
        e = model.embed_tokens(tokens[:, :3])
        prefill = model.forward_iterations(model.plain_seed(e), e, cache=cache)[-1]
        decoded = model.step(tokens[:, 3:], prefill.payload[:, -1:], cache)
        assert cache.pos == 4 and decoded.h_top.shape == (1, 1, args.dim)
        assert torch.isfinite(decoded.h_top).all()
    torch.cuda.synchronize()
    distributed.shutdown()
    if topology.main:
        print(f"cuda probe passed | {time.monotonic() - started:.1f}s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser("delta probe", description=__doc__)
    parser.add_argument(
        "--ranks",
        type=int,
        default=1,
        help="probe on this many CUDA devices at once through the launcher, "
        "one rank per device, with the same collectives a training step "
        "makes (default: 1)",
    )
    argv = list(sys.argv[1:] if argv is None else argv)
    ranks = parser.parse_args(argv).ranks
    if ranks < 1:
        raise SystemExit("--ranks must be positive")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("delta probe requires CUDA; run pytest for portable tests")
    from . import distributed

    distributed.relaunch(__spec__.name, argv, ranks)
    # Exercise real CUDA kernels and graph replay without compiling a matrix
    # of Inductor specializations. Cached FlexAttention uses its reference.
    with torch.compiler.set_stance("force_eager"), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"flex_attention called without torch\.compile\(\)"
        )
        cuda_probe()


if __name__ == "__main__":
    main()

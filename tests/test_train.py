"""Training controls and one exact checkpoint-resume trajectory."""

from dataclasses import replace

import pytest
import torch

from delta_feedback_experiment.data import DEFAULT_SOURCE, write_synthetic
from delta_feedback_experiment.optim import (
    DEFAULT_NADAM_BETAS,
    DEFAULT_NADAM_LR,
    NorMuonH,
    orthogonalize,
)
from delta_feedback_experiment.train import (
    BATCH_TOKENS,
    CONTRACT,
    GRAD_CLIP_NORM,
    CudaGraphTrainer,
    build_parser,
    build_schedule,
    clip_gradients,
    draw_iterations,
    draw_passes,
    mix,
    model_fields,
    parse_run_args,
    reference_active,
    resolve_run_args,
)


def test_normuonh_applies_nesterov_before_orthogonalization():
    torch.manual_seed(3)
    weight = torch.nn.Parameter(torch.randn(8, 6))
    radius = weight.detach().norm()
    momentum = torch.zeros_like(weight)
    row_moment = torch.zeros(weight.shape[0], 1)
    beta1, beta2, lr, eps = 0.8, 0.7, 0.03, 1e-8
    optimizer = NorMuonH([weight], lr=lr, momentum=beta1, beta2=beta2, eps=eps)

    for gradient in (torch.randn_like(weight), torch.randn_like(weight)):
        weight.grad = gradient.clone()
        optimizer.step()

        momentum = torch.lerp(momentum, gradient, 1 - beta1)
        direction = torch.lerp(gradient, momentum, beta1)
        update = orthogonalize(direction)
        row_moment = torch.lerp(
            row_moment, update.square().mean(dim=-1, keepdim=True), 1 - beta2
        )
    torch.testing.assert_close(weight.norm(), radius)
    assert torch.allclose(optimizer.state[weight]["momentum"], momentum)
    assert torch.allclose(optimizer.state[weight]["row_moment"], row_moment)


def test_global_gradient_clip_uses_one_accumulated_vector():
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([12.0])

    preclip = clip_gradients([first, second])

    assert preclip == pytest.approx(13.0)
    clipped = torch.cat([first.grad, second.grad])
    assert clipped.norm().item() == pytest.approx(GRAD_CLIP_NORM)
    assert clipped.tolist() == pytest.approx(
        [3 * GRAD_CLIP_NORM / 13, 4 * GRAD_CLIP_NORM / 13, 12 * GRAD_CLIP_NORM / 13]
    )


def test_optimizer_materialization_restores_fresh_nadam_state():
    embedding = torch.nn.Parameter(torch.randn(4))
    other = torch.nn.Parameter(torch.randn(4))
    embedding.grad = torch.zeros_like(embedding)
    other.grad = torch.zeros_like(other)
    nadam = torch.optim.NAdam(
        [embedding, other],
        lr=DEFAULT_NADAM_LR,
        betas=DEFAULT_NADAM_BETAS,
    )
    nadam.param_groups[0]["stable_lr"] = DEFAULT_NADAM_LR
    runner = object.__new__(CudaGraphTrainer)
    runner.optimizers = [nadam]

    runner._initialize_optimizers()

    assert [group["lr"] for group in nadam.param_groups] == [DEFAULT_NADAM_LR]
    for parameter in (embedding, other):
        state = nadam.state[parameter]
        assert state["step"].item() == 0
        assert state["mu_product"].item() == 1
        assert torch.count_nonzero(state["exp_avg"]) == 0
        assert torch.count_nonzero(state["exp_avg_sq"]) == 0


def test_resolved_arguments_pin_what_the_operator_fixed():
    """A resume validates what the operator pinned and inherits the rest: an
    explicit --scale pins its fields, an explicit --tokens-per-param pins the
    derived steps, and an untyped schedule stays open for the checkpoint."""
    parser = build_parser()
    args, pinned = resolve_run_args(parser, ["x", "--resume"])
    assert pinned == frozenset({"resume"}) and args.steps is None
    _, pinned = resolve_run_args(parser, ["x", "--resume", "--scale", "bridge"])
    assert {
        "scale",
        "dim",
        "layers",
        "seq_len",
        "batch_rows",
        "micro_rows",
        "expert_intermediate",
        "num_routed_experts",
        "experts_per_token",
    } <= pinned
    assert "steps" not in pinned
    _, pinned = resolve_run_args(parser, ["x", "--tokens-per-param", "400"])
    assert {"tokens_per_param", "steps"} <= pinned and "dim" not in pinned
    _, pinned = resolve_run_args(parser, ["x", "--seq-len", "2048"])
    assert {"seq_len", "batch_rows"} <= pinned


def test_preset_accounting_counts_shared_and_selected_experts():
    scale, dim, selected, routed, total, active = (
        "screen",
        768,
        3,
        15,
        630606216,
        200919432,
    )
    from delta_feedback_experiment.model import DeltaModel, condition_config

    args = parse_run_args(["geometry", "--scale", scale])
    cfg = condition_config("fl", **model_fields(args))
    assert cfg.layers == 16 and cfg.core_layers == range(4, 12)
    assert cfg.executed_layers(4) == 40 and cfg.routing_blocks == 4
    assert cfg.dim == dim and cfg.expert_intermediate == 832
    assert cfg.heads * cfg.head_dim == dim
    assert cfg.heads == 2 * cfg.kv_heads
    assert cfg.pkda_heads * cfg.pkda_head_dim * 3 == dim * 5
    assert cfg.mup_ratio == 1536 / dim
    assert cfg.expert_lr_scale == pytest.approx((8 / (selected + 1)) ** 0.5)
    assert cfg.experts_per_token == selected and cfg.num_routed_experts == routed
    assert (selected + 1) * 832 * 3 == 13 * dim
    assert routed + 1 == 4 * (selected + 1)
    with torch.device("meta"):
        model = DeltaModel(cfg)
    assert sum(p.numel() for p in model.parameters()) == total
    assert len(model.expert_banks) == 17
    assert all(
        bank.intermediate == 832
        and len(bank.experts) == routed
        and bank.experts_per_token == selected
        for bank in model.expert_banks
    )
    assert reference_active(args) == active
    assert args.steps * BATCH_TOKENS >= 25 * active
    assert (args.steps - 1) * BATCH_TOKENS < 25 * active


def test_keyed_schedule_draws_are_reproducible():
    args = parse_run_args(["draws", "--steps", "100"])
    schedule = build_schedule(args)
    assert schedule.total == args.steps
    assert schedule.rate_at(schedule.total, 1.0) < 1e-6
    for step in (1, 17, 99):
        assert mix(0, step, 1) == mix(0, step, 1)
        assert mix(0, step, 1) != mix(0, step, 2)
        iterations = draw_iterations(args, step, True)
        assert iterations == draw_iterations(args, step, True)
        assert 1 <= iterations <= args.loop_max_iterations
        assert draw_iterations(args, step, False) == 1
        passes = draw_passes(args, step, args.steps)
        assert passes == draw_passes(args, step, args.steps)
        assert 1 <= passes <= 3


def assert_identical(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_identical(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_identical(a, b)
    else:
        assert left == right


def test_training_resume_preserves_the_exact_next_update(tmp_path, monkeypatch):
    from transformer_experiments.spool import Spool

    from delta_feedback_experiment import cli
    from delta_feedback_experiment import train as trainer

    write_synthetic(
        tmp_path / "data" / DEFAULT_SOURCE, train_tokens=80, val_tokens=20, vocab=31
    )
    settings = {
        "condition": "fl",
        "data-root": tmp_path / "data",
        "out-dir": tmp_path / "runs",
        "vocab-size": 31,
        "dim": 16,
        "layers": 16,
        "heads": 2,
        "kv-heads": 2,
        "head-dim": 8,
        "expert-intermediate": 8,
        "num-routed-experts": 3,
        "experts-per-token": 2,
        "pkda-heads": 2,
        "pkda-head-dim": 8,
        "seq-len": 4,
        "batch-rows": 2,
        "micro-rows": 1,
        "loop-iterations": 2,
        "loop-max-iterations": 2,
        "steps": 2,
        "warmup-frac": 0,
        "cooldown-frac": 0,
        "feedback-start": 0,
        "three-pass": 0,
        "eval-every": 2,
        "eval-rows": 1,
        "snapshot-every": 1,
        "device": "cpu",
    }
    flags = [
        item for key, value in settings.items() for item in (f"--{key}", str(value))
    ]
    monkeypatch.setattr(
        trainer, "draw_iterations", lambda args, step, loop: 2 if loop else 1
    )
    full = trainer.train(["full", *flags])
    half = trainer.train(["half", *flags, "--max-steps", "1"])
    assert half["step"] == 1
    Spool(replace(cli.LAYOUT, root=tmp_path), cli.PIPELINE).move("half", "renamed")
    with pytest.raises(ValueError, match="conflicts"):
        trainer.train(["renamed", *flags, "--resume", "--experts-per-token", "1"])
    resumed = trainer.train(["renamed", *flags, "--resume"])
    for key in ("step", "loss", "val", "val_fused", "val_mtp", "val_mtp_fused"):
        assert resumed[key] == full[key]
    complete = trainer.read_checkpoint(tmp_path / "runs/full.pt.2")
    restored = trainer.read_checkpoint(tmp_path / "runs/renamed.pt.2")
    assert complete["version"] == CONTRACT.version
    assert_identical(complete["state"], restored["state"])
    assert_identical(complete["optimizer"], restored["optimizer"])
    biases = [
        value
        for name, value in restored["state"].items()
        if name.endswith("expert_bias")
    ]
    assert any(value.count_nonzero() for value in biases)
    assert restored["state"]["mtp.block.mlp.expert_bias"].count_nonzero()

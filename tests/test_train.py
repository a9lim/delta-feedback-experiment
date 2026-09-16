"""Training controls and one exact checkpoint-resume trajectory."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from delta_feedback_experiment.data import DEFAULT_SOURCE, write_synthetic
from delta_feedback_experiment.model import (
    DEFAULT_LOOP_ITERATIONS,
    LOOP_MAX_ITERATIONS,
    ModelConfig,
    condition_config,
)
from delta_feedback_experiment.optim import (
    DEFAULT_NADAM_BETAS,
    DEFAULT_NADAM_LR,
    NorMuonH,
    orthogonalize,
)
from delta_feedback_experiment.train import (
    BATCH_TOKENS,
    CONTRACT,
    CudaGraphTrainer,
    GraphSpec,
    Trainer,
    build_parser,
    build_schedule,
    draw_recurrence,
    micro_draws,
    mix,
    model_fields,
    parse_run_args,
    reference_active,
    replay_widths,
    resolve_run_args,
    step_shape,
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
    runner = object.__new__(Trainer)
    runner.optimizers = [nadam]

    runner._initialize_optimizers()

    assert [group["lr"] for group in nadam.param_groups] == [DEFAULT_NADAM_LR]
    for parameter in (embedding, other):
        state = nadam.state[parameter]
        assert state["step"].item() == 0
        assert state["mu_product"].item() == 1
        assert torch.count_nonzero(state["exp_avg"]) == 0
        assert torch.count_nonzero(state["exp_avg_sq"]) == 0


@pytest.mark.parametrize("scale", ["screen", "bridge", "flagship", "extension"])
def test_resolved_arguments_pin_what_the_operator_fixed(scale):
    """A resume validates what the operator pinned and inherits the rest: an
    explicit --scale pins its fields, an explicit --tokens-per-param pins the
    derived steps, and an untyped schedule stays open for the checkpoint."""
    parser = build_parser()
    args, pinned = resolve_run_args(parser, ["x", "--resume"])
    assert pinned == frozenset({"resume"}) and args.steps is None
    assert args.loop_iterations == DEFAULT_LOOP_ITERATIONS == 2
    args, pinned = resolve_run_args(parser, ["x", "--resume", "--scale", scale])
    assert args.loop_iterations == 2
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
    assert "loop_iterations" not in pinned
    assert "steps" not in pinned
    args, pinned = resolve_run_args(
        parser, ["x", "--steps", "1", "--loop-iterations", "1", "--scale", scale]
    )
    assert args.loop_iterations == 1 and "loop_iterations" in pinned
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
        638862216,
        209175432,
    )
    from delta_feedback_experiment.cli import stream_target
    from delta_feedback_experiment.data import (
        CANONICAL_TARGET_TOKENS,
        CANONICAL_VAL_TOKENS,
    )
    from delta_feedback_experiment.model import (
        DeltaModel,
        ModelConfig,
        condition_config,
    )

    args = parse_run_args(["geometry", "--scale", scale])
    cfg = condition_config("fl", **model_fields(args))
    assert cfg.loop_iterations == ModelConfig().loop_iterations == 2
    assert cfg.layers == 16 and cfg.pkda_layers == 12
    assert cfg.routing_blocks == 4
    assert cfg.dim == dim and cfg.expert_intermediate == 832
    assert cfg.head_dim == ModelConfig().head_dim == 192
    assert cfg.heads * cfg.head_dim == 2 * dim
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
    # The canonical corpus identity stays fixed when parameter accounting
    # shrinks; it still covers the current 400-token-per-parameter schedule.
    assert CANONICAL_TARGET_TOKENS >= stream_target(scale, 400, CANONICAL_VAL_TOKENS)


def test_keyed_schedule_draws_are_reproducible():
    args = parse_run_args(["draws", "--steps", "100"])
    schedule = build_schedule(args)
    assert schedule.total == args.steps
    assert schedule.rate_at(schedule.total, 1.0) < 1e-6
    for step in (1, 17, 99):
        assert mix(0, step, 1) == mix(0, step, 1)
        assert mix(0, step, 1) != mix(0, step, 2)
        roll = draw_recurrence(args, step, args.steps)
        assert roll == draw_recurrence(args, step, args.steps)
        assert roll in {(1, 1), (2, 2), (3, 2), (2, 3)}
        # Every condition projects the same roll onto the letters it has.
        for condition in ("f", "l", "fl"):
            cfg = condition_config(condition)
            passes, iterations = step_shape(args, step, args.steps, cfg)
            assert passes == (roll[0] if cfg.feedback else 1)
            assert iterations == (roll[1] if cfg.loop else 1)
            assert iterations <= LOOP_MAX_ITERATIONS == 3


def test_recurrence_roll_assigns_exact_probability_intervals(monkeypatch):
    args = parse_run_args(["draws", "--steps", "100", "--recurrence-start", "0"])
    draws = iter((index + 0.5) / 100 for index in range(100))
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *args, **kwargs: torch.tensor(next(draws), dtype=torch.float64),
    )
    rolls = [draw_recurrence(args, step, 100) for step in range(1, 101)]
    assert rolls.count((3, 2)) == 12
    assert rolls.count((2, 3)) == 12
    assert rolls.count((2, 2)) == 76

    for draw, expected in (
        (0.0, (3, 2)), (0.12 - 1e-9, (3, 2)), (0.12, (2, 3)),
        (0.24 - 1e-9, (2, 3)), (0.24, (2, 2)), (1.0 - 1e-9, (2, 2)),
    ):
        monkeypatch.setattr(
            torch,
            "rand",
            lambda *args, **kwargs: torch.tensor(draw, dtype=torch.float64),
        )
        assert draw_recurrence(args, 1, 100) == expected
    # Before the boundary no step is rolled; after it none is single-column.
    args = parse_run_args(["draws", "--steps", "100"])
    assert all(draw_recurrence(args, step, 100) == (1, 1) for step in range(1, 76))
    assert all(draw_recurrence(args, step, 100) == (2, 2) for step in range(76, 101))
    with pytest.raises(SystemExit):
        parse_run_args(["draws", "--steps", "100", "--three-rate", "0.51"])


def test_recurrence_rolls_are_shared_across_scales_and_independent_of_eval_depth():
    steps = range(1, 129)
    expected = None
    for scale in ("screen", "bridge", "flagship", "extension"):
        args = parse_run_args([
            "draws", "--steps", "128", "--scale", scale, "--recurrence-start", "0",
        ])
        assert args.loop_iterations == 2
        for eval_depth in (1, 2, 3):
            args.loop_iterations = eval_depth
            cfg = condition_config("fl", **model_fields(args))
            assert cfg.loop_iterations == eval_depth
            rolls = [draw_recurrence(args, step, 128) for step in steps]
            if expected is None:
                expected = rolls
            assert rolls == expected
    assert set(expected) == {(2, 2), (3, 2), (2, 3)}


def test_roll_rng_is_addressed_by_step_and_preserves_other_streams():
    args = parse_run_args([
        "draws", "--steps", "128", "--seq-len", "4", "--recurrence-start", "0",
    ])
    expected = [draw_recurrence(args, step, 128) for step in range(1, 129)]
    micro = micro_draws(args, 17, 0, 3, 2, 2, 16, torch.device("cpu"))
    global_state = torch.get_rng_state().clone()
    # Reordered calls model restarting at an arbitrary step without carrying
    # sampler state, while unrelated microbatch draws intervene.
    for step in reversed(range(1, 129)):
        assert draw_recurrence(args, step, 128) == expected[step - 1]
    repeated = micro_draws(args, 17, 0, 3, 2, 2, 16, torch.device("cpu"))
    assert torch.equal(global_state, torch.get_rng_state())
    for left, right in zip(micro, repeated, strict=True):
        assert torch.equal(left, right)
    args.data_seed += 1
    assert [draw_recurrence(args, step, 128) for step in range(1, 129)] != expected


def test_reachable_graphs_cover_every_rolled_shape_without_cuda():
    args = parse_run_args([
        "graphs", "--condition", "fl", "--steps", "400",
        "--recurrence-start", "0.25", "--three-rate", "0.3",
    ])
    runner = object.__new__(CudaGraphTrainer)
    runner.args = args
    runner.model = SimpleNamespace(cfg=condition_config("fl", **model_fields(args)))
    schedule = build_schedule(args)
    assert runner._reachable_specs(schedule) == [
        GraphSpec(1, 1), GraphSpec(2, 2), GraphSpec(2, 3), GraphSpec(3, 2),
    ]
    runner.model.cfg = replace(runner.model.cfg, loop=False)
    assert runner._reachable_specs(schedule) == [
        GraphSpec(1, 1), GraphSpec(2, 1), GraphSpec(3, 1),
    ]
    runner.model.cfg = replace(runner.model.cfg, feedback=False, loop=True)
    assert runner._reachable_specs(schedule) == [
        GraphSpec(1, 1), GraphSpec(1, 2), GraphSpec(1, 3),
    ]


@pytest.mark.parametrize("depth", [0, 4])
def test_eval_depth_stays_within_the_trained_range(depth):
    with pytest.raises(ValueError, match="loop.*iterations"):
        ModelConfig(loop=True, loop_iterations=depth)
    with pytest.raises(SystemExit):
        parse_run_args(["draws", "--steps", "1", "--loop-iterations", str(depth)])


def test_keyed_jitter_covers_single_and_final_passes_and_replay_buffers():
    args = parse_run_args(["draws", "--steps", "3", "--seq-len", "4"])
    device = torch.device("cpu")
    for count in (1, 3):
        prefix, jitter, loop = micro_draws(args, 2, 0, count, 1, 2, 16, device)
        assert loop is None
        assert prefix.shape == (count - 1, 2)
        assert jitter.shape == (count, 2, args.seq_len + 1, 16)
        assert torch.all((prefix >= 1) & (prefix < args.seq_len))
        assert torch.all(jitter.abs() <= args.jitter)
        assert jitter.abs().max() > 0.9 * args.jitter
        assert all(draw.count_nonzero() for draw in jitter)
        destination_prefix, destination_jitter = (
            torch.empty_like(prefix), torch.empty_like(jitter)
        )
        repeated = micro_draws(
            args, 2, 0, count, 1, 2, 16, device,
            prefix_out=destination_prefix, jitter_out=destination_jitter,
            generator=torch.Generator(),
        )
        assert repeated[0] is destination_prefix and repeated[1] is destination_jitter
        torch.testing.assert_close(repeated[0], prefix, atol=0, rtol=0)
        torch.testing.assert_close(repeated[1], jitter, atol=0, rtol=0)
        _, changed, _ = micro_draws(args, 2, 2, count, 1, 2, 16, device)
        assert not torch.equal(jitter, changed)
        # Every row is keyed by its own global row: a two-row draw is the two
        # one-row draws side by side, whatever the replay width or rank.
        for row in (0, 1):
            single_prefix, single_jitter, _ = micro_draws(
                args, 2, row, count, 1, 1, 16, device
            )
            torch.testing.assert_close(single_prefix[:, 0], prefix[:, row], atol=0, rtol=0)
            torch.testing.assert_close(single_jitter[:, 0], jitter[:, row], atol=0, rtol=0)
        # Looped columns draw their jitter after the pass draws, so a condition
        # with l shares its prefix and pass jitter with the one without it.
        looped_prefix, looped_jitter, loop = micro_draws(
            args, 2, 0, count, 3, 2, 16, device
        )
        assert torch.equal(looped_prefix, prefix) and torch.equal(looped_jitter, jitter)
        assert loop.shape == (count, 2, 2, args.seq_len + 1, 16)
        assert torch.all(loop.abs() <= args.jitter) and loop.count_nonzero()
        destination_loop = torch.empty_like(loop)
        again = micro_draws(
            args, 2, 0, count, 3, 2, 16, device,
            prefix_out=destination_prefix, jitter_out=destination_jitter,
            loop_jitter_out=destination_loop, generator=torch.Generator(),
        )
        assert again[2] is destination_loop
        torch.testing.assert_close(again[2], loop, atol=0, rtol=0)


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
        "steps": 2,
        "warmup-frac": 0,
        "cooldown-frac": 0,
        "recurrence-start": 0,
        "three-rate": 0,
        "eval-every": 2,
        "eval-rows": 1,
        "snapshot-every": 1,
        "device": "cpu",
    }
    flags = [
        item for key, value in settings.items() for item in (f"--{key}", str(value))
    ]
    sampled_shapes = []
    original_shape = trainer.step_shape

    def record_shape(args, step, total, cfg):
        shape = original_shape(args, step, total, cfg)
        sampled_shapes.append((step, shape))
        return shape

    monkeypatch.setattr(trainer, "step_shape", record_shape)
    full = trainer.train(["full", *flags])
    half = trainer.train(["half", *flags, "--max-steps", "1"])
    assert half["step"] == 1
    # Resume must restore a nonzero auxiliary controller state. Later updates
    # can legitimately bring these signed count corrections back to zero.
    halfway = trainer.read_checkpoint(tmp_path / "runs/half.pt.1")
    assert halfway["state"]["mtp.block.mlp.expert_bias"].count_nonzero()
    Spool(replace(cli.LAYOUT, root=tmp_path), cli.PIPELINE).move("half", "renamed")
    with pytest.raises(ValueError, match="conflicts"):
        trainer.train(["renamed", *flags, "--resume", "--experts-per-token", "1"])
    # Saved geometry and evaluation depth override unpinned new-run defaults.
    resume_flags = [
        item
        for key, value in settings.items()
        if key not in {"head-dim", "loop-iterations"}
        for item in (f"--{key}", str(value))
    ]
    resumed = trainer.train(["renamed", *resume_flags, "--resume"])
    for key in ("step", "loss", "val", "val_fused", "val_mtp", "val_mtp_fused"):
        assert resumed[key] == full[key]
    complete = trainer.read_checkpoint(tmp_path / "runs/full.pt.2")
    restored = trainer.read_checkpoint(tmp_path / "runs/renamed.pt.2")
    assert complete["version"] == CONTRACT.version == 42
    assert (
        complete["args"]["loop_iterations"] == restored["args"]["loop_iterations"] == 2
    )
    assert sampled_shapes == [(1, (2, 2)), (2, (2, 2)), (1, (2, 2)), (2, (2, 2))]
    assert "recurrence_start" in restored["args"] and "three_rate" in restored["args"]
    assert "loop_max_iterations" not in restored["args"]
    assert_identical(complete["state"], restored["state"])
    assert_identical(complete["optimizer"], restored["optimizer"])
    biases = [
        value
        for name, value in restored["state"].items()
        if name.endswith("expert_bias")
    ]
    assert any(value.count_nonzero() for value in biases)


def test_replay_plan_fits_rows_then_recomputes_blocks():
    """Every graph takes the largest divisor of the rank's rows that fits raw;
    a shortfall recomputes exactly as many block invocations as the calibrated
    release of one recomputed block needs, never more than are eligible."""
    from delta_feedback_experiment.model import condition_config
    from delta_feedback_experiment.train import (
        Calibration,
        GraphSpec,
        block_invocations,
        plan_replay,
    )

    args = parse_run_args(["plan", "--condition", "fl", "--steps", "5"])
    cfg = condition_config("fl", **model_fields(args))
    assert block_invocations(cfg, GraphSpec(1, 1)) == (17, 13)
    assert block_invocations(cfg, GraphSpec(2, 1)) == (34, 26)
    assert block_invocations(cfg, GraphSpec(1, 3)) == (51, 39)
    assert block_invocations(cfg, GraphSpec(2, 3)) == (102, 78)
    block = 300 * 2**20
    release = 320 * 2**20
    # A retaining recurrence keeps 30% more per block than a rebuilding one.
    calibration = Calibration(1.3 * block, block, release)

    def budget(rows, spec):
        blocks, _ = block_invocations(cfg, spec)
        return rows * blocks * block

    def plan(spec, budget_bytes):
        return plan_replay(cfg, args, spec, calibration, budget_bytes)

    def chosen(replay):
        return replay.rows_per_replay, "lean" if replay.lean else "full"

    flat = GraphSpec(1, 1)
    assert replay_widths(16, 1) == [16, 8, 4, 2, 1]
    assert replay_widths(12, 2) == [12, 6, 4, 2]
    with pytest.raises(ValueError, match="multiple"):
        replay_widths(6, 4)
    # Keeping the intermediates at a narrower replay beats rebuilding them at
    # a wider one; the one-row replay is the last resort either way.
    assert chosen(plan(flat, budget(8, flat))) == (4, "full")
    assert chosen(plan(flat, budget(7, flat))) == (4, "full")
    assert chosen(plan(flat, budget(2, flat))) == (2, "lean")
    assert chosen(plan(flat, budget(2, flat) - 1)) == (1, "full")
    assert chosen(plan(flat, budget(1, flat))) == (1, "lean")
    looped = GraphSpec(1, 2)
    assert chosen(plan(looped, budget(8, looped))) == (4, "full")
    two = GraphSpec(2, 1)
    generous = plan(two, budget(8, two))
    assert (generous.rows_per_replay, generous.checkpoint_blocks) == (4, 0)
    # A rank's rows bound the width: sixteen rows of a step never replay more.
    sixteen = plan_replay(cfg, args, flat, calibration, budget(128, flat), 16)
    assert chosen(sixteen) == (16, "full")
    # A shortfall is sized by what a recomputed block releases, not by the
    # average block: five releases cover a five-average-block shortfall.
    short = plan(two, budget(1, two) - 5 * block)
    assert (short.rows_per_replay, short.checkpoint_blocks, short.lean) == (1, 5, True)
    assert plan(two, budget(1, two) - 5 * release).checkpoint_blocks == 5
    assert plan(two, budget(1, two) - 5 * release - 1).checkpoint_blocks == 6
    starved = plan(GraphSpec(1, 3), 0)
    assert starved.checkpoint_blocks == starved.eligible_blocks == 39


def test_mode_activity_keeps_replicated_parameters_with_zero_gradients():
    """Warm-up decides sharded activity from the sinks, but a replicated
    parameter whose warm-up gradient is exactly zero (a routing site's
    key-norm gain behind a zero-initialized query) stays active."""
    from delta_feedback_experiment.train import mode_activity

    gain = torch.nn.Parameter(torch.ones(4))
    query = torch.nn.Parameter(torch.zeros(4))
    dense = torch.nn.Parameter(torch.ones(4, 4))
    idle = torch.nn.Parameter(torch.ones(4, 4))
    expert = torch.nn.Parameter(torch.ones(4, 4))

    class Bank(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = expert

    gradients = {
        gain: torch.zeros(4),
        query: torch.ones(4),
        dense: torch.ones(4, 4),
        idle: torch.zeros(4, 4),
        expert: torch.zeros(4, 4),
    }
    active = mode_activity(gradients, frozenset({dense, idle, expert}), [Bank()])
    assert active == {gain, query, dense, expert}


def test_pick_device_refuses_a_foreign_index_under_several_ranks():
    from delta_feedback_experiment.distributed import Topology
    from delta_feedback_experiment.train import pick_device

    with pytest.raises(ValueError, match="own local device"):
        pick_device("cuda:0", Topology(rank=3, world=8, local_rank=3))
    assert pick_device("cpu", Topology(rank=3, world=8, local_rank=3)).type == "cpu"


def test_plan_graphs_sets_inputs_aside_at_the_planned_widths():
    """The inputs follow the widths and the widths the budget: planning
    settles where no graph's inputs exceed what was set aside, starting from
    the smallest replays rather than the widest."""
    from types import SimpleNamespace

    from delta_feedback_experiment.train import (
        GraphSpec,
        ReplayPlan,
        input_bytes,
        plan_graphs,
    )

    args = SimpleNamespace(seq_len=8, dim=4, micro_rows=1)
    specs = [GraphSpec(1, 1), GraphSpec(2, 1)]
    unit = input_bytes(args, GraphSpec(1, 1), 1)
    free = 40 * unit
    budgets = []

    def budget_for(inputs):
        budgets.append(inputs)
        return free - inputs

    def plan(spec, budget):
        # A replay of ``rows`` rows needs rows * columns units of budget.
        columns = spec.n_passes * spec.iterations
        rows = next(r for r in (8, 4, 2, 1) if r * columns * unit <= budget)
        return ReplayPlan(rows, 0, 0, 0.0)

    plans, inputs, budget = plan_graphs(args, specs, budget_for, plan)
    widths = {spec: plans[spec].rows_per_replay for spec in specs}
    assert inputs >= sum(input_bytes(args, s, widths[s]) for s in specs)
    assert budget == free - inputs
    assert widths[GraphSpec(2, 1)] <= widths[GraphSpec(1, 1)]
    assert budgets[0] == sum(input_bytes(args, s, 1) for s in specs)
    assert len(budgets) >= 2


def test_input_bytes_counts_tokens_prefix_and_both_jitters():
    from types import SimpleNamespace

    from delta_feedback_experiment.train import GraphSpec, input_bytes

    args = SimpleNamespace(seq_len=8, dim=4)
    columns = 9
    assert input_bytes(args, GraphSpec(1, 1), 2) == 2 * columns * 8 + 2 * columns * 4 * 2
    assert input_bytes(args, GraphSpec(3, 2), 2) == (
        2 * columns * 8 + 2 * 2 * 8 + 3 * 2 * 2 * columns * 4 * 2
    )

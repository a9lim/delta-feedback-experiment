"""Offline checks for the data pipeline, optimizer stack, and trainer.

The heavyweight claim here is resume exactness: a run interrupted and
resumed must be bit-identical to an uninterrupted one — the property the
whole paired-comparison design leans on for multi-day jobe runs.
"""

import numpy as np
import pytest
import torch

from delta_feedback_experiment.data import TokenData, write_synthetic
from delta_feedback_experiment.optim import NorMuon, orthogonalize, split_parameters
from delta_feedback_experiment.train import (
    automatic_checkpoint,
    build_parser,
    build_schedule,
    draw_passes,
    mix,
    route_summary,
    train,
)

TINY_ARGS = [
    "--vocab-size",
    "97",
    "--dim",
    "32",
    "--layers",
    "2",
    "--heads",
    "2",
    "--kv-heads",
    "2",
    "--head-dim",
    "16",
    "--intermediate",
    "64",
    "--pkda-heads",
    "2",
    "--pkda-head-dim",
    "16",
    "--seq-len",
    "16",
    "--batch-rows",
    "4",
    "--micro-rows",
    "2",
    "--steps",
    "8",
    "--warmup-steps",
    "2",
    "--cooldown-frac",
    "0.25",
    "--feedback-start",
    "0.5",
    "--feedback-batch-prob",
    "1",
    "--eval-every",
    "4",
    "--snapshot-every",
    "100",
    "--eval-rows",
    "4",
    "--device",
    "cpu",
]


def corpus(tmp_path):
    directory = tmp_path / "tokens"
    write_synthetic(directory, train_tokens=1200, val_tokens=120, vocab=97)
    return directory


# -- data ----------------------------------------------------------------------


def test_token_data_stitches_shards(tmp_path):
    directory = corpus(tmp_path)
    data = TokenData.load(directory, "train", seq_len=16)
    whole = np.concatenate(
        [
            np.fromfile(directory / "train.0000.bin", dtype=np.uint32),
            np.fromfile(directory / "train.0001.bin", dtype=np.uint32),
        ]
    )
    assert data.total_tokens == whole.size
    batch = data.batch(0, data.rows)
    assert torch.equal(
        batch.flatten(), torch.from_numpy(whole[: data.rows * 17].astype(np.int64))
    )
    with pytest.raises(IndexError):
        data.batch(data.rows, 1)


# -- optimizer -----------------------------------------------------------------


def test_orthogonalize_singular_values():
    """NS5 with Muon's loose quintic: the bulk lands near 1 (median well
    up, max bounded); a square Gaussian's smallest values lag — that's
    the known worst case, not a bug."""
    torch.manual_seed(0)
    for shape in ((8, 24), (24, 8), (16, 16)):
        matrix = torch.randn(shape)
        singular = torch.linalg.svdvals(orthogonalize(matrix))
        assert singular.max() < 1.4, shape
        assert singular.median() > 0.65, shape
        assert singular.min() > 0.2, shape


def test_normuon_descends():
    torch.manual_seed(0)
    target = torch.randn(16, 8)
    weight = torch.nn.Parameter(torch.zeros(16, 8))
    optimizer = NorMuon([weight], lr=0.1, weight_decay=0.0)
    first = None
    for _ in range(300):
        loss = (weight - target).square().mean()
        first = loss.item() if first is None else first
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Constant-RMS updates oscillate near the optimum at ~(0.2*lr) scale.
    assert loss.item() < 0.02 * first


def test_normuon_rejects_vectors():
    with pytest.raises(ValueError):
        NorMuon([torch.nn.Parameter(torch.zeros(8))])


def test_batched_normuon_matches_independent_parameters():
    torch.manual_seed(4)
    left = [torch.nn.Parameter(torch.randn(12, 8)) for _ in range(3)]
    right = [torch.nn.Parameter(parameter.detach().clone()) for parameter in left]
    gradients = [torch.randn_like(parameter) for parameter in left]
    batched = NorMuon(left, lr=0.03, weight_decay=0.02)
    singles = [NorMuon([parameter], lr=0.03, weight_decay=0.02) for parameter in right]
    for parameter, gradient in zip(left, gradients, strict=True):
        parameter.grad = gradient.clone()
    for parameter, gradient in zip(right, gradients, strict=True):
        parameter.grad = gradient.clone()
    batched.step()
    for optimizer in singles:
        optimizer.step()
    for actual, expected in zip(left, right, strict=True):
        assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_attention_gates_use_adam_while_fbt_fusion_uses_normuon():
    from delta_feedback_experiment.model import DFModel, arm_config

    model = DFModel(
        arm_config(
            "df",
            vocab_size=97,
            dim=32,
            layers=4,
            heads=2,
            kv_heads=2,
            head_dim=16,
            intermediate=64,
            pkda_heads=2,
            pkda_head_dim=16,
            max_seq_len=17,
        )
    )
    normuon, adam = split_parameters(model)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    normuon_names = {names[id(parameter)] for parameter in normuon}
    adam_names = {names[id(parameter)] for parameter in adam}

    assert "attention_gates.0.weight" in adam_names
    assert "embed_tokens.weight" in adam_names
    assert "blocks.0.attn_router.query" in adam_names
    assert "blocks.0.attn_router.key_norm.weight" in adam_names
    assert "fuse_value.weight" in normuon_names
    assert "fuse_gate.weight" in normuon_names
    assert "blocks.0.attn.q_proj.weight" in normuon_names
    assert "blocks.3.attn.qkv_proj.weight" in normuon_names
    assert "blocks.0.attn.decay_down.weight" in adam_names
    assert "blocks.0.attn.precond_decay_proj.weight" in adam_names
    assert "blocks.0.attn.output_gate_up.weight" in adam_names
    assert "blocks.0.attn.q_conv.weight" in adam_names
    assert "blocks.0.mlp.gate_up_proj.weight" in normuon_names
    assert not (normuon_names & adam_names)
    assert len(normuon_names) + len(adam_names) == len(names)


def test_route_summary_reports_universal_nulls_and_payload_seed():
    from types import SimpleNamespace

    from delta_feedback_experiment.model import DFModel, arm_config

    class Validation:
        rows = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(3))

        def batch(self, first, count, device=None):
            rows = self.rows[first : first + count]
            return rows.to(device) if device is not None else rows

    args = SimpleNamespace(eval_rows=2)
    for arm in ("mhdb", "df"):
        model = DFModel(
            arm_config(
                arm,
                vocab_size=97,
                dim=32,
                layers=2,
                heads=2,
                kv_heads=2,
                head_dim=16,
                intermediate=64,
                pkda_heads=2,
                pkda_head_dim=16,
                max_seq_len=16,
            )
        )
        records = route_summary(model, Validation(), args, torch.device("cpu"))
        assert records
        assert all("null" in record and record["null_rms"] == 0 for record in records)
        assert all("seed" in record for record in records)


def test_execution_telemetry_supports_a_pkda_first_layer():
    from types import SimpleNamespace

    from delta_feedback_experiment.model import (
        DFModel,
        arm_config,
        flash_attn_func,
    )
    from delta_feedback_experiment.train import GraphSpec, execution_fields

    model = DFModel(
        arm_config(
            "base",
            vocab_size=97,
            dim=32,
            layers=4,
            heads=2,
            kv_heads=2,
            head_dim=16,
            intermediate=64,
            pkda_heads=2,
            pkda_head_dim=16,
            max_seq_len=17,
        )
    )
    trainer = SimpleNamespace(
        states={
            GraphSpec(1, False, False): None,
            GraphSpec(1, True, False): None,
        }
    )
    evaluator = SimpleNamespace(states={4: None})
    fields = execution_fields(model, trainer, evaluator)
    assert fields == {
        "flash": int(flash_attn_func is not None),
        "cce": 1,
        "cuda_graphs": 3,
        "eval_graphs": 1,
        "checkpoint_modes": 0,
    }


# -- schedule and derived randomness -------------------------------------------


def test_build_schedule_screen_shape():
    args = build_parser().parse_args(["x"])  # defaults: 6700 steps
    schedule = build_schedule(args)
    assert schedule.spans == (200, 0, 4825, 1675)
    assert schedule.total == 6700
    assert schedule.phase(200)[0] == "warmup"
    assert schedule.phase(201)[0] == "heat"
    assert schedule.phase(5026)[0] == "cooldown"
    assert schedule.rate_at(5025, 1.0) == 1.0
    assert schedule.rate_at(6700, 1.0) < 1e-6


def test_log_every_flag_is_removed():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["x", "--log-every", "2"])


@pytest.mark.parametrize(
    "flag",
    ["--feedback-start", "--feedback-batch-prob", "--three-pass"],
)
@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf"])
def test_pass_probabilities_reject_values_outside_unit_interval(flag, value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["x", flag, value])


def test_mix_is_stable():
    assert mix(0, 5, 1) == mix(0, 5, 1)
    assert mix(0, 5, 1) != mix(0, 5, 2)
    assert mix(1, 5, 1) != mix(0, 5, 1)


def test_pass_mixture_fractions():
    args = build_parser().parse_args(["x", "--steps", "4000"])
    assert args.feedback_start == 0.5
    assert args.feedback_batch_prob == 0.5
    assert args.three_pass == 0.12

    first_half = [draw_passes(args, step, 4000) for step in range(1, 2001)]
    second_half = [draw_passes(args, step, 4000) for step in range(2001, 4001)]
    assert set(first_half) == {1}

    second_counts = {k: second_half.count(k) for k in (1, 2, 3)}
    assert 0.45 < second_counts[1] / 2000 < 0.55
    assert 0.39 < second_counts[2] / 2000 < 0.49
    assert 0.04 < second_counts[3] / 2000 < 0.08

    counts = {1: 0, 2: 0, 3: 0}
    for n_passes in first_half + second_half:
        counts[n_passes] += 1
    assert 0.72 < counts[1] / 4000 < 0.78
    assert 0.20 < counts[2] / 4000 < 0.24
    assert 0.02 < counts[3] / 4000 < 0.04
    mean_passes = sum(n_passes * count for n_passes, count in counts.items()) / 4000
    assert mean_passes == pytest.approx(1.28, abs=0.03)


# -- end-to-end ----------------------------------------------------------------


def run(tmp_path, tag, extra):
    directory = (
        corpus(tmp_path) if not (tmp_path / "tokens").exists() else tmp_path / "tokens"
    )
    return train(
        [
            tag,
            "--data-dir",
            str(directory),
            "--out-dir",
            str(tmp_path / "runs"),
            *TINY_ARGS,
            *extra,
        ]
    )


@pytest.mark.parametrize("arm", ["vanilla", "base", "mhdb", "fbt", "df"])
def test_tiny_run_completes(tmp_path, capsys, arm):
    summary = run(tmp_path, f"t-{arm}", ["--arm", arm])
    assert summary["step"] == 8
    assert np.isfinite(summary["loss"])
    assert np.isfinite(summary["val"])
    if arm in ("fbt", "df"):
        assert np.isfinite(summary["val_fused"])
    snapshots = list((tmp_path / "runs").glob(f"t-{arm}.pt.*"))
    assert {int(p.name.rsplit(".", 1)[1]) for p in snapshots} == {6, 8}
    step_records = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("step ")
    ]
    assert len(step_records) == 8


def test_resume_is_exact(tmp_path, capsys):
    full = run(tmp_path, "full", ["--arm", "df"])
    half = run(tmp_path, "half", ["--arm", "df", "--max-steps", "5"])
    assert half["step"] == 5
    resumed = run(tmp_path, "half", ["--arm", "df", "--resume"])
    assert resumed["step"] == 8
    assert resumed["loss"] == full["loss"]
    assert resumed["val"] == full["val"]
    # The spool folds the log at the resume record and requires its path.
    assert any(
        line.startswith("resume") and "path=" in line
        for line in capsys.readouterr().out.splitlines()
    )


def test_resume_rejects_conflicting_exact_field(tmp_path):
    run(tmp_path, "conf", ["--arm", "df", "--max-steps", "2"])
    with pytest.raises(ValueError, match="conflicts"):
        run(tmp_path, "conf", ["--arm", "df", "--resume", "--dim", "64"])


def rewrite_latest_as_v8(tmp_path, tag):
    snapshots = list((tmp_path / "runs").glob(f"{tag}.pt.*"))
    path = max(snapshots, key=lambda item: int(item.name.rsplit(".", 1)[1]))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["version"] = 8
    for field in ("pkda_heads", "pkda_head_dim", "pkda_conv_size"):
        payload["args"].pop(field)
    torch.save(payload, path)


def test_checkpoint_v8_resume_is_vanilla_only(tmp_path, capsys):
    run(tmp_path, "old-vanilla", ["--arm", "vanilla", "--max-steps", "5"])
    rewrite_latest_as_v8(tmp_path, "old-vanilla")
    resumed = run(tmp_path, "old-vanilla", ["--arm", "vanilla", "--resume"])
    assert resumed["step"] == 8

    run(tmp_path, "old-df", ["--arm", "df", "--max-steps", "2"])
    rewrite_latest_as_v8(tmp_path, "old-df")
    with pytest.raises(ValueError, match="only for the preserved vanilla"):
        run(tmp_path, "old-df", ["--arm", "df", "--resume"])


def test_multipass_checkpoint_parity():
    """The guarded larger modes preserve plain-path loss and gradients."""
    from delta_feedback_experiment.model import (
        DFModel,
        arm_config,
        multipass,
        multipass_loss,
    )

    cfg = arm_config(
        "df",
        vocab_size=97,
        dim=32,
        layers=2,
        heads=2,
        kv_heads=2,
        head_dim=16,
        intermediate=64,
        pkda_heads=2,
        pkda_head_dim=16,
        max_seq_len=17,
    )
    torch.manual_seed(0)
    tokens = torch.randint(0, 97, (2, 17))
    prefix = torch.ones((1, 2), dtype=torch.long)

    def run(flag):
        torch.manual_seed(1)
        model = DFModel(cfg)
        model.grad_checkpoint = flag
        outs = multipass(model, tokens, 2, prefix_lens=prefix)
        loss, _ = multipass_loss(model, tokens, outs)
        loss.backward()
        grads = torch.cat(
            [p.grad.flatten() for p in model.parameters() if p.grad is not None]
        )
        return loss.item(), grads

    plain_loss, plain_grads = run(False)
    checked_loss, checked_grads = run(True)
    assert checked_loss == pytest.approx(plain_loss, rel=1e-6)
    assert torch.allclose(plain_grads, checked_grads, rtol=1e-5, atol=1e-7)


def test_checkpoint_policy_is_internal_and_screen_measured():
    from delta_feedback_experiment.model import DFModel, arm_config

    args = build_parser().parse_args(["x"])
    with torch.device("meta"):
        model = DFModel(arm_config("df"))
    assert not automatic_checkpoint(model, 2, args, torch.device("cuda"))
    assert not automatic_checkpoint(model, 3, args, torch.device("cuda"))
    args.micro_rows = 8
    assert automatic_checkpoint(model, 3, args, torch.device("cuda"))
    with pytest.raises(SystemExit):
        build_parser().parse_args(["x", "--grad-checkpoint"])

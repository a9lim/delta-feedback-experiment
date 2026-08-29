"""Offline checks for the data pipeline, optimizer stack, and trainer.

The heavyweight claim here is resume exactness: a run interrupted and
resumed must be bit-identical to an uninterrupted one — the property the
whole paired-comparison design leans on for multi-day jobe runs.
"""

import numpy as np
import pytest
import torch

from delta_feedback_experiment.data import TokenData, write_synthetic
from delta_feedback_experiment.optim import NorMuon, orthogonalize
from delta_feedback_experiment.train import (
    automatic_checkpoint,
    build_parser,
    build_schedule,
    draw_passes,
    mix,
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
    "1",
    "--head-dim",
    "16",
    "--intermediate",
    "64",
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
    "--log-every",
    "4",
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


def test_mix_is_stable():
    assert mix(0, 5, 1) == mix(0, 5, 1)
    assert mix(0, 5, 1) != mix(0, 5, 2)
    assert mix(1, 5, 1) != mix(0, 5, 1)


def test_pass_mixture_fractions():
    args = build_parser().parse_args(["x", "--steps", "4000"])
    counts = {1: 0, 2: 0, 3: 0}
    for step in range(1, 4001):
        counts[draw_passes(args, step, 4000)] += 1
    assert counts[1] == 3000  # feedback starts at 75% exactly
    assert 0.06 < counts[3] / (counts[2] + counts[3]) < 0.20


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


@pytest.mark.parametrize("arm", ["vanilla", "dar", "fbt", "df", "df_soft"])
def test_tiny_run_completes(tmp_path, arm):
    summary = run(tmp_path, f"t-{arm}", ["--arm", arm])
    assert summary["step"] == 8
    assert np.isfinite(summary["loss"])
    assert np.isfinite(summary["val"])
    if arm in ("fbt", "df", "df_soft"):
        assert np.isfinite(summary["val_fused"])
    snapshots = list((tmp_path / "runs").glob(f"t-{arm}.pt.*"))
    assert {int(p.name.rsplit(".", 1)[1]) for p in snapshots} == {6, 8}


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


def test_multipass_checkpoint_parity():
    """Multi-pass steps checkpoint unconditionally (the k>=2 graphs OOM'd
    the 4090 otherwise); loss and grads must match the plain path."""
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
        kv_heads=1,
        head_dim=16,
        intermediate=64,
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
    with torch.device("meta"):
        soft = DFModel(arm_config("df_soft"))
    assert automatic_checkpoint(soft, 2, args, torch.device("cuda"))
    args.micro_rows = 8
    assert automatic_checkpoint(model, 3, args, torch.device("cuda"))
    with pytest.raises(SystemExit):
        build_parser().parse_args(["x", "--grad-checkpoint"])

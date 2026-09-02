"""Offline checks for the data pipeline, optimizer stack, and trainer.

The heavyweight claim here is resume exactness: a run interrupted and
resumed must be bit-identical to an uninterrupted one — the property the
whole paired-comparison design leans on for multi-day jobe runs.
"""

import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from delta_feedback_experiment import data as data_module
from delta_feedback_experiment.cli import tokenize_command
from delta_feedback_experiment.data import (
    CANONICAL_CONFIG,
    CANONICAL_DATA_PACKAGES,
    CANONICAL_DATASET,
    CANONICAL_DATASET_REVISION,
    CANONICAL_TARGET_TOKENS,
    CANONICAL_TOKENIZER,
    CANONICAL_TOKENIZER_REVISION,
    CANONICAL_VAL_TOKENS,
    TokenData,
    data_package_versions,
    tokenize,
    write_synthetic,
)
from delta_feedback_experiment.optim import (
    DEFAULT_ADAM_LR,
    DEFAULT_NORMUONH_LR,
    NorMuonH,
    build_optimizers,
    orthogonalize,
    split_parameters,
)
from delta_feedback_experiment.train import (
    CONTRACT,
    GRAD_CLIP_NORM,
    automatic_checkpoint,
    build_parser,
    build_schedule,
    clip_gradients,
    draw_passes,
    feedback_boundary,
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


def test_data_build_versions_are_exact(monkeypatch):
    installed = dict(CANONICAL_DATA_PACKAGES)
    monkeypatch.setattr(data_module, "version", installed.__getitem__)
    assert data_package_versions() == installed

    installed["datasets"] = "0.0.0"
    with pytest.raises(RuntimeError, match="datasets==5.0.1"):
        data_package_versions()


def test_tokenize_materializes_pinned_hf_stream(tmp_path, monkeypatch):
    calls = {}
    datasets = ModuleType("datasets")
    transformers = ModuleType("transformers")

    def load_dataset(dataset, *, name, split, streaming, revision):
        calls["dataset"] = (dataset, name, split, streaming, revision)
        return iter([{"text": "a"}, {"text": "b"}, {"text": "c"}])

    class FakeTokenizer:
        eos_token_id = 3

        def __call__(self, texts, *, add_special_tokens):
            assert not add_special_tokens
            return {"input_ids": [[1, 2] for _ in texts]}

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, name, *, revision):
            calls["tokenizer"] = (name, revision)
            return FakeTokenizer()

    datasets.load_dataset = load_dataset
    transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        data_module, "data_package_versions", lambda: dict(CANONICAL_DATA_PACKAGES)
    )

    meta = tokenize(tmp_path / "tokens", target_tokens=9, val_tokens=3, batch_docs=1)

    assert calls["dataset"] == (
        CANONICAL_DATASET,
        CANONICAL_CONFIG,
        "train",
        True,
        CANONICAL_DATASET_REVISION,
    )
    assert calls["tokenizer"] == (
        CANONICAL_TOKENIZER,
        CANONICAL_TOKENIZER_REVISION,
    )
    assert meta["revision"] == CANONICAL_DATASET_REVISION
    assert meta["tokenizer_revision"] == CANONICAL_TOKENIZER_REVISION
    assert meta["packages"] == CANONICAL_DATA_PACKAGES
    assert meta["val_tokens"] == 3
    assert meta["train_tokens"] == 6


def test_tokenize_command_uses_canonical_defaults(tmp_path, monkeypatch):
    captured = {}

    def fake_tokenize(out_dir, **kwargs):
        captured["out_dir"] = out_dir
        captured.update(kwargs)

    monkeypatch.setattr(data_module, "tokenize", fake_tokenize)
    tokenize_command(["--out", str(tmp_path / "tokens")])

    assert captured["target_tokens"] == CANONICAL_TARGET_TOKENS
    assert captured["val_tokens"] == CANONICAL_VAL_TOKENS
    assert captured["config"] == CANONICAL_CONFIG
    assert captured["revision"] == CANONICAL_DATASET_REVISION
    assert captured["tokenizer_revision"] == CANONICAL_TOKENIZER_REVISION


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


def test_normuonh_descends_on_its_initial_frobenius_sphere():
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(16, 8))
    initial_radius = weight.detach().norm()
    target = torch.randn_like(weight)
    target.mul_(initial_radius / target.norm())
    optimizer = NorMuonH([weight], lr=0.05)
    first = None
    for _ in range(200):
        loss = (weight - target).square().mean()
        first = loss.item() if first is None else first
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.allclose(weight.norm(), initial_radius, rtol=2e-6, atol=2e-6)
    assert loss.item() < 0.05 * first
    assert torch.equal(optimizer.state[weight]["radius"], initial_radius)


def test_normuonh_rejects_vectors_and_zero_radius():
    with pytest.raises(ValueError):
        NorMuonH([torch.nn.Parameter(torch.zeros(8))])
    with pytest.raises(ValueError, match="nonzero initial Frobenius radius"):
        NorMuonH([torch.nn.Parameter(torch.zeros(8, 8))])


def test_batched_normuonh_matches_independent_parameters():
    torch.manual_seed(4)
    left = [torch.nn.Parameter(torch.randn(12, 8)) for _ in range(3)]
    right = [torch.nn.Parameter(parameter.detach().clone()) for parameter in left]
    gradients = [torch.randn_like(parameter) for parameter in left]
    batched = NorMuonH(left, lr=0.03)
    singles = [NorMuonH([parameter], lr=0.03) for parameter in right]
    for parameter, gradient in zip(left, gradients, strict=True):
        parameter.grad = gradient.clone()
    for parameter, gradient in zip(right, gradients, strict=True):
        parameter.grad = gradient.clone()
    batched.step()
    for optimizer in singles:
        optimizer.step()
    for actual, expected in zip(left, right, strict=True):
        assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_global_gradient_clip_uses_one_accumulated_vector():
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([12.0])

    preclip = clip_gradients([first, second])

    assert GRAD_CLIP_NORM == 1.0
    assert CONTRACT.version == 17
    assert CONTRACT.resumable == frozenset({17})
    assert CONTRACT.surface_version == 16
    assert preclip == pytest.approx(13.0)
    clipped = torch.cat([first.grad, second.grad])
    assert clipped.norm().item() == pytest.approx(1.0)
    assert clipped.tolist() == pytest.approx([3 / 13, 4 / 13, 12 / 13])


def test_semantic_scale_gates_use_adam_and_value_matrices_use_normuonh():
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
    normuonh, adam = split_parameters(model)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    normuonh_names = {names[id(parameter)] for parameter in normuonh}
    adam_names = {names[id(parameter)] for parameter in adam}

    assert "attention_gates.0.weight" in adam_names
    assert "embed_tokens.weight" in adam_names
    assert "blocks.0.attn_router.query" in adam_names
    assert "blocks.0.attn_router.key_norm.weight" in adam_names
    assert "fuse_value.weight" in normuonh_names
    assert "fuse_gate.weight" in adam_names
    assert "blocks.0.attn.q_proj.weight" in normuonh_names
    assert "blocks.3.attn.qkv_proj.weight" in normuonh_names
    assert "blocks.0.attn.control_proj.weight" in adam_names
    assert "blocks.0.attn.decay_up.weight" in adam_names
    assert "blocks.0.attn.output_gate_up.weight" in adam_names
    assert "blocks.0.attn.q_conv.weight" in adam_names
    assert "blocks.0.mlp.gate_up_proj.weight" in normuonh_names
    assert not (normuonh_names & adam_names)
    assert len(normuonh_names) + len(adam_names) == len(names)

    normuonh_optimizer, adam_optimizer = build_optimizers(model)
    assert isinstance(normuonh_optimizer, NorMuonH)
    assert DEFAULT_NORMUONH_LR == 2e-2
    assert normuonh_optimizer.param_groups[0]["lr"] == DEFAULT_NORMUONH_LR
    assert normuonh_optimizer.param_groups[0]["stable_lr"] == DEFAULT_NORMUONH_LR
    assert "weight_decay" not in normuonh_optimizer.param_groups[0]
    assert adam_optimizer.param_groups[0]["lr"] == DEFAULT_ADAM_LR
    assert adam_optimizer.param_groups[0]["stable_lr"] == DEFAULT_ADAM_LR
    assert adam_optimizer.param_groups[0]["weight_decay"] == 0


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
            GraphSpec(1, False): None,
            GraphSpec(2, False): None,
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
    args = build_parser().parse_args(["x"])  # defaults: 10,745 steps
    assert args.batch_rows == 320
    assert args.batch_rows // args.micro_rows == 80
    schedule = build_schedule(args)
    assert schedule.spans == (200, 0, 7859, 2686)
    assert schedule.total == 10745
    assert schedule.phase(200)[0] == "warmup"
    assert schedule.phase(201)[0] == "heat"
    assert schedule.phase(8060)[0] == "cooldown"
    assert schedule.rate_at(8059, 1.0) == 1.0
    assert schedule.rate_at(10745, 1.0) < 1e-6
    assert feedback_boundary(args, schedule.total) == schedule.heat_end == 8059


def test_registered_fresh_screen_budgets_match_active_parameter_ratios():
    tokens_per_step = 320 * 1024
    df_active_non_embedding = 140_827_944
    trials = {25: 10_745, 400: 171_909}

    for target_ratio, steps in trials.items():
        realized_ratio = steps * tokens_per_step / df_active_non_embedding
        assert realized_ratio == pytest.approx(target_ratio, abs=0.002)

    prime = build_parser().parse_args(["x", "--steps", "171909"])
    schedule = build_schedule(prime)
    assert schedule.spans == (200, 0, 128_732, 42_977)
    assert feedback_boundary(prime, schedule.total) == schedule.heat_end == 128_932


def test_fresh_run_uses_authoritative_optimizer_defaults():
    args = build_parser().parse_args(["x"])
    assert args.lr_h == DEFAULT_NORMUONH_LR == 2e-2
    assert args.lr_adam == DEFAULT_ADAM_LR == 5e-4


def test_log_every_flag_is_removed():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["x", "--log-every", "2"])


@pytest.mark.parametrize("flag", ["--lr-muon", "--wd-muon"])
def test_legacy_optimizer_flags_are_removed(flag):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["x", flag, "0.01"])


@pytest.mark.parametrize(
    "flag",
    ["--feedback-start", "--three-pass"],
)
@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf"])
def test_pass_probabilities_reject_values_outside_unit_interval(flag, value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["x", flag, value])


def test_feedback_batch_prob_flag_is_removed():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["x", "--feedback-batch-prob", "0.5"])


def test_mix_is_stable():
    assert mix(0, 5, 1) == mix(0, 5, 1)
    assert mix(0, 5, 1) != mix(0, 5, 2)
    assert mix(1, 5, 1) != mix(0, 5, 1)


def test_pass_mixture_fractions():
    args = build_parser().parse_args(["x", "--steps", "4000"])
    assert args.feedback_start == 0.75
    assert args.three_pass == 0.12
    assert feedback_boundary(args, 4000) == 3000

    heat = [draw_passes(args, step, 4000) for step in range(1, 3001)]
    feedback = [draw_passes(args, step, 4000) for step in range(3001, 4001)]
    assert set(heat) == {1}
    assert 1 not in feedback

    feedback_counts = {k: feedback.count(k) for k in (2, 3)}
    assert 0.84 < feedback_counts[2] / 1000 < 0.92
    assert 0.08 < feedback_counts[3] / 1000 < 0.16

    counts = {1: 0, 2: 0, 3: 0}
    for n_passes in heat + feedback:
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
    # Protected: feedback boundary (4), cooldown boundary (6), end (8).
    assert {int(p.name.rsplit(".", 1)[1]) for p in snapshots} == {4, 6, 8}
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


def rewrite_latest_version(tmp_path, tag, version):
    snapshots = list((tmp_path / "runs").glob(f"{tag}.pt.*"))
    path = max(snapshots, key=lambda item: int(item.name.rsplit(".", 1)[1]))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["version"] = version
    torch.save(payload, path)


@pytest.mark.parametrize("version", [9, 10, 11, 12, 13, 14, 15, 16])
def test_resume_rejects_every_legacy_checkpoint(tmp_path, version):
    tag = f"legacy-v{version}"
    run(tmp_path, tag, ["--arm", "vanilla", "--max-steps", "5"])
    rewrite_latest_version(tmp_path, tag, version)
    with pytest.raises(ValueError, match="resumable versions \\[17\\]"):
        run(tmp_path, tag, ["--arm", "vanilla", "--resume"])


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


def test_prefix_draw_leaves_every_row_one_fused_position():
    """Plain-prefix lengths are drawn in 1..seq_len-1: position 0 is always
    plain, the last executed position is always fused."""
    from types import SimpleNamespace

    from delta_feedback_experiment.train import micro_draws

    args = SimpleNamespace(data_seed=0, seq_len=16, jitter=0.02)
    seen = set()
    for step in range(64):
        prefix, jitter = micro_draws(
            args, step, step * 7, 3, 8, 4, torch.device("cpu")
        )
        assert prefix.shape == (2, 8)
        assert jitter.shape == (2, 8, args.seq_len + 1, 4)
        assert int(prefix.min()) >= 1
        assert int(prefix.max()) <= args.seq_len - 1
        seen.update(prefix.flatten().tolist())
    assert args.seq_len - 1 in seen and 1 in seen
    assert args.seq_len not in seen

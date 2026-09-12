"""Small data lifecycle, recipe, optimizer, and exact-resume contracts."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from delta_feedback_experiment import data as data_module
from delta_feedback_experiment.data import (
    CANONICAL_SHUFFLE_SEED,
    DEFAULT_SOURCE,
    LocalSource,
    Shuffle,
    TokenData,
    data_package_versions,
    source_address,
    tokenize,
    verify,
    write_synthetic,
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


def corpus(tmp_path):
    directory = tmp_path / DEFAULT_SOURCE
    write_synthetic(directory, train_tokens=1200, val_tokens=120, vocab=97)
    return directory


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


def test_data_build_records_newer_package_versions(monkeypatch):
    installed = {
        "transformers": "5.17.0",
        "tokenizers": "0.23.2",
        "huggingface-hub": "1.31.0",
        "pyarrow": "25.0.1",
    }
    monkeypatch.setattr(data_module, "version", installed.__getitem__)
    assert data_package_versions() == installed


def test_data_build_reports_a_missing_package(monkeypatch):
    def installed_version(package):
        if package == "pyarrow":
            raise data_module.PackageNotFoundError(package)
        return "99.0.0"

    monkeypatch.setattr(data_module, "version", installed_version)
    with pytest.raises(
        RuntimeError, match="requires pyarrow; install the data-build extra"
    ):
        data_package_versions()


@pytest.mark.parametrize("size", [1, 7, 64])
def test_shuffle_is_a_keyed_bijection(size):
    everything = np.arange(size)
    shuffle = Shuffle(size, seed=CANONICAL_SHUFFLE_SEED)
    positions = shuffle(everything)
    assert sorted(positions.tolist()) == everything.tolist()
    assert np.array_equal(shuffle.inverse(positions), everything)
    assert np.array_equal(shuffle(everything), positions)  # stateless
    assert np.array_equal(shuffle(everything[::7]), positions[::7])  # batch-free
    if size >= 64:
        assert not np.array_equal(positions, everything)
        assert not np.array_equal(Shuffle(size, seed=1)(everything), positions)
    with pytest.raises(ValueError):
        shuffle([size])


def test_shuffle_can_preserve_source_order():
    addresses = np.array([0, 9, 3, 9, 99])
    identity = Shuffle(100, seed=42, enabled=False)
    assert np.array_equal(identity(addresses), addresses)
    assert np.array_equal(identity.inverse(addresses), addresses)
    with pytest.raises(ValueError):
        identity([100])


EOS = 3


class CharTokenizer:
    eos_token_id = EOS

    def __call__(self, texts, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [[4 + ord(c) % 90 for c in text] for text in texts]}


def char_tokenizer():
    return CharTokenizer()


def char_ids(text: str) -> list[int]:
    return CharTokenizer()([text], add_special_tokens=False)["input_ids"][0] + [EOS]


def parquet_source(root: Path, files: int = 3, rows: int = 40) -> LocalSource:
    """Tiny DCLM-shaped parquet files, with varying lengths and one empty text."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = root / "filtered" / "shard_00000"
    directory.mkdir(parents=True)
    for f in range(files):
        texts = []
        for i in range(rows):
            texts.append(
                "" if (f, i) == (1, 5) else f"d{f}-{i}-" + "x" * ((7 * i + 3 * f) % 25)
            )
        table = pa.table(
            {
                "text": texts,
                "url": [f"https://example.com/{f}/{i}" for i in range(rows)],
                "id": [f"{f}-{i}" for i in range(rows)],
                "language": ["en"] * rows,
                "language_score": [0.99] * rows,
                "fasttext_score": [0.95] * rows,
            }
        )
        pq.write_table(table, directory / f"{f:03d}.parquet", row_group_size=17)
    return LocalSource(root)


def build(out: Path, source: LocalSource, **overrides) -> dict:
    settings = {
        "target_tokens": 600,
        "val_tokens": 200,
        "seed": 3,
        "tokens_per_doc": 10,
        "source": source,
        "tokenizer": char_tokenizer,
        "check_packages": False,
        "shuffle": True,
    }
    settings.update(overrides)
    return tokenize(out, **settings)


def stream(directory: Path, split: str) -> np.ndarray:
    data = TokenData.load(directory, split, 1)
    return np.asarray(data.read(0, data.total_tokens))


def test_tokenize_writes_a_shuffled_prefix_with_provenance(tmp_path):
    import pyarrow.parquet as pq

    source = parquet_source(tmp_path / "source")
    out = tmp_path / "tokens"
    meta = build(out, source)
    index = json.loads((out / "source.json").read_text())

    assert meta["universe_docs"] == 120
    assert meta["selected_docs"] == 60
    assert meta["eos_id"] == EOS
    assert "config" not in meta and "dumps" not in meta
    assert set(index) == {"files"}
    assert not (out / "parts").exists()
    assert verify(out)["train"]["docs"] == meta["train_docs"]

    rows = {}
    for path in {f["path"] for f in index["files"]}:
        table = pq.ParquetFile(source.root / path).read(columns=["text", "url", "id"])
        rows[path] = table.to_pylist()

    seen = []
    for split in ("val", "train"):
        tokens = stream(out, split)
        docs = TokenData.load(out, split, 1).docs
        assert docs.dtype.names == ("start", "source")
        assert tokens.size == meta[f"{split}_tokens"]
        assert tokens[-1] == EOS  # splits end at document boundaries
        starts = docs["start"].tolist() + [tokens.size]
        for record, start, end in zip(docs, starts, starts[1:]):
            path, row = source_address(index, int(record["source"]))
            original = rows[path][row]
            text = original["text"]
            assert text, "an empty document was written"
            assert tokens[start:end].tolist() == char_ids(text)
            file_number = int(Path(path).stem)
            assert original["id"] == f"{file_number}-{row}"
            assert original["url"] == f"https://example.com/{file_number}/{row}"
            seen.append(int(record["source"]))
    assert meta["val_target"] == 200 >= meta["val_tokens"] > 150
    assert meta["train_tokens"] >= 400 and meta["target_tokens"] == 600
    assert len(seen) == len(set(seen)) <= 60
    assert seen != sorted(seen)  # the stream is not in source order
    assert 45 not in seen  # file 1 row 5, the empty text, is never written


@dataclass
class FlakySource:
    """A local source whose second fetch fails once."""

    inner: LocalSource
    marker: Path

    def list_files(self):
        return self.inner.list_files()

    def row_groups(self, path):
        return self.inner.row_groups(path)

    def fetch(self, path):
        if path.endswith("001.parquet") and self.marker.exists():
            self.marker.unlink()
            raise OSError("simulated download failure")
        return self.inner.fetch(path)

    def release(self, path):
        return None


def test_tokenize_prefixes_agree_and_a_partial_build_resumes(tmp_path):
    workers, shuffle = 1, True
    source = parquet_source(tmp_path / "source")
    short = build(tmp_path / "short", source, shuffle=shuffle)
    long = build(tmp_path / "long", source, target_tokens=900, shuffle=shuffle)
    assert long["selected_docs"] == 90 > short["selected_docs"]
    assert np.array_equal(
        stream(tmp_path / "short", "val"), stream(tmp_path / "long", "val")
    )
    short_train, long_train = (
        stream(tmp_path / "short", "train"),
        stream(tmp_path / "long", "train"),
    )
    assert long_train.size > short_train.size
    assert np.array_equal(short_train, long_train[: short_train.size])
    short_docs = TokenData.load(tmp_path / "short", "train", 1).docs
    long_docs = TokenData.load(tmp_path / "long", "train", 1).docs
    assert np.array_equal(short_docs, long_docs[: short_docs.size])

    marker = tmp_path / "fail-once"
    marker.touch()
    flaky = FlakySource(source, marker)
    with pytest.raises(OSError, match="simulated"):
        build(tmp_path / "resumed", flaky, workers=workers, shuffle=shuffle)
    assert (tmp_path / "resumed" / "parts" / "0000.index.npy").exists()
    assert not (tmp_path / "resumed" / "parts" / "0001.index.npy").exists()
    resumed = build(tmp_path / "resumed", flaky, workers=workers, shuffle=shuffle)
    assert resumed["train_tokens"] == short["train_tokens"]
    assert np.array_equal(stream(tmp_path / "resumed", "train"), short_train)
    assert {p.name: p.read_bytes() for p in (tmp_path / "resumed").iterdir()} == {
        p.name: p.read_bytes() for p in (tmp_path / "short").iterdir()
    }
    with pytest.raises(FileExistsError):
        build(tmp_path / "resumed", source)


def test_tokenize_partial_build_rejects_changed_settings(tmp_path, monkeypatch):
    changed = {"seed": 4}
    source = parquet_source(tmp_path / "source")
    marker = tmp_path / "fail-once"
    marker.touch()
    flaky = FlakySource(source, marker)
    out = tmp_path / "tokens"
    with pytest.raises(OSError, match="simulated"):
        build(out, flaky)
    assert (out / "build.json").is_file()
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}

    with monkeypatch.context() as patch:
        patch.setattr(
            flaky,
            "fetch",
            lambda path: pytest.fail("incompatible partial build fetched a source"),
        )
        with pytest.raises(ValueError, match="partial build:"):
            build(out, flaky, **changed)
    assert {
        p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()
    } == before
    build(out, flaky)
    assert not (out / "build.json").exists()


def test_verify_catches_a_broken_document_boundary(tmp_path):
    write_synthetic(tmp_path, train_tokens=1200, val_tokens=120, vocab=97)
    assert (
        verify(tmp_path)["val"]["docs"] == data_module.read_meta(tmp_path)["val_docs"]
    )
    docs = TokenData.load(tmp_path, "train", 16).docs
    assert docs is not None and docs["start"][0] == 0
    shard = np.memmap(tmp_path / "train.0000.bin", dtype=np.uint32, mode="r+")
    shard[int(docs["start"][1]) - 1] = 5
    shard.flush()
    with pytest.raises(RuntimeError, match="no EOS"):
        verify(tmp_path)


def test_tokenize_continue_recovers_an_interrupted_append(tmp_path, monkeypatch):
    shuffle = False
    source = parquet_source(tmp_path / "source")
    monkeypatch.setattr(data_module, "SHARD_TOKENS", 71)
    monkeypatch.setattr(data_module, "WRITE_BUFFER_TOKENS", 97)
    out = tmp_path / "resumed"
    initial = build(out, source, shuffle=shuffle)
    before = sum(p.stat().st_size for p in out.glob("train.*.bin"))
    original_write = data_module._ShardWriter.write

    def append_then_fail(writer, array):
        original_write(writer, array)
        if writer.handle is not None:
            writer.handle.flush()
        raise OSError("simulated append failure")

    with monkeypatch.context() as patch:
        patch.setattr(data_module._ShardWriter, "write", append_then_fail)
        with pytest.raises(OSError, match="simulated append"):
            build(out, source, target_tokens=900, extend=True, shuffle=shuffle)
    assert sum(p.stat().st_size for p in out.glob("train.*.bin")) > before
    assert data_module.read_meta(out)["train_tokens"] == initial["train_tokens"]
    build(out, source, target_tokens=900, extend=True, shuffle=shuffle)

    # Match every byte against an uninterrupted extension, including metadata.
    control = tmp_path / "control"
    build(control, source, shuffle=shuffle)
    build(control, source, target_tokens=900, extend=True, shuffle=shuffle)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == {
        p.name: p.read_bytes() for p in control.iterdir()
    }
    # Its shards, sidecars, and source index also match a fresh larger build.
    fresh = tmp_path / "fresh"
    build(fresh, source, target_tokens=900, shuffle=shuffle)
    assert {p.name: p.read_bytes() for p in out.iterdir() if p.name != "meta.json"} == {
        p.name: p.read_bytes() for p in fresh.iterdir() if p.name != "meta.json"
    }


def test_normuonh_applies_nesterov_before_orthogonalization():
    torch.manual_seed(3)
    weight = torch.nn.Parameter(torch.randn(8, 6))
    expected = weight.detach().clone()
    radius = expected.norm()
    momentum = torch.zeros_like(expected)
    row_moment = torch.zeros(expected.shape[0], 1)
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
        update = update / (row_moment.sqrt() + eps)
        update = update / update.norm().clamp_min(eps)
        trial = expected - lr * radius * update
        expected = radius * trial / trial.norm().clamp_min(eps)

    assert torch.allclose(weight, expected, rtol=2e-5, atol=2e-6)
    assert torch.allclose(optimizer.state[weight]["momentum"], momentum)
    assert torch.allclose(optimizer.state[weight]["row_moment"], row_moment)


def test_global_gradient_clip_uses_one_accumulated_vector():
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([12.0])

    preclip = clip_gradients([first, second])

    assert GRAD_CLIP_NORM == 10.0
    assert CONTRACT.version == 32
    assert CONTRACT.resumable == frozenset({32})
    assert CONTRACT.surface_version == 32
    assert preclip == pytest.approx(13.0)
    clipped = torch.cat([first.grad, second.grad])
    assert clipped.norm().item() == pytest.approx(10.0)
    assert clipped.tolist() == pytest.approx([30 / 13, 40 / 13, 120 / 13])


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
        "scale", "dim", "layers", "seq_len", "batch_rows", "micro_rows",
        "expert_intermediate", "num_routed_experts", "experts_per_token",
    } <= pinned
    assert "steps" not in pinned
    _, pinned = resolve_run_args(parser, ["x", "--tokens-per-param", "400"])
    assert {"tokens_per_param", "steps"} <= pinned and "dim" not in pinned
    _, pinned = resolve_run_args(parser, ["x", "--seq-len", "2048"])
    assert {"seq_len", "batch_rows"} <= pinned


@pytest.mark.parametrize(
    "scale,dim,selected,routed,total,active",
    [
        ("screen", 768, 3, 15, 630606216, 200919432),
        ("bridge", 1152, 5, 23, 1384528652, 446708492),
        ("flagship", 1536, 7, 31, 2430864784, 789384592),
    ],
)
def test_width_ladder_preserves_depth_and_ffn_capacity(
    scale, dim, selected, routed, total, active
):
    from delta_feedback_experiment.model import DeltaModel, condition_config

    args = parse_run_args(["geometry", "--scale", scale])
    cfg = condition_config("fl", **model_fields(args))
    assert cfg.layers == 16 and cfg.core_layers == range(4, 12)
    assert cfg.executed_layers(4) == 40 and cfg.routing_blocks == 4
    assert cfg.dim == dim and cfg.expert_intermediate == 832
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


def test_expert_geometry_overrides_replace_preset_fields():
    args, pinned = resolve_run_args(
        build_parser(),
        [
            "custom", "--scale", "flagship", "--expert-intermediate", "64",
            "--num-routed-experts", "9", "--experts-per-token", "2",
        ],
    )
    assert (args.expert_intermediate, args.num_routed_experts, args.experts_per_token) == (
        64, 9, 2
    )
    assert {"expert_intermediate", "num_routed_experts", "experts_per_token"} <= pinned
    with pytest.raises(SystemExit):
        parse_run_args(["retired", "--intermediate", "3328"])


def test_default_token_store_covers_current_screen_schedule():
    from delta_feedback_experiment.cli import stream_target
    from delta_feedback_experiment.data import (
        CANONICAL_TARGET_TOKENS,
        CANONICAL_VAL_TOKENS,
    )

    assert CANONICAL_TARGET_TOKENS == stream_target("screen", 400, CANONICAL_VAL_TOKENS)


def test_trainer_uses_named_source_paths():
    args = parse_run_args(["x"])
    assert args.data_root == "data"
    assert args.source == DEFAULT_SOURCE
    parser = build_parser()
    args, pinned = resolve_run_args(
        parser, ["x", "--data-root", "/data/delta", "--source", "fineweb-edu-10b"]
    )
    assert Path(args.data_root) / args.source == Path("/data/delta/fineweb-edu-10b")
    assert {"data_root", "source"} <= pinned


def test_current_recipe_and_keyed_draws():
    default = parse_run_args(["recipe"])
    assert default.condition == "f" and default.mtp_weight == 0.3
    assert parse_run_args(["recipe", "--condition", "lf"]).condition == "fl"
    for invalid in ("", "a", "r", "e", "m", "arf", "ff"):
        with pytest.raises(SystemExit):
            parse_run_args(["recipe", "--condition", invalid])
    for condition in ("f", "l", "fl"):
        assert (
            parse_run_args(["recipe", "--condition", condition]).steps == default.steps
        )
    schedule = build_schedule(default)
    assert schedule.total == default.steps
    assert schedule.rate_at(schedule.total, 1.0) < 1e-6
    assert default.batch_rows * default.seq_len == BATCH_TOKENS
    for step in (1, 17, 101):
        assert mix(0, step, 1) == mix(0, step, 1)
        assert mix(0, step, 1) != mix(0, step, 2)
        assert 1 <= draw_iterations(default, step, True) <= default.loop_max_iterations
        assert draw_iterations(default, step, False) == 1
    assert draw_passes(default, 1, default.steps) == 1
    assert draw_passes(default, default.steps, default.steps) in (2, 3)
    for flags in (["--mtp-weight", "nan"], ["--seq-len", "1"], ["--three-pass", "1.1"]):
        with pytest.raises(SystemExit):
            parse_run_args(["recipe", *flags])


def test_full_training_resume_and_rename_preserve_model_optimizer_and_bias(
    tmp_path, monkeypatch
):
    """One two-step run covers the full recurrent training/checkpoint surface."""
    from transformer_experiments.spool import Spool

    from delta_feedback_experiment import cli
    from delta_feedback_experiment import train as trainer
    from delta_feedback_experiment.model import DeltaModel

    write_synthetic(
        tmp_path / "data" / DEFAULT_SOURCE, train_tokens=80, val_tokens=20, vocab=31
    )
    settings = [
        "--condition",
        "fl",
        "--data-root",
        str(tmp_path / "data"),
        "--out-dir",
        str(tmp_path / "runs"),
        "--vocab-size",
        "31",
        "--dim",
        "16",
        "--layers",
        "16",
        "--heads",
        "2",
        "--kv-heads",
        "2",
        "--head-dim",
        "8",
        "--expert-intermediate",
        "8",
        "--num-routed-experts",
        "23",
        "--experts-per-token",
        "5",
        "--pkda-heads",
        "2",
        "--pkda-head-dim",
        "8",
        "--seq-len",
        "4",
        "--batch-rows",
        "2",
        "--micro-rows",
        "1",
        "--loop-iterations",
        "2",
        "--loop-max-iterations",
        "2",
        "--steps",
        "2",
        "--warmup-frac",
        "0",
        "--cooldown-frac",
        "0",
        "--feedback-start",
        "0",
        "--three-pass",
        "0",
        "--eval-every",
        "2",
        "--eval-rows",
        "1",
        "--snapshot-every",
        "1",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(
        trainer, "draw_iterations", lambda args, step, loop: 2 if loop else 1
    )
    update_bias = DeltaModel.update_expert_bias
    optimizer_events = []
    for optimizer_class in (NorMuonH, torch.optim.NAdam):
        original = optimizer_class.step

        def observe_step(optimizer, *args, _original=original, **kwargs):
            result = _original(optimizer, *args, **kwargs)
            optimizer_events.append(type(optimizer).__name__)
            return result

        monkeypatch.setattr(optimizer_class, "step", observe_step)

    def observe_bias(model, counts, **kwargs):
        assert optimizer_events[-2:] == ["NorMuonH", "NAdam"]
        per_layer = 5 * 2 * 4 * 2  # top-k * rows * input length * feedback passes
        assert (
            counts.sum(-1).tolist()
            == [per_layer] * 4 + [2 * per_layer] * 8 + [per_layer] * 4
            + [5 * 2 * 3 * 2]  # MTP predicts one fewer token on each pass.
        )
        before = torch.stack([bank.expert_bias.clone() for bank in model.expert_banks])
        update_bias(model, counts, **kwargs)
        after = torch.stack([bank.expert_bias for bank in model.expert_banks])
        torch.testing.assert_close(
            after,
            before + 0.001 * (counts.sum(-1, keepdim=True) - 23 * counts).sign(),
            rtol=0,
            atol=0,
        )
        optimizer_events.append("bias")

    monkeypatch.setattr(DeltaModel, "update_expert_bias", observe_bias)
    summarize_experts = trainer.expert_summary

    def normalized_expert_summary(model, *args, **kwargs):
        records = summarize_experts(model, *args, **kwargs)
        assert any(record["site"] == "mtp.experts" for record in records)
        for record in records:
            mass = sum(record[f"expert{i}"] for i in range(model.cfg.num_routed_experts))
            # Summaries round each assignment fraction to four decimal places.
            assert mass == pytest.approx(1, abs=model.cfg.num_routed_experts * 5e-5)
        return records

    monkeypatch.setattr(trainer, "expert_summary", normalized_expert_summary)

    full = trainer.train(["full", *settings])
    half = trainer.train(["half", *settings, "--max-steps", "1"])
    assert half["step"] == 1
    Spool(replace(cli.LAYOUT, root=tmp_path), cli.PIPELINE).move("half", "renamed")
    with pytest.raises(ValueError, match="conflicts"):
        trainer.train(["renamed", *settings, "--resume", "--mtp-weight", "0.17"])
    for flag, value in (
        ("--expert-intermediate", "9"),
        ("--num-routed-experts", "31"),
        ("--experts-per-token", "7"),
    ):
        with pytest.raises(ValueError, match="conflicts"):
            trainer.train(["renamed", *settings, "--resume", flag, value])
    resumed = trainer.train(["renamed", *settings, "--resume"])
    for key in ("step", "loss", "val", "val_fused", "val_mtp", "val_mtp_fused"):
        assert resumed[key] == full[key]
    complete = trainer.read_checkpoint(tmp_path / "runs" / "full.pt.2")
    restored = trainer.read_checkpoint(tmp_path / "runs" / "renamed.pt.2")

    def identical(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                identical(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right, strict=True):
                identical(a, b)
        else:
            assert left == right

    identical(complete["state"], restored["state"])
    identical(complete["optimizer"], restored["optimizer"])
    assert complete["version"] == CONTRACT.version == 32
    assert any(
        value.count_nonzero()
        for name, value in restored["state"].items()
        if name.endswith("expert_bias")
    )
    assert restored["state"]["mtp.block.mlp.expert_bias"].count_nonzero()
    assert [name for name in optimizer_events if name == "bias"] == ["bias"] * 4

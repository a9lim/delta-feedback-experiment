"""Offline checks for the data pipeline, optimizer stack, and trainer.

The heavyweight claim here is resume exactness: a run interrupted and
resumed must be bit-identical to an uninterrupted one — the property the
whole paired-comparison design leans on for multi-day jobe runs.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from delta_feedback_experiment import data as data_module
from delta_feedback_experiment.cli import stream_target, tokenize_command
from delta_feedback_experiment.data import (
    CANONICAL_CONFIG,
    CANONICAL_DATA_PACKAGES,
    CANONICAL_DATASET_REVISION,
    CANONICAL_SHUFFLE_SEED,
    CANONICAL_TARGET_TOKENS,
    CANONICAL_TOKENIZER_REVISION,
    CANONICAL_TOKENS_PER_DOC,
    CANONICAL_VAL_TOKENS,
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
    DEFAULT_NORMUONH_LR,
    NADAM_MOMENTUM_DECAY,
    NorMuonH,
    apply_schedule,
    build_optimizers,
    orthogonalize,
    split_parameters,
)
from delta_feedback_experiment.train import (
    CONTRACT,
    GRAD_CLIP_NORM,
    CudaGraphTrainer,
    automatic_checkpoint,
    BATCH_TOKENS,
    build_parser,
    build_schedule,
    parse_run_args,
    resolve_run_args,
    clip_gradients,
    draw_iterations,
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
    "--warmup-frac",
    "0.25",
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

    installed["pyarrow"] = "0.0.0"
    with pytest.raises(RuntimeError, match="pyarrow==25.0.1"):
        data_package_versions()


@pytest.mark.parametrize("size", [1, 2, 3, 7, 64, 1000, 4097])
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


# -- a tiny parquet source with a character tokenizer --------------------------

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


DUMPS = ("CC-MAIN-2013-20", "CC-MAIN-2019-35", "CC-MAIN-2024-10")


def parquet_source(root: Path, files: int = 3, rows: int = 40) -> LocalSource:
    """Files of single-crawl runs, like FineWeb-Edu's, with one empty text."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "sample" / "tiny").mkdir(parents=True)
    for f in range(files):
        texts, dumps = [], []
        for i in range(rows):
            texts.append("" if (f, i) == (1, 5) else f"d{f}-{i}-" + "x" * ((7 * i + 3 * f) % 25))
            dumps.append(DUMPS[(i // 14 + f) % len(DUMPS)])
        table = pa.table({"text": texts, "dump": dumps})
        pq.write_table(table, root / "sample" / "tiny" / f"{f:03d}.parquet", row_group_size=17)
    return LocalSource(root)


def build(out: Path, source: LocalSource, **overrides) -> dict:
    settings = dict(
        target_tokens=600,
        val_tokens=200,
        seed=3,
        tokens_per_doc=10,
        source=source,
        tokenizer=char_tokenizer,
        check_packages=False,
        config="sample-tiny",
    )
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
    assert meta["config"] == "sample-tiny"
    assert not (out / "parts").exists()
    assert verify(out)["train"]["docs"] == meta["train_docs"]

    rows = {}
    for path in {f["path"] for f in index["files"]}:
        table = pq.ParquetFile(source.root / path).read(columns=["text", "dump"])
        rows[path] = (table.column("text").to_pylist(), table.column("dump").to_pylist())

    seen = []
    for split in ("val", "train"):
        tokens = stream(out, split)
        docs = TokenData.load(out, split, 1).docs
        assert tokens.size == meta[f"{split}_tokens"]
        assert tokens[-1] == EOS  # splits end at document boundaries
        starts = docs["start"].tolist() + [tokens.size]
        for record, start, end in zip(docs, starts, starts[1:]):
            path, row = source_address(index, int(record["source"]))
            text, dump = rows[path][0][row], rows[path][1][row]
            assert text, "an empty document was written"
            assert tokens[start:end].tolist() == char_ids(text)
            assert meta["dumps"][record["dump"]] == dump
            seen.append(int(record["source"]))
    assert meta["val_target"] == 200 >= meta["val_tokens"] > 150
    assert meta["train_tokens"] >= 400 and meta["target_tokens"] == 600
    assert len(seen) == len(set(seen)) <= 60
    assert seen != sorted(seen)  # the stream is not in source order
    assert 45 not in seen  # file 1 row 5, the empty text, is never written
    crawls = [meta["dumps"][d] for d in TokenData.load(out, "train", 1).docs["dump"]]
    assert len(set(crawls[:8])) > 1  # neighbours come from different crawls


@dataclass
class FlakySource:
    """A local source whose second fetch fails once."""

    inner: LocalSource
    marker: Path

    def list_files(self):
        return self.inner.list_files()

    def dumps(self):
        return self.inner.dumps()

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
    source = parquet_source(tmp_path / "source")
    short = build(tmp_path / "short", source)
    long = build(tmp_path / "long", source, target_tokens=900)
    assert long["selected_docs"] == 90 > short["selected_docs"]
    assert np.array_equal(stream(tmp_path / "short", "val"), stream(tmp_path / "long", "val"))
    short_train, long_train = stream(tmp_path / "short", "train"), stream(tmp_path / "long", "train")
    assert long_train.size > short_train.size
    assert np.array_equal(short_train, long_train[: short_train.size])
    short_docs = TokenData.load(tmp_path / "short", "train", 1).docs
    long_docs = TokenData.load(tmp_path / "long", "train", 1).docs
    assert np.array_equal(short_docs, long_docs[: short_docs.size])

    marker = tmp_path / "fail-once"
    marker.touch()
    flaky = FlakySource(source, marker)
    with pytest.raises(OSError, match="simulated"):
        build(tmp_path / "resumed", flaky)
    assert (tmp_path / "resumed" / "parts" / "0000.index.npy").exists()
    assert not (tmp_path / "resumed" / "parts" / "0001.index.npy").exists()
    resumed = build(tmp_path / "resumed", flaky)
    assert resumed["train_tokens"] == short["train_tokens"]
    assert np.array_equal(stream(tmp_path / "resumed", "train"), short_train)
    with pytest.raises(FileExistsError):
        build(tmp_path / "resumed", source)


def test_tokenize_reports_an_exhausted_selection(tmp_path):
    source = parquet_source(tmp_path / "source")
    with pytest.raises(RuntimeError, match="selection exhausted"):
        build(tmp_path / "tokens", source, tokens_per_doc=100)


def test_verify_catches_a_broken_document_boundary(tmp_path):
    write_synthetic(tmp_path, train_tokens=1200, val_tokens=120, vocab=97)
    assert verify(tmp_path)["val"]["docs"] == data_module.read_meta(tmp_path)["val_docs"]
    docs = TokenData.load(tmp_path, "train", 16).docs
    assert docs is not None and docs["start"][0] == 0
    shard = np.memmap(tmp_path / "train.0000.bin", dtype=np.uint32, mode="r+")
    shard[int(docs["start"][1]) - 1] = 5
    shard.flush()
    with pytest.raises(RuntimeError, match="no EOS"):
        verify(tmp_path)


def test_tokenize_command_uses_canonical_defaults(tmp_path, monkeypatch):
    captured = {}

    def fake_tokenize(out_dir, **kwargs):
        captured["out_dir"] = out_dir
        captured.update(kwargs)

    monkeypatch.setattr(data_module, "tokenize", fake_tokenize)
    tokenize_command(["--out", str(tmp_path / "tokens")])

    assert captured["target_tokens"] == CANONICAL_TARGET_TOKENS
    assert captured["val_tokens"] == CANONICAL_VAL_TOKENS
    assert captured["seed"] == CANONICAL_SHUFFLE_SEED
    assert captured["tokens_per_doc"] == CANONICAL_TOKENS_PER_DOC
    assert captured["workers"] == 1
    assert captured["scratch"] is None
    assert captured["config"] == CANONICAL_CONFIG
    assert captured["revision"] == CANONICAL_DATASET_REVISION
    assert captured["tokenizer_revision"] == CANONICAL_TOKENIZER_REVISION
    assert captured["extend"] is False


def test_tokenize_continue_lands_on_the_fresh_build(tmp_path):
    source = parquet_source(tmp_path / "source")
    build(tmp_path / "grown", source)
    fresh = build(tmp_path / "fresh", source, target_tokens=900)
    with pytest.raises(ValueError, match="shuffle_seed"):
        build(tmp_path / "grown", source, target_tokens=900, seed=4, extend=True)
    grown = build(tmp_path / "grown", source, target_tokens=900, extend=True)
    for split in ("val", "train"):
        assert np.array_equal(stream(tmp_path / "grown", split), stream(tmp_path / "fresh", split))
        assert np.array_equal(
            TokenData.load(tmp_path / "grown", split, 1).docs,
            TokenData.load(tmp_path / "fresh", split, 1).docs,
        )
    assert {k: v for k, v in grown.items() if k != "unused_selected"} == {
        k: v for k, v in fresh.items() if k != "unused_selected"
    }
    assert not (tmp_path / "grown" / "parts").exists()
    assert verify(tmp_path / "grown")["train"]["tokens"] == fresh["train_tokens"]
    # A target the store already covers is a no-op.
    assert build(tmp_path / "grown", source, target_tokens=700, extend=True) == grown


def test_stream_target_rounds_the_schedule_up_to_a_billion():
    assert stream_target("screen", 25, CANONICAL_VAL_TOKENS) == 4_000_000_000
    assert stream_target("screen", 400, CANONICAL_VAL_TOKENS) == CANONICAL_TARGET_TOKENS
    assert stream_target("bridge", 400, CANONICAL_VAL_TOKENS) == 167_000_000_000


def test_tokenize_command_derives_the_target_from_a_scale(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(data_module, "tokenize", lambda out, **kw: captured.update(kw))
    out = str(tmp_path / "tokens")
    tokenize_command(["--out", out, "--scale", "screen", "--tokens-per-param", "25", "--continue"])
    assert captured["target_tokens"] == 4_000_000_000 and captured["extend"] is True
    with pytest.raises(SystemExit):
        tokenize_command(["--out", out, "--scale", "screen"])
    with pytest.raises(SystemExit):
        tokenize_command(["--out", out, "--scale", "screen", "--tokens-per-param", "25", "--target", "1e9"])


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

    assert GRAD_CLIP_NORM == 10.0
    assert CONTRACT.version == 26
    assert CONTRACT.resumable == frozenset({26})
    assert CONTRACT.surface_version == 26
    assert preclip == pytest.approx(13.0)
    clipped = torch.cat([first.grad, second.grad])
    assert clipped.norm().item() == pytest.approx(10.0)
    assert clipped.tolist() == pytest.approx([30 / 13, 40 / 13, 120 / 13])


def test_semantic_scale_gates_use_nadam_and_value_matrices_use_normuonh():
    from delta_feedback_experiment.model import (
        MUP_BASE_DIM,
        DeltaModel,
        condition_config,
    )

    model = DeltaModel(
        condition_config(
            "arf",
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
    normuonh, nadam, width = split_parameters(model)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    normuonh_names = {names[id(parameter)] for parameter in normuonh}
    nadam_names = {names[id(parameter)] for parameter in nadam}
    width_names = {names[id(parameter)] for parameter in width}

    assert "attention_gates.0.weight" in width_names
    assert "embed_tokens.weight" in nadam_names
    assert "blocks.0.attn_router.query" in nadam_names
    assert "blocks.0.attn_router.key_norm.weight" in nadam_names
    assert "fuse_value.weight" in normuonh_names
    assert "fuse_gate.weight" in width_names
    assert "blocks.0.attn.q_proj.weight" in normuonh_names
    assert "blocks.3.attn.qkv_proj.weight" in normuonh_names
    assert "blocks.0.attn.control_proj.weight" in width_names
    assert "blocks.0.attn.decay_up.weight" in nadam_names
    assert "blocks.0.attn.output_gate_up.weight" in nadam_names
    assert "blocks.0.attn.q_conv.weight" in nadam_names
    assert "blocks.0.mlp.gate_up_proj.weight" in normuonh_names
    # The width-scaled group is exactly the NAdam matrices whose fan-in is D.
    assert width_names == {
        "attention_gates.0.weight",
        "fuse_gate.weight",
        "blocks.0.attn.control_proj.weight",
        "blocks.1.attn.control_proj.weight",
        "blocks.2.attn.control_proj.weight",
    }
    assert all(parameter.shape[1] == 32 for parameter in width)
    assert not (normuonh_names & nadam_names) and not (nadam_names & width_names)
    assert len(normuonh_names) + len(nadam_names) + len(width_names) == len(names)

    normuonh_optimizer, nadam_optimizer = build_optimizers(model)
    assert isinstance(normuonh_optimizer, NorMuonH)
    assert isinstance(nadam_optimizer, torch.optim.NAdam)
    assert DEFAULT_NORMUONH_LR == 6e-3
    assert normuonh_optimizer.param_groups[0]["lr"] == DEFAULT_NORMUONH_LR
    assert normuonh_optimizer.param_groups[0]["stable_lr"] == DEFAULT_NORMUONH_LR
    assert "weight_decay" not in normuonh_optimizer.param_groups[0]
    nadam_group, width_group = nadam_optimizer.param_groups
    assert model.cfg.mup_ratio == MUP_BASE_DIM / 32 == 48
    assert nadam_group["lr"] == DEFAULT_NADAM_LR == 3e-4
    assert nadam_group["stable_lr"] == DEFAULT_NADAM_LR
    assert nadam_group["rate_name"] == "nadam"
    assert width_group["lr"] == width_group["stable_lr"] == DEFAULT_NADAM_LR * 48
    assert width_group["rate_name"] == "nadam_width"
    assert [id(p) for p in width_group["params"]] == [id(p) for p in width]
    for group in nadam_optimizer.param_groups:
        assert group["betas"] == DEFAULT_NADAM_BETAS
        assert group["momentum_decay"] == NADAM_MOMENTUM_DECAY
        assert group["weight_decay"] == 0

    class HalfSchedule:
        @staticmethod
        def rate_at(_step, stable_lr):
            return stable_lr / 2

    rates = apply_schedule(
        [normuonh_optimizer, nadam_optimizer], HalfSchedule(), step=1
    )
    assert rates == {
        "normuonh": DEFAULT_NORMUONH_LR / 2,
        "nadam": DEFAULT_NADAM_LR / 2,
        "nadam_width": DEFAULT_NADAM_LR * 48 / 2,
    }
    assert normuonh_optimizer.param_groups[0]["lr"] == rates["normuonh"]
    assert nadam_group["lr"] == rates["nadam"]
    assert width_group["lr"] == rates["nadam_width"]


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


def test_route_summary_reports_universal_nulls_and_payload_seed():
    from types import SimpleNamespace

    from delta_feedback_experiment.model import DeltaModel, condition_config

    class Validation:
        rows = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(3))

        def batch(self, first, count, device=None):
            rows = self.rows[first : first + count]
            return rows.to(device) if device is not None else rows

    args = SimpleNamespace(eval_rows=2)
    for condition in ("ar", "arf", "arfl"):
        model = DeltaModel(
            condition_config(
                condition,
                vocab_size=97,
                dim=32,
                layers=12 if "l" in condition else 2,
                loop_iterations=2,
                loop_max_iterations=2,
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
        DeltaModel,
        condition_config,
    )
    from delta_feedback_experiment.train import GraphSpec, execution_fields

    model = DeltaModel(
        condition_config(
            "a",
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
        },
        head_flush_every=16,
    )
    evaluator = SimpleNamespace(states={4: None})
    fields = execution_fields(model, trainer, evaluator)
    assert fields == {
        "flex": 1,
        "cce": 1,
        "cuda_graphs": 3,
        "eval_graphs": 1,
        "checkpoint_modes": 0,
        "head_flush_every": 16,
    }


# -- schedule and derived randomness -------------------------------------------


def test_build_schedule_screen_shape():
    args = parse_run_args(["x"])  # the screen at 25 tokens per parameter: 6,716 steps
    assert args.batch_rows == 128 and args.seq_len == 4096
    assert args.batch_rows // args.micro_rows == 128
    schedule = build_schedule(args)
    assert schedule.spans == (134, 0, 5239, 1343)
    assert schedule.total == 6716
    assert schedule.phase(134)[0] == "warmup"
    assert schedule.phase(135)[0] == "heat"
    assert schedule.phase(5374)[0] == "cooldown"
    assert schedule.rate_at(5373, 1.0) == 1.0
    assert schedule.rate_at(6716, 1.0) < 1e-6
    assert feedback_boundary(args, schedule.total) == 5037
    assert schedule.heat_end == 5373


def test_fresh_screen_budgets_match_active_parameter_ratios():
    tokens_per_step = 128 * 4096
    assert tokens_per_step == BATCH_TOKENS == 2**19
    df_active_non_embedding = 140_827_944
    trials = {25: 6_716, 400: 107_444}

    for target_ratio, steps in trials.items():
        realized_ratio = steps * tokens_per_step / df_active_non_embedding
        assert target_ratio <= realized_ratio < target_ratio + tokens_per_step / df_active_non_embedding
        assert parse_run_args(["x", "--tokens-per-param", str(target_ratio)]).steps == steps

    prime = parse_run_args(["x", "--tokens-per-param", "400"])
    schedule = build_schedule(prime)
    assert schedule.spans == (134, 0, 85_821, 21_489)
    assert feedback_boundary(prime, schedule.total) == 80_583
    assert schedule.heat_end == 85_955


def test_warmup_is_fixed_per_scale():
    """The warmup fraction applies to the shorter of the run and the 25x
    recipe at its geometry: a longer run keeps the 25x warmup, a shorter run
    keeps the fraction."""
    def warmup(argv):
        return build_schedule(parse_run_args(["x", *argv])).warmup_steps
    assert warmup([]) == 134
    assert warmup(["--tokens-per-param", "400"]) == 134
    assert warmup(["--tokens-per-param", "50", "--condition", "arfl"]) == 134
    assert warmup(["--steps", "100"]) == 2
    assert warmup(["--scale", "bridge"]) == 397
    assert warmup(["--scale", "bridge", "--tokens-per-param", "400"]) == 397
    assert warmup(["--scale", "flagship", "--tokens-per-param", "400"]) == 1_051


def test_scale_presets_and_ratio_derive_the_schedule():
    """--condition, --scale, and --tokens-per-param address every planned run:
    the preset fills the geometry and batch, the ratio derives the steps from
    the flat stack's active count at that scale, and every condition at a
    scale shares the count so paired runs keep one schedule."""
    screen = parse_run_args(["x"])
    assert (screen.dim, screen.layers, screen.seq_len, screen.batch_rows, screen.micro_rows) == (768, 12, 4096, 128, 1)
    assert screen.steps == 6_716
    bridge = parse_run_args(["x", "--scale", "bridge"])
    assert (bridge.dim, bridge.layers, bridge.heads, bridge.kv_heads) == (1152, 16, 12, 6)
    assert (bridge.intermediate, bridge.pkda_heads) == (4992, 15)
    assert (bridge.seq_len, bridge.batch_rows, bridge.micro_rows, bridge.steps) == (4096, 128, 1, 19_867)
    assert bridge.batch_rows * bridge.seq_len == BATCH_TOKENS
    assert parse_run_args(["x", "--scale", "bridge", "--tokens-per-param", "400"]).steps == 317_867
    flagship = parse_run_args(["x", "--scale", "flagship", "--tokens-per-param", "400"])
    assert (flagship.seq_len, flagship.batch_rows, flagship.micro_rows) == (4096, 128, 1)
    assert flagship.steps == 840_795
    assert flagship.steps * BATCH_TOKENS == 440_818_728_960
    for letters in ("", "a", "arf", "arfl"):
        assert parse_run_args(["x", "--condition", letters]).steps == 6_716
    deeper = parse_run_args(["x", "--scale", "bridge", "--layers", "20"])
    assert (deeper.layers, deeper.dim) == (20, 1152)
    shorter = parse_run_args(["x", "--scale", "bridge", "--seq-len", "2048"])
    assert shorter.batch_rows == 256 and shorter.batch_rows * shorter.seq_len == BATCH_TOKENS
    assert parse_run_args(["x", "--steps", "100"]).steps == 100
    with pytest.raises(SystemExit):
        parse_run_args(["x", "--steps", "100", "--tokens-per-param", "25"])
    with pytest.raises(SystemExit):
        parse_run_args(["x", "--seq-len", "1000"])


def test_resolved_arguments_pin_what_the_operator_fixed():
    """A resume validates what the operator pinned and inherits the rest: an
    explicit --scale pins its fields, an explicit --tokens-per-param pins the
    derived steps, and an untyped schedule stays open for the checkpoint."""
    parser = build_parser()
    args, pinned = resolve_run_args(parser, ["x", "--resume"])
    assert pinned == frozenset({"resume"}) and args.steps is None
    _, pinned = resolve_run_args(parser, ["x", "--resume", "--scale", "bridge"])
    assert {"scale", "dim", "layers", "seq_len", "batch_rows", "micro_rows"} <= pinned
    assert "steps" not in pinned
    _, pinned = resolve_run_args(parser, ["x", "--tokens-per-param", "400"])
    assert {"tokens_per_param", "steps"} <= pinned and "dim" not in pinned
    _, pinned = resolve_run_args(parser, ["x", "--seq-len", "2048"])
    assert {"seq_len", "batch_rows"} <= pinned


def test_fresh_run_uses_authoritative_optimizer_defaults():
    args = build_parser().parse_args(["x"])
    assert args.lr_normuonh == DEFAULT_NORMUONH_LR == 6e-3
    assert args.lr_nadam == DEFAULT_NADAM_LR == 3e-4
    assert not hasattr(args, "lr_embedding")
    assert args.warmup_frac == 0.02
    assert args.cooldown_frac == 0.2


def test_optimizer_learning_rate_flags_are_independent():
    args = build_parser().parse_args(
        [
            "x",
            "--lr-normuonh",
            "0.03",
            "--lr-nadam",
            "0.0002",
        ]
    )
    assert args.lr_normuonh == 0.03
    assert args.lr_nadam == 0.0002


def test_condition_flag_canonicalizes_and_rejects_names():
    assert build_parser().parse_args(["x"]).condition == ""
    assert build_parser().parse_args(["x", "--condition", ""]).condition == ""
    assert build_parser().parse_args(["x", "--condition", "fra"]).condition == "arf"
    for name in ("vanilla", "base", "mhdb", "fbt", "df"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["x", "--condition", name])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["x", "--arm", "df"])


def test_fixed_warmup_steps_flag_is_removed():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["x", "--warmup-steps", "200"])


def test_log_every_flag_is_removed():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["x", "--log-every", "2"])


@pytest.mark.parametrize(
    "flag", ["--lr-h", "--lr-muon", "--wd-muon", "--lr-adam", "--lr-embedding"]
)
def test_legacy_optimizer_flags_are_removed(flag):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["x", flag, "0.01"])


@pytest.mark.parametrize(
    "flag",
    ["--warmup-frac", "--cooldown-frac", "--feedback-start", "--three-pass"],
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


def test_iteration_draw_statistics():
    """The recurrent-depth log-normal Poisson draw, rate shifted by one and
    capped: at the screen defaults E[r] = 3.88, median 4, P(r = 1) = 10.3%,
    P(r = 8) = 8.0%. Its sub-stream leaves the pass draw untouched."""
    args = build_parser().parse_args(["x"])
    assert (args.loop_iterations, args.loop_max_iterations) == (4, 8)
    draws = [draw_iterations(args, step, True) for step in range(1, 20001)]
    assert min(draws) == 1 and max(draws) == 8
    assert sum(draws) / len(draws) == pytest.approx(3.88, abs=0.05)
    assert sorted(draws)[len(draws) // 2] == 4
    assert draws.count(1) / len(draws) == pytest.approx(0.103, abs=0.01)
    assert draws.count(8) / len(draws) == pytest.approx(0.080, abs=0.01)
    assert draws[:8] == [draw_iterations(args, step, True) for step in range(1, 9)]
    assert all(draw_iterations(args, step, False) == 1 for step in range(1, 50))
    args.loop_iterations = 1
    assert all(draw_iterations(args, step, True) == 1 for step in range(1, 50))
    args.loop_iterations, args.loop_max_iterations = 4, 3
    assert max(draw_iterations(args, step, True) for step in range(1, 500)) == 3


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


LOOP_ARGS = ["--layers", "12", "--loop-iterations", "2", "--loop-max-iterations", "3"]
"""The tiny loop geometry: three whole cells and a small draw."""


def condition_args(condition: str) -> list[str]:
    return ["--condition", condition, *(LOOP_ARGS if "l" in condition else [])]


@pytest.mark.parametrize(
    "condition",
    ["", "a", "r", "f", "ar", "af", "rf", "arf", "l", "arl", "afl", "arfl"],
    ids=lambda condition: condition or "plain",
)
def test_tiny_run_completes(tmp_path, capsys, condition):
    tag = f"t-{condition or 'plain'}"
    summary = run(tmp_path, tag, condition_args(condition))
    assert summary["step"] == 8
    assert np.isfinite(summary["loss"])
    assert np.isfinite(summary["val"])
    if "f" in condition:
        assert np.isfinite(summary["val_fused"])
    snapshots = list((tmp_path / "runs").glob(f"{tag}.pt.*"))
    # Protected: feedback boundary (4), cooldown boundary (6), end (8).
    assert {int(p.name.rsplit(".", 1)[1]) for p in snapshots} == {4, 6, 8}
    step_records = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("step ")
    ]
    assert len(step_records) == 8
    assert all("lr_normuonh=" in line for line in step_records)
    assert all("lr_nadam=" in line for line in step_records)
    assert all("lr_nadam_width=" in line for line in step_records)
    assert all("lr_embedding=" not in line for line in step_records)
    assert all("lr_h=" not in line for line in step_records)
    assert all("cell_tok_s=" in line for line in step_records)
    assert all(("| r=" in line) == ("l" in condition) for line in step_records)


@pytest.mark.parametrize("condition", ["arf", "arfl"])
def test_resume_is_exact(tmp_path, capsys, condition):
    full = run(tmp_path, "full", condition_args(condition))
    scrambled = condition_args(condition[::-1])
    half = run(tmp_path, "half", [*scrambled, "--max-steps", "5"])
    assert half["step"] == 5
    resumed = run(tmp_path, "half", [*condition_args(condition), "--resume"])
    assert resumed["step"] == 8
    assert resumed["loss"] == full["loss"]
    assert resumed["val"] == full["val"]
    # The spool folds the log at the resume record and requires its path.
    assert any(
        line.startswith("resume") and "path=" in line
        for line in capsys.readouterr().out.splitlines()
    )


def test_resume_rejects_conflicting_exact_field(tmp_path):
    run(tmp_path, "conf", ["--condition", "arf", "--max-steps", "2"])
    with pytest.raises(ValueError, match="conflicts"):
        run(tmp_path, "conf", ["--condition", "arf", "--resume", "--dim", "64"])
    with pytest.raises(ValueError, match="conflicts"):
        run(tmp_path, "conf", ["--condition", "ar", "--resume"])


def test_continuation_reproduces_the_longer_run(tmp_path, capsys):
    """A finished eight-step arf run continued to sixteen starts from the last
    snapshot both schedules reproduce, the feedback boundary at step 4, and
    from there is the fresh sixteen-step run exactly."""
    base = [*condition_args("arf"), "--warmup-frac", "0"]
    run(tmp_path, "src", base)
    full = run(tmp_path, "full", [*base, "--steps", "16"])
    capsys.readouterr()
    longer = run(tmp_path, "src-16", [*base, "--continue", "src", "--steps", "16"])
    assert longer["step"] == 16
    assert longer["loss"] == full["loss"]
    assert longer["val"] == full["val"]
    out = capsys.readouterr().out.splitlines()
    record = next(line for line in out if line.startswith("continue"))
    assert "source=src" in record and "step=04/16" in record and "src.pt.4" in record
    assert "exact=True" in record
    assert any(line.startswith("run") for line in out)
    assert not any(line.startswith("resume") for line in out)
    steps = [int(line.split("step=")[1].split("/")[0]) for line in out if line.startswith("step ")]
    assert steps == list(range(5, 17))
    # The continuation protects its own boundaries under the longer schedule.
    snapshots = (tmp_path / "runs").glob("src-16.pt.*")
    assert {int(p.name.rsplit(".", 1)[1]) for p in snapshots} == {8, 12, 16}


def test_continuation_keeps_every_setting_but_the_length(tmp_path):
    base = condition_args("arf")
    run(tmp_path, "src2", base)
    with pytest.raises(ValueError, match="longer schedule"):
        run(tmp_path, "src2-same", [*base, "--continue", "src2"])
    with pytest.raises(ValueError, match="keeps every setting"):
        run(tmp_path, "src2-dim", [*base, "--continue", "src2", "--steps", "16", "--dim", "64"])
    with pytest.raises(SystemExit):
        parse_run_args(["x", "--resume", "--continue", "src2"])


def rewrite_latest_version(tmp_path, tag, version):
    snapshots = list((tmp_path / "runs").glob(f"{tag}.pt.*"))
    path = max(snapshots, key=lambda item: int(item.name.rsplit(".", 1)[1]))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["version"] = version
    torch.save(payload, path)


@pytest.mark.parametrize("version", range(9, 26))
def test_resume_rejects_every_legacy_checkpoint(tmp_path, version):
    tag = f"legacy-v{version}"
    run(tmp_path, tag, ["--condition", "", "--max-steps", "5"])
    rewrite_latest_version(tmp_path, tag, version)
    with pytest.raises(ValueError, match="resumable versions \\[26\\]"):
        run(tmp_path, tag, ["--condition", "", "--resume"])


def test_multipass_checkpoint_parity():
    """The guarded larger modes preserve plain-path loss and gradients."""
    from delta_feedback_experiment.model import (
        DeltaModel,
        condition_config,
        multipass,
        multipass_loss,
    )

    geometry = dict(
        vocab_size=97,
        dim=32,
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

    def run(cfg, flag, iterations):
        torch.manual_seed(1)
        model = DeltaModel(cfg)
        model.grad_checkpoint = flag
        outs = multipass(model, tokens, 2, prefix_lens=prefix, iterations=iterations)
        loss, _ = multipass_loss(model, tokens, outs)
        loss.backward()
        grads = torch.cat(
            [p.grad.flatten() for p in model.parameters() if p.grad is not None]
        )
        return loss.item(), grads

    for condition, layers, iterations in (("arf", 2, None), ("arfl", 12, 3)):
        cfg = condition_config(condition, layers=layers, **geometry)
        plain_loss, plain_grads = run(cfg, False, iterations)
        checked_loss, checked_grads = run(cfg, True, iterations)
        assert checked_loss == pytest.approx(plain_loss, rel=1e-6), condition
        assert torch.allclose(plain_grads, checked_grads, rtol=1e-5, atol=1e-7)


def test_checkpoint_policy_is_internal_and_screen_measured():
    from delta_feedback_experiment.model import DeltaModel, condition_config

    args = build_parser().parse_args(["x"])
    cuda = torch.device("cuda")
    with torch.device("meta"):
        model = DeltaModel(condition_config("arf"))
        looped = DeltaModel(condition_config("arfl"))
    assert (args.micro_rows, args.seq_len) == (1, 4096)
    assert not automatic_checkpoint(model, 2, 1, args, cuda)
    assert not automatic_checkpoint(model, 3, 1, args, cuda)
    args.micro_rows = 2
    assert automatic_checkpoint(model, 3, 1, args, cuda)
    args.micro_rows = 1
    # The loop counts executed layers, 8 + 4r per pass at the screen, against
    # the measured raw budget of forty layer-passes (ten cells) at one
    # 4,096-token row, the same tokens as the four 1,024-token rows measured.
    assert not automatic_checkpoint(looped, 3, 1, args, cuda)
    assert not automatic_checkpoint(looped, 1, 8, args, cuda)
    assert not automatic_checkpoint(looped, 2, 3, args, cuda)
    assert automatic_checkpoint(looped, 2, 4, args, cuda)
    assert automatic_checkpoint(looped, 3, 2, args, cuda)
    assert not automatic_checkpoint(looped, 2, 8, args, torch.device("cpu"))
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
        prefix, jitter = micro_draws(args, step, step * 7, 3, 8, 4, torch.device("cpu"))
        assert prefix.shape == (2, 8)
        assert jitter.shape == (2, 8, args.seq_len + 1, 4)
        assert int(prefix.min()) >= 1
        assert int(prefix.max()) <= args.seq_len - 1
        seen.update(prefix.flatten().tolist())
    assert args.seq_len - 1 in seen and 1 in seen
    assert args.seq_len not in seen

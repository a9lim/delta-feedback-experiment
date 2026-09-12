"""Token identity, document boundaries, and interrupted build recovery."""

import json
from dataclasses import dataclass
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
    source_address,
    tokenize,
    verify,
    write_synthetic,
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


def test_tokenized_documents_match_their_source(tmp_path):
    import pyarrow.parquet as pq

    source = parquet_source(tmp_path / "source")
    out = tmp_path / "tokens"
    meta = build(out, source)
    index = json.loads((out / "source.json").read_text())

    assert meta["universe_docs"] == 120
    assert meta["selected_docs"] == 60
    assert meta["eos_id"] == EOS
    assert set(index) == {"files"}
    assert not (out / "parts").exists()
    assert verify(out)["train"]["docs"] == meta["train_docs"]

    rows = {}
    for path in {f["path"] for f in index["files"]}:
        table = pq.ParquetFile(source.root / path).read(columns=["text"])
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

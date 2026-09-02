"""One fixed token stream, tokenized once, read as a prefix by every run.

Layout under a data directory (default ``data/tokens``):

    meta.json            tokenizer/dataset revisions, build versions, counts
    val.bin              the held-out slice — the stream's first tokens
    train.0000.bin ...   uint32 shards, one contiguous stream

The val slice comes first so the train stream can cover both registered
screen budgets without ever touching held-out documents.  Rows are
non-overlapping ``seq_len+1``-token windows addressed by a global row
index, so batch ``step`` is the same bytes for every arm — the paired
data order contract — and a resumed run addresses the identical rows.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from transformer_experiments import telemetry

META = "meta.json"
SHARD_TOKENS = 1 << 28
"""Tokens per train shard (~1 GiB as uint32)."""

CANONICAL_TARGET_TOKENS = 57_000_000_000
CANONICAL_VAL_TOKENS = 30_000_000
CANONICAL_DATASET = "HuggingFaceFW/fineweb-edu"
CANONICAL_CONFIG = "sample-100BT"
CANONICAL_DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
CANONICAL_TOKENIZER = "Qwen/Qwen3-0.6B"
CANONICAL_TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
CANONICAL_DATA_PACKAGES = {
    "datasets": "5.0.1",
    "transformers": "5.16.1",
    "tokenizers": "0.23.1",
    "huggingface-hub": "1.29.0",
}


def data_package_versions() -> dict[str, str]:
    """Return and validate the exact packages that compile the token stream."""
    installed = {}
    for package, expected in CANONICAL_DATA_PACKAGES.items():
        try:
            installed[package] = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"canonical data build requires {package}=={expected}; "
                "install the data-build extra"
            ) from exc
    mismatches = {
        package: (CANONICAL_DATA_PACKAGES[package], actual)
        for package, actual in installed.items()
        if actual != CANONICAL_DATA_PACKAGES[package]
    }
    if mismatches:
        detail = ", ".join(
            f"{package}=={expected} (found {actual})"
            for package, (expected, actual) in mismatches.items()
        )
        raise RuntimeError(
            f"canonical data build package mismatch: {detail}; "
            "install the data-build extra"
        )
    return installed


# -- tokenization --------------------------------------------------------------


def tokenize(
    out_dir: str | Path,
    *,
    target_tokens: int,
    val_tokens: int,
    tokenizer_name: str = CANONICAL_TOKENIZER,
    tokenizer_revision: str = CANONICAL_TOKENIZER_REVISION,
    dataset: str = CANONICAL_DATASET,
    config: str = CANONICAL_CONFIG,
    revision: str = CANONICAL_DATASET_REVISION,
    batch_docs: int = 256,
) -> dict:
    """Stream, tokenize, and shard the corpus until the target is reached.

    Deterministic given the pinned dataset, tokenizer, and build packages:
    documents are taken in the dataset's canonical streaming order, each
    followed by the tokenizer's EOS.  Writes val.bin from the head of the
    stream, then train shards.  Idempotent completion: refuses to run if
    meta.json already exists.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / META).exists():
        raise FileExistsError(f"{out / META} exists; delete the directory to redo")

    packages = data_package_versions()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, revision=tokenizer_revision
    )
    eos = tokenizer.eos_token_id
    assert eos is not None
    stream = load_dataset(
        dataset, name=config, split="train", streaming=True, revision=revision
    )

    telemetry.log(
        "tokenize",
        dataset=dataset,
        config=config,
        revision=revision,
        tokenizer=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
        target=target_tokens,
        val=val_tokens,
    )

    written = 0  # total tokens emitted (val + train)
    val_count = 0
    val_open = True
    shard_index = 0
    buffer: list[np.ndarray] = []
    buffered = 0

    def flush_train() -> None:
        nonlocal buffer, buffered, shard_index
        if not buffered:
            return
        array = np.concatenate(buffer)
        array.tofile(out / f"train.{shard_index:04d}.bin")
        telemetry.log("shard", index=shard_index, tokens=array.size, total=written)
        buffer, buffered = [], 0
        shard_index += 1

    documents = iter(stream)
    done = False
    while not done:
        texts = []
        for _ in range(batch_docs):
            try:
                sample = next(documents)
            except StopIteration:
                done = True
                break
            text = sample.get("text") or ""
            if text:
                texts.append(text)
        if not texts:
            continue
        for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
            ids.append(eos)
            array = np.asarray(ids, dtype=np.uint32)
            written += array.size
            if val_open:
                buffer.append(array)
                val_count += array.size
                if val_count >= val_tokens:
                    np.concatenate(buffer).tofile(out / "val.bin")
                    telemetry.log("val", tokens=val_count)
                    buffer, buffered, val_open = [], 0, False
            else:
                buffer.append(array)
                buffered += array.size
                if buffered >= SHARD_TOKENS:
                    flush_train()
        if written >= target_tokens:
            done = True
    if not val_open:
        flush_train()

    meta = {
        "tokenizer": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "eos_id": int(eos),
        "dataset": dataset,
        "config": config,
        "revision": revision,
        "packages": packages,
        "val_tokens": int(val_count),
        "train_tokens": int(written - val_count),
        "vocab_size": 151936,
    }
    (out / META).write_text(json.dumps(meta, indent=2))
    telemetry.log("tokenized", train=meta["train_tokens"], val=meta["val_tokens"])
    return meta


# -- reading -------------------------------------------------------------------


class TokenData:
    """Row-addressed reader over one split's shard files.

    Row r is tokens [r·(seq_len+1), (r+1)·(seq_len+1)) of the split's
    contiguous stream; reads spanning shard boundaries are stitched.
    """

    def __init__(self, paths: list[Path], seq_len: int):
        if not paths:
            raise FileNotFoundError("no token shards found")
        self.seq_len = seq_len
        self.maps = [np.memmap(path, dtype=np.uint32, mode="r") for path in paths]
        self.offsets = np.concatenate([[0], np.cumsum([m.size for m in self.maps])])
        self.total_tokens = int(self.offsets[-1])

    @classmethod
    def load(cls, directory: str | Path, split: str, seq_len: int) -> TokenData:
        directory = Path(directory)
        if split == "val":
            paths = [directory / "val.bin"]
        else:
            paths = sorted(directory.glob("train.*.bin"))
        return cls([p for p in paths if p.exists()], seq_len)

    @property
    def rows(self) -> int:
        return self.total_tokens // (self.seq_len + 1)

    def read(self, start: int, length: int) -> np.ndarray:
        """length tokens from global offset start, across shards."""
        pieces = []
        while length > 0:
            shard = int(np.searchsorted(self.offsets, start, side="right")) - 1
            local = start - int(self.offsets[shard])
            take = min(length, self.maps[shard].size - local)
            pieces.append(self.maps[shard][local : local + take])
            start += take
            length -= take
        return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]

    def batch(self, first_row: int, n_rows: int, device=None) -> torch.Tensor:
        """Rows [first_row, first_row+n_rows) as int64 [n, seq_len+1]."""
        width = self.seq_len + 1
        if (first_row + n_rows) * width > self.total_tokens:
            raise IndexError(
                f"rows {first_row}..{first_row + n_rows} exceed the stream "
                f"({self.rows} rows of {width})"
            )
        flat = self.read(first_row * width, n_rows * width)
        array = np.asarray(flat, dtype=np.int64).reshape(n_rows, width)
        tensor = torch.from_numpy(array)
        return tensor.to(device) if device is not None else tensor


def write_synthetic(
    directory: str | Path,
    *,
    train_tokens: int,
    val_tokens: int,
    vocab: int = 97,
    seed: int = 0,
    shards: int = 2,
) -> None:
    """A tiny fake corpus for offline tests and smoke runs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rng.integers(0, vocab, val_tokens, dtype=np.uint32).tofile(directory / "val.bin")
    per_shard = train_tokens // shards
    for index in range(shards):
        count = (
            per_shard if index < shards - 1 else train_tokens - per_shard * (shards - 1)
        )
        rng.integers(0, vocab, count, dtype=np.uint32).tofile(
            directory / f"train.{index:04d}.bin"
        )
    (directory / META).write_text(
        json.dumps(
            {
                "tokenizer": "synthetic",
                "tokenizer_revision": None,
                "eos_id": 0,
                "dataset": "synthetic",
                "config": None,
                "revision": None,
                "packages": {},
                "val_tokens": val_tokens,
                "train_tokens": train_tokens,
                "vocab_size": vocab,
            },
            indent=2,
        )
    )


def read_meta(directory: str | Path) -> dict:
    return json.loads((Path(directory) / META).read_text())

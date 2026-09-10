"""One shuffled token stream over a fixed document universe, read as a prefix
by every run.

Layout under a store directory (default ``data/tokens``):

    meta.json            source, tokenizer, packages, seed, counts, dump names
    source.json          the document universe: every source file's row groups
    val.bin              the held-out slice: the shuffled stream's first documents
    val.docs.npy         one record per held-out document (start, source, dump)
    train.0000.bin ...   uint32 shards, one contiguous stream after the slice
    train.docs.npy       one record per training document

The universe is every document of the pinned parquet source, addressed by
its row in file order. A keyed bijection (:class:`Shuffle`) sends addresses
to stream positions, so consecutive documents come from unrelated files and
crawls, and a store of any size is a prefix of the same stream: a build that
stops at 57B tokens is byte-identical to the first 57B tokens of one that
stops at 170B. Rows are non-overlapping ``seq_len+1``-token windows addressed
by a global row index, so batch ``step`` is the same bytes for every
condition (the paired data order contract) and a resumed run addresses the
identical rows.

The sidecar records where every document starts in its split's stream, its
address in the universe (which names the parquet row that holds its text,
URL, and score), and its crawl, as an index into ``meta["dumps"]``.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from itertools import pairwise
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from transformer_experiments import telemetry

META = "meta.json"
SOURCE = "source.json"
SHARD_TOKENS = 1 << 28
"""Tokens per train shard (~1 GiB as uint32)."""
WRITE_BUFFER_TOKENS = 1 << 24
"""Gather up to about 64 MiB before each sequential output write."""

DOC_DTYPE = np.dtype([("start", "<i8"), ("source", "<i8"), ("dump", "<i2")])
"""One sidecar record: split-local start offset, universe address, dump id."""

CANONICAL_TARGET_TOKENS = 57_000_000_000
"""Stored tokens, held-out slice included, for the screen's 400x schedule;
the bridge's 400x schedule needs 167B (``docs/scaling.md``)."""
CANONICAL_VAL_TOKENS = 30_000_000
CANONICAL_SHUFFLE_SEED = 0
CANONICAL_TOKENS_PER_DOC = 900
"""A lower bound on the mean document length in stored tokens, which sizes
the selection; ``sample-350BT`` documents average about 1,090."""
CANONICAL_DATASET = "HuggingFaceFW/fineweb-edu"
CANONICAL_CONFIG = "sample-350BT"
CANONICAL_DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
CANONICAL_TOKENIZER = "Qwen/Qwen3-0.6B-Base"
CANONICAL_TOKENIZER_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
VOCAB_SIZE = 151_936
CANONICAL_DATA_PACKAGES = {
    "transformers": "5.16.1",
    "tokenizers": "0.23.2",
    "huggingface-hub": "1.30.0",
    "pyarrow": "25.0.1",
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


# -- the shuffle ---------------------------------------------------------------

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _mix(x: np.ndarray) -> np.ndarray:
    """The splitmix64 finalizer on a uint64 array; wraps modulo 2^64."""
    x = x ^ (x >> np.uint64(30))
    x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    x = x * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


class Shuffle:
    """A keyed bijection on ``[0, size)``: addresses to stream positions.

    A balanced Feistel network over the next even power of two, walked
    until it lands inside the domain (cycle walking), so it is a permutation
    of exactly ``size`` elements, stateless, and the same function on every
    machine that shares the seed. ``inverse`` recovers the address of a
    position.
    """

    ROUNDS = 8

    def __init__(self, size: int, seed: int):
        if size < 1:
            raise ValueError("a shuffle needs a non-empty domain")
        self.size = size
        self.seed = seed
        bits = max(2, (size - 1).bit_length())
        self.half = np.uint64((bits + 1) // 2)
        self.mask = np.uint64((1 << int(self.half)) - 1)
        rounds = np.arange(self.ROUNDS, dtype=np.uint64)
        seeds = np.full(self.ROUNDS, seed, dtype=np.uint64)
        self.keys = _mix(seeds * _GOLDEN + _mix(rounds + np.uint64(1)))

    def _feistel(self, x: np.ndarray, forward: bool) -> np.ndarray:
        left, right = x >> self.half, x & self.mask
        order = range(self.ROUNDS) if forward else range(self.ROUNDS - 1, -1, -1)
        for r in order:
            if forward:
                left, right = right, left ^ (_mix(right ^ self.keys[r]) & self.mask)
            else:
                left, right = right ^ (_mix(left ^ self.keys[r]) & self.mask), left
        return (left << self.half) | right

    def _walk(self, values, forward: bool) -> np.ndarray:
        x = np.atleast_1d(np.asarray(values, dtype=np.uint64))
        if x.size and int(x.max()) >= self.size:
            raise ValueError("value outside the shuffle's domain")
        y = self._feistel(x, forward)
        outside = y >= np.uint64(self.size)
        while outside.any():
            y[outside] = self._feistel(y[outside], forward)
            outside = y >= np.uint64(self.size)
        return y.astype(np.int64)

    def __call__(self, addresses) -> np.ndarray:
        """Stream positions of universe addresses."""
        return self._walk(addresses, True)

    def inverse(self, positions) -> np.ndarray:
        """Universe addresses of stream positions."""
        return self._walk(positions, False)


# -- the source ----------------------------------------------------------------


def source_prefix(config: str) -> str:
    """The repository path holding a configuration's parquet files."""
    if config.startswith("sample-"):
        return "sample/" + config.removeprefix("sample-")
    if config == "default":
        return "data"
    raise ValueError(f"unknown source configuration {config!r}")


class Source(Protocol):
    """Where parquet files come from; picklable so workers can share it."""

    def list_files(self) -> list[str]: ...

    def dumps(self) -> list[str]: ...

    def row_groups(self, path: str) -> list[int]: ...

    def fetch(self, path: str) -> Path: ...

    def release(self, path: str) -> None: ...


@dataclass(frozen=True)
class HubSource:
    """The pinned Hugging Face dataset, staged one file at a time."""

    dataset: str
    config: str
    revision: str
    scratch: Path

    def list_files(self) -> list[str]:
        from huggingface_hub import HfApi

        entries = HfApi().list_repo_tree(
            self.dataset,
            repo_type="dataset",
            path_in_repo=source_prefix(self.config),
            revision=self.revision,
            recursive=True,
        )
        return sorted(e.path for e in entries if e.path.endswith(".parquet"))

    def dumps(self) -> list[str]:
        """Every crawl of the dataset at this revision, the ``data/`` folders."""
        from huggingface_hub import HfApi

        entries = HfApi().list_repo_tree(
            self.dataset,
            repo_type="dataset",
            path_in_repo="data",
            revision=self.revision,
        )
        names = [e.path.split("/")[-1] for e in entries]
        return sorted(name for name in names if "." not in name)

    def row_groups(self, path: str) -> list[int]:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        remote = f"datasets/{self.dataset}@{self.revision}/{path}"
        for attempt in range(5):
            try:
                with HfFileSystem().open(remote, "rb") as handle:
                    metadata = pq.ParquetFile(handle).metadata
                return [
                    metadata.row_group(i).num_rows
                    for i in range(metadata.num_row_groups)
                ]
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def fetch(self, path: str) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                self.dataset,
                path,
                repo_type="dataset",
                revision=self.revision,
                local_dir=self.scratch,
            )
        )

    def release(self, path: str) -> None:
        # Only this file: the scratch's download cache is shared by workers
        # with files in flight, and the build removes the scratch at the end.
        (self.scratch / path).unlink(missing_ok=True)


@dataclass(frozen=True)
class LocalSource:
    """Parquet files under a directory, for tests and offline builds."""

    root: Path

    def list_files(self) -> list[str]:
        return sorted(
            str(p.relative_to(self.root)) for p in self.root.rglob("*.parquet")
        )

    def dumps(self) -> list[str]:
        import pyarrow.parquet as pq

        names: set[str] = set()
        for path in self.list_files():
            column = pq.read_table(self.root / path, columns=["dump"]).column("dump")
            names.update(column.unique().to_pylist())
        return sorted(names)

    def row_groups(self, path: str) -> list[int]:
        import pyarrow.parquet as pq

        metadata = pq.ParquetFile(self.root / path).metadata
        return [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]

    def fetch(self, path: str) -> Path:
        return self.root / path

    def release(self, path: str) -> None:
        return None


def index_source(source: Source) -> dict:
    """The document universe: every file's row-group row counts, and the
    crawl names the sidecar's dump ids index, fixed before any text is read."""
    files = []
    for path in source.list_files():
        files.append({"path": path, "row_groups": source.row_groups(path)})
        telemetry.log("index", file=path, rows=sum(files[-1]["row_groups"]))
    return {"files": files, "dumps": source.dumps()}


def universe_size(index: dict) -> int:
    return sum(sum(f["row_groups"]) for f in index["files"])


def file_bases(index: dict) -> list[int]:
    """The universe address of each file's first row."""
    bases, total = [], 0
    for file in index["files"]:
        bases.append(total)
        total += sum(file["row_groups"])
    return bases


def source_address(index: dict, address: int) -> tuple[str, int]:
    """The (file path, row within file) a universe address names."""
    bases = file_bases(index)
    which = int(np.searchsorted(bases, address, side="right")) - 1
    return index["files"][which]["path"], address - bases[which]


def load_tokenizer(
    name: str = CANONICAL_TOKENIZER, revision: str = CANONICAL_TOKENIZER_REVISION
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    assert tokenizer.eos_token_id is not None
    return tokenizer


# -- the select stage ----------------------------------------------------------


@dataclass(frozen=True)
class _Build:
    source: Source
    tokenizer: Callable[[], object]
    parts: Path
    seed: int
    universe: int
    first: int
    selected: int
    bases: tuple[int, ...]
    row_groups: tuple[tuple[int, ...], ...]
    paths: tuple[str, ...]
    dumps: tuple[str, ...]


PART_DTYPE = np.dtype(
    [("position", "<i8"), ("offset", "<i8"), ("length", "<i4"), ("dump", "<i2")]
)

_TOKENIZERS: dict[str, object] = {}


def _select_file(
    build: _Build, which: int, local: Path | None = None
) -> tuple[int, int]:
    """Tokenize one file's selected documents into its part; returns counts."""
    import pyarrow.parquet as pq

    key = repr(build.tokenizer)  # one load per worker process, not per file
    tokenizer = _TOKENIZERS.get(key)
    if tokenizer is None:
        tokenizer = _TOKENIZERS[key] = build.tokenizer()
    eos = tokenizer.eos_token_id
    shuffle = Shuffle(build.universe, build.seed)
    path = build.paths[which]
    if local is None:
        local = build.source.fetch(path)
    tokens_path = build.parts / f"{which:04d}.tokens.bin"
    index_path = build.parts / f"{which:04d}.index.npy"
    crawl_ids = {name: i for i, name in enumerate(build.dumps)}
    records: list[tuple[int, int, int, int]] = []
    offset = docs = 0
    parquet = pq.ParquetFile(local)
    expected = build.row_groups[which]
    if parquet.metadata.num_row_groups != len(expected):
        raise RuntimeError(f"{path}: row groups differ from the source index")
    with tokens_path.open("wb") as out:
        base = build.bases[which]
        for group, rows in enumerate(expected):
            table = parquet.read_row_group(group, columns=["text", "dump"])
            if table.num_rows != rows:
                raise RuntimeError(f"{path}: row group {group} differs from the index")
            positions = shuffle(base + np.arange(rows, dtype=np.int64))
            keep = np.nonzero(
                (positions >= build.first) & (positions < build.selected)
            )[0]
            base += rows
            if keep.size == 0:
                continue
            texts = table.column("text").take(keep).to_pylist()
            crawls = table.column("dump").take(keep).to_pylist()
            encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
            for text, ids, crawl, position in zip(
                texts, encoded, crawls, positions[keep]
            ):
                if not text:
                    continue
                ids.append(eos)
                array = np.asarray(ids, dtype=np.uint32)
                array.tofile(out)
                if crawl not in crawl_ids:
                    raise RuntimeError(f"{path}: crawl {crawl!r} is not in the index")
                records.append((int(position), offset, array.size, crawl_ids[crawl]))
                offset += array.size
                docs += 1
    index = np.array(records, dtype=PART_DTYPE)
    staging = index_path.with_suffix(".npy.tmp")
    with staging.open("wb") as handle:  # np.save would append its own suffix
        np.save(handle, index)
    staging.replace(index_path)
    build.source.release(path)
    telemetry.log("part", file=path, docs=docs, tokens=offset)
    return docs, offset


def _select_files(build: _Build, pending: list[int]) -> None:
    """One worker, with at most one upcoming file downloading during encoding.

    The fetch thread lives inside the worker, after the process pool forks.
    Completed parts remain the resume markers; a prefetched source file left
    by a failure is reusable on the next attempt.
    """
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=1) as downloads:
        fetched = downloads.submit(build.source.fetch, build.paths[pending[0]])
        for i, which in enumerate(pending):
            local = fetched.result()
            if i + 1 < len(pending):
                fetched = downloads.submit(
                    build.source.fetch, build.paths[pending[i + 1]]
                )
            _select_file(build, which, local)


def _select(build: _Build, workers: int) -> None:
    pending = [
        which
        for which in range(len(build.paths))
        if not (build.parts / f"{which:04d}.index.npy").exists()
    ]
    if not pending:
        return
    if workers <= 1:
        _select_files(build, pending)
        return
    # The Rust tokenizer threads every worker across all cores by default;
    # leave half the machine to whatever else is running.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault(
        "RAYON_NUM_THREADS", str(max(1, (os.cpu_count() or 2) // (2 * workers)))
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        chunks = [pending[i::workers] for i in range(min(workers, len(pending)))]
        for _ in pool.map(_select_files, [build] * len(chunks), chunks):
            pass


# -- the write stage -----------------------------------------------------------


class _ShardWriter:
    """A contiguous uint32 stream cut into files at exact shard boundaries."""

    def __init__(self, directory: Path, name: str, shard_tokens: int | None):
        self.directory = directory
        self.name = name
        self.shard_tokens = shard_tokens
        self.written = 0
        self.handle = None
        self.in_shard = 0
        self.shards = 0

    @classmethod
    def continuing(cls, directory: Path, name: str, shard_tokens: int) -> _ShardWriter:
        """A writer positioned after a finished split's shards, appending to
        the last one while it is short of a whole shard."""
        writer = cls(directory, name, shard_tokens)
        shards = sorted(directory.glob(f"{name}.*.bin"))
        sizes = [path.stat().st_size // 4 for path in shards]
        writer.written = sum(sizes)
        writer.shards = len(shards)
        if shards and sizes[-1] < shard_tokens:
            writer.shards -= 1
            writer.handle = shards[-1].open("ab")
            writer.in_shard = sizes[-1]
        return writer

    def _open(self):
        if self.shard_tokens is None:
            path = self.directory / f"{self.name}.bin"
        else:
            path = self.directory / f"{self.name}.{self.shards:04d}.bin"
        self.handle = path.open("wb")
        self.in_shard = 0

    def write(self, array: np.ndarray) -> None:
        while array.size:
            if self.handle is None:
                self._open()
            room = (
                array.size
                if self.shard_tokens is None
                else min(array.size, self.shard_tokens - self.in_shard)
            )
            array[:room].tofile(self.handle)
            self.in_shard += room
            self.written += room
            array = array[room:]
            if self.shard_tokens is not None and self.in_shard == self.shard_tokens:
                self.close()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
            self.shards += 1


def _gather_tokens(
    pieces: list[np.ndarray], pool: ThreadPoolExecutor, readers: int
) -> np.ndarray:
    """Fault in independent document ranges concurrently, retaining stream order."""
    readers = min(readers, len(pieces))
    if readers <= 1:
        return np.concatenate(pieces)
    offsets = np.empty(len(pieces) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum([piece.size for piece in pieces], out=offsets[1:])
    output = np.empty(int(offsets[-1]), dtype=np.uint32)
    bounds = np.linspace(0, len(pieces), readers + 1, dtype=int)
    futures = [
        pool.submit(
            np.concatenate,
            pieces[first:last],
            out=output[offsets[first] : offsets[last]],
        )
        for first, last in pairwise(bounds)
    ]
    for future in futures:
        future.result()
    return output


def _write(
    out: Path,
    parts: Path,
    n_files: int,
    *,
    target_tokens: int,
    val_tokens: int,
    shuffle: Shuffle,
    extend: bool,
    readers: int = 8,
) -> dict:
    """Walk the selection in stream order into val.bin, shards, and sidecars.

    The held-out slice is the stream's first documents up to ``val_tokens``;
    training then runs until it holds ``target_tokens - val_tokens`` tokens,
    ending on a document boundary. Extending appends to a finished store's
    train shards and sidecar under the same rule, so it lands on the bytes a
    fresh build at the larger target would.
    """
    started = time.monotonic()
    telemetry.log("assemble", files=n_files, target=target_tokens, readers=readers)
    indices, maps = [], []
    for which in range(n_files):
        index_path = parts / f"{which:04d}.index.npy"
        if not index_path.exists():
            raise FileNotFoundError(
                f"part {which} missing: the select stage is incomplete"
            )
        part = np.load(index_path)
        merged = np.empty(part.size, dtype=_MERGED_DTYPE)
        merged["position"] = part["position"]
        merged["part"] = which
        merged["offset"] = part["offset"]
        merged["length"] = part["length"]
        merged["dump"] = part["dump"]
        indices.append(merged)
        tokens_path = parts / f"{which:04d}.tokens.bin"
        tokens = (
            np.memmap(tokens_path, dtype=np.uint32, mode="r")
            if tokens_path.stat().st_size
            else np.empty(0, dtype=np.uint32)
        )
        # Stream order jumps between small documents across every source part.
        # Sequential mmap readahead pulls in mostly unused neighbouring pages
        # when the parts exceed RAM. The hint changes paging, never token bytes.
        if tokens.size and hasattr(mmap, "MADV_RANDOM"):
            tokens._mmap.madvise(mmap.MADV_RANDOM)
        maps.append(tokens)
    index = np.concatenate(indices) if indices else np.empty(0, dtype=_MERGED_DTYPE)
    index = index[np.argsort(index["position"], kind="stable")]
    del indices
    positions, parts_col = index["position"], index["part"]
    offsets, lengths, dumps = index["offset"], index["length"], index["dump"]

    if extend:
        writers = {"train": _ShardWriter.continuing(out, "train", SHARD_TOKENS)}
        split, val_docs = "train", 0
    else:
        writers = {
            "val": _ShardWriter(out, "val", None),
            "train": _ShardWriter(out, "train", SHARD_TOKENS),
        }
        split, val_docs = "val", None
    train_target = target_tokens - val_tokens
    doc_start = np.empty(index.size, dtype=np.int64)
    doc_position = np.empty(index.size, dtype=np.int64)
    doc_dump = np.empty(index.size, dtype=np.int16)
    count = 0
    buffer: list[np.ndarray] = []
    buffered = 0
    initial_tokens = sum(writer.written for writer in writers.values())
    next_report = started + 30

    def flush() -> None:
        nonlocal buffer, buffered, next_report
        if buffered:
            writers[split].write(_gather_tokens(buffer, reads, readers))
            now = time.monotonic()
            if now >= next_report:
                written = sum(writer.written for writer in writers.values())
                telemetry.log(
                    "assemble_progress",
                    train=writers["train"].written,
                    target_train=train_target,
                    tokens_per_second=(written - initial_tokens) / (now - started),
                    elapsed_s=now - started,
                )
                next_report = now + 30
        buffer, buffered = [], 0

    with ThreadPoolExecutor(max_workers=readers) as reads:
        for i in range(index.size):
            length = int(lengths[i])
            if (
                split == "val"
                and writers["val"].written + buffered + length > val_tokens
            ):
                flush()
                writers["val"].close()
                split = "train"
                val_docs = count
            if split == "train" and writers["train"].written + buffered >= train_target:
                break
            doc_start[count] = writers[split].written + buffered
            doc_position[count] = positions[i]
            doc_dump[count] = dumps[i]
            count += 1
            offset = int(offsets[i])
            buffer.append(maps[int(parts_col[i])][offset : offset + length])
            buffered += length
            if buffered >= WRITE_BUFFER_TOKENS:
                flush()
        flush()
    for writer in writers.values():
        writer.close()
    if split == "val":
        raise RuntimeError(
            f"selection exhausted inside the held-out slice at "
            f"{writers['val'].written:,} of {val_tokens:,} tokens; rebuild with "
            "a smaller --tokens-per-doc"
        )
    if not extend and not val_docs:
        raise RuntimeError("no document fits the held-out slice")
    if writers["train"].written < train_target:
        raise RuntimeError(
            f"selection exhausted at {writers['train'].written:,} of "
            f"{train_target:,} training tokens; rebuild with a smaller "
            "--tokens-per-doc"
        )

    def sidecar(first: int, last: int) -> np.ndarray:
        records = np.empty(last - first, dtype=DOC_DTYPE)
        records["start"] = doc_start[first:last]
        records["source"] = shuffle.inverse(doc_position[first:last])
        records["dump"] = doc_dump[first:last]
        return records

    counts = {}
    if not extend:
        np.save(out / "val.docs.npy", sidecar(0, val_docs))
        counts["val_docs"] = val_docs
        counts["val_tokens"] = writers["val"].written
    train_docs = sidecar(val_docs, count)
    if extend:
        train_docs = np.concatenate([np.load(out / "train.docs.npy"), train_docs])
    np.save(out / "train.docs.npy", train_docs)
    counts["train_docs"] = int(train_docs.size)
    counts["train_tokens"] = writers["train"].written
    counts["unused_selected"] = int(index.size - count)
    telemetry.log(
        "assembled", train=counts["train_tokens"], elapsed_s=time.monotonic() - started
    )
    return counts


_MERGED_DTYPE = np.dtype(
    [
        ("position", "<i8"),
        ("part", "<i4"),
        ("offset", "<i8"),
        ("length", "<i4"),
        ("dump", "<i2"),
    ]
)


# -- the build -----------------------------------------------------------------


def tokenize(
    out_dir: str | Path,
    *,
    target_tokens: int,
    val_tokens: int,
    seed: int = CANONICAL_SHUFFLE_SEED,
    tokens_per_doc: int = CANONICAL_TOKENS_PER_DOC,
    workers: int = 1,
    readers: int = 8,
    source: Source | None = None,
    tokenizer: Callable[[], object] | None = None,
    tokenizer_name: str = CANONICAL_TOKENIZER,
    tokenizer_revision: str = CANONICAL_TOKENIZER_REVISION,
    dataset: str = CANONICAL_DATASET,
    config: str = CANONICAL_CONFIG,
    revision: str = CANONICAL_DATASET_REVISION,
    scratch: str | Path | None = None,
    check_packages: bool = True,
    extend: bool = False,
) -> dict:
    """Build the shuffled stream's first ``target_tokens`` stored tokens.

    Three resumable stages under ``out_dir``: index the source into
    ``source.json``; select the documents whose stream position falls below
    ``target_tokens / tokens_per_doc`` and tokenize them into per-file parts;
    then write the parts in stream order, the held-out slice first. The
    stream is a pure function of the source revision, the tokenizer revision,
    and the seed: any two builds agree on their common prefix. Refuses to run
    if ``meta.json`` already exists unless ``extend`` continues that store to
    a larger target under the same settings, selecting from the position
    after its last document and appending; a partial build resumes.
    """
    if readers < 1:
        raise ValueError("readers must be at least 1")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    previous = None
    if (out / META).exists():
        if not extend:
            raise FileExistsError(f"{out / META} exists; delete the directory to redo")
        previous = read_meta(out)
        settings = {
            "dataset": dataset,
            "config": config,
            "revision": revision,
            "tokenizer": tokenizer_name,
            "tokenizer_revision": tokenizer_revision,
            "shuffle_seed": seed,
            "val_target": val_tokens,
        }
        for key, value in settings.items():
            if previous.get(key) != value:
                raise ValueError(
                    f"--continue: {key} is {previous.get(key)!r} in the store, "
                    f"{value!r} requested"
                )
        if previous["train_tokens"] >= target_tokens - val_tokens:
            telemetry.log(
                "tokenized",
                train=previous["train_tokens"],
                val=previous["val_tokens"],
                docs=previous["train_docs"] + previous["val_docs"],
                unchanged=True,
            )
            return previous
    elif extend:
        raise FileNotFoundError(f"--continue: no store at {out}")
    packages = data_package_versions() if check_packages else {}
    if previous is not None and check_packages and packages != previous["packages"]:
        raise ValueError(
            f"--continue: the store was built with {previous['packages']}, "
            f"this environment has {packages}"
        )
    if source is None:
        scratch = Path(scratch) if scratch is not None else out / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        source = HubSource(dataset, config, revision, scratch)
    if tokenizer is None:
        tokenizer = _HubTokenizer(tokenizer_name, tokenizer_revision)
    eos = int(tokenizer().eos_token_id)
    telemetry.log(
        "tokenize",
        dataset=dataset,
        config=config,
        revision=revision,
        tokenizer=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
        seed=seed,
        target=target_tokens,
        val=val_tokens,
        workers=workers,
        readers=readers,
    )

    source_path = out / SOURCE
    if source_path.exists():
        index = json.loads(source_path.read_text())
    else:
        index = index_source(source)
        source_path.write_text(json.dumps(index))
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if previous is not None and previous["source_sha256"] != source_sha256:
        raise ValueError("--continue: source.json differs from the store's")
    universe = universe_size(index)
    if universe == 0:
        raise RuntimeError("the source holds no documents")
    shuffle = Shuffle(universe, seed)
    selected = min(universe, math.ceil(target_tokens / tokens_per_doc))
    first = 0
    if previous is not None:
        last = int(np.load(out / "train.docs.npy")["source"][-1])
        first = int(shuffle([last])[0]) + 1
        if selected <= first:
            raise ValueError(
                "--continue: the selection ends inside the store; lower "
                "--tokens-per-doc"
            )
    telemetry.log(
        "universe",
        docs=universe,
        first=first,
        selected=selected,
        files=len(index["files"]),
    )

    parts = out / "parts"
    parts.mkdir(exist_ok=True)
    build = _Build(
        source=source,
        tokenizer=tokenizer,
        parts=parts,
        seed=seed,
        universe=universe,
        first=first,
        selected=selected,
        bases=tuple(file_bases(index)),
        row_groups=tuple(tuple(f["row_groups"]) for f in index["files"]),
        paths=tuple(f["path"] for f in index["files"]),
        dumps=tuple(index["dumps"]),
    )
    _select(build, workers)

    counts = _write(
        out,
        parts,
        len(index["files"]),
        target_tokens=target_tokens,
        val_tokens=val_tokens,
        shuffle=shuffle,
        extend=previous is not None,
        readers=readers,
    )
    if previous is None:
        meta = {
            "tokenizer": tokenizer_name,
            "tokenizer_revision": tokenizer_revision,
            "eos_id": eos,
            "dataset": dataset,
            "config": config,
            "revision": revision,
            "packages": packages,
            "source_sha256": source_sha256,
            "shuffle_seed": seed,
            "universe_docs": universe,
            "dumps": index["dumps"],
            "vocab_size": VOCAB_SIZE,
            "val_target": val_tokens,
            "val_tokens": counts["val_tokens"],
            "val_docs": counts["val_docs"],
        }
    else:
        meta = dict(previous)
    meta.update(
        target_tokens=target_tokens,
        tokens_per_doc=tokens_per_doc,
        selected_docs=selected,
        unused_selected=counts["unused_selected"],
        train_tokens=counts["train_tokens"],
        train_docs=counts["train_docs"],
    )
    (out / META).write_text(json.dumps(meta, indent=2))
    shutil.rmtree(parts)
    if isinstance(source, HubSource):
        shutil.rmtree(source.scratch, ignore_errors=True)
    telemetry.log(
        "tokenized",
        train=meta["train_tokens"],
        val=meta["val_tokens"],
        docs=meta["train_docs"] + meta["val_docs"],
    )
    return meta


@dataclass(frozen=True)
class _HubTokenizer:
    name: str
    revision: str

    def __call__(self):
        return load_tokenizer(self.name, self.revision)


# -- reading -------------------------------------------------------------------


class TokenData:
    """Row-addressed reader over one split's shard files.

    Row r is tokens [r·(seq_len+1), (r+1)·(seq_len+1)) of the split's
    contiguous stream; reads spanning shard boundaries are stitched.
    """

    def __init__(self, paths: list[Path], seq_len: int, docs: Path | None = None):
        if not paths:
            raise FileNotFoundError("no token shards found")
        self.seq_len = seq_len
        self.maps = [np.memmap(path, dtype=np.uint32, mode="r") for path in paths]
        self.offsets = np.concatenate([[0], np.cumsum([m.size for m in self.maps])])
        self.total_tokens = int(self.offsets[-1])
        self._docs_path = docs

    @classmethod
    def load(cls, directory: str | Path, split: str, seq_len: int) -> TokenData:
        directory = Path(directory)
        if split == "val":
            paths = [directory / "val.bin"]
        else:
            paths = sorted(directory.glob("train.*.bin"))
        docs = directory / f"{split}.docs.npy"
        return cls(
            [p for p in paths if p.exists()], seq_len, docs if docs.exists() else None
        )

    @property
    def docs(self) -> np.ndarray | None:
        """The split's document sidecar (``DOC_DTYPE``), or None without one."""
        if self._docs_path is None:
            return None
        return np.load(self._docs_path, mmap_mode="r")

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


def verify(directory: str | Path, *, sample_docs: int = 200_000) -> dict:
    """Check a store against its meta and sidecars; raise on any mismatch.

    Token counts must match the files, sidecar starts must be increasing and
    begin at zero, and the token before each sampled document's start must be
    the EOS that closed its predecessor.
    """
    directory = Path(directory)
    meta = read_meta(directory)
    summary = {"directory": str(directory)}
    rng = np.random.default_rng(0)
    for split in ("val", "train"):
        data = TokenData.load(directory, split, 1)
        expected = meta[f"{split}_tokens"]
        if data.total_tokens != expected:
            raise RuntimeError(
                f"{split}: {data.total_tokens:,} tokens on disk, meta says {expected:,}"
            )
        docs = data.docs
        if docs is None:
            raise RuntimeError(f"{split}: no sidecar")
        if docs.size != meta[f"{split}_docs"]:
            raise RuntimeError(
                f"{split}: {docs.size:,} sidecar records, meta says {meta[f'{split}_docs']:,}"
            )
        starts = np.asarray(docs["start"])
        if docs.size and (
            starts[0] != 0 or np.any(np.diff(starts) <= 0) or starts[-1] >= expected
        ):
            raise RuntimeError(f"{split}: sidecar starts are not increasing from zero")
        if docs.size and (
            docs["dump"].min() < 0 or docs["dump"].max() >= len(meta["dumps"])
        ):
            raise RuntimeError(f"{split}: sidecar dump ids outside meta['dumps']")
        later = starts[1:]
        if later.size:
            picked = rng.choice(later, min(sample_docs, later.size), replace=False)
            for start in np.sort(picked):
                if int(data.read(int(start) - 1, 1)[0]) != meta["eos_id"]:
                    raise RuntimeError(
                        f"{split}: no EOS before the document at {start}"
                    )
        summary[split] = {"tokens": data.total_tokens, "docs": int(docs.size)}
    return summary


def write_synthetic(
    directory: str | Path,
    *,
    train_tokens: int,
    val_tokens: int,
    vocab: int = 97,
    seed: int = 0,
    shards: int = 2,
) -> None:
    """A tiny fake corpus for offline tests and smoke runs.

    Random tokens cut into documents of 20 to 60 tokens, each closed by EOS
    id 0, with sidecars, so readers and ``verify`` see a real store's shape.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    eos = 0

    def stream(total: int) -> tuple[np.ndarray, np.ndarray]:
        tokens = rng.integers(1, vocab, total, dtype=np.uint32)
        starts = [0]
        while True:
            nxt = starts[-1] + int(rng.integers(20, 61))
            if nxt >= total:
                break
            tokens[nxt - 1] = eos
            starts.append(nxt)
        docs = np.empty(len(starts), dtype=DOC_DTYPE)
        docs["start"] = starts
        docs["source"] = np.arange(len(starts))
        docs["dump"] = 0
        return tokens, docs

    val, val_docs = stream(val_tokens)
    val.tofile(directory / "val.bin")
    np.save(directory / "val.docs.npy", val_docs)
    train, train_docs = stream(train_tokens)
    per_shard = train_tokens // shards
    for index in range(shards):
        piece = (
            train[index * per_shard :]
            if index == shards - 1
            else train[index * per_shard : (index + 1) * per_shard]
        )
        piece.tofile(directory / f"train.{index:04d}.bin")
    np.save(directory / "train.docs.npy", train_docs)
    (directory / META).write_text(
        json.dumps(
            {
                "tokenizer": "synthetic",
                "tokenizer_revision": None,
                "eos_id": eos,
                "dataset": "synthetic",
                "config": None,
                "revision": None,
                "packages": {},
                "shuffle_seed": seed,
                "val_tokens": val_tokens,
                "train_tokens": train_tokens,
                "val_docs": int(val_docs.size),
                "train_docs": int(train_docs.size),
                "dumps": ["synthetic"],
                "vocab_size": vocab,
            },
            indent=2,
        )
    )


def read_meta(directory: str | Path) -> dict:
    return json.loads((Path(directory) / META).read_text())

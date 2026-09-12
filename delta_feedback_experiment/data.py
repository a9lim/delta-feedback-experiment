"""One reproducible token stream over a fixed document universe, read as a prefix
by every run.

Layout under a store directory (default ``data/dclm-100b``):

    meta.json            source, tokenizer, packages, ordering, seed, counts
    source.json          the document universe: every source file's row groups
    val.bin              the held-out slice: the shuffled stream's first documents
    val.docs.npy         one record per held-out document (start, source)
    train.0000.bin ...   uint32 shards, one contiguous stream after the slice
    train.docs.npy       one record per training document

The universe is every document of the pinned parquet source, addressed by
its row in file order. Optional keyed shuffling (:class:`Shuffle`) sends
addresses to stream positions; otherwise published file/row order is kept.
Within one source, ordering mode, tokenizer, build package versions and seed,
every store is a prefix of the same stream: a build that
stops at 85B tokens is byte-identical to the first 85B tokens of one that
stops at 188B. Rows are non-overlapping ``seq_len+1``-token windows addressed
by a global row index, so batch ``step`` is the same bytes for every
condition (the paired data order contract) and a resumed run addresses the
identical rows.

The sidecar records where every document starts in its split's stream, its
address in the universe (which names the parquet row that holds its text,
URL, id, scores, and any source-specific metadata).
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
from multiprocessing import get_context
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from transformer_experiments import telemetry

from .tokenizer import (
    SYNTHETIC_TOKENIZER_ID,
    TOKENIZER_ID,
    VOCAB_SIZE,
    load_tokenizer,
    tokenizer_metadata,
)

META = "meta.json"
SOURCE = "source.json"
BUILD = "build.json"
STORE_FORMAT = "document-stream-v1"
SHARD_TOKENS = 1 << 28
"""Tokens per train shard (~1 GiB as uint32)."""
WRITE_BUFFER_TOKENS = 1 << 24
"""Gather up to about 64 MiB before each sequential output write."""

DOC_DTYPE = np.dtype([("start", "<i8"), ("source", "<i8")])
"""One sidecar record: split-local start offset and universe address."""

CANONICAL_TARGET_TOKENS = 85_000_000_000
"""Stored tokens, held-out slice included, for the screen's 400x schedule;
the bridge's 400x schedule needs 188B (``docs/scaling.md``)."""
CANONICAL_VAL_TOKENS = 30_000_000
CANONICAL_SHUFFLE_SEED = 0
CANONICAL_TOKENS_PER_DOC = 900
"""A conservative mean document length used to size the selection.
This does not truncate documents; a selection that runs short fails."""


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    revision: str
    prefix: str
    shuffle: bool


DEFAULT_SOURCE = "dclm-100b"
SOURCES = {
    "dclm": SourceSpec(
        "mlfoundations/dclm-baseline-1.0-parquet",
        "817d6752765f6a41261085171dd546b104f60626",
        "filtered",
        True,
    ),
    "dclm-100b": SourceSpec(
        "HuggingFaceFW/dclm_100BT-shuffled",
        "2fa015e4044ec442a0734e89658cdcc538d10dd4",
        "data",
        False,
    ),
    **{
        name: SourceSpec(
            "HuggingFaceFW/fineweb-edu",
            "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
            prefix,
            True,
        )
        for name, prefix in (
            ("fineweb-edu", "data"),
            ("fineweb-edu-350b", "sample/350BT"),
            ("fineweb-edu-100b", "sample/100BT"),
            ("fineweb-edu-10b", "sample/10BT"),
        )
    },
}
DATA_PACKAGES = ("transformers", "tokenizers", "huggingface-hub", "pyarrow")


def data_package_versions() -> dict[str, str]:
    """Record installed build versions; dependency lower bounds live in pyproject.toml."""
    installed = {}
    for package in DATA_PACKAGES:
        try:
            installed[package] = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"data build requires {package}; "
                "install the data-build extra"
            ) from exc
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

    def __init__(self, size: int, seed: int, enabled: bool = True):
        if size < 1:
            raise ValueError("a shuffle needs a non-empty domain")
        self.size = size
        self.seed = seed
        self.enabled = enabled
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
        if not self.enabled:
            return x.astype(np.int64)
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


class Source(Protocol):
    """Where parquet files come from; picklable so workers can share it."""

    def list_files(self) -> list[str]: ...

    def row_groups(self, path: str) -> list[int]: ...

    def fetch(self, path: str) -> Path: ...

    def release(self, path: str) -> None: ...


@dataclass(frozen=True)
class HubSource:
    """The pinned Hugging Face dataset, staged one file at a time."""

    dataset: str
    prefix: str
    revision: str
    scratch: Path

    def list_files(self) -> list[str]:
        from huggingface_hub import HfApi

        entries = HfApi().list_repo_tree(
            self.dataset,
            repo_type="dataset",
            path_in_repo=self.prefix,
            revision=self.revision,
            recursive=True,
        )
        return sorted(e.path for e in entries if e.path.endswith(".parquet"))

    def row_groups(self, path: str) -> list[int]:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        remote = f"datasets/{self.dataset}@{self.revision}/{path}"
        for attempt in range(5):
            try:
                with HfFileSystem().open(remote, "rb", block_size=65536) as handle:
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

    def row_groups(self, path: str) -> list[int]:
        import pyarrow.parquet as pq

        metadata = pq.ParquetFile(self.root / path).metadata
        return [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]

    def fetch(self, path: str) -> Path:
        return self.root / path

    def release(self, path: str) -> None:
        return None


def _write_json(path: Path, value: dict) -> None:
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(value, indent=2))
    staging.replace(path)


def _write_array(path: Path, value: np.ndarray) -> None:
    staging = path.with_suffix(path.suffix + ".tmp")
    with staging.open("wb") as handle:
        np.save(handle, value)
    staging.replace(path)


def index_source(source: Source, cache: Path | None = None) -> dict:
    """Index pinned row counts in sorted file order, with resumable footers."""
    if cache is not None:
        cache.mkdir(exist_ok=True)

    def entry(item: tuple[int, str]) -> dict:
        which, path = item
        cached = cache / f"{which:05d}.json" if cache is not None else None
        if cached is not None and cached.exists():
            record = json.loads(cached.read_text())
            if record["path"] != path:
                raise ValueError("source file listing changed during indexing")
        else:
            record = {"path": path, "row_groups": source.row_groups(path)}
            if cached is not None:
                _write_json(cached, record)
        return record

    files = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for record in pool.map(entry, enumerate(source.list_files())):
            files.append(record)
            telemetry.log("index", file=record["path"], rows=sum(record["row_groups"]))
    return {"files": files}


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
    shuffle: bool = True


PART_DTYPE = np.dtype([("position", "<i8"), ("offset", "<i8"), ("length", "<i4")])

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
    shuffle = Shuffle(build.universe, build.seed, build.shuffle)
    path = build.paths[which]
    if local is None:
        local = build.source.fetch(path)
    tokens_path = build.parts / f"{which:04d}.tokens.bin"
    index_path = build.parts / f"{which:04d}.index.npy"
    records: list[tuple[int, int, int]] = []
    offset = docs = 0
    parquet = pq.ParquetFile(local)
    expected = build.row_groups[which]
    if parquet.metadata.num_row_groups != len(expected):
        raise RuntimeError(f"{path}: row groups differ from the source index")
    with tokens_path.open("wb") as out:
        base = build.bases[which]
        for group, rows in enumerate(expected):
            positions = shuffle(base + np.arange(rows, dtype=np.int64))
            keep = np.nonzero(
                (positions >= build.first) & (positions < build.selected)
            )[0]
            base += rows
            if keep.size == 0:
                continue
            table = parquet.read_row_group(group, columns=["text"])
            if table.num_rows != rows:
                raise RuntimeError(f"{path}: row group {group} differs from the index")
            texts = table.column("text").take(keep).to_pylist()
            encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
            for text, ids, position in zip(texts, encoded, positions[keep]):
                if not text:
                    continue
                ids.append(eos)
                array = np.asarray(ids, dtype=np.uint32)
                array.tofile(out)
                records.append((int(position), offset, array.size))
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

    The fetch thread lives inside the worker, after the process pool starts.
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
    if not build.shuffle:
        # Published order is already mixed for sources such as dclm-100b.
        # Files outside this prefix need no download or tokenization.
        for which, base in enumerate(build.bases):
            end = base + sum(build.row_groups[which])
            if end <= build.first or base >= build.selected:
                (build.parts / f"{which:04d}.tokens.bin").write_bytes(b"")
                np.save(
                    build.parts / f"{which:04d}.index.npy",
                    np.empty(0, dtype=PART_DTYPE),
                )
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
    # Tokenizer/Arrow/Hub initialization can leave native threads in the parent.
    # Spawn avoids inheriting their locks into the encoding workers.
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn")
    ) as pool:
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
        indices.append(merged)
        tokens_path = parts / f"{which:04d}.tokens.bin"
        tokens = np.empty(0, dtype=np.uint32)
        if tokens_path.stat().st_size:
            # Full DCLM has tens of thousands of parts. The mapping owns its
            # pages without retaining a file descriptor per source shard.
            with tokens_path.open("rb") as handle:
                mapping = mmap.mmap(
                    handle.fileno(), 0, access=mmap.ACCESS_READ, trackfd=False
                )
            tokens = np.frombuffer(mapping, dtype=np.uint32)
            if shuffle.enabled and hasattr(mmap, "MADV_RANDOM"):
                mapping.madvise(mmap.MADV_RANDOM)
        # Stream order jumps between small documents across every source part.
        # Sequential mmap readahead pulls in mostly unused neighbouring pages
        # when the parts exceed RAM. The hint changes paging, never token bytes.
        maps.append(tokens)
    index = np.concatenate(indices) if indices else np.empty(0, dtype=_MERGED_DTYPE)
    if shuffle.enabled:
        index = index[np.argsort(index["position"], kind="stable")]
    del indices
    positions, parts_col = index["position"], index["part"]
    offsets, lengths = index["offset"], index["length"]

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
        return records

    counts = {}
    if not extend:
        _write_array(out / "val.docs.npy", sidecar(0, val_docs))
        counts["val_docs"] = val_docs
        counts["val_tokens"] = writers["val"].written
    train_docs = sidecar(val_docs, count)
    if extend:
        train_docs = np.concatenate([np.load(out / "train.docs.npy"), train_docs])
    _write_array(out / "train.docs.npy", train_docs)
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
    ]
)


# -- the build -----------------------------------------------------------------


def _clean_committed_build(out: Path, meta: dict) -> None:
    """Finish cleanup if the metadata commit succeeded before an interruption."""
    path = out / BUILD
    if not path.exists():
        return
    recorded = json.loads(path.read_text())
    if not all(
        meta.get(key) == value
        for key, value in recorded.items()
        if key != "extend_from"
    ):
        return  # An extension is still in progress, not committed.
    if meta["train_tokens"] < meta["target_tokens"] - meta["val_target"]:
        return
    parts = out / "parts"
    if parts.exists():
        shutil.rmtree(parts)
    path.unlink()


def _restore_train_prefix(out: Path, meta: dict) -> None:
    """Drop a failed extension's uncommitted tail before appending again.

    meta.json is the commit marker. Atomic sidecars can be ahead of it when
    assembly is interrupted, but their original prefix remains intact.
    """
    remaining = meta["train_tokens"]
    paths = sorted(out.glob("train.*.bin"))
    if sum(path.stat().st_size for path in paths) < remaining * 4:
        raise RuntimeError("the store is shorter than its committed training prefix")
    for path in paths:
        size = path.stat().st_size // 4
        if remaining == 0:
            path.unlink()
        elif size > remaining:
            with path.open("r+b") as handle:
                handle.truncate(remaining * 4)
        remaining = max(0, remaining - size)
    docs_path = out / "train.docs.npy"
    docs = np.load(docs_path)
    if docs.size < meta["train_docs"]:
        raise RuntimeError("the store is missing committed document records")
    if docs.size != meta["train_docs"]:
        _write_array(docs_path, docs[: meta["train_docs"]])


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
    source_name: str = DEFAULT_SOURCE,
    shuffle: bool | None = None,
    revision: str | None = None,
    scratch: str | Path | None = None,
    check_packages: bool = True,
    extend: bool = False,
) -> dict:
    """Build the chosen stream's first ``target_tokens`` stored tokens.

    Three resumable stages under ``out_dir``: index the source into
    ``source.json``; select the documents whose stream position falls below
    ``target_tokens / tokens_per_doc`` and tokenize them into per-file parts;
    then write the parts in stream order, the held-out slice first. The
    stream is a pure function of the source, its revision, the tokenizer,
    build package versions, ordering mode and seed: matching builds agree on
    their common prefix. Refuses to run if ``meta.json`` already exists unless
    ``extend`` continues that store to
    a larger target under the same settings, selecting from the position
    after its last document and appending; a partial build resumes.
    """
    if readers < 1:
        raise ValueError("readers must be at least 1")
    if workers < 1 or tokens_per_doc < 1:
        raise ValueError("workers and tokens_per_doc must be at least 1")
    if not 0 < val_tokens < target_tokens:
        raise ValueError("target_tokens must exceed a positive val_tokens")
    if source_name not in SOURCES:
        raise ValueError(
            f"unknown source {source_name!r}; choose from {', '.join(SOURCES)}"
        )
    spec = SOURCES[source_name]
    revision = spec.revision if revision is None else revision
    shuffle = spec.shuffle if shuffle is None else shuffle
    settings = {
        "format": STORE_FORMAT,
        "source": source_name,
        "dataset": spec.dataset,
        "source_prefix": spec.prefix,
        "revision": revision,
        **(tokenizer_metadata() if tokenizer is None else {
            "tokenizer": "synthetic", "tokenizer_revision": None, "vocab_size": VOCAB_SIZE,
        }),
        "tokenizer_id": TOKENIZER_ID if tokenizer is None else SYNTHETIC_TOKENIZER_ID,
        "shuffle": shuffle,
        "shuffle_seed": seed,
        "val_target": val_tokens,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    previous = None
    if (out / META).exists():
        if not extend:
            raise FileExistsError(f"{out / META} exists; delete the directory to redo")
        previous = read_meta(out)
        for key, value in settings.items():
            if previous.get(key) != value:
                raise ValueError(
                    f"--continue: {key} is {previous.get(key)!r} in the store, "
                    f"{value!r} requested"
                )
        _clean_committed_build(out, previous)
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
        source = HubSource(spec.dataset, spec.prefix, revision, scratch)
    if tokenizer is None:
        tokenizer = _HubTokenizer()
    eos = int(tokenizer().eos_token_id)
    telemetry.log(
        "tokenize",
        source=source_name,
        dataset=spec.dataset,
        shuffle=shuffle,
        revision=revision,
        tokenizer=settings["tokenizer"],
        tokenizer_id=settings["tokenizer_id"],
        seed=seed,
        target=target_tokens,
        val=val_tokens,
        workers=workers,
        readers=readers,
    )

    # Completed parts may only resume under exactly the compilation settings
    # that produced them. Worker/reader counts can change without changing bytes.
    build_settings = {
        **settings,
        "packages": packages,
        "eos_id": eos,
        "target_tokens": target_tokens,
        "tokens_per_doc": tokens_per_doc,
        "extend_from": previous["train_tokens"] if previous is not None else None,
    }
    build_path = out / BUILD
    if build_path.exists():
        recorded = json.loads(build_path.read_text())
        for key, value in build_settings.items():
            if recorded.get(key) != value:
                raise ValueError(
                    f"partial build: {key} differs; use a new output directory"
                )
    else:
        if (out / SOURCE).exists() and previous is None:
            raise ValueError(
                "partial build has no build.json; use a new output directory"
            )
        _write_json(build_path, build_settings)

    source_path = out / SOURCE
    index_cache = out / "source-index"
    if source_path.exists():
        index = json.loads(source_path.read_text())
    else:
        index = index_source(source, index_cache)
        _write_json(source_path, index)
    shutil.rmtree(index_cache, ignore_errors=True)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if previous is not None and previous["source_sha256"] != source_sha256:
        raise ValueError("--continue: source.json differs from the store's")
    universe = universe_size(index)
    if universe == 0:
        raise RuntimeError("the source holds no documents")
    order = Shuffle(universe, seed, shuffle)
    selected = min(universe, math.ceil(target_tokens / tokens_per_doc))
    first = 0
    if previous is not None:
        last = int(
            np.load(out / "train.docs.npy")["source"][previous["train_docs"] - 1]
        )
        first = int(order([last])[0]) + 1
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
        shuffle=shuffle,
    )
    _select(build, workers)

    if previous is not None:
        _restore_train_prefix(out, previous)
    counts = _write(
        out,
        parts,
        len(index["files"]),
        target_tokens=target_tokens,
        val_tokens=val_tokens,
        shuffle=order,
        extend=previous is not None,
        readers=readers,
    )
    if previous is None:
        meta = {
            **settings,
            "eos_id": eos,
            "packages": packages,
            "source_sha256": source_sha256,
            "universe_docs": universe,
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
    _write_json(out / META, meta)
    shutil.rmtree(parts)
    build_path.unlink()
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
    def __call__(self):
        return load_tokenizer()


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
        read_meta(directory)
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


def verify(directory: str | Path, *, sample_docs: int = 2_048) -> dict:
    """Check a store against its meta and sidecars; raise on any mismatch.

    Token counts must match the files, sidecar starts must be increasing and
    begin at zero, and the token before each sampled document's start must be
    the EOS that closed its predecessor. Boundary checks sample up to 2,048
    documents per split by default.
    """
    directory = Path(directory)
    meta = read_meta(directory)
    if "source_sha256" in meta:
        source_path = directory / SOURCE
        if (
            hashlib.sha256(source_path.read_bytes()).hexdigest()
            != meta["source_sha256"]
        ):
            raise RuntimeError("source.json differs from the store's pinned index")
        if universe_size(json.loads(source_path.read_text())) != meta["universe_docs"]:
            raise RuntimeError("source index size differs from the store's universe")
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
        if docs.dtype != DOC_DTYPE:
            raise RuntimeError(f"{split}: unsupported document sidecar format")
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
            docs["source"].min() < 0 or docs["source"].max() >= meta["universe_docs"]
        ):
            raise RuntimeError(
                f"{split}: sidecar source addresses outside the universe"
            )
        later = starts[1:]
        if later.size:
            picked = np.sort(
                rng.choice(later, min(sample_docs, later.size), replace=False)
            ) - 1
            for shard, (first, last) in zip(
                data.maps, pairwise(data.offsets), strict=True
            ):
                lower, upper = np.searchsorted(picked, (first, last))
                positions = picked[lower:upper]
                invalid = np.flatnonzero(shard[positions - first] != meta["eos_id"])
                if invalid.size:
                    start = int(positions[invalid[0]]) + 1
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
                "format": STORE_FORMAT,
                "tokenizer": "synthetic",
                "tokenizer_id": SYNTHETIC_TOKENIZER_ID,
                "tokenizer_revision": None,
                "eos_id": eos,
                "dataset": "synthetic",
                "revision": None,
                "packages": {},
                "shuffle_seed": seed,
                "val_tokens": val_tokens,
                "train_tokens": train_tokens,
                "val_docs": int(val_docs.size),
                "train_docs": int(train_docs.size),
                "universe_docs": max(int(val_docs.size), int(train_docs.size)),
                "vocab_size": vocab,
            },
            indent=2,
        )
    )


def read_meta(directory: str | Path) -> dict:
    meta = json.loads((Path(directory) / META).read_text())
    if meta.get("format") != STORE_FORMAT:
        raise ValueError("unsupported token store format; rebuild with delta tokenize")
    if meta.get("tokenizer_id") == TOKENIZER_ID:
        if any(meta.get(key) != value for key, value in tokenizer_metadata().items()):
            raise ValueError("token store tokenizer metadata differs; rebuild with delta tokenize")
    elif meta.get("tokenizer_id") != SYNTHETIC_TOKENIZER_ID:
        raise ValueError("token store uses a different tokenizer; rebuild with delta tokenize")
    return meta

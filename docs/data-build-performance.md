# Token-store build performance

The build has three costs: source indexing, download/tokenization, and
assembly in shuffled document order. Profile them separately. A faster
tokenizer does not fix download waits or random-read amplification.

## Current execution

The default `dclm-100b` source is already shuffled by its publisher. Its
published-order path indexes 100 file footers, downloads only files that
intersect the selected document prefix, and assembles in file/row order
without a global sort or random-read advice. `--data-root ROOT` writes to
`ROOT/dclm-100b`; other source names use the same path convention.

`dclm` defaults to a keyed document shuffle across the full source, requiring
a scan of all 27,938 parquet files (7.42 TB at the pinned revision). File
footers are indexed with eight threads and cached individually for restart;
small remote reads avoid fetching full files for metadata. `--shuffle` and
`--no-shuffle` override either default. Both paths record the source and
ordering in the store and require matching settings for resume/extension.

Each `--workers` process encodes one source file and prefetches at most one
upcoming file in a thread. Encoding processes use `spawn` so that parent
tokenizer/Arrow/Hub threads and locks are not inherited. Completed part indices remain the resume markers;
an interrupted download or encode leaves completed parts reusable. At most
two source files per worker occupy download scratch. Multiple workers divide
half the host's logical CPU threads between their Rust tokenizers by default;
set `RAYON_NUM_THREADS` explicitly to tune that budget.

Shuffled assembly sorts document indices by stream position, then gathers roughly
64 MiB at a time into the final uint32 stream. Source part mappings receive
`MADV_RANDOM` where available, suppressing inappropriate sequential read-ahead
([Linux semantics](https://man7.org/linux/man-pages/man2/madvise.2.html)).
`--readers` (default 8) threads copy disjoint document ranges into disjoint
output ranges. Shard writes stay sequential. Mappings do not retain one file descriptor per
part, so the full DCLM source does not exhaust the descriptor limit. Document
boundaries and file/row provenance are preserved in either ordering mode.
`assemble_progress` reports train tokens, the target, elapsed seconds, and
average assembled tokens per second every 30 seconds.

## Measuring the current build

End-to-end build speed with the current GPT-NeoX/ChatML tokenizer is not yet
measured. Record indexing, encoding, and assembly separately from the current
build logs, including source pins, package versions, worker/thread counts,
selected documents, token counts, and elapsed time.

Compare read-only gathers while a build still has its completed parts:

```bash
python scripts/token_assembly_bench.py OUT/parts --docs-per-part 32 --readers 8
```

The script leaves shared caches alone, reports physical read bytes on Linux,
and checks identical token hashes. Use distinct seeded document selections
and alternate comparison order. Parts are deleted after a successful build,
so this is a staging benchmark. Hash equality is a correctness check and does
not establish a speedup. Complete-build timing must include downloads and
encoding as well as gathers.

## Prime staging

Build on a CPU staging instance with local NVMe and sufficient network
throughput, before attaching the finished store to the GPU node. Start with
eight assembly readers; compare four/eight/sixteen on that actual disk. Increase encoding workers only while network
throughput or encoding throughput improves, with an explicit per-process
Rust thread budget. Preserve tokenizer/source revisions, build package versions
and ordering. Compare checksums only across builds of the same source;
the full DCLM store and Jobe's 100B subset do not
share a promised prefix.

Peak storage includes both selected token parts and the finished stream on
the output filesystem. `--scratch` only relocates source downloads, not
`OUT/parts`. Allow roughly twice the final token-store size for token parts
and final output, plus sidecars, selection headroom, and bounded download
scratch; measure the actual peak before treating it as a capacity bound.

Use `--source dclm` for the bridge's 167B and the flagship's 441B stores.
The published 100B subset is a separate source for smaller builds; its name
is a publisher token estimate, not an exact count with the current tokenizer.
A 15B-token store occupies about 60 GB of uint32 tokens plus sidecars.

If larger stores remain I/O-bound after these changes, the next architectural
step is an external shuffle by stream-position ranges: write documents into
bounded buckets, then assemble each bucket in order. That can improve read
locality and bound index memory, at the cost of another temporary-data pass.
It is not implemented or benchmarked. Provider qualification should determine
whether that extra complexity is worthwhile.

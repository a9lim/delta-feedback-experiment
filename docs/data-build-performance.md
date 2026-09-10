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

## Jobe evidence, September 10, 2026

The former FineWeb-Edu 57B-token build completed all 472 source parts at
04:36:47 EDT. It was stopped and its data deleted when the project switched
to DCLM. These measurements describe that FineWeb-Edu working set. Its
original assembly process produced about 11.5 MB/s while reading roughly
280 MB/s from the NVMe drive. A stack sample found it in the buffer gather;
process counters showed about 2,300 major page faults per second and only
6% of one CPU used. Small shuffled documents were pulling in much larger
neighbouring ranges from a working set larger than RAM.

Bounded read-only gathers from the real 472-part store, under the concurrent
assembly load, measured:

| Gather mode | Output MB/s | Disk bytes / output byte |
|---|---:|---:|
| Default mmap advice, one reader | 10.4–11.0 | 24.5–25.8 |
| Random advice, one reader | 18.5–19.5 | about 1.6 |
| Random advice, four readers | 46.5–47.5 | about 1.6 |
| Random advice, eight readers | 53.4–55.9 | about 1.6 |

Timing samples used distinct seeded document selections and reversed mode
order. Separate repeated-input SHA-256 checks agreed exactly; the warm-cache
repeat is a correctness check, not a speed comparison. These are sampled
gather rates, not complete-build or Prime benchmarks. Download prefetch
overlap is verified by a synchronized test; its end-to-end speedup has not
been measured. This does not establish throughput for either DCLM source.

Reproduce the gather comparison while a build still has its completed parts:

```bash
python scripts/token_assembly_bench.py OUT/parts --docs-per-part 32 --readers 8
```

The script reads only, leaves shared caches alone, reports physical read
bytes on Linux, and checks identical token hashes. Parts are deleted after a
successful build, so this is a staging benchmark. Tests also compare exact
store bytes across worker/reader counts, different flush sizes, shard
boundaries, empty source parts, interrupted builds, and extension.

## Prime staging

Build on a CPU staging instance with local NVMe and sufficient network
throughput, before attaching the finished store to the GPU node. Start with
eight assembly readers; compare four/eight/sixteen on that actual disk rather
than extrapolating Jobe's result. Increase encoding workers only while network
throughput or encoding throughput improves, with an explicit per-process
Rust thread budget. Preserve tokenizer/source pins and ordering. Compare checksums only across
builds of the same source; the full DCLM store and Jobe's 100B subset do not
share a promised prefix.

Peak storage includes both selected token parts and the finished stream on
the output filesystem. The measured FineWeb-Edu 57B build had 260.9 GB of parts and about 229 GB
of final tokens/sidecars: roughly 490 GB before cleanup. `--scratch` only
relocates source downloads. Budget roughly 2.2 times final size plus headroom
for a comparable build; a separate scratch disk does not relocate `OUT/parts`.

Use `--source dclm` for the bridge's 167B and the flagship's 441B stores.
The published 100B subset is a separate source for smaller builds; its name
is a publisher token estimate, not an exact count with the Qwen tokenizer.
A 15B-token store occupies about 60 GB of uint32 tokens plus sidecars, and
construction also needs selected parts and bounded download scratch.
The selection estimate determines extra staging, so the FineWeb-Edu peak
ratio above is a planning guide, not a measured DCLM storage bound.

If larger stores remain I/O-bound after these changes, the next architectural
step is an external shuffle by stream-position ranges: write documents into
bounded buckets, then assemble each bucket in order. That can improve read
locality and bound index memory, at the cost of another temporary-data pass.
It is not implemented or benchmarked. Provider qualification should determine
whether that extra complexity is worthwhile.

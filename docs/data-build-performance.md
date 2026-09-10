# Token-store build performance

The build has three costs: source indexing, download/tokenization, and
assembly in shuffled document order. Profile them separately. A faster
tokenizer does not fix download waits or random-read amplification.

## Current execution

Each `--workers` process encodes one source file and prefetches at most one
upcoming file in a thread. Completed part indices remain the resume markers;
an interrupted download or encode leaves completed parts reusable. At most
two source files per worker occupy download scratch. Multiple workers divide
half the host's logical CPU threads between their Rust tokenizers by default;
set `RAYON_NUM_THREADS` explicitly to tune that budget.

Assembly sorts document indices by stream position, then gathers roughly
64 MiB at a time into the final uint32 stream. Source part mappings receive
`MADV_RANDOM` where available, suppressing inappropriate sequential read-ahead
([Linux semantics](https://man7.org/linux/man-pages/man2/madvise.2.html)).
`--readers` (default 8) threads copy disjoint document ranges into disjoint
output ranges. Shard writes stay sequential. The source/tokenizer revisions,
shuffle, document boundaries, sidecars, and store format are unchanged.
`assemble_progress` reports train tokens, the target, elapsed seconds, and
average assembled tokens per second every 30 seconds.

## Jobe evidence, September 10, 2026

The 57B-token build completed all 472 source parts at 04:36:47 EDT. Its
original assembly process produced about 11.5 MB/s while reading roughly
280 MB/s from the NVMe drive. A stack sample found it in the buffer gather;
process counters showed about 2,300 major page faults per second and only
6% of one CPU used. Small shuffled documents were pulling in much larger
neighbouring ranges from a working set larger than RAM.

Bounded read-only gathers from the real 472-part store, under the concurrent
assembly load, measured:

| Gather mode | Output MB/s | Disk bytes / output byte |
|---|---:|---:|
| Default mmap advice, one reader | 10.6–11.0 | 24.5–25.4 |
| Random advice, one reader | 18.5–19.5 | about 1.6 |
| Random advice, four readers | 46.5–47.5 | about 1.6 |
| Random advice, eight readers | 54.3–55.6 | about 1.6 |

Timing samples used distinct seeded document selections and reversed mode
order. Separate repeated-input SHA-256 checks agreed exactly; the warm-cache
repeat is a correctness check, not a speed comparison. These are sampled
gather rates, not complete-build or Prime benchmarks. Download prefetch
overlap is verified by a synchronized test; its end-to-end speedup has not
been measured. The running build keeps its already-loaded implementation.

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
Rust thread budget. Preserve tokenizer/source pins and compare the finished
store's common prefix to Jobe by checksum.

Peak storage includes both selected token parts and the finished stream on
the output filesystem. The 57B build has 260.9 GB of parts and about 229 GB
of final tokens/sidecars: roughly 490 GB before cleanup. `--scratch` only
relocates source downloads. Budget roughly 2.2 times final size plus headroom
for a comparable build; a separate scratch disk does not relocate `OUT/parts`.

The bridge's 167B target fits the current source. A 441B flagship store needs
a larger document universe: the 63.3M selected documents produced 64.9B Qwen
tokens, estimating only about 348B over the full 339.3M-document sample.
That is a sample-based estimate, not a full token count. Changing the source
universe changes the keyed permutation, so expanding the source also requires
an explicit decision about the shared-prefix comparison.

If larger stores remain I/O-bound after these changes, the next architectural
step is an external shuffle by stream-position ranges: write documents into
bounded buckets, then assemble each bucket in order. That can improve read
locality and bound index memory, at the cost of another temporary-data pass.
It is not implemented or benchmarked. Provider qualification should determine
whether that extra complexity is worthwhile.

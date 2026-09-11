# Growing recipe: conditions, data, schedule, evaluation

This page describes how a specimen is grown: the condition letters and how
conditions pair, the token stream, the optimizer batch, the feedback passes, the learning-rate schedule, and the numbers we look at afterwards.
[Architecture](architecture.md) owns the equations;
[scaling](scaling.md) owns the two geometries, their accounting, and their
budgets; [interpretability](interpretability.md) owns the analysis scripts.

## Conditions

A condition is a string of letters from `arfl`, each one change from the
plain twelve-layer gated GQA decoder. `parse_condition` accepts the
letters in any order and returns them in that order; the empty string is the
plain decoder, and `ModelConfig.condition` renders a configuration's letters
back.

| Letter | Change | `ModelConfig` flag |
|---|---|---|
| `a` | Kimi Delta Attention: replace the three RoPE-GGQA layers per cell with PKDA, giving `[PKDA, PKDA, PKDA, NoPE-GGQA] x 3` | `hybrid` |
| `r` | MHDB: transient grouped reads of the seed and block deltas before every sublayer; with `f`, routed enrichment of the payload | `block_routing` |
| `f` | FBT: token-gated latent payload transfer between token columns | `feedback` |
| `l` | Huginn loop: the cells between the first and last become one tied core, iterated a drawn number of times per column ([architecture.md](architecture.md#letter-l-the-tied-depth-loop)) | `loop` |

Every subset of `arfl` builds; their parameter counts at the screen are in
[scaling.md](scaling.md#the-screen).

Without `a`, each cell is `[RoPE-GGQA, RoPE-GGQA, RoPE-GGQA, NoPE-GGQA]`.
This partial use of RoPE selects layers, with full-head Q/K rotation after
per-head RMSNorm at theta 10,000. The fourth layer stays NoPE in every
condition. The pattern holds at every scale and core iteration and adds no
learned parameters or recipe knob.

`arfl` is the full built stack and `arf` the flat column. `a` alone already
has recurrent mixer memory. `r` without `f` seeds from the plain embedding and
emits no payload; `f` without `r` emits `payload_norm(h_top)`; `r` with `f`
uses the fused seed as a routing source and enriches the payload with routed
sources. `l` adds no parameters: it holds the first and last cells and runs
the cells between them as one tied core, so every looped condition has its
unlooped condition's counts. The specimens so far are `ar` and `arf` runs.

Pairing is built in. Two conditions on the same trunk letter initialize every
parameter they share byte-identically for a given seed
([architecture.md](architecture.md#precision-and-initialization)); parameters
do not pair across `a`. Two conditions trained with the same seed and data seed therefore
see the same rows in the same order with the same keyed feedback draws, and
can be compared token by token. A looped condition pairs with its unlooped
one the same way: the iteration draw is its own keyed sub-stream, so `arfl`
and `arf` share every pass, prefix, and jitter draw.

Initialization uses the single `BASE_NORMAL_INIT_STD = 0.02` constant in
`delta_feedback_experiment/model.py` for the tied embedding and NAdam dense
matrices. Gate/control matrices with fan-in `D` multiply that standard
deviation by `sqrt(1536 / D)`; PKDA's fixed-head-width expansions and the
embedding use it directly. NorMuonH matrices use `1 / sqrt(fan_in)`
independently of the base constant.

`l` needs whole cells and at least three of them. At one core iteration
`arfl` is exactly `arf`: the same layers, parameters, banks, routes, losses,
and gradients.

## Geometry

The screen, the bridge, and the flagship, with their parameter and cache
accounting and their budgets, are in [scaling.md](scaling.md). The screen is width 768,
twelve layers in three cells, and context 4,096; under `a` the trunk is
`[PKDA, PKDA, PKDA, NoPE-GGQA] x 3`, and under `l` the first cell is
the prelude, the last the coda, and the one between them the tied core, run
`r` times per column.

## Data

The source is chosen with `delta tokenize --source NAME`. All sources use
`Qwen/Qwen3-0.6B-Base` at commit `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.
The base tokenizer's EOS, `<|endoftext|>` (151643), closes every non-empty
document. The `data-build` extra sets minimum versions for the four packages
that compile the stream; `meta.json` records their realized versions. New
stores can use newer packages. Resuming or extending a store requires its
recorded build package versions so one store never mixes compilation stacks.

| Source | Dataset and parquet directory | Default ordering |
|---|---|---|
| `dclm-100b` (default; screen and Jobe) | `HuggingFaceFW/dclm_100BT-shuffled`, `data/` | Published order |
| `dclm` (larger stores and flagship) | `mlfoundations/dclm-baseline-1.0-parquet`, `filtered/` | Keyed document shuffle |
| `fineweb-edu` | `HuggingFaceFW/fineweb-edu`, `data/` | Keyed document shuffle |
| `fineweb-edu-350b` | Same dataset, `sample/350BT/` | Keyed document shuffle |
| `fineweb-edu-100b` | Same dataset, `sample/100BT/` | Keyed document shuffle |
| `fineweb-edu-10b` | Same dataset, `sample/10BT/` | Keyed document shuffle |

The pinned revisions are `2fa015e4044ec442a0734e89658cdcc538d10dd4` for
`dclm-100b`, `817d6752765f6a41261085171dd546b104f60626` for `dclm`, and
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` for the FineWeb-Edu sources.
The [100B subset](https://huggingface.co/datasets/HuggingFaceFW/dclm_100BT-shuffled)
has already been globally shuffled by its publisher with seed 42.
`--shuffle` applies our keyed document shuffle to any source;
`--no-shuffle` preserves any source's published file/row order. The full
[DCLM release](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet)
retains source clustering and defaults to shuffling.

The universe is every document under the chosen source prefix, addressed by
its row in sorted file order (`source.json`). With shuffling enabled, a keyed
Feistel bijection (`Shuffle`, seed 0) sends addresses to stream positions.
Without it, the address is the position. Every store built with the same
source, revision, tokenizer, build package versions, ordering mode and seed is
a prefix of one stream.
**Different sources do not promise matching prefixes:** the screen's 100B
subset and the flagship's full DCLM are distinct data streams. Their validation
slices also differ, so their losses are not a paired data comparison.

`delta tokenize` indexes source row groups, selects documents whose position
falls below `--target / --tokens-per-doc`, tokenizes those documents into
per-file parts, and assembles them in stream order. The default
`--tokens-per-doc 900` is a conservative mean-length estimate used to size
that selection; it never truncates documents. A selection that runs short
fails. Published-order builds download only files intersecting the selected
prefix. Shuffled builds scan the full source. Both produce a contiguous
uint32 store, so training rows, validation and resume are local reads.

The held-out slice contains the stream's first documents up to 30M tokens;
training follows until it holds the target less that cap, ending on a
document boundary. A screen at 100 tokens per parameter requires 14.083B
predicted tokens and a rounded 15B-token store. `--data-root ROOT` places the
store at `ROOT/NAME` (default `data/dclm-100b`); `--out` overrides that path.
On Jobe the root is `/data/delta`, so the subset store is
`/data/delta/dclm-100b` and a full-source store is `/data/delta/dclm`.
The trainer uses the same `--data-root` and `--source` pair. Resume and
continuation inherit each setting independently unless it is retyped; the
store path is resolved after that inheritance.

`delta tokenize --continue` appends to a finished store under matching source,
build package versions and ordering settings, landing on the bytes a fresh
build at that target writes. Partial builds bind those settings in `build.json`; source footer
indexing and token parts are resumable. An interrupted extension discards
its uncommitted tail before appending again. Each split's sidecar
(`val.docs.npy`, `train.docs.npy`; `DOC_DTYPE`) records document start and
universe address. That address resolves to the pinned parquet file and row
holding its text, URL, ID, scores and any source-specific metadata, including
crawl information where available. `delta verify DIR` checks the store's
counts, source index, sidecars and sampled EOS boundaries.

Each of the `--workers` encoding processes prefetches one upcoming source
file while encoding its current file, bounding downloaded scratch to at most
two files per worker. `RAYON_NUM_THREADS` controls tokenizer threads per
process; with multiple workers its default divides half the host's CPU
threads between them. Assembly uses random-access mmap advice where supported
and `--readers` concurrent document readers (default 8) into disjoint ranges
of a roughly 64 MiB output buffer, followed by sequential shard writes.
These settings change throughput, not document order, token bytes, or
provenance. Assembly logs progress every 30 seconds.
See [data-build performance](data-build-performance.md) for the measured
bottleneck, tuning procedure, and storage requirements.

One row is a non-overlapping `seq_len + 1` window, so 4,097 stored tokens give
4,096 predictions. Rows may cross document boundaries, and attention crosses
them too: with shuffled documents the neighbours are unrelated, which is the
honest packing regime, and an intra-document mask would be an architecture
question for PKDA's recurrent state rather than a data one. Step `n`
addresses:

```text
first_row(n) = (n - 1) * batch_rows
```

Feedback pass counts are keyed by data seed and step; prefix lengths and
jitter are additionally keyed by the microbatch's first global row. None of
them depend on ambient RNG state, so a resume returns to the same row and the
same draws.

`delta tokenize --scale S --tokens-per-param R` sizes a store for a planned
run: that schedule's rows of `seq_len + 1` tokens plus the slice's cap,
rounded up to the next billion. The default target is the screen at 400x,
57B stored tokens with about 584M of headroom; the bridge's 400x rung is
167B from the same stream.

## Training

### Optimizer batch

Every optimizer update at every scale sees 524,288 predicted tokens, `2^19`:
128 rows of 4,096 predictions. The context is the same at every scale, so the
rungs of the ladder differ only in size, and the batch is uniform, so equal
steps are equal tokens at every rung. The efficient batch grows with the
token budget rather than the model, and `2^19` sits inside the flat basin of
the published fits at the screen's 25x budget and under it at every larger
rung ([scaling.md](scaling.md)).

| Surface | Realization |
|---|---|
| Jobe, any scale | 128 one-row microbatches |
| Prime, any scale | 8 ranks x 1 row x 16 accumulation microsteps |

Every condition uses the NorMuonH/NAdam partition in [architecture.md](architecture.md).
After all microbatches have accumulated, the single global FP32 gradient
vector is clipped to L2 norm 10.0 before both optimizer steps; telemetry
reports the pre-clip norm.

### Feedback passes

Conditions with `f` train with parallel Jacobi passes over a full sequence. Pass 1
is ordinary teacher forcing with plain embeddings. Every later pass:

1. takes the preceding pass's payload without detaching it;
2. adds keyed uniform jitter, `[-0.02, 0.02]` by default;
3. shifts the payload one position right and inserts zero at position 0;
4. draws a per-row plain-prefix length uniformly from `1..seq_len-1`;
5. uses plain embeddings on that prefix and FBT-fused inputs on the suffix;
6. runs the complete stack again.

Position 0 is always plain and the last executed position is always fused. A
`k`-pass batch trains a feedback horizon of `k-1` transitions and costs `k`
transformer evaluations. With `ell_k` the mean next-token cross-entropy on
pass `k`:

```text
K = 1:  loss = ell_1
K > 1:  loss = ell_1 + mean(ell_2, ..., ell_K)
```

During cooldown the same combination applies to the squared log-partition
penalty `mean(logsumexp(logits)^2)` with coefficient `1e-5`.

### Core iterations

Conditions with `l` draw the core iteration count `r` once per step from the
recurrent-depth log-normal Poisson draw with mean `r_mean = 4` and cap
`r_max = 8`, `E[r] = 3.88` under the cap, shared by every pass and microbatch
of the step; the draw is its own keyed sub-stream. A pass at `r` iterations
executes `2 + r` cells, so a `k`-pass batch costs `k (2 + r)` cell evaluations
per predicted token. The trainer logs the realized `r` and reports
cell-tokens beside pass-tokens.

### Schedule

All three parameter groups share one warmup-stable-cooldown multiplier. Warmup
rises linearly over `round(warmup_frac * min(steps, steps at 25x))` updates:
the fraction applies to the shorter of the run and the 25x recipe at its
geometry, so warmup is fixed per scale, 134 steps at the screen, 397 at the
bridge, 1,051 at the flagship, and a longer run does not spend more of it.
Warmup guards the optimizer's first steps at the batch and learning rate,
which do not depend on the horizon; cooldown does, and occupies
`round(cooldown_frac * steps)` updates with multiplier `1 - sqrt(u)` for
local progress `u`, reaching zero at the last step. Defaults are 0.02 and
0.20. The default 6,716-step Jobe schedule:

| Phase | Steps | Passes with `f` |
|---|---:|---|
| Warmup | 1–134 | one |
| Stable heat | 135–5,373 | one through 5,037, then two or three |
| Cooldown | 5,374–6,716 | two or three |

The feedback boundary is `round(feedback_start * steps)`, independent of the
learning-rate phases, and defaults to three quarters of the schedule. After
it every step draws three passes with probability `three_pass = 0.12` and two
otherwise, which targets a 75% / 22% / 3% pass mixture over the run and 1.28
expected pass-tokens per predicted token. Conditions without `f` use one pass
throughout.

The run is 3,521,118,208 predicted tokens: 25.0–25.2 per active non-embedding
parameter with `a`, 29.0–29.3 without. The count is derived: 25 predicted
tokens per active parameter of the `arf` stack at the screen, 140,827,944,
rounded up to whole steps, and every condition at a scale shares it.

### Knobs

| Flag | Default | What it changes |
|---|---:|---|
| `--condition` | `""` | letters from `arfl` in any order; empty is the plain decoder with three RoPE-GGQA layers and one NoPE-GGQA layer per cell |
| `--scale` | `screen` | geometry and batch preset from [scaling.md](scaling.md): `screen`, `bridge`, or `flagship`; a trunk or recipe flag typed alongside overrides its field |
| `--tokens-per-param` | 25 | predicted tokens per active non-embedding parameter of the flat full stack at the scale; derives `--steps`, rounded up to whole steps, so every condition at a scale shares one schedule (25 is the screen recipe, 400 the Prime recipes) |
| `--steps` | derived | schedule length, typed instead of derived |
| `--continue TAG` | | extend finished run TAG to this longer schedule under a new tag: its last snapshot that the longer schedule reproduces is restored, every setting but the length inherited |
| `--seq-len`, `--batch-rows`, `--micro-rows` | 4,096, 128, 1 at every scale | predictions per row, rows per step, and the microbatch; the scale keeps 524,288 predictions per step, as does a retyped `--seq-len` alone |
| `--seed`, `--data-seed` | | initialization pairing and the keyed data/feedback streams |
| `--lr-normuonh`, `--lr-nadam` | `6e-3`, `3e-4` | the NorMuonH relative step, and the base NAdam rate at the muP reference width 1536; the fan-in-`D` NAdam matrices run at `lr_nadam x 1536 / D` and the tied readout carries the same ratio |
| `--feedback-start` | 0.75 | fraction of the schedule before the feedback boundary; 0 trains fused from step 0 |
| `--three-pass` | 0.12 | probability of three passes after the boundary; 1 makes every feedback step three-pass |
| `--loop-iterations`, `--loop-max-iterations` | 4, 8 | `l`: mean and cap of the per-step core iteration draw; the mean is also the fixed evaluation and decode count |
| `--jitter` | 0.02 | payload jitter half-width |
| `--warmup-frac`, `--cooldown-frac` | 0.02, 0.20 | schedule shape; the warmup fraction applies to the shorter of the run and the 25x recipe, the cooldown fraction to the run |
| `--max-steps` | | caps this invocation without changing the schedule |
| `--resume` | | continues a tag from its latest supported snapshot: v27, or v26 with `a` |

The specimens so far used the default recipe (`ar`) and
`--feedback-start 0 --three-pass 1` (`arf`); see [findings.md](findings.md).

### Checkpoints and queue

New snapshots use checkpoint contract v27. Resume, evaluation, and forks
accept v27 for every condition and v26 for conditions with `a`, whose
computation is unchanged. A v26 snapshot without `a` is rejected because its
all-NoPE trunk does not match the current RoPE/NoPE pattern. Every snapshot
records its condition as letters. A snapshot holds the model, both
optimizer states, the fixed NorMuonH radii, the state-defining arguments, the
cumulative step, and Python/Torch/CUDA RNG state. A resume inherits every
state-defining field and rejects explicit conflicts; runtime paths, device,
evaluation cadence, snapshot cadence, and evaluation-row count may change.

Each run keeps the latest two snapshots plus protected ones at the cooldown
boundary, the feedback boundary (the last one-pass state), and the end of the
run, so cooldown and feedback variants can fork from the exact pre-boundary
state. `--continue TAG` is their built-in use: it extends a finished run to
a longer schedule under a new tag, `--tokens-per-param 50` from a 25x run,
by restoring TAG's last snapshot at or before the last step both schedules
reproduce and training on under the longer schedule with every other setting
inherited. That step is the feedback boundary when the boundary moves and the
cooldown boundary otherwise. Warmup is fixed per scale, so from that step on
a continuation of any run at or above 25x is the longer run exactly, and a
shorter source differs only in the warmup it inherited, which the `continue`
record reports as `exact`; at the screen a 25x `arf` run
continued to 50x restores step 5,037 and trains 8,394 new steps
of a 13,431-step schedule. The queue stores arguments, not Git state; a
source change never stops an active child, and the worker refreshes before
the next job.

## Evaluation

### Language-model numbers

Every evaluation point reports pass-1 held-out cross-entropy as `val` over
the slice's first `--eval-rows` rows, 512 by default: 524,288 predictions at
the screen, a quarter of the standard error of the 32 rows it replaces, a
forward-only few seconds every `--eval-every` steps. Conditions with `f` also report `val_fused`, a
second pass with plain-prefix length 1. Conditions with `l` evaluate at the fixed count `r = r_mean`.
Decoding has three modes, each at one fixed `r` under `l`:

- **Standard:** one plain prompt prefill, no feedback during decode.
- **Soft:** one plain prefill, then one feedback transition per generated
  token.
- **Fused:** an additional fused prompt pass, then the same feedback decode.

### Comparing conditions

For validation loss `L`, a letter's gain is the paired difference between a
condition with it and the same condition without it, and two letters interact
by the difference of their gains. On the `a` trunk, with `G_x = L_a - L_x`:

```text
I = G_arf - G_ar - G_af = L_ar + L_af - L_arf - L_a
```

Positive `I` is superadditive loss reduction. Computed on paired checkpoints
by mode; writing `plain` for the empty condition, `L_plain - L_a` is the
whole-trunk contrast. Report parameters, predicted tokens, pass-tokens, and
cell-tokens with every number; a matched-compute view compares at equal
cumulative cell-tokens, since a loop pass is deeper than a flat one.
`L_arf - L_arfl` at equal steps is the loop's matched-data contrast. The full two-seed screen over the `a` conditions and the plain
decoder is the natural complete comparison and has not been run; it is one
option among several for what to train next, not a prerequisite.

### Downstream tasks

`scripts/downstream_eval.py` scores a snapshot on the workspace's pinned
zero-shot suite (HellaSwag, ARC-Easy/Challenge, PIQA, WinoGrande, BoolQ,
OpenBookQA, SciQ, LAMBADA) with the evaluation harness's prompts and
normalization, in Standard, Soft, or Fused mode. Accuracy comes with a
standard error; pairwise comparison on identical documents through the
workspace module resolves gold log-probability and margin differences far
smaller than accuracy can. At this scale the tasks resolve a few accuracy
points and anchor the specimen against published models of similar size.

### Recurrent dynamics

`iterate_fused` repeatedly applies fully fused prefill to fixed held-out
tokens with plain-prefix length 1 and records loss and
`mean_token ||h_top^(k) - h_top^(k-1)||_2` per iteration. Mixer caches reset
within each prefill; only the shifted payload passes between iterations. The
training monitor runs eight iterations; the analysis scripts run thirty.
Under `l` the trace runs at `r = r_mean`.

For conditions with `l`, `depth_trace` sweeps the fixed iteration count from
1 to `r_max` and records the held-out loss after each count and the size of
each iteration's core update; the trainer logs it as the `depth` record and
`scripts/depth_trace.py` runs it on a snapshot.

### Routing and interventions

`scripts/route_report.py` measures per-site/group source mass, entropy,
cross-group divergence, source and null scale, and query geometry.
`scripts/payload_swap.py` replaces the payload enrichment with top-only,
uniform, or forced-source choices. The rest of the toolset is in
[interpretability.md](interpretability.md). A routing weight is a mixing
coefficient; learned nulls can carry nonzero values; top-only enrichment
ablation keeps `h_top` and so keeps most of the payload channel.

## Not in the current recipe

Adaptive pause or halting tokens, token-conditioned payload queries, multiple
explicit previous-column payloads, per-layer cross-column banks, alternative
optimizers, long-context continuation, instruction tuning, and any task
beyond next-token prediction on web text. Several of these are candidate
moves in [findings.md](findings.md).

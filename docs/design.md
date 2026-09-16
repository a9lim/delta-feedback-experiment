# Training recipe

Conditions, token streams, and schedules for the current model.
[Architecture](architecture.md) defines the computation;
[scaling](scaling.md) lists presets and parameter accounting.

## Conditions

| Condition | Feedback between positions | Looped columns per position |
|---|---|---|
| `f` (default) | Yes | One |
| `l` | No | Rolled during training |
| `fl` | Yes | Rolled during training |

Every condition includes PKDA/GQA cells, MHDB routing, shared and routed
experts, and auxiliary second-token prediction (MTP). MTP, feedback, and the
loop share the same concat-linear fusion, including its parameters and
per-column jitter. Shared parameters pair identically for a given
initialization seed. Data rows and recurrence draws use keyed streams; one
roll per step shapes every condition. At one column, `fl` and `f` have
identical values and gradients.

## Data

The pinned GPT-NeoX tokenizer adds `<|im_start|>` and `<|im_end|>` to its
50,277 base IDs. The model uses 50,279 token IDs and a 50,304-row tied head.
`tokenizer.py` owns the vocabulary revision and ChatML template. Pretraining
uses raw text with `<|endoftext|>` after every nonempty document.

| `--source` | Dataset / directory | Default ordering |
|---|---|---|
| `dclm-100b` (default) | `HuggingFaceFW/dclm_100BT-shuffled` / `data/` | Published |
| `dclm` | `mlfoundations/dclm-baseline-1.0-parquet` / `filtered/` | Keyed shuffle |
| `fineweb-edu` | `HuggingFaceFW/fineweb-edu` / `data/` | Keyed shuffle |
| `fineweb-edu-350b`, `fineweb-edu-100b`, `fineweb-edu-10b` | Same dataset / `sample/350BT/`, `sample/100BT/`, `sample/10BT/` | Keyed shuffle |

`data.py` owns source revisions. `--shuffle` and `--no-shuffle` override the
default. Matching source, revision, tokenizer, build-package versions, order,
and seed define one stream; larger stores extend its prefix. Different
sources have different validation slices and do not form a paired comparison.

Tokenization indexes parquet rows, encodes a selected document prefix into
parts, then assembles a contiguous uint32 stream. `--tokens-per-doc` sizes
the document selection; it never truncates a document. The first documents
up to 30M tokens form validation; training follows to the requested target.
`--data-root ROOT` stores each source at `ROOT/SOURCE`. On Jobe, use
`/data/delta`. `--out` overrides the tokenizer output path.

`meta.json` records the realized build-package versions. Resume and extension
require matching build settings; training only reads the resulting store.
Interrupted builds reuse the source index and encoded parts. Each split's
`.docs.npy` sidecar maps document starts to source rows. `delta verify DIR`
checks counts, sidecars, and up to 2,048 sampled EOS boundaries per split.

A row is a nonoverlapping window of `seq_len+1` tokens. Ordinary prediction
uses `seq_len` targets; MTP runs over the same `seq_len` positions and
supervises the `seq_len-1` of them that have a second token. Rows and
attention can cross document boundaries. Step `n` starts at row
`(n-1)*batch_rows`; with several ranks each takes a contiguous
`batch_rows/ranks` slice of it. The recurrence roll depends on data seed and
step; each row's prefix and jitter draws depend on data seed, step, and its
own global row, so they are the same bytes whatever the replay width or the
rank that runs the row.

`delta tokenize --scale S --tokens-per-param R` includes validation and
extra row targets, then rounds storage up to a billion tokens. A new screen
400x store requires 84B tokens under the current geometry. Allow roughly twice the final store size
while parts and output coexist. `--scratch` moves downloads only.
`--workers` controls encoding processes; `--readers` controls assembly
concurrency; `RAYON_NUM_THREADS` controls tokenizer threads per process.

## Training

All presets accumulate 128 rows of 4,096 predictions: 524,288 predicted
tokens per optimizer update. Every graph replays a divisor of a rank's rows
whose raw activations fit, at least `--micro-rows`, the widest first, its
PKDA recurrences keeping their backward intermediates when that fits too
([architecture](architecture.md#depth-caches-and-training)); each replay
enters the step in proportion to its rows, so the arithmetic is unchanged.
The full FP32 gradient, summed across ranks, has its L2 norm measured,
unclipped, before NorMuonH and NAdam update; a non-finite norm stops the run.
Expert-selection biases update once from the whole step's assignment counts.

Every column trains ordinary next-token prediction and MTP. For each head,
`combine(CE)` applies FBT's first-plus-mean rule along passes within each
column and then along columns, so a rolled `fl` step gives the plain
column, the loop alone, feedback alone, and both together one unit of
weight each; with one axis absent it is that axis's own rule. The
objective is:

```text
loss = combine(CE_ntp) + mtp_weight * combine(CE_mtp)
     + z_coef * (combine(z_ntp) + mtp_weight * combine(z_mtp))
     + 1e-4 * expert_balance
```

`mtp_weight` defaults to 0.3. Each head averages over its own valid positions.
`z` is the mean squared log-partition; its coefficient is `1e-5` in cooldown
and zero otherwise. Expert balancing averages across executed trunk and MTP
invocations, then columns; its coefficient is independent of `mtp_weight`.
[Architecture](architecture.md#two-token-prediction) defines target alignment
and balancing equations.

### Feedback passes

A feedback step starts with plain teacher forcing, using token embeddings
fused with the blank payload as column seeds. Every supervised column applies
the writer's learned RMSNorm, whose gain starts at one, then adds keyed
uniform jitter without detaching it. `--jitter` sets the half-width in
payload units, default `0.02`; the draw is uniform in `[-jitter, jitter]`.
It concatenates the next-token embedding and jittered payload,
then applies the shared bias-free `2D -> D` projection. Fusion applies no
normalization or second scaling to either input.
This includes single-pass batches, the final pass, and `l` without `f`. The
independent MTP block consumes that fused tensor. When another feedback pass
follows, it right-shifts the same tensor and restores plain seeds
before a per-row prefix drawn uniformly from `1..seq_len-1`. Position zero stays
plain. When another looped column follows at the same position, the same
jittered payload is fused once more with the position's own embedding
to seed it; no prefix applies, since every position has its own preceding
column. Mixer states restart each pass; every consumer backpropagates
through the shared fusion.

One `embed_tokens` call looks up the whole stored row before slicing, serving
plain seeds and every fusion consumer across all passes. The lookup is the
tied row times `1/BASE_NORMAL_INIT_STD`; the classifier reads the raw
weight. The jitter generator writes the draw directly in payload units, so
the fusion path adds the buffer without another scale factor. The pass jitter has shape
`[n_passes, B, seq_len+1, dim]`, where `B` is the number of rows in the
microbatch or replay, and perturbs each pass's last column; the loop jitter
`[n_passes, r-1, B, seq_len+1, dim]` perturbs the columns before it and is
drawn after the prefix and pass jitter, so a condition with `l` shares those
with the condition without it. The first `seq_len` positions of each draw
perturb its column's payload. Evaluation and diagnostics use no jitter.

### The recurrence roll

One keyed uniform draw per step, shared by every pass, column, and
microbatch, sets the step's shape once the roll begins. Before the boundary
every step is one pass of one column. After it, no step is single-column:
with probability `three_rate` (default `0.12`) the step runs three passes of
two columns, with the same probability two passes of three columns, and
otherwise two passes of two columns. `f` reads the pass count, `l` the
column count, and `fl` both, so the three conditions align step by step and
the pass projection is the feedback draw a condition without `l` makes:

| Roll | Probability | `f` | `l` | `fl` |
|---|---:|---|---|---|
| three passes, two columns | 0.12 | 3 passes | 2 columns | 3 x 2 |
| two passes, three columns | 0.12 | 2 passes | 3 columns | 2 x 3 |
| two passes, two columns | 0.76 | 2 passes | 2 columns | 2 x 2 |

Every scale uses the same roll. Evaluation and decode use two columns by
default; `--loop-iterations` changes only that fixed count within `1..3`.
A step of `k` passes and `r` columns executes `4kr` cells; on rolled steps
`f` and `l` each run 2.12 columns per position on average and `fl` 4.48.
See [scaling.md](scaling.md#loop-compute-and-decode-state).

### Schedule

`--tokens-per-param` defaults to 25 and derives whole steps from the flat
`f` training-active non-embedding parameter count, including MTP. Every
condition at one geometry receives the same predicted-token budget.
`--steps` overrides that length.

All optimizer groups share a warmup-stable-cooldown multiplier. Warmup is
linear over 2% of the shorter of the run and its 25x schedule. Cooldown takes
the final 20%, using `1-sqrt(u)` and reaching zero at the final update.

The recurrence roll starts after 75% of the schedule (`--recurrence-start`).
Over a whole run this gives approximately 75% / 22% / 3% one-/two-/three-pass
steps under `f`, the same split of one-/two-/three-column steps under `l`,
and 1.28 pass-tokens per predicted token for `f`; `fl` multiplies the two.
Screen runs at this token budget have rolled from the first step with
`--recurrence-start 0`. `l` without `f` always uses one pass. MTP adds
computation without increasing the ordinary predicted-token budget.

### Knobs

`delta train --help` is the complete flag reference. The main controls are:

| Flags | Purpose |
|---|---|
| `--scale`, geometry/expert flags | Select a preset and override dimensions |
| `--condition`, `--seed`, `--data-seed` | Computation, paired initialization, keyed rows/draws |
| `--tokens-per-param`, `--steps` | Schedule length |
| `--lr-normuonh`, `--lr-nadam` | Peak optimizer rates; defaults `0.006`, `0.0003` |
| `--warmup-frac`, `--cooldown-frac` | Schedule shape |
| `--recurrence-start`, `--three-rate` | Recurrence roll: its boundary and the probability of each three-deep outcome |
| `--jitter` | Payload-jitter half-width in payload units, default `0.02` |
| `--loop-iterations` | Fixed evaluation/decode columns per position, default 2 within `1..3`; training draws from the roll |
| `--mtp-weight` | Auxiliary prediction weight |
| `--seq-len`, `--batch-rows` | Batch geometry |
| `--ranks`, `--micro-rows` | Processes per invocation and the smallest replay; runtime, not state |
| `--precision` | CUDA GEMM recipe, `fp8` (default) or `bf16`; runtime, inherited by a resume unless retyped |
| `--resume`, `--continue TAG`, `--max-steps` | Run lifecycle |

### Checkpoints and queue

Checkpoint v42 binds model, both optimizers, arguments, step, RNG state, and
tokenizer identity. Resume inherits state-defining settings and rejects
explicit conflicts. Device, paths, evaluation cadence, snapshot cadence, the
smallest replay, and the rank count can change: a snapshot is one file of
FP32 masters and whole optimizer state whatever number of ranks wrote it,
and any number of ranks resumes it, each taking the matrices it owns. The
checkpoint includes expert-selection biases and NorMuonH radius/spectral
state; transient counts and working copies are rebuilt. Only v42 snapshots
are accepted. The recurrence roll's boundary and rate are
saved schedule arguments, independent of the saved evaluation-count
argument. Token lookups multiply the tied table by
`1/BASE_NORMAL_INIT_STD`, payloads use the writer's learned RMSNorm with a
gain initialized to one, and jitter buffers are sampled in those payload
units, so both fusion inputs are unit RMS at initialization. Fusion concatenates its
inputs and projects them without further scaling, and every column seed is
one such product: positions without an incoming payload fuse the learned
blank payload `blank_payload`. This defines the current
parameter names and optimizer ownership. The fusion projection belongs to
ordinary NorMuonH; the writer's payload norm and the blank payload belong to
base NAdam.

The trainer keeps the latest two snapshots and protected recurrence,
cooldown, and final boundaries. `--continue TAG` extends a finished run under a new tag
by restoring its last snapshot that the longer schedule reproduces. Other
settings are inherited. Warmup is fixed per scale at or above 25x; shorter
source schedules may differ in warmup, as reported by the continuation record.

The queue saves arguments and refreshes the checkout before each job. Source
changes do not stop an active child. Operational commands are in
[operations.md](operations.md#runs).

## Evaluation

`val` is plain held-out next-token CE, read after the evaluation column
count. With `f`, `val_fused` is a second pass with plain-prefix length 1.
With `l`, `val_one` is the first column's CE, the single-column model.
`val_mtp` and `val_mtp_fused` measure the separate second-token predictor
over its supervised positions, before weights and z-loss. All come from the
same per-row head results the training objective uses, with jitter disabled
for both heads. Evaluation reads the first `--eval-rows` validation rows,
128 by default.

Generation uses three modes: **Standard** prefills and decodes without
feedback; **Soft** prefills plainly and feeds payloads back during decode;
**Fused** adds a fused prompt pass before that feedback decode. Ordinary
generation uses only the main next-token head.

Compare conditions on matching source, rows, seed, schedule, and evaluation
mode. Report predicted tokens, pass-tokens, cell-tokens, and measured device
time: equal data does not imply equal compute. The `f`, `l`, and `fl`
conditions isolate conditional additions of feedback or depth; they omit a
no-recurrence control.

The trainer records fixed-token payload self-composition and per-column
depth traces. [Interpretability](interpretability.md) describes the retained
checkpoint tools and how to read their measurements.

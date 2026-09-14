# Training recipe

Conditions, token streams, and schedules for the current model.
[Architecture](architecture.md) defines the computation;
[scaling](scaling.md) lists presets and parameter accounting.

## Conditions

| Condition | Feedback between columns | Tied core depth |
|---|---|---|
| `f` (default) | Yes | One iteration |
| `l` | No | Sampled during training |
| `fl` | Yes | Sampled during training |

Every condition includes PKDA/GQA cells, MHDB routing, shared and routed
experts, and auxiliary second-token prediction (MTP). MTP and feedback share
the same concat-linear fusion, including its parameters and
per-pass jitter. Shared parameters pair identically for a given initialization
seed. Data rows and feedback draws use keyed streams; tied depth uses a
separate stream. At one core iteration,
`fl` and `f` have identical values and gradients.

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
`(n-1)*batch_rows`. Feedback draws depend on data seed and step;
prefix/jitter draws also depend on the first global row of the microbatch.

`delta tokenize --scale S --tokens-per-param R` includes validation and
extra row targets, then rounds storage up to a billion tokens. A new screen
400x store requires 84B tokens under the current geometry. Allow roughly twice the final store size
while parts and output coexist. `--scratch` moves downloads only.
`--workers` controls encoding processes; `--readers` controls assembly
concurrency; `RAYON_NUM_THREADS` controls tokenizer threads per process.

## Training

All presets accumulate 128 one-row microbatches of 4,096 predictions:
524,288 predicted tokens per optimizer update. A one-pass graph may replay
several microbatches at once when their raw activations fit; each replay
enters the step in proportion to its rows, so the arithmetic is unchanged.
The full FP32 gradient is clipped to L2 norm 10 before NorMuonH and NAdam
update. Expert-selection biases update once from the whole step's assignment
counts.

Every pass trains ordinary next-token prediction and MTP. For each head,
`combine(CE)` is pass-1 CE plus the mean later-pass CE, or just pass-1 CE
when only one pass runs. The objective is:

```text
loss = combine(CE_ntp) + mtp_weight * combine(CE_mtp)
     + z_coef * (combine(z_ntp) + mtp_weight * combine(z_mtp))
     + 1e-4 * expert_balance
```

`mtp_weight` defaults to 0.3. Each head averages over its own valid positions.
`z` is the mean squared log-partition; its coefficient is `1e-5` in cooldown
and zero otherwise. Expert balancing averages across executed trunk and MTP
invocations, then passes; its coefficient is independent of `mtp_weight`.
[Architecture](architecture.md#two-token-prediction) defines target alignment
and balancing equations.

### Feedback passes

A feedback step starts with plain teacher forcing, using raw token embeddings
as column seeds. Every supervised pass applies the writer's learned RMSNorm
with its fixed output scale `BASE_NORMAL_INIT_STD=0.02`, then adds keyed
uniform jitter without detaching it. `--jitter` sets the relative half-width,
default `0.02`; the actual draw is uniform in
`[-BASE_NORMAL_INIT_STD * jitter, BASE_NORMAL_INIT_STD * jitter]`, or
`[-0.0004,0.0004]` by default. It concatenates the raw next-token embedding and jittered payload,
then applies the shared bias-free `2D -> D` projection. Fusion applies no
normalization or second scaling to either input.
This includes single-pass batches, the final pass, and `l` without `f`. The
independent MTP block consumes that fused tensor. When another feedback pass
follows, it right-shifts the same tensor and restores raw embeddings
before a per-row prefix drawn uniformly from `1..seq_len-1`. Position zero stays
plain. Mixer states restart each pass; both consumers backpropagate through
the shared fusion.

One `embed_tokens` call looks up the whole stored row before slicing, serving
plain seeds and both fusion consumers across all passes. The tied classifier
uses the same raw embedding weight. The jitter generator writes the draw
directly in scaled payload units, so the fusion path adds the buffer without
another scale factor. Jitter has shape
`[n_passes, B, seq_len+1, dim]`, where `B` is the number of rows in the
microbatch or replay; the first `seq_len` positions of each pass's draw
perturb its payload. Evaluation and diagnostics use no jitter.

### Core iterations

With `l`, one independent keyed categorical draw sets the core depth for the
whole optimizer step, shared by every microbatch and feedback pass. All four
scales use probabilities `30/30/20/10/10%` for one through five total core
visits, giving an actual mean of 2.4. Evaluation and decode use three visits
by default; `--loop-iterations` changes only that fixed depth within `1..5`.
For `C` cells, each pass executes `2+(C-2)r` cells. Every preset has four
cells, so expected training depth is 27.2 trunk layers, default evaluation
depth is 32, and maximum depth is 48. See
[scaling.md](scaling.md#loop-compute-and-decode-state).

### Schedule

`--tokens-per-param` defaults to 25 and derives whole steps from the flat
`f` training-active non-embedding parameter count, including MTP. Every
condition at one geometry receives the same predicted-token budget.
`--steps` overrides that length.

All optimizer groups share a warmup-stable-cooldown multiplier. Warmup is
linear over 2% of the shorter of the run and its 25x schedule. Cooldown takes
the final 20%, using `1-sqrt(u)` and reaching zero at the final update.

Feedback starts after 75% of the schedule. Subsequent steps use three passes
with probability 0.12, otherwise two. This gives approximately 75% / 22% / 3%
one-/two-/three-pass steps and 1.28 pass-tokens per predicted token. `l`
without `f` always uses one pass. MTP adds computation without increasing the
ordinary predicted-token budget.

### Knobs

`delta train --help` is the complete flag reference. The main controls are:

| Flags | Purpose |
|---|---|
| `--scale`, geometry/expert flags | Select a preset and override dimensions |
| `--condition`, `--seed`, `--data-seed` | Computation, paired initialization, keyed rows/draws |
| `--tokens-per-param`, `--steps` | Schedule length |
| `--lr-normuonh`, `--lr-nadam` | Peak optimizer rates; defaults `0.006`, `0.0003` |
| `--warmup-frac`, `--cooldown-frac` | Schedule shape |
| `--feedback-start`, `--three-pass` | Feedback mixture |
| `--jitter` | Relative payload-jitter half-width, default `0.02`; actual amplitude is `0.02 * jitter` |
| `--loop-iterations` | Fixed evaluation/decode core depth, default 3 within `1..5`; training distribution is fixed |
| `--mtp-weight` | Auxiliary prediction weight |
| `--seq-len`, `--batch-rows`, `--micro-rows` | Batch geometry |
| `--resume`, `--continue TAG`, `--max-steps` | Run lifecycle |

### Checkpoints and queue

Checkpoint v39 binds model, both optimizers, arguments, step, RNG state, and
tokenizer identity. Resume inherits state-defining settings and rejects
explicit conflicts. Device, paths, evaluation cadence, and snapshot cadence
can change. The checkpoint includes expert-selection biases and NorMuonH
radius/spectral state; transient counts and classifier shadows are rebuilt.
Only v39 snapshots are accepted. The core-depth training recipe is the fixed
`30/30/20/10/10%` distribution, independent of the saved evaluation-depth
argument. Token lookups are raw, payloads use the
writer's learned RMSNorm with fixed output scale `BASE_NORMAL_INIT_STD=0.02`,
and jitter buffers are sampled in those scaled payload units. The same
constant sets token embedding initialization. Fusion concatenates its
inputs and projects them without further scaling. This defines the current
parameter names and optimizer ownership. The fusion projection belongs to
ordinary NorMuonH; the writer's payload norm belongs to base NAdam.

The trainer keeps the latest two snapshots and protected feedback, cooldown,
and final boundaries. `--continue TAG` extends a finished run under a new tag
by restoring its last snapshot that the longer schedule reproduces. Other
settings are inherited. Warmup is fixed per scale at or above 25x; shorter
source schedules may differ in warmup, as reported by the continuation record.

The queue saves arguments and refreshes the checkout before each job. Source
changes do not stop an active child. Operational commands are in
[operations.md](operations.md#runs).

## Evaluation

`val` is plain held-out next-token CE. With `f`, `val_fused` is a second pass
with plain-prefix length 1. `val_mtp` and `val_mtp_fused` measure the separate
second-token predictor over its supervised positions, before weights and
z-loss. All four come from the same per-row head results the training
objective uses, with jitter disabled for both heads. Evaluation reads the
first `--eval-rows` validation rows, 128 by default.

Generation uses three modes: **Standard** prefills and decodes without
feedback; **Soft** prefills plainly and feeds payloads back during decode;
**Fused** adds a fused prompt pass before that feedback decode. Ordinary
generation uses only the main next-token head.

Compare conditions on matching source, rows, seed, schedule, and evaluation
mode. Report predicted tokens, pass-tokens, cell-tokens, and measured device
time: equal data does not imply equal compute. The `f`, `l`, and `fl`
conditions isolate conditional additions of feedback or depth; they omit a
no-recurrence control.

The trainer records fixed-token payload self-composition and tied-depth
traces. [Interpretability](interpretability.md) describes the retained
checkpoint tools and how to read their measurements.

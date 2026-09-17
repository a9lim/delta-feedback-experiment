# Training recipe

The recipe defines the data, objective, recurrence distribution, and evaluation
for the model in [architecture](architecture.md). [Scaling](scaling.md) gives
geometry and compute budgets; [operations](operations.md) covers execution and
checkpoints.

## Conditions

| Condition | Feedback between positions | Looped columns per position |
|---|---|---|
| `f` (default) | Enabled | One |
| `l` | Disabled | Sampled during training |
| `fl` | Enabled | Sampled during training |

All conditions retain the same PKDA/GQA stack, MHDB readers, shared and routed
experts, payload writer, fusion, and auxiliary second-token predictor (MTP).
The letters select recurrence axes, not component ablations. Shared parameters
initialize identically at a given `--seed`; at one column, `fl` and `f` have
identical values and gradients. These conditions omit a no-recurrence control.

## Data

The pinned GPT-NeoX tokenizer adds `<|im_start|>` and `<|im_end|>` to its
50,277 base IDs. It has 50,279 token IDs; the tied embedding/head has 50,304
rows for alignment. [tokenizer.py](../delta_feedback_experiment/tokenizer.py)
owns its revision, identity, and ChatML template, which preserves role names
and turn boundaries without imposing speaker alternation. Pretraining uses
raw text with `<|endoftext|>` after each nonempty document, not ChatML.

| `--source` | Dataset and directory | Default ordering |
|---|---|---|
| `dclm-100b` (default) | `HuggingFaceFW/dclm_100BT-shuffled`, `data/` | Published |
| `dclm` | `mlfoundations/dclm-baseline-1.0-parquet`, `filtered/` | Keyed shuffle |
| `fineweb-edu` | `HuggingFaceFW/fineweb-edu`, `data/` | Keyed shuffle |
| `fineweb-edu-350b`, `fineweb-edu-100b`, `fineweb-edu-10b` | Same dataset, `sample/350BT/`, `sample/100BT/`, `sample/10BT/` respectively | Keyed shuffle |

[data.py](../delta_feedback_experiment/data.py) pins source revisions.
`--shuffle` and `--no-shuffle` override the ordering. Source, revision,
tokenizer, build-package versions, ordering, and shuffle seed identify one
stream; larger stores extend its prefix when the holdout is unchanged.
Different sources have different validation slices and do not form a paired
comparison.

The builder indexes parquet rows, encodes complete documents, and assembles
a contiguous uint32 stream. `--tokens-per-doc` estimates how many documents
to select; it never truncates them. `--all` selects every source document
and consumes the entire stream. Validation takes the initial complete
documents that fit within `--val` (30M by default); the publication script
uses 100M. Training takes the following documents to the requested target,
or to source exhaustion with `--all`. Empty texts are skipped. `meta.json`
records build identity and counts; each split's `.docs.npy` sidecar maps
document starts to source rows.
Build resume and extension require matching settings. Store commands and
verification are in [operations](operations.md#tokenize).

A training row contains `T+1` tokens, where `T = --seq-len`. Rows are
nonoverlapping; both rows and attention can cross document boundaries.
[Architecture](architecture.md#auxiliary-two-token-prediction) defines
trunk/MTP target alignment and valid positions.

At step `s`, the global batch starts at row `(s-1) * batch_rows`. Ranks take
contiguous equal slices. The recurrence roll is keyed by `--data-seed` and
step; prefix and jitter draws are keyed additionally by each global row.
Changing the rank count or replay width preserves rows and per-row draws.
The data-build shuffle seed selects stream order separately from this
training randomness seed.

## Training

### The recurrence roll

Let `k` be feedback passes and `r` columns per pass. Through step
`round(recurrence_start * steps)`, every condition executes `(k,r) = (1,1)`.
Every subsequent step draws one shared shape:

| Draw | Probability | `f`: `(k,r)` | `l`: `(k,r)` | `fl`: `(k,r)` |
|---|---:|---|---|---|
| Three passes, two columns | `three_rate` = 0.12 | `(3,1)` | `(1,2)` | `(3,2)` |
| Two passes, three columns | `three_rate` = 0.12 | `(2,1)` | `(1,3)` | `(2,3)` |
| Two passes, two columns | `1 - 2*three_rate` = 0.76 | `(2,1)` | `(1,2)` | `(2,2)` |

The draw is shared across the whole optimizer step and all scales. Thus
passes and columns are coupled, never independent; each enabled axis takes
two or three on rolled steps. `--three-rate` must be at most 0.5.
`--recurrence-start` defaults to 0.75; setting it to zero rolls from step 1.
Compute expectations are in [scaling](scaling.md#loop-compute-and-decode-state).

Every feedback pass after the first draws a separate plain-prefix length
per row, uniformly in `1..T-1`. Position zero stays plain and the final
position receives feedback. The loop has no prefix: each later column uses
its own position's preceding payload. These differentiable paths and their
causal alignment are defined in [architecture](architecture.md).

Every supervised column adds uniform payload jitter in
`[-jitter, jitter]`, with `--jitter 0.02` by default. One draw serves that
column's MTP, feedback, and loop consumers, including on single-pass batches
and final columns. Prefix draws and last-column jitter precede earlier-column
jitter in each row's keyed stream, so adding `l` preserves the draws shared
with `f`. Evaluation and diagnostics disable jitter.

### Objective

Every executed column trains next-token prediction and MTP. Define
`F(x) = x[0]` for one element, otherwise
`F(x) = x[0] + mean(x[1:])`. `combine` applies `F` over passes for each
column, then over columns. On rolled `fl` steps this assigns one unit of
weight to each of four groups: the plain first column, later columns of
pass 1, column 1 of later passes, and later columns of later passes. It
does not average the four groups into one unit. With a single recurrence
axis, the corresponding two groups each receive one unit.

```text
loss = combine(CE_ntp) + mtp_weight * combine(CE_mtp)
     + z_coef * (combine(z_ntp) + mtp_weight * combine(z_mtp))
     + 1e-4 * expert_balance
```

`--mtp-weight` defaults to 0.3. Each `z` is the mean squared log-partition;
`z_coef` is `--zloss` (default `1e-5`) during cooldown and zero otherwise.
Expert balance averages equally over executed trunk and MTP layer
invocations, then over all pass/column pairs. Its coefficient is independent
of MTP's prediction weight. [Architecture](architecture.md) defines MTP and
the per-layer balance loss.

Replays contribute in proportion to their row count; gradients accumulate
in FP32 and sum across ranks before the update. The trainer measures the full L2 gradient norm without clipping
and stops on a non-finite norm. Expert-selection biases update once from
the whole step's assignment counts, at rate `0.001`.

### Optimizer settings

| Optimizer | Base peak rate | Fixed settings |
|---|---|---|
| NorMuonH | `--lr-normuonh 0.006` | Nesterov momentum `0.95`, row second moment `0.95`, epsilon `1e-8`, 5 Newton–Schulz steps, 3 spectral power iterations |
| NAdam | `--lr-nadam 0.0003` | Betas `(0.9, 0.95)`, epsilon `1e-8`, momentum decay `0.004` |

Neither optimizer uses weight decay. [Architecture](architecture.md) defines
parameter ownership and update equations; [scaling](scaling.md) gives the
rate multipliers, batch geometry, and memory accounting.

### Schedule

[Token budgets](scaling.md#token-budgets) gives the schedule-length calculation
for `--tokens-per-param` or `--steps`. The resulting `steps` and the geometry's
25x reference length, `steps_at_25x`, set the phase boundaries below.

Both optimizers use a warmup-stable-cooldown multiplier:

| Phase | Duration | Multiplier |
|---|---|---|
| Warmup | `round(warmup_frac * min(steps, steps_at_25x))`, default fraction 0.02 | Linear rise |
| Stable | Remaining steps before cooldown | 1 |
| Cooldown | `round(cooldown_frac * steps)`, default fraction 0.20 | `1-sqrt(u)` for cooldown progress `u`, reaching zero at the final update |

Warmup therefore stays fixed at a geometry for budgets at or above 25x;
cooldown and the recurrence boundary remain fractions of total steps.
Resume and schedule extension are described in
[operations](operations.md#checkpoints). `delta train --help` is the complete
flag reference.

## Evaluation

Evaluation uses the first `--eval-rows` validation rows (default 128), no
jitter, and unregularized per-head cross-entropy. For `l` and `fl`,
`--loop-iterations` sets a fixed evaluation/decode count in `1..3`, default
2; it does not control training draws. `f` always uses one column per pass.

| Metric | Measurement |
|---|---|
| `val` | Main-head CE after the final column of a plain pass |
| `val_fused` | Main-head CE after the final column of a second pass with plain-prefix length 1; `f`/`fl` only |
| `val_one` | Main-head CE after the first column of the plain pass; `l`/`fl` only |
| `val_mtp` | MTP CE on valid second-token targets of the plain pass's final column |
| `val_mtp_fused` | Corresponding MTP CE of the second pass; `f`/`fl` only |

Generation uses the main next-token head. **Standard** prefills and decodes
without feedback; **Soft** prefills plainly and uses feedback during decode;
**Fused** adds a fused prompt pass before feedback decode. Looped conditions
use their configured columns in each mode. The [analysis guide](interpretability.md)
distinguishes sequential generation from downstream teacher-forced scoring.

Compare conditions on matching source, rows, initialization and data seeds,
schedule, and evaluation mode. Report predicted tokens, pass-tokens,
cell-tokens, and measured device time: equal data does not imply equal compute.

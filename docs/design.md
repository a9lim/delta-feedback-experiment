# Growing recipe: conditions, data, schedule, evaluation

This page describes how a specimen is grown: the condition letters and how
conditions pair, the token stream, optimizer batch, feedback passes, schedule,
and evaluation.
[Architecture](architecture.md) owns the equations;
[scaling](scaling.md) owns the geometry presets and their accounting; [interpretability](interpretability.md) owns the analysis scripts.

## Conditions

Conditions are `f`, `l`, and `fl`. `parse_condition` accepts their letters in
any order and canonicalizes to `fl` order. Empty input and other letters are
rejected. The default is `f`.

| Letter | Behavior | `ModelConfig` flag |
|---|---|---|
| `f` | FBT token-gated latent payload transfer between columns | `feedback` |
| `l` | Repeated application of the tied middle cells | `loop` |

The model always uses `[PKDA, PKDA, PKDA, NoPE-GGQA]` cells, MHDB source
routing, one shared plus top-`k`-of-`n` routed experts, and an
auxiliary PKDA/expert two-token predictor. Every condition produces a
normalized top state with routed enrichment as its payload. MTP reads this
payload; `f` also transfers it to the next column. `l` alone uses embedding
seeds. The presets fix expert intermediate width at 832 and use `(k,n)`
of `(3,15)`, `(5,23)`, `(7,31)`, and `(11,47)`. Experts have no capacity limit
or token dropping. The auxiliary block has independent PKDA state that resets for
each row and pass.

Shared parameters initialize identically for a given seed. Data order and
feedback pass, prefix, and jitter draws use keyed streams. The tied-core
depth draw uses an independent stream, so `f` and `fl` can be compared on
identical rows and feedback draws. At one core iteration, `fl` equals `f`,
including routes, losses, and gradients. Tied depth requires at least three
whole cells and adds no parameters.

## Geometry

[Scaling](scaling.md) gives the screen, bridge, flagship, and extension geometry,
parameter counts, token budgets, and cache accounting. The screen has width
768 and context 4,096. All four scales have sixteen layers in four cells.
Under `l`, the first cell is the prelude, the middle two form the tied core,
and the last is the coda.
[Architecture](architecture.md#precision-and-initialization) defines the
shared initialization, numerical, and optimizer contracts.

## Data

The source is chosen with `delta tokenize --source NAME`. All sources use
`EleutherAI/gpt-neox-20b` at commit
`c292233c833e336628618a88a648727eb3dff0a7`. Its 50,277 base IDs are extended
with `<|im_start|>` (50,277) and `<|im_end|>` (50,278), giving 50,279 token
IDs and a model vocabulary padded to 50,304 rows. Loading these tokenizer
files loads no model weights. Document EOS is `<|endoftext|>` (0); every
non-empty pretraining document ends with it. ChatML's message end is distinct.

The generic ChatML template serializes each message as
`<|im_start|>ROLE\nCONTENT<|im_end|>\n`. Roles remain exactly as supplied,
including arbitrary names and repeated consecutive roles; the template does
not enforce user/assistant alternation or inject a system message. With a
generation prompt it appends `<|im_start|>NEXT_ROLE\n`, where `next_role`
defaults to `self`. This formatting contract does not itself train
conversational behavior; pretraining continues to use raw web documents.

The `data-build` extra sets minimum versions for the four packages that
compile the stream; `meta.json` records their realized versions. New stores
can use newer packages. Resuming or extending a store requires its recorded
build package versions so one store never mixes compilation stacks.

| Source | Dataset and parquet directory | Default ordering |
|---|---|---|
| `dclm-100b` (default; screen and Jobe) | `HuggingFaceFW/dclm_100BT-shuffled`, `data/` | Published order |
| `dclm` (larger stores, including flagship and extension) | `mlfoundations/dclm-baseline-1.0-parquet`, `filtered/` | Keyed document shuffle |
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
document boundary. A screen at 100 tokens per parameter requires about
20.092B predicted tokens and a rounded 21B-token store. `--data-root ROOT`
places the store at `ROOT/NAME` (default `data/dclm-100b`); `--out` overrides that path.
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
document addressing. Assembly logs progress every 30 seconds. Peak storage
includes selected token parts and the final stream on the output filesystem;
allow roughly twice the finished store plus sidecars and download scratch.
`--scratch` relocates downloads only, not token parts.

One row is a non-overlapping `seq_len + 1` window, so 4,097 stored tokens give
4,096 predictions. Rows and attention may cross document boundaries. Step `n`
addresses:

```text
first_row(n) = (n - 1) * batch_rows
```

The same row also supplies `seq_len - 1` auxiliary second-token
targets. Cropping keeps every auxiliary target supervised and preserves the
stored-row format, row order, and ordinary predicted-token budget.

Feedback pass counts are keyed by data seed and step; prefix lengths and
jitter are additionally keyed by the microbatch's first global row. None of
them depend on ambient RNG state, so a resume returns to the same row and the
same draws.

`delta tokenize --scale S --tokens-per-param R` sizes a store for a planned
run: that schedule's rows of `seq_len + 1` tokens plus the slice's cap,
rounded up to the next billion. The default target is the screen at 400x,
81B stored tokens; the bridge's 400x rung is 179B.

## Training

### Optimizer batch

Every optimizer update at every scale sees 524,288 predicted tokens, `2^19`:
128 rows of 4,096 predictions, accumulated in 128 one-row microbatches.

Every condition uses the NorMuonH/NAdam partition in [architecture.md](architecture.md).
After all microbatches have accumulated, the single global FP32 gradient
vector is clipped to L2 norm 10.0 before both optimizer steps; telemetry
reports the pre-clip norm.

### Two-token prediction

Every run trains one DeepSeek-style auxiliary prediction depth. On every
pass, it combines the payload at `t` with the shared embedding of the
ground-truth next token and predicts the token at `t + 2`. Gradients flow
through both inputs, directly training the payload writer in every condition.
This requires payload construction on single-pass batches and the final
feedback pass too. [Architecture](architecture.md#two-token-prediction)
defines the causal PKDA/expert block and exact target alignment.

The auxiliary cross-entropy averages over `seq_len - 1` positions per row.
It uses the same pass-1-plus-mean-feedback combination as ordinary
cross-entropy and the same cooldown z-loss coefficient, then multiplies both
by `--mtp-weight` (default 0.3). The weight is constant for the run, finite,
and nonnegative; it is stored among the exact-resume settings. The reported
`ntp` and `mtp` metrics are the combined ordinary and auxiliary
cross-entropies before coefficients or z-loss. `pass1` remains ordinary
first-pass cross-entropy; `loss` reports the complete optimized objective,
including MTP, z-loss, and expert balancing. On multiple-pass steps,
`ntp - pass1` gives the mean feedback-pass next-token cross-entropy.

The extra block and vocabulary loss run once per pass, regardless of the
trunk's tied-core iteration count. They add training computation without
changing the schedule's predicted-token count. Include that work in compute
accounting alongside trunk cell-tokens.

### Expert balancing

A sigmoid router selects the top `k = experts_per_token` scores after adding
each expert's persistent selection bias. The output mixture uses the original
scores, normalized over the selected experts. Each physical bank accumulates
assignment counts over the update's microbatches, feedback passes, and repeated
core invocations. The auxiliary bank contributes once per pass over its
`seq_len - 1` positions. After the optimizer step, overused experts decrease their
bias by `0.001` and underused experts increase it by `0.001`; equal load leaves
it unchanged. Forward execution and activation recomputation never update the
bias, and evaluation holds it fixed.

A complementary load-balance loss is computed separately for each sequence,
using sigmoid scores normalized over all routed experts and a detached top-`k`
selection fraction computed before adding the bias. Actual biased dispatch
counts drive the separate bias controller. The loss is averaged over
sequences within each bank, over all executed trunk invocations plus the one
auxiliary invocation, then over feedback passes, and added with coefficient
`1e-4`. The auxiliary contribution is independent of `--mtp-weight`, which
weights only its prediction cross-entropy and z-loss. The balancing weight stays
fixed as depth or pass count changes. Reported cross-entropy and perplexity
exclude it. [Architecture](architecture.md#shared-and-routed-experts)
defines the exact equations and count normalization.

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
of the step; the draw is its own keyed sub-stream. With `C` cells, a pass at
`r` iterations executes `2 + (C - 2) r` cells. A `k`-pass batch multiplies
that count by `k` for its cell evaluations per predicted token. The trainer logs the realized `r` and reports
cell-tokens beside pass-tokens.

### Schedule

All five parameter groups across the two optimizers share one
warmup-stable-cooldown multiplier.
Warmup rises linearly over
`round(warmup_frac * min(steps, steps at 25x))` updates. The default 2%
warmup is therefore fixed per scale for runs at or above 25x: 192 updates
at screen, 426 at bridge, 753 at flagship, and 1,684 at extension. Cooldown
occupies `round(cooldown_frac * steps)` updates, default 20%, with multiplier
`1-sqrt(u)` at local progress `u`, reaching zero at the final step.

The feedback boundary is `round(feedback_start * steps)`, independent of the
learning-rate phases, and defaults to three quarters of the schedule. After
it every step draws three passes with probability `three_pass = 0.12` and two
otherwise, which targets a 75% / 22% / 3% pass mixture over the run and 1.28
expected pass-tokens per predicted token. Conditions without `f` use one pass
throughout.

The default screen schedule is 9,581 steps, or 5,023,203,328 predicted
tokens. Its budget is derived from 25 tokens per training-active non-embedding
parameter of `f`, including MTP, rounded up to whole steps. All conditions
at a geometry share the same schedule; [scaling.md](scaling.md) gives the
reference counts and larger budgets.

### Knobs

| Flag | Default | What it changes |
|---|---:|---|
| `--condition` | `f` | `f`, `l`, or `fl`; at least one recurrence must be selected |
| `--scale` | `screen` | geometry and batch preset from [scaling.md](scaling.md): `screen`, `bridge`, `flagship`, or `extension`; a trunk or recipe flag typed alongside overrides its field |
| `--expert-intermediate` | 832 | each shared and routed expert's intermediate width, in both trunk and MTP |
| `--num-routed-experts`, `--experts-per-token` | 15, 3 at screen | routed bank size and selected routed experts per token; bridge uses 23, 5; flagship 31, 7; extension 47, 11 |
| `--tokens-per-param` | 25 | predicted tokens per training-active non-embedding parameter of flat `f`, including MTP; derives `--steps`, rounded up to whole steps, so every condition at a scale shares one schedule |
| `--steps` | derived | schedule length, typed instead of derived |
| `--continue TAG` | | extend finished run TAG to this longer schedule under a new tag: its last snapshot that the longer schedule reproduces is restored, every setting but the length inherited |
| `--seq-len`, `--batch-rows`, `--micro-rows` | 4,096, 128, 1 at every scale | predictions per row, rows per step, and the microbatch; the scale keeps 524,288 predictions per step, as does a retyped `--seq-len` alone |
| `--seed`, `--data-seed` | | initialization pairing and the keyed data/feedback streams |
| `--lr-normuonh` | `6e-3` | base NorMuonH relative step; expert gate/up matrices multiply it by `sqrt(1536/D)` and expert down matrices by `sqrt(8/(k+1))`, where `k` is the selected routed expert count |
| `--lr-nadam` | `3e-4` | base NAdam rate at reference width 1536; fan-in-`D` NAdam matrices run at `lr_nadam x 1536 / D` and the tied readout carries the same ratio |
| `--feedback-start` | 0.75 | fraction of the schedule before the feedback boundary; 0 trains fused from step 0 |
| `--three-pass` | 0.12 | probability of three passes after the boundary; 1 makes every feedback step three-pass |
| `--loop-iterations`, `--loop-max-iterations` | 4, 8 | `l`: mean and cap of the per-step core iteration draw; the mean is also the fixed evaluation and decode count |
| `--jitter` | 0.02 | payload jitter half-width |
| `--mtp-weight` | 0.3 | finite nonnegative weight for auxiliary cross-entropy and its z-loss, constant across the run |
| `--warmup-frac`, `--cooldown-frac` | 0.02, 0.20 | schedule shape; the warmup fraction applies to the shorter of the run and the 25x recipe, the cooldown fraction to the run |
| `--max-steps` | | caps this invocation without changing the schedule |
| `--resume` | | continues a tag from its latest v33 snapshot |

### Checkpoints and queue

Checkpoint v33 is the only accepted contract for resume, evaluation, and
forks. Its optimizer contract has three NorMuonH groups (`normuonh`,
`normuonh_expert_in`, `normuonh_expert_out`) and two NAdam groups (`nadam`,
`nadam_width`). Conditions are `f`, `l`, or `fl`; every state dictionary
includes the payload writer and auxiliary PKDA/expert prediction parameters, and
state-defining arguments include `mtp_weight`, `expert_intermediate`,
`num_routed_experts`, and `experts_per_token`. Snapshots bind to the current
GPT-NeoX/ChatML tokenizer and 50,304-row head. The snapshot and token
store carry the same deterministic
`tokenizer_id`, derived from the pinned vocabulary, delimiter IDs, and full
ChatML template; mismatches are rejected. Every snapshot records its condition
as letters. A snapshot holds the model, both optimizer states, the fixed NorMuonH radii, the state-defining arguments, the
cumulative step, and Python/Torch/CUDA RNG state. A resume inherits every
state-defining field and rejects explicit conflicts; runtime paths, device,
evaluation cadence, snapshot cadence, and evaluation-row count may change.
Snapshots include the current per-bank `expert_bias` buffers, including MTP's;
strict state restoration requires them. Assignment counts are transient and are
consumed before a step's snapshot is staged.

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
record reports as `exact`. A screen 25x run continued to 50x restores step
7,186 and trains 11,976 more steps of a 19,162-step schedule. The queue stores
arguments, not Git state; a source change never stops an active child, and the
worker refreshes before the next job.

## Evaluation

### Language-model numbers

Every evaluation point reports pass-1 held-out cross-entropy as `val` over
the slice's first `--eval-rows` rows, 128 by default: 524,288 predictions at
the screen, every `--eval-every` steps. Conditions with `f` also report `val_fused`, a
second pass with plain-prefix length 1. Conditions with `l` evaluate at the fixed count `r = r_mean`.
Every condition reports `val_mtp` over the valid second-token targets;
with `f`, `val_mtp_fused` reads the second pass's payloads. These auxiliary
metrics exclude their training weight and z-loss. `val` and `val_fused` remain
ordinary next-token losses over all `seq_len` positions.
Decoding has three modes, each at one fixed `r` under `l`:

- **Standard:** one plain prompt prefill, no feedback during decode.
- **Soft:** one plain prefill, then one feedback transition per generated
  token.
- **Fused:** an additional fused prompt pass, then the same feedback decode.

All three modes use ordinary next-token prediction. The MTP module is a
training and auxiliary-validation branch; speculative decoding is not
implemented.

### Comparing conditions

`L_f - L_fl` at equal steps is the loop's paired loss reduction at matched
data. `L_l - L_fl` measures adding feedback to the loop. There is no
no-recurrence control, so these three conditions do not identify a complete
two-factor interaction. Keep evaluation mode, initialization seed, source,
rows, and schedule aligned.

Report parameters, predicted tokens, pass-tokens, and cell-tokens with every
comparison. Equal data does not imply equal compute: repeated cells and
feedback passes add work, and each pass also executes MTP and its vocabulary
loss. Measured device time captures those costs and routing overhead.

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

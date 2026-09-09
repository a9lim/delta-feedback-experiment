# Growing recipe: conditions, data, schedule, evaluation

This page describes how a specimen is grown: the condition letters and how
conditions pair, the small geometry, the token stream, the optimizer batch, the feedback
passes, the learning-rate schedule, and the numbers we look at afterwards.
[Architecture](architecture.md) owns the equations;
[interpretability](interpretability.md) owns the analysis scripts;
[scaling](scaling.md) keeps the longer and larger recipes.

## Conditions

A condition is a string of letters from `arfl`, each one change from the
plain twelve-layer gated NoPE GQA decoder. `parse_condition` accepts the
letters in any order and returns them in that order; the empty string is the
plain decoder, and `ModelConfig.condition` renders a configuration's letters
back.

| Letter | Change | `ModelConfig` flag |
|---|---|---|
| `a` | Kimi Delta Attention: PKDA in three of every four attention layers, `[PKDA, PKDA, PKDA, gated global GQA] x 3` | `hybrid` |
| `r` | MHDB: transient grouped reads of the seed and block deltas before every sublayer; with `f`, routed enrichment of the payload | `block_routing` |
| `f` | FBT: token-gated latent payload transfer between token columns | `feedback` |
| `l` | Huginn loop: the cells between the first and last become one tied core, iterated a drawn number of times per column ([depth-architecture.md](depth-architecture.md)) | `loop` |

Every subset of `arfl` builds. At the screen geometry:

| Condition | Parameters | Active non-embedding |
|---|---:|---:|
| `""` (plain) | 237,032,448 | 120,345,600 |
| `r` | 237,087,744 | 120,400,896 |
| `f` | 238,214,400 | 121,527,552 |
| `rf` | 238,272,000 | 121,585,152 |
| `a` | 256,275,240 | 139,588,392 |
| `ar` | 256,330,536 | 139,643,688 |
| `af` | 257,457,192 | 140,770,344 |
| `arf` | 257,514,792 | 140,827,944 |
| `arfl` | 257,514,792 | 140,827,944 |

`arfl` is the full built stack and `arf` the flat column. `a` alone already
has recurrent mixer memory. `r` without `f` seeds from the plain embedding and
emits no payload; `f` without `r` emits `payload_norm(h_top)`; `r` with `f`
uses the fused seed as a routing source and enriches the payload with routed
sources. `l` adds no parameters: it holds the first and last cells and runs
the cells between them as one tied core, so every looped condition has its
unlooped condition's counts. The specimens so far are `ar` and `arf` runs.

Pairing is built in. Two conditions on the same trunk letter initialize every
parameter they share byte-identically for a given seed: the trunk from the
common stream, the attention gates from their own deterministic stream, and
`f`'s fusion matrices from `f`'s own, neither of which advances the common
one; routers initialize to zero. The plain trunk and the
`a` trunk consume the common stream differently, so parameters do not pair
across `a`. Two conditions trained with the same seed and data seed therefore
see the same rows in the same order with the same keyed feedback draws, and
can be compared token by token. A looped condition pairs with its unlooped
one the same way: the iteration draw is its own keyed sub-stream, so `arfl`
and `arf` share every pass, prefix, and jitter draw.

`l` needs whole cells and at least three of them. At one core iteration
`arfl` is exactly `arf`: the same layers, parameters, banks, routes, losses,
and gradients.

## Geometry

The trunk is twelve gated NoPE GQA layers; under `a` it is:

```text
[PKDA, PKDA, PKDA, gated global GQA] x 3
```

| Field | Screen value |
|---|---:|
| Vocabulary | 151,936, Qwen3 tokenizer |
| Width | 768 |
| Layers / four-layer cells | 12 / 3 |
| SwiGLU intermediate width | 3,328 |
| Context / predictions per row | 1,024 |
| Explicit position encoding | none |
| GQA query / KV heads / head width | 8 / 4 / 96 |
| PKDA Q/K/V heads / head width | 10 / 128 |
| PKDA Q/K/V projection width | 1,280 |
| PKDA convolution width | 4 |
| Routing groups | 4, the KV-head count |
| RMSNorm epsilon | `1e-6` everywhere, including PKDA output |

Under `l` the first of those cells is the prelude, the last the coda, and the
one between them the tied core, run `r` times per column.

Relative to the larger geometry in `architecture.md`, the screen halves the
residual, SwiGLU, and PKDA projection widths (`768 / 3,328 / 1,280` against
`1,536 / 6,656 / 2,560`), so the 10-by-128 PKDA geometry keeps the Kimi `5/3`
recurrent-projection ratio. Every dense attention layer, the fourth of each
cell under `a` and all twelve without it, is bias-free causal NoPE GQA with
packed QKV, per-head Q/K RMSNorm, and a sigmoid output gate. MHDB has four
groups because its groups follow the global KV-head count.

## Data

The corpus is `HuggingFaceFW/fineweb-edu`, configuration `sample-100BT`, in
its canonical streaming order at dataset commit
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, tokenized with `Qwen/Qwen3-0.6B`
at commit `c1899de289a04d12100db370d81485cdf75e47ca`. The `data-build` extra
pins the four packages that compile the stream and `meta.json` records their
realized versions. Every non-empty document is followed by EOS.

`delta tokenize` materializes the stream as a local contiguous uint32 store so
that step-addressed rows, validation, and resume never depend on network or
iterator state. The held-out validation slice is the stream head; training
follows in contiguous shards. One row is a non-overlapping `seq_len + 1`
window, so 1,025 stored tokens give 1,024 predictions. Rows may cross
document boundaries. Step `n` addresses:

```text
first_row(n) = (n - 1) * batch_rows
```

Feedback pass counts are keyed by data seed and step; prefix lengths and
jitter are additionally keyed by the microbatch's first global row. None of
them depend on ambient RNG state, so a resume returns to the same row and the
same draws.

The canonical stream target is 57B stored tokens including the held-out
prefix, enough for the 400x Prime schedule in [scaling.md](scaling.md) with
about 584M tokens of headroom.

## Training

### Optimizer batch

Every optimizer update sees 327,680 predicted tokens:

| Surface | Realization |
|---|---|
| Jobe screen | 320 rows x 1,024 predictions; 80 four-row microbatches |
| Prime screen | 8 ranks x 4 rows x 10 accumulation microsteps |
| Larger geometry | 8 ranks x 1 row x 8,192 predictions x 5 accumulation microsteps |

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

Both parameter groups share one warmup-stable-cooldown multiplier. Warmup
occupies `round(warmup_frac * steps)` updates and rises linearly; cooldown
occupies `round(cooldown_frac * steps)` updates with multiplier `1 - sqrt(u)`
for local progress `u`, reaching zero at the last step. Defaults are 0.02 and
0.20. The default 10,745-step Jobe schedule:

| Phase | Steps | Passes with `f` |
|---|---:|---|
| Warmup | 1–215 | one |
| Stable heat | 216–8,596 | one through 8,059, then two or three |
| Cooldown | 8,597–10,745 | two or three |

The feedback boundary is `round(feedback_start * steps)`, independent of the
learning-rate phases, and defaults to three quarters of the schedule. After
it every step draws three passes with probability `three_pass = 0.12` and two
otherwise, which targets a 75% / 22% / 3% pass mixture over the run and 1.28
expected pass-tokens per predicted token. Conditions without `f` use one pass
throughout.

The run is 3,520,921,600 predicted tokens: 25.0–25.2 per active non-embedding
parameter with `a`, 30.7–31.1 without.

### Knobs

| Flag | Default | What it changes |
|---|---:|---|
| `--condition` | `""` | letters from `arfl` in any order; empty is the plain decoder |
| `--steps`, `--batch-rows` | 10,745, 320 | schedule length and optimizer batch; the 25x recipe |
| `--seed`, `--data-seed` | | initialization pairing and the keyed data/feedback streams |
| `--lr-normuonh`, `--lr-nadam` | `6e-3`, `3e-4` | the two group learning rates |
| `--feedback-start` | 0.75 | fraction of the schedule before the feedback boundary; 0 trains fused from step 0 |
| `--three-pass` | 0.12 | probability of three passes after the boundary; 1 makes every feedback step three-pass |
| `--loop-iterations`, `--loop-max-iterations` | 4, 8 | `l`: mean and cap of the per-step core iteration draw; the mean is also the fixed evaluation and decode count |
| `--jitter` | 0.02 | payload jitter half-width |
| `--warmup-frac`, `--cooldown-frac` | 0.02, 0.20 | schedule shape |
| `--max-steps` | | caps this invocation without changing the schedule |
| `--resume` | | continues a tag from its latest v25 snapshot |

The specimens so far used the default recipe (`ar`) and
`--feedback-start 0 --three-pass 1` (`arf`); see [findings.md](findings.md).

### Checkpoints and queue

Snapshots use checkpoint contract v25, and only v25 resumes; v16 through v24
stay readable for evaluation and forks. Every snapshot records its condition
as letters. A snapshot holds the model, both
optimizer states, the fixed NorMuonH radii, the state-defining arguments, the
cumulative step, and Python/Torch/CUDA RNG state. A resume inherits every
state-defining field and rejects explicit conflicts; runtime paths, device,
evaluation cadence, snapshot cadence, and evaluation-row count may change.

Each run keeps the latest two snapshots plus protected ones at the cooldown
boundary, the feedback boundary (the last one-pass state), and the end of the
run, so cooldown and feedback variants can fork from the exact pre-boundary
state. The queue stores arguments, not Git state; a source change never stops
an active child, and the worker refreshes before the next job.

## Evaluation

### Language-model numbers

Every evaluation point reports pass-1 held-out cross-entropy as `val`.
Conditions with `f` also report `val_fused`, a second pass with plain-prefix
length 1. Conditions with `l` evaluate at the fixed count `r = r_mean`.
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
beyond next-token prediction on FineWeb-Edu. Several of these are candidate
moves in [findings.md](findings.md).

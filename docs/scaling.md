# Geometries and budgets: the screen, the bridge, and the flagship

Three geometries of the same architecture. The **screen** is the small
organism we grow on Jobe: width 768, three cells, context 1,024, the
specimens in [findings.md](findings.md). The **bridge** is the rung between:
width 1,152, four cells, context 4,096, about three times the screen's
active parameters and the geometric midpoint of the ladder, the largest
column the single-process trainer can grow on Jobe today and the one meant
for publication as an organism. The **flagship** is the 1B column at width
1,536, six cells, context 8,192, worked out here and not scheduled: it needs
the distributed path, which is not implemented, and Prime time is real money,
so a9 decides when to spend it. Every condition builds at every geometry,
and the loop at each is the flat column with its middle cells tied.

"Screen", "25x", and "400x" name recipes by predicted tokens per active
non-embedding parameter. Every feedback pass and every core iteration adds
compute on top of that count, so results carry pass-tokens and cell-tokens
beside predicted tokens; equal steps are matched data, not matched compute.

## The three geometries

| Field | Screen | Bridge | Flagship |
|---|---:|---:|---:|
| Vocabulary, Qwen3 tokenizer | 151,936 | 151,936 | 151,936 |
| Residual width `D` | 768 | 1,152 | 1,536 |
| Layers / four-layer cells | 12 / 3 | 16 / 4 | 24 / 6 |
| SwiGLU intermediate width | 3,328 | 4,992 | 6,656 |
| Context, predictions per row | 1,024 | 4,096 | 8,192 |
| Global query / KV heads, head width 96 | 8 / 4 | 12 / 6 | 16 / 8 |
| Routing groups, the KV-head count | 4 | 6 | 8 |
| PKDA heads, head width 128 | 10 | 15 | 20 |
| PKDA Q/K/V projection width | 1,280 | 1,920 | 2,560 |
| Active non-embedding parameters | 140,827,944 | 416,634,192 | 1,102,046,496 |

The widths step by half the screen's, so every geometry keeps the Kimi `5/3`
recurrent-projection ratio, the `13/3` SwiGLU ratio, and the head widths, and
MHDB's groups follow the global KV-head count at each. Convolution width,
RMSNorm epsilon, and every other constant are the architecture's and do not
vary. Every planned run is addressed by `--condition`, `--scale`, and
`--tokens-per-param`: `--scale screen|bridge|flagship` fills the column, the
row length, and the batch rows of a geometry, any trunk or recipe flag typed
alongside overrides its field, and `--tokens-per-param` derives the schedule
length from the flat full stack's active count at that scale, rounded up to
whole steps, so every condition at a scale shares one schedule. 25 is the
screen recipe and 400 the Prime recipes; `--steps` types a length instead.
Warmup is 2% of the 25x length at the scale whatever the ratio, 215, 636,
and 1,682 steps; cooldown is 20% of the run.

Under `a` the trunk is `[PKDA, PKDA, PKDA, gated global GQA] x C`. Under
`l` the first cell is the prelude, the last the coda, and the cells between
them the tied core:

| | Screen | Bridge | Flagship |
|---|---|---|---|
| Prelude | layers 0–3 | layers 0–3 | layers 0–3 |
| Core | layers 4–7, one cell | layers 4–11, two cells | layers 4–19, four cells |
| Coda | layers 8–11 | layers 12–15 | layers 20–23 |
| Compute depth per pass | `4 + 4r + 4` layers, `2 + r` cells | `4 + 8r + 4` layers, `2 + 2r` cells | `4 + 16r + 4` layers, `2 + 4r` cells |

The draw is `r_mean = 4`, `r_max = 8` at every geometry, so `E[r] = 3.88`,
median 4, `P(r = 1) = 10.3%`, and `P(r = 8) = 8.0%` throughout; only the
cells per iteration change.

## The screen

### Parameters

Every subset of `arfl` builds. `l` adds no parameters, so every looped
condition has its unlooped condition's counts:

| Condition | Parameters | Active non-embedding |
|---|---:|---:|
| `""` (plain) | 237,032,448 | 120,345,600 |
| `r` | 237,087,744 | 120,400,896 |
| `f` | 238,214,400 | 121,527,552 |
| `rf` | 238,272,000 | 121,585,152 |
| `a` | 256,275,240 | 139,588,392 |
| `ar` | 256,330,536 | 139,643,688 |
| `af` | 257,457,192 | 140,770,344 |
| `arf`, `arfl` | 257,514,792 | 140,827,944 |

The full stack by component:

| Component | Parameters |
|---|---:|
| Tied embedding and readout | 116,686,848 |
| 12 SwiGLU channel mixers | 92,012,544 |
| 9 PKDA mixers | 40,478,184 |
| 3 gated global GQA mixers, including Q/K norms and gates | 7,078,464 |
| FBT fusion, `f` | 1,179,648 |
| Trunk, entry, and payload norms | 21,504 |
| 24 within-column routers and one payload router | 57,600 |
| **Total** | **257,514,792** |
| **Active non-embedding** | **140,827,944** |

Relative to unpreconditioned KDA, PKDA's two width-to-head projections and
three learned per-head vectors add 15,390 parameters per PKDA layer, or
138,510 in all; the three GGQA gates are 1,769,472 of the GGQA count.

### Decode cache

At 1,024 positions one cell's token-mixer cache per sequence is 3.456 MiB:
three FP32 PKDA matrix and diagonal states plus BF16 convolution histories,
1.956 MiB, and one BF16 GQA K/V cache, 1.500 MiB. The flat column's three
cells hold 10.37 MiB. Every decode mode under `l` holds `2 + r` cells at the
request's `r`:

| Fixed `r` | Cells cached | Cache |
|---|---:|---:|
| 1 | 3 | 10.4 MiB |
| 4 | 6 | 20.7 MiB |
| 8 | 10 | 34.6 MiB |

Payload, logits, allocator overhead, and serving metadata are additional. This
is decode-state accounting, not training memory; Jobe's captured training graph
pool reserves about 23 GiB.

### Budget

The default recipe is 10,745 steps of 327,680 predicted tokens, or
3,520,921,600 in all: 25.0–25.2 per active non-embedding parameter with `a`,
30.7–31.1 without. Its schedule shape, pass mixture, and knobs are in
[design.md](design.md#training); the pass mixture targets 1.28 expected
pass-tokens per predicted token with `f`.

A finished run extends to a longer ratio with `--continue`, which restores
the last snapshot the longer schedule reproduces and pays only for the new
heat and cooldown; warmup is fixed per scale, so the continuation of a run
at or above 25x is the longer run exactly: at the screen, 25x to 50x restores step 8,059 of
10,745 and trains 13,430 of the 50x recipe's 21,489 steps, and 50x to
100x restores step 16,117 and trains 26,861 of 42,978.

Under `l` at the default draw, `r_mean = 4` and `r_max = 8`:

| Statistic | Value |
|---|---:|
| `E[r]` | 3.88 |
| median `r` | 4 |
| `P(r = 1)` | 10.3% |
| `P(r = 8)` | 8.0% |
| expected layers per pass | 23.5 |
| expected cells per pass | 5.88 |

So `arfl` spends 5.88 expected cell-tokens per pass-token against `arf`'s
three: the loop about doubles the recipe's compute at matched data.

### Cost of the loop at the screen

Training memory and step time are measured, not derived
([runtime qualification](runtime-qualification.md#the-loop), record
`data/summary/loop-stage-2026-09-09.json`). Capturing the twenty-four train
graphs and the evaluation graph took 67 s and peaked at 15.21 GiB allocated
and 23.02 GiB reserved, against 13.85 and 22.99 GiB for the flat column's four
graphs. Eager three-pass microbatches peak at 12.74 GiB raw at `r = 1` and
4.02 GiB checkpointed at `r = 8`; the one-pass microbatch at `r = 8` peaks
at 13.11 GiB raw. The trainer's activation policy counts executed layers,
`8 + 4r` per pass at the screen, against a measured raw budget of forty
layer-passes (ten cells: one pass through `r = 8`, two passes through
`r = 3`), and checkpoints every mode deeper than that. Replay per four-row
microbatch, in milliseconds:

| Passes \ `r` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 53.5 | 67.9 | 82.4 | 96.9 | 111.3 | 125.8 | 140.1 | 154.6 |
| 2 | 107.8 | 136.7 | 165.4 | 248.4 | 286.3 | 324.3 | 362.2 | 400.2 |
| 3 | 162.2 | 259.3 | 316.5 | 373.5 | 430.6 | 487.6 | 544.8 | 601.6 |

One pass costs one cell, 14.4 ms, per iteration all the way to the cap. The
jumps at `r = 4` for two passes and `r = 2` for three are where checkpointing
switches on and recomputes each block once. At the schedule's realized draws
a step's replay averages 7.6, 18.1, and 29.1 s at one, two, and three
passes. With one second of optimizer, clipping, shadow, and staging overhead
per step, which projects the flat three-pass step to 41.7 h against the
42.3 h measured, the schedule projects to:

| Recipe | `arfl` | `arf` |
|---|---:|---:|
| default mixture | 34.8 h | ~19 h |
| two passes on every step | 57.0 h | ~28 h |
| three passes on every step | 89.8 h | 42.3 h |

## The bridge

The bridge is the flat column at width 1,152, four cells, and context 4,096:
the screen's architecture at one and a half times the width and one more
cell, on rows four times longer. Under `l` its core is two cells, so a bridge
`arfl` is the first specimen whose core has more than one cell, and its
`r = 1` column is the bridge `arf`. It is grown with the single-process
trainer on Jobe at 25x, and it is the middle rung of the 400x ladder on
Prime between the screen pair and the flagship. Its rows are 4,096
predictions where the screen's are 1,024 and the flagship's 8,192, so the
ladder's rungs differ in context as well as size.

```bash
delta queue bridge-delta-arf-s1 --condition arf --scale bridge --seed 1 --data-seed 0 \
  --data-dir /data/delta/tokens
```

`--scale bridge` is the column, row length, and batch above; the 25x schedule
is derived, and `--tokens-per-param 400` gives the Prime schedule.

### Parameters

| Component | Parameters |
|---|---:|
| Tied embedding and readout | 175,030,272 |
| 16 SwiGLU channel mixers | 276,037,632 |
| 12 PKDA mixers | 116,552,400 |
| 4 gated global GQA mixers, including Q/K norms and gates | 21,234,432 |
| FBT fusion, `f` | 2,654,208 |
| Trunk, entry, and payload norms | 41,472 |
| 32 within-column routers and one payload router | 114,048 |
| **Total** | **591,664,464** |
| **Active non-embedding** | **416,634,192** |

Relative to unpreconditioned KDA, PKDA's two width-to-head projections and
three learned per-head vectors add 34,605 parameters per PKDA layer, or
415,260 in all; the four GGQA gates are 5,308,416 of the GGQA count. `l`
adds nothing, as everywhere.

### Decode cache

At a full 4,096-token prompt, one sequence's token-mixer cache is:

| Cache | Size |
|---|---:|
| 12 FP32 PKDA matrix states `[15,128,128]` | 11.250 MiB |
| 12 FP32 PKDA diagonal states `[15,128]` | 0.088 MiB |
| 12 BF16 Q/K/V convolution histories `[1920,3]` | 0.396 MiB |
| 4 BF16 global-GQA KV caches `[4096,6,96]` | 36.000 MiB |
| **Token-mixer total** | **47.733 MiB** |

One cell is 11.9 MiB. Every decode mode under `l` holds `2 + 2r` cells at
the request's `r`: 47.7 MiB at `r = 1`, 119.3 MiB at `r = 4`, 214.8 MiB at
the cap. Payload, logits, allocator overhead, and serving metadata are
additional.

### Budgets

Both recipes keep the family's 327,680 predictions per optimizer step, which
at 4,096 predictions per row is 80 rows: on Jobe 80 one-row microbatches, on
Prime 8 ranks x microbatch 1 x accumulation 10. `--scale bridge` with
`--tokens-per-param 25` or `400` names them.

| Quantity | 25x, Jobe | 400x, Prime |
|---|---:|---:|
| Optimizer steps | 31,787 | 508,587 |
| Aligned budget | 10,415,964,160 predicted tokens | 166,653,788,160 predicted tokens |
| Predicted tokens per active parameter | 25.0003 | 400.0003 |
| Warmup | steps 1–636 | steps 1–636 |
| Stable heat | steps 637–25,430 | steps 637–406,870 |
| Cooldown | steps 25,431–31,787 | steps 406,871–508,587 |
| Feedback boundary | after step 23,840 | after step 381,440 |
| Expected pass-tokens | about 13.33B | about 213.3B |
| Expected cell-tokens, `arf` / `arfl` | about 53.3B / 130.1B | about 853B / 2.08T |

The 25x run is the screen recipe at the bridge's size: the same schedule
shape, pass mixture, and knobs, 2.96x the screen's tokens on a column that
costs about 2.3x per token, so about 6.8x the screen's arithmetic for `arf`
and about 9x for `arfl`. Scaling the screen's measured hours by those ratios
projects roughly six days for the default-mixture `arf` and two weeks for
`arfl`, before checkpoint recompute; the memory and step time are not yet
staged, and `scripts/loop_memory_stage.py` at this geometry is the first
thing to run. The parameters and optimizer state alone are about 2.3x the
screen's, so the one-row microbatch is expected to train checkpointed.

The 400x run is the flagship recipe at the bridge's size: about 0.17x the
flagship's arithmetic flat and 0.34x looped, so the pair is the cheap first
exercise of the distributed path and the middle rung of the ladder, screen
pair at 56B tokens, bridge at 167B, flagship at 441B, on which a letter's
gain can be seen to grow or shrink with scale before the flagship is rented.

## Longer training at the same size

A fresh `{a, arf}` pair on one 8xH100-80GB Prime node, both conditions seed 1,
data seed 0, global row zero, `--tokens-per-param 400` at the screen, which
is 171,910 steps and 56,331,468,800 predicted tokens each, nothing loaded
from Jobe.

| Condition | Predicted tokens | Active-token ratio |
|---|---:|---:|
| `a` | 56,331,468,800 | 403.55 |
| `arf` | 56,331,468,800 | 400.00 |

| Phase | Steps | Passes, `arf` |
|---|---:|---|
| Warmup | 1–215 | one |
| Stable heat | 216–137,528 | one through 128,932, then two or three |
| Cooldown | 137,529–171,910 | two or three |

The pair is 112.66B predicted tokens and about 128.44B expected pass-tokens.
It compares the complete `arf` package against its shared hybrid baseline.

Data staging on Prime: after choosing a provider and location and before
provisioning the H100 node, create a provider-local persistent disk sized for
the compiled stream and run artifacts, attach it to a cheap compatible
staging instance, install the `data-build` extra, and run `delta tokenize` from
the pinned Hugging Face dataset and tokenizer into the mounted path. The
resulting `meta.json` and byte checksums match Jobe's prefix; then detach the
disk and attach it to the H100 node. No live Hugging Face reads during
training and no re-hosted copy of the store.

The single-process trainer expresses this schedule already. Distributed
execution needs exact row sharding, DDP accumulation with synchronized global
clipping, rank-safe telemetry and snapshots, persistent artifact staging,
cross-rank optimizer parity, restart tests, persistent-disk read checks, and
H100 memory and throughput measurement. Hopper also needs its own `delta probe`
run and FLA kernel constants swept; see
[runtime-qualification.md](runtime-qualification.md).

## The flagship

The flagship is `arf` at six cells and width 1,536, with a budget of 400
predicted tokens per active non-embedding parameter. The execution target is
replicated DDP over eight H100 80GB GPUs with BF16 autocast, FP32 parameters
and optimizer state, no tensor, pipeline, context, or parameter sharding,
every block activation-checkpointed on every pass, and payload and source-bank
graphs kept differentiable. Building it means reproducing the parameter and
compute accounting below, PKDA parity across the portable, chunk, and
recurrent paths, cache continuation, gated-GQA parity, MHDB source
identities, optimizer partition and radius invariants, distributed row
assignment and clipping, checkpoint portability, restart behavior, memory, and
throughput at that geometry.

### Parameters

| Component | Parameters |
|---|---:|
| Tied embedding and readout | 233,373,696 |
| 24 SwiGLU channel mixers | 736,100,352 |
| 18 PKDA mixers | 304,297,632 |
| 6 gated global GQA mixers, including Q/K norms and gates | 56,624,256 |
| FBT fusion, `f` | 4,718,592 |
| Trunk, entry, and payload norms | 79,872 |
| 48 within-column routers and one payload router | 225,792 |
| **Total** | **1,335,420,192** |
| **Active non-embedding** | **1,102,046,496** |

The active non-embedding count is the denominator for the data budget.
Relative to unpreconditioned KDA, PKDA's two width-to-head projections and
three learned per-head vectors add 61,500 parameters per PKDA layer, or
1,107,000 in all; the six GGQA gates are 14,155,776 of the GGQA count.

### Decode cache

At a full 8,192-token prompt, one sequence's token-mixer cache is:

| Cache | Size |
|---|---:|
| 18 FP32 PKDA matrix states `[20,128,128]` | 22.500 MiB |
| 18 FP32 PKDA diagonal states `[20,128]` | 0.176 MiB |
| 18 BF16 Q/K/V convolution histories `[2560,3]` | 0.791 MiB |
| 6 BF16 global-GQA KV caches `[8192,8,96]` | 144.000 MiB |
| **Token-mixer total** | **167.467 MiB** |

One cell is 27.9 MiB. This excludes allocator overhead, the width-1,536
payload, logits, and serving metadata. An implementation at this geometry
reproduces these parameter counts, state shapes, and cache continuation
semantics.

### Budget

| Quantity | Value |
|---|---:|
| Sequence length | 8,192 predictions |
| Global batch | 40 sequences = 327,680 predictions |
| Distributed batch | 8 ranks x microbatch 1 x accumulation 5 |
| Unrounded target | 440,818,598,400 predicted tokens |
| Optimizer steps | 1,345,272 |
| Aligned budget | 440,818,728,960 predicted tokens |
| Warmup | steps 1–1,682 |
| Stable heat | steps 1,683–1,076,218 |
| Cooldown | steps 1,076,219–1,345,272 |
| Feedback boundary | after step 1,008,954 |
| Whole-run pass mixture | expected 75% / 22% / 3% |
| Expected compute | about 564.25B pass-tokens, 3.39T cell-tokens |

The aligned schedule overshoots the 400x target by 130,560 tokens, less than
one batch, at 400.000118 predicted tokens per active parameter.

## The flagship loop

One `arfl` run at the flagship geometry: the six cells above with the middle
four tied as the core, so the loop and the flat specimen have the same
1,335,420,192 parameters, the same 1,102,046,496 active non-embedding, and
the same 400x schedule, and at `r = 1` the loop is the flat specimen. The
draw is the screen's, `r_mean = 4` and `r_max = 8`, with the screen's
statistics: `E[r] = 3.88`, median 4, `P(r = 1) = 10.3%`, `P(r = 8) = 8.0%`.
Compute depth per pass is `4 + 16r + 4` layers, mean 70.1 and cap 136, and
`2 + 4r` cells, 17.5 expected. Metrics use fixed `r = 4`; the sweep runs to 8.

| Quantity | Value |
|---|---:|
| Global batch | 327,680 predictions |
| Optimizer steps | 1,345,272 |
| Aligned budget | 440,818,728,960 predicted tokens |
| Warmup / stable heat / cooldown / feedback boundary | as the flagship |
| Expected pass-tokens | about 564.25B |
| Expected cell-tokens | about 9.88T |

The flagship spends about 3.39T cell-tokens, so the loop spends about 2.9x
the cell-tokens and, with the head executed once per pass, about 2.6x the
arithmetic; that is accounting, not measured device time. The two runs pair
as the screen's do: byte-identical initialization, the same stream, row
order, schedule, and pass-count, prefix, and jitter draws, with the `r` draw
the loop's only extra randomness. Every decode mode holds `2 + 4r` cells at
the request's `r`, 502 MiB at `r = 4` and 949 MiB at the cap. The shared
fused decode cache belongs to `L`.

## When one of these would be worth it

When a question about the organism cannot be answered at the screen: a
behavior that never appears at 25x, a channel that stays unused, or a
mechanism whose dependence on training duration is the question. Before
renting, write down what the extra training or size should show, including
what a null would mean, and have the distributed path and the target hardware
qualified.

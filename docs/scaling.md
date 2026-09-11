# Geometries and budgets: the screen, the bridge, and the flagship

Three geometries of the same architecture, every one configured for 4,096-token
rows in batches of 128 rows, `2^19` predictions per step. The **screen** is
the small organism we grow on Jobe: width 768 and three cells. There are no
current trained specimens; [findings.md](findings.md) states the evidence
boundary. The **bridge** is the rung between: width 1,152,
four cells, about three times the screen's active parameters and the
geometric midpoint of the ladder. It is the largest column targeted for
single-process training on Jobe and the one meant for publication as an
organism; its training memory and throughput remain unqualified. The **flagship** is the 1B column at width 1,536, six cells, worked
out here and not scheduled: it needs the distributed path, which is not
implemented, and Prime time is real money, so a9 decides when to spend it.
Every condition builds at every geometry, and the loop at each is the flat
column with its middle cells tied.

"Screen", "25x", and "400x" name recipes by predicted tokens per active
non-embedding parameter. Every feedback pass and every core iteration adds
compute on top of that count, so results carry pass-tokens and cell-tokens
beside predicted tokens; equal steps are matched data, not matched compute.

## The three geometries

| Field | Screen | Bridge | Flagship |
|---|---:|---:|---:|
| Padded model vocabulary, GPT-NeoX + ChatML | 50,304 | 50,304 | 50,304 |
| Residual width `D` | 768 | 1,152 | 1,536 |
| Layers / four-layer cells | 12 / 3 | 16 / 4 | 24 / 6 |
| SwiGLU intermediate width | 3,328 | 4,992 | 6,656 |
| Context, predictions per row | 4,096 | 4,096 | 4,096 |
| Rows per step, one-row microbatches | 128 | 128 | 128 |
| Global query / KV heads, head width 96 | 8 / 4 | 12 / 6 | 16 / 8 |
| Routing groups, the KV-head count | 4 | 6 | 8 |
| PKDA heads, head width 128 | 10 | 15 | 20 |
| PKDA Q/K/V projection width | 1,280 | 1,920 | 2,560 |
| Active non-embedding parameters | 140,827,944 | 416,634,192 | 1,102,046,496 |
| muP width ratio `1536 / D`: fan-in-`D` NAdam rate multiplier and readout multiplier | 2 | 4/3 | 1 |
| Fan-in-`D` NAdam initialization standard deviation | 0.02828 | 0.02309 | 0.02 |

The widths step by half the screen's, so every geometry keeps the Kimi `5/3`
recurrent-projection ratio, the `13/3` SwiGLU ratio, and the head widths, and
MHDB's groups follow the global KV-head count at each. Convolution width,
RMSNorm epsilon, and every other constant are the architecture's and do not
vary. The flagship is the muP reference width: the NAdam matrices with fan-in
`D` and the tied readout carry the ratio `1536 / D`, so `--lr-nadam` names
the flagship's rate and the screen runs its gates and controls at twice it
under a doubled readout ([architecture.md](architecture.md#nadam-parameters));
their initialization standard deviation is
`BASE_NORMAL_INIT_STD * sqrt(1536 / D)`, preserving initial gate/control
variance across widths. `BASE_NORMAL_INIT_STD = 0.02` in `model.py` also sets
the width-independent embedding and PKDA head-expansion scale.
NorMuonH's relative step needs no rule. Every planned run is addressed by `--condition`, `--scale`, and
`--tokens-per-param`: `--scale screen|bridge|flagship` fills the column, the
row length, and the batch rows of a geometry, any trunk or recipe flag typed
alongside overrides its field, and `--tokens-per-param` derives the schedule
length from the flat full stack's active count at that scale, rounded up to
whole steps, so every condition at a scale shares one schedule. 25 is the
screen recipe and 400 the Prime recipes; `--steps` types a length instead.
Warmup is 2% of the 25x length at the scale whatever the ratio, 134, 397,
and 1,051 steps; cooldown is 20% of the run.

The context and the batch are uniform across the ladder on purpose. Rows of
4,096 give the recurrent state and dense read room for multiple web
documents at every scale. Screen/Jobe use the published shuffle of the
100B DCLM subset; stores exceeding that subset use shuffled full DCLM.
The source change prevents token-by-token comparisons across those stores. The batch of `2^19` follows the
standard result that the efficient batch grows with the token budget rather
than the model: the DeepSeek fit `B = 0.29 C^0.33` puts the screen's 25x
optimum near 0.32M tokens, the bridge's near 0.66M, and the flagship's 400x
near 3M, so `2^19` sits inside the flat basin at the screen and under the
optimum everywhere larger, where token-efficiency is unharmed and the
sequential step count stays bounded. FBT trained at 300K and moved to 1.2M
for its 1T-token baseline. A uniform batch keeps equal steps meaning equal
tokens at every rung. Batch is the one knob muP does not transfer, so a rate
sweep belongs at this batch.

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
| `""` (plain) | 158,979,072 | 120,345,600 |
| `r` | 159,034,368 | 120,400,896 |
| `f` | 160,161,024 | 121,527,552 |
| `rf` | 160,218,624 | 121,585,152 |
| `a` | 178,221,864 | 139,588,392 |
| `ar` | 178,277,160 | 139,643,688 |
| `af` | 179,403,816 | 140,770,344 |
| `arf`, `arfl` | 179,461,416 | 140,827,944 |

The full stack by component:

| Component | Parameters |
|---|---:|
| Tied embedding and readout | 38,633,472 |
| 12 SwiGLU channel mixers | 92,012,544 |
| 9 PKDA mixers | 40,478,184 |
| 3 gated global GQA mixers, including Q/K norms and gates | 7,078,464 |
| FBT fusion, `f` | 1,179,648 |
| Trunk, entry, and payload norms | 21,504 |
| 24 within-column routers and one payload router | 57,600 |
| **Total** | **179,461,416** |
| **Active non-embedding** | **140,827,944** |

Relative to unpreconditioned KDA, PKDA's two width-to-head projections and
three learned per-head vectors add 15,390 parameters per PKDA layer, or
138,510 in all; the three GGQA gates are 1,769,472 of the GGQA count.

### Decode cache

At 4,096 positions one cell's token-mixer cache per sequence is 7.956 MiB:
three FP32 PKDA matrix and diagonal states plus BF16 convolution histories,
1.956 MiB, and one BF16 GQA K/V cache, 6.000 MiB. The flat column's three
cells hold 23.87 MiB. Every decode mode under `l` holds `2 + r` cells at the
request's `r`:

| Fixed `r` | Cells cached | Cache |
|---|---:|---:|
| 1 | 3 | 23.9 MiB |
| 4 | 6 | 47.7 MiB |
| 8 | 10 | 79.6 MiB |

Payload, logits, allocator overhead, and serving metadata are additional. This
is decode-state accounting; training memory depends on graph capture and
requires qualification with the current vocabulary.

### Budget

The default recipe is 6,716 steps of 524,288 predicted tokens, or
3,521,118,208 in all: 25.0–25.2 per active non-embedding parameter with `a`,
29.0–29.3 without. Its schedule shape, pass mixture, and knobs are in
[design.md](design.md#training); the pass mixture targets 1.28 expected
pass-tokens per predicted token with `f`.

A finished run extends to a longer ratio with `--continue`, which restores
the last snapshot the longer schedule reproduces and pays only for the new
heat and cooldown; warmup is fixed per scale, so the continuation of a run
at or above 25x is the longer run exactly: at the screen, 25x to 50x restores step 5,037 of
6,716 and trains 8,394 of the 50x recipe's 13,431 steps, and 50x to
100x restores step 10,073 and trains 16,788 of 26,861.

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

### Runtime qualification

The current 50,304-row head has not yet been assigned a measured training
speed. Qualify the screen's full captured graph family, accumulated gradients,
and complete optimizer updates on Jobe before projecting a schedule duration.
The reproducible checks are in
[runtime-qualification.md](runtime-qualification.md).

The activation policy counts executed layers, `8 + 4r` per pass, against a
raw budget of forty layer-passes: one pass through `r = 8`, two through
`r = 3`, and three at `r = 1`. In deeper modes, each cell retains the
activations of its final compiled block, including every tied-core iteration.
Earlier blocks are checkpointed outside compilation. Every core iteration and
feedback pass remains differentiable. This policy does not establish peak
memory or execution time for the new head.

## The bridge

The bridge is the flat column at width 1,152 and four cells: the screen's
architecture at one and a half times the width and one more cell, on the
same rows and batch. Under `l` its core is two cells, so a bridge `arfl` is
the first specimen whose core has more than one cell, and its `r = 1` column
is the bridge `arf`. It targets single-process training on Jobe at 25x,
subject to memory qualification, and is the middle rung of the 400x ladder
on Prime between the screen pair and the flagship.

```bash
delta queue bridge-delta-arf-s1 --condition arf --scale bridge --seed 1 --data-seed 0 \
  --data-root /data/delta --source dclm-100b
```

`--scale bridge` is the column, row length, and batch above; the 25x schedule
is derived, and `--tokens-per-param 400` gives the Prime schedule.

### Parameters

| Component | Parameters |
|---|---:|
| Tied embedding and readout | 57,950,208 |
| 16 SwiGLU channel mixers | 276,037,632 |
| 12 PKDA mixers | 116,552,400 |
| 4 gated global GQA mixers, including Q/K norms and gates | 21,234,432 |
| FBT fusion, `f` | 2,654,208 |
| Trunk, entry, and payload norms | 41,472 |
| 32 within-column routers and one payload router | 114,048 |
| **Total** | **474,584,400** |
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

Both recipes keep the family's 524,288 predictions per optimizer step, 128
rows of 4,096: on Jobe 128 one-row microbatches, with a planned Prime layout
of 8 ranks x microbatch 1 x accumulation 16. `--scale bridge` with `--tokens-per-param 25`
or `400` names them.

| Quantity | 25x, Jobe | 400x, Prime |
|---|---:|---:|
| Optimizer steps | 19,867 | 317,867 |
| Aligned budget | 10,416,029,696 predicted tokens | 166,653,853,696 predicted tokens |
| Predicted tokens per active parameter | 25.0004 | 400.0002 |
| Warmup | steps 1–397 | steps 1–397 |
| Stable heat | steps 398–15,894 | steps 398–254,294 |
| Cooldown | steps 15,895–19,867 | steps 254,295–317,867 |
| Feedback boundary | after step 14,900 | after step 238,400 |
| Expected pass-tokens | about 13.33B | about 213.3B |
| Expected cell-tokens, `arf` / `arfl` | about 53.3B / 130.1B | about 853B / 2.08T |

The bridge's training memory and throughput remain unqualified on both Jobe
and H100. Staging its full graph family must establish memory before training.
Screen measurements cannot establish bridge speed: the head, cell arithmetic,
activation policy, and kernel efficiency scale differently.

A continuation ladder, 25x to 100x to 200x to 400x, restores each rung's
feedback boundary. At the expected 2.12 passes after that boundary, it costs
25,430, 86,818, 143,836, and 287,670 pass-steps, about 285B pass-tokens in
all. This is about 1.34 times the pass-tokens of a single 400x run and 0.74
times four independent runs; the loop's additional cell-tokens follow its
`2 + 2r` executed cells per pass. Convert this accounting to hardware time
only after measuring the current execution path.

## Longer training at the same size

A fresh `{a, arf}` pair on one 8xH100-80GB Prime node, both conditions seed 1,
data seed 0, global row zero, `--tokens-per-param 400` at the screen, which
is 107,444 steps and 56,331,599,872 predicted tokens each, nothing loaded
from Jobe.

| Condition | Predicted tokens | Active-token ratio |
|---|---:|---:|
| `a` | 56,331,599,872 | 403.55 |
| `arf` | 56,331,599,872 | 400.00 |

| Phase | Steps | Passes, `arf` |
|---|---:|---|
| Warmup | 1–134 | one |
| Stable heat | 135–85,955 | one through 80,583, then two or three |
| Cooldown | 85,956–107,444 | two or three |

The pair is 112.66B predicted tokens and about 128.44B expected pass-tokens.
It compares the complete `arf` package against its shared hybrid baseline.

Data staging on Prime: after choosing a provider and location and before
provisioning the H100 node, create a provider-local persistent disk sized for
the compiled stream and run artifacts (the bridge ladder's 167B-token store
is about 690 GB, the screen's 57B about 230 GB). During construction the
output filesystem must also hold the selected token parts under `OUT/parts`;
budget roughly 2.2 times the finished store plus headroom. `--scratch` holds
downloaded parquet files, not token parts. Attach the disk to a compatible
CPU staging instance, install the `data-build` extra, and run
`delta tokenize --source dclm --data-root ROOT --scale bridge --tokens-per-param 400 --scratch SCRATCH --workers N --readers 8`.
The full-source shuffle reads the pinned DCLM release (27,938 parquet files,
7.42 TB) and tokenizes selected documents into `ROOT/dclm`. The default
`dclm-100b` source is suitable for smaller Jobe stores but cannot supply
the bridge's 167B or flagship's 441B target. Measure download and assembly throughput on the chosen provider
before estimating build time; more CPU cores alone do not establish it.
[Data-build performance](data-build-performance.md) gives the profiling
and tuning procedure. The resulting
store should pass `delta verify`; checksum comparisons apply only against
another build of the same source, ordering, pins and seed. The full-source
store does not share a prefix or validation slice with Jobe's `dclm-100b`
store. Then detach the disk and attach it to the H100 node. No live Hugging Face reads during training
and no re-hosted copy of the store.

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
activation checkpointing with each cell's final block retained on every pass,
and payload and source-bank graphs kept differentiable. This retention policy
still needs memory and throughput qualification at the flagship geometry.
Building it means reproducing the parameter and
compute accounting below, PKDA parity across the portable, chunk, and
recurrent paths, cache continuation, gated-GQA parity, MHDB source
identities, optimizer partition and radius invariants, distributed row
assignment and clipping, checkpoint portability, restart behavior, memory, and
throughput at that geometry.

### Parameters

| Component | Parameters |
|---|---:|
| Tied embedding and readout | 77,266,944 |
| 24 SwiGLU channel mixers | 736,100,352 |
| 18 PKDA mixers | 304,297,632 |
| 6 gated global GQA mixers, including Q/K norms and gates | 56,624,256 |
| FBT fusion, `f` | 4,718,592 |
| Trunk, entry, and payload norms | 79,872 |
| 48 within-column routers and one payload router | 225,792 |
| **Total** | **1,179,313,440** |
| **Active non-embedding** | **1,102,046,496** |

The active non-embedding count is the denominator for the data budget.
Relative to unpreconditioned KDA, PKDA's two width-to-head projections and
three learned per-head vectors add 61,500 parameters per PKDA layer, or
1,107,000 in all; the six GGQA gates are 14,155,776 of the GGQA count.

### Decode cache

At a full 4,096-token prompt, one sequence's token-mixer cache is:

| Cache | Size |
|---|---:|
| 18 FP32 PKDA matrix states `[20,128,128]` | 22.500 MiB |
| 18 FP32 PKDA diagonal states `[20,128]` | 0.176 MiB |
| 18 BF16 Q/K/V convolution histories `[2560,3]` | 0.791 MiB |
| 6 BF16 global-GQA KV caches `[4096,8,96]` | 72.000 MiB |
| **Token-mixer total** | **95.467 MiB** |

One cell is 15.9 MiB. This excludes allocator overhead, the width-1,536
payload, logits, and serving metadata. An implementation at this geometry
reproduces these parameter counts, state shapes, and cache continuation
semantics.

### Budget

| Quantity | Value |
|---|---:|
| Sequence length | 4,096 predictions |
| Global batch | 128 sequences = 524,288 predictions |
| Distributed batch | 8 ranks x microbatch 1 x accumulation 16 |
| Unrounded target | 440,818,598,400 predicted tokens |
| Optimizer steps | 840,795 |
| Aligned budget | 440,818,728,960 predicted tokens |
| Warmup | steps 1–1,051 |
| Stable heat | steps 1,052–672,636 |
| Cooldown | steps 672,637–840,795 |
| Feedback boundary | after step 630,596 |
| Whole-run pass mixture | expected 75% / 22% / 3% |
| Expected compute | about 564.25B pass-tokens, 3.39T cell-tokens |

The aligned schedule overshoots the 400x target by 130,560 tokens, less than
one batch, at 400.000118 predicted tokens per active parameter.

## The flagship loop

One `arfl` run at the flagship geometry: the six cells above with the middle
four tied as the core, so the loop and the flat specimen have the same
1,179,313,440 parameters, the same 1,102,046,496 active non-embedding, and
the same 400x schedule, and at `r = 1` the loop is the flat specimen. The
draw is the screen's, `r_mean = 4` and `r_max = 8`, with the screen's
statistics: `E[r] = 3.88`, median 4, `P(r = 1) = 10.3%`, `P(r = 8) = 8.0%`.
Compute depth per pass is `4 + 16r + 4` layers, mean 70.1 and cap 136, and
`2 + 4r` cells, 17.5 expected. Metrics use fixed `r = 4`; the sweep runs to 8.

| Quantity | Value |
|---|---:|
| Global batch | 524,288 predictions |
| Optimizer steps | 840,795 |
| Aligned budget | 440,818,728,960 predicted tokens |
| Warmup / stable heat / cooldown / feedback boundary | as the flagship |
| Expected pass-tokens | about 564.25B |
| Expected cell-tokens | about 9.88T |

The flagship spends about 3.39T cell-tokens, so the loop spends about 2.9x
the cell-tokens. The head executes once per pass, so cell-token ratios alone
do not give total arithmetic or device time. The two runs pair
as the screen's do: byte-identical initialization, the same stream, row
order, schedule, and pass-count, prefix, and jitter draws, with the `r` draw
the loop's only extra randomness. Every decode mode holds `2 + 4r` cells at
the request's `r`, 286 MiB at `r = 4` and 541 MiB at the cap. The shared
fused decode cache belongs to `L`.

## When one of these would be worth it

When a question about the organism cannot be answered at the screen: a
behavior that never appears at 25x, a channel that stays unused, or a
mechanism whose dependence on training duration is the question. Before
renting, write down what the extra training or size should show, including
what a null would mean, and have the distributed path and the target hardware
qualified.

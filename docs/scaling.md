# Geometries and budgets: the screen and the flagship

Two geometries of the same architecture. The **screen** is the small organism
we grow on Jobe: width 768, three cells, the specimens in
[findings.md](findings.md). The **flagship** is the 1B column at width 1,536
and six cells, worked out here and not scheduled: it needs the distributed
path, which is not implemented, and Prime time is real money, so a9 decides
when to spend it. Every condition builds at both geometries, and the loop at
each is the flat column with its middle cells tied.

"Screen", "25x", and "400x" name recipes by predicted tokens per active
non-embedding parameter. Every feedback pass and every core iteration adds
compute on top of that count, so results carry pass-tokens and cell-tokens
beside predicted tokens; equal steps are matched data, not matched compute.

## The two geometries

| Field | Screen | Flagship |
|---|---:|---:|
| Vocabulary, Qwen3 tokenizer | 151,936 | 151,936 |
| Residual width `D` | 768 | 1,536 |
| Layers / four-layer cells | 12 / 3 | 24 / 6 |
| SwiGLU intermediate width | 3,328 | 6,656 |
| Context, predictions per row | 1,024 | 8,192 |
| Global query / KV heads, head width 96 | 8 / 4 | 16 / 8 |
| Routing groups, the KV-head count | 4 | 8 |
| PKDA heads, head width 128 | 10 | 20 |
| PKDA Q/K/V projection width | 1,280 | 2,560 |

The flagship doubles the residual, SwiGLU, and PKDA projection widths and
the head counts, so both keep the Kimi `5/3` recurrent-projection ratio and
the head widths, and MHDB's groups follow the global KV-head count at both.
Convolution width, RMSNorm epsilon, and every other constant are the
architecture's and do not vary.

Under `a` the trunk is `[PKDA, PKDA, PKDA, gated global GQA] x C`. Under
`l` the first cell is the prelude, the last the coda, and the cells between
them the tied core:

| | Screen | Flagship |
|---|---|---|
| Prelude | layers 0–3 | layers 0–3 |
| Core | layers 4–7, one cell | layers 4–19, four cells |
| Coda | layers 8–11 | layers 20–23 |
| Compute depth per pass | `4 + 4r + 4` layers, `2 + r` cells | `4 + 16r + 4` layers, `2 + 4r` cells |

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

## Longer training at the same size

A fresh `{a, arf}` pair on one 8xH100-80GB Prime node, both conditions seed 1,
data seed 0, global row zero, 171,909 steps, 56,331,141,120 predicted tokens
each, nothing loaded from Jobe.

| Condition | Predicted tokens | Active-token ratio |
|---|---:|---:|
| `a` | 56,331,141,120 | 403.55 |
| `arf` | 56,331,141,120 | 400.00 |

| Phase | Steps | Passes, `arf` |
|---|---:|---|
| Warmup | 1–3,438 | one |
| Stable heat | 3,439–137,527 | one through 128,932, then two or three |
| Cooldown | 137,528–171,909 | two or three |

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
| Warmup | steps 1–26,905 |
| Stable heat | steps 26,906–1,076,218 |
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

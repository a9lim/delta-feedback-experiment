# Working journal

Dated working notes: hypotheses, measurements, and readings as they happened.
Newer entries supersede older ones; the distilled picture lives in
[findings.md](findings.md). Git history keeps what gets cut.

## 2026-09-09 — muP for the NAdam group, pinned to the flagship width

a9 asked whether the NAdam parameters should be muP'd, given that NorMuonH is
width-invariant on its own. Working the NAdam group against the ladder: head
width and routing-group width are fixed across the three geometries, so only
four NAdam families have a fan-in that grows with `D`: the tied embedding in
its readout role, the GGQA gate matrices, the FBT token gate, and PKDA's
packed control projection. Everything else NAdam owns (the embedding lookup,
router queries over 192-wide groups, the 128-in PKDA expansions,
convolutions, norms, vectors) already transfers under a constant rate.
NorMuonH's step is `lr · R · Normalize_F(U)` with `R = ||W_0||_F`, so its
update's spectral norm is `lr · sqrt(d_out / min(d_out, d_in))`, the spectral
condition up to a constant the aspect ratio fixes, and every aspect ratio is
fixed across the ladder.

The rule landed: a second NAdam group for the three fan-in-`D` matrix
families at `lr_nadam x 1536 / D`, and a `1536 / D` multiplier on the readout
input, the mup shared-readout convention, with the flagship as the reference
width. a9 chose the flagship rather than the screen: the recipe's rates came
from FBT's 1B fit (1e-2 / 5e-4) and were brought down to 6e-3 / 3e-4 together
when 1e-2 ran hot, so the NAdam rate was never tuned at the screen and reads
best as the flagship's. The screen now runs its gates and controls at 6e-4
under a doubled readout, the bridge at 4e-4 and 4/3.

The readout multiplier changes what a screen snapshot means and the NAdam
state dict grows a group, so the checkpoint contract is v26 and only v26
loads. The three specimens (`screen-delta-ar-s1`, `screen-delta-arf-s1`,
`screen-delta-arf-s1-highLR`) and `screen-delta-arl-s1` are not loadable by
the current code; they were due for a rerun on the keyed stream anyway.

Not done, and worth doing before the bridge: a coord check at widths 384
through 1536, some twenty steps at a tiny batch, logging the per-step RMS of
the gate pre-activations, the PKDA control logits, the routing scores, and
the logits. Flat across widths confirms the rule and measures the NorMuonH
invariance instead of assuming it. Depth is the other axis: the
`1 / sqrt(2L)` branch scale is already the depth-muP residual multiplier,
and depth-muP for normalized optimizers would also scale both rates by
`1 / sqrt(L)`, about 0.7 at 24 layers. Neither is scheduled.

## 2026-09-09 — The loss bumps are the source's crawl order; the stream becomes a keyed document shuffle

a9 noticed that every run's training loss drifts up and down over a few
hundred steps and suspected the data. Detrending each run's pass-1 training
loss (a 21-step mean minus a 401-step trend) over steps 900–4,590, which the
three runs on the old stream share:

| pair | residual correlation at lag 0 |
|---|---:|
| `arf-s1-highLR` x `arf-s1` | 0.996 |
| `arf-s1-highLR` x `arl-s1` | 0.987 |
| `arf-s1` x `arl-s1` | 0.991 |

Different learning rate, different condition, identical bumps at identical
steps: the residual is the data, not the optimizer. Its autocorrelation peaks
at 742 and 1,492 steps, its std is 0.027 nats, its peak-to-trough about 0.1,
and each cycle holds one dip of about 0.05 nats lasting about 175 steps.

The mechanism is in the source files. `sample-100BT` file `000_00001` (727k
documents, 727 row groups of 1,000, 756M GPT-2 tokens) is thirteen
contiguous single-dump runs of about 60M tokens: four crawls (2013-20,
2017-26, 2020-05, 2023-14) in turn, twice, then the next four. One cycle is
245M GPT-2 tokens; 742 steps of 328,000 stored tokens is 243M, and the
Qwen/GPT-2 token ratio measured on that file is 0.993. Rows were read in
source order (`first_row = (n - 1) * batch_rows`, no shuffle at any level),
so every 60M-token stretch of training was one CommonCrawl dump and the dip
is one crawl being easier than its neighbours. The held-out slice was the
head of the stream: the first half of one 2013 crawl run. The in-run eval
read 32 rows of it, 32,768 predictions, a standard error about the size of
the bumps.

Two more things surfaced on the way. The store's EOS was the chat
tokenizer's `<|im_end|>` (151645); `Qwen/Qwen3-0.6B-Base` shares every id and
ends on `<|endoftext|>` (151643), which is the convention an instruct-tuned
descendant would expect. And Jobe's store held 35B train tokens of a
`sample-100BT` that is only about 99B Qwen tokens in total, while the bridge's
400x rung needs 167B: the Prime ladder needed a new source anyway.

What changed, all at once: the stream is now a keyed Feistel shuffle of every
document in `sample-350BT` (346B tokens, of which `sample-100BT` was itself a
sample), under the base tokenizer, with any build a byte-identical prefix of
one stream; `delta tokenize` indexes the source's row groups, selects and
tokenizes the documents whose position falls below `target / tokens-per-doc`
one file at a time, and writes them in stream order with a per-split sidecar
of document starts, universe addresses, and crawls; `delta verify` checks a
store; `delta tokenize --scale S --tokens-per-param R` sizes a store for a
planned run, rounded up to the next billion, and `--continue` extends a
finished store in place onto the bytes a fresh build would write (the
held-out slice is the first documents *up to* its cap so both rules agree);
`datasets` left the `data-build` extra for `pyarrow` and the pins moved to
what Jobe runs; `--eval-rows` defaults to 512; and
`scripts/store_cutover.py` retires a store once the spool stops reading it.
The Jobe rebuild to `/data/delta/tokens-350B` runs beside
`screen-delta-arl-s1`, which stays on the old store until it lands; the
three finished specimens keep their old-order comparability among
themselves and are not comparable token by token with anything trained on
the new stream. The bridge store on Prime is `--target 167e9` of the same
stream.

## 2026-09-09 — The core generalizes to any cell count; the larger loop is the larger geometry tied

a9 asked whether the `arfl` scaling sketch should be reworked: it was
striking that the larger loop had only 553M active parameters against the
larger `arf`'s 1.1B, and the question was whether to widen or deepen toward
parity, with 4/8/4 at a draw of 8/16 or 4/8 as the candidate. What the
arithmetic and the running specimen said:

- The 553M was an accounting artifact. The sketch executed about 3.4B
  parameters and 70 layers per pass; it was cheap because the 400x rule prices
  unique parameters, so the token budget halved. 4/4/4 at `r_mean` 16, 4/8/4
  at 8, and 4/16/4 at 4 all execute the same 70 layers and 3.4B parameters
  per pass; they differ only in how much of that computation is unique weight,
  and every ~1B looped specimen costs about 2.3–2.6x the flat 1B's arithmetic
  whatever the shape (scripts in the session scratchpad, exact counts from
  `DeltaModel` on the meta device).
- The 16/32 draw was the one part without evidence. `screen-delta-arl-s1` at
  step 3500 puts the loss on the depth-trace rows at 3.806 / 3.748 / 3.755 for
  `r` = 1 / 4 / 8: about 0.05 nats from one iteration to the trained mean and
  nothing past it, with the core's update at `r = 8` still at norm 1.5, down
  from 14 at step 250. Against its paired `ar` at matched data it leads 3.463
  to 3.477. At 16/32 the `r = 1` case is drawn on 0.07% of steps, so the
  coincidence with `arf` would never be exercised.

a9 chose the larger geometry with its middle four cells tied at the screen's
4/8 draw: exact parameter parity, the same schedule, and the screen's own
pairing carried up, at about 2.6x the flat run's arithmetic. That needed a
core of more than one cell, which the spec had ruled a three-cell fact, and
a9 supplied the rule: fill the core cells' deltas progressively on the first
iteration, then loop back and keep adding to each cell's own delta. Each core
cell keeps its own block delta across iterations; a core site reads the seed,
the prelude delta, every other core cell's delta so far, and its own as the
live partial; the coda and payload read one delta per core cell. On the first
iteration the cells ahead are absent, so the banks are the flat column's, and
at one iteration the column is the flat column at any cell count. With one
core cell it is exactly the previous rule.

Landed together: `forward_column` runs the core cells in order with each
cell's delta measured from an origin pinned at its first entry and advanced
by what the other cells add between visits, `routing_blocks` is one per cell
for every condition, and the depth-trace script labels whatever sources the
core routers read. The three-cell screen is bitwise unchanged across 3,102
observables at `r` = 1, 2, 3 for `arl`, `arfl`, and `afl` (values, payloads,
core state, route weights and names, losses, every gradient) against the
previous commit, the `r = 1` coincidence is tested at four and five cells,
and a two-core-cell test pins the banks by name. `scaling.md`'s larger loop is
now the tied larger geometry, and the loop pages describe the general rule.

Later the same day the three architecture pages became two, at a9's
suggestion: `architecture.md` holds the four letters and the column, the
equations and state contracts that do not vary with geometry, with the former
depth page folded into its `l` section and `L` kept there as the unbuilt
extension; `scaling.md` holds the two geometries, now called the screen and
the flagship, with their accounting, budgets, the loop's measured cost, and
the longer and larger recipes. `design.md` keeps the recipe and points at
them for counts; the loop's sources table moved to `literature.md`.

a9 then asked about a bridge geometry between the screen and the flagship
and a super-flagship stretch goal. The ladder arithmetic: width 1,152 makes
every ratio exact (12/6 heads, 15 PKDA heads at projection width 1,920,
SwiGLU 4,992, six routing groups), and sixteen layers put it on the
geometric midpoint of the ladder at 417M active, three times the screen and
0.38 of the flagship, with a two-cell core. A super-flagship at twice the
flagship's parameters, 1920 x 32, would cost four times the flagship's
arithmetic at 400x, roughly seventy-five node-days on eight H100s; a9 parked
it until funding. The bridge is registered at context 4,096 rather than the
screen's 1,024 because a9 wants to publish it as an organism, with a 25x
recipe for Jobe and a 400x recipe for Prime.

a9 then asked what the commands would be and whether `--scale` and a
tokens-per-parameter flag should address every planned run. Landed as
`--scale screen|bridge|flagship` and `--tokens-per-param`, default 25, with
`--steps` derived from it unless typed. Two design points: the ratio's
denominator is the flat full stack's active count at the scale, not the
run's own, so `ar`, `arf`, and `arfl` keep one schedule and stay paired,
which a per-condition count would have broken; and the unit is tokens per
parameter rather than a multiple of the Chinchilla twenty, because that is
the number every table here reports. The count rounds up, which moves the
screen's 400x pair from 171,909 to 171,910 steps. A resume validates what
`--scale` or `--tokens-per-param` pinned and inherits the rest.

Then `--continue TAG`, a9's ask for the WSD continuation the protected
snapshots were kept for: a finished run extended to a longer schedule under
a new tag. The fork step is the last step both schedules reproduce, heat in
both and on the same side of both feedback boundaries, which is the feedback
boundary when it moves and the cooldown boundary otherwise; the continuation
restores the source's last snapshot at or before it, inherits every setting
but the length, and from there is the longer run exactly apart from the
inherited warmup. The tiny end-to-end test shows an eight-step `arf` run
continued to sixteen matching the fresh sixteen-step run's loss and
validation bit for bit. At the screen, 25x to 50x trains 13,430 new
steps instead of 21,489.

a9 then asked whether warmup should scale with the run or be fixed per
scale, proposing 2% of the 25x length. Adopted as the fraction applied to
the shorter of the run and the 25x recipe at its geometry: warmup guards the
optimizer's first steps at the batch and learning rate, which do not depend
on the horizon, while cooldown and the feedback boundary stay fractions of
the run. No existing schedule changes, since every specimen is at or below
25x; the 400x recipes lose most of their warmup, 3,438 to 215 at the screen
and 26,905 to 1,682 at the flagship; and a continuation of any run at or
above 25x is now the longer run exactly, which the `continue` record reports
as `exact`.

a9 then asked, before building the distributed path, what an `arfl` bridge
ladder on Prime, 25x to 100x to 200x to 400x by continuation, would take and
cost on the cards there. The arithmetic is in `scaling.md`: 2.15e21 FLOPs,
1.34x a single 400x run. Calibration: after seven speed rounds the screen
runs at 60–70% of the 4090's dense BF16 peak on Jobe, so the estimate takes
40% on Hopper and Blackwell untuned, 50% on Ampere, and 60% on the Ada cards
whose kernels are already the tuned ones. Refined the same evening with bandwidth in the model, since about a third of
a step at the bridge scales with memory bandwidth rather than tensor rate:
the loop's ladder is about 1,500 GPU-hours on H100 SXM, 3,600 on A100,
3,900 on L40S, whose tensor rate is twice the L40's on the same 864 GB/s,
5,500 on L40 or RTX 6000 Ada, 7,500 on A6000, and 3,200 on RTX PRO 6000 if
its BF16 rate with FP32 accumulation is the full 503 TF; the `arf` ladder is
about half of each. The 100x rung of the loop on an 8xL40S node is about
three and a half days as a continuation, the flat partner under two. At the day's market prices from an
aggregator, Prime's own marketplace could not be fetched, every card lands
between about $2.3k and $7k for the loop's ladder, H100 SXM cheapest at
$2.3–3.8k and fastest at eight days on one node, the `arf` partner about
half again. The price-per-hour of a card tracks its throughput closely
enough that the choice is wall-clock and porting risk, not cost.

## 2026-09-09 — Round seven: the loop's cost, measured, and what moves it

a9 paused `screen-delta-arl-s1` at step 1846 for a seventh speed round, the
first since the `l` letter made the middle cell run `r` times per column.
Everything was measured on that snapshot with real rows, restored optimizer
state, and the production CUDA graphs (scripts under `jobe:~/tmp/r7/`, the
report "The Loop Ledger"). Two independent Codex reads and two kernel-fork
reads with file:line citations fed the ranking.

The step is what the cost model says it is. A one-pass microbatch replays in
`58.7 + 14.6 (r - 1)` ms through `r = 7` and 205.8 ms at `r = 8`, where the
activation policy recomputed every block; the head is the one part that does
not scale with `r` (12.6 ms at `r = 1`, 13.5 at `r = 8`). Over the schedule's
realized draws (`r = 8` on 8.0% of steps, mean 3.90), eighty replays plus
about 0.14 s of clipping, optimizer, and shadow refresh reproduce the logged
8.3 s per step. At `r = 4` a microbatch is 42% GEMMs (all at BF16 peak), 25%
PKDA, 14% pointwise and eager elementwise, 13% head, 6% routers and
attention, across 3,504 kernels; the checkpoint at `r = 8` adds a full
forward, 45 ms.

What moves it, in the order it is being taken:

- **Ten cell-passes fit raw.** The whole one-pass loop family captured
  without checkpointing at 14.6 GiB allocated (the pool reserves the card
  either way, with the same allocator retry warnings every configuration
  logs), and the `r = 8` replay is 154 ms raw against 199 recomputing. The
  policy now admits forty layer-passes, so one pass through `r = 8` and two
  passes through `r = 3` keep their activations; the three-pass family raw
  runs out of memory at 22.4 GiB, so deeper modes still checkpoint. Exact,
  and 3.5% of an `arl` run.
- **Eager gradient sums between compiled graphs.** Autograd added 150
  residual-shaped BF16 tensors per `r = 4` microbatch (2.2 ms), summing the
  gradient contributions that reach one tensor from several compiled
  graphs. `torch._dynamo.explain` named the second cause exactly: a PKDA
  block compiled into **fifteen** graphs joined by fourteen breaks, every one
  of them at a `torch.compiler.disable`d FLA entry point (four at the fused
  convolution, one at the recurrence, three at the norm/gate), so each of the
  block's two routed reads sat in a different graph and every source entered
  twice per block. Wrapping those three kernels as custom operators with
  exact fake implementations makes a PKDA block one graph (0 breaks, 57 ops)
  and took the count to 63; it is bit-identical to calling FLA directly
  except for `d log_atk_scale`, whose atomic FP32 reduction differs by
  4.5e-8 against a run-to-run spread of 7.6e-6. It bought no time on its own
  (+0.3 ms at `r = 4`): the sums it removed came back as in-graph adds. What
  bought time was the bank underneath it — each within-column source now
  hands its readers an alias plus one BF16 accumulator that the routing
  backward adds into in place, one program per token, and returns no
  gradient for. That took 63 adds to 22 (2.18 ms to 0.42) for 0.78 ms of
  read-modify-write in the router kernel, and the whole round runs `r = 1/4/8`
  at 58.4/101.6/159.4 ms against 58.8/102.7/161.1, with 3,215 kernels
  against 3,504. The 22 that remain are all cell-entry fan-out: the tied
  core's entry residual is `block_start` in sixteen block calls because the
  cell delta is `h - start`. Writing it as `partial + a + m` instead would
  delete the input, but it is a different rounding of the delta and so a
  different model.
- **The classifier gradient's round trip.** Each microbatch zero-fills a
  233 MB BF16 classifier gradient and adds it into the FP32 sink (1.17 GB of
  traffic, 1.2 ms). Landed: the fork's backward accumulates into a
  caller-owned persistent BF16 buffer (`CCEParams.c_grad_accum`) and the
  trainer flushes it into the FP32 sink every four head calls
  (`--head-flush-every`, a runtime setting; 1 is the old path exactly, and a
  resumed snapshot without the field takes the default). Against a dense
  FP32 classifier gradient over the 80 real microbatches of step 1847 the
  per-call path sits at 7.9e-3 relative, cosine 0.999969, norm ratio
  1.00084; window 4 at 1.9e-2, 0.99982, 1.0021; the excess is a coherent
  norm bias, round four's signature at a thirtieth of its size, so the
  window is a judgement: three quarters of the saving for a 0.2% bias. Head
  call 15.16 to 13.68 ms, replay 102.6 to 101.5 ms at `r = 4`.
- **The FLA tail.** The fork read found `bwd_dhu` compiling with a single
  pipeline stage on Ada (an A100 shared-memory gate), FP32 intermediates the
  intra backward re-reads with amplification, a gate cumsum computed twice,
  and a `qg` that the forward o-kernel already holds; then two structural
  candidates, un-fusing the warp-starved `inter_solve_fused` and a
  tensor-core forward diagonal whose safety needs the within-sub-chunk gate
  span measured on real checkpoints first. Landed on the fork's `patches`:
  pipeline stages for the Ada scans, `dAqk`/`dAkk` and the ATK `gk`
  hand-off in the activation dtype, `g` kept, `qg` from the o-kernel, dead
  arguments and a memset gone: the PKDA layer's in-kernel time 2.348 to
  2.226 ms (5.2%), drift 7.44e-3 to 7.52e-3, whole-model replay 1.4%. Both
  structural candidates died on measurement. The un-fused
  `inter_solve_fused` split is bitwise identical and its own kernels are
  9 us slower; its apparent win was an allocation-address effect on a
  neighbouring kernel that a control with only the allocation reproduced.
  The tensor-core diagonal is unsafe here by an order of magnitude: the
  within-16-token gate span reaches 701 log2 on `arl` and 908 on `arf`
  against a ceiling near 85, with single-step decays of 159, so one factor
  overflows while its partner underflows. And `dg2` in BF16 breaks
  `dt_bias`'s gradient nineteen-fold, a 4,096-term cancelling sum.

Merged and qualified in one piece: the probe passes on the combined state
(145 tests; loop gradient parity 0.0163 against a 0.0164 floor; the flush
window's own check 9.18e-3 against 8.94e-3 per call; replays 50.8, 102.3,
and 155.3 ms for one, two, and three passes against 53.5, 107.7, and 161.9
at the start of the round), and a paired 24-step continuation from the
step-1846 snapshot tracks the pre-round kernels to 3.2e-5 relative in loss
with matching gradient norms, 3.6% faster on that stretch of draws on top of
the raw budget's 3.5%. `screen-delta-arl-s1` resumed from step 1846 on the
merged state with the default flush window: 47.8k tokens per second at
`r = 3` against 46.5k before the pause, 36.2k against 34.9k at `r = 5`, and
the same losses as the continuation check to three decimals.

What was measured and parked for the Hopper run: FP8. At the trunk's exact
shapes the 4090's tensor cores run 250 to 334 TFLOPS through
`torch._scaled_mm` against 140 to 172 in BF16, halving a cell's GEMM time in
isolation (6.92 to 3.78 ms) while naive per-call quantization makes it slower
than BF16 (7.20 ms); the design that survives review freezes every
activation and gradient scale for a whole optimizer step (the tied core runs
up to eight forwards before any backward), calibrates on real rows (warm-up
runs zero token rows), and pilots the two MLP projections first. The head's
two large dots are already in the layout FP8 needs on this card, and an
LSE-shaped proxy ran 1.63x BF16 at 64x128x64 tiles, so the head's FP8 pass
belongs to the same program.

Also measured and not taken now: the head's filter at 2^-10 costs 12% less
than at 2^-12 with a per-microbatch gradient error still inside the BF16
lock-accumulation floor (the unfiltered head is the least accurate on `dE`),
but the step-level check that caught round four's coherent bias has not been
run for it; the chunk size of the PKDA kernels is a hard-wired local optimum
(`inter_solve_fused` fixes four diagonal and six off-diagonal sub-blocks, the
ATK kernels hold a `BT x BT` decay tile in registers, and the state traffic
grows as `1/BT`); Triton FP8 with a transposed operand load runs at a third
of the K-contiguous rate on Ada.

## 2026-09-08 — The `l` letter is built

The tied-depth loop of [architecture.md](architecture.md#letter-l-the-tied-depth-loop) exists:
`arfl` builds, trains, evaluates, and decodes on the portable path, and every
subset of `arfl` builds and pairs with its unlooped condition. What was built
and the decisions behind it:

- The loop holds the first and last cells and runs the cell between them as
  one tied core `r` times per column. `r` is drawn once per step from the
  recurrent-depth log-normal Poisson draw with mean 4 and cap 8 on its own
  keyed sub-stream (`--loop-iterations` and `--loop-max-iterations`, both
  state-defining; snapshots move to v25). The core's partial is measured from
  the prelude output across iterations, so the core routers see an `arf`
  second-cell bank at every iteration and answer one stationary question.
- Mixing is same-depth: iteration `i` reads earlier positions' iteration-`i`
  writes, which is the recurrent-depth paper's training regime. The FBT
  payload is the only channel between columns. This needed no new kernel;
  each iteration is one more full-sequence pass through the same four blocks.
  The shared core cache of the earlier spec is deferred to an `L` letter, `l`
  plus the cache, specified on the same page and not built: it is the
  project's own departure from the paper, it needs a fused-position PKDA
  operator and a two-piece attention merge, and it would add a second, wide
  cross-column channel before we know what the first one carries under a
  loop.
- The presence mask at the core's iteration-1 attention entry is gone from
  the spec. A router scores whatever sources it is handed, so that site reads
  a three-source bank exactly as a cell entry does today. The one code
  consequence is that a block's "has a prior partial" is now whether it was
  handed a cell entry rather than a static per-layer flag, which leaves the
  flat column byte-identical.
- a9 declined a trained warm start, the previous column's core delta landed
  at the current core entry: it does not line up with existing work and would
  make the model less legible on purpose, which is not the goal. The
  zero-shot warm-start sweep went with it.
- At `r = 1` a looped condition coincides with its unlooped one in values,
  routes, losses, and gradients. The suite checks this on CPU for `al`,
  `afl`, `arl`, and `arfl`; the CUDA gate checks loss and gradient at the
  screen geometry. Decode parity through per-iteration core cache tracks
  holds to FP32 accumulation noise, relative error under `1e-4` at twenty
  executed layers. The draw's statistics match the spec (`E[r]` 3.88, median
  4, 10.3% at 1, 8.0% at 8), and resume is exact for `arfl`.
- New diagnostics: `depth_trace`, the fixed-`r` sweep, which on plain
  positions is one trajectory read out after every iteration, as a `depth`
  telemetry record and as `scripts/depth_trace.py`; and
  `scripts/loop_memory_stage.py` for the staged memory and time measurement
  on Jobe. The trainer reports cell-tokens per second beside pass-tokens and
  logs the realized `r`.

Two things the Jobe gate taught while landing. BF16 decode parity between
the parallel forward and the stepped cache drifts with executed depth, not
with the loop: the flat twelve-layer column already sits at 7% on the gate's
small geometry where the four-layer case sits at 3.5%, and in FP32 every
twelve-layer condition, loop included, sits at 0.17%; the loop's decode check
runs in FP32. And the CUDA path's gradient is nondeterministic at 1.8%
relative, the flat column against itself, because the head's lock-ordered
BF16 accumulation reaches every parameter through the backward; the loop at
`r = 1` sits exactly on that floor with an identical loss, so the gate
measures the floor and holds the loop to it. Numbers in
[runtime-qualification.md](runtime-qualification.md#the-loop).

The staged measurement (`scripts/loop_memory_stage.py`, record
`data/summary/loop-stage-2026-09-08.json`) says the whole family fits: the
twenty-four train graphs capture at 14.34 GiB allocated against the flat
column's 13.85, and the deepest eager mode, three passes at `r = 8`, peaks
at 4.02 GiB under block checkpointing. Replay runs from 53.5 ms per
microbatch at one pass and `r = 1`, the flat column's own number, to
601.8 ms at three passes and `r = 8`. Projected with the measured overhead,
the full schedule is 35.8 h on the default mixture, 59.1 h at two passes on
every step, and 89.8 h at three, against 42.3 h for the dense `arf`
specimens. The recipe for the first `arfl` specimen is a9's call.

Nothing was trained.

## 2026-09-08 — The plain trunk is gated NoPE GQA

Every dense attention layer is now the same layer: bias-free causal GQA with
per-head Q/K RMSNorm, no position encoding, and a sigmoid output gate. The
empty condition is twelve of them; `a` replaces three of every four with PKDA
and nothing else, so `a` isolates the mixer swap and every non-`a` condition
gets the gate and NoPE the same way. RoPE is gone from the code along with
`rope_theta`; a plain-trunk model infers position from the causal mask alone,
which is the known-viable but usually slightly weaker NoPE baseline for a
pure-attention decoder.

The attention gates keep their own seeded stream. They are no longer
`a`-private, but that stream is what the Jobe specimens initialized from, so
`a` conditions initialize exactly as before and a new `a`, `af`, or `arf` run
at seed 1 still pairs with `screen-delta-ar-s1`. The plain conditions gain
7,077,888 gate parameters (screen counts in
[design.md](design.md#conditions)). The `a` specimens' snapshots are
untouched and still load. No plain-trunk snapshot exists, so the contract
stays v24; one written before this change would fail on its missing gate keys
rather than silently read as the new trunk. Nothing was trained.

## 2026-09-08 — The operator is `delta`, and the specimens are named for their conditions

The console script `df` is now `delta`, which also stops shadowing the
disk-free command. The run tags follow the letters: `screen-delta-ar-s1`,
`screen-delta-arf-s1`, and `screen-delta-arf-s1-highLR`. NorMuonH `6e-3` is the
default, so it is the `2e-2` run that carries an LR suffix. The compiled token
store on Jobe is `/data/delta/tokens`.

The three Jobe specimens were rewritten in place: each snapshot now records its
`condition`, its new tag, and the new store path, and its run log records
`condition=` where it recorded `arm=`. With no arm-named artifact left, the
loader's `NAMED_ARMS` translation and its counterpart in `training_curves.py`
are gone; `saved_args` reads `condition` and nothing else. The snapshots keep
their v22/v23 contract versions, which is what their state is.

Deleted: the Mac's three snapshots, which were contract v1, v4, and v5 and so
already below the v16 surface. One of them was the earlier `arf` run on the
75/22/3 pass mixture, whose row leaves [findings.md](findings.md); its numbers
survive in the 2026-09-02 and 2026-09-08 entries below, where it is now the
earlier Mac `arf` run rather than a tag. Its logs and its `fused-`/`route-`
figure directories went with it.

Also renamed: `DFModel` to `DeltaModel`, `DFScorer` to `DeltaScorer`,
`DELTA_INDUCTOR_CACHE_DIR` for the durable Inductor cache, and `delta-loop` for
the tied-depth design. Figure directories took the new tags and their JSON
records were rewritten, but the PNGs keep the old tags in their titles until
they are regenerated. Nothing was trained.

## 2026-09-08 — Conditions become letters

The five named arms are replaced by a modular condition string: one letter per
change from the plain RoPE GQA decoder, `a` for the PKDA/gated-GQA trunk, `r`
for MHDB reads, `f` for FBT feedback, and `l` reserved for the tied-depth loop
(it parses, `DeltaModel` refuses it). `vanilla`, `base`, `mhdb`, `fbt`, and `df`
are now `""`, `a`, `ar`, `af`, and `arf`; the specimens are `ar` and `arf`
runs. Every subset of `arf` builds, so `r`, `f`, and `rf` on the plain trunk
exist for the first time (screen counts in [design.md](design.md#conditions)).
Snapshots move to v24 with a `condition` field; the analysis loader translates
the arm names in v16–v23 snapshots and `training_curves.py` translates them in
old run logs. `compare_arms.py` is now `compare_conditions.py` and writes
`compare_conditions.json`. Nothing was trained; the letter grammar is the
ground for the `l` design that follows.

## 2026-09-08 — `ar` against full-feedback `arf`: what the feedback package changed inside

Two completed runs on Jobe, paired by construction (seed 1, data
seed 0, identical rows and schedule, NorMuonH `6e-3`, 10,745 steps):

| run | condition | passes | pass-tokens | s/step | val (32 rows) | val_fused |
|---|---|---|---:|---:|---:|---:|
| `screen-delta-ar-s1` | `ar` | 1 throughout | 3.52B | 4.9 | 3.069 | — |
| `screen-delta-arf-s1` | `arf` | 3 from step 0 | 10.56B | 14.2 | 3.076 | 3.077 |

The `arf` run is the fresh-run form of the question left open on 2026-09-02:
does the FBT benefit form when the fused mode is trained densely from the
start? It is the recipe variant `--feedback-start 0 --three-pass 1`, and equal
steps here are 3x the compute. Everything below is one paired observation plus
same-checkpoint interventions on the final snapshots. Scripts:
`scripts/compare_conditions.py`, `weight_divergence.py`, `fused_diagnostics.py`,
`entry_sweeps.py`, `feedback_followups.py`, `payload_swap.py`,
`route_report.py`, `training_curves.py`, `analysis_figures.py`; records and
figures under `figures/curves-ar-vs-arf/`,
`figures/compare-screen-delta-ar-s1-vs-screen-delta-arf-s1/`,
`figures/weights-…/`, `figures/fused-screen-delta-arf-s1/`, and
`figures/route-…/` (ignored); run logs copied to `logs/`.

### The gap is small and mostly a small-sample draw

On 256 held-out rows (262k tokens; the first 32 reproduce the logged numbers
to the third decimal):

| predictor | CE | difference | s.e. |
|---|---:|---:|---:|
| `ar` pass 1 | 2.8862 | | |
| `arf` pass 1 | 2.8887 | +0.0025 vs `ar` | 0.0013 |
| `arf` fused (prefix 1) | 2.8899 | +0.0012 vs `arf` pass 1 | 0.0006 |
| `arf` fused, second iteration | 2.8915 | +0.0016 vs fused | |

Top-1 accuracy is 42.92% for both pass-1 predictors. The per-step pass-1
training loss on identical rows (exact pairing, EMA 300) was +0.0054 for the
first 37% of the schedule, −0.0007 through the second half of the heat, and
+0.0013 in the cooldown: the plain-mode tax of serving two input regimes is
real, small, and largest while the fused mode is forming.

### The two arms differ like two seeds

KL(`ar` ‖ `arf` pass 1) is 0.215 nats, argmax agreement 77.6%, per-token CE
correlation 0.972; a split-validated 50/50 probability mixture of the two
gains 0.049 nats. Within `arf`, KL(pass 1 ‖ fused) is 0.039, agreement 89.4%,
mixture gain 0.011. In weight space every shared trunk matrix sits 85–87°
from the paired initialization and 87° from its twin in the other run
(cosine 0.06–0.08; total-update cosine about 0.5), while norms, router
vectors, and PKDA vectors stay within a few degrees and the embedding at
cosine 0.77. Under fixed-radius normalized updates the two trajectories
separate like different seeds, so the +0.0025 is a mean shift riding on a
±0.24 interquartile per-token spread and no per-token difference is
attributable to the package. Conditioning on the reference's own loss shows
the expected regression toward the mean (`arf` looks better exactly where
`ar` is worst, and worse where `ar` is confident); that panel is an
artifact of the conditioning, so the tables condition on the reference's
entropy and on token frequency instead.

Where the +0.0025 concentrates: positions 1–15 carry +0.02 to +0.06 (few
tokens, 1–2 s.e.), positions beyond 64 carry +0.0015 to +0.002; rare target
tokens +0.012, frequent ones −0.001 to +0.002; input-frequency and surprise
bins are flat within noise.

### Dense exposure closed the fused deficit and formed no benefit

Against the earlier Mac `arf` run (75/22/3 recipe, +0.024 fused gap):

| diagnostic | earlier Mac `arf` run | `screen-delta-arf-s1` |
|---|---:|---:|
| fused − pass 1 | +0.024 | +0.0012 |
| KL(pass 1 ‖ fused) / argmax agreement | 0.057 / 87.5% | 0.039 / 89.4% |
| fused wins on | 47.6% of tokens | 50.4% |
| split-validated mixture gain | 0.006 | 0.011 |
| rarest 10% input tokens / most frequent 10% | +0.047 / +0.011 | +0.015 / −0.004 |
| fused-in token in the top 1% of surprise | +0.085 | +0.024 |
| positions 1–3 / 4–15 / 64+ | +0.065 / +0.058 / +0.024 | +0.025 / +0.014 / +0.001 |
| second fused iteration | +0.007 worse | +0.0016 worse |
| gradient cosine, pass 1 vs fused (trunk matrices / all shared) | 0.75 / 0.86 | 0.82 / 0.90 |

The plain-prefix-512 and random-prefix fused passes agree with prefix 1 to
±0.001, jitter changes nothing, and the 30-iteration self-composition
contracts cleanly (relative top-state update 0.16 → 0.037 → 0.013 → 0.009 and
flat; loss within 0.001 of pass 1 on 8 rows). The residual cost keeps its
decode signature (rare and surprising fused-in tokens, rarely fused early
positions) at three to ten times smaller amplitude. The benefit side did not
appear: fused wins half the tokens by tiny margins, iterating hurts, and the
mixture gain of 0.011 is the whole complementary information.

### Mechanism: the first cell cancels the fused seed, the rest runs plain

On the same tokens, `arf`'s fused pass differs from its plain pass almost only
in cell 0. The fused seed enters at 23x the plain seed's RMS (0.85 against
0.04); the first completed block delta then changes by 3.7x its own RMS and is
orthogonal to its plain-pass self (cosine −0.02, CKA 0.57, RMS 0.76 ≈ the
seed's), while block 1 changes by 0.24x (cosine 0.97, CKA 0.974), block 2 by
0.15x (0.988, 0.992), and `h_top` by 0.15x (0.987, 0.992). Cell 0 is a decoder
that maps the fused seed back onto the plain representation; cells 1–2 then
compute what they compute on a plain pass. Between the two arms the same
sources have CKA 0.92 (block 0), 0.81 (block 1), 0.95 (block 2 and `h_top`)
on plain passes: the middle cell is where the two trainings diverged most.

Routing is consistent with that: `arf`'s early MLP sites read the null on
both passes (L1–L3.mlp 0.78–0.89, L4.attn 0.84–0.90) where `ar`'s read the
partial and seed (0.44–0.75); on the fused pass `arf`'s L0.mlp reads the fused
seed at 0.86, and the top attention sites switch from block 0 to the seed
(L9/L10/L11.attn seed mass 0.25/0.23/0.00 → 0.39/0.38/0.53), a direct read of
the previous column's state just before the readout. The payload router's
head 0 reads block 0 on pass 1 and the seed on pass 2 at weight 1.0, one head
sits on its null (a learned constant of RMS 0.33), and per-head ablation costs
at most +0.006; forcing every head to the null costs +0.023, removing the
routed term costs +3.3. The payload itself is linearly the readout state:
ridge R² 0.978 from `final_norm(h_top)`, 0.96 from raw `h_top`, 0.69 from the
other arm's readout state. The entry is fully co-adapted: a constant gate
costs +8.4, gate temperature 2 costs +0.09, seed scale 0.5 or 2 costs +7.3 or
+2.5, a unit embedding bypass +0.023, a payload from another row +0.46, a zero
payload +32. The token is recoverable from the fused seed (ridge probe R²
0.78, nearest-embedding hit 98%); the gate's token-dependent share of the
pre-norm input is 15% at the median (36% at p90, up from 8% / 15%).

### Reading

1. With every step three-pass from step 0, the feedback package costs the
   plain mode +0.0025 ± 0.0013 nats at 25 tokens per parameter and 3x the
   pass-tokens; the logged +0.007 was a 32-row draw. The paired training
   trace puts the cost early and in the cooldown, not in a competition
   signature.
2. Dense exposure answered the 2026-09-02 question: the fused deficit was a
   recipe artifact (0.024 → 0.001), and the FBT benefit still does not form
   on this trunk at this scale. The model neutralizes the fused seed in cell
   0 and reproduces the plain computation; the previous column's state is
   load-bearing only as the carrier through which the current token is
   decoded.
3. For the organism the channel is stable, harmless, and token-legible: the
   payload is the readout basis and the seed decodes the token at 98%. That is
   the opposite of an opaque cross-column channel. The 2026-09-02 decision
   tree's "redesign the entry" branch is the one this selects; whether to
   take it, and whether the `ar` run doubles as the reference `ar`
   specimen, are a9's calls.

### Downstream zero-shot tasks

The same two snapshots on the workspace's pinned nine-task suite
(`scripts/downstream_eval.py`; harness prompts, `acc_norm` by character
length, batch 16, buckets 128–1024), with `EleutherAI/pythia-160m` (300B
tokens) scored by the same code as an anchor. The plumbing reproduces
EleutherAI's published pythia-160m harness numbers to within about one
standard error on ARC-Easy (43.6 / 39.6 against 43.5 / 39.7), ARC-Challenge
(19.5 / 23.6 against 18.8 / 23.3), PIQA (62.3 / 61.9 against 62.7 / 61.6),
SciQ (75.4 / 67.7 against 74.1 / 66.8) and LAMBADA perplexity (37.3 against
38.1); WinoGrande sits 1.8 points low and LAMBADA accuracy 2.6 points high
(35.4 against 32.8), unresolved. Records:
`figures/downstream-<tag>/downstream_<mode>.json`, comparisons beside them.

| task (metric) | pythia-160m | `ar` | `arf` Standard | `arf` Fused |
|---|---:|---:|---:|---:|
| HellaSwag (acc_norm) | 30.3 | 34.6 ± 0.5 | 34.8 | 35.2 |
| ARC-Easy (acc_norm) | 39.6 | 47.4 ± 1.0 | 48.8 | 48.3 |
| ARC-Challenge (acc_norm) | 23.6 | 26.3 ± 1.3 | 26.6 | 26.0 |
| PIQA (acc_norm) | 61.9 | 63.7 ± 1.1 | 63.4 | 64.0 |
| WinoGrande (acc) | 51.3 | 50.0 ± 1.4 | 51.2 | 50.7 |
| BoolQ (acc) | 56.5 | 59.1 ± 0.9 | 61.1 | 61.7 |
| OpenBookQA (acc_norm) | 26.8 | 31.8 ± 2.1 | 31.8 | 32.0 |
| SciQ (acc_norm) | 67.7 | 68.7 ± 1.5 | 69.8 | 69.3 |
| LAMBADA (acc / ppl) | 35.4 / 37.3 | 29.9 / 56.9 | 30.5 / 58.1 | 32.0 / 56.2 |

Both specimens beat the 300B-token pythia-160m on the educational and
science-flavored tasks (ARC-Easy by 8–9 points, HellaSwag by 4, OpenBookQA by
5, SciQ by 1–2) and trail it badly on LAMBADA, which is fiction and
long-range: FineWeb-Edu at 3.5B tokens, in one line. ARC-Challenge and
WinoGrande are at chance for every model, and BoolQ is below the 62.2%
majority class for every model.

**`ar` against `arf` Standard, paired on identical documents.** Accuracy
differs by less than two standard errors on every task except BoolQ (+2.1 ±
0.6) and SciQ (+2.1 ± 1.0); `arf` leads on seven of nine, which is weak
evidence in itself. The gold-continuation log-probability rises by 0.4–0.9
nats per document on ARC, BoolQ, SciQ and PIQA, but the shift over *all*
choices is the same size: `arf` assigns more mass to short answers after
`Answer:`, a prompt-format calibration difference. The discriminative margin
(gold minus best distractor) moves by less than 0.05 nats on every task but
BoolQ (+0.13 ± 0.02) and SciQ (+0.15 ± 0.05), and the BoolQ margin is a pure
"yes" bias: `arf` answers yes on 94.7% of documents against `ar`'s 86.7%
(62.2% are yes), so the margin rises +0.58 on yes-gold and falls −0.60 on
no-gold documents. Downstream, then, the arms are indistinguishable at this
resolution apart from a calibration idiosyncrasy of the kind two seeds also
show.

**`arf` Standard against `arf` Fused, paired within one model.** This pairing
resolves far smaller effects, and the fused pass is not a no-op downstream:

| task | acc diff | gold logp | all choices | margin |
|---|---:|---:|---:|---:|
| LAMBADA | +1.47 ± 0.36 | +0.033 ± 0.009 | | |
| HellaSwag (acc_norm) | +0.39 ± 0.19 | +0.232 ± 0.018 | +0.105 ± 0.009 | +0.160 ± 0.022 |
| PIQA | +1.03 ± 0.54 | +0.050 ± 0.042 | +0.038 | +0.025 ± 0.031 |
| OpenBookQA | +0.40 ± 0.85 | +0.119 ± 0.043 | +0.079 | +0.019 ± 0.048 |
| BoolQ | +0.61 ± 0.28 | −0.107 ± 0.006 | −0.108 | +0.002 ± 0.004 |
| ARC-Easy | −0.17 ± 0.50 | −0.070 ± 0.016 | −0.056 | −0.012 ± 0.016 |
| SciQ | −0.30 ± 0.59 | −0.332 ± 0.019 | −0.300 | −0.040 ± 0.020 |
| ARC-Challenge | −0.34 ± 0.57 | −0.042 ± 0.027 | −0.043 | +0.026 ± 0.027 |
| WinoGrande | −0.47 ± 1.11 | −0.020 ± 0.020 | −0.022 | +0.005 ± 0.013 |

The fused pass predicts LAMBADA's last word better (+1.5 points at four
standard errors, 208 documents gained against 132 lost) and discriminates
HellaSwag endings better (+0.16 nats of margin over 29-token continuations,
+0.005 per token), while it lowers the probability of every short answer after
a QA prompt by 0.05–0.3 nats with no change in margin. So the +0.0012 mean gap
on FineWeb tokens is a net of a small benefit on long-range, narrative
continuation and a small calibration cost on out-of-distribution answer
formats. That is the first downstream behavior that separates the two modes; it
is a paired observation on one checkpoint, and a second seed would say whether
it is general.

**More fused passes.** FBT reports that further prefill passes keep helping
with diminishing returns. Here the first pass is the whole effect:

| paired on identical documents | pooled acc gain | LAMBADA acc | HellaSwag margin |
|---|---:|---:|---:|
| Standard → fused ×1 | +0.42 ± 0.12 | +1.47 ± 0.36 | +0.160 ± 0.022 |
| Standard → fused ×2 | +0.36 ± 0.13 | +1.30 ± 0.37 | +0.146 ± 0.023 |
| Standard → fused ×3 | +0.38 ± 0.13 | +1.16 ± 0.37 | +0.145 ± 0.023 |
| fused ×1 → ×2 | −0.07 ± 0.06 | −0.17 ± 0.17 | −0.014 ± 0.006 |
| fused ×2 → ×3 | +0.04 ± 0.04 | −0.14 ± 0.12 | −0.001 ± 0.003 |

The second pass lowers the gold log-probability on every task but SciQ
(HellaSwag −0.034 ± 0.005, PIQA −0.100 ± 0.012, LAMBADA −0.005 ± 0.003) and
the third changes almost nothing, the downstream face of the self-composition
trace: the map contracts by the second iteration to a fixed point a hair
below the first pass. Front-loaded, as in FBT; not continuing.

**Soft against Fused: where the gain comes from.** `--mode soft` runs the
feedback pass with the plain prefix set to each item's context, so only the
scored continuation receives feedback (exact Soft decoding for the first
continuation token; a second pass makes the second exact and changed nothing:
pooled +0.04 ± 0.03, LAMBADA +0.0000 ± 0.0002 nats).

| paired on identical documents | pooled acc | LAMBADA acc | LAMBADA gold logp | HellaSwag margin |
|---|---:|---:|---:|---:|
| Standard → Soft | −0.00 ± 0.04 | −0.02 ± 0.04 | +0.0006 ± 0.0011 | +0.114 ± 0.020 |
| Soft → Fused | +0.19 ± 0.10 | +1.49 ± 0.36 | +0.032 ± 0.009 | +0.045 ± 0.012 |
| Standard → Fused | +0.42 ± 0.12 | +1.47 ± 0.36 | +0.033 ± 0.009 | +0.160 ± 0.022 |

On LAMBADA the feedback transition itself is worth nothing: Soft equals
Standard to four decimals (3 documents flip each way), and the whole +1.5
points appears only when the context is re-processed under its own top
states. On HellaSwag about 70% of the margin gain survives with a plain
context, so there the feedback along the 29-token continuation is what
discriminates the endings. The cross-column channel as trained is a
per-position depth extension: it helps when the token being scored has
already been fused into (a continuation) or when the context has been
(fused prefill), and a single transition into a fresh position carries
nothing usable. That is the sharpest behavioral statement about the channel
so far, and it is consistent with cell 0 cancelling the seed: what survives
the cancellation is what the second pass computed, not what the payload
carried.

Repo notes: checkpoint-analysis helpers now live in
`delta_feedback_experiment.analysis` (loader, trainer numerics, per-token
losses, fused inputs; `tests/test_analysis.py`), the September-2 scratch
scripts were promoted to `scripts/` under the names above with a shared
`scripts/figstyle.py`. The downstream suite is the workspace
module `transformer_experiments.downstream` (pinned Hub revisions, a scorer
protocol, paired comparison, an HF reference scorer); this experiment's
scorer is `scripts/downstream_eval.py`.

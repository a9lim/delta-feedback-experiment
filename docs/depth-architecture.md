# The `l` letter: a tied-depth loop

This document specifies the `l` condition letter, which is built, and the
`L` letter, its shared-cache extension, which is specified and not built.
Where this document and [architecture.md](architecture.md) both speak, this
one describes the loop only.

Under `l` the column's first cell is a prelude, its last cell a coda, and the
cells between them become one weight-tied core that runs `r` times per
column, with `r` drawn once per optimizer step. At the screen geometry that is
one four-layer core between one prelude cell and one coda cell. Recurrent-depth
language models supply the tied core and the iteration draw; the horizontal
channel between columns stays exactly the FBT payload of `f`, and the mixer
caches stay same-depth: iteration `i` of a column reads earlier columns'
iteration-`i` writes, as they do in that paper's training. `arfl` is the full
built stack.

## Why

The tied core adds a second axis of recurrence: what changes within a column
as the same weights are reused, what an intervention at a particular iteration
does, and which information is newly computed versus carried. The current
`arf` channel turned out token-legible ([findings.md](findings.md)); a core
that refines a column over several iterations before the payload is emitted is
one candidate way to give the channel something to carry. At `r = 1` the loop
is exactly its unlooped condition, so the loop is the one thing an `arfl`
specimen adds over an `arf` specimen. Every iteration-indexed trace says
whether its index is core depth, Jacobi pass number, or token position.

## Geometry

The loop keeps the screen geometry in [design.md](design.md) (width 768,
twelve unique layers, three cells, vocabulary 151,936) and reinterprets the
three cells:

| Field | `arfl` value |
|---|---:|
| Prelude `P` | one cell, layers 0–3 |
| Recurrent core `R` | one cell, layers 4–7, weight-tied across iterations |
| Coda `C` | one cell, layers 8–11 |
| Iterations per column `r` | drawn per step, `1 ≤ r ≤ 8` |
| Mean iteration rate `r_mean` | 4 |
| Compute depth per pass | `4 + 4r + 4` layers, `2 + r` cells |
| Unique parameters | identical to `arf` |

Every cell is `[PKDA, PKDA, PKDA, gated global GQA]` with the mixer, gate,
routing-group, and precision contracts of `architecture.md`. At `r = 1` the
column executes the same twelve layers as `arf` with the same parameters and
the same source banks, so `arfl` at one iteration coincides with `arf` in
values, routes, losses, and gradients.

The rule generalizes: a model of `4n` layers holds its first and last cells
and loops the `c = n - 2` cells between them, in order, as one tied core. `l`
requires whole cells and at least three of them. Each core cell keeps its own
block delta, accumulated across iterations, so the looped column has exactly
the unlooped column's cells and block sources and the one-iteration
coincidence holds at every cell count. The three-cell geometry is the one
trained; the larger loop in [scaling.md](scaling.md#larger-loop) ties four
cells.

A `k`-pass batch at iteration count `r` costs `k (2 + c r)` cell evaluations
per predicted token. Results report **cell-tokens** beside pass-tokens: `arf`
spends three cell-tokens per pass-token, `arfl` spends `2 + c r`, which is
`2 + r` at the screen.

## Column

```text
token embedding e_t -------------------+
                                       +-- FBT fuse --> seed s_t
previous payload p_(t-1) --------------+                  |
                                                          v
        +----------------------- prelude P ------------------------+  Delta_P
        |                                                          |
        |   +--------------- core R, tied, r iterations --------+  |
        |   | iteration i reads: seed, Delta_P, core partial    |  |  Delta_R
        |   +---------------------------------------------------+  |
        |                                                          |
        +------------------------- coda C --------------------------+  Delta_C
                          |                         |
                       h_top                    block deltas
                          +------------+------------+
                                       v
                                routed payload p_t
```

An equivalent high-level column, with each cell running its four layers and
per-layer partial sources exactly as in `architecture.md`:

```python
e = embed(tokens)
seed = e if previous_payload is None else fbt_fuse(previous_payload, e)

h = prelude_cell(seed, sources=[seed])
Delta_P = h - seed

core_entry = h
for i in range(r):                                   # same r for every pass of a step
    h = core_cell(h, sources=[seed, Delta_P], entry=core_entry)
Delta_R = h - core_entry

h = coda_cell(h, sources=[seed, Delta_P, Delta_R])
Delta_C = h - (core_entry + Delta_R)

payload = payload_norm(h + mhdb([seed, Delta_P, Delta_R, Delta_C], site="payload"))
logits = tied_readout(final_norm(h))
```

Each `mhdb` call prepends its site's learned null. Every cell measures its
partial from its own entry; the core's entry is pinned to the prelude output
for all `r` iterations, so the core is one cell of `4r` layers whose partial
accumulates across iterations.

With `c` core cells the core runs cells `R_1 … R_c` in order on every
iteration, and each keeps its own delta `Delta_R_j`, accumulated across
iterations: cell `j` measures from an origin pinned at its first entry and
advanced by whatever the other core cells add between its visits. Its bank at
every site is the seed, `Delta_P`, every other core cell's delta so far, and
its own as the live partial; the coda and the payload read
`Delta_R_1 … Delta_R_c` in place of `Delta_R`, whose sum they are. With one
core cell nothing runs between visits, the origin is the prelude output, and
this rule is the one above. There is no adapter and no random initial state:
the seed and prelude delta are re-readable at every iteration through the
routers. At a fused position the token enters only through the FBT gate, as in
`arf`; a plain position seeds from the embedding itself.

In code, `forward_column` runs the prelude, the core cells `iterations`
times, and the coda as `2 + c` routing cells, numbered as the unlooped column
numbers them; `ColumnOutput` carries `iterations`,
`core_entry`, and `core_state`, and a core site's route weights are keyed by
iteration (`L4i2.attn`).

## Residual shell in the core

Every cell, the core included, keeps the pre-norm shell of `architecture.md`
with branch scale `1/sqrt(2L)` at `L = 12`, the unique layer count. No
branch-output or residual-state norm is added, and the telescoping identity
holds in every cell:

```text
h_top = seed + Delta_P + Delta_R + Delta_C
```

with `Delta_R` the sum of the core cells' deltas. The shell is chosen because
it has no mechanism that forces a nonzero update: a branch-output norm would
pin every iteration's update size and make a fixed point unreachable by
construction. Whether the tied core actually contracts is
not assumed. Hyperball radii bound the NorMuonH matrices, but norm scales,
router values, and the NAdam-owned gates and controls are not radius-bounded,
so whether it contracts is measured: the depth trace below is the guard, and a
diverging trace means the specification changes.

## Iteration count

`r` is drawn once per optimizer step and shared by every pass and microbatch
of that step:

```text
tau ~ Normal(log(r_mean - 1) - sigma^2 / 2, sigma),   sigma = 1/2
r   = min(1 + Poisson(exp(tau)), r_max),               r_mean = 4, r_max = 8
```

This is the recurrent-depth log-normal Poisson draw with the rate shifted by
one so that the uncapped mean of `r` is `r_mean`. Under the cap:

| Statistic | Value |
|---|---:|
| `E[r]` | 3.88 |
| median `r` | 4 |
| `P(r = 1)` | 10.3% |
| `P(r = 8)` | 8.0% |
| expected layers per pass | 23.5 |
| expected cells per pass | 5.88 |

The draw comes from its own keyed sub-stream of the data seed and step, so `arf`
and `arfl` share byte-identical pass-count, prefix, and jitter draws.
`--loop-iterations` and `--loop-max-iterations` set `r_mean` and `r_max`; both
are state-defining. Realized `r` is logged per step and enters the cell-token
rate.

Evaluation and decoding use one fixed `r` for the prompt passes and the entire
decode trajectory; `r` never changes within a request. Metrics use
`r = r_mean`; the fixed-`r` sweep over `1..r_max` is the depth trace.

## MHDB source banks

Routers follow `architecture.md` exactly: zero-initialized width-`D` query,
full-width RMS key statistics, raw values, one softmax per routing group, a
site-local null prepended to every bank. A core cell's eight sites are shared
across iterations because they are core weights. At the screen the core is one
cell, so its bank is a cell's bank:

| Site | Sources after the null |
|---|---|
| prelude, attention entry | seed |
| prelude, other sublayers | seed, partial |
| core, attention entry of iteration 1 | seed, `Delta_P` |
| core, every other sublayer | seed, `Delta_P`, core partial |
| coda, attention entry | seed, `Delta_P`, `Delta_R` |
| coda, other sublayers | seed, `Delta_P`, `Delta_R`, partial |
| payload | seed, `Delta_P`, `Delta_R`, `Delta_C` |

The core partial is `h - core_entry`, the core's progress since the prelude
output, accumulated across iterations rather than reset at each; `Delta_R` is
its value at core exit. The core's bank therefore has the shape and meaning of
an `arf` second-cell bank at every iteration, and the tied routers answer one
stationary question rather than one per iteration index. Every bank's non-null
sources reconstruct the current residual exactly, as in `architecture.md`, so
the uniform zero-query mixture stays collinear with the residual and pass-1
routing is functionally inert at initialization at every iteration. The
residual is the state; the routers are the learned input injection of the seed
and the prelude. Per-iteration deltas are never sources: an iteration boundary
is a boundary in weights, not in state.

A core cell's partial is exactly absent at one site, its attention entry on
iteration 1, where the cell has not yet moved. A router scores whatever
sources it is handed, so that site reads a three-source bank exactly as the
unlooped cell entry does, and from iteration 2 on the same router reads the
partial as a fourth source. No placeholder and no presence mask are involved.

With `c` core cells the bank keeps the unlooped column's names. Core cell `j`
at iteration `i` reads the seed, `Delta_P`, the deltas so far of the cells
before it from this iteration and of the cells after it from the previous
one, and its own accumulated delta as the partial; the coda and the payload
read `Delta_R_1 … Delta_R_c`. On the first iteration the cells after `j` have
written nothing and are absent, so the first iteration's banks are the
unlooped column's banks exactly, and on the last iteration a cell that has
finished enters the bank as its completed block delta. A core cell's delta is
never reset and never split by iteration: it is the tensor the coda reads,
caught midway.

## Input injection and column start

The recurrent-depth decoder computes `e = P(x)`, draws a random state `s_0`,
iterates `s_i = R(e, s_(i-1))` where `R` opens with an adapter that maps the
concatenation of the state and `e` back to the residual width once per
iteration, and reads out `C(s_r)`. Its warm start is an inference-only
substitution of the previous token's `s_r` for the noise. `arfl` realizes
each piece differently:

| Recurrent-depth decoder | `arfl` |
|---|---|
| injected input `e = P(x)` | `seed + Delta_P`, the prelude output |
| adapter on `[s; e]`, once per iteration | router reads of seed and `Delta_P` at all eight core sites |
| separate latent stream `s` | the residual itself; `h_top = seed + Delta_P + Delta_R + Delta_C` |
| random `s_0` | the prelude output |
| zero-shot warm start from the previous token's `s_r` | the trained FBT payload at the seed |

The two designs differ in one structural place: where the previous position's
state enters. The recurrent-depth warm start hands the core its previous fixed
point directly, untrained. Here the previous column enters at the seed,
through the token-gated FBT fusion, and the prelude reprocesses the fused seed
before the core starts, so the vertical iteration restarts from the prelude
output at every column. The previous column's converged core state reaches the
current core through one trained channel, the payload, compressed by the
payload norm and the token gate.

Two expressivity gaps against the adapter are known and accepted:

- The router is a per-group convex mixture of raw values, and the branch's
  projections are shared across every source; the adapter is a dedicated
  linear map that can transform `e` independently of the state. The
  recurrent-depth paper found additive re-injection matched concatenation at
  small scale and lost at scale; the router is closer to addition with
  learned gates.
- The core carries no state across positions in its residual. Whether the
  restart costs anything is a question for the trained specimen.

Landing the previous column's state at the core entry, trained or zero-shot,
would move the horizontal channel for the whole family and break `r = 1`
coincidence with `arf`. It is a different design rather than a loop variant
and is not specified here.

## Horizontal channel and mixing

The payload is unchanged from `architecture.md` and `design.md`: asymmetric
FBT fusion at the seed, routed payload over the null, seed, and completed
deltas, `payload_norm`, keyed uniform jitter, the one-position shift, and the
per-row plain prefix. It is the only channel between columns besides the mixer
caches.

Mixing is **same-depth** at every position: at core iteration `i` a position
reads earlier positions' iteration-`i` writes of the current pass. Each
iteration is one more full-sequence evaluation of the same four blocks, so the
PKDA chunk operator, FlexAttention, and the router run unchanged; a PKDA state
at iteration `i` starts from zero and advances across the pass's token order
exactly as it does for a pass today. Plain and fused positions differ only in
their seed, as in `arf`.

Decoding keeps one core cache track per iteration: track `i` holds the PKDA
matrix and diagonal states, the convolution histories, and the GQA K/V that
iteration `i` wrote at every earlier position. `forward_column` selects the
track before each core iteration, and every track shares the column position.
Standard, Soft, and Fused decoding all hold `2 + r` cells of cache at the
request's fixed `r`; a `KVCache` is allocated for that `r`.

## Training

Passes and iterations nest: each pass runs the prelude once, the core `r`
times, and the coda once; passes follow the Jacobi scheme of `design.md` with
the same pass mixture, prefix mixin, jitter, and loss:

```text
K = 1:  loss = ell_1
K > 1:  loss = ell_1 + mean(ell_2, ..., ell_K)
```

Readout happens only after the coda at the final iteration. The cooldown
log-partition penalty applies unchanged.

Backpropagation is complete: every iteration of every pass is in the graph.
The trainer's activation policy counts executed layers, `8 + 4r` per pass at
the screen, against a measured raw budget of forty layer-passes (ten cells:
one pass through `r = 8`, two passes through `r = 3`), so block-level
checkpointing switches on for every mode deeper than that. Truncated
backpropagation through the last iterations is excluded.

Optimizer ownership follows `architecture.md`: the tied core's matrices are
NorMuonH parameters whose gradients sum across iterations and passes, with one
fixed Hyperball radius each. The global FP32 gradient is clipped to norm
10.0 before both steps.

Execution follows the [runtime graph contract](runtime-qualification.md): one
fixed-address train CUDA graph per reachable `(pass count, r)` pair, at most
twenty-four at the screen, all sharing one pool, plus the no-grad evaluation
graphs at `r = r_mean`. `scripts/loop_memory_stage.py` measures the eager
modes and the captured family in stages.

`arfl` pairs with `arf`: byte-identical initialization of every parameter, the
same stream, row order, schedule, and pass-count, prefix, and jitter draws. The
`r` draw is the only additional randomness.

## Evaluation and diagnostics

`val` is pass-1 held-out cross-entropy at `r = r_mean`, every position plain.
`val_fused` is a second pass with plain-prefix length 1 at `r = r_mean`, so
every position but the first is fused. Both report cell-tokens.

`arfl - arf` at equal steps is a **matched-data** contrast on the loop at
unequal compute. A matched-compute view compares checkpoints or curves at
equal cumulative cell-tokens or measured device-time; equal pass-tokens are
never matched compute for this condition.

**Depth trace.** `depth_trace` runs the column at every iteration count from 1
to `r_max` and records the held-out loss with the coda applied after that many
iterations and `mean_token ||core_state(r) - core_state(r-1)||_2`, the size of
iteration `r`'s update, measured from the core entry at `r = 1`. Under
same-depth mixing the state after iteration `i` does not depend on how many
iterations follow, so on plain positions the sweep is one trajectory read out
after every iteration; the fused sweep runs both passes at each count. The
trainer logs a `depth` record at every evaluation point with the loss at one
iteration, at `r_mean`, and at `r_max`, and the last update norm.
`scripts/depth_trace.py` runs the sweep on a snapshot over more rows, in both
label assignments, together with the core routers' mass on every source by
iteration, which is the learned input-injection profile.

The payload self-composition trace of `design.md` runs at `r = r_mean` and
tests the horizontal channel exactly as it does for `arf`. The recurrent-depth
KL exit rule can run as a diagnostic; metrics use fixed `r`.

## Parameter, cache, and cost accounting

`arfl` has exactly the parameters of `arf`: 257,514,792 total and 140,827,944
active non-embedding.

Per sequence at 1,024 context, one cell's mixer cache is 3.456 MiB (three FP32
PKDA matrix and diagonal states plus BF16 convolution histories, 1.956 MiB; one
BF16 KV cache, 1.500 MiB). Every decode mode holds `2 + c r` cells at the
request's `r`, `2 + r` at the screen:

| Fixed `r` | Cells cached | Cache |
|---|---:|---:|
| 1 | 3 | 10.4 MiB |
| 4 | 6 | 20.7 MiB |
| 8 | 10 | 34.6 MiB |

Training memory and step time are measured, not derived
([runtime qualification](runtime-qualification.md#the-loop), record
`data/summary/loop-stage-2026-09-09.json`). Capturing the twenty-four train
graphs and the evaluation graph took 67 s and peaked at 15.21 GiB allocated
and 23.02 GiB reserved, against 13.85 and 22.99 GiB for the flat column's four
graphs. Eager three-pass microbatches peak at 12.74 GiB raw at `r = 1` and
4.02 GiB checkpointed at `r = 8`; the one-pass microbatch at `r = 8` peaks
at 13.11 GiB raw. Replay per four-row microbatch, in milliseconds, with the
trainer's activation policy checkpointing every mode deeper than ten
cell-passes:

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

## Larger geometry

The larger loop, the larger geometry with its middle four cells tied, is in
[scaling.md](scaling.md#larger-loop).

## The `L` letter: a shared core cache

`L` is `l` plus a second trained horizontal channel: a shared core cache
through which every core iteration of a fused position reads earlier
positions' final-iteration mixer writes. It is the recurrent-depth "attend to
later iterations of earlier tokens" mechanism trained rather than zero-shot.
It is specified here and not built; `parse_condition` does not accept it yet.
In the condition grammar a capital letter is its lowercase letter plus one
package, and `l` and `L` are exclusive occupants of the loop slot, so `arfL`
would name the full stack with the cache.

### Plain and fused positions

Every executed position is either **plain** or **fused**, and under `L` the
label governs both horizontal channels at once:

- A plain position seeds from its embedding and mixes **same-depth**: at core
  iteration `i` it reads earlier positions' iteration-`i` writes of the
  current pass.
- A fused position seeds from the FBT fusion and mixes **shared**: at every
  core iteration it reads earlier positions' **final-iteration** writes plus
  its own current-iteration write.

Plain positions always form a prefix of the row, so a plain position only ever
reads plain positions and same-depth mixing over the prefix is self-contained.
Position 0 is always plain. The assignment per context:

| Context | Plain | Fused |
|---|---|---|
| training pass 1 | every position | none |
| training pass `k ≥ 2` | the drawn prefix | the suffix |
| Standard decoding | prompt and every generated position | none |
| Soft decoding | the prompt prefill | every generated position |
| Fused decoding | the prompt prefill; position 0 of the fused prompt pass | the rest of the fused prompt pass; every generated position |

"Final" means, in training pass `k ≥ 2`, pass `k-1`'s last iteration at each
earlier position, whatever that position's label was in pass `k-1`; in
decoding, each earlier position's last executed iteration. The prefix therefore
keeps the meaning it has in `design.md`: a plain position computes exactly what
pass 1 computes. The price is that a feedback pass evaluates each core mixer
twice per iteration, once same-depth over the prefix and once shared, and
selects by position. At `r = 1`, `arfL` differs from `arf` only by this
shared mixing on feedback passes.

### Shared core cache

A position's **write** at a core layer is everything later positions consume
there. For PKDA: the pre-convolution Q/K/V projections that form later
positions' convolution history, the post-convolution `k_t` and `v_t`, and the
per-head transition controls `alpha_t`, `beta_t`, `alphaP_t`, `betaP_t`. For
gated GQA: the normalized `k_t` and `v_t`. The write bank of one pass is:

| Tensor, per core layer | Shape | Dtype |
|---|---|---|
| PKDA pre-convolution Q/K/V | `[B, T, 3 x 1280]` | BF16 |
| PKDA post-convolution K, V | `[B, T, 2 x 1280]` | BF16 |
| PKDA transition controls | `[B, T, 4 x 10]` | FP32 |
| GGQA K, V | `[B, T, 2 x 4 x 96]` | BF16 |

The bank is produced by the final core iteration of a pass at every position,
is part of the pass's `ColumnOutput`, stays in the autograd graph, and is
consumed by the next pass exactly as the payload is:

```python
outs = [forward_column(e, writes=None)]                 # pass 1: all plain
for k in range(2, K + 1):
    p = shift_right(outs[-1].payload + jitter[k - 2])
    x = where(plain, e, fuse(p, e))
    outs.append(forward_column(x, plain=plain, writes=outs[-1].writes))
```

**PKDA at a fused position.** Let `S̄_(t-1)`, `Ā_(t-1)` be the recurrent matrix
and preconditioner state accumulated over positions `< t` from the write bank
only, with the transition equations of `architecture.md`; at `t = 0` both are
zero. Position `t` then applies its own current-iteration transition:

```text
A_t        = alphaP_t Ā_(t-1) + betaP_t (k_t ⊙ k_t)
k_write_t  = B(A_t) ⊙ k_t
S_tilde_t  = Diag(alpha_t) S̄_(t-1)
S_t        = (I - beta_t k_write_t k_t^T) S_tilde_t + beta_t k_write_t v_t^T
o_t        = S_t^T q_t
           = S̄_(t-1)^T Diag(alpha_t) (q_t - beta_t (k_write_t · q_t) k_t)
             + beta_t (k_write_t · q_t) v_t
```

The bank-side scan does not depend on the iteration, so its exclusive states
and chunk factors are formed once per pass per layer; each iteration performs
an exclusive read at the modified query and adds the own term. The convolution
at position `t` uses its own current pre-convolution projections with the
bank's projections as history. This is a new operator, forward and backward,
with gradient flowing into the current writes and into the bank. Its parity
oracle is the literal sequential recurrence over the bank followed by the own
transition, run on the portable path.

**Gated GQA at a fused position.** Position `t` attends over the bank's `K̄`,
`V̄` at positions `< t` and over its own current `k_t`, `v_t`; the two softmax
pieces are merged in FP32 by log-sum-exp with the layer's usual scale, and at
`t = 0` only the own term exists. Gradient flows into both.

No jitter is added directly to write tensors; later-pass writes are functions
of the jittered payload seeds. Pass `k`'s loss reaches pass `k-1` through both
the payload and the bank, and neither is ever detached.

### Jacobi horizon

The shared cache inherits the payload's training horizon. A `k`-pass batch
exposes chains of at most `k-1` transitions through both channels, while
decoding runs unbounded chains in which each generated position's finals were
themselves produced from a cache of mixed age: prompt finals from the prefill,
generated finals from the decode trajectory. Training pass 2 reads pass-1
finals, pass 3 reads pass-2 finals produced against pass-1 finals, and no
training pass constructs a deeper joint distribution. This is the same mismatch
the payload already has, and a two-channel version of the self-composition
trace watches it: repeated fully fused prefill with prefix length 1 in which
both the shifted payload and the write bank of the previous iteration advance,
where a rising or oscillating trace outranks any one-step metric.

### Decoding and cost

Fused decoding under `L` keeps one compact core cache that advances once per
position after its final iteration, three cells in all; Standard decoding
keeps the `l` tracks. Training holds two write banks and the bank-side scan
intermediates on top of `l`'s activations, and each feedback pass evaluates
the core mixers twice per iteration, so `L` is more expensive than `l` in
every mode.

### Building it

Building `L` touches the following together:

- `CONDITION_LETTERS` gains `L` as the loop slot's superset letter, exclusive
  with `l`; `ModelConfig` gains the shared-cache flag; `ColumnOutput` gains
  the write bank; `forward_column` takes a per-position plain mask and a
  previous bank.
- The workspace FLA fork gains the fused-position PKDA operator, forward and
  backward; the gated GQA layer gains the two-piece merge. `delta probe`
  checks each against its portable oracle.
- The trainer threads the bank through the passes; the monitor gains the
  two-channel sequence trace.
- A checkpoint contract bump; the documents that describe `l` gain `L`.

## Sources

| Source | Adopted | Replaced here |
|---|---|---|
| Recurrent-depth language models | tied core between prelude and coda, log-normal Poisson iteration draw, same-depth per-iteration caches, effective-depth accounting | adapter by router reads of the prelude output at every core site; random initial state by the prelude output; sandwich norms by the unchanged pre-norm shell; zero-shot warm start by the trained payload; truncated backpropagation by full backpropagation; zero-shot cache sharing kept for `L` as a trained channel with a budget of one |
| Full-Bandwidth Transformer | asymmetric fusion, payload, Jacobi passes, prefix mixin, jitter, pass mixture, loss | — |
| This project | per-cell core deltas accumulated across iterations, iteration-tagged route sites and cache tracks, cell-token accounting, the depth trace; for `L`, plain/fused position labels governing both channels and the fused-position PKDA read with own-term substitution | — |

## Not in this specification

Left out: truncated backpropagation; branch-output or residual-state norms in
the core; a random or learned initial core state; any warm start of the core
from the previous column, trained or zero-shot; a raw embedding path into the
core; a payload landing at the core entry, which is a different design rather
than a loop variant; a shared-cache budget above one; per-iteration halting
readouts and pause tokens, which belong to a separate specification; adaptive
exit as anything but a diagnostic; and a decode `r` that varies within a
request. `l` on the plain
trunk or without `r` builds and pairs like every other subset, and is
unstudied: without `r` the core has no input injection at all.

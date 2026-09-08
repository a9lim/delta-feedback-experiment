# The `l` letter: a tied-depth loop

This document specifies the `l` condition letter and its full-stack condition
`arfl`, a candidate next organism: the small model with a weight-tied
recurrent core, so that repeated within-column computation exists to look at.
The letter parses today and `DeltaModel` refuses to build it; the list at the end
is what building it touches. Where this document and
[architecture.md](architecture.md) both speak, this one describes the loop
only.

`arfl` combines two recurrences on the `a` trunk:

- **Vertical.** A tied four-layer core is iterated `r` times per column, with
  `r` drawn per optimizer step. Recurrent-depth language models supply the
  loop, the iteration draw, and the shared-cache idea.
- **Horizontal.** The previous column reaches the current one through two
  trained channels: the FBT payload at the seed, and a shared core cache that
  every core iteration of a fused position reads. FBT supplies the payload,
  its fusion, and the Jacobi multi-pass training that makes both channels
  trainable in parallel.

## Why

The tied core adds a second axis of recurrence: what changes within a column
as the same weights are reused, what an intervention at a particular iteration
does, and which information is newly computed versus carried. The horizontal
payload and the shared cache let that computation persist across tokens
through two trained channels instead of one. The current `arf` channel turned
out token-legible ([findings.md](findings.md)); a core that refines a column
over several iterations before the payload is emitted is one candidate way to
give the channel something to carry. The shared cache, the source-bank
changes, and the tied depth are a package, and at `r = 1` the loop is exactly
`arf`, so the pieces can be switched on separately. Every iteration-indexed
trace says whether its index is core depth, Jacobi pass number, or token
position.

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
column executes the same twelve layers as `arf` with the same parameters and the
same source banks; plain positions then coincide with `arf` exactly, and fused
positions differ from `arf` only by the shared mixing defined below.

A `k`-pass batch at iteration count `r` costs `k (2 + r)` cell evaluations per
predicted token. Results report **cell-tokens** beside pass-tokens: `arf` spends
three cell-tokens per pass-token, `arfl` spends `2 + r`.

## Column

```text
token embedding e_t -------------------+
                                       +-- FBT fuse --> seed s_t
previous payload p_(t-1) --------------+                  |
                                                          v
        +----------------------- prelude P ------------------------+  Delta_P
        |                                                          |
        |   +--------------- core R, tied, r iterations --------+  |
        |   | iteration i reads: seed, Delta_P, core partial,   |  |  Delta_R
        |   |                    core cache (mode by position)  |  |
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
    h = core_cell(h, sources=[seed, Delta_P], entry=core_entry, cache=core_cache)
Delta_R = h - core_entry

h = coda_cell(h, sources=[seed, Delta_P, Delta_R])
Delta_C = h - (core_entry + Delta_R)

payload = payload_norm(h + mhdb([seed, Delta_P, Delta_R, Delta_C], site="payload"))
logits = tied_readout(final_norm(h))
```

Each `mhdb` call prepends its site's learned null. Every cell measures its
partial from its own entry; the core's entry is pinned to the prelude output
for all `r` iterations, so the core is one cell of `4r` layers whose partial
accumulates across iterations. There is no adapter and no random initial state:
the seed and prelude delta are re-readable at every iteration through the
routers. At a fused position the token enters only through the FBT gate, as in
`arf`; a plain position seeds from the embedding itself.

## Residual shell in the core

Every cell, the core included, keeps the pre-norm shell of `architecture.md`
with branch scale `1/sqrt(2L)` at `L = 12`, the unique layer count. No
branch-output or residual-state norm is added, and the telescoping identity
holds in every cell:

```text
h_top = seed + Delta_P + Delta_R + Delta_C
```

The shell is chosen because it has no mechanism that forces a nonzero update: a
branch-output norm would pin every iteration's update size and make a fixed
point unreachable by construction. Whether the tied core actually contracts is
not assumed. Hyperball radii bound the NorMuonH matrices, but norm scales,
router values, and the NAdam-owned gates and controls are not radius-bounded,
so whether it contracts is measured: per-iteration residual RMS and the
depth-contraction trace below are the guards, and a diverging trace means the
specification changes.

## Iteration count

`r` is drawn once per optimizer step and shared by every pass of that step:

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
Realized `r` is logged per step and enters the cell-token total.

Evaluation and decoding use one fixed `r` for the prompt passes and the entire
decode trajectory; `r` never changes within a request. Metrics use
`r = r_mean`; a fixed-`r` sweep over `{1, 2, 4, 8}` is a diagnostic.

## MHDB source banks

Routers follow `architecture.md` exactly: zero-initialized width-`D` query,
full-width RMS key statistics, raw values, one softmax per routing group, a
site-local null prepended to every bank. The core's eight sites are shared
across iterations because they are core weights. The core is one cell, so its
bank is a cell's bank:

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
a `arf` second-cell bank at every iteration, and the tied routers answer one
stationary question rather than one per iteration index. Every bank's non-null
sources reconstruct the current residual exactly, as in `architecture.md`, so
the uniform zero-query mixture stays collinear with the residual and pass-1
routing is functionally inert at initialization at every iteration. The
residual is the state; the routers are the learned input injection of the seed
and the prelude. Per-iteration deltas are never sources: an iteration boundary
is a boundary in weights, not in state.

The core partial is exactly absent at one site, the attention entry of
iteration 1, where the core has not yet moved. The fixed-capacity router there
carries a zero-valued placeholder together with a boolean source presence mask;
an absent source has logit `-inf`, contributes no value, and receives no
gradient. The mask is load-bearing rather than cosmetic: the routed mixture is
added to the residual before the norm, so an unmasked zero source at logit zero
would take softmax mass `1 / (S + 1)` from the live sources, `S` the sum of
their exponentiated scores, and shrink the routed term relative to `h` once the
query has trained. Router weights, values, and gradients at `r = 1` must match
`arf` exactly, and the telemetry labels the slot `core_partial`.

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
| warm start from the previous token's `s_r` | the FBT payload at the seed, and the shared core cache at fused positions |

The two designs differ in one structural place: where the previous position's
state enters. The recurrent-depth warm start hands the core its previous fixed
point directly. Here the previous column enters at the seed, through the
token-gated FBT fusion, and the prelude reprocesses the fused seed before the
core starts, so the vertical iteration restarts from the prelude output at
every column. The previous column's converged core state reaches the current
core through two trained channels only: the payload, compressed by the payload
norm and the token gate, and the shared cache, which is the recurrent-depth
"attend to later iterations of earlier tokens" mechanism trained rather than
zero-shot.

Two expressivity gaps against the adapter are known and accepted:

- The router is a per-group convex mixture of raw values, and the branch's
  projections are shared across every source; the adapter is a dedicated
  linear map that can transform `e` independently of the state. The
  recurrent-depth paper found additive re-injection matched concatenation at
  small scale and lost at scale; the router is closer to addition with
  learned gates.
- The core carries no state across positions in its residual. Whether the
  restart costs anything is what the warm-start sweep measures.

Landing the payload at the core entry instead would move the horizontal
channel for the whole family and break `r = 1` parity with `arf`. That is the
entry-redesign branch in [findings.md](findings.md), a different design rather
than a loop variant, and is not specified here.

## Horizontal channels

### Payload

Unchanged from `architecture.md` and `design.md`: asymmetric FBT fusion at the
seed, routed payload over the null, seed, and completed deltas, `payload_norm`,
keyed uniform jitter, the one-position shift, and the per-row plain prefix. The
payload is the trained form of warm-starting a column from the previous
column's final state; no untrained warm start exists.

### Plain and fused positions

Every executed position is either **plain** or **fused**, and the label governs
both horizontal channels at once:

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
selects by position.

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
the payload already has, and the same trace watches it: the
sequence-contraction trace over both channels, where a rising or oscillating
trace outranks any one-step metric.

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

Backpropagation is complete: every iteration of every pass is in the graph,
with block-level activation checkpointing at every pass. Truncated
backpropagation through the last iterations is excluded.

Optimizer ownership follows `architecture.md`: the tied core's matrices are
NorMuonH parameters whose gradients sum across iterations and passes, with one
fixed Hyperball radius each. The global FP32 gradient is clipped to norm
10.0 before both steps.

Execution follows the [runtime graph contract](runtime-qualification.md): one
fixed-address train CUDA graph per reachable `(pass count, r)` pair,
twenty-four at the screen, all sharing one pool sized by the three-pass `r_max`
case, plus the no-grad evaluation graphs at `r = r_mean`. Preparation therefore
costs about six times today's capture. Autograd sums the tied gradients inside
the captured step; no per-cell graph composition is used.

`arfl` pairs with `arf`: byte-identical initialization of every shared
parameter, the same stream, row order, schedule, and pass-count, prefix, and
jitter draws. The `r` draw is the only additional randomness.

## Evaluation and diagnostics

`val` is pass-1 held-out cross-entropy at `r = r_mean`, every position plain.
`val_fused` is a second pass with plain-prefix length 1 at `r = r_mean`, so
every position but the first is fused. Both report cell-tokens.

`arfl - arf` at equal steps is a **matched-data** contrast on the complete
loop package: vertical depth, the shared cache, and the changed source banks
together, at unequal compute. A matched-compute view compares checkpoints or
curves at equal cumulative cell-tokens or measured device-time; equal
pass-tokens are never matched compute for this condition. Depth alone is not
attributable without the unrouted loop, `afl`.

Two stability traces go with any loop run:

- **Depth contraction.** Run one column pass over held-out rows in a fixed
  label assignment (all plain, and prefix-1 fused) at `r = r_max`, record
  `||h^(i) - h^(i-1)||_2` per position for every iteration, the increment of
  the core partial, and record the loss with the coda applied after each
  iteration `i`. The loss curve is the fixed-`r` sweep.
- **Sequence contraction.** Repeated fully fused prefill with prefix length 1
  in which **both** channels advance each iteration: the shifted payload and
  the write bank of the previous iteration. Record held-out loss and mean
  `||h_top^(j) - h_top^(j-1)||_2` per iteration: eight iterations in the
  monitor, thirty in analysis. The existing payload-only iteration is not
  this trace.

Additional diagnostics: residual RMS per iteration; the core routers' mass on
the null, seed, `Delta_P`, and the core partial by iteration, which is the
learned input-injection profile; the own-term share of PKDA output at fused
positions. The recurrent-depth KL exit rule can run as a diagnostic; metrics
use fixed `r`.

One same-checkpoint diagnostic is the **warm-start sweep**, the recurrent-depth
warm start applied zero-shot to a model trained without it. At fused evaluation
with prefix length 1, every fused position's core starts from
`h_entry + lambda * Delta_R` of the previous pass at the position before it,
shifted exactly as the payload is, for `lambda` in `{0, 0.5, 1}`. The warm
term counts as core progress: the core partial is still measured from the prelude output, so
iteration 1's attention entry reads a live partial and the reconstruction
identity holds. The fixed-`r` sweep is repeated at each `lambda`. A gain at
`lambda > 0` says more core-state continuity is worth trying as a trained
channel; no gain leaves the question open, since co-adaptation, injection
scale, and the task can all limit a zero-shot intervention.

## Parameter, cache, and cost accounting

`arfl` has exactly the parameters of `arf`: 257,514,792 total and 140,827,944
active non-embedding.

Per sequence at 1,024 context, one cell's mixer cache is
3.456 MiB (three FP32 PKDA matrix and diagonal states plus BF16 convolution
histories, 1.956 MiB; one BF16 KV cache, 1.500 MiB). Standard decoding keeps
`r` live per-iteration tracks for the core, allocated at `r_max`; fused
decoding keeps one compact core cache that advances once per position after its
final iteration:

| Decode mode | Cells cached | Cache |
|---|---:|---:|
| Standard at `r_max` | `1 + 8 + 1` | 34.6 MiB |
| Soft and Fused | 3 | 10.4 MiB |

Training memory is not established by this arithmetic. The three-pass `r_max`
batch holds 120 checkpointed block inputs, two write banks, and the bank-side
scan intermediates on a card whose qualified pool already reserves
23.0 GiB; measure it in stages before capturing graphs.

Jobe step times are estimates, not measurements. The anchor is the measured
14.21 s three-pass `arf` step in the short
[runtime qualification](runtime-qualification.md), approximated conservatively
as `k x (1.3 s + 0.30 s per layer evaluation)` per step, with block
checkpointing adding one third to the layer term. At the expected `E[r] =
3.88`:

| Condition | One pass | Two passes | Three passes | Mean step | Full schedule |
|---|---:|---:|---:|---:|---:|
| `arf` | 5.0 s | 9.9 s | 14.9 s | 6.3 s | ~19 h |
| `arfl` | 10.8 s | 21.7 s | 32.5 s | 13.9 s | ~41 h |

The model assigns no cost to the bank-side scan, the write bank, the two-piece
attention merge, the new backward, or the second mixer evaluation on feedback
passes, so the loop figure is a lower bound until eager and captured prototypes
are measured.

## Larger geometry

The wider loop budget and geometry are in [scaling.md](scaling.md#larger-loop).

## Sources

| Source | Adopted | Replaced here |
|---|---|---|
| Recurrent-depth language models | tied core between prelude and coda, log-normal Poisson iteration draw, cache shared across iterations, effective-depth accounting | adapter by router reads of the prelude output at every core site; random initial state by the prelude output; sandwich norms by the unchanged pre-norm shell; untrained warm start by the trained payload and shared cache, kept only as the warm-start sweep; truncated backpropagation by full backpropagation; a cache budget of several slots by a trained budget of one |
| Full-Bandwidth Transformer | asymmetric fusion, payload, Jacobi passes, prefix mixin, jitter, pass mixture, loss | — |
| This project | one-cell core partial, plain/fused position labels governing both channels, fused-position PKDA read with own-term substitution, cell-token accounting, warm-start sweep | — |

## Not in this specification

Left out: `l` on any condition but `arf` (`al`, `arl`, `afl`, and the
plain-trunk loops parse and are unspecified); truncated backpropagation;
branch-output or residual-state norms in the core; a random or learned initial
core state; an untrained warm start anywhere but the warm-start sweep; a raw
embedding path into the core; a payload landing at the core entry, which is a
different design rather than a loop variant; a shared-cache budget above one;
per-iteration halting readouts and pause tokens, which belong to a separate
specification; adaptive exit as anything but a diagnostic; a decode `r` that
varies within a request; and any loop mapping onto the six-cell larger
reference.

## Building it

Building `l` touches the following together:

- `ModelConfig.loop` gains the core recurrence, `r_mean`, and `r_max`, and
  `DeltaModel` stops refusing it; `ColumnOutput` gains the write bank and
  `forward_column` takes a per-position plain mask and a previous bank;
  `AGENTS.md` and `README.md` mark `l` built.
- A checkpoint contract beyond v24 with `r_mean`, `r_max`, and the `r`
  sub-stream as state-defining fields; realized `r` and cell-tokens in the run
  log.
- The workspace FLA fork gains the fused-position PKDA operator, forward and
  backward; the gated GQA layer gains the two-piece merge; the router gains the
  presence mask. `delta probe` checks each against its portable oracle and
  checks `r = 1` value, weight, and gradient parity with `arf`.
- A staged memory measurement before any graph work: eager `r = 1`, eager
  `r_max`, three-pass backward at `r_max`, then the captured `(k, r)` graph
  family; allocated and reserved memory reported separately, and the pool
  reservation recorded.
- `design.md` adds the `l` counts to the condition table, cell-tokens and the matched-data
  versus matched-compute definitions to the accounting, `r = r_mean` to the
  evaluation section, and the two-channel sequence trace in place of the
  payload-only iteration for loop conditions.

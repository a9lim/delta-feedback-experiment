# Depth-recurrent architecture (`df-loop`)

This document specifies `df-loop`, a sixth arm that adds a weight-tied
recurrent core to Hard Delta Feedback. It is fully specified and not yet
implemented; it is not part of the registered screen until the adoption list
at the end has landed. Where this document and [architecture.md](architecture.md)
both speak, this document governs the loop arm only.

`df-loop` combines two recurrences on one hybrid trunk:

- **Vertical.** A tied four-layer core is iterated `r` times per column, with
  `r` drawn per optimizer step. Recurrent-depth language models supply the
  loop, the iteration draw, and the shared-cache idea.
- **Horizontal.** The previous column reaches the current one through two
  trained channels: the FBT payload at the seed, and a shared core cache that
  every core iteration of a fused position reads. FBT supplies the payload,
  its fusion, and the Jacobi multi-pass training that makes both channels
  trainable in parallel.

## Geometry

The loop arm keeps the screen geometry in [design.md](design.md) (width 768,
twelve unique layers, three cells, vocabulary 151,936) and reinterprets the
three cells:

| Field | `df-loop` value |
|---|---:|
| Prelude `P` | one cell, layers 0–3 |
| Recurrent core `R` | one cell, layers 4–7, weight-tied across iterations |
| Coda `C` | one cell, layers 8–11 |
| Iterations per column `r` | drawn per step, `1 ≤ r ≤ 8` |
| Mean iteration rate `r_mean` | 4 |
| Compute depth per pass | `4 + 4r + 4` layers, `2 + r` cells |
| Unique parameters | identical to `df` |

Every cell is `[PKDA, PKDA, PKDA, gated global GQA]` with the mixer, gate,
routing-group, and precision contracts of `architecture.md`. At `r = 1` the
column executes the same twelve layers as `df` with the same parameters and
the same source banks; plain positions then coincide with `df` exactly, and
fused positions differ from `df` only by the shared mixing defined below.

A `k`-pass batch at iteration count `r` costs `k (2 + r)` cell evaluations
per predicted token. Results report **cell-tokens** beside pass-tokens: `df`
spends three cell-tokens per pass-token, `df-loop` spends `2 + r`.

## Column

```text
token embedding e_t -------------------+
                                       +-- FBT fuse --> seed s_t
previous payload p_(t-1) --------------+                  |
                                                          v
        +----------------------- prelude P ------------------------+  Delta_P
        |                                                          |
        |   +--------------- core R, tied, r iterations --------+  |
        |   | iteration i reads: seed, Delta_P, Delta_R^(<i),   |  |  Delta_R^(i)
        |   |                    core cache (mode by position)  |  |
        |   +---------------------------------------------------+  |
        |                                                          |
        +------------------------- coda C --------------------------+  Delta_C
                          |                         |
                       h_top                    block deltas
                          +------------+------------+
                                       v
                              routed DF payload p_t
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
    prior = [h - core_entry] if i > 0 else []        # Delta_R^(<i), the core's progress
    h = core_cell(h, sources=[seed, Delta_P, *prior], cache=core_cache)
Delta_R = h - core_entry

h = coda_cell(h, sources=[seed, Delta_P, Delta_R])
Delta_C = h - (core_entry + Delta_R)

payload = payload_norm(h + mhdb([seed, Delta_P, Delta_R, Delta_C], site="payload"))
logits = tied_readout(final_norm(h))
```

Each `mhdb` call prepends its site's learned null. There is no adapter and no
random initial state: the core's entry state is the prelude output, and the
seed and prelude delta are re-readable at every iteration through the
routers. At a fused position the token enters only through the FBT gate, as
in `df`; a plain position seeds from the embedding itself.

## Residual shell in the core

Every cell, the core included, keeps the pre-norm shell of `architecture.md`
with branch scale `1/sqrt(2L)` at `L = 12`, the unique layer count. No
branch-output or residual-state norm is added, and the telescoping identity
holds in every cell:

```text
h_top = seed + Delta_P + sum_i Delta_R^(i) + Delta_C
```

The shell is chosen because it has no mechanism that forces a nonzero update:
a branch-output norm would pin every iteration's update size and make a
fixed point unreachable by construction. Whether the tied core actually
contracts is not assumed. Hyperball radii bound the NorMuonH matrices, but
norm scales, router values, and the NAdam-owned gates and controls are not
radius-bounded, so the safety argument is the measured one: per-iteration
residual RMS and the depth-contraction trace below are registered guards, and
a diverging trace invalidates the loop comparison and forces a new
specification.

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

The draw comes from its own keyed sub-stream of the data seed and step, so
`df` and `df-loop` share byte-identical pass-count, prefix, and jitter draws.
Realized `r` is logged per step and enters the cell-token total.

Evaluation and decoding use one fixed `r` for the prompt passes and the
entire decode trajectory; `r` never changes within a request. Registered
metrics use `r = r_mean`. A fixed-`r` sweep over `{1, 2, 4, 8}` at the screen
is a diagnostic, not a metric.

## MHDB source banks

Routers follow `architecture.md` exactly: zero-initialized width-`D` query,
full-width RMS key statistics, raw values, one softmax per routing group, a
site-local null prepended to every bank. The core's eight sites are shared
across iterations because they are core weights. What changes is the bank:

| Site | Sources after the null |
|---|---|
| prelude, attention entry | seed |
| prelude, other sublayers | seed, partial |
| core iteration 1, attention entry | seed, `Delta_P` |
| core iteration `i > 1`, attention entry | seed, `Delta_P`, `Delta_R^(<i)` |
| core, other sublayers | the entry bank plus the current partial |
| coda, attention entry | seed, `Delta_P`, `Delta_R` |
| coda, other sublayers | seed, `Delta_P`, `Delta_R`, partial |
| payload | seed, `Delta_P`, `Delta_R`, `Delta_C` |

`Delta_R^(<i)` is the core's completed progress before iteration `i`, the sum
of the earlier iteration deltas; `Delta_R` is the whole core's completed
delta. Every bank's non-null sources therefore reconstruct the current
residual exactly, as in `architecture.md`, so the uniform zero-query mixture
stays collinear with the residual and pass-1 routing is functionally inert at
initialization at every iteration. The core reads a fixed-size window rather
than every past iteration: the residual is the state, the window is only the
read, and the routers act as the learned input injection of the seed and the
prelude.

At iteration 1 the `Delta_R^(<i)` slot is absent. The fixed-capacity router
carries a zero-valued placeholder in that slot together with a boolean source
presence mask; an absent source has logit `-inf`, contributes no value, and
receives no gradient. Router weights, values, and gradients at `r = 1` must
match `df` exactly, and the telemetry labels the slot `core_prior`.

## Horizontal channels

### Payload

Unchanged from `architecture.md` and `design.md`: asymmetric FBT fusion at the
seed, routed payload over the null, seed, and completed deltas,
`payload_norm`, keyed uniform jitter, the one-position shift, and the per-row
plain prefix. The payload is the trained form of warm-starting a column from
the previous column's final state; no untrained warm start exists.

### Plain and fused positions

Every executed position is either **plain** or **fused**, and the label
governs both horizontal channels at once:

- A plain position seeds from its embedding and mixes **same-depth**: at core
  iteration `i` it reads earlier positions' iteration-`i` writes of the
  current pass.
- A fused position seeds from the FBT fusion and mixes **shared**: at every
  core iteration it reads earlier positions' **final-iteration** writes plus
  its own current-iteration write.

Plain positions always form a prefix of the row, so a plain position only
ever reads plain positions and same-depth mixing over the prefix is
self-contained. Position 0 is always plain. The assignment per context:

| Context | Plain | Fused |
|---|---|---|
| training pass 1 | every position | none |
| training pass `k ≥ 2` | the drawn prefix | the suffix |
| Standard decoding | prompt and every generated position | none |
| Soft decoding | the prompt prefill | every generated position |
| Fused decoding | the prompt prefill; position 0 of the fused prompt pass | the rest of the fused prompt pass; every generated position |

"Final" means, in training pass `k ≥ 2`, pass `k-1`'s last iteration at each
earlier position, whatever that position's label was in pass `k-1`; in
decoding, each earlier position's last executed iteration. The prefix
therefore keeps the meaning it has in `design.md`: a plain position computes
exactly what pass 1 computes. The price is that a feedback pass evaluates
each core mixer twice per iteration, once same-depth over the prefix and once
shared, and selects by position.

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

**PKDA at a fused position.** Let `S̄_(t-1)`, `Ā_(t-1)` be the recurrent
matrix and preconditioner state accumulated over positions `< t` from the
write bank only, with the transition equations of `architecture.md`; at
`t = 0` both are zero. Position `t` then applies its own current-iteration
transition:

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
an exclusive read at the modified query and adds the own term. The
convolution at position `t` uses its own current pre-convolution projections
with the bank's projections as history. This is a new operator, forward and
backward, with gradient flowing into the current writes and into the bank.
Its parity oracle is the literal sequential recurrence over the bank followed
by the own transition, run on the portable path.

**Gated GQA at a fused position.** Position `t` attends over the bank's `K̄`,
`V̄` at positions `< t` and over its own current `k_t`, `v_t`; the two softmax
pieces are merged in FP32 by log-sum-exp with the layer's usual scale, and
at `t = 0` only the own term exists. Gradient flows into both.

No jitter is added directly to write tensors; later-pass writes are functions
of the jittered payload seeds. Pass `k`'s loss reaches pass `k-1` through both
the payload and the bank, and neither is ever detached.

### Jacobi horizon

The shared cache inherits the payload's training horizon. A `k`-pass batch
exposes chains of at most `k-1` transitions through both channels, while
decoding runs unbounded chains in which each generated position's finals were
themselves produced from a cache of mixed age: prompt finals from the
prefill, generated finals from the decode trajectory. Training pass 2 reads
pass-1 finals, pass 3 reads pass-2 finals produced against pass-1 finals, and
no training pass constructs a deeper joint distribution. This is the same
mismatch `design.md` already accepts for the payload, and it is gated the same
way: the sequence-contraction trace over both channels is the registered
test, and a rising or oscillating trace invalidates the feedback comparison
regardless of any one-step metric.

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
NorMuonH parameters whose gradients sum across iterations and passes, with
one fixed Hyperball radius each. The global FP32 gradient is clipped to norm
1.0 before both steps.

Execution keeps the whole-step capture of `design.md`: one fixed-address
train CUDA graph per reachable `(pass count, r)` pair, twenty-four at the
screen, all sharing one pool sized by the three-pass `r_max` case, plus the
no-grad evaluation graphs at `r = r_mean`. Preparation therefore costs about
six times today's capture. Autograd sums the tied gradients inside the
captured step; no per-cell graph composition is used.

`df-loop` pairs with `df`: byte-identical initialization of every shared
parameter, the same stream, row order, schedule, and pass-count, prefix, and
jitter draws. The `r` draw is the only additional randomness.

## Evaluation and diagnostics

`val` is pass-1 held-out cross-entropy at `r = r_mean`, every position plain.
`val_fused` is a second pass with plain-prefix length 1 at `r = r_mean`, so
every position but the first is fused. Both report cell-tokens.

`df-loop - df` at equal steps is a **matched-data** contrast on the complete
loop package: vertical depth, the shared cache, and the changed source
banks together, at unequal compute. A matched-compute view compares
checkpoints or curves at equal cumulative cell-tokens or measured
device-time; equal pass-tokens are never matched compute for this arm.
Depth alone is not attributable without an unrouted loop arm.

A loop result is uninterpretable without two stability traces:

- **Depth contraction.** Run one column pass over held-out rows in a fixed
  label assignment (all plain, and prefix-1 fused) at `r = r_max`, record
  `||h^(i) - h^(i-1)||_2` per position for every iteration, and record the
  loss with the coda applied after each iteration `i`. The loss curve is the
  fixed-`r` sweep.
- **Sequence contraction.** Repeated fully fused prefill with prefix length 1
  in which **both** channels advance each iteration: the shifted payload and
  the write bank of the previous iteration. Record held-out loss and mean
  `||h_top^(j) - h_top^(j-1)||_2` per iteration: eight iterations in the
  monitor, at least thirty for promotion. The existing payload-only iteration
  is not this trace.

Additional registered diagnostics: residual RMS per iteration; the core
routers' mass on seed, `Delta_P`, and `Delta_R^(<i)` by iteration, which is
the learned input-injection profile; the own-term share of PKDA output at
fused positions. Routing summaries remain diagnostics, not wins. The
recurrent-depth KL exit rule may be run as a diagnostic; adaptive exit is not
part of any metric.

## Parameter, cache, and cost accounting

`df-loop` has exactly the parameters of `df`: 257,514,792 total and
140,827,944 active non-embedding.

Per sequence at 1,024 context, one cell's registered mixer cache is
3.456 MiB (three FP32 PKDA matrix and diagonal states plus BF16 convolution
histories, 1.956 MiB; one BF16 KV cache, 1.500 MiB). Standard decoding keeps
`r` live per-iteration tracks for the core, allocated at `r_max`; fused
decoding keeps one compact core cache that advances once per position after
its final iteration:

| Decode mode | Cells cached | Cache |
|---|---:|---:|
| Standard at `r_max` | `1 + 8 + 1` | 34.6 MiB |
| Soft and Fused | 3 | 10.4 MiB |

Training memory is not established by this arithmetic. The three-pass
`r_max` batch holds 120 checkpointed block inputs, two write banks, and the
bank-side scan intermediates on a card whose qualified pool already reserves
22.98 GiB; the adoption list requires a staged memory gate.

Jobe step times are **unqualified estimates**. The anchor is the measured
14.8 s three-pass `df` step of the current kernels, read as
`k x (1.3 s + 0.30 s per layer evaluation)` per step, with block
checkpointing adding one third to the layer term. At the expected
`E[r] = 3.88`:

| Arm | One pass | Two passes | Three passes | Mean step | Full schedule |
|---|---:|---:|---:|---:|---:|
| `df` | 5.0 s | 9.9 s | 14.9 s | 6.3 s | ~19 h |
| `df-loop` | 10.8 s | 21.7 s | 32.5 s | 13.9 s | ~41 h |

The model assigns no cost to the bank-side scan, the write bank, the
two-piece attention merge, the new backward, or the second mixer evaluation
on feedback passes, so the loop figure is a lower bound until eager and
captured prototypes are measured.

## Flagship-width loop run

Beyond the screen, one loop run is outlined at flagship width with the
screen's naive depth: the same three cells, one each for prelude, core, and
coda, with the flagship's layer geometry from `architecture.md` and a deeper
draw. It is a loop counterpart to the Stage 3 plan in `design.md`, not a
replacement for the flagship, and it is gated the same way.

| Field | Value |
|---|---:|
| Residual width / SwiGLU intermediate | 1,536 / 6,656 |
| PKDA heads x width, projection width | 20 x 128, 2,560 |
| Global query / KV heads, head width | 16 / 8, 96 |
| Unique layers / cells | 12 / 3 |
| Context | 8,192 |
| `r_mean` / `r_max` | 16 / 32 |
| Compute depth per pass | `4 + 4r + 4` layers, mean 70.2, cap 136 |

| Component | Parameters |
|---|---:|
| Tied embedding and readout | 233,373,696 |
| 12 SwiGLU channel mixers | 368,050,176 |
| 9 PKDA mixers | 152,148,816 |
| 3 gated global GQA mixers, including Q/K norms | 28,312,128 |
| Hard-DF fusion | 4,718,592 |
| Trunk, entry, and payload norms | 43,008 |
| 24 within-column routers and one payload router | 115,200 |
| **Total** | **786,761,616** |
| **Active non-embedding** | **553,387,920** |

Under the cap the draw gives `E[r] = 15.56`, median 14, `P(r = 1) = 0.1%`,
`P(r = 32) = 5.8%`, and 17.56 expected cells per pass. Metrics use fixed
`r = 16`; the sweep runs to 32.

The run inherits the flagship recipe: 400 predicted tokens per active
non-embedding parameter aligned to the flagship batch of 40 sequences by
8,192 predictions, the same optimizer, pass mixture, and schedule shape, and
the same 8xH100 DDP target with every block checkpointed on every pass.

| Quantity | Value |
|---|---:|
| Global batch | 327,680 predictions |
| Optimizer steps | 675,523 |
| Exact aligned budget | 221,355,376,640 predicted tokens (400.000377 per active parameter) |
| Warmup / stable heat / cooldown | steps 1–13,510 / 13,511–540,418 / 540,419–675,523 |
| Feedback boundary | after step 506,642 |
| Expected pass-tokens | approximately 283.3B |
| Expected cell-tokens | approximately 4.97T |

The flagship spends about 3.39T cell-tokens, so this run costs about 1.5x the
flagship's compute with 50% of its active parameters. Per sequence at 8,192
context one cell's mixer cache is 27.9 MiB: Standard decoding at `r_max`
holds 34 cells, 949 MiB, and Soft or Fused decoding holds 3 cells, 83.7 MiB.

Entry requires an admissible `df-loop - df` screen result, the Prime gates
of `design.md`, the loop-specific gates in the adoption list, and explicit
spend confirmation.

## Sources

| Source | Adopted | Replaced here |
|---|---|---|
| Recurrent-depth language models | tied core between prelude and coda, log-normal Poisson iteration draw, cache shared across iterations, effective-depth accounting | adapter by MHDB reads of seed and `Delta_P`; random initial state by the prelude output; sandwich norms by the unchanged pre-norm shell; untrained warm start by the trained payload; truncated backpropagation by full backpropagation; a cache budget of several slots by a trained budget of one |
| Full-Bandwidth Transformer | asymmetric fusion, payload, Jacobi passes, prefix mixin, jitter, pass mixture, loss | — |
| This project | cumulative-progress source window, plain/fused position labels governing both channels, fused-position PKDA read with own-term substitution, cell-token accounting | — |

## Exclusions

Not part of this specification: loop variants of `base`, `mhdb`, or `fbt`;
truncated backpropagation; branch-output or residual-state norms in the core;
a random or learned initial core state; a raw embedding path into the core; a
shared-cache budget above one; per-iteration halting readouts and pause
tokens, which belong to a separate specification; adaptive exit as anything
but a diagnostic; a decode `r` that varies within a request; and any loop
mapping onto the six-cell flagship.

## Adoption

Registering `df-loop` changes the following together:

- `ARMS` gains `df-loop`; `ModelConfig` gains the core recurrence, `r_mean`,
  and `r_max`; `ColumnOutput` gains the write bank and `forward_column` takes
  a per-position plain mask and a previous bank; `AGENTS.md` and `README.md`
  state six arms.
- Checkpoint contract v20 with `r_mean`, `r_max`, and the `r` sub-stream as
  state-defining fields; realized `r` and cell-tokens in the run log.
- The workspace FLA fork gains the fused-position PKDA operator, forward and
  backward; the gated GQA layer gains the two-piece merge; the router gains
  the presence mask. `df probe` gates each against its portable oracle and
  checks `r = 1` value, weight, and gradient parity with `df`.
- A staged memory gate before any graph work: eager `r = 1`, eager `r_max`,
  three-pass backward at `r_max`, then the captured `(k, r)` graph family;
  allocated and reserved memory reported separately, and the pool
  reservation recorded.
- `design.md` adds `df-loop` to the arm table, cell-tokens and the
  matched-data versus matched-compute definitions to the accounting, two
  registered runs (`df-loop`, seeds 1 and 2) to Stage 1, `r = r_mean` to the
  evaluation contract, the two-channel sequence trace in place of the
  payload-only iteration for loop arms, and the flagship-width loop run to
  the scale plan.

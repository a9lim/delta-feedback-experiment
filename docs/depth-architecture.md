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
  every core iteration reads. FBT supplies the payload, its fusion, and the
  Jacobi multi-pass training that makes both channels trainable in parallel.

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
column executes the same twelve layers as `df` with the same parameters, and
`df-loop` at `r = 1` is `df`.

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
        |   | iteration i reads: seed, Delta_P, Delta_R^(i-1),  |  |  Delta_R^(i)
        |   |                    shared core cache              |  |
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
prev = None
for i in range(r):                                   # same r for every pass of a step
    entry = h
    h = core_cell(h, sources=[seed, Delta_P] + ([prev] if prev is not None else []),
                  mixing=mode)                       # same-depth or shared, see below
    prev = h - entry                                 # Delta_R^(i)
Delta_R = h - core_entry                             # sum of the iteration deltas

h = coda_cell(h, sources=[seed, Delta_P, Delta_R])
Delta_C = h - (core_entry + Delta_R)

payload = payload_norm(h + mhdb([seed, Delta_P, Delta_R, Delta_C], site="payload"))
logits = tied_readout(final_norm(h))
```

Each `mhdb` call prepends its site's learned null. There is no adapter, no
random initial state, and no raw-embedding bypass: the core's entry state is
the prelude output, the token enters only through the FBT gate, and the seed
and prelude delta are re-readable at every iteration through the routers.

## Residual shell in the core

Every cell, the core included, keeps the pre-norm shell of `architecture.md`
with branch scale `1/sqrt(2L)` at `L = 12`, the unique layer count. No
branch-output or residual-state norm is added. The bound that recurrent-depth
models obtain from their sandwich norms is already present here: Hyperball
pins every matrix at its initial Frobenius radius, PKDA and gated GQA carry
sigmoid output gates, and every branch reads a unit-RMS pre-normalized input,
so branch outputs are bounded a priori. Because those gates and the SwiGLU
can drive a branch toward zero input-dependently, the depth iteration can
reach a fixed point in the state; a branch-output norm would pin the update
size and forbid that. The telescoping identity holds in every cell:

```text
h_top = seed + Delta_P + sum_i Delta_R^(i) + Delta_C
```

Per-iteration residual RMS and the depth-contraction trace are the registered
guards. If a run shows the iteration diverging, the fallback is a residual
state norm in the core, which caps the state but makes each delta carry a
rescaling of everything before it; a branch-output norm is not a fallback.

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
| `P(r = 1)` | 10.2% |
| `P(r = 8)` | 8.0% |
| expected layers per pass | 23.5 |
| expected cells per pass | 5.88 |

The draw comes from its own keyed sub-stream of the data seed and step, so
`df` and `df-loop` share byte-identical pass-count and jitter draws. Realized
`r` is logged per step and enters the cell-token total.

Evaluation uses fixed `r = r_mean` in every mode. A fixed-`r` sweep over
`{1, 2, 4, 8}` at the screen is a diagnostic, not a metric.

## MHDB source banks

Routers follow `architecture.md` exactly: zero-initialized width-`D` query,
full-width RMS key statistics, raw values, one softmax per routing group, a
site-local null prepended to every bank. The core's sixteen sites are shared
across iterations because they are core weights. What changes is the bank:

| Site | Sources after the null |
|---|---|
| prelude, attention entry | seed |
| prelude, other sublayers | seed, partial |
| core iteration 1, attention entry | seed, `Delta_P` |
| core iteration `i > 1`, attention entry | seed, `Delta_P`, `Delta_R^(i-1)` |
| core, other sublayers | the entry bank plus the current partial |
| coda, attention entry | seed, `Delta_P`, `Delta_R` |
| coda, other sublayers | seed, `Delta_P`, `Delta_R`, partial |
| payload | seed, `Delta_P`, `Delta_R`, `Delta_C` |

`Delta_R^(i-1)` is the previous iteration's completed delta; `Delta_R` is the
whole core's completed delta, their sum. The core therefore reads a Markov
window rather than every past iteration: the residual is the state, the
window is only the read, and the routers act as the learned input injection
of the seed and the previous step. The fixed-capacity router carries a masked
slot for `Delta_R^(i-1)` at iteration 1. At `r = 1` every bank equals the
corresponding `df` bank.

## Horizontal channels

### Payload

Unchanged from `architecture.md` and `design.md`: asymmetric FBT fusion at the
seed, routed payload over the null, seed, and completed deltas,
`payload_norm`, keyed uniform jitter, the one-position shift, and the per-row
plain prefix. The payload is the trained form of warm-starting a column from
the previous column's final state; no untrained warm start exists.

### Shared core cache

A position's **write** at a core layer is everything later positions consume
there. For PKDA: the pre-convolution Q/K/V projections that form later
positions' convolution history, and the post-convolution `k_t`, `v_t`, and
control logits that drive the state transition. For gated GQA: the normalized
`k_t` and `v_t`.

The core's four token mixers run in one of two modes per pass:

- **Same-depth mode.** At iteration `i`, position `t` reads earlier
  positions' iteration-`i` writes. This is ordinary layer execution and needs
  no cache during training.
- **Shared mode.** At every iteration, position `t` reads earlier positions'
  **final-iteration** writes plus its own current-iteration write. In
  training pass `k ≥ 2`, final means pass `k-1`'s last iteration at each
  position. In decoding, final means each earlier position's last executed
  iteration.

Mixing mode is a property of the pass, not of the position: pass 1 and
Standard decoding run same-depth; every later pass and Soft or Fused decoding
run shared. The plain prefix of a later pass changes only the seed.

**PKDA in shared mode.** Let `S̄_(t-1)`, `Ā_(t-1)` be the recurrent matrix and
preconditioner state accumulated over positions `< t` from shared writes
only, with the transition equations of `architecture.md`. Position `t` then
applies its own current-iteration transition:

```text
A_t        = alphaP_t Ā_(t-1) + betaP_t (k_t ⊙ k_t)
k_write_t  = B(A_t) ⊙ k_t
S_tilde_t  = Diag(alpha_t) S̄_(t-1)
S_t        = (I - beta_t k_write_t k_t^T) S_tilde_t + beta_t k_write_t v_t^T
o_t        = S_t^T q_t
           = S̄_(t-1)^T Diag(alpha_t) (q_t - beta_t (k_write_t · q_t) k_t)
             + beta_t (k_write_t · q_t) v_t
```

The shared scan (`S̄`, `Ā`, and the chunk factors over shared writes) does not
depend on the iteration, so it is computed once per pass per layer; each
iteration performs an exclusive read at the modified query and adds the own
term. In chunk form this is the existing kernel with a strictly lower
intra-chunk mask, a rank-one modified query, and an explicit diagonal term.
The convolution at position `t` uses its own current pre-convolution
projections and earlier positions' shared ones as history.

**Gated GQA in shared mode.** Position `t` attends over earlier positions'
shared `K̄`, `V̄` under a strictly causal mask and over its own current `k_t`,
`v_t`; the two softmax pieces are merged by log-sum-exp.

Shared writes are never detached and never jittered. Pass `k`'s loss reaches
pass `k-1` through both the payload and the final writes.

### Modes

| Mode | Prompt | Generated positions | Core cache per sequence |
|---|---|---|---:|
| Standard | one plain pass, same-depth | same-depth, no payload | `r_max`-fold per-iteration |
| Soft | one plain pass, same-depth | shared finals plus payload | one-fold |
| Fused | plain pass, then one fused pass (prefix 1) in shared mode | shared finals plus payload | one-fold |

Training covers both kinds of finals: pass 2 reads same-depth finals from
pass 1, and pass 3 reads shared finals from pass 2.

## Training

Passes and iterations nest: each pass runs the prelude once, the core `r`
times in the pass's mixing mode, and the coda once; passes follow the Jacobi
scheme of `design.md` with the same pass mixture, prefix mixin, jitter, and
loss:

```text
K = 1:  loss = ell_1
K > 1:  loss = ell_1 + mean(ell_2, ..., ell_K)
```

Readout happens only after the coda at the final iteration. The cooldown
log-partition penalty applies unchanged.

Backpropagation is complete: every iteration of every pass is in the graph,
with block-level activation checkpointing at every pass. Truncated
backpropagation through the last iterations is excluded at both scales below;
if it were ever adopted, the seed and `Delta_P` reads inside the retained
iterations would be the prelude's only gradient path, so an unrouted core
could not truncate.

Optimizer ownership follows `architecture.md`: the tied core's matrices are
NorMuonH parameters receiving summed gradients across iterations, with one
fixed Hyperball radius each. The global FP32 gradient is clipped to norm 1.0
before both steps.

`df-loop` pairs with `df`: byte-identical initialization of every shared
parameter, the same stream, row order, schedule, and pass-count and jitter
draws. The `r` draw is the only additional randomness.

## Evaluation and diagnostics

`val` is pass-1 held-out cross-entropy at `r = r_mean` in same-depth mode.
`val_fused` is a second pass with plain-prefix length 1 at `r = r_mean` in
shared mode with the payload. Both report cell-tokens.

A loop result is uninterpretable without two stability traces:

- **Depth contraction.** At fixed neighbours, per-position
  `||h^(i) - h^(i-1)||_2` across core iterations for `i` up to `r_max`, with
  loss at each fixed power-of-two `r` up to `r_max` in both modes.
- **Sequence contraction.** Repeated fully fused prefill in shared mode with
  the payload, as in `design.md`: eight iterations in the monitor, at least
  thirty for promotion.

Additional registered diagnostics: residual RMS per iteration; the core
routers' mass on seed, `Delta_P`, and `Delta_R^(i-1)` by iteration, which is
the learned input-injection profile; the own-term share of PKDA output in
shared mode. Routing summaries remain diagnostics, not wins. The
recurrent-depth KL exit rule may be run as a diagnostic; adaptive exit is not
part of any metric.

## Parameter, cache, and cost accounting

`df-loop` has exactly the parameters of `df`: 257,514,792 total and
140,827,944 active non-embedding.

Per sequence at 1,024 context, one cell's registered mixer cache is
3.456 MiB (three FP32 PKDA matrix and diagonal states plus BF16 convolution
histories, 1.956 MiB; one BF16 KV cache, 1.500 MiB):

| Decode mode | Cells cached | Cache |
|---|---:|---:|
| Standard | `1 + 8 + 1` | 34.6 MiB |
| Soft and Fused | 3 | 10.4 MiB |

Jobe step-time estimates from the measured twelve-layer replay (about 3.8 ms
per layer evaluation plus a fixed head cost per pass, block checkpointing
adding one third to the trunk), at the expected `E[r] = 3.88`:

| Arm | One pass | Two passes | Three passes | Mean step | Full schedule |
|---|---:|---:|---:|---:|---:|
| `df` | 4.8 s | 9.7 s | 14.5 s | 6.2 s | ~19 h |
| `df-loop` | 10.7 s | 21.5 s | 32.2 s | 13.7 s | ~41 h |

The implementation must reproduce the parameter counts, the cache shapes, and
the mode semantics above before the arm is registered.

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
| Compute depth per pass | `4 + 4r + 4` layers, mean 70.2 |

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
`P(r = 32) = 5.9%`, and 17.56 expected cells per pass. Evaluation uses fixed
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
| Warmup / stable heat / cooldown | steps 1–200 / 201–506,642 / 506,643–675,523 |
| Feedback boundary | after step 506,642, with the cooldown |
| Expected pass-tokens | approximately 283.3B |
| Expected cell-tokens | approximately 4.98T |

The flagship spends about 3.39T cell-tokens, so this run costs about 1.5x the
flagship's compute with 50% of its active parameters. Per sequence at 8,192
context one cell's mixer cache is 27.9 MiB: Standard decoding at `r_max`
holds 34 cells, 949 MiB, and Soft or Fused decoding holds 3 cells, 83.7 MiB.

Entry requires an admissible `df-loop - df` screen result, the Prime gates
of `design.md`, the loop-specific parity gates below, and explicit spend
confirmation.

## Sources

| Source | Adopted | Replaced here |
|---|---|---|
| Recurrent-depth language models | tied core between prelude and coda, log-normal Poisson iteration draw, cache shared across iterations, effective-depth accounting | adapter by MHDB reads of seed and `Delta_P`; random initial state by the prelude output; sandwich norms by the unchanged pre-norm shell under Hyperball radii and gates; untrained warm start by the trained payload; truncated backpropagation by full backpropagation; a cache budget of several slots by a trained budget of one |
| Full-Bandwidth Transformer | asymmetric fusion, payload, Jacobi passes, prefix mixin, jitter, pass mixture, loss | — |
| This project | Markov source window, pass-level mixing mode, shared-mode PKDA read with own-term substitution, cell-token accounting | — |

## Exclusions

Not part of this specification: loop variants of `base`, `mhdb`, or `fbt`;
truncated backpropagation; branch-output or residual-state norms in the core;
a random or learned initial core state; a raw embedding path into the core; a
shared-cache budget above one; per-iteration halting readouts and pause
tokens, which belong to a separate specification; adaptive exit as anything
but a diagnostic; and any loop mapping onto the six-cell flagship.

## Adoption

Registering `df-loop` changes the following together:

- `ARMS` gains `df-loop`; `ModelConfig` gains the core recurrence, `r_mean`,
  `r_max`, and the mixing mode; `AGENTS.md` and `README.md` state six arms.
- Checkpoint contract v17 with `r_mean`, `r_max`, and the `r` sub-stream as
  state-defining fields; realized `r` and cell-tokens in the run log.
- The workspace FLA fork gains the shared-mode PKDA forward and backward; the
  gated GQA layer gains the two-piece merge; `df probe` gains parity gates for
  both against the literal recurrence, and the router's masked slot.
- CUDA graphs capture the prelude, the core cell per mixing mode and per
  bank capacity, and the coda with the head, replayed with alternating
  buffers; the graph pool is requalified and its reservation recorded.
- `design.md` adds `df-loop` to the arm table, cell-tokens to the accounting,
  two registered runs (`df-loop`, seeds 1 and 2) to Stage 1, `r = r_mean` to
  the evaluation contract, and the flagship-width loop run to the scale plan.
  `df-loop - df` is a paired contrast on the complete package only; depth
  alone is not attributable without an unrouted loop arm.

# Architecture: the four letters and the column

This page defines the computation and state of the `DeltaModel` family: the
column every condition shares and the four packages the condition letters
add. `a` puts Preconditioned Kimi Delta Attention (PKDA) in three of every
four token mixers, `r` adds Multi-Head Delta Block routing (MHDB), `f` adds
Full-Bandwidth Transformer (FBT) feedback between token columns, and `l` ties
the cells between the first and last into one core iterated within the
column. `arfl` is the full built stack, `arf` the flat column, and the empty
condition the plain decoder; dropping a letter removes that package from the
computation. `L`, the loop's shared-cache extension, is specified at the end
of the `l` section and not built.

What this page holds is invariant across geometries: equations, state, source
identities, initialization, precision, and optimizer ownership. The widths,
head counts, layer counts, parameter and cache accounting, and budgets of the
two geometries we grow are in [scaling.md](scaling.md); the training recipe,
data, schedule, and evaluation modes are in [design.md](design.md);
[interpretability.md](interpretability.md) lists the scripts that use the
named states; [literature.md](literature.md) separates source mechanisms from
local choices.

## The column

```text
token embedding e_t -------------------+
                                       +-- FBT fuse --> seed s_t          (f)
previous payload p_(t-1) --------------+                  |
                                                          v
        [ PKDA - PKDA - PKDA - gated global GQA ] x C     (a)
           each sublayer transiently reads MHDB sources   (r)
           the cells between the first and last tied and
           iterated r times                               (l)
                          |                         |
                       h_top                    block deltas
                          +------------+------------+
                                       |
                                routed payload p_t
```

The decoder is a bias-free pre-norm stack with tied embedding and readout, a
final RMSNorm, packed SwiGLU channel mixers, and no dropout. Every RMSNorm,
including PKDA's gated output norm, uses epsilon `1e-6`. There is no explicit
position encoding: no sliding window, MLA, RoPE, or additive position
embedding. Layers come in four-layer **cells**, the unit MHDB addresses and
the unit the loop ties. Every dense attention layer is gated causal NoPE GQA;
under `a` each cell is `[PKDA, PKDA, PKDA, gated global GQA]`, so PKDA carries
token-mixer state, order, and recency and every fourth layer supplies a dense
causal global read. Without `a` every layer is that dense read, and position
is whatever the causal mask lets the model infer.

On pass 1 and in Standard decoding the seed is the token embedding. The
readout uses `final_norm(h_top)`; the outgoing payload has its own routing
and normalization. The mixer caches are additional state paths outside the
diagram's explicit payload edge.

### Residual shell

For layer `l` of `L` unique layers, the token mixer and MLP produce
already-scaled branch deltas `a_l` and `m_l`:

```text
a_l = mixer_l(rmsnorm(routed_read(h_l))) / sqrt(2L)
h'_l = h_l + a_l
m_l = swiglu_l(rmsnorm(routed_read(h'_l))) / sqrt(2L)
h_(l+1) = h'_l + m_l
```

MHDB changes only the temporary input read. It never writes its routed mixture
directly into the residual stream. Consequently, for column seed `s`, the top
state always has the exact telescoping form:

```text
h_top = s + sum_l (a_l + m_l)
```

This identity is the source and causality contract for within-column routing,
the recurrent payload, and the loop.

### States

| State | Lifetime and causal role |
|---|---|
| Column seed | Current embedding or token-gated incoming payload; the actual residual origin |
| Completed block deltas and current partial | Contributions within one column; MHDB reads them without replacing the residual identity |
| `h_top` | Completed column state before final readout normalization |
| Payload | Normalization of top state plus routed enrichment under `r` with `f`, shifted to the next token column |
| Core entry and core state | `l`: the prelude output the tied core starts from and the residual after its last iteration; their difference is the sum of the core cells' deltas |
| PKDA matrix, preconditioner, convolution history | Token-mixer memory, continued during decode and reset within each current Jacobi prefill pass |
| GQA K/V | Visible-prefix cache for the global layers |

Three recurrences are distinct. Every `a` condition has PKDA recurrence inside
its token mixers whether or not it has `f`; `f` adds explicit payload
feedback across token columns; `l` adds a tied core iterated within the
column, a third recurrence with iteration-indexed mixer states. Every contract
on this page holds inside the loop.

## Letter a: PKDA and gated global GQA

### PKDA mixer

Each PKDA layer is a Kimi Delta Attention recurrence with the stable diagonal
apply-to-key preconditioner from Preconditioned DeltaNet. Its heads have
key/value width 128, and its Q/K/V projection width is five thirds of the
residual width at both geometries: the residual width does not constrain the
recurrent projection width.

#### Projection and controls

Bias-free Q/K/V projections are followed by separate causal depthwise
convolutions of width 4 and SiLU. Q and K are L2-normalized per head, and Q is
additionally scaled by `1/sqrt(128)`. A single packed hidden-width projection
produces five non-overlapping control slices:

1. the rank-128 hidden input to the channel-wise main decay;
2. one delta-update logit per head;
3. one preconditioner-decay logit per head;
4. one preconditioner-gain logit per head;
5. the rank-128 hidden input to the output gate.

Packing changes neither parameter ownership nor optimizer semantics. On CUDA,
the corresponding backward packs the five slice gradients directly into one
contiguous buffer before the shared projection GEMM.

The main decay and diagonal preconditioner have independent parameter slices
and downstream parameters inside the packed implementation. They are not tied.
The preconditioner uses:

- squash bound `x = 1.5` and epsilon `1e-6`;
- learned per-head log-space center initialized to `-0.2`;
- decay rates sampled on `[1, 16]`;
- softplus time constants sampled log-uniformly on `[0.001, 0.1]`;
- negative-eigenvalue and safe-gate modes disabled.

#### Recurrent equation

For one head, let `S_t` be the key-by-value recurrent matrix and `A_t` the
nonnegative diagonal preconditioner state. Both begin at zero. In the project's
key-first convention:

```text
log alpha_t   = -exp(A) * softplus(decay_t + b)
alpha_t       = exp(log alpha_t)
beta_t        = sigmoid(update_t)

log alphaP_t  = -exp(A_P) * softplus(decayP_t + b_P)
alphaP_t      = exp(log alphaP_t)
betaP_t       = sigmoid(updateP_t)

A_t           = alphaP_t A_(t-1) + betaP_t (k_t ⊙ k_t)
r_t           = log(A_t + 1e-6) - center
s_t           = r_t / (1 + abs(r_t))
B_t           = exp(-log(1.5) * s_t)
k_write_t     = B_t ⊙ k_t

S_tilde_t     = Diag(alpha_t) S_(t-1)
S_t           = (I - beta_t k_write_t k_t^T) S_tilde_t
                + beta_t k_write_t v_t^T
o_t           = S_t^T q_t
```

Every coordinate of `B_t` lies in `[2/3, 3/2]`. The output is normalized
head-wise, multiplied by a rank-128 sigmoid output gate, flattened, and sent
through a bias-free output projection. The output-gate expansion has a
zero-initialized bias.

#### Training and decoding state

CUDA training and long-prompt prefill use the workspace FLA fork's chunk
operator at chunk size 64. CPU and MPS execute the literal recurrence. CUDA
never silently substitutes the sequential fallback.

Autoregressive decoding continues, per PKDA layer:

- the FP32 recurrent matrix `S_t`;
- the FP32 diagonal state `A_t`;
- three BF16 convolution histories of length 3.

A Jacobi or fused-prefill pass starts those states from zero and advances them
once across that pass's causal token order. PKDA state is not carried between
repeated passes over the same positions; the FBT payload is the sole cross-pass
state. After the final prefill, the materialized mixer caches advance normally
during decoding.

### Gated global GQA

Every dense attention layer, the fourth of each cell under `a` and every
layer without it, is causal NoPE GQA with head width 96 and an output gate.
For pre-normalized input `x`, a packed bias-free projection produces Q/K/V, Q
and K receive per-head RMSNorm, and a separate bias-free projection produces
one gate coordinate per query-head output coordinate:

```text
q, k, v = split(W_qkv x)
z       = concat(GQA(q, k, v))
o       = W_o(sigmoid(W_g x) * z)
```

The gate acts on the concatenated attention result before the output projection
and residual-branch scaling. It does not change attention logits or softmax
weights. It is distinct from every PKDA control and from the FBT entry gate.

Full-sequence and prefill CUDA execution use compiled PyTorch FlexAttention
with a shared causal block mask and native GQA. Cached decoding writes BF16 K/V
explicitly, then attends only to the valid prefix; a one-token query sees every
key in that prefix. Only the dense attention layers own KV storage: one per
cell under `a`, every layer without it. The external `flash-attn` extension is
not required.

## Letter r: Multi-Head Delta Block routing

MHDB widens residual access along depth while preserving a clean residual
stream. It uses the multi-head source-selection operation from multi-head Delta
Attention Residuals, but its addressable deltas are four-layer cells, not
individual attention and MLP branches. `block_routing` is the flag and `r` the
letter.

### Router

One routing site owns:

- a width-`D` query `q`, initialized to zero;
- a learnable width-`D` RMS key scale `g`, initialized to one;
- a width-`D` null value, initialized to zero;
- `H = kv_heads` contiguous feature groups.

For raw source values `v_i`:

```text
k_i       = g * v_i / sqrt(mean(v_i^2) + eps)
score_i,h = dot(q_h, k_i,h)
weight_i,h = softmax_i(score_i,h)
route_h   = sum_i weight_i,h * v_i,h
```

The RMS statistic spans the full width, so it couples the groups; each group
then has its own softmax over sources and mixes only its own raw value slice.
There is no score factor `1/sqrt(head_dim)` and no output projection.

Every site prepends its own null. At initialization, the zero query makes the
source distribution uniform. The null initially contributes no value, but it is
learnable, so null mass must later be interpreted together with null-vector
scale.

### Block sources

Let `c_b` be the residual at entry to four-layer cell `b`. Define:

```text
partial_b = h_current - c_b
Delta_b   = h_cell_exit - c_b
```

At a site in cell `b`, the source bank is:

```text
[site-local null, column seed, completed Delta_0 .. Delta_(b-1), partial_b?]
```

The current partial is omitted when it is exactly absent at cell entry. The MLP
site sees the partial after its layer's attention delta has been added. At the
cell boundary, the evolving partial becomes the one completed block delta.
Individual branch deltas never become addressable sources.

The non-null sources therefore reconstruct the current residual exactly:

```text
h_current = seed + sum(completed block deltas) + current partial delta
```

The routed value is added only to the next sublayer's pre-norm read. On the
initial zero-query pass, the non-null mixture is collinear with `h_current`, so
the following RMSNorm makes routed pass-1 reads functionally inert up to its
epsilon.

With `C` cells, the deepest within-column site and the payload router each see
at most `C + 2` sources. The former has null, seed, `C - 1` completed cells,
and one live partial; the latter has null, seed, and all `C` completed cells.

## Letter f: Full-Bandwidth Transformer feedback

FBT adds recurrence across token columns. Let `e_t` be the sampled token
embedding and `p_(t-1)` the previous column's payload. A feedback position
constructs the next seed with the asymmetric FBT GLU:

```text
u_t = entry_norm(
    W_U p_(t-1) * sigmoid(W_G gate_norm(e_t))
)
```

The payload is the value path and the current token controls the gate. There is
no additive embedding bypass on a feedback position. Plain-prefix positions,
pass-1 positions, and Standard decoding use `e_t` instead of the fused seed.

With `r` as well, `u_t` is both the residual seed and the first non-null
source for every within-column MHDB router, and after the final cell a
dedicated `H`-group payload router reads its own null, the seed, and all `C`
completed block deltas. The recurrent payload is:

```text
r_payload = route(null, seed, Delta_0 .. Delta_(C-1))
p_t       = payload_norm(h_top + r_payload)
```

The additive `h_top` retains a direct top-state contribution alongside the
routed enrichment. The mixture and normalization need not preserve every
feature or its scale. At initialization, the uniform mixture, including the
zero-valued null, equals `h_top / (C + 2)`; after the payload RMSNorm it
therefore matches a bare normalized top state up to epsilon. The payload query
is not conditioned on the next token; token-dependent control happens only
after the payload shifts into the next column's FBT entry gate. Without `r`,
the payload is `payload_norm(h_top)`.

Training runs parallel Jacobi passes over a full sequence: pass 1 is plain
teacher forcing, and every later pass takes the preceding pass's payload
without detaching it, jitters it, shifts it one position right, and fuses it
on the suffix after a per-row plain prefix. Only the payload crosses passes;
every mixer state restarts within each pass. The pass mixture, prefix draw,
jitter, and loss are recipe and live in [design.md](design.md#feedback-passes).

Sequential generation retains only the immediately previous payload outside the
mixer caches. Each new column consumes it once, advances all PKDA and GQA
caches, and emits the next payload.

An equivalent high-level flat column is:

```python
e = embed(tokens)
seed = e if previous_payload is None else fbt_fuse(previous_payload, e)
h = seed
completed = []

for cell in four_layer_cells:  # C cells
    cell_start = h
    for layer_index, layer in enumerate(cell):
        partial = [] if layer_index == 0 else [h - cell_start]
        sources = [seed, *completed, *partial]
        h = h + scaled_mixer(rmsnorm(h + mhdb(sources)))
        partial = h - cell_start
        h = h + scaled_mlp(rmsnorm(h + mhdb([seed, *completed, partial])))
    completed.append(h - cell_start)

payload = payload_norm(h + mhdb([seed, *completed], site="payload"))
logits = tied_readout(final_norm(h))
```

Each `mhdb` call above implicitly prepends its own learned null.

## Letter l: the tied-depth loop

Under `l` the column's first cell is a prelude, its last cell a coda, and the
cells between them become one weight-tied core that runs `r` times per
column, with `r` drawn once per optimizer step. Recurrent-depth language
models supply the tied core and the iteration draw; the horizontal channel
between columns stays exactly the FBT payload of `f`, and the mixer caches
stay same-depth: iteration `i` of a column reads earlier columns'
iteration-`i` writes, as they do in that paper's training. `l` adds no
parameters.

### Why

The tied core adds a second axis of recurrence: what changes within a column
as the same weights are reused, what an intervention at a particular iteration
does, and which information is newly computed versus carried. The `arf`
channel turned out token-legible ([findings.md](findings.md)); a core that
refines a column over several iterations before the payload is emitted is one
candidate way to give the channel something to carry. At `r = 1` the loop is
exactly its unlooped condition, so the loop is the one thing an `arfl`
specimen adds over an `arf` specimen. Every iteration-indexed trace says
whether its index is core depth, Jacobi pass number, or token position.

### The looped column

A column of `C` cells holds its first and last cells and loops the
`c = C - 2` cells between them, in order, as one tied core; `l` requires
whole cells and at least three of them. Each core cell keeps its own block
delta, accumulated across iterations, so the looped column has exactly the
unlooped column's cells and block sources, and at one iteration it coincides
with the unlooped column in values, routes, losses, and gradients at every
cell count. A pass at `r` iterations executes `2 + c r` cells, so a `k`-pass
batch costs `k (2 + c r)` cell evaluations per predicted token; results report
**cell-tokens** beside pass-tokens.

```text
token embedding e_t -------------------+
                                       +-- FBT fuse --> seed s_t
previous payload p_(t-1) --------------+                  |
                                                          v
        +----------------------- prelude P ------------------------+  Delta_P
        |                                                          |
        |   +--------------- core R, tied, r iterations --------+  |
        |   | iteration i reads: seed, Delta_P, core deltas     |  |  Delta_R
        |   +---------------------------------------------------+  |
        |                                                          |
        +------------------------- coda C --------------------------+  Delta_C
                          |                         |
                       h_top                    block deltas
                          +------------+------------+
                                       v
                                routed payload p_t
```

With one core cell, the screen's shape, the column is:

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
numbers them; `ColumnOutput` carries `iterations`, `core_entry`, and
`core_state`, and a core site's route weights are keyed by iteration
(`L4i2.attn`).

### Residual shell in the core

Every cell, the core included, keeps the pre-norm shell above with branch
scale `1/sqrt(2L)` at `L` the unique layer count. No branch-output or
residual-state norm is added, and the telescoping identity holds in every
cell:

```text
h_top = seed + Delta_P + Delta_R + Delta_C
```

with `Delta_R` the sum of the core cells' deltas. The shell is chosen because
it has no mechanism that forces a nonzero update: a branch-output norm would
pin every iteration's update size and make a fixed point unreachable by
construction. Whether the tied core actually contracts is not assumed.
Hyperball radii bound the NorMuonH matrices, but norm scales, router values,
and the NAdam-owned gates and controls are not radius-bounded, so whether it
contracts is measured: the depth trace below is the guard, and a diverging
trace means the specification changes.

### Iteration count

`r` is drawn once per optimizer step and shared by every pass and microbatch
of that step:

```text
tau ~ Normal(log(r_mean - 1) - sigma^2 / 2, sigma),   sigma = 1/2
r   = min(1 + Poisson(exp(tau)), r_max)
```

This is the recurrent-depth log-normal Poisson draw with the rate shifted by
one so that the uncapped mean of `r` is `r_mean`. `--loop-iterations` and
`--loop-max-iterations` set `r_mean` and `r_max`; both are state-defining,
their defaults are in [design.md](design.md#knobs), and the statistics of the
draw under the cap are in [scaling.md](scaling.md). The draw comes from its
own keyed sub-stream of the data seed and step, so `arf` and `arfl` share
byte-identical pass-count, prefix, and jitter draws. Realized `r` is logged
per step and enters the cell-token rate.

Evaluation and decoding use one fixed `r` for the prompt passes and the entire
decode trajectory; `r` never changes within a request. Metrics use
`r = r_mean`; the fixed-`r` sweep over `1..r_max` is the depth trace.

### Source banks

Routers follow the `r` section exactly: zero-initialized width-`D` query,
full-width RMS key statistics, raw values, one softmax per routing group, a
site-local null prepended to every bank. A core cell's eight sites are shared
across iterations because they are core weights. With one core cell, its bank
is a cell's bank:

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
sources reconstruct the current residual exactly, so the uniform zero-query
mixture stays collinear with the residual and pass-1 routing is functionally
inert at initialization at every iteration. The residual is the state; the
routers are the learned input injection of the seed and the prelude.
Per-iteration deltas are never sources: an iteration boundary is a boundary in
weights, not in state.

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

### Input injection and column start

The recurrent-depth decoder computes `e = P(x)`, draws a random state `s_0`,
iterates `s_i = R(e, s_(i-1))` where `R` opens with an adapter that maps the
concatenation of the state and `e` back to the residual width once per
iteration, and reads out `C(s_r)`. Its warm start is an inference-only
substitution of the previous token's `s_r` for the noise. `arfl` realizes
each piece differently:

| Recurrent-depth decoder | `arfl` |
|---|---|
| injected input `e = P(x)` | `seed + Delta_P`, the prelude output |
| adapter on `[s; e]`, once per iteration | router reads of seed and `Delta_P` at every core site |
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

### Horizontal channel and mixing

The payload is unchanged from the `f` section: asymmetric FBT fusion at the
seed, routed payload over the null, seed, and completed deltas,
`payload_norm`, keyed uniform jitter, the one-position shift, and the per-row
plain prefix. It is the only channel between columns besides the mixer caches.

Mixing is **same-depth** at every position: at core iteration `i` a position
reads earlier positions' iteration-`i` writes of the current pass. Each
iteration is one more full-sequence evaluation of the same core cells, so the
PKDA chunk operator, FlexAttention, and the router run unchanged; a PKDA state
at iteration `i` starts from zero and advances across the pass's token order
exactly as it does for a pass today. Plain and fused positions differ only in
their seed, as in `arf`.

Decoding keeps one core cache track per iteration: track `i` holds the PKDA
matrix and diagonal states, the convolution histories, and the GQA K/V that
iteration `i` wrote at every earlier position. `forward_column` selects the
track before each core iteration, and every track shares the column position.
Standard, Soft, and Fused decoding all hold `2 + c r` cells of cache at the
request's fixed `r`; a `KVCache` is allocated for that `r`.

### Training

Passes and iterations nest: each pass runs the prelude once, the core `r`
times, and the coda once; passes follow the Jacobi scheme of `f` with the
same pass mixture, prefix mixin, jitter, and loss:

```text
K = 1:  loss = ell_1
K > 1:  loss = ell_1 + mean(ell_2, ..., ell_K)
```

Readout happens only after the coda at the final iteration. The cooldown
log-partition penalty applies unchanged.

Backpropagation is complete: every iteration of every pass is in the graph.
The trainer's activation policy counts executed layers per pass against a
measured raw budget, and block-level checkpointing switches on for every mode
deeper than that budget; the budget and the measured costs are in
[scaling.md](scaling.md#cost-of-the-loop-at-the-screen). Truncated
backpropagation through the last iterations is excluded.

Optimizer ownership follows the partition below: the tied core's matrices are
NorMuonH parameters whose gradients sum across iterations and passes, with one
fixed Hyperball radius each. The global FP32 gradient is clipped to norm 10.0
before both steps.

Execution follows the [runtime graph contract](runtime-qualification.md): one
fixed-address train CUDA graph per reachable `(pass count, r)` pair, all
sharing one pool, plus the no-grad evaluation graphs at `r = r_mean`.

`arfl` pairs with `arf`: byte-identical initialization of every parameter, the
same stream, row order, schedule, and pass-count, prefix, and jitter draws. The
`r` draw is the only additional randomness.

### Evaluation and diagnostics

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

### Not in this specification

Left out: truncated backpropagation; branch-output or residual-state norms in
the core; a random or learned initial core state; any warm start of the core
from the previous column, trained or zero-shot; a raw embedding path into the
core; a payload landing at the core entry, which is a different design rather
than a loop variant; a shared-cache budget above one; per-iteration halting
readouts and pause tokens, which belong to a separate specification; adaptive
exit as anything but a diagnostic; and a decode `r` that varies within a
request. `l` on the plain trunk or without `r` builds and pairs like every
other subset, and is unstudied: without `r` the core has no input injection at
all.

### Letter L: a shared core cache, specified and unbuilt

`L` is `l` plus a second trained horizontal channel: a shared core cache
through which every core iteration of a fused position reads earlier
positions' final-iteration mixer writes. It is the recurrent-depth "attend to
later iterations of earlier tokens" mechanism trained rather than zero-shot.
It is specified here and not built; `parse_condition` does not accept it yet.
In the condition grammar a capital letter is its lowercase letter plus one
package, and `l` and `L` are exclusive occupants of the loop slot, so `arfL`
would name the full stack with the cache.

#### Plain and fused positions

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

#### Shared core cache

A position's **write** at a core layer is everything later positions consume
there. For PKDA: the pre-convolution Q/K/V projections that form later
positions' convolution history, the post-convolution `k_t` and `v_t`, and the
per-head transition controls `alpha_t`, `beta_t`, `alphaP_t`, `betaP_t`. For
gated GQA: the normalized `k_t` and `v_t`. The write bank of one pass, per
core layer, holds the BF16 pre-convolution Q/K/V, the BF16 post-convolution K
and V, the FP32 transition controls, and the BF16 GGQA K and V at every
position.

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
only, with the transition equations of the `a` section; at `t = 0` both are
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

#### Jacobi horizon

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

#### Decoding and cost

Fused decoding under `L` keeps one compact core cache that advances once per
position after its final iteration, `2 + c` cells in all; Standard decoding
keeps the `l` tracks. Training holds two write banks and the bank-side scan
intermediates on top of `l`'s activations, and each feedback pass evaluates
the core mixers twice per iteration, so `L` is more expensive than `l` in
every mode.

#### Building it

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
- A checkpoint contract bump; the pages that describe `l` gain `L`.

## Precision and initialization

Each NorMuonH-owned matrix `W` with shape `[d_out, d_in]` is initialized from
`Normal(0, 1 / sqrt(d_in))`, following the Hyperball parameterization. The tied
embedding/readout and NAdam-owned dense matrices use `Normal(0, 0.02)`.
Depthwise convolution weights keep PyTorch Conv1d's Kaiming-uniform
initialization at fan-in 4. RMSNorm scales initialize to one; routing queries
and nulls initialize to zero. PKDA rates, time constants, centers, and
output-gate bias use the special initializations stated above.

Pairing is built in. Two conditions on the same trunk letter initialize every
parameter they share byte-identically for a given seed: the trunk from the
common stream, the attention gates from their own deterministic stream, and
`f`'s fusion matrices from `f`'s own, neither of which advances the common
one; routers initialize to zero. The plain trunk and the `a` trunk consume
the common stream differently, so parameters do not pair across `a`.

Embedding parameters and optimizer state are FP32. On CUDA, the residual
stream, routed values, payloads, mixer activations, and caches are BF16 except
for the PKDA matrix and diagonal boundary states. Cut cross-entropy reads an
address-stable BF16 classifier shadow that is refreshed from the tied embedding
once after every optimizer update and is neither a parameter nor checkpoint
state. Its authoritative tied parameter and accumulated gradient remain FP32.
The head's own classifier gradient reaches that FP32 gradient through a second
address-stable BF16 buffer: every head call lock-adds into it, and the captured
trainer adds it into the FP32 sink and clears it whenever another microbatch
would take the buffer past `--head-flush-every` head calls, and once more before
the optimizer reads the step. A feedback pass is a head call, so the number of
BF16 additions between two FP32 flushes is the same under every pass count. The
buffer is a runtime operand like the shadow, not state.

## NorMuonH and NAdam

Every condition at every geometry uses one optimizer recipe with two disjoint
parameter groups: one NorMuonH group and one NAdam group. Their public
controls are `--lr-normuonh` and `--lr-nadam`, with defaults in
[design.md](design.md#knobs). No group uses weight decay.

### NorMuonH matrices

NorMuonH owns ordinary trainable two-dimensional hidden weights, including:

- attention and PKDA Q/K/V/output projections;
- SwiGLU input and output matrices;
- the FBT payload-value projection.

For matrix `W`, its initial FP32 Frobenius radius `R = ||W_0||_F` is fixed for
the life of the run. With the fan-in initialization above, `E[R^2] = d_out`;
the realized sampled radius, rather than that expectation, is the exact
constraint. NorMuonH forms an update direction by:

1. EMA momentum with coefficient `0.95`;
2. the released NorMuon Nesterov blend of `0.05` times the current gradient
   and `0.95` times the updated momentum;
3. five Newton-Schulz orthogonalization steps;
4. row-wise second-moment normalization with beta `0.95` and epsilon `1e-8`;
5. unit-Frobenius normalization;
6. a Hyperball trial step and exact radial projection.

For current gradient `G_t` and stored momentum `M_t`:

```text
M_t = 0.95 M_{t-1} + 0.05 G_t
N_t = 0.05 G_t + 0.95 M_t
```

Newton-Schulz and the NorMuon row normalization act on `N_t`.

With the dimensionless learning rate `lr_normuonh` and normalized direction
`U`:

```text
W_next = R * Normalize_F(W - lr_normuonh * R * Normalize_F(U))
```

The initial radius is checkpointed and must remain invariant across eager,
compiled, captured, staged, and resumed updates.

### NAdam parameters

NAdam owns parameters whose norm carries semantic scale and every non-matrix
parameter:

- tied embedding/readout;
- GGQA gate matrices;
- the FBT token-gate matrix;
- PKDA's packed control projection, main-decay expansion, and output-gate
  expansion;
- RMSNorm weights, router queries and nulls, depthwise convolutions, biases,
  rates, time constants, and preconditioner centers.

Every NAdam-owned parameter, including the tied embedding/readout, belongs to
one parameter group with learning rate `lr_nadam`. It uses moment betas
`(0.9, 0.95)`, momentum decay `psi = 0.004`, epsilon `1e-8`, and no weight
decay. At optimizer step `t`, PyTorch NAdam uses:

```text
m_t = beta1 m_{t-1} + (1 - beta1) G_t
v_t = beta2 v_{t-1} + (1 - beta2) G_t^2
mu_t = beta1 (1 - 0.5 * 0.96^(t psi))
P_t = product_{i=1}^t mu_i
U_t = ((1 - mu_t) G_t / (1 - P_t)
       + mu_{t+1} m_t / (1 - P_t mu_{t+1}))
      / (sqrt(v_t / (1 - beta2^t)) + eps)
```

The update is `theta_t = theta_{t-1} - lr * U_t`. Both parameter groups receive
the same warmup-stable-cooldown multiplier defined by the current scale's
schedule. CUDA uses PyTorch's foreach NAdam path outside the captured
forward/backward graphs; its scalar step and momentum-product state stay on
CPU, while both moment tensors and all parameters remain FP32 on-device.
NorMuonH compiles each active shape bucket's packing, mathematical update, and
in-place state writeback together. Its checkpoint state remains ordinary
per-parameter tensors; the compiled path retains no second packed state.

After synchronized microbatch accumulation, the single global FP32 gradient
vector is clipped to L2 norm 10.0 immediately before both optimizer steps. The
reported gradient norm is the pre-clip norm, and a non-finite norm terminates
the run.

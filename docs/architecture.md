# Architecture: the six letters and the column

This page defines the computation and state of the `DeltaModel` family: the
column every condition shares and the six packages the condition letters
add. `a` puts Preconditioned Kimi Delta Attention (PKDA) in three of every
four token mixers, `e` adds shared and routed quarter-width SwiGLU experts,
`r` adds Multi-Head Delta Block routing (MHDB), `f` adds
Full-Bandwidth Transformer (FBT) feedback between token columns, `l` ties
the cells between the first and last into one core iterated within the
column, and `m` adds an auxiliary second-token predictor after the column.
`aerflm` is the full built stack, `aerfm` its flat column, and the empty
condition the plain decoder; dropping a letter removes that package from the
computation.

What this page holds is invariant across geometries: equations, state, source
identities, initialization, precision, and optimizer ownership. The widths,
head counts, layer counts, parameter and cache accounting, and budgets of the
three geometries we grow are in [scaling.md](scaling.md); the training recipe,
data, schedule, and evaluation modes are in [design.md](design.md);
[interpretability.md](interpretability.md) lists the analysis interfaces, and
[references/refs.yaml](../references/refs.yaml) identifies the sources.

## The column

```text
token embedding e_t -------------------+
                                       +-- FBT fuse --> seed s_t          (f)
previous payload p_(t-1) --------------+                  |
                                                          v
        [ PKDA - PKDA - PKDA - gated global GQA ] x C     (a)
           each FFN selects 1 shared + 3/15 experts       (e)
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
final RMSNorm, packed SwiGLU channel mixers (shared and routed under `e`),
and no dropout. Every RMSNorm, including PKDA's gated output norm, uses
epsilon `1e-6`. Layers come in
four-layer **cells**, the unit MHDB addresses and the unit the loop ties.
Under `a` each cell is `[PKDA, PKDA, PKDA, NoPE-GGQA]`, so PKDA carries
token-mixer state, order, and recency and every fourth layer supplies a gated
dense causal global read. Without `a` each cell is `[RoPE-GGQA, RoPE-GGQA,
RoPE-GGQA, NoPE-GGQA]`: the RoPE layers occupy exactly the PKDA sites. The
pattern holds at every geometry and core iteration.

On pass 1 and in Standard decoding the seed is the token embedding. The
readout uses `final_norm(h_top)` times the muP readout multiplier `1536 / D`;
the outgoing payload has its own routing and normalization. With `m`, an
auxiliary branch reads `h_top` together with the next token's embedding to
predict a second token. The mixer caches are additional state paths outside the
diagram's explicit payload edge.

### Residual shell

For layer `l` of `L` unique layers, the token mixer and MLP produce
already-scaled branch deltas `a_l` and `m_l`:

```text
a_l = mixer_l(rmsnorm(routed_read(h_l))) / sqrt(2L)
h'_l = h_l + a_l
m_l = ffn_l(rmsnorm(routed_read(h'_l))) / sqrt(2L)
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
| GQA K/V | Visible-prefix cache for the global layers; RoPE layers store already-rotated keys |

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
residual width at all three scale presets.

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

CUDA full rows and cached decoding share one fused convolution/SiLU/QK-norm
kernel, keeping those operations in FP32 until the final Q/K/V cast. Cached
rows prepend their raw projected history, discard the history's outputs, and
retain the final `conv_size - 1` projected inputs for the next call (three at
the default kernel size).

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

Every dense attention layer uses causal GQA with head width 96 and an output
gate. The fourth layer of each cell is NoPE; without `a`, the other three
layers use RoPE. For pre-normalized input `x`, a packed bias-free projection
produces Q/K/V, Q and K receive learned per-head RMSNorm, and a separate
bias-free projection produces one gate coordinate per query-head output
coordinate:

```text
q, k, v = split(W_qkv x)
q, k    = rmsnorm_q(q), rmsnorm_k(k)
q, k    = rope(q, position), rope(k, position)  # RoPE layers only
z       = concat(GQA(q, k, v))
o       = W_o(sigmoid(W_g x) * z)
```

Partial RoPE here means selecting three layers per cell; each selected layer
rotates the full head width. For head width `d`, pair `j` is the adjacent
coordinates `(2j, 2j + 1)`, with angle `position * 10000^(-2j / d)`. Phases,
sine/cosine, and the rotation are computed in FP32, then Q and K are cast
back to their activation dtype. The base is fixed at 10,000, with no learned
parameters or CLI knob. Values are not rotated.

Positions count from zero within each token row. Every feedback pass and
core iteration reuses those same positions, and the plain/fused split does
not restart them. Cached execution offsets new queries and keys by
`cache.pos` and stores the keys after rotation, so a prefix is never rotated
again when the next token arrives. Every iteration's cache track shares the
same token position.

The gate acts on the concatenated attention result before the output projection
and residual-branch scaling. It does not change attention logits or softmax
weights. It is distinct from every PKDA control and from the FBT entry gate.

Full-sequence and prefill CUDA execution use compiled PyTorch SDPA with
native GQA. BF16 and FP16 explicitly select its built-in Flash backend; FP32
diagnostics explicitly select math. The backend choice is scoped inside the
compiled region and restores the caller's preferences. Cached decoding writes
BF16 K/V explicitly, then uses compiled FlexAttention over the valid prefix;
a one-token query sees every key in that prefix. Only the dense attention
layers own KV storage: one per cell under `a`, every layer without it.

## Letter e: shared and routed experts

Every dense SwiGLU is replaced by sixteen independent SwiGLUs: one shared
expert `S` and fifteen routed experts `E_j`. Each reads and writes the full
residual width `D`, with intermediate width `H / 4` for dense width `H`.
`H` must be divisible by four. Every token uses the shared expert and exactly
three routed experts. The expert bank is tied wherever its enclosing core
layer is tied; routing is recomputed from the current state at each iteration.

For normalized FFN input `x`, a bias-free `D -> 15` projection and a separate
expert-selection bias `b` compute:

```text
s = sigmoid(W_router x)
J = top3(s + b)
w_j = s_j / sum_(i in J) s_i  if j in J, else 0
ffn(x) = (S(x) + 3 * sum_j w_j E_j(x)) / 2
```

Scores are evaluated in FP32 during ordinary training. The bias affects
selection only; mixture weights use the original sigmoid scores. Selection is
token-local, has no expert capacity limit, and never drops tokens. The selected
gate values remain differentiable; the discrete choice has no gradient. The factor of
three keeps each selected expert's coefficient at one under equal gates;
division by two matches the variance of four independent, equal-variance
expert outputs to one dense output at equal gates. This is an initialization
scaling rationale, not a guarantee about learned output variance.

Hard top-three routing is discontinuous at expert-selection boundaries.
Differences between full-row and cached CUDA mixers can swap near-tied
experts and amplify hidden-state differences, including with FP32 activations
and router logits. The probe checks causal prefixes, portable FP32 cache
agreement, and CUDA FP32/BF16 cache arithmetic with expert choices held fixed;
unrestricted CUDA decode
drift is reported separately.

The expert matrices have four times the dense FFN's stored parameters and
the same active matrix arithmetic per token. The router adds `15D` parameters
per layer and dispatch overhead. Shared and finely segmented experts follow
the architectural ideas in [DeepSeekMoE](https://arxiv.org/abs/2401.06066);
the specific output normalization and fixed configuration are local choices.

Each physical expert bank has a zero-initialized persistent `expert_bias[15]`
buffer, outside the optimizer parameter groups. Training sums actual assignment
counts `C_j` over every microbatch, pass, and invocation of that bank in the
optimizer update, then changes the bias once after the optimizer step:

```text
b_j += 0.001 * sign(sum_i C_i - 15 * C_j)
```

Underloaded experts receive a larger selection bias; overloaded experts receive
a smaller one. No assignments or exactly balanced counts leave it unchanged.
The bias stays fixed during forward, backward, activation recomputation, and
evaluation. Counts are returned values rather than forward-side mutations, so
checkpoint recomputation cannot count a token twice. This follows
[auxiliary-loss-free balancing](https://arxiv.org/abs/2408.15664).

A complementary sequence regularizer normalizes the original sigmoid scores
over all fifteen experts, including unselected ones. Its load counts use the
top three unbiased affinities, distinct from the biased dispatch counts `C`
that drive the bias controller. For each sequence of `T` tokens, compute:

```text
q_j = s_j / sum_i s_i
U = top3(s)
P_j = mean_tokens(q_j)
F_j = stop_gradient(count_tokens(j in U) / (3T))
aux_layer = mean_sequences(15 * sum_j P_j F_j)
aux = mean_passes(mean_executed_layers(aux_layer))
training_loss += 1e-4 * aux
```

Sigmoid affinities, selection-only bias, and a small per-sequence regularizer
follow [DeepSeek-V3](https://arxiv.org/abs/2412.19437). Each recurrent invocation
is one term in the layer mean. The pass mean is independent of feedback
cross-entropy weighting; neither more depth nor more passes increases the
regularizer's coefficient. Cross-entropy reporting excludes it. Sequence
statistics couple the regularizer's gradients within each sequence without
changing causal forward activations.

`ColumnOutput.expert_aux_loss` holds the layer mean (`None` without `e`).
`expert_counts` holds integer counts `[layers, 15]` summed over each physical
bank's invocations (`None` without `e`); it is not persistent model state.
With `want_weights=True`, `expert_weights` maps each layer invocation to a
`[B, T, 15]` tensor containing the sparse normalized `w`: three nonzero values
per token summing to one. These are separate from MHDB's source weights.

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
logits = tied_readout(final_norm(h) * mup_ratio)  # 1536 / D
```

Each `mhdb` call above implicitly prepends its own learned null.

## Letter l: the tied-depth loop

Under `l` the column's first cell is a prelude, its last cell a coda, and the
cells between them become one weight-tied core that runs `r` times per
column, with `r` drawn once per optimizer step. The mixer caches stay
same-depth: iteration `i` reads earlier columns' iteration-`i` writes.
Under `f`, the FBT payload also crosses columns. `l` adds no parameters.

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
logits = tied_readout(final_norm(h) * mup_ratio)  # 1536 / D
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
this rule is the one above. The seed and prelude delta remain readable
through the routers at every iteration. At a fused position the token enters only through the FBT gate, as in
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

with `Delta_R` the sum of the core cells' deltas. The depth trace measures whether the core contracts. Fixed NorMuonH matrix
radii alone do not establish contraction: norms, router values, gates, and
controls are unconstrained NAdam parameters.

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
its value at core exit. Every bank's non-null sources reconstruct the current residual, so the
uniform zero-query mixture is collinear with that residual at initialization.
The routers inject the seed and prelude output at every core site. Sources
accumulate by cell, not by iteration.

A core cell's partial is exactly absent at one site, its attention entry on
iteration 1, where the cell has not yet moved. A router scores whatever
sources it is handed, so that site reads a three-source bank exactly as the
unlooped cell entry does, and from iteration 2 on the same router reads the
partial as a fourth source.
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

### Horizontal channel and mixing

The payload is unchanged from the `f` section: asymmetric FBT fusion at the
seed, routed payload over the null, seed, and completed deltas,
`payload_norm`, keyed uniform jitter, the one-position shift, and the per-row
plain prefix. It is the only channel between columns besides the mixer caches.

Mixing is **same-depth** at every position: at core iteration `i` a position
reads earlier positions' iteration-`i` writes of the current pass. Each
iteration is one more full-sequence evaluation of the same core cells, so the
PKDA chunk operator, causal GQA, and the router run unchanged; a PKDA state
at iteration `i` starts from zero and advances in token order. Plain and
fused positions differ only in their seed, as in `arf`.

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
log-partition penalty applies unchanged. With `e`, the expert balancing term
is added with the layer and pass normalization defined above.

Backpropagation is complete: every iteration of every pass is in the graph.
The trainer's activation policy counts executed layers per pass against a
configured raw budget. Deeper modes retain the final block of each four-layer
cell and checkpoint its preceding three blocks. With `a`, the retained block
is dense attention; without `a`, the same quarter of blocks is retained.
With `m`, the budget also counts the auxiliary block executed once per pass.
`e` halves the raw activation budget (20 executed layers at the screen,
versus 40 for dense FFNs) to allow for expert dispatch and parameter storage.
The checkpoint wrapper stays outside each compiled block, so both retained
and recomputed blocks use the same compilation boundaries. Every iteration
remains differentiable.

Optimizer ownership follows the partition below: the tied core's matrices are
NorMuonH parameters whose gradients sum across iterations and passes, with one
fixed Hyperball radius each. The global FP32 gradient is clipped to norm 10.0
before both steps.

CUDA captures one fixed-address training graph per reachable `(pass count, r)`
pair, sharing one pool, plus no-grad evaluation graphs at `r = r_mean`.
See [operations.md](operations.md#cuda-execution) for the runtime path.

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
tests the horizontal channel at fixed core depth.

## Letter `m`: two-token prediction

`m` adds one auxiliary prediction depth following
[DeepSeek-V3 section 2.2](https://arxiv.org/html/2412.19437v2#S2.SS2).
At input position `t`, the trunk still predicts `x_(t+1)` from `h_top,t`.
The auxiliary branch also receives the ground-truth `x_(t+1)` embedding and
predicts `x_(t+2)`:

```text
u_t = M concat(RMSNorm_h(h_top,t), RMSNorm_e(Emb(x_(t+1))))
v = DenseCausalTransformer(u)
logits_mtp,t = (final_norm(v_t) * 1536 / D) @ Emb.weight.T
```

The two entry norms are separate learned RMSNorms. `M` is a bias-free
`2D -> D` projection. The auxiliary transformer is one dedicated pre-norm
block at the trunk width, with dense gated GQA, full-head RoPE at theta
10,000, and a dense SwiGLU at the configured intermediate width. It uses
causal attention across the cropped sequence. Its weights are separate from
the trunk; it has no PKDA, experts, MHDB routing, feedback payload, or tied
core iterations. `e` and `l` continue to describe the trunk. Both the final
norm and the tied embedding/readout are shared with ordinary prediction.
Its two residual branches use the trunk's `1 / sqrt(2L)` scaling, where `L`
is the configured unique trunk layer count.

For a stored row of `T + 1` tokens, the trunk processes `tokens[:, :-1]`.
The auxiliary inputs are `h_top[:, :-1]` and
`Emb(tokens[:, 1:-1])`, and the targets are `tokens[:, 2:]`. This yields
`T - 1` valid auxiliary predictions without padding, ignored labels, or a
change to token stores. The branch can see the next token provided to it, but
never its second-token target or a later token. Its state does not feed back
into the column or its payload. Both inputs remain differentiable, so the
auxiliary loss updates the trunk, shared embedding/readout, and auxiliary
parameters.

Every training pass computes both ordinary and auxiliary cross-entropy.
Each head averages over its own valid positions: `T` for ordinary prediction,
`T - 1` for MTP. This valid-position normalization is a Delta choice;
DeepSeek-V3 equation 24 divides its cropped sum by the original length.
For either head, `combine(ell) = ell_1` at one pass and
`ell_1 + mean(ell_2, ..., ell_K)` at multiple passes. The objective is:

```text
loss = combine(CE_ntp) + z_coef * combine(z_ntp)
     + mtp_weight * (combine(CE_mtp) + z_coef * combine(z_mtp))
     + expert_balance_weight * expert_balance       # with e
```

The auxiliary head uses the existing cooldown log-partition penalty and its
own position mean. `--mtp-weight` is a finite nonnegative run setting,
constant across the schedule, defaulting to 0.3. The dense auxiliary block,
valid-position normalization, and integration with Delta's feedback and
z-loss recipe are local choices. The `ntp` and `mtp` metrics are the combined
ordinary and auxiliary cross-entropies before coefficients or z-loss. `loss`
reports the full optimized objective; `pass1` remains ordinary first-pass
cross-entropy. `val_mtp` and, with `f`, `val_mtp_fused` report separate
auxiliary validation losses.

Ordinary Standard, Soft, and Fused inference does not execute this branch or
allocate an auxiliary cache. MTP is an additional training objective here;
speculative candidate generation, verification, and cache rollback are not
implemented.

## Precision and initialization

Each NorMuonH-owned matrix `W` with shape `[d_out, d_in]` is initialized from
`Normal(0, 1 / sqrt(d_in))`, following the Hyperball parameterization. Normal
distributions here are specified by standard deviation. The single tuning
constant `BASE_NORMAL_INIT_STD = 0.02` in `delta_feedback_experiment/model.py`
sets the base scale for NAdam-owned matrices:

- GGQA gates, the expert router, the FBT token gate, and PKDA's packed control
  projection use
  `BASE_NORMAL_INIT_STD * sqrt(1536 / D)`: approximately 0.02828 at the
  screen, 0.02309 at the bridge, and 0.02 at the flagship.
- The tied embedding/readout and PKDA's fixed-head-width main-decay and
  output-gate expansions use `BASE_NORMAL_INIT_STD` directly.

For independent unit-RMS inputs, the width adjustment preserves the initial
gate/control logit variance. Keeping the head-width expansions at the base
scale also preserves the combined variance of PKDA's two-stage controls.
The scaling uses the same parameter predicate as the NAdam width-scaled
learning-rate group and draws no additional randomness. Changing the base
constant changes these NAdam matrix scales; NorMuonH's fan-in scale stays
independent of it.

The readout multiplies `final_norm(h_top)` by the muP width ratio `1536 / D`
before the tied classifier product, one at the flagship and two at the
screen; the multiplier is a fixed part of the parametrization, not a
parameter, and the CUDA classifier shadow reads the scaled input unchanged.
Depthwise convolution weights keep PyTorch Conv1d's Kaiming-uniform
initialization at fan-in 4. RMSNorm scales initialize to one; routing queries
and nulls initialize to zero. PKDA rates, time constants, centers, and
output-gate bias use the special initializations stated above.

Pairing is built in. Two conditions on the same trunk letter initialize every
parameter they share byte-identically for a given seed: the trunk from the
common stream, the attention gates from their own deterministic stream, and
`f`'s fusion matrices from `f`'s own, neither of which advances the common
one; MHDB routing queries initialize to zero. The plain trunk and the `a` trunk consume
the common stream differently, so parameters do not pair across `a`. The
expert bank and router use a separate deterministic stream after common
initialization, preserving every shared non-FFN parameter when `e` is added.
Expert weights pair across `r`, `f`, and `l`; MHDB's zero-initialized queries
are distinct from the randomly initialized expert router.
The MTP module also uses its own deterministic initialization stream, so
adding `m` preserves all shared parameters byte-identically.

Embedding parameters and optimizer state are FP32. On CUDA, the residual
stream, routed values, payloads, mixer activations, and caches are BF16 except
for the PKDA matrix and diagonal boundary states. Cut cross-entropy reads an
address-stable BF16 classifier shadow that is refreshed from the tied embedding
once after every optimizer update and is neither a parameter nor checkpoint
state. Its authoritative tied parameter and accumulated gradient remain FP32.

Projection gradients also accumulate directly into persistent FP32 sinks.
PKDA Q/K/V gradients occupy disjoint row views of one allocation; dense QKV
and gate gradients share another. Each packed projection's backward writes
one GEMM into that allocation, while its parameters and optimizer states
remain separate. Each parameter view is clipped and updated once; the shared
allocation adds no duplicate gradient or checkpoint state.

The head's own classifier gradient reaches that FP32 gradient through a second
address-stable BF16 buffer: every head call lock-adds into it, and the captured
trainer adds it into the FP32 sink and clears it whenever another microbatch
would take the buffer past `--head-flush-every` head calls, and once more before
the optimizer reads the step. A pass makes one head call without `m` and two
with it. Flush scheduling counts the actual head calls, including auxiliary
heads, rather than treating a pass as one call. A captured microbatch whose
head count exceeds `--head-flush-every` uses per-call accumulation into the
FP32 sink instead; setting the limit to one gives per-call precision in every mode.
The BF16 buffer is a runtime operand like the shadow, not state.

## NorMuonH and NAdam

Every condition at every geometry uses one optimizer recipe with three
disjoint parameter groups: one NorMuonH group and two NAdam groups, base and
width-scaled. Their public controls are `--lr-normuonh` and `--lr-nadam`,
with defaults in [design.md](design.md#knobs); `--lr-nadam` is the base rate
at the muP reference width. No group uses weight decay.

### NorMuonH matrices

NorMuonH owns ordinary trainable two-dimensional hidden weights, including:

- attention and PKDA Q/K/V/output projections;
- dense, shared, and routed SwiGLU input and output matrices;
- the FBT payload-value projection;
- the MTP concatenation projection and auxiliary attention/SwiGLU matrices.

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
parameter, in two groups. The **width-scaled group** holds the NAdam matrices
whose fan-in is the residual width `D`:

- GGQA gate matrices;
- expert router matrices;
- the FBT token-gate matrix;
- PKDA's packed control projection.

The **base group** holds everything else NAdam owns:

- tied embedding/readout;
- PKDA's main-decay and output-gate expansions, fan-in 128;
- RMSNorm weights, router queries and nulls, depthwise convolutions, biases,
  rates, time constants, and preconditioner centers.

The width-scaled group runs at `lr_nadam * 1536 / D`; the base group runs
at `lr_nadam`. The tied readout carries the same width ratio while its
embedding lookup keeps the base learning rate. NorMuonH uses one relative
learning rate across widths. These are the current width-transfer rules;
their effectiveness is an experimental question.

| | Screen | Bridge | Flagship |
|---|---:|---:|---:|
| Width ratio `1536 / D` | 2 | 4/3 | 1 |
| Gate/control initialization standard deviation | 0.02828 | 0.02309 | 0.02 |
| Width-scaled NAdam rate at `lr_nadam = 3e-4` | 6e-4 | 4e-4 | 3e-4 |
| Readout multiplier | 2 | 4/3 | 1 |

Both groups use moment betas `(0.9, 0.95)`, momentum decay `psi = 0.004`,
epsilon `1e-8`, and no weight decay. At optimizer step `t`, PyTorch NAdam
uses:

```text
m_t = beta1 m_{t-1} + (1 - beta1) G_t
v_t = beta2 v_{t-1} + (1 - beta2) G_t^2
mu_t = beta1 (1 - 0.5 * 0.96^(t psi))
P_t = product_{i=1}^t mu_i
U_t = ((1 - mu_t) G_t / (1 - P_t)
       + mu_{t+1} m_t / (1 - P_t mu_{t+1}))
      / (sqrt(v_t / (1 - beta2^t)) + eps)
```

The update is `theta_t = theta_{t-1} - lr * U_t`. All three parameter groups
receive the same warmup-stable-cooldown multiplier defined by the current
scale's schedule. CUDA uses PyTorch's foreach NAdam path outside the captured
forward/backward graphs; its scalar step and momentum-product state stay on
CPU, while both moment tensors and all parameters remain FP32 on-device.
NorMuonH compiles each active shape bucket's packing, mathematical update, and
in-place state writeback together. Its checkpoint state remains ordinary
per-parameter tensors; the compiled path retains no second packed state.
With `e`, packing is split into batches of at most 33,554,432 matrix elements
(128 MiB per FP32 packed tensor), except that a larger individual matrix
remains whole. This bounds each temporary without splitting a matrix's
NorMuonH update.

After synchronized microbatch accumulation, the single global FP32 gradient
vector is clipped to L2 norm 10.0 immediately before both optimizer steps. The
reported gradient norm is the pre-clip norm, and a non-finite norm terminates
the run.

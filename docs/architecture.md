# Architecture

`DeltaModel` has one core architecture: PKDA/GQA token mixers, Multi-Head Delta
Block (MHDB) routing, shared and routed SwiGLU experts, and auxiliary two-token
prediction. Conditions select cross-column feedback (`f`), looped depth (`l`),
or both (`fl`); the default is `f`. Both recurrences are one operator: a
column writes a payload, and the shared DeepSeek-style concat-linear fusion
seeds the next column from that payload and a raw token embedding. Feedback
fuses the payload with the next position's token, the Full-Bandwidth
Transformer (FBT) recurrence with Jacobi training; the loop fuses it with the
same position's token and re-runs the whole column, in the Ouro style. The
payload is normalized once at its writer with a learned gain, and a token
lookup is the tied embedding row times the fixed `1/BASE_NORMAL_INIT_STD`,
so both fusion inputs are unit RMS at initialization and the seed enters the
residual stream at unit scale. There is no empty condition.

[Scaling](scaling.md) owns geometry and accounting, [design](design.md) owns
data and training, and [references](../references/refs.yaml) records sources.

The [TikZ figure](architecture.tex) shows `fl` as one vertical diagram, with
the shared fusion feeding MTP and both recurrences, a cell-delta source bank,
and one layer cutaway.
Build its vector PDF from the repository root with Tectonic:

```bash
mkdir -p figures/architecture
tectonic --outdir figures/architecture docs/architecture.tex
```

The editable source is tracked; generated files in `figures/` are not.

## The column

The pre-norm decoder has tied embedding/readout, a final RMSNorm, and no
dropout. Every RMSNorm uses epsilon `1e-6`. `ResidualEmbedding` returns raw
token features, cast to the residual dtype:

```text
e_t = embed_tokens(x_t) = E(x_t)
```

The tied classifier uses the same raw `embed_tokens.weight = E` matrix.
CUDA lookups cast to BF16 and accumulate tied embedding gradients in FP32;
there is no input token normalization.
Layers form four-layer cells `[PKDA, PKDA, PKDA, NoPE-GGQA]`.
Each trunk FFN uses one shared expert
and a configurable number of selected routed experts. All presets have four
cells. Under `l` the whole column runs `r` times per position, each run
seeded by the preceding run's payload.

```text
token embedding + incoming payload (f), the preceding column's own
payload (l), or the learned blank payload
  -> shared concat-linear fusion -> column seed
  -> four cells
  -> top state -> tied readout -> next-token logits
       + routed block deltas -> learned RMSNorm -> payload
                                  (+ shared jitter in training)
                                  + same-token embedding
                                  -> shared concat-linear fusion
                                     -> next column seed at this position (l)
                                  + raw next-token embedding
                                  -> shared concat-linear fusion
                                     -> next column seed at the next position (f)
                                     -> independent auxiliary PKDA/expert block
                                        -> tied readout -> second-token logits
```

Each cell has four residual layers, each with its own token mixer and expert
FFN. An MHDB read enriches each branch's input with the seed and cell deltas.
Linear maps are bias-free except PKDA's output-gate expansion.

Every column seed is the shared fusion of the token embedding `e_t` with
a payload. Pass 1, plain-prefix positions, and Standard decoding, which have
no incoming payload, fuse the learned blank payload `p_0` instead. The readout applies
`final_norm(h_top) * 1536 / D` before the tied classifier.
PKDA and GQA caches carry additional state along tokens; the payload carries
feedback between positions, between Jacobi passes, and between the looped
columns of one position.

### Residual shell and sources

For `L` unique layers, every mixer and FFN branch scales by `1/sqrt(2L)`:

```text
a_l = mixer_l(rmsnorm(h_l + route_l(sources))) / sqrt(2L)
h'_l = h_l + a_l
m_l = ffn_l(rmsnorm(h'_l + route'_l(sources'))) / sqrt(2L)
h_(l+1) = h'_l + m_l
```

Routing changes only the temporary input read. It never replaces the residual.
For seed `s`, the top therefore telescopes as
`h_top = s + sum_l(a_l + m_l) = s + sum(completed cell deltas)`.

A cell delta is its residual contribution. While a cell runs, that contribution
is the current partial; at cell exit it becomes a completed source. A site's
bank contains its own learned null, the column seed, completed cell deltas,
and the current partial when present. At the first attention entry the partial
is absent. The FFN site sees the newly added attention contribution. Branch
deltas never become individually addressable sources.

## PKDA and gated global GQA

### PKDA mixer

Each PKDA layer combines Kimi Delta Attention with the stable diagonal
apply-to-key preconditioner from Preconditioned DeltaNet. Key/value head width
is 128; projection width is five thirds of residual width at the presets.
Bias-free Q/K/V projections pass through separate causal width-4 depthwise
convolutions and SiLU. Q and K receive per-head L2 normalization, and Q scales
by `1/sqrt(128)`.

A packed projection produces disjoint slices for the rank-128 main decay,
per-head delta update, preconditioner decay and gain, and rank-128 output gate.
Packing preserves independent parameter ownership. Main and preconditioner
controls are not tied. The preconditioner uses squash bound 1.5, epsilon
`1e-6`, learned log-space center initialized to `-0.2`, decay rates sampled
on `[1,16]`, and softplus time constants sampled log-uniformly on `[.001,.1]`.
Negative-eigenvalue and safe-gate modes are disabled.

For a head's key-by-value matrix `S_t` and nonnegative diagonal state `A_t`,
both initially zero:

```text
alpha_t      = exp(-exp(A) * softplus(decay_t + b))
beta_t       = sigmoid(update_t)
alphaP_t     = exp(-exp(A_P) * softplus(decayP_t + b_P))
betaP_t      = sigmoid(updateP_t)
A_t          = alphaP_t A_(t-1) + betaP_t (k_t ⊙ k_t)
r_t          = log(A_t + 1e-6) - center
B_t          = exp(-log(1.5) * r_t / (1 + abs(r_t)))
k_write_t    = B_t ⊙ k_t
S_tilde_t    = Diag(alpha_t) S_(t-1)
S_t          = (I - beta_t k_write_t k_t^T) S_tilde_t
               + beta_t k_write_t v_t^T
o_t          = S_t^T q_t
```

Each coordinate of `B_t` lies in `[2/3,3/2]`. The output receives head-wise
normalization, a rank-128 sigmoid output gate, and a bias-free output
projection. The output-gate expansion bias initializes to zero.

CUDA training/prefill uses the workspace FLA chunk operator at chunk size 64;
decode uses its recurrent operator. Training's backward rebuilds the
recurrence's WY representation, chunk states, and gate cumsum from the
retained intra-chunk products and preconditioner scan, bitwise identically to
keeping them. CPU/MPS use the literal recurrence.
Convolution, SiLU, and Q/K normalization stay FP32 until the final activation
cast. Decode retains the FP32 matrix and diagonal states plus three BF16
projected convolution histories of length 3. Each Jacobi pass restarts those
states; only the payload crosses passes. The final prefill's caches then
advance normally during generation.

### Gated global GQA

The fourth layer of each cell uses causal NoPE GQA, head width 192. At every
preset, the query/output-gate projection width is `2D` and each K/V projection
width is `D`; the output projection returns to residual width `D`:

```text
q, k, v = split(W_qkv x)
q, k    = rmsnorm_q(q), rmsnorm_k(k)
z       = concat(GQA(q, k, v))
o       = W_o(sigmoid(W_g x) * z)
```

The output gate acts before projection and residual scaling. It changes
neither attention logits nor softmax weights. CUDA full rows use native
Flash SDPA for BF16/FP16 and math SDPA for FP32 diagnostics. Cached decoding
stores BF16 K/V and uses FlexAttention over the valid prefix. Only the single
GQA layer in each trunk cell owns K/V storage.

## Shared and routed experts

Every trunk and auxiliary FFN holds one shared SwiGLU `S` and `n` routed
SwiGLUs `E_0..E_(n-1)`, selecting `k` routed experts per token. Each reads and
writes residual width `D` and uses actual intermediate width `h`.
`ModelConfig` names these fields `expert_intermediate`, `num_routed_experts`,
and `experts_per_token`; their CLI flags are `--expert-intermediate`,
`--num-routed-experts`, and `--experts-per-token`. Width and bank size must be
positive and `1 <= k <= n`. Presets fix `h=832` and use `(k,n)` of `(3,15)`,
`(5,23)`, `(7,31)`, and `(11,47)`. For normalized input:

```text
s = sigmoid(W_router x)
J = topk(s + b)
w_j = s_j / sum_(i in J) s_i  if j in J, else 0
ffn(x) = (S(x) + k * sum_j w_j E_j(x)) / sqrt(k+1)
```

Router scores are FP32. The persistent bias `b` affects selection only. The
selected scores remain differentiable; the discrete choice does not. There
is no capacity limit or token dropping. CUDA dispatches every token's `k`
assignments in expert-major order, stable in token and slot, through a
counting sort, and each expert's rows form one tile range of the grouped
GEMMs; the outputs are gathered back per token and mixed in FP32.
Multiplying by `k` gives unit selected coefficients at equal scores; division
by `sqrt(k+1)` matches the variance of `k+1` independent equal-variance
expert outputs to one expert output at initialization. This does not
guarantee learned variance.

The active dense-equivalent width is derived as `H=(k+1)h`: expert matrices
perform `3DH` multiply-accumulates per token and store `3D(n+1)h` parameters.
The presets store four times the active expert matrix parameters. The router
adds `nD` parameters per bank. The shared expert accounts for `1/(k+1)` of
active expert width: 25%, 16.7%, 12.5%, and 8.3% across the presets. Fixed expert
intermediate width does not fix each expert's parameter count as `D` grows.
Memory and dispatch overhead still matter. Near-tied expert rankings can amplify numerical differences between cached
and full-row evaluation.

### Load balancing

Each physical bank's `expert_bias[n]` starts at zero and stays outside the
optimizer groups. Training sums actual integer assignment counts `C_j` over
all microbatches, passes, and invocations, then updates once after the step:

```text
b_j += 0.001 * sign(sum_i C_i - n * C_j)
```

Evaluation and recomputation leave it fixed. Counts are returned values, so
activation checkpointing cannot double-count.

A separate sequence regularizer uses unbiased top-`k` preferences:

```text
q_j = s_j / sum_i s_i
U = topk(s)
P_j = mean_tokens(q_j)
F_j = stop_gradient(count_tokens(j in U) / (kT))
aux_layer = mean_sequences(n * sum_j P_j F_j)
aux_pass = (sum_executed_trunk_layers(aux_layer) + aux_mtp) / (N_trunk + 1)
aux = mean_passes(aux_pass)
training_loss += 1e-4 * aux
```

Layer/pass averaging keeps the
coefficient independent of depth and pass count. The auxiliary bank contributes
once per pass, with its own `T` positions, including the padded last one whose
cross-entropy is unweighted: it still selects experts, so it enters that bank's
counts and sequence balance as a one-in-`T` perturbation. The bank's
within-sequence gradient coupling does not change causal forward activations.
Cross-entropy excludes it.

`ColumnOutput.expert_aux_loss` holds the trunk layer mean and `expert_counts`
holds transient `[L,n]` counts summed over physical-bank invocations.
`LossOutput.expert_aux_loss` includes the auxiliary bank using the averaging
above; its `[L+1,n]` counts place that bank last. Bias updates include every
bank once per optimizer step. With `want_weights=True`, column `expert_weights`
maps each trunk invocation to `[B,T,n]` sparse normalized weights, distinct
from MHDB's source weights. `forward_mtp` can return the auxiliary block's
weights over its `T` positions.

## Multi-Head Delta Block routing

Each site owns a zero-initialized width-`D` query `q`, a learned RMS key scale
`g` initialized to one, and a zero-initialized learned null value. For
`H = kv_heads` contiguous feature groups and raw source values `v_i`:

```text
k_i       = g * v_i / sqrt(mean(v_i^2) + eps)
score_i,h = dot(q_h, k_i,h)
weight_i,h = softmax_i(score_i,h)
route_h   = sum_i weight_i,h * v_i,h
```

The RMS statistic spans the full width; each group's softmax mixes only its
own raw feature slice. There is no head-dimension score factor or output
projection. The site prepends its own null, whose learned value must be
considered alongside its mass when interpreting routes.

The non-null bank reconstructs the current residual. At initialization the
zero query makes the mixture uniform and collinear with that residual, so
the following RMSNorm makes pass-1 routed reads inert up to epsilon. With
`C` cells, at most `C+2` sources appear: null, seed, and completed/partial
cell deltas. MHDB uses the source-selection idea of multi-head Delta Attention
Residuals with cell-level addresses.

## Payload and letter f: feedback

At a feedback position, fusion concatenates the token embedding `e_t`
with incoming payload `p_(t-1)` and projects back to width `D`. The payload
was normalized once at its writer. With jitter disabled,
the same fusion used by MTP gives the column seed:

```text
seed_t = fuse_proj(concat(e_t, p_(t-1)))
```

The bias-free `fuse_proj: 2D -> D` is shared by MTP and feedback. Fusion is
only concatenation and projection: it applies no normalization, activation,
or extra scaling. Splitting its matrix into two width-`D` blocks gives
`W_e e_t + W_p p_(t-1)`, so both inputs contribute directly to the seed. This adapts
[DeepSeek-V3's MTP entry, Equation 21](https://arxiv.org/html/2412.19437v2#S2.SS2):
the routed payload replaces its preceding hidden state and is normalized at
the writer before training jitter, the token input is the scaled lookup
without normalization, and the result is shared with cross-column feedback. Plain-prefix, pass-1,
and Standard-decoding positions use the same product with the learned blank
payload `p_0` in place of `p_(t-1)`:

```text
seed_t = fuse_proj(concat(e_t, p_0))          (no incoming payload)
```

`p_0` is a width-`D` vector in payload units, zero-initialized and
present in every condition, so every seed passes through `fuse_proj` and a
feedback position differs from a plain one only by `W_p (p_(t-1) - p_0)`.
Neither the blank nor its fusion receives jitter. The actual seed is both residual
origin and MHDB source. Every condition has a dedicated payload router that reads its
null, seed, and every completed cell delta:

```text
r_payload = route(null, seed, Delta_0 .. Delta_(C-1))
payload_norm(z) = BASE_NORMAL_INIT_STD * RMSNorm_g(z)
p_t = payload_norm(h_top + r_payload)
```

The writer's `payload_norm` is a learned RMSNorm whose gain `g` starts at one.
Its fixed output multiplier is `payload_norm.scale = BASE_NORMAL_INIT_STD`,
currently `0.02`. The same constant sets raw token embedding initialization
and the reference amplitude for training jitter. Initially the payload has
RMS approximately `0.02`; its learned gain can change that amplitude during
training. The direct `h_top` term remains alongside routed enrichment.
Initially the uniform enrichment is
`h_top/(C+2)`, making the payload equal to `0.02` times a normalized top state
up to epsilon. Payload routing is not conditioned on the next token; that
token enters through the shared fusion at the next column.
Every supervised column constructs this payload for MTP, including
single-pass batches, `l` without `f`, and the final feedback pass. The `f`
letter controls consumption by the next position's column and the `l` letter
by the same position's next column. Generation can skip payload construction
when no feedback, loop, or auxiliary read needs it.

Every training column adds its keyed jitter, already in payload units,
to the undetached payload and fuses it with the raw next-token embedding
once.
There is no second payload normalization after jitter. MTP consumes the
resulting tensor directly. If another Jacobi pass follows, it shifts that
same tensor right and restores the per-row plain prefix. If another looped
column follows at the same position, the same jittered payload is fused once
more with the position's own raw embedding to seed it. Mixer states restart
inside each pass.
Sequential generation instead retains the preceding payload and advances all
mixer caches once per new token, computing one fusion with that token's
raw embedding and no jitter, and under `l` running the evaluation column
count before the caches advance; Standard decoding fuses the blank payload.

## Letter l: the looped column

Under `l`, every position runs the whole column `r` times per pass. The
first column is seeded as in the flat model. Each later column is seeded by
the same shared fusion feedback uses, with the preceding column's payload in
the payload slot and the position's own raw embedding in the token slot:

```text
seed_t^(1)   = fuse_proj(concat(e_t, p_(t-1)))            (or p_0)
p_t^(i)      = payload_norm(h_top,t^(i) + r_payload,t^(i))
seed_t^(i+1) = fuse_proj(concat(e_t, p_t^(i) + jitter_t^(i)))
```

The loop adds no parameters, and every column is a plain column: the same
bank of null, seed, and cell deltas at every site, the same telescoping
`h_top = seed + sum(cell deltas)` within each column, and a fresh residual
origin at every column. Feedback and the loop are one operator applied at
different positions: `f` fuses the payload with the next position's token,
`l` with the same position's token. A position's payload slot therefore
always holds the most recent payload for that position, whether it came
from the left neighbor or from the preceding column, and under `fl` a later
column carries the cross-column payload through its seed. At one column,
`fl` equals `f` in values, routes, losses, and gradients. A column never
depends on the columns after it, so the first column of a looped run is the
single-column model exactly, and reading out after every column is one
trajectory.

Every column is read out and supervised, and every column's payload feeds
MTP. Contraction is not established by construction: the payload writer's
normalization and fixed scale bound every column's seed, but the column's
own contribution is unconstrained. [Ouro](../references/refs.yaml) loops
the whole transformer stack with an exit at every step; this model re-enters
through the shared fusion, so the raw token embedding is re-injected and the
state crosses the boundary as a normalized payload rather than a raw
residual.

### Depth, caches, and training

The recurrence roll draws once per optimizer step, and every pass, column,
and microbatch of the step shares the result;
[design](design.md#the-recurrence-roll) defines it. `f` reads its pass
component, `l` its column component, and `fl` both, so a step has the same
shape in every condition that shares the roll. Evaluation and decode hold
`r = r_eval` fixed for the request, defaulting to two columns.
`--loop-iterations` sets this fixed count within `1..3` and does not change
the training distribution.
Each column has its own mixer cache track at every layer and reads earlier
token positions' writes at the same column index. Every current prefill pass
starts those tracks from zero, and sequential decode advances the shared
position once per token, after its last column. A pass executes `4r` cells.

All passes and columns remain differentiable. Readout occurs after every
column, and auxiliary prediction runs once per column. On CUDA the
per-column epilogues around the compiled blocks (the payload's routed add,
norm, and scale, the shared jittered fusions, feedback shift/prefix
selection, and both heads' readout) compile as their own small regions. The
compiled blocks carry an Inductor activation memory budget of 0.9, the
partitioner tier that recomputes cheap fused tensors in backward and never a
custom operator. CUDA captures one training graph per reachable
`(pass count,r)` shape and no-grad evaluation graphs at the fixed count. Every
graph, the optimizer step, and the periodic monitors share one private memory
pool: the eager work runs on the capture stream inside it, so nothing between
replays grows memory outside the pool, which needs the allocator's expandable
segments. At
start-up the trainer measures, from two eager forwards, the activation bytes
one block invocation retains and the bytes one recomputed block releases,
takes the device memory still free once the static footprint exists less a
2 GiB default margin for backward workspaces, recomputation, allocator
rounding, and graph instantiation, and plans each graph: a single-column
graph replays the largest row multiple whose raw
activations fit, and otherwise the first PKDA and auxiliary block invocations
of the logical forward, as many as the shortfall needs, recompute in
backward. Global-attention blocks are always retained. Checkpoint wrappers
remain outside compiled blocks.

`multipass` returns `[pass][column]` outputs, and `forward_iterations` runs
every column of one position range. `depth_trace` runs the maximum count
and reads out after every column, reporting each column's held-out loss and
top-state update norm. The depth telemetry labels the fixed evaluation
count `r_eval` and its held-out loss `loss_eval`, beside `loss_one` for the
first column and `loss_max` for the last. The payload self-composition
trace holds the count fixed. Paired `f`/`fl` runs share initialization,
rows, schedule, the roll, and the feedback draws; the loop's own jitter is
drawn after them. Equal steps match data; compute comparisons need
cell-tokens and auxiliary work or device time.

## Two-token prediction

One independent auxiliary prediction block runs after every column.
Its input is computed by the model's shared concat-linear fusion:

```text
e_(t+1) = embed_tokens(x_(t+1))
p_t = payload_norm(h_top,t + r_payload,t)
jitter_t ~ Uniform(-BASE_NORMAL_INIT_STD * jitter, BASE_NORMAL_INIT_STD * jitter)
u_t = fuse_proj(concat(e_(t+1), p_t + jitter_t))
v = PKDAExpertBlock(u)
logits_mtp,t = (final_norm(v_t) * 1536 / D) @ embed_tokens.weight.T
```

`fuse_proj` is the exact projection used by feedback, present in every
condition. The two inputs each have width `D`, and the concatenation places
the token first and payload second. One `embed_tokens` call looks up the whole
stored row before slicing; those raw embeddings serve the blank-fused plain
seeds, MTP, feedback, and the loop across all passes. Each column adds
jitter after the writer's payload normalization and fixed scale, then
computes `u` once for MTP and the next pass; a following looped column
fuses the same jittered payload with the position's own embedding.
Fusion accepts these inputs directly, with no normalization or second scaling.

The auxiliary block has its own PKDA mixer and one shared plus top-`k`-of-`n`
routed SwiGLU experts, using the trunk's residual width, expert intermediate
width, expert counts, output normalization, and `1/sqrt(2L)` branch scale.
It shares the final norm and embedding/readout.
It has no MHDB read or tied loop. Its independent matrix, diagonal, and
convolution states start from zero for each row and pass.

`--jitter` is a half-width in payload units, which are unit RMS at
initialization, with default `0.02`. Training samples the buffer directly
in those units: uniform `[-jitter, jitter]`. This draw happens for every supervised
column, including the final or only pass and `l` without `f`. MTP, the next
looped column, and the next feedback pass use the same realization. For the
latter, `u_t` becomes the seed at position `t+1`, with positions inside the
selected plain prefix restored to their blank-fused seeds. The auxiliary block's output `v`
is used only for MTP; it is not an early trunk readout or an extra feedback layer.
Evaluation and diagnostics set jitter to zero.

For a stored row of `T+1` tokens, ordinary inputs/targets are
`tokens[:,:-1]` / `tokens[:,1:]`. Auxiliary inputs fuse the whole `payload`
and the raw embeddings for `tokens[:,1:]`, with targets `tokens[:,2:]`
plus one dummy target: `T-1` supervised positions and a padded last one,
whose second token lies past the end of the stored row and which therefore
carries no loss weight. That row
is causally last in the auxiliary recurrence, so the supervised rows are the
same ones the cropped geometry produced. One lookup of the whole stored row
serves both heads, the blank-fused plain seed taking positions `0..T-1` and the
payload fusion taking next-token features at `1..T`.
Both inputs and the reused fused tensor remain differentiable, so the
auxiliary objective trains the payload router, payload normalization, trunk,
shared fusion, and token embedding.
The causal PKDA recurrence never sees the second-token target. The auxiliary
block's output never feeds into the column or payload.
`multipass` exposes each column's shared tensor as `ColumnOutput.fused_input`.
`DeltaModel.forward_mtp_fused` runs the block on this tensor;
`forward_mtp(payload, next_tokens)` embeds and fuses supplied next-token IDs
as a convenience. Both return balancing loss, assignment counts, and optional
expert weights.

Each head averages over its own valid positions, and both heads' rows go
through one unreduced vocabulary pass per column: their readout rows are
concatenated into a single cut cross-entropy request, and the column scalars
are weighted sums of the returned rows. For either head, `combine` applies
FBT's first-plus-mean rule along both recurrence axes: each column's series
is combined along passes, `ell_1 + mean(ell_2,...,ell_k)`, or `ell_1` alone
at one pass, and the per-column results are combined the same way along
columns. At `k` passes and `r` columns the plain first column, pass 1's
later columns, later passes' first columns, and the remaining block each
carry unit weight, spread evenly inside the block, so with one axis absent
the rule is the other axis's own combine:

```text
loss = combine(CE_ntp) + z_coef * combine(z_ntp)
     + mtp_weight * (combine(CE_mtp) + z_coef * combine(z_mtp))
     + 1e-4 * expert_balance
```

The cooldown z-loss is the mean squared log-partition. `--mtp-weight` is a
finite nonnegative constant, default 0.3. `ntp` and `mtp` report cross-entropies
before coefficients/z-loss, `pass1` is the first column's ordinary CE, and
`loss` is the optimized objective. `val_mtp` and feedback's `val_mtp_fused` remain
separate from ordinary validation. Generation does not run the auxiliary
block or allocate its cache; speculative decoding is not implemented.

## Precision and initialization

NorMuonH matrices initialize from `Normal(0,1/sqrt(fan_in))`. The shared
`BASE_NORMAL_INIT_STD=0.02` sets NAdam matrix standard deviations. GGQA gates,
expert routers, and PKDA's packed control projection multiply
it by `sqrt(1536/D)`. Embeddings and PKDA fixed-head-width expansions use
0.02 directly. This preserves initial control-logit variance across widths.
`MUP_BASE_DIM` stays at the flagship width 1,536: extension uses a width ratio
of `1536/2304 = 2/3` for NAdam rates and readout scaling, and its square root
for the specified initialization scales. The reference does not depend on
the largest configured preset. Expert optimizer rates have their own factors
below; they do not change initialization or forward computation.
The fusion projection uses NorMuonH initialization with fan-in `2D`, hence
standard deviation `1/sqrt(2D)`. The same `BASE_NORMAL_INIT_STD` supplies the
payload RMSNorm's fixed output scale and the jitter generator's reference
amplitude; it does not change fusion projection initialization. Depthwise
convolutions retain Kaiming-uniform initialization. Learned RMSNorm gains
start at one; MHDB queries and nulls and the blank payload start at zero.

Shared parameters pair byte-identically across conditions for a given seed.
Experts are constructed directly as part of the common model initialization.
Attention gates, the fusion matrix, and the auxiliary module use separate
deterministic streams. Fusion initialization does not advance shared draws.
Feedback/depth recipe draws are separately keyed.

Parameters, accumulated gradients, and optimizer state are FP32, except
NorMuonH's momentum, which CUDA stores in BF16 rounded to nearest with its
update computed in FP32. CUDA residuals,
routed values, payloads, and mixer caches use BF16, except PKDA matrix/diagonal
boundaries. CCE reads an address-stable BF16 classifier shadow refreshed after
each optimizer step; it is runtime state, excluded from snapshots.

Packed projection backpropagation writes into persistent FP32 sinks. PKDA
Q/K/V row views share an allocation, as do dense QKV and gate gradients;
each parameter is updated once. The routed experts' backward keeps
the SwiGLU derivative in FP32 inside the epilogue of the activation-gradient
GEMM and recomputes the activation there, so the forward retains only the
pre-activation; expert weight gradients accumulate in FP32 sinks, and an
expert without assignments leaves its sink untouched. Head calls first accumulate classifier
gradients in a BF16 buffer, flushed into the FP32 sink at the configured
`--head-flush-every` cadence and before the optimizer update. Each column
makes one head call, covering both prediction depths. A captured microbatch
exceeding the cadence uses the FP32 sink per call; cadence 1 therefore gives
per-call precision throughout.

## NorMuonH and NAdam

Five disjoint parameter groups across two optimizers share the
warmup-stable-cooldown multiplier with no weight decay:

| Group | Parameters | Peak learning rate |
|---|---|---|
| `normuonh` | Ordinary NorMuonH matrices outside experts | `lr_normuonh` |
| `normuonh_expert_in` | Shared and routed expert gate/up matrices | `lr_normuonh * sqrt(8/(k+1))` |
| `normuonh_expert_out` | Shared and routed expert down matrices | `lr_normuonh * sqrt(8/(k+1))` |
| `nadam` | Base NAdam parameters | `lr_nadam` |
| `nadam_width` | NAdam matrices with residual-width fan-in | `lr_nadam * 1536/D` |

Here `k` is the selected routed expert count, so `k+1` includes the shared
expert. Both expert groups include every trunk bank and the auxiliary MTP
bank. Both expert groups use the same active-count factor, anchored at
flagship's eight active experts. Coherently aligned branch changes add as
`sqrt(k+1)` after the bank's forward normalization; the count factor offsets
that growth. Each matrix's spectral normalization separately handles its
fan-in/fan-out, including geometry overrides. No extra `sqrt(1536/D)` expert
input factor is applied.

### NorMuonH matrices

NorMuonH owns ordinary 2D hidden matrices: token-mixer projections, expert and
auxiliary FFNs, and the entire shared `D x 2D` fusion projection. Every matrix
keeps its initial FP32 Frobenius radius `R`, stored in the checkpoint:

```text
M_t = .95 M_(t-1) + .05 G_t
N_t = .05 G_t + .95 M_t
U = five_Newton_Schulz_steps(N_t)          # BF16 on CUDA, FP32 elsewhere
U = row_second_moment_normalize(U, beta=.95, eps=1e-8)
U = Normalize_F(U)
T = U - <W,U>_F / <W,W>_F * W
T = Normalize_F(T)
sigma_hat, v_next = three_power_iterations_with_restart(T, v)
eta_t = schedule_multiplier(t) * group_peak_lr
W_next = R * Normalize_F(W - eta_t * sqrt(fan_out/fan_in) * T / sigma_hat)
```

The default base rate is `6e-3`, an estimated RMS-to-RMS operator budget for
the tangent trial step; the expert groups apply the factors above. The
spectral estimate follows row adaptation and removal of the radial component.
Each update compares the image of the saved right vector with the image of
the normalized largest-energy row, then performs three paired power
iterations from the better start and one final norm evaluation. The restart
can recover a newly rotated direction orthogonal to the saved vector and
consumes no RNG. The right vector is FP32 optimizer state, saved alongside
momentum, row moments, and radius for exact resume.

Zero and numerically radial directions produce no weight movement; after
normalizing `U`, tangent norms at or below `32 * finfo(dtype).eps` are treated
as cancellation residue. Zero-rate updates also preserve weights exactly.
An exact spectral norm would set the trial's RMS-to-RMS norm to the group
rate. Power iteration can underestimate it, and the sphere retraction changes
the finite displacement, so this is not a strict final-step bound. Full-model transfer of these rates across scales remains unmeasured.
[Scaling](scaling.md#expert-learning-rates) lists the preset rates.
CUDA compiles the update arithmetic by shape bucket within each group and
runs the five Newton-Schulz iterations on a BF16 copy of the normalized
direction, as reference Muon implementations do. CUDA also stores each
bucket's momentum in BF16: the EMA and the Nesterov direction are computed in
FP32 and only the writeback rounds to nearest, so a snapshot carries a BF16
momentum and resumes exactly. The result returns to FP32
before row adaptation, the spectral/tangent step, and the retraction; CPU and
MPS stay FP32 throughout.

Bucket membership is fixed at construction from device, dtype, shape, and a
packing bound of 33,554,432 matrix elements (128 MiB per FP32 tensor), except
larger individual matrices remain whole. Each bucket holds its momenta, row
moments, radii, and right vectors in one packed tensor per state kind, and
each parameter's state entries are views into them, so a step packs only the
parameters and gradients, which are separate tensors owned elsewhere.
The saved schema stays per-parameter, and a restored state is copied into
the packed storage rather than replacing the views. A step that reaches only
some members of a bucket gathers the active rows and scatters the results
back, leaving absent members' weights and state untouched. All results
materialize before state and parameter writebacks outside the compiled
boundary, preserving reads of the previous momentum.

### NAdam parameters

The width-scaled group owns matrices with fan-in `D`: GGQA gates,
expert routers, and PKDA packed controls. It uses `lr_nadam * 1536/D`; the
base group uses `lr_nadam`, default `3e-4`. The base group owns the tied
embedding/readout, fixed-head-width PKDA expansions, and every non-matrix
parameter, including the payload-writer RMSNorm gain.
The readout also multiplies by `1536/D`, while embedding lookup keeps its
base rate.

NAdam uses betas `(.9,.95)`, momentum decay `psi=.004`, epsilon `1e-8`:

```text
m_t = beta1*m_(t-1) + (1-beta1)*G_t
v_t = beta2*v_(t-1) + (1-beta2)*G_t^2
mu_t = beta1*(1-.5*.96^(t*psi))
P_t = product_(i=1..t) mu_i
U_t = ((1-mu_t)*G_t/(1-P_t) + mu_(t+1)*m_t/(1-P_t*mu_(t+1)))
      / (sqrt(v_t/(1-beta2^t)) + eps)
theta_t = theta_(t-1) - lr*U_t
```

CUDA NAdam and the global FP32 gradient norm run outside forward/backward
graphs. The scalar step and momentum-product state remain on CPU; moments
and parameters remain FP32 on-device. Before both optimizers, the complete
accumulated gradient's L2 norm is measured and reported; a non-finite norm
stops the run. Nothing is clipped: NorMuonH's spectral step is scale-free,
NAdam's nearly so, and with the residual stream entering at unit scale the
graph shapes' norms sit within a factor of two of each other.

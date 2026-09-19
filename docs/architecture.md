# Architecture

The full `fl` model combines two differentiable recurrences around one shared
causal decoder: feedback (`f`) passes a predictive payload to the next token
position, and looping (`l`) re-enters the decoder at the same position. A
**column** is one execution of the entire trunk; a **pass** runs the requested
columns over a sequence. Columns share all parameters, but keep separate
mixer-state tracks across token positions.

The trunk uses Preconditioned Kimi Delta Attention (PKDA), gated global grouped
query attention (GQA), shared and routed SwiGLU experts, and Multi-Head Delta
Block (MHDB) reads. A routed payload and one shared fusion connect the trunk,
both recurrences, and auxiliary two-token prediction (MTP):

```text
scaled token lookup or blank embedding + incoming or blank payload
                   │ shared concat-linear fusion
                   ▼
                column seed
                   │
       four cells of [PKDA, PKDA, PKDA, gated GQA]
       each layer: MHDB read → mixer → MHDB read → experts
                   │
                top state ───────────────→ tied next-token readout
                   │ + MHDB read of seed and completed cell deltas
                   ▼
             payload RMSNorm
                   │ + one training jitter draw
                   ├─ fuse with blank embedding → next column at this position (l)
                   └─ fuse with next token ─┬→ next position's first column (f)
                                           └→ independent PKDA/expert block
                                              → tied second-token readout
```

[Scaling](scaling.md) owns preset sizes, parameter/compute accounting, and
width-dependent rates; the [training recipe](design.md) owns conditions,
sampling, schedules, and objective coefficients; [operations](operations.md)
owns execution and precision. [Mechanism references](../references/refs.yaml)
and the editable [architecture diagram](architecture.tex), with recurrence
paths and matrix-level component cutaways, accompany this specification. The
implementation is in
[model.py](../delta_feedback_experiment/model.py),
[pkda.py](../delta_feedback_experiment/pkda.py),
[moe.py](../delta_feedback_experiment/moe.py), and
[optim.py](../delta_feedback_experiment/optim.py).

## Column and residual decomposition

A column is a pre-norm decoder with `L` unique layers, tied embedding/readout,
a shared final RMSNorm, and no dropout or positional embeddings. The presets
use four four-layer cells. All RMSNorms use epsilon `1e-6`. Linear maps are
bias-free except PKDA's output-gate expansion.

Let `E` be the tied vocabulary table, `D` the residual width, and
`s₀ = BASE_NORMAL_INIT_STD = 0.02`. Token input and classifier use different
scalings of the same table:

```text
e_t          = E[x_t] / s₀
logits(h_t)  = (1536/D) · RMSNorm_final(h_t) · Eᵀ
```

The lookup is unnormalized: learned token magnitudes survive the fixed
multiplier. `E` initializes with standard deviation `s₀`, so lookup features
begin at approximately unit RMS. Both prediction heads share `E`, the final
norm, and the readout multiplier.

Each layer has two residual branches. MHDB enriches their temporary inputs;
only the branch outputs enter the residual stream:

```text
a_l       = Mixer_l(RMSNorm_attn,l(h_l + Route_attn,l(bank))) / sqrt(2L)
h'_l      = h_l + a_l
m_l       = Experts_l(RMSNorm_ffn,l(h'_l + Route_ffn,l(bank'))) / sqrt(2L)
h_(l+1)   = h'_l + m_l
```

For seed `s` and cell deltas `Δ_c`, this gives the exact algebraic decomposition
(up to floating-point arithmetic):

```text
Δ_c   = residual_at_cell_exit − residual_at_cell_entry
h_top = s + Σ_l(a_l + m_l) = s + Σ_c Δ_c
```

A read's source bank contains its own learned null, the column seed, completed
cell deltas, and the current cell's partial delta when one exists. At a cell's
first mixer read there is no partial. Its FFN read adds the new mixer delta
to that partial; subsequent reads see the accumulated contribution of the
cell so far. Branch outputs are never separate persistent source addresses.
With `C` cells, a bank has at most `C+2` sources.

## Multi-Head Delta Block routing

MHDB lets a branch choose which earlier cell contributions to emphasize
without changing the residual decomposition. Each site owns a width-`D`
query `q`, learned RMS key gain `g`, and learned null value. The query and
null start at zero; the key gain starts at one. For raw source values `v_i`
and `H = kv_heads` contiguous feature groups:

```text
key_i       = g ⊙ v_i / sqrt(mean_D(v_i²) + ε)
score_i,h   = dot(q_h, key_i,h)
weight_i,h  = softmax_over_sources(score_i,h)
route_h     = Σ_i weight_i,h · v_i,h
route       = concatenate_h(route_h)
```

Keys normalize over the entire residual width; each group's softmax mixes
only that group's raw value slice. There is no score divisor or output
projection. Routing groups partition the residual features, independently
of the token mixers' projected heads.

The zero query initially makes every source weight uniform. The null is
zero and the remaining sources sum to the current residual, so routing
initially adds a scalar multiple of that residual. The following RMSNorm
makes this enrichment inert up to epsilon. Learned queries then select
content-dependent mixtures. A learned null can itself carry content, so
its weight alone does not measure an absence of routing.

The payload writer uses the same primitive with its own parameters, over
its null, the seed, and every completed cell delta. MHDB source weights
are distinct from the expert-selection weights described below.

## Token mixers

### Preconditioned Kimi Delta Attention

PKDA mixes tokens through a recurrent key-by-value matrix instead of storing
all previous K/V rows. Each head has width 128 at the presets. Separate Q/K/V
projections pass through causal width-4 depthwise convolutions and SiLU.
Queries and keys receive per-head L2 normalization; queries additionally
scale by `1/sqrt(128)`. Values remain unnormalized.

One packed control projection has five disjoint slices: a rank-128 main
decay input, per-head update logits, preconditioner decay logits,
preconditioner update logits, and a rank-128 output-gate input. The two
rank-128 inputs expand to all head channels through separate learned maps.
Main decay and preconditioner controls have independent parameters.

For one head, let `S_t` be its key-by-value matrix and `A_t` its nonnegative
diagonal second-moment state. Both start at zero. Main decay `α_t` is a
vector over key channels; `αP_t`, `β_t`, and `βP_t` are scalars:

```text
α_t      = exp(−exp(a)  · softplus(decay_t  + b))
β_t      = sigmoid(update_t)
αP_t     = exp(−exp(aP) · softplus(decayP_t + bP))
βP_t     = sigmoid(updateP_t)

A_t      = αP_t A_(t−1) + βP_t (k_t ⊙ k_t)
r_t      = log(A_t + 1e−6) − center
B_t      = exp(−log(1.5) · r_t / (1 + |r_t|))
k_write  = B_t ⊙ k_t

S_decay  = Diag(α_t) S_(t−1)
error_t  = v_t − S_decayᵀ k_t
S_t      = S_decay + β_t k_write error_tᵀ
o_t      = S_tᵀ q_t
```

The delta update writes the error between the current value and the memory's
prediction of it. Its diagonal preconditioner tracks key-coordinate activity
and rescales only the **write key**: the prediction still reads with `k_t`.
Low-activity coordinates receive larger writes and high-activity coordinates
smaller ones, relative to the learned log-space center. The squash confines
each coordinate of `B_t` to `[2/3, 3/2]`, preventing unbounded inverse-moment
amplification. This bounds the preconditioner, not the complete recurrent
operator.

Each head's output receives learned RMS normalization and a sigmoid gate,
then the concatenated heads project back to residual width:

```text
PKDA(x)_t = W_o concat_heads(RMSNorm_head(o_t) ⊙ sigmoid(gate_t))
```

The head norm shares its channel gain across heads. The output gate acts on
retrieved content, independently of the gates controlling memory retention
and writing. Matrix and diagonal states accumulate in FP32. Cached execution
also retains the three projected convolution histories, each of length 3.
The literal portable recurrence and CUDA chunk/recurrent operators implement
this same state transition.

### Gated global GQA

The fourth layer of each cell supplies direct attention over the causal
prefix, including the current position. It has no positional encoding (NoPE), uses
head width 256 at the presets, and shares each K/V head between query heads:

```text
q, k, v = split(W_qkv x)
q, k    = RMSNorm_q(q), RMSNorm_k(k)
z       = concat_heads(softmax(q kᵀ / sqrt(d_head) + causal_mask) v)
GQA(x)  = W_o(sigmoid(W_gate x) ⊙ z)
```

Q/K normalization is per head. The sigmoid gate acts after attention and
before the output projection; it does not alter logits or softmax weights.
Unlike PKDA's fixed-size memory, this layer stores K/V for every position
in its cache track. Interleaving the two mixers combines compressed
recurrent memory with direct retrieval of earlier token representations.

## Shared and routed experts

Every trunk FFN and the auxiliary FFN has one shared expert `S` and `n`
routed experts, selecting `k` per token. Each expert maps `D → h → D`:

```text
Expert(x) = W_down(SiLU(W_gate x) ⊙ W_up x)
```

A token-local FP32 router computes sigmoid affinities. A persistent
non-gradient bias changes selection but does not enter the mixture weights:

```text
s_j       = sigmoid((W_router x)_j)
J         = topk_j(s_j + b_j)
w_j       = s_j / Σ_(i∈J) s_i               for j ∈ J
Experts(x)= (S(x) + k Σ_(j∈J) w_j E_j(x)) / sqrt(k+1)
```

All selected assignments execute; there is no capacity limit or token
dropping. The selected affinities remain differentiable through their
normalization, while top-`k` membership is discrete. At equal affinities,
`k w_j = 1`, so the shared and selected branches have equal coefficients.
The final divisor preserves one expert's variance when the `k+1` outputs
are independent with equal variance; it does not fix learned variance.

Two mechanisms balance utilization at different levels:

- **Step-level controller:** aggregate actual integer assignment counts
  `C_j` for each physical bank across all microbatches, passes, columns, and
  ranks, then update once after the optimizer step:
  `b_j ← b_j + η_bias sign(Σ_i C_i − n C_j)`, then `b ← b − mean(b)`.
  Top-`k` reads only differences between biases, so the mean is a free
  offset. Step counts are right-skewed, leaving more experts under target
  than over it, and the sign rule alone integrates that imbalance into the
  offset at a constant rate; removing it changes no routing decision.
  Biases start at zero, remain outside optimizer groups, and stay fixed in
  evaluation and backward recomputation.
- **Sequence regularizer:** let `q_j = s_j / Σ_i s_i`,
  `P_j = mean_tokens(q_j)`, and
  `F_j = stop_gradient(count_tokens(j ∈ topk(s)) / (kT))`.
  The bank's auxiliary loss is `mean_sequences(n Σ_j P_j F_j)`.
  These preferences omit the selection bias, so the regularizer acts on
  learned affinities independently of the controller.

The model averages the regularizer over trunk and auxiliary banks, then
over executed columns. Actual counts sum instead of averaging. Returned
counts describe logical forwards, so activation recomputation cannot count
a dispatch twice. The last padded MTP position still selects experts and
enters balancing, although it has no prediction-loss weight. Sequence
regularization couples gradients within a row; forward activations remain
causal. Coefficients live in the [training recipe](design.md).

## Payload and shared fusion

Every supervised column writes a width-`D` predictive payload:

```text
r_t = MHDB_payload(null, seed_t, Δ_0,t, …, Δ_(C−1),t)
p_t = RMSNorm_payload(h_top,t + r_t)
```

The direct top-state term preserves access to the complete residual;
routing can emphasize particular cell contributions. The writer does not
see the next token. Its learned norm gain begins at one, so payload RMS
starts near one and can change with training. Normalization alone does not
guarantee contraction of the column-to-column map.

One bias-free projection implements every entry to a column and to MTP:

```text
F(e, p) = W_fuse concatenate(e, p) = W_e e + W_p p
```

The concatenation places the token first. Fusion adds no normalization,
activation, gate, or scaling: learned token magnitudes and payload gains
reach its output directly. A learned width-`D` blank payload `p₀`, initialized
to zero, supplies positions with no incoming payload. Their seed is
`F(e_t,p₀)`; replacing the blank with a payload changes it by
`W_p(p − p₀)`. Its counterpart on the token side is a learned width-`D`
blank embedding `e₀`, in post-lookup token units and also initialized to
zero, which stands in for the token wherever a column re-enters at its own
position. Only `l` has it.

During training, each column adds one keyed perturbation `ξ_t` **after**
payload normalization. Its amplitude is specified directly in payload units
by the [recipe](design.md). The same jittered payload
serves all consumers of that column: the next looped column, MTP, and,
for the pass's last column, the next feedback pass. There is no detach or
second normalization. The blank receives no jitter; evaluation and decoding
use zero jitter.

## Feedback passes, looping, and decoding

Let `p_t^(a,i)` denote the payload at position `t`, pass `a`, column `i`,
and `R` the number of columns per pass. The first column of pass 1 uses
`F(e_t,p₀)`. Later columns of any pass use the blank embedding:

```text
seed_t^(a,i+1) = F(e₀, p_t^(a,i) + ξ_t^(a,i))
```

The first column consumes the token lookup `e_t`; every later column uses
the learned baseline `e₀`. Those later columns receive token-dependent
information through their payload and mixer caches, with no fresh token
lookup.

Later feedback passes instead take the preceding position's final-column
payload from the previous pass:

```text
seed_t^(a+1,1) = F(e_t, p_(t−1)^(a,R) + ξ_(t−1)^(a,R))
```

A per-row plain prefix overrides that equation with `F(e_t,p₀)`; position 0
is always plain. This is Jacobi execution: all positions in one pass use
the preceding pass's payloads, allowing parallel processing of the row.
Each pass starts its mixer states afresh, while gradients remain connected
through every payload and fusion.

Every loop iteration is a complete column with a fresh seed and source bank;
no cell delta is carried across its boundary as a separate source. Looping
adds only `e₀`. Its first column does not depend on later columns, and
setting `R=1` makes `fl` equal to `f` in values, routes, losses, and gradients
for matched inputs and randomness, with `e₀` outside the graph. Every
executed column is supervised.

For sequential decoding, the first column at position `t` consumes the
final payload from `t−1`; later columns loop at `t`. Each layer owns one
cache track per column index: column `i` reads earlier positions' column-`i`
writes. All tracks share a token position counter, which advances only after
the last column. PKDA tracks hold matrix, diagonal, and convolution states;
GQA tracks hold K/V rows. Standard decoding supplies the blank instead of
the preceding position's payload, while retaining the configured loop.
MTP is unnecessary for generation and has no decode cache.

The [analysis guide](interpretability.md#checkpoint-and-tensor-apis) lists
execution APIs; the [recipe](design.md#the-recurrence-roll) defines training
shapes and evaluation/decode depth.

## Auxiliary two-token prediction

Each column predicts `x_(t+1)` from its top state. Its independent auxiliary
block predicts `x_(t+2)` after conditioning the payload on `x_(t+1)`:

```text
u_t          = F(e_(t+1), p_t + ξ_t)
v            = AuxiliaryPKDAExpertBlock(u)
logits_mtp,t = logits(v_t)
```

The auxiliary block has its own PKDA mixer and expert bank with the trunk's
geometry and `1/sqrt(2L)` branch scaling. It shares the fusion, final norm,
and classifier, but has no MHDB reads or loop. Its causal recurrent states
start from zero for each row and invocation. It can see next-token inputs
through position `t+1`, never its `t+2` target. Its output serves only the
auxiliary readout and never feeds the trunk or payload.

For a stored row of `T+1` tokens, the trunk executes `T` positions and
predicts tokens `1..T`. MTP executes those same `T` positions, conditioning
on tokens `1..T` and supervising targets `2..T`. Its final position has a
dummy target and zero prediction-loss weight. This causally last padding
leaves all earlier auxiliary states unchanged. Each head averages over its
own valid targets.

The next-token fusion `u` is computed once per column and exposed as
`ColumnOutput.fused_input`. MTP reads it directly. After the pass's last
column, feedback shifts that same tensor right and restores the plain
prefix. A later looped column reuses the jittered payload but fuses it with
the blank embedding instead. The auxiliary objective therefore trains the
trunk, payload router and gain, fusion, embedding, and auxiliary block,
including on a single-pass batch or the final pass.

The [training objective](design.md) defines how the main and auxiliary
cross-entropies, squared-log-partition z-losses, and expert balancing combine
across the supervised passes and columns.

## Initialization and optimization

Initialization follows parameter ownership rather than applying one normal
scale to every matrix. Write `ρ = 1536/D`:

| Parameters | Initialization |
|---|---|
| NorMuonH matrices: Q/K/V and mixer output projections, expert gate/up and down maps, shared fusion | `Normal(0, 1/sqrt(fan_in))`; fusion has `fan_in=2D` |
| NAdam matrices with residual-width fan-in: GQA gates, expert routers, PKDA packed controls | `Normal(0, 0.02 sqrt(ρ))` |
| Tied embedding/readout, PKDA decay and output-gate expansions | `Normal(0, 0.02)` |
| RMSNorm gains, including payload and PKDA output norms | One |
| MHDB queries/nulls, blank payload, blank embedding, linear biases, main PKDA decay bias | Zero |
| PKDA depthwise convolutions | Kaiming-uniform |

PKDA's main and preconditioner decay-rate parameters initialize independently
with `exp(a), exp(aP) ~ Uniform(1,16)`. Its preconditioner bias satisfies
`softplus(bP) ~ LogUniform(0.001,0.1)`, and the learned log-space center starts
at `−0.2`.

A payload-bearing fusion combines two approximately unit-RMS inputs with
fan-in `2D`, giving an order-one seed. The zero-blank plain seed has only the
token contribution, and the zero-blank loop seed only the payload's; each has
expected RMS near `1/sqrt(2)`. These are initialization
scales, not restrictions on learned magnitudes. Shared parameters initialize
byte-identically across `f`, `l`, and `fl` for a given seed. Attention gates,
fusion, and the auxiliary module use separate deterministic streams;
training recurrence and jitter draws are keyed separately.

### NorMuonH

Ordinary hidden matrices use NorMuonH: Nesterov momentum, approximate
orthogonalization, row-wise second-moment adaptation, and a spectral tangent
step on a fixed Frobenius sphere. Each matrix retains its initial FP32 radius
`R = ‖W_initial‖_F`. With gradient `G`, momentum coefficient `μ`, and row
second-moment coefficient `β`:

```text
M       ← μM + (1−μ)G
N        = (1−μ)G + μM
U        = NewtonSchulz_5(N)
V       ← βV + (1−β) mean_columns(U²)
U        = Normalize_F(U / (sqrt(V) + ε))
T        = Normalize_F(U − ⟨W,U⟩_F W / ‖W‖_F²)
σ̂, v'   = spectral_estimate(T, saved_right_vector)
W_next   = R Normalize_F(W − η sqrt(fan_out/fan_in) T / σ̂)
```

Newton-Schulz works on the smaller Gram matrix, transposing tall matrices
and restoring their orientation afterward. The spectral estimate chooses
between the saved right vector and a restart from the largest-energy row,
then runs three paired power iterations. This deterministic restart can
recover a direction orthogonal to the previous estimate without consuming
randomness. Momentum, row moments, radius, and right vector are checkpointed.

The tangent projection removes radial movement; retraction restores the
fixed radius. Zero-rate, zero, and numerically radial updates preserve the
weights exactly. The matrix aspect ratio converts spectral scale to an
RMS-to-RMS trial-step scale. Power iteration can underestimate the spectral
norm, and retraction changes finite displacement, so the group rate is an
estimated trial-step budget rather than a certified final-step bound.

### NAdam and parameter ownership

NAdam owns parameters whose magnitudes remain learnable: the tied table,
GQA gates, expert routers, PKDA controls/expansions, and all non-matrix
parameters. It applies elementwise second-moment normalization to a
bias-corrected, time-varying Nesterov momentum update. Expert-selection
biases are buffers controlled by dispatch counts, outside both optimizers.
Neither optimizer uses weight decay.

There are five disjoint rate groups: ordinary NorMuonH matrices, expert
input maps, expert output maps, base NAdam parameters, and residual-width
NAdam matrices. The two expert groups include shared, routed, and MTP
experts. Their width/count factors are specified in [scaling](scaling.md),
and optimizer hyperparameters and the common learning-rate schedule in the
[recipe](design.md). [Operations](operations.md) describes FP32 masters and
gradient accumulation, CUDA working precision, and distributed ownership.

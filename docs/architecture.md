# Flagship architecture

This document is the authoritative specification of the flagship model and
optimizer. The flagship is a hard Delta Feedback decoder built from gated
global grouped-query attention (GGQA), Preconditioned Kimi Delta Attention
(PKDA), Multi-Head Delta Block routing (MHDB), and Full-Bandwidth Transformer
(FBT) recurrence. MHDB is this project's block-source specialization of
Multi-Head Delta Attention Residual routing (MHDAR); `mhdb` remains the code
and experiment name.

The architecture is fully specified, but the exact distributed flagship is
not yet runnable. Generic mixer, routing, feedback, cache, and optimizer
semantics exist in `DFModel`; exact-geometry distributed execution and its
qualification belong to the implementation gate in [design.md](design.md).
Primary source roles are indexed in
[../references/refs.yaml](../references/refs.yaml).

## Geometry

| Field | Flagship value |
|---|---:|
| Vocabulary | 151,936, Qwen3 tokenizer |
| Residual width | 1,536 |
| Decoder layers | 24 |
| Four-layer cells | 6 |
| SwiGLU intermediate width | 6,656 |
| Context | 8,192 |
| Global query heads | 16 |
| Global KV heads / routing groups | 8 |
| Global head width | 96 |
| PKDA query/key heads | 20 |
| PKDA value heads | 20 |
| PKDA key/value head width | 128 |
| PKDA Q/K/V projection width | 2,560 |
| PKDA convolution width | 4 |
| RMSNorm epsilon | `1e-6` at every norm site |
| Explicit position encoding | none |

The token-mixing schedule is exactly:

```text
[PKDA, PKDA, PKDA, gated global GQA] x 6
```

There is no sliding-window attention, MLA, RoPE, or additive position
embedding. PKDA's causal convolution and recurrent transition carry local
order and recency; every fourth layer supplies a dense causal global read.

At a high level, one recurrent column is:

```text
token embedding e_t -------------------+
                                       +-- FBT fuse --> seed s_t
previous payload p_(t-1) --------------+                  |
                                                          v
        +------------------------------------------------------+
        | [ PKDA - PKDA - PKDA - gated global GQA ] x 6       |
        |   each sublayer may transiently read MHDB sources    |
        +------------------------------------------------------+
                          |                         |
                       h_top                    block deltas
                          +------------+------------+
                                       |
                                       v
                              routed DF payload p_t
```

On the first pass or in Standard decoding, no previous payload exists and the
seed is the token embedding itself.

### Synthesis boundary

The flagship combines established components but is not a reproduction of any
one source model:

| Source family | Adopted contract |
|---|---|
| Full-Bandwidth Transformer | outer 1B geometry, latent-feedback entry and training objective, WSD/NorMuon lineage |
| Kimi Linear | 20-by-128 KDA substrate, causal convolution, NoPE, sigmoid output gate, and 3:1 local/global cadence |
| Preconditioned DeltaNet | apply-to-key diagonal preconditioner, independent gates, bounded squash, chunk/recurrent forms, and FP32 boundary states |
| Qwen3-Next | sigmoid-gated global GQA operation and interval-four precedent |
| Multi-head and Delta Attention Residuals | grouped source selection and additive delta-source semantics |
| This project | PKDA/GGQA composition, four-layer MHDB sources, hard-DF payload routing, tokenizer, exact optimizer partition, and scale schedule |

These precedents motivate each piece; only the registered experiments can
establish their interaction.

## Residual shell

The model is a bias-free pre-norm decoder with tied embedding/readout, a final
RMSNorm, packed SwiGLU channel mixers, and no dropout. Every RMSNorm, including
PKDA's gated output norm, uses epsilon `1e-6`. For layer `l`, the token
mixer and MLP produce already-scaled branch deltas `a_l` and `m_l`:

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

This identity is the source and causality contract for both within-column
routing and the recurrent payload.

## PKDA mixer

Each PKDA layer is a Kimi Delta Attention recurrence with the stable diagonal
apply-to-key preconditioner from Preconditioned DeltaNet. The model width does
not constrain its recurrent projection width: the flagship deliberately uses
20 heads by 128 coordinates, or 2,560 projected coordinates, at residual width
1,536.

### Projection and controls

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
and downstream parameters inside the packed implementation. They are not
tied. The preconditioner uses:

- squash bound `x = 1.5` and epsilon `1e-6`;
- learned per-head log-space center initialized to `-0.2`;
- decay rates sampled on `[1, 16]`;
- softplus time constants sampled log-uniformly on `[0.001, 0.1]`;
- negative-eigenvalue and safe-gate modes disabled.

### Recurrent equation

For one head, let `S_t` be the key-by-value recurrent matrix and `A_t` the
nonnegative diagonal preconditioner state. Both begin at zero. In the
project's key-first convention:

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

### Training and decoding state

CUDA training and long-prompt prefill use the workspace FLA fork's chunk
operator at chunk size 64. CPU and MPS execute the literal recurrence. CUDA
never silently substitutes the sequential fallback.

Autoregressive decoding continues, per PKDA layer:

- the FP32 recurrent matrix `S_t`;
- the FP32 diagonal state `A_t`;
- three BF16 convolution histories of length 3.

A Jacobi or fused-prefill pass starts those states from zero and advances them
once across that pass's causal token order. PKDA state is not carried between
repeated passes over the same positions; the DF payload is the sole
cross-pass state. After the final prefill, the materialized mixer caches
advance normally during decoding.

## Gated global GQA

The fourth layer of every cell is dense causal NoPE GQA. For pre-normalized
input `x`, a packed bias-free projection produces Q/K/V, Q and K receive
per-head RMSNorm, and a separate bias-free projection produces one gate
coordinate per query-head output coordinate:

```text
q, k, v = split(W_qkv x)
z       = concat(GQA(q, k, v))
o       = W_o(sigmoid(W_g x) * z)
```

The gate acts on the concatenated attention result before the output
projection and residual-branch scaling. It does not change attention logits
or softmax weights. It is distinct from every PKDA control and from the FBT
entry gate.

Full-sequence, prefill, and cached CUDA execution use FlashAttention. Cached
decoding stores BF16 K/V only for the six global layers.

## Multi-Head Delta Block routing

MHDB widens residual access along depth while preserving a clean residual
stream. It uses the multi-head source-selection operation from multi-head
Delta Attention Residuals, but its addressable deltas are four-layer blocks,
not individual attention and MLP branches.

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
source distribution uniform. The null initially contributes no value, but it
is learnable, so null mass must later be interpreted together with null-vector
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

The current partial is omitted when it is exactly absent at cell entry. The
MLP site sees the partial after its layer's attention delta has been added. At
the cell boundary, the evolving partial becomes the one completed block delta.
Individual branch deltas never become addressable sources.

The non-null sources therefore reconstruct the current residual exactly:

```text
h_current = seed + sum(completed block deltas) + current partial delta
```

The routed value is added only to the next sublayer's pre-norm read. On the
initial zero-query pass, the non-null mixture is collinear with `h_current`, so
the following RMSNorm makes routed pass-1 reads functionally inert up to its
epsilon.

The deepest within-column site and the payload router each see at most eight
sources. The former has null, seed, five completed cells, and one live partial;
the latter has null, seed, and six completed cells.

## FBT recurrence and hard Delta Feedback

FBT adds recurrence across token columns. Let `e_t` be the sampled token
embedding and `p_(t-1)` the previous column's payload. A feedback position
constructs the next seed with the asymmetric FBT GLU:

```text
u_t = entry_norm(
    W_U p_(t-1) * sigmoid(W_G gate_norm(e_t))
)
```

The payload is the value path and the current token controls the gate. There is
no additive embedding bypass on a feedback position. Prompt positions, pass-1
positions, and Standard decoding use `e_t` because no payload is supplied.

Hard DF uses `u_t` as both the residual seed and the first non-null source for
every within-column MHDB router. After the sixth cell, a dedicated eight-group
payload router reads its own null, the seed, and the six completed block
deltas. The recurrent payload is:

```text
r_payload = route(null, seed, Delta_0 .. Delta_5)
p_t       = payload_norm(h_top + r_payload)
```

The additive `h_top` guarantees that the complete column state survives even
if the learned router specializes. At initialization, the uniform non-null
mixture equals `h_top / 8`; after the final RMSNorm, the payload therefore
matches a bare normalized top state up to epsilon. The payload query is not
conditioned on the next token; token-dependent control happens only after the
payload shifts into the next column's FBT entry gate.

Sequential generation retains only the immediately previous payload outside
the mixer caches. Each new column consumes it once, advances all PKDA and GQA
caches, and emits the next payload.

An equivalent high-level column is:

```python
e = embed(tokens)
seed = e if previous_payload is None else fbt_fuse(previous_payload, e)
h = seed
completed = []

for cell in six_four_layer_cells:
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

## Precision and initialization

Each NorMuonH-owned matrix `W` with shape `[d_out, d_in]` is initialized from
`Normal(0, 1 / sqrt(d_in))`, following the Hyperball parameterization. The tied
embedding/readout and NAdam-owned dense matrices use `Normal(0, 0.02)`.
Depthwise convolution weights keep PyTorch Conv1d's Kaiming-uniform
initialization at fan-in 4. RMSNorm scales initialize to one; routing queries
and nulls initialize to zero. PKDA rates, time constants, centers, and
output-gate bias use the special initializations stated above.

Embedding parameters and optimizer state are FP32. On CUDA, the residual
stream, routed values, payloads, mixer activations, and caches are BF16 except
for the PKDA matrix and diagonal boundary states. Cut cross-entropy reads an
address-stable BF16 classifier shadow that is refreshed from the tied embedding
once after every optimizer update and is neither a parameter nor checkpoint
state. Its authoritative tied parameter and accumulated gradient remain FP32.

## NorMuonH and NAdam

The flagship uses one optimizer recipe with three disjoint parameter groups:
one NorMuonH group and two NAdam groups. Their public controls are
`--lr-normuonh`, `--lr-embedding`, and `--lr-nadam`. No group uses weight
decay.

### NorMuonH matrices

NorMuonH owns ordinary trainable two-dimensional hidden weights, including:

- attention and PKDA Q/K/V/output projections;
- SwiGLU input and output matrices;
- the FBT payload-value projection.

For matrix `W`, its initial FP32 Frobenius radius `R = ||W_0||_F` is fixed for
the life of the run. With the fan-in initialization above,
`E[R^2] = d_out`; the realized sampled radius, rather than that expectation, is
the exact constraint. NorMuonH forms an update direction by:

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

With dimensionless learning rate `lr_normuonh = 2e-2` and normalized direction
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

The tied embedding/readout is its own parameter group with learning rate
`6e-4`; every other NAdam-owned parameter uses `3e-4`. Both groups use moment
betas `(0.9, 0.95)`, momentum decay `psi = 0.004`, epsilon `1e-8`, and no weight
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

The update is `theta_t = theta_{t-1} - lr * U_t`. All three parameter groups
receive the same warmup-stable-cooldown multiplier defined by the current
scale's schedule.
CUDA uses PyTorch's foreach NAdam path outside the captured forward/backward
graphs; its scalar step and momentum-product state stay on CPU, while both
moment tensors and all parameters remain FP32 on-device.

After synchronized microbatch accumulation, the single global FP32 gradient
vector is clipped to L2 norm 3.0 immediately before both optimizer steps. The
reported gradient norm is the pre-clip norm, and a non-finite norm terminates
the run.

## Parameter and cache accounting

| Component | Parameters |
|---|---:|
| Tied embedding and readout | 233,373,696 |
| 24 SwiGLU channel mixers | 736,100,352 |
| 18 PKDA mixers | 304,297,632 |
| 6 gated global GQA mixers, including Q/K norms | 56,624,256 |
| Hard-DF fusion | 4,718,592 |
| Trunk, entry, and payload norms | 79,872 |
| 48 within-column routers and one payload router | 225,792 |
| **Total** | **1,335,420,192** |
| **Active non-embedding** | **1,102,046,496** |

The active non-embedding count is the denominator for the flagship data
budget. Relative to unpreconditioned KDA, PKDA's two width-to-head projections
and three learned per-head vectors add 61,500 parameters per PKDA layer, or
1,107,000 total. The six GGQA gates add 14,155,776 weights.

At a full 8,192-token prompt, one sequence's registered token-mixer cache is:

| Cache | Size |
|---|---:|
| 18 FP32 PKDA matrix states `[20,128,128]` | 22.500 MiB |
| 18 FP32 PKDA diagonal states `[20,128]` | 0.176 MiB |
| 18 BF16 Q/K/V convolution histories `[2560,3]` | 0.791 MiB |
| 6 BF16 global-GQA KV caches `[8192,8,96]` | 144.000 MiB |
| **Token-mixer total** | **167.467 MiB** |

This excludes allocator overhead, the width-1,536 payload, logits, and serving
metadata. The instantiated implementation must reproduce the parameter count,
state shapes, and cache continuation semantics before the flagship can pass
its implementation gate.

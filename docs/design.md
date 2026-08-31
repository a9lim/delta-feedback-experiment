# Design

This document is the authoritative contract for the current experiment. It
defines the model family, training recipe, comparisons, diagnostics, scale
plan, and promotion gates. Accepted scientific evidence belongs in
[findings.md](findings.md); active scratch work belongs in
[journal.md](journal.md); primary source roles are indexed in
[../references/refs.yaml](../references/refs.yaml).

## Objective

The experiment tests two innovation packages in an autoregressive transformer:

1. **Architecture innovation.** Multi-Head Delta Attention Residuals (MHDAR)
   let each sublayer read an additive mixture of the current column's input
   seed and earlier residual deltas, with independent depth selection by
   contiguous feature group. Every architecture-package layer also uses
   sigmoid-gated grouped-query attention (GGQA).
2. **Recurrence innovation.** Full-Bandwidth Transformer (FBT) feedback returns
   the previous column's top state to layer 0 of the next column through a
   token-gated fusion, giving latent state another full pass through depth.

The primary question is whether these packages are complementary, independent,
or redundant when pretrained together. The primary four-cell factorial is:

| Arm | Architecture package | Recurrence package | GQA | Parameters |
|---|---:|---:|---|---:|
| `vanilla` | no | no | ungated | 222,876,672 |
| `mhdar` | MHDAR | no | gated | 229,991,424 |
| `fbt` | no | FBT | ungated | 224,058,624 |
| `df` | MHDAR | FBT | gated | 231,174,912 |

`df_soft` is a 230,012,928-parameter screen diagnostic. It retains the
architecture package's GGQA, keeps a plain token entry, and puts a
zero-initialized learnable null source in every router so that within-column
routing, previous-payload access, and payload enrichment can each be declined
by routing mass. It tests adoption; it does not replace the hard hybrid in the
factorial.

The claim boundary is pretraining behavior under the registered recipe and
data. No result is a general claim about recurrent transformers, depth routing,
reasoning, or adaptive computation without a dedicated evaluation that supports
that claim.

## Common transformer trunk

Every arm uses one `DFModel` implementation and differs only through
`ModelConfig` flags. The screen trunk is:

| Field | Value |
|---|---:|
| Vocabulary | 151,936, Qwen3 tokenizer |
| Width | 768 |
| Layers | 12 |
| Query heads | 8 |
| KV heads | 4 |
| Attention head width | 96 |
| SwiGLU intermediate width | 3,072 |
| Context / predicted tokens per row | 1,024 |
| RoPE theta | 1,000,000 |
| RMSNorm epsilon | 1e-6 |
| Routing heads | 4, exactly the KV-head count |

The base trunk is a bias-free pre-norm decoder with packed QKV projection,
grouped-query attention, per-head Q/K RMSNorm, rotary positions, packed SwiGLU
gate/up projection, tied embedding/unembedding, and a final RMSNorm. Each
attention and MLP branch output is scaled by `1/sqrt(2L)` before it is added to
the residual stream. There is no dropout.

The architecture package adds one bias-free gate projection to every attention
layer. For pre-normalized attention input `x`, it computes:

```text
q, k, v = split(W_qkv x)
z       = concat(GQA(q, k, v))
o       = W_o(sigmoid(W_g x) * z)
```

The gate has one coordinate per query-head output coordinate and acts before
the output projection and branch scaling; it does not modify attention logits
or softmax weights. `mhdar`, `df`, and `df_soft` use this GGQA path.
`vanilla` and `fbt` omit `W_g` and pass `z` directly to `W_o`. The gate is part
of the registered architecture axis, not an independently interpreted factor.

Embedding weights and optimizer state remain FP32. Under CUDA autocast, the
embedding output, residual stream, recurrent payload, and routed values are
BF16. CPU and MPS use the same semantics through portable PyTorch operations.

## Architecture package: MHDAR and gated GQA

The screen and its same-geometry token ladder use the per-sublayer source
layout in this section. The flagship keeps the same routing primitive and
residual identity but coarsens its source bank into four-layer block deltas as
specified in the scale plan.

### Router

For source vectors `v_i in R^D`, one routing site owns:

- a learned query `q in R^D`, initialized to zero;
- a learned RMS key scale `g in R^D`, initialized to one;
- `H = kv_heads` contiguous feature groups of width `D/H`.

The source key is normalized across the full width before the grouped score is
formed:

```text
k_i = g * v_i / sqrt(mean(v_i^2) + eps)
s_i,h = dot(q_h, k_i,h)
a_i,h = softmax_i(s_i,h)
route(v_1..v_N)_h = sum_i a_i,h * v_i,h
```

Keys are normalized; values are raw. There is no output projection and no
`1/sqrt(head_dim)` score factor. Full-width RMS statistics couple the groups,
while the source softmax and value mixture are independent per group. Splitting
one width-`D` query into groups adds no query parameters relative to a
single-head delta router.

Zero query initialization makes every active head uniform over its available
sources. A hard router with fewer than two sources is a no-op. In `df_soft`, a
learnable width-`D` null vector is prepended at every site; it is initialized to
zero, so even the first attention site has two choices.

### Sources and residual identity

For a column input seed `s`, layer `l` produces scaled attention and MLP deltas
`a_l` and `m_l`. The residual stream is always:

```text
h_top = s + sum_l (a_l + m_l)
```

At the attention read in layer `l`, the available sources are the seed and all
completed earlier deltas. At the MLP read, `a_l` is also available. A routed
mixture is added only to that sublayer's pre-norm read:

```text
x_attn = h + route(s, a_0, m_0, ..., m_(l-1))
a_l = scale * attention(rmsnorm(x_attn))
h = h + a_l

x_mlp = h + route(s, a_0, m_0, ..., m_(l-1), a_l)
m_l = scale * mlp(rmsnorm(x_mlp))
h = h + m_l
```

The routed value is not accumulated directly into `h`. This transient-read
rule preserves the exact telescoping identity and makes every stored source a
real component of the column state rather than another cumulative state.

For `mhdar`, the seed is the token embedding `e`. For hard `df`, the seed is
`e` on pass 1 and the fused input `u` on feedback passes. For `df_soft`, the
stream seed is always `e`; the previous payload is an additional standing
source only where the prefix mask marks it present.

## FBT latent feedback

Let `e_t` be the sampled token embedding for column `t`, and let `p_(t-1)` be
the payload produced by the previous column. Hard feedback arms construct the
next column input as:

```text
u_t = entry_norm(
    W_U p_(t-1) * sigmoid(W_G gate_norm(e_t))
)
```

The payload occupies the value path and the token embedding controls the gate.
There is no additive token-embedding bypass on a feedback position. Prompt
positions and all pass-1 positions use `e_t` directly because no recurrent
payload is present.

The cache stores attention keys and values for the actual column inputs. During
sequential feedback decoding, only the immediately previous payload is carried
outside the cache; each generated column consumes it once and emits the next
payload.

The `fbt` payload is:

```text
p_t = payload_norm(h_top,t)
```

## Hard Delta Feedback

Hard `df` combines the two packages without adding an alternate route around
either one:

1. On a feedback position, fuse the shifted previous payload with the token
   embedding to obtain `u`.
2. Use `u` as both the residual seed and the first source for every
   within-column MHDAR site, and use GGQA in every attention layer.
3. Preserve the clean residual identity while attention and MLP sites read
   routed mixtures of the seed and completed deltas.
4. Route over this column's deltas with a dedicated multi-head payload router.
5. Add the routed enrichment to the full top state and normalize the result.

For the screen and token ladder, with
`Delta = [a_0, m_0, ..., a_(L-1), m_(L-1)]`:

```text
r_payload = route(Delta; q_payload)
p = payload_norm(h_top + r_payload)
```

The payload router excludes the seed. The base `h_top` guarantees that the full
column state is retained; the routed term selects which changes made in this
column receive an additional cross-column path. The router has no null source,
so delta enrichment carries unit softmax mass in every head. At initialization
it is the mean of the scaled deltas.

The payload query is not conditioned on the next token. Token-dependent control
occurs in the FBT entry gate after the payload shifts to the next position.

An equivalent high-level pass is:

```python
e = embed(tokens)
s = e if payload is None else fuse(payload, e)
sources = [s]
h = s
for block in blocks:
    a = scaled_gated_attention(rmsnorm(h + route(sources)))
    h = h + a
    sources.append(a)
    m = scaled_mlp(rmsnorm(h + route(sources)))
    h = h + m
    sources.append(m)
payload = payload_norm(h + route(sources[1:], q_payload))
logits = tied_head(final_norm(h))
```

The first attention router sees only the seed and therefore acts as a no-op in
hard MHDAR/DF. All later sites have at least two sources.

## Soft adoption diagnostic

`df_soft` uses the same GGQA architecture package, multi-pass loop, and routing
primitive but removes the mandatory FBT entry:

```text
stream seed: e
standing sources on a fused suffix: [p_previous, e]
router sources at every site: [learnable_null, standing_sources, deltas_so_far]
payload: payload_norm(h_top + route([learnable_null, deltas]))
```

The previous-payload source is masked out on the plain prefix. The null and
token seed remain available everywhere. On pass 1 there is no previous-payload
source, but null-sourced within-column routing is already active. This creates a
within-run adoption contrast: depth routing trains from step 1, whereas the
feedback source appears only in the feedback phase.

The learned null is a trainable source, not a hard zero clamp. Its mass, the
previous-payload mass, the token-seed mass, and the payload-router null mass are
observables. A high null mass shows non-adoption at that site; it does not by
itself establish that the corresponding mechanism is useless under a mandatory
entry or a different formation schedule.

## Multi-pass training

### Jacobi passes

Feedback arms use parallel Jacobi passes over a full token sequence. Pass 1 is
ordinary teacher forcing with plain embeddings. For each later pass:

1. take the preceding pass's payload tensor;
2. add keyed uniform jitter;
3. shift it one position right, inserting zero at position 0;
4. sample a per-row plain-prefix length;
5. use plain embeddings on the prefix and feedback inputs on the suffix;
6. run the complete stack again without detaching the payload graph.

`k` passes train a feedback horizon of `k-1` token transitions and cost `k`
transformer evaluations. The shift and prefix mask preserve token causality.
Gradients from later-pass losses reach the earlier states, payload router, and
entry gate.

For hard feedback arms, suffix inputs are FBT-fused. For `df_soft`, suffix
inputs remain plain embeddings and the shifted payload becomes a masked routing
source. Non-feedback arms always use one pass.

### Prefix mixin and jitter

For every feedback pass and row, the plain-prefix length is drawn uniformly
from `1..seq_len`. Position 0 is therefore always plain, and every row retains
at least one feedback position. Payload jitter is drawn uniformly from
`[-0.02, 0.02]` by default and is added before shifting.

Pass count is keyed by `data_seed` and step. Prefix lengths and jitter are keyed
by `data_seed`, step, and the microbatch's first global row. They do not depend
on ambient RNG state and are shared across paired feedback arms.

### Objective

Let `ell_k` be mean next-token cross-entropy on pass `k`. The loss is:

```text
K = 1:  L = ell_1
K > 1:  L = ell_1 + mean(ell_2, ..., ell_K)
```

The pass-1 term always has unit weight; all feedback passes together have unit
weight. No pass is detached. During cooldown, the same combination is applied
to the squared log-partition penalty
`mean(logsumexp(logits)^2)` with coefficient `1e-5`.

Pass-1 loss is logged separately because it is the shared Standard-mode metric
across all arms.

## Data and pairing

The corpus is FineWeb-Edu in canonical streaming order, tokenized with
`Qwen/Qwen3-0.6B`. If no dataset revision is supplied, tokenization resolves and
records the current dataset commit before writing data. Each non-empty document
is followed by EOS.

The held-out validation slice is taken from the head of the stream. Training
tokens follow it in contiguous uint32 shards. A training row is a non-overlapping
`seq_len + 1` window: 1,025 stored tokens yield 1,024 predictions. Rows may
cross document boundaries, and each step directly addresses its global row
range:

```text
first_row(step) = (step - 1) * batch_rows
```

Paired arms and seeds use the same data directory, `data_seed`, batch geometry,
and step addresses. One initialization seed produces byte-identical common
parameters in every arm. Factor-private streams produce identical attention
gate matrices in `mhdar`, `df`, and `df_soft`, and identical FBT fusion matrices
in `fbt` and `df`; conditional modules never advance the common stream.
Initialization seeds differ only when registering a new paired seed. Resumption
returns to the same row and the same keyed feedback draws.

## Optimizer and schedule

Every arm uses the same recipe.

### Parameter groups

- **NorMuon:** every trainable two-dimensional weight except the tied
  embedding/unembedding and sigmoid attention-gate projections. This includes
  the FBT value and token-gate fusion matrices. Defaults: learning rate `1e-2`,
  momentum `0.95`, row second-moment beta `0.95`, five Newton-Schulz steps,
  epsilon `1e-8`, decoupled weight decay `0.01`.
- **Adam:** sigmoid attention-gate projections, tied embeddings, RMSNorm
  weights, routing queries, null vectors, and all other non-matrix parameters.
  Defaults: learning rate `5e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, no
  weight decay.

NorMuon orthogonalizes the momentum, normalizes rows by their second moments,
and globally rescales the update to Frobenius norm `0.2 * sqrt(m*n)` before
applying learning rate and decoupled decay.

### Default screen schedule

The default 6,700-step WSD schedule is:

| Phase | Steps | Pass behavior |
|---|---:|---|
| Warmup | 1–200 | one pass |
| Stable heat | 201–5,025 | one pass |
| Cooldown | 5,026–6,700 | feedback arms draw 2 or 3 passes |

Within cooldown, `P(k=3) = 0.12`; otherwise `k=2`. Across the full run this
targets the 75%/22%/3% one-/two-/three-pass mixture. The exact draws are
deterministic for the registered data seed. The expected compute multiplier for
a feedback arm is 1.28 transformer passes per predicted token, while
non-feedback arms remain at 1.0.

Learning rates follow the shared WSD multiplier. NorMuon weight decay stays at
its stable value through warmup and heat, then is multiplied by the normalized
learning-rate factor during cooldown. The z-loss is active only during
cooldown.

`--max-steps` caps the number of additional steps in one process. It does not
change the schedule, feedback boundary, protected checkpoints, or any
state-defining field.

## Execution and checkpoints

### Portable path

CPU and MPS use PyTorch scaled-dot-product attention, the algebraic MHDAR
router, chunked tied-head cross-entropy, eager execution, and the same model,
loss, optimizer, data, and checkpoint semantics. This path owns fast invariant
tests and analysis.

### CUDA path

The authoritative Jobe screen path uses:

- BF16 trunk activations with FP32 weights and optimizer state;
- FlashAttention for full-sequence, prefill, GQA, and cached decoding;
- a fixed-capacity Triton MHDAR router over source pointers, with full-width RMS
  scores, per-head masked softmaxes, FP32 value accumulation, and an analytic
  backward that retains cross-head RMS coupling;
- cut cross-entropy for the tied head, including the exact squared
  log-partition gradient without materializing vocabulary-wide logits;
- full-block compilation around attention kernels;
- fixed-address forward/backward CUDA graphs for every schedule-reachable train
  mode and shared-pool no-grad validation graphs;
- BF16 keyed jitter drawn directly into graph input buffers;
- internal activation checkpointing for `df_soft` feedback modes and geometries
  above the measured screen work threshold;
- asynchronous snapshot staging to pinned host memory followed by atomic
  background serialization.

The public experiment surface does not expose kernel, graph, or activation
checkpoint policy switches. Those are execution choices, not factorial axes.

### Checkpoint contract

Snapshots use checkpoint contract v6 and resume only v6. They contain model,
both optimizer states, exact state-defining arguments, step, and Python/Torch/
CUDA RNG state. A resume inherits all state-defining fields and rejects an
explicit conflict. Runtime paths, device, evaluation cadence, snapshot cadence,
and evaluation-row count may change per invocation.

The run retains the latest two snapshots plus the protected end-of-heat and
end-of-run snapshots. `df queue` records exact arguments and runs the probe
before every training job. Git state is not part of the queue schema: source
changes do not stop an active child, and the worker refreshes before the next
queued job so it uses the current checkout.

## Evaluation

### Language-model metrics

At each evaluation point:

- every arm reports `val`, the pass-1 held-out cross-entropy;
- every feedback arm also reports `val_fused`, a second pass whose plain prefix
  has length 1.

The inference modes are:

- **Standard:** one prompt prefill and no latent feedback during generation;
- **Soft:** one plain prompt prefill, then one latent-feedback transition per
  generated token;
- **Fused:** an additional fused prompt pass, then the same latent-feedback
  decode loop.

Here “Soft” names the FBT decoding mode and is unrelated to the `df_soft` arm.
No-feedback arms use their Standard metric when a factorial table is shown for
Soft or Fused mode.

### Effect and interaction

For validation loss `L` (lower is better), define the gain of arm `A` over
vanilla as:

```text
G_A = L_vanilla - L_A
```

`G_mhdar` is the architecture-package effect, `G_fbt` is the
recurrence-package effect, and `G_df` is their joint effect. The factorial
interaction is:

```text
I = G_df - G_mhdar - G_fbt
  = L_mhdar + L_fbt - L_df - L_vanilla
```

`I > 0` is superadditive loss reduction, `I = 0` is additive, and `I < 0` is
subadditive. Compute the statistic on paired checkpoints and per decode mode,
then aggregate paired seed differences. Do not mix Standard, Soft, and Fused
losses inside one interaction estimate.

Equal step counts are matched-data comparisons, not matched-compute
comparisons. Report predicted tokens and pass-tokens for every point. A
matched-compute view must compare curves or checkpoints at equal cumulative
pass-tokens; feedback batches count once per pass.

### Contraction

For every feedback-bearing checkpoint, repeatedly apply fully fused prefill
passes with prefix length 1. At iteration `k`, record held-out loss and:

```text
mean_token ||h_top^(k) - h_top^(k-1)||_2
```

The standing training monitor runs eight iterations. Promotion uses at least 30
self-compositions. Stable loss and a decaying update norm support safe
composition; oscillating updates or rising loss make feedback comparisons
untrustworthy even if a one-step metric improves.

### Routing observables and causal probes

For each routing site and head, record source weights separately rather than
only their mean. Current summaries include mean source mass, per-token maximum,
normalized entropy, normalized cross-head Jensen-Shannon divergence, source RMS
scale, query norm, and query cosine similarity.

The important source labels are:

- within-column seed mass (`e` for `mhdar`, `u` for hard `df` feedback passes);
- previous-payload, token-seed, and null mass in `df_soft`;
- per-delta mass at attention, MLP, and payload sites.

`scripts/route_report.py` produces held-out route maps and query geometry from
a hard-DF checkpoint. `scripts/payload_swap.py` replaces the learned payload
mixture with top-only, uniform, or single-delta alternatives on the same
weights. The sweep is a co-adapted same-checkpoint intervention: its landscape
identifies sensitive payload content but does not estimate the effect of
training an alternative payload rule from scratch.

## Screen and scale plan

### 223–231M screen

The active screen uses the geometry above, 6,700 steps, 2.003B predicted tokens
per run, FineWeb-Edu, two paired initialization seeds, and Jobe's RTX 4090. The
primary four factorial arms run before the conditional `df_soft` diagnostic.

The screen is designed to establish:

1. that the shared recipe learns a healthy vanilla baseline;
2. that the harness can resolve the MHDAR-plus-GGQA architecture package under
   this recipe;
3. the signs and paired magnitudes of the architecture, recurrence, and joint
   effects;
4. whether hard DF is stable under recurrent self-composition;
5. whether optional within-column and cross-column routes are adopted in
   `df_soft`, if that diagnostic is reached.

The screen runs at 8.7–9.0 predicted tokens per parameter across its arms. It
is a sensitivity and interaction screen, not a decisive test of FBT formation
at the high token-per-parameter regime.

### Token ladder

The registered ladder keeps the screen geometry and extends selected arms through
2B, 8B, and 32B predicted tokens, taking a WSD cooldown branch at each rung and
continuing the next rung from that rung's protected pre-cooldown checkpoint.
Finalist arms share the stream prefix and use one paired seed, with the screen's
two-seed spread retained as the noise estimate.

This ladder is not currently runnable through exact resume: `steps` is a v6
state-defining field, and no tested branch-from-heat-end continuation command
exists. Before ladder launch, code and tests must define a new run address,
preserve model/optimizer/RNG and row continuity, extend the stable phase without
re-warming, and create a new cooldown branch without weakening exact resume.

### Flagship

The registered flagship is hard DF at approximately 1.22B parameters and 400B
predicted tokens. It has width 1,536, 24 decoder layers, SwiGLU width 6,656,
context 8,192, and six identical four-layer cells. Its token-mixing schedule is
exactly:

```text
[KDA, KDA, KDA, gated global GQA] x 6
```

There is no sliding-window attention. Each KDA layer uses 12 heads with
`d_k = d_v = 128`, so its concatenated head width is 1,536. It follows the
released Kimi Linear parameterization: bias-free Q/K/V projections; separate
causal depthwise convolutions of width 4 followed by SiLU; L2-normalized Q and
K; a rank-128 channel-wise decay projection; one sigmoid delta-update gate per
head; the KDA recurrent update; head-wise RMSNorm; a rank-128 sigmoid output
gate; and a bias-free output projection. Training uses the chunkwise-parallel
form and cached decoding uses the mathematically equivalent recurrent form.
For each KDA head, with zero initial state, the semantic recurrence is:

```text
log_alpha_t = -exp(A) * softplus(W_f_up W_f_down x_t + b_f)
alpha_t     = exp(log_alpha_t)
beta_t      = sigmoid(W_beta x_t)
S_tilde_t   = Diag(alpha_t) S_(t-1)
S_t         = (I - beta_t k_t k_t^T) S_tilde_t + beta_t k_t v_t^T
o_t         = S_t^T q_t
```

The fourth layer of each cell uses dense causal, sigmoid-output-gated GQA: 16
query heads, 8 KV heads, head width 96, per-head Q/K RMSNorm, and a full 8,192-
token receptive field. For the pre-normalized layer input `x_t`, a bias-free
projection produces one query and one same-width gate coordinate per query-head
coordinate:

```text
q_t, g_t = split(W_qg x_t)
z_t      = concat(GQA(q_t, W_k x_t, W_v x_t))
o_t      = W_o(sigmoid(g_t) * z_t)
```

The gate acts elementwise on the concatenated attention output before the
output projection and before the branch's `1/sqrt(2L)` scaling. It does not
modify attention logits or softmax weights, and it is distinct from KDA's
delta-update and output gates and from the FBT entry gate. Because the gated
`o_t` is still the single attention branch delta, the seed-plus-deltas residual
identity is unchanged.

The flagship has no explicit positional embedding in either mixer: KDA's
causal convolution and data-dependent recurrent transition carry order and
recency, and the global GQA layers use NoPE. This is a deliberate synthesis.
Kimi Linear supplies KDA, NoPE global attention, and the empirically selected
3:1 cadence, but uses global MLA; Qwen3-Next supplies independent interval-four
Gated DeltaNet/gated-global-GQA precedent, but does not use KDA.

The approximately 1.22B count assumes the geometry and dense KDA projections
above, tied embeddings, hard-DF modules, and six width-1,536 GQA gate
projections. Those gates add 14,155,776 weights to the ungated hybrid. The
implementation gate must record the exact instantiated count before spend
approval.

#### Flagship block-delta routing

The four-layer attention cell is also the routing block. A cell contains four
token mixers and four MLPs, but none of their eight individual branch deltas is
retained as an addressable routing source. Let `c_b` be the residual at entry to
cell `b`. While that cell is executing, define:

```text
partial_b = h_current - c_b
Delta_b   = h_cell_exit - c_b
```

At a routing site in cell `b`, the source bank is the column seed, one completed
`Delta_j` for every earlier cell, and one `partial_b` when it is nonzero. The
first mixer in a cell therefore sees only the seed and completed earlier cells;
later sites see those sources plus one evolving aggregate for the current cell.
At the boundary, `partial_b` becomes the single completed `Delta_b`. This keeps
the exact decomposition:

```text
h_current = seed + sum(completed cell deltas) + current partial delta
```

Routing remains a transient pre-norm read and never accumulates directly into
the residual stream. The hard-DF payload router excludes the seed and routes
over the six completed cell deltas `[Delta_0, ..., Delta_5]`. The deepest
within-column router therefore has at most seven sources: the seed, five
completed cells, and one current partial cell. This is the block form described
by the Delta Attention Residuals and Attention Residuals papers, rather than an
ad hoc bank of individual attention and MLP outputs.

KDA state is local to one transformer evaluation. A Jacobi or fused-prefill
pass starts every KDA recurrent and convolution state from zero and advances it
once across that pass's causal token order. Autoregressive decoding retains one
KDA state and convolution history per KDA layer alongside the GQA KV caches and
advances both once per generated token. Recurrent state is never carried from
one repeated pass over the same token positions into the next; the DF payload
is the only cross-pass state. After the final prefill pass, the KDA and GQA
caches advance normally over newly generated positions.

#### Flagship implementation gate

The exact flagship implementation must pass portable/chunkwise/recurrent KDA
value and gradient parity, convolution- and recurrent-cache continuation
parity, gated-GQA value and gradient parity against an explicit
sigmoid-times-attention reference, block-source and payload-source identity
tests, exact compute and parameter accounting, optimizer partition tests, the
width-1,536/H=8 router gate, distributed execution, checkpoint portability,
and restart behavior on the selected hardware. Until those contracts land in
code and tests, the flagship is specified but not runnable.

## Gates

### Engineering gate

`df probe` must pass at the queued commit. On Jobe this includes portable tests,
Triton router value/weight/gradient parity for H=4 and H=8, cut-cross-entropy
z-loss parity, cached FlashAttention recurrence, captured/eager evaluation
parity, every schedule-reachable CUDA graph, finite optimizer updates,
contraction-monitor execution, and production-scale checkpoint staging.

Engineering qualification establishes implementation sensitivity and numerical
coherence. It is not an experiment finding.

### Screen admissibility and ladder entry

A screen comparison is admissible only when the registered paired runs finish
with the same token stream, batch order, recipe, seeds, and schedule, and all
feedback arms have healthy contraction traces.

The MHDAR-plus-GGQA architecture package must clearly improve on vanilla in
Standard mode for the screen to serve as a sensitivity gate for few-percent
architecture effects. A stable FBT null at screen scale does not by itself
exclude DF from the ladder because the screen is far below the registered
token-per-parameter regime. DF enters the ladder only if its paired effect is
credible enough that additional tokens can resolve the interaction between the
two parent packages.

### Flagship promotion

Promote only if:

1. DF's advantage over both parent arms at matched token-equivalent compute
   holds or grows from 2B through 32B rather than appearing at one rung;
2. the top-rung feedback map remains stable for at least 30 fused
   self-compositions;
3. routing and same-checkpoint ablations do not reveal a trivial unused or
   bypassed mechanism;
4. the flagship implementation gate passes on its exact distributed geometry;
5. a9 explicitly approves the flagship spend with the ladder evidence in hand.

## Scope exclusions

The current experiment does not include adaptive pause or halting tokens,
token-conditioned payload queries, more than one explicit previous-column
payload, per-layer cross-column source banks, alternative optimizer arms,
long-context continuation, instruction tuning, or downstream reasoning claims.
Any of those changes requires a separately specified experiment and cannot be
introduced into the registered factorial.

## Risks and interpretation

- Expected effects are small relative to run noise. Paired data order, paired
  seeds, complete runs, and pass-token accounting are part of the causal
  design.
- The shared FBT recipe may not reproduce the best standalone MHDAR-plus-GGQA
  setting. The `mhdar` cell therefore acts as a sensitivity control for this
  exact architecture package, not as a numerical reproduction target.
- A low-token FBT null is compatible with missing formation conditions. A null
  that persists across the registered ladder is stronger evidence against the
  current feedback recipe at this model scale.
- `df_soft` non-adoption can reflect optimization path dependence. Compare it
  with hard DF before interpreting null mass as lack of utility.
- The flagship hybrid is a synthesis rather than a reproduced architecture:
  Kimi's 3:1 evidence used KDA with MLA at a different scale, while Qwen3-Next
  used Gated DeltaNet with gated global GQA. Those deployed precedents motivate
  the mixer and gate choices but do not establish their interaction with DF;
  realized stability, throughput, and quality remain empirical.
- KDA's asymptotic cache advantage does not guarantee a realized speedup at an
  8,192-token context. Promotion uses measured end-to-end training, prefill,
  decode, memory, and pass-token costs on the selected hardware.
- Parameter counts differ because architecture gates, feedback fusion, and
  routers add parameters. Always report exact arm parameter counts alongside
  token and pass-token budgets.
- Contraction is a stability condition, not evidence that latent feedback
  improves language modeling or performs serial reasoning.

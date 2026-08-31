# Design

This document is the authoritative contract for the current experiment. It
defines the model family, training recipe, comparisons, diagnostics, scale
plan, and promotion gates. Accepted scientific evidence belongs in
[findings.md](findings.md); active scratch work belongs in
[journal.md](journal.md); primary source roles are indexed in
[../references/refs.yaml](../references/refs.yaml).

## Objective

The registered screen asks whether Multi-Head Delta Block routing (MHDB) and
Full-Bandwidth Transformer (FBT) feedback are complementary, independent, or
redundant when added to one common hybrid sequence-model trunk:

1. **MHDB** lets each sublayer read an additive mixture of a learned null, the
   current column's input seed, completed four-layer block deltas, and the
   current block's partial delta, with independent depth selection by
   contiguous feature group.
2. **FBT** returns the previous column's top state to layer 0 of the next
   column through a token-gated fusion, giving latent state another full pass
   through depth.

The four hybrid arms form the primary MHDB x FBT factorial. The completed
pure-GQA `vanilla` run remains a fifth, external trunk control:

| Arm | Hybrid trunk | MHDB | FBT | Parameters |
|---|---:|---:|---:|---:|
| `vanilla` | no | no | no | 222,876,672 |
| `base` | yes | no | no | 241,455,840 |
| `mhdb` | yes | yes | no | 241,511,136 |
| `fbt` | yes | no | yes | 242,637,792 |
| `df` | yes | yes | yes | 242,695,392 |

The factorial contrasts use `base`, not `vanilla`, as their shared baseline.
`vanilla` versus `base` isolates the whole PKDA/GGQA trunk replacement. The
old vanilla state layout and initialization are byte-identical under the
current code, so its completed screen record remains admissible without a
rerun. There is no optional-feedback arm.

The claim boundary is pretraining behavior under the registered recipe and
data. No result is a general claim about recurrent transformers, depth routing,
reasoning, or adaptive computation without a dedicated evaluation that supports
that claim.

## Screen trunks

Every arm uses one `DFModel` implementation and exact `ModelConfig` flags.
Shared channel mixing and outer geometry are:

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
| Routing block size | 4 layers |
| PKDA heads | 8 |
| PKDA key/value head width | 128 |
| PKDA Q/K/V projection width | 1,024 |
| PKDA convolution width | 4 |

`vanilla` is the preserved bias-free pre-norm GQA decoder: twelve RoPE GQA
layers, packed QKV projection, per-head Q/K RMSNorm, and no output gate. The
four factorial arms replace its token mixers with exactly:

```text
[PKDA, PKDA, PKDA, gated global GQA] x 3
```

All hybrid global-attention layers are dense, causal, and NoPE. For their
pre-normalized input `x` they compute:

```text
q, k, v = split(W_qkv x)
z       = concat(GQA(q, k, v))
o       = W_o(sigmoid(W_g x) * z)
```

The gate has one coordinate per query-head output coordinate and acts before
the output projection and branch scaling; it does not modify attention logits
or softmax weights. PKDA's causal short convolution carries order and recency,
so the hybrid has no explicit positional embedding in either mixer.

Each screen PKDA mixer uses 8 query/key heads and 8 value heads with
`d_k = d_v = 128`. This is the directly reproduced 340M Preconditioned
DeltaNet geometry, with Kimi Linear's fixed 128-dimensional KDA heads and 3:1
hybrid cadence. Its base parameterization is bias-free Q/K/V projection;
separate causal depthwise convolutions followed by SiLU; L2-normalized Q/K;
rank-128 channel-wise decay; one sigmoid delta-update gate per head; head-wise
RMSNorm at epsilon `1e-5`; a rank-128 sigmoid output gate; and a bias-free
output projection. One packed Adam-side hidden-width projection supplies five
non-overlapping control slices: the main-decay bottleneck, delta-update logits,
preconditioner-decay logits, preconditioner-gain logits, and output-gate
bottleneck. Packing changes neither parameter count nor optimizer semantics.

The diagonal apply-to-key preconditioner has independent scalar decay and gain
projections, `x = 1.5`, `eps = 1e-6`, a learned log-space center initialized to
`-0.2`, decay rates sampled on `[1,16]`, and softplus time constants sampled
log-uniformly on `[0.001,0.1]`. CUDA training uses the pinned upstream FLA
chunk operator at chunk size 64 with internal backward recomputation. CPU and
MPS execute the literal recurrent equations. CUDA cannot silently fall back to
that sequential reference.

Every arm retains the same packed SwiGLU, tied embedding/readout, and final
RMSNorm. Each token-mixer and MLP branch output is scaled by `1/sqrt(2L)`
before residual addition. There is no dropout.

Embedding weights and optimizer state remain FP32. Under CUDA autocast, the
embedding output, residual stream, recurrent payload, and routed values are
BF16. CPU and MPS use the same semantics through portable PyTorch operations.

## MHDB package

The screen, same-geometry token ladder, and flagship all use the same
four-layer block-delta source contract. Individual attention and MLP branch
deltas are never retained as separate addressable sources.

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

Every routing site owns a learnable width-`D` null vector initialized to zero
and prepends it to the source bank. Zero query initialization makes every head
uniform over its available sources, including the null. Null mass initially
adds nothing; after training, its mass must be interpreted together with the
null vector's RMS because the learned value need not remain zero.

### Sources and residual identity

For a column input seed `s`, layer `l` produces scaled attention and MLP deltas
`a_l` and `m_l`. The residual stream is always:

```text
h_top = s + sum_l (a_l + m_l)
```

Partition the layers into consecutive four-layer cells. If `c_b` is the
residual at entry to cell `b`, define:

```text
partial_b = h_current - c_b
Delta_b   = h_cell_exit - c_b
```

At a site in cell `b`, the bank is the site-local null, the column seed, one
completed `Delta_j` for every earlier cell, and `partial_b` when it is nonzero.
The MLP site sees the partial after adding its layer's attention delta. At the
cell boundary, the partial becomes the single completed `Delta_b`. Thus:

```text
h_current = seed + sum(completed block deltas) + current partial delta
```

The routed mixture is added only to the sublayer's pre-norm read; it is never
accumulated directly into `h`. This transient-read rule preserves the exact
telescoping identity. On routed pass-1 sites, the zero-query non-null
mixture is a scalar multiple of `h`, so the following RMSNorm makes routing
functionally inert up to its epsilon.

For `mhdb`, the seed is the token embedding `e`. For `df`, the seed is `e` on
pass 1 and the fused input `u` on feedback passes.

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

During sequential hybrid decoding, the cache stores KV only for the three
global GQA layers. Each PKDA layer instead retains its fixed-size FP32 recurrent
matrix, FP32 diagonal-preconditioner state, and three short-convolution
histories. Only the immediately previous FBT payload is carried outside the
mixer caches; each generated column consumes it once and emits the next.

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
   within-column MHDB site, and run the common PKDA/GGQA hybrid trunk.
3. Preserve the clean residual identity while attention and MLP sites read
   routed mixtures of the null, seed, completed block deltas, and current
   partial block delta.
4. Route over the null, seed, and completed block deltas with a dedicated
   multi-head payload router.
5. Add the routed enrichment to the full top state and normalize the result.

With `Delta = [Delta_0, ..., Delta_(B-1)]` for the completed four-layer cells:

```text
r_payload = route(null, seed, Delta; q_payload)
p = payload_norm(h_top + r_payload)
```

The base `h_top` guarantees that the full column state is retained; the routed
term selects which component receives an additional cross-column path. At
zero-query initialization, the null contributes zero and the other sources sum
to `h_top`, so the routed addition is `h_top / (B + 2)`. The following RMSNorm
therefore makes the initial payload equal to the bare normalized top state up
to its epsilon, matching the FBT payload functionally at initialization.

The payload query is not conditioned on the next token. Token-dependent control
occurs in the FBT entry gate after the payload shifts to the next position.

An equivalent high-level pass is:

```python
e = embed(tokens)
s = e if payload is None else fuse(payload, e)
h = s
completed = []
for cell in four_layer_cells:
    cell_start = h
    for layer in cell:
        partial = [] at cell entry else [h - cell_start]
        sources = [null, s, *completed, *partial]
        a = scaled_hybrid_mixer(rmsnorm(h + route(sources)))
        h = h + a
        partial = h - cell_start
        m = scaled_mlp(rmsnorm(h + route([null, s, *completed, partial])))
        h = h + m
    completed.append(h - cell_start)
payload = payload_norm(h + route([payload_null, s, *completed], q_payload))
logits = tied_head(final_norm(h))
```

The first attention router sees `[null, seed]`; all routed sites are active and
can learn to select their null.

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

Feedback suffix inputs are FBT-fused. Non-feedback arms always use one pass.

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
and step addresses. One initialization seed produces a byte-identical hybrid
trunk in `base`, `mhdb`, `fbt`, and `df`, identical MHDB parameters in `mhdb`
and `df`, and identical FBT fusion weights in `fbt` and `df`; conditional
modules never advance a shared stream. The structurally different `vanilla`
control retains its exact previous initialization. Initialization seeds differ
only when registering a new paired seed. Resumption returns to the same row and
the same keyed feedback draws.

## Optimizer and schedule

Every arm uses the same recipe.

### Parameter groups

- **NorMuon:** every trainable two-dimensional weight except the tied
  embedding/unembedding and scale-sensitive gate-producing projections. This
  includes the FBT value and token-gate fusion matrices. Defaults: learning
  rate `1e-2`, momentum `0.95`, row second-moment beta `0.95`, five
  Newton-Schulz steps, epsilon `1e-8`, decoupled weight decay `0.01`.
- **Adam:** scale-sensitive gate-producing projections, tied embeddings,
  RMSNorm weights, routing queries, null vectors, depthwise convolution
  weights, and all other non-matrix parameters. This includes every GGQA gate
  and PKDA's packed control projection, main-decay expansion, and output-gate
  expansion. Defaults: learning rate `5e-4`, betas `(0.9, 0.95)`, epsilon
  `1e-8`, no weight decay.

NorMuon orthogonalizes the momentum, normalizes rows by their second moments,
and globally rescales the update to Frobenius norm `0.2 * sqrt(m*n)` before
applying learning rate and decoupled decay.

### Default screen schedule

The default 6,700-step WSD schedule is:

| Phase | Steps | Pass behavior |
|---|---:|---|
| Warmup | 1–200 | one pass |
| Stable heat | 201–3,350 | one pass |
| Stable heat | 3,351–5,025 | feedback arms draw 1, 2, or 3 passes |
| Cooldown | 5,026–6,700 | feedback arms draw 1, 2, or 3 passes |

After step 3,350, `P(k>1) = 0.50` and `P(k=3 | k>1) = 0.12`, giving the
50%/44%/6% one-/two-/three-pass mixture through the second half of the run.
Across the full run this targets the 75%/22%/3% mixture. The exact draws are
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

CPU and MPS use PyTorch scaled-dot-product attention, the literal recurrent
PKDA equations, the algebraic MHDB router, chunked tied-head cross-entropy,
eager execution, and the same model, loss, optimizer, data, and checkpoint
semantics. This path owns fast invariant tests and analysis.

### CUDA path

The authoritative Jobe screen path uses:

- BF16 trunk activations with FP32 weights and optimizer state;
- the pinned upstream FLA PKDA chunk kernel with chunk size 64, FP32 boundary
  states, and recomputed backward intermediates;
- FLA's fused head-wise RMSNorm-plus-sigmoid-output-gate operator after every
  PKDA recurrence;
- FlashAttention for full-sequence, prefill, GQA, and cached decoding;
- a fixed-capacity Triton MHDB router over source pointers, with full-width RMS
  scores, per-head masked softmaxes, FP32 value accumulation, and an analytic
  backward that retains cross-head RMS coupling;
- cut cross-entropy for the tied head, including the exact squared
  log-partition gradient without materializing vocabulary-wide logits;
- full-block compilation around global-attention kernels and segmented PKDA
  block compilation that graph-breaks at the opaque FLA recurrence while
  compiling the projections, controls, routing, MLP, and residual work around
  it;
- fixed-address forward/backward CUDA graphs for every schedule-reachable train
  mode and shared-pool no-grad validation graphs;
- BF16 keyed jitter drawn directly into graph input buffers;
- internal activation checkpointing above the measured screen work threshold;
- asynchronous snapshot staging to pinned host memory followed by atomic
  background serialization.

The public experiment surface does not expose kernel, graph, or activation
checkpoint policy switches. Those are execution choices, not factorial axes.

Fresh-process capture measurements at the default screen geometry on Jobe are:

| Arm | Peak allocated | Peak reserved | Captured train/eval graphs |
|---|---:|---:|---:|
| `base` | 6.43 GiB | 13.65 GiB | 3 |
| `mhdb` | 6.58 GiB | 13.97 GiB | 3 |
| `df` | 12.73 GiB | 23.00 GiB | 7 |

The reserved graph pool, not live allocated tensors, is the concurrency
boundary. Even the two lightest hybrid processes exceed the RTX 4090's usable
23.50 GiB when their reservations are combined. Screen jobs therefore remain
serial on Jobe; use the durable queue rather than process-level co-training.
PKDA's memory savings are retained as safety headroom and make the registered
three-pass DF modes fit, but do not justify a concurrent-run mode.

### Checkpoint contract

New snapshots use checkpoint contract v10, and only v10 is resumable.
Snapshots contain model, both optimizer states, exact state-defining arguments,
step, and Python/Torch/CUDA RNG state. A resume inherits all state-defining
fields and rejects an explicit conflict. Runtime paths, device, evaluation
cadence, snapshot cadence, and evaluation-row count may change per invocation.

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

No-feedback arms use their Standard metric when a factorial table is shown for
Soft or Fused mode.

### Effect and interaction

For validation loss `L` (lower is better), define factorial gains over the
hybrid `base` arm:

```text
G_A = L_base - L_A
```

`G_mhdb` is the MHDB effect, `G_fbt` is the recurrence effect, and `G_df` is
their joint effect. The factorial interaction is:

```text
I = G_df - G_mhdb - G_fbt
  = L_mhdb + L_fbt - L_df - L_base
```

`I > 0` is superadditive loss reduction, `I = 0` is additive, and `I < 0` is
subadditive. Compute the statistic on paired checkpoints and per decode mode,
then aggregate paired seed differences. Do not mix Standard, Soft, and Fused
losses inside one interaction estimate.

Report `L_vanilla - L_base` separately as the whole hybrid-trunk contrast. It
is not an MHDB or FBT main effect and does not enter `I`.

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

- null mass and null-vector RMS at every routed site;
- within-column seed mass (`e` for `mhdb`, `u` for `df` feedback passes);
- completed-block and current-partial mass at attention and MLP sites;
- seed and completed-block mass at payload sites.

`scripts/route_report.py` produces held-out route maps and query geometry from
a hard-DF checkpoint. `scripts/payload_swap.py` replaces the learned payload
mixture with top-only, uniform, or single-source alternatives on the same
weights. The sweep is a co-adapted same-checkpoint intervention: its landscape
identifies sensitive payload content but does not estimate the effect of
training an alternative payload rule from scratch.

## Screen and scale plan

### 223–243M screen

The screen uses the geometry above, 6,700 steps, 2.003B predicted tokens per
run, FineWeb-Edu, two paired initialization seeds, and Jobe's RTX 4090. The
completed `vanilla` run is retained; the four hybrid factorial arms are new
runs under checkpoint-v10.

The screen is designed to establish:

1. whether the hybrid trunk improves on the completed vanilla control;
2. the signs and paired magnitudes of the MHDB, recurrence, and joint effects
   within one shared hybrid trunk;
3. whether hard DF is stable under recurrent self-composition;
4. whether learned null, seed, block, and payload routing paths are adopted.

The screen runs at 8.25–8.99 predicted tokens per parameter across its arms. It
is a sensitivity and interaction screen, not a decisive test of FBT formation
at the high token-per-parameter regime.

### Token ladder

The registered ladder keeps the screen geometry and extends selected arms through
2B, 8B, and 32B predicted tokens, taking a WSD cooldown branch at each rung and
continuing the next rung from that rung's protected pre-cooldown checkpoint.
Finalist arms share the stream prefix and use one paired seed, with the screen's
two-seed spread retained as the noise estimate.

This ladder is not currently runnable through exact resume: `steps` is a v10
state-defining field, and no tested branch-from-heat-end continuation command
exists. Before ladder launch, code and tests must define a new run address,
preserve model/optimizer/RNG and row continuity, extend the stable phase without
re-warming, and create a new cooldown branch without weakening exact resume.

### Flagship

The registered flagship is hard DF with an analytical parameter count of
1,335,420,192 and a 441B-predicted-token budget, defined as 400 predicted
tokens per active non-embedding parameter before batch alignment. It has width
1,536, 24 decoder layers, SwiGLU width 6,656, the project's tied 151,936-token
vocabulary, context 8,192, and six identical four-layer cells. Its token-mixing
schedule is exactly:

```text
[PKDA, PKDA, PKDA, gated global GQA] x 6
```

The operating point has four explicit authority classes:

| Surface | Authority |
|---|---|
| Width 1,536, 24 layers, SwiGLU 6,656, GQA 16/8-by-96, context 8,192, nominal 1B/400B-token scale, WSD/NorMuon recipe, and latent-feedback schedule | Full-Bandwidth Transformer, inherited directly except for the project's tokenizer and vocabulary |
| PKDA 20-by-128 geometry, convolution width 4, Q/K L2 normalization, sigmoid output gate, NoPE, and 3:1 hybrid cadence | Kimi Linear paper and released configuration |
| Apply-to-key recurrence, independent preconditioner gates, `x = 1.5`, bounded squash, initialization, chunk/recurrent forms, FP32 boundary states, and global gradient clip 1.0 | Preconditioned DeltaNet paper and upstream FLA implementation |
| Sigmoid-gated global GQA | Released Qwen3-Next configuration and implementation |
| PKDA replacing KDA inside the 3:1 cadence, global GQA replacing MLA, hard DF plus block-delta routing, 151,936-token vocabulary, 400 predicted tokens per active non-embedding parameter, exact mid-run feedback boundary, 441B batch-aligned budget, and replicated eight-rank execution | Registered project synthesis; these interactions require the promotion and implementation gates |

There is no sliding-window attention. Each PKDA layer uses 20 query/key heads,
20 value heads, `expand_v = 1`, and `d_k = d_v = 128`, so its key and value
projection widths are both 2,560. This is not constrained to equal the residual
width. Kimi Linear's scaling-law table pairs 20 heads with hidden width 1,536
and fixes `d_k = d_v = 128` throughout its experiments; its released model also
uses a KDA projection wider than its residual stream. The 20-by-128 geometry is
therefore the registered operating point rather than a width-preserving
12-by-128 convenience.

The base recurrence keeps the released Kimi Linear parameterization:
bias-free Q/K/V projections; separate bias-free causal depthwise convolutions
of width 4 followed by SiLU; L2-normalized Q and K; a rank-128 channel-wise
decay projection; one sigmoid delta-update gate per head; head-wise RMSNorm
with epsilon `1e-5`; a rank-128 sigmoid output gate whose final projection has
a zero-initialized bias; and a bias-free output projection. All ordinary dense
and tied-embedding weights are initialized from `Normal(0, 0.02)`. The three
depthwise convolution weights retain the released implementation's inherited
`Conv1d` Kaiming-uniform initialization at fan-in 4. RMSNorm weights initialize
to one; routing queries and nulls initialize to zero. The special KDA and
preconditioner gate initializations below override the ordinary rule where
stated.

PKDA applies the stable diagonal apply-to-key preconditioner from
Preconditioned DeltaNet. Each head maintains an auxiliary nonnegative diagonal
state `A_t` with its own scalar decay `alpha_P_t` and gain `beta_P_t`. Their
input projections and learned gate parameters are independent of the main KDA
decay and update gate; they are not tied. The learned per-head log-space center
`c` is initialized to `-0.2`.
The preconditioner decay rate is initialized with `exp(A_P)` sampled uniformly
on `[1, 16]`; its softplus time constant is initialized log-uniformly on
`[0.001, 0.1]`. The squash uses `x = 1.5`, `eps = 1e-6`, disables the optional
negative-eigenvalue and safe-gate modes, and bounds every coordinate of the
write-key preconditioner to `[2/3, 3/2]`. For each head, with zero initial
matrix and preconditioner states, the semantic recurrence in the project's
key-first state convention is:

```text
log_alpha_t = -exp(A) * softplus(W_f_up W_f_down x_t + b_f)
alpha_t     = exp(log_alpha_t)
beta_t      = sigmoid(W_beta x_t)
log_alpha_P_t = -exp(A_P) * softplus(W_alpha_P x_t + b_alpha_P)
alpha_P_t     = exp(log_alpha_P_t)
beta_P_t      = sigmoid(W_beta_P x_t)
A_t           = alpha_P_t A_(t-1) + beta_P_t (k_t ⊙ k_t)
c              = learned log-space center, initialized to -0.2
r_t            = log(A_t + eps) - c
s_t            = r_t / (1 + abs(r_t))
B_t            = exp(-log(1.5) * s_t)
k_write_t      = B_t ⊙ k_t
S_tilde_t   = Diag(alpha_t) S_(t-1)
S_t         = (I - beta_t k_write_t k_t^T) S_tilde_t
              + beta_t k_write_t v_t^T
o_t         = S_t^T q_t
```

Training and long-prompt prefill use the chunkwise-parallel PKDA form with
fixed chunk size 64. Backward recomputes the large chunk intermediates rather
than retaining them. Cached autoregressive decoding uses the mathematically
equivalent single-token recurrent form and advances both `S_t` and `A_t`.

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
modify attention logits or softmax weights, and it is distinct from PKDA's
main delta-update, preconditioner, and output gates and from the FBT entry gate.
Because the gated `o_t` is still the single attention branch delta, the
seed-plus-deltas residual identity is unchanged.

The flagship has no explicit positional embedding in either mixer: PKDA's
causal convolution and data-dependent recurrent transition carry order and
recency, and the global GQA layers use NoPE. This is a deliberate synthesis.
Kimi Linear supplies PKDA's KDA substrate, NoPE global attention, and the
empirically selected 3:1 cadence, but uses unpreconditioned KDA with global MLA.
Preconditioned DeltaNet supplies the PKDA recurrence and stable preconditioner
parameterization, but evaluates pure PKDA rather than this hybrid. Qwen3-Next
supplies independent interval-four Gated DeltaNet/gated-global-GQA precedent,
but does not use PKDA.

The analytical count is:

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

The total includes the 233,373,696-parameter tied embedding/readout. Removing
that term leaves 1,102,046,496 active non-embedding parameters, which is the
registered denominator for the flagship data budget.

Relative to unpreconditioned KDA, the two width-to-head preconditioner
projections and three learned per-head vectors add 61,500 parameters per PKDA
layer and 1,107,000 across the 18 PKDA layers. Relative to the discarded
12-by-128 sketch, 20-by-128 adds 118,886,976 parameters. The GQA gates add
14,155,776 weights to the ungated hybrid. The implementation gate must make an
instantiated model reproduce the analytical total exactly before spend
approval.

#### Flagship training and execution

The flagship retains the experiment's FineWeb-Edu stream, Qwen3 tokenizer,
row addressing, paired initialization, hard-DF objective, and optimizer
partition. Its exact operating schedule is:

| Quantity | Registered value |
|---|---:|
| Sequence length | 8,192 predictions |
| Global batch | 40 sequences = 327,680 predicted tokens |
| Distributed batch | 8 ranks x microbatch 1 x accumulation 5 |
| Active non-embedding parameters | 1,102,046,496 |
| Budget rule | 400 predicted tokens per active non-embedding parameter |
| Unrounded token target | 440,818,598,400 |
| Optimizer steps | 1,345,272 |
| Exact predicted tokens | 440,818,728,960 |
| Warmup | steps 1-200 |
| Stable heat | steps 201-1,008,954 |
| Cooldown | steps 1,008,955-1,345,272 |
| Feedback boundary | after step 672,636 |
| Whole-run pass mixture | expected 75% / 22% / 3% for 1 / 2 / 3 passes |
| Expected token-equivalent compute | approximately 564.248B pass-tokens |

The mathematical 400-token target is 440,818,598,400 predicted tokens. The
nearest whole-step schedule at the registered global batch exceeds it by only
130,560 tokens, less than one optimizer batch, and realizes 400.000118
predicted tokens per active non-embedding parameter.

The 327,680-token batch is the clean eight-rank realization nearest the FBT
paper's approximately 300K-token batch: every rank processes one full sequence
per microstep and synchronizes after five microsteps. The first half of the run
is one-pass. In the second half, hard DF draws 1, 2, or 3 passes with
probabilities 50%, 44%, and 6%, using the same keyed schedule as the screen;
the exact realized pass-token count is recorded rather than inferred from the
expectation.

Learning rates, decay, and the z-loss follow the shared NorMuon/Adam WSD recipe
above. The flagship additionally clips the accumulated global FP32 gradient
norm to 1.0 immediately before the optimizer step, matching the 1B PKDA
training precedent. The packed PKDA control projection, main-decay expansion,
and output-gate expansion remain in Adam; Q/K/V/output projections remain in
NorMuon. The three-dimensional depthwise convolution weights are non-matrix
parameters and remain in Adam with no weight decay. All learned KDA and
preconditioner rate parameters are also exempt from weight decay.

The registered distributed target is eight H100 80GB GPUs under replicated
DDP, BF16 autocast, FP32 parameters, FP32 gradients at the optimizer boundary,
and FP32 optimizer state. It uses no tensor, pipeline, context, or parameter
sharding. Every transformer block is activation-checkpointed on every pass;
feedback payloads and source banks remain differentiable, and checkpoint
recomputation does not preserve RNG state because all stochastic feedback
choices arrive as keyed tensor inputs. Global GQA uses BF16 FlashAttention.
PKDA uses the upstream-equivalent Triton chunk kernel for training/prefill, the
fused recurrent kernel for single-token decode, and FLA's fused head-wise
RMSNorm-plus-sigmoid-gate operator. This target remains
non-runnable until its per-rank memory, graph behavior, parity, and throughput
pass the implementation gate.

PKDA recurrent matrix and diagonal-preconditioner boundary states are FP32;
PKDA outputs and convolution caches are BF16. Global-GQA KV caches are BF16.
At a full 8,192-token prompt, the registered per-sequence token-mixer cache is:

| Cache | Size |
|---|---:|
| 18 PKDA matrix states `[20,128,128]` | 22.500 MiB |
| 18 PKDA preconditioner states `[20,128]` | 0.176 MiB |
| 18 Q/K/V convolution caches, width 2,560 and history 3 | 0.791 MiB |
| 6 global-GQA BF16 KV caches `[8192,8,96]` | 144.000 MiB |
| **Token-mixer total** | **167.467 MiB** |

This total excludes allocator overhead, the width-1,536 DF payload, logits,
and serving-runtime metadata. A prefill or Jacobi pass still begins with zero
PKDA state as specified below; FP32 boundary-state precision applies whenever
a state is materialized or carried across decode calls.

#### Flagship block-delta routing

The four-layer attention cell is also the routing block. A cell contains four
token mixers and four MLPs, but none of their eight individual branch deltas is
retained as an addressable routing source. Let `c_b` be the residual at entry to
cell `b`. While that cell is executing, define:

```text
partial_b = h_current - c_b
Delta_b   = h_cell_exit - c_b
```

At a routing site in cell `b`, the source bank is the site-local null, column
seed, one completed `Delta_j` for every earlier cell, and one `partial_b` when
it is nonzero. The first mixer in a cell therefore sees the null, seed, and
completed earlier cells; later sites also see one evolving aggregate for the
current cell. At the boundary, `partial_b` becomes the single completed
`Delta_b`. The non-null column sources keep the exact decomposition:

```text
h_current = seed + sum(completed cell deltas) + current partial delta
```

Routing remains a transient pre-norm read and never accumulates directly into
the residual stream. The hard-DF payload router routes over its null, the seed,
and the six completed cell deltas `[Delta_0, ..., Delta_5]`. The deepest hard
within-column router and the payload router therefore each have at most eight
sources. This is the block form described by the Delta Attention Residuals and
Attention Residuals papers, rather than an ad hoc bank of individual attention
and MLP outputs.

PKDA state is local to one transformer evaluation. A Jacobi or fused-prefill
pass starts every PKDA matrix state, diagonal preconditioner state, and
convolution state from zero and advances them once across that pass's causal
token order. Autoregressive decoding retains all three states per PKDA layer
alongside the GQA KV caches and advances them once per generated token. No PKDA
state is carried from one repeated pass over the same token positions into the
next; the DF payload is the only cross-pass state. After the final prefill pass,
the PKDA and GQA caches advance normally over newly generated positions.

#### Flagship implementation gate

The exact flagship implementation must pass portable/chunkwise/recurrent PKDA
value and gradient parity; bounded-preconditioner and independent-gate tests;
matrix-state, diagonal-preconditioner-state, and convolution-cache continuation
parity; gated-GQA value and gradient parity against an explicit
sigmoid-times-attention reference; block-source and payload-source identity
tests; exact compute and parameter accounting; optimizer partition tests; the
width-1,536/H=8 router gate; distributed execution; checkpoint portability;
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

The hybrid `base` arm must clearly improve on vanilla in Standard mode for the
screen to justify replacing the old trunk. Within the hybrid factorial, MHDB
must produce a resolvable paired effect for the screen to serve as a
sensitivity gate. A stable FBT null at screen scale does not by itself exclude
DF from the ladder because the screen is far below the registered
token-per-parameter regime. DF enters the ladder only if its paired effect is
credible enough that additional tokens can resolve the interaction between the
two packages.

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
- The 8-by-128 PKDA/GGQA screen is a scale-adapted synthesis. The `base` cell
  measures that trunk directly, and `mhdb` measures routing conditional on it;
  neither is a numerical reproduction target for its source paper.
- A low-token FBT null is compatible with missing formation conditions. A null
  that persists across the registered ladder is stronger evidence against the
  current feedback recipe at this model scale.
- The flagship hybrid is a synthesis rather than a reproduced architecture:
  Kimi's 3:1 evidence used unpreconditioned KDA with MLA, Preconditioned
  DeltaNet evaluated pure PKDA at different context and training budgets, and
  Qwen3-Next used Gated DeltaNet with gated global GQA. Those precedents
  motivate the mixer and gate choices but do not establish their interaction
  with DF; realized stability, throughput, and quality remain empirical.
- PKDA's asymptotic cache advantage does not guarantee a realized speedup at
  an 8,192-token context, and its diagonal preconditioner adds work to every
  recurrent layer. Promotion uses measured end-to-end training, prefill,
  decode, memory, and pass-token costs on the selected hardware.
- Parameter counts differ because architecture gates, feedback fusion, and
  routers add parameters. Always report exact arm parameter counts alongside
  token and pass-token budgets.
- Contraction is a stability condition, not evidence that latent feedback
  improves language modeling or performs serial reasoning.

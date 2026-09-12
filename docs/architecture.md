# Architecture

`DeltaModel` has one core architecture: PKDA/GQA token mixers, Multi-Head Delta
Block (MHDB) routing, shared and routed SwiGLU experts, and auxiliary two-token
prediction. Conditions select Full-Bandwidth Transformer (FBT) feedback (`f`),
tied depth (`l`), or both (`fl`); the default is `f`. There is no empty condition.

[Scaling](scaling.md) owns geometry and accounting, [design](design.md) owns
data and training, and [references](../references/refs.yaml) records sources.

## The column

The pre-norm decoder has tied embedding/readout, a final RMSNorm,
and no dropout. Every RMSNorm uses epsilon `1e-6`. Layers form four-layer
cells `[PKDA, PKDA, PKDA, NoPE-GGQA]`. Each trunk FFN uses one shared expert
and a configurable number of selected routed experts. All presets have four
cells; the middle two form the tied core under `l`.

The diagram expands **`fl`**, with both feedback and tied depth enabled. Panel
A follows one column from input to both outputs; B-H expand its modules and
state. Panel I connects columns, passes, and training objectives. Dimensions
are symbolic, with the screen preset as a concrete example. Arrows carry
activations unless labeled as state, weights, or supervision; `(*)` is an
elementwise product and `(+)` is addition. Linear maps are bias-free except
PKDA's output-gate expansion.

```text
A. fl: COMPLETE COLUMN / PASS
============================================================================================
  B = batch rows; T = input positions; D = residual width; V = vocabulary size
  L = unique trunk layers = 4C; C = unique cells; c = C-2 = core cells
  r = core iterations; k = Jacobi pass; t = token position; h_ff = per-expert width
  Screen: D=768, V=50,304, L=16, C=4, c=2, h_ff=832; default eval/decode r=4
  Residuals, seeds, deltas, payloads: [B,T,D] in a full row; [B,1,D] in decode

  token x_t --> Emb.weight [V,D] --> e_t -------------------------------+
                                      |                               |
                                      v                               |
                           gate_norm --> W_G [D->D]                   |
                                      |                               |
                                   sigmoid                            |
                                      |                               |
  incoming payload --> W_U [D->D] --> (*) --> entry_norm --> fused_t    |
       ^                                                      |       |
       |                                                      v       v
       |                                            +-------------------------+
       |                                            | select column seed s_t  |
       |                                            | plain: e_t              |
       |                                            | feedback: fused_t       |
       |                                            +------------+------------+
       |                                                         |
       |     Plain = pass 1 / plain prefix / Standard decode.    h=s
       |     At fused positions e_t controls the gate only.       |
       |                                                         v
       |       s ---------------------------> [MHDB source bank: seed]
       |                                                         |
       |                          +------------------------------v------------------+
       |                          | PRELUDE P: one four-layer cell [B]              |
       |                          | PKDA+FFN -> PKDA+FFN -> PKDA+FFN -> GGQA+FFN    |
       |                          +------------------------------+------------------+
       |                                                         |  emit Delta_P
       |                                                   core_entry
       |                                                         |
       |                          +------------------------------v------------------+
       |                          | TIED CORE: R_0 -> ... -> R_(c-1)                |
       |                          | Each R_j is a distinct four-layer cell [B].     |
       |                          |                                                 |
       |                          |  +--> R_0 --> ... --> R_(c-1) --+               |
       |                          |  |                              |               |
       |                          |  +--- h continues; repeat r ----+               |
       |                          |                                                 |
       |                          | Same R_j weights on every visit.                |
       |                          | One accumulated Delta_Rj per cell [C].          |
       |                          | Separate mixer cache for each iteration [I].    |
       |                          +------------------------------+------------------+
       |                                                         |
       |                                                    core_state
       |                                                         |
       |                          +------------------------------v------------------+
       |                          | CODA: one four-layer cell [B]                   |
       |                          | PKDA+FFN -> PKDA+FFN -> PKDA+FFN -> GGQA+FFN    |
       |                          +------------------------------+------------------+
       |                                                         |  emit Delta_Coda
       |                                                       h_top
       |                                                         |
       |                        +--------------------------------+
       |                        |
       |                        v                      seed and emitted cell deltas
       |                 final_norm                               |
       |                        |                                 v
       |                    * 1536/D                   final source bank [D]
       |                        |                     {seed, Delta_P, all Delta_Rj,
       |                 @ Emb.weight.T                       Delta_Coda}
       |                        |                                 |
       |                 NTP logits [B,T,V]                       v
       |                        |                         payload MHDB
       |                 target x_(t+1)                     (own null)
       |                                                          |
       |                                      h_top ------------>(+)
       |                                                          |
       |                                                     payload_norm
       |                                                          |
       +---- next feedback entry [I] <-------- p_t [B,T,D] <-------+
                                                |
                                                +--> auxiliary MTP [H]

  Identity: h_top = seed + Delta_P + sum_j Delta_Rj + Delta_Coda
  Executed trunk: 2+c*r cells, 4*(2+c*r) layers; parameters remain L unique layers.
  All presets at r=4: P -> (R_0 -> R_1) x 4 -> Coda = 40 executed layers.
  At r=1, fl matches f in values, routes, losses, and gradients.


B. EVERY TRUNK CELL: FOUR LAYERS, EIGHT ROUTED READS
============================================================================================
  h_in --> [PKDA layer] --> [PKDA layer] --> [PKDA layer] --> [NoPE-GGQA layer] --> h_out
            each layer has the complete two-branch residual shell below

                         attention source bank [C,D]
                                     |
                                  MHDB_attn
                                     |
  h ------------------------------->(+) --> RMSNorm_attn --> mixer [E or F]
  |                                                                     |
  |                                                             * b, b=1/sqrt(2L)
  |                                                                     | a
  +------------------------------------------------------------------->(+) --> h'

  FFN bank: replace current partial by partial+a; create it as a if absent
                                     |
                                  MHDB_ffn
                                     |
  h' ------------------------------>(+) --> RMSNorm_ffn --> shared+routed FFN [G]
  |                                                                     |
  |                                                                    * b
  |                                                                     | m
  +------------------------------------------------------------------->(+) --> h_next

  h' = h+a; h_next = h'+m. The route is consumed only by the temporary pre-norm read.
  Cell partial grows by a+m after each layer; the cell emits its final partial.
  b uses UNIQUE L, including in the tied core and MTP; it does not depend on r.
  Each physical layer has its own mixer, FFN, two norms, and two MHDB sites.


C. TIED-CORE DELTA BOOKKEEPING: ONE ADDRESS PER CELL, ACROSS ALL VISITS
============================================================================================
  At entry to core cell R_j on iteration i:

     seed  +  Delta_P  +  sum_(q != j) Delta_Rq_so_far  +  Delta_Rj_previous  =  h
       |         |                     |                       |
       +---------+---------------------+                       |
                        |                                     |
                 effective origin o_j                  live partial d_j
                        |                                     |
                        |           +-------------------------v----------------+
                        |           | R_j: four layers                         |
                        |           | other-cell deltas stay fixed this visit  |
                        |           | own partial grows after each branch      |
                        |           +-------------------------+----------------+
                        |                                     |
                        +----------> Delta_Rj_new = h_exit - o_j
                                                              |
                               next visit reuses this delta <--+

  First visit: own partial is absent at the first mixer; as-yet-unvisited cells are absent.
  Later visits: earlier core cells contribute this iteration's deltas; later cells contribute
  their previous iteration's deltas. Own accumulated delta appears only as the live partial.
  o_j advances by other cells' additions between visits; weights and source identities persist.
  For c=1: o_0 = core_entry for every visit, so Delta_R0 = h-core_entry throughout the loop.
  After the last iteration: commit each final Delta_Rj for the coda and payload writer.
  No iteration-boundary reset, extra normalization, or extra residual scale is applied to h.


D. MHDB: TOKEN-LOCAL DEPTH ROUTING, SEPARATE FROM TOKEN ATTENTION AND EXPERT ROUTING
============================================================================================
  one site's learned null --+
  actual column seed -------+
  completed/other deltas ---+--> raw values v_i [N,B,T,D] ----------------------------+
  current partial, if any --+                |                                      |
                                            v                                      |
                             full-D RMS statistic + learned scale g                |
                                            |                                      |
                                   keys k_i [N,B,T,D]                              |
                                            |                                      |
                         split into G=kv_heads contiguous groups                   |
                                            |                                      |
  learned query q [D] --> split q_h --> dot(q_h,key_(i,h))                         |
                                            |                                      |
                               softmax over source index i                         |
                                            |                                      |
                               weights [N,B,T,G]                                   |
                                            |                                      |
                                            +--> weighted raw slices <--------------+
                                                            |
                                      concatenate groups --> route [B,T,D]

  key_i = g*v_i/sqrt(mean_D(v_i^2)+1e-6); route_h = sum_i weight_i,h * v_i,h
  No token mixing, head-dimension score factor, or output projection in this router.
  Groups partition residual features; they do not identify PKDA or GQA projection heads.
  At most C+2 sources, even at depth r>1: null + seed + one delta/partial per cell.
  Payload site: exactly C+2 sources, all cells completed; no dependence on next-token input.
  Each site owns q, g, null; initialization q=0, g=1, null=0 gives uniform source weights.
  Non-null values sum to h: routed enrichment is initially collinear with the residual.


E. PKDA MIXER: CAUSAL CONVOLUTIONS + PRECONDITIONED MATRIX MEMORY
============================================================================================
  z = normalized routed read [B,T,D]; P = n_pkda*128 = 5D/3 at the presets

  z --+--> W_Q: D->P --> causal DWConv1d(4) --> SiLU --> head L2 --> /sqrt(128) --> q_t
      +--> W_K: D->P --> causal DWConv1d(4) --> SiLU --> head L2 ----------------> k_t
      +--> W_V: D->P --> causal DWConv1d(4) --> SiLU ----------------------------> v_t
      |
      +--> packed control projection: D -> 256+3*n_pkda, five disjoint slices
                |
                +--> decay hidden [128] --> W_decay: 128->P --> alpha_t [n_pkda,128]
                +--> update [n_pkda] --> sigmoid ----------------> beta_t [n_pkda]
                +--> decayP [n_pkda] --> independent decay map --> alphaP_t [n_pkda]
                +--> updateP [n_pkda] --> sigmoid --------------> betaP_t [n_pkda]
                +--> gate hidden [128] --> W_gate: 128->P + bias --> sigmoid --> gate_t

  q_t, k_t, v_t and gate_t have shape [B,T,n_pkda,128]. The following is per head/token:

  DIAGONAL MEMORY A                              KEY-BY-VALUE MATRIX MEMORY S
  -----------------                              ----------------------------
  A_(t-1), alphaP_t, betaP_t, k_t                 S_(t-1), alpha_t
                |                                         |
                v                                         v
  A_t = alphaP_t*A_(t-1)                        S_tilde = Diag(alpha_t)*S_(t-1)
        + betaP_t*(k_t*k_t)                               |
                |                                         +---------------------+
                +--> retain A_t for next token            |                     |
                |                                         v                     |
                v                               k_t --> S_tilde^T k_t           |
  u = log(A_t+1e-6)-center                                |                     |
                |                                         v                     |
                v                             error = v_t - S_tilde^T*k_t       |
  B_t = exp(-log(1.5)*u/(1+abs(u)))                       |                     |
                |                                         v                     |
  k_t -------->(*) --> k_write ------> beta_t*k_write*error^T                   |
                                                          |                     |
                                                          v                     |
                                                         (+) <------------------+
                                                          |
                                                          S_t
                                                          |
                                                          +--> retain S_t for next token
                                                          |
                                                q_t --> S_t^T q_t --> o_t
                                                                      |
                                                               per-head RMSNorm
                                                                      |
                                                gate_t ------------->(*)
                                                                      |
                                                concatenate --> W_O: P->D --> mixer output

  alpha = exp(-exp(A_log)*softplus(decay+b)); beta = sigmoid(update).
  alphaP/betaP use their own learned controls, rates, and time biases; B_t is in [2/3,3/2].
  Per layer/iteration: S [B,n_pkda,128,128], A [B,n_pkda,128], three conv histories [B,P,3].
  S and A start at zero and stay FP32; convolution histories store projected inputs.
  The recurrence reads the UPDATED S_t; only its key-side write is preconditioned.
  Output RMSNorm has a learned length-128 scale shared across heads; gate bias starts at zero.
  Q/K/V convolution, SiLU, and L2 normalization compute in FP32 before activation cast.
  CUDA: FLA chunk-64 for training/prefill; recurrent operator for decode. No K/V token bank.


F. NoPE-GGQA MIXER: CAUSAL GLOBAL TOKEN ATTENTION, EVERY FOURTH LAYER
============================================================================================
  z [B,T,D] --> packed Q/K/V + gate projection (independently owned gate weights)
                        |                                              |
         +--------------+---------------+           +--> sigmoid gate [B,T,n_q*96]
         |              |               |                              |
         Q              K               V                              |
    [B,n_q,T,96]   [B,n_kv,T,96]   [B,n_kv,T,96]                       |
         |              |               |                              |
    head RMSNorm   head RMSNorm         |                              |
         |              |               |                              |
         |              +------> per-layer/iteration K/V cache         |
         |                              |                              |
         +--> softmax(Q K^T / sqrt(96) + causal mask) --> weighted V   |
                                                         |             |
                                                   concatenate heads   |
                                                         |             |
                                                        (*) <----------+
                                                         |
                                               W_O: n_q*96 -> D

  GQA shares n_kv K/V heads across n_q query heads; presets have n_q=2*n_kv.
  No positional embedding or rotary transform. The gate scales output features after attention.
  CUDA full rows: native Flash SDPA (BF16/FP16); cached single-token: FlexAttention valid prefix.
  K/V storage per track: two [B,T_cache,n_kv,96] BF16 tensors, written before the causal read.


G. EVERY FFN: ONE SHARED + TOP-k_e-OF-n ROUTED SwiGLU EXPERTS
============================================================================================
  k_e = selected routed count; n = routed bank size; h_ff = per-expert width
  Screen/bridge/flagship/extension: (k_e,n) = (3,15)/(5,23)/(7,31)/(11,47)
  h_ff=832 throughout the presets.
  z [B,T,D] -----------------------+------------------------------------+
       |                          |                                     |
       v                          v                                    v
  shared expert S          W_router: D->n                         E_0 ... E_(n-1)
  (always active)                 |                          (execute selected rows)
       |                   sigmoid scores s_j                           |
       |                      FP32 |                                    |
       |              +------------+-------------+                      |
       |              |                          |                      |
       |      topk_e(s_j + b_j)          gather original s_j             |
       |              |                          |                      |
       |         selected J ------------> w_j = s_j/sum_(i in J)s_i     |
       |              |                          |                      |
       |              +------ dispatch ----------|--------------------->|
       |                                         |                      |
       |                                         +--> weighted sum <----+
       |                                                     |
       |                                                   * k_e
       |                                                     |
       +--------------------------------------------------->(+) --> /sqrt(k_e+1) --> output

  Each of n+1 independent experts:
       z --> W_gate: D->h_ff --> SiLU --+
       |                               (*) --> W_down: h_ff->D --> expert output
       +---> W_up:   D->h_ff -----------+
  Gate/up matrices are packed; each expert retains its own parameters and optimizer state.
  k_e+1 active experts per token; no capacity limit or token dropping.
  Gradients flow through selected scores and experts; selection indices are discrete.

  Actual selections --> integer counts summed across microbatches/passes/core visits
                    --> once per optimizer step: b_j += .001*sign(sum_i C_i - n*C_j)
  Unbiased topk_e(s) + normalized all-expert scores --> sequence balance loss --> [I]
  One b[n] buffer per PHYSICAL FFN, shared across its tied visits; fixed in evaluation.


H. AUXILIARY SECOND-TOKEN PREDICTION: ONE BLOCK, ONCE PER PASS
============================================================================================
  p_t from payload writer [A]                      x_(t+1) --> same Emb.weight
             |                                                   |
       MTP payload RMSNorm                                MTP embedding RMSNorm
             |                                                   |
             +----------------------> concat <-------------------+
                                        |
                                   [B,T-1,2D]
                                        |
                                  M: 2D->D
                                        |
                                        u -------------------------------+
                                        |                                |
                                 own attn RMSNorm                        |
                                        |                                |
                                 own PKDA mixer [E]                      |
                                        |                                |
                              * 1/sqrt(2L) ---------------------------->(+)
                                                                         |
                                        +--------------------------------+
                                        |                                |
                                  own FFN RMSNorm                        |
                                        |                                |
                                 own expert FFN [G]                      |
                                        |                                |
                              * 1/sqrt(2L) ---------------------------->(+)
                                                                         |
                                   shared final_norm <--------------------+
                                        |
                                    * 1536/D
                                        |
                                  @ Emb.weight.T
                                        |
                              MTP logits [B,T-1,V] --> target x_(t+2)

  Positions: payload[:,:-1] + Emb(tokens[:,1:-1]) predict tokens[:,2:].
  Independent entry norms, PKDA, FFN, and fresh zero mixer state for every row/pass.
  No MHDB, trunk-cache input, or tied-depth loop; auxiliary states do not feed the trunk.
  Both inputs remain differentiable. MTP trains the payload writer even on the final pass.
  Generation uses NTP only: no auxiliary execution or auxiliary cache allocation.


I. THE THREE AXES: TOKEN HISTORY t, FEEDBACK PASS k, AND TIED ITERATION i
============================================================================================
  JACOBI TRAINING (each box is the WHOLE column stack [A], evaluated over a causal row):

    stored tokens [B,T+1] --> inputs x_0 ... x_(T-1); ordinary targets x_1 ... x_T
                |
                +--> embeddings e ------------------------------------------------+
                            |                                                     |
                   pass k=1: seed=e --> [P -> core x r -> Coda] --> h^1, p^1        |
                                                                         |        |
                                                    add keyed jitter ----+        |
                                                                         |        |
                                                    shift right by one position   |
                                                                         |        |
                                                              FBT gate <----------+
                                                                         |
                    plain-prefix e / fused-suffix selection <-------------+
                            |
                   pass k=2: seed   --> [P -> core x r -> Coda] --> h^2, p^2
                                                                         |
                                         repeat feedback construction for k=3..K

    Feedback at (k,t) consumes p^(k-1)_(t-1) + jitter^(k-1)_(t-1), never p^k_(t-1).
    Per-row/per-feedback-pass prefix length is drawn in 1..T-1; position 0 stays plain.
    Every pass restarts mixer state, core deltas, and the column source bank.
    Only the undetached payload crosses passes; all passes/iterations backpropagate.
    One keyed r draw per optimizer step, shared by all passes and microbatches;
    independent of feedback draws. Default uncapped mean 4, cap 8; eval/decode fix r=4.

  SEQUENTIAL FEEDBACK DECODE:

    (x_(t-1), p_(t-2)) --> [column t-1] --> p_(t-1) --+
                                                    |
                                     x_t --> Emb --> FBT --> [column t] --> p_t
                                                                 |
                                                           NTP predicts x_(t+1)

    A prefill pass starts caches empty; final prefill caches continue into generation.
    A core layer's SAME weights have DIFFERENT token-history tracks:

                  token t-1                     token t                     token t+1
       i=1:    cache(layer,1) ----------------> cache(layer,1) -----------> cache(layer,1)
       i=2:    cache(layer,2) ----------------> cache(layer,2) -----------> cache(layer,2)
        ...                                 ...
       i=r:    cache(layer,r) ----------------> cache(layer,r) -----------> cache(layer,r)

    Within a column h flows down iterations; token history flows right at the same iteration.
    Prelude/coda have one track per layer. All tracks share position, advanced once per token.
    Payload is a separate D-wide channel; it does not replace PKDA or GQA cache state.

  SUPERVISION AND NUMERICS:

    every h^k --> tied NTP head --> mean CE_ntp,k + z_coef*mean(logZ_ntp,k^2) --+
    every p^k --> MTP [H]       --> mean CE_mtp,k + z_coef*mean(logZ_mtp,k^2) --|--+
                                                                                |  |
       combine(ell) = ell_1 if K=1, else ell_1 + mean(ell_2,...,ell_K)          |  |
                                                                                v  v
       loss = combine(NTP) + mtp_weight*combine(MTP) + 1e-4*expert_balance
                             default 0.3

    Each head averages its own positions: T for NTP, T-1 for MTP; all are supervised.
    Expert balance averages executed trunk layers plus one MTP bank, then averages passes.
    CUDA activations/payloads are BF16; PKDA S/A, parameters, optimizer state, and accumulated
    gradients are FP32. The two heads use the tied classifier through a refreshed BF16 shadow.
    CUDA loss uses cut cross-entropy: the [B,T,V] and [B,T-1,V] logits above are conceptual.
    Gradients traverse FBT, MHDB values/queries, all tied visits, and both MTP inputs.
    Before NorMuonH + NAdam updates: global FP32 gradient clipping to norm 10.
```

Pass 1 and Standard decoding seed the column from the token embedding. The
readout applies `final_norm(h_top) * 1536 / D` before the tied classifier.
PKDA and GQA caches carry additional state along tokens; the payload carries
feedback between columns and between Jacobi passes.

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
decode uses its recurrent operator. CPU/MPS use the literal recurrence.
Convolution, SiLU, and Q/K normalization stay FP32 until the final activation
cast. Decode retains the FP32 matrix and diagonal states plus three BF16
projected convolution histories of length 3. Each Jacobi pass restarts those
states; only the payload crosses passes. The final prefill's caches then
advance normally during generation.

### Gated global GQA

The fourth layer of each cell uses causal NoPE GQA, head width 96:

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
is no capacity limit or token dropping. Multiplying by `k` gives unit
selected coefficients at equal scores; division by `sqrt(k+1)` matches the
variance of `k+1` independent equal-variance expert outputs to one expert
output at initialization. This does not guarantee learned variance.

The active dense-equivalent width is derived as `H=(k+1)h`: expert matrices
perform `3DH` multiply-accumulates per token and store `3D(n+1)h` parameters.
The presets store four times the active expert matrix parameters. The router
adds `nD` parameters per bank. The shared expert accounts for `1/(k+1)` of
active expert width: 25%, 16.7%, 12.5%, and 8.3% across the presets. Fixed expert
intermediate width does not fix each expert's parameter count as `D` grows.
Memory and dispatch overhead still matter. The design follows
[DeepSeekMoE](https://arxiv.org/abs/2401.06066).
Hard selection can amplify small full-row/cached numerical differences when
near-tied experts exchange rank; a compact execution smoke does not establish
unrestricted long-horizon decode agreement.

### Load balancing

Each physical bank's `expert_bias[n]` starts at zero and stays outside the
optimizer groups. Training sums actual integer assignment counts `C_j` over
all microbatches, passes, and invocations, then updates once after the step:

```text
b_j += 0.001 * sign(sum_i C_i - n * C_j)
```

Evaluation and recomputation leave it fixed. Counts are returned values, so
activation checkpointing cannot double-count. This follows
[auxiliary-loss-free balancing](https://arxiv.org/abs/2408.15664).

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

Selection-only bias, sigmoid scores, and sequence regularization follow
[DeepSeek-V3](https://arxiv.org/abs/2412.19437). Layer/pass averaging keeps the
coefficient independent of depth and pass count. The auxiliary bank contributes
once per pass, with its own `T-1` positions. Its within-sequence gradient
coupling does not change causal forward activations. Cross-entropy excludes it.

`ColumnOutput.expert_aux_loss` holds the trunk layer mean and `expert_counts`
holds transient `[L,n]` counts summed over physical-bank invocations.
`LossOutput.expert_aux_loss` includes the auxiliary bank using the averaging
above; its `[L+1,n]` counts place that bank last. Bias updates include every
bank once per optimizer step. With `want_weights=True`, column `expert_weights`
maps each trunk invocation to `[B,T,n]` sparse normalized weights, distinct
from MHDB's source weights. `forward_mtp` can return the auxiliary block's
weights over its `T-1` positions.

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

At a feedback position, token embedding `e_t` controls the gate on incoming
payload `p_(t-1)`:

```text
seed_t = entry_norm(W_U p_(t-1) * sigmoid(W_G gate_norm(e_t)))
```

There is no additive embedding bypass at that position. Plain-prefix, pass-1,
and Standard-decoding positions use `e_t`. The actual seed is both residual
origin and MHDB source. Every condition has a dedicated payload router that
reads its null, seed, and every completed cell delta:

```text
r_payload = route(null, seed, Delta_0 .. Delta_(C-1))
p_t = payload_norm(h_top + r_payload)
```

The direct `h_top` term remains alongside routed enrichment. Initially the
uniform enrichment is `h_top/(C+2)`, making the normalized payload equal to a
normalized top state up to epsilon. Payload routing is not conditioned on
the next token; token control happens at the next column's entry gate.
Every supervised pass constructs this payload for MTP, including single-pass
batches, `l` without `f`, and the final feedback pass. The `f` letter controls
consumption by the next column. Generation can skip payload construction when
no feedback or auxiliary read needs it.

Jacobi training adds keyed jitter to the previous pass's undetached payload,
shifts it right, and fuses a suffix after a per-row plain prefix. Mixer states
restart inside each pass. Sequential generation instead retains the preceding
payload and advances all mixer caches once per new token.

## Letter l: the tied-depth loop

Whole cells and at least three cells are required. Of `C` unique cells, the
first is the prelude, the last the coda, and the `c=C-2` middle cells run in
order as one tied core, repeated `r` times. It adds no parameters. At `r=1`,
`fl` equals `f` in values, routes, losses, and gradients.

Every core cell keeps one delta accumulated across iterations. A visit reads
the seed, prelude delta, every other core cell's contribution so far, and its
own accumulated contribution as the partial. Its effective origin advances
by whatever other cells add between visits. Contributions not yet produced
on the first iteration are absent. The coda and payload read one final delta
per core cell, retaining the flat column's source identities:

```text
h_top = seed + Delta_P + sum_j Delta_R_j + Delta_C
```

With one core cell, its origin remains the prelude output and its partial is
`h - core_entry` across all iterations. No additional residual normalization
or branch scaling is introduced. Fixed NorMuonH radii alone do not establish
contraction because norms, router values, gates, and controls are unconstrained.

### Depth, caches, and training

The keyed depth stream draws once per optimizer step, independently of the
feedback draws, and every pass/microbatch shares the result:

```text
tau ~ Normal(log(r_mean - 1) - sigma^2 / 2, sigma), sigma = 1/2
r = min(1 + Poisson(exp(tau)), r_max)
```

The uncapped mean is `r_mean`; defaults are 4 and cap 8. Evaluation/decode
hold `r = r_mean` fixed for the request. Each iteration has its own mixer
cache track and reads earlier token positions' writes at the same iteration.
Every current prefill pass starts those tracks from zero. Cache position is
shared across tracks. A pass executes `2+c*r` cells.

All passes and iterations remain differentiable. Readout occurs after the
coda, and auxiliary prediction runs once per pass. Above the raw activation
budget, each cell retains its final GQA block and checkpoints its preceding
three PKDA blocks. Checkpoint wrappers remain outside compiled blocks.
CUDA captures one training graph per reachable `(pass count,r)` pair and
no-grad evaluation graphs at the fixed depth.

`ColumnOutput` exposes `iterations`, `core_entry`, and `core_state`; route
keys include iteration, such as `L4i2.attn`. `depth_trace` sweeps `1..r_max`,
reading out after the coda at each depth and reporting core-update norms.
The payload self-composition trace holds depth fixed. Paired `f`/`fl` runs
share initialization, rows, schedule, and feedback draws. Equal steps match
data; compute comparisons need cell-tokens and auxiliary work or device time.

## Two-token prediction

One auxiliary prediction depth follows
[DeepSeek-V3 section 2.2](https://arxiv.org/html/2412.19437v2#S2.SS2):

```text
u_t = M concat(RMSNorm_p(p_t), RMSNorm_e(Emb(x_(t+1))))
v = PKDAExpertBlock(u)
logits_mtp,t = (final_norm(v_t) * 1536 / D) @ Emb.weight.T
```

`M` is bias-free `2D -> D`. The two entry norms are independent. The auxiliary
block has its own PKDA mixer and one shared plus top-`k`-of-`n` routed
SwiGLU experts, using the trunk's residual width, expert intermediate width,
expert counts, output normalization, and `1/sqrt(2L)` branch scale. It shares the final norm and embedding/readout.
It has no MHDB read or tied loop. Its independent matrix, diagonal, and
convolution states start from zero for each row and pass.

For a stored row of `T+1` tokens, ordinary inputs/targets are
`tokens[:,:-1]` / `tokens[:,1:]`. Auxiliary inputs are `payload[:,:-1]` and
`Emb(tokens[:,1:-1])`, with targets `tokens[:,2:]`: `T-1` fully supervised
positions, no padding or ignored labels. Both inputs remain differentiable,
so the auxiliary objective trains the payload router, payload normalization,
and trunk as well as the shared token embedding. The causal PKDA recurrence
never sees the second-token target, and auxiliary activations never feed into
the column or payload. `DeltaModel.forward_mtp` exposes the block output,
including balancing loss, assignment counts, and optional expert weights.

Each head averages over its own valid positions. For either head,
`combine(ell)=ell_1` with one pass, otherwise
`ell_1 + mean(ell_2,...,ell_K)`:

```text
loss = combine(CE_ntp) + z_coef * combine(z_ntp)
     + mtp_weight * (combine(CE_mtp) + z_coef * combine(z_mtp))
     + 1e-4 * expert_balance
```

The cooldown z-loss is the mean squared log-partition. `--mtp-weight` is a
finite nonnegative constant, default 0.3. `ntp` and `mtp` report cross-entropies
before coefficients/z-loss, `pass1` is ordinary first-pass CE, and `loss` is
the optimized objective. `val_mtp` and feedback's `val_mtp_fused` remain
separate from ordinary validation. Generation does not run the auxiliary
block or allocate its cache; speculative decoding is not implemented.

## Precision and initialization

NorMuonH matrices initialize from `Normal(0,1/sqrt(fan_in))`. The shared
`BASE_NORMAL_INIT_STD=0.02` sets NAdam matrix standard deviations. GGQA gates,
expert routers, the FBT gate, and PKDA's packed control projection multiply
it by `sqrt(1536/D)`. Embeddings and PKDA fixed-head-width expansions use
0.02 directly. This preserves initial control-logit variance across widths.
`MUP_BASE_DIM` stays at the flagship width 1,536: extension uses a width ratio
of `1536/2304 = 2/3` for NAdam rates and readout scaling, and its square root
for the specified initialization scales. The reference does not depend on
the largest configured preset. Expert optimizer rates have their own factors
below; they do not change initialization or forward computation.
Depthwise convolutions retain Kaiming-uniform initialization. RMSNorm scales
start at one; MHDB queries and nulls start at zero.

Shared parameters pair byte-identically across conditions for a given seed.
Experts are constructed directly as part of the common model initialization.
Attention gates, feedback matrices, and the auxiliary module use separate
deterministic streams. The FBT stream does not advance shared draws. Feedback/depth recipe draws are separately keyed.

Parameters, optimizer state, and accumulated gradients are FP32. CUDA residuals,
routed values, payloads, and mixer caches use BF16, except PKDA matrix/diagonal
boundaries. CCE reads an address-stable BF16 classifier shadow refreshed after
each optimizer step; it is runtime state, excluded from snapshots.

Packed projection backpropagation writes into persistent FP32 sinks. PKDA
Q/K/V row views share an allocation, as do dense QKV and gate gradients;
each parameter is clipped/updated once. Head calls first accumulate classifier
gradients in a BF16 buffer, flushed into the FP32 sink at the configured
`--head-flush-every` cadence and before the optimizer update. Each pass makes
two head calls. A captured microbatch exceeding the cadence uses the FP32
sink per call; cadence 1 therefore gives per-call precision throughout.

## NorMuonH and NAdam

Five disjoint parameter groups across two optimizers share the
warmup-stable-cooldown multiplier with no weight decay:

| Group | Parameters | Peak learning rate |
|---|---|---|
| `normuonh` | Ordinary NorMuonH matrices outside experts | `lr_normuonh` |
| `normuonh_expert_in` | Shared and routed expert gate/up matrices | `lr_normuonh * sqrt(1536/D)` |
| `normuonh_expert_out` | Shared and routed expert down matrices | `lr_normuonh * sqrt(8/(k+1))` |
| `nadam` | Base NAdam parameters | `lr_nadam` |
| `nadam_width` | NAdam matrices with residual-width fan-in | `lr_nadam * 1536/D` |

Here `k` is the selected routed expert count, so `k+1` includes the shared
expert. Both expert groups include every trunk bank and the auxiliary MTP
bank. Input and output factors are computed separately: the former follows
residual width and the latter active expert count. They coincide at the
presets but can differ for custom geometry. The references are the flagship
width 1,536 and its eight active experts.

### NorMuonH matrices

NorMuonH owns ordinary 2D hidden matrices: token-mixer projections, expert and
auxiliary FFNs, the FBT value projection, and MTP concatenation. Every matrix
keeps its initial FP32 Frobenius radius `R`, stored in the checkpoint:

```text
M_t = .95 M_(t-1) + .05 G_t
N_t = .05 G_t + .95 M_t
U = five_Newton_Schulz_steps(N_t)
U = row_second_moment_normalize(U, beta=.95, eps=1e-8)
eta_t = schedule_multiplier(t) * group_peak_lr
W_next = R * Normalize_F(W - eta_t * R * Normalize_F(U))
```

The default base relative rate is `6e-3`; the expert groups apply the factors
above. A Frobenius-relative step alone does not establish equal functional
updates across widths or expert counts. These factors are an implemented
scaling candidate; full-model hyperparameter transfer remains unestablished.
[Scaling](scaling.md#expert-learning-rates) lists the preset rates.
CUDA compiles packing and update arithmetic by shape bucket within each
group. All results materialize before state and parameter writebacks outside
the compiled boundary, preserving reads of the previous momentum.
Packing is bounded to 33,554,432 matrix elements (128 MiB per FP32 tensor),
except larger individual matrices remain whole. State stays per-parameter.

### NAdam parameters

The width-scaled group owns matrices with fan-in `D`: GGQA and FBT gates,
expert routers, and PKDA packed controls. It uses `lr_nadam * 1536/D`; the
base group uses `lr_nadam`, default `3e-4`. The base group owns the tied
embedding/readout, fixed-head-width PKDA expansions, and every non-matrix
parameter. The readout also multiplies by `1536/D`, while embedding lookup
keeps its base rate.

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

CUDA NAdam and global FP32 clipping run outside forward/backward graphs.
The scalar step and momentum-product state remain on CPU; moments and
parameters remain FP32 on-device. Before both optimizers, the complete
accumulated gradient is clipped to L2 norm 10. A non-finite norm stops the
run. Telemetry reports the pre-clip norm.

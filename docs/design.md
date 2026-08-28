# Design ledger

Numbered decisions (D*) are settled unless explicitly reopened; later entries
supersede earlier ones. Origin: the 2026-08-27 scoping session (a9 + Claude).
Confidence marks: [paper] = claim from a source, [synthesis] = our reasoning
on top of sources, [speculation] = prior/bet, on the record to be tested.

## Question

Autoregressive transformers under-route information along two axes of the
(position t, layer l) compute lattice. **Delta Attention Residuals** (DAR,
arXiv:2605.18855) widens the *vertical* axis: each sublayer routes additively
over the RMS-normed *deltas* of its own column via zero-init softmax attention
— cumulative sources collapse to near-uniform routing at depth, deltas stay
sharp. **Full-bandwidth transformers** (FBT, arXiv:2608.08888) widen the
*horizontal* axis: the previous column's top-layer state is fed back to layer
0, fused with the sampled token embedding through an asymmetric GLU, so
non-verbalized state re-enters the stack with a renewed depth budget.

Primary question: **are the depth-axis and time-axis widenings complementary
or redundant?** Secondary: does the feedback *payload* benefit from DAR-style
delta routing (FBT hardcodes the top state; the FBT paper explicitly leaves
"which form of past hidden state injection is best" open for lack of
resources) — that is the A1-vs-A2 comparison.

Priors on record [speculation]: a9 expects superadditive; Claude hedges toward
additive-to-mildly-sub (the channels may partially substitute for the same
limited per-column capacity). Either sign is a finding. If superadditivity is
real it should concentrate in the **Soft-decoding delta** (Standard→Soft gap
on the same weights), since DAR keeps early-layer information alive in the
payload.

## Architecture

One decode step at position t (trial config: A2-hybrid, per-sublayer):

```python
e = embed(tok)
u = rmsnorm(glu(p_prev, e))          # FBT gate: W_U p_prev * sigmoid(W_G e)
srcs = [u, p_prev, e]                # standing sources (see D4, D5)
h = u
for l in layers:
    h = h + route(srcs, q_attn[l])   # DAR: + sum softmax(q . rmsnorm(v)) v
    a = attn(norm(h)); h = h + a; srcs.append(a)
    h = h + route(srcs, q_mlp[l])
    m = mlp(norm(h));  h = h + m; srcs.append(m)
p_prev = payload(srcs)               # A1: rmsnorm(h_top)
                                     # A2: rmsnorm(sum softmax(q_p . rmsnorm(v)) v)
tok = sample(lm_head(h))
```

Training is FBT's Jacobi multi-pass loop unchanged: pass k shifts pass k-1's
payloads one position right, fuses, re-runs the stack (DAR active), prefix
mixin, jitter noise on the payload, NTP loss on every pass, no detach. Pass 1
of a combined arm is exactly the DAR-only model; the combined model's
"Standard decoding" mode is therefore a DAR transformer whose training carried
an extra objective. [synthesis]

## Decisions

- **D1 — Name and provenance.** `delta-feedback-experiment`. Deliberate
  greenfield: predecessors (stateful-thought-experiment/CoST,
  chain-of-dots-experiment) were cleared by a9; their design decisions are
  *not* inherited. The platform verdict (custom architecture excludes managed
  fine-tuning services; Prime Intellect marketplace pods for large runs, jobe
  for trials) carries over from the 2026-07-11 investigation.

- **D2 — Payload arms.** A1: payload = top-layer state (FBT verbatim, the
  combination baseline). A2 "delta feedback": payload = softmax-routed
  combination of the column's deltas under a dedicated static learned query.
  A3 (all prev-column block deltas as separate sources at every layer) is
  **dropped** for cost; revisit only if A2 wins decisively and the routing
  weights suggest unmet demand for finer cross-column access.

- **D3 — A2 payload router is static-query.** Routing computed at the end of
  column t-1 with a learned query, not conditioned on the next token (the GLU
  already gates the payload by e elementwise; token-conditioned selection is
  A3-creep). Ledger note: token-conditioned routing is affordable at decode
  time (deltas are per-sequence, not per-position) — a variant, not v1.

- **D4 — Entry is hybrid.** Mandatory GLU fusion at layer 0 (FBT's shortcut
  argument: on passes k>1 the loss can be minimized by imitating pass 1 and
  ignoring state unless reading is forced) **plus** the raw payload as one
  optional zero-init routed source at every sublayer (ungated access; its
  routing weight per layer is a direct observable of cross-column demand).

- **D5 — Raw token embedding as a standing source.** DAR's strongest learned
  pattern is deep layers re-injecting the token embedding [paper]; FBT's gate
  destroys additive access to e (the token survives only as a multiplicative
  pattern) [paper]. Without this source the combined model loses the shortcut
  DAR found most valuable — so `e` rides as an optional routed source at
  every sublayer. [synthesis]

- **D6 — Payload RMSNorm.** The payload is RMS-normed before the GLU in both
  arms. Feeds FBT's O(1)-state-norm stability requirement and scale-matches
  A1 and A2 at init: A2's zero-init router yields the uniform delta average,
  which telescopes to (h_top - u)/N — normalized, approximately A1-minus-
  input. The arms diverge only as routing sharpens, so differences are
  attributable to learned selection. [synthesis]

- **D7 — Delta granularity.** Per-sublayer sources (2L) wherever affordable:
  all trial-scale runs. Flagship coarsens to Delta-Block-style block deltas —
  supported by DAR's 533M ablation (PPL flat, 31.18-31.27, across B=2-24)
  [paper]. The A2 payload router routes over whichever source set is active.
  Memory pressure to respect: 2L sources x k passes x no-detach; gradient
  checkpointing at trials, blocks at flagship.

- **D8 — Schedule.** DAR routing active from step 0 (zero-init is exactly
  identity; proven from-scratch [paper]). Feedback passes introduced late,
  default mixture 75% one-pass / 22% two-pass / 3% three-pass (the 3%
  three-pass fraction is what makes the feedback map a contraction [paper]).
  The contraction diagnostic — iterate fused prefill passes, watch
  ||h(k) - h(k-1)|| and val loss — is a **standing monitor** during and after
  training. Adaptive pass-mixing triggered by that monitor (the FBT
  limitations section asks for exactly this) is in scope if cheap, aspiration
  otherwise.

- **D9 — Arms and comparisons.** Trials: {vanilla, DAR, FBT, A1, A2}, all
  under one recipe (D10), matched tokens *and* reported at matched
  token-equivalent compute (FBT accounting: an n-pass batch costs n).
  Interaction := (combined - vanilla) - [(DAR - vanilla) + (FBT - vanilla)],
  evaluated per decode mode (Standard / Soft / Fused). Flagship: the better
  of A1/A2 only — one big run, not a factorial at scale.

- **D10 — One recipe everywhere; FBT's is binding.** Match the papers as
  closely as possible; where they conflict, FBT wins (the feedback channel
  needs its stability kit; the flagship matches their scale): NorMuon for
  matrices (lr 1e-2, wd 0.01) + Adam for vectors (lr 5e-4), WSD schedule
  (200 warmup, 25% cooldown), z-loss 1e-5 + AdamC-style wd decay in cooldown,
  jitter sigma=0.02 on the carried state, depth scaling for O(1) top-state
  norm, tied embed/unembed, prefix mixin. DAR module conventions nest inside
  (zero-init routing queries, RMSNorm on routing keys, raw values). WSD also
  enables extending token budgets for matched-compute baselines without
  re-warming.

- **D11 — Tokenizer and trunks.** Qwen3 tokenizer (~151k, tied) everywhere —
  one tokenizer across our runs beats matching FBT's phi-4 100k. Trials
  reproduce DAR's exact 220M config (d=768, L=12; their "220M" is precisely
  this with the Qwen3 vocab) [paper]. Flagship takes FBT's trunk (d=1536,
  L=24, GQA 16q/8kv with headwise gates, QK-norm, SiLU GLU 6656, RoPE, ctx
  8192, 2048-token SWA on 5/6 layers with full attention every 6th) with
  vocab swapped to Qwen3's → ~1.08B params.

- **D12 — Data.** FineWeb-Edu throughout (DAR parity; FBT's Phi-4 mixture is
  unavailable). Held-out FineWeb-Edu val split shared across all arms.

- **D13 — Trial budget.** jobe (1x4090): 5 arms x 2 seeds x ~1B tokens at the
  220M config, paired data order across arms (same batches, same order, so
  loss curves difference cleanly). Rough cost: 4-6 h/run with the ~1.5x
  DAR+multipass overhead → ~2.5-3 days total. Contingency mid-rung: ~300M
  params, finalists only, ~1 day/run, before any flagship spend.

- **D14 — Flagship.** ~1.08B params, **400B tokens** (matching FBT's largest
  run; their decode-time behavior lives at 100-400B). Prime Intellect
  marketplace pods. Cost on record: ~2.4e21 FLOPs → roughly 2,700 H100-hours
  with overhead ≈ **$6-7k** at July-2026 H100 rates (~$2.43/GPU-hr), ~$3-4k
  on H200 spot with checkpoint-tolerant preemption; ~2 weeks on 8xH100.
  Optional FBT-style follow-through: long-context extension (12B tokens,
  8K→32K) + instruction tune (6B tokens), three-pass throughout, if the base
  result justifies it.

- **D15 — Flagship controls.** No matched 1B vanilla (cost). Instead: (a)
  decode-mode ablations on the same weights — Standard/Soft/Fused isolate the
  channel's inference contribution; (b) pass-1 loss tracked throughout
  training as the "as ordinary transformer" mode; (c) one or two small
  matched-data vanilla rungs (<=300M) anchoring a scaling extrapolation.
  Published 1B-class models (TinyLlama, Llama-3.2-1B, Qwen3-1.7B, SmolLM2)
  appear as *context rows only* — 2-36T tokens of unknowable data mixture
  makes them incomparable as controls.

- **D16 — Promotion gate (pre-registered).** Promote to the flagship only if
  the better of A1/A2 beats **both** parents on val loss at matched
  token-equivalent compute with a consistent sign across seeds, **and** the
  contraction diagnostic is clean past 30 self-compositions. If FBT's main
  effect is null at 220M but the combined arms are healthy, that reads as a
  formation-conditions result (cf. acot at 135M) — the DAR arm doubles as the
  positive control that the harness detects effects of this size.

- **D17 — Phase-2 extension: pause tokens.** Contingent on the factorial.
  Add pause-training (Goyal et al., arXiv:2310.02226) to the winning model.
  Mechanism bet [synthesis]: in a vanilla transformer pauses add parallel
  width only (constant input embedding; columns chain only via attention);
  under latent feedback, consecutive pauses chain through the payload at full
  depth — k pauses ≈ a k-step loop transformer unrolled horizontally, paid in
  KV entries. Pause supplies decode-time serial depth; FBT supplies the carry
  channel pauses lack. Complementary dials: fused prefill scales prompt-side
  compute, pauses scale decode-side. Goyal et al.'s core finding — pauses
  must be pretrained in — matches our from-scratch setting [paper]. Parallel
  to recirculated-dot-experiment is **loose and conceptual** (no shared
  protocol obligation).

- **D18 — Interp layer.** First-class, not an afterthought: (a) routing
  weights are direct observables — per-layer weight on the raw-payload source
  (D4) measures cross-column demand; per-layer weight on `e` (D5) tracks
  whether DAR's embedding-prominence survives or migrates to the fused input;
  (b) FBT Appendix-F state-tracking synthetics (completion tracking, delayed
  memory, multi-register latest-write) reimplemented with linear probes
  across depth — what rides the payload, and does A2's routed payload carry
  different state than A1's top state; (c) contraction diagnostics as
  standing evals.

## Risks

- FBT is unproven below 1B params (their smallest run); the channel may not
  form at 220M — mitigations in D16. DAR is proven at exactly our trial scale
  and anchors the harness.
- Both parents are single-group arXiv v1s; DAR's AttnRes baseline is their
  own reimplementation. Effect sizes are a few percent PPL — seeds, paired
  data order, and matched-compute accounting are load-bearing, not hygiene.
- Memory: no-detach multi-pass x per-sublayer sources compounds; if trials
  OOM at k=3 with checkpointing, coarsen sources before detaching (detaching
  changes the objective — FBT's data-efficiency story runs through gradients
  into earlier passes' states [paper]).
- DAR has public code (github.com/wdlctc/delta-attention-residuals-code) to
  borrow; FBT has none — the multi-pass loop is small but the stability kit
  has several interacting pieces; reimplement against their listings and
  verify the contraction signature before trusting any arm comparison.

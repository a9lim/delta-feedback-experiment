# Design

This file is the authoritative architecture and operating contract, and
defines the current experiment only. Evidence lives in
[findings.md](findings.md); disposable working notes in
[journal.md](journal.md); source roles in
[../references/refs.yaml](../references/refs.yaml). Confidence marks:
[paper] = claim from a parent paper, [synthesis] = our reasoning on top,
[speculation] = pre-registered bet.

## Research question

Autoregressive transformers under-route information along two axes of the
(position t, layer l) compute lattice. **Delta Attention Residuals** (DAR,
arXiv:2605.18855) widens the *vertical* axis: each sublayer routes additively
over the RMS-normed *deltas* of its own column via zero-init softmax
attention — cumulative sources collapse to near-uniform routing at depth,
deltas stay sharp [paper]. **Full-bandwidth transformers** (FBT,
arXiv:2608.08888) widen the *horizontal* axis: the previous column's
top-layer state is fed back to layer 0, fused with the sampled token
embedding through an asymmetric GLU, so non-verbalized state re-enters the
stack with a renewed depth budget [paper].

- **Primary:** pretrained together from scratch, are the depth-axis and
  time-axis widenings complementary or redundant?
- **Secondary:** should the feedback *payload* be the raw top-layer state or
  a routed combination of the column's deltas? FBT hardcodes the top state
  and explicitly leaves the injection-form question open [paper]; DAR's
  thesis is that deltas are the routable decomposition of a column.

Pre-registered predictions [speculation]: a9 expects superadditive gains;
Claude hedges toward additive-to-mildly-sub (the channels may partially
substitute for the same limited per-column capacity). Either sign is a
finding. If superadditivity is real it should concentrate in the
Standard→Soft decoding gap on the same weights, since depth routing keeps
early-layer information alive in the payload [synthesis].

## Architecture

One decode step at position t (trial configuration: per-sublayer sources,
A2 payload):

```python
e = embed(tok)
u = rmsnorm(glu(p_prev, e))          # FBT gate: W_U p_prev * sigmoid(W_G e)
srcs = [u, p_prev, e]                # standing sources
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

**Depth routing (DAR side).** Before every attention and MLP sublayer, a
zero-init learned query per site routes softmax attention over the source
list and adds the convex combination to the residual stream. Keys are
RMS-normed; values are raw. Zero-init makes routing exactly the identity at
step 0, so it is active from the start of training [paper]. Sources are
per-sublayer deltas (2L per column) wherever affordable; the flagship
coarsens to Delta-Block-style block deltas, supported by DAR's 533M
ablation showing block-size insensitivity (PPL 31.18–31.27 across B=2–24)
[paper].

**Feedback channel (FBT side).** The payload from column t−1 enters column t
two ways:

1. *Mandatory* — fused into the layer-0 input through FBT's asymmetric GLU
   (payload on the value path, token embedding as the gate). This closes the
   shortcut in which multi-pass losses are minimized by imitating the
   no-feedback pass and ignoring state [paper].
2. *Optional* — the raw payload rides as one standing zero-init routed
   source at every sublayer, giving ungated access; its per-layer routing
   weight is a direct observable of cross-column demand [synthesis].

**Payload variants.** A1: the RMS-normed top-layer state (FBT verbatim; the
combination baseline). A2 "delta feedback": a softmax-routed combination of
the column's delta sources under a dedicated static learned query, RMS-
normed. The payload router is *not* conditioned on the next token — the GLU
already gates the payload elementwise by the token. With payload RMSNorm,
A2's zero-init router yields the uniform delta average, which telescopes to
approximately A1-minus-input: the arms are near-identical at init and
diverge only as routing sharpens, so differences are attributable to
learned selection [synthesis].

**Standing sources.** The source list opens with the fused input `u`, the
raw payload `p_prev`, and the raw token embedding `e`. The `e` source
exists because DAR's strongest learned pattern is deep layers re-injecting
the token embedding [paper], while the FBT gate destroys additive access to
it (the token survives only as a multiplicative pattern) [paper]; the
optional source restores that shortcut without weakening the mandatory gate
[synthesis].

**Scope boundaries.** Feeding multiple prev-column deltas as separate
per-layer sources (the full-lattice variant), token-conditioned payload
routing, and any cross-column access deeper than one step are *not* part of
this design; each defines a different experiment. Revisit only with routing-
weight evidence of unmet cross-column demand.

## Training

**Multi-pass regime.** FBT's Jacobi-style parallel training, unchanged:
pass k shifts pass k−1's payloads one position right, fuses, and re-runs the
stack (depth routing active) in parallel over positions; k passes train a
(k−1)-step feedback horizon. NTP loss on every pass, no detach — gradients
from later passes supervise earlier passes' states, which is part of FBT's
data-efficiency mechanism [paper]. Prefix mixin (random plain-embedding
prefix per pass) matches the prompt-then-generate structure of inference.
Pass 1 of a combined arm is exactly the DAR-only model, so the combined
model's Standard-decoding mode is a DAR transformer trained with an extra
objective [synthesis].

**Schedule.** Depth routing from step 0. Feedback passes late, default
mixture 75% one-pass / 22% two-pass / 3% three-pass — the small three-pass
fraction is what makes the learned feedback map a contraction rather than a
divergence under self-composition [paper]. The contraction diagnostic
(iterate fused prefill passes; watch ||h(k) − h(k−1)|| and val loss) is a
standing monitor during and after training. Adaptive pass-mixing triggered
by that monitor is in scope if cheap.

**One recipe everywhere; FBT's is binding.** Every arm, including vanilla,
trains under FBT's published recipe: NorMuon for matrices (lr 1e-2, wd
0.01) + Adam for vectors (lr 5e-4), WSD schedule (200 warmup, 25% cooldown),
z-loss 1e-5 and AdamC-style weight-decay decay in cooldown, jitter
sigma=0.02 on the carried state, depth scaling for O(1) top-state norm,
tied embed/unembed. DAR module conventions (zero-init queries, RMS-normed
keys, raw values) nest inside. Divergences from the parent papers are
recorded here when made. WSD permits extending token budgets for matched-
compute baselines without re-warming.

**Memory.** No-detach multi-pass times per-sublayer sources compounds
activation memory. Gradient checkpointing at trial scale; block deltas at
flagship scale. If trials OOM at k=3 with checkpointing, coarsen sources
before detaching — detaching changes the objective.

## Arms and comparisons

Screen arms: **{vanilla, DAR, FBT, A1, A2}**, one recipe, matched tokens,
paired data order (same batches, same order — loss curves difference
cleanly), 2 seeds. Finalists (~3–4 arms: vanilla, the better combined arm,
parents as budget allows) then extend along the token ladder below.
Reported at matched tokens *and* matched token-equivalent compute (FBT
accounting: an n-pass batch costs n).

Interaction := (combined − vanilla) − [(DAR − vanilla) + (FBT − vanilla)],
evaluated per decode mode: **Standard** (no feedback), **Soft** (feedback
during generation), **Fused** (extra fused prefill pass + Soft).

The flagship runs only the better of A1/A2 — one large run, not a factorial
at scale.

## Configurations and scale plan

The flagship trains at ~370 tokens/param (400B on 1.08B) — far past
compute-optimal, and the regime where FBT's decode-time behavior actually
emerged [paper]. A short screen cannot reach that regime, and the feedback
phase is a *late fraction* of training under the pass schedule, so a
screen-scale FBT null is ambiguous rather than damning. The plan therefore
measures the **trend** of the combined advantage along a token ladder,
using WSD's extend-without-re-warming property (a cooldown branch at each
rung gives a measurement point; the extension continues from the
pre-cooldown checkpoint). De-risking spend totals ~$0.6–1k, ~10–15% of the
flagship; program total ≈ $7–8k.

| Stage | Model | Tokens | tok/param | Where | Rough cost |
|---|---|---|---|---|---|
| Smoke / dev | 220M (DAR's config: d=768, L=12, Qwen3-style) | ≤0.3B | — | jobe (1×4090) | free |
| Screen | 220M, 5 arms × 2 seeds | 2B / run | 9 | jobe, ~1 wk background (or rented, ~$10/run, if wall-clock matters) | free–$100 |
| Token ladder | 220M, finalists, 1 seed | 2B → 8B → 32B via WSD extension, cooldown branch per rung | 36 → 145 | rented single H100/H200 | ~$120–150/arm; $400–600 total |
| Mid-rung (params axis, optional) | ~300M | ~30B | 100 | rented | ~$150/run |
| Flagship | ~1.08B, FBT trunk: d=1536, L=24, GQA 16q/8kv headwise-gated, QK-norm, SiLU GLU 6656, RoPE, ctx 8192, 2048-SWA on 5/6 layers | 400B (FBT's largest) | 370 | Prime Intellect marketplace pods | ~2,700 H100-h ≈ $6–7k at 2026-07 rates (~$3–4k H200 spot, checkpoint-tolerant); ~2 wk on 8×H100 |

Sources are per-sublayer at every stage except the flagship, which coarsens
to block deltas. The ladder's top rung (145 tok/param) lands within ~2.5×
of the flagship's ratio; ladder arms run 1 seed with paired data order,
using the screen's seed spread as the noise estimate.

Tokenizer: Qwen3's (~151k, tied) everywhere — one tokenizer across our runs
beats matching FBT's phi-4 100k; DAR's "220M" is exactly the Qwen3-vocab
d=768/L=12 model [paper]. Data: FineWeb-Edu throughout, shared held-out val
split (FBT's Phi-4 mixture is unavailable). Optional flagship
follow-through, FBT-style, if the base result justifies it: long-context
extension (12B tokens, 8K→32K) then instruction tune (6B tokens),
three-pass throughout.

**Flagship controls.** No matched 1B vanilla (cost). Instead: decode-mode
ablations on the same weights isolate the channel's inference contribution;
pass-1 loss is tracked throughout training as the "as ordinary transformer"
mode; the ladder's vanilla arm and the optional 300M mid-rung anchor the
scaling extrapolation. Published 1B-class models (TinyLlama, Llama-3.2-1B,
Qwen3-1.7B, SmolLM2) appear as context rows only — 2–36T tokens of
unknowable data mixture makes them incomparable as controls.

## Evaluation and diagnostics

- **Language modeling:** paired val-loss curves per arm; token-equivalent
  compute isoclines.
- **Contraction:** the standing monitor above; also the stability gate
  before any flagship spend.
- **Decode modes:** Standard/Soft/Fused on every feedback-bearing arm.
- **Interpretability (first-class):** routing weights are direct
  observables — per-layer weight on the raw-payload source measures
  cross-column demand; per-layer weight on `e` tracks whether DAR's
  embedding-prominence pattern survives, or migrates to the fused input
  (deep layers reaching the previous column through layer 0). FBT's
  Appendix-F state-tracking synthetics (completion tracking, delayed
  memory, multi-register latest-write) reimplemented with linear probes
  across depth: what rides the payload, and does A2's routed payload carry
  different state than A1's top state?

## Gates

**Ladder entry** (from the screen): DAR must reproduce its published
ordering at the screen, or the harness is suspect and nothing else is
interpretable. Finalists are vanilla, the better combined arm, and parents
as budget allows. One deliberate asymmetry: a null-but-stable FBT side at
the screen does **not** exclude the best combined arm from the ladder — the
screen runs at 9 tok/param and formation-with-scale is precisely the
hypothesis it cannot test; the ladder is a ~$130 question.

**Promotion to flagship** (pre-registered): the combined advantage over
**both** parents at matched token-equivalent compute **holds or grows
across the ladder** (2B → 32B) — a trend, not a point estimate — **and**
the contraction diagnostic is clean past 30 self-compositions at the top
rung. The flagship spend is not authorized by default; confirm with a9 at
promotion time with ladder evidence in hand.

**Null reading:** FBT is unproven below 1B params (their smallest run). A
screen-level FBT null with healthy DAR reads as ambiguous (regime, not
refutation); a null that *persists across the ladder* reads as a
formation-conditions result (cf. acot at 135M). The DAR arm, proven at
exactly screen scale [paper], is the positive control that the harness
detects effects of this size.

## Extension (phase 2, contingent on the factorial)

Add pause-token pretraining (Goyal et al., arXiv:2310.02226) to the winning
model. Mechanism bet [synthesis]: in a vanilla transformer pauses add
parallel width only (constant input embedding; pause columns chain only via
attention); under latent feedback, consecutive pauses chain through the
payload at full depth — k pauses approximate a k-step loop transformer
unrolled horizontally, paid in KV entries. Pause supplies decode-time
serial depth; the feedback channel supplies the carry pauses lack.
Complementary dials: fused prefill scales prompt-side compute, pauses scale
decode-side. Goyal et al.'s core finding — pauses must be pretrained in —
matches this from-scratch setting [paper]. The parallel to
recirculated-dot-experiment is loose and conceptual; no shared protocol
obligation.

## Risks

- Both parents are single-group arXiv v1s; DAR's AttnRes baseline is their
  own reimplementation. Effect sizes are a few percent PPL — seeds, paired
  data order, and matched-compute accounting are load-bearing, not hygiene.
- DAR has public code (github.com/wdlctc/delta-attention-residuals-code) to
  borrow; FBT has none — the multi-pass loop is small but the stability kit
  has several interacting pieces. Reproduce the contraction signature
  before trusting any arm comparison.

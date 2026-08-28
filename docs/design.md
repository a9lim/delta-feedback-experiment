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
- **Secondary:** what happens when adoption is *free*? The screen-only
  DF-soft arm makes both channels optional (null sources everywhere,
  vanilla-reachable); its routing weights are a continuous readout of
  channel demand — including the injection-form question FBT explicitly
  leaves open [paper] — while the spine forces adoption and measures only
  benefit.

Pre-registered predictions [speculation]: a9 expects superadditive gains;
Claude hedges toward additive-to-mildly-sub (the channels may partially
substitute for the same limited per-column capacity). Either sign is a
finding. If superadditivity is real it should concentrate in the
Standard→Soft decoding gap on the same weights, since depth routing keeps
early-layer information alive in the payload [synthesis]. For DF-soft,
Claude predicts within-column adoption but payload-source non-adoption
(performance ≈ the DAR arm) — the same-arm split being the
committed-valley signature; DF-soft closing on DF instead would be the
surprising result (the gate was never necessary).

## Architecture

One decode step of the full model at position t (trial configuration:
per-sublayer sources). `route(vs, q)` returns
`sum softmax_i(q . rmsnorm(vs_i)) vs_i` with `q` zero-init.

```python
e = embed(tok)
u = rmsnorm(glu(p_prev, e))              # FBT gate: W_U p_prev * sigmoid(W_G e)
srcs = [u]                               # input seed + deltas: complete decomposition
h = u
for l in layers:
    h = h + route(srcs, q_attn[l])       # depth routing (no-op while len(srcs) < 2)
    a = attn(norm(h)); h = h + a; srcs.append(a)
    h = h + route(srcs, q_mlp[l])        # depth routing
    m = mlp(norm(h));  h = h + m; srcs.append(m)
p_prev = rmsnorm(h + route(srcs[1:], q_p))   # additive payload over deltas (no null)
tok = sample(lm_head(h))

# Parent arms delete from this model:
#   vanilla: u = e; no route() calls; no payload
#   DAR:     u = e; srcs = [e] (its input seed); no payload
#   FBT:     no route() calls; p_prev = rmsnorm(h)   (bare top state)
#
# DF-soft (screen-only fifth arm) replaces the hard choices with soft ones:
#   h = e (no GLU anywhere); srcs = [p_prev, e]; every route() — depth and
#   payload — carries a null source, so it can regress to vanilla.
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

**Feedback channel (FBT side).** The payload from column t−1 enters column
t one way: fused into the layer-0 input through FBT's asymmetric GLU
(payload on the value path, token embedding as the gate). This closes the
shortcut in which multi-pass losses are minimized by imitating the
no-feedback pass and ignoring state [paper] — load-bearing under our
schedule, where feedback passes arrive late, i.e. at an effectively
well-trained checkpoint, exactly the regime where optional paths go
unadopted (DenseFormer's committed valley; FBT's own rationale for the
gate) [paper]. Ungated access and adoption-by-choice are DF-soft's job,
not the spine's.

**Payload.** The payload is the top state *plus* a softmax-routed
combination of the column's delta sources under a dedicated static learned
query, RMS-normed — DAR's additive routing applied at the cross-column
site exactly as within the column: base signal preserved by default,
routing re-weights on top. No null source: the routed enrichment carries
fixed unit mass, forcing delta content into the recurrence (the hard
philosophy; the gain dial and the regress-to-bare-FBT option live in
DF-soft). Why keep the base: a pure routed mixture is structurally unable
to transmit the full column state — softmax weights are convex, and h_top
is the *sum* of the deltas, outside their convex hull — and repeats the
replacement-routing pattern DAR shows fails within the column [synthesis];
the base also keeps every delta on a direct cross-column gradient path
even under sharp routing, preserving the multi-pass auxiliary-supervision
mechanism [synthesis]. The payload router is *not* conditioned on the next
token — the GLU already gates the payload elementwise. At zero-init the
router is uniform over the N deltas, so the payload is
rmsnorm(h_top + (h_top − u)/N) ≈ the bare top state: the model starts as
approximately FBT-with-depth-routing and diverges only as payload routing
sharpens [synthesis].

**Sources.** Every routed arm carries the *complete decomposition* of its
stream: the source list is the column's input seed plus the per-sublayer
deltas, telescoping to the full hidden state (seed + Σv = h_top). The seed
is whatever the column's input actually is — `u` in DF, `e` in the DAR
arm (`srcs = [e]`), `e` in DF-soft (already present). This deliberately
departs from the paper's per-sublayer variant, which routes deltas only:
we read that omission as unprincipled — their Block variant seeds with the
embedding, completing the decomposition, and their embedding-prominence
finding was only *observable* in the variant that offered the option
[synthesis]. The seed closes that observability gap at trial granularity:
per-layer seed weight is the input-re-injection readout in every routed
arm. The depth-routing module remains identical between the DAR arm and
DF — only the seed's content differs, and that difference is entailed by
the feedback apparatus itself. Raw `e` and raw `p_prev` remain absent from
the spine's list (token identity passes only through the gate's
multiplicative pattern, FBT's own working regime [paper]; ungated payload
access is DF-soft's measurement), and the spine/DF-soft asymmetry stays
principled: minimal mandatory machinery vs a maximal optional menu. The
payload router is deliberately exempt from the completeness rule — it
routes deltas only, since a payload that re-amplifies its own carried
input would open a self-reinforcing persistence loop across steps, and
"what changed this column" is the enrichment's semantics [synthesis]. On
single-pass batches (no payload yet) the spine's input, and hence its
seed, is plain `e`.

**The hard/soft design space, and DF-soft.** Entry (gated vs plain input)
and routing nulls (absent vs present) span a design plane whose coherent
points are the diagonals: **hard-everywhere** — the spine above — forces
maximal adoption of both mechanisms and cannot give either up;
**soft-everywhere** makes both free choices and can regress all the way to
a vanilla transformer. Mixed corners (e.g. gated entry with a null-sourced
payload) trade coherence for site-local optimizations and are dominated by
the pair [synthesis]. The spine must be hard: under the late-feedback
schedule a soft feedback channel arrives at a well-trained checkpoint —
the committed-valley regime where the parents' evidence predicts
non-adoption. That prediction is itself worth testing, so **DF-soft** runs
as a screen-only fifth arm:

- `h = e`, no GLU anywhere — feedback is just one more routed source; one
  primitive everywhere, no W_U/W_G.
- Standing sources `[p_prev, e]` — ungated payload access and raw token
  re-injection, the two observables the spine gives up.
- A null source in *every* router, within-column and payload — the model
  can regress to vanilla, and every routing weight (null mass included) is
  a continuous readout of channel demand.

DF-soft contains its own control: within-column routing starts at step 0
(the regime where DAR's optional routing is known to adopt [paper]), while
the `p_prev` source structurally cannot appear before the late multi-pass
batches — the same mechanism predicts opposite fates for the two channels
in a single run. DF-soft is never a ladder candidate unless it matches DF
outright, which would itself be a headline result (the gate was never
necessary).

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
Pass 1 of DF is the depth-routing model with no feedback (u = e, payload
unused), so DF's Standard-decoding mode is a DAR-style transformer trained
with an extra objective [synthesis].

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

Screen arms: **{vanilla, DAR, FBT, DF, DF-soft}**, one recipe, matched
tokens, paired data order (same batches, same order — loss curves
difference cleanly), 2 seeds. **DF** ("delta feedback") is the
hard-everywhere model of the Architecture section; the parents are its
ablations per the pseudocode flags; **DF-soft** is the soft-everywhere
companion, screen-only — its adoption question resolves in the routing
weights at screen scale. Finalists (~3–4 arms: vanilla, DF, parents as
budget allows) then extend along the token ladder below.
Reported at matched tokens *and* matched token-equivalent compute (FBT
accounting: an n-pass batch costs n).

Interaction := (DF − vanilla) − [(DAR − vanilla) + (FBT − vanilla)],
evaluated per decode mode: **Standard** (no feedback), **Soft** (feedback
during generation), **Fused** (extra fused prefill pass + Soft).

The flagship runs DF only — one large run, not a factorial at scale.

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
  observables. Spine: the payload router's distribution over deltas (what
  rides the recurrence) and per-layer depth weight on the `u` seed (does
  the paper's Block-mode embedding-prominence appear at per-sublayer
  granularity, and does it migrate to the fused input?) — with the DAR
  arm's `e`-seed weight as the feedback-free baseline for the same
  question. DF-soft: per-layer weight on `p_prev` (does a
  free model demand the previous column?), weight on `e`
  (embedding-prominence under recurrence), and null masses everywhere —
  the adoption readout, including the injection-form question. FBT's
  Appendix-F state-tracking synthetics (completion tracking, delayed
  memory, multi-register latest-write) reimplemented with linear probes
  across depth: what rides the payload, and does the routed delta
  enrichment carry state the bare top state doesn't (probe with the
  enrichment term ablated vs as learned, same weights)?

## Gates

**Ladder entry** (from the screen): the DAR arm must clearly beat vanilla
at the screen, or the harness is suspect and nothing else is
interpretable. (It validates harness *sensitivity* to DAR-class effects
rather than reproducing the paper numerically — it runs the shared
FBT-binding recipe plus the input seed, neither of which the paper's
per-sublayer runs used.) Finalists are vanilla, DF, and parents as budget allows.
One deliberate asymmetry: a null-but-stable FBT side at the screen does
**not** exclude DF from the ladder — the screen runs at 9 tok/param and
formation-with-scale is precisely the hypothesis it cannot test; the
ladder is a ~$130 question.

**Promotion to flagship** (pre-registered): DF's advantage over
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

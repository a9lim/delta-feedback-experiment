# Journal

Disposable working space, periodically cleared. Keep only notes needed for
active work; anything durable graduates to `design.md` or `findings.md`.

## 2026-08-27 — scoping session (a9 + Claude)

Both parent papers read in full; design settled into `design.md` the same
day. Condensed record of forks considered and how they resolved, kept here
until the harness lands in case implementation reopens one:

- **Payload arms:** A1 (top state) and A2 (routed delta summary) kept; A3
  (all prev-column block deltas as separate per-layer sources) dropped for
  cost — the full-lattice variant is a different experiment.
- **Entry:** hybrid won over pure-mandatory (FBT) and pure-optional (DAR
  ethos): the layer-0 GLU stays mandatory because of the multi-pass
  shortcut problem; the raw payload additionally rides as an optional
  routed source, making adoption measurable instead of assumed.
- **Raw `e` as a standing source:** added after noticing DAR's
  embedding-prominence pattern collides with the FBT gate destroying
  additive access to the token embedding.
- **Payload RMSNorm:** added for FBT's O(1)-norm stability requirement;
  bonus discovery that it init-matches A1 and A2 (zero-init A2 payload
  telescopes to ~A1-minus-input).
- **Recipe conflicts:** papers disagree on optimizer (AdamW vs
  NorMuon+Adam) and tokenizer (Qwen3 151k vs phi-4 100k). Resolved: FBT
  recipe binding everywhere (its stability kit is load-bearing); Qwen3
  tokenizer everywhere (trials then reproduce DAR's 220M config exactly;
  one tokenizer across our runs beats matching FBT's).
- **Scale plan:** trials {vanilla, DAR, FBT, A1, A2} × 2 seeds × ~1B tokens
  on jobe; flagship = better of A1/A2 only, 1.08B / 400B tokens (a9 chose
  matching FBT's largest run over cheaper options; ~$6–7k H100-rate cost
  acknowledged). No matched 1B vanilla — decode-mode ablations +
  pass-1 tracking + small vanilla rungs instead (a9 agreed published
  models are context, not controls).
- **Priors registered:** a9 superadditive; Claude additive-to-mildly-sub,
  with the Standard→Soft gap named as where superadditivity should show.
- **Predecessors:** stateful-thought-experiment (CoST) and
  chain-of-dots-experiment deliberately cleared by a9 before this session —
  greenfield contract, decisions not inherited. Platform verdict (Prime
  pods; Tinker excluded for custom architectures) carried over from the
  2026-07-11 investigation.

Next: harness (`model.py` trunk + routing + gate + payload behind arm
flags; multi-pass loop + stability kit), then jobe smoke of vanilla and DAR
arms first — DAR is the positive control that must reproduce before any
feedback arm means anything.

## 2026-08-28 — hard/soft diagonals; DF goes hard-everywhere, DF-soft added

a9 generalized the entry/payload choices into a hard/soft design plane and
leaned diagonal: soft-everywhere (h = e, no GLU, srcs = [p_prev, e], nulls
in every router, vanilla-reachable) or hard-everywhere (h = u, srcs = [u],
no nulls, forced adoption of both). Claude updated — the prior mixed
corner (hard entry, soft payload) is dominated by the pair. Settling
argument: the late-feedback schedule means a soft feedback channel arrives
at a well-trained checkpoint — the committed-valley regime where both
parents' evidence predicts non-adoption — so the spine must be hard; but
that prediction is testable at screen scale, so both diagonals run:
**DF = hard-everywhere** (spine: screen → ladder → flagship candidate),
**DF-soft = soft-everywhere** (screen-only fifth arm, carrying all the
soft observables: p_prev demand, e-prominence, null masses). Casualties of
the hard spine: the e standing source (D5-era; FBT's multiplicative-only
token access suffices) and the payload null (the injection-form readout
moves to DF-soft). Registered prediction (Claude): DF-soft ≈ DAR arm,
within-column routing sharpens (step-0 start) while p_prev stays flat
(late arrival) — same mechanism, opposite fates, one run; DF-soft closing
on DF would mean the gate was never necessary (headline surprise).
Implementation note: route() must no-op with <2 sources (a singleton
softmax would double u at layer 0).

## 2026-08-28 — sources reversed: complete decomposition everywhere

The srcs = [] trim below lasted one turn. Discussing *why* the paper's
variants disagree (Block seeds with the embedding, per-sublayer doesn't)
produced the telescoping read: Block's {embed, Δ_1..Δ_B} is the complete
decomposition of the stream (sums to h_top); per-sublayer silently drops
the seed term — the omission, not the seed, is the inconsistency. Also:
embedding-prominence was only *observable* in the variant offering the
option, so the paper's headline interp finding is partly an artifact of
this asymmetry. a9 reversed the shape: **DF: srcs = [u]; DAR arm:
srcs = [e]** — complete decomposition in every routed arm, closing the
observability gap at trial granularity (seed weight = input-re-injection
readout everywhere). Costs accepted: the DAR arm is now the untested cell
(per-sublayer + seed) — its positive-control claim softened to
harness-sensitivity (it already ran the shared recipe, so numeric
replication was never on the table). Payload router deliberately exempt
from completeness (deltas only): a payload re-amplifying its own carried
input opens a persistence loop across steps; revisit flag. Singleton
no-op guard restored (len < 2 — the seed would otherwise be doubled at
layer 0), superseding the entry below's guard note.

## 2026-08-28 — spine sources trimmed to [] (a9's nit)

a9 re-read the paper: per-sublayer Delta AttnRes routes deltas only — the
input-as-first-source pattern is Delta Block's. srcs = [u] was a Block
feature transplanted into a per-sublayer configuration; dropped. Rule now:
sources are whatever the paper's variant at the active granularity uses
(per-sublayer: none; Block: the input seed — which at the flagship is u,
so the input-re-injection mechanism and its migrate-to-fused-input
observable return there, paper-licensed). Bonus: DAR arm and DF now share
an identical depth-routing module — DF−DAR isolates {gate, payload}
exactly. Spine/DF-soft source asymmetry declared principled: minimal
mandatory vs maximal optional menu. Supersedes the earlier <2-sources
no-op note: paper's depth_route softmaxes over a singleton (weight 1), so
only the empty list needs a guard.

## 2026-08-27 — arms collapsed to one hybrid, renamed DF

a9: A1 (bare-top-state payload) is the *less* parsimonious design once the
payload is additive — drop A1 and A2r, focus on the single coherent
hybrid. Ratified; combined arm renamed **DF** (delta feedback) since the
A-numbering counted nothing anymore. Claude's addition (veto-able): a
**null source** in the payload router only (from DAR's fine-tuning setup;
within-column routing stays paper-verbatim without one), which makes "at
worst the deltas are ignored" exact — bare-h_top is now a reachable point
and the router's null mass is a continuous readout of the injection-form
question, replacing the dropped A1 comparison with an observable.
Accepted tradeoff: no deconfounded attribution between payload enrichment
and mechanism coexistence (screen is 4 arms: vanilla, DAR, FBT, DF).
Pseudocode now shows the full DF model inline with per-arm ablation flags.

## 2026-08-27 — A2 corrected to additive payload (a9's catch)

The original A2 (payload = routed delta mixture alone) mis-transcribed
DAR's principle: DAR routes deltas *additively onto a preserved base*, and
the pure mixture is the replacement pattern DAR argues against. Structural
proof it mattered: softmax weights are convex, h_top is the *sum* of the
deltas — outside the convex hull — so old-A2 could not transmit the full
column state except by flattening routing (the very contrast collapse DAR
diagnosed). A2 is now h_top + sum softmax(q_p . rmsnorm(v)) v, RMS-normed.
Bonuses: A1 becomes A2's routing-ablated nested baseline (zero-init A2 =
A1 + O(1/N)), and every delta keeps a cross-column gradient path through
h_top under sharp routing. Caveat kept honest: DAR's
additive-beats-replacement evidence is within-column; the payload-site
version is suspect-by-analogy, not refuted — hence the optional
screen-only control arm A2r (pure mixture), a direct transfer test.

## 2026-08-27 — scale-plan revision (ratified)

a9 flagged the regime mismatch: flagship ~370 tok/param vs trials at ~4.5.
Restructured into screen (2B on jobe) + WSD token ladder (finalists
2B→8B→32B on rented single GPUs, cooldown branch per rung) + optional 300M
mid-rung. Gate became trend-based (advantage holds or grows across the
ladder); added the asymmetry that a screen-null FBT side doesn't exclude
the best combined arm from the ladder (the screen can't test
formation-with-scale). De-risking ≈ $0.6–1k; program total ≈ $7–8k.

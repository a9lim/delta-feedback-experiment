# Journal

Disposable working space, periodically cleared. Keep only notes needed for
active work; anything durable graduates to `design.md` or `findings.md`.

## 2026-08-29 — smoke-df1 landed; all three smoke gates read clean

Exit 0 at 00:29, queue DONE; snapshots at 825/1000/1100 (2.2G each);
multipass phase held 9.8G and ~4k tok/s on the checkpointed path.
Readings against the pre-registered smoke gates:

- **Contraction decays** — and dramatically. At the boundary (step 850)
  the iterated fused map was divergent: loss0=5.585 < loss8=7.656,
  upd8=4650. Fifty steps later loss8−loss0=0.054, upd8=6.5; by 1100
  loss0=4.122, loss8=4.124, upd8=**0.042** — five orders of magnitude
  down, effectively a fixed point. The standing stability gate passes
  at smoke scale.
- **val_fused → val**: 8.80 at the boundary, 5.17@850, 4.06@900, then a
  smooth glide to **3.795** vs plain val **3.774** — a 0.021-nat gap,
  closing monotonically through cooldown. Fused did not cross *below*
  plain within the 275-step feedback phase; whether it does is a
  screen-scale question (longer feedback phase), not a smoke failure.
- **Pass-1 recovery**: val 3.982@850 → 3.774@1100, final **+0.002 nats
  vs smoke-v1's 3.772** at matched tokens/steps/seed. Feedback training
  left the plain-LM path essentially untouched (df's multipass steps do
  cost extra compute — the matched-compute view is the screen's job).
- Payload router ended sharpened (max 0.183 from uniform 0.042 pre-
  boundary) with seed/null still near zero — feedback gradient reached
  the routing, as designed.

Smoke-level read, kept soft: formation conditions are all present —
feedback trains, contracts, and doesn't tax the base LM. No advantage
claim at this scale; that's the screen's question. Next session: read
these against design gates in full, then screen prep on the code a9 +
Codex optimize.

### Routing autopsy of the final checkpoint

`scripts/route_report.py` (new; figures/route-smoke-df1/, re-runnable at
screen scale) dissects the 1100-step snapshot: static query geometry +
mean routing maps over 32 val rows, plain and fused pass, with per-token
max/entropy to separate static wiring from dynamic selection. Verified
against step-1100 route telemetry (payload maxT 0.981 vs logged 0.9805).

- **Scale context first**: the raw embedding has RMS 0.04 vs deltas at
  10–305, so *routing mass on the seed is a learned off-switch* (the
  closest non-soft analogue of DAR's null source) — L0.mlp put 1.000
  there, i.e. turned itself off. m0 is anomalously large (RMS 44 vs
  ~12 for m1/m2): layer-0's MLP is the de facto embedding table.
- **attn readers**: recency band — each reads the 1–2 immediately
  preceding deltas — except L9/L10/L11.attn, which reach back to **m0**
  (0.34/0.48/0.81): a learned depth shortcut re-injecting token-local
  lexical state where the telescoping sum has diluted it.
- **mlp readers**: near-universal *self-read of their own layer's fresh
  attn delta* (a_i at 0.78–0.95 for L5–L9), i.e. routing amplifies
  attn→MLP coupling within the layer; they read attn deltas almost
  exclusively, rarely MLP deltas. Deep ones (L10/L11.mlp) split over
  a8/a9/a10 per token (maxT ≫ max of mean → genuine dynamic selection;
  mid-stack attn sites likewise).
- **payload router**: collapsed to a delta function on **a8** (0.981;
  was 0.18 mean-max at step 850 — cooldown sharpened it), identical on
  plain and fused passes. The feedback channel ships
  payload_norm(h_top + a8). Router choices are not norm-chasing — the
  biggest sources (m11 RMS 305, m10 146) get ~0 everywhere.
- Query norms grow with depth (~0.3→1.3, sharper softmax deep); deep-site
  queries (L5+) are mutually correlated in direction, payload's included.
  Pass-1 vs fused routing differs little; largest shift is L1.attn giving
  the fused entry u 10% (vs 1% to plain e).

**Why a8** — counterfactual sweep (`scripts/payload_swap.py`, forcing
the payload enrichment to each single delta; caveat: entry weights
co-adapted to a8, so alternatives are handicapped — read shape, not
gaps). Trained/forced-a8 fused 3.7948; **h_top-only and uniform both
3.8001** — the uniform init is a structural no-op (Σdeltas ≈ h_top up to
the tiny seed, and payload_norm scales it away), so the router had to
break symmetry to contribute anything, and its whole contribution is
~5 mn of the 21 mn fused-vs-plain gap. The landscape is family-
structured: every attn delta ties or beats no-enrichment (a8 3.7948 <
a6 3.7968 < a9 3.7976), every MLP delta *hurts*, monotonically worse
with depth (m8 3.8037 → m10 3.8102) — worse than sending nothing.
Read (synthesis): position t's fused entry already knows token t, so
late-MLP content — next-token-prediction features of position t−1 —
is redundant and is also the amplifying direction the contraction
fight is about; contextual attention summaries are the non-redundant
family. Within the attn family the preference for a8 over a6/a9 is
~2–3 mn under co-adapted weights — plausibly semi-contingent
(rich-get-richer during the short feedback phase; a different seed
might crown a6 or a9). Screen-scale question: does the family
structure persist and the specific choice stay stable across arms and
seeds?

## 2026-08-28 (night) — resume fix; runs.a9l.im tracking page live

Plan settled with a9: once smoke-df1 lands, a9 runs the optimization
pass with Codex directly (no gaslamp handoff needed from this side);
the correctness contract for that pass is the invariant suite, the
bit-exact-resume test, and the smoke logs/snapshots as parity
references. Next session picks up at: read smoke-df1 against its gates
(val_fused → val, contraction decay, pass-1 recovery), then screen prep
on the optimized code.

- First resume attempt under the spool FAILED in seconds: the spool folds
  the log at the `resume` record and requires it to carry `path=` (the
  source checkpoint) — df logged only the step. Fixed (0cf23c1): resume
  records carry their snapshot path; pinned by an assertion in
  `test_resume_is_exact`. Second attempt resumed cleanly from 825 and is
  through the feedback boundary on the checkpointed multipass path.
- **runs.a9l.im is now the multi-experiment tracker**:
  `runs.a9l.im/EXPERIMENT#RUN` (e.g. `/delta-feedback#smoke-df1`,
  `/recirculated-dot#TAG`), `/` a cross-experiment index. Shared
  `monitor.serve` rewritten for path mounts (root repo 780a428):
  repeatable `--exp NAME=REPO`, per-repo `monitor/serve.json`,
  mount-relative chassis fetches. df's monitor page:
  loss/pass1/val/val_fused, lr, gnorm, routing max/seed/null for payload
  + deepest sites, contraction loss + update norm; overlays across arms
  work (paired-comparison view). Trainer telemetry gained `schedule`
  (phase bands), `elapsed` (pace/ETA), `kind=snapshot` on checkpoints,
  and the `run` record now carries all EXACT fields (config card) — old
  logs degrade gracefully. Deployed as `runs-monitor.service` on jobe
  (unit lives in the monitor package; bootstrap step_monitor updated;
  rd-monitor.service removed, rd repo carries only serve.json now).

## 2026-08-28 (evening) — smoke-v1 done; df1 OOM at feedback boundary; GPU fell off bus

- **smoke-v1 complete**: final val **3.772** after clean cooldown to
  lr 0 (train 3.63), 22k tok/s solo, 7.9G peak. Sane vs DAR's 220M
  anchor (~3.61 at their budget/recipe). Vanilla gate passes.
- **Byte-prefix determinism check passes**: full 35B stream landed
  (34.97B train + 30M val, 131 shards, revision 87f09149); val.bin
  identical to smoke's, both smoke shards exact byte-prefixes of the
  full stream. Smoke runs literally trained on a prefix of ladder data.
- **smoke-df1 OOM'd at step 826 — the first two-pass step.** Single-pass
  DF sits at 18.9G (router source-stacks are the hog); holding two full
  activation graphs blew 24G. Honest allocator OOM; the protected
  step-825 (heat_end) snapshot was written first, so resume loses
  nothing. Fix (8e2c546): **multi-pass steps checkpoint blocks
  unconditionally** — `--grad-checkpoint` still forces it for k=1 —
  with a model-level parity test (loss + all grads, k=2). 32 invariants
  green.
- **Three minutes after the OOM exit, the idle GPU dropped off the PCIe
  bus** (Xid 79 → Xid 154 "GPU Reset Required"; nvidia-smi saw no
  devices). Likely load→idle power transient, a known 4090 mode.
  Rebooted jobe to recover. Watch for recurrence — if it repeats under
  the screen it's a hardware/PSU conversation, not a software one.
- df1 telemetry before the crash, all as expected: attn/mlp routers
  sharpening (L11.attn max 0.40 vs uniform 0.043, seed 0.037), payload
  router exactly uniform (no gradient until feedback), val_fused ~8.8
  and contraction absent (entry weights untrained pre-825).

## 2026-08-28 — harness complete; smoke live on jobe

Second build block landed (commits through 80251df): data pipeline
(uint32 memmap shards, val-first layout, revision-pinned FineWeb-Edu
stream, row-addressed reader), NorMuon implemented from arXiv:2510.05491
(added to refs.yaml) + Adam split, WSD via the shared Schedule with
AdamC wd-decay and cooldown z-loss, trainer with keyed derived
randomness (pass counts per step, prefix/jitter per microbatch — arms
must share data_seed/batch_rows/micro_rows), chunked+checkpointed head
loss (vocab 151936 logits were the memory hog), telemetry/checkpoints/
runs integration with **bit-exact resume** (tested), block-level
activation checkpointing (parity-tested), and `df` CLI with the shared
spool queue. 31 offline invariants green on Mac and jobe.

Operational state on jobe (all under `df status` / logs/):

- `/data/df/tokens-smoke`: 420M train + 30M val, Qwen3 tokenizer,
  fineweb-edu sample-100BT @ 87f09149, eos 151645. Tokenizer throughput
  ~3M tok/s single-process.
- `/data/df/tokens`: full 35B stream tokenizing in background (~3.5h).
  Same revision → must be a byte-prefix superset of the smoke stream
  (check with cmp when done — standing determinism check).
- Queue: `smoke-v1` (vanilla, 1100 steps = DAR's 0.33B-token budget at
  our geometry) running — init loss 11.96 ≈ ln(151936) sane, 14–18k
  tok/s, 7.3G peak; `smoke-df1` (df, same budget; feedback phase covers
  final 275 steps) queued behind it.
- GPU is shared with an active recirculated-dot run (~8.5G). Its 9.5h
  direct invocation ended (DONE 08:24:17, *before* our first GPU
  allocation ~08:25:30) and was immediately re-queued `--resume` under
  its own spool at 08:25:24 — looks like concurrent operator action on
  that experiment, not our interference; no OOM/traceback in its logs.
  Contention costs us ~20% (17.8k → 14.2k tok/s).

Screen projection from measured smoke throughput: 2B tokens ≈ 39h/run
vanilla-solo at 14k tok/s, likely ~2× for DF pre-optimization — the
Codex optimization pass (compile, flash-attn, fused draws) matters
before the screen. No torch.compile yet by design: correctness first.

Pending: smoke gates (loss sanity vs DAR's 220M numbers, routing
sharpness rising, DF contraction telemetry), Codex handoff, monitor
page, root-repo submodule pointer bump.

## 2026-08-28 — build forks resolved; DAR code archaeology

Pre-build fork review with a9; all four resolved and recorded in design/
AGENTS: (1) sub-flagship geometry ctx 1024 + FBT 300K-token batches
(divergence recorded; smoke validates NorMuon at this geometry); (2)
greenfield plain-PyTorch harness, DAR repo cloned to references/ as parity
reference; (3) screen on jobe (~3 days/run accepted), triage order
vanilla → DF → DAR/FBT → DF-soft with results gating each next spend; (4)
operational layer from the root package (telemetry/runs/checkpoints/spool/
Schedule/monitor). Claude writes first iteration, Codex optimizes
(recirculated-dot convention; flash-attn on jobe, SDPA fallback for Mac).

Cloning DAR's code paid off immediately:

- **The per-sublayer `delta` mode seeds sources with the embedding**
  (`if not blocks: blocks = [partial_block]` before the first delta
  append). The paper's Fig. 3 pseudocode omits the seed, but the code that
  produced the published numbers has it. Yesterday's complete-decomposition
  reversal is therefore code-verbatim, not a departure — and the DAR arm is
  *not* an untested cell: per-sublayer + seed is exactly what they ran.
  Design's Sources and Gates sections updated accordingly.
- **Queries are not zero-initialized in the released code** (HF default
  normal 0.02, no override), despite the paper's "zero-initialized
  queries" claim everywhere. Near-uniform routing at init either way at
  from-scratch scale; we keep exact zero-init as designed (the
  identity-at-init invariant is load-bearing for DF's init-matching
  argument). Recorded as a paper/code discrepancy, not adopted.
- **Null source implementation** (their fine-tuning setup): a *learnable*
  per-site d-vector, zero-init, prepended to the source list; key
  rmsnorm(0)=0 → logit exactly 0 at init. DF-soft adopts this exact
  mechanism.
- Their `final_res_proj` routing site exists only for replacement-routing
  modes (block/full) where the stream doesn't carry full state; delta
  modes end at norm(h) directly — no analogue to our payload site in their
  code.
- 220M run names confirm effective batch 32 sequences (bs2 × ga2 × 8
  GPUs); exact head split for d=768 is unrecoverable (script defaults are
  d=512-shaped) — pinned ours as 8H/4KV/head_dim 96/SwiGLU 3072 ≈ 223M.
- Their data pipeline (streaming shuffle, val from a different shuffle
  seed of the *same* split) is neither deterministic nor properly held
  out — our memmap-prefix design replaces it, one more reason numeric
  replication was never on the table.

Feedback-phase placement pinned provisionally in design (a knob):
single-pass first 75% of steps, 88/12 two-/three-pass mixture in the final
25%, coinciding with WSD cooldown — reproduces FBT's overall 75/22/3 and
keeps ladder rungs structurally comparable (every pre-cooldown checkpoint
is single-pass-trained). Cost flagged: feedback never sees stable LR;
feedback_start moves earlier if contraction/screen look unhealthy. FBT's
Appendix-C pseudocode details captured for the implementation: jitter
*before* shift, rmsnorm on the gate's embedding input and on the fused
input, prefix mixin reverts prefix positions to plain e.

**Design pseudocode bug found and fixed while implementing:** design.md
had `h = h + route(...)` — routing accumulated into the residual stream.
Paper Fig. 3 *and* code agree the routed sum feeds only the sublayer's
pre-norm input read (`attn(norm(h + routed))`); the stream accumulates
sublayer outputs only. The stream-accumulate version would have broken
the telescoping identity (seed + Σv = h_top) that the Sources section's
complete-decomposition argument rests on — the two sections were mutually
inconsistent until now. Corrected to transient-read semantics
(code-verbatim); Architecture pseudocode and Depth-routing prose updated.
Corollary: the len<2 no-op guard is belt-and-suspenders — a singleton
seed's weight-1 route gives norm(2u) = norm(u) under RMSNorm scale
invariance, so it was never able to do damage at the only site where a
singleton occurs. Guard kept (explicit, and keeps telemetry clean).
Also pinned: depth scaling = 1/√(2L) branch-output multipliers (FBT names
the property, not the formula); the entry gate reads rmsnorm(e) per
Appendix C; prefix mixin reverts prefix positions to *plain* e (bypassing
the entry norm) so prompt positions match pass-1 distribution exactly.

Next: harness first iteration (model.py + invariant tests → check-in),
then data pipeline, optimizer stack, multi-pass trainer.

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

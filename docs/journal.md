# Journal

## 2026-09-02 — screen-df-s1: why the fused pass underperforms pass 1

Same-checkpoint examination of the completed `df` run (step 10,745). Nothing
here is an arm comparison; every number is an intervention or a diagnostic on
one checkpoint and belongs to this scratch page until a matched comparison
exists. Scripts and raw outputs: `jobe:~/tmp/df-fused/` (`fused_diag.py`,
`gate_ensemble.py`, `dense_feedback_continue.py`, their JSON/logs); figures
under `figures/fused-screen-df-s1/` (ignored).

### The gap

| Metric (final checkpoint) | 32 eval rows (logged) | 256 rows |
|---|---:|---:|
| pass 1 (Standard) | 3.098 | 2.919 |
| fused prefill, prefix 1 | 3.122 | 2.944 |
| gap | +0.024 | +0.025 |

The 256-row pass reproduces the logged 32-row numbers to the fourth decimal
on its first 32 rows; top-1 accuracy also falls, 42.4% to 42.1%. The gap
fell monotonically from +0.56 at step 5,600 to +0.042 at the end of heat and
+0.024 at the end of the run; against cumulative feedback passes it follows
`n^-1.04` from step 7,000 on. That fit does not identify exposure as the
cause: feedback passes, optimizer step, plain-token training, and the
cooldown are collinear over that range, and a floor of 0.008–0.015 fits
almost as well. The train-time pass-2 minus pass-1 penalty on random-prefix
rows ended at +0.015, consistent with +0.024 once all positions are fused.
Gradient-weighted, the fusion parameters were updated on 2,996 of 10,745
steps and the payload/gate path received its loss at unit weight only on
those steps.

### What the penalty is made of

- **Local to the fused position, not accumulated through context.** With a
  plain 512-token prefix the suffix penalty is +0.0229; with prefix 1 the same
  positions carry +0.0239. Under a training-style random prefix the plain
  prefix is bit-identical to pass 1 and the fused suffix carries +0.024.
- **Flat over position beyond 64.** +0.024 for positions 64–1023; +0.031 for
  16–63; +0.058 for 4–15; +0.065 for 1–3. Position `t` is fused with
  probability `t/1024` in a feedback pass, so the early-position excess sits
  exactly where fused exposure is rarest, but those positions are 6% of tokens
  and explain ~0.002 of the mean gap.
- **Grows with the surprise of the fused-in token and with token rarity.**
  Binned by the previous column's `-log p` of the token that gets fused in:
  +0.021 for the bottom 80%, +0.044 at 8.6–11.6 nats, +0.085 above 11.6. The
  fused seed's nearest embedding is the true token 48% of the time for
  expected tokens and 11% for very surprising ones. Binned by input-token
  count in the 30M held-out slice: +0.047 for the rarest 10%, +0.011 for the
  most frequent 10%.
- **A different predictor, not a degraded copy.** Mean KL(pass 1 ‖ fused) is
  0.057 nats, argmax agreement 87.5%, and 47.6% of tokens have *lower* loss
  under the fused pass (median per-token delta +0.002, IQR −0.107…+0.137).
  Fused helps where pass 1 is badly wrong (own CE > 8.6: −0.03 to −0.07) and
  hurts where pass 1 is confident (CE < 0.4: +0.025, only 43% improve).
- **Little complementary information.** A probability mixture with 30%
  weight on the fused pass beats pass 1 by 0.006 and the geometric mean by
  0.002, with the weight chosen on the same 128 rows and no temperature
  scaling; the split-validated, temperature-calibrated version is in the
  follow-ups below.
- **A second fused iteration hurts further** (+0.032 versus +0.025), the
  opposite of FBT's "further passes continue to help".
- **Self-composition is a clean contraction to a worse fixed point.** Over 30
  fully fused iterations on the first 8 rows (harder than average: pass 1
  3.397 against 2.919 overall) the loss goes 3.397 → 3.429 → 3.440 → 3.442
  and stays flat; the relative top-state update decays 0.20 → 0.058 → 0.017
  → 0.008 and plateaus (a small cycle is not excluded at that floor).
  Stable, and the fixed point is 0.046 above pass 1; the third column is
  the least-trained one, so this says the map is stable and unhelpful on
  this checkpoint, not that recurrence is harmful in itself.
- **Jitter is irrelevant.** Adding the training jitter to the payload at eval
  changes the fused loss by 0.0001; every payload coordinate's across-token
  std (≥ 0.29) is 25× the jitter RMS (0.0115).

### Mechanism at the entry and the payload

- **The token gate is nearly flat in magnitude but functionally decisive.**
  Gate-logit RMS is 0.67 (sigmoid range roughly 0.34–0.66), per-coordinate
  gate std across tokens 0.135, and the explicit token-dependent part of the
  pre-norm fused input is 8% of its RMS (median; p90 15%). `fuse_gate` grew
  from Frobenius 15.36 to 19.05 over the whole feedback phase; `fuse_value`
  is pinned at its Hyperball radius. This is an energy split, not an
  information split: the token is linearly recoverable from the seed (ridge
  probe R² 0.79, nearest-token hit 98.4% on held-out tokens) and replacing
  the gate by a constant costs 10 nats. The raw seed's nearest embedding is
  right only 35% of the time and its cosine with the embedding is 0.10,
  which says the seed lives in a learned basis, not that the token is lost.
- **The fused seed enters at 14× the plain scale** (RMS 0.58 versus 0.04;
  `h_top` RMS 0.48). The first cell's block delta changes by 3.6× its own RMS
  between pass 1 and pass 2, the second by 1.0×, the third by 0.36×, the top
  state by 0.17×: the trunk largely undoes the fused seed with depth rather
  than exploiting it. Within-column routers cut their seed weight on fused
  passes at every early site (L1.mlp 0.46 → 0.04, L3.mlp 0.90 → 0.19), which
  roughly holds the *absolute* seed contribution constant; only L10.attn and
  L11.attn read the seed more on fused passes.
- **The fused pathway is complete and tightly co-adapted, not a perturbed
  plain pass.** Eval-time entry interventions on 64 rows (pass 1 3.001,
  fused 3.027): a constant 0.5 gate, which removes the token from fused
  positions, gives 13.36; a payload taken from a different row gives 3.471,
  so the previous column's state is worth 0.44 nats at a fused position
  and the token identity reaches the trunk through an 8%-RMS modulation
  that is nonetheless decoded; scaling the fused seed by 0.5 or 2 costs
  5.3 and 1.0 nats; gate temperature 2 costs 0.13 and 0.5 costs 0.43;
  adding the raw embedding as a bypass costs 0.10 at unit weight and more
  beyond. Every deviation from the trained input distribution is expensive,
  so the +0.024 is the residual of a fully formed second pathway, and the
  rare-token and surprising-token excess is the lossy part of decoding the
  token identity through the multiplicative gate.
- **The payload router is a sparse one-head solution.** Three of four heads
  put 100% of their mass on the site null (a learned constant, RMS 0.14) on
  both passes; the fourth reads `block2` on pass 1 and mostly `seed` on pass
  2. Same-checkpoint swaps (32 rows): trained router 3.122, all heads on null
  3.173, `block2` 3.367, `seed` 3.735, uniform 3.562, `h_top` only 3.864 —
  the co-adapted fusion needs the constant offset, the one active head is
  worth 0.05, and forcing every head to one source is far out of
  distribution, so those numbers bound nothing about source usefulness.
  Per-head ablation is in the follow-ups below. For promotion gate 3 this
  is three idle heads and one used one, not a wholly bypassed mechanism.

### Exposure accounting against the FBT recipe

The run realized 2,996 feedback passes over 10,745 steps: 0.279 feedback
passes per step, 1.279 pass-tokens per predicted token, and 4.9e8 fused
position-exposures against 3.5e9 plain ones (ratio 0.139). FBT's recipes by
scale (1B parameters):

| FBT run | mixture | feedback passes / step | fused ÷ plain exposure |
|---|---|---:|---:|
| 10B (10 tok/param) | 100% three-pass | 2.00 | 1.00 |
| 100B (100 tok/param) | 75/0/25 | 0.50 | 0.25 |
| 200B, 400B | 75/22/3 | 0.28 | 0.14 |
| this run (25 tok/param) | 75/22/3 realized | 0.279 | 0.139 |

So the screen used the ≥200-token-per-parameter recipe at 25 tokens per
parameter; the fused path saw 0.49B fused positions where FBT's own
low-ratio recipes would have given it 0.88B (75/0/25) or 3.5B (0/0/100).
The gate matrix is Adam-owned and only receives gradient on feedback steps,
which is where its 24% growth and flat logits come from.

### Follow-ups after an adversarial read (Codex, gaslamp job cx-20260902-045727)

- **No gradient conflict between the two losses.** On 32 fresh training
  rows at the final checkpoint, the cosine between the gradients of `ell_1`
  and `ell_2` is 0.75 on trunk matrices, 0.98 on the embedding, 0.90 on the
  attention gates, 0.87 on the remaining trunk parameters, 0.86 over every
  shared parameter, with similar norms (0.46 versus 0.56 on trunk
  matrices). The fused objective pulls the same way as the plain one; it is
  not a mode fighting the plain pass. The fusion matrices receive the
  largest gradient per unit weight (0.24 on norm 48.6); the payload router
  receives almost none (0.005 on norm 30.3) because its softmax is
  saturated, so its null solution is absorbing.
- **Complementary information is 0.006 nats, split-validated.** Choosing
  temperatures and mixture weight on 64 rows (best: both temperatures 1.0,
  30% fused) and scoring on the other 192 gives 2.8856 against 2.8914 for
  pass 1, whose best temperature is also 1.0.
- **The payload router's one active head is mostly a bias.** Replacing head
  2's routed slice with the null costs 0.046 (3.027 → 3.072); replacing it
  with another row's head-2 read costs 0.0045. Its row-specific content is
  worth 0.005; the other three heads are already null. Functionally the DF
  payload enrichment on this checkpoint is `payload_norm(h_top + constant)`.
- **A fused column writes a state that plain columns cannot read.** Fusing a
  block of 1, 8, or 64 positions and returning to plain embeddings costs the
  next plain position +0.18, +1.09, and +1.16 nats respectively, decaying to
  noise over 8–64 positions. That boundary (fused then plain) never occurs in
  training or in Soft decoding, so this is an out-of-distribution probe, not
  an in-regime propagation measurement; the in-regime test remains the equal
  penalties under prefix 1 and prefix 512. It does show the fused regime's
  recurrent and attention state is a different object, not a perturbation
  of the plain one. Single-position penalties in this test have a standard
  error near 0.06 at 64 rows and are not individually interpretable.

### Dense-feedback continuation (same checkpoint, not an arm)

Pending: `dense_feedback_continue.py` restores the final snapshot with both
optimizer states and trains every step with two passes on fresh rows past the
schedule at 10% of the stable learning rates (10-step warmup), first with
every parameter trainable (150 steps), then with only the fusion parameters
(`fuse_value`, `fuse_gate`, entry/gate/payload norms, payload router; 100
steps). Eval every 25 steps under the run's convention.

### Reading

Partly an exposure artifact, partly not. The exposure signature is real: the
gap decays as `n^-1` in feedback passes with no floor in sight, the penalty
is largest exactly where fused exposure is smallest (early positions, rare
tokens), and the entry gate has barely moved off its initialization. But the
penalty at well-exposed positions on frequent tokens is still +0.011–0.024,
the fused pass changes the first cell's computation by 3.6× and the trunk
spends its depth undoing that rather than using it, the extra information in
the fused pass is worth 0.006 in ensemble, and iterating makes it worse.
The FBT mechanism (a shallow layer reading full-depth past states through
dense attention) enters a PKDA recurrence here, at 14× the plain seed scale
and 92% previous-state content, and this checkpoint has learned to tolerate
that input, not to profit from it. The continuation says how much of the
remaining gap more feedback exposure alone can buy.

### Options (decisions for a9, not taken)

1. **Match FBT's recipe to the token-per-parameter regime.** FBT chose its
   mixture per scale; at 25x the closest precedents are 75/0/25 (100 tok/param)
   and 100% three-pass (10 tok/param). A screen-wide change to
   `feedback_batch_prob`/`three_pass` (and possibly `feedback_start`) is
   contract-neutral for the factorial because only `df-s1` is complete, but
   it changes the pass-token cost of every feedback arm and the design's
   whole-run 75/22/3 statement, and the flagship at 400x keeps 75/22/3
   either way.
2. **Leave the recipe and accept a fused deficit at 25x.** The design already
   states a stable FBT null at 25x does not exclude Prime; the factorial's
   pass-1 effects (MHDB, FBT, interaction) are unaffected by which fused
   metric is reported alongside.
3. **Architectural changes are out of contract.** An additive embedding
   bypass would remove the token-decode cost but is exactly the shortcut
   FBT's asymmetry exists to close; a payload router that cannot collapse
   to a constant is a separate MHDB question. Neither belongs in the
   screen without a registered decision.

### Repo notes

- `scripts/payload_swap.py` and `scripts/route_report.py` were broken on
  CUDA since the classifier shadow and BF16 sink contracts landed (no
  shadow prepared; FP32 activations against the BF16 shadow in CCE). Fixed
  by preparing the shadow and autocasting on CUDA; both reproduce the
  trainer's eval numbers.

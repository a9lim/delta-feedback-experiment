# Journal

## Screen midpoint diagnosis: gated arms trail vanilla (2026-08-30)

Status: `vanilla` and `df` complete; the matching `mhdar` run was stopped at
step 4095 after reproducing the gated-arm dynamics; `fbt` and `df_soft` are not
queued. Working diagnosis, not an accepted finding.

### Observed ordering

- Final pass-1 val: `vanilla` 3.276, `df` 3.313 (fused 3.317). Gap 0.037.
- Matched-step evals: gated arms lead vanilla for the first ~300 steps, are
  overtaken by ~400, then track parallel — the gap neither closes nor grows
  through heat and cooldown.
- `mhdar` vs `df` during heat is not an architecture comparison: `df`'s fusion
  and payload parameters are bit-exact at init through step 5025 (checked at
  the heat-end snapshot: fuse row-norms 0.554 = init, payload |q| = 0), so the
  two runs compute the same function on the same data. Their persistent
  ~0.007–0.01 offset is an execution-reproducibility floor (atomics/graphs),
  and calibrates noise: the vanilla gap is ~4x it. Expect `mhdar` to finish
  ≈ `df` ± 0.01.

### Same-checkpoint causal probes (16 val rows, CPU, pass-1)

`df.6700` baseline 3.330 on the probe subset:

- **Gates binarized** (`gate -> 1[gate>0.5]`): 3.331. The sigmoid gates are
  functionally binary.
- **Gates made static** (per-coordinate batch mean, token dependence removed):
  5.724. The binary masks are strongly token-dependent — learned hard routing,
  not static pruning.
- **Routing queries zeroed** (uniform-source intervention): 3.907. MHDAR routing is
  adopted and load-bearing.

Gate activation stats (`df.6700`, identical shape at `df.5025` and
`mhdar.3500`, so locked in by mid-heat): ~95% of gate coordinates saturated
(<0.1 or >0.9) in most layers; deeper layers ~90% open. **L0 attention is
amputated**: mean gate 0.003, 99.7% of coordinates < 0.1; its branch delta RMS
is 0.5 vs 174 for its MLP. In `vanilla`, L0 attention is the *largest*
attention contribution in the network by delta RMS (306). With sigmoid
saturated, gradient through the gate is ~0 and gradients into L0 attention are
attenuated ~300x: the decision is frozen.

### Mechanism

`W_g` sits in the NorMuon group. NorMuon rescales every update to Frobenius
norm `0.2*sqrt(m*n)` = 153.6 for 768x768; at lr 1e-2 that is 1.54 per step
against an init norm of 15.4 — ~10% of init per step, blind to gradient scale.
The sigmoid is the one scale-sensitive nonlinearity fed by a NorMuon matrix
(the router queries, by contrast, are Adam at 5e-4 and learned moderate norms,
|q| per head 1–2 over RMS-normed keys). `W_g` row norms grew 0.554 → ~16
(pre-activation std ~16 logits on unit-RMS input) within the first few hundred
steps, saturating the gates and freezing them. The crossover at step ~400
coincides with this window. Gated arms also show correlated instability the
vanilla arm lacks on the same rows: at steps ~3169–3243 `mhdar` hit gnorm 39
(vs its usual ~0.17) and both gated arms spiked the step-3200 eval; vanilla
stayed at ~0.15.

The gate reference behavior (Qwen gated attention) was established under
AdamW, where gate logits grow slowly and reversibly. The deviation here is the
optimizer treatment, not the gate design.

### Read on "data vs setup"

- Architecture package: setup, not data. The gap is flat from step 400 through
  2B tokens including cooldown, and the mechanism (saturated gates, zero gate
  gradient, dead L0 attention) predicts it stays. All arms are undertrained
  (2B tokens vs ~4.5B Chinchilla-optimal for 224M) but matched.
- Recurrence package: genuinely schedule-starved, by design. Feedback trains
  only in cooldown (1675 steps, k=2/3); fused val went from divergent (8.9,
  untrained) to 0.004 above pass-1 and was still closing at run end;
  contraction trace clean (loss8 ≈ loss0, upd8 stable). No verdict available;
  `fbt` (ungated, unaffected by the gate pathology) will give the clean
  recurrence-only cell.

### Current decision

- Sigmoid attention-gate projections use Adam at `5e-4`, while FBT's value and
  token-gate fusion matrices retain their paper-aligned NorMuon treatment. The
  optimizer-state change advances the checkpoint contract to v6; saturated v5
  runs remain analysis artifacts and are not resumable into the new recipe.
- The next gated run starts from initialization. Its early kill test is the
  step-~400 crossover, with gate mean, saturated fraction, sigmoid derivative,
  weight norm, and attention-delta scale inspected before full-screen spend.
- `df_soft` remains held until the attention-gate treatment passes that test.

Probe scripts are session scratch (`/tmp/ckpt_diag.py`, `/tmp/causal_probe.py`
on Jobe); promote into `scripts/` only if they earn a place.

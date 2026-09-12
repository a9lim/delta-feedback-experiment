# Looking inside

The organism is being grown so latent computation can be observed, explained,
and intervened on. This page lists the implemented analysis tools and explains
their measurement boundaries.

## Three kinds of recurrence

Token position, full-sequence feedback pass, and tied-core iteration are
separate axes. They execute different computations and a trace has to say
which one it is indexed by.

| Axis | State and transition | Status |
|---|---|---|
| Token-mixer memory | PKDA matrix, diagonal preconditioner, and convolution history advance along tokens; GQA retains prefix K/V | Every condition |
| Latent feedback | FBT transfers the previous column's payload through the current token's gate | Every condition with `f`; Jacobi passes train it in parallel |
| Tied depth | A shared core repeatedly updates one column's residual state; iteration `i` mixes over earlier columns' iteration-`i` writes | Every condition with `l` |

In repeated-prefill diagnostics each pass recomputes its mixer states from
zero and consumes the preceding pass's shifted payload. In sequential feedback
decode, the mixer caches and the immediately previous payload both continue
across generated tokens. The two execution modes can behave differently.

## Tools

| Surface | What it shows | Notes |
|---|---|---|
| `DeltaModel.forward_column` / `ColumnOutput` | Top residual state, feedback payload, final source bank, optional per-site/group route weights with source labels; expert weights and mean balancing loss | The analysis interface everything below builds on |
| `scripts/route_report.py` | Plain and fused route summaries, source and null scales, query geometry | Route mass is a mixing weight, not an importance score |
| `scripts/payload_swap.py` | Top-only, uniform, and forced-source payload enrichment, optionally one routing group at a time | Top-only keeps the payload's `h_top` term |
| `scripts/fused_diagnostics.py` | Fused minus pass-1 loss by position, fused-in-token surprise, and token frequency; gate, seed-decodability, and scale statistics; a 30-iteration self-composition trace | Conditioning on a position's own pass-1 loss selects on noise; condition on entropy or frequency instead |
| `scripts/entry_sweeps.py` | Gate temperature, fused-seed scale, embedding bypass, zero and foreign payloads at the FBT entry | Out-of-distribution perturbations of a co-adapted entry; they bound dependence |
| `scripts/feedback_followups.py` | Fused-block impulse response, payload-head ablation, split-validated pass mixtures, pass-1 versus fused gradient alignment | The fused-then-plain boundary never occurs in training |
| `scripts/compare_conditions.py` | Two checkpoints on identical rows: per-token loss structure, predictor KL and agreement, mixtures, residual-stream CKA per source, payload redundancy with the readout | Paired runs differ like two seeds |
| `scripts/weight_divergence.py` | Angular movement of every shared parameter from the paired initialization and between two runs | Under fixed-radius updates angle says little about function |
| `scripts/dense_feedback_continue.py` | Continued training of one snapshot with dense feedback passes or the plain-only control | A recipe intervention on a checkpoint |
| `scripts/downstream_eval.py` | The workspace zero-shot suite in Standard, Soft, or Fused mode, with per-document records for paired comparison | Soft feeds back only along the scored continuation |
| `scripts/training_curves.py`, `scripts/analysis_figures.py` | Figures from run logs and from the JSON records above | Rendering only |
| `iterate_fused` | Per-iteration held-out loss and mean top-state update norm on fixed tokens | Payload-only repeated prefill |
| `depth_trace`, `scripts/depth_trace.py` | Loss after each core iteration count, the size of each iteration's update, and the core routers' source mass by iteration | Fixed-`r` sweep; on plain positions one trajectory read out after every iteration |
| Trainer telemetry / `delta watch` | Training health, validation, routing summaries, recurrent dynamics | Operational |
| `delta probe` and portable tests | Tiny CUDA execution smoke; reduced portable suite on CPU/MPS | Engineering |

Every script rebuilds the condition from a snapshot through
`delta_feedback_experiment.analysis`, evaluates under the trainer's numerics
(BF16 autocast on CUDA, classifier shadow prepared), and writes a JSON record
beside its figures under `figures/<kind>-<tag>/`. Commands are in
[operations.md](operations.md#inspect-a-checkpoint).

## Reading a trace

Start with the actual column seed, the completed block deltas, `h_top`, the
payload, and the mixer states. The reconstruction identity
`h_top = seed + sum(completed block deltas)` checks a source interpretation.
Norms, distances, route entropy, and low-dimensional projections locate an
effect; identifying the computation takes an intervention.

For matched replay hold tokens, positions, pass and prefix choices, and cache
history fixed. When testing autoregressive consequences, let generated tokens
diverge and treat it as a separate experiment, since later differences then
include feedback through changed text.

Keep the snapshot, data rows, mode, state coordinates, and precision with an
analysis result so the measured intervention can be reproduced.

Request `want_weights=True` to obtain `expert_weights`, keyed by
layer invocation, alongside the separate MHDB `route_weights`. Each expert
tensor is `[B, T, 15]`: three nonzero routed weights per token, normalized to
sum to one. The shared expert always runs and is absent from that axis. Align
core iteration as well as token and pass when comparing expert selections.
Each physical bank also carries a persistent `expert_bias` that affects expert
selection without entering the mixture weights. Hold that bias fixed during
matched replay. `ColumnOutput.expert_counts` sums actual selections by physical
bank across repeated invocations; the trainer combines these counts across the
whole update before adjusting the biases once.
The auxiliary loss regularizes unbiased expert preferences within sequences;
actual dispatch can differ because of the selection bias. Expert choice and
gate mass alone do not establish specialization or causal importance.
Trainer `expert_balance` telemetry reports the unweighted mean sequence
auxiliary loss, while the expert summary reports per-site assignment fractions and
selected-gate entropy and `bias0` through `bias14` on up to two validation
rows. Its Standard-mode sample does not describe expert use during later
feedback passes. Step-level `expert_max_violation` uses actual whole-update
assignment counts: the worst physical bank's `max / mean - 1` load ratio.
`expert_bias_max` records the maximum absolute post-update bias.

## Interventions

Useful distinctions when perturbing the model:

- Changing route weights against changing source values.
- Removing enrichment against removing the whole payload.
- Losing behavior under disruption against recovering it with an informative
  state; the first is necessity, the second sufficiency.
- A decodable variable against a used one. Recovering a token or readout
  state from the payload does not establish how the column uses it.

An identity patch is the execution control. A matched unrelated donor, a
magnitude control, and a restoration where feasible separate a targeted
effect from generic damage. Donors align in position, mode, shape, and
history, and never encode future information relative to the patched
position; a zero vector or shuffled row creates out-of-distribution damage on
its own. Report effect sizes over paired examples and the range, not just the
prettiest trajectory.

## Dynamics

Auxiliary validation predicts a second token after receiving the
ground-truth next token's embedding. `val_mtp` and `val_mtp_fused` therefore
measure a different predictor and target alignment from `val` and
`val_fused`. Compare the ordinary next-token metrics between paired runs to
assess the effect on the model used for generation. Auxiliary accuracy alone
does not establish speculative-decoding speed or payload usefulness. The MTP
block adds no recurrent state or payload path to the ordinary model's
inference computation.

Track state movement and behavior together. Small updates can mean a settled
representation, an unproductive fixed point, numerical resolution, or an
unused channel; large updates can mean purposeful change or instability. Loss
and task behavior separate some of these; interventions separate the rest.
The fixed-token self-composition trace is the first diagnostic; perturbation
propagation, recovery, and dependence on history are natural follow-ons.

The `l` letter adds iteration-indexed states; the depth trace is its
within-column diagnostic, and the payload trace still tests its horizontal
channel at fixed `r`.

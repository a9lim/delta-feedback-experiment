# Looking inside

The organism exists so that latent computation can be observed, explained,
and intervened on. This page lists the analysis tools that exist today, keeps
the axes of recurrence straight, and sketches the shape of the study the
organism is being grown for. The current picture from these tools is in
[findings.md](findings.md).

## Three kinds of recurrence

Token position, full-sequence feedback pass, and tied-core iteration are
separate axes. They execute different computations and a trace has to say
which one it is indexed by.

| Axis | State and transition | Status |
|---|---|---|
| Token-mixer memory | PKDA matrix, diagonal preconditioner, and convolution history advance along tokens; GQA retains prefix K/V | Every condition with `a`, whether or not it has `f` |
| Latent feedback | FBT transfers the previous column's payload through the current token's gate | Every condition with `f`; Jacobi passes train it in parallel |
| Tied depth | A shared core repeatedly updates one column's residual state | The `l` letter, specified and not built |

In repeated-prefill diagnostics each pass recomputes its mixer states from
zero and consumes the preceding pass's shifted payload. In sequential feedback
decode, the mixer caches and the immediately previous payload both continue
across generated tokens. The two execution modes can behave differently.

## Tools

| Surface | What it shows | Notes |
|---|---|---|
| `DFModel.forward_column` / `ColumnOutput` | Top residual state, feedback payload, final source bank, optional per-site/group route weights with source labels | The analysis interface everything below builds on |
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
| Trainer telemetry / `delta watch` | Training health, validation, routing summaries, recurrent dynamics | Operational |
| `delta probe` and portable semantics | Numerical, causal, gradient, cache, and execution invariants | Engineering |

Every script rebuilds the condition from a snapshot through
`delta_feedback_experiment.analysis`, evaluates under the trainer's numerics
(BF16 autocast on CUDA, classifier shadow prepared), and writes a JSON record
beside its figures under `figures/<kind>-<tag>/`. Commands are in
[operations.md](operations.md#inspect-a-checkpoint).

What does not exist yet: a persistent trajectory format, general activation
patching between examples, controlled behavioral tasks, and any learned
monitor. Each is a natural next tool once there is a channel worth pointing
it at.

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

A trace worth keeping records the code and vendor revisions, snapshot and its
training arguments, the rows or task items, the mode (Standard, Soft, Fused),
pass count, prefix masks, jitter, the state coordinates (position, layer or
cell, site and source label, pass, iteration), device and precision, the
command, and where the raw output went.

## Interventions

Useful distinctions when perturbing the model:

- Changing route weights against changing source values.
- Removing enrichment against removing the whole payload.
- Losing behavior under disruption against recovering it with an informative
  state; the first is necessity, the second sufficiency.
- A decodable variable against a used one. The seed decodes the token at 98%
  and the payload is the readout basis; neither says the column uses them
  beyond reconstruction.

An identity patch is the execution control. A matched unrelated donor, a
magnitude control, and a restoration where feasible separate a targeted
effect from generic damage. Donors align in position, mode, shape, and
history, and never encode future information relative to the patched
position; a zero vector or shuffled row creates out-of-distribution damage on
its own. Report effect sizes over paired examples and the range, not just the
prettiest trajectory.

## Dynamics

Track state movement and behavior together. Small updates can mean a settled
representation, an unproductive fixed point, numerical resolution, or an
unused channel; large updates can mean purposeful change or instability. Loss
and task behavior separate some of these; interventions separate the rest.
The fixed-token self-composition trace is the first diagnostic; perturbation
propagation, recovery, and dependence on history are natural follow-ons.

The `l` letter adds iteration-indexed states and a shared core cache. Its
two-channel sequence trace advances both the payload and the write bank, so
the current payload-only trace does not test its dynamics. See
[depth-architecture.md](depth-architecture.md).

## The study the organism is for

Sketched, so that growing decisions point at it:

- **A behavior that needs the channel.** A small, externally scored task
  where the answer depends on state carried between columns and not on the
  visible prefix or PKDA memory alone. Candidate families: retaining a
  controlled fact across distractors, updating a latent variable after a rule
  change, tracking a short sequence of operations. Paired items so surface
  cues do not give the answer away.
- **A causal account.** State what should carry what, when it should matter,
  and what a targeted intervention should change, then patch payloads between
  paired examples, replace block contributions, or disrupt mixer state at a
  defined boundary, with the controls above.
- **A monitor.** A narrow independently defined target (a controlled
  overwrite, a known loss of retained information, a trajectory headed for a
  specified failure), a declared observation budget, probes fit on
  development data and frozen, evaluation on disjoint items split by
  template, comparison against output-only and simple internal-statistic
  baselines, and error rates, calibration, and lead time reported.

That study is future work. The organism is ready for it when the properties
in the [README](../README.md#what-ready-looks-like) hold.

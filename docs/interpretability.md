# Interpreting and monitoring the recurrent organism

The purpose of this organism is to make latent computation experimentally
accessible: observe a trajectory, propose an explanation, intervene on its
state, and test the predicted behavioral consequence. Monitoring then asks
whether limited observations can reliably identify a specified internal event
or failure. These are research objectives, not established model properties.

The architecture combines published mechanisms to create a tractable object of
study. Its exposed state boundaries make experiments possible; they do not
guarantee that features, routes, or iterations have human-readable meanings.
The small system is intended to help develop methods for understanding models
capable of opaque reasoning. Transfer to those models must be tested
separately.

## Three kinds of recurrence

Keep token position, full-sequence feedback pass, and tied-core iteration as
separate axes. They execute different computations and support different
claims.

| Axis | State and transition | Current status |
|---|---|---|
| Token-mixer memory | PKDA matrix, diagonal preconditioner, and convolution history advance along tokens; GQA retains prefix K/V | Implemented in every hybrid arm, including `base` |
| Latent feedback | FBT transfers the previous token column's payload through the current token's gate | Implemented in `fbt` and `df`; Jacobi passes train this causal dependency in parallel |
| Tied depth | A shared core repeatedly updates one column's residual state | Specified only in `df-loop`; not implemented or registered |

During the current repeated-prefill diagnostic, each pass recomputes its mixer
states from zero and consumes the preceding pass's shifted payload. During
sequential feedback decode, both mixer caches and the immediately previous
payload continue across generated tokens. A conclusion from one execution mode
does not automatically describe the other.

## What exists today

| Surface | What it exposes or tests | Limit |
|---|---|---|
| `DFModel.forward_column` / `ColumnOutput` | Top residual state, feedback payload, final source bank, and optional per-site/group route weights with source labels | An analysis interface, not a complete trajectory recorder or intervention framework |
| `scripts/route_report.py` | Plain/fused route summaries, source scales, null scales, and query geometry on checkpointed models | Observational; large route mass does not establish behavioral importance |
| `scripts/payload_swap.py` | Top-only, uniform, and forced-source payload enrichment, optionally one routing group at a time | A co-adapted same-checkpoint perturbation; top-only preserves the payload's top-state term |
| `scripts/fused_diagnostics.py` | Fused-pass minus pass-1 loss by position, fused-in-token surprise, and token frequency; gate, seed-decodability, and scale statistics; a 30-iteration self-composition trace | Repeated-prefill diagnostics on one checkpoint; conditioning on a position's own pass-1 loss selects on noise |
| `scripts/entry_sweeps.py` | Gate temperature, fused-seed scale, embedding bypass, zero and foreign payloads at the FBT entry | Out-of-distribution perturbations of a co-adapted entry; they bound dependence, not design alternatives |
| `scripts/feedback_followups.py` | Fused-block impulse response, payload-head ablation, split-validated pass mixtures, pass-1 versus fused-pass gradient alignment | The fused-then-plain boundary never occurs in training; mixture gain measures complementary information, not use |
| `scripts/compare_arms.py` | Two checkpoints on identical rows: per-token loss structure, predictor KL and agreement, mixtures, residual-stream CKA per source, payload redundancy with the readout | Paired observation; independently trained runs differ like two seeds, so per-token differences are not attributable to one package |
| `scripts/weight_divergence.py` | Angular movement of every shared parameter from the paired initialization and between two runs | Weight-space distance says nothing about function under fixed-radius normalized updates |
| `scripts/dense_feedback_continue.py` | Continued training of one snapshot with dense feedback passes or the plain-only control | A same-checkpoint intervention on the recipe, not a registered arm |
| `scripts/downstream_eval.py` | Harness-identical zero-shot tasks from the workspace `downstream` module in Standard or Fused mode, with per-document records for paired comparison | Specimen characterization; at screen scale the tasks resolve differences of a few points, not the sub-0.01-nat effects the factorial produces |
| `scripts/training_curves.py`, `scripts/analysis_figures.py` | Figures from run logs (paired validation and per-row training differences, matched compute, routing and contraction monitors) and from the JSON records above | Rendering only; no new measurement |
| `iterate_fused` | Per-iteration held-out loss and mean top-state update norm on fixed tokens | A payload-only repeated-prefill diagnostic; no general contraction proof |
| Trainer telemetry / `df watch` | Training health, validation, routing summaries, and recurrent dynamics | Operational monitoring, without labels for hidden reasoning |
| `df probe` and portable semantics | Numerical, causal, gradient, cache, and execution invariants | Engineering evidence, not a mechanistic explanation |

The next work includes a persistent trajectory format, general state patching,
controlled behavioral tasks, held-out causal replication, and learned-monitor
evaluation. None is a supported command merely because it is specified here.
Analysis must respect the precision and cache contracts of the trained model; a
convenient portable replay must be checked against the relevant CUDA path.

## Start with a behavioral phenomenon

Choose a small, externally scored behavior before deciding which activation
“means” something. Candidate tasks include retaining a controlled fact across
distractors, updating a latent variable after an explicit rule change, or
tracking a short sequence of operations. These are proposed task families, not
benchmarks the current checkpoints have passed. Pair examples so that surface
cues alone do not identify the answer or the experimental condition.

For a first study, choose one task, one state boundary, and one causal
contrast. Record ordinary task accuracy and the errors being explained.
Establish that the specimen can perform the behavior often enough for the
comparison to be informative. Compare successful and failed cases, and test
whether the chosen recurrent channel actually matters. PKDA memory or the
visible token prefix may suffice even when an FBT payload is present.

A failed task is not evidence that reasoning is hidden in another subspace. It
may mean the behavior has not been acquired. Conversely, task success does not
establish recurrent reasoning: the causal channel still needs testing. Any
targeted training or new curriculum belongs to a separately specified
comparison with its own controls and provenance.

## Trace a computation

A retained analysis should be a reproducible specimen record, containing:

- Exact code and vendor revisions, checkpoint identity/hash and original
  training contract, arm, geometry, seed, and cumulative token/pass-token cost.
- Token IDs, row/task addresses, expected outputs, evaluation split, and
  Standard/Soft/Fused mode; pass count, prefix masks, and jitter settings.
- State coordinates: token position, layer/cell, route site and source label,
  pass number, and core iteration if a future loop is used.
- State capture boundary, device/precision, cache initialization/continuation,
  deterministic choices, command, and raw output location.
- Baseline behavior, measured observables, intervention specification, and
  counterfactual outcomes, including failures and numerical tolerances.

Start with the actual column seed, completed block deltas, `h_top`, payload,
and mixer states. Use the reconstruction identity `h_top = seed + sum(completed
block deltas)` to check the source interpretation. Norms, distances, route
entropy, and low-dimensional projections can locate an effect. They do not
identify the computation on their own.

For matched replay, hold tokens, positions, pass/prefix choices, and cache
history fixed. When testing autoregressive consequences, allow generated tokens
to diverge but report that as a separate experiment: subsequent differences
then include feedback through changed text. For differentiable analyses, keep
that discrete schedule fixed and preserve the intended recurrent graph.

## Test the explanation causally

State the expected result before examining the confirmatory examples: which
state carries what information, when it affects the answer, and what a targeted
intervention should change. Useful next experiments include patching a payload
from a paired example, replacing one block contribution, or disrupting a mixer
state at a defined boundary. These need implementation and validation beyond
the current payload-enrichment sweep.

Use an identity patch as an execution control. Compare the targeted patch with
a matched unrelated donor or site, appropriate magnitude controls, and a
restoration/rescue when feasible. Donors must be aligned in position, mode,
shape, and history; a donor must not encode future information relative to the
patched position. A zero vector or shuffled row can create out-of-distribution
damage, so a loss increase alone does not identify the erased feature.

Distinguish changing route weights from changing source values. Distinguish
removing enrichment from removing the full payload. Distinguish loss of
behavior under disruption from recovering the behavior with an informative
state. These answer different questions about necessity, sufficiency, and
redundancy. A decodable variable can be present without being used.

Report effect sizes over paired examples and the range of outcomes, not only an
attractive trajectory. Separate exploratory selection from held-out
confirmation; compare across seeds/checkpoints before claiming a recurring
mechanism. A result on one checkpoint is still useful when explicitly scoped to
that specimen. The trained factorial is needed for claims about how adding a
package changes learning; patching alone does not supply that comparison.

## Study dynamics without calling stability reasoning

Track state movement and behavior together. Small updates can indicate a
settled representation, an unproductive fixed point, numerical resolution, or a
nearly unused feedback channel. Large updates can reflect purposeful change or
instability. Loss and task behavior distinguish some of these possibilities;
interventions are needed to establish the causal process.

The existing fixed-token fused-prefill trace is a first diagnostic. Future work
can compare perturbation propagation, recovery, and dependence on state
history. Report the observation interval and tested inputs. Neither finite
trajectories nor a decaying average norm proves a global contraction, a
completed reasoning process, or safe behavior.

The proposed `df-loop` adds iteration-indexed states and a different shared
cache transition. Its two-channel sequence trace must advance both payload and
write bank. It cannot reuse the present payload-only trace as if it tested the
loop's dynamics. See [depth-architecture.md](depth-architecture.md).

## Evaluate a monitor

Define a narrow target first: for example, a controlled task-state overwrite, a
known loss of retained information, or a trajectory that will enter a specified
failure condition. Labels should come from task construction, independent
outcomes, or controlled interventions, with the label's limits stated. Avoid
using the same activation threshold both to define an event and to “detect” it
unless the claim is explicitly just threshold reconstruction.

Specify what the monitor may see and when: visible tokens/output uncertainty,
selected internal states, or a limited history of internal observations. A
claim of early detection must evaluate predictions before the outcome or
intervention label is available. Fit representations, thresholds, and probes on
development data, freeze them, and evaluate on disjoint examples. Split by
underlying task/template as needed so near-duplicate trajectories do not leak
between training and evaluation.

Compare against an output-only baseline and simple internal-statistic
baselines. Report false positives, misses, precision/recall at the chosen
operating point, calibration where probabilities are used, and lead time for
early warnings. Include the event prevalence and monitoring cost. Test on
held-out templates, sequence lengths, seeds or checkpoints where available, and
interventions that break a convenient proxy while preserving the target.

A monitor can predict a failure without explaining its mechanism. A causal
explanation can be valid without enabling a cheap monitor. Report these as
separate achievements. Success on this organism is a method-development result;
deployment or transfer claims require evaluation on the target model and its
actual conditions.

## Deliverables and acceptance

The first useful deliverable is one replayable checkpoint analysis with a clear
behavioral contrast and a controlled causal intervention. Next comes
replication and, where the phenomenon supports it, a frozen monitor with an
honest error profile. A larger model is not a required deliverable.

Keep exploratory notes in [journal.md](journal.md). An accepted result in
[findings.md](findings.md) states its bounded claim, specimen, method,
controls, held-out evidence, uncertainty, and artifacts. A negative result or
an exposed monitoring failure can qualify. Route plots, engineering parity, and
pretraining metrics alone do not constitute an explanation of opaque reasoning.

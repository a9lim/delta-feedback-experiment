# Inspecting the model

Delta exposes its computation for checkpoint analysis and interventions.
General capture and readout tooling belongs to the sibling
`interpretability-experiments` workspace; training does not depend on it.

## State and traces

| Axis | State | Conditions |
|---|---|---|
| Token position | PKDA matrix/diagonal/convolution state and GQA prefix K/V | All |
| Latent feedback | Previous column's payload enters the next token's gate | `f`, `fl` |
| Core depth | Tied middle cells repeatedly update the column | `l`, `fl` |

A Jacobi pass recomputes mixer state from zero and consumes the prior pass's
shifted payload. Sequential feedback decode advances mixer caches and carries
the previous token's payload. Core iterations have separate mixer-cache
tracks. Label token, pass, and depth coordinates when comparing states.

`DeltaModel.forward_column` returns `ColumnOutput`: top residual, payload,
source bank and labels, expert balance loss/counts, and optional route/expert
weights. The sources obey `h_top = seed + sum(completed cell deltas)`.
`want_weights=True` exposes MHDB source weights separately from sparse routed
expert weights `[B,T,n]`; the always-active shared expert is outside that axis.
`forward_mtp` exposes the auxiliary block over the column's own rows; its
last row has no second token and carries no training weight.

## Tools

| Script | Output |
|---|---|
| `scripts/route_report.py` | Plain/fused source mass and entropy by site/group, source/null scale, query geometry |
| `scripts/payload_swap.py` | Trained, top-only, uniform, and forced-source payload enrichment; optional single-group intervention |
| `scripts/depth_trace.py` | Held-out loss, core updates, and route mass over fixed depths `1..r_max` |
| `scripts/downstream_eval.py` | Workspace zero-shot tasks in Standard, Soft, or Fused mode |
| `scripts/training_curves.py` | Main/auxiliary training and validation CE, gradient norm, and throughput from current logs |

Checkpoint scripts use `delta_feedback_experiment.analysis` to load current
snapshots with trainer numerics, including CUDA BF16. Commands are in
[operations.md](operations.md#inspect-a-checkpoint). JSON records and figures
under `figures/` can be regenerated from snapshots or logs.

The trainer also records fixed-token payload self-composition and tied-depth
summaries in `delta watch`. A small update indicates little state movement;
loss and interventions are needed to determine why.

## Interpreting results

A route weight is a mixing coefficient. Account for source values and learned
null magnitude before assigning importance. Top-only payload ablation removes
routed enrichment while retaining `h_top`. Expert gate mass and selection
frequency alone do not establish specialization.

Match tokens, rows, positions, pass/depth choices, precision, and cache history
for replay. Keep expert-selection biases fixed. Use an identity patch to
check execution, and matched donor/magnitude controls to distinguish a
specific effect from generic damage. Donors must contain no future information
relative to the patched position. Autoregressive comparisons allow text to
diverge and measure that additional feedback as part of the outcome.

MTP predicts a second token using the payload and the ground-truth next-token
embedding, with its own PKDA memory. Its accuracy measures that auxiliary
predictor. Compare ordinary next-token behavior and payload interventions to
assess usefulness for generation; recoverability from the payload alone does
not show how the main column uses it.

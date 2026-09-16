# Inspecting the model

Delta exposes its computation for checkpoint analysis and interventions.
General capture and readout tooling belongs to the sibling
`interpretability-experiments` workspace; training does not depend on it.

## State and traces

| Axis | State | Conditions |
|---|---|---|
| Token position | PKDA matrix/diagonal/convolution state and GQA prefix K/V | All |
| Latent feedback | Previous column's payload is fused with the next token's embedding | `f`, `fl` |
| Looped depth | The whole column re-runs from its own payload | `l`, `fl` |

A Jacobi pass recomputes mixer state from zero and consumes the prior pass's
shifted payload. Sequential feedback decode advances mixer caches and carries
the previous token's payload. Looped columns have separate mixer-cache
tracks at every layer. Label token, pass, and column coordinates when
comparing states.

`embed_tokens(tokens)` returns token features in the residual dtype, the
tied row times `1/BASE_NORMAL_INIT_STD`, so unit RMS at initialization;
they enter the column only through the shared fusion. `plain_seed(e)` is the
seed of pass 1, plain-prefix positions, and Standard decoding: the fusion of
`e` with the learned blank payload `blank_payload`, so a feedback position
differs from a plain one only by `W_p (p_(t-1) - p_0)`. Replacing a payload
with the blank is therefore an in-distribution ablation, and patches to plain
seeds act either on `e` before fusion or on the projected seed.
`embed_tokens.weight` is the tied embedding/classifier matrix, read raw by
the classifier. Payloads are `payload_norm(h_top + routed)`: unit RMS times
a learned gain initialized to one.

`DeltaModel.forward_column` returns `ColumnOutput`: top residual, payload,
source bank and labels, expert balance loss/counts, and optional route/expert
weights. The sources obey `h_top = seed + sum(completed cell deltas)`.
`want_weights=True` exposes MHDB source weights separately from sparse routed
expert weights `[B,T,n]`; the always-active shared expert is outside that axis.
`forward_iterations` runs every column of one position range and returns
them in order; `multipass` returns `[pass][column]`.
Outputs from `multipass` also expose `fused_input`: the shared projected tensor
supplied to MTP and the following feedback pass; a following looped column's
seed is the same jittered payload fused with the position's own embedding.
`forward_mtp_fused`
runs the independent auxiliary block directly on that tensor; `forward_mtp`
accepts payloads and next-token IDs and computes their fusion with token
embeddings first. The
auxiliary block's last row has no second token and carries no training weight.

## Tools

| Script | Output |
|---|---|
| `scripts/route_report.py` | Plain/fused source mass and entropy by site/group, source/null scale, query geometry |
| `scripts/payload_swap.py` | Trained, top-only, uniform, and forced-source payload enrichment; optional single-group intervention |
| `scripts/depth_trace.py` | Held-out loss, top-state updates, and route mass after every column, default `1..3` |
| `scripts/downstream_eval.py` | Workspace zero-shot tasks in Standard, Soft, or Fused mode |
| `scripts/training_curves.py` | Main/auxiliary training and validation CE, gradient norm, and throughput from current logs |

Checkpoint scripts use `delta_feedback_experiment.analysis` to load current
snapshots with trainer numerics, including CUDA BF16. Commands are in
[operations.md](operations.md#inspect-a-checkpoint). JSON records and figures
under `figures/` can be regenerated from snapshots or logs.

The trainer also records fixed-token payload self-composition and per-column
depth summaries in `delta watch`. A small update indicates little state movement;
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

MTP predicts a second token using the same concat-linear fusion of payload
and raw ground-truth next-token embedding that feedback consumes.
The payload receives its learned RMSNorm at the writer. Training jitter is
sampled directly in payload units, with default range `[-0.02,0.02]`; fusion
adds it without another scale factor.
Training shares the fused tensor between
both consumers; diagnostics and evaluation disable jitter. The auxiliary
block retains its own PKDA memory, so its
accuracy measures that auxiliary predictor. Sharing the fusion aligns the
input interface but does not establish useful feedback. Compare ordinary
next-token behavior and payload interventions to assess usefulness for
generation; recoverability from the payload alone does not show how the main
column uses it. The projection can supply token-derived features even when
the payload contribution vanishes, so assess payload use through interventions.

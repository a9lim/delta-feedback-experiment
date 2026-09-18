# Inspecting the model

This guide covers checkpoint APIs, diagnostics, and interventions.
[Architecture](architecture.md) defines the tensors and recurrence;
[the recipe](design.md#evaluation) defines evaluation metrics. General
capture and readout tooling belongs to the sibling
`interpretability-experiments` workspace; training does not depend on it.

## Checkpoint and tensor APIs

[analysis.py](../delta_feedback_experiment/analysis.py) provides the common
entry points:

| API | Contract |
|---|---|
| `load_checkpoint(path, device=None)` | Load a current snapshot into an evaluation-mode model; return `(model, saved_args)` with cumulative `step`. Prepare working readout copies and discard optimizer state. |
| `autocast(device)` | Match trainer evaluation numerics: BF16 activations on CUDA, ordinary execution elsewhere. |
| `logprob_chunks(model, h_top, chunk=256)` | Yield `(start, log_softmax)` over position chunks, avoiding a full `[B,T,V]` allocation. |
| `token_ce(model, h_top, targets)` | Return main-head per-token CE `[B,T]`. |
| `fused_inputs(model, e, payload, prefix=1)` | Build the next feedback pass's seeds from prior-pass payloads, shifting them right and restoring the blank-fused prefix. `e` is the output of `model.embed_tokens`. |

Use model entry points to preserve the trained fusion and readout conventions:

| API | Result |
|---|---|
| `embed_tokens(tokens)`, `plain_seed(e)` | Token features and their blank-payload fusion, respectively; the embedding's `.weight` is the tied classifier parameter. |
| `forward_column(x, want_weights=True)` | One `ColumnOutput` with top residual, payload, source bank, routing data, expert balance, and assignment counts. |
| `forward_iterations(x, iterations=...)` | Every looped column for one position range, returned in order. Later columns re-enter through `loop_seed(payload)`, the payload's fusion with `blank_embedding`. |
| `multipass(model, tokens, n_passes, ...)` | Outputs indexed `[pass][column]`; `tokens` has shape `[B,T+1]` and the trunk executes `T` positions. |
| `forward_mtp_fused(fused_input)` | The auxiliary block on an already projected fusion tensor. `forward_mtp(payload, next_tokens)` computes that fusion first. |

`ColumnOutput.h_top` is the pre-final-norm residual; `payload` may be absent
only when explicitly skipped. `sources` contains the seed and completed
cell deltas, with labels in `source_names`. With `want_weights=True`,
`route_weights[site]` has shape `[sources,B,T,groups]`, and
`route_source_names[site]` labels that exact axis, including the learned null
and any partial-cell source. The labels vary by site; align by name, not by
column index. Sparse `expert_weights[site]` has shape `[B,T,routed_experts]`;
the always-active shared expert is outside that axis.

`multipass` attaches each column's `fused_input`, consumed by MTP and, for
the last column of a pass, the following feedback pass. A bare
`forward_column` leaves it unset. To patch a payload, distinguish its
post-writer value from the projected fusion tensor: those interventions
affect different interfaces. Replacing an incoming payload with
`blank_payload` restores the plain seed at that position; top-only payload
ablation instead removes routed enrichment and retains the normalized top
residual. They test different hypotheses. A loop re-entry differs: its seed
is `fuse(payload, blank_embedding)` with no token term, so the payload is the
later column's only view of its position. Patching it sets that column's
whole input, and the blank payload there leaves a constant, token-free seed
rather than a plain one. The informative loop question is what the payload
holds beyond the current token and the draft readout, not whether it is read.

## Diagnostics and tools

| Tool | Output and scope |
|---|---|
| [`route_report.py`](../scripts/route_report.py) | Plain/fused routing at the configured final column: source mass, per-token and mean-distribution entropy, cross-group divergence, source/null magnitudes, and query geometry. |
| [`payload_swap.py`](../scripts/payload_swap.py) | Fused CE with trained, top-only, uniform, or forced-source payload enrichment. `--head` restricts forced-source cases to one routing group; top-only and uniform still replace all groups. Choose `--rows` divisible by `--micro-rows`, because the script averages microbatch means equally. |
| [`depth_trace.py`](../scripts/depth_trace.py) | Held-out loss and top-state update at every column, default `1..3`, in plain and eligible fused modes. Route mass is sampled from the first microbatch in plain mode. |
| [`delta eval`](operations.md#downstream-evaluation) | Workspace zero-shot tasks with Standard, Soft, or Fused teacher-forced continuation scoring; the queue runs it after a finished schedule. |
| [`training_curves.py`](../scripts/training_curves.py) | Main/auxiliary CE, gradient norm, and throughput from logs; repeated steps retain their final record. |

[Operations](operations.md#inspect-a-checkpoint) gives commands. Checkpoint
tools use the loader and evaluation numerics above. JSON and figures under
`figures/` are generated outputs and remain untracked.

The payload sweep overrides the writer wherever it runs, including loop
seeds under `fl`; its loss measures that combined intervention rather than
one isolated feedback edge.

Two model-level diagnostics also appear in trainer telemetry:

- `iterate_fused` keeps tokens fixed, starts from a plain pass, and repeatedly
  composes feedback with prefix length 1. It reports main-head CE and mean
  tokenwise `||h_new-h_old||_2` after each composition, at the requested
  fixed loop depth.
- `depth_trace` reports main-head CE and the same top-state update norm at
  each column; column 1 compares against the input seed. In fused mode its
  first pass uses the requested column cap, then its second pass is read
  after every column. Changing the cap also changes the donor payload, so
  fused traces from different caps are not the same depth sweep.

Downstream **Soft** scoring keeps each item's context plain and applies
Jacobi feedback to the scored continuation. **Fused** scoring uses prefix
length 1. `--passes k` means `k` feedback passes after the initial plain
pass; it scores the last pass. These use fixed ground-truth tokens, so
finite-pass Soft scores are not a full sequential feedback decode. Their
causal results agree with sequential Soft on the first `k` continuation
tokens after `k` feedback passes. Each pass uses the snapshot's configured
loop count.

Training curves need the objective's [recurrence weighting](design.md#objective).
The plotted `ntp - pass1` remainder includes loop-only and joint terms under
`fl`, even though the plot labels it feedback. Combined training CE has a
different total weight across recurrence shapes; compare matching `(k,r)`
or use per-head held-out CE. The plotted gradient norm is the full norm;
training does not clip it.

## Interpreting interventions

A route weight is a mixing coefficient, not a causal importance score.
Read it alongside source values and learned null magnitude. Entropy of the
mean route distribution can hide sharp routing that changes across tokens;
per-token entropy distinguishes that from diffuse routing at every token.
Expert gate mass and selection frequency alone do not establish specialization.

Label token position, feedback pass, and loop column in every comparison.
Match tokens, rows, precision, and cache history; keep expert-selection
biases fixed. Full passes restart mixer state, while sequential decode
advances it on a separate track for each loop column. An identity patch
checks the intervention path. Matched donor and magnitude controls help
separate a specific effect from generic disruption; donors must contain
no information from the patched position's future. Generated text may
diverge after a patch, and that downstream divergence is part of the effect.

A small update norm establishes little movement on the measured inputs;
it does not establish useful computation, convergence on other inputs, or
a contraction bound. Read loss and interventions alongside the update.
Likewise, MTP accuracy measures an auxiliary predictor with its own PKDA
memory and a ground-truth next-token input. Its shared fusion interface
does not establish that the main column uses payload information. Compare
ordinary next-token behavior with payload ablations, swaps, and controls;
the fusion can carry token-derived features even when its payload
contribution is absent.

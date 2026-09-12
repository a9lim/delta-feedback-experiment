# AGENTS.md

This repository grows a recurrent language model for later interpretability
and monitoring work. It is an experimental sandbox: train variants, inspect
them, and revise the architecture or recipe. No result needs to pass a
predeclared gate before it can inform the next experiment.

## Repository map

- `delta_feedback_experiment/`: model, training, data, analysis, and CLI.
- `tests/`: executable numerical, causal, state, and lifecycle contracts.
- `scripts/`: maintained checkpoint analysis and plotting workflows.
- [README.md](README.md): purpose and current scientific state.
- [docs/architecture.md](docs/architecture.md): implemented equations and state.
- [docs/design.md](docs/design.md): data and training recipe.
- [docs/scaling.md](docs/scaling.md): implemented geometry presets and accounting.
- [docs/operations.md](docs/operations.md): operator commands and CUDA execution.
- [docs/interpretability.md](docs/interpretability.md): analysis interfaces and interpretation.
- [references/refs.yaml](references/refs.yaml): source attribution.

## Editing

Keep one current implementation and document what it does. Remove superseded
code, stale outputs, historical qualification records, and unimplemented
specifications. Git retains history. Update code, tests, CLI help, and affected
docs together when a contract changes; do not duplicate model details here.
Generated runs, logs, figures, token stores, and fetched papers are untracked.

Preserve the model's causal and numerical contracts: differentiable feedback
and core iterations, telescoping block deltas, paired initialization and keyed
randomness, FP32 accumulated gradients, and exact resume of current snapshots.
Checkpoint v33 and the tokenizer identity define the current stored format.
Use the existing readers and CLI for checkpoints and run lifecycle actions.

## Runtime

The workspace root package owns telemetry, schedules, snapshots, monitoring,
and the spool. CUDA kernels come from its editable FLA and CCE vendor forks.
Use Python 3.13 and the project's declared dependency extras; machine-wide
constraints own Jobe's PyTorch/CUDA installation.

Before touching Jobe, inspect `delta status`, the active log, and GPU ownership.
Keep GPU jobs serial and preserve active data builds. Run focused checks while
editing and the compact `delta probe` on the final candidate for cross-cutting
changes; hardware-specific claims need the CUDA path. The queue starts
training directly. Keep cyclic Python garbage collection outside train/eval
graph capture.

The queue stores arguments and refreshes the checkout before its next job.
Source edits do not stop the active child. Distributed training is not
implemented. Prime spending requires a9's decision.

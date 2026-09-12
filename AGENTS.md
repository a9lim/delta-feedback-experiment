# AGENTS.md

This repository trains a recurrent model for interpretability experiments.
No result needs to pass a predeclared gate before it can inform the next experiment.
Keep one current implementation. Remove superseded code, compatibility paths,
stale artifacts, and historical qualification records; Git retains history.
Update affected code, CLI help, tests, and docs together.

## Map

- `delta_feedback_experiment/`: model, training, data, analysis, and CLI.
- `tests/`: small portable numerical, causal, state, and lifecycle contracts.
- `scripts/`: current checkpoint analysis and training plots.
- `docs/`: architecture, recipe, scaling, operations, and analysis.
- `references/refs.yaml`: concise mechanism sources.

## Contracts

Preserve differentiable feedback/core iterations, telescoping cell deltas,
paired initialization and keyed randomness, FP32 accumulated gradients, and
exact resume of checkpoint v34 with the current tokenizer. Use the existing
checkpoint readers and run lifecycle CLI. Generated data, logs, snapshots,
figures, and fetched papers remain untracked.

## Runtime and verification

Use Python 3.13 and the declared extras. The parent workspace owns telemetry,
schedules, snapshots, monitoring, the spool, and editable FLA/CCE forks.
Interpretation dependencies belong to `interpretability-experiments`.

Before using Jobe, inspect `delta status`, the active log, and GPU ownership.
Keep GPU jobs serial and preserve active data builds. Run focused tests while
editing, the fast default `pytest` suite for a final portable check, and
`delta probe` for changes that need CUDA execution. The probe is a small CUDA
smoke. Queue startup runs training directly. Keep cyclic Python garbage
collection outside train/eval graph capture.

The queue stores arguments and refreshes the checkout before each job;
source changes do not stop an active child. Distributed training is absent.
Prime spending requires a9's decision.

# AGENTS.md

This repository grows a small recurrent language model with a latent channel
between token columns, so that a later interpretability study has an organism
worth studying. The organism is not the object of study yet. This is a
sandbox: train variants, look inside, change the recipe or the architecture,
write down what happened. Nothing is pre-registered and no result needs to
clear a gate before it can be used to decide what to try next.

One `DeltaModel` family, addressed by condition letters: `a` makes three of
every four attention layers PKDA, `r` adds MHDB block-delta reads, `f` adds
FBT feedback between token columns, and `l` makes the middle cell a tied core
iterated a drawn number of times per column. `--condition arfl` is the full
built stack and `arf` the flat column; the empty condition is the plain
decoder. Three full-schedule specimens exist
on Jobe (`screen-delta-ar-s1`, an `ar` run; `screen-delta-arf-s1-highLR` and
`screen-delta-arf-s1`, `arf` runs); the current read of what they show is
in `docs/findings.md`.

## Documents

- [README.md](README.md): what we are growing, what ready looks like, where
  things stand, and the doc map.
- [docs/design.md](docs/design.md): conditions, geometry, data, schedule, feedback
  passes, evaluation modes, and the recipe knobs.
- [docs/architecture.md](docs/architecture.md): the four letters and the
  column: equations, state, initialization, precision, and optimizer
  ownership; `L` specified and unbuilt.
- [docs/interpretability.md](docs/interpretability.md): the analysis scripts,
  what each shows, and the shape of the future study.
- [docs/findings.md](docs/findings.md): the distilled current picture of the
  specimens. [docs/journal.md](docs/journal.md): dated working notes.
- [docs/operations.md](docs/operations.md): operator commands.
  [docs/runtime-qualification.md](docs/runtime-qualification.md): the CUDA
  execution path and its numerical evidence.
- [docs/scaling.md](docs/scaling.md): the screen and the flagship: geometry,
  accounting, budgets, the loop's measured cost, and the longer and larger
  recipes, worked out but not scheduled.
- [docs/literature.md](docs/literature.md) and
  [references/refs.yaml](references/refs.yaml): sources and departures.
- [figures/README.md](figures/README.md): the figure directories.

Keep the docs current-only: they describe what the code does and what we have
seen, not the history of either. When a contract changes, update code, tests,
CLI help, and the affected pages together. Delete superseded material rather
than keeping aliases or compatibility notes for hypothetical readers.

## How the model fits together

These describe the code as it stands. Any of them can change; when one does,
change it everywhere at once.

- One `DeltaModel` and `CONDITION_LETTERS` define the family. A condition is the
  canonical string `parse_condition` returns (letters in `arfl` order) and
  `ModelConfig.condition` renders it back; every subset of `arfl` builds. Every dense attention layer is gated NoPE
  GQA. Under `a` the trunk is `[PKDA, PKDA, PKDA, gated global GQA] x 3`;
  without `a` it is twelve such layers. The flagship in
  `scaling.md` is the same architecture at six cells and width 1,536.
- PKDA runs the literal recurrence on CPU/MPS and the workspace FLA fork's
  chunk and recurrent kernels on CUDA, with FP32 recurrent-matrix and
  preconditioner boundaries. CUDA training does not fall back to the
  sequential path.
- MHDB sources are the column seed, the completed four-layer block deltas, and
  at most one current-cell partial; every site prepends its own learned
  zero-initialized null. Routers use a zero-initialized width-`D` query,
  full-width RMS key statistics, raw values, and one softmax per contiguous
  feature group, with `kv_heads` groups. Routing is a transient pre-norm read
  and leaves the telescoping residual identity intact.
- `f` seeds a fused position with the FBT payload-value/token-gate fusion and
  emits `payload_norm(h_top)`; with `r` as well, the fused seed is the first
  non-null routing source and the payload is
  `payload_norm(h_top + route(null, seed, block deltas))`.
- Feedback passes are causal and differentiable across passes; the payload is
  never detached.
- Under `l` the first cell is the prelude, the last the coda, and the cells
  between them one tied core run `r` times per column, `r` drawn once per
  step from a log-normal Poisson with mean 4 and cap 8. Each core cell keeps
  its own block delta across iterations, the screen's one core cell measuring
  from the prelude output; mixing is same-depth (iteration `i` reads
  iteration-`i` writes, one decode cache track per iteration), and the
  payload is the only channel between columns. At `r = 1` a looped condition
  is its unlooped condition exactly, at any cell count.
- NorMuonH owns ordinary hidden matrices, initialized `Normal(0, 1/sqrt(d_in))`
  with fixed realized Frobenius radii; NAdam owns gates, embeddings, norms,
  routing parameters, PKDA controls, convolutions, and vectors in one group.
  No weight decay. The global FP32 gradient is clipped to norm 10.0 before
  both steps.

## Useful bookkeeping

- Conditions trained with the same seed and data seed share tokenizer, stream,
  row order, schedule, optimizer, batch geometry, and keyed feedback
  randomness; conditions on the same trunk letter pair every parameter they
  share byte-identically, each letter's private weights from that letter's own
  stream. Two such runs can be compared token by token.
- A `k`-pass batch costs `k` transformer evaluations, and under `l` a pass
  executes `2 + r` cells. Report pass-tokens and cell-tokens beside predicted
  tokens; equal steps are matched data, not matched compute.
- Pass-1 validation is the common number. Conditions with `f` also report the fused
  number and the self-composition trace.
- Independently trained runs differ like two seeds even when paired, so a
  small per-token difference between two runs is a mean shift on a wide
  spread. Same-checkpoint interventions resolve much finer effects.

## Runtime and Jobe

- Telemetry, schedules, run/snapshot addressing, checkpoint staging, monitor
  serving, and spool orchestration come from the workspace root package.
- Jobe is the single-GPU CUDA surface. Keep GPU jobs serial; the captured
  graph pool reserves about 23.0 GiB. `docs/runtime-qualification.md` holds
  the PyTorch 2.14 / CUDA 13.2 evidence. Keep cyclic Python garbage
  collection outside train/eval graph capture.
- Snapshots are v25. Only v25 resumes; v16 through v24 stay readable for
  evaluation and forks, and every specimen records its condition as letters.
- Before touching Jobe, look at `delta status`, the active log, and GPU
  ownership. The queue stores arguments rather than Git state; a worker
  refreshes before the next job and runs the current checkout's probe.
- Prime DDP and the larger reference are worked out in `docs/scaling.md` but
  not implemented. Prime time is real money; a9 decides when to spend it.

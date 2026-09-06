# AGENTS.md

This repository develops a small recurrent model organism for interpretability.
It synthesizes already-published mechanisms so we can inspect latent
computation, test causal explanations, and develop methods for monitoring
models capable of opaque reasoning. Capabilities serve task acquisition and
experimental usefulness; advancing the capabilities frontier is not the goal.

The implemented five-arm pretraining study produces controlled specimens:
`{base, mhdb, fbt, df}` is the primary factorial and pure-GQA `vanilla` is the
external trunk control. No completed comparison under the current screen
contract or accepted scientific finding is recorded. Diagnostic checkpoints may
support explicitly scoped exploratory analysis with original provenance. Do not
present the organism as already exhibiting understood reasoning or validated
monitoring, or infer live Jobe availability from documentation.

## Documents

- [README.md](README.md) is the research purpose, organism overview, and entry point.
- [docs/design.md](docs/design.md) owns the controlled study, data, schedules,
  evaluation, evidence, and research decision gates.
- [docs/interpretability.md](docs/interpretability.md) owns behavioral,
  tracing, causal-intervention, and monitoring protocols and implementation status.
- [docs/architecture.md](docs/architecture.md) is the sole model, state,
  parameter, and optimizer contract, with small and optional larger geometry.
- [docs/depth-architecture.md](docs/depth-architecture.md) specifies the
  unimplemented `df-loop`; it is not registered until its adoption list lands.
- [docs/operations.md](docs/operations.md) owns operator commands;
  [docs/runtime-qualification.md](docs/runtime-qualification.md) owns numerical
  and execution qualification.
- [docs/scaling.md](docs/scaling.md) preserves optional longer/larger recipes,
  gated on a named interpretability need, implementation, and spend approval.
- [docs/literature.md](docs/literature.md) maps published components to the
  organism; [references/refs.yaml](references/refs.yaml) owns primary-source roles.
- [docs/findings.md](docs/findings.md) and [figures/README.md](figures/README.md)
  hold accepted scientific findings and their figures.
- [docs/journal.md](docs/journal.md) is disposable active scratch space.

Keep active documentation current-only. Remove superseded configurations,
chronology, compatibility aliases, speculative extensions, and unearned claims.
Update code, tests, CLI help, and all affected documents together when a
contract changes.

## Nonnegotiable model contract

- One `DFModel` and the five exact `ARMS` define the family. Do not add parallel
  implementations or aliases.
- Every hybrid screen arm is exactly
  `[PKDA, PKDA, PKDA, gated global GQA] x 3`; `vanilla` is twelve-layer RoPE
  GQA. The optional larger reference is the corresponding six-cell hard-DF
  architecture in `architecture.md`; it is not the project objective.
- PKDA uses the literal portable recurrence off CUDA and the workspace FLA
  fork's chunk and recurrent kernels on CUDA. Recurrent matrix and
  preconditioner boundaries are FP32. Never silently use the sequential
  fallback for CUDA training.
- MHDB sources are the actual column seed, completed four-layer block deltas,
  and at most one current-cell partial. Individual branch deltas are not
  sources. Every site prepends its own learned zero-initialized null.
- Each router uses a zero-initialized width-`D` query, full-width RMS key
  statistics, raw values, and one source softmax per contiguous feature group;
  the group count equals `kv_heads`. Routing is a transient pre-norm read and
  never changes the telescoping residual identity.
- Hard DF uses the FBT asymmetric payload-value/token-gate fusion as its MHDB
  seed. Its payload normalizes the sum of the top state and a routed mixture over
  the null, seed, and completed block deltas.
- Multi-pass feedback stays causal and differentiable across passes. Never
  detach the payload to solve memory pressure.
- NorMuonH owns ordinary hidden matrices and uses the released NorMuon
  Nesterov blend before orthogonalization. Its matrices initialize from
  `Normal(0, 1/sqrt(d_in))`; their realized initial Frobenius radii remain fixed.
  NAdam owns semantic-scale gates, embeddings, norms, routing parameters, PKDA
  controls, convolutions, and vectors in one parameter group. No group uses
  weight decay. Clip the global FP32 gradient to norm 10.0 immediately before
  both steps.

## Evidence discipline

- Registered arms share tokenizer, stream, row order, schedule, optimizer,
  batch geometry, and keyed feedback randomness. Hybrid trunk weights are
  paired across all four factorial cells; factor-private weights pair across
  their parent and `df`.
- Report exact parameters, predicted tokens, and pass-tokens. A `k`-pass batch
  costs `k` transformer evaluations; equal steps are not matched compute.
- Attribute MHDB, FBT, and their interaction only on the Jobe factorial.
  Prime's `{base, df}` pair tests the complete package only. Report
  `vanilla - base` separately as a whole-trunk contrast.
- Pass-1 validation is the common metric. Feedback arms also report the fused
  metric and contraction trace. Routing summaries are diagnostics, not wins.
- Training-effect claims require completed registered comparisons. A scoped
  same-checkpoint causal finding requires reproducible interventions, controls,
  and held-out confirmation; it does not require an architecture win.
- Routing weights and decodable features do not establish causal importance.
  Empirical settling does not prove contraction, reasoning, or safe behavior.
  The training dashboard is not a validated monitor of hidden computation.
- Monitoring claims need an independent target, declared observation budget,
  held-out evaluation, calibration/error reporting, and shift tests. Treat
  transfer to larger reasoning models as a separate empirical question.
- Prefer the smallest specimen that answers the question. Useful null or
  negative results can qualify; capabilities improvements alone do not.
  Keep engineering qualification separate from scientific findings.

## Runtime and scale boundary

- Import telemetry, schedules, run/snapshot addressing, checkpoint staging,
  monitor serving, and spool orchestration from the workspace root package.
- Jobe is the authoritative single-GPU CUDA surface. Keep screen jobs serial;
  the qualified DF graph pool reserves about 23.0 GiB. See
  `docs/runtime-qualification.md` for the PyTorch 2.14 / CUDA 13.2 evidence.
  Keep cyclic Python garbage collection outside train/eval graph capture.
  New snapshots are v23;
  only v23 resumes, and v16-v22 stay readable for evaluation and forks.
- Before operating Jobe, inspect `df status`, the active log, and GPU ownership.
  The queue stores arguments rather than Git state; a worker refreshes before
  the next job and runs the current checkout's probe.
- The implemented local program is the two-seed 25x Jobe screen plus a fresh
  single-process 400x schedule. Prime DDP and the exact larger reference remain gated
  on pinned direct-HF data materialization with byte-checksum parity,
  distributed parity, durable artifacts, memory/throughput, checkpoint
  portability, and restart tests.
- Larger runs require a named interpretability question that the smaller
  specimen cannot resolve. The larger reference also requires an admissible
  Jobe factorial, a fresh Prime `{base, df}` result, stable long-horizon
  recurrence, the exact implementation gate, and explicit spend confirmation
  from a9. A useful small organism is a valid endpoint.

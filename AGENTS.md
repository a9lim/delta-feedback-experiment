# AGENTS.md

This repository owns a five-arm pretraining study of MHDB depth routing and FBT
latent recurrence on a common PKDA/gated-GQA hybrid. The primary factorial is
`{base, mhdb, fbt, df}`; pure-GQA `vanilla` is the external trunk control.

No run is complete under the current screen contract. Jobe is between runs;
existing diagnostic checkpoints are engineering inputs, not registered
comparisons. Keep
[docs/findings.md](docs/findings.md) empty of architecture claims until a
matched comparison is admissible.

## Documents

- [README.md](README.md) is the concise operator entry point and live status.
- [docs/architecture.md](docs/architecture.md) is the sole flagship model,
  state, parameter, and optimizer contract.
- [docs/design.md](docs/design.md) owns comparisons, data, schedules,
  execution, evaluation, scaling, and gates.
- [docs/depth-architecture.md](docs/depth-architecture.md) specifies the
  unimplemented depth-recurrent `df-loop` arm; nothing in it is registered
  until its adoption list lands.
- [docs/findings.md](docs/findings.md) contains accepted scientific findings
  only; engineering tests, throughput, isolated checkpoints, and plans do not
  belong there.
- [docs/journal.md](docs/journal.md) is disposable active scratch space.
- [references/refs.yaml](references/refs.yaml) records current primary-source
  roles; [figures/README.md](figures/README.md) indexes accepted-result figures.

Keep active documentation current-only. Remove superseded configurations,
chronology, compatibility aliases, speculative extensions, and unearned
claims. Update code, tests, CLI help, and all affected documents together when
a contract changes.

## Nonnegotiable model contract

- One `DFModel` and the five exact `ARMS` define the family. Do not add parallel
  implementations or aliases.
- Every hybrid screen arm is exactly
  `[PKDA, PKDA, PKDA, gated global GQA] x 3`; `vanilla` is twelve-layer RoPE
  GQA. The flagship is the corresponding six-cell hard-DF architecture in
  `architecture.md`.
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
  seed. Its payload is the normalized top state plus a routed mixture over the
  null, seed, and completed block deltas.
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
- Accept claims only from completed registered comparisons or clearly labeled
  same-checkpoint interventions. Keep engineering qualification separate.

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
  single-process 400x schedule. Prime DDP and the exact flagship remain gated
  on pinned direct-HF data materialization with byte-checksum parity,
  distributed parity, durable artifacts, memory/throughput, checkpoint
  portability, and restart tests.
- Flagship promotion requires an admissible Jobe factorial, a fresh Prime
  `{base, df}` result, stable long-horizon contraction, the exact implementation
  gate, and explicit spend confirmation from a9.

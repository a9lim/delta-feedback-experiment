# AGENTS.md

This repository owns a from-scratch pretraining factorial over two innovation
packages. The architecture package combines multi-head routing over
within-column residual deltas with gated GQA; the recurrence package is
full-bandwidth latent feedback between adjacent token columns. The hard hybrid
is `df`; `mhdb` and `fbt` are its parent-package deletions; `vanilla` is the
shared baseline; `df_soft` is an optional-channel diagnostic.

The 223–231M screen campaign is active. No matched scientific comparison is
complete, and [docs/findings.md](docs/findings.md) must remain empty of
architecture claims until one is admissible.

## Documentation contract

- [README.md](README.md) is the concise operator entry point and live status.
- [docs/design.md](docs/design.md) is the sole architecture, training,
  evaluation, scale-plan, and promotion contract.
- [docs/findings.md](docs/findings.md) contains accepted experiment findings
  only. Do not put unit tests, kernel parity, smoke behavior, throughput,
  isolated checkpoints, plans, or predictions there.
- [docs/journal.md](docs/journal.md) is disposable working space. Clear it when
  notes graduate or become inactive; Git history is the archive.
- [references/refs.yaml](references/refs.yaml) records the primary sources and
  only the roles needed by the current design.
- [figures/README.md](figures/README.md) indexes committed figures that support
  accepted findings. Generated diagnostics remain ignored until promoted.

Keep every active document current-only. State the present contract directly;
do not retain chronology, provenance narrative, superseded configurations,
legacy aliases, decision journals, or future extensions outside the defined
experiment. Update code, tests, CLI help, and documentation together when the
contract changes.

## Architecture invariants

- The model family is one plain-PyTorch `DFModel` configured by the five exact
  arm names in `ARMS`; do not add parallel model implementations or aliases.
- Screen and token-ladder `mhdb`, `df`, and `df_soft` use gated GQA as part of
  the architecture package; `vanilla` and `fbt` use ungated GQA. Every gate is
  a bias-free projection of the attention input followed by an elementwise
  sigmoid on the concatenated attention output before output projection and
  residual-branch scaling. Flagship global GQA uses the same gate semantics.
- Every scale uses MHDB sources: the column's actual input seed, one delta per
  completed four-layer cell, and at most one aggregate partial delta for the
  current cell. Individual branch deltas are not sources. Every router prepends
  its own learnable zero-initialized null. Routing is always a transient
  pre-norm read, so the seed-plus-block-deltas decomposition of the current
  residual remains exact.
- Each routing site uses a zero-initialized width-`D` query, a learnable
  full-width RMS key normalization, raw values, and one source softmax per
  contiguous feature group. The number of routing groups equals `kv_heads`.
- Hard DF fuses the shifted payload with the token embedding through the FBT
  asymmetric GLU, then uses the fused input as the MHDB seed. Its payload is
  the normalized top state plus a routed mixture over the null, seed, and this
  column's completed block deltas.
- `df_soft` keeps the plain token embedding as the stream seed and exposes the
  shifted payload as a masked standing source. Its payload router uses the same
  null, seed, and completed-block bank as hard DF.
- Multi-pass feedback is causal, differentiable across passes, and uses the
  shared prefix-mixin and jitter streams. Do not detach the payload to solve a
  memory problem; checkpoint blocks or coarsen the source representation.

## Experimental discipline

- Every arm uses the same tokenizer, token stream, row order, paired common
  initialization, feedback random stream, base trunk geometry, optimizer
  recipe, and schedule. Architecture gates are paired across `mhdb`, `df`,
  and `df_soft`; FBT fusion weights are paired across `fbt` and `df`. The
  registered architecture package is the only deliberate trunk divergence.
- Report both predicted tokens and token-equivalent compute. A `k`-pass batch
  costs `k` transformer passes; equal steps are matched-data, not matched-FLOP,
  comparisons.
- The primary factorial is `{vanilla, mhdb, fbt, df}`. `df_soft` is run only
  as the adoption diagnostic defined in the design and is not substituted for
  hard DF.
- Pass-1 validation is the common Standard-mode metric. Feedback arms also
  report the fully fused second-pass metric. Routing observables and
  contraction traces are diagnostics, not architecture wins by themselves.
- A feedback result is not interpretable without the contraction trace. Run
  the repeated fused-prefill diagnostic throughout training and require stable
  long-horizon self-composition before scale promotion.
- Accept a scientific claim only from completed, matched comparisons at the
  registered seeds/tokens or from a clearly labeled same-checkpoint causal
  ablation. Keep engineering qualification separate.

## Runtime and scale boundary

- The project imports telemetry, run/snapshot addressing, checkpoint staging,
  spool orchestration, `Schedule`, and monitor serving from the workspace root
  package. Do not reimplement those facilities here.
- CPU/MPS runs use the semantic PyTorch fallbacks. Jobe is the authoritative
  single-GPU CUDA surface: BF16 activations, FlashAttention, cut cross-entropy,
  the Triton MHDB router, compiled blocks, CUDA graphs, and asynchronous atomic
  snapshots. Respect Jobe's pinned Torch/FlashAttention environment.
- Inspect `df status`, the active log, and GPU ownership before operating Jobe.
  The queue records exact arguments without inspecting Git state. Source
  changes never stop an active child; the worker refreshes before the next job
  and runs that job's probe from the current checkout.
- The screen and same-geometry continuation surfaces are implemented. The
  2B→8B→32B WSD ladder still requires an explicit tested branch-from-heat-end
  continuation path. The 24-layer flagship is the exact
  `[PKDA, PKDA, PKDA, gated global GQA] x 6` hard-DF design in the scale plan;
  each PKDA mixer uses 20 query/key heads and 20 value heads at
  `d_k = d_v = 128`, for width-2,560 projections. It also requires PKDA kernels
  and recurrent/preconditioner cache semantics, gated GQA, block-delta
  routing, and distributed execution. Do not describe either path as runnable
  until those contracts land in code and tests.
- Flagship promotion requires the registered ladder trend, the matched
  factorial evidence, a clean contraction gate, the exact implementation gate,
  and explicit spend confirmation from a9.

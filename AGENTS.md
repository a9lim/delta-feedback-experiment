# AGENTS.md

This repository owns a from-scratch MHDB x FBT pretraining factorial on a
common PKDA/gated-GQA hybrid trunk. The hard hybrid is `df`; `mhdb` and `fbt`
are its parent-package deletions; `base` is their shared hybrid baseline. The
completed pure-GQA `vanilla` run is an external trunk control.

The 223–243M screen has no active job. No matched hybrid comparison is complete,
and [docs/findings.md](docs/findings.md) must remain empty of architecture
claims until one is admissible.

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
- Screen and token-ladder `base`, `mhdb`, `fbt`, and `df` use exact
  `[PKDA, PKDA, PKDA, gated global GQA] x 3` trunks: 8 PKDA heads at
  `d_k = d_v = 128`, causal convolution width 4, NoPE global attention, and
  8/4-by-96 GQA. `vanilla` remains the exact twelve-layer RoPE GQA trunk. Every
  global gate is a bias-free projection of the attention input followed by an
  elementwise sigmoid on the concatenated attention output before output
  projection and residual-branch scaling.
- Screen PKDA uses the literal portable recurrence off CUDA and the pinned FLA
  recomputing chunk kernel on CUDA. Its main and preconditioner gates are
  independent, recurrent and preconditioner boundary states are FP32, and
  sequential decoding continues both states plus all three convolution
  histories. Never silently run the sequential fallback for CUDA training.
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
- Multi-pass feedback is causal, differentiable across passes, and uses the
  shared prefix-mixin and jitter streams. Do not detach the payload to solve a
  memory problem; checkpoint blocks or coarsen the source representation.

## Experimental discipline

- Every arm uses the same tokenizer, token stream, row order, feedback random
  stream, outer geometry, optimizer recipe, and schedule. The complete hybrid
  trunk is byte-identical across `base`, `mhdb`, `fbt`, and `df`; MHDB weights
  are paired across `mhdb`/`df`, and FBT weights across `fbt`/`df`. The
  structurally distinct `vanilla` control retains its exact completed-run
  initialization and state layout.
- Report both predicted tokens and token-equivalent compute. A `k`-pass batch
  costs `k` transformer passes; equal steps are matched-data, not matched-FLOP,
  comparisons.
- The primary factorial is `{base, mhdb, fbt, df}`. Report `vanilla` versus
  `base` separately as the whole-trunk contrast.
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
  single-GPU CUDA surface: BF16 activations, pinned FLA PKDA, FlashAttention,
  FLA's fused PKDA RMSNorm/output gate, BF16-operand cut cross-entropy, the
  Triton PKDA control-gradient packer and MHDB router, compiled
  global-attention blocks, segmented PKDA block compilation, one train CUDA
  graph per pass count, and asynchronous atomic snapshots. Respect Jobe's pinned
  Torch/FlashAttention environment.
- Keep Jobe screen execution serial. Fresh default captures reserve 9.96 GiB
  for `base`, 10.12 GiB for `mhdb`, and 22.76 GiB for `df`; concurrent
  execution is outside the qualified deterministic single-GPU path.
- New snapshots are checkpoint-v11. Reject every older checkpoint version.
- Inspect `df status`, the active log, and GPU ownership before operating Jobe.
  The queue records exact arguments without inspecting Git state. Source
  changes never stop an active child; the worker refreshes before the next job
  and runs that job's probe from the current checkout.
- The screen and same-geometry continuation surfaces are implemented. The
  2B→8B→32B WSD ladder still requires an explicit tested branch-from-heat-end
  continuation path. The 24-layer flagship is the exact
  `[PKDA, PKDA, PKDA, gated global GQA] x 6` hard-DF design in the scale plan;
  each PKDA mixer uses 20 query/key heads and 20 value heads at
  `d_k = d_v = 128`, for width-2,560 projections. Its batch-aligned budget is
  440,818,728,960 predicted tokens: 400 per 1,102,046,496 active non-embedding
  parameters, conventionally 441B. The generic mixers and cache semantics now
  exist, but the flagship still requires exact-geometry distributed execution,
  memory/throughput qualification, checkpoint portability, and restart tests.
  Do not describe it as runnable until those contracts land.
- Flagship promotion requires the registered ladder trend, the matched
  factorial evidence, a clean contraction gate, the exact implementation gate,
  and explicit spend confirmation from a9.

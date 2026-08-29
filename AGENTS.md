# AGENTS.md

This repository owns a from-scratch pretraining factorial over two mechanisms:
multi-head routing over within-column residual deltas and full-bandwidth latent
feedback between adjacent token columns. The hard hybrid is `df`; `mhdar` and
`fbt` are its parent-factor deletions; `vanilla` is the shared baseline;
`df_soft` is an optional-channel diagnostic.

The 220M screen campaign is active. No matched scientific comparison is
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
- MHDAR sources are the column's actual input seed followed by scaled attention
  and MLP deltas. Routing is a transient pre-norm read; routed values never
  enter the residual stream directly, so `seed + sum(deltas) == h_top` remains
  exact.
- Each routing site uses a zero-initialized width-`D` query, a learnable
  full-width RMS key normalization, raw values, and one source softmax per
  contiguous feature group. The number of routing groups equals `kv_heads`.
- Hard DF fuses the shifted payload with the token embedding through the FBT
  asymmetric GLU, then uses the fused input as the MHDAR seed. Its payload is
  the normalized top state plus a routed mixture over this column's deltas.
- `df_soft` keeps the plain token embedding as the stream seed, exposes the
  shifted payload as a masked standing source, and prepends a learnable null to
  every router, including the payload router.
- Multi-pass feedback is causal, differentiable across passes, and uses the
  shared prefix-mixin and jitter streams. Do not detach the payload to solve a
  memory problem; checkpoint blocks or coarsen the source representation.

## Experimental discipline

- Every arm uses the same tokenizer, token stream, row order, initialization
  seed pairing, feedback random stream, trunk geometry, optimizer recipe, and
  schedule. Treat any divergence as a different experiment.
- Report both predicted tokens and token-equivalent compute. A `k`-pass batch
  costs `k` transformer passes; equal steps are matched-data, not matched-FLOP,
  comparisons.
- The primary factorial is `{vanilla, mhdar, fbt, df}`. `df_soft` is run only
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
  the Triton MHDAR router, compiled blocks, CUDA graphs, and asynchronous atomic
  snapshots. Respect Jobe's pinned Torch/FlashAttention environment.
- Inspect `df status`, the active log, and GPU ownership before operating Jobe.
  Queue entries are commit-addressed; do not disturb an active run to update
  documentation or code.
- The screen and same-geometry continuation surfaces are implemented. The
  2B→8B→32B WSD ladder still requires an explicit tested branch-from-heat-end
  continuation path. The 24-layer flagship additionally requires a defined
  block-delta source partition and distributed execution. Do not describe
  either path as runnable until those contracts land in code and tests.
- Flagship promotion requires the registered ladder trend, a clean contraction
  gate, and explicit spend confirmation from a9.

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
decoder. There are no current trained specimens. The GPT-NeoX/ChatML
15B-token DCLM-100B store is being rebuilt on Jobe; no training has been
launched. `docs/findings.md` states the current evidence boundary.

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
- [docs/scaling.md](docs/scaling.md): the screen, the bridge, and the
  flagship: geometry, accounting, budgets, runtime qualification needs, and the
  longer and larger recipes, worked out but not scheduled.
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
  `ModelConfig.condition` renders it back; every subset of `arfl` builds.
  Under `a` each cell is `[PKDA, PKDA, PKDA, NoPE-GGQA]`; without `a` it is
  `[RoPE-GGQA, RoPE-GGQA, RoPE-GGQA, NoPE-GGQA]`. All GGQA layers retain
  learned per-head Q/K RMSNorm. RoPE follows that norm, rotates adjacent pairs
  across the full head width with theta 10,000 in FP32, and casts back to the
  activation dtype. Positions are absolute within a token row, shared across
  feedback passes and core iterations; cached keys are already rotated and
  new queries/keys use the cache position. The screen has three cells. The bridge and the
  flagship in `scaling.md` are the same architecture at four cells and width
  1,152 and at six cells and width 1,536.
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
  routing parameters, PKDA controls, convolutions, and vectors in two groups:
  the matrices with fan-in `D` (GGQA gates, the FBT token gate, PKDA control
  projections) at `lr_nadam x 1536 / D`, everything else at `lr_nadam`, and
  the tied readout multiplies its logits by the same ratio. The flagship is
  the muP reference width, so its ratio is one. `BASE_NORMAL_INIT_STD = 0.02`
  in `model.py` is the NAdam matrix initialization knob: fan-in-`D` gates and
  controls multiply it by `sqrt(1536 / D)`; embeddings and fixed-head-width
  expansions use it directly. No weight decay. The global
  FP32 gradient is clipped to norm 10.0 before both steps.

## Useful bookkeeping

- The tokenizer is `EleutherAI/gpt-neox-20b` pinned at
  `c292233c833e336628618a88a648727eb3dff0a7`, with 50,277 base IDs plus
  `<|im_start|>` (50,277) and `<|im_end|>` (50,278). The tokenizer has
  50,279 IDs and the model vocabulary is padded to 50,304 rows. Document EOS
  is `<|endoftext|>` (0), separate from the ChatML message end. Generic ChatML
  preserves arbitrary and repeated role strings; generation defaults to
  `next_role="self"`. Loading the tokenizer loads no model weights.

- Sources are named by `--source`: the default `dclm-100b` preserves its
  publisher-shuffled order for screen/Jobe; `dclm` shuffles the full source
  for larger stores. FineWeb-Edu full/350B/100B/10B sources remain selectable.
  `--shuffle` / `--no-shuffle` override ordering. Stores live at
  `DATA_ROOT/SOURCE`, on Jobe `/data/delta/dclm-100b` or `/data/delta/dclm`.
  The trainer uses the same `--data-root` / `--source` pair; both inherit
  independently on resume or continuation unless explicitly overridden.
  Prefix identity requires the same source, revision, tokenizer, build package
  versions, ordering and seed; it does not hold between the subset and full DCLM.
  A sidecar maps
  every document to its pinned parquet file and row. Conditions trained with
  the same seed and data seed share tokenizer, stream,
  row order, schedule, optimizer, batch geometry, and keyed feedback
  randomness; conditions on the same trunk letter pair every parameter they
  share byte-identically, each letter's private weights from that letter's own
  stream. Two such runs can be compared token by token.
- A `k`-pass batch costs `k` transformer evaluations, and under `l` a pass
  executes `2 + r` cells. Report pass-tokens and cell-tokens beside predicted
  tokens; equal steps are matched data, not matched compute.
- Every planned run is addressed by `--condition`, `--scale`, and
  `--tokens-per-param`: the preset fills the geometry and batch, the ratio
  derives the step count from the flat stack's active parameters at that
  scale, rounded up, and every condition at a scale shares the schedule.
  `--continue TAG` extends a finished run to a longer schedule from the last
  snapshot both schedules reproduce, every other setting inherited.
- Pass-1 validation is the common number. Conditions with `f` also report the fused
  number and the self-composition trace.
- Independently trained runs differ like two seeds even when paired, so a
  small per-token difference between two runs is a mean shift on a wide
  spread. Same-checkpoint interventions resolve much finer effects.

## Runtime and Jobe

- Telemetry, schedules, run/snapshot addressing, checkpoint staging, monitor
  serving, and spool orchestration come from the workspace root package.
- BF16 CUDA causal training/prefill uses native PyTorch Flash SDPA; FP32
  diagnostics use math, and cached single-query decode uses FlexAttention over
  its valid prefix. No external `flash-attn` package is used.
- Projection gradients accumulate into persistent FP32 banks: PKDA Q/K/V and
  dense QKV/gate each share a contiguous allocation with disjoint parameter
  views, so their backward uses one packed GEMM. Optimizer state stays separate.
- Above the raw activation budget, each cell retains its final compiled block's
  activations and checkpoints the preceding blocks. The checkpoint wrapper
  stays outside block compilation, preserving its numerical boundaries.
  Every feedback pass and core iteration remains differentiable.
- Jobe is the single-GPU CUDA surface. Keep GPU jobs serial.
  `docs/runtime-qualification.md` tracks qualification of the current
  tokenizer and head on PyTorch 2.14 / CUDA 13.2. Keep cyclic Python garbage
  collection outside train/eval graph capture.
- Checkpoint v28 is the only accepted contract for resume, evaluation, and
  forks. Every specimen records its condition as letters.
- Before touching Jobe, look at `delta status`, the active log, and GPU
  ownership. The queue stores arguments rather than Git state; a worker
  refreshes before the next job and runs the current checkout's probe.
- Prime DDP and the larger reference are worked out in `docs/scaling.md` but
  not implemented. Prime time is real money; a9 decides when to spend it.

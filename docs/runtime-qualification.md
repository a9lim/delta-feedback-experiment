# Runtime qualification

Analysis depends on the measured computation matching the trained
computation. This page records the CUDA execution path and its engineering
evidence.

The execution timing measurements below are dated 2026-09-09 on Jobe's RTX 4090
with PyTorch 2.14.0+cu132, CUDA 13.2, Triton 3.8.0, and standard GIL-enabled
CPython 3.13.15. The Mac uses PyTorch 2.14.0 and Python 3.13.15. Records carry
exact sources and inputs; Hopper has not been through the same checks.

## Evidence records

| Record | What it establishes |
|---|---|
| [Cached PKDA precision](../data/summary/cache-parity-2026-09-10.json) | Shared convolution precision brings the four-layer BF16 cache error to 3.08% within the unchanged 4% bound; focused cache-history and decode checks pass |
| [Initialization qualification](../data/summary/initialization-qualification-2026-09-10.json) | Width-scaled gate/control initialization, unchanged flagship and RNG streams, passing test suites and Ruff |
| [Current execution optimizations](../data/summary/runtime-optimizations-2026-09-09.json) | Flash SDPA, packed projection gradients, PKDA tiling, selective block retention, full loop timings and accumulated-gradient comparisons |
| [Python/runtime qualification](../data/summary/python-runtime-2026-09-06.json) | Environment migration and its dependency/capture checks |
| [FlexAttention migration](../data/summary/flexattention-runtime-2026-09-06.json) | Earlier paired six-update diagnostic from a trained checkpoint |
| [Earlier execution optimizations](../data/summary/runtime-optimizations-2026-09-06.json) | Paired 18-update traces and component measurements on the earlier stack |
| [Loop staging baseline](../data/summary/loop-stage-2026-09-09.json) | Pre-optimization memory and timing measurements |

The width-scaled initializer passed 239 portable tests with 16 CUDA cases
skipped and all 255 tests on Jobe. CUDA cached PKDA now shares the full-row
fused convolution/SiLU/QK-normalization kernel, retaining FP32 arithmetic until
the final Q/K/V cast. This removes extra BF16 rounding from the cached path.
The four-layer, width-128 `a` model's full-sequence versus cached-decode
relative hidden-state L2 error is 0.03081, within the unchanged 0.04 bound.
Plain-decoder BF16 and looped-model FP32 comparisons also pass, as do five
focused CUDA projection/history cases, eleven portable cache/initialization
cases, and Ruff on both machines. The full suite and full screen graph gate
were not rerun for this cache fix; the timing evidence below is from September 9.
These records predate the partial-RoPE path for conditions without `a` and
do not establish its numerical parity or performance.

The 2026-09-09 execution record's four train/eval graphs peak at 14.46 GiB
allocated and 22.96 GiB reserved, with
one/two/three-pass replay at 64.6/129.9/195.9 ms. The loop-versus-flat gradient
error is 1.64%, matching its 1.64% repeat floor; checkpoint staging leaves
0.02 GiB of residual device allocation. Keep GPU work serial on Jobe's
24 GiB card. New snapshots use v27; loading accepts v27 for every condition
and v26 only for conditions with `a`, whose computation is unchanged.

## Maintained execution contract

Portable CPU/MPS execution uses PyTorch scaled-dot-product attention, the
literal PKDA recurrence, algebraic MHDB routing, chunked tied-head loss, and
eager execution. It is the numerical reference for focused invariants and
analysis. CUDA training uses the same equations through the following path:

- BF16 trunk activations with FP32 parameters, recurrent/preconditioner
  boundaries, accumulated gradients, and optimizer state.
- Exactly `seq_len` executed inputs per stored `seq_len + 1` row; the final
  token is a target. Keyed jitter retains the stored-row draw width so data
  and randomness addresses do not change with the executed length.
- Workspace FLA PKDA chunk kernels for training/prefill and recurrent kernels
  for cached decode. Q/K/V projection and convolution work is packed; output
  norm/gate and control-gradient packing are fused. CUDA training never falls
  back silently to a sequential implementation. State-kernel tuning keys
  include sequence count and length, so one long row has its own choice.
  Ada additionally searches 16-wide state tiles; the recurrence and FP32 state
  boundaries are unchanged.
- The dense convolution, recurrence, and norm/gate reach FLA through custom
  operators with exact fake implementations and their own backward operators,
  so Dynamo keeps them in the graph and a PKDA block compiles as one graph
  instead of fifteen joined by breaks at FLA's disabled entry points. Each
  forward asserts the metadata its fake promises. Cached decode calls FLA
  directly.
- Compiled native PyTorch SDPA with GQA for causal training and prefill:
  BF16/FP16 explicitly selects Flash, while FP32 diagnostics select math.
  The backend scope is inside the compiled region and restores the caller's
  preferences. Cached decode retains compiled FlexAttention, writing BF16 K/V
  and exposing only the valid prefix. No external `flash-attn` package is used.
- Without `a`, the first three layers of each cell apply full-head,
  adjacent-pair RoPE after learned per-head Q/K RMSNorm, with theta 10,000,
  FP32 phases and rotation, and a cast back to the activation dtype. Every
  fourth layer stays NoPE. Positions are absolute within the token row and
  reused across feedback passes and core iterations. Cached new queries and
  keys use `cache.pos`; stored keys are already rotated. Under `a`, the
  `[PKDA, PKDA, PKDA, NoPE-GGQA]` computation is unchanged.
- Triton MHDB routing with site-local nulls, raw values, a source softmax per
  group, and full-width RMS coupling in backward. Each compiled block emits
  its distance from the cell entry for the current/completed delta bank.
- Each within-column source is banked at its birth: readers receive an alias
  and the routing backward adds their contribution into the source's own BF16
  accumulator, one program per token so no atomics, and returns no gradient
  for it. The bank rides the residual stream, so its backward runs once every
  reader has, and it hands the finished accumulator on as the source's
  gradient. The accumulator is allocated where the source is born, which under
  capture is one address in the graph's pool and one memset per replay. A
  reader that does not bank (the portable and non-Triton routers) still
  returns an ordinary gradient and the bank adds it; the gate compares the two
  accumulations on the same model.
- Persistent FP32 projection/embedding gradient sinks and address-stable BF16
  weight shadows refreshed once per optimizer update. PKDA Q/K/V gradients
  share one contiguous allocation, as do each dense layer's QKV and gate
  gradients. The parameter gradients are disjoint row views, so one backward
  GEMM accumulates each packed projection directly into its bank. Parameters,
  clipping, and optimizer state remain per-parameter; no duplicate gradient
  or checkpoint state is added. Shadows are runtime operands.
- Workspace cut-cross-entropy with BF16 operands, capture-safe preprocessing,
  differentiable log-partition for z-loss, ascending mean-logit vocabulary
  tiling, and backward filtering equivalent to its late-filter decision.
- One persistent BF16 classifier-gradient buffer for the head. The backward
  lock-adds every call's `dC` into it rather than zero-filling and returning a
  233 MB tensor per microbatch, so the classifier receives no autograd
  gradient; the trainer adds the buffer into the FP32 embedding sink and clears
  it whenever another microbatch would take it past `--head-flush-every` head
  calls, and once before the optimizer reads the step. One feedback pass is one
  head call, so the number of BF16 additions a flush carries does not change
  with the pass count. Vocabulary rows are addressed by original id, so the
  per-batch permutation does not move a contribution. The gate measures the
  window's classifier gradient against the exact FP32 gradient of the same
  operands and against the per-call path, and checks that the captured body
  reaches the buffer and the flush reaches the sink.
- Fixed-shape Inductor tuning; one train graph per reachable (pass count,
  core iteration count) pair and shared-pool no-grad validation graphs at the
  evaluation count. Cyclic Python garbage is collected
  before capture and automatic collection is suspended through capture entry,
  body, and exit, restoring the caller's setting even on exceptions.
- One pinned host batch and one device batch per update. An event fences host
  reuse during asynchronous transfer; graph-input copies remain stream ordered.
  Every scale uses one 4,096-token row per microbatch and 128 microbatches per
  update, with BF16 keyed jitter written directly into graph inputs.
- NorMuonH shape-bucket compilation including packing, update, and state
  writeback; ordinary per-parameter checkpoint state with no persistent packed
  duplicate. NAdam and global FP32 clipping follow
  [architecture.md](architecture.md).
- Selective block activation retention above the measured work threshold.
  Each four-layer cell retains its final compiled block and checkpoints the
  preceding three. The checkpoint wrapper stays outside block compilation,
  preserving its numerical boundaries. The raw-work threshold is unchanged,
  and all feedback passes and core iterations remain differentiable. Snapshot
  staging is asynchronous to pinned host memory, with atomic background writes.

Inductor artifacts live at `~/.cache/delta-feedback/torchinductor` by default.
The first probe performs the fixed-shape search; later processes reuse it. Set
`DELTA_INDUCTOR_CACHE_DIR` only to relocate that durable cache. Run the probe on
the exact source and hardware before a training job; compilation and graph
capture are part of what gets reproduced.

For analysis, prepare the classifier shadow and use the model's activation
precision. The maintained route and payload scripts do this. New hooks or
portable replays reproduce the relevant baseline first, so that a changed
output is attributable to the change.

## The loop

The CUDA gate checks the `l` letter three ways:

- **Cached decode through per-iteration core tracks.** At width 128, twelve
  layers, and twelve positions, the full parallel forward and stepped
  `arfl` and `rfl` decode at two iterations must agree within 1% relative
  error in FP32. Separate four-layer BF16 checks of `a` and the partial-RoPE
  plain decoder keep their 4% bound.
- **One iteration is the flat column.** At the screen geometry, eager
  two-pass `arfl` at `r = 1` and `arf` agree in loss. The gate compares their
  gradients against a repeated `arf` measurement because the head's BF16
  accumulation order introduces variation. It also checks banked versus
  ordinary routed-source gradient accumulation.
- **The iteration cap runs.** One eager one-pass microbatch at `r = 8`,
  forward and backward under the trainer's activation policy, with its peak
  allocation reported as `loop_cap_peak`.

The captured `(pass count, r)` family is measured separately from the probe.
At the screen geometry (`arfl`, one 4,096-token row per microbatch), all 24
training graphs plus evaluation captured in 92.5 s, peaking at 15.84 GiB
allocated and 23.02 GiB reserved. Raw replay ranges from 64.6 ms at one pass,
`r = 1`, to 623.5 ms at three passes, `r = 8`. The full table and schedule
projection are in [scaling.md](scaling.md#cost-of-the-loop-at-the-screen).

For 128 distinct synthetic rows, the two-pass `r = 4` batch took 34.01 s
versus 37.42 s on the baseline; three-pass `r = 8` took 80.00 s versus
88.54 s. These include input replay and head-gradient flushing, excluding
optimizer, data loading, evaluation, and snapshots. The realized default
schedule projects to 34.89 replay hours, 6.35% less than the baseline.

The accumulated-gradient relative L2 differences at `(k,r) = (1,1), (1,4),
(2,4), (3,8)` are 2.83%, 3.63%, 4.67%, and 13.64%; corresponding baseline
repeats differ by 1.79%, 2.25%, 2.84%, and 8.47%. All values are finite,
gradient norm ratios remain within 0.032% of one, and the lowest cosine is
0.9907. This is measurable numerical drift beyond repeat variation,
particularly at the deepest mode. It is accepted engineering evidence, not
bitwise equivalence or evidence about training quality. These comparisons
start from paired fresh weights; no trained specimen was available.

Reproduce the current graph family and gradient dumps with:

```bash
python scripts/loop_optimization_check.py --full-family --modes 1:1,1:2,1:3,1:4,1:5,1:6,1:7,1:8,2:1,2:2,2:3,2:4,2:5,2:6,2:7,2:8,3:1,3:2,3:3,3:4,3:5,3:6,3:7,3:8 --batch-modes 1:1,1:4,2:4,3:8 --gradients /tmp/current-gradients --output /tmp/current-loop.json
python scripts/compare_loop_gradients.py /tmp/baseline-gradients /tmp/current-gradients --output /tmp/gradient-comparison.json
```

## Short-update evidence and its limits

The current record compares the baseline and candidate through two optimizer
updates on the same synthetic 128-row batch at two passes and `r = 4`.
Both paths keep finite losses and gradients at the recipe's learning rates;
the maximum loss difference is 0.000452 and gradient-norm relative difference
0.120%. Fresh weights and repeated synthetic rows cannot establish
training quality, retained capability, or equivalent learning trajectories.
The older trained-checkpoint traces are linked as dated evidence for their
own source revisions; they do not qualify this intervention.

## Backend constraints

Causal BF16/FP16 execution explicitly selects the native Flash SDPA backend;
unsupported Flash inputs fail instead of choosing another kernel. FP32
reference diagnostics select math. Cached single-query decoding uses
FlexAttention with no causal mask, because every key in the valid prefix is
visible to that query. Its cache tests poison unwritten storage and exercise
prefixes across kernel boundaries.

`scripts/attention_check.py` compares the production path against diagnostic
FlexAttention variants at one 4,096-token row, eight query heads, four KV
heads, and head dimension 96. It uses normalized BF16 Q/K and the projection's
strided V layout, checks output and Q/K/V gradient differences, and alternates
variant order when timing captured forward/backward work. FlexAttention's
prescaling and mask hints are comparison options, not training settings.

NVGEMM is a diagnostic option through `kernel-bench`, not a training backend.
The record documents candidate selection and compilation failures on Ada; it
does not support adopting that backend. Re-run the probe after hardware or
kernel changes.

## Reproduce a diagnostic

Run GPU checks serially after inspecting live status, the active log, and GPU
ownership.

```bash
python scripts/kernel_training_check.py runs/screen-delta-arf-s1-highLR.pt.10745 --updates 18
python scripts/attention_check.py
python scripts/attention_check.py --length 4097
python scripts/gemm_backend_check.py --backends ATEN,TRITON --output-dir tmp/gemm-base
```

Use the optimization record's exact baseline revision and explicit variant
flags for a paired reproduction. Trained-input and full-gradient checks also
live in `scripts/kernel_inputs.py` and `scripts/kernel_qualification.py`. A new
intervention keeps the numerical, cache, and state-boundary invariants the
probe checks.

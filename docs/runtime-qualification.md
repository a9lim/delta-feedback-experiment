# Runtime qualification

Analysis depends on the measured computation matching the trained
computation. This page records the CUDA execution path and its engineering
evidence.

The pinned GPT-NeoX/ChatML tokenizer, 50,304-row head, and checkpoint v28
passed `delta probe` on Jobe at `e422d078` on 2026-09-11: 333 tests and the
production-shape CUDA gate. The [machine-readable record](../data/summary/neox-tokenizer-qualification-2026-09-11.json)
identifies exact source/vendor revisions, runtime, tokenizer identity, and
printed numerical results.

The gate captured three `arf` training graphs and one validation graph at
width 768 and sequence length 4,096. Peak allocated memory was 12.99 GiB and
reserved memory 22.99 GiB; the separate one-pass loop-cap check reached
12.88 GiB allocated. One 20 MiB allocation request logged OOM during the
probe; execution continued and exited successfully. The reserved pool still
approaches card capacity, so GPU work remains serial.

Median synthetic one-row replay times were 53.8, 108.4, and 163.3 ms for one,
two, and three passes. These exclude optimizer, data-loading, validation,
and snapshot work. They are not trained throughput or a measured tokenizer
speedup. The loop-versus-flat gradient difference was 1.67%, matching the
measured repeat floor; cache and head-gradient checks passed their bounds.
The full 24-graph loop family and complete 128-row update throughput still
need their own measurements.

The target runtime is Jobe's RTX 4090 with PyTorch 2.14.0+cu132, CUDA 13.2,
Triton 3.8.0, and standard GIL-enabled CPython 3.13.15. Qualification records
must identify exact project/vendor revisions and inputs. Hopper requires its
own checks. Keep GPU work serial on Jobe's 24 GiB card.

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
  full classifier-gradient tensor per microbatch, so the classifier receives
  no autograd gradient; the trainer adds the buffer into the FP32 embedding sink and clears
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
- Selective block activation retention above the configured work threshold.
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

The full captured `(pass count, r)` family must be measured separately from
the probe at the screen geometry: one 4,096-token row per microbatch, all
24 training graphs, and evaluation. Report capture time, peak allocated and
reserved memory, raw replay time, and the time for 128 distinct synthetic
rows including keyed input replay and head-gradient flushing. A complete
training-update timing must additionally include optimizer work; report data
loading, evaluation, and snapshot costs separately when projecting a run.

Compare accumulated FP32 gradients at `(k,r) = (1,1), (1,4), (2,4), (3,8)`
with repeated baselines at the same settings. Finite values, relative L2,
norm ratios, and cosine similarities distinguish numerical changes from the
head's repeat variation. Synthetic fresh-weight checks establish execution
properties and do not establish training quality.

Reproduce the current graph family and gradient dumps with:

```bash
python scripts/loop_optimization_check.py --full-family --modes 1:1,1:2,1:3,1:4,1:5,1:6,1:7,1:8,2:1,2:2,2:3,2:4,2:5,2:6,2:7,2:8,3:1,3:2,3:3,3:4,3:5,3:6,3:7,3:8 --batch-modes 1:1,1:4,2:4,3:8 --gradients /tmp/current-gradients --output /tmp/current-loop.json
python scripts/compare_loop_gradients.py /tmp/baseline-gradients /tmp/current-gradients --output /tmp/gradient-comparison.json
```

## Short-update checks

After isolated loss and gradient checks, exercise complete optimizer updates
on the same synthetic 128-row batches at the recipe's learning rates. Record
losses, gradient norms, and updated-parameter differences against a paired
baseline. Fresh weights and repeated synthetic rows cannot establish retained
capability or equivalent learning trajectories. Trained-input comparisons
require a current checkpoint and matching token store.

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
Re-run the probe after hardware or kernel changes.

## Reproduce a diagnostic

Run GPU checks serially after inspecting live status, the active log, and GPU
ownership.

```bash
python scripts/kernel_training_check.py runs/TAG.pt.STEP --updates 18
python scripts/attention_check.py
python scripts/attention_check.py --length 4097
python scripts/gemm_backend_check.py --backends ATEN,TRITON --output-dir tmp/gemm-base
```

Use exact baseline revisions and explicit variant flags for paired checks. Trained-input and full-gradient checks also
live in `scripts/kernel_inputs.py` and `scripts/kernel_qualification.py`. A new
intervention keeps the numerical, cache, and state-boundary invariants the
probe checks.

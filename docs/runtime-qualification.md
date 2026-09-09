# Runtime qualification

Analysis depends on the measured computation matching the trained
computation. This page records the CUDA execution path and its engineering
evidence.

The committed qualification records dated 2026-09-06 cover Jobe's RTX 4090 with
PyTorch 2.14.0+cu132, CUDA 13.2, Triton 3.8.0, and standard GIL-enabled CPython
3.13.15. The Mac uses PyTorch 2.14.0 and Python 3.13.15. These records describe
their exact sources and inputs, not live machine status. Hopper has not been
through the same checks.

## Evidence records

| Record | What it establishes |
|---|---|
| [Python/runtime qualification](../data/summary/python-runtime-2026-09-06.json) | Exact environment selection, dependency checks, portable/CUDA gates, and capture memory |
| [FlexAttention migration](../data/summary/flexattention-runtime-2026-09-06.json) | Paired six-update diagnostic from a trained checkpoint, with runtime versions, revisions, loss/gradient differences, and limitations |
| [Execution optimizations](../data/summary/runtime-optimizations-2026-09-06.json) | Paired 18-update traces, component measurements, profiler counts, backend decisions, and reproduction commands |
| [Loop staging](../data/summary/loop-stage-2026-09-09.json) | Eager and captured memory of every `arfl` mode, replay time per (pass count, `r`), and the projected schedule |

Jobe passed 114 experiment tests and the full `delta probe` CUDA gate. The Mac
passed 106 tests with eight CUDA cases skipped. The workspace's 122 tests
passed on each machine, and both environments passed `uv pip check`. Checkpoint
v23 is unchanged.

The Python 3.13 gate covers PKDA and fused-kernel parity, causal/cached
attention, all four train/eval graphs, replay, optimizer state and radius
invariants, post-capture reporting, and checkpoint staging. Capture peaked at
13.85 GiB allocated and 23.01 GiB reserved. One/two/three-pass microbatch
replay took 53.4/107.7/161.9 ms. Keep GPU work serial on Jobe's 24 GiB card.

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
- Selective activation checkpointing above the measured work threshold.
  The checkpoint wrapper is inside full-block compilation so AOTAutograd
  retains native Flash-attention outputs and `mm`/`addmm` outputs whose width
  is at most six times their input width. The screen's packed PKDA Q/K/V
  projection fits that bound; its expanded MLP gate/up projection does not.
  Expanded MLP values, PKDA forward auxiliaries, and other recomputable
  activations are reconstructed. The raw-work threshold is unchanged, and
  all feedback passes and core iterations remain differentiable. Snapshot
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

The CUDA gate checks the `l` letter three ways, all on 2026-09-08 with the
runtime above:

- **Cached decode through per-iteration core tracks.** On the gate's small
  geometry (width 128, twelve layers, twelve positions) the full parallel
  forward and the stepped decode differ, in FP32, by 0.17% relative for
  `arfl` at two iterations, the same as the flat `a` and `arf` columns at
  twelve layers. Under BF16 the same comparison drifts with executed depth:
  3.5% for `a` at four layers, 6.9% for `a` and 7.6% for `arf` at twelve,
  7.6% for `arfl` at one iteration and 8.7% at two, and 3.2% for the plain
  GQA loop. The loop check therefore runs in FP32 with a 1% bound; the
  four-layer BF16 checks keep their 4% bound.
- **One iteration is the flat column.** At the screen geometry an eager
  two-pass microbatch gives `arfl` at `r = 1` and `arf` the same loss to a
  millionth. Their gradients differ by 1.78% relative, and `arf` against
  itself differs by 1.78%: the head accumulates its BF16 gradient through
  locks and that order reaches every parameter (embedding 1.84%, trunk
  1.72%). The gate measures that floor and holds the loop to it.
- **The iteration cap runs.** One eager one-pass microbatch at `r = 8`,
  forward and backward under the trainer's activation policy, with its peak
  allocation reported as `loop_cap_peak`.

The captured `(pass count, r)` family is not part of the probe; it is
measured by `scripts/loop_memory_stage.py` and captured again by each run.
The 2026-09-09 staging record for `arfl` at the default recipe, one
4,096-token row per microbatch: eager one-, two-, and three-pass
microbatches peak at 5.46, 9.59, and 13.33 GiB allocated raw at `r = 1`; at
`r = 8` the one-pass microbatch peaks at 13.70 GiB raw and the two- and
three-pass microbatches at 3.43 and 4.08 GiB checkpointed; the twenty-four
train graphs and the evaluation graph capture in 119 s at 15.85 GiB
allocated and 22.92 GiB reserved; replay runs from 67.2 ms (one pass,
`r = 1`, the flat column's own time) to 690.3 ms (three passes, `r = 8`)
per microbatch, the full table in
[scaling.md](scaling.md#cost-of-the-loop-at-the-screen).
Reproduce with:

```bash
python scripts/loop_memory_stage.py --out data/summary/loop-stage-DATE.json --condition arfl
```

## Short-update evidence and its limits

The 18-update optimization comparison starts from the same diagnostic
checkpoint on each side with fresh optimizer state and identical data
addresses, learning rates, pass schedule, and keyed randomness. Both sides use
Python 3.13 and the current runtime; the baseline selects plain causal
attention, per-microbatch uploads, and the baseline optimizer recorded in the
artifact. Timing excludes the first update.

| Passes | Baseline update | Combined update | Time reduction | Samples per side |
|---|---:|---:|---:|---:|
| 1 | 4.749 s | 4.731 s | 0.38% | 8 |
| 2 | 9.471 s | 9.449 s | 0.24% | 6 |
| 3 | 14.278 s | 14.252 s | 0.19% | 3 |

The largest update-loss difference was 0.000118, the largest gradient-norm
relative difference 0.98%, and final pass-one/fused validation differences
0.000128/0.000160. Both eight-pass traces remained finite. Combined capture
peaked at 13.83 GiB allocated and 22.91 GiB reserved. These are small
sequential samples: a modest throughput change, not long-run parity.

The separate optimizer comparison reduced warmed NorMuonH time from 62.15 to
50.43 ms. In the profiled clipping/optimizer/shadow/zeroing interval, launches
fell from 629 to 582 and eager `stack`/`cat` calls from 31 each to one each.
Eager allocation counts did not fall. Staging alone stayed within 0.2% of the
baseline and adds about 2.5 MiB each of pinned host and device storage.

The six-update whole-runtime migration comparison used Python 3.12.13 on both
sides, the same checkpoint and rows, and pass counts `1,1,1,2,2,3`. Candidate
one/two/three-pass update medians were 4.741/9.470/14.209 s. Maximum
update-loss difference was 0.000277 and gradient-norm relative difference
0.229%. The record contains the baseline stack and full traces. It does not
isolate FlexAttention's contribution. The distinct Python 3.13 gate is recorded
separately.

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

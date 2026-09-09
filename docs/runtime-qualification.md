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
  back silently to a sequential implementation.
- Compiled native FlexAttention with causal masks shared across layers and
  passes. Cached decode writes BF16 K/V and exposes only the valid prefix.
  Global Q/K/V and gate projections share a GEMM. No external `flash-attn`
  package or FlashAttention-4 backend is part of this runtime.
- Triton MHDB routing with site-local nulls, raw values, a source softmax per
  group, and full-width RMS coupling in backward. Each compiled block emits
  its distance from the cell entry for the current/completed delta bank.
- Persistent FP32 projection/embedding gradient sinks and address-stable BF16
  weight shadows refreshed once per optimizer update. Shadows are runtime
  operands, not additional learned or checkpointed state.
- Workspace cut-cross-entropy with BF16 operands, capture-safe preprocessing,
  differentiable log-partition for z-loss, ascending mean-logit vocabulary
  tiling, and backward filtering equivalent to its late-filter decision.
- Fixed-shape Inductor tuning; one train graph per reachable (pass count,
  core iteration count) pair and shared-pool no-grad validation graphs at the
  evaluation count. Cyclic Python garbage is collected
  before capture and automatic collection is suspended through capture entry,
  body, and exit, restoring the caller's setting even on exceptions.
- One pinned host batch and one device batch per update. An event fences host
  reuse during asynchronous transfer; graph-input copies remain stream ordered.
  The screen uses four 1,024-token rows per microbatch and 80 microbatches per
  update, with BF16 keyed jitter written directly into graph inputs.
- NorMuonH shape-bucket compilation including packing, update, and state
  writeback; ordinary per-parameter checkpoint state with no persistent packed
  duplicate. NAdam and global FP32 clipping follow
  [architecture.md](architecture.md).
- Activation checkpointing above the measured work threshold, preserving the
  full feedback graph, and asynchronous pinned-host snapshot staging with
  atomic background writes.

Inductor artifacts live at `~/.cache/delta-feedback/torchinductor` by default.
The first probe performs the fixed-shape search; later processes reuse it. Set
`DELTA_INDUCTOR_CACHE_DIR` only to relocate that durable cache. Run the probe on
the exact source and hardware before a training job; compilation and graph
capture are part of what gets reproduced.

For analysis, prepare the classifier shadow and use the model's activation
precision. The maintained route and payload scripts do this. New hooks or
portable replays reproduce the relevant baseline first, so that a changed
output is attributable to the change.

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

Causal row-safety hints and forward-only contiguous-block hints are enabled.
Backward keeps indexed traversal: a 1,025-position prefill can have
noncontiguous partial query-block lists, and a global contiguous hint produced
large gradient errors. The scoped hint had zero output/gradient drift against
plain FlexAttention in that case. At production `high` matrix precision,
component timing showed no material isolated screen gain. `PRESCALE_QK` stays
off because its small measured gain did not justify the numerical change.

NVGEMM is a diagnostic option through `kernel-bench`, not a training backend.
The record documents candidate selection and compilation failures on Ada; it
does not support adopting that backend. Re-run the probe after hardware or
kernel changes.

## Reproduce a diagnostic

Run GPU checks serially after inspecting live status, the active log, and GPU
ownership.

```bash
python scripts/kernel_training_check.py runs/screen-delta-arf-s1-highLR.pt.10745 --updates 18
python scripts/attention_check.py --length 1024
python scripts/attention_check.py --length 1025
python scripts/gemm_backend_check.py --backends ATEN,TRITON --output-dir tmp/gemm-base
```

Use the optimization record's exact baseline revision and explicit variant
flags for a paired reproduction. Trained-input and full-gradient checks also
live in `scripts/kernel_inputs.py` and `scripts/kernel_qualification.py`. A new
intervention keeps the numerical, cache, and state-boundary invariants the
probe checks.

# Runtime qualification

The maintained Jobe runtime passed the full gate on 2026-09-06: PyTorch
2.14.0+cu132, CUDA 13.2, Triton 3.8.0, and RTX 4090. CUDA GQA uses native
compiled FlexAttention. The separate FLA
PKDA and cut-cross-entropy vendor revisions are unchanged. The Mac uses
PyTorch 2.14.0, with portable SDPA for CPU/MPS attention.

The machine interpreter is standard, GIL-enabled CPython 3.13.15 on both
machines. The [Python migration record](../data/summary/python-runtime-2026-09-06.json)
records its separate qualification. The six-update runtime-migration comparison
below was collected on Python 3.12.13; the execution-optimization measurements
use Python 3.13.15.

The [machine-readable record](../data/summary/flexattention-runtime-2026-09-06.json)
contains both diagnostic traces, exact revisions, inputs, and gate results.
This is engineering qualification; there are still no accepted experiment
findings. Hopper remains unqualified until the same checks run there.

## Correctness and capture

Jobe passed 114 experiment tests and the full `df probe` CUDA gate. The Mac
passed 106 tests with eight CUDA cases skipped. The workspace's own 122 tests
passed on each machine, and both environments passed `uv pip check`.

The Python 3.13 gate covers PKDA and fused-kernel parity, causal and cached attention,
all four train/eval graphs, replay, optimizer state and radius invariants,
post-capture reporting, and checkpoint staging. Capture peaked at 13.85 GiB
allocated and 23.01 GiB reserved. One/two/three-pass microbatch replay took
53.4/107.7/161.9 ms. Keep Jobe runs serial. Checkpoint v23 remains unchanged.

Both machines select only the Python 3.13 environment. Obsolete Python 3.12
environments and rejected Python 3.14 candidates were removed. The latest
Prime, Verifiers, and Renderers metadata still excludes Python 3.14, which
sets the common machine pin. The migration record includes these constraints
and the Mac MPS, MLX, and orchestration checks.

The initial migration failed during three-pass capture. Instrumentation showed
a generation-1 Python garbage collection changing the CUDA stream from active
capture to invalidated capture. PyTorch 2.14 no longer unconditionally collects
warm-up garbage before capture. Both train and evaluation now collect those
cycles before capture and suspend automatic collection through capture entry,
body, and exit, restoring the caller's setting even on an exception. The full
gate and the independent trained-checkpoint capture both passed with this fix.

## Execution optimizations

The Python 3.13 / PyTorch 2.14 screen runtime uses causal row-safety and
forward-only contiguous-block hints, one pinned whole-update input upload, and
NorMuonH bucket compilation that includes packing and state writeback. The
[optimization record](../data/summary/runtime-optimizations-2026-09-06.json)
contains the component measurements, paired update traces, profiler counts,
and rejected backend diagnostics. Model equations, optimizer ownership, and
checkpoint v23 are unchanged.

The 18-update comparison loads the same diagnostic checkpoint as below, with
fresh optimizer state and identical data addresses, learning rates, pass
schedule, and keyed randomness. Both sides use the same current runtime; the
baseline selects plain causal attention, per-microbatch uploads, and the
pre-change optimizer. Timing excludes the first update.

| Passes | Baseline update | Combined update | Time reduction | Samples per side |
|---|---:|---:|---:|---:|
| 1 | 4.749 s | 4.731 s | 0.38% | 8 |
| 2 | 9.471 s | 9.449 s | 0.24% | 6 |
| 3 | 14.278 s | 14.252 s | 0.19% | 3 |

These are small sequential diagnostic samples. The observed whole-update gain
is modest and does not establish a long-run throughput or training result.
The largest update-loss difference was 0.000118, the largest gradient-norm
relative difference was 0.98%, and final pass-one/fused validation differed by
0.000128/0.000160. Both eight-pass contraction traces remained finite. Combined
capture peaked at 13.83 GiB allocated and 22.91 GiB reserved.
The separate optimizer comparison reduced warmed NorMuonH time from 62.15 to
50.43 ms (18.9%). In the profiled clipping/optimizer/shadow/zeroing interval,
CUDA kernel launches fell from 629 to 582, and eager `stack`/`cat` calls from
31 each to one each. Eager allocation counts did not fall; the supported gain
is reduced dispatch and compiled packing/writeback. Optimizer state remains
ordinary per-parameter tensors, with no additional persistent packed copy.

The staging-only comparison stayed within 0.2% of baseline. It replaces 80
small host-to-device uploads with one pinned asynchronous upload per update,
then copies device slices into graph inputs. An event fences reuse of the
pinned storage; uploads and replays remain ordered on one stream. The screen
adds about 2.5 MiB each of pinned host and device storage. Tests delay DMA to
exercise host-buffer lifetime and compare addressed rows and feedback draws.

Contiguity is a forward-only promise: a 1,025-position causal prefill has
noncontiguous backward partial query-block lists, and enabling the global
hint produced roughly 44%/38% relative dK/dV error. The adopted scoped hint
had zero output/gradient drift against plain FlexAttention in that case.
Screen training attends over 1,024 positions; its stored 1,025-token rows
include the next-token target. Both geometries are recorded separately.
At production's `high` matrix precision, plain/hinted forward-backward graph
times were 0.453/0.453 ms for 1,024 positions and 0.561/0.556 ms for 1,025.
The hints therefore show no material isolated screen gain. Earlier
highest-precision component runs selected different kernels and are retained
as separate diagnostics. `PRESCALE_QK` added about 0.1–0.6% component speed,
within the timing spread, with output/gradient relative L2 drift below 0.5%;
it remains off because there is no measured material benefit.

NVGEMM is available only through the `kernel-bench` optional dependencies and
diagnostic flags. It is not enabled in training. The native Ada discovery path
requests nonexistent `89a`; the benchmark-only correction retains real SM89
capabilities without editing installed packages. After that correction, four
NVGEMM candidates per GEMM were timed in projection and SwiGLU
forward/backward comparisons, but none were selected. Projection graph time
was 0.193 versus 0.211 ms and SwiGLU 1.304 versus 1.304 ms. Full training
compilation failed on symbolic optimizer BMM dimensions; restricting NVGEMM
to trunk regions still failed with `Invalid leading dimension: 2`. These
results do not support adopting this backend on Jobe.

Reproduce the current combined diagnostic and component checks with:

```bash
python scripts/kernel_training_check.py runs/screen-df-full-s1.pt.10745 --updates 18
python scripts/attention_check.py --length 1024
python scripts/attention_check.py --length 1025
python scripts/gemm_backend_check.py --backends ATEN,TRITON --output-dir tmp/gemm-base
python scripts/gemm_backend_check.py --backends ATEN,TRITON,NVGEMM --ada-target-workaround --output-dir tmp/gemm-nv
```

Run GPU diagnostics serially. The record preserves the exact baseline
optimizer revision and explicit variant flags needed for the paired control.

## Runtime migration comparison

Both stacks loaded `runs/screen-df-full-s1.pt.10745`, used fresh identical
optimizer settings, the same token rows and keyed randomness, and six updates
with pass counts `1, 1, 1, 2, 2, 3`. Each update contains 320 rows of 1,024
tokens in microbatches of four. Reproduce the candidate with:

```bash
python scripts/kernel_training_check.py runs/screen-df-full-s1.pt.10745 --updates 6
```

The baseline used PyTorch 2.9.1+cu130, CUDA 13.0, Triton 3.5.1, and external
FlashAttention 2.8.3. These measurements compare the whole runtime migration;
they cannot isolate FlexAttention's contribution.

| Passes | Baseline update | New update | Time reduction |
|---|---:|---:|---:|
| 1 | 4.830 s | 4.741 s | 1.8% |
| 2 | 9.686 s | 9.470 s | 2.2% |
| 3 | 14.538 s | 14.209 s | 2.3% |

Times are medians excluding the first update: only two, two, and one samples,
respectively. They establish no observed slowdown in this diagnostic, not a
precise performance estimate.

The largest update-loss difference was 0.000277 and the largest gradient-norm
relative difference was 0.229%. Final validation differed by 0.0000074 for
pass one and 0.0000372 for the fused pass. Both contraction traces remained
finite. This short check does not establish long-run training equivalence.

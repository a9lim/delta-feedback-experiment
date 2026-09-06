# Runtime qualification

The maintained Jobe runtime passed the full gate on 2026-09-06: PyTorch
2.14.0+cu132, CUDA 13.2, Triton 3.8.0, and RTX 4090. CUDA GQA uses native
compiled FlexAttention. The separate FLA
PKDA and cut-cross-entropy vendor revisions are unchanged. The Mac uses
PyTorch 2.14.0, with portable SDPA for CPU/MPS attention.

The machine interpreter is standard, GIL-enabled CPython 3.13.15 on both
machines. The [Python migration record](../data/summary/python-runtime-2026-09-06.json)
records its separate qualification. The short paired update measurements below
were collected on Python 3.12.13 before the interpreter migration.

The [machine-readable record](../data/summary/flexattention-runtime-2026-09-06.json)
contains both diagnostic traces, exact revisions, inputs, and gate results.
This is engineering qualification; there are still no accepted experiment
findings. Hopper remains unqualified until the same checks run there.

## Correctness and capture

Jobe passed 109 experiment tests and the full `df probe` CUDA gate. The Mac
passed 104 tests with five CUDA cases skipped. The workspace's own 122 tests
passed on each machine, and both environments passed `uv pip check`.

The Python 3.13 gate covers PKDA and fused-kernel parity, causal and cached attention,
all four train/eval graphs, replay, optimizer state and radius invariants,
post-capture reporting, and checkpoint staging. Capture peaked at 13.84 GiB
allocated and 22.71 GiB reserved; the earlier trained-checkpoint diagnostic
reserved 23.01 GiB. One/two/three-pass microbatch replay took
53.5/107.8/162.2 ms. Keep Jobe runs serial. Checkpoint v23 remains unchanged.

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

## Short paired update check

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

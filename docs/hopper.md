# Hopper execution

The single GH200 at `ssh rental1` is the reference machine for Hopper work.
The canonical geometry uses GQA head width 256 with query/KV head
counts 4/2, 6/3, 8/4, and 12/6 at screen, bridge, flagship, and extension;
PKDA uses 8, 12, 16, and 24 heads of width 128. MHDB uses one group per KV
head. The complete geometry and accounting are in [scaling](scaling.md).

See [runtime estimates](runtime.md) for GH200 and eight-H100 run durations.

## Execution choices

| Path | Current choice |
|---|---|
| Dense projections | Stateless rowwise FP8 forward/dX; BF16 operands into FP32 dW |
| Routed experts | FP8 forward/dX; Hopper dW tiles `(BM,BN,BK,warps,stages) = (64,128,128,4,3)` |
| Classifier | BF16 with vocabulary filtering and FP32 gradient accumulation; Hopper `B=256,V=128,D=64`, eight warps, three stages |
| Causal GQA | cuDNN first, Flash fallback; head width 256 |
| PKDA | Hopper ATK inter-chunk scan `BK=128`, four warps |
| Replay planning | Widest fitting replay; retain intermediates when they fit |

Accumulated gradients stay FP32, and the classifier stays BF16 even under
`--precision fp8`. The checkpoint contract is v42 with the canonical
token-store identity. Hardware-specific tile choices are selected by the
installed workspace forks. Geometry affects model capacity and fresh-run
schedules; execution measurements do not establish learning quality.

## Current-geometry measurements

All measurements here use 4,096-token rows on the GH200. The attention pilot
covers forward, backward, and warm CUDA graph replay on initialized BF16
inputs, with output/input-gradient checks against Flash.

| Scale | Rows | Flash, ms | cuDNN, ms |
|---|---:|---:|---:|
| Screen | 4 | 2.3026 | 1.4215 |
| Bridge | 2 | 1.8069 | 1.1154 |
| Flagship | 1 | 1.3041 | 0.8137 |

cuDNN reduces attention time by 37.6–38.3% against Flash at the same new
geometry.
The largest output/input-gradient relative L2 difference is 0.196%; captured
gradients pass the same checks. Results:
`logs/hopper-tuning/attention-geometry-{scale}.json`.

### Full-model screen

The complete `fl` graph uses two passes, two loops, four rows per replay,
and the rowwise FP8/BF16-head recipe. Five warm samples gave a median
**707.97 ms per replay**, or **22.66 seconds per 128-row update** before
optimizer overhead. Four complete optimizer updates took **22.74–22.78
seconds** each, with finite losses and pre-clip gradient norms of 6.80–7.50.

The graph retains recurrence intermediates and recomputes zero blocks.
Its planner estimates 59.11 GiB of activations and 8.33 GiB of static state;
these are not peak-allocation measurements. Trace totals per replay include
243.69 ms for the classifier, 173.14 ms for recurrence, 120.27 ms for experts,
and 22.80 ms for attention. The trace confirms cuDNN forward/backward kernels.

The fixed four-column graph represents 76% of the recurrent phase's draws.
The first 75% of the default run uses one column; complete schedule-weighted
throughput remains an estimate. Bridge, flagship, and extension have no
full-model measurement at the canonical geometry.

The CUDA probe passed evaluation capture and cached decode. Six CUDA routing
tests passed, covering all four 384-wide group counts, the maximum source
bank, and repeated BF16 graph replay with gradient accumulation. The portable
suite passes 113 tests with seven hardware-dependent skips. Full-model
optimizer replay is an execution check; a complete train/eval/snapshot/resume/
stop lifecycle at this geometry remains unmeasured.

Current raw evidence:

- `logs/replay-bench/gh200-screen-geometry.json`
- `logs/hopper-tuning/screen-geometry.log`
- `logs/hopper-tuning/geometry-probe.log`
- `logs/hopper-tuning/geometry-routing-final.log`
- `logs/hopper-tuning/portable-final.log`

## Machine and operation

The rental reports `NVIDIA GH200 480GB`, compute capability 9.0, with
**94.50 GiB CUDA-visible GPU memory** on an aarch64 Grace host. Observed
runtime versions were Torch 2.14.0+cu132, CUDA 13.2, Triton 3.8.0, cuDNN
9.24, and NCCL 2.30.7. The product name does not describe available GPU HBM.

`inductor.py` supplies `min_rblock` to the split-scan tuner so the launcher's
workspace allocation bounds the selected reduction block. Compiler caches
must correspond to the installed launcher and forks.

Warmup can generate pointwise/reduction kernels on resume despite cached
GEMM tuning, and allocation warnings can precede successful capture. Inspect
process activity, cache writes, and subsequent capture/step records before
calling startup stalled. Keep cyclic garbage collection outside capture and
GPU jobs serial.

Use `scripts/replay_bench.py --replay-rows N` to fix actual replay width;
`--micro-rows` is a minimum. `--condition f --specs 1:1` isolates a
single-column width trial. `--expert-tiles ada`, `--cce-config bf16-base`,
`--attention-backend`, and `--fp8-head` support scoped comparisons.
Component tools override configurations only in their own process:

| Tool | Scope |
|---|---|
| `scripts/attention_bench.py` | Attention backend, output/gradients, graph capture |
| `scripts/pkda_bench.py --heads 8,12,16 --rows 2` | Current PKDA head counts and all input gradients |
| `scripts/moe_bench.py` | Balanced/skewed expert routing and GEMM tiles |
| `scripts/cce_bench.py --dim 1152 --rows 4 --candidates bf16-base,bf16-b256` | Paired head for a two-row bridge replay |
| `scripts/dense_fp8_bench.py` | Dense FP8 scaling, quantization, weight refresh |

`scripts/first_hour.sh` exercises training, evaluation, snapshots, resume,
and cooperative stop, checking the final checkpoint before writing success.
Generated logs, traces, snapshots, and figures remain untracked. Keep raw
evidence for the current geometry and execution choices; remove superseded
runs, tuning candidates, and completed orchestration files. Each retained
JSON's source revisions and geometry bound its claims.

## Multi-GPU node verification

A single GH200 establishes neither multi-GPU throughput nor communication
cost. The existing `--ranks 8` path shards optimizer ownership while retaining
replicated working weights and gradient slabs. Before relying on a node:

1. Confirm distinct devices, usable GPU memory, NVSwitch/peer topology,
   runtime and fork revisions; run `delta probe --ranks 8`.
2. Measure reduce-scatter and all-gather on production slabs. Verify
   gradient sums, gathered weights, and expert-bias agreement across ranks.
3. Capture the deepest screen and target-scale graphs. Record each rank's
   plan, peak allocation, reserved memory, and free memory.
4. Exercise evaluation, snapshots, cooperative stop, and resume across the
   intended rank counts.
5. Measure complete warm replays and schedule-weighted step time. Include
   startup compilation and monitoring in complete-run estimates.

Shared compiler caches can reduce startup work but do not eliminate graph
reconstruction or duplicate compilation by cold processes. Choose overlap
or further sharding from node measurements.

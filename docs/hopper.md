# Hopper execution

The single GH200 at `ssh rental1` is the reference machine for Hopper work.
The canonical geometry uses GQA head width 256 with query/KV head
counts 4/2, 6/3, 8/4, and 12/6 at screen, bridge, flagship, and extension;
PKDA uses 8, 12, 16, and 24 heads of width 128. MHDB uses one group per KV
head. The complete geometry and accounting are in [scaling](scaling.md).

Revision `3a66a5f` selects cuDNN attention first, with Flash fallback. The
initialized screen replay improved from 825.91 to 707.97 ms at the same
replay width, about 16.7% higher throughput. Attention pilots support the
backend choice at screen, bridge, and flagship. These measurements do not
establish model quality equivalence or whole-training throughput.

## Execution choices

| Path | Choice | Measurement boundary |
|---|---|---|
| Dense projections | Rowwise FP8 forward/dX; BF16 operands into FP32 dW | The blockwise pilot did not improve the complete path consistently |
| Routed experts | FP8 forward/dX; Hopper dW tiles `(64,128,128,4,3)` | Component gains at screen and bridge; effectively flat at flagship, one row |
| Classifier | BF16; Hopper `B=256,V=128,D=64`, eight warps, three stages | Faster at all three measured widths; tile filtering changes gradients slightly |
| Causal GQA | cuDNN first, Flash fallback; head width 256 | Screen, bridge, and flagship pass numerical/capture checks and favor cuDNN |
| PKDA | Hopper ATK inter-chunk scan `BK=128`, four warps | About 1% faster complete recurrence with the earlier head counts |
| Replay planning | Widest fitting replay; retain intermediates when they fit | Actual fit and timing require the complete captured graph |

The kernel retunes preserve Ada choices. FP8 scaling remains stateless and
rowwise, accumulated gradients remain FP32, and the classifier remains BF16
even under `--precision fp8`. Checkpoint v42 and the canonical token-store
identity are unchanged. Geometry changes affect model capacity and derived
fresh-run schedules; performance measurements alone do not establish equal
learning behavior.

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
The earlier bridge geometry (12 query / 6 KV heads, width 192) took 2.4515 ms
with Flash at two rows. That comparison also changes head counts and
projection widths;
it combines an architecture change with a backend change. The installed
Torch/cuDNN path rejects backward at head width 192.

### Full-model screen result

The complete `fl` graph used two passes, two loops, four rows per replay,
and the production rowwise FP8/BF16-head recipe. Five warm samples gave:

| Geometry / kernels | Median replay, ms | Replay-derived 128-row step, s |
|---|---:|---:|
| Earlier geometry, previous tiles | 844.35 | 27.02 |
| Earlier geometry, Hopper tiles | 825.91 | 26.43 |
| Current geometry, Hopper tiles and cuDNN | 707.97 | 22.66 |

The geometry/backend change reduces replay time by 14.3%, or increases
throughput by 16.7%, relative to the retuned earlier geometry. The combined
change is 19.3% higher throughput than the pre-retune baseline. All three
use retained recurrence intermediates and zero recomputed blocks. The
current planner estimate is 59.11 GiB, down from 67.90 GiB; static allocation
is 8.33 GiB. These estimates are not peak-allocation measurements.

The trace confirms cuDNN forward/backward kernels. Attention falls from
49.28 to 22.80 ms and recurrence from 224.65 to 173.14 ms. The classifier is
effectively unchanged at 243.69 ms. Architecture changes also change keyed
initialization and expert assignments, so component differences are not
isolated kernel comparisons. This common graph covers about 76% of the
schedule; a complete schedule-weighted training speedup was not measured.

Four full 128-row optimizer updates then completed in 22.74–22.78 seconds
each, with finite losses and pre-clip gradient norms (6.80–7.50). They use
the same fixed graph; this is an execution check, not a learning-quality
comparison or a repeat of the complete checkpoint lifecycle. The new-geometry
CUDA probe also passed, including evaluation capture and cached decode.
The final portable suite passed 113 tests, with seven hardware-dependent skips.
Six CUDA routing tests then passed: all four 384-wide group counts, the
maximum source bank, and repeated BF16 graph replay with gradient accumulation.
They also exposed and fixed shared return storage in the FP32 routing backward;
the BF16 dtype conversion already produced independent outputs.

Results: `logs/replay-bench/gh200-screen-geometry.json`, alongside the
earlier `gh200-screen-{before,final}.json` comparisons. Bridge, flagship,
and extension have no new-geometry full-model measurement.

## Component selection evidence

These measurements selected the retained kernel choices before the geometry
change. They used GQA query/KV counts 8/4, 12/6, and 16/8 at width 192, and
PKDA counts 10, 15, and 20 at width 128 for screen, bridge, and flagship.
Residual widths, experts, and the classifier match the current scales.
Component times are warm captured synthetic forward/backward comparisons;
they are not complete training steps. Ignored JSON artifacts retain source
revisions, configurations, numerical checks, and timing samples.

### Experts

The two BF16 weight-gradient GEMMs use
`(BM,BN,BK,warps,stages) = (64,128,128,4,3)` on sm90, replacing
`(64,64,64,4,3)`. FP8 activation tiles and their scale granularity stay fixed.
Complete six-GEMM path timings were:

| Scale / replay rows | Routing | Previous tiles, ms | Hopper tiles, ms |
|---|---|---:|---:|
| Screen / 4 | Balanced | 1.746 | 1.639 |
| Screen / 4 | Skewed | 1.711 | 1.632 |
| Bridge / 2 | Balanced | 2.014 | 1.923 |
| Bridge / 2 | Skewed | 2.004 | 1.943 |
| Flagship / 1 | Balanced | 1.941 | 1.945 |
| Flagship / 1 | Skewed | 1.931 | 1.932 |

Outputs, input gradients, and weight gradients matched the comparison tiles
exactly. Dispatch, external quantization, and sink reset are excluded.
Results: `logs/hopper-tuning/moe-{scale}.json` and `moe-short-*.json`.

### Classifier

Both BF16 directions changed from `B=128,V=128,D=32`, four warps/four stages,
to `B=256,V=128,D=64`, eight warps/three stages. Vocabulary filtering and
FP32 classifier-gradient accumulation remain enabled.

| Scale | CCE input rows | Previous BF16, ms | Hopper BF16, ms |
|---|---:|---:|---:|
| Screen | 8 | 63.513 | 61.062 |
| Bridge | 4 | 43.226 | 41.568 |
| Flagship | 2 | 26.698 | 26.387 |

CCE combines NTP and MTP, so these approximate four, two, and one training
replay rows respectively, without the small boundary crop. Timings include
forward, backward, and vocabulary ordering; classifier refresh and sink
reset are separate. Inputs model initialized normalized readouts, not a
trained checkpoint's filtering distribution.

At flagship, changed token partitions produced relative L2 differences of
**0.216% for dE** and **0.137% for dC** against the previous filtered BF16
configuration. Numerical and captured-replay checks passed; trajectories
are not bitwise identical. The fastest tested FP8 heads took 263.474,
193.916, and 123.350 ms respectively, so the classifier stays BF16.
Results: `logs/hopper-tuning/head-{scale}.json`.

Focused CUDA validation passed eight accumulator cases in 5.90 seconds;
twelve tile-metadata cases passed separately. FP32 sinks may accumulate
inside the MMA dot operation, so a repeated random contribution need not
match adding two independently rounded gradients bit for bit. The tests
retain exact first-write coverage and exact repeated accumulation with
representable inputs.

### PKDA and dense FP8

The ATK inter-chunk forward scan now traverses a 128-channel head once with
`BK=128`, replacing four `BK=32` traversals. Complete PKDA forward/backward
at two rows changed from 2.024 to 2.002 ms at ten heads, 2.869 to 2.836 ms at
fifteen, and 3.705 to 3.668 ms at twenty. Other intra-chunk, state, WY, and
gate-scan candidates were flat or slower. All ten recurrence input gradients
and captured replay were checked; the two focused old/new scan tests passed
with exact outputs and gradients. Results: `logs/hopper-tuning/atk.json`,
`atk-wide.json`, and `pkda.json`.

At ten heads, increasing from two to sixteen rows reduced recurrence time
per row from 1.012 to 0.853 ms. This is component occupancy evidence, not
proof that a sixteen-row model replay fits or improves a complete step.
Results: `logs/hopper-tuning/pkda-width.json`.

For the earlier flagship packed QKV projection at 8,192 tokens, combined
forward/dX and unchanged BF16-to-FP32 dW took 0.7655 ms in BF16, 0.6567 ms
with rowwise FP8, 0.6629 ms with activation `1×128` / weight `128×128`
scales, and 0.7535 ms with `1×128` scales on both operands. Per-call
activation/gradient quantization is included; weight refresh was measured
separately. Other blockwise cases were slower or failed the numerical check.
This limited pilot supports retaining rowwise scaling.
Results: `logs/hopper-tuning/dense.json`.

## Earlier integrated replay and lifecycle

The earlier width-192 screen `fl` graph, two passes/two loops/four rows,
fell from **844.35 to 825.91 ms** with the component retunes: 2.18% less
replay time. Five warm samples used retained recurrence intermediates,
zero recomputed blocks, and a 67.90 GiB planner estimate. The corresponding
retuned bridge graph at two rows measured **656.44 ms**. These are the
comparison points for the new geometry, not its results. The common graph
covers 76.04% of the configured recurrence roll; these times exclude a
schedule-weighted mean and full training-job overhead.

Screen trace totals attributed the main reductions to the classifier
(253.23 to 243.79 ms) and experts (146.60 to 139.74 ms); recurrence was
224.86 versus 224.65 ms. Before/after FLA revisions were `f208d549` and
`9a1c09bd`, with the latter adding the ATK change. Both used Flash and
rowwise FP8. Results: `logs/replay-bench/gh200-screen-before.json` and
`gh200-screen-final.json`.

The earlier-geometry screen `fl` lifecycle completed steps 1–12 with
evaluation/snapshots, resumed through steps 13–15, and stopped cooperatively
at step 18. The v42 reader loaded the final checkpoint. It reported 60.1 GiB
peak allocation, 61.21 GiB reserved, and 8.57 GiB static allocation. Captured
plans retained recurrence intermediates and recomputed zero blocks:

| Passes | Loops | Rows per replay |
|---:|---:|---:|
| 2 | 2 | 4 |
| 2 | 3 | 2 |
| 3 | 2 | 2 |

This establishes the launcher/evaluation/snapshot/resume/stop path at its
measured geometry. It preceded the final retunes and does not certify the
new geometry. `scripts/first_hour.sh` checks the expected lifecycle records
and final checkpoint before writing success.

## Machine and operation

The rental reports `NVIDIA GH200 480GB`, compute capability 9.0, with
**94.50 GiB CUDA-visible GPU memory** on an aarch64 Grace host. Observed
runtime versions were Torch 2.14.0+cu132, CUDA 13.2, Triton 3.8.0, cuDNN
9.24, and NCCL 2.30.7. The product name does not describe available GPU HBM.

The GH200 exposed an Inductor split-scan workspace mismatch:
`min_split_scan_rblock` metadata did not bound `R0_BLOCK` in the tuner,
which reads `min_rblock`. `inductor.py` supplies that key. The lifecycle ran
after removing faulty caches and applying the correction. Old cache entries
cannot establish correctness of the corrected launcher.

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

Generated logs, traces, snapshots, and figures remain untracked. Each JSON's
source revisions and geometry bound its claims; a later fork or different
shape needs its own evidence.

## Multi-GPU node verification

A single GH200 establishes neither multi-GPU throughput nor communication
cost. Before relying on an eight-H100 node:

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

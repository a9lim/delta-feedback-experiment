# Hopper node

What changes when training moves from the 4090 to an 8×H100 SXM node, what
is worth doing about it, and which of that can be built and tested before a
node is rented. Numbers are from the 2026-09-16 audit; the 4090 ledgers it
extrapolates from are in [operations](operations.md#cuda-execution) and the
journal. Nothing here has run on Hopper yet: every H100 figure is a
projection until the first rental session replaces it.

## The card and the node

| | RTX 4090 | H100 SXM | Ratio |
|---|---:|---:|---:|
| SMs | 128 | 132 | 1.03 |
| BF16 dense tensor peak | 165 TF | 989 TF | 6.0 (realistic GEMM ~4.5) |
| HBM | 1.0 TB/s | 3.35 TB/s | 3.35 |
| Memory | 24 GiB | 80 GiB | 3.3 |
| Boost clock | 2.5 GHz | 1.98 GHz | 0.8 |
| Inter-GPU | none | NVSwitch, ~350–450 GB/s bus | |

The profile does not scale uniformly. GEMMs gain the most, memory-bound
kernels about 3×, and latency-bound kernels (the sub-8 µs tail, serial
scans) little, since the SM count is the same and the clock lower. So the
H100 profile is less GEMM-dominated than the 4090 census, and anything that
speeds only GEMMs caps out lower than the 4090 split suggests.

Class-wise projection of the round-11 census (one 2-row k=1 screen replay,
181.4 ms on the 4090). Two independent sets of assumed ratios bracket the
uncertainty; the rental's first trace replaces both.

| Class | 4090 ms | Share | Ratio A / B | H100 ms A / B |
|---|---:|---:|---:|---:|
| Dense + expert GEMMs | 70.5 | 39% | 4.5 / 3.5–4.5 | 15.7 / 18.6 |
| CCE head | 34.1 | 19% | 3.5 / 2.5 | 9.7 / 13.6 |
| FLA recurrence | 35.4 | 20% | 3.0 / 1.8 | 11.8 / 19.7 |
| Dense attention (head 192) | 12.5 | 7% | 3.5 / 2.5 | 3.6 / 5.0 |
| Pointwise + conv/norm | 19.0 | 10% | 3.3 / 2.0 | 5.8 / 9.5 |
| Routers (FP32) | 5.5 | 3% | 3.0 / 1.5 | 1.8 / 3.7 |
| Tail (<8 µs kernels) | 4.4 | 2% | 1.5 / 1.2 | 2.9 / 3.7 |
| Total | 181.4 | | 3.5 / 2.5 | 51 / 74 |

A is the audit's own derivation, B the Codex thread's more pessimistic one;
the earlier flat 3.0× sits between them. Under A the H100 GEMM share is 31%
(dense + expert) plus 19% CCE; under B, 25% and 18%. Flagship (D = 1536)
roughly doubles the D-linear classes and quadruples the D² classes, so its
GEMM share is ~40–50%.

Per-rank step time at 16 rows per rank, k = r = 1, from the two brackets:
screen ~0.4–0.6 s, flagship ~1.5–2.1 s; mean f steps under the default roll
(~0.75 k=1 + 0.22 k=2 + 0.03 k=3) about 0.5–0.8 s and 2.0–2.9 s, fl about
40% more. The screen 25× schedule is 9,975 steps at the live preset.

## Execution on the node

Every rank runs the same graphs on 16 of the step's 128 rows and meets the
others only in the per-site collectives ([design](design.md#training)).
Sharded masters and optimizer state put the static footprint per rank at
about 4.3 GiB screen, 9.2 bridge, 16.1 flagship, 35.5 extension, before
the working copies of the graphs' inputs, which the budget now sets aside
(`reserved_gib` in `memory_plan`). With ~70 GiB of activation budget the
largest raw replay per graph is about:

| Columns per logical forward | 1 | 2 | 3 | 4 | 6 |
|---|---:|---:|---:|---:|---:|
| Screen rows per replay | 16 | 8 | 4 | 4 | 2 |
| Flagship rows per replay | 8 | 4 | 2 | 2 | 1 |

These are fit estimates from 209.7 MiB per block invocation per row at
screen scaled with D, not captured plans. The per-row cost of a wider
replay is not expected to fall much: the 4090 measured no gain from 2 to 4
rows, only a ~4% penalty at one row, and the SM count is the same. The FLA
scan kernels launch B × H CTAs (20 at screen B = 2, 160 at B = 16), so
their occupancy is the one place a wider replay could pay; it is a
measurement, not a projection.

Communication per step, per rank, ring-equivalent: about 5.25 S + 7 R
bytes for S sharded and R replicated parameters (FP32 reduce-scatter plus
BF16 all-gather of the sites, FP32 all-reduce of the arena): 3.4 GB at
screen, 13.1 GB at flagship, i.e. about 14 ms and 53 ms at 250 GB/s
effective, 1.7% and 2.6% of a mean step. Nothing overlaps today. On a PCIe
box without NVSwitch (30–50 GB/s) the same traffic is 10–20% of a step: the
rental has to be SXM with NVSwitch, and the first-hour check measures the
actual reduce-scatter and all-gather bandwidth on the production slabs.

Startup: every rank compiles the same Inductor and Triton kernels; a shared
`TORCHINDUCTOR_CACHE_DIR` deduplicates across runs, not within the first
one, and eight compiling processes contend for the host CPUs. At 0.6 s
steps ten minutes of cold compilation is ~10% of a screen 25× run.

## Levers, ranked

Reductions of mean node step time; the ranges overlap and do not add.

| # | Lever | Screen | Flagship | Needs Hopper to build? |
|---|---|---:|---:|---|
| 1 | FP8 GEMM class (below) | 4–13% (+4–8% CCE) | 8–28% (+3–6%) | No; to measure, yes |
| 2 | FLA retune of the preconditioned-KDA chain | 4–10% | 3–8% | Yes |
| 3 | Expert grouped-GEMM retile | 2–5% | 3–7% | Yes |
| 4 | CCE fixed-configuration retune | 3–6% | 2–5% | Yes |
| 5 | Attention backend at head width 192 | 2–4% | 1.5–3% | No; to measure, yes |
| 6 | Replay width and per-graph saved set | 0–10% | 0–8% | Partly |
| 7 | Communication layout and overlap | 0.5–2% | 1–3% | No |
| 8 | Pointwise, MHDB, dispatch, tail | 1–5% | 1–4% | Yes |
| 9 | Optimizer step, host gaps | <1% | <1% | No |

1. **FP8 for the GEMM class.** Forward and dX GEMMs of the dense sites
   through `torch._scaled_mm`, the six grouped expert GEMMs in Triton
   `fp8e4nv`, and separately the CCE head. dW stays a BF16 → FP32
   accumulation in `dw_accum` (the `rowwise_with_gw_hp` shape): the FP32
   gradient contract is untouched and no dY^T transposes are materialized.
   The weight is quantized once per step by its owner from the FP32 master,
   so the FP8 bytes and scales are what the gather ships: the working copy
   becomes FP8 (plus a transposed copy for dX, so no memory is saved). The
   activation and gradient scales must be reconstructible from the step
   (stateless current scaling, or scales computed per invocation and kept
   for that step's recompute); delayed-scaling histories are new state the
   v42 checkpoint does not carry. The router GEMMs stay FP32. Quantization
   overhead is the whole design problem: on the 4090 prequantized operands
   ran 3.78 ms per cell against 6.92 BF16, but naive per-call amax + cast
   made it 7.20. torchao's own filters reject most of our shapes rowwise
   (gate/up N = 1664, down K = 832 at every scale; K = 768 at screen), so
   the pilot is tensorwise or per-block, and a measured 4090 win is the
   gate for building the rest. FP8 changes training numerics: runs paired
   against the BF16 recipe cannot mix it in.
2. **FLA retune.** The fork already carries Hopper tables (`IS_NVIDIA_HOPPER`
   warp lists and the 128-wide `CONST_TILING` in `precond_kda/chunk_bwd.py`)
   and the Triton #984 guard in `delta_rule/wy_fast.py` (`num_warps=4`
   miscompiles on sm_90). Ada-tuned constants remain in
   `precond_kda/chunk_intra.py` (`BWD_INTRA_BK=32`),
   `precond_kda/wy_fast.py` (BK = BV = 64), the `chunk_delta_h` state kernels,
   the ATK scans, and the fused conv / L2-norm kernels. Several autotune keys
   omit the batch, so a choice made at calibration width is reused at B = 16.
   Sweep with `pkda_bench.py` at H = 10 and 20 on real projected tensors,
   requiring gradients for gates, beta, the preconditioner scale, and
   `dt_bias`, not q/k/v agreement alone. Do not enable `safe_gate`, round
   `dg2`, or revisit the tensor-core diagonal.
3. **Expert retile.** `moe_kernels.py`'s `TILE_*` are Ada measurements
   (BK = 32, four warps). Tune all six GEMMs together on one real bank at
   production B under balanced and skewed routing; a faster forward that
   loses the fused SwiGLU backward is not a win.
4. **CCE retune.** `CCE_AUTOTUNE` cannot be used with the persistent
   classifier accumulator (the fork rejects it), so this is a fixed
   forward/backward configuration swept in the fork with disposable sinks,
   keeping the 128 × 128 token/vocabulary partition so the tile filter's
   semantics do not move.
5. **Attention backend.** `attention.py` pins `SDPBackend.FLASH_ATTENTION`,
   the FA2-derived in-tree kernel; torch 2.14 selects cuDNN attention on
   sm_90 by default (head 192 forward and backward, GQA) and reports up to
   1.75× over the flash backend on H100. FA3 has abi3 wheels at
   download.pytorch.org (torch ≥ 2.9), head 192 both directions, FP8 forward
   only. Compare forward + backward under the compiled wrapper and a graph
   at the real strides; an unsupported configuration must fail, not fall to
   math.
6. **Replay geometry.** `plan_replay` already takes the largest divisor that
   fits raw. The open choice is the saved set per graph: FLA saved set C
   costs a measured 2.9 ms per row-pass on the 4090 and buys ~80 MiB per
   invocation; A costs 0.6 ms for 21 MiB. Where 16-row replays fit raw
   without C, A is ~3% at screen.
7. **Communication.** Grouped launches that keep the site-major storage,
   then the weight all-gather overlapping later optimizer buckets after the
   finite-norm check. Gradient overlap needs last-use events or graph
   segmentation and is not worth it at the ceiling above. NCCL symmetric
   memory (torch 2.14, NCCL ≥ 2.27) covers all-gather, float all-reduce and
   reduce-scatter but not the dense `reduce`/`broadcast` or integer counts,
   and needs its own registered pool, not the activation pool.

Non-levers for the node, with the reason: FlashKDA (forward-only, K = V = 128,
no released backward; the fork's operator has an asymmetric preconditioned
update it does not implement); FA4 first (beta, no cp313 wheel, no compile
custom op); blind torchao conversion (the custom sink and grouped operators
are the paths that matter); BF16 or FP8 gradient communication (breaks the
FP32 contract for ~2%); more sharding or offload (memory is not the binding
constraint below extension); rank-0 monitor and snapshot restructuring
(measured 2.2 s per eval and ~2.5 s per screen snapshot on the 4090, under
1% at the default cadences); vocab-parallel CCE; `max-autotune` (already
on).

## What can be built without a Hopper card

The 4090 has FP8 tensor cores (sm_89) and runs cuDNN attention, so the
code paths of levers 1, 5, 6, 7, and 9 can be built, tested for numerics,
and even timed on Jobe; only their Hopper throughput is unknown.
Tensorwise and rowwise `_scaled_mm` work on sm_89; DeepSeek-style
1 × 128 / 128 × 128 blockwise scaling is sm_90-only, so the pilot develops on
the former and the recipe choice waits for the node. Levers 2, 3, 4, and 8
are tile constants and traces that only mean something on the target
architecture. NCCL multi-rank execution cannot run on one GPU: the plumbing
is covered by the two-rank gloo tests and `delta probe --ranks N` exists
for the first node hour.

| Work | Where it can be verified now |
|---|---|
| FP8 dense sites, FP8 working copies, expert GEMMs, CCE | Jobe: numerics (paired steps), 4090 throughput as an Ada signal |
| Attention backend selection by architecture | Jobe: correctness with cuDNN forced; speed only on the node |
| Per-graph saved set in the planner | Jobe: memory and time, fully |
| Communication grouping and gather overlap | Mac/Jobe: gloo two-rank tests; bandwidth only on the node |
| Compile-cache prewarm across ranks, snapshot gather batching | Mac/Jobe |
| FLA, expert, CCE retunes; FA3; NVLS/symmetric memory; the real profile | Node only |

## Kernel stack state (2026-09-16)

- flash-linear-attention upstream v0.5.2 (2026-07-27); the fork at 05ba6d83
  is synced with it and keeps the PKDA kernel patches. No FP8 anywhere in
  the KDA or gated-delta chains.
- FlashKDA (MoonshotAI): CUTLASS, sm_90a and Blackwell, forward only,
  K = V = 128, dispatched by FLA's `chunk_kda` under `use_qk_l2norm_in_kernel`.
  NVIDIA cudnn-frontend PR #1061 is the first sm90 KDA backward, unmerged.
- FlashAttention: FA4 4.0.0b30 (Hopper + Blackwell, beta, Python ≤ 3.12
  wheels, compile through `flex_attention(kernel_options={"BACKEND": "FLASH"})`
  only); FA3 3.0.0 abi3 wheels for CUDA 12.6–13.0.
- torch 2.14: cuDNN 9.24 with SDPA head 256 on sm_90; cuDNN attention
  default-selected on sm_90; GQA on every SDPA backend; blockwise
  `_scaled_mm` on sm_90 (CUDA ≥ 12.9); cuBLASLt grouped GEMM backend
  (default for FP16 on Hopper with CUDA ≥ 13.3); NVGEMM CuTeDSL templates in
  Inductor; symmetric memory for the standard collectives; an in-tree
  `nccl2` preview backend.
- torchao float8: recipes `tensorwise`, `rowwise`, `rowwise_with_gw_hp`;
  `mxfp8` Blackwell-only; `moe_training._scaled_grouped_mm` (rowwise on
  sm_89+); reported 1.25–1.5× end to end on Llama-3 8B with FSDP2.
- DeepGEMM: sm_90/sm_100, CUDA ≥ 12.9, dense `fp8_gemm_{nt,nn,tn,tt}`,
  m-grouped (MoE forward) and k-grouped (weight gradient) kernels; contiguous
  segment layouts, not our unpadded stable permutation.
- Transformer Engine 2.x: recommends blockwise scaling on Hopper; reports
  1.41× (blockwise) to 1.69× (delayed) on a 5B dense model on H200; grouped
  FP8 GEMMs on Hopper.
- No public FP8 cross-entropy head results.

## The first rental session

An eight-GPU node costs eight GPU-hours per hour even when one card is
busy: scripts, reference tensors, constraints, and data go on before the
clock starts. Ordered by information per GPU-hour:

1. **First ten minutes.** Eight distinct H100 SXM devices, NVSwitch topology
   and peer access, no MIG or sharing, exact driver / torch / Triton / NCCL /
   fork / data revisions.
2. **Minutes 10–20.** `delta probe --ranks 8`: NCCL through the real
   collectives, both graphs, the cross-rank digest of gathered weights.
   Then the production slabs' reduce-scatter and all-gather bandwidth.
3. **Minutes 20–40.** A short screen `fl` run on eight ranks reaching the
   `(2,2)` and `(3,2)` graphs; every rank's `memory_plan`, peak, reserved and
   free; gradient sums (not averages); finite gradients on every PKDA
   control and preconditioner parameter; replicated masters and expert
   biases identical across ranks after the gather.
4. **Minutes 40–60.** Save on eight ranks, restore on one and two, then the
   reverse; a forced eval, monitors, and snapshot; `delta stop` through the
   spool, the snapshot landing inside the launcher's 110 s.
5. **Then.** Warm screen and flagship traces per kernel class (replaces the
   table above); replay-width sweep with C versus A retention; the FLA,
   expert, CCE, and attention comparisons; one FP8 dense or expert pilot
   with real scales and backward; finally the integrated candidate on a
   paired short train and the deepest captures. Score candidates with the
   roll's weights, about 0.75 t₁ + 0.22 t₂ + 0.03 t₃ for f.

Stop debugging performance if collective correctness or restart fails;
that evidence is what the session is for.

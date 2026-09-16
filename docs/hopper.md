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
the graphs' persistent inputs, which the budget sets aside at the planned
widths (`inputs_gib` in `memory_plan`). With ~70 GiB of activation budget the
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

## A single GH200 first

A rented GH200 (the H100 die with 132 SMs and 96 GB of HBM3 at 4.0 TB/s, on
an aarch64 Grace host) is one H100 SXM's throughput at about half the
node's price per GPU-hour, and it runs the single-process path Jobe has been
running, so it is where the Hopper qualification starts: everything in the
rental session below except the collectives is a single-GPU measurement.
The machine profile lives in the meta repository's `bootstrap/rental.sh`
(aarch64 CUDA 13.2 lock, workspace, the token store from the private
bucket); `scripts/first_hour.sh` runs the session's first hour and writes
`logs/first-hour/<tag>/summary.json`. The Grace cores are also where the
whole dclm-100b stream gets built and published (`scripts/publish_store.sh`,
[operations](operations.md#tokenize)), overlapping the GPU work.

Unsharded, with the FP8 copies, the static footprint is 8.5 GiB at screen,
18.6 bridge, 32.5 flagship, and 72.1 extension. Against about 94.5 GiB
usable, the widest raw replay per graph (rows; 128 rows per step) from the
same block ledger as the table above:

| Scale | k=1 | k=2 | k=3 | (2,2) | (3,2) |
|---|---:|---:|---:|---:|---:|
| Screen | 16 | 8 | 4 | 4 | 4 |
| Bridge | 8 | 4 | 4 | 2 | 2 |
| Flagship | 8 | 4 | 2 | 2 | 1 |
| Extension | 1 | 1, 4 of 26 recomputed | 1, 21 of 39 | 1, 38 of 52 | 1, 72 of 78 |

Screen through flagship run raw with every intermediate kept; extension
wants the 144 GB variant or BF16 (about 10 GiB of static back). At the H100
factor of 3.0 the 25x schedules take about 14 h (screen f), 21 (screen fl),
58 / 85 (bridge), 160 / 234 (flagship) on the one card; the faster HBM and
FP8 make these slightly conservative.

## Levers, ranked

Reductions of mean node step time; the ranges overlap and do not add.

| # | Lever | Screen | Flagship | Needs Hopper to build? |
|---|---|---:|---:|---|
| 1 | FP8 GEMM class (landed: dense sites and experts; the head measured as a loss on the 4090, below) | 4–13% | 8–28% | No; to measure, yes |
| 2 | FLA retune of the preconditioned-KDA chain | 4–10% | 3–8% | Yes |
| 3 | Expert grouped-GEMM retile | 2–5% | 3–7% | Yes |
| 4 | CCE fixed-configuration retune | 3–6% | 2–5% | Yes |
| 5 | Attention backend at head width 192 (landed: cuDNN first, flash fallback) | 2–4% | 1.5–3% | No; to measure, yes |
| 6 | Replay width and per-graph saved set (landed; retaining measured ~0.2% on the 4090) | 0–10% | 0–8% | Partly |
| 7 | Communication layout and overlap | 0.5–2% | 1–3% | No |
| 8 | Pointwise, MHDB, dispatch, tail | 1–5% | 1–4% | Yes |
| 9 | Optimizer step, host gaps | <1% | <1% | No |

1. **FP8 for the GEMM class** (landed, `--precision fp8`, the default;
   [architecture](architecture.md#precision-and-initialization)). Forward
   and dX GEMMs of the dense sites through `torch._scaled_mm` with rowwise
   scales and the experts' two forward and two activation-gradient GEMMs in
   Triton `fp8e4nv`; the CCE fork can run the head on FP8 too
   (`CCEParams.fp8_classifier`, reached through `DeltaModel.fp8_classifier`)
   but the recipe leaves it in BF16, below. dW stays a BF16 → FP32
   accumulation (the `rowwise_with_gw_hp` shape): the FP32 gradient
   contract is untouched and no dY^T transposes are materialized.
   Every rank rewrites each site's FP8 copies (the matrix and its
   transpose, a scale per row of each) from the BF16 working copy after
   the gather, so nothing FP8 is communicated and the working copy the
   optimizer writes is unchanged; the FP8 copies cost one byte per element
   beyond it (screen static footprint 7.4 → 8.6 GiB on the 4090). Scaling
   is current and stateless (a row's scale is its own amax), so a
   recomputed forward quantizes identically and the v42 checkpoint carries
   nothing new; the recipe is a runtime field a resume inherits. The
   router GEMMs stay FP32. On the 4090 the quantization overhead costs
   most of the small-shape gain: forward + dX of the dense sites at
   M = 8,192 (two screen rows) 2.19 → 1.84 ms (1.19×; 1.29× at one row),
   1.11× per site once the BF16 dW is counted; at flagship shapes 8.78 →
   5.92 ms (1.48×, 1.26× with dW). Tensorwise scaling was slower than
   rowwise at every screen shape (1.06× against 1.19×) and worse
   numerically, so it is not offered. The routed expert kernels at the
   screen geometry (8,192 tokens, top-3): forward 0.75 → 0.51 ms (1.46×),
   backward without the weight gradients 0.79 → 0.66 ms (1.20×); the two
   BF16 dW GEMMs are now the larger half of the expert backward. Per-GEMM
   relative error 3.7e-2, expert output 6.5e-2. A paired eight-step screen
   run (`f`, one row per replay) tracked the BF16 recipe's loss to four
   digits at 25.4 against 26.9 s per k = 2 step (5.6%) and 38.8 against
   40.4 at k = 3. The head is the exception: its FP8 kernels are right
   (loss and log-partition within 2e-3 relative, gradients 2.7e-2) but its
   backward is bound by the lock-added partial gradients, dE per vocabulary
   tile and dC per token tile, whose bytes FP8 does not touch, and on sm_89
   the 8-bit MMA fragments and the in-register quantization of the
   probability tile spill at the 128 × 128 tile: the shapes that fit
   (256 × 64 × 64 forward, 64 × 64 × 32 backward) double that traffic. In
   the model's own head at 8,192 rows the backward took 77.6 ms against
   23.4 in BF16, and the eight-step run 39.3 s per k = 2 step against 25.4,
   so the recipe keeps the head in BF16; `DeltaModel.fp8_classifier` is the
   one-line hook for trying it on Hopper, where wgmma keeps the operands in
   shared memory and 128-wide tiles may hold. FP8 changes training
   numerics: runs paired against the BF16 recipe cannot mix it in, and
   Hopper's own choices (blockwise scaling is sm_90-only; the head) are
   first-hour measurements against this rowwise recipe.
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
   forward/backward configuration swept in the fork with disposable sinks.
   The BF16 halves share the 128 × 128 token/vocabulary partition; the FP8
   halves (`_cce_best_config_fp8` and its backward twin in
   `tl_autotune.py`) are 4090 measurements that need only agree on the
   vocabulary tile, and Hopper's wgmma wants larger ones.
5. **Attention backend.** `attention.py` now asks for cuDNN attention first
   and the flash kernel second; on the 4090 cuDNN has no kernel at head
   width 192 and the fallback is bitwise flash, on sm_90 torch 2.14 serves
   head 192 forward and backward with GQA through cuDNN and reports up to
   1.75× over the flash backend. Whether it wins there is a first-hour
   measurement (forward + backward under the compiled wrapper and a graph);
   if it loses, the order flips. FA3 has abi3 wheels at
   download.pytorch.org (torch ≥ 2.9), head 192 both directions, FP8 forward
   only.
6. **Replay geometry.** `plan_replay` takes the widest replay that fits,
   its recurrences keeping their WY representation and chunk states when
   that fits too, else rebuilding them, then block recomputation. Measured
   on the 4090 (screen f, k = 1, two-row replays, eight 128-row steps):
   keeping 108.9 s, rebuilding 109.1 s, losses identical to four digits, so
   the rebuild costs ~0.4 ms per two-row replay, not the 2.9 ms per row-pass
   round 11 measured before the fork's upstream sync; the choice is free
   where it fits and worth nothing else. With it, screen f k = 3 fits raw
   with every intermediate kept (12.7 of 13.5 GiB on the 4090). What remains
   to measure is whether width beyond the smallest replay buys anything on
   Hopper (the FLA scans), and where the 106 ms per row of a fresh run
   comes from against the ledger's 94.4 after twenty steps (an untrained
   head filters no CCE tiles; the head sink is now FP32).
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
1 × 128 / 128 × 128 blockwise scaling is sm_90-only, so the landed recipe is
rowwise and the node measures whether blockwise pays. Levers 2, 3, 4, and 8
are tile constants and traces that only mean something on the target
architecture. NCCL multi-rank execution cannot run on one GPU: the plumbing
is covered by the two-rank gloo tests and `delta probe --ranks N` exists
for the first node hour.

| Work | Where it can be verified now |
|---|---|
| FP8 dense sites and expert GEMMs (landed); FP8 head (built, off) | Jobe: numerics (paired steps), 4090 throughput as an Ada signal |
| Attention backend selection by architecture | Jobe: correctness with cuDNN forced; speed only on the node |
| Per-graph saved set in the planner (landed) | Jobe: memory and time, fully |
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
- Found on the GH200 (2026-09-16): Inductor's split-scan kernels record
  their workspace's block minimum as `min_split_scan_rblock` while the
  coordinate-descent tuner and the spill halving floor `R0_BLOCK` on
  `min_rblock`, so the tuner can pick a block whose extra programs write
  past the workspace. The 4090 never crossed the bound; the GH200 faulted in
  the routers' rank scan under every recipe. `inductor.py` records the
  minimum under the key the tuner reads; a 15 x 12,288 `cumsum` under
  max-autotune with coordinate descent reproduces it upstream.

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

On one card, `scripts/first_hour.sh` covers items 1, 3, and 4 without the
collectives: inventory, probe, a short `fl` run from the recurrence roll's
start with an eval and snapshot midway, a resume, a SIGINT stop, and the
summary (memory plan, peak, seconds per step by graph).

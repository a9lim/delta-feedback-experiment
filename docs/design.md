# Experimental design

This document is the authoritative research and scale plan. It defines the
comparisons, data, training schedules, execution stages, evaluation, evidence
rules, and promotion gates. The exact flagship model and optimizer are defined
in [architecture.md](architecture.md). Accepted scientific evidence belongs in
[findings.md](findings.md); active scratch work belongs in
[journal.md](journal.md).

## Question and claim boundary

The experiment asks whether two packages improve a common PKDA/GGQA hybrid
independently or interact when pretrained together:

1. **MHDB** gives each sublayer a transient multi-head read over the current
   column's seed and four-layer block deltas.
2. **FBT** carries a latent payload from one token column to the next through a
   mandatory token-gated fusion.

Hard Delta Feedback (`df`) combines both. The primary comparison is a two by
two factorial on one byte-identical hybrid trunk:

| Arm | Hybrid trunk | MHDB | FBT | Parameters | Active non-embedding |
|---|---:|---:|---:|---:|---:|
| `base` | yes | no | no | 256,275,240 | 139,588,392 |
| `mhdb` | yes | yes | no | 256,330,536 | 139,643,688 |
| `fbt` | yes | no | yes | 257,457,192 | 140,770,344 |
| `df` | yes | yes | yes | 257,514,792 | 140,827,944 |

`base` deletes both packages. `mhdb` routes within a plain-embedding column
but emits no recurrent payload. `fbt` uses the asymmetric entry fusion and a
bare `payload_norm(h_top)` payload. `df` uses the fused input as its MHDB seed
and adds routed block-source enrichment to the normalized top-state payload.

Pure-GQA `vanilla` is a fifth external control with 229,954,560 parameters,
113,267,712 active non-embedding parameters, and a structurally different
trunk. `vanilla` versus `base` tests the whole PKDA/GGQA trunk replacement; it
does not enter the MHDB by FBT interaction estimate. There is no optional-
feedback arm.

The registered claim surface is pretraining behavior under the exact data,
geometry, optimizer, and schedules below. No result alone establishes a
general claim about recurrent transformers, depth routing, reasoning, or
adaptive computation.

## Discovery model

All five arms are configurations of one plain-PyTorch `DFModel`. The four
factorial arms are a scale-adapted projection of the flagship architecture:

```text
[PKDA, PKDA, PKDA, gated global GQA] x 3
```

| Field | Jobe and Prime screen value |
|---|---:|
| Vocabulary | 151,936, Qwen3 tokenizer |
| Width | 768 |
| Layers / four-layer cells | 12 / 3 |
| SwiGLU intermediate width | 3,328 |
| Context / predictions per row | 1,024 |
| Vanilla RoPE theta | 1,000,000 |
| GQA query / KV heads / head width | 8 / 4 / 96 |
| PKDA Q/K/V heads / head width | 10 / 128 |
| PKDA Q/K/V projection width | 1,280 |
| PKDA convolution width | 4 |
| Routing groups | 4, exactly the KV-head count |
| RMSNorm epsilon | `1e-6` everywhere, including PKDA output |

The screen halves the flagship's residual, SwiGLU, and PKDA projection widths:
`768 / 3,328 / 1,280` versus `1,536 / 6,656 / 2,560`. The 10-by-128 PKDA
geometry therefore preserves the flagship's `5/3` recurrent-projection ratio.
This is the Kimi scaling prior; Preconditioned DeltaNet supplies the
preconditioner rather than an independently validated head-count optimum. The
single `1e-6` RMSNorm epsilon is invariant across scale and norm sites; it
deliberately replaces upstream PrecondKDA's `1e-5` output-norm default. Hybrid
global layers are dense causal NoPE GGQA. `vanilla` instead retains twelve
bias-free RoPE GQA layers with packed QKV, per-head Q/K RMSNorm, and no
attention-output gate. MHDB remains four-headed because its groups follow the
global KV-head count, not the PKDA head count.

The complete hybrid trunk initializes byte-identically across `base`, `mhdb`,
`fbt`, and `df`. MHDB parameters are paired across `mhdb` and `df`; FBT
parameters are paired across `fbt` and `df`. Factor-private modules use
independent deterministic initialization streams and never advance the common
stream. `vanilla` retains its own exact initialization and state layout.

Mechanism semantics, initialization, optimizer ownership, and cache contracts
are defined once in [architecture.md](architecture.md). The screen changes the
geometry and cell count, not those semantics.

## Data and pairing

Hugging Face's `HuggingFaceFW/fineweb-edu` `sample-100BT` configuration is the
corpus authority. The registered input is its canonical streaming order at
dataset commit `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, tokenized with
`Qwen/Qwen3-0.6B` at commit
`c1899de289a04d12100db370d81485cdf75e47ca`. The `data-build` extra pins the
four packages that compile this stream, and `meta.json` records those realized
versions. Every non-empty document is followed by EOS.

Training never consumes the live iterable directly. Each execution surface
materializes a local contiguous uint32 representation from those pinned Hugging
Face sources so that step-addressed rows, rank sharding, validation, and resume
do not depend on network or iterator state. This store is a compiled training
representation, not a separately curated or re-hosted corpus.

The held-out validation slice is the stream head. Training follows in
contiguous uint32 shards. One row is a non-overlapping `seq_len + 1` window, so
1,025 stored tokens provide 1,024 predictions at screen scale. Rows may cross
document boundaries. Step `n` directly addresses:

```text
first_row(n) = (n - 1) * batch_rows
```

Every arm in a registered comparison shares tokenizer, resolved data, global
row order, initialization seed pairing, data seed, batch geometry, optimizer
recipe, and schedule. Feedback pass counts are keyed by data seed and step;
prefix lengths and jitter are additionally keyed by the microbatch's first
global row. They do not depend on ambient RNG state. Resume returns to the same
row and the same keyed feedback draws.

The canonical stream target is 57B stored tokens including the held-out
prefix. The fresh Prime schedule needs exactly 55,010,880 training rows, or
56,386,152,000 stored training tokens, leaving about 584M tokens of headroom
after the held-out prefix.

## Training protocol

### Optimizer batch

Every scale uses 327,680 predicted tokens per optimizer update:

| Surface | Realization |
|---|---|
| Jobe screen | 320 rows x 1,024 predictions; 80 four-row microbatches |
| Prime screen | 8 ranks x 4 rows x 10 accumulation microsteps |
| Flagship | 8 ranks x 1 row x 8,192 predictions x 5 accumulation microsteps |

All arms use the NorMuonH/NAdam partition and defaults in
[architecture.md](architecture.md). After all microbatches—and, for
distributed runs, all ranks—have accumulated, the single global FP32 gradient
vector is clipped to L2 norm 10.0 immediately before both optimizer steps.
Telemetry reports the pre-clip norm. Equal optimizer steps are matched-data,
not matched-compute, comparisons.

### Feedback passes

Feedback arms train with parallel Jacobi passes over a full sequence. Pass 1
is ordinary teacher forcing with plain embeddings. For every later pass:

1. take the preceding pass's payload without detaching it;
2. add keyed uniform jitter, `[-0.02, 0.02]` by default;
3. shift the payload one position right and insert zero at position 0;
4. draw a per-row plain-prefix length uniformly from `1..seq_len-1`;
5. use plain embeddings on that prefix and FBT-fused inputs on the suffix;
6. run the complete stack again.

The shift and prefix preserve token causality. Position 0 is always plain,
the last executed position is always fused, and every row therefore has at
least one feedback position. A `k`-pass batch trains a
feedback horizon of `k-1` transitions and costs `k` transformer evaluations.
Non-feedback arms always use one pass; feedback arms use one pass before the
feedback boundary and two or three passes on every step after it.

Let `ell_k` be mean next-token cross-entropy on pass `k`:

```text
K = 1:  loss = ell_1
K > 1:  loss = ell_1 + mean(ell_2, ..., ell_K)
```

Pass 1 always has unit weight; all feedback passes together have unit weight.
During cooldown, the same combination applies to the squared log-partition
penalty `mean(logsumexp(logits)^2)` with coefficient `1e-5`.

### WSD schedule

Both parameter-group learning rates use one warmup-stable-cooldown multiplier.
Warmup and cooldown occupy `round(warmup_frac * steps)` and
`round(cooldown_frac * steps)` updates. Their defaults are 0.02 and 0.20.
Warmup rises linearly to the stable rate. For local cooldown progress `u` in
`(0, 1]`, the multiplier is `1 - sqrt(u)` and therefore reaches zero at the
terminal step. The default 10,745-step Jobe schedule is:

| Phase | Steps | Pass behavior |
|---|---:|---|
| Warmup | 1–215 | one pass |
| Stable heat | 216–8,596 | one pass through 8,059; then feedback arms draw 2 or 3 passes |
| Cooldown | 8,597–10,745 | feedback arms draw 2 or 3 passes |

The feedback boundary is `round(feedback_start * steps)`, independently of
the learning-rate phases, and defaults to three quarters of the schedule.
Before it every step is one pass. After it there are no one-pass steps: each
step draws three passes with probability `three_pass = 0.12` and two passes
otherwise, so the fused mode is trained on every update rather than eroded
between feedback steps. Across the full run this targets 75% / 22% / 3% and
an expected feedback-arm compute multiplier of 1.28 pass-tokens per predicted
token. Exact realized pass-tokens are recorded.

`--max-steps` caps additional steps in one process. It never rescales the
schedule, feedback boundary, protected checkpoints, or any state-defining
field.

## Execution contract

### Portable semantics

CPU and MPS use PyTorch scaled-dot-product attention, the literal recurrent
PKDA equations, the algebraic MHDB router, chunked tied-head cross-entropy, and
eager execution. This path owns fast invariant tests and analysis while using
the same model, loss, optimizer, data, and checkpoint semantics.

### Jobe CUDA path

Jobe is the authoritative single-GPU screen surface, on PyTorch 2.14 and
CUDA 13.2. It uses:

- BF16 trunk activations with FP32 parameters, accumulated gradients, and
  optimizer state;
- execution of exactly the `seq_len` input positions of every stored row; the
  final stored token is only ever a target, and keyed jitter is still drawn at
  the stored-row width so the draws do not depend on the executed length;
- persistent FP32 gradient buffers: the tied embedding scatters lookup
  contributions into its buffer and adds the CCE classifier's BF16 gradient;
  large projections accumulate through cuBLAS with the FP32 sink as both the
  unit-beta addend and output, avoiding a separate projection weight gradient.
  These parameters hand autograd no further gradient;
- address-stable BF16 shadows of every sink-fed projection's concatenated
  weights and of PKDA's three NAdam-owned control matrices, refreshed together
  with the classifier shadow once per optimizer update, so no replay casts or
  concatenates FP32 parameters;
- the workspace FLA fork's PKDA chunk kernel retaining its WY and chunk-state
  intermediates for backward and recomputing only the gated query, one GEMM
  over the concatenated Q/K/V weights per PKDA layer feeding one causal
  convolution launch that also applies the SiLU and the per-head Q/K L2
  normalization, recomputes its nonlinear derivative once per backward tile
  plus a compact halo and reuses it across convolution taps, and writes Q,
  K, and V as three contiguous slabs whose gradients return separately, so the
  recurrence reads them without contiguity copies and the backward never
  concatenates them. The screen's Ada backward uses 32-row tiles while its
  forward retains 64-row tiles; heads narrower than 32 use per-tap
  recomputation. Intra-chunk backward evaluates two direct gate differences
  together, retaining the unbounded-gate semantics and FP32 state boundaries.
  The fork also provides Ada-tuned inter/solve forward
  kernels, a WY/inter backward that keeps its value-side tiles across its
  key loop at 128-wide value tiles, a gate backward that folds the decay
  gradient's reverse cumulative sum and narrowing into its own kernel, a
  preconditioner whose forward products are kept for backward instead of
  recomputed, whose chunk kernels run without register spills, and whose
  summary kernel folds in the intra-chunk key gradient, and fused PKDA
  output norm/gate;
- compiled FlexAttention for full-sequence, prefill, native GQA, and cached
  decode, with each gated global layer projecting Q/K/V and its gate in one
  GEMM. Causal block masks are shared across layers and passes; decoding
  explicitly updates the KV cache and exposes only its valid prefix. The
  Triton training backend ships with PyTorch, and short-query decoding uses
  FlexAttention's automatic backend selection without external `flash-attn`;
- a fixed-capacity Triton MHDB router that reads every source once per token
  with the source softmax folded in online, holds each site's width-`D` null in
  place, keeps full-width RMS coupling in its analytic backward, and reduces the
  query and null gradients from FP32 per-program partials;
- an address-stable, non-checkpoint BF16 classifier shadow refreshed from its
  FP32 tied-embedding master once per optimizer update;
- the workspace cut-cross-entropy fork with BF16 operands, capture-safe
  fixed-shape no-ignore preprocessing, and its native differentiable
  log-partition output for the exact squared-log-partition gradient, tiling
  the classifier in both halves through the microbatch's own ascending
  mean-logit ordering of the vocabulary (the mean logit is linear in the
  embeddings, so one classifier product with the mean embedding yields it
  before the forward, inside the graph and without state; ascending because
  the backward accumulates each row's embedding gradient across vocabulary
  tiles in BF16 through locks and the smallest contributions must arrive
  first), storing each tile's per-row maximum logit and each target's tile
  index in the forward, and
  skipping every backward tile the gradient filter would drop before
  recomputing its logits, a decision identical to the late filter's. Skipped
  tiles need no vocabulary-permutation load or target-membership matrix;
- the Triton PKDA control-gradient packer;
- each compiled block also emitting the residual's distance from its cell
  entry, so partial and completed block deltas are never formed eagerly
  between blocks;
- fixed-shape, exhaustive Inductor autotuning for compiled global-attention
  blocks and segmented PKDA blocks around the opaque FLA recurrence, with one
  process-wide Dynamo recompile budget covering every block, source-count,
  mode, and grad-state specialization so no path silently demotes to eager;
- one fixed-address train CUDA graph per reachable pass count and shared-pool
  no-grad validation graphs, with Python cyclic garbage collection kept
  outside capture;
- BF16 keyed jitter written directly into graph input buffers;
- internal activation checkpointing above the measured work threshold;
- asynchronous pinned-host snapshot staging and atomic background writes.

These choices are not exposed as experiment axes. The default geometry uses
four 1,024-token rows per microbatch and 80 microbatches per update.
The PyTorch 2.14 / FlexAttention runtime passed the full CUDA gate and a
trained-checkpoint short-update comparison on Jobe. The graph pool reserves
about 23.0 GiB; paired one/two/three-pass updates took 4.74/9.47/14.21 s.
See [runtime qualification](runtime-qualification.md) for exact revisions,
numerical differences, and measurement limits. These are engineering checks,
not training-quality findings. Reproducible trained-input, full-gradient, and short-update checks
are in `scripts/kernel_inputs.py`, `scripts/kernel_qualification.py`, and
`scripts/kernel_training_check.py`.

Hopper uses the same FlexAttention API but remains hardware-unqualified until
the forward/backward, cache, graph, memory, and throughput gates pass there.
The optional external FlashAttention-4 backend is not part of this runtime.

Inductor artifacts live in
`~/.cache/delta-feedback/torchinductor` by default; the probe performs the
fixed-shape search once and later processes reuse the cache with no autotuning
work. The reserved graph pool is the concurrency boundary. Screen runs remain
serial; concurrent execution is outside the qualified deterministic path.

### Checkpoints and queue

New snapshots use checkpoint contract v23, and only v23 is resumable. V16-v22
snapshots remain readable for evaluation and forks but cannot be continued:
v16 used different feedback-prefix draws, v16-v17 predate the Nesterov
NorMuonH update, v16-v18 predate NAdam, and v19 stores the retired fixed-step
warmup contract. V20 predates the split NAdam rates and the global clipping
contract.
V21 predates inverse-square-root fan-in initialization of NorMuonH matrices,
and v22 predates the unified NAdam rate and reduced NorMuonH rate. A
snapshot contains the model,
both optimizer states, fixed
NorMuonH radii, state-defining arguments, cumulative step, and
Python/Torch/CUDA RNG state. A resume inherits every state-defining field and
rejects explicit conflicts. Runtime paths, device, evaluation cadence, snapshot
cadence, and evaluation-row count may change between invocations.

Each run retains the latest two snapshots plus protected snapshots at the
cooldown boundary, the feedback boundary (the last one-pass state), and the
end of the run, so cooldown and feedback variants can continue from the
exact pre-boundary state. The durable queue stores exact arguments, not Git state.
Source changes do not stop an active child; the worker refreshes before the
next job and runs that job's probe from the current checkout. `df stop queue`
atomically removes every pending job without touching the active child or
worker; `df stop live` does the converse.

## Evaluation

### Language-model modes

Every evaluation point reports pass-1 held-out cross-entropy as `val`. Feedback
arms also report `val_fused`, a second pass with plain-prefix length 1.

- **Standard:** one plain prompt prefill and no latent feedback during decode.
- **Soft:** one plain prefill, then one feedback transition per generated
  token.
- **Fused:** an additional fused prompt pass, then the same feedback decode.

No-feedback arms retain their Standard metric when a factorial table is shown
for Soft or Fused mode. Modes are never mixed inside one effect estimate.

### Factorial effects

For validation loss `L`, where lower is better, define gain over `base`:

```text
G_A = L_base - L_A
```

The interaction is:

```text
I = G_df - G_mhdb - G_fbt
  = L_mhdb + L_fbt - L_df - L_base
```

`I > 0` is superadditive loss reduction, `I = 0` additive, and `I < 0`
subadditive. Compute it on paired checkpoints and separately by inference mode,
then aggregate paired-seed differences. Report `L_vanilla - L_base` separately
as the whole-trunk contrast.

Every result reports parameters, predicted tokens, and pass-tokens. A
matched-compute view compares curves or checkpoints at equal cumulative
pass-tokens; each feedback pass counts in full.

### Contraction

A feedback result is uninterpretable without its recurrent stability trace.
Repeatedly apply fully fused prefill with prefix length 1 and record held-out
loss plus:

```text
mean_token ||h_top^(k) - h_top^(k-1)||_2
```

The training monitor uses eight iterations; promotion uses at least 30. Stable
loss and decaying update norm support safe self-composition. Oscillation or
rising loss invalidates the feedback comparison even when a one-step metric
improves. Contraction is a stability condition, not a quality or reasoning
claim.

### Routing and causal probes

Routing observables remain per-site and per-head. They include source mass,
per-token maximum, normalized entropy, cross-head Jensen-Shannon divergence,
source RMS, null-vector RMS, query norm, and query cosine similarity. Source
labels distinguish null, seed, completed blocks, current partial, and payload.

`scripts/route_report.py` produces held-out route maps and query geometry.
`scripts/payload_swap.py` replaces the learned payload mixture with top-only,
uniform, or single-source alternatives on the same checkpoint. This is a
co-adapted causal intervention, not an independently trained arm comparison.

## Scale plan

There is no continuation ladder and no 100x run. The program has three stages:
a two-seed Jobe discovery factorial, a fresh one-seed Prime minimum contrast,
and the separately implemented flagship.

### Stage 1: Jobe 25x discovery

Each Jobe arm trains for 10,745 steps, 3,520,921,600 predicted tokens, and a
320-row global batch on the RTX 4090. The four factorial cells receive
25.00–25.22 predicted tokens per active non-embedding parameter; the smaller
pure-GQA control receives 31.08. Two paired initialization seeds are registered.

The stage asks:

1. whether the hybrid trunk improves on the pure-GQA control;
2. the signs and paired magnitudes of MHDB, FBT, joint, and interaction effects;
3. whether every feedback arm remains stable under self-composition;
4. whether null, seed, block, and payload paths are actually used.

No run is complete under the current screen contract; `screen-df-s1` ran
under the superseded mixed-heat recipe and remains a diagnostic checkpoint
only. This is a
sensitivity and interaction screen, not a decisive test of FBT formation at
high token-per-parameter ratio.

### Stage 2: fresh Prime 400x minimum contrast

After an admissible Jobe result, only `{base, df}` is pretrained from fresh
paired initialization on one 8xH100-80GB Prime node. Both arms use seed 1,
data seed 0, global row zero, 171,909 steps, and 56,331,141,120 predicted tokens.
Neither loads a Jobe model or optimizer checkpoint.

| Arm | Exact predicted tokens | Active-token ratio |
|---|---:|---:|
| `base` | 56,331,141,120 | 403.551759 |
| `df` | 56,331,141,120 | 399.999741 |

The fresh WSD schedule is:

| Phase | Steps | Pass behavior |
|---|---:|---|
| Warmup | 1–3,438 | one pass |
| Stable heat | 3,439–137,527 | one pass through 128,932; then feedback arms draw 2 or 3 passes |
| Cooldown | 137,528–171,909 | feedback arms draw 2 or 3 passes |

The pair consumes 112.662B predicted tokens and approximately 128.435B
expected pass-tokens. It tests only the complete DF package against its shared
hybrid baseline; component attribution remains a Jobe factorial claim.

Prime data staging occurs only after selecting an 8xH100 provider and location,
but before provisioning the H100 node. Create a provider-local persistent disk
sized for the compiled stream and run artifacts, attach it to an inexpensive
compatible staging instance, install the exact `data-build` stack, and run
`df tokenize` directly from the pinned Hugging Face dataset and tokenizer into
the mounted data path. The resulting metadata and byte-checksum manifest must
match the canonical Jobe prefix before the disk is detached and attached to
the H100 node. The registered path requires neither live training reads from
Hugging Face nor a separately hosted copy of the compiled store.

The schedule is expressible by the current single-process trainer, but Prime
distributed execution is not runnable. Its gate must add exact row sharding,
DDP accumulation and synchronized global clipping, rank-safe telemetry and
snapshots, persistent artifact staging, cross-rank optimizer parity, restart
tests, persistent-disk read qualification, and H100 memory/throughput
qualification. The qualified configuration and projected rental cost are
reviewed before provisioning.

### Stage 3: flagship

The flagship is the exact hard-DF architecture in
[architecture.md](architecture.md): 1,335,420,192 total parameters and
1,102,046,496 active non-embedding parameters. Its budget is 400 predicted
tokens per active non-embedding parameter before batch alignment.

| Quantity | Registered value |
|---|---:|
| Sequence length | 8,192 predictions |
| Global batch | 40 sequences = 327,680 predictions |
| Distributed batch | 8 ranks x microbatch 1 x accumulation 5 |
| Unrounded target | 440,818,598,400 predicted tokens |
| Optimizer steps | 1,345,272 |
| Exact aligned budget | 440,818,728,960 predicted tokens |
| Warmup | steps 1–26,905 |
| Stable heat | steps 26,906–1,076,218 |
| Cooldown | steps 1,076,219–1,345,272 |
| Feedback boundary | after step 1,008,954 |
| Whole-run pass mixture | expected 75% / 22% / 3% |
| Expected compute | approximately 564.248B pass-tokens |

The aligned schedule exceeds the mathematical 400x target by 130,560 tokens,
less than one optimizer batch, and realizes 400.000118 predicted tokens per
active parameter.

The registered target is replicated DDP over eight H100 80GB GPUs with BF16
autocast, FP32 parameters/optimizer state, and no tensor, pipeline, context, or
parameter sharding. Every block is activation-checkpointed on every pass;
payload and source-bank graphs remain differentiable. The exact implementation
must establish parameter and compute accounting, portable/chunk/recurrent PKDA
parity, cache continuation, gated-GQA parity, MHDB source identities,
optimizer partition and radius invariants, distributed row assignment and
clipping, checkpoint portability, restart behavior, memory, and throughput.

## Gates

### Engineering qualification

`df probe` must pass from the source that will run the job. On Jobe it covers
portable tests, Triton router value/weight/gradient parity for four and eight
groups, PKDA forward/backward and cache parity, cut-cross-entropy z-loss
parity, causal/prefix FlexAttention values and gradients, cached decoding,
eager/captured evaluation parity, every
schedule-reachable graph, finite optimizer updates, exact NorMuonH radius
preservation, contraction monitoring, and production-scale snapshot staging.

Engineering qualification establishes implementation and numerical coherence.
It is never an experiment finding.

### Jobe admissibility and Prime entry

A Jobe comparison is admissible only when all registered paired runs complete
with the same stream, order, recipe, seeds, and schedule, and every feedback
arm has a healthy contraction trace.

`base` must clearly improve on `vanilla` in Standard mode to justify the
hybrid trunk replacement. MHDB must produce a resolvable paired effect for the
factorial to serve as a sensitivity gate. A stable FBT null at 25x does not by
itself exclude Prime because the discovery screen is far below the registered
token-per-parameter regime. Prime proceeds only when the complete DF package
is credible enough to justify a fresh high-token trial.

The Prime pair is admissible only when both arms start from fresh paired
initialization, consume the same row-zero stream prefix, and complete on one
qualified distributed recipe. Jobe checkpoints may be cross-hardware
diagnostics but never initialize Prime.

### Flagship promotion

Promote only when:

1. the fresh Prime pair shows a DF advantage over `base` at matched
   token-equivalent compute, while Jobe supplies coherent two-seed component
   attribution;
2. Prime feedback remains stable for at least 30 fused self-compositions;
3. routing and same-checkpoint interventions do not reveal a trivial unused or
   bypassed mechanism;
4. the exact flagship implementation and distributed restart gates pass;
5. a9 explicitly approves the flagship spend with both screen results in hand.

## Exclusions and interpretation

The registered experiment excludes adaptive pause or halting tokens,
token-conditioned payload queries, more than one explicit previous-column
payload, per-layer cross-column source banks, alternative optimizer arms,
long-context continuation, instruction tuning, and downstream reasoning
claims.

Interpret every result with these constraints:

- expected effects may be small relative to run noise, so paired order, seeds,
  complete runs, and pass-token accounting are part of the causal design;
- the screen and flagship are syntheses, not numerical reproductions of any
  one source architecture;
- a 25x FBT null may reflect missing formation conditions, while a fresh 400x
  `df - base` null is stronger evidence against the complete package at this
  model scale but cannot identify the responsible component;
- PKDA's asymptotic cache advantage does not guarantee a realized speedup at
  8,192 context; promotion uses measured end-to-end train, prefill, decode,
  memory, and pass-token costs;
- exact arm parameter counts accompany every reported token budget because
  gates, fusion, and routers make the arms unequal.

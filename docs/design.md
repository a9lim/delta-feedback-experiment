# Research design: a recurrent model organism

This project synthesizes published architectures into a small, reproducible
organism for mechanistic interpretability. We want to inspect latent
computation, explain its causal role in behavior, and test ways to monitor it.
The research product is the organism together with analysis methods and
well-supported explanations. Capabilities are an enabling condition for
interesting behavior, not the objective or a claim to frontier progress.

[Architecture](architecture.md) owns the model and optimizer equations.
[Interpretability](interpretability.md) owns the analysis and monitoring
protocol. This document owns the controlled specimen-building experiment, data,
schedules, evaluation, and decision rules. Optional longer and larger training
specifications live in [scaling.md](scaling.md).

## Questions and success criteria

The central question is whether we can build a tractable recurrent system whose
internal computations support reproducible causal investigation. “Model
organism” describes its intended experimental role; it is not a claim that it
already reasons like a larger model or represents every form of opaque
computation.

The program asks how information is retained and transformed across token
columns, which paths affect behavior, how recurrent dynamics depend on
training, and which internal observables support reliable monitoring. Success
requires the following evidence, developed incrementally:

1. **A reproducible specimen:** a checkpoint with exact training provenance,
   faithful replay, and documented state boundaries.
2. **A behavioral phenomenon:** a repeatable task or contrast that actually
   depends on the recurrent channel being studied.
3. **A causal account:** interventions with controls that test a specific
   explanation of the phenomenon, including the explanation's failures.
4. **A monitoring evaluation:** an explicitly defined target, frozen monitor,
   held-out evaluation, and measured false positives and misses.

The latter three are research work to do, not properties established by the
current trainer. A stable null effect on validation loss can coexist with a
useful mechanistic contrast. An unused channel is an informative negative
result and a reason to reconsider that channel, not evidence of a hidden
reasoning mechanism. Better loss alone satisfies none of these criteria.

## Controlled specimen family

The implemented `DFModel` has five exact arms. The main two-by-two factorial
holds the hybrid trunk fixed and varies two packages:

- **MHDB:** transient grouped reads of the seed and block deltas within a column.
- **FBT:** token-gated latent payload transfer between token columns.

| Arm | MHDB | FBT | Parameters | Active non-embedding |
|---|---:|---:|---:|---:|
| `base` | no | no | 256,275,240 | 139,588,392 |
| `mhdb` | yes | no | 256,330,536 | 139,643,688 |
| `fbt` | no | yes | 257,457,192 | 140,770,344 |
| `df` | yes | yes | 257,514,792 | 140,827,944 |
| `vanilla` | no | no | 229,954,560 | 113,267,712 |

`vanilla` is a separate twelve-layer RoPE GQA control. The other four share
PKDA/GQA: even `base` has recurrent mixer memory. Thus `vanilla - base` tests
the whole trunk change, and the factorial tests additional depth routing and
latent feedback. It cannot isolate every architectural ingredient.

`mhdb` uses a plain-embedding seed and emits no payload. `fbt` emits
`payload_norm(h_top)`. `df` uses the fused seed for MHDB and enriches its
payload with routed sources. Consequently the interaction concerns the complete
packages, including `df`'s payload enrichment. It is not an estimate of one
isolated routing edge. Same-checkpoint interventions answer narrower questions
about a trained specimen; separately trained arms answer training effects.

The tied-depth `df-loop` is a separate, unimplemented specification. It does
not enter this factorial. Its use must be justified by an iteration-level
question and pass its [adoption gates](depth-architecture.md#adoption).

## Small-organism geometry

All five arms are configurations of one plain-PyTorch `DFModel`. The four
factorial arms are the small hybrid architecture:

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

Relative to the optional larger reference, the screen halves the residual,
SwiGLU, and PKDA projection widths: `768 / 3,328 / 1,280` versus `1,536 / 6,656
/ 2,560`. The 10-by-128 PKDA geometry therefore preserves the `5/3`
recurrent-projection ratio. This is the Kimi scaling prior; Preconditioned
DeltaNet supplies the preconditioner rather than an independently validated
head-count optimum. The single `1e-6` RMSNorm epsilon is invariant across scale
and norm sites; it deliberately replaces upstream PrecondKDA's `1e-5`
output-norm default. Hybrid global layers are dense causal NoPE GGQA. `vanilla`
instead retains twelve bias-free RoPE GQA layers with packed QKV, per-head Q/K
RMSNorm, and no attention-output gate. MHDB remains four-headed because its
groups follow the global KV-head count, not the PKDA head count.

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
`Qwen/Qwen3-0.6B` at commit `c1899de289a04d12100db370d81485cdf75e47ca`. The
`data-build` extra pins the four packages that compile this stream, and
`meta.json` records those realized versions. Every non-empty document is
followed by EOS.

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

The canonical stream target is 57B stored tokens including the held-out prefix.
The fresh Prime schedule needs exactly 55,010,880 training rows, or
56,386,152,000 stored training tokens, leaving about 584M tokens of headroom
after the held-out prefix.

## Training protocol

### Optimizer batch

Every scale uses 327,680 predicted tokens per optimizer update:

| Surface | Realization |
|---|---|
| Jobe screen | 320 rows x 1,024 predictions; 80 four-row microbatches |
| Prime screen | 8 ranks x 4 rows x 10 accumulation microsteps |
| Optional larger reference | 8 ranks x 1 row x 8,192 predictions x 5 accumulation microsteps |

All arms use the NorMuonH/NAdam partition and defaults in
[architecture.md](architecture.md). After all microbatches—and, for distributed
runs, all ranks—have accumulated, the single global FP32 gradient vector is
clipped to L2 norm 10.0 immediately before both optimizer steps. Telemetry
reports the pre-clip norm. Equal optimizer steps are matched-data, not
matched-compute, comparisons.

### Feedback passes

Feedback arms train with parallel Jacobi passes over a full sequence. Pass 1 is
ordinary teacher forcing with plain embeddings. For every later pass:

1. take the preceding pass's payload without detaching it;
2. add keyed uniform jitter, `[-0.02, 0.02]` by default;
3. shift the payload one position right and insert zero at position 0;
4. draw a per-row plain-prefix length uniformly from `1..seq_len-1`;
5. use plain embeddings on that prefix and FBT-fused inputs on the suffix;
6. run the complete stack again.

The shift and prefix preserve token causality. Position 0 is always plain, the
last executed position is always fused, and every row therefore has at least
one feedback position. A `k`-pass batch trains a feedback horizon of `k-1`
transitions and costs `k` transformer evaluations. Non-feedback arms always use
one pass; feedback arms use one pass before the feedback boundary and two or
three passes on every step after it.

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

The feedback boundary is `round(feedback_start * steps)`, independently of the
learning-rate phases, and defaults to three quarters of the schedule. Before it
every step is one pass. After it there are no one-pass steps: each step draws
three passes with probability `three_pass = 0.12` and two passes otherwise, so
every update in this phase trains the fused mode. Across the full run this
targets 75% / 22% / 3% and an expected feedback-arm compute multiplier of 1.28
pass-tokens per predicted token. Exact realized pass-tokens are recorded.

`--max-steps` caps additional steps in one process. It never rescales the
schedule, feedback boundary, protected checkpoints, or any state-defining
field.

## Execution and provenance

CPU and MPS provide portable PyTorch semantics for fast invariant tests and
analysis. Jobe provides the authoritative CUDA training surface. Both implement
the same model, loss, optimizer, data, and checkpoint contracts; optimized
execution must be checked against the portable computation before analysis
relies on it. [Runtime qualification](runtime-qualification.md) owns the
kernel/capture contract and its numerical evidence.

Use [operations](operations.md) for installation and commands. The qualified
Jobe graph pool reserves about 23 GiB, so screen jobs and GPU analyses remain
serial. Read live status, the active log, and GPU ownership before operating
it. A cached status statement in a document is not a scheduling decision.

### Checkpoints and queue

New snapshots use checkpoint contract v23, and only v23 is resumable. V16–v22
remain readable for evaluation and forks; their original training contracts
must accompany any analysis. A snapshot contains the model, both optimizer
states, fixed NorMuonH radii, state-defining arguments, cumulative step, and
Python/Torch/CUDA RNG state. A resume inherits every state-defining field and
rejects explicit conflicts. Runtime paths, device, evaluation cadence, snapshot
cadence, and evaluation-row count may change between invocations.

Each run retains the latest two snapshots plus protected snapshots at the
cooldown boundary, the feedback boundary (the last one-pass state), and the end
of the run, so cooldown and feedback variants can continue from the exact
pre-boundary state. The durable queue stores exact arguments, not Git state.
Source changes do not stop an active child; the worker refreshes before the
next job and runs that job's probe from the current checkout. `df stop queue`
atomically removes every pending job without touching the active child or
worker; `df stop live` does the converse.

## Evaluation

Language-model metrics characterize the specimen and expose confounds. They
remain the common numerical measures of the pretraining factorial. Behavioral,
causal, and monitoring results must be reported separately under the
[interpretability protocol](interpretability.md).

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

### Recurrent dynamics

Repeatedly apply fully fused prefill to fixed held-out tokens with plain-prefix
length 1. Record loss and `mean_token ||h_top^(k) - h_top^(k-1)||_2` at each
iteration. The current `iterate_fused` diagnostic resets mixer caches within
each full prefill; only the shifted payload passes between iterations. It is
not an experiment on freely generated trajectories or tied depth.

The training monitor uses eight iterations. Characterization for any larger run
uses at least 30. Report trajectories, precision, and token rows. Decaying
updates are empirical evidence of settling on those inputs, not a proof of
contraction over a state space. Increasing loss can occur even as states
settle; loss and state stability must be interpreted separately. Oscillation,
divergence, and stable but behaviorally unhelpful states are valid outcomes to
explain. Only finite, sufficiently stable trajectories qualify a specimen for
analyses that assume stable recurrence. Failures remain reportable results.

### Routing and interventions

The current route report measures per-site/group source mass, entropy,
cross-group divergence, source/null scale, and query geometry. The payload
sweep replaces enrichment with top-only, uniform, or forced-source choices on
one checkpoint. These are the implemented entry points for analysis.

A routing coefficient is a mixing weight, not a causal importance score.
Learned nulls can have nonzero values; top-only enrichment ablation retains
`h_top` and therefore does not remove the whole payload channel. Changes to
co-adapted representations can cause generic damage. Interpret each result at
the level of the actual intervention and use the controls in
[interpretability.md](interpretability.md).

## Primary study: two-seed Jobe screen

Each of the five arms trains for 10,745 steps with a 320-row global batch:
3,520,921,600 predicted tokens per run. Two paired initialization seeds make
ten registered runs. Hybrid arms receive 25.00–25.22 predicted tokens per
active non-embedding parameter; `vanilla` receives 31.08. Record realized
pass-tokens, parameters, and device time rather than treating equal steps as
equal compute.

The study produces controlled checkpoints and asks:

1. How do the trunk and added packages affect learning and recurrent dynamics?
2. Which paths show causal use, and which are redundant or bypassed?
3. Which checkpoints and behavioral contrasts are suitable for mechanistic work?
4. Which conclusions repeat across paired seeds, and which are specimen-specific?

No completed comparison under this contract is recorded. Existing diagnostic
checkpoints may be inspected now with their original provenance attached;
exploratory work does not have to wait for the full factorial. Training-effect
claims do require the completed registered comparison.

The 25x recipe may fail to produce the behavior we want to study. Before
responding with more scale, distinguish missing task acquisition, unused
recurrence, inadequate measurement, and an incorrect mechanistic hypothesis.
Develop the smallest focused behavioral test first. Changing training tasks or
the recipe requires a new registered comparison; it does not silently relabel
existing runs.

## Evidence and decision gates

### Engineering

Run `df probe` on the exact source and hardware that will train the model. Its
CUDA gate covers portable/kernel values and gradients, cache continuation,
routing, losses, capture/replay, optimizer and radius invariants, recurrence
diagnostics, and checkpoint staging. Engineering qualification supports
faithful measurement. It is not a scientific explanation or a reasoning result.

### Scientific evidence

A complete training-effect comparison requires all registered paired runs under
the same stream, order, recipe, seeds, and schedule. Report unstable or failed
runs and their consequences instead of selecting only favorable arms. State any
stability requirement of the particular claim.

A same-checkpoint causal result needs a reproducible baseline, an explicit
intervention, appropriate controls, held-out confirmation, and a bounded claim.
It can be accepted without an architecture win or completion of unrelated arms.
An operational diagnostic or an exploratory sweep alone is not such a result.
Record accepted evidence in [findings.md](findings.md).

A monitoring result additionally needs an independently defined event or state,
a specified information budget and decision time, disjoint fitting and
evaluation data, calibration, error rates, and shift tests. Training telemetry
and a probe's decoding accuracy do not meet that gate by themselves.

### Further development

Prefer the smallest specimen and intervention that answer the question. A
useful causal result at current scale is sufficient reason to deepen analysis;
a capabilities advantage is not required. Introduce tied depth only for an
iteration-level question. Consider longer training or larger geometry only when
a documented limitation prevents that question from being tested.
[Scaling](scaling.md) preserves exact optional recipes and all distributed,
artifact, restart, and spend gates.

## Scope and limits

The current pretraining contract excludes adaptive pause/halting tokens,
token-conditioned payload queries, multiple explicit previous-column payloads,
per-layer cross-column banks, alternative optimizer arms, long-context
continuation, and instruction tuning. Targeted behavioral evaluations and
analysis tools are planned separately; no reasoning benchmark is registered.

This is a synthesis of published mechanisms, not a numerical reproduction of
any source model. It supports claims about the observed organism and tested
interventions. Whether any resulting explanation or monitor transfers to larger
models capable of opaque reasoning is a separate empirical question.

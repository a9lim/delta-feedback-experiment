# Growing recipe: arms, data, schedule, evaluation

This page describes how a specimen is grown: the five arms and how they pair,
the small geometry, the token stream, the optimizer batch, the feedback
passes, the learning-rate schedule, and the numbers we look at afterwards.
[Architecture](architecture.md) owns the equations;
[interpretability](interpretability.md) owns the analysis scripts;
[scaling](scaling.md) keeps the longer and larger recipes.

## Arms

`DFModel` has five arms. The four hybrid arms share one trunk and vary two
packages:

- **MHDB:** transient grouped reads of the seed and block deltas within a
  column, plus routed enrichment of the payload in `df`.
- **FBT:** token-gated latent payload transfer between token columns.

| Arm | MHDB | FBT | Parameters | Active non-embedding |
|---|---:|---:|---:|---:|
| `base` | no | no | 256,275,240 | 139,588,392 |
| `mhdb` | yes | no | 256,330,536 | 139,643,688 |
| `fbt` | no | yes | 257,457,192 | 140,770,344 |
| `df` | yes | yes | 257,514,792 | 140,827,944 |
| `vanilla` | no | no | 229,954,560 | 113,267,712 |

`vanilla` is a separate twelve-layer RoPE GQA decoder for a whole-trunk
contrast. The other four share PKDA/GQA, so even `base` has recurrent mixer
memory. `mhdb` seeds from the plain embedding and emits no payload; `fbt`
emits `payload_norm(h_top)`; `df` uses the fused seed as an MHDB source and
enriches its payload with routed sources.

Pairing is built in. The hybrid trunk initializes byte-identically across
`base`, `mhdb`, `fbt`, and `df` for a given seed; MHDB parameters pair across
`mhdb` and `df`, FBT parameters across `fbt` and `df`. Factor-private modules
draw from their own deterministic streams and never advance the common one.
`vanilla` keeps its own initialization and state layout. Two arms trained
with the same seed and data seed therefore see the same rows in the same
order with the same keyed feedback draws, and can be compared token by token.

The tied-depth [`df-loop`](depth-architecture.md) is specified and not built.
At `r = 1` it is exactly `df`.

## Geometry

The four hybrid arms are:

```text
[PKDA, PKDA, PKDA, gated global GQA] x 3
```

| Field | Screen value |
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
| Routing groups | 4, the KV-head count |
| RMSNorm epsilon | `1e-6` everywhere, including PKDA output |

Relative to the larger geometry in `architecture.md`, the screen halves the
residual, SwiGLU, and PKDA projection widths (`768 / 3,328 / 1,280` against
`1,536 / 6,656 / 2,560`), so the 10-by-128 PKDA geometry keeps the Kimi `5/3`
recurrent-projection ratio. Hybrid global layers are dense causal NoPE gated
GQA. `vanilla` keeps twelve bias-free RoPE GQA layers with packed QKV,
per-head Q/K RMSNorm, and no attention-output gate. MHDB has four groups
because its groups follow the global KV-head count.

## Data

The corpus is `HuggingFaceFW/fineweb-edu`, configuration `sample-100BT`, in
its canonical streaming order at dataset commit
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, tokenized with `Qwen/Qwen3-0.6B`
at commit `c1899de289a04d12100db370d81485cdf75e47ca`. The `data-build` extra
pins the four packages that compile the stream and `meta.json` records their
realized versions. Every non-empty document is followed by EOS.

`df tokenize` materializes the stream as a local contiguous uint32 store so
that step-addressed rows, validation, and resume never depend on network or
iterator state. The held-out validation slice is the stream head; training
follows in contiguous shards. One row is a non-overlapping `seq_len + 1`
window, so 1,025 stored tokens give 1,024 predictions. Rows may cross
document boundaries. Step `n` addresses:

```text
first_row(n) = (n - 1) * batch_rows
```

Feedback pass counts are keyed by data seed and step; prefix lengths and
jitter are additionally keyed by the microbatch's first global row. None of
them depend on ambient RNG state, so a resume returns to the same row and the
same draws.

The canonical stream target is 57B stored tokens including the held-out
prefix, enough for the 400x Prime schedule in [scaling.md](scaling.md) with
about 584M tokens of headroom.

## Training

### Optimizer batch

Every optimizer update sees 327,680 predicted tokens:

| Surface | Realization |
|---|---|
| Jobe screen | 320 rows x 1,024 predictions; 80 four-row microbatches |
| Prime screen | 8 ranks x 4 rows x 10 accumulation microsteps |
| Larger geometry | 8 ranks x 1 row x 8,192 predictions x 5 accumulation microsteps |

All arms use the NorMuonH/NAdam partition in [architecture.md](architecture.md).
After all microbatches have accumulated, the single global FP32 gradient
vector is clipped to L2 norm 10.0 before both optimizer steps; telemetry
reports the pre-clip norm.

### Feedback passes

Feedback arms train with parallel Jacobi passes over a full sequence. Pass 1
is ordinary teacher forcing with plain embeddings. Every later pass:

1. takes the preceding pass's payload without detaching it;
2. adds keyed uniform jitter, `[-0.02, 0.02]` by default;
3. shifts the payload one position right and inserts zero at position 0;
4. draws a per-row plain-prefix length uniformly from `1..seq_len-1`;
5. uses plain embeddings on that prefix and FBT-fused inputs on the suffix;
6. runs the complete stack again.

Position 0 is always plain and the last executed position is always fused. A
`k`-pass batch trains a feedback horizon of `k-1` transitions and costs `k`
transformer evaluations. With `ell_k` the mean next-token cross-entropy on
pass `k`:

```text
K = 1:  loss = ell_1
K > 1:  loss = ell_1 + mean(ell_2, ..., ell_K)
```

During cooldown the same combination applies to the squared log-partition
penalty `mean(logsumexp(logits)^2)` with coefficient `1e-5`.

### Schedule

Both parameter groups share one warmup-stable-cooldown multiplier. Warmup
occupies `round(warmup_frac * steps)` updates and rises linearly; cooldown
occupies `round(cooldown_frac * steps)` updates with multiplier `1 - sqrt(u)`
for local progress `u`, reaching zero at the last step. Defaults are 0.02 and
0.20. The default 10,745-step Jobe schedule:

| Phase | Steps | Passes, feedback arms |
|---|---:|---|
| Warmup | 1–215 | one |
| Stable heat | 216–8,596 | one through 8,059, then two or three |
| Cooldown | 8,597–10,745 | two or three |

The feedback boundary is `round(feedback_start * steps)`, independent of the
learning-rate phases, and defaults to three quarters of the schedule. After
it every step draws three passes with probability `three_pass = 0.12` and two
otherwise, which targets a 75% / 22% / 3% pass mixture over the run and 1.28
expected pass-tokens per predicted token. Non-feedback arms use one pass
throughout.

The run is 3,520,921,600 predicted tokens: 25.0–25.2 per active non-embedding
parameter for the hybrid arms, 31.1 for `vanilla`.

### Knobs

| Flag | Default | What it changes |
|---|---:|---|
| `--arm` | | one of the five arms |
| `--steps`, `--batch-rows` | 10,745, 320 | schedule length and optimizer batch; the 25x recipe |
| `--seed`, `--data-seed` | | initialization pairing and the keyed data/feedback streams |
| `--lr-normuonh`, `--lr-nadam` | `6e-3`, `3e-4` | the two group learning rates |
| `--feedback-start` | 0.75 | fraction of the schedule before the feedback boundary; 0 trains fused from step 0 |
| `--three-pass` | 0.12 | probability of three passes after the boundary; 1 makes every feedback step three-pass |
| `--jitter` | 0.02 | payload jitter half-width |
| `--warmup-frac`, `--cooldown-frac` | 0.02, 0.20 | schedule shape |
| `--max-steps` | | caps this invocation without changing the schedule |
| `--resume` | | continues a tag from its latest v23 snapshot |

The specimens so far used the default recipe (`mhdb`) and
`--feedback-start 0 --three-pass 1` (`df`); see [findings.md](findings.md).

### Checkpoints and queue

Snapshots use checkpoint contract v23, and only v23 resumes; v16 through v22
stay readable for evaluation and forks. A snapshot holds the model, both
optimizer states, the fixed NorMuonH radii, the state-defining arguments, the
cumulative step, and Python/Torch/CUDA RNG state. A resume inherits every
state-defining field and rejects explicit conflicts; runtime paths, device,
evaluation cadence, snapshot cadence, and evaluation-row count may change.

Each run keeps the latest two snapshots plus protected ones at the cooldown
boundary, the feedback boundary (the last one-pass state), and the end of the
run, so cooldown and feedback variants can fork from the exact pre-boundary
state. The queue stores arguments, not Git state; a source change never stops
an active child, and the worker refreshes before the next job.

## Evaluation

### Language-model numbers

Every evaluation point reports pass-1 held-out cross-entropy as `val`.
Feedback arms also report `val_fused`, a second pass with plain-prefix
length 1. Decoding has three modes:

- **Standard:** one plain prompt prefill, no feedback during decode.
- **Soft:** one plain prefill, then one feedback transition per generated
  token.
- **Fused:** an additional fused prompt pass, then the same feedback decode.

### Comparing arms

For validation loss `L`, gain over `base` is `G_A = L_base - L_A` and the
interaction of the two packages is:

```text
I = G_df - G_mhdb - G_fbt = L_mhdb + L_fbt - L_df - L_base
```

Positive `I` is superadditive loss reduction. Computed on paired checkpoints
by mode; `L_vanilla - L_base` is the whole-trunk contrast. Report parameters,
predicted tokens, and pass-tokens with every number; a matched-compute view
compares at equal cumulative pass-tokens. The full two-seed, five-arm screen
is the natural complete comparison and has not been run; it is one option
among several for what to train next, not a prerequisite.

### Downstream tasks

`scripts/downstream_eval.py` scores a snapshot on the workspace's pinned
zero-shot suite (HellaSwag, ARC-Easy/Challenge, PIQA, WinoGrande, BoolQ,
OpenBookQA, SciQ, LAMBADA) with the evaluation harness's prompts and
normalization, in Standard, Soft, or Fused mode. Accuracy comes with a
standard error; pairwise comparison on identical documents through the
workspace module resolves gold log-probability and margin differences far
smaller than accuracy can. At this scale the tasks resolve a few accuracy
points and anchor the specimen against published models of similar size.

### Recurrent dynamics

`iterate_fused` repeatedly applies fully fused prefill to fixed held-out
tokens with plain-prefix length 1 and records loss and
`mean_token ||h_top^(k) - h_top^(k-1)||_2` per iteration. Mixer caches reset
within each prefill; only the shifted payload passes between iterations. The
training monitor runs eight iterations; the analysis scripts run thirty.

### Routing and interventions

`scripts/route_report.py` measures per-site/group source mass, entropy,
cross-group divergence, source and null scale, and query geometry.
`scripts/payload_swap.py` replaces the payload enrichment with top-only,
uniform, or forced-source choices. The rest of the toolset is in
[interpretability.md](interpretability.md). A routing weight is a mixing
coefficient; learned nulls can carry nonzero values; top-only enrichment
ablation keeps `h_top` and so keeps most of the payload channel.

## Not in the current recipe

Adaptive pause or halting tokens, token-conditioned payload queries, multiple
explicit previous-column payloads, per-layer cross-column banks, alternative
optimizers, long-context continuation, instruction tuning, and any task
beyond next-token prediction on FineWeb-Edu. Several of these are candidate
moves in [findings.md](findings.md).

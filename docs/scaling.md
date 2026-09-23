# Scaling and resource accounting

`--scale screen|bridge|flagship|extension` selects geometry and batch settings;
explicit flags override individual fields. The conditions share every
parameter; loops and feedback reuse weights. Source definitions are
[`SCALES` and `reference_active`](../delta_feedback_experiment/train.py).

## Geometry and parameters

All presets share **16 layers in four `[PKDA, PKDA, PKDA, NoPE-GGQA]` cells**,
a 50,304-row tied embedding/readout, and a global batch of 128 rows × 4,096
predictions = **524,288 tokens per update**. Each stored row has one extra
target. `--ranks N` divides the global rows among devices; replay width is
chosen by the [CUDA planner](operations.md#cuda-execution).

| Field | Screen | Bridge | Flagship | Extension |
|---|---:|---:|---:|---:|
| Residual width `D` | 768 | 1,152 | 1,536 | 2,304 |
| Shared + selected / routed experts, width 832 | 1 + 3 / 15 | 1 + 5 / 23 | 1 + 7 / 31 | 1 + 11 / 47 |
| Active / stored FFN width | 3,328 / 13,312 | 4,992 / 19,968 | 6,656 / 26,624 | 9,984 / 39,936 |
| GQA query / KV heads, width 256 | 4 / 2 | 6 / 3 | 8 / 4 | 12 / 6 |
| MHDB groups, width 384 | 2 | 3 | 4 | 6 |
| PKDA heads, width 128 | 8 | 12 | 16 | 24 |
| Total parameters | 621,389,088 | 1,364,464,240 | 2,395,794,368 | 5,323,219,552 |
| Training-active non-embedding parameters | 191,702,304 | 426,644,080 | 754,314,176 | 1,687,839,328 |
| Included MTP total / active parameters | 34,321,312 / 11,318,176 | 76,867,376 / 25,110,320 | 136,337,088 / 44,324,544 | 306,047,456 / 99,019,232 |

GQA query and PKDA projection widths are `4D/3`; each GQA K/V projection is
`2D/3`. Every trunk and MTP expert bank stores one shared plus `n` routed
experts, selects `k` routed experts per token, and uses intermediate width
`h=832`. Its active/stored widths are `(k+1)h` and `(n+1)h`: every preset
executes one quarter of its expert matrices per token.

Totals and active counts cover the parameters every condition shares, so
one denominator sets each scale's schedule. Active counts include the selected experts and every
router, mixer, payload, and fusion parameter, excluding only idle experts
and the tied embedding matrix. They are a per-token accounting convention:
a batch can touch all experts, and training must store all weights,
gradients, and optimizer state.
Selection biases are buffers, excluded from parameter counts.

The auxiliary block contributes `P_PKDA + 3D(n+1)h + (n+2)D` parameters;
replace `n+1` with `k+1` only in the expert-matrix term for its active count.
This covers its mixer, experts, router, and two norms. The shared `2D -> D`
fusion contributes `2D²` parameters once, outside MTP; output norm and tied
readout are shared with the trunk. MTP runs once per training column and is
absent from generation. Its supervision does not add recorded input tokens.

## Width and expert learning rates

Let `r_mu=1536/D`, with flagship as the fixed width reference, and
`r_expert=sqrt(8/(k+1))`. Peak rates follow optimizer ownership:

| Parameter group | Peak learning rate |
|---|---|
| Ordinary NorMuonH matrices | `lr_normuonh` |
| Expert gate/up and down matrices, including shared and MTP experts | `lr_normuonh * r_expert` |
| Full-residual-width NAdam matrices: GQA gates, PKDA controls, expert routers | `lr_nadam * r_mu` |
| Other NAdam parameters | `lr_nadam` |

| Scale | `r_mu` | `r_expert` | Expert peak LR at base `0.006` |
|---|---:|---:|---:|
| Screen | 2 | 1.414214 | 0.00848528 |
| Bridge | 4/3 | 1.154701 | 0.00692820 |
| Flagship | 1 | 1 | 0.00600000 |
| Extension | 2/3 | 0.816497 | 0.00489898 |

Both expert projection groups use the same count factor; NorMuonH handles
matrix aspect ratio separately. All five groups share the recipe's schedule
multiplier. `r_mu` also scales normalized hidden states before the tied
readout. [Architecture](architecture.md) defines initialization and optimizer
mechanics. These are implemented scaling rules, not demonstrated
hyperparameter transfer across widths.

## Token budgets

`--tokens-per-param R` uses the flat `f` training-active non-embedding count
`A`, including MTP, for every condition at the selected geometry:

```text
steps = ceil(R * A / (batch_rows * seq_len))
predicted_tokens = steps * batch_rows * seq_len
stored_tokens = ceil((steps * batch_rows * (seq_len+1) + val_tokens) / 1e9) * 1e9
```

The default is `R=25`. Alternatively, `--steps` specifies a length directly;
the two flags are mutually exclusive. The store formula is used by `delta tokenize --scale S --tokens-per-param R`. These values use
the default 30M validation tokens:

| Scale | 25x steps | 25x predicted tokens | 400x steps | 400x predicted tokens | 400x store |
|---|---:|---:|---:|---:|---:|
| Screen | 9,142 | 4,793,040,896 | 146,258 | 76,681,314,304 | 77B |
| Bridge | 20,344 | 10,666,115,072 | 325,504 | 170,657,841,152 | 171B |
| Flagship | 35,969 | 18,858,115,072 | 575,497 | 301,726,171,136 | 302B |
| Extension | 80,483 | 42,196,271,104 | 1,287,720 | 675,136,143,360 | 676B |

The canonical store target is 85B tokens. Larger budgets can use full DCLM;
changing the source changes the stream and validation slice, so it breaks
per-token pairing. The [recipe](design.md#data) defines stream identity, and
[operations](operations.md#tokenize) covers build storage and commands.

## Measured loss scaling

[`scaling_curves.py`](../scripts/scaling_curves.py) fits run logs and writes
`figures/scaling/`; the current read uses `screen-f-25x-feedback0p6-s1`,
`screen-n-50x-s2`, and `bridge-f-25x-feedback0p6-s1`. Runs with one
`--data-seed` read identical batches at each step, so step-matched differences
cancel the data-order bumps (detrended residuals correlate 0.998 across scales
and seeds). Two screen seeds differ by 0.002 nats of validation loss, and
enabling feedback moves the plain pass by under 0.003.

The stable phase flattens toward a floor set by the peak learning rate while
annealed finals keep falling. Cooldown lowers validation loss by 0.150 at 4.8B
tokens (0.155 measured directly against a run still at peak rate on the same
batches) and by 0.18–0.19 at 9.6–10.7B tokens, so within-run stable curves do
not extrapolate; annealed finals do.

Annealed finals on seed 1 anchor `E + A·N^-α + B·D^-β`, with `N` the
training-active count and `D` predicted tokens. Three anchors leave the
exponents free: the table holds α = 0.3, β = 0.4 and brackets β over 0.3–0.7
and α over 0.25–0.35. The anchors imply a screen–bridge capacity gap of
0.096–0.098 for any β in that range, matching the step-matched stable-phase
gap, and a fall of 0.093 per doubling of tokens at screen. The learning-rate
annealing law of Tissue et al., fitted to the 25x screen run alone, predicted
the 50x final within 0.002; its own extrapolations sit at the high-β edge of
the brackets.

| Scale | 25x | 50x | 100x | 400x |
|---|---:|---:|---:|---:|
| Screen | 2.931 (measured) | 2.838 (measured) | 2.767 [2.762–2.780] | 2.673 [2.650–2.723] |
| Bridge | 2.729 (measured) | 2.661 [2.656–2.676] | 2.610 [2.596–2.643] | 2.541 [2.508–2.610] |
| Flagship | 2.616 [2.610–2.629] | 2.562 [2.548–2.593] | 2.521 [2.498–2.571] | 2.467 [2.424–2.549] |
| Extension | 2.490 [2.471–2.528] | 2.451 [2.422–2.507] | 2.422 [2.383–2.495] | 2.382 [2.325–2.482] |

Downstream, paired per document, the 50x screen run scores 3.4 ± 0.2 accuracy
points above the 25x run pooled over the suite, and the 25x bridge run a
further 4 above that.

## Loop compute and decode state

A step with `p` passes and `r` columns per pass executes `pr` columns,
`4pr` cells, and `16pr` trunk layers per token position. The
[recurrence roll](design.md#the-recurrence-roll) gives these expectations:

| Quantity | `n` | `f` | `v` | `fv` |
|---|---:|---:|---:|---:|
| Mean columns per position on rolled steps | 1 | 2.12 | 2.12 | 4.48 |
| Maximum columns per position on rolled steps | 1 | 3 | 3 | 6 |
| Mean columns per position over the default schedule | 1 | 1.28 | 1.28 | 1.87 |
| Columns per pass in evaluation/decode | 1 | 1 | 2 | 2 |

For looped conditions, a rolled pass averages 33.92 executed trunk layers; fixed
evaluation/decode uses 32, and a three-column pass uses 48.
`--loop-iterations` changes only the evaluation/decode count within `1..3`.
Feedback evaluation additionally runs a second pass. Decode carries feedback
sequentially between positions and does not replay full-prefix Jacobi passes.

Predicted-token counters count data once, pass-tokens multiply by `p`, and
cell-tokens multiply by `4pr`. Report all three plus device time: MTP and
vocabulary loss add work beyond these trunk counters. The full `fv` mean
uses the joint roll, not a product of independent pass/column means.

Each column has separate mixer caches at every layer. For one sequence and
4,096 cached positions, BF16 GQA K/V and convolution histories plus FP32
PKDA matrix/diagonal states require:

| Scale | One column | Default loop depth, 2 columns | Maximum loop depth, 3 columns |
|---|---:|---:|---:|
| Screen | 38.26 MiB | 76.52 MiB | 114.77 MiB |
| Bridge | 57.39 MiB | 114.77 MiB | 172.16 MiB |
| Flagship | 76.52 MiB | 153.03 MiB | 229.55 MiB |
| Extension | 114.77 MiB | 229.55 MiB | 344.32 MiB |

For cache length `T`, GQA KV-head count `Hkv`, PKDA head count `Hp`, and PKDA
head width `d=128`, one cell costs
`4*T*Hkv*256 + 3*(4*Hp*(d²+d) + 18*Hp*d)` bytes: two BF16 K/V arrays and
three PKDA layers, each with FP32 matrix/diagonal state and three BF16
convolution histories of length three. Multiply by four cells, column count,
and batch size. Payload, logits, allocator overhead, and serving metadata
are excluded. This is decode-state arithmetic, not training-memory usage.

## Runtime estimates

These durations come from replay benches on **one 80 GB H100 PCIe** (sm_90)
with the current kernels. `scripts/replay_bench.py` captures every graph a
schedule reaches at its planned replay width and times its replays. An update
is one replay's time multiplied by the number of replays in a 128-row batch.
Screen replays used trained `v` weights. Bridge and flagship used
initialization, which ran about 3% slower than trained weights at screen.
Update seconds, before the optimizer step:

| Scale | `(1,1)` | `f`: `(2,1)` / `(3,1)` | `fv`: `(2,2)` / `(2,3)` / `(3,2)` | Rolled mean, `f` / `fv` |
|---|---:|---:|---:|---:|
| Screen | 5.98 | 11.85 / 18.11 | 24.10 / 38.42 / 38.49 | 12.60 / 27.55 |
| Bridge | 10.66 | 20.21 / 31.85 | 42.38 / 69.31 / 69.39 | 21.60 / 48.86 |
| Flagship | 15.18 | 30.69 / 49.99 | 66.56 / 103.00 / 103.12 | 33.00 / 75.32 |

Rolled means weight each shape by its roll probability: 0.88 / 0.12 for `f`'s
two and three passes, and 0.76 / 0.12 / 0.12 for `fv`; `n` runs
`(1,1)` on every step. The planner
narrows replays as scale grows: `(2,2)` runs four, two, then one row per
replay. Flagship's six-column graphs keep lean intermediates and recompute
five blocks. Extension does not fit one 80 GB card; see the memory boundary
below.

```text
mean_update = s * t(1,1) + (1 - s) * t_rolled + t_optimizer
H100 PCIe seconds = steps * mean_update * 1.10
node seconds = H100 PCIe seconds / assumed_node_speedup
```

`s` is the recurrence start: 0.75 by default. The repository's current runs
use 0.6, where screen `fv` averages 14.7 s. The eager optimizer step measured
about 0.1 s at screen and is scaled by stored parameters at the other scales.
At `s = 0.75`:

| Scale | H100 PCIe mean update, `fv` / `f` | H100 PCIe at 25x, `fv` / `f` | 8×H100 SXM at 25x, `fv` / `f` |
|---|---:|---:|---:|
| Screen | 11.5 / 7.7 s | 1.3 d / 21.6 h | 3.6 / 2.4 h |
| Bridge | 20.4 / 13.6 s | 5.3 / 3.5 d | 14.1 / 9.4 h |
| Flagship | 30.6 / 20.0 s | 14.0 / 9.2 d | 1.6 / 1.0 d |
| Extension | 58.4 / 38.3 s, extrapolated | does not fit | 6.0 / 3.9 d |

The node is **8×80 GB H100 SXM with NVSwitch**, using `--ranks 8` and
optimizer ownership sharding. Its assumed speedup over one H100 PCIe is 9× for
screen, bridge and flagship, and 10× for extension. That chains two figures. A
GH200, whose GPU is H100 SXM-class, ran the same screen `(2,2)` graph 1.4×
faster than the PCIe card. The node is modeled at 6.5× one GH200, or 7× for
extension, where sharding also relieves memory pressure. These are modeling
inputs, not node benchmarks. Extension multiplies each flagship time by
`(A_extension / A_flagship)^0.8 = 1.9`. That exponent is somewhat above the
screen-to-flagship exponents of 0.62–0.76, because replays narrow further and
extension needs recomputation.

The table includes 10% for evaluation and checkpointing. At 100x or 400x,
durations are approximately 4× or 16× the 25x values; use the formula to avoid
multiplying rounded table entries. Add 0.5–2 hours for cold setup and
compilation, potentially more for extension; one condition's graphs compiled
in 20–30 minutes on the bench host. Data preparation, queueing, failures, and
offline downstream evaluation are excluded.

The single-card rows are measured graphs, not measured runs. Allow about ±10%
for screen and bridge, and ±15% for flagship, whose weights were at
initialization and whose plans are lean. The PCIe card's sustained clocks
under its 350 W limit can also move them. Extension spans roughly 0.6–1.6× the
point estimate, conditional on fit. Node estimates span roughly 0.7–1.5×,
wider for extension. These are judgment ranges, not confidence intervals; see
the [node workflow](operations.md#hopper-and-node-workflow) for measuring a
new host.

### Training memory boundary

NorMuonH matrices keep replicated BF16 working weights and FP32 gradients;
FP8 additionally keeps quantized matrices and transposes. Only FP32 masters,
BF16 momentum, and small optimizer statistics are partitioned among ranks.
NAdam weights, gradients, and moments remain replicated. Thus eight ranks do
not divide the complete training footprint by eight.

With `N` NorMuonH elements, `A` NAdam elements, and `R` ranks, a leading-order
FP8 persistent-state estimate is `8N + 6N/R + 16A` bytes per rank. Add working
copies for NAdam-owned GEMM operands, classifier shadows, optimizer vectors,
rank padding, and allocation overhead. Initialization also holds the full
FP32 NorMuonH parameters before adopting the working views, adding about `4N`.
Activations, temporary tensors, graph pools, and communication buffers are
additional.

For extension this gives approximately **70 GiB persistent / 89 GiB during
initialization** on one GPU and **45 / 64 GiB per GPU** on eight ranks,
before activations and transient allocations. A GH200 scenario with about
94.5 GiB usable HBM therefore has a narrow initialization margin; the
marketing product name is not the usable CUDA allocation budget. An 80 GB
H100 needs the existing ownership sharding for this extension scenario.
Only target-hardware capture and complete training steps establish fit.

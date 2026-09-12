# Scale presets and accounting

`--scale screen|bridge|flagship|extension` selects geometry and batch settings.
Explicit geometry or recipe flags override their preset fields. Conditions are `f`,
`l`, and `fl`; tied depth adds no parameters. The trainer is single-process.
Preset availability does not establish GPU fit or throughput.

## Geometry and parameters

| Field | Screen | Bridge | Flagship | Extension |
|---|---:|---:|---:|---:|
| Residual width `D` | 768 | 1,152 | 1,536 | 2,304 |
| Layers / four-layer cells | 16 / 4 | 16 / 4 | 16 / 4 | 16 / 4 |
| Prelude / core / coda cells | 1 / 2 / 1 | 1 / 2 / 1 | 1 / 2 / 1 | 1 / 2 / 1 |
| Trunk and MTP per-expert width `h` | 832 | 832 | 832 | 832 |
| Shared + selected / routed experts | 1 + 3 / 15 | 1 + 5 / 23 | 1 + 7 / 31 | 1 + 11 / 47 |
| Active FFN width `H = (k+1)h` | 3,328 | 4,992 | 6,656 | 9,984 |
| Stored FFN width `(n+1)h` | 13,312 | 19,968 | 26,624 | 39,936 |
| GQA query / KV heads, width 96 | 8 / 4 | 12 / 6 | 16 / 8 | 24 / 12 |
| MHDB groups | 4 | 6 | 8 | 12 |
| PKDA heads, width 128 | 10 | 15 | 20 | 30 |
| PKDA projection width | 1,280 | 1,920 | 2,560 | 3,840 |
| `f` / `fl` total parameters | 630,606,216 | 1,384,528,652 | 2,430,864,784 | 5,400,778,136 |
| `f` training-active non-embedding parameters | 200,919,432 | 446,708,492 | 789,384,592 | 1,765,397,912 |
| Included auxiliary MTP total parameters | 36,362,664 | 81,407,420 | 144,361,168 | 323,995,640 |
| Included auxiliary MTP active parameters | 13,359,528 | 29,650,364 | 52,348,624 | 116,967,416 |

Every scale has a 50,304-row tied embedding/readout, 4,096 predictions per
row, and 128 rows per update in one-row microbatches: 524,288 predicted tokens
per step. Each stored row includes one additional target. Width multipliers
and optimizer ownership are in [architecture.md](architecture.md#nadam-parameters).
The muP reference stays at the flagship width 1,536. Extension's `1536/D`
multiplier is `2/3` for NAdam width rates and readout scaling. Expert
NorMuonH rates use the separate factors below.

Each trunk and MTP FFN stores one shared plus `n` routed experts, with `k`
routed experts selected per token. `expert_intermediate` is the actual
per-expert width `h`; the active dense-equivalent width `H = (k+1)h` is derived.
The presets keep `(k+1)/(n+1) = 1/4`, so a quarter of the stored expert
matrices run per token. Active counts include those experts and the full router
in every bank, plus all mixer and payload-writer parameters.
A microbatch can touch all experts; all parameters, gradients, and optimizer
state occupy memory. Per-bank selection biases are buffers, excluded from
parameter counts. Training's transient assignment counts have shape
`[layers+1,n]`, with MTP last; column-only counts are `[layers,n]`.

The auxiliary module adds `P_PKDA + 3D(n+1)h + 2D^2 + (n+4)D` parameters:
its PKDA mixer, expert matrices, concatenation projection, expert router, and
four RMSNorm scales. Its active count replaces `(n+1)` with `(k+1)` in the
expert term. It runs once per training pass over `seq_len-1` positions, with
a second vocabulary-loss call, and shares the embedding/final norm/readout.
It does not execute in generation. Condition `l` retains the payload writer
and omits the FBT entry; shared token budgets still use `f`.

## Expert learning rates

Shared and routed experts in the trunk and MTP use the same NorMuonH
operator-step budget for their gate/up and down groups:

```text
gate/up peak LR = lr_normuonh * sqrt(8 / (k+1))
down peak LR    = lr_normuonh * sqrt(8 / (k+1))
```

`k+1` counts the shared plus selected routed experts. The count factor
offsets coherent aggregation of expert changes after the bank's `1/sqrt(k+1)`
forward normalization. Both groups retain this factor under overrides;
per-matrix spectral normalization handles residual/expert width separately.
With the default `lr_normuonh = 0.006`:

| Scale | Gate/up factor | Down factor | Peak expert LR, both groups |
|---|---:|---:|---:|
| Screen | 1.414214 | 1.414214 | 0.00848528 |
| Bridge | 1.154701 | 1.154701 | 0.00692820 |
| Flagship | 1.000000 | 1.000000 | 0.00600000 |
| Extension | 0.816497 | 0.816497 | 0.00489898 |

All five groups share the schedule multiplier. Ordinary NorMuonH retains
the base rate; NAdam uses its base or `1536/D` rate. The expert factors
change optimizer updates only: forward normalization and initialization
retain their own contracts. This is an implemented scaling candidate,
without demonstrated full-model hyperparameter transfer across scales.

`python scripts/spectral_optimizer_check.py --device cuda --benchmark --output /tmp/spectral.json`
compares the three-step spectral estimates with exact SVD on synthetic
screen-shaped directions, including rotating low-rank gradients and unequal
row histories. It also times bounded optimizer buckets and weights their
timings by screen matrix counts. Exact SVD is diagnostic only. Results measure
estimate error and optimizer cost, not full training throughput, model quality,
or a guaranteed norm bound. Omit `--benchmark` for accuracy alone; omit
`--device cuda` for CPU execution.

`python scripts/expert_scaling_check.py --device cuda --output /tmp/expert-scaling.json`
records a bounded coordinate check with learned routing. It compares the
implemented rates against base expert rates on identical initial weights and
a fixed synthetic batch, using one screen bank per run at one-eighth residual
and expert widths (`D=96`, `h=104`), screen expert counts, and the real 1,536
reference. This diagnostic runs screen only; larger geometries are outside
Jobe's execution scope.
The JSON reports output changes, router logits/entropy, and gradient norms
over four steps and two seeds; it is a diagnostic, not a transfer benchmark.
Omit `--device cuda` for portable FP32 execution.

## Token budgets

`--tokens-per-param R` derives steps from the current flat `f` training-active
non-embedding count, including MTP:

```text
steps = ceil(R * reference_active / (batch_rows * seq_len))
```

Every condition at a geometry shares that schedule. Auxiliary second-token
targets add supervision on the same rows without increasing the recorded
ordinary predicted-token count. `--steps` sets a length directly. The default
ratio is 25; larger ratios are supported arithmetic scenarios.

| Scale | 25x steps | 25x predicted tokens | 400x steps | 400x predicted tokens |
|---|---:|---:|---:|---:|
| Screen | 9,581 | 5,023,203,328 | 153,290 | 80,368,107,520 |
| Bridge | 21,301 | 11,167,858,688 | 340,812 | 178,683,641,856 |
| Flagship | 37,641 | 19,734,724,608 | 602,253 | 315,754,020,864 |
| Extension | 84,181 | 44,135,088,128 | 1,346,892 | 706,159,312,896 |

Warmup is 2% of the shorter of the run and its 25x length: 192, 426, 753,
and 1,684 steps at or above 25x. Cooldown occupies 20%. Feedback begins at 75%;
the default mixture costs about 1.28 pass-tokens per prediction. See
[design.md](design.md#schedule).

`--continue TAG` extends a finished run by restoring the last snapshot the
longer schedule reproduces. A 25x screen run continued to 50x restores step
7,186 and trains 11,976 more steps to reach 19,162.

Token stores include validation and each row's extra target.
`delta tokenize --scale S --tokens-per-param R` computes that requirement and
rounds up to the next billion stored tokens. Screen 100x needs 21B, screen
400x needs 81B, bridge 400x needs 179B, flagship 400x needs 316B, and extension
400x needs 707B.
Full DCLM supports larger stores. Its stream and held-out slice differ from
the publisher's DCLM-100B subset, preventing paired per-token comparisons.

## Loop compute and decode state

With `C` unique cells, the core contains `C-2` cells. At depth `r`, a pass
executes `2+(C-2)r` cells. Every preset has `C=4`, so all scales execute
`2+2r` cells: four at flat depth, ten at `r=4`, and eighteen at `r=8`. The
default capped draw has uncapped mean 4, cap 8, and actual mean about 3.88.
Report predicted tokens, pass-tokens, and cell-tokens together.
These counters describe the trunk; total compute also includes MTP and its
vocabulary loss. Measured device time includes routing and execution overhead.

For 4,096 cached positions, BF16 GQA K/V and convolution histories plus FP32
PKDA matrix/diagonal states cost:

| Scale | One cell | Flat / `r=1` | `r=4` | `r=8` |
|---|---:|---:|---:|---:|
| Screen | 7.96 MiB | 31.82 MiB | 79.56 MiB | 143.20 MiB |
| Bridge | 11.93 MiB | 47.73 MiB | 119.33 MiB | 214.80 MiB |
| Flagship | 15.91 MiB | 63.64 MiB | 159.11 MiB | 286.40 MiB |
| Extension | 23.87 MiB | 95.47 MiB | 238.67 MiB | 429.60 MiB |

Each core iteration keeps its own mixer cache. These per-sequence state
counts exclude payload, logits, allocator overhead, and serving metadata.
Training also holds activations, gradients, optimizer state, and CUDA graph
pools; decode-state arithmetic is not a training-memory estimate.

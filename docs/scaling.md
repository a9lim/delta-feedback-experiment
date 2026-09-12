# Scale presets and accounting

`--scale screen|bridge|flagship` selects geometry and batch settings. Explicit
geometry or recipe flags override the corresponding preset fields.
Every condition builds at each scale; `l` reuses the existing middle cells
and adds no parameters. The trainer is single-process. Larger presets do not
imply that their memory use or throughput fits a particular GPU; distributed
training is not implemented.

## Geometry

| Field | Screen | Bridge | Flagship |
|---|---:|---:|---:|
| Residual width `D` | 768 | 1,152 | 1,536 |
| Layers / four-layer cells | 12 / 3 | 16 / 4 | 24 / 6 |
| SwiGLU intermediate width | 3,328 | 4,992 | 6,656 |
| Per-expert intermediate width under `e` | 832 | 1,248 | 1,664 |
| Global query / KV heads, head width 96 | 8 / 4 | 12 / 6 | 16 / 8 |
| Routing groups | 4 | 6 | 8 |
| PKDA heads, head width 128 | 10 | 15 | 20 |
| PKDA Q/K/V projection width | 1,280 | 1,920 | 2,560 |
| `arf` / `arfl` total parameters | 179,461,416 | 474,584,400 | 1,179,313,440 |
| Active non-embedding parameters | 140,827,944 | 416,634,192 | 1,102,046,496 |
| `aerf` / `aerfl` total parameters | 455,637,288 | 1,302,973,776 | 3,388,167,456 |
| `aerf` active non-embedding per token | 140,966,184 | 416,910,672 | 1,102,599,456 |

Every scale uses a 50,304-row tied embedding/readout, 4,096 predictions per
row, and 128 rows per update in one-row microbatches: 524,288 predicted tokens
per step. Each stored row contains one additional target token.

The PKDA projection-to-residual ratio is `5/3`; the SwiGLU ratio is `13/3`.
Head widths, convolution width, and normalization epsilon stay fixed.
The NAdam gate/control initialization, learning rate, and readout multipliers
are defined in [architecture.md](architecture.md#nadam-parameters), with
width 1,536 as the reference.

### Parameter breakdown

Counts below are for `arf`; `arfl` has identical parameters.

| Component | Screen | Bridge | Flagship |
|---|---:|---:|---:|
| Tied embedding/readout | 38,633,472 | 57,950,208 | 77,266,944 |
| SwiGLU mixers | 92,012,544 | 276,037,632 | 736,100,352 |
| PKDA mixers | 40,478,184 | 116,552,400 | 304,297,632 |
| Gated global GQA mixers | 7,078,464 | 21,234,432 | 56,624,256 |
| FBT fusion | 1,179,648 | 2,654,208 | 4,718,592 |
| Trunk, entry, and payload norms | 21,504 | 41,472 | 79,872 |
| Within-column and payload routers | 57,600 | 114,048 | 225,792 |

Under `e`, every FFN stores one shared plus fifteen routed quarter-width
experts. Its stored matrix parameters are four times the dense row above;
one shared plus three routed experts keep active matrix parameters equal to
the dense FFN. Each layer adds a `D x 15` router: 138,240, 276,480, or 552,960
parameters over the full model. Active counts refer to one token; a microbatch
can touch all experts, whose gradients and optimizer state still occupy memory.
Each physical layer also stores fifteen FP32 selection biases as model buffers,
excluded from parameter counts. Per-update assignment counts have the same
small `[layers, 15]` shape and are transient.
Routing and optimizer overhead mean equal active arithmetic does not imply
equal training time or measured GPU fit.

### The screen

Every looped condition has the same count as the corresponding row without `l`:

| Condition | Parameters | Active non-embedding |
|---|---:|---:|
| `""` (plain) | 158,979,072 | 120,345,600 |
| `r` | 159,034,368 | 120,400,896 |
| `f` | 160,161,024 | 121,527,552 |
| `rf` | 160,218,624 | 121,585,152 |
| `a` | 178,221,864 | 139,588,392 |
| `ar` | 178,277,160 | 139,643,688 |
| `af` | 179,403,816 | 140,770,344 |
| `arf` | 179,461,416 | 140,827,944 |

Adding `e` to any row adds 276,175,872 stored parameters and 138,240 active
parameters per token. `l` continues to add no parameters, including when its
core contains experts.

`m` adds one dense gated-GQA/SwiGLU block, two entry norms, and a `2D -> D`
concatenation projection. Its embedding, final norm, and vocabulary projection
are shared with the trunk. The tables above exclude this auxiliary module.
At the presets its parameter addition is `19D^2 + 4D + 192`:

| Auxiliary MTP module | Screen | Bridge | Flagship |
|---|---:|---:|---:|
| Additional stored and training-active parameters | 11,209,920 | 25,219,776 | 44,832,960 |

It runs once per training pass over `seq_len - 1` positions and adds a second
vocabulary-loss call per pass, even with `e` or `l`. Ordinary inference does
not execute it and needs no auxiliary decode cache.

## Token budgets

`--tokens-per-param R` derives the step count from the flat `arf` active
non-embedding parameter count at the selected geometry:

```text
steps = ceil(R * reference_active / (batch_rows * seq_len))
```

Every condition at a scale shares that schedule, including `e` and `m`; the
dense `arf` reference remains fixed and does not count the expert bank, its
router, or the auxiliary MTP module. Auxiliary second-token targets are
additional supervision on the same rows and do not increase the recorded
ordinary predicted-token budget.
`--steps` sets a length
directly instead. The default ratio is 25; 400 is another supported ratio,
not a scheduled run.

| Scale | 25x steps | 25x predicted tokens | 400x steps | 400x predicted tokens |
|---|---:|---:|---:|---:|
| Screen | 6,716 | 3,521,118,208 | 107,444 | 56,331,599,872 |
| Bridge | 19,867 | 10,416,029,696 | 317,867 | 166,653,853,696 |
| Flagship | 52,550 | 27,551,334,400 | 840,795 | 440,818,728,960 |

Warmup is 2% of the shorter of the run and its 25x length: 134, 397, and
1,051 steps for runs at or above 25x. Cooldown is 20% of the run. The
feedback boundary is at 75%; the default feedback mixture costs about 1.28
pass-tokens per predicted token. See [design.md](design.md#schedule).

`--continue TAG` extends a finished run by restoring the last snapshot the
longer schedule reproduces. For example, a 25x screen run continued to 50x
restores step 5,037 and trains 8,394 additional steps to reach 13,431.

Token stores also hold validation data and each row's extra target.
`delta tokenize --scale S --tokens-per-param R` computes that requirement and
rounds up to the next billion stored tokens. The screen at 100x needs a 15B
store; the bridge at 400x needs 167B, and the flagship at 400x needs 441B.
Full DCLM supplies larger stores. Its stream and held-out slice differ from
the publisher's DCLM-100B subset, so those stores cannot support a paired
per-token comparison.

## Loop compute and decode state

For `C` unique cells, the tied core contains `C - 2` cells. A pass at fixed
iteration count `r` executes `2 + (C - 2) r` cells:

| Scale | Core cells | Cells per pass |
|---|---:|---|
| Screen | 1 | `2 + r` |
| Bridge | 2 | `2 + 2r` |
| Flagship | 4 | `2 + 4r` |

The default capped log-normal Poisson draw has uncapped mean 4, cap 8, and
actual mean about 3.88. Report realized pass-tokens and cell-tokens with
predicted tokens; equal data does not imply equal compute.
These counters describe the trunk. With `m`, include the auxiliary block and
second vocabulary loss when accounting for compute; equal cell-tokens alone
do not match compute between conditions with and without MTP.

For the `a` trunk at 4,096 cached positions, BF16 GQA K/V and convolution
histories plus FP32 PKDA matrix/diagonal states cost:

| Scale | One cell | Flat / `r = 1` | `r = 4` | `r = 8` |
|---|---:|---:|---:|---:|
| Screen | 7.96 MiB | 23.87 MiB | 47.73 MiB | 79.56 MiB |
| Bridge | 11.93 MiB | 47.73 MiB | 119.33 MiB | 214.80 MiB |
| Flagship | 15.91 MiB | 95.47 MiB | 286.40 MiB | 540.98 MiB |

Each core iteration keeps its own mixer cache. These are per-sequence
state counts, excluding payload, logits, allocator overhead, and serving
metadata. Training also holds activations, gradients, optimizer state, and
CUDA graph pools; decode-state arithmetic is not a training-memory estimate.

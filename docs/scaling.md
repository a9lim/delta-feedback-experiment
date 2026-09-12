# Scale presets and accounting

`--scale screen|bridge|flagship` selects geometry and batch settings. Explicit
geometry or recipe flags override their preset fields. Conditions are `f`,
`l`, and `fl`; tied depth adds no parameters. The trainer is single-process.
Preset availability does not establish GPU fit or throughput.

## Geometry and parameters

| Field | Screen | Bridge | Flagship |
|---|---:|---:|---:|
| Residual width `D` | 768 | 1,152 | 1,536 |
| Layers / four-layer cells | 12 / 3 | 16 / 4 | 24 / 6 |
| Dense-equivalent FFN width `H` | 3,328 | 4,992 | 6,656 |
| Trunk and MTP per-expert width | 832 | 1,248 | 1,664 |
| GQA query / KV heads, width 96 | 8 / 4 | 12 / 6 | 16 / 8 |
| MHDB groups | 4 | 6 | 8 |
| PKDA heads, width 128 | 10 | 15 | 20 |
| PKDA projection width | 1,280 | 1,920 | 2,560 |
| `f` / `fl` total parameters | 491,999,952 | 1,384,371,980 | 3,532,504,048 |
| `f` training-active non-embedding parameters | 154,325,712 | 446,551,820 | 1,154,923,504 |
| Included auxiliary MTP total parameters | 36,362,664 | 81,398,204 | 144,336,592 |
| Included auxiliary MTP active parameters | 13,359,528 | 29,641,148 | 52,324,048 |

Every scale has a 50,304-row tied embedding/readout, 4,096 predictions per
row, and 128 rows per update in one-row microbatches: 524,288 predicted tokens
per step. Each stored row includes one additional target. Width multipliers
and optimizer ownership are in [architecture.md](architecture.md#nadam-parameters).

Each trunk and MTP FFN stores one shared plus fifteen routed quarter-width
experts. One shared and three selected experts run per token. Active counts
include those four experts and the router in every bank, plus all mixer and
payload-writer parameters.
A microbatch can touch all experts; all parameters, gradients, and optimizer
state occupy memory. Per-bank selection biases are buffers, excluded from
parameter counts. Training's transient assignment counts have shape
`[layers+1,15]`, with MTP last; column-only counts remain `[layers,15]`.

The auxiliary module adds `P_PKDA + 12DH + 2D^2 + 19D` parameters: its PKDA
mixer, expert matrices, concatenation projection, expert router, and four
RMSNorm scales. Its active count replaces `12DH` with `3DH`. It runs once per training
pass over `seq_len-1` positions, with a second vocabulary-loss call, and shares
the embedding/final norm/readout. It does not execute in generation. Condition
`l` retains the payload writer and omits the FBT entry; shared token budgets
still use `f`.

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
| Screen | 7,359 | 3,858,235,392 | 117,742 | 61,730,717,696 |
| Bridge | 21,294 | 11,164,188,672 | 340,693 | 178,621,251,584 |
| Flagship | 55,072 | 28,873,588,736 | 881,137 | 461,969,555,456 |

Warmup is 2% of the shorter of the run and its 25x length: 147, 426, and
1,101 steps at or above 25x. Cooldown occupies 20%. Feedback begins at 75%;
the default mixture costs about 1.28 pass-tokens per prediction. See
[design.md](design.md#schedule).

`--continue TAG` extends a finished run by restoring the last snapshot the
longer schedule reproduces. A 25x screen run continued to 50x restores step
5,519 and trains 9,199 more steps to reach 14,718.

Token stores include validation and each row's extra target.
`delta tokenize --scale S --tokens-per-param R` computes that requirement and
rounds up to the next billion stored tokens. Screen 100x needs 16B, screen
400x needs 62B, bridge 400x needs 179B, and flagship 400x needs 463B.
Full DCLM supports larger stores. Its stream and held-out slice differ from
the publisher's DCLM-100B subset, preventing paired per-token comparisons.

## Loop compute and decode state

With `C` unique cells, the core contains `C-2` cells. At depth `r`, a pass
executes `2+(C-2)r` cells: `2+r` at screen, `2+2r` at bridge, and `2+4r` at
flagship. The default capped draw has uncapped mean 4, cap 8, and actual mean
about 3.88. Report predicted tokens, pass-tokens, and cell-tokens together.
These counters describe the trunk; total compute also includes MTP and its
vocabulary loss. Measured device time includes routing and execution overhead.

For 4,096 cached positions, BF16 GQA K/V and convolution histories plus FP32
PKDA matrix/diagonal states cost:

| Scale | One cell | Flat / `r=1` | `r=4` | `r=8` |
|---|---:|---:|---:|---:|
| Screen | 7.96 MiB | 23.87 MiB | 47.73 MiB | 79.56 MiB |
| Bridge | 11.93 MiB | 47.73 MiB | 119.33 MiB | 214.80 MiB |
| Flagship | 15.91 MiB | 95.47 MiB | 286.40 MiB | 540.98 MiB |

Each core iteration keeps its own mixer cache. These per-sequence state
counts exclude payload, logits, allocator overhead, and serving metadata.
Training also holds activations, gradients, optimizer state, and CUDA graph
pools; decode-state arithmetic is not a training-memory estimate.

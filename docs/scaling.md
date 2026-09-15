# Scale presets and accounting

`--scale screen|bridge|flagship|extension` selects geometry and batch settings.
Explicit geometry or recipe flags override their preset fields. Conditions are `f`,
`l`, and `fl`; looped depth adds no parameters. The trainer is single-process.
Preset availability does not establish GPU fit or throughput.

## Geometry and parameters

| Field | Screen | Bridge | Flagship | Extension |
|---|---:|---:|---:|---:|
| Residual width `D` | 768 | 1,152 | 1,536 | 2,304 |
| Layers / four-layer cells | 16 / 4 | 16 / 4 | 16 / 4 | 16 / 4 |
| Columns per position on rolled steps, mean / maximum (`l`, `fl`) | 2.12 / 3 | 2.12 / 3 | 2.12 / 3 | 2.12 / 3 |
| Default evaluation / decode columns per position | 2 | 2 | 2 | 2 |
| Executed trunk layers per pass, rolled mean / evaluation / maximum | 33.9 / 32 / 48 | 33.9 / 32 / 48 | 33.9 / 32 / 48 | 33.9 / 32 / 48 |
| Trunk and MTP per-expert width `h` | 832 | 832 | 832 | 832 |
| Shared + selected / routed experts | 1 + 3 / 15 | 1 + 5 / 23 | 1 + 7 / 31 | 1 + 11 / 47 |
| Active FFN width `H = (k+1)h` | 3,328 | 4,992 | 6,656 | 9,984 |
| Stored FFN width `(n+1)h` | 13,312 | 19,968 | 26,624 | 39,936 |
| GQA query / KV heads, width 192 | 8 / 4 | 12 / 6 | 16 / 8 | 24 / 12 |
| GQA query projection width `2D` | 1,536 | 2,304 | 3,072 | 4,608 |
| GQA K/V projection width, each `D` | 768 | 1,152 | 1,536 | 2,304 |
| MHDB groups | 4 | 6 | 8 | 12 |
| PKDA heads, width 128 | 10 | 15 | 20 | 30 |
| PKDA projection width | 1,280 | 1,920 | 2,560 | 3,840 |
| Total parameters, every condition | 638,862,216 | 1,403,105,420 | 2,463,891,088 | 5,475,089,816 |
| Training-active non-embedding parameters | 209,175,432 | 465,285,260 | 822,410,896 | 1,839,709,592 |
| Included auxiliary MTP total parameters | 35,181,480 | 78,750,908 | 139,639,504 | 313,374,200 |
| Included auxiliary MTP active parameters | 12,178,344 | 26,993,852 | 47,626,960 | 106,345,976 |

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
in every bank, plus all mixer, payload-writer, and fusion parameters. Only
the raw tied embedding weight is excluded from non-embedding counts.
A microbatch can touch all experts; all parameters, gradients, and optimizer
state occupy memory. Per-bank selection biases are buffers, excluded from
parameter counts. Training's transient assignment counts have shape
`[layers+1,n]`, with MTP last; column-only counts are `[layers,n]`.

The auxiliary block adds `P_PKDA + 3D(n+1)h + (n+2)D` parameters:
its PKDA mixer, expert matrices, expert router, and two RMSNorm scales.
Its active count replaces `(n+1)` with `(k+1)` in the expert term. It runs
once per training pass over `seq_len` positions, of
which `seq_len-1` are supervised, and shares the embedding/final norm/readout
with the main head, including that pass's single vocabulary-loss call.
It does not execute in generation. MTP and feedback share a concat-linear
entry containing one `2D -> D` matrix, or `2D^2` parameters. Token embeddings
enter directly; the payload receives a learned RMSNorm and fixed `0.02`
scale at its writer, then training jitter in those scaled units. The fusion
matrix is counted once outside the auxiliary block, and the fixed payload
scale adds no parameters. Every condition retains the payload writer and
shared entry, so `f`, `l`, and `fl` have identical parameter counts.
MTP trains the shared entry
even on single-pass batches; `f` selects its consumption by the trunk.
The entire fusion matrix uses ordinary NorMuonH; the payload writer's norm
uses base NAdam.

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
| Screen | 9,975 | 5,229,772,800 | 159,589 | 83,670,597,632 |
| Bridge | 22,187 | 11,632,377,856 | 354,985 | 186,114,375,680 |
| Flagship | 39,216 | 20,560,478,208 | 627,450 | 328,964,505,600 |
| Extension | 87,725 | 45,993,164,800 | 1,403,588 | 735,884,345,344 |

Warmup is 2% of the shorter of the run and its 25x length: 200, 444, 784,
and 1,754 steps at or above 25x. Cooldown occupies 20%. Feedback begins at 75%;
the default mixture costs about 1.28 pass-tokens per prediction. See
[design.md](design.md#schedule).

`--continue TAG` extends a finished run by restoring the last snapshot the
longer schedule reproduces.
Token stores include validation and each row's extra target.
`delta tokenize --scale S --tokens-per-param R` computes that requirement and
rounds up to the next billion stored tokens. Screen 100x needs 21B, screen
400x needs 84B, bridge 400x needs 187B, flagship 400x needs 330B, and extension
400x needs 737B.
Full DCLM supports larger stores. Its stream and held-out slice differ from
the publisher's DCLM-100B subset, preventing paired per-token comparisons.

## Loop compute and decode state

With `C` unique cells, a pass of `r` columns executes `Cr` cells. Every
preset has `C=4`, so a column is 16 layers and a pass `16r`. Every scale
uses the same recurrence roll, defined in
[design.md](design.md#the-recurrence-roll); on rolled steps it gives:

| Step shape (passes x columns) | 3 x 2 | 2 x 3 | 2 x 2 |
|---|---:|---:|---:|
| Probability | 12% | 12% | 76% |
| Columns per position, `f` / `l` / `fl` | 3 / 2 / 6 | 2 / 3 / 6 | 2 / 2 / 4 |

On rolled steps `f` and `l` each average 2.12 columns per position and `fl`
4.48; the maximum is six columns, or 96 layers, and `l` alone peaks at three
columns, or 48 layers. Evaluation and decode default to two columns, or 32
layers. `--loop-iterations` overrides only their fixed count within `1..3`;
scale and geometry overrides do not change the roll. Without `l`, every
scale executes its 16 unique layers once per pass.

Resumes and continuations inherit the saved evaluation count unless
explicitly pinned; conflicting overrides are rejected. `--scale` pins
geometry and batch settings, while the roll and the evaluation-count default
are common to every scale. The roll's arguments and the evaluation-count
argument are part of the checkpoint v41 contract.

Report predicted tokens, pass-tokens, and cell-tokens together.
These counters describe the trunk; total compute also includes MTP and its
vocabulary loss. Measured device time includes routing and execution overhead.

For 4,096 cached positions, BF16 GQA K/V and convolution histories plus FP32
PKDA matrix/diagonal states cost:

| Scale | One cell | Flat / `r=1` | Default evaluation `r=2` | Maximum `r=3` |
|---|---:|---:|---:|---:|
| Screen | 13.96 MiB | 55.82 MiB | 111.64 MiB | 167.47 MiB |
| Bridge | 20.93 MiB | 83.73 MiB | 167.47 MiB | 251.20 MiB |
| Flagship | 27.91 MiB | 111.64 MiB | 223.29 MiB | 334.93 MiB |
| Extension | 41.87 MiB | 167.47 MiB | 334.93 MiB | 502.40 MiB |

Each column keeps its own mixer cache at every layer. These per-sequence state
counts exclude payload, logits, allocator overhead, and serving metadata.
Training also holds activations, gradients, optimizer state, and CUDA graph
pools; decode-state arithmetic is not a training-memory estimate.

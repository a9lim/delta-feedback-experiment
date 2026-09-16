# Scale presets and accounting

`--scale screen|bridge|flagship|extension` selects geometry and batch settings.
Explicit geometry or recipe flags override their preset fields. Conditions are `f`,
`l`, and `fl`; looped depth adds no parameters. The trainer runs one process
per device; `--ranks` splits each step's rows across them.
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
| GQA query / KV heads, width 256 | 4 / 2 | 6 / 3 | 8 / 4 | 12 / 6 |
| GQA query projection width `4D/3` | 1,024 | 1,536 | 2,048 | 3,072 |
| GQA K/V projection width, each `2D/3` | 512 | 768 | 1,024 | 1,536 |
| MHDB groups, width 384 | 2 | 3 | 4 | 6 |
| PKDA heads, width 128 | 8 | 12 | 16 | 24 |
| PKDA projection width `4D/3` | 1,024 | 1,536 | 2,048 | 3,072 |
| Total parameters, every condition | 621,389,088 | 1,364,464,240 | 2,395,794,368 | 5,323,219,552 |
| Training-active non-embedding parameters | 191,702,304 | 426,644,080 | 754,314,176 | 1,687,839,328 |
| Included auxiliary MTP total parameters | 34,321,312 | 76,867,376 | 136,337,088 | 306,047,456 |
| Included auxiliary MTP active parameters | 11,318,176 | 25,110,320 | 44,324,544 | 99,019,232 |

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
enter through the fixed lookup multiplier `1/BASE_NORMAL_INIT_STD`; the
payload receives a learned RMSNorm at its writer, then training jitter in
payload units. The fusion matrix is counted once outside the auxiliary
block, and the lookup multiplier adds no parameters. Every condition retains the payload writer and
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
| Screen | 9,142 | 4,793,040,896 | 146,258 | 76,681,314,304 |
| Bridge | 20,344 | 10,666,115,072 | 325,504 | 170,657,841,152 |
| Flagship | 35,969 | 18,858,115,072 | 575,497 | 301,726,171,136 |
| Extension | 80,483 | 42,196,271,104 | 1,287,720 | 675,136,143,360 |

Warmup is 2% of the shorter of the run and its 25x length: 183, 407, 719,
and 1,610 steps at or above 25x. Cooldown occupies 20%. Feedback begins at 75%;
the default mixture costs about 1.28 pass-tokens per prediction. See
[design.md](design.md#schedule).

`--continue TAG` extends a finished run by restoring the last snapshot the
longer schedule reproduces.
Token stores include validation and each row's extra target.
`delta tokenize --scale S --tokens-per-param R` computes that requirement and
rounds up to the next billion stored tokens. Screen 100x needs 20B, screen
400x needs 77B, bridge 400x needs 171B, flagship 400x needs 302B, and extension
400x needs 676B. The canonical corpus retains its fixed 85B-token identity.
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
argument are part of the checkpoint v42 contract.

Report predicted tokens, pass-tokens, and cell-tokens together.
These counters describe the trunk; total compute also includes MTP and its
vocabulary loss. Measured device time includes routing and execution overhead.

For 4,096 cached positions, BF16 GQA K/V and convolution histories plus FP32
PKDA matrix/diagonal states cost:

| Scale | One cell | Flat / `r=1` | Default evaluation `r=2` | Maximum `r=3` |
|---|---:|---:|---:|---:|
| Screen | 9.56 MiB | 38.26 MiB | 76.52 MiB | 114.77 MiB |
| Bridge | 14.35 MiB | 57.39 MiB | 114.77 MiB | 172.16 MiB |
| Flagship | 19.13 MiB | 76.52 MiB | 153.03 MiB | 229.55 MiB |
| Extension | 28.69 MiB | 114.77 MiB | 229.55 MiB | 344.32 MiB |

Each column keeps its own mixer cache at every layer. These per-sequence state
counts exclude payload, logits, allocator overhead, and serving metadata.
Training also holds activations, gradients, optimizer state, and CUDA graph
pools; decode-state arithmetic is not a training-memory estimate.

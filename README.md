# delta-feedback-experiment

This repository tests whether two innovation packages improve a common
PKDA/GGQA hybrid independently or interact when pretrained together:

- **Depth routing** uses Multi-Head Delta Block routing (MHDB) over a column's
  seed and four-layer block deltas.
- **Recurrence innovation** uses Full-Bandwidth Transformer (FBT) feedback to
  carry the previous column's top state into the next through a mandatory
  token-gated fusion.
- **Delta Feedback (DF)** combines both packages and applies the multi-head
  delta router to enrich the recurrent payload between columns.

The primary factorial is `{base, mhdb, fbt, df}`. All four use
`[PKDA, PKDA, PKDA, gated global GQA] x 3`; pure-GQA `vanilla` is the external
trunk control. Every MHDB router has a learnable zero-initialized null source.

## Status

No screen job is active and no registered screen run is complete, so there are
no accepted experiment findings. The 8-by-128 PKDA/GGQA screen, deterministic
trainer, portable
recurrence, and optimized single-GPU CUDA path are implemented and
Jobe-qualified. The full DF train/eval graph pool peaks at 12.87 GiB allocated
and 22.76 GiB reserved across four graphs, so screen jobs remain serial on the
24 GiB RTX 4090.
The token-ladder continuation and the distributed
1.335B-parameter, 441B-token, 20-by-128-PKDA
`[PKDA, PKDA, PKDA, gated global GQA]` block-delta flagship path are specified
but not yet implemented.

See [docs/findings.md](docs/findings.md) for the scientific result surface and
[docs/design.md](docs/design.md) for the complete experiment contract.

## Install

Python 3.12 is required. The shared operational package is installed from the
workspace root; this project owns the model and trainer.

```bash
cd /path/to/transformer-experiments
uv pip install -e .
cd delta-feedback-experiment
uv pip install -e .
```

On Jobe, install the CUDA extras into its shared environment without changing
the pinned Torch/FlashAttention pair:

```bash
uv pip install -e '.[cuda]'
```

## Operate

```bash
# Portable invariant suite; adds the full CUDA execution gate on a CUDA host.
df probe

# Build the fixed FineWeb-Edu/Qwen3 token stream once.
df tokenize --out /data/df/tokens

# Run or queue one arm.
df train example-df-s1 --arm df --seed 1 --data-seed 0 \
  --data-dir /data/df/tokens
df queue example-df-s1 --arm df --seed 1 --data-seed 0 \
  --data-dir /data/df/tokens

# Inspect or control the detached queue.
df status
df watch
df stop TAG|live|all [--at STEP]
df clear TAG|all
```

The queue records exact arguments but does not freeze or inspect Git state.
Changing the checkout never stops an active child; the worker refreshes before
starting the next queued job, and every job runs its probe on the then-current
source.

The default screen run is 9,614 steps, 320 rows per step, and sequence length
1,024: 3,150,315,520 predicted training tokens. Its 327,680-token optimizer
batch exactly matches the flagship, and this is the batch-aligned 25x rung for
the 126,008,544 active non-embedding parameters in `df`, the largest hybrid
arm. Feedback arms use one pass through step 4,807, then draw one, two,
or three passes at a 50%/44%/6% mixture through the rest of heat and cooldown.
Every step is addressed directly into one fixed token stream. Pass counts are
keyed by data seed and step; prefix lengths and jitter additionally use the
global row, so paired arms see identical examples and feedback draws.

The same-geometry ladder uses exact 25x, 100x, and 400x rungs at 9,614,
38,455, and 153,819 optimizer steps. Its branch-from-heat-end continuation path
remains an implementation gate; only the first rung is currently runnable.

New snapshots are immutable checkpoint-contract v12 files under `runs/`.
Only v12 is resumable. `--max-steps` limits only the current invocation; it
never rescales the state-defining schedule.

## Analysis

```bash
python scripts/route_report.py runs/TAG.pt.STEP --data-dir /data/df/tokens
python scripts/payload_swap.py runs/TAG.pt.STEP --data-dir /data/df/tokens
```

The route report summarizes a hard-DF snapshot's per-head block-source weights,
entropy, head divergence, source and null scale, and query geometry. The payload sweep
is a same-checkpoint counterfactual for routed feedback content; it is not an
independently trained arm comparison.

## Documentation

- [docs/design.md](docs/design.md): authoritative architecture, training,
  comparison, evaluation, scale, and gate contract.
- [docs/findings.md](docs/findings.md): accepted experiment findings only.
- [docs/journal.md](docs/journal.md): disposable active notes.
- [references/refs.yaml](references/refs.yaml): primary references and their
  current roles.
- [figures/README.md](figures/README.md): committed experiment-result figures.

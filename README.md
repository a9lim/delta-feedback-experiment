# delta-feedback-experiment

This repository tests whether depth routing and latent recurrence help a common
hybrid decoder independently or become more useful together.

The hybrid trunk repeats three recurrent PKDA layers and one dense gated global
GQA layer. PKDA supplies efficient causal state; global GQA periodically gives
every token a full-prefix read. Multi-Head Delta Block routing (MHDB) lets each
sublayer transiently revisit the column seed and completed four-layer deltas.
Full-Bandwidth Transformer (FBT) carries the previous token column's top state
through a token-gated entry. Hard Delta Feedback (`df`) combines both and uses
MHDB to enrich the payload sent to the next column.

```text
previous payload + token embedding
                │
              FBT gate
                │
                v
 [PKDA, PKDA, PKDA, gated global GQA] x cells
                │
        MHDB-routed payload
                │
                v
          next token column
```

The primary screen is the factorial `{base, mhdb, fbt, df}`. All four share
the same hybrid trunk; `base` deletes both packages, `mhdb` and `fbt` retain
one each, and `df` retains both. Pure twelve-layer RoPE GQA `vanilla` is a
separate whole-trunk control.

## Status

No run under the current screen contract is complete, no screen job is active,
and there are no accepted experiment findings.

The 230–258M screen model, deterministic single-process trainer, portable
semantics, and optimized Jobe CUDA path are implemented and qualified. Jobe
runs remain serial because the full v15 DF graph pool reserves 23.00 GiB on
its 24 GiB RTX 4090.

The next stages are specified but not runnable: a fresh `{base, df}` 400x
comparison on one 8xH100-80GB Prime node, followed—only after the evidence and
implementation gates—by the 1.335B-parameter, 441B-token flagship. See
[docs/architecture.md](docs/architecture.md) for the exact flagship and
NorMuonH/Adam specification, and [docs/design.md](docs/design.md) for the
experiment and scale plan.

## Install

Python 3.12 is required. Install the shared operational package from the
workspace root, then this project:

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
# Portable invariant suite; includes the full CUDA gate on a CUDA host.
df probe

# Build the canonical FineWeb-Edu/Qwen3 stream.
df tokenize --out /data/df/tokens

# Run or queue one arm.
df train example-df-s1 --arm df --seed 1 --data-seed 0 \
  --data-dir /data/df/tokens
df queue example-df-s1 --arm df --seed 1 --data-seed 0 \
  --data-dir /data/df/tokens

# Inspect and control the detached queue.
df status
df watch
df stop TAG|live|all [--at STEP]
df clear TAG|all
```

The queue records exact arguments, not Git state. A source change never stops
an active child; the worker refreshes before the next queued job and runs its
probe from the current checkout.

The default Jobe run uses 10,745 steps, 320 rows per step, and 1,024 predictions
per row: 3,520,921,600 predicted tokens. Feedback starts halfway through the
schedule and draws one, two, or three passes. Every arm sees the same addressed
token rows; feedback arms also share deterministic pass, prefix, and jitter
streams. The global FP32 gradient is clipped to norm 1.0 before the shared
NorMuonH/Adam update.

Snapshots use checkpoint contract v15, and only v15 is resumable.
`--max-steps` limits the current invocation without changing the registered
schedule.

## Analysis

```bash
python scripts/route_report.py runs/TAG.pt.STEP --data-dir /data/df/tokens
python scripts/payload_swap.py runs/TAG.pt.STEP --data-dir /data/df/tokens
```

The route report summarizes per-head block-source selection, null/source
scale, and query geometry. The payload sweep is a same-checkpoint intervention,
not a separately trained arm.

## Documentation

- [docs/architecture.md](docs/architecture.md): exact flagship architecture,
  state, parameter accounting, and optimizer contract.
- [docs/design.md](docs/design.md): comparisons, data, schedules, execution,
  evaluation, scale plan, and promotion gates.
- [docs/findings.md](docs/findings.md): accepted experiment findings only.
- [docs/journal.md](docs/journal.md): disposable active notes.
- [references/refs.yaml](references/refs.yaml): primary references and their
  current roles.
- [figures/README.md](figures/README.md): committed finding figures.

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

No run is complete under the current screen contract: all ten registered
runs are pending, and there are no accepted experiment findings. The
unregistered full-feedback diagnostic `screen-df-fullN-s1` is active on Jobe.
The earlier `screen-df-s1` run used the superseded mixed-heat feedback recipe
and is retained only as a diagnostic checkpoint; its examination is in
[docs/journal.md](docs/journal.md).

The 230–258M screen model, deterministic single-process trainer, portable
semantics, and optimized Jobe CUDA path are implemented and qualified. Jobe
runs remain serial because the qualified DF graph pool reserves 22.98 GiB on
its 24 GiB RTX 4090.

The realized Jobe token store is a 35.000B-token prefix of the pinned stream.
The canonical 57B store has not yet been materialized.

The next stages are specified but not runnable: a fresh `{base, df}` 400x
comparison on one 8xH100-80GB Prime node, followed—only after the evidence and
implementation gates—by the 1.335B-parameter, 441B-token flagship. See
[docs/architecture.md](docs/architecture.md) for the exact flagship and
NorMuonH/NAdam specification, and [docs/design.md](docs/design.md) for the
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

On Jobe, initialize the workspace's kernel forks and install the CUDA extras
into its shared environment without changing the pinned Torch/FlashAttention
pair:

```bash
git -C .. submodule update --init vendor/flash-linear-attention vendor/ml-cross-entropy
uv pip install -e '.[cuda]'
```

CCE and FLA install editable from the workspace forks under `../vendor/`
through `tool.uv.sources` in `pyproject.toml`, so the root repository's
submodule pointers are their only pin and kernel edits there are live without
reinstalling. The compatible FlashAttention version is pinned in the extra;
machine-wide constraints continue to own Torch itself. The first `df probe` performs the fixed-shape Inductor search and preserves its
generated artifacts at `~/.cache/delta-feedback/torchinductor`; later probes
and training processes reuse that cache. Set `DF_INDUCTOR_CACHE_DIR` only when
the durable cache belongs elsewhere.

Install the exact data-build stack on any staging host that materializes the
canonical token stream:

```bash
uv pip install -e '.[data-build]'
```

## Operate

```bash
# Portable invariant suite; includes the full CUDA gate on a CUDA host.
df probe

# Materialize the canonical 57B stream from pinned HF dataset/tokenizer commits.
df tokenize --out /data/df/tokens

# Run or queue one arm.
df train example-df-s1 --arm df --seed 1 --data-seed 0 \
  --data-dir /data/df/tokens
df queue example-df-s1 --arm df --seed 1 --data-seed 0 \
  --data-dir /data/df/tokens

# Inspect and control the detached queue.
df status
df watch
df stop TAG|live [--at STEP]
df stop queue
df stop all
df clear TAG|all
```

The queue records exact arguments, not Git state. A source change never stops
an active child; the worker refreshes before the next queued job and runs its
probe from the current checkout. `df stop queue` removes every pending job but
leaves the active run and worker untouched; it is the complement of `df stop
live`, which stops only the active run and preserves the pending queue.

The default Jobe run uses 10,745 steps, 320 rows per step, and 1,024 predictions
per row: 3,520,921,600 predicted tokens. Linear warmup occupies the first 2%
of updates and the `1 - sqrt(u)` cooldown occupies the final 20%. Feedback
starts independently at three quarters of the schedule, and every later step
draws two or three passes. Every arm sees the same addressed
token rows; feedback arms also share deterministic pass, prefix, and jitter
streams. The global FP32 gradient is clipped to norm 3.0 before the shared
NorMuonH/NAdam update. NorMuonH uses a `2e-2` stable rate; the tied
embedding/readout and remaining NAdam parameters use `6e-4` and `3e-4`.
Both sides apply their specified Nesterov construction:
NorMuonH before orthogonalization and NAdam through its scheduled first moment.

Snapshots use checkpoint contract v21, and only v21 is resumable. V16-v20
remain readable for evaluation and forks; v19 retains the retired fixed-step
warmup contract.
Protected snapshots persist at the cooldown boundary, the
feedback boundary, and the end of the run.
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
- [docs/depth-architecture.md](docs/depth-architecture.md): specified,
  unimplemented `df-loop` arm: tied recurrent core, shared core cache,
  and its contract deltas.
- [docs/findings.md](docs/findings.md): accepted experiment findings only.
- [docs/journal.md](docs/journal.md): disposable active notes.
- [references/refs.yaml](references/refs.yaml): primary references and their
  current roles.
- [figures/README.md](figures/README.md): committed finding figures.

# Operating the model organism

Use this page to reproduce training and inspect checkpoints. The research
purpose and current study are in [design.md](design.md); the analysis protocol
is in [interpretability.md](interpretability.md). Training creates specimens
for that work. A completed job is not itself a mechanistic result.

Commands below run from this repository unless they explicitly change
directory. Before using Jobe, inspect `df status`, the active log, and GPU
ownership with `nvidia-smi`. Keep GPU jobs serial and preserve active work.

## Install

The workspace runtime is Python 3.13. Install the shared operational package
from the workspace root, then this project:

```bash
cd /path/to/transformer-experiments
uv pip install -e .
cd delta-feedback-experiment
uv pip install -e .
```

On Jobe, initialize the workspace's kernel forks and install the CUDA extras
into its shared PyTorch 2.14 / CUDA 13.2 environment:

```bash
git -C .. submodule update --init vendor/flash-linear-attention vendor/ml-cross-entropy
uv pip install -e '.[cuda]'
```

CCE and FLA install editable from the workspace forks under `../vendor/`
through `tool.uv.sources` in `pyproject.toml`, so the root repository's
submodule pointers are their only pin and kernel edits there are live without
reinstalling. CUDA GQA uses compiled PyTorch FlexAttention; the external
`flash-attn` package is not used. Machine-wide constraints own Torch itself.
The first `df probe` performs the fixed-shape Inductor search and preserves its
generated artifacts at `~/.cache/delta-feedback/torchinductor`; later probes
and training processes reuse that cache. Set `DF_INDUCTOR_CACHE_DIR` only when
the durable cache belongs elsewhere.

Install the exact data-build stack on any staging host that materializes the
canonical token stream:

```bash
uv pip install -e '.[data-build]'
```

## Operate

Choose the command for the intended action; `TAG` and `STEP` are placeholders.
Foreground training and queueing are alternative ways to run an arm.

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
df stop TAG --at STEP
df stop live
df stop queue
df stop all
df clear TAG
df clear all
```

The queue records exact arguments, not Git state. A source change never stops
an active child; the worker refreshes before the next queued job and runs its
probe from the current checkout. `df stop queue` removes every pending job but
leaves the active run and worker untouched; it is the complement of `df stop
live`, which stops only the active run and preserves the pending queue.

The default Jobe run uses 10,745 steps, 320 rows per step, and 1,024
predictions per row: 3,520,921,600 predicted tokens. Linear warmup occupies the
first 2% of updates and the `1 - sqrt(u)` cooldown occupies the final 20%.
Feedback starts independently at three quarters of the schedule; feedback arms
draw two or three passes on every later step. Every arm sees the same addressed
token rows; feedback arms also share deterministic pass, prefix, and jitter
streams. The global FP32 gradient is clipped to norm 10.0 before the shared
NorMuonH/NAdam update. NorMuonH uses a `6e-3` stable rate; every NAdam-owned
parameter, including the tied embedding/readout, uses `3e-4`. Both sides apply
their specified Nesterov construction: NorMuonH before orthogonalization and
NAdam through its scheduled first moment.

Snapshots use checkpoint contract v23, and only v23 is resumable. V16–v22
remain readable for evaluation and forks. Reading an older diagnostic snapshot
does not make it a member of the current registered comparison. Protected
snapshots persist at the cooldown boundary, the feedback boundary, and the end
of the run. `--max-steps` limits the current invocation without changing the
registered schedule.

## Inspect a checkpoint

```bash
python scripts/route_report.py runs/TAG.pt.STEP --data-dir /data/df/tokens
python scripts/payload_swap.py runs/TAG.pt.STEP --data-dir /data/df/tokens
python scripts/payload_swap.py runs/TAG.pt.STEP --data-dir /data/df/tokens --head 2
```

The route report writes figures and prints routing summaries. The payload sweep
prints held-out losses for the trained router, top-only, uniform, and
forced-source enrichment; `--head` restricts source replacement to one routing
group. Use a `df` checkpoint for the payload sweep. These tools are diagnostic
starting points, not a complete causal-analysis or learned-monitor suite.
Record the exact checkpoint, data rows, device, precision, and command with any
retained output. The shared analysis environment supplies Matplotlib for the
route report.

`df watch` shows training health, validation, and recurrence diagnostics. It is
an operational monitor; it does not classify hidden reasoning or establish that
internal computation is understood. See
[interpretability.md](interpretability.md) for the separate monitoring research
objective.

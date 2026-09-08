# Operating the model organism

Use this page to train specimens and inspect checkpoints. The recipe is in
[design.md](design.md); the analysis scripts are in
[interpretability.md](interpretability.md).

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
Foreground training and queueing are alternative ways to run a condition.

```bash
# Portable invariant suite; includes the full CUDA gate on a CUDA host.
df probe

# Materialize the canonical 57B stream from pinned HF dataset/tokenizer commits.
df tokenize --out /data/df/tokens

# Run or queue one condition: letters from arfl in any order, empty for the
# plain RoPE GQA decoder (`df train --help` lists the letters).
df train example-arf-s1 --condition arf --seed 1 --data-seed 0 \
  --data-dir /data/df/tokens
df queue example-arf-s1 --condition arf --seed 1 --data-seed 0 \
  --data-dir /data/df/tokens
df queue example-plain-s1 --condition "" --seed 1 --data-seed 0 \
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
Feedback starts independently at three quarters of the schedule; conditions
with `f` draw two or three passes on every later step. Every condition sees the
same addressed token rows; conditions with `f` also share deterministic pass,
prefix, and jitter streams. The global FP32 gradient is clipped to norm 10.0 before the shared
NorMuonH/NAdam update. NorMuonH uses a `6e-3` stable rate; every NAdam-owned
parameter, including the tied embedding/readout, uses `3e-4`. Both sides apply
their specified Nesterov construction: NorMuonH before orthogonalization and
NAdam through its scheduled first moment.

Snapshots use checkpoint contract v24, and only v24 is resumable. V16–v23
remain readable for evaluation and forks; they recorded the condition as an
arm name, which the analysis loader translates. Protected snapshots persist at the
cooldown boundary, the feedback boundary, and the end of the run. `--max-steps`
limits the current invocation without changing the schedule.

## Inspect a checkpoint

Every analysis script rebuilds the condition from a snapshot through
`delta_feedback_experiment.analysis`, evaluates under the trainer's numerics
(BF16 autocast on CUDA), and writes a JSON record beside its figures under
`figures/<kind>-<tag>/`. Those directories stay ignored; the [figure
index](../figures/README.md) maps them.

```bash
# Routing: per-site/group source mass, entropy, query geometry (any r).
python scripts/route_report.py runs/TAG.pt.STEP --data-dir /data/df/tokens

# Payload enrichment swaps (r with f): trained router, top-only, uniform, forced source.
python scripts/payload_swap.py runs/TAG.pt.STEP --data-dir /data/df/tokens
python scripts/payload_swap.py runs/TAG.pt.STEP --data-dir /data/df/tokens --head 2

# Fused pass against pass 1 on one feedback snapshot: position, surprise, and
# frequency structure, gate and seed statistics, self-composition.
python scripts/fused_diagnostics.py runs/TAG.pt.STEP --data-dir /data/df/tokens

# Entry interventions: gate temperature, seed scale, embedding bypass,
# zero or foreign payload.
python scripts/entry_sweeps.py runs/TAG.pt.STEP --data-dir /data/df/tokens

# Impulse response, payload-head ablation, split-validated ensemble, and
# pass-1 versus fused-pass gradient alignment.
python scripts/feedback_followups.py runs/TAG.pt.STEP --data-dir /data/df/tokens

# Two checkpoints on the same rows: per-token loss structure, predictor
# divergence, mixtures, residual-stream CKA, payload redundancy.
python scripts/compare_conditions.py --reference runs/A.pt.STEP --feedback runs/B.pt.STEP \
  --data-dir /data/df/tokens

# Paired weight-space divergence from the shared initialization (CPU).
python scripts/weight_divergence.py runs/A.pt.STEP runs/B.pt.STEP

# Continue a feedback snapshot with dense feedback passes (or --passes 1 as
# the plain-only erosion control) at a fraction of the stable learning rate.
python scripts/dense_feedback_continue.py runs/TAG.pt.STEP --data-dir /data/df/tokens \
  --steps 150 --passes 2 --out figures/fused-TAG/dense_all.json

# Downstream zero-shot tasks (workspace `transformer_experiments.downstream`),
# in Standard mode or the Fused mode of a feedback snapshot; compare two runs
# pairwise with the workspace command.
python scripts/downstream_eval.py runs/TAG.pt.STEP --mode standard
python scripts/downstream_eval.py runs/TAG.pt.STEP --mode fused
python -m transformer_experiments.downstream --compare \
  figures/downstream-A/downstream_standard.json figures/downstream-B/downstream_standard.json

# Figures: training dynamics from run logs, and panels from the JSON records.
python scripts/training_curves.py logs/A.log logs/B.log --out-dir figures/curves-A-vs-B
python scripts/analysis_figures.py --compare figures/compare-A-vs-B/compare_conditions.json \
  --weights figures/weights-A-vs-B/weight_divergence.json \
  --fused figures/fused-B/fused_diagnostics.json --entry figures/fused-B/entry_sweeps.json \
  --followups figures/fused-B/feedback_followups.json --swap figures/fused-B/payload_swap.json \
  --out-dir figures/compare-A-vs-B
```

Payload and entry swaps perturb a co-adapted pathway, so they bound what the
trained model depends on rather than what an alternative design would achieve.
Record the checkpoint, data rows, device, precision, and command with any
output you keep. The shared analysis environment supplies Matplotlib.

`df watch` shows training health, validation, and recurrence diagnostics.

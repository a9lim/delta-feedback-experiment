# Operating the model organism

Use this page to train specimens and inspect checkpoints. The recipe is in
[design.md](design.md); the analysis scripts are in
[interpretability.md](interpretability.md).

Commands below run from this repository unless they explicitly change
directory. Before using Jobe, inspect `delta status`, the active log, and GPU
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
reinstalling. BF16 CUDA training and prefill use PyTorch's native Flash SDPA
with GQA; FP32 diagnostics use its math backend. Cached single-token decoding
uses compiled FlexAttention over the valid prefix. The external `flash-attn`
package is not used. Machine-wide constraints own Torch itself.
The first `delta probe` performs the fixed-shape Inductor search and preserves
its generated artifacts at `~/.cache/delta-feedback/torchinductor`; later
probes and training processes reuse that cache. Set `DELTA_INDUCTOR_CACHE_DIR` only when
the durable cache belongs elsewhere.

Install the data-build requirements on any staging host that materializes the
canonical token stream:

```bash
uv pip install -e '.[data-build]'
```

The project's version constraints are lower bounds, including the `data-build` and
`kernel-bench` extras. Jobe's machine constraints select its qualified Torch
wheel. Each token store records the build packages actually used; partial
resumption and `--continue` require those recorded versions. After a package
upgrade, build into a new directory or reproduce the original store's build
environment before extending it. Reading an existing store for training does
not require its build package versions.

## Format a conversation

Load the experiment's tokenizer through its shared entry point so the two
ChatML delimiters, template, and pinned identity are installed together:

```python
from delta_feedback_experiment.tokenizer import load_tokenizer

tokenizer = load_tokenizer()
messages = [
    {"role": "researcher", "content": "State the hypothesis."},
    {"role": "critic", "content": "Identify a counterexample."},
    {"role": "critic", "content": "Then check the boundary case."},
]
token_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, next_role="researcher"
)
```

Role names are ordinary text and stay as supplied, including repeated roles.
Omitting `next_role` opens a `self` turn. `<|im_end|>` ends a message;
`<|endoftext|>` is the separate pretraining document EOS. This loads tokenizer
files only. The web-text pretraining pipeline does not apply ChatML.

## Operate

Choose the command for the intended action; `TAG` and `STEP` are placeholders.
Foreground training and queueing are alternative ways to run a condition.

```bash
# Portable invariant suite; includes the full CUDA gate on a CUDA host.
delta probe

# Screen at 100 tokens per parameter: 14.083B predictions, a 15B store.
# dclm-100b is already shuffled; its default keeps published order and
# downloads only the source files intersecting the selected prefix.
delta tokenize --source dclm-100b --data-root /data/delta \
  --scale screen --tokens-per-param 100 \
  --scratch /data/delta/scratch/dclm-100b --workers 3 --readers 8
delta verify /data/delta/dclm-100b
# Extend the same source and ordering in place, e.g. screen 400x (57B).
delta tokenize --source dclm-100b --data-root /data/delta --continue \
  --scale screen --tokens-per-param 400 \
  --scratch /data/delta/scratch/dclm-100b --workers 3
# Full DCLM for larger stores: defaults to keyed document shuffling.
# This scans the complete source. Run on suitably sized CPU storage before
# provisioning a GPU node; Prime spending is not scheduled here.
delta tokenize --source dclm --data-root /data/delta \
  --scale flagship --tokens-per-param 400 \
  --scratch /data/delta/scratch/dclm --workers 3 --readers 8
# --shuffle and --no-shuffle explicitly override a source's default.
# Sources and pins are listed in docs/design.md. Different sources have
# different streams and held-out slices. --out overrides DATA_ROOT/SOURCE.

# Run or queue one condition: letters from arfl in any order, empty for the
# plain gated GQA decoder (`delta train --help` lists the letters).
delta train example-arf-s1 --condition arf --seed 1 --data-seed 0 \
  --data-root /data/delta --source dclm-100b
delta queue example-arf-s1 --condition arf --seed 1 --data-seed 0 \
  --data-root /data/delta --source dclm-100b
delta queue example-plain-s1 --condition "" --seed 1 --data-seed 0 \
  --data-root /data/delta --source dclm-100b
delta queue example-arfl-s1 --condition arfl --seed 1 --data-seed 0 \
  --data-root /data/delta --source dclm-100b

# Every planned run is --condition, --scale, and --tokens-per-param: the
# preset fills the geometry and batch, the ratio derives the schedule (25 is
# the screen recipe, 400 the Prime recipes; docs/scaling.md has each budget).
# The bridge on Jobe, and the memory staging to run before it:
delta queue bridge-delta-arf-s1 --condition arf --scale bridge --seed 1 --data-seed 0 \
  --data-root /data/delta --source dclm-100b
delta queue bridge-delta-arfl-s1 --condition arfl --scale bridge --seed 1 --data-seed 0 \
  --data-root /data/delta --source dclm-100b
python scripts/loop_memory_stage.py --out data/summary/loop-stage-bridge-DATE.json \
  --condition arfl --scale bridge
# The 400x recipes, which the single-process trainer expresses and Prime's
# unbuilt distributed path would run:
delta train screen-delta-arf-400x-s1 --condition arf --tokens-per-param 400 \
  --seed 1 --data-seed 0 --data-root /data/delta --source dclm-100b
delta train flagship-delta-arfl-s1 --condition arfl --scale flagship --tokens-per-param 400 \
  --seed 1 --data-seed 0 --data-root /data/delta --source dclm
# Extend a finished run to a longer schedule under a new tag: its stable
# phase resumes from the last snapshot the longer schedule reproduces.
delta queue screen-delta-arf-s1-50x --continue screen-delta-arf-s1 --tokens-per-param 50 \
  --data-root /data/delta --source dclm-100b

# Inspect and control the detached queue.
delta status
delta watch
delta stop TAG --at STEP
delta stop live
delta stop queue
delta stop all
delta clear TAG
delta clear all

# Rename a finished or stopped run, keeping its saved trajectory.
delta move screen-delta-r-s1 screen-delta-r-s1-nope
```

The queue records exact arguments, not Git state. A source change never stops
an active child; the worker refreshes before the next queued job and runs its
probe from the current checkout. `delta stop queue` removes every pending job
but leaves the active run and worker untouched; it is the complement of
`delta stop live`, which stops only the active run and preserves the pending
queue. A stop is SIGINT: the trainer snapshots the completed step and the tag
resumes from it. A child that has not exited 120 s later is killed and the
marker records `TAG KILLED`; `delta stop` holds that deadline itself when no
worker is alive. The worker restores Python's SIGINT handling before its first
job, so a launcher that ignored the signal (`nohup`, a backgrounded command in
a non-interactive shell) cannot leave a run that no stop can reach.

`delta move OLD NEW` renames the run's snapshots, training/probe logs, and
standard `figures/<kind>-<tag>` and comparison directories. It updates text
records, continuation links in idle logs, and monitor status. Checkpoint bytes
stay intact: the canonical `TAG.pt.STEP` filename supplies the current tag
when loaded, while the serialized invocation remains as originally recorded.
Rendered binary figures retain their existing labels until regenerated.
Use `--out-dir DIR` for snapshots trained outside `runs/`; arbitrary custom
analysis output paths and shared queue output are not renamed.
Both tags must be idle and the destination unused; queued or active jobs
referencing the source also block the move. Ordinary I/O failures roll back
the changes, though a multi-file rename is not atomic against machine failure.

The default Jobe run uses 6,716 steps, 128 rows per step, and 4,096
predictions per row: 3,521,118,208 predicted tokens. Linear warmup occupies the
first 2% of updates and the `1 - sqrt(u)` cooldown occupies the final 20%.
Feedback starts independently at three quarters of the schedule; conditions
with `f` draw two or three passes on every later step. Conditions with `l`
draw the core iteration count once per step, mean 4 and cap 8, and log it as
`r`. Every condition sees the same addressed token rows; conditions with `f`
also share deterministic pass, prefix, and jitter streams, and the iteration
draw is its own stream. The global FP32 gradient is clipped to norm 10.0 before the shared
NorMuonH/NAdam update. NorMuonH uses a `6e-3` stable rate; NAdam uses `3e-4`
at the muP reference width 1536, the fan-in-`D` gate and control matrices
`3e-4 x 1536 / D`, and the tied readout multiplies its logits by the same
ratio, so the screen runs those at `6e-4` under a doubled readout. Both sides apply
their specified Nesterov construction: NorMuonH before orthogonalization and
NAdam through its scheduled first moment.

Checkpoint v28 is the only accepted contract for resume, evaluation, and
forks; it uses the pinned GPT-NeoX tokenizer with generic ChatML delimiters
and a 50,304-row model vocabulary. Token stores must match that tokenizer
identity; the source path alone does not make a store compatible.
Protected snapshots persist at the
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
python scripts/route_report.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b

# Payload enrichment swaps (r with f): trained router, top-only, uniform, forced source.
python scripts/payload_swap.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b
python scripts/payload_swap.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b --head 2

# Fused pass against pass 1 on one feedback snapshot: position, surprise, and
# frequency structure, gate and seed statistics, self-composition.
python scripts/fused_diagnostics.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b

# Loop depth trace (l): loss after each iteration count, core update sizes,
# and core router mass by iteration, plain and fused.
python scripts/depth_trace.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b

# Entry interventions: gate temperature, seed scale, embedding bypass,
# zero or foreign payload.
python scripts/entry_sweeps.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b

# Impulse response, payload-head ablation, split-validated ensemble, and
# pass-1 versus fused-pass gradient alignment.
python scripts/feedback_followups.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b

# Two checkpoints on the same rows: per-token loss structure, predictor
# divergence, mixtures, residual-stream CKA, payload redundancy.
python scripts/compare_conditions.py --reference runs/A.pt.STEP --feedback runs/B.pt.STEP \
  --data-dir /data/delta/dclm-100b

# Paired weight-space divergence from the shared initialization (CPU).
# Use the initialization source revision and constants that trained both runs.
python scripts/weight_divergence.py runs/A.pt.STEP runs/B.pt.STEP

# Continue a feedback snapshot with dense feedback passes (or --passes 1 as
# the plain-only erosion control) at a fraction of the stable learning rate.
python scripts/dense_feedback_continue.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b \
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

`delta watch` shows training health, validation, and recurrence diagnostics.

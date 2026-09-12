# Operations

Before using Jobe, inspect `delta status`, the active log, and `nvidia-smi`.
Keep GPU jobs serial and preserve active training and data builds.

## Install and check

Use the shared Python 3.13 environment. Install the workspace before Delta:

```bash
cd /path/to/transformer-experiments
uv pip install -e .
cd delta-feedback-experiment
uv pip install -e .
```

CUDA uses the workspace's editable FLA and CCE forks; tokenization needs the
`data-build` extra. Machine constraints own Jobe's PyTorch/CUDA versions.

```bash
git -C .. submodule update --init vendor/flash-linear-attention vendor/ml-cross-entropy
uv pip install -e '.[cuda,data-build]'
uv pip check
python -m pytest
delta probe
```

The default tests cover small portable model, training, and lifecycle cases.
`delta probe` requires CUDA and runs one small train/eval/decode smoke,
exercising kernels and graph replay. First-use kernel JIT adds startup
time. Neither command measures production throughput or memory fit. The
queue starts training directly without running tests or a probe.

## Tokenize

```bash
delta tokenize --source dclm-100b --data-root /data/delta \
  --scale screen --tokens-per-param 100 \
  --scratch /data/delta/scratch/dclm-100b --workers 3 --readers 8
delta verify /data/delta/dclm-100b
```

`--data-root ROOT` writes `ROOT/SOURCE`; `--out` overrides the path.
`--continue` extends a finished store with matching build settings.
`--shuffle` / `--no-shuffle` override source ordering. Use full `dclm` for
stores beyond the 100B subset. Source identity, row layout, and temporary
storage requirements are in [design.md](design.md#data).

## Runs

```bash
delta train example-f-s1 --condition f --seed 1 --data-seed 0 \
  --data-root /data/delta --source dclm-100b
delta queue example-fl-s1 --condition fl --seed 1 --data-seed 0 \
  --data-root /data/delta --source dclm-100b
delta train example-f-s1 --resume
delta queue example-f-s1-50x --continue example-f-s1 --tokens-per-param 50

delta status
delta watch
delta stop TAG --at STEP
delta stop live
delta stop queue
delta stop all
delta clear TAG
delta move old-tag new-tag
```

`stop live` preserves pending jobs; `stop queue` preserves the active job.
Stopping sends SIGINT so the trainer snapshots its completed step; a child
that does not exit within 120 seconds is killed. `--max-steps` caps one
invocation without shortening the schedule. `clear` moves an idle run's artifacts into timestamped recovery. `move` renames idle snapshots, logs, and standard analysis paths;
both tags must have no active or queued references and the destination must
be free. Use `--out-dir` for snapshots outside `runs/`; custom outputs are
not renamed. Existing figures retain their labels until regenerated.

Resume and continuation use [current snapshots](design.md#checkpoints-and-queue).
The queue stores arguments and refreshes the checkout before each job.
Use `delta train --help` for recipe and runtime overrides.

## Telemetry

| Field | Meaning |
|---|---|
| `loss` | Full optimized objective |
| `ntp`, `mtp` | Combined main / auxiliary CE before weights and z-loss |
| `pass1` | Main first-pass CE; `ntp - pass1` is mean feedback CE on multipass steps |
| `val`, `val_fused` | Main plain / fused validation CE |
| `val_mtp`, `val_mtp_fused` | Auxiliary plain / fused validation CE |
| `expert_balance` | Unweighted mean sequence balance loss |
| `expert_max_violation` | Worst bank's whole-update `max(load)/mean(load)-1` |
| `expert_bias_max` | Maximum absolute post-update selection bias |

`delta watch` streams run milestones. The browser monitor plots training
health, validation, routing, and recurrence.
Expert route summaries use up to two validation rows and describe that sample.

## Conversation formatting

```python
from delta_feedback_experiment.tokenizer import load_tokenizer

tokenizer = load_tokenizer()
token_ids = tokenizer.apply_chat_template(
    [
        {"role": "researcher", "content": "State the hypothesis."},
        {"role": "critic", "content": "Check the boundary case."},
    ],
    add_generation_prompt=True,
    next_role="researcher",
    return_dict=False,
)
```

Roles remain as supplied, including repeated names. The default next role is
`self`. `<|im_end|>` ends a message; `<|endoftext|>` ends a pretraining document.
Web pretraining does not apply ChatML.

## Inspect a checkpoint

```bash
python scripts/route_report.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b
python scripts/payload_swap.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b
python scripts/depth_trace.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b
python scripts/downstream_eval.py runs/TAG.pt.STEP --mode standard
python scripts/downstream_eval.py runs/TAG.pt.STEP --mode fused
python scripts/training_curves.py logs/A.log logs/B.log --out-dir figures/curves-A-vs-B
```

Checkpoint tools use `delta_feedback_experiment.analysis` and the trainer's
numerics. Outputs under `figures/` are regenerable. Plotting uses Matplotlib
from the shared analysis environment. The [analysis guide](interpretability.md)
defines each retained tool's measurement.

## CUDA execution

CUDA uses BF16 activations with FP32 parameters, accumulated gradients,
optimizer state, and PKDA recurrent boundaries. FLA handles PKDA; native Flash
SDPA handles full-row attention and FlexAttention handles cached prefixes.
CCE reads a BF16 classifier shadow and flushes gradients to the FP32 sink at
`--head-flush-every` calls and before each optimizer update. A pass makes one
head call, covering both prediction depths, so the default cadence of 2 holds
one two-pass microbatch.

Training compiles blocks and captures fixed-address graphs for reachable
pass/depth pairs. Inductor caches persist at
`~/.cache/delta-feedback/torchinductor`; `DELTA_INDUCTOR_CACHE_DIR` relocates
them. Before capture the trainer measures the activation bytes one block
invocation retains and plans each graph against device memory minus the
static footprint and `--checkpoint-margin-gib` (default 3.5): rows per
replay for one-pass graphs, otherwise how many leading PKDA and auxiliary
block invocations recompute in backward. The `execution` record reports the
static footprint, the budget, and the bytes per block; one `plan` record per
graph reports its rows and recomputed block count. Raise the margin if
capture or the optimizer step runs out of memory. Keep cyclic Python garbage
collection outside train/eval capture.
CPU/MPS use eager attention, literal PKDA, and chunked tied-head loss.

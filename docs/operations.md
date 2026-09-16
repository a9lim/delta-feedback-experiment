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
`delta probe` requires CUDA and runs small one-pass and two-pass training
graphs plus evaluation and decode, exercising kernels, shared fusion, and
graph replay. First-use kernel JIT adds startup
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
stores beyond the 100B subset.

`scripts/publish_store.sh --disk DISK` builds the whole dclm-100b stream on a
box with fast cores and a fast uplink (100e9 stored tokens, about 400 GB, a
few hours on the rental's Grace CPUs), verifies it, checks every full shard
and the held-out slice against a reference manifest, writes the dataset
card, and uploads it as the public dataset `a9lim/dclm-100b-neox`. Any box
then pulls a prefix with `hf download` instead of copying from Jobe. Source identity, row layout, and temporary
storage requirements are in [design.md](design.md#data).

## Runs

```bash
delta train example-f-s1 --condition f --seed 1 --data-seed 0 \
  --data-root /data/delta --source dclm-100b
delta queue example-fl-s1 --condition fl --seed 1 --data-seed 0 \
  --data-root /data/delta --source dclm-100b
delta queue node-fl-s1 --condition fl --scale flagship --ranks 8 \
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

`--ranks N` runs `N` processes, one per CUDA device, through `torchrun`:
`delta train` replaces itself with the launcher, so the spool keeps the
process it started. Each rank takes `batch-rows/ranks` rows of every step,
which must divide evenly. The rank count is per invocation, never inherited
by a resume, and a snapshot written by any rank count resumes under any
other. `stop live` preserves pending jobs; `stop queue` preserves the active
job. Stopping sends SIGINT, which the launcher forwards to every rank; the
ranks finish the step they are on, agree to stop, snapshot it, and exit. A
second SIGINT aborts without a snapshot. `stop TAG --at STEP` signals once,
on the record of the step before `STEP`, so the ranks read the request
inside `STEP` and snapshot it. The launcher gives its ranks 110 seconds to
snapshot before it kills them, inside the spool's 120-second grace, and a
kill by the spool reaches the ranks' own sessions. `delta probe --ranks N`
runs the CUDA smoke on `N` devices with the collectives a step makes. `--max-steps` caps one
invocation without shortening the schedule. `clear` moves an idle run's artifacts into timestamped recovery. `move` renames idle snapshots, logs, and standard analysis paths;
both tags must have no active or queued references and the destination must
be free. Use `--out-dir` for snapshots outside `runs/`; custom outputs are
not renamed. Existing figures retain their labels until regenerated.

Resume and continuation accept only [v42 snapshots](design.md#checkpoints-and-queue).
The queue stores arguments and refreshes the checkout before each job.
Use `delta train --help` for recipe and runtime overrides.

## Telemetry

| Field | Meaning |
|---|---|
| `loss` | Full optimized objective |
| `ntp`, `mtp` | Combined main / auxiliary CE before weights and z-loss |
| `pass1` | Main CE of the first column; `ntp - pass1` is the mean over the other columns on rolled steps |
| `k`, `r` | The step's passes and columns per pass; `r` appears under `l` |
| `val`, `val_fused` | Main plain / fused validation CE at the evaluation column count |
| `val_one` | Main plain validation CE after the first column, under `l` |
| `val_mtp`, `val_mtp_fused` | Auxiliary plain / fused validation CE |
| `expert_balance` | Unweighted mean sequence balance loss |
| `expert_max_violation` | Worst bank's whole-update `max(load)/mean(load)-1` |
| `expert_bias_max` | Maximum absolute post-update selection bias |

`delta watch` streams run milestones. The browser monitor groups raw learning
curves, layer anatomy, optimization, and recurrence. The overview holds progress,
pace, cumulative token counts, and the host's current GPU reading; configuration
and non-step logs expand on demand. Token totals require the complete addressed
step history, including inherited fork steps, and exclude the MTP branch.

The **Eval step** slider moves all layer profiles and expert heatmaps together.
Arrow buttons and keyboard arrows select recorded evals. A selection stays pinned
while the log refreshes; **Follow latest** resumes tracking new evals. Switching
runs resets to the latest eval. A resume that removes a selected eval clamps the
selection to the nearest surviving earlier eval (or the first remaining eval).
Routing overlays use the exact selected step; absent sites or evals remain blank.
The heatmaps show the selected run. Assignment colors use multiples of uniform
load; bias colors and null-RMS axes keep a fixed range across recorded evals.

Routing profiles expose null, seed and previous-cell mass, maximum weight with
its `1/n` reference, head divergence, and learned-null RMS. Expert assignment
heatmaps show each executed site's share per expert; hovering or focusing a cell
also reports expert coverage and gate entropy. Bias heatmaps show each bank's
signed selection biases. Both diagnostics use up to two validation rows:
routing uses the fused pass at its last column with feedback, while expert
loads use pass-1 and the teacher-forced MTP block. Both diagnostics disable
payload jitter. These are sample diagnostics. The looped-column section
plots the per-column depth readout, and the training feedback gain is shown
only on single-column steps, since looped steps fold the loop blocks into
the combined CE.

Portable monitor data/lifecycle checks run with `node --test tests/monitor.test.cjs`
from this checkout in the parent workspace. Browser checks cover slider input,
overlays, refresh, heatmap inspection, and responsive layout.

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

CUDA uses BF16 activations with FP32 masters, accumulated gradients,
optimizer state, and PKDA recurrent boundaries. FLA handles PKDA; native
fused SDPA handles full-row attention and FlexAttention handles cached
prefixes. CCE reads the classifier working copy and accumulates its gradient
straight into the tied embedding's FP32 sink on every call. A column makes
one head call, covering both prediction depths. The projection and expert
GEMMs run on FP8 tensor cores under the default `--precision fp8` recipe
([architecture](architecture.md#precision-and-initialization)): each site
carries an FP8 copy of its working copy, both layouts with per-row scales,
about one byte per element beyond the BF16 copy; `--precision bf16` keeps
every GEMM in BF16. The recipe is a runtime setting: a resume keeps the
checkpoint's unless retyped, and it is recorded in the `run` record.

Every parameter is a view of its site's two slabs, a working copy the
kernels read and an FP32 gradient sink ([architecture](architecture.md#precision-and-initialization)).
With several ranks, an expert bank is chunked along its expert axis so each
rank owns a contiguous run of experts, every other NorMuonH site is owned
whole by one rank, and the NAdam parameters are replicated. A step reduces
the sinks onto their owners in place (reduce-scatter for banks, reduce for
dense sites, all-reduce for the replicated arena), each rank steps the
matrices it owns against their FP32 masters, and the updated working copies
gather back. Per rank a NorMuonH matrix costs two bytes per element for the
working copy, two for its FP8 copies under the FP8 recipe, and four for its
gradient on every rank, plus four for the master and two for the momentum
on its owner; the static footprint the `execution` record reports is
therefore this rank's. The communicator's
buffers are allocated before that footprint is measured, and the activation
budget is the smallest across ranks, so every rank replays the same plan.

Training compiles blocks and captures fixed-address graphs for reachable
pass/column shapes. Inductor caches persist at
`~/.cache/delta-feedback/torchinductor`; `DELTA_INDUCTOR_CACHE_DIR` relocates
them. Before capture the trainer measures, from eager one-pass, one-column
forwards, the activation bytes one block invocation retains and the bytes
one recomputed block releases, and plans each graph against the device memory still free
once the static footprint exists minus `--checkpoint-margin-gib` (default
2): the largest divisor of the rank's rows that fits raw, at least
`--micro-rows`, otherwise the smallest replay with as many leading PKDA and
auxiliary block invocations recomputing in backward as the shortfall needs.
The `execution` record reports the rank count and rows per rank, the static
footprint, the budget, the bytes per block, the bytes per recomputed block,
the peak bytes allocated through calibration, warm-up, and capture, and the
bytes reserved and free once capture ends; the peak minus the static
footprint and the deepest graph's retained activations is the transient the
margin covered. These measurements also appear in `memory_plan` before
warm-up, with `cached_gib`, the memory the allocator still holds beyond the
static footprint when the budget is read, `inputs_gib`, the graphs'
persistent inputs set aside before the activation budget (planned to a fixed
point, since the inputs follow the replay widths), and the bytes per
block with the recurrences keeping (`block_full_mib`) or rebuilding
(`block_mib`) their intermediates; each `plan` record precedes its graph's
warm-up and reports its rows, whether its recurrences keep (`saved=full`) or
rebuild (`saved=lean`) their intermediates, and its recomputed block count,
and `capture` identifies each graph before capture starts. The margin covers backward workspaces, checkpoint
recomputation, allocator rounding, and graph instantiation that the
retained-forward measurements do not include, and the CUDA context grows as
kernels compile after the budget is measured. On the 24 GiB card a 1 GiB
margin ran out of memory in the eager warm-up backward of the deepest screen
fl graph and 3.5 GiB ran; raise the margin if warm-up or capture runs out of
memory. The optimizer step and the periodic monitors run inside the graphs'
memory pool on the capture stream, which requires the allocator's expandable
segments: the package sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
on import, and the trainer refuses to run without them. The collectives and
the evaluation replays stay outside the pool. Keep cyclic Python garbage
collection outside train/eval capture.
CPU/MPS use eager attention, literal PKDA, and chunked tied-head loss with
the same sites and, under gloo, the same collectives.

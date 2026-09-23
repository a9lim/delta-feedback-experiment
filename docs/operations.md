# Operations

This guide covers setup, data preparation, run lifecycle, execution, and
inspection. Model mechanisms are in [architecture](architecture.md), the
training recipe in [design](design.md), and capacity/runtime planning in
[scaling](scaling.md).

Before using Jobe, inspect `delta status`, the active log, and `nvidia-smi`.
Keep GPU jobs serial and preserve active training and data builds.

## Install and check

Use the shared Python 3.13 environment and the machine's constraints:

```bash
cd /path/to/transformer-experiments
uv pip install -e . -e delta-feedback-experiment
uv pip check
cd delta-feedback-experiment
python -m pytest
```

For CUDA and tokenization, install the declared extras and editable workspace
forks; the parent repository's submodule pointers select FLA and CCE:

```bash
git -C .. submodule update --init vendor/flash-linear-attention vendor/ml-cross-entropy
uv pip install -e '../vendor/flash-linear-attention[cuda]' \
  -e ../vendor/ml-cross-entropy -e '.[cuda,data-build]'
uv pip check
delta probe
```

The default tests cover portable numerical, state, and lifecycle contracts.
`delta probe` exercises small CUDA train/eval/decode graphs, recomputation,
shared fusion, and replay under `fv`; `--ranks N` adds the step's collectives on N devices.
Neither is a production memory or throughput measurement. The queue starts
training directly. Kernel compilation and graph capture add startup time.

## Tokenize

```bash
delta tokenize --source dclm-100b --data-root /data/delta \
  --scale screen --tokens-per-param 100 \
  --scratch /data/delta/scratch/dclm-100b --workers 3 --readers 8
delta verify /data/delta/dclm-100b
```

`--data-root ROOT` writes `ROOT/SOURCE`; `--out` overrides that path.
`--scale` and `--tokens-per-param` derive a schedule-sized target, while
`--target` sets stored tokens directly. Interrupted builds resume;
`--continue` extends a finished store with matching build settings.
`--shuffle` / `--no-shuffle` override source ordering. Use full `dclm` for
stores beyond the 100B subset. See [data](design.md#data) for source identity,
row layout, and validation semantics.

Allow roughly twice the final store size while tokenized parts and the
assembled output coexist, plus source downloads and sidecars. `--scratch`
relocates downloaded parquet files, not tokenized parts. `--workers` controls
concurrent files; `RAYON_NUM_THREADS` controls tokenizer threads per worker;
`--readers` controls document reads during shuffled assembly. Published-order
builds do not need shuffled reads. Scratch downloads are removed on success.

`scripts/publish_store.sh --disk DISK` processes every document in the pinned
`dclm-100b` source with `--all` and a 100M-token holdout target (`--val 1e8`).
The holdout ends at the last whole document before that limit; training starts
with the next document. The script writes `DISK/build/dclm-100b-val100m`,
leaving earlier builds and the training store at `DISK/delta/dclm-100b` intact.
Use `--out PATH` for another destination, `--target N` for a finite token
prefix, or `--val N` for another holdout target. Rerunning resumes a partial
build or reuses/extends a completed store with matching settings.

Publication verifies the store, writes checksums and a dataset card with exact
token and document counts, and uploads to `a9lim/dclm-100b-neox`.
`--reference MANIFEST` compares held-out files and complete prefix shards only
when the accompanying `meta.json` has the same source, tokenizer, packages,
ordering and holdout target. Changing the holdout from 30M to 100M changes the
training offset, so the old split's shard-prefix checks are explicitly skipped.
`--skip-upload` keeps the result local.

Build and upload output remains untruncated, with informational upload progress
in unattended logs. The script publishes training shards in batches of eight,
then validation and document indexes, then metadata and the card. The current
`hf upload` pipeline resumes through Hub commits and Xet's content-addressed
storage; keep `HF_HOME/xet` between attempts. Files already committed with
matching sizes and content hashes are skipped. The script defaults Xet upload concurrency to
at most 8 and concurrent file ingestion to 2 using
`HF_XET_CLIENT_AC_MAX_UPLOAD_CONCURRENCY` and
`HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION`; explicit environment values
override these defaults. It retries each failed batch up to three times with
backoff, then exits with the failure code. The final publication is checked
against committed Hub file sizes and hashes. Rerun the same command
to retry without rebuilding. Other machines can download a training-shard prefix
together with the metadata, validation files, and document sidecars.

## Runs

```bash
delta train example-fv-s1 --condition fv --seed 1 --data-seed 0 \
  --data-root /data/delta --source dclm-100b
delta queue node-fv-s1 --condition fv --scale flagship --ranks 8 \
  --data-root /data/delta --source dclm-100b
delta train example-fv-s1 --resume
delta queue example-fv-s1-50x --continue example-fv-s1 --tokens-per-param 50
delta queue example-v-s1 --fork example-fv-s1 --condition v

delta status
delta watch
delta stop live --at 10k
```

The queue stores arguments and refreshes the worker from the checkout before
each job; edits do not stop an active child. `delta queue FILE` reads one
`TAG FLAGS` job per line. `--max-steps` limits additional steps in one
invocation without shortening the schedule. `delta train --help` lists the
recipe and runtime controls.

A queued job trains, then scores its final snapshot with
[`delta eval`](#downstream-evaluation) once the trainer's `done` record
closes the schedule. A stopped, interrupted, or `--max-steps` invocation
leaves a skip note in `logs/TAG.eval.log` instead. Flags after `--` (`|` in
a queue file) go to that evaluation, and `--skip` alone opts out:

```bash
delta queue example-f-s1 --condition f -- --mode standard fused
delta queue node-fv-s1 --condition fv --scale flagship --ranks 8 -- --skip
```

The evaluation is one process on one device, at most twenty minutes for
every mode at screen scale on a 4090; a multi-GPU node idles its other
devices meanwhile. Its failure marks the job `EVAL FAILED` without touching the
finished snapshots, and `delta eval TAG` repeats it.

| Command | Effect |
|---|---|
| `stop TAG` | Remove that pending job or interrupt its active child; later jobs remain |
| `stop live` | Interrupt the active job and preserve pending jobs |
| `stop queue` | Remove pending jobs and preserve the active job |
| `stop all` | Remove pending jobs and interrupt the active job |
| `stop TAG\|live --at STEP` | Arm a cooperative stop after an absolute optimizer step |
| `clear TAG\|all` | Move idle snapshots and logs into timestamped `tmp/cleared` recovery |
| `move OLD NEW` | Rename idle snapshots, logs, and standard analysis paths |

A cooperative stop sends SIGINT; all ranks finish a step, agree to stop,
snapshot, and exit. A second SIGINT aborts immediately. An armed stop signals
after the preceding step's record so the request is read inside the target
step. Multi-rank jobs receive 110 seconds to exit within the spool's
120-second grace period. `watch` exits when the queue is idle; Ctrl-C stops
watching without stopping training.

`move` requires both tags to have no active or queued references and the
destination to be free. Supply `--out-dir DIR` for a custom snapshot root.
Arbitrary custom analysis outputs are not renamed, and existing figures keep
their labels until regenerated.

## Checkpoints

Only **v42** snapshots with the current tokenizer identity are accepted.
Each contains model FP32 masters, both optimizers, arguments, step, RNG state,
expert-selection biases, and NorMuonH radius/spectral state. Rank zero
assembles one whole snapshot; transient counts and working copies are rebuilt
on load. Any supported rank count can resume it.

`--resume` loads the latest snapshot for the tag and inherits state-defining
settings; explicit conflicts are rejected, except that `--recurrence-start`
may increase while the saved step is at or before the old recurrence boundary
(`round(old_fraction * total_steps)`, the last single-column update). This
postpones recurrence without changing any completed update. Decreasing the
fraction or changing it after a recurrent update is rejected. Paths, device,
evaluation/snapshot cadence, evaluation rows, replay minimum, memory margin, and precision are
inherited unless overridden. `--ranks` always describes the new invocation
and defaults to one. Changing precision changes subsequent training numerics.

For example, stop `OLD`, wait for its checkpoint and exit, then rename and
resume with a later recurrence boundary and new reporting cadence:

```bash
delta stop OLD
# Once the run is idle:
delta move OLD NEW
delta queue NEW --resume --recurrence-start 0.6 --eval-every 500 --snapshot-every 1000
```

The resumed log records the resolved settings and schedule; later snapshots
inherit the new recurrence fraction and cadence.

Snapshots are written under `runs/TAG.pt.STEP` unless `--out-dir` is supplied.
The trainer retains the latest two plus the recurrence boundary, cooldown
boundary, and final step. Writes stage immutable state before background
atomic serialization; cooperative exits wait for the writer.

Three flags start a run from a snapshot, and each holds a different thing
fixed:

| Flag | Tag | Restores | May change |
|---|---|---|---|
| `--resume` | Same | The tag's latest snapshot | Nothing state-defining, except postponing recurrence before it begins |
| `--continue SOURCE` | New | The source's last snapshot the longer schedule reproduces | The schedule length |
| `--fork SOURCE` | New | The source's last snapshot before recurrence begins in either run | The condition and recurrence recipe |

`--continue SOURCE` starts a longer schedule under a new tag, keeping all
other state-defining settings. It restores the latest source snapshot before
the schedules diverge in recurrence or cooldown, rather than appending after
the source's final step. Short source schedules can have a different warmup;
the `continue` record's `exact` field reports whether warmup lengths match.

`--fork SOURCE` branches another condition off a run at the same length.
Before the recurrence boundary every update is one plain column in every
condition, with one data order and keyed randomness, so runs that differ only
in `--condition`, `--recurrence-start`, `--three-rate`, and
`--loop-iterations` are one trajectory up to the earlier of their boundaries.
A fork restores the source's protected boundary snapshot, or its latest one
before that step, and trains on: it ends as the target condition trained
from scratch, to the run's numerical floor, without repeating the shared
steps. Every other state-defining setting is inherited, and typing a
different one is refused, as is a fork that changes nothing. Every condition
has the same parameter set, so the snapshot restores unchanged.
`n` never rolls but still keeps its boundary snapshot, so it forks both ways. Recurrence can start no earlier
than the source's retained snapshots allow. The `fork` record names the
source snapshot and the changed settings, and the monitor draws the source's
log up to that step as the fork's own history.

## CUDA execution

`--precision fp8` is the default CUDA GEMM recipe; `bf16` disables FP8
operands. The precision boundaries are:

| State or operation | CUDA precision |
|---|---|
| Model masters, accumulated gradients, NAdam state | FP32 |
| Working weights, activations, classifier, evaluation | BF16; RMSNorm computes in FP32 and casts back |
| Projection/expert forward and input-gradient GEMMs | Rowwise FP8 under `fp8`; BF16 under `bf16` |
| Weight-gradient GEMMs | BF16 operands, FP32 accumulation |
| Tied-readout logits (Hopper) | FP32 from BF16 operands, 4,096-row chunks; BF16 logit gradient |
| PKDA recurrent boundaries | FP32 |
| NorMuonH momentum storage and Newton–Schulz iterations | BF16 |
| Momentum EMA, Nesterov direction, row moments, radii, spectral/tangent update and retraction | FP32 |

The tied readout's cross-entropy accumulates the classifier gradient
directly into the tied embedding's FP32 sink. On Hopper it is a dense
cuBLAS head (`head.py`): each chunk's FP32 logits give the row statistics
and the logit gradient in one fused pass, and because the objective's row
weights are fixed before the forward, both gradient GEMMs run in the forward
and the backward returns the saved row gradient. Nothing is filtered. Other
CUDA devices run CCE with its gradient filter, which won on Ada. Causal GQA prefers cuDNN with Flash fallback; cached prefixes use
FlexAttention. CPU/MPS use FP32 eager attention, literal PKDA, chunked
tied-head loss, and FP32 optimizer arithmetic.

### Data parallelism

`--ranks N` launches one process per device through `torchrun`; one rank uses
the same training path. `batch-rows` must be divisible by `ranks × micro-rows`.
Working weights and gradient slabs are replicated. Expert banks are padded
and split along the expert axis; dense NorMuonH sites have one owner each;
NAdam state is replicated. A step reduces FP32 gradients to owners, updates
owned matrices, gathers working weights, and refreshes local FP8 copies.
Expert counts are summed globally before selection-bias updates. CUDA uses
NCCL, other devices gloo.

### Replay planning and memory

Training compiles blocks and captures fixed-address graphs for reachable
pass/column shapes. After allocating optimizer state and communication
buffers, calibration measures retained activation bytes per block and the
bytes released by recomputation. The planner reserves persistent graph inputs
and `--checkpoint-margin-gib` (default **2 GiB**) from available device memory,
using the smallest rank budget for every rank.

For each shape it tries divisors of the rank's rows from widest to narrowest,
never below `--micro-rows`. At each width it first retains recurrence
intermediates, then tries rebuilding them. If the smallest replay still does
not fit, it recomputes leading PKDA and auxiliary block invocations during
backward; global-attention blocks stay retained. Replay width does not change
the keyed per-row draws. `--micro-rows` also sets evaluation microbatch size.

| Record | What to inspect |
|---|---|
| `memory_plan` | Rank-local static and cached memory, persistent inputs, activation budget, full/lean bytes per block, bytes released per recomputed block |
| `plan` | Shape `k:r`, replay rows, `saved=full\|lean`, recomputed/eligible blocks, retained-activation estimate |
| `capture` | Graph whose capture is starting |
| `execution` | Rank count/rows, graph counts, plan measurements, peak allocation through startup, post-capture reserved/free memory |
| Step `mem` | Peak allocated CUDA memory since startup capture ended |

The margin covers backward workspaces, recomputation, allocator rounding,
CUDA context growth, and graph instantiation beyond retained-forward
estimates. Raise it after a warm-up/capture OOM and inspect the resulting
plan. Optimizer updates and periodic diagnostics reuse the graphs' private
memory pool; collectives and evaluation replays stay outside that allocation
scope. Expandable allocator segments are required: the package sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless already configured.
Keep cyclic Python garbage collection outside train/eval capture.

Compiler caches persist at `~/.cache/delta-feedback/torchinductor`;
`DELTA_INDUCTOR_CACHE_DIR` relocates them. Keep caches consistent with the
installed runtime and forks. Resume still reconstructs graphs and may compile
new kernels. Inspect process activity, cache writes, and subsequent
capture/step records before treating a quiet startup as stalled.

## Telemetry

| Field | Meaning |
|---|---|
| `loss` | Full optimized objective |
| `ntp`, `mtp` | Main / auxiliary CE combined with the [objective's recurrence weighting](design.md#objective), before prediction weights and z-loss |
| `pass1` | Main CE of the plain first column: pass 1, column 1 |
| `k`, `r` | Passes and columns per pass; `r` appears for looped conditions |
| `val*` | Validation metrics defined by the [evaluation recipe](design.md#evaluation) |
| `gnorm` | Unclipped global L2 gradient norm |
| `expert_balance` | Unweighted mean sequence-balance loss |
| `expert_max_violation` | Worst bank's whole-update `max(load)/mean(load)-1` |
| `expert_bias_max` | Largest absolute post-update selection bias; each bank's biases are zero-mean |
| `tok_s`, `pass_tok_s`, `cell_tok_s` | Predicted-token, pass-token, and cell-token rates; their counts exclude MTP |

The [Jobe monitor](https://runs.a9l.im/delta-feedback/) and
[rental monitor](https://rentalruns.a9l.im/delta-feedback/) use separate
Cloudflare Access tunnels. Rental provisioning is owned by
`~/Work/meta/bootstrap/rental.sh --only monitor` and its `bootstrap/MANUAL.md`.
The overview shows progress, pace, token totals, and the host's GPU reading;
configuration and non-step logs expand on demand. Exact token totals require
complete addressed step history, including inherited fork steps.

**Log x** gives training-step charts a logarithmic x axis; step zero is omitted
while it is enabled. **Log y** independently controls positive-valued y axes.
Layer-site axes, masses, counts, and signed differences retain their linear
scales where appropriate.

**Scale steps** changes training-step x coordinates to predicted tokens per
reference active non-embedding parameter, using the same denominator as
`--tokens-per-param`. Equal token budgets line up across model sizes; a
100 tok/param run spans four times a 25 tok/param run on the linear axis
(up to whole-step rounding). Batch size and sequence length enter the
conversion; feedback passes and loop visits do not. Schedule bands, recurrence
and fork boundaries, and checkpoint markers use the same conversion. Log x
can be enabled independently. Layer profiles and eval selectors retain their
recorded step addresses.

**Downstream** reads saved `figures/downstream-TAG/*.json` results and
`figures/baseline/**/*.json` reference scores through the monitor's
`api/results` endpoint. It updates when each mode's JSON is saved; an eval log
is unnecessary, smoke runs write no results, and a fork does not inherit its
parent's scores. Each row is one run's evaluation mode and feedback pass count,
or one saved baseline. **Compare runs** adds selected runs with a separate
checkpoint choice for each. By default, each run's modes sit together under
one shared run name. Mode labels use × for feedback passes. Baselines show
only the model name; hover reveals the full model ID. Run names expose the
full tag and checkpoint on hover, including when a narrow column truncates
the label. Model and mode columns stay pinned during horizontal scrolling.

Benchmarks group their reported metrics into columns: **acc % / norm %** for
HellaSwag, **acc %** for BoolQ, and **acc % / ppl** for LAMBADA. Percent units
appear only in headers; perplexity remains per word. Unreported scores show a
dash. Document counts sit beneath each benchmark name. If counts differ
across runs, modes, or baselines, the header shows **n varies** and a tooltip
identifies each result's count.

Click a metric header to cycle through **best first → worst first → default**.
Accuracy sorts highest first; perplexity sorts lowest first. Sorting ranks
each mode independently and repeats run names so every row stays identifiable.
It uses full precision, preserves default order for ties, and keeps missing
scores last in either direction. The active sort survives result refreshes
and checkpoint changes; choosing another metric starts with best first.
The table shows saved scores without cross-run deltas; paired comparisons
remain available through the evaluation CLI. Replacing or deleting a result
file replaces or removes its scores on refresh. An unreadable file is reported
and retried without retaining stale scores. The server caches summaries of
unchanged files and omits per-document records from browser responses.

The **Eval step** slider synchronizes layer profiles and expert heatmaps;
arrows select recorded evaluations, and **Follow latest** resumes tracking.
A selection stays pinned through refresh. Switching runs selects the latest
evaluation; a resume that removes the selection clamps it to a surviving
earlier evaluation. Overlays use the exact selected step, leaving missing
data blank. Heatmaps show the selected run with stable color ranges; hover
or keyboard focus exposes assignment share, coverage, entropy, and bias.

Routing profiles show null/seed/previous-cell mass, maximum weight against
`1/n`, head divergence, and learned-null RMS. They sample up to two validation
rows from the final column of the fused pass when feedback is enabled. Expert
profiles use the first column of pass 1 and the teacher-forced MTP block.
Both disable jitter. These are sample diagnostics, not whole-corpus loads.
The Looped column section appears for `v` and `fv`, including
when a looped run is an overlay. It shows per-column depth readouts and
executed recurrence; training feedback gain
appears only on single-column steps, where the combined CE isolates it.

For monitor changes, run
`node --test tests/monitor.test.cjs ../tests/test_monitor_chassis.mjs` from this
checkout and check the sliders, overlays, eval comparison, axis controls,
refresh, heatmaps, and responsive layout in the browser.

## Hopper and node workflow

On a provisioned Hopper host, inventory CUDA-visible HBM, runtime/fork
versions, and device topology; product names can include memory outside GPU
HBM. Kernel choices are automatic:

| Kernel | Hopper choice |
|---|---|
| Routed expert weight gradients | `(BM, BN, BK, warps, stages) = (64, 128, 128, 4, 3)` |
| Classifier head | Dense cuBLAS logits instead of CCE, about twice as fast |
| PKDA ATK inter-chunk scan at head width 128 | `BK=128`, four warps |
| PKDA WY backward | Autotuned over two to eight warps |
| MHDB routing backward | Half the forward's warps |
| Fused Q/K/V convolution | 32-row tiles, autotuned per tile in both directions |
| PKDA ATK reverse chunk scan | Elementwise carry over 32-wide state slices; its gate gradient runs as its own chunk-parallel pass |

For a single GPU, run the lifecycle script with a fresh tag:

```bash
scripts/first_hour.sh --data-root /data/delta --tag first-hour \
  --scale screen --condition fv --precision fp8 --steps 12
```

It inventories the host, runs the CUDA probe, trains with recurrence from
step zero and frequent evaluation/snapshots, resumes, requests a cooperative
stop, and reads the final checkpoint before writing success. Results go to
`logs/first-hour/TAG/`, including per-shape timing summaries; snapshots use
`runs/`. This checks lifecycle behavior on the selected geometry.

For a multi-GPU node:

1. Confirm distinct devices, usable memory, and NVSwitch/peer topology; run
   `delta probe --ranks N`.
2. Measure reductions and gathers on production slabs; check gradient sums,
   gathered weights, and expert-bias agreement across ranks.
3. Capture the deepest screen and target-scale graphs; inspect every rank's
   plan, peak allocation, and reserved/free memory.
4. Exercise evaluation, snapshots, cooperative stop, and resume across the
   intended rank counts.
5. Measure complete warm steps across schedule shapes, including startup and
   monitoring when estimating total run time. See [scaling](scaling.md).

Shared caches reduce compilation work but do not eliminate per-process graph
reconstruction. Choose further sharding or communication overlap from node
measurements. Generated data, logs, traces, snapshots, and figures stay
untracked.

## Benchmarks

`python scripts/replay_bench.py --data-root /data/delta` times the production
trainer's captured graphs. `--replay-rows N` fixes actual width;
`--condition f --specs 1:1` isolates a single-column trial. `--trace` groups
kernel time, while `--train-steps N` adds full-batch optimizer updates for
one selected graph. The latter does not exercise evaluation or checkpoints.
Use `--attention-backend`, `--expert-tiles`, and `--cce-config` (off Hopper)
for process-local comparisons.

| Script | Measurement |
|---|---|
| `attention_bench.py` | Attention backend, outputs/gradients, graph capture |
| `pkda_bench.py` | PKDA head counts, recurrence saving modes, input gradients; `fix:`/`set:` candidates pin any kernel's launch or a wrapper's tile |
| `moe_bench.py` | Balanced/skewed routing and expert GEMM tiles |
| `cce_bench.py` | CCE configurations off Hopper; use twice the replay rows for paired NTP/MTP |
| `route_bench.py` | MHDB routing forward/backward over the screen site profile |
| `dense_fp8_bench.py` | Dense FP8 scaling, quantization, weight refresh |

All are under `scripts/`; use `--help` for dimensions and output controls.
Component timing and numerical agreement bound execution choices; assess
learning behavior with the training/evaluation recipe.

## Inspect a checkpoint

```bash
python scripts/route_report.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b
python scripts/payload_swap.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b
python scripts/depth_trace.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b
python scripts/training_curves.py logs/A.log logs/B.log --out-dir figures/curves-A-vs-B
python scripts/scaling_curves.py logs/A.log logs/B.log logs/C.log
python scripts/optimizer_state.py runs/TAG.pt.A runs/TAG.pt.B --log logs/TAG.log
```

Checkpoint tools use `delta_feedback_experiment.analysis` and trainer
evaluation numerics. Figures are regenerable; plotting uses Matplotlib from
the shared analysis environment. The [analysis guide](interpretability.md)
defines each tool's measurement.

### Downstream evaluation

```bash
delta eval TAG                          # latest snapshot, every eligible mode
delta eval TAG --step 5485 --mode standard
delta eval TAG --mode fused --passes 2
delta eval TAG --tasks piqa --limit 32  # smoke; writes nothing
delta eval TAG --baseline EleutherAI/pythia-410m HuggingFaceTB/SmolLM2-360M
python -m transformer_experiments.downstream --compare \
  figures/downstream-TAG/standard.STEP.json figures/downstream-TAG/fused.STEP.json
```

`delta eval` scores the workspace's pinned zero-shot tasks against one loaded
model: Standard for every condition, plus Fused and Soft for conditions with
`f`. Each task emits a `downstream` record (`delta watch` shows them), and
each mode writes `figures/downstream-TAG/MODE.STEP.json`, or
`MODEk.STEP.json` for `--passes k`; a `--tasks` subset updates its tasks
there and keeps the rest. `--out-dir` names a custom snapshot root.
The first use downloads the task datasets from the Hub. The queue runs the
same command [after a finished schedule](#runs).

`--baseline MODEL` compares each mode against a published Hub model on the
same documents. The model is scored once in FP32, a few minutes below 1B
parameters, into `figures/baseline/ORG/NAME.json`, which belongs to no run
and survives `clear`. Later evaluations reuse it without loading the model
or reaching the Hub, and score only tasks it lacks; a model whose Hub commit
has moved is rescored. The snapshot's own results are written first. Each
mode prints a paired table and a `downstream` record with `against=MODEL` and
the pooled accuracy difference, snapshot minus baseline.

Reference tokenization preserves a model's native beginning-of-sequence token
(including Gemma's tokenizer post-processor) without appending an end token.

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

Role names and repeated speakers are preserved; the default next role is
`self`. `<|im_end|>` ends a message and `<|endoftext|>` ends a pretraining
document. Web pretraining does not apply this chat template.

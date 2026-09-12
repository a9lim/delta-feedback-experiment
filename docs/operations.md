# Operations

Commands run from this repository unless shown otherwise. Before using Jobe,
inspect `delta status`, the active log, and `nvidia-smi`. Keep GPU jobs serial
and preserve active training and data builds.

## Install

Use the shared Python 3.13 environment. Install the workspace package, then
this project:

```bash
cd /path/to/transformer-experiments
uv pip install -e .
cd delta-feedback-experiment
uv pip install -e .
```

On a CUDA host, initialize and install the workspace's kernel forks:

```bash
git -C .. submodule update --init vendor/flash-linear-attention vendor/ml-cross-entropy
uv pip install -e '.[cuda]'
```

The `tool.uv.sources` entries resolve editable FLA and CCE under `../vendor/`.
Machine-wide constraints own Jobe's PyTorch 2.14 / CUDA 13.2 installation.
For a host that builds token stores, also install:

```bash
uv pip install -e '.[data-build]'
```

Token stores record the build package versions. Resuming or extending a store
requires those versions; reading it for training does not. After upgrading
build packages, use a new store or restore its build environment.

## CUDA execution

CPU/MPS use eager PyTorch attention, literal PKDA recurrence, algebraic routing,
and chunked tied-head loss. CUDA uses BF16 activations with FP32 parameters,
accumulated gradients, optimizer state, and PKDA recurrent boundaries:

- FLA chunk kernels run PKDA training/prefill; recurrent kernels run decode.
- Native Flash SDPA runs causal BF16/FP16 attention; FP32 diagnostics use math.
  Cached decode uses FlexAttention over the valid prefix.
- Triton MHDB routers bank source gradients until the corresponding source's
  backward runs. Packed projection gradients accumulate into FP32 sinks.
- CCE reads an address-stable BF16 classifier shadow. Its BF16 gradient buffer
  flushes into the FP32 embedding sink at `--head-flush-every` head calls and
  before the optimizer update. Each pass makes two head calls; auxiliary calls
  count toward the flush cadence. A captured
  microbatch that would exceed the limit uses the FP32 sink per call;
  `--head-flush-every 1` gives per-call precision in every mode.
- Compiled blocks sit inside fixed-address CUDA graphs, one per reachable
  `(pass count, core iteration count)`, plus no-grad evaluation graphs.
  All iterations and feedback passes remain differentiable. Above the raw
  activation budget, each cell retains its final block's activations and
  checkpoints the preceding blocks outside compilation.
- NorMuonH compiles shape-bucket updates; NAdam and global FP32 clipping run
  outside the forward/backward graphs. Snapshot staging uses pinned host
  memory and atomic background writes.

On CPU/MPS, `delta probe` runs the reduced portable test suite. On CUDA it
runs one tiny captured `fl` train/eval/decode smoke directly, without pytest.
The portable suite covers focused model, training, and lifecycle invariants;
neither path sweeps full geometry or condition matrices. Queue startup runs
neither tests nor a probe. Run the probe explicitly after changing execution
code or hardware. The CUDA smoke skips Inductor compilation and uses eager
cached attention; it checks kernel execution and graph replay, not compiler
output or production memory fit. Exact cache parity is tested on CPU; CUDA
decode checks shapes, position, and finite output. First-use kernel JIT can still add startup
time. Training's Inductor artifacts persist
at `~/.cache/delta-feedback/torchinductor`; `DELTA_INDUCTOR_CACHE_DIR` relocates
that cache. Keep cyclic Python garbage collection outside graph capture.

## Tokenize

The source and ordering contract is in [design.md](design.md#data).
The screen at 100 tokens per parameter needs a 21B-token store:

```bash
delta tokenize --source dclm-100b --data-root /data/delta \
  --scale screen --tokens-per-param 100 \
  --scratch /data/delta/scratch/dclm-100b --workers 3 --readers 8
delta verify /data/delta/dclm-100b
```

`--data-root ROOT` writes `ROOT/SOURCE`; `--out` overrides the output path.
`--continue` extends a completed store with matching settings. `--shuffle`
and `--no-shuffle` override the source's ordering default. Full `dclm` supplies
stores larger than the publisher's 100B subset.

`--workers` controls file encoders; each prefetches one upcoming source file.
`RAYON_NUM_THREADS` controls tokenizer threads per process. `--readers`
controls parallel document assembly. Budget storage for both token parts and
the final stream plus sidecars; `--scratch` moves downloads only. Completed
parts and source indexes allow interrupted builds to resume.

## Train and control runs

`TAG` and `STEP` below are placeholders. The scale presets and derived budgets
are in [scaling.md](scaling.md); recipe flags are in
[design.md](design.md#knobs) and `delta train --help`.

Choose `--condition f` (default), `l`, or `fl`. All runs include PKDA,
MHDB routing, experts, and auxiliary two-token prediction. The presets use
sixteen layers and expert intermediate width 832.
`--expert-intermediate`, `--num-routed-experts`, and `--experts-per-token`
override expert width, routed bank size, and selected count for both trunk and
MTP; the selected count must be between one and the routed bank size. The finite
nonnegative `--mtp-weight` defaults to 0.3 and is restored on resume. MTP uses
the payload and next-token embedding from existing token-store rows in its
own PKDA/expert block. Every supervised pass constructs the payload, including
without `f`; `f` controls feedback consumption. Ordinary generation does not
execute MTP.

```bash
delta probe
# Foreground training or detached queueing.
delta train example-f-s1 --condition f --seed 1 --data-seed 0 \
  --data-root /data/delta --source dclm-100b
delta queue example-fl-s1 --condition fl --seed 1 --data-seed 0 \
  --data-root /data/delta --source dclm-100b
# Resume this tag, or extend a completed run under a new tag.
delta train example-f-s1 --resume
delta queue example-f-s1-50x --continue example-f-s1 --tokens-per-param 50

delta status
delta watch
delta stop TAG --at STEP
delta stop live
delta stop queue
delta stop all
delta clear TAG
delta clear all
delta move old-tag new-tag
```

The queue stores arguments and refreshes the checkout before starting each
new training job. It does not run tests or a probe as preflight. `stop live`
stops the active run and keeps pending jobs;
`stop queue` removes pending jobs and keeps the active run. A stop sends SIGINT
so the trainer snapshots its completed step; a child still running after
120 seconds is killed. The current checkpoint resumes through `--resume`.
`--max-steps` caps an invocation without changing its schedule.

`delta move OLD NEW` renames idle snapshots, training/probe logs, standard
analysis directories, and comparison directories. It updates text records,
continuation links, and monitor status. The canonical `TAG.pt.STEP` filename
supplies the current tag while checkpoint bytes stay intact. Binary figures
retain their labels until regenerated. Use `--out-dir DIR` for snapshots
outside `runs/`; custom output paths are not renamed. Both tags must be idle,
with no queued references and no destination collision. Ordinary I/O failures
roll back; a multi-file rename is not crash-atomic.

Current snapshots use checkpoint v33 and the pinned tokenizer identity.
Resume inherits state-defining settings and rejects explicit conflicts;
runtime paths and evaluation/snapshot cadence may change. Latest snapshots
and the protected feedback, cooldown, and final boundaries support resume and
continuation. See [design.md](design.md#checkpoints-and-queue).

Training logs `ntp`, the combined ordinary cross-entropy before z-loss, and
`mtp`, the combined auxiliary cross-entropy before its weight and z-loss. `loss` reports the full optimized objective. `pass1`,
`val`, and `val_fused` keep their ordinary next-token meaning; auxiliary
validation is `val_mtp` and, with feedback, `val_mtp_fused`. On feedback
steps, use `ntp - pass1` for mean feedback next-token cross-entropy.

Training logs `expert_balance`, the unweighted per-sequence auxiliary loss
averaged over executed trunk invocations plus one MTP invocation per pass,
then over passes and the update. Its coefficient is `1e-4`, independent of
`--mtp-weight`; the monitor plots it separately from cross-entropy.
Expert selection biases, including the auxiliary bank's, update once after the
optimizer step at rate `0.001`, using the whole update's assignment counts,
and remain fixed during evaluation. `expert_max_violation` is the largest
`max(load) / mean(load) - 1` across physical banks for that update, and
`expert_bias_max` is the largest absolute bias after its update. Evaluation
also records expert assignment fractions, current biases, and selected-gate
entropy by trunk layer invocation and at `mtp.experts` on up to two validation
rows; those summaries describe that sample,
not the full training distribution.

## Format a conversation

The shared tokenizer entry point installs the pinned vocabulary, ChatML
delimiters, and template together:

```python
from delta_feedback_experiment.tokenizer import load_tokenizer

tokenizer = load_tokenizer()
token_ids = tokenizer.apply_chat_template(
    [
        {"role": "researcher", "content": "State the hypothesis."},
        {"role": "critic", "content": "Identify a counterexample."},
        {"role": "critic", "content": "Then check the boundary case."},
    ],
    add_generation_prompt=True,
    next_role="researcher",
    return_dict=False,
)
```

Roles stay as supplied, including repeated names. Omitting `next_role` opens
a `self` turn. `<|im_end|>` ends a message; `<|endoftext|>` ends a pretraining
document. This loads tokenizer files only. Web-text pretraining does not apply
ChatML.

## Inspect a checkpoint

Every analysis script rebuilds the condition from a snapshot through
`delta_feedback_experiment.analysis`, evaluates under the trainer's numerics
(BF16 autocast on CUDA), and writes a JSON record beside its figures under
`figures/<kind>-<tag>/`. Those generated directories stay ignored and can be
regenerated from current snapshots or logs.

```bash
# Routing: per-site/group source mass, entropy, query geometry (any condition).
python scripts/route_report.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b

# Payload enrichment swaps (f): trained router, top-only, uniform, forced source.
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
The shared analysis environment supplies Matplotlib.

`delta watch` shows training health, validation, and recurrence diagnostics.

# delta-feedback-experiment

We are developing a small recurrent **model organism for interpretability**: a
miniature language model whose latent computation we can analyze, inspect, and
intervene on. The goal is to better understand and eventually monitor models
that reason through internal states whose computation is opaque to us.
Capabilities matter insofar as the organism needs meaningful behavior to study.
Advancing the capabilities frontier is not this research's objective.

The architecture synthesizes already-published work on latent feedback,
recurrent memory, and depth routing. We are assembling an experimental object,
not claiming a new frontier architecture or reproducing any one paper's
results. The intended contribution is a reproducible organism, causal
explanations of its behavior, and monitoring methods tested against those
explanations.

## What we want to understand

- What information survives in recurrent state, and how is it transformed?
- When does later behavior depend on the payload, mixer memory, or depth routes?
- Does repeated computation preserve, refine, overwrite, or destabilize state?
- Can a monitor detect a specific internal change before it appears in outputs,
  and does that detection survive controlled interventions and held-out tasks?

Readable activations and routing plots are starting points. Understanding
requires interventions that predictably change behavior. A monitor requires an
independently defined target and measured errors. Neither lower loss nor a
decaying state-update norm demonstrates interpretable reasoning.

## The organism

The implemented model is a 230–258M-parameter decoder family that can be
trained and examined on one RTX 4090. “Small” is relative to frontier models:
this is still a substantial pretraining workload, and smaller diagnostic
geometries are useful for implementation checks.

```text
previous token's latent payload + current token embedding
                         │
                 FBT token-gated entry
                         │
        [PKDA, PKDA, PKDA, gated global GQA] × 3
           MHDB reads seed and block deltas at each site
                         │
                 top state + routed sources
                         │
                   next latent payload
```

PKDA supplies recurrent token-mixer memory; GQA supplies periodic full-prefix
reads. FBT carries a latent payload across token columns. MHDB makes the seed
and block contributions addressable inside a column and in the outgoing `df`
payload. These are concrete places to observe and perturb computation; their
presence does not make their learned contents interpretable.

The five arms provide controls for studying those mechanisms:

| Arm | Role |
|---|---|
| `base` | Common PKDA/GQA trunk, with neither MHDB nor FBT |
| `mhdb` | Depth routing without latent feedback |
| `fbt` | Latent feedback without depth routing |
| `df` | Both, including routed payload enrichment |
| `vanilla` | Separate pure-GQA whole-trunk control |

`base` already has PKDA recurrence. The factorial isolates the MHDB and FBT
packages; it does not compare all recurrence against none. The proposed
[`df-loop`](docs/depth-architecture.md) adds a tied core iterated within each
column. It is specified but unimplemented and is not a sixth registered arm.

## Current status

The repository has a single `DFModel`, deterministic single-process training,
portable semantics, a qualified Jobe CUDA path, route reports, payload-source
interventions, and recurrent stability diagnostics. No completed comparison
under the current screen contract or accepted scientific finding is recorded.
Existing diagnostic checkpoints can support explicitly scoped exploratory
analysis; they do not constitute the registered factorial.

The next work is to build and characterize reproducible specimens, establish
behavior that causally uses recurrent state, and develop the analysis and
monitoring protocol. There is no registered reasoning benchmark or validated
hidden-state monitor yet. The model has not been shown to be a faithful proxy
for opaque reasoning in larger systems.

The primary training recipe is a two-seed, five-arm Jobe screen. Longer
training and larger geometries are [optional references](docs/scaling.md),
gated on a specific interpretability need, implementation qualification, and
spend approval. A useful organism at the current size is a successful endpoint.

## Start here

Read the [research design](docs/design.md) and [interpretability
program](docs/interpretability.md) for the questions, controls, deliverables,
and evidence rules. The [architecture](docs/architecture.md) defines the exact
state and computation; the [literature map](docs/literature.md) explains the
source roles and synthesis.

For installation, data materialization, training, queue control, and analysis
commands, use [operations](docs/operations.md). Run `df probe` before training.
Jobe's qualified graph pool reserves about 23 GiB, so GPU work stays serial;
[runtime qualification](docs/runtime-qualification.md) records its evidence.

[Findings](docs/findings.md) contains accepted scientific results,
[figures](figures/README.md) indexes their figures, and [the
journal](docs/journal.md) holds disposable working notes.

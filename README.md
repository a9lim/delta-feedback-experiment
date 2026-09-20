# delta-feedback-experiment

A recurrent language model for interpretability experiments. The full `fl`
model combines cross-token latent feedback with repeated execution of a
shared column. Each column contains four `[PKDA, PKDA, PKDA, NoPE-GGQA]`
cells, shared and routed SwiGLU experts, and Multi-Head Delta Block routing
over the seed and cell deltas. A shared fusion projection connects token
embeddings, payloads, and an auxiliary second-token predictor.

The conditions select feedback (`f`), looped depth re-entering with a blank
embedding (`l`) or with the token again (`v`), or feedback with either (`fl`,
`fv`). They share parameters, paired initialization, and keyed training draws.
Four presets span 621M–5.32B stored parameters; the CLI defaults to screen
scale and `f`. Ordinary generation uses the main next-token head.

Start with [installation and run commands](docs/operations.md). Training
uses Python 3.13 and the parent `transformer-experiments` workspace;
general capture/readout tooling lives in `interpretability-experiments`.

| Document | Owns |
|---|---|
| [Architecture](docs/architecture.md) | The full `fl` computation, component equations, state, initialization, and optimizer mechanics |
| [Training recipe](docs/design.md) | Conditions, data identity, objectives, randomness, schedules, and evaluation |
| [Scaling](docs/scaling.md) | Presets, parameter/token/compute accounting, decode memory, and runtime estimates |
| [Operations](docs/operations.md) | Installation, tokenization, run/checkpoint control, telemetry, CUDA, and Hopper/node workflows |
| [Interpretability](docs/interpretability.md) | Analysis APIs, tools, interventions, and measurement limits |
| [Mechanism sources](references/refs.yaml) | References for implemented components |

Documentation describes the current implementation. Generated data,
checkpoints, logs, figures, and fetched papers remain untracked; Git retains
project history.

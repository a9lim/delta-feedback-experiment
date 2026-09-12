# delta-feedback-experiment

A recurrent language model with a latent channel between token columns,
grown for interpretability and monitoring experiments.

The model uses four-layer `[PKDA, PKDA, PKDA, NoPE-GGQA]` cells, Multi-Head
Delta Block (MHDB) routing over the seed and cell deltas, and shared plus
selected routed SwiGLU experts. An auxiliary PKDA/expert block trains each
column's payload to predict a second token.

| Condition | Computation |
|---|---|
| `f` (default) | Latent feedback between columns |
| `l` | Tied middle cells repeated within a column |
| `fl` | Both |

All presets have sixteen unique layers and expert intermediate width 832.
Screen, bridge, flagship, and extension use residual widths 768, 1,152,
1,536, and 2,304. Shared parameters initialize identically for a seed;
data and recurrent recipe draws use keyed streams.

The pinned GPT-NeoX/ChatML tokenizer supports arbitrary and repeated roles.
Pretraining uses raw web documents. Ordinary generation uses the main
next-token head; MTP is auxiliary supervision.

## Use

Install the workspace package, then this project in the shared Python 3.13
environment. CUDA and tokenization extras are described in
[operations](docs/operations.md).

```bash
uv pip install -e .
python -m pytest
delta probe  # CUDA smoke; run when the GPU is idle
delta train example-f-s1 --condition f --seed 1 --data-seed 0 \
  --data-root /data/delta --source dclm-100b
delta status
delta watch
```

- [Architecture](docs/architecture.md): equations, state, precision, and optimizers.
- [Recipe](docs/design.md): data, schedules, snapshots, and evaluation.
- [Scaling](docs/scaling.md): geometry, parameter/token counts, and decode state.
- [Operations](docs/operations.md): installation, tokenization, and run control.
- [Analysis](docs/interpretability.md): checkpoint tools and measurement boundaries.
- [Sources](references/refs.yaml): mechanisms used by the model.

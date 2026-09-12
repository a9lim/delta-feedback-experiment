# delta-feedback-experiment

A small recurrent language model with a latent channel between token columns,
grown as a model organism for interpretability and monitoring. This repository
builds the model, trains variants, and provides tools to inspect and intervene
on their computation.

## Model

The core model always uses:

- four-layer cells `[PKDA, PKDA, PKDA, NoPE-GGQA]`;
- Multi-Head Delta Block (MHDB) reads over the seed and block deltas;
- one shared plus a scale-dependent selection of routed SwiGLU experts per layer;
- a routed payload trained by an auxiliary PKDA/expert block for second-token
  prediction.

Two condition letters control its recurrent computation:

| Condition | Computation |
|---|---|
| `f` (default) | FBT latent feedback between token columns |
| `l` | Repeated application of the tied core between the first and last cells |
| `fl` | Both feedback and tied depth |

A condition must include at least one letter. Shared parameters initialize
identically for a given seed, and data and feedback draws use keyed streams.

All three scales have sixteen layers in a one-cell prelude, two-cell core,
and one-cell coda. Residual widths are 768, 1,152, and 1,536; every expert has
intermediate width 832. Screen selects three of fifteen routed experts, bridge
five of twenty-three, and flagship seven of thirty-one, alongside one shared
expert. All experts occupy parameter and optimizer memory.
The auxiliary prediction block combines the payload with the next token's
embedding and trains on existing rows; `--mtp-weight` defaults to 0.3. It
shares the embedding/readout and uses its own PKDA recurrence and expert bank.
Every condition produces payloads for this supervision; `f` also feeds them
into later columns. Ordinary generation uses the main column.
[Scaling](docs/scaling.md) gives parameter and token accounting.

The tokenizer is pinned GPT-NeoX with two generic ChatML delimiters, 50,279
token IDs, and a 50,304-row tied embedding/readout. It formats arbitrary and
repeated roles; pretraining uses raw web text.

## Current state

There are no trained specimens under the current tokenizer and checkpoint
contract, and no measurements of payload usefulness, token recoverability, or
learned recurrent dynamics. Use `delta status` and the active log for live
run and data-build progress.

The scientific target is a channel that carries behaviorally useful
information beyond the visible tokens and mixer caches. Payload interventions,
readout comparisons, and recurrence traces test that target. Training and
engineering checks alone do not establish it.

## Use

```bash
uv pip install -e .
delta probe
delta train example-f-s1 --condition f --seed 1 --data-seed 0 \
  --data-root /data/delta --source dclm-100b
```

Install the workspace package first; CUDA and token-store builds need the
extras described in [operations.md](docs/operations.md). Keep GPU work serial
on Jobe and inspect active work before starting a job.

- [Architecture](docs/architecture.md): equations, state, initialization, precision, and optimizer ownership.
- [Recipe](docs/design.md): condition pairing, data, schedule, and evaluation modes.
- [Scale presets](docs/scaling.md): geometry, parameters, token budgets, and decode-state accounting.
- [Operations](docs/operations.md): installation, tokenization, training, queue control, and checkpoint analysis.
- [Interpretability](docs/interpretability.md): existing measurements and how to interpret them.
- [Sources](references/refs.yaml): papers and implementations used by the current model.

# delta-feedback-experiment

A small recurrent language model with a latent channel between token columns,
grown as a model organism for interpretability and monitoring. This repository
builds the model, trains variants, and provides tools to inspect and intervene
on their computation.

## Model

One `DeltaModel` implements every subset of six condition letters:

| Letter | Change |
|---|---|
| `a` | Preconditioned Kimi Delta Attention (PKDA) in three layers of each four-layer cell; the fourth is gated global GQA |
| `e` | Quarter-width SwiGLU experts: one always shared plus three selected from fifteen routed experts in every layer |
| `r` | Multi-Head Delta Block (MHDB) routing over the column seed and block deltas before every sublayer |
| `f` | Full-Bandwidth Transformer (FBT) fusion of the previous column's latent payload with the current token embedding |
| `l` | Repeated application of the tied core between the first and last cells |
| `m` | One auxiliary transformer block predicts a second token from the top state and the next token's shared embedding |

`aerflm` is the full stack, `aerfm` its flat column, and the empty condition the
plain gated GQA decoder. Without `a`, the first three layers of each cell use
full-head RoPE after Q/K RMSNorm; every fourth layer uses no positional
encoding. Conditions on the same attention trunk pair their shared parameters
at initialization and can train on identical rows and keyed feedback draws.

The default screen has width 768, twelve layers, and 455,637,288 parameters
under `aerf` or `aerfl` (179,461,416 without `e`). Four quarter-width experts
are active per token; their matrix arithmetic matches the dense FFN before
routing overhead. Token budgets retain the dense `arf` reference. The model
uses a pinned GPT-NeoX tokenizer with two generic ChatML delimiters, 50,279
token IDs, and a 50,304-row tied embedding/readout.
The tokenizer formats arbitrary and repeated roles; pretraining uses raw web
text.

Expert selection uses sigmoid scores with a separate load-balancing bias
updated after each optimizer step. A small per-sequence regularizer supplements
that update; reported cross-entropy excludes it.

With `m`, DeepSeek-style two-token prediction adds a dense causal RoPE-GGQA
and SwiGLU block after each pass's top state. It shares the embedding and
readout, and its loss trains the main model as well as the auxiliary block.
`--mtp-weight` defaults to 0.3. Ordinary next-token validation remains
separate from MTP validation; generation uses the main model alone.

## Current state

There are no trained specimens under the current tokenizer and checkpoint
contract, and no measurements of payload usefulness, token recoverability, or
learned recurrent dynamics. Jobe is building the 15B-token DCLM-100B store.
Use `delta status` and the active log for live run and build progress.

The scientific target is a channel that carries behaviorally useful
information beyond the visible tokens and mixer caches. Payload interventions,
readout comparisons, and recurrence traces test that target. Training and
engineering checks alone do not establish it.

## Use

```bash
uv pip install -e .
delta probe
delta train example-arf-s1 --condition arf --seed 1 --data-seed 0 \
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

# delta-feedback-experiment

This is a nursery for a small recurrent language model with a latent channel
between token columns. The eventual goal is a **model organism**: a model of
roughly 250M parameters whose cross-token latent computation is real enough,
and opaque enough, to be worth studying with interpretability and monitoring
methods. That study is future work with its own design. This repository is
where the organism gets grown: train variants, look inside, change the recipe
or the architecture, repeat.

Nothing here is pre-registered. Runs are experiments in the ordinary sense of
trying things. [The journal](docs/journal.md) records what we saw as we went,
[findings](docs/findings.md) keeps the distilled current picture, and the
rest of the docs describe what the code does today.

## What we are growing

The model is a decoder with three kinds of recurrence available to it:

- **Token-mixer memory.** Preconditioned Kimi Delta Attention (PKDA) layers
  carry a recurrent matrix state along the token axis; every fourth layer is a
  gated global GQA read over the whole prefix.
- **Latent feedback across columns.** The Full-Bandwidth Transformer (FBT)
  entry fuses the previous token column's payload with the current token
  embedding, so a column can receive computation the previous column finished
  after it emitted its token.
- **Depth routing.** Multi-Head Delta Block routing (MHDB) lets every sublayer
  read the column seed and the completed four-layer block deltas, and lets the
  outgoing payload be a routed mixture of them instead of just the top state.

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

One `DeltaModel` implements the whole family. A condition is a string of
letters, each one change from the plain twelve-layer RoPE GQA decoder, so any
two conditions can be trained on identical rows from paired initializations
and compared:

| Letter | Change |
|---|---|
| `a` | Kimi Delta Attention: the `[PKDA, PKDA, PKDA, gated global GQA]` trunk replaces RoPE GQA |
| `r` | MHDB residual reads of the seed and block deltas before every sublayer; with `f`, a routed payload |
| `f` | Full-bandwidth feedback: the FBT entry and a payload for the next column |
| `l` | Huginn loop: a tied-depth core, [specified](docs/depth-architecture.md) and not built |

`--condition arf` is the full built stack; `ar` and `af` each drop one
package; `a` is the bare hybrid trunk; the empty condition is the plain
decoder. The [architecture page](docs/architecture.md) has the exact
equations, geometry, and optimizer.

## What "ready" looks like

The organism is ready for a real experiment when its latent channel has
properties worth studying rather than properties we would have to assume:

- The payload carries something the token stream and the mixer caches do not,
  so later behavior changes when it is perturbed.
- That content is not simply a re-encoding of the emitted token or the readout
  state. A channel the tokenizer could reconstruct is not opaque.
- Repeated application of the feedback map settles instead of drifting or
  blowing up, so trajectories can be traced.
- Some behavior, ideally an externally scored task, depends on the channel.

These are growth targets. Measuring them is the job of the analysis scripts;
hitting them is the job of the recipe and the architecture.

## Where things stand

Three full-schedule specimens exist on Jobe, all seed 1 on the same rows:
an `ar` run on the default recipe, and two `arf` runs trained with three
feedback passes on every step from step 0, at two NorMuonH learning rates.

The current read, from [findings](docs/findings.md): the channel as trained is
live, stable, and harmless, and it is token-legible. The first cell decodes
the fused seed back onto the plain representation and the rest of the column
runs as it would on a plain pass; the payload is linearly the readout state
and the token is recoverable from the fused seed at 98%. A second fused
prefill pass buys a small gain on long-range continuation and nothing on
FineWeb tokens. That is the opposite of an opaque channel, which tells us what
to grow next.

Directions on the bench, none decided:

- Redesign the entry so the previous column's state lands somewhere the
  current column cannot simply cancel, for example at a core entry rather than
  the seed.
- Build the `l` letter, the tied core in
  [depth-architecture.md](docs/depth-architecture.md), so computation can
  refine within a column and persist across columns through two trained
  channels.
- Separate a persistent state stream from a read-only prediction stream, the
  Free Pause Tokens idea assessed in [literature](docs/literature.md).
- Give the model a task that needs the channel, rather than hoping FineWeb
  induces one.

## Start here

- [design.md](docs/design.md): the conditions, geometry, data, schedule, feedback
  passes, evaluation modes, and the knobs on the recipe.
- [architecture.md](docs/architecture.md): the exact model, state, and
  optimizer.
- [interpretability.md](docs/interpretability.md): the analysis scripts,
  what each one shows, and the shape of the future study.
- [operations.md](docs/operations.md): install, tokenize, train, queue,
  inspect. Run `delta probe` before training; Jobe's captured graph pool
  reserves about 23 GiB, so GPU work stays serial.
- [runtime-qualification.md](docs/runtime-qualification.md): the CUDA
  execution path and its numerical evidence.
- [scaling.md](docs/scaling.md): longer and larger recipes, worked out but
  not scheduled.
- [literature.md](docs/literature.md) and
  [references/refs.yaml](references/refs.yaml): where each mechanism comes from
  and what we changed.

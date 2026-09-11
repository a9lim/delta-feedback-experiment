# delta-feedback-experiment

This is a nursery for a small recurrent language model with a latent channel
between token columns. The eventual goal is a **model organism**: a model of
roughly 180M parameters whose cross-token latent computation is real enough,
and opaque enough, to be worth studying with interpretability and monitoring
methods. That study is future work with its own design. This repository is
where the organism gets grown: train variants, look inside, change the recipe
or the architecture, repeat.

Nothing here is pre-registered. Runs are experiments in the ordinary sense of
trying things. [The journal](docs/journal.md) records the working state,
[findings](docs/findings.md) keeps the distilled current picture, and the
rest of the docs describe what the code does today.

## What we are growing

The model is a decoder with four kinds of recurrence available to it:

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
- **Tied depth.** Under `l` the middle cell is one weight-tied core that runs
  a drawn number of times per column, so a column can refine its state before
  it emits a payload.

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
letters, each one change from the plain twelve-layer gated GQA decoder,
so any two conditions can be trained on identical rows from paired
initializations and compared:

| Letter | Change |
|---|---|
| `a` | Kimi Delta Attention: replace the three RoPE-GGQA layers per cell with PKDA, giving `[PKDA, PKDA, PKDA, NoPE-GGQA]` |
| `r` | MHDB residual reads of the seed and block deltas before every sublayer; with `f`, a routed payload |
| `f` | Full-bandwidth feedback: the FBT entry and a payload for the next column |
| `l` | Huginn loop: the cells between the first and last become one tied core iterated a drawn number of times per column ([architecture.md](docs/architecture.md#letter-l-the-tied-depth-loop)) |

`--condition arfl` is the full built stack and `arf` the flat column; `ar`
and `af` each drop one package; `a` is the bare hybrid trunk; the empty
condition is the plain decoder, with `[RoPE-GGQA, RoPE-GGQA, RoPE-GGQA,
NoPE-GGQA]` cells. RoPE rotates the full Q/K head width after per-head RMSNorm
in those three layers; the fourth layer stays NoPE in every condition. The
[architecture page](docs/architecture.md) has the exact equations, geometry,
and optimizer.

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

The tokenizer is pinned GPT-NeoX with two generic ChatML delimiters: 50,279
token IDs and a 50,304-row tied embedding/readout. The screen's full stack
has 179,461,416 parameters. ChatML preserves arbitrary role names and repeated
roles; it supplies formatting for future conversational data, not instruction
tuning or pretrained behavior.

There are no current trained specimens. The token store is being rebuilt on
Jobe for a 15B-token target under this tokenizer, and training has not been
launched. Checkpoint v28 is the only accepted format. The current CUDA path
must be qualified with the new head before assigning it a measured training
speed; [runtime qualification](docs/runtime-qualification.md) tracks that gate.

The next scientific question is whether the current architecture and recipe
grow a useful cross-column channel. [Findings](docs/findings.md) states the
current evidence boundary, and the existing analysis tools can measure
payload use, token recoverability, routing, and recurrence once a trained
checkpoint exists.

## Start here

- [design.md](docs/design.md): the conditions, geometry, data, schedule, feedback
  passes, evaluation modes, and the knobs on the recipe.
- [architecture.md](docs/architecture.md): the four letters and the column,
  their equations and state contracts, initialization, and the optimizer;
  `L` specified and unbuilt.
- [interpretability.md](docs/interpretability.md): the analysis scripts,
  what each one shows, and the shape of the future study.
- [operations.md](docs/operations.md): install, tokenize, train, queue,
  inspect. Run `delta probe` before training and keep GPU work serial on Jobe.
- [data-build-performance.md](docs/data-build-performance.md): token-store
  profiling, assembly tuning, and Prime staging/storage requirements.
- [runtime-qualification.md](docs/runtime-qualification.md): the CUDA
  execution path and its numerical evidence.
- [scaling.md](docs/scaling.md): the screen, the bridge, and the flagship,
  their geometry, accounting, and budgets, and the longer and larger recipes,
  worked out but not scheduled.
- [literature.md](docs/literature.md) and
  [references/refs.yaml](references/refs.yaml): where each mechanism comes from
  and what we changed.

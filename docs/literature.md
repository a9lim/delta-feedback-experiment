# Literature and synthesis

The organism assembles mechanisms from published papers and released
implementations. It is a synthesis rather than a reproduction of any one
source model, and source-paper gains are not assumed to carry over.

[references/refs.yaml](../references/refs.yaml) is the machine-readable primary
source index. Its entries identify the papers and implementation references,
the exact roles used here, and deliberate departures. The map below connects
those roles to the organism.

| Source family | Role in the organism | Why it is here |
|---|---|---|
| Full-Bandwidth Transformer | Asymmetric payload-value/token-gate entry, latent feedback, Jacobi training, and recipe lineage | A named latent channel across token columns, with plain and fused execution modes |
| Kimi Linear / Preconditioned DeltaNet | PKDA recurrence, short convolution, preconditioner, and recurrent-state boundaries | Explicit token-mixer memory that can be distinguished from the FBT payload |
| Qwen3-Next released configuration and attention | Gated global GQA and interval-four precedent | A full-prefix read pathway alongside recurrent mixer memory |
| Multi-head and Delta Attention Residuals | Grouped source selection and additive delta-source semantics | Addressable seed and block contributions, with a residual reconstruction identity |
| Attention Residuals | Cumulative-state routing comparison | Clarifies how additive delta routing differs from replacing the residual read |
| NorMuon / Hyperball / NAdam implementation | Matrix update direction, fixed realized radii, and semantic-scale optimization | A fixed training recipe across controls, with explicit parameter ownership |
| Recurrent-depth language models | Prelude/tied-core/coda and the iteration draw for the `l` letter; zero-shot cache sharing as the trained `L` channel | An iteration axis, `l`, built; `L` specified |
| Free Pause Tokens | Design comparison: separate persistent state from a read-only prediction stream | A candidate way to separate memory from prediction; assessed below |

The local synthesis choices include the PKDA/GQA composition, four-layer MHDB
source banks, routed enrichment of the payload under `r` with `f`, exact optimizer partition,
geometry, and paired training schedules. They are specified in
[architecture.md](architecture.md) and [design.md](design.md). The loop's
adoptions and replacements are recorded separately in
[depth-architecture.md](depth-architecture.md#sources).

Source precedent motivates a component; the probe checks that we implemented
the chosen equations; what the trained organism does with them is what the
analysis scripts are for.

Fetched paper Markdown/PDF files are regenerable and ignored. From the
workspace reference workflow, use:

```bash
python -m transformer_experiments.references
```

Keep the source index to mechanisms and methods used here or assessed for a
design decision.

## Free Pause Tokens and the loop

[Free Pause Tokens](https://arxiv.org/abs/2609.03807), Langford et al., v1,
is indexed under that title; its HTML body is titled *Almost Free State
Prediction Separation*. The local full text is regenerated with
`python -m transformer_experiments.references langford2026-free-pause-tokens`.

The paper's ordinary state stream writes per-layer K/V. A second stream starts
each position from one learned vector, shares backbone weights, reads state
K/V causally through the current position, and supplies the only LM loss. It
writes no prediction K/V. Training gradients still reach the state stream.
Optional shared FFNs and late activation of the split reduce training cost.

At 1B parameters and 100B tokens, reported CE is 2.8957 for the control and
2.8673 for the full split. Activating the split after 75% of training gives
2.8756 at 1.14 times training wall-clock. Equal-FLOP full-split performance is
slightly worse than the control; phased variants retain smaller gains. The
roughly 1% decode overhead is a fused B200 microbenchmark. Evidence uses one
primary seed; individual downstream-task differences are unresolved.
[Method and results](https://arxiv.org/html/2609.03807v1).

### Relationship to the loop

Huginn supplies repeated application of a tied core before readout. Free Pause
Tokens supplies a second stream alongside a fixed-depth backbone. Weight
sharing between streams does not itself introduce depth recurrence. Our loop
already replaces much of Huginn: MHDB input injection, deterministic prelude
initialization, full backpropagation, PKDA/GQA, and the trained FBT payload
are local choices; the shared cache is the unbuilt `L`. This is a decision about computational structure, not a
choice between loading two pretrained models.

For the specified question of how iterated computation persists across tokens,
retain the tied core as the starting point. The new paper motivates a separate
question: does a prediction-oriented computation need to enter persistent
memory? A read-only prediction stream would create a useful intervention
boundary. Under fixed token replay, changing that stream could affect the
current logits while leaving future state, payload, and cache unchanged. Free
generation would still propagate the change through the emitted token.

### Candidate integration

First isolate separation at `r = 1`, before combining it with variable depth:

- Let the state stream retain the existing delta column, MHDB banks, cache writes,
  and FBT payload. Let prediction use its own residual and MHDB source banks,
  with shared backbone/router parameters and one learned initial vector. Feed
  the tied readout from prediction. Apply the existing pass-loss weighting to
  prediction outputs; do not detach state or earlier Jacobi passes.
- In GQA, form prediction Q and its output gate, then read state K/V through
  the current position. There is no prediction own-K/V term and no cache
  append. The live attention method combines projection and cache mutation,
  so it would need an explicit read-only execution path.
- In PKDA, state alone computes K/V, decay, write strength, preconditioner
  updates, and the recurrent matrix. The proposed prediction read is
  `o_pred,t = S_state,t^T q_pred,t`, with the usual query scale, output norm,
  gate, and projection. This uses the inclusive state after its token update;
  it is different from the loop's exclusive bank plus prediction-own update.
  The current recurrence already separates Q reads algebraically from writes,
  but efficient two-query training/backward still needs CUDA qualification.
- Resolve query convolution explicitly. The live PKDA Q projection has a
  temporal convolution. Applying it to prediction history would add persistent
  prediction state and break the proposed intervention boundary. One candidate
  uses state Q-projection history plus the current prediction projection;
  another removes temporal convolution from prediction Q. Either is a local
  departure requiring a declared equation and portable/decode parity.
- Keep two separate FFN evaluations initially, even though their weights are
  tied. The paper's pooled shared FFN depends on both streams and updates both,
  creating an indirect prediction-to-state path. Its state-only prefill
  shortcut cannot be assumed for that coupled variant.

Use paired continuations from the same checkpoint and data position, comparing
the ordinary model with the declared split at matched tokens and measured device-time;
report actual FLOPs and stream/cell evaluations separately, with a matched
extra-compute control before attributing a gain to separation. The
interesting analyses are fixed-token prediction interventions, state/payload
interventions with held-out behavioral witnesses, and checks of which future
paths change.

Only then consider a recurrent prediction core over a fixed state bank. That
would combine the paper's separation with Huginn-style iteration and simplify
which stream can write memory, but it changes the loop's scientific object:
refinement would not reach later tokens through a latent payload. Conversely,
letting the state core iterate and adding a prediction reader preserves carried
refinement but adds compute and, under `L`, the cache complexity.
Neither alternative inherits the paper's loss or latency results. For Jobe,
measure prefill, decode across batch sizes, full backward memory, and captured
execution before calling either cheaper.

# Literature and synthesis

This organism assembles mechanisms from already-published papers and released
implementations. The research aim is a small system on which to develop
interpretability and monitoring methods. We do not claim that combining the
components advances the capabilities frontier, that source-paper gains carry
over, or that this is a numerical reproduction of any one source model.

[references/refs.yaml](../references/refs.yaml) is the machine-readable primary
source index. Its entries identify the papers and implementation references,
the exact roles used here, and deliberate departures. The map below connects
those roles to the experimental object; the interpretability motivations are
this project's design rationale, not results established by the source papers.

| Source family | Role in the organism | Why it is useful to study |
|---|---|---|
| Full-Bandwidth Transformer | Asymmetric payload-value/token-gate entry, latent feedback, Jacobi training, and recipe lineage | A named latent channel across token columns, with plain and fused execution modes |
| Kimi Linear / Preconditioned DeltaNet | PKDA recurrence, short convolution, preconditioner, and recurrent-state boundaries | Explicit token-mixer memory that can be distinguished from the FBT payload |
| Qwen3-Next released configuration and attention | Gated global GQA and interval-four precedent | A full-prefix read pathway alongside recurrent mixer memory |
| Multi-head and Delta Attention Residuals | Grouped source selection and additive delta-source semantics | Addressable seed and block contributions, with a residual reconstruction identity |
| Attention Residuals | Cumulative-state routing comparison | Clarifies how additive delta routing differs from replacing the residual read |
| NorMuon / Hyperball / NAdam implementation | Matrix update direction, fixed realized radii, and semantic-scale optimization | A fixed training recipe across controls, with explicit parameter ownership |
| Recurrent-depth language models | Prelude/tied-core/coda, iteration draw, and shared-cache ideas for the proposed `df-loop` | A future iteration-level experimental surface; not implemented in current arms |

The local synthesis choices include the PKDA/GQA composition, four-layer MHDB
source banks, routed enrichment of the `df` payload, exact optimizer partition,
geometry, and paired training schedules. They are specified in
[architecture.md](architecture.md) and [design.md](design.md). The loop's
adoptions and replacements are recorded separately in
[depth-architecture.md](depth-architecture.md#sources).

Source precedent motivates a component. Local numerical tests establish that we
implemented the chosen equations. Controlled behavioral and causal studies must
establish what the trained organism actually does. These are separate kinds of
evidence; none substitutes for the others.

Fetched paper Markdown/PDF files are regenerable and ignored. From the
workspace reference workflow, use:

```bash
python -m transformer_experiments.references
```

Keep the source index focused on mechanisms and methods actually used. Add new
interpretability references when a method is selected, with its adopted
protocol and limits; a reading list is not an implemented analysis pipeline.

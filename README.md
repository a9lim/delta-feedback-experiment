# delta-feedback-experiment

Do the two axes of a transformer's compute lattice — depth-wise routing
(**Delta Attention Residuals**, arXiv:2605.18855) and token-time latent
feedback (**full-bandwidth transformers**, arXiv:2608.08888) — help
complementarily or redundantly when pretrained together? The combined model
("delta feedback", DF) commits hard to both: gated latent feedback whose
payload is the top state plus routed deltas — DAR's additive routing
applied at the cross-column site. A soft screen-only companion arm makes
every channel optional and null-sourced, its routing weights reading out
what a free model actually adopts.

**Status:** design phase. Design settled (2026-08-27); no training code, no
results yet. See [`docs/findings.md`](docs/findings.md).

## Plan shape

- Screen on jobe (1x4090): {vanilla, DAR, FBT, DF, DF-soft} at DAR's 220M
  config, ~2B FineWeb-Edu tokens, 2 seeds, one recipe (FBT's, binding).
- Token ladder on rented single GPUs: finalists extended 2B -> 8B -> 32B via
  WSD, testing whether the combined advantage holds with training scale
  (the flagship regime is ~370 tok/param; the screen alone cannot reach it).
- Flagship on rented pods: DF at ~1.08B params / 400B tokens,
  gated on the ladder trend (pre-registered gate in the design doc).
- Phase-2 extension (contingent): pause-token pretraining on the winner.

## Install

```bash
# from the workspace root, once:
uv pip install -e .
# then:
cd delta-feedback-experiment
uv pip install -e .
```

## Run

Nothing runnable yet. `scripts/` will hold numbered entry points as the
harness lands.

## Docs map

- [`docs/design.md`](docs/design.md) — the authoritative current design:
  architecture, training, arms, scale plan, evaluation, gates, risks.
- [`docs/findings.md`](docs/findings.md) — claims and limitations (currently:
  none).
- [`docs/journal.md`](docs/journal.md) — disposable working notes,
  periodically cleared.
- [`references/refs.yaml`](references/refs.yaml) — load-bearing papers;
  fetch markdown copies with `python -m transformer_experiments.references`.
- [`figures/README.md`](figures/README.md) — figure map (currently empty).

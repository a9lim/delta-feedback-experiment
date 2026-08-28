# delta-feedback-experiment

Do the two axes of a transformer's compute lattice — depth-wise routing
(**Delta Attention Residuals**, arXiv:2605.18855) and token-time latent
feedback (**full-bandwidth transformers**, arXiv:2608.08888) — help
complementarily or redundantly when pretrained together? Secondary: should
the feedback payload be the raw top-layer state (A1) or a routed combination
of the column's deltas (A2, "delta feedback") — the injection-form question
the FBT paper leaves open.

**Status:** design phase. Ledger settled (2026-08-27); no training code, no
results yet. See [`docs/findings.md`](docs/findings.md).

## Plan shape

- Trials on jobe (1x4090): {vanilla, DAR, FBT, A1, A2} at DAR's 220M config,
  ~1B FineWeb-Edu tokens, 2 seeds, one recipe (FBT's, binding).
- Flagship on rented pods: better of A1/A2 at ~1.08B params / 400B tokens,
  gated on the trials (pre-registered gate in the ledger).
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

- [`docs/design.md`](docs/design.md) — the design ledger (D1-D18):
  architecture, arms, recipe, budgets, promotion gate, risks.
- [`docs/findings.md`](docs/findings.md) — claims and limitations (currently:
  none).
- [`references/refs.yaml`](references/refs.yaml) — load-bearing papers;
  fetch markdown copies with `python -m transformer_experiments.references`.
- [`figures/README.md`](figures/README.md) — figure map (currently empty).

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

**Status:** the plain-PyTorch harness and optimized Jobe CUDA path are ready;
no scientific training result exists yet. See
[`docs/findings.md`](docs/findings.md).

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

# Jobe CUDA kernels (the shared environment already carries the pinned wheels):
uv pip install -e '.[cuda]'
```

## Run

```bash
df probe
df train TAG --arm df --data-dir /data/df/tokens
df queue TAG --arm df --data-dir /data/df/tokens
df status
df watch
```

`df probe` runs the invariant suite everywhere and, when CUDA is present, the
full 220M/B4/T1025 graph-capture gate. Training uses fixed CUDA graphs,
FlashAttention, CCE, BF16 residuals, and an internal measured activation plan;
there are no public kernel or checkpoint-policy switches.

## Docs map

- [`docs/design.md`](docs/design.md) — the authoritative current design:
  architecture, training, arms, scale plan, evaluation, gates, risks.
- [`docs/findings.md`](docs/findings.md) — scientific and systems evidence,
  with limitations.
- [`docs/journal.md`](docs/journal.md) — disposable working notes,
  periodically cleared.
- [`references/refs.yaml`](references/refs.yaml) — load-bearing papers;
  fetch markdown copies with `python -m transformer_experiments.references`.
- [`figures/README.md`](figures/README.md) — figure map (currently empty).

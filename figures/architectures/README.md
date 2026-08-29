# Architecture diagrams

These are model-contract figures, not result figures. They are generated from
the current architecture in [`../../docs/design.md`](../../docs/design.md) and
the live arm definitions in
[`../../delta_feedback_experiment/model.py`](../../delta_feedback_experiment/model.py).

- Vanilla — [SVG](vanilla.svg) · [PNG](vanilla.png)
- DAR — [SVG](dar.svg) · [PNG](dar.png)
- FBT — [SVG](fbt.svg) · [PNG](fbt.png)
- DF — [SVG](df.svg) · [PNG](df.png)
- DF-soft — [SVG](df-soft.svg) · [PNG](df-soft.png)

Regenerate all five with:

```bash
python figures/architectures/render.py
```

The SVGs are canonical. The committed 1700×950 PNGs are raster exports for
places that do not render SVG directly.

The visual grammar is shared across the set: blue is the ordinary residual
path, purple is depth/payload routing, orange is recurrent feedback, and gray
is DF-soft's null escape.

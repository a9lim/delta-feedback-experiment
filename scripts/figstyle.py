"""Shared figure style for the analysis scripts: palette, rc defaults, helpers.

The categorical order is fixed (never cycled) so a run or a predictor keeps its
hue across every figure of one analysis.  Marks are thin, grids are hairlines,
and text stays in ink colors rather than series colors.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
SERIES = (BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED)

INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

SEQUENTIAL = "Blues"
DIVERGING = "RdBu_r"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "font.size": 9,
        "font.family": "sans-serif",
        "figure.dpi": 100,
        "savefig.dpi": 170,
    }
)


def zero_line(ax, axis: str = "y") -> None:
    """A recessive reference line at zero."""
    if axis == "y":
        ax.axhline(0, color=BASELINE, lw=0.8, zorder=1)
    else:
        ax.axvline(0, color=BASELINE, lw=0.8, zorder=1)


def mark_step(ax, step: float, label: str | None = None, y: float = 0.98) -> None:
    """A vertical schedule marker with a small muted label."""
    ax.axvline(step, color=MUTED, lw=0.9, ls=":", zorder=1)
    if label:
        ax.text(
            step,
            y,
            f" {label}",
            color=MUTED,
            fontsize=7,
            ha="left",
            va="top",
            transform=ax.get_xaxis_transform(),
        )


def save(fig, path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

"""Per-column readout of a loop snapshot: loss, update size, and router mass by iteration.

    python scripts/depth_trace.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b

Runs ``depth_trace`` on held-out rows in both label assignments (all plain,
and prefix-1 fused when the condition has ``f``), records every router's
mean mass on each source at every column up to the cap, and writes
``depth_trace.json`` and ``depth-trace.png`` under ``figures/depth-TAG/``.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import figstyle as fs
import matplotlib.pyplot as plt
import torch

from delta_feedback_experiment.analysis import autocast, load_checkpoint
from delta_feedback_experiment.data import TokenData
from delta_feedback_experiment.model import LOOP_MAX_ITERATIONS, depth_trace


@torch.no_grad()
def route_mass_by_column(model, rows, iterations: int) -> dict[str, dict[int, dict[str, float]]]:
    """Mean route mass per source at every site, keyed by column index."""
    device = rows.device
    with autocast(device):
        e = model.embed_tokens(rows[:, :-1])
        columns = model.forward_iterations(
            model.plain_seed(e), e, iterations=iterations, want_weights=True
        )
    routes: dict[str, dict[int, dict[str, float]]] = {}
    for iteration, out in enumerate(columns):
        for site, weights in out.route_weights.items():
            names = out.route_source_names[site]
            mean = weights.float().mean(dim=(1, 2, 3))
            routes.setdefault(site, {})[iteration] = dict(
                zip(names, mean.tolist(), strict=True)
            )
    return routes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("snapshot")
    parser.add_argument("--data-dir", default="data/dclm-100b")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--micro", type=int, default=4)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=f"column cap (default: {LOOP_MAX_ITERATIONS})",
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    model, saved = load_checkpoint(args.snapshot)
    cfg = model.cfg
    if not cfg.loop:
        raise SystemExit(f"{args.snapshot} is a {cfg.condition!r} snapshot, not a loop")
    device = next(model.parameters()).device
    data = TokenData.load(args.data_dir, "val", saved["seq_len"])
    cap = LOOP_MAX_ITERATIONS if args.iterations is None else args.iterations
    tag = Path(args.snapshot).name.split(".pt.")[0]
    out_dir = Path(args.out_dir or f"figures/depth-{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = ["plain"] + (["fused"] if cfg.feedback else [])
    sums = {mode: {"loss": [0.0] * cap, "update_norm": [0.0] * cap} for mode in modes}
    counted = 0
    for first in range(0, args.rows, args.micro):
        rows = data.batch(first, min(args.micro, args.rows - first), device)
        for mode in modes:
            records = depth_trace(model, rows, cap, fused=mode == "fused")
            for index, record in enumerate(records):
                for key in ("loss", "update_norm"):
                    sums[mode][key][index] += record[key] * rows.shape[0]
        counted += rows.shape[0]
    sweep = {
        mode: {key: [value / counted for value in values] for key, values in stats.items()}
        for mode, stats in sums.items()
    }
    routes = route_mass_by_column(model, data.batch(0, min(args.micro, args.rows), device), cap)

    record = {
        "snapshot": str(args.snapshot),
        "condition": cfg.condition,
        "step": saved["step"],
        "rows": counted,
        "iterations": cap,
        "r_eval": cfg.loop_iterations,
        "device": str(device),
        "sweep": sweep,
        "routes": routes,
    }
    (out_dir / "depth_trace.json").write_text(json.dumps(record, indent=2))

    depth = list(range(1, cap + 1))
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    colors = {"plain": fs.BLUE, "fused": fs.ORANGE}
    for mode in modes:
        axes[0].plot(depth, sweep[mode]["loss"], "o-", color=colors[mode], label=mode)
        axes[1].plot(
            depth, sweep[mode]["update_norm"], "o-", color=colors[mode], label=mode
        )
    for ax in axes[:2]:
        ax.axvline(cfg.loop_iterations, color="0.6", ls=":", lw=1)
        ax.set_xlabel("columns r")
        ax.set_xticks(depth)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("held-out loss after r columns")
    axes[1].set_ylabel("mean ||top-state update|| at column r")
    labels: list[str] = []
    for by_iteration in routes.values():
        for masses in by_iteration.values():
            labels += [label for label in masses if label not in labels]
    for label, color in zip(labels, itertools.cycle(fs.SERIES)):
        series = []
        for iteration in range(cap):
            masses = [
                by_iteration[iteration][label]
                for by_iteration in routes.values()
                if iteration in by_iteration and label in by_iteration[iteration]
            ]
            series.append(sum(masses) / len(masses) if masses else float("nan"))
        axes[2].plot(depth, series, "o-", color=color, label=label)
    axes[2].set_xlabel("column")
    axes[2].set_xticks(depth)
    axes[2].set_ylabel("mean route mass over sites")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{tag} step {saved['step']}: depth trace on {counted} rows", fontsize=10)
    fs.save(fig, out_dir / "depth-trace.png")

    print(f"{tag} step {saved['step']} ({cfg.condition}), {counted} rows, r_eval {cfg.loop_iterations}")
    print(f"{'r':>3} " + " ".join(f"{mode + ' loss':>12} {mode + ' upd':>11}" for mode in modes))
    for index, r in enumerate(depth):
        cells = " ".join(
            f"{sweep[mode]['loss'][index]:12.4f} {sweep[mode]['update_norm'][index]:11.4f}"
            for mode in modes
        )
        print(f"{r:>3} {cells}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

"""Fixed-r sweep of a loop snapshot: loss, update size, and router mass by iteration.

    python scripts/depth_trace.py runs/TAG.pt.STEP --data-dir /data/delta/tokens

Runs ``depth_trace`` on held-out rows in both label assignments (all plain,
and prefix-1 fused when the condition has ``f``), records the core routers'
mean mass on each source by iteration at the cap, and writes
``depth_trace.json`` and ``depth-trace.png`` under ``figures/depth-TAG/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

from delta_feedback_experiment.analysis import autocast, load_checkpoint  # noqa: E402
from delta_feedback_experiment.data import TokenData  # noqa: E402
from delta_feedback_experiment.model import depth_trace  # noqa: E402


def core_route_mass(model, rows, iterations: int) -> dict[str, dict[int, dict[str, float]]]:
    """Mean route mass per source at every core site, keyed by iteration."""
    device = rows.device
    with autocast(device):
        e = model.embed_tokens(rows[:, :-1])
        out = model.forward_column(e, want_weights=True, iterations=iterations)
    routes: dict[str, dict[int, dict[str, float]]] = {}
    for site, weights in out.route_weights.items():
        layer_kind, _, sublayer = site.partition(".")
        layer, tagged, iteration = layer_kind[1:].partition("i")
        if not tagged:
            continue
        names = out.route_source_names[site]
        mean = weights.float().mean(dim=(1, 2, 3))
        routes.setdefault(f"L{layer}.{sublayer}", {})[int(iteration)] = dict(
            zip(names, mean.tolist(), strict=True)
        )
    return routes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("snapshot")
    parser.add_argument("--data-dir", default="data/tokens")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--micro", type=int, default=4)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="sweep cap (default: the snapshot's r_max)",
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    model, saved = load_checkpoint(args.snapshot)
    cfg = model.cfg
    if not cfg.loop:
        raise SystemExit(f"{args.snapshot} is a {cfg.condition!r} snapshot, not a loop")
    device = next(model.parameters()).device
    data = TokenData.load(args.data_dir, "val", saved["seq_len"])
    cap = args.iterations or cfg.loop_max_iterations
    tag = Path(args.snapshot).name.split(".pt.")[0]
    out_dir = Path(args.out_dir or f"figures/depth-{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = ["plain"] + (["fused"] if cfg.feedback_active else [])
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
    routes = (
        core_route_mass(model, data.batch(0, min(args.micro, args.rows), device), cap)
        if cfg.routing_active
        else {}
    )

    record = {
        "snapshot": str(args.snapshot),
        "condition": cfg.condition,
        "step": saved["step"],
        "rows": counted,
        "iterations": cap,
        "r_mean": cfg.loop_iterations,
        "device": str(device),
        "sweep": sweep,
        "core_routes": routes,
    }
    (out_dir / "depth_trace.json").write_text(json.dumps(record, indent=2))

    depth = list(range(1, cap + 1))
    panels = 3 if routes else 2
    fig, axes = plt.subplots(1, panels, figsize=(4.2 * panels, 3.4))
    colors = {"plain": fs.BLUE, "fused": fs.ORANGE}
    for mode in modes:
        axes[0].plot(depth, sweep[mode]["loss"], "o-", color=colors[mode], label=mode)
        axes[1].plot(
            depth, sweep[mode]["update_norm"], "o-", color=colors[mode], label=mode
        )
    for ax in axes[:2]:
        ax.axvline(cfg.loop_iterations, color="0.6", ls=":", lw=1)
        ax.set_xlabel("core iterations r")
        ax.set_xticks(depth)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("held-out loss after r iterations")
    axes[1].set_ylabel("mean ||core update|| at iteration r")
    if routes:
        labels = ("null", "seed", "block0", "partial1")
        for label, color in zip(labels, (fs.BLUE, fs.ORANGE, fs.AQUA, "0.3"), strict=True):
            series = []
            for iteration in range(cap):
                masses = [
                    by_iteration[iteration][label]
                    for by_iteration in routes.values()
                    if iteration in by_iteration and label in by_iteration[iteration]
                ]
                series.append(sum(masses) / len(masses) if masses else float("nan"))
            axes[2].plot(depth, series, "o-", color=color, label=label)
        axes[2].set_xlabel("core iteration")
        axes[2].set_xticks(depth)
        axes[2].set_ylabel("mean core route mass")
        axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{tag} step {saved['step']}: depth trace on {counted} rows", fontsize=10)
    fs.save(fig, out_dir / "depth-trace.png")

    print(f"{tag} step {saved['step']} ({cfg.condition}), {counted} rows, r_mean {cfg.loop_iterations}")
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

"""Plot main/auxiliary losses, gradient norm, and throughput from current logs.

    python scripts/training_curves.py logs/A.log logs/B.log --out-dir figures/curves-A-vs-B

Resumed logs keep the final record for each step. Comparing curves does not
assert that the runs use matching data or schedules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import figstyle as fs
import matplotlib.pyplot as plt
import numpy as np


def parse_log(path: Path) -> dict:
    records = {"step": {}, "eval": {}}
    run = {}
    for line in path.read_text().splitlines():
        kind, _, tail = line.partition("|")
        kind = kind.strip()
        if kind not in ("run", "step", "eval"):
            continue
        fields = {}
        for piece in tail.split("|"):
            key, sep, value = piece.strip().partition("=")
            if sep:
                fields[key.strip()] = value.strip()
        if kind == "run":
            run = fields
        else:
            step = int(fields["step"].split("/")[0])
            records[kind][step] = fields
    if not run or not records["step"]:
        raise ValueError(f"{path}: expected a current run header and training steps")
    return {"run": run, **records}


def values(records: dict, key: str) -> np.ndarray:
    return np.array([float(records[step][key]) for step in sorted(records)])


def ema(values: np.ndarray, span: int) -> np.ndarray:
    result = values.copy()
    alpha = 2.0 / (span + 1)
    for index in range(1, len(result)):
        result[index] = alpha * values[index] + (1 - alpha) * result[index - 1]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--ema", type=int, default=300)
    args = parser.parse_args()
    if args.ema < 1:
        parser.error("--ema must be positive")
    runs = [parse_log(path) for path in args.logs]
    labels = [run["run"]["tag"] for run in runs]
    out_dir = args.out_dir or Path("figures") / ("curves-" + "-vs-".join(labels))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    titles = (
        "Main training CE", "Auxiliary training CE (combined)", "Pre-clip gradient norm",
        "Main validation CE", "Auxiliary validation CE", "Predicted tokens / second",
    )
    summary = []
    for index, (path, run, label) in enumerate(zip(args.logs, runs, labels, strict=True)):
        color = fs.SERIES[index % len(fs.SERIES)]
        steps, evals = run["step"], run["eval"]
        x = np.array(sorted(steps))
        axes[0, 0].plot(x, ema(values(steps, "pass1"), args.ema), color=color, label=label)
        fused = values(steps, "k") > 1
        if fused.any():
            ce = values(steps, "ntp") - values(steps, "pass1")
            axes[0, 0].plot(x[fused], ema(ce[fused], args.ema), "--", color=color, label=f"{label}: feedback")
        axes[0, 1].plot(x, ema(values(steps, "mtp"), args.ema), color=color, label=label)
        axes[0, 2].plot(x, ema(values(steps, "gnorm"), args.ema), color=color, label=label)
        axes[1, 2].plot(x, ema(values(steps, "tok_s"), args.ema), color=color, label=label)
        e = np.array(sorted(evals))
        for axis, metric in zip(axes[1, :2], ("val", "val_mtp"), strict=True):
            axis.plot(e, values(evals, metric), color=color, label=label)
            if "f" in run["run"]["condition"]:
                axis.plot(e, values(evals, metric + "_fused"), "--", color=color, label=f"{label}: fused")
        summary.append({
            "log": str(path), "tag": label, "condition": run["run"]["condition"],
            "step": int(x[-1]), "latest_training": steps[int(x[-1])],
            "latest_validation": evals[int(e[-1])] if len(e) else None,
        })
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set(title=title, xlabel="optimizer step")
        axis.legend()
    axes[0, 2].set_yscale("log")
    fs.save(fig, out_dir / "training-curves.png")
    (out_dir / "training_curves.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(out_dir)


if __name__ == "__main__":
    main()

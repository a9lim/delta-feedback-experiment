"""Training-dynamics figures from one or more run logs.

Parses the trainer's telemetry (`step`, `eval`, `route`, `contract`,
`schedule`, `run` lines) and draws validation curves, paired differences
against a reference run, matched-compute curves, exponentially smoothed
training losses (including the per-step paired pass-1 difference, which is
exact because paired runs share every row), gradient norms, routing
trajectories per site, and the repeated-fused-prefill monitor.

Usage:
    python scripts/training_curves.py logs/A.log logs/B.log --out-dir figures/curves-A-vs-B

The first log is the reference for every difference panel.  Runs that were
resumed keep the last record for each step.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import figstyle as fs
import matplotlib.pyplot as plt


STEP_RE = re.compile(r"^(\w+)\s*\|")


def parse_log(path: Path) -> dict:
    """Return {'run': {...}, 'schedule': {...}, 'steps': {step: {...}}, 'evals': [...], 'routes': [...], 'contracts': [...]}."""
    out = {"run": {}, "schedule": {}, "steps": {}, "evals": {}, "routes": [], "contracts": {}}
    for line in path.read_text().splitlines():
        m = STEP_RE.match(line)
        if not m:
            continue
        kind = m.group(1)
        fields = {}
        for piece in line.split("|")[1:]:
            piece = piece.strip()
            if "=" in piece:
                k, v = piece.split("=", 1)
                fields[k.strip()] = v.strip()
        if "step" in fields and "/" in fields["step"]:
            fields["step"] = int(fields["step"].split("/")[0])
        for k, v in list(fields.items()):
            if k in ("site", "phase", "tag", "condition", "path", "kind", "device"):
                continue
            try:
                fields[k] = float(v)
            except (TypeError, ValueError):
                pass
        if kind == "run":
            out["run"] = fields
        elif kind == "schedule":
            out["schedule"] = fields
        elif kind == "step":
            out["steps"][int(fields["step"])] = fields
        elif kind == "eval":
            out["evals"][int(fields["step"])] = fields
        elif kind == "route":
            out["routes"].append(fields)
        elif kind == "contract":
            out["contracts"][int(fields["step"])] = fields
    out["evals"] = [out["evals"][s] for s in sorted(out["evals"])]
    out["contracts"] = [out["contracts"][s] for s in sorted(out["contracts"])]
    return out


def ema(x: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    y = np.empty_like(x, dtype=float)
    acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1 - alpha) * acc
        y[i] = acc
    return y


def step_arrays(steps: dict) -> dict[str, np.ndarray]:
    keys = sorted(steps)
    cols = {"step": np.array(keys, dtype=float)}
    for name in ("loss", "pass1", "k", "gnorm", "pass_tok_s", "tok_s"):
        cols[name] = np.array([steps[s].get(name, np.nan) for s in keys], dtype=float)
    return cols


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", type=Path, nargs="+", help="run logs; the first is the reference")
    ap.add_argument("--labels", nargs="*", default=None, help="legend labels (default: run tags)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--ema", type=int, default=300, help="EMA span in steps for training losses")
    ap.add_argument("--zoom", type=float, default=0.4, help="fraction of the schedule shown in the zoom panel")
    ap.add_argument("--batch-rows", type=int, default=320)
    ap.add_argument("--seq-len", type=int, default=1024)
    args = ap.parse_args()

    runs = [parse_log(p) for p in args.logs]
    labels = args.labels or [r["run"].get("tag", p.stem) for r, p in zip(runs, args.logs)]
    if len(labels) != len(runs):
        raise SystemExit("one label per log")
    out_dir = args.out_dir or Path("figures") / ("curves-" + "-vs-".join(labels))
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = fs.SERIES[: len(runs)]
    ref = runs[0]
    total = int(ref["schedule"].get("total_steps", max(ref["steps"])))
    boundary = ref["schedule"].get("feedback_boundary")
    cooldown_start = total - int(ref["schedule"].get("cooldown_steps", 0)) + 1
    tokens_per_step = args.batch_rows * args.seq_len
    summary = {"reference": labels[0], "runs": {}}

    # -- per-run arrays --------------------------------------------------------
    arrays = []
    for run in runs:
        a = step_arrays(run["steps"])
        a["cum_pass_tokens"] = np.cumsum(np.nan_to_num(a["k"], nan=1.0)) * tokens_per_step
        a["cum_tokens"] = a["step"] * tokens_per_step
        a["fused_train"] = np.where(a["k"] > 1, a["loss"] - a["pass1"], np.nan)
        arrays.append(a)
    evals = []
    for run in runs:
        e = run["evals"]
        evals.append(
            {
                "step": np.array([r["step"] for r in e], dtype=float),
                "val": np.array([r["val"] for r in e], dtype=float),
                "val_fused": np.array([r.get("val_fused", np.nan) for r in e], dtype=float),
            }
        )
    feedback = [not np.all(np.isnan(e["val_fused"])) for e in evals]

    # -- validation -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    ax = axes[0]
    for e, lab, col, fb in zip(evals, labels, colors, feedback):
        ax.plot(e["step"], e["val"], color=col, label=f"{lab}: pass 1")
        if fb:
            ax.plot(e["step"], e["val_fused"], color=col, ls="--", lw=1.2, label=f"{lab}: fused")
    ax.set(xlabel="optimizer step", ylabel="held-out CE", title="Validation through training")
    if boundary:
        fs.mark_step(ax, boundary, "feedback", y=0.98)
    fs.mark_step(ax, cooldown_start, "cooldown", y=0.92)
    ax.legend()
    ax = axes[1]
    start = total * (1 - args.zoom)
    for e, lab, col, fb in zip(evals, labels, colors, feedback):
        m = e["step"] >= start
        ax.plot(e["step"][m], e["val"][m], "o-", color=col, ms=3, label=f"{lab}: pass 1")
        if fb:
            ax.plot(e["step"][m], e["val_fused"][m], "s--", color=col, ms=3, lw=1.2, label=f"{lab}: fused")
    ax.set(xlabel="optimizer step", ylabel="held-out CE", title=f"Last {int(args.zoom * 100)}% of the schedule")
    fs.mark_step(ax, cooldown_start, "cooldown")
    ax.legend()
    ax = axes[2]
    for i, (e, lab, col, fb) in enumerate(zip(evals, labels, colors, feedback)):
        if i > 0:
            common = np.intersect1d(e["step"], evals[0]["step"])
            d = np.array([e["val"][np.searchsorted(e["step"], s)] - evals[0]["val"][np.searchsorted(evals[0]["step"], s)] for s in common])
            ax.plot(common, d, "o-", color=col, ms=3, label=f"{lab} − {labels[0]}, pass 1")
            summary["runs"].setdefault(lab, {})["val_minus_reference"] = [[float(s), float(v)] for s, v in zip(common, d)]
        if fb:
            ax.plot(e["step"], e["val_fused"] - e["val"], "s--", color=col, ms=3, lw=1.2, label=f"{lab}: fused − pass 1")
            summary["runs"].setdefault(lab, {})["fused_minus_pass1"] = [[float(s), float(v)] for s, v in zip(e["step"], e["val_fused"] - e["val"])]
    fs.zero_line(ax)
    ax.set(xlabel="optimizer step", ylabel="CE difference", title="Paired differences (same 32 rows)")
    ax.set_ylim(-0.06, 0.12)
    if boundary:
        fs.mark_step(ax, boundary, "feedback", y=0.98)
    fs.mark_step(ax, cooldown_start, "cooldown", y=0.92)
    ax.legend(loc="lower left")
    fs.save(fig, out_dir / "validation.png")

    # -- matched compute -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for e, a, lab, col, fb in zip(evals, arrays, labels, colors, feedback):
        idx = np.searchsorted(a["step"], e["step"]).clip(0, len(a["step"]) - 1)
        axes[0].plot(a["cum_tokens"][idx], e["val"], color=col, label=f"{lab}: pass 1")
        axes[1].plot(a["cum_pass_tokens"][idx], e["val"], color=col, label=f"{lab}: pass 1")
        if fb:
            axes[1].plot(a["cum_pass_tokens"][idx], e["val_fused"], color=col, ls="--", lw=1.2, label=f"{lab}: fused")
    axes[0].set(xlabel="predicted tokens (matched data)", ylabel="held-out CE", title="Against predicted tokens", xscale="log")
    axes[1].set(xlabel="pass-tokens (matched compute)", ylabel="held-out CE", title="Against pass-tokens", xscale="log")
    for ax in axes:
        ax.set_ylim(min(np.nanmin(e["val"]) for e in evals) - 0.05, 4.2)
        ax.legend()
    fs.save(fig, out_dir / "matched-compute.png")

    # -- training losses -------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ax = axes[0, 0]
    for a, lab, col in zip(arrays, labels, colors):
        ax.plot(a["step"], ema(a["pass1"], args.ema), color=col, label=f"{lab}: pass 1")
    ax.set(xlabel="optimizer step", ylabel=f"train CE (EMA {args.ema})", title="Pass-1 training loss")
    ax.set_ylim(top=4.5)
    ax.legend()
    ax = axes[0, 1]
    ref_steps = arrays[0]["step"]
    for a, lab, col in list(zip(arrays, labels, colors))[1:]:
        common, ia, ib = np.intersect1d(a["step"], ref_steps, return_indices=True)
        d = a["pass1"][ia] - arrays[0]["pass1"][ib]
        ax.plot(common, ema(d, args.ema), color=col, label=f"{lab} − {labels[0]}")
        summary["runs"].setdefault(lab, {})["train_pass1_minus_reference_last1000"] = float(d[-1000:].mean())
        summary["runs"][lab]["train_pass1_minus_reference_by_phase"] = {
            "warmup+heat_first_half": float(d[common <= total * 0.375].mean()),
            "heat_second_half": float(d[(common > total * 0.375) & (common < cooldown_start)].mean()),
            "cooldown": float(d[common >= cooldown_start].mean()),
        }
    fs.zero_line(ax)
    ax.set(xlabel="optimizer step", ylabel=f"paired CE difference (EMA {args.ema})", title="Pass-1 loss difference on identical rows")
    if boundary:
        fs.mark_step(ax, boundary, "feedback", y=0.98)
    fs.mark_step(ax, cooldown_start, "cooldown", y=0.92)
    ax.legend()
    ax = axes[1, 0]
    any_fb = False
    for a, lab, col, fb in zip(arrays, labels, colors, feedback):
        if not fb:
            continue
        any_fb = True
        m = ~np.isnan(a["fused_train"])
        ax.plot(a["step"][m], ema((a["fused_train"] - a["pass1"])[m], args.ema), color=col, label=f"{lab}: mean fused pass − pass 1")
        summary["runs"].setdefault(lab, {})["train_fused_minus_pass1_last1000"] = float((a["fused_train"] - a["pass1"])[m][-1000:].mean())
    fs.zero_line(ax)
    ax.set(xlabel="optimizer step", ylabel=f"CE difference (EMA {args.ema})", title="Training-time fused gap")
    if any_fb:
        ax.set_ylim(-0.05, 0.15)
        ax.legend()
    ax = axes[1, 1]
    for a, lab, col in zip(arrays, labels, colors):
        ax.plot(a["step"], ema(a["gnorm"], args.ema), color=col, label=lab)
    ax.set(xlabel="optimizer step", ylabel=f"pre-clip gradient norm (EMA {args.ema})", title="Gradient norm", yscale="log")
    ax.legend()
    fs.save(fig, out_dir / "training-loss.png")

    # -- routing trajectories -------------------------------------------------
    routed = [r for r in runs if r["routes"]]
    if routed:
        sites = []
        for r in routed:
            for rec in r["routes"]:
                if rec["site"] not in sites:
                    sites.append(rec["site"])
        column_sites = [s for s in sites if s != "payload"]
        fig, axes = plt.subplots(2, len(routed), figsize=(5.2 * len(routed), 7.5), squeeze=False, constrained_layout=True)
        for j, (run, lab) in enumerate([(r, l) for r, l in zip(runs, labels) if r["routes"]]):
            steps_r = sorted({rec["step"] for rec in run["routes"]})
            for i, key in enumerate(("seed", "null")):
                grid = np.full((len(column_sites), len(steps_r)), np.nan)
                for rec in run["routes"]:
                    if rec["site"] in column_sites and key in rec:
                        grid[column_sites.index(rec["site"]), steps_r.index(rec["step"])] = rec[key]
                ax = axes[i, j]
                im = ax.imshow(grid, aspect="auto", cmap=fs.SEQUENTIAL, vmin=0, vmax=1, interpolation="nearest")
                ax.set_yticks(range(len(column_sites)), column_sites, fontsize=6)
                xt = np.linspace(0, len(steps_r) - 1, 6).astype(int)
                ax.set_xticks(xt, [int(steps_r[t]) for t in xt], fontsize=7)
                ax.set(title=f"{lab}: mean {key} mass per site", xlabel="optimizer step")
                ax.grid(False)
        fig.colorbar(im, ax=axes, shrink=0.6, label="mean routing weight (2 held-out rows)")
        fs.save(fig, out_dir / "routing-sites.png")

        payload_runs = [(r, l, c) for r, l, c in zip(runs, labels, colors) if any(rec["site"] == "payload" for rec in r["routes"])]
        if payload_runs:
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
            for run, lab, col in payload_runs:
                recs = [rec for rec in run["routes"] if rec["site"] == "payload"]
                s = np.array([rec["step"] for rec in recs], dtype=float)
                axes[0].plot(s, [rec.get("seed", np.nan) for rec in recs], color=col, label=f"{lab}: seed")
                axes[0].plot(s, [rec.get("null", np.nan) for rec in recs], color=col, ls="--", lw=1.2, label=f"{lab}: null")
                axes[1].plot(s, [rec.get("max", np.nan) for rec in recs], color=col, label=f"{lab}: mean token-wise max")
                axes[1].plot(s, [rec.get("head_js", np.nan) for rec in recs], color=col, ls="--", lw=1.2, label=f"{lab}: cross-head JS")
                axes[2].plot(s, [rec.get("null_rms", np.nan) for rec in recs], color=col, label=lab)
            axes[0].set(xlabel="optimizer step", ylabel="mean routing weight", title="Payload router: seed and null mass", ylim=(0, 1))
            axes[1].set(xlabel="optimizer step", ylabel="statistic", title="Payload router: sharpness and head disagreement", ylim=(0, 1))
            axes[2].set(xlabel="optimizer step", ylabel="RMS of the learned null", title="Payload null scale")
            for ax in axes:
                ax.legend()
            fs.save(fig, out_dir / "routing-payload.png")

    # -- contraction monitor --------------------------------------------------
    contract_runs = [(r, l, c) for r, l, c in zip(runs, labels, colors) if r["contracts"]]
    if contract_runs:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        for run, lab, col in contract_runs:
            s = np.array([r["step"] for r in run["contracts"]], dtype=float)
            l0 = np.array([r["loss0"] for r in run["contracts"]])
            l8 = np.array([r["loss8"] for r in run["contracts"]])
            u8 = np.array([r["upd8"] for r in run["contracts"]])
            axes[0].plot(s, l8 - l0, "o-", color=col, ms=3, label=lab)
            axes[1].plot(s, u8, "o-", color=col, ms=3, label=lab)
            summary["runs"].setdefault(lab, {})["contract_final"] = {"loss0": float(l0[-1]), "loss8": float(l8[-1]), "upd8": float(u8[-1])}
        fs.zero_line(axes[0])
        axes[0].set(xlabel="optimizer step", ylabel="CE(iteration 8) − CE(iteration 1)", title="Repeated fused prefill: loss drift (2 rows)")
        axes[1].set(xlabel="optimizer step", ylabel="mean ‖h₈ − h₇‖₂", title="Repeated fused prefill: iteration-8 update", yscale="log")
        for ax in axes:
            ax.legend()
        fs.save(fig, out_dir / "contraction.png")

    # -- summary ---------------------------------------------------------------
    for run, a, e, lab in zip(runs, arrays, evals, labels):
        entry = summary["runs"].setdefault(lab, {})
        entry.update(
            {
                "condition": run["run"].get("condition"),
                "final_val": float(e["val"][-1]),
                "final_val_fused": None if np.isnan(e["val_fused"][-1]) else float(e["val_fused"][-1]),
                "steps": int(a["step"][-1]),
                "predicted_tokens": float(a["cum_tokens"][-1]),
                "pass_tokens": float(a["cum_pass_tokens"][-1]),
                "pass_token_multiplier": float(a["cum_pass_tokens"][-1] / a["cum_tokens"][-1]),
                "mean_step_seconds": float(tokens_per_step / np.nanmean(a["tok_s"])),
            }
        )
    (out_dir / "training_curves.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in summary["runs"].items()}, indent=2))
    print(f"figures -> {out_dir}/")


if __name__ == "__main__":
    main()

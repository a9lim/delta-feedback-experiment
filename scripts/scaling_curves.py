"""Loss scaling from run logs: stable-phase fits, the cooldown bonus, seed
offsets, and a predicted-final-loss table for every preset.

    python scripts/scaling_curves.py logs/screen-f-25x-feedback0p6-s1.log \\
        logs/screen-n-50x-s2.log logs/bridge-f-25x-feedback0p6-s1.log

Runs sharing ``data_seed`` and batch geometry read identical batches at each
step, so step-matched differences cancel the data-order bumps: a same-scale
pair gives the seed offset over the window where both are plain and stable,
and the cooldown bonus directly where one has begun cooling. Each run's
stable-phase validation curve is fitted as ``a + B·D^-β`` and extrapolated to
its final step; its gap to the annealed final is that run's cooldown bonus.
The stable curve flattens toward a learning-rate floor while annealed finals
keep falling, so only annealed finals are extrapolated: seed-corrected onto
``--seed-basis``, they anchor ``E + A·N^-α + B·D^-β`` with the exponents held
on grids, because three anchors do not determine them. The learning-rate
annealing law of Tissue et al. is fitted per run and jointly as a
schedule-aware cross-check. Downstream ``standard`` results under
``figures/`` are paired per document against the first log's run.
Repeated steps retain their final record.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import figstyle as fs
import matplotlib.pyplot as plt
import numpy as np
from transformer_experiments import downstream, telemetry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from delta_feedback_experiment.downstream import result_path
from delta_feedback_experiment.train import SCALES, reference_active

MODEL_KEYS = (
    "vocab_size", "dim", "layers", "heads", "kv_heads", "head_dim",
    "expert_intermediate", "num_routed_experts", "experts_per_token",
    "pkda_heads", "pkda_head_dim", "pkda_conv_size", "seq_len", "loop_iterations",
)
BETA_GRID = np.round(np.arange(0.05, 2.5, 0.005), 3)
ALPHA_GRID = np.round(np.arange(0.1, 1.5, 0.005), 3)


# -- logs ---------------------------------------------------------------------------


def parse_log(path: Path) -> dict:
    """One run's header, schedule, and per-step training/evaluation records."""
    run, schedule, steps, evals, done = {}, {}, {}, {}, {}
    with path.open("rb") as handle:
        for raw in handle:
            record = telemetry.parse_record(raw.decode("utf8", "replace"))
            if record is None:
                continue
            event = record.pop("event")
            if event == "run":
                run = record
            elif event == "schedule":
                schedule = record
            elif event in ("step", "eval", "done"):
                step = telemetry.record_step({"step": record["step"]})
                target = {"step": steps, "eval": evals, "done": done}[event]
                target[step] = {k: v for k, v in record.items() if k != "step"}
    if not run or not schedule or not evals:
        raise ValueError(f"{path}: expected run and schedule headers and evaluations")
    dim = int(run["dim"])
    scale = next((name for name, geo in SCALES.items() if geo["dim"] == dim), f"dim{dim}")
    fields = SimpleNamespace(**{key: int(run[key]) for key in MODEL_KEYS})
    total = int(schedule["total_steps"])
    cooldown = int(schedule["cooldown_steps"])
    return {
        "path": str(path), "tag": run["tag"], "scale": scale, "condition": run["condition"],
        "seed": int(run["seed"]), "data_seed": int(run["data_seed"]),
        "tokens_per_step": int(run["batch_rows"]) * int(run["seq_len"]),
        "active": reference_active(fields), "fields": fields,
        "steps": total, "warmup": int(schedule["warmup_steps"]), "cooldown": cooldown,
        "stable_end": total - cooldown,
        "boundary": math.inf if run["condition"] == "n" else int(schedule["recurrence_boundary"]),
        "train": {s: float(r["pass1"]) for s, r in steps.items()},
        "val": {s: float(r["val"]) for s, r in evals.items()},
        "final": float(done[max(done)]["val"]) if done else float(evals[max(evals)]["val"]),
        "complete": bool(done),
    }


def tokens(run: dict, step) -> np.ndarray:
    """Predicted tokens through *step*, in billions."""
    return np.asarray(step, dtype=float) * run["tokens_per_step"] / 1e9


def steps_for(active: int, tokens_per_step: int, tokens_per_param: float) -> int:
    return math.ceil(Fraction(str(tokens_per_param)) * active / tokens_per_step)


# -- fits ---------------------------------------------------------------------------


def power_fit(D: np.ndarray, y: np.ndarray, betas: np.ndarray = BETA_GRID) -> tuple[float, float, float, float]:
    """Least-squares ``a + B·D^-β`` with β chosen on a grid: ``(a, B, β, rms)``."""
    best = None
    for beta in betas:
        X = np.stack([np.ones_like(D), D ** -beta], 1)
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        rms = float(np.sqrt(np.mean((X @ c - y) ** 2)))
        if best is None or rms < best[3]:
            best = (float(c[0]), float(c[1]), float(beta), rms)
    return best


def schedule_multiplier(steps: int, warmup: int, cooldown: int) -> np.ndarray:
    """The recipe's warmup-stable-cooldown learning-rate multiplier per step."""
    s = np.arange(1, steps + 1, dtype=float)
    m = np.ones(steps)
    m[:warmup] = s[:warmup] / warmup
    u = (s[steps - cooldown:] - (steps - cooldown)) / cooldown
    m[steps - cooldown:] = 1 - np.sqrt(u)
    return m


def annealing_areas(m: np.ndarray, decay: float) -> tuple[np.ndarray, np.ndarray]:
    """Tissue et al.'s forward area ``S1`` and annealing area ``S2`` with momentum *decay*."""
    S1 = np.cumsum(m)
    drops = np.concatenate([[0.0], np.clip(m[:-1] - m[1:], 0, None)])
    momentum = np.zeros_like(m)
    acc = 0.0
    for index, drop in enumerate(drops):
        acc = acc * decay + drop
        momentum[index] = acc
    return S1, np.cumsum(momentum)


def annealing_fit(S1: np.ndarray, S2: np.ndarray, y: np.ndarray, alphas: np.ndarray = ALPHA_GRID):
    """``L0 + A·S1^-α − C·S2`` with α on a grid; areas in thousands of steps."""
    best = None
    for alpha in alphas:
        X = np.stack([np.ones_like(S1), (S1 / 1000) ** -alpha, -S2 / 1000], 1)
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        rms = float(np.sqrt(np.mean((X @ c - y) ** 2)))
        if best is None or rms < best[4]:
            best = (float(c[0]), float(c[1]), float(alpha), float(c[2]), rms)
    return best


def annealing_predict(fit, steps: int, warmup: int, decay: float) -> float:
    L0, A, alpha, C, _ = fit
    S1, S2 = annealing_areas(schedule_multiplier(steps, warmup, round(0.2 * steps)), decay)
    return L0 + A * (S1[-1] / 1000) ** -alpha - C * S2[-1] / 1000


def separable_solve(anchors: list[tuple[float, float, float]], alpha: float, beta: float):
    """``E + A·N^-α + B·D^-β`` through ``(N, D, L)`` anchors (N in 1e8 parameters,
    D in billions of tokens); ``None`` when the anchors do not span both axes."""
    N, D, L = (np.array(column, dtype=float) for column in zip(*anchors, strict=True))
    X = np.stack([np.ones_like(N), N ** -alpha, D ** -beta], 1)
    if np.linalg.matrix_rank(X) < 3:
        return None
    c = np.linalg.lstsq(X, L, rcond=None)[0]
    return tuple(float(v) for v in c) + (float(np.sqrt(np.mean((X @ c - L) ** 2))),)


def separable_predict(coefficients, N: float, D: float, alpha: float, beta: float) -> float:
    E, A, B, _ = coefficients
    return E + A * N ** -alpha + B * D ** -beta


# -- analyses -----------------------------------------------------------------------


def stable_window(run: dict, from_tokens: float) -> np.ndarray:
    steps = np.array(sorted(run["val"]))
    return steps[(tokens(run, steps) >= from_tokens) & (steps <= run["stable_end"])]


def stable_fit(run: dict, from_tokens: float) -> dict:
    steps = stable_window(run, from_tokens)
    y = np.array([run["val"][s] for s in steps])
    a, B, beta, rms = power_fit(tokens(run, steps), y)
    extrapolated = a + B * tokens(run, run["steps"]) ** -beta
    return {
        "window": [int(steps[0]), int(steps[-1])], "a": a, "B": B, "beta": beta, "rms": rms,
        "stable_at_end": float(extrapolated), "bonus": float(extrapolated - run["final"]),
    }


def pair_analysis(a: dict, b: dict, window: int) -> dict | None:
    """Step-matched ``b − a`` where both runs read the same batches."""
    if a["data_seed"] != b["data_seed"] or a["tokens_per_step"] != b["tokens_per_step"]:
        return None
    shared = sorted(set(a["train"]) & set(b["train"]))
    if len(shared) < 2 * window:
        return None
    gap = np.array([b["train"][s] - a["train"][s] for s in shared])
    steps = np.array(shared)
    plain_end = min(a["boundary"], b["boundary"], a["stable_end"], b["stable_end"])
    plain = (steps >= 8 * max(a["warmup"], b["warmup"])) & (steps <= plain_end)
    out = {"a": a["tag"], "b": b["tag"], "same_scale": a["scale"] == b["scale"], "windows": []}
    for lo in range(int(steps[0]), int(steps[-1]), window):
        sel = (steps >= lo) & (steps < lo + window)
        if sel.sum() < window // 2:
            continue
        note = []
        for run, name in ((a, "a"), (b, "b")):
            if lo + window > run["stable_end"]:
                note.append(f"{name} cooling")
            elif lo + window > run["boundary"]:
                note.append(f"{name} recurrent")
        out["windows"].append({
            "steps": [lo, min(lo + window, int(steps[-1]))], "train_gap": float(gap[sel].mean()),
            "se": float(gap[sel].std() / np.sqrt(sel.sum())), "note": ", ".join(note),
        })
    common = sorted(set(a["val"]) & set(b["val"]))
    val_gap = {s: b["val"][s] - a["val"][s] for s in common}
    out["val_gap"] = {str(s): float(g) for s, g in val_gap.items()}
    if out["same_scale"] and plain.any():
        window_vals = [val_gap[s] for s in common if plain_end >= s >= 8 * max(a["warmup"], b["warmup"])]
        out["seed_offset"] = {
            "window": [int(steps[plain][0]), int(plain_end)], "train": float(gap[plain].mean()),
            "val": float(np.mean(window_vals)) if window_vals else math.nan,
            "val_std": float(np.std(window_vals)) if window_vals else math.nan,
        }
        cooling = [s for s in common if (s > a["stable_end"]) != (s > b["stable_end"])]
        if cooling:
            s = max(cooling)
            sign = 1 if s > a["stable_end"] else -1
            out["direct_bonus"] = {
                "step": int(s), "cooling": a["tag"] if sign > 0 else b["tag"],
                "val": float(sign * (val_gap[s] - out["seed_offset"]["val"])),
            }
    return out


def seed_corrections(runs: list[dict], pairs: list[dict], basis: int) -> dict[str, float]:
    """Each run's plain-window val offset from a basis-seed run at its scale."""
    offsets = {}
    for run in runs:
        if run["seed"] == basis:
            offsets[run["tag"]] = 0.0
            continue
        for pair in pairs:
            names = (pair["a"], pair["b"])
            if run["tag"] not in names or "seed_offset" not in pair:
                continue
            other = next(r for r in runs if r["tag"] in names and r is not run)
            if other["seed"] == basis:
                offsets[run["tag"]] = pair["seed_offset"]["val"] * (1 if pair["b"] == run["tag"] else -1)
                break
    return offsets


def downstream_pairs(runs: list[dict]) -> list[dict]:
    """Paired ``standard`` comparisons of every later run against the first."""
    stored = {}
    for run in runs:
        path = result_path(run["tag"], run["steps"], "standard")
        if path.is_file():
            stored[run["tag"]] = downstream.from_json(json.loads(path.read_text()))
    if runs[0]["tag"] not in stored:
        return []
    reference = stored[runs[0]["tag"]]
    out = []
    for run in runs[1:]:
        if run["tag"] not in stored:
            continue
        comparison = downstream.compare(reference, stored[run["tag"]])
        out.append({"a": runs[0]["tag"], "b": run["tag"], "comparison": comparison, "pooled": downstream.pooled(comparison)})
    return out


# -- report -------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=Path("figures") / "scaling")
    parser.add_argument("--stable-from", type=float, default=1.3, help="first stable-fit token count, billions (default: 1.3)")
    parser.add_argument("--window", type=int, default=500, help="step window for step-matched train gaps")
    parser.add_argument("--seed-basis", type=int, help="seed the anchors are corrected onto (default: the first log's)")
    parser.add_argument("--alpha", type=float, default=0.3, help="central parameter exponent")
    parser.add_argument("--beta", type=float, default=0.4, help="central token exponent")
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=(0.25, 0.3, 0.35))
    parser.add_argument("--beta-grid", type=float, nargs="+", default=(0.3, 0.4, 0.5, 0.7))
    parser.add_argument("--tokens-per-param", type=float, nargs="+", default=(25, 50, 100, 400))
    parser.add_argument("--decay", type=float, nargs="+", default=(0.99, 0.995), help="annealing-law momentum decays")
    args = parser.parse_args()
    runs = [parse_log(path) for path in args.logs]
    basis = runs[0]["seed"] if args.seed_basis is None else args.seed_basis
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"runs": [], "pairs": [], "annealing": [], "anchors": [], "table": {}}

    print("== runs and stable-phase fits: a + B·D^-β on val, extrapolated to the final step ==")
    fits = {}
    for run in runs:
        fits[run["tag"]] = fit = stable_fit(run, args.stable_from)
        state = "" if run["complete"] else " (incomplete)"
        print(
            f"  {run['tag']}: {run['scale']} {run['condition']} seed {run['seed']} data_seed {run['data_seed']}, "
            f"{run['steps']} steps = {tokens(run, run['steps']):.2f}B tokens = {run['steps'] * run['tokens_per_step'] / run['active']:.1f} tok/param, "
            f"final val {run['final']:.4f}{state}"
        )
        print(
            f"    window {fit['window'][0]}-{fit['window'][1]}: a={fit['a']:.3f} B={fit['B']:.3f} β={fit['beta']:.3f} rms={fit['rms']:.4f}; "
            f"stable curve at the end {fit['stable_at_end']:.3f} → cooldown bonus {fit['bonus']:.3f}"
        )
        report["runs"].append({k: run[k] for k in ("tag", "scale", "condition", "seed", "data_seed", "steps", "active", "final", "complete")} | {"fit": fit})

    pairs = [p for i, a in enumerate(runs) for b in runs[i + 1:] if (p := pair_analysis(a, b, args.window))]
    for pair in pairs:
        print(f"\n== step-matched gap {pair['b']} − {pair['a']} (identical batches) ==")
        for w in pair["windows"]:
            print(f"  steps {w['steps'][0]:6d}-{w['steps'][1]:6d}: train {w['train_gap']:+.4f} ± {w['se']:.4f}  {w['note']}")
        if "seed_offset" in pair:
            o = pair["seed_offset"]
            print(f"  seed offset over the plain window {o['window'][0]}-{o['window'][1]}: val {o['val']:+.4f} (std {o['val_std']:.4f}), train {o['train']:+.4f}")
        if "direct_bonus" in pair:
            d = pair["direct_bonus"]
            print(f"  direct cooldown bonus at step {d['step']} ({d['cooling']} cooling, the other stable), seed-corrected: {d['val']:.4f}")
        report["pairs"].append(pair)

    print(f"\n== annealing law L0 + A·S1^-α − C·S2 (Tissue et al.), evals from 8×warmup; predictions on seed {basis} ==")
    offsets = seed_corrections(runs, pairs, basis)
    for run in runs:
        if run["tag"] not in offsets:
            print(f"  {run['tag']}: seed {run['seed']} has no basis-seed run at its scale; anchored uncorrected")
            offsets[run["tag"]] = 0.0
    by_scale = {}
    for run in runs:
        by_scale.setdefault(run["scale"], []).append(run)
    for decay in args.decay:
        for scale, group in by_scale.items():
            designs = []
            for run in group:
                S1, S2 = annealing_areas(schedule_multiplier(run["steps"], run["warmup"], run["cooldown"]), decay)
                steps = np.array([s for s in sorted(run["val"]) if s >= 8 * run["warmup"]])
                designs.append((S1[steps - 1], S2[steps - 1], np.array([run["val"][s] for s in steps]) - offsets[run["tag"]]))
            fitted = [(run["tag"], annealing_fit(*d)) for run, d in zip(group, designs, strict=True)]
            if len(group) > 1:
                fitted.append(("+".join(r["tag"] for r in group), annealing_fit(*(np.concatenate(c) for c in zip(*designs, strict=True)))))
            active, warmup, per_step = group[0]["active"], group[0]["warmup"], group[0]["tokens_per_step"]
            for name, fit in fitted:
                preds = {R: annealing_predict(fit, steps_for(active, per_step, R), warmup, decay) for R in args.tokens_per_param}
                cross = "; ".join(
                    f"{r['tag']} {annealing_predict(fit, r['steps'], r['warmup'], decay):.3f} vs {r['final'] - offsets[r['tag']]:.3f}"
                    for r in group if r["tag"] != name
                )
                print(
                    f"  decay {decay} {name}: L0={fit[0]:.3f} A={fit[1]:.3f} α={fit[2]:.3f} C={fit[3]:.3f} rms={fit[4]:.4f} → "
                    + " ".join(f"{R:g}x {v:.3f}" for R, v in preds.items()) + (f"  [cross: {cross}]" if cross else "")
                )
                report["annealing"].append({"decay": decay, "fit": name, "scale": scale, "coefficients": fit[:4], "rms": fit[4], "predictions": {str(R): v for R, v in preds.items()}})

    anchors = [(run["active"] / 1e8, float(tokens(run, run["steps"])), run["final"] - offsets[run["tag"]]) for run in runs if run["complete"]]
    report["anchors"] = [{"tag": run["tag"], "N": a[0], "D": a[1], "L": a[2]} for run, a in zip([r for r in runs if r["complete"]], anchors, strict=True)]
    central = separable_solve(anchors, args.alpha, args.beta)
    if central is None:
        print("\n== annealed anchors span one axis only; no separable table ==")
    else:
        print(f"\n== annealed finals on seed {basis} anchor E + A·N^-α + B·D^-β; central α={args.alpha} β={args.beta}, range over the grids ==")
        print(f"  central E={central[0]:.3f} rms={central[3]:.4f}; anchors " + ", ".join(f"{a['tag']} {a['L']:.4f}" for a in report["anchors"]))
        grid = [(al, be, separable_solve(anchors, al, be)) for al in args.alpha_grid for be in args.beta_grid]
        base = runs[0]["fields"]
        scales = {name: reference_active(SimpleNamespace(**(vars(base) | {k: v for k, v in geo.items() if k in MODEL_KEYS}))) for name, geo in SCALES.items()}
        per_step = runs[0]["tokens_per_step"]
        anchored = {(run["scale"], round(run["steps"] * per_step / run["active"])) for run in runs if run["complete"]}
        print(f"  {'scale':10s}" + "".join(f"{R:>8g}x{'':13s}" for R in args.tokens_per_param))
        for scale, active in scales.items():
            cells = []
            for R in args.tokens_per_param:
                D = steps_for(active, per_step, R) * per_step / 1e9
                value = separable_predict(central, active / 1e8, D, args.alpha, args.beta)
                spread = [separable_predict(c, active / 1e8, D, al, be) for al, be, c in grid if c is not None]
                mark = "*" if (scale, R) in anchored else " "
                cells.append(f"{value:.3f} [{min(spread):.3f}-{max(spread):.3f}]{mark}")
                report["table"][f"{scale}|{R:g}"] = {"central": value, "low": min(spread), "high": max(spread), "tokens": D}
            print(f"  {scale:10s}" + " ".join(cells))
        print("  * anchored on a measured final")

    pairs_ds = downstream_pairs(runs)
    for pair in pairs_ds:
        print(f"\n== downstream standard, {pair['b']} − {pair['a']}, paired per document ==")
        print(downstream.format_comparison(pair["comparison"], pair["a"], pair["b"]))

    panels = 2 + bool(pairs_ds)
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 4.6), constrained_layout=True)
    colors = {name: fs.SERIES[i % len(fs.SERIES)] for i, name in enumerate(SCALES)}
    for index, run in enumerate(runs):
        color = fs.SERIES[index % len(fs.SERIES)]
        steps = np.array(sorted(run["val"]))
        axes[0].plot(tokens(run, steps), [run["val"][s] for s in steps], color=color, label=run["tag"])
        fit = fits[run["tag"]]
        x = np.linspace(tokens(run, fit["window"][0]), tokens(run, run["steps"]), 100)
        axes[0].plot(x, fit["a"] + fit["B"] * x ** -fit["beta"], ":", color=color, linewidth=1)
        axes[0].annotate(f"bonus {fit['bonus']:.3f}", (tokens(run, run["steps"]), run["final"]), textcoords="offset points", xytext=(-52, 8), fontsize=8, color=color)
    top = max(run["val"][fits[run["tag"]]["window"][0]] for run in runs)
    axes[0].set(title="Validation CE, stable-phase fit dotted", xlabel="predicted tokens (B)", ylabel="val", ylim=(min(r["final"] for r in runs) - 0.05, top + 0.08))
    axes[0].legend()
    if central is not None:
        for scale, active in scales.items():
            R = np.geomspace(min(args.tokens_per_param), max(args.tokens_per_param), 40)
            D = np.array([steps_for(active, per_step, float(r)) * per_step / 1e9 for r in R])
            band = np.array([[separable_predict(c, active / 1e8, d, al, be) for al, be, c in grid if c is not None] for d in D])
            axes[1].fill_between(D, band.min(1), band.max(1), color=colors[scale], alpha=0.18)
            axes[1].plot(D, [separable_predict(central, active / 1e8, d, args.alpha, args.beta) for d in D], color=colors[scale], label=scale)
        axes[1].plot([a["D"] for a in report["anchors"]], [a["L"] for a in report["anchors"]], "k*", markersize=10, label="measured", zorder=5)
        axes[1].set(xscale="log", title=f"Predicted final val; line α={args.alpha} β={args.beta}, band over the grids", xlabel="predicted tokens (B)", ylabel="val")
        axes[1].legend()
    if pairs_ds:
        names = sorted({t for pair in pairs_ds for t in pair["comparison"]})
        y = np.arange(len(names))
        width = 0.8 / len(pairs_ds)
        for index, pair in enumerate(pairs_ds):
            diffs = [100 * pair["comparison"][t][downstream.headline_metric(pair["comparison"][t])]["diff"] if t in pair["comparison"] else 0 for t in names]
            errors = [100 * pair["comparison"][t][downstream.headline_metric(pair["comparison"][t])]["se"] if t in pair["comparison"] else 0 for t in names]
            axes[2].barh(y + (index - (len(pairs_ds) - 1) / 2) * width, diffs, width, xerr=errors, color=fs.SERIES[(index + 1) % len(fs.SERIES)], label=f"{pair['b']} − {pair['a']}")
        axes[2].set(yticks=y, yticklabels=names, title="Downstream standard, paired accuracy difference (pp)")
        axes[2].axvline(0, color=fs.INK, linewidth=0.6)
        axes[2].invert_yaxis()
        axes[2].legend()
    fs.save(fig, args.out_dir / "scaling-curves.png")
    (args.out_dir / "scaling_curves.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"\n{args.out_dir}")


if __name__ == "__main__":
    main()

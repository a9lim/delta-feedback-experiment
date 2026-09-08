"""Figures from the checkpoint-analysis JSON outputs.

Each input is optional; the script draws whatever it is given:

    --compare   compare_arms.json         (scripts/compare_arms.py)
    --weights   weight_divergence.json    (scripts/weight_divergence.py)
    --fused     fused_diagnostics.json    (scripts/fused_diagnostics.py)
    --entry     entry_sweeps.json         (scripts/entry_sweeps.py)
    --followups feedback_followups.json   (scripts/feedback_followups.py)
    --swap      payload_swap.json         (scripts/payload_swap.py)
    --downstream A.json B.json ...        (scripts/downstream_eval.py; one bar group per run)
    --downstream-modes standard.json soft.json fused.json ...   (paired gains against the first file)

Usage:
    python scripts/analysis_figures.py --compare figures/compare-A-vs-B/compare_arms.json \\
        --fused figures/fused-B/fused_diagnostics.json --out-dir figures/compare-A-vs-B

Every panel is observational or a same-checkpoint intervention; none is a
training-effect estimate or a causal finding on its own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import figstyle as fs
import matplotlib.pyplot as plt
import numpy as np

REF, FB1, FB2 = fs.BLUE, fs.ORANGE, fs.AQUA  # reference pass 1, feedback pass 1, feedback fused


def human(x: float) -> str:
    """Compact tick text: 0.35, 12, 1.2k, 1.2M."""
    if abs(x) >= 1e6:
        return f"{x / 1e6:.3g}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.3g}k"
    return f"{x:.2g}"


def range_labels(rows: list[dict], fmt=None) -> list[str]:
    return [f"{human(r['range'][0])}–{human(r['range'][1])}" for r in rows]


def conditioning_panel(ax, rows, labels, title, xlabel, keys=("d_df1_mhdb", "d_fused_df1"), names=("feedback pass 1 − reference", "fused − feedback pass 1")):
    x = np.arange(len(rows))
    for key, name, col in zip(keys, names, (FB1, FB2)):
        ax.plot(x, [r[key] for r in rows], "o-", color=col, label=name)
    fs.zero_line(ax)
    ax.set_xticks(x, range_labels(rows), rotation=35, ha="right", fontsize=7)
    ax.set(title=title, xlabel=xlabel, ylabel="CE difference")
    ax.legend()


def compare_figures(report: dict, out_dir: Path) -> None:
    L = report.get("labels", {"m": "reference pass 1", "d1": "feedback pass 1", "d2": "feedback fused"})
    M = report["means"]
    curve = report["position_curve"]
    x = np.array([c["pos_lo"] for c in curve]) + 16

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(x, [c["mhdb"] for c in curve], color=REF, label=L["m"])
    ax.set(title="Reference loss by position", xlabel="position in row (32-wide bins)", ylabel="held-out CE", xscale="log")
    ax.legend()
    for ax, key, col, title in ((axes[1], "d_df1_mhdb", FB1, f"{L['d1']} − {L['m']}"), (axes[2], "d_fused_df1", FB2, f"{L['d2']} − {L['d1']}")):
        y = np.array([c[key] for c in curve])
        se = np.array([c.get(f"{key}_se", 0.0) for c in curve])
        ax.fill_between(x, y - se, y + se, color=col, alpha=0.18, lw=0)
        ax.plot(x, y, "o-", color=col, ms=3)
        fs.zero_line(ax)
        ax.set(title=title, xlabel="position in row (32-wide bins)", ylabel="CE difference (±1 s.e.)", xscale="log")
        mean = M["delta_df1_minus_mhdb"] if key == "d_df1_mhdb" else M["delta_fused_minus_df1"]
        ax.axhline(mean, color=col, lw=0.9, ls=":")
        ax.text(x[0], mean, f" mean {mean:+.4f}", color=fs.SECONDARY, fontsize=7, va="bottom")
    fs.save(fig, out_dir / "pair-position.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), constrained_layout=True)
    names = (f"{L['d1']} − {L['m']}", f"{L['d2']} − {L['d1']}")
    conditioning_panel(axes[0, 0], report["by_target_token_frequency"], L, "By target-token frequency", "target-token count in the held-out slice (deciles)", names=names)
    conditioning_panel(axes[0, 1], report["by_input_token_frequency"], L, "By input-token frequency", "input-token count in the held-out slice (deciles)", names=names)
    conditioning_panel(axes[1, 0], report["by_prev_surprise"], L, "By the reference's surprise at the fused-in token", "reference −log p of the current input token (quantiles)", names=names)
    conditioning_panel(axes[1, 1], report["by_mhdb_entropy"], L, "By the reference's predictive entropy", "reference entropy (quantiles)", names=names)
    fs.save(fig, out_dir / "pair-conditioning.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    ax = axes[0]
    pairs = [("KL(ref ‖ fb pass 1)", M["kl_mhdb_to_df1"], M["argmax_agree_mhdb_df1"]), ("KL(fb pass 1 ‖ fused)", M["kl_df1_to_fused"], M["argmax_agree_df1_fused"]),
             ("KL(ref ‖ fused)", M["kl_mhdb_to_fused"], None)]
    ax.bar(range(3), [p[1] for p in pairs], color=[FB1, FB2, fs.MAGENTA], width=0.6)
    for i, (_, kl, agree) in enumerate(pairs):
        ax.text(i, kl, f"{kl:.3f}" + (f"\nargmax agree {agree:.1%}" if agree is not None else ""), ha="center", va="bottom", fontsize=7, color=fs.SECONDARY)
    ax.set_xticks(range(3), [p[0] for p in pairs], fontsize=8)
    ax.set(title="How different are the predictors?", ylabel="mean KL (nats)")
    ax.set_ylim(0, max(p[1] for p in pairs) * 1.35)
    ax = axes[1]
    for key, col, name in (("m+d1", FB1, f"{L['m']} + {L['d1']}"), ("d1+d2", FB2, f"{L['d1']} + {L['d2']}"), ("m+d2", fs.MAGENTA, f"{L['m']} + {L['d2']}")):
        e = report["ensembles"][key]
        ws = sorted(float(w) for w in e["test_by_w"])
        ax.plot(ws, [e["test_by_w"][f"{w:.1f}"] for w in ws], "o-", color=col, label=f"{name} (gain {e['test_gain_over_better_single']:+.4f})")
    ax.set(title="Probability mixtures, scored on held-out test rows", xlabel="weight on the second predictor", ylabel="test CE")
    ax.legend()
    ax = axes[2]
    for key, col, name in (("df1_minus_mhdb", FB1, names[0]), ("fused_minus_df1", FB2, names[1])):
        q = report["delta_quantiles"][key]
        ps = [int(k[1:]) for k in q]
        ax.plot(ps, list(q.values()), "o-", color=col, label=name)
    fs.zero_line(ax)
    ax.set(title="Per-token CE difference, quantiles", xlabel="percentile of tokens", ylabel="CE difference")
    ax.legend()
    fs.save(fig, out_dir / "pair-predictors.png")

    rep = report["representation"]
    sources = list(rep)
    x = np.arange(len(sources))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    ax = axes[0]
    w = 0.26
    for i, (key, col, name) in enumerate((("cka_m_d1", FB1, f"{L['m']} vs {L['d1']}"), ("cka_d1_d2", FB2, f"{L['d1']} vs {L['d2']}"), ("cka_m_d2", fs.MAGENTA, f"{L['m']} vs {L['d2']}"))):
        ax.bar(x + (i - 1) * w, [rep[s][key] for s in sources], w, color=col, label=name)
    ax.set_xticks(x, sources)
    ax.set(title="Linear CKA of residual-stream sources on the same tokens", ylabel="CKA", ylim=(0, 1.05))
    ax.legend(loc="lower right")
    ax = axes[1]
    for key, col, name in (("cos_m_d1", FB1, f"{L['m']} vs {L['d1']}"), ("cos_d1_d2", FB2, f"{L['d1']} vs {L['d2']}")):
        ax.plot(x, [rep[s][key] for s in sources], "o-", color=col, label=name)
    fs.zero_line(ax)
    ax.set_xticks(x, sources)
    ax.set(title="Per-token cosine between the same source", ylabel="mean cosine", ylim=(-0.1, 1.05))
    ax.legend()
    ax = axes[2]
    for key, col, name in (("rms_m", REF, L["m"]), ("rms_d1", FB1, L["d1"]), ("rms_d2", FB2, L["d2"])):
        ax.plot(x, [rep[s][key] for s in sources], "o-", color=col, label=name)
    ax.set_xticks(x, sources)
    ax.set(title="Source scale", ylabel="RMS per coordinate", yscale="log")
    ax.legend()
    fs.save(fig, out_dir / "pair-representation.png")

    red = report["payload_redundancy"]
    items = [(k, v) for k, v in red.items() if k.startswith("r2_")]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(items) + 1.2), constrained_layout=True)
    y = np.arange(len(items))
    ax.barh(y, [v for _, v in items], color=FB1, height=0.6)
    for i, (_, v) in enumerate(items):
        ax.text(max(v, 0) + 0.01, i, f"{v:.3f}", va="center", fontsize=8, color=fs.SECONDARY)
    ax.set_yticks(y, [k[3:].replace("_", " ") for k, _ in items], fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    ax.set(title="Ridge R² (fit on half the sampled tokens, scored on the other half)", xlabel="R²", xlim=(0, 1.08))
    fs.save(fig, out_dir / "pair-redundancy.png")


def weight_figures(report: dict, out_dir: Path) -> None:
    labels = report.get("labels", {"a": "a", "b": "b"})
    grouped = report["grouped"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), constrained_layout=True)
    for ax, fam, title in ((axes[0, 0], "attn_matrix", "Token-mixer matrices"), (axes[0, 1], "mlp_matrix", "SwiGLU matrices")):
        rows = sorted((e for e in grouped.values() if e["family"] == fam and e["layer"] is not None), key=lambda e: e["layer"])
        layers = [e["layer"] for e in rows]
        ax.plot(layers, [e["angle_deg_a_0"] for e in rows], "o-", color=REF, label=f"{labels['a']} from init")
        ax.plot(layers, [e["angle_deg_b_0"] for e in rows], "s-", color=FB1, label=f"{labels['b']} from init")
        ax.plot(layers, [e["angle_deg_a_b"] for e in rows], "^-", color=fs.MAGENTA, label="between the two runs")
        ax.set(title=f"{title}: angular distance", xlabel="layer", ylabel="angle (degrees)", ylim=(0, 95))
        ax.legend(loc="lower right")
    fam_rows = sorted(report["families"].items(), key=lambda t: -t[1]["numel"])
    names = [k for k, _ in fam_rows]
    x = np.arange(len(names))
    ax = axes[1, 0]
    w = 0.27
    ax.bar(x - w, [e["move_a"] for _, e in fam_rows], w, color=REF, label=f"{labels['a']} moved")
    ax.bar(x, [e["move_b"] for _, e in fam_rows], w, color=FB1, label=f"{labels['b']} moved")
    ax.bar(x + w, [e["gap"] for _, e in fam_rows], w, color=fs.MAGENTA, label="gap between runs")
    ax.set_xticks(x, names, rotation=35, ha="right", fontsize=8)
    ax.set(title="Displacement relative to ‖W₀‖ by parameter family", ylabel="‖ΔW‖ / ‖W₀‖", yscale="log")
    ax.legend()
    ax = axes[1, 1]
    ax.bar(x, [e["cos_upd"] for _, e in fam_rows], 0.6, color=fs.VIOLET)
    ax.set_xticks(x, names, rotation=35, ha="right", fontsize=8)
    ax.set(title="Alignment of the two runs' total updates", ylabel="cos(Wₐ − W₀, W_b − W₀)", ylim=(0, 1.05))
    fs.save(fig, out_dir / "weight-divergence.png")


def fused_figures(report: dict, out_dir: Path) -> None:
    rows = report["position_bins"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(rows))
    w = 0.27
    ax.bar(x - w, [r["delta"] for r in rows], w, color=FB2, label="fused, prefix 1", yerr=[r.get("delta_se", 0) for r in rows], ecolor=fs.MUTED, capsize=2)
    ax.bar(x, [r["delta_prefix512"] for r in rows], w, color=fs.YELLOW, label="fused, plain prefix 512")
    ax.bar(x + w, [r["delta_iter2"] for r in rows], w, color=fs.MAGENTA, label="two fused iterations")
    fs.zero_line(ax)
    ax.set_xticks(x, [r["bin"] for r in rows])
    ax.set(title="Fused − pass 1 by position", xlabel="position in row", ylabel="CE difference")
    ax.legend()
    ax = axes[1]
    first = report["first_positions"]
    ax.plot([r["pos"] for r in first], [r["delta"] for r in first], "o-", color=FB2)
    fs.zero_line(ax)
    ax.set(title="Fused − pass 1 at the first 16 positions", xlabel="position", ylabel="CE difference")
    fs.save(fig, out_dir / "fused-position.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    ax = axes[0]
    rows = report["by_prev_surprise"]
    x = np.arange(len(rows))
    ax.plot(x, [r["delta"] for r in rows], "o-", color=FB2, label="fused − pass 1")
    ax.plot(x, [r["u_nearest_hit"] / 10 for r in rows], "s--", color=fs.MUTED, lw=1.1, label="nearest-token hit of the fused seed ÷ 10")
    fs.zero_line(ax)
    ax.set_xticks(x, range_labels(rows), rotation=35, ha="right", fontsize=7)
    ax.set(title="By the previous column's surprise at the fused-in token", xlabel="−log p (quantiles)", ylabel="CE difference")
    ax.legend()
    ax = axes[1]
    rows = report["by_input_token_frequency"]
    x = np.arange(len(rows))
    ax.plot(x, [r["delta"] for r in rows], "o-", color=FB2)
    fs.zero_line(ax)
    ax.set_xticks(x, range_labels(rows), rotation=35, ha="right", fontsize=7)
    ax.set(title="By input-token frequency", xlabel="input-token count in the held-out slice (deciles)", ylabel="CE difference")
    ax = axes[2]
    rows = report["by_own_ce1"]
    x = np.arange(len(rows))
    ax.plot(x, [r["delta"] for r in rows], "o-", color=FB2)
    fs.zero_line(ax)
    ax.set_xticks(x, range_labels(rows), rotation=35, ha="right", fontsize=7)
    ax.set(title="By own pass-1 CE (selects on pass-1 noise: regression to the mean)", xlabel="pass-1 CE (quantiles)", ylabel="CE difference")
    fs.save(fig, out_dir / "fused-conditioning.png")

    sc = report["self_composition"]
    trace = sc["trace"]
    its = [r["iter"] for r in trace]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    ax = axes[0]
    ax.plot(its, [r["loss"] for r in trace], "o-", color=FB2, ms=3, label="fully fused, iterated")
    ax.axhline(sc["pass1"], color=FB1, lw=1, ls="--", label="pass 1")
    ax.set(title=f"Self-composition loss ({sc['rows']} rows)", xlabel="fused prefill iteration", ylabel="held-out CE")
    ax.legend()
    ax = axes[1]
    ax.plot(its, [r["update_rel_rms"] for r in trace], "o-", color=FB2, ms=3, label="top state")
    ax.plot(its, [r["payload_update_rms"] for r in trace], "s-", color=fs.MAGENTA, ms=3, label="payload")
    ax.set(title="Relative update per iteration", xlabel="fused prefill iteration", ylabel="RMS(update) / RMS(previous)", yscale="log")
    ax.legend()
    fs.save(fig, out_dir / "fused-self-composition.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), constrained_layout=True)
    ax = axes[0]
    g = report["gate"]
    for key, col, name in (("token_mean_quantiles", FB2, "token-mean gate"), ("token_dependent_share_of_prenorm_input_quantiles", fs.MAGENTA, "token-dependent share of the pre-norm fused input")):
        q = g[key]
        ax.plot([int(k[1:]) for k in q], list(q.values()), "o-", color=col, label=name)
    ax.set(title="FBT entry gate", xlabel="percentile of tokens", ylabel="value", ylim=(0, 1))
    ax.legend()
    ax = axes[1]
    q = g["logit_rms_quantiles"]
    ax.plot([int(k[1:]) for k in q], list(q.values()), "o-", color=FB2)
    ax.set(title="Gate-logit RMS per token", xlabel="percentile of tokens", ylabel="RMS of W_G norm(e)")
    ax = axes[2]
    if "per_source_rel_change_pass2_vs_pass1" in report:
        src = report["per_source_rel_change_pass2_vs_pass1"]
        ax.bar(range(len(src)), list(src.values()), 0.6, color=FB2)
        for i, v in enumerate(src.values()):
            ax.text(i, v, f"{v:.2f}×", ha="center", va="bottom", fontsize=8, color=fs.SECONDARY)
        ax.set_xticks(range(len(src)), list(src))
        ax.set(title="Pass-2 change of each source, relative to its pass-1 RMS", ylabel="‖Δ‖ / RMS", yscale="log")
    fs.save(fig, out_dir / "fused-gate.png")


def entry_figures(report: dict, out_dir: Path) -> None:
    pen = {k: v for k, v in report["penalty_vs_pass1"].items() if k != "pass1"}
    names = list(pen)
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(names) + 1.4), constrained_layout=True)
    y = np.arange(len(names))
    ax.barh(y, [max(v, 1e-4) for v in pen.values()], color=[FB2 if k == "fused" else fs.VIOLET for k in names], height=0.6)
    for i, v in enumerate(pen.values()):
        ax.text(max(v, 1e-4) * 1.15, i, f"{v:+.4f}", va="center", fontsize=8, color=fs.SECONDARY)
    ax.set_yticks(y, [n.replace("_", " ") for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", which="both")
    ax.grid(False, axis="y")
    ax.set(title=f"Entry interventions: CE penalty over pass 1 ({report['rows']} rows)", xlabel="CE penalty (nats, log scale)", xscale="log", xlim=(5e-4, 1e2))
    fs.save(fig, out_dir / "entry-interventions.png")


def followup_figures(report: dict, out_dir: Path) -> None:
    imp = report["impulse"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), constrained_layout=True)
    ax = axes[0]
    for L, col in zip(sorted(imp, key=int), (FB2, fs.YELLOW, fs.MAGENTA)):
        starts = imp[L]
        lags = sorted({int(k) for d in starts.values() for k in d}, key=int)
        mean = [np.nanmean([starts[t0][str(lag)] for t0 in starts]) for lag in lags]
        ax.plot(range(len(lags)), mean, "o-", color=col, label=f"{L} fused position{'s' if L != '1' else ''}")
    fs.zero_line(ax)
    ax.set_xticks(range(len(lags)), lags)
    ax.set(title="Fused block, then plain: penalty after the block", xlabel="lag after the last fused position", ylabel="CE penalty (mean over starts)")
    ax.legend()
    ax = axes[1]
    if "payload_head_ablation" in report:
        abl = report["payload_head_ablation"]
        base = report.get("impulse_fused_prefix1")
        names = list(abl)
        vals = [abl[k] - base for k in names] if base is not None else list(abl.values())
        ax.bar(range(len(names)), vals, 0.6, color=fs.VIOLET)
        ax.set_xticks(range(len(names)), [n.replace("_", " ") for n in names], rotation=25, ha="right", fontsize=8)
        fs.zero_line(ax)
        ax.set(title="Payload-head ablation" + (" (vs the trained fused pass)" if base is not None else ""), ylabel="CE difference" if base is not None else "CE")
    ax = axes[2]
    gc = report["gradient_conflict"]
    fams = [f for f in gc if "cosine" in gc[f]]
    ax.bar(range(len(fams)), [gc[f]["cosine"] for f in fams], 0.6, color=fs.VIOLET)
    ax.set_xticks(range(len(fams)), [f.replace("_", " ") for f in fams], rotation=25, ha="right", fontsize=8)
    ax.set(title="cos(∇ pass-1 loss, ∇ fused-pass loss) on fresh training rows", ylabel="cosine", ylim=(0, 1.05))
    fs.save(fig, out_dir / "feedback-followups.png")


def swap_figures(report: dict, out_dir: Path) -> None:
    items = [("pass 1", report["pass1"]), ("trained router", report["trained_router"]), ("h_top only", report["h_top_only"]), ("uniform", report["uniform"])]
    items += [(f"force {k}", v) for k, v in report["forced"].items()]
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(items) + 1.4), constrained_layout=True)
    y = np.arange(len(items))
    colors = [REF, FB2] + [fs.VIOLET] * (len(items) - 2)
    ax.barh(y, [v for _, v in items], color=colors, height=0.6)
    for i, (_, v) in enumerate(items):
        ax.text(v + 0.05, i, f"{v:.3f}", va="center", fontsize=8, color=fs.SECONDARY)
    ax.set_yticks(y, [k for k, _ in items], fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    head = report.get("head")
    ax.set(title=f"Payload enrichment swaps ({report['rows']} rows" + (f", head {head} only)" if head is not None else ")"), xlabel="fused CE, prefix 1")
    fs.save(fig, out_dir / ("payload-swap.png" if head is None else f"payload-swap-head{head}.png"))


CHANCE = {"hellaswag": 0.25, "arc_easy": 0.25, "arc_challenge": 0.25, "piqa": 0.5, "winogrande": 0.5, "boolq": 0.5,
          "openbookqa": 0.25, "sciq": 0.25, "lambada_openai": 0.0}


def downstream_figures(paths: list[Path], labels: list[str] | None, out_dir: Path) -> None:
    runs = [json.loads(p.read_text()) for p in paths]
    labels = labels or [f"{r['meta'].get('tag', p.stem)} {r['meta'].get('mode', '')}".strip() for r, p in zip(runs, paths)]
    tasks = [t for t in runs[0]["tasks"] if all(t in r["tasks"] for r in runs)]
    metric_of = {t: ("acc_norm" if "acc_norm" in runs[0]["tasks"][t]["metrics"] else "acc") for t in tasks}
    fig, ax = plt.subplots(figsize=(1.35 * len(tasks) + 2, 4.2), constrained_layout=True)
    x = np.arange(len(tasks))
    w = 0.8 / len(runs)
    for i, (run, label) in enumerate(zip(runs, labels)):
        means = [run["tasks"][t]["metrics"][metric_of[t]]["mean"] for t in tasks]
        ses = [run["tasks"][t]["metrics"][metric_of[t]]["se"] for t in tasks]
        ax.bar(x + (i - (len(runs) - 1) / 2) * w, means, w, yerr=ses, color=fs.SERIES[i % len(fs.SERIES)], ecolor=fs.MUTED, capsize=2, label=label)
    for j, t in enumerate(tasks):
        if CHANCE.get(t, 0) > 0:
            ax.plot([j - 0.42, j + 0.42], [CHANCE[t]] * 2, color=fs.MUTED, lw=1, ls=":")
    ax.set_xticks(x, [f"{t}\n({metric_of[t]})" for t in tasks], fontsize=8)
    ax.set(title="Downstream zero-shot tasks (dotted: chance; bars ±1 s.e.)", ylabel="accuracy", ylim=(0, 1))
    ax.legend()
    fs.save(fig, out_dir / "downstream-tasks.png")


def downstream_modes_figure(paths: list[Path], out_dir: Path) -> None:
    """Paired gains over the first file (Standard) for each later mode or pass count."""
    from transformer_experiments import downstream as ds

    payloads = [json.loads(p.read_text()) for p in paths]
    runs = [ds.from_json(d) for d in payloads]
    names = [f"{d['meta'].get('mode', p.stem)}" + (f" ×{d['meta']['passes']}" if d["meta"].get("passes", 1) > 1 else "") for d, p in zip(payloads, paths)]
    passes = list(range(len(runs)))
    comparisons = [ds.compare(runs[0], r) for r in runs[1:]]
    tasks = sorted(comparisons[0])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    ax = axes[0]
    pooled = [ds.pooled(c) for c in comparisons]
    ax.errorbar(passes, [0.0] + [100 * p["diff"] for p in pooled], yerr=[0.0] + [100 * p["se"] for p in pooled], fmt="o-", color=FB2, capsize=3, label="pooled over tasks (inverse-variance)")
    for t, col in (("lambada_openai", fs.MAGENTA), ("hellaswag", fs.YELLOW)):
        if t in tasks:
            key = ds.headline_metric(comparisons[0][t])
            ax.errorbar(passes, [0.0] + [100 * c[t][key]["diff"] for c in comparisons], yerr=[0.0] + [100 * c[t][key]["se"] for c in comparisons], fmt="s--", color=col, lw=1.2, capsize=3, label=f"{t} ({key})")
    fs.zero_line(ax)
    ax.set_xticks(passes, names)
    ax.set(title="Accuracy gain over Standard, paired on identical documents", ylabel="accuracy points (±1 s.e.)")
    ax.legend()
    ax = axes[1]
    for i, t in enumerate(tasks):
        vals = [0.0] + [c[t]["gold_logprob"]["diff"] for c in comparisons]
        ax.plot(passes, vals, "o-" if i < len(fs.SERIES) else "s--", color=fs.SERIES[i % len(fs.SERIES)], lw=1.2, ms=3, label=t)
    fs.zero_line(ax)
    ax.set_xticks(passes, names)
    ax.set(title="Gold-continuation log-probability shift over Standard", ylabel="nats per document")
    ax.legend(fontsize=7, ncols=2)
    fs.save(fig, out_dir / "downstream-modes.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--fused", type=Path)
    parser.add_argument("--entry", type=Path)
    parser.add_argument("--followups", type=Path)
    parser.add_argument("--swap", type=Path)
    parser.add_argument("--downstream", type=Path, nargs="*")
    parser.add_argument("--downstream-labels", nargs="*")
    parser.add_argument("--downstream-modes", type=Path, nargs="*")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.downstream:
        downstream_figures(args.downstream, args.downstream_labels, args.out_dir)
    if args.downstream_modes:
        downstream_modes_figure(args.downstream_modes, args.out_dir)
    for path, fn in ((args.compare, compare_figures), (args.weights, weight_figures), (args.fused, fused_figures),
                     (args.entry, entry_figures), (args.followups, followup_figures), (args.swap, swap_figures)):
        if path is not None:
            fn(json.loads(path.read_text()), args.out_dir)
    print(f"figures -> {args.out_dir}/")


if __name__ == "__main__":
    main()

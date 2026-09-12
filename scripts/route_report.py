"""Dissect a trained snapshot's routing: what each router learned.

Rebuilds the condition's model from a checkpoint and reads its routing components,
the per-layer attention readers, the per-layer MLP readers, and the payload
router, two ways:

- statically: query-vector geometry (norms set the effective softmax
  temperature since keys are RMS-normed; pairwise cosines say whether sites
  learned a shared reading direction);
- empirically: per-head mean routing distributions over held-out rows, on the
  plain pass and, for feedback conditions, on a fused pass (prefix length 1, the
  training eval convention), plus entropy and cross-head Jensen-Shannon
  divergence.

Writes ``route_report.json`` and figures under ``figures/route-TAG/`` and
prints a per-site table.

Usage:
    python scripts/route_report.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import figstyle as fs
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors

from delta_feedback_experiment import analysis
from delta_feedback_experiment.data import TokenData
from delta_feedback_experiment.model import multipass


def site_names(cfg) -> list[str]:
    """Every routing site in execution order; a core site under ``l`` is
    tagged by iteration (``L4i2.attn``) at the evaluation count."""
    names = []
    for layer in range(cfg.layers):
        labels = (
            [f"i{i}" for i in range(cfg.loop_iterations)]
            if cfg.is_core_layer(layer)
            else [""]
        )
        names += [f"L{layer}{label}.{kind}" for label in labels for kind in ("attn", "mlp")]
    return names


@torch.no_grad()
def collect(model, data_val, device, rows: int, micro: int):
    """Routing statistics per (pass, site) over `rows` val rows.

    Returns (mean weights [N,H], per-head token stats {max, H_tok},
    normalized cross-head JS) keyed by (pass, site).  Mean weights say
    *which* sources a site reads; the per-token mean max and mean normalized
    entropy say whether that read is static wiring (sharp and identical
    everywhere) or token-dependent (sharp per token, varied across tokens).
    """
    sums: dict[tuple[int, str], torch.Tensor] = {}
    stats: dict[tuple[int, str], torch.Tensor] = {}
    divergences: dict[tuple[int, str], torch.Tensor] = {}
    norms: dict[int, torch.Tensor] = {}
    final_source_names: dict[int, tuple[str, ...]] = {}
    route_source_names: dict[tuple[int, str], tuple[str, ...]] = {}
    counted = 0
    feedback = model.cfg.feedback
    for first in range(0, rows, micro):
        batch = data_val.batch(first, min(micro, rows - first), device)
        n = batch.shape[0]
        prefix = torch.ones((1, n), dtype=torch.long, device=device)
        with analysis.autocast(device):
            # Conditions without f have only the plain pass to read.
            outs = multipass(
                model,
                batch,
                2 if feedback else 1,
                prefix_lens=prefix if feedback else None,
                want_weights=True,
            )
        for p, out in enumerate(outs):
            values = torch.stack(out.sources)  # [N, B, T, D]
            rms = values.float().pow(2).mean(-1).sqrt().mean((1, 2)).cpu() * n
            norms[p] = norms.get(p, 0.0) + rms
            final_source_names[p] = out.source_names
            for site, weights in out.route_weights.items():
                route_source_names[p, site] = out.route_source_names[site]
                w = weights.float()
                per_token = -(w.clamp_min(1e-12).log() * w).sum(dim=0)
                head_mean = w.mean(dim=-1, keepdim=True)
                head_js = (
                    (w * (w.clamp_min(1e-12).log() - head_mean.clamp_min(1e-12).log()))
                    .sum(dim=0)
                    .mean()
                )
                head_js /= math.log(min(w.shape[0], w.shape[-1]))
                batch_stats = (
                    torch.stack(
                        [
                            w.max(dim=0).values.mean(dim=(0, 1)),
                            per_token.mean(dim=(0, 1)) / math.log(w.shape[0]),
                        ],
                        dim=-1,
                    ).cpu()
                    * n
                )
                key = (p, site)
                sums[key] = sums.get(key, 0.0) + w.mean(dim=(1, 2)).cpu() * n
                stats[key] = stats.get(key, 0.0) + batch_stats
                divergences[key] = divergences.get(key, 0.0) + head_js.cpu() * n
        counted += n
    return (
        {key: (value / counted).numpy() for key, value in sums.items()},
        {key: (value / counted).numpy() for key, value in stats.items()},
        {key: float(value / counted) for key, value in divergences.items()},
        {key: (value / counted).numpy() for key, value in norms.items()},
        final_source_names,
        route_source_names,
    )


def entropy(weights: np.ndarray) -> float:
    """Normalized entropy of a mean distribution (1 = uniform)."""
    w = weights[weights > 0]
    return float(-(w * np.log(w)).sum() / math.log(len(weights)))


def site_matrix(means, route_names, cfg, passes=(0, 1)):
    """[site, source] mean-weight matrices per pass and head; NaN = absent.

    Columns cover the union of null, seed, completed-block, and current-partial
    labels.  Each row is placed by label rather than source position because
    the bank changes at block boundaries.
    """
    sites = site_names(cfg)
    columns = ["null", "seed"]
    columns += [f"block{i}" for i in range(cfg.routing_blocks)]
    columns += [f"partial{i}" for i in range(cfg.routing_blocks)]
    column_index = {name: index for index, name in enumerate(columns)}
    matrices = {}
    for p in passes:
        for head in range(cfg.kv_heads):
            matrix = np.full((len(sites), len(columns)), np.nan)
            for row, site in enumerate(sites):
                if (p, site) not in means:
                    continue
                w = means[(p, site)][:, head]
                for name, value in zip(route_names[p, site], w, strict=True):
                    matrix[row, column_index[name]] = value
            matrices[p, head] = matrix
    return sites, columns, matrices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--data-dir", default="data/dclm-100b")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--micro-rows", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    model, saved = analysis.load_checkpoint(args.snapshot, args.device)
    device = next(model.parameters()).device
    cfg = model.cfg
    layers = cfg.layers
    tag = saved["tag"]
    out_dir = args.out_dir or Path("figures") / f"route-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    passes = (0, 1) if cfg.feedback else (0,)
    pass_titles = {0: "pass 1 (plain)", 1: "pass 2 (fused)"}
    print(f"# {tag} — condition {saved['condition']!r}, {layers} layers, dim {cfg.dim}, device {device}")

    # -- static: query geometry ------------------------------------------------
    queries, null_rms, labels = [], [], []
    for i, block in enumerate(model.blocks):
        for kind, router in (("attn", block.attn_router), ("mlp", block.mlp_router)):
            if router is not None:
                queries.append(router.query.detach().float().cpu())
                null_rms.append(router.null.detach().float().square().mean().sqrt())
                labels.append(f"L{i}.{kind}")
    if cfg.feedback:
        queries.append(model.payload_router.query.detach().float().cpu())
        null_rms.append(
            model.payload_router.null.detach().float().square().mean().sqrt()
        )
        labels.append("payload")
    q = torch.stack(queries)
    norms = q.norm(dim=1)
    unit = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cosine = (unit @ unit.T).numpy()

    # -- empirical: routing distributions --------------------------------------
    data_val = TokenData.load(args.data_dir, "val", saved["seq_len"])
    (
        means,
        stats,
        divergences,
        norms_by_pass,
        final_source_names,
        route_source_names,
    ) = collect(model, data_val, device, args.rows, args.micro_rows)
    sites, columns, matrices = site_matrix(means, route_source_names, cfg, passes)

    # The routed values are raw (only keys are normed), so source scale
    # matters for what a read actually adds.
    print("\nfinal-bank source RMS norms:")
    for p, rms in sorted(norms_by_pass.items()):
        shown = "  ".join(
            f"{name}={value:.2f}"
            for name, value in zip(final_source_names[p], rms, strict=True)
        )
        print(f"  p{p + 1}: {shown}")

    print(
        f"\n{'site':<10}{'|q|':>7}{'nullRMS':>9}   "
        "pass head H_mean  maxT  H_tok head-JS  top sources"
    )
    report = {"snapshot": str(args.snapshot), "condition": saved["condition"], "rows": args.rows, "sites": {},
              "query_norms": dict(zip(labels, norms.tolist())), "query_cosine": cosine.tolist(), "labels": labels,
              "null_rms": dict(zip(labels, [float(v) for v in null_rms])),
              "source_rms": {f"p{p + 1}": dict(zip(final_source_names[p], rms.tolist())) for p, rms in norms_by_pass.items()}}
    for label, norm, null_scale in zip(labels, norms, null_rms, strict=True):
        for p in passes:
            key = (p, label)
            if key not in means:
                continue
            local = route_source_names[p, label]
            report["sites"].setdefault(label, {})[f"p{p + 1}"] = {
                "sources": list(local),
                "mean_weights": means[key].T.tolist(),
                "per_head_max": stats[key][:, 0].tolist(),
                "per_head_token_entropy": stats[key][:, 1].tolist(),
                "head_js": divergences[key],
            }
            for route_head in range(cfg.kv_heads):
                w = means[key][:, route_head]
                max_t, h_tok = stats[key][route_head]
                top = sorted(zip(local, w), key=lambda t: -t[1])[:3]
                shown = "  ".join(f"{n}={v:.3f}" for n, v in top)
                first_row = p == passes[0] and route_head == 0
                site = (
                    f"{label:<10}{norm:>7.2f}{null_scale:>9.3f}"
                    if first_row
                    else " " * 26
                )
                js = f"{divergences[key]:.3f}" if route_head == 0 else "  ·  "
                print(
                    f"{site}   p{p + 1}   h{route_head:<2}  {entropy(w):5.3f}  "
                    f"{max_t:.3f}  {h_tok:.3f}   {js}   {shown}"
                )
    (out_dir / "route_report.json").write_text(json.dumps(report, indent=2) + "\n")

    # -- figures ---------------------------------------------------------------
    norm_map = colors.PowerNorm(
        0.5,
        vmin=0,
        vmax=np.nanmax([np.nanmax(matrix) for matrix in matrices.values()]),
    )
    fig, axes = plt.subplots(
        cfg.kv_heads,
        len(passes),
        figsize=(7.5 * len(passes), 3.1 * cfg.kv_heads),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for route_head in range(cfg.kv_heads):
        for column, p in enumerate(passes):
            ax = axes[route_head, column]
            image = ax.imshow(
                matrices[p, route_head], aspect="auto", cmap=fs.SEQUENTIAL, norm=norm_map
            )
            ax.grid(False)
            ax.set_yticks(range(len(sites)), sites, fontsize=7)
            ax.set_title(f"head {route_head}, {pass_titles[p]}", fontsize=10)
            if column == 0:
                ax.set_ylabel("reading site")
            if route_head == cfg.kv_heads - 1:
                ax.set_xticks(range(len(columns)), columns, fontsize=7, rotation=90)
                ax.set_xlabel("MHDB source")
    fig.colorbar(image, ax=axes, label="mean routing weight", shrink=0.8)
    fig.suptitle(f"{tag}: MHDB read maps", fontsize=12)
    fs.save(fig, out_dir / "site-routing.png")

    if (0, "payload") in means:
        fig, axes = plt.subplots(
            len(passes), 1, figsize=(11, 2.75 * len(passes)), sharex=True, constrained_layout=True, squeeze=False
        )
        payload_max = max(means[(p, "payload")].max() for p in passes)
        for ax, p in zip(axes[:, 0], passes):
            image = ax.imshow(
                means[(p, "payload")].T,
                aspect="auto",
                cmap=fs.SEQUENTIAL,
                vmin=0,
                vmax=payload_max,
            )
            ax.grid(False)
            ax.set_yticks(range(cfg.kv_heads), range(cfg.kv_heads))
            ax.set_ylabel("routing head")
            ax.set_title(pass_titles[p])
        payload_names = route_source_names[0, "payload"]
        axes[-1, 0].set_xticks(
            range(len(payload_names)), payload_names, fontsize=7, rotation=90
        )
        axes[-1, 0].set_xlabel("payload source")
        fig.colorbar(image, ax=axes, label="mean routing weight", shrink=0.8)
        fig.suptitle(f"{tag}: MHDB payload — what rides to the next column")
        fs.save(fig, out_dir / "payload-routing.png")

    fig, (ax_norm, ax_cos) = plt.subplots(
        1, 2, figsize=(11, 4.2), width_ratios=(1, 1.2), constrained_layout=True
    )
    depth = np.arange(layers)
    attn_norms = [norms[labels.index(f"L{i}.attn")] for i in depth]
    mlp_norms = [norms[labels.index(f"L{i}.mlp")] for i in depth]
    ax_norm.plot(depth, attn_norms, "o-", color=fs.BLUE, label="attn reader")
    ax_norm.plot(depth, mlp_norms, "s-", color=fs.ORANGE, label="mlp reader")
    if "payload" in labels:
        ax_norm.axhline(
            norms[labels.index("payload")], color=fs.AQUA, lw=1, ls=":", label="payload"
        )
    ax_norm.set_xlabel("layer")
    ax_norm.set_ylabel("|query|")
    ax_norm.set_title("query norms (softmax sharpness)", fontsize=10)
    ax_norm.legend()
    image = ax_cos.imshow(cosine, cmap=fs.DIVERGING, vmin=-1, vmax=1)
    ax_cos.grid(False)
    ax_cos.set_xticks(range(len(labels)), labels, fontsize=6, rotation=90)
    ax_cos.set_yticks(range(len(labels)), labels, fontsize=6)
    ax_cos.set_title("query cosine similarity", fontsize=10)
    fig.colorbar(image, ax=ax_cos, shrink=0.9)
    fig.suptitle(f"{tag}: router query geometry", fontsize=12)
    fs.save(fig, out_dir / "query-geometry.png")
    print(f"\nfigures -> {out_dir}/")


if __name__ == "__main__":
    main()

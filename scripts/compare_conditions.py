"""Paired interior comparison of a reference checkpoint and a feedback checkpoint.

Both models see identical held-out rows.  For every token the script keeps the
cross-entropy of the reference (``m``: its pass 1), the feedback model's pass 1
(``d1``), its fused pass with plain-prefix length 1 (``d2``, the eval
convention) and a second fused iteration (``d3``); the KL between the
predictors; argmax agreement; and the token's position, frequency, and the
reference's previous-column surprise.  It compares the residual-stream sources
(seed, completed block deltas, ``h_top``) between the models on the same tokens
(relative difference, cosine, linear CKA), tests whether the payload is
linearly redundant with the readout state, and split-validates probability
mixtures of the predictors (weights chosen on the calibration rows, scored on
the rest).

Conditioning on one model's own loss selects on that model's noise, so the
tables are binned on the *reference* model's loss and entropy; read the
``d_df1_mhdb`` column of those tables as regression toward the mean, and the
``d_fused_df1`` column as a genuine independent conditioning.

Usage:
    python scripts/compare_conditions.py --reference runs/A.pt.10745 --feedback runs/B.pt.10745 \\
        --data-dir /data/delta/tokens
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from delta_feedback_experiment import analysis
from delta_feedback_experiment.data import TokenData
from delta_feedback_experiment.model import shift_right

POS_BINS = [(0, 0), (1, 3), (4, 15), (16, 63), (64, 255), (256, 511), (512, 1023)]
MIX_W = (0.1, 0.2, 0.3, 0.4, 0.5)
PAIRS = (("m", "d1"), ("d1", "d2"), ("m", "d2"))


def quantiles(x, ps=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {f"p{int(p * 100):02d}": float(np.quantile(x, p)) for p in ps}


def binned(values: np.ndarray, edges: np.ndarray, columns: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for i in range(len(edges) - 1):
        last = i == len(edges) - 2
        mm = (values >= edges[i]) & ((values <= edges[i + 1]) if last else (values < edges[i + 1]))
        row = {"range": [float(edges[i]), float(edges[i + 1])], "n": int(mm.sum())}
        for name, column in columns.items():
            row[name] = float(column[mm].mean()) if mm.any() else float("nan")
        rows.append(row)
    return rows


def cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA in feature space (both ``[n, d]``, centered per column)."""
    x = x.double() - x.double().mean(0, keepdim=True)
    y = y.double() - y.double().mean(0, keepdim=True)
    return float((x.T @ y).norm().square() / ((x.T @ x).norm() * (y.T @ y).norm()))


def ridge_r2(x: torch.Tensor, y: torch.Tensor, lam_scale: float = 1e-3) -> float:
    """Ridge fit of ``y`` from ``x`` on the first half, R^2 on the second half."""
    x, y = x.double(), y.double()
    n = x.shape[0] // 2
    xt, yt, xv, yv = x[:n], y[:n], x[n:], y[n:]
    xm, ym = xt.mean(0, keepdim=True), yt.mean(0, keepdim=True)
    xt, xv = xt - xm, xv - xm
    lam = lam_scale * xt.shape[1]
    w = torch.linalg.solve(xt.T @ xt + lam * torch.eye(x.shape[1], dtype=x.dtype), xt.T @ (yt - ym))
    pred = xv @ w + ym
    return float(1 - (pred - yv).square().sum() / (yv - yv.mean(0, keepdim=True)).square().sum())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", type=Path, required=True, help="reference snapshot (any condition; its pass 1 is used)")
    parser.add_argument("--feedback", type=Path, required=True, help="feedback snapshot (a condition with f)")
    parser.add_argument("--data-dir", default="data/tokens")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--calib-rows", type=int, default=64, help="leading rows used to choose mixture weights")
    parser.add_argument("--micro-rows", type=int, default=4)
    parser.add_argument("--stride", type=int, default=8, help="position stride for CKA/ridge feature samples")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    torch._dynamo.config.recompile_limit = 64
    ref, saved_ref = analysis.load_checkpoint(args.reference, args.device)
    fb, saved_fb = analysis.load_checkpoint(args.feedback, args.device)
    device = next(fb.parameters()).device
    if not fb.cfg.feedback_active:
        raise SystemExit("--feedback must be a snapshot of a condition with f")
    if saved_ref["seq_len"] != saved_fb["seq_len"]:
        raise SystemExit("the two snapshots have different sequence lengths")
    cfg = fb.cfg
    T = saved_fb["seq_len"]
    tag_ref, tag_fb = saved_ref.get("tag", args.reference.stem), saved_fb.get("tag", args.feedback.stem)
    out_dir = args.out_dir or Path("figures") / f"compare-{tag_ref}-vs-{tag_fb}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = TokenData.load(args.data_dir, "val", T)
    print(f"reference={args.reference} ({saved_ref['condition']!r}) feedback={args.feedback} ({saved_fb['condition']!r}) rows={args.rows}", flush=True)

    val_tokens = data.read(0, data.rows * (T + 1))
    counts = np.bincount(val_tokens, minlength=cfg.vocab_size).astype(np.float64)
    del val_tokens

    positions = torch.arange(T, device=device)
    chunk = 256
    keys = ("pos", "ce_m", "ce_d1", "ce_d2", "ce_d3", "kl_m_d1", "kl_d1_d2", "kl_m_d2", "ent_m", "ent_d1",
            "acc_m", "acc_d1", "acc_d2", "agree_m_d1", "agree_d1_d2", "inp_freq", "tgt_freq",
            "prev_surprise_m", "gate_mean", "gate_logit_rms", "htop_rel_m_d1", "htop_rel_d1_d2",
            "htop_cos_m_d1", "htop_cos_d1_d2")
    keep = {k: [] for k in keys}
    source_rel: dict = {}
    source_cos: dict = {}
    n_source = 0
    feats: dict[str, list] = {}
    source_names: list[str] = []
    mix_sums = {split: {pair: np.zeros(len(MIX_W)) for pair in PAIRS} for split in ("calib", "test")}
    single_sums = {split: {name: 0.0 for name in ("m", "d1", "d2")} for split in ("calib", "test")}
    mix_counts = {"calib": 0, "test": 0}
    rel = lambda a, b: (b - a).square().mean(-1).sqrt() / a.square().mean(-1).sqrt().clamp_min(1e-6)
    flat = lambda x: x.reshape(-1).detach().cpu().numpy()
    t0 = time.time()
    with analysis.autocast(device):
        for first in range(0, args.rows, args.micro_rows):
            split = "calib" if first < args.calib_rows else "test"
            tokens = data.batch(first, min(args.micro_rows, args.rows - first), device)
            B = tokens.shape[0]
            inp, tgt = tokens[:, :-1], tokens[:, 1:]

            out_m = ref.forward_column(ref.embed_tokens(inp))
            e_d = fb.embed_tokens(inp)
            out_d1 = fb.forward_column(e_d, need_payload=True)
            gate_logit = fb.fuse_gate(fb.gate_norm(e_d)).float()
            out_d2 = fb.forward_column(analysis.fused_inputs(fb, e_d, out_d1.payload), need_payload=True)
            out_d3 = fb.forward_column(analysis.fused_inputs(fb, e_d, out_d2.payload), need_payload=False)

            heads = {"m": (ref, out_m.h_top), "d1": (fb, out_d1.h_top), "d2": (fb, out_d2.h_top), "d3": (fb, out_d3.h_top)}
            ce = {k: [] for k in heads}
            kl = {k: [] for k in ("m_d1", "d1_d2", "m_d2")}
            ent = {"m": [], "d1": []}
            arg = {k: [] for k in ("m", "d1", "d2")}
            iterators = {k: analysis.logprob_chunks(model, h, chunk) for k, (model, h) in heads.items()}
            for s in range(0, T, chunk):
                t = tgt[:, s : s + chunk]
                lp = {k: next(it)[1] for k, it in iterators.items()}
                for k in heads:
                    ce[k].append(-lp[k].gather(-1, t[..., None]).squeeze(-1))
                for a, b in (("m", "d1"), ("d1", "d2"), ("m", "d2")):
                    kl[f"{a}_{b}"].append((lp[a].exp() * (lp[a] - lp[b])).sum(-1))
                for k in ent:
                    ent[k].append(-(lp[k].exp() * lp[k]).sum(-1))
                for k in arg:
                    arg[k].append(lp[k].argmax(-1))
                for pair in PAIRS:
                    a, b = pair
                    for i, w in enumerate(MIX_W):
                        lm = torch.logaddexp(math.log(1 - w) + lp[a], math.log(w) + lp[b])
                        mix_sums[split][pair][i] += -lm.gather(-1, t[..., None]).squeeze(-1).sum().item()
                for k in single_sums[split]:
                    single_sums[split][k] += ce[k][-1].sum().item()
                del lp
            mix_counts[split] += tgt.numel()
            ce = {k: torch.cat(v, 1) for k, v in ce.items()}
            kl = {k: torch.cat(v, 1) for k, v in kl.items()}
            ent = {k: torch.cat(v, 1) for k, v in ent.items()}
            arg = {k: torch.cat(v, 1) for k, v in arg.items()}

            prev_surprise = torch.zeros_like(ce["m"])
            prev_surprise[:, 1:] = ce["m"][:, :-1]
            gate = torch.sigmoid(gate_logit)
            hm, hd1, hd2 = out_m.h_top.float(), out_d1.h_top.float(), out_d2.h_top.float()
            keep["pos"].append(flat(positions[None, :].expand(B, -1)))
            for k in ("m", "d1", "d2", "d3"):
                keep[f"ce_{k}"].append(flat(ce[k]))
            for k in ("m_d1", "d1_d2", "m_d2"):
                keep[f"kl_{k}"].append(flat(kl[k]))
            keep["ent_m"].append(flat(ent["m"]))
            keep["ent_d1"].append(flat(ent["d1"]))
            for k in ("m", "d1", "d2"):
                keep[f"acc_{k}"].append(flat(arg[k] == tgt))
            keep["agree_m_d1"].append(flat(arg["m"] == arg["d1"]))
            keep["agree_d1_d2"].append(flat(arg["d1"] == arg["d2"]))
            keep["inp_freq"].append(counts[flat(inp)])
            keep["tgt_freq"].append(counts[flat(tgt)])
            keep["prev_surprise_m"].append(flat(prev_surprise))
            keep["gate_mean"].append(flat(gate.mean(-1)))
            keep["gate_logit_rms"].append(flat(gate_logit.square().mean(-1).sqrt()))
            keep["htop_rel_m_d1"].append(flat(rel(hm, hd1)))
            keep["htop_rel_d1_d2"].append(flat(rel(hd1, hd2)))
            keep["htop_cos_m_d1"].append(flat(F.cosine_similarity(hm, hd1, dim=-1)))
            keep["htop_cos_d1_d2"].append(flat(F.cosine_similarity(hd1, hd2, dim=-1)))

            # residual-stream sources on the same tokens; without routing the
            # bank is absent, so fall back to the seed and top state only
            def bank(out, e):
                if out.sources is not None:
                    return list(out.source_names) + ["h_top"], [*out.sources, out.h_top]
                return ["seed", "h_top"], [e, out.h_top]

            names_m, srcs_m = bank(out_m, ref.embed_tokens(inp))
            names_d, srcs_d1 = bank(out_d1, e_d)
            _, srcs_d2 = bank(out_d2, analysis.fused_inputs(fb, e_d, out_d1.payload))
            common = [n for n in names_m if n in names_d]
            source_names = common
            srcs = {
                "m": {n: s for n, s in zip(names_m, srcs_m)},
                "d1": {n: s for n, s in zip(names_d, srcs_d1)},
                "d2": {n: s for n, s in zip(names_d, srcs_d2)},
            }
            for a, b in (("m", "d1"), ("d1", "d2")):
                for name in common:
                    sa, sb = srcs[a][name].float()[:, 1:], srcs[b][name].float()[:, 1:]
                    source_rel[(a, b, name)] = source_rel.get((a, b, name), 0.0) + rel(sa, sb).mean().item() * B
                    source_cos[(a, b, name)] = source_cos.get((a, b, name), 0.0) + F.cosine_similarity(sa, sb, dim=-1).mean().item() * B
            n_source += B
            coll = {
                "fn_m": ref.final_norm(out_m.h_top),
                "fn_d1": fb.final_norm(out_d1.h_top),
                "payload_d1": out_d1.payload,
                "htop_d1": out_d1.h_top,
                "e": e_d,
            }
            if cfg.routing_active:
                payload_sources = [out_d1.sources[0], *out_d1.sources[1:]]
                routed, _ = fb.payload_router(payload_sources, [None] * len(payload_sources), False)
                coll["routed_d1"] = routed
            for k, table in srcs.items():
                for name in common:
                    coll[f"{name}_{k}"] = table[name]
            for k, v in coll.items():
                feats.setdefault(k, []).append(v[:, 1 :: args.stride].reshape(-1, cfg.dim).half().cpu())
            if first % 32 == 0:
                print(f"  rows {first}.. {time.time() - t0:.0f}s", flush=True)

    K = {k: np.concatenate(v) for k, v in keep.items()}
    for k in ("acc_m", "acc_d1", "acc_d2", "agree_m_d1", "agree_d1_d2"):
        K[k] = K[k].astype(np.float64)
    pos = K["pos"]
    d_m_d1 = K["ce_d1"] - K["ce_m"]
    d_d1_d2 = K["ce_d2"] - K["ce_d1"]
    d_m_d2 = K["ce_d2"] - K["ce_m"]
    d_d2_d3 = K["ce_d3"] - K["ce_d2"]
    se = lambda x: float(np.std(x) / math.sqrt(x.size))
    report: dict = {
        "reference": str(args.reference), "feedback": str(args.feedback), "reference_condition": saved_ref["condition"],
        "feedback_condition": saved_fb["condition"], "rows": args.rows, "calib_rows": args.calib_rows, "tokens": int(pos.size),
        "labels": {"m": f"{tag_ref} pass 1", "d1": f"{tag_fb} pass 1", "d2": f"{tag_fb} fused", "d3": f"{tag_fb} fused, iteration 2"},
    }
    report["means"] = {
        "mhdb": float(K["ce_m"].mean()), "df_pass1": float(K["ce_d1"].mean()),
        "df_fused": float(K["ce_d2"].mean()), "df_fused_iter2": float(K["ce_d3"].mean()),
        "delta_df1_minus_mhdb": float(d_m_d1.mean()), "delta_df1_minus_mhdb_se": se(d_m_d1),
        "delta_fused_minus_df1": float(d_d1_d2.mean()), "delta_fused_minus_df1_se": se(d_d1_d2),
        "delta_fused_minus_mhdb": float(d_m_d2.mean()), "delta_iter2_minus_fused": float(d_d2_d3.mean()),
        "first32rows_mhdb": float(K["ce_m"][: 32 * T].mean()), "first32rows_df1": float(K["ce_d1"][: 32 * T].mean()),
        "first32rows_df_fused": float(K["ce_d2"][: 32 * T].mean()),
        "kl_mhdb_to_df1": float(K["kl_m_d1"].mean()), "kl_df1_to_fused": float(K["kl_d1_d2"].mean()),
        "kl_mhdb_to_fused": float(K["kl_m_d2"].mean()),
        "acc_mhdb": float(K["acc_m"].mean()), "acc_df1": float(K["acc_d1"].mean()), "acc_fused": float(K["acc_d2"].mean()),
        "argmax_agree_mhdb_df1": float(K["agree_m_d1"].mean()), "argmax_agree_df1_fused": float(K["agree_d1_d2"].mean()),
        "corr_ce_mhdb_df1": float(np.corrcoef(K["ce_m"], K["ce_d1"])[0, 1]),
        "corr_ce_df1_fused": float(np.corrcoef(K["ce_d1"], K["ce_d2"])[0, 1]),
        "entropy_mhdb": float(K["ent_m"].mean()), "entropy_df1": float(K["ent_d1"].mean()),
        "htop_rel_mhdb_df1": float(K["htop_rel_m_d1"].mean()), "htop_cos_mhdb_df1": float(K["htop_cos_m_d1"].mean()),
        "htop_rel_df1_fused": float(K["htop_rel_d1_d2"][pos >= 1].mean()), "htop_cos_df1_fused": float(K["htop_cos_d1_d2"][pos >= 1].mean()),
    }
    m = pos >= 1
    report["delta_quantiles"] = {
        "df1_minus_mhdb": quantiles(d_m_d1), "fused_minus_df1": quantiles(d_d1_d2[m]), "fused_minus_mhdb": quantiles(d_m_d2[m]),
    }
    report["frac_better"] = {
        "df1_beats_mhdb": float((d_m_d1 < 0).mean()), "fused_beats_df1": float((d_d1_d2[m] < 0).mean()),
        "fused_beats_mhdb": float((d_m_d2[m] < 0).mean()),
    }
    cols = {
        "mhdb": K["ce_m"], "df1": K["ce_d1"], "fused": K["ce_d2"],
        "d_df1_mhdb": d_m_d1, "d_fused_df1": d_d1_d2, "d_fused_mhdb": d_m_d2, "d_iter2_fused": d_d2_d3,
        "kl_m_d1": K["kl_m_d1"], "kl_d1_d2": K["kl_d1_d2"],
        "agree_m_d1": K["agree_m_d1"], "agree_d1_d2": K["agree_d1_d2"],
        "htop_rel_m_d1": K["htop_rel_m_d1"], "htop_rel_d1_d2": K["htop_rel_d1_d2"],
        "frac_df1_better": (d_m_d1 < 0).astype(np.float64), "frac_fused_better": (d_d1_d2 < 0).astype(np.float64),
        "gate_mean": K["gate_mean"],
    }
    rows = []
    for lo, hi in POS_BINS:
        mm = (pos >= lo) & (pos <= hi)
        row = {"bin": f"{lo}-{hi}", "n": int(mm.sum())}
        row.update({k: float(v[mm].mean()) for k, v in cols.items()})
        row["d_df1_mhdb_se"] = se(d_m_d1[mm])
        row["d_fused_df1_se"] = se(d_d1_d2[mm])
        rows.append(row)
    report["position_bins"] = rows
    width = 32
    report["position_curve"] = [
        {
            "pos_lo": int(b * width),
            "d_df1_mhdb": float(d_m_d1[(pos >= b * width) & (pos < (b + 1) * width)].mean()),
            "d_df1_mhdb_se": se(d_m_d1[(pos >= b * width) & (pos < (b + 1) * width)]),
            "d_fused_df1": float(d_d1_d2[(pos >= max(b * width, 1)) & (pos < (b + 1) * width)].mean()),
            "d_fused_df1_se": se(d_d1_d2[(pos >= max(b * width, 1)) & (pos < (b + 1) * width)]),
            "mhdb": float(K["ce_m"][(pos >= b * width) & (pos < (b + 1) * width)].mean()),
        }
        for b in range(T // width)
    ]
    sub = {k: v[m] for k, v in cols.items()}
    edges = np.quantile(K["ce_m"][m], [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    report["by_mhdb_ce"] = binned(K["ce_m"][m], edges, sub)
    edges = np.quantile(K["prev_surprise_m"][m], [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    report["by_prev_surprise"] = binned(K["prev_surprise_m"][m], edges, sub)
    edges = np.quantile(K["inp_freq"][m], [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    report["by_input_token_frequency"] = binned(K["inp_freq"][m], edges, sub)
    edges = np.quantile(K["tgt_freq"][m], [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    report["by_target_token_frequency"] = binned(K["tgt_freq"][m], edges, sub)
    edges = np.quantile(K["ent_m"][m], [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    report["by_mhdb_entropy"] = binned(K["ent_m"][m], edges, sub)

    ens = {}
    for pair in PAIRS:
        a, b = pair
        calib = mix_sums["calib"][pair] / mix_counts["calib"]
        test = mix_sums["test"][pair] / mix_counts["test"]
        single_c = {k: single_sums["calib"][k] / mix_counts["calib"] for k in (a, b)}
        single_t = {k: single_sums["test"][k] / mix_counts["test"] for k in (a, b)}
        cands = [(single_c[a], 0.0, single_t[a])] + [(calib[i], w, test[i]) for i, w in enumerate(MIX_W)]
        best = min(cands, key=lambda t: t[0])
        ens[f"{a}+{b}"] = {
            "single_test": single_t, "best_w_on_calib": best[1], "test_at_best_w": best[2],
            "test_gain_over_first": single_t[a] - best[2], "test_gain_over_better_single": min(single_t.values()) - best[2],
            "test_by_w": {"0.0": single_t[a], **{f"{w:.1f}": float(test[i]) for i, w in enumerate(MIX_W)}},
        }
    report["ensembles"] = ens

    Fe = {k: torch.cat(v).float() for k, v in feats.items()}
    rep = {}
    for name in source_names:
        entry = {}
        for a, b in (("m", "d1"), ("d1", "d2"), ("m", "d2")):
            entry[f"cka_{a}_{b}"] = cka(Fe[f"{name}_{a}"], Fe[f"{name}_{b}"])
        for a, b in (("m", "d1"), ("d1", "d2")):
            entry[f"rel_{a}_{b}"] = source_rel[(a, b, name)] / n_source
            entry[f"cos_{a}_{b}"] = source_cos[(a, b, name)] / n_source
        for k in ("m", "d1", "d2"):
            entry[f"rms_{k}"] = float(Fe[f"{name}_{k}"].square().mean().sqrt())
        rep[name] = entry
    report["representation"] = rep
    report["feature_tokens"] = int(Fe["fn_d1"].shape[0])

    redundancy = {
        "r2_payload_from_final_norm_df1": ridge_r2(Fe["fn_d1"], Fe["payload_d1"]),
        "r2_payload_from_final_norm_mhdb": ridge_r2(Fe["fn_m"], Fe["payload_d1"]),
        "r2_payload_from_htop_df1": ridge_r2(Fe["htop_d1"], Fe["payload_d1"]),
        "r2_payload_from_embedding": ridge_r2(Fe["e"], Fe["payload_d1"]),
        "r2_final_norm_df1_from_final_norm_mhdb": ridge_r2(Fe["fn_m"], Fe["fn_d1"]),
        "r2_final_norm_mhdb_from_final_norm_df1": ridge_r2(Fe["fn_d1"], Fe["fn_m"]),
        "payload_rms": float(Fe["payload_d1"].square().mean().sqrt()),
        "payload_norm_weight_quantiles": quantiles(fb.payload_norm.weight.detach().float().cpu().numpy()),
    }
    if "routed_d1" in Fe:
        redundancy.update({
            "r2_routed_from_final_norm_df1": ridge_r2(Fe["fn_d1"], Fe["routed_d1"]),
            "r2_routed_from_embedding": ridge_r2(Fe["e"], Fe["routed_d1"]),
            "routed_rms_over_htop_rms": float((Fe["routed_d1"].square().mean(-1).sqrt() / Fe["htop_d1"].square().mean(-1).sqrt()).mean()),
            "routed_rms": float(Fe["routed_d1"].square().mean().sqrt()),
        })
    report["payload_redundancy"] = redundancy
    report["gate"] = {
        "mean": float(K["gate_mean"][m].mean()), "token_mean_quantiles": quantiles(K["gate_mean"][m]),
        "logit_rms_quantiles": quantiles(K["gate_logit_rms"][m]),
    }
    out_path = out_dir / "compare_conditions.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    M = report["means"]
    print("\n== headline ==")
    for k, v in M.items():
        print(f"  {k:<40} {v:.5f}")
    print("\n== per-token delta quantiles ==")
    for k, v in report["delta_quantiles"].items():
        print(f"  {k:<18}", {kk: round(vv, 3) for kk, vv in v.items()})
    print("  frac better:", report["frac_better"])
    print("\n== by position ==")
    print(f"  {'bin':<10}{'n':>8}{'mhdb':>8}{'df1':>8}{'fused':>8}{'d_df1':>9}{'d_fus':>9}{'d_it2':>8}{'kl_md1':>8}{'kl_d12':>8}{'agr_md1':>8}{'rel_md1':>8}{'rel_d12':>8}")
    for r in rows:
        print(f"  {r['bin']:<10}{r['n']:>8}{r['mhdb']:>8.3f}{r['df1']:>8.3f}{r['fused']:>8.3f}{r['d_df1_mhdb']:>+9.4f}{r['d_fused_df1']:>+9.4f}{r['d_iter2_fused']:>+8.4f}{r['kl_m_d1']:>8.3f}{r['kl_d1_d2']:>8.3f}{r['agree_m_d1']:>8.3f}{r['htop_rel_m_d1']:>8.3f}{r['htop_rel_d1_d2']:>8.3f}")
    for title, key in (("by reference own CE (d_df1 column is regression to the mean)", "by_mhdb_ce"),
                       ("by previous-column surprise (reference)", "by_prev_surprise"),
                       ("by input-token frequency", "by_input_token_frequency"), ("by target-token frequency", "by_target_token_frequency"),
                       ("by reference entropy", "by_mhdb_entropy")):
        print(f"\n== {title} ==")
        for r in report[key]:
            print(f"  [{r['range'][0]:9.2f},{r['range'][1]:9.2f}) n={r['n']:>7} mhdb={r['mhdb']:.3f} d_df1={r['d_df1_mhdb']:+.4f} ({r['frac_df1_better']:.3f} better) d_fused={r['d_fused_df1']:+.4f} ({r['frac_fused_better']:.3f} better) kl_md1={r['kl_m_d1']:.3f} kl_d12={r['kl_d1_d2']:.3f}")
    print("\n== ensembles (split-validated) ==")
    for k, v in ens.items():
        print(f"  {k:<8} singles={ {kk: round(vv, 4) for kk, vv in v['single_test'].items()} } best_w={v['best_w_on_calib']} test={v['test_at_best_w']:.4f} gain_over_better_single={v['test_gain_over_better_single']:+.4f}")
    print("\n== representation (positions >= 1; CKA on stride-sampled tokens) ==")
    for name, e in rep.items():
        print(f"  {name:<8} " + "  ".join(f"{k}={v:.3f}" for k, v in e.items()))
    print("\n== payload redundancy ==", json.dumps(redundancy, indent=1))
    print("== gate ==", json.dumps(report["gate"]))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

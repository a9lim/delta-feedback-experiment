"""Same-checkpoint structure of the fused pass against pass 1 on a feedback snapshot.

Held-out diagnostics on one snapshot of a condition with `f`:

* per-position fused penalty under the eval convention (prefix 1), a plain
  prefix of 512, a training-style random prefix, a second fused iteration,
  and a jittered payload;
* per-token penalty conditioned on how surprising the fused-in token was under
  the previous column's prediction (token-identity bottleneck test), on the
  input token's frequency, and on the position's own pass-1 loss (that last
  conditioning selects on pass-1 noise and reads as regression to the mean);
* FBT entry-gate statistics, the token-dependent share of the pre-norm fused
  input, fusion weight norms, and token decodability from the fused seed;
* scale statistics of ``h_top`` and the payload, and the pass-2 change of every
  residual-stream source;
* a 30-iteration fully fused self-composition trace.

Usage:
    python scripts/fused_diagnostics.py runs/TAG.pt.STEP --data-dir /data/df/tokens
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


@torch.no_grad()
def head_stats(model, h_a, h_b, targets, chunk=256):
    """Per-token CE for two top states plus KL(a||b), argmax_a, argmax_b, entropy_a."""
    ce_a, ce_b, kl, arg_a, arg_b, ent_a = [], [], [], [], [], []
    for (s, la), (_, lb) in zip(analysis.logprob_chunks(model, h_a, chunk), analysis.logprob_chunks(model, h_b, chunk)):
        t = targets[:, s : s + chunk]
        ce_a.append(-la.gather(-1, t[..., None]).squeeze(-1))
        ce_b.append(-lb.gather(-1, t[..., None]).squeeze(-1))
        pa = la.exp()
        kl.append((pa * (la - lb)).sum(-1))
        ent_a.append(-(pa * la).sum(-1))
        arg_a.append(la.argmax(-1))
        arg_b.append(lb.argmax(-1))
    cat = lambda xs: torch.cat(xs, 1)
    return cat(ce_a), cat(ce_b), cat(kl), cat(arg_a), cat(arg_b), cat(ent_a)


@torch.no_grad()
def nearest_token(model, u, chunk=256):
    """argmax_v <u, E_v> for a seed u [B, T, D]."""
    W = model.embed_tokens.weight.to(u.dtype)
    return torch.cat([F.linear(u[:, s : s + chunk], W).argmax(-1) for s in range(0, u.shape[1], chunk)], 1)


def quantiles(x, ps=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)):
    x = np.asarray(x, dtype=np.float64)
    return {f"p{int(p * 100):02d}": float(np.quantile(x, p)) for p in ps}


def binned(values, edges, columns: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for i in range(len(edges) - 1):
        last = i == len(edges) - 2
        mm = (values >= edges[i]) & ((values <= edges[i + 1]) if last else (values < edges[i + 1]))
        row = {"range": [float(edges[i]), float(edges[i + 1])], "n": int(mm.sum())}
        row.update({k: float(v[mm].mean()) if mm.any() else float("nan") for k, v in columns.items()})
        rows.append(row)
    return rows


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--data-dir", default="data/tokens")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--micro-rows", type=int, default=4)
    parser.add_argument("--self-rows", type=int, default=8)
    parser.add_argument("--self-iters", type=int, default=30)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    torch._dynamo.config.recompile_limit = 64
    model, saved = analysis.load_checkpoint(args.snapshot, args.device)
    device = next(model.parameters()).device
    if not model.cfg.feedback_active:
        raise SystemExit("the fused diagnostics need a snapshot of a condition with f")
    cfg = model.cfg
    T = saved["seq_len"]
    tag = saved.get("tag", args.snapshot.stem)
    out_dir = args.out_dir or Path("figures") / f"fused-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = TokenData.load(args.data_dir, "val", T)
    print(f"loaded {args.snapshot} condition={saved['condition']!r} rows={args.rows}", flush=True)

    val_tokens = data.read(0, data.rows * (T + 1))
    counts = np.bincount(val_tokens, minlength=cfg.vocab_size).astype(np.float64)
    del val_tokens

    gen = torch.Generator(device=device).manual_seed(1234)
    positions = torch.arange(T, device=device)
    keep = {k: [] for k in ("pos", "ce1", "ce2", "ce2b", "ce2r", "ce3", "ce2j", "kl12", "acc1", "acc2", "agree12", "prev_hit",
                            "prev_surprise", "ent1", "gate_mean", "gate_lo", "gate_hi", "cos_ue", "u_nearest_hit", "inp_freq",
                            "h1_rms", "h1_top8", "p_top8", "state_upd_rel", "prefix_r", "token_frac")}
    u_sub, e_sub, logit_rms, gate_coord_std = [], [], [], []
    payload_sum = payload_sq = None
    n_payload = 0
    source_rel = None
    source_names: list[str] = []
    n_batches = 0
    flat = lambda t: t.reshape(-1).detach().cpu().numpy()
    t0 = time.time()
    with analysis.autocast(device):
        for first in range(0, args.rows, args.micro_rows):
            tokens = data.batch(first, min(args.micro_rows, args.rows - first), device)
            B = tokens.shape[0]
            inp, tgt = tokens[:, :-1], tokens[:, 1:]
            e = model.embed_tokens(inp)
            out1 = model.forward_column(e, need_payload=True)
            p1 = out1.payload
            ps = shift_right(p1)
            g_logit = model.fuse_gate(model.gate_norm(e)).float()
            gate = torch.sigmoid(g_logit)
            value = model.fuse_value(ps).float()
            fused_all = model.fuse(ps, e)

            out2 = model.forward_column(analysis.fused_inputs(model, e, p1, 1), need_payload=True)
            out2b = model.forward_column(analysis.fused_inputs(model, e, p1, 512), need_payload=False)
            pref = torch.randint(1, T, (B,), generator=gen, device=device)
            out2r = model.forward_column(analysis.fused_inputs(model, e, p1, pref), need_payload=False)
            out3 = model.forward_column(analysis.fused_inputs(model, e, out2.payload, 1), need_payload=False)
            jit = torch.empty_like(p1).uniform_(-saved["jitter"], saved["jitter"], generator=gen)
            out2j = model.forward_column(analysis.fused_inputs(model, e, p1 + jit, 1), need_payload=False)

            ce1, ce2, kl12, arg1, arg2, ent1 = head_stats(model, out1.h_top, out2.h_top, tgt)
            ce2b = analysis.token_ce(model, out2b.h_top, tgt)
            ce2r = analysis.token_ce(model, out2r.h_top, tgt)
            ce3 = analysis.token_ce(model, out3.h_top, tgt)
            ce2j = analysis.token_ce(model, out2j.h_top, tgt)

            prev_hit = torch.zeros_like(inp, dtype=torch.bool)
            prev_hit[:, 1:] = arg1[:, :-1] == inp[:, 1:]
            prev_surprise = torch.zeros_like(ce1)
            prev_surprise[:, 1:] = ce1[:, :-1]
            u, ef = fused_all.float(), e.float()
            h1 = out1.h_top.float()
            h1_sq, p_sq = h1.square(), p1.float().square()
            gbar = gate[:, 1:].reshape(-1, cfg.dim).mean(0)
            tok_part = (value[:, 1:] * (gate[:, 1:] - gbar)).square().mean(-1).sqrt()
            all_part = (value[:, 1:] * gate[:, 1:]).square().mean(-1).sqrt()

            keep["pos"].append(flat(positions[None, :].expand(B, -1)))
            keep["prefix_r"].append(flat(pref[:, None].expand(B, T)))
            for k, v in (("ce1", ce1), ("ce2", ce2), ("ce2b", ce2b), ("ce2r", ce2r), ("ce3", ce3), ("ce2j", ce2j),
                         ("kl12", kl12), ("ent1", ent1), ("prev_surprise", prev_surprise),
                         ("cos_ue", F.cosine_similarity(u, ef, dim=-1)),
                         ("h1_rms", h1_sq.mean(-1).sqrt()), ("h1_top8", h1_sq.topk(8, dim=-1).values.sum(-1) / h1_sq.sum(-1)),
                         ("p_top8", p_sq.topk(8, dim=-1).values.sum(-1) / p_sq.sum(-1)),
                         ("state_upd_rel", (out2.h_top.float() - h1).square().mean(-1).sqrt() / h1_sq.mean(-1).sqrt()),
                         ("gate_mean", gate.mean(-1)), ("gate_lo", (gate < 0.1).float().mean(-1)), ("gate_hi", (gate > 0.9).float().mean(-1))):
                keep[k].append(flat(v))
            keep["token_frac"].append(flat(tok_part / all_part))
            keep["acc1"].append(flat(arg1 == tgt))
            keep["acc2"].append(flat(arg2 == tgt))
            keep["agree12"].append(flat(arg1 == arg2))
            keep["prev_hit"].append(flat(prev_hit))
            keep["u_nearest_hit"].append(flat(nearest_token(model, fused_all) == inp))
            keep["inp_freq"].append(counts[flat(inp)])
            logit_rms.append(flat(g_logit[:, 1:].square().mean(-1).sqrt()))
            gate_coord_std.append(gate[:, 1:].reshape(-1, cfg.dim).std(0).cpu().numpy())
            u_sub.append(u[:, 1::8].reshape(-1, cfg.dim).half().cpu())
            e_sub.append(ef[:, 1::8].reshape(-1, cfg.dim).half().cpu())
            pflat = p1.float()[:, :-1].reshape(-1, cfg.dim)
            payload_sum = pflat.sum(0) if payload_sum is None else payload_sum + pflat.sum(0)
            payload_sq = pflat.square().sum(0) if payload_sq is None else payload_sq + pflat.square().sum(0)
            n_payload += pflat.shape[0]
            if out1.sources is not None:
                rel = []
                for s1, s2 in zip(out1.sources, out2.sources):
                    rel.append(((s2.float() - s1.float()).square().mean(-1).sqrt() / s1.float().square().mean(-1).sqrt().clamp_min(1e-6))[:, 1:].mean().item())
                source_names = list(out1.source_names)
                source_rel = rel if source_rel is None else [a + b for a, b in zip(source_rel, rel)]
            n_batches += 1
            if first % 64 == 0:
                print(f"  rows {first}.. {time.time() - t0:.0f}s", flush=True)

    K = {k: np.concatenate(v) for k, v in keep.items()}
    for k in ("acc1", "acc2", "agree12", "prev_hit", "u_nearest_hit"):
        K[k] = K[k].astype(np.float64)
    pos = K["pos"]
    d = K["ce2"] - K["ce1"]
    report: dict = {"snapshot": str(args.snapshot), "condition": saved["condition"], "rows": args.rows, "tokens": int(pos.size)}
    n32 = 32 * T
    m512 = pos >= 512
    mr = pos >= K["prefix_r"]
    report["means"] = {
        "pass1": float(K["ce1"].mean()), "fused_prefix1": float(K["ce2"].mean()),
        "fused_prefix512_all": float(K["ce2b"].mean()), "fused_random_prefix_all": float(K["ce2r"].mean()),
        "fused_two_iterations": float(K["ce3"].mean()), "fused_prefix1_jittered": float(K["ce2j"].mean()),
        "first32_pass1": float(K["ce1"][:n32].mean()), "first32_fused": float(K["ce2"][:n32].mean()),
        "kl_pass1_to_fused": float(K["kl12"].mean()), "acc1": float(K["acc1"].mean()), "acc2": float(K["acc2"].mean()),
        "argmax_agreement": float(K["agree12"].mean()),
        "suffix512_pass1": float(K["ce1"][m512].mean()), "suffix512_fused_prefix1": float(K["ce2"][m512].mean()),
        "suffix512_fused_prefix512": float(K["ce2b"][m512].mean()),
        "randprefix_fusedpos_pass1": float(K["ce1"][mr].mean()), "randprefix_fusedpos_fused": float(K["ce2r"][mr].mean()),
        "randprefix_fusedpos_fused_prefix1_same_positions": float(K["ce2"][mr].mean()),
        "randprefix_plainpos_pass1": float(K["ce1"][~mr].mean()), "randprefix_plainpos_pass2": float(K["ce2r"][~mr].mean()),
    }
    m = pos >= 1
    report["delta_quantiles"] = quantiles(d[m])
    report["delta_frac_negative"] = float((d[m] < 0).mean())
    report["delta_mean_pos_ge1"] = float(d[m].mean())
    rows = []
    for lo, hi in POS_BINS:
        mm = (pos >= lo) & (pos <= hi)
        rows.append({"bin": f"{lo}-{hi}", "n": int(mm.sum()), "pass1": float(K["ce1"][mm].mean()), "fused": float(K["ce2"][mm].mean()),
                     "delta": float(d[mm].mean()), "delta_se": float(d[mm].std() / math.sqrt(mm.sum())),
                     "delta_prefix512": float((K["ce2b"] - K["ce1"])[mm].mean()), "delta_iter2": float((K["ce3"] - K["ce1"])[mm].mean()),
                     "delta_jitter": float((K["ce2j"] - K["ce2"])[mm].mean()), "kl": float(K["kl12"][mm].mean()),
                     "frac_better": float((d[mm] < 0).mean()), "state_upd_rel": float(K["state_upd_rel"][mm].mean())})
    report["position_bins"] = rows
    report["first_positions"] = [{"pos": int(p), "delta": float(d[pos == p].mean()), "pass1": float(K["ce1"][pos == p].mean())} for p in range(16)]
    sub = {"pass1": K["ce1"][m], "delta": d[m], "frac_better": (d[m] < 0).astype(np.float64), "kl": K["kl12"][m],
           "u_nearest_hit": K["u_nearest_hit"][m], "cos_ue": K["cos_ue"][m]}
    s = K["prev_surprise"][m]
    report["by_prev_surprise"] = binned(s, np.quantile(s, [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]), sub)
    hit = K["prev_hit"][m] > 0.5
    report["by_prev_hit"] = {name: {"n": int(sel.sum()), "delta": float(d[m][sel].mean()), "pass1": float(K["ce1"][m][sel].mean()),
                                    "u_nearest_hit": float(K["u_nearest_hit"][m][sel].mean())} for name, sel in (("hit", hit), ("miss", ~hit))}
    c = K["ce1"][m]
    report["by_own_ce1"] = binned(c, np.quantile(c, [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]), sub)
    f = K["inp_freq"][m]
    report["by_input_token_frequency"] = binned(f, np.quantile(f, [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]), sub)

    lr = np.concatenate(logit_rms)
    report["gate"] = {
        "mean": float(K["gate_mean"][m].mean()), "token_mean_quantiles": quantiles(K["gate_mean"][m]),
        "frac_coords_below_0.1": float(K["gate_lo"][m].mean()), "frac_coords_above_0.9": float(K["gate_hi"][m].mean()),
        "logit_rms_quantiles": quantiles(lr),
        "coord_std_across_tokens_quantiles": quantiles(np.mean(gate_coord_std, 0)),
        "token_dependent_share_of_prenorm_input_quantiles": quantiles(K["token_frac"]),
    }
    init_fro = 0.02 * cfg.dim
    Wg, Wu = model.fuse_gate.weight.float(), model.fuse_value.weight.float()
    report["weights"] = {
        "fuse_gate_fro": Wg.norm().item(), "fuse_value_fro": Wu.norm().item(), "fuse_gate_init_fro": init_fro,
        "fuse_gate_row_norm_quantiles": quantiles(Wg.norm(dim=1).cpu().numpy(), (0.01, 0.5, 0.99)),
        "gate_norm_weight_quantiles": quantiles(model.gate_norm.weight.float().cpu().numpy(), (0.01, 0.5, 0.99)),
        "embedding_rms": model.embed_tokens.weight.float().square().mean().sqrt().item(),
    }
    U, E = torch.cat(u_sub).float(), torch.cat(e_sub).float()
    n = U.shape[0] // 2
    lam = 1e-3 * U.shape[1]
    Wp = torch.linalg.solve(U[:n].T @ U[:n] + lam * torch.eye(cfg.dim), U[:n].T @ E[:n])
    pred = U[n:] @ Wp
    r2 = 1 - ((pred - E[n:]).square().sum() / (E[n:] - E[n:].mean(0)).square().sum()).item()
    Wemb = model.embed_tokens.weight.detach().float().cpu()
    inp_all = torch.cat([data.batch(first, min(args.micro_rows, args.rows - first))[:, :-1][:, 1::8].reshape(-1) for first in range(0, args.rows, args.micro_rows)])
    inp_v = inp_all[n:]
    hits = sum((F.linear(pred[s : s + 4096], Wemb).argmax(-1) == inp_v[s : s + 4096]).sum().item() for s in range(0, pred.shape[0], 4096))
    report["seed"] = {"cos_u_e_quantiles": quantiles(K["cos_ue"][m]), "u_nearest_token_hit": float(K["u_nearest_hit"][m].mean()),
                      "linear_probe_u_to_e_r2": r2, "linear_probe_nearest_token_hit": hits / pred.shape[0], "probe_train_tokens": int(n)}
    pmean = payload_sum / n_payload
    pstd = (payload_sq / n_payload - pmean.square()).clamp_min(0).sqrt().cpu().numpy()
    jit_rms = saved["jitter"] / math.sqrt(3)
    report["scale"] = {
        "h_top_rms_quantiles": quantiles(K["h1_rms"][m]), "h_top_top8_energy_share_quantiles": quantiles(K["h1_top8"][m]),
        "payload_top8_energy_share_quantiles": quantiles(K["p_top8"][m]), "payload_coord_std_quantiles": quantiles(pstd),
        "jitter_rms": jit_rms, "frac_payload_coords_std_below_jitter": float((pstd < jit_rms).mean()),
        "payload_norm_weight_quantiles": quantiles(model.payload_norm.weight.detach().float().cpu().numpy()),
        "entry_norm_weight_quantiles": quantiles(model.entry_norm.weight.detach().float().cpu().numpy()),
        "state_update_rel_rms_quantiles": quantiles(K["state_upd_rel"][m]),
    }
    if source_rel is not None:
        report["per_source_rel_change_pass2_vs_pass1"] = dict(zip(source_names, [v / n_batches for v in source_rel]))

    trace = []
    with analysis.autocast(device):
        tokens = data.batch(0, args.self_rows, device)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        e = model.embed_tokens(inp)
        out = model.forward_column(e, need_payload=True)
        base = analysis.token_ce(model, out.h_top, tgt).mean().item()
        for it in range(1, args.self_iters + 1):
            prev_h, prev_p = out.h_top, out.payload
            out = model.forward_column(analysis.fused_inputs(model, e, prev_p, 1), need_payload=True)
            loss = analysis.token_ce(model, out.h_top, tgt).mean().item()
            upd = (out.h_top.float() - prev_h.float()).norm(dim=-1).mean().item()
            rel = ((out.h_top.float() - prev_h.float()).square().mean(-1).sqrt() / prev_h.float().square().mean(-1).sqrt()).mean().item()
            prel = (out.payload.float() - prev_p.float()).square().mean(-1).sqrt().mean().item()
            trace.append({"iter": it, "loss": loss, "update_l2": upd, "update_rel_rms": rel, "payload_update_rms": prel})
    report["self_composition"] = {"rows": args.self_rows, "pass1": base, "trace": trace}

    out_path = out_dir / "fused_diagnostics.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    M = report["means"]
    print("\n== headline (all positions) ==")
    for k, v in M.items():
        print(f"  {k:<52} {v:.4f}")
    print(f"\n== per-token delta (fused - pass1), pos>=1: mean {report['delta_mean_pos_ge1']:+.4f}, frac better {report['delta_frac_negative']:.3f}")
    print("  quantiles:", {k: round(v, 3) for k, v in report["delta_quantiles"].items()})
    print("\n== by position ==")
    print(f"  {'bin':<10}{'n':>8}{'pass1':>8}{'fused':>8}{'delta':>8}{'se':>8}{'d_pre512':>9}{'d_iter2':>9}{'d_jit':>8}{'kl':>7}{'better':>8}{'upd':>7}")
    for r in rows:
        print(f"  {r['bin']:<10}{r['n']:>8}{r['pass1']:>8.3f}{r['fused']:>8.3f}{r['delta']:>+8.4f}{r['delta_se']:>8.4f}{r['delta_prefix512']:>+9.4f}{r['delta_iter2']:>+9.4f}{r['delta_jitter']:>+8.4f}{r['kl']:>7.3f}{r['frac_better']:>8.3f}{r['state_upd_rel']:>7.3f}")
    print("  first positions:", " ".join(f"{r['pos']}:{r['delta']:+.3f}" for r in report["first_positions"]))
    for title, key in (("by previous-column surprise of the fused-in token", "by_prev_surprise"),
                       ("by own pass-1 CE (regression to the mean)", "by_own_ce1"), ("by input-token frequency", "by_input_token_frequency")):
        print(f"\n== {title} ==")
        for r in report[key]:
            print(f"  [{r['range'][0]:9.2f},{r['range'][1]:9.2f}) n={r['n']:>7} pass1={r['pass1']:.3f} delta={r['delta']:+.4f} better={r['frac_better']:.3f} kl={r['kl']:.3f} uhit={r['u_nearest_hit']:.3f} cos={r['cos_ue']:.3f}")
    print("  prev-argmax hit:", report["by_prev_hit"])
    print("\n== gate ==", json.dumps(report["gate"]))
    print("== weights ==", json.dumps(report["weights"]))
    print("== seed ==", json.dumps(report["seed"]))
    print("== scale ==", json.dumps(report["scale"]))
    if source_rel is not None:
        print("== per-source relative change, pass 2 vs pass 1 ==", json.dumps(report["per_source_rel_change_pass2_vs_pass1"]))
    print("\n== self-composition ==  pass1", f"{base:.4f}")
    for r in trace:
        print(f"  it {r['iter']:>2} loss {r['loss']:.4f} upd_l2 {r['update_l2']:.3f} rel {r['update_rel_rms']:.4f} payload_upd {r['payload_update_rms']:.4f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

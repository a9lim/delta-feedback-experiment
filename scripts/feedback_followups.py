"""Follow-up same-checkpoint tests on a feedback snapshot.

* **Impulse response:** fuse a block of positions and return to plain
  embeddings; the per-lag penalty after the block measures what a fused column
  writes into the mixer states that plain columns then read.  That boundary
  never occurs in training, so this is an out-of-distribution probe.
* **Payload-head ablation** (`df` only): replace one routing head's payload
  slice with the site null, or with another row's slice.
* **Split-validated ensemble:** temperatures and a mixture weight for the
  pass-1 and fused predictors are chosen on the calibration rows and scored on
  the rest.
* **Gradient alignment:** cosine between the gradients of the pass-1 loss and
  the fused-pass loss on fresh training rows, per parameter family.

Usage:
    python scripts/feedback_followups.py runs/TAG.pt.STEP --data-dir /data/df/tokens
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from delta_feedback_experiment import analysis
from delta_feedback_experiment.data import TokenData
from delta_feedback_experiment.model import multipass, multipass_loss, shift_right
from delta_feedback_experiment.train import micro_draws

FUSION_NAMES = {"fuse_value.weight", "fuse_gate.weight", "gate_norm.weight", "entry_norm.weight", "payload_norm.weight"}


def family(name: str) -> str:
    if name in FUSION_NAMES:
        return "fusion"
    if name.startswith("payload_router."):
        return "payload_router"
    if name.startswith("embed_tokens"):
        return "embedding"
    if "_router." in name:
        return "column_routers"
    if name.startswith("attention_gates"):
        return "attention_gates"
    if name.startswith("blocks.") and name.endswith("weight") and "norm" not in name and "conv" not in name and "proj" in name:
        return "trunk_matrices"
    return "trunk_other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--data-dir", default="data/tokens")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--calib-rows", type=int, default=64)
    parser.add_argument("--impulse-rows", type=int, default=64)
    parser.add_argument("--micro-rows", type=int, default=4)
    parser.add_argument("--gradient-micros", type=int, default=8, help="fresh training microbatches for the gradient test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    torch._dynamo.config.recompile_limit = 64
    torch.set_float32_matmul_precision("high")
    model, saved = analysis.load_checkpoint(args.snapshot, args.device)
    device = next(model.parameters()).device
    if not model.cfg.feedback_active:
        raise SystemExit("the follow-ups need an fbt or df snapshot")
    cfg = model.cfg
    run = SimpleNamespace(**saved)
    T = saved["seq_len"]
    tag = saved.get("tag", args.snapshot.stem)
    out_dir = args.out_dir or Path("figures") / f"fused-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = TokenData.load(args.data_dir, "val", T)
    data_train = TokenData.load(args.data_dir, "train", T)
    positions = torch.arange(T, device=device)
    plain1 = analysis.plain_mask(T, 1, device)
    rep: dict = {"snapshot": str(args.snapshot), "rows": args.rows, "impulse_rows": args.impulse_rows}
    out_path = out_dir / "feedback_followups.json"

    # -- impulse response and payload-head ablation --------------------------------
    starts = (64, 256, 512, 768)
    lags = (0, 1, 2, 4, 8, 16, 32, 64, 128)
    imp = {L: {t0: {lag: 0.0 for lag in lags} for t0 in starts} for L in (1, 8, 64)}
    routed_payload = cfg.routing_active
    head_abl = {f"head{h}_to_null": 0.0 for h in range(cfg.routing_heads)} if routed_payload else {}
    if routed_payload:
        head_abl["head2_shuffled_rows"] = 0.0
    fused_sum = 0.0
    n_imp = 0
    with torch.no_grad(), analysis.autocast(device):
        for first in range(0, args.impulse_rows, args.micro_rows):
            tokens = data.batch(first, min(args.micro_rows, args.impulse_rows - first), device)
            inp, tgt = tokens[:, :-1], tokens[:, 1:]
            e = model.embed_tokens(inp)
            out1 = model.forward_column(e, need_payload=True)
            ce1 = analysis.token_ce(model, out1.h_top, tgt)
            fused_all = model.fuse(shift_right(out1.payload), e)
            fused_sum += analysis.token_ce(model, model.forward_column(torch.where(plain1, e, fused_all), need_payload=False).h_top, tgt).sum().item()
            for L in imp:
                for t0 in starts:
                    mask = ((positions >= t0) & (positions < t0 + L))[None, :, None]
                    out = model.forward_column(torch.where(mask, fused_all, e), need_payload=False)
                    d = analysis.token_ce(model, out.h_top, tgt) - ce1
                    for lag in lags:
                        imp[L][t0][lag] += d[:, t0 + L - 1 + lag].sum().item() if t0 + L - 1 + lag < T else float("nan")
            if routed_payload:
                payload_sources = [out1.sources[0], *out1.sources[1:]]
                routed, _ = model.payload_router(payload_sources, [None] * len(payload_sources), False)
                hd = cfg.dim // cfg.routing_heads
                null = model.payload_router.null.to(routed.dtype)
                for h in range(cfg.routing_heads):
                    r = routed.clone()
                    r[..., h * hd : (h + 1) * hd] = null[h * hd : (h + 1) * hd]
                    p = model.payload_norm(out1.h_top + r)
                    out = model.forward_column(torch.where(plain1, e, model.fuse(shift_right(p), e)), need_payload=False)
                    head_abl[f"head{h}_to_null"] += analysis.token_ce(model, out.h_top, tgt).sum().item()
                r = routed.clone()
                perm = torch.roll(torch.arange(tokens.shape[0], device=device), 1)
                r[..., 2 * hd : 3 * hd] = routed[perm][..., 2 * hd : 3 * hd]
                p = model.payload_norm(out1.h_top + r)
                out = model.forward_column(torch.where(plain1, e, model.fuse(shift_right(p), e)), need_payload=False)
                head_abl["head2_shuffled_rows"] += analysis.token_ce(model, out.h_top, tgt).sum().item()
            n_imp += tokens.shape[0]
    rep["impulse"] = {str(L): {str(t0): {str(lag): v / n_imp for lag, v in d.items()} for t0, d in dd.items()} for L, dd in imp.items()}
    rep["impulse_fused_prefix1"] = fused_sum / (n_imp * T)
    if routed_payload:
        rep["payload_head_ablation"] = {k: v / (n_imp * T) for k, v in head_abl.items()}
    out_path.write_text(json.dumps(rep, indent=2) + "\n")

    # -- split-validated calibrated ensemble ------------------------------------------
    T1s, T2s, ws = (1.0, 1.05, 1.1), (1.0, 1.05, 1.1, 1.2), (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    grid = list(itertools.product(T1s, T2s, ws))
    sums = {"calib": np.zeros(len(grid)), "test": np.zeros(len(grid))}
    counts = {"calib": 0, "test": 0}
    with torch.no_grad(), analysis.autocast(device):
        for first in range(0, args.rows, args.micro_rows):
            split = "calib" if first < args.calib_rows else "test"
            tokens = data.batch(first, min(args.micro_rows, args.rows - first), device)
            inp, tgt = tokens[:, :-1], tokens[:, 1:]
            e = model.embed_tokens(inp)
            out1 = model.forward_column(e, need_payload=True)
            out2 = model.forward_column(analysis.fused_inputs(model, e, out1.payload, 1), need_payload=False)
            W = model.embed_tokens.weight.to(out1.h_top.dtype)
            for s in range(0, T, 256):
                t = tgt[:, s : s + 256]
                z1 = F.linear(model.final_norm(out1.h_top[:, s : s + 256]), W).float()
                z2 = F.linear(model.final_norm(out2.h_top[:, s : s + 256]), W).float()
                l1 = {T1: (z1 / T1).log_softmax(-1) for T1 in T1s}
                l2 = {T2: (z2 / T2).log_softmax(-1) for T2 in T2s}
                for gi, (T1, T2, w) in enumerate(grid):
                    lm = l1[T1] if w == 0.0 else torch.logaddexp(math.log(1 - w) + l1[T1], math.log(w) + l2[T2])
                    sums[split][gi] += -lm.gather(-1, t[..., None]).squeeze(-1).sum().item()
            counts[split] += tgt.numel()
    calib, test = sums["calib"] / counts["calib"], sums["test"] / counts["test"]
    best = int(np.argmin(calib))
    p1_only = [gi for gi, (T1, T2, w) in enumerate(grid) if w == 0.0]
    best_p1 = p1_only[int(np.argmin(calib[p1_only]))]
    raw_p1 = grid.index((1.0, 1.0, 0.0))
    rep["ensemble_split"] = {
        "grid_best_on_calib": {"T1": grid[best][0], "T2": grid[best][1], "w": grid[best][2], "calib": float(calib[best]), "test": float(test[best])},
        "pass1_best_temperature_on_calib": {"T1": grid[best_p1][0], "calib": float(calib[best_p1]), "test": float(test[best_p1])},
        "pass1_raw": {"calib": float(calib[raw_p1]), "test": float(test[raw_p1])},
        "test_gain_over_calibrated_pass1": float(test[best_p1] - test[best]),
        "test_by_w_at_unit_temperature": {f"{w:.1f}": float(test[grid.index((1.0, 1.0, w))]) for w in ws},
    }
    out_path.write_text(json.dumps(rep, indent=2) + "\n")
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # -- gradient alignment on fresh training rows ------------------------------------
    model.train()
    model.grad_checkpoint = True
    fams = sorted({family(n) for n, _ in model.named_parameters()})
    acc: dict[str, dict[str, torch.Tensor]] = {"ell1": {}, "ell2": {}}
    start_step = saved["steps"] + 1
    for which, n_passes in (("ell1", 1), ("ell2", 2)):
        model.zero_grad(set_to_none=True)
        for micro in range(args.gradient_micros):
            first_row = (start_step - 1) * saved["batch_rows"] + micro * args.micro_rows
            rows = data_train.batch(first_row, args.micro_rows, device)
            prefix = jitter = None
            if n_passes > 1:
                prefix, jitter = micro_draws(run, start_step, first_row, n_passes, rows.shape[0], cfg.dim, device)
            with analysis.autocast(device):
                outs = multipass(model, rows, n_passes, prefix_lens=prefix, jitter=jitter)
                _, losses = multipass_loss(model, rows, outs)
            (losses[-1] / args.gradient_micros).backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                acc[which][n] = p.grad.detach().float().cpu()
        model.zero_grad(set_to_none=True)
    conflict = {}
    for fam in fams:
        names = [n for n, _ in model.named_parameters() if family(n) == fam]
        g1 = torch.cat([acc["ell1"][n].reshape(-1) for n in names if n in acc["ell1"]]) if any(n in acc["ell1"] for n in names) else None
        g2 = torch.cat([acc["ell2"][n].reshape(-1) for n in names if n in acc["ell2"]]) if any(n in acc["ell2"] for n in names) else None
        entry = {"weight_norm": math.sqrt(sum(p.detach().float().square().sum().item() for n, p in model.named_parameters() if family(n) == fam))}
        if g1 is not None:
            entry["grad_norm_ell1"] = g1.norm().item()
        if g2 is not None:
            entry["grad_norm_ell2"] = g2.norm().item()
        if g1 is not None and g2 is not None and g1.numel() == g2.numel():
            entry["cosine"] = F.cosine_similarity(g1, g2, dim=0).item()
        conflict[fam] = entry
    shared = [n for n in acc["ell1"] if n in acc["ell2"]]
    all1 = torch.cat([acc["ell1"][n].reshape(-1) for n in shared])
    all2 = torch.cat([acc["ell2"][n].reshape(-1) for n in shared])
    conflict["all_shared"] = {"grad_norm_ell1": all1.norm().item(), "grad_norm_ell2": all2.norm().item(),
                              "cosine": F.cosine_similarity(all1, all2, dim=0).item()}
    rep["gradient_conflict"] = conflict
    out_path.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "impulse"}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

"""Same-checkpoint interventions at the FBT entry of a feedback snapshot.

Runs one fused pass (prefix 1) while replacing the fused seed: gate
temperature (0 = a constant 0.5 gate, which removes the token from fused
positions), fused-seed rescaling, an additive raw-embedding bypass, a zero
payload, and a payload taken from a different row.  Every variant is scored
as held-out cross-entropy against the same rows' pass 1.  These are
co-adapted perturbations: they bound what the trained fused pathway depends
on, not what an alternative design would achieve.

Usage:
    python scripts/entry_sweeps.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from delta_feedback_experiment import analysis
from delta_feedback_experiment.data import TokenData
from delta_feedback_experiment.model import shift_right

GATE_TEMPERATURES = (0.0, 0.5, 2.0, 4.0, 8.0)
SEED_SCALES = (0.07, 0.25, 0.5, 2.0)
BYPASS_WEIGHTS = (1.0, 4.0, 14.0)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--data-dir", default="data/dclm-100b")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--micro-rows", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    torch._dynamo.config.recompile_limit = 64
    model, saved = analysis.load_checkpoint(args.snapshot, args.device)
    device = next(model.parameters()).device
    if not model.cfg.feedback_active:
        raise SystemExit("entry sweeps need a snapshot of a condition with f")
    T = saved["seq_len"]
    tag = saved.get("tag", args.snapshot.stem)
    out_dir = args.out_dir or Path("figures") / f"fused-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = TokenData.load(args.data_dir, "val", T)
    plain1 = analysis.plain_mask(T, 1, device)

    variants = ["pass1", "fused"]
    variants += [f"gate_tau_{t}" for t in GATE_TEMPERATURES]
    variants += [f"seed_scale_{s}" for s in SEED_SCALES]
    variants += [f"bypass_beta_{b}" for b in BYPASS_WEIGHTS]
    variants += ["payload_zero", "payload_shuffled_rows"]
    sums = {v: 0.0 for v in variants}
    n = 0
    with analysis.autocast(device):
        for first in range(0, args.rows, args.micro_rows):
            tokens = data.batch(first, min(args.micro_rows, args.rows - first), device)
            inp, tgt = tokens[:, :-1], tokens[:, 1:]
            e = model.embed_tokens(inp)
            out1 = model.forward_column(e, need_payload=True)
            sums["pass1"] += analysis.token_ce(model, out1.h_top, tgt).sum().item()
            ps = shift_right(out1.payload)
            logit = model.fuse_gate(model.gate_norm(e))
            value = model.fuse_value(ps)

            def run(u):
                out = model.forward_column(torch.where(plain1, e, u), need_payload=False)
                return analysis.token_ce(model, out.h_top, tgt).sum().item()

            base = model.entry_norm(value * torch.sigmoid(logit))
            sums["fused"] += run(base)
            for t in GATE_TEMPERATURES:
                sums[f"gate_tau_{t}"] += run(model.entry_norm(value * torch.sigmoid(t * logit)))
            for s in SEED_SCALES:
                sums[f"seed_scale_{s}"] += run(base * s)
            for b in BYPASS_WEIGHTS:
                sums[f"bypass_beta_{b}"] += run(base + b * e)
            sums["payload_zero"] += run(model.entry_norm(model.fuse_value(torch.zeros_like(ps)) * torch.sigmoid(logit)))
            perm = torch.roll(torch.arange(tokens.shape[0], device=device), 1)
            sums["payload_shuffled_rows"] += run(model.entry_norm(model.fuse_value(ps[perm]) * torch.sigmoid(logit)))
            n += tgt.numel()
    report = {"snapshot": str(args.snapshot), "rows": args.rows, "losses": {v: sums[v] / n for v in variants}}
    report["penalty_vs_pass1"] = {v: report["losses"][v] - report["losses"]["pass1"] for v in variants}
    out_path = out_dir / "entry_sweeps.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    for v in variants:
        print(f"{v:<24} {report['losses'][v]:.4f}  ({report['penalty_vs_pass1'][v]:+.4f} vs pass 1)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

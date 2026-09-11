"""Counterfactual payload sweep: what should ride to the next column?

Forces the MHDB payload enrichment to each null, seed, or completed-block
source in turn (plus none / uniform), either across all heads or in one
selected head, and measures fused val loss under the training eval convention
(one fused pass, prefix length 1).  Answers whether the trained router's choice
is a family preference, a specific peak, or a no-op.

Caveat: the fuse/entry weights co-adapted to the *trained* payload, so
alternatives are handicapped: read the landscape's shape (which family helps,
which hurts, how peaked), not absolute gaps.

Usage:
    python scripts/payload_swap.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b
    python scripts/payload_swap.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b --head 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from delta_feedback_experiment import analysis
from delta_feedback_experiment.data import TokenData
from delta_feedback_experiment.model import multipass, multipass_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--data-dir", default="data/dclm-100b")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--micro-rows", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="replace only this routing head; default replaces every head",
    )
    args = parser.parse_args()

    model, saved = analysis.load_checkpoint(args.snapshot, args.device)
    device = next(model.parameters()).device
    cfg = model.cfg
    if not cfg.feedback or not cfg.block_routing:
        raise SystemExit("the payload sweep needs a routed payload: a condition with r and f")
    if args.head is not None and not 0 <= args.head < cfg.kv_heads:
        raise SystemExit(
            f"--head must be in [0, {cfg.kv_heads - 1}], got {args.head}"
        )
    tag = saved["tag"]
    out_dir = args.out_dir or Path("figures") / f"fused-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_val = TokenData.load(args.data_dir, "val", saved["seq_len"])

    @torch.no_grad()
    def losses() -> tuple[float, float]:
        sums = [0.0, 0.0]
        batches = 0
        for first in range(0, args.rows, args.micro_rows):
            rows = data_val.batch(
                first, min(args.micro_rows, args.rows - first), device
            )
            prefix = torch.ones((1, rows.shape[0]), dtype=torch.long, device=device)
            with analysis.autocast(device):
                outs = multipass(model, rows, 2, prefix_lens=prefix)
                _, per_pass = multipass_loss(model, rows, outs)
            sums[0] += per_pass[0].item()
            sums[1] += per_pass[1].item()
            batches += 1
        return sums[0] / batches, sums[1] / batches

    names = ["null", "seed"] + [f"block{i}" for i in range(cfg.routing_blocks)]
    val, fused = losses()
    report = {"snapshot": str(args.snapshot), "rows": args.rows, "head": args.head,
              "pass1": val, "trained_router": fused}
    print(f"trained router : val={val:.4f}  fused={fused:.4f}")

    trained_forward = model.payload_router.forward
    model.payload_router.forward = lambda sources, masks, want: (None, None)
    report["h_top_only"] = losses()[1]
    print(f"h_top only     : fused={report['h_top_only']:.4f}")
    model.payload_router.forward = lambda sources, masks, want: (
        torch.stack([model.payload_router.null.expand_as(sources[0]), *sources]).mean(
            0
        ),
        None,
    )
    report["uniform"] = losses()[1]
    print(f"uniform        : fused={report['uniform']:.4f}")

    def selected_source(sources, index):
        if index == 0:
            return model.payload_router.null.expand_as(sources[0])
        return sources[index - 1]

    results = []
    for i, name in enumerate(names):
        if args.head is None:
            model.payload_router.forward = lambda sources, masks, want, i=i: (
                selected_source(sources, i),
                None,
            )
        else:

            def force_one_head(sources, masks, want, i=i):
                routed, _ = trained_forward(sources, masks, want)
                batch, length, dim = routed.shape
                head_dim = dim // cfg.kv_heads
                routed = routed.reshape(batch, length, cfg.kv_heads, head_dim)
                routed = routed.clone()
                chosen = selected_source(sources, i)
                routed[:, :, args.head] = chosen.reshape(
                    batch, length, cfg.kv_heads, head_dim
                )[:, :, args.head]
                return routed.reshape(batch, length, dim), None

            model.payload_router.forward = force_one_head
        results.append((name, losses()[1]))
        print(f"force {name:<4}     : fused={results[-1][1]:.4f}", flush=True)
    report["forced"] = dict(results)

    print("\nsorted:")
    for name, value in sorted(results, key=lambda t: t[1]):
        print(f"  {name:<4} {value:.4f}")
    suffix = "" if args.head is None else f"_head{args.head}"
    out_path = out_dir / f"payload_swap{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

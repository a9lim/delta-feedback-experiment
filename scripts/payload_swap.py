"""Counterfactual payload sweep: what should ride to the next column?

Forces the MHDAR payload enrichment to each single delta in turn (plus none /
uniform), either across all heads or in one selected head, and measures fused val loss under the
training eval convention (one fused pass, prefix length 1).  Answers
whether the trained router's choice is a family preference, a specific
peak, or a no-op.

Caveat: the fuse/entry weights co-adapted to the *trained* payload, so
alternatives are handicapped — read the landscape's shape (which family
helps, which hurts, how peaked), not absolute gaps.

Usage:
    python scripts/payload_swap.py runs/ladder-df.pt.6700
    python scripts/payload_swap.py runs/ladder-df.pt.6700 --head 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformer_experiments import checkpoints

from delta_feedback_experiment.data import TokenData
from delta_feedback_experiment.model import (
    DFModel,
    arm_config,
    multipass,
    multipass_loss,
)
from delta_feedback_experiment.train import CONTRACT, pick_device

GEOMETRY = (
    "vocab_size",
    "dim",
    "layers",
    "heads",
    "kv_heads",
    "head_dim",
    "intermediate",
)


def main() -> None:
    parser = argparse.ArgumentParser("payload swap")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--data-dir", default="data/tokens")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--micro-rows", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="replace only this routing head; default replaces every head",
    )
    args = parser.parse_args()

    device = pick_device(args.device)
    payload = checkpoints.read(args.snapshot, CONTRACT, map_location="cpu")
    saved = payload["args"]
    cfg = arm_config(
        saved["arm"],
        max_seq_len=saved["seq_len"] + 1,
        **{f: saved[f] for f in GEOMETRY},
    )
    if not cfg.feedback_active or not cfg.routing_active:
        raise SystemExit("the payload sweep needs a payload router (df/df_soft)")
    if args.head is not None and not 0 <= args.head < cfg.routing_heads:
        raise SystemExit(
            f"--head must be in [0, {cfg.routing_heads - 1}], got {args.head}"
        )
    model = DFModel(cfg)
    model.load_state_dict(payload["state"])
    model = model.to(device).eval()
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
            outs = multipass(model, rows, 2, prefix_lens=prefix)
            _, per_pass = multipass_loss(model, rows, outs)
            sums[0] += per_pass[0].item()
            sums[1] += per_pass[1].item()
            batches += 1
        return sums[0] / batches, sums[1] / batches

    names = [f"{kind}{i}" for i in range(cfg.layers) for kind in ("a", "m")]
    val, fused = losses()
    print(f"trained router : val={val:.4f}  fused={fused:.4f}")

    trained_forward = model.payload_router.forward
    model.payload_router.forward = lambda sources, masks, want: (None, None)
    print(f"h_top only     : fused={losses()[1]:.4f}")
    model.payload_router.forward = lambda sources, masks, want: (
        torch.stack(sources).mean(0),
        None,
    )
    print(f"uniform        : fused={losses()[1]:.4f}")

    results = []
    for i, name in enumerate(names):
        if args.head is None:
            model.payload_router.forward = lambda sources, masks, want, i=i: (
                sources[i],
                None,
            )
        else:

            def force_one_head(sources, masks, want, i=i):
                routed, _ = trained_forward(sources, masks, want)
                batch, length, dim = routed.shape
                head_dim = dim // cfg.routing_heads
                routed = routed.reshape(batch, length, cfg.routing_heads, head_dim)
                routed = routed.clone()
                routed[:, :, args.head] = sources[i].reshape(
                    batch, length, cfg.routing_heads, head_dim
                )[:, :, args.head]
                return routed.reshape(batch, length, dim), None

            model.payload_router.forward = force_one_head
        results.append((name, losses()[1]))
        print(f"force {name:<4}     : fused={results[-1][1]:.4f}", flush=True)

    print("\nsorted:")
    for name, value in sorted(results, key=lambda t: t[1]):
        print(f"  {name:<4} {value:.4f}")


if __name__ == "__main__":
    main()

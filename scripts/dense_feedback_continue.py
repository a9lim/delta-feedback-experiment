"""Same-checkpoint intervention: continue a feedback snapshot with dense feedback passes.

Restores the snapshot (model and both optimizer states) and trains every step
with ``k`` feedback passes on fresh rows past the schedule at a
small constant learning rate, tracking ``val`` and ``val_fused`` under the
run's evaluation convention.  ``--passes 1`` is the plain-only control that
measures erosion of the fused mode; ``--trainable fusion`` restricts updates to
the FBT interface.  The output is a JSON trace.

Usage:
    python scripts/dense_feedback_continue.py runs/TAG.pt.STEP --data-dir /data/delta/dclm-100b \\
        --steps 150 --passes 2 --out figures/fused-TAG/dense_all.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformer_experiments import checkpoints

from delta_feedback_experiment import analysis
from delta_feedback_experiment.data import TokenData
from delta_feedback_experiment.model import (
    DeltaModel,
    combine_pass_losses,
    multipass,
    multipass_loss,
)
from delta_feedback_experiment.optim import OptimizerPair, build_optimizers
from delta_feedback_experiment.train import (
    CONTRACT,
    clip_gradients,
    micro_draws,
    read_checkpoint,
)

FUSION_NAMES = ("fuse_value.weight", "fuse_gate.weight", "gate_norm.weight", "entry_norm.weight",
                "payload_norm.weight", "payload_router.query", "payload_router.null", "payload_router.key_norm.weight")


@torch.no_grad()
def evaluate(model, data_val, rows, micro, device, *, mtp_weight):
    was_training = model.training
    model.eval()
    sums = {"val": 0.0, "val_fused": 0.0}
    if model.cfg.mtp:
        sums.update(val_mtp=0.0, val_mtp_fused=0.0)
    for first in range(0, rows, micro):
        batch = data_val.batch(first, min(micro, rows - first), device)
        prefix = torch.ones((1, batch.shape[0]), dtype=torch.long, device=device)
        with analysis.autocast(device):
            outs = multipass(model, batch, 2, prefix_lens=prefix)
            result = multipass_loss(model, batch, outs, mtp_weight=mtp_weight)
        sums["val"] += result.ntp[0].item() * batch.shape[0]
        sums["val_fused"] += result.ntp[1].item() * batch.shape[0]
        if model.cfg.mtp:
            sums["val_mtp"] += result.mtp[0].item() * batch.shape[0]
            sums["val_mtp_fused"] += result.mtp[1].item() * batch.shape[0]
    model.train(was_training)
    return {name: value / rows for name, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--data-dir", default="data/dclm-100b")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--trainable", choices=("all", "fusion"), default="all")
    parser.add_argument("--lr-scale", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-rows", type=int, default=32)
    parser.add_argument("--batch-rows", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=Path("dense_feedback.json"))
    args = parser.parse_args()

    torch._dynamo.config.recompile_limit = 64
    torch.set_float32_matmul_precision("high")
    payload = read_checkpoint(args.snapshot)
    saved = analysis.saved_args(payload)
    run = SimpleNamespace(**saved)
    cfg = analysis.config_from_args(saved)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not cfg.feedback:
        raise SystemExit("continuation needs a snapshot of a condition with f")
    torch.manual_seed(saved["seed"])
    model = DeltaModel(cfg).to(device)
    optimizers = build_optimizers(model, lr_normuonh=saved["lr_normuonh"], lr_nadam=saved["lr_nadam"])
    pair = OptimizerPair(optimizers)
    start_step = checkpoints.restore(payload, CONTRACT, model, pair, current_optimizer_groups=False)
    del payload
    print(f"restored step {start_step}; trainable={args.trainable} passes={args.passes} lr_scale={args.lr_scale}", flush=True)

    if args.trainable == "fusion":
        n_train = 0
        for name, p in model.named_parameters():
            p.requires_grad_(name in FUSION_NAMES)
            n_train += p.numel() if name in FUSION_NAMES else 0
        print(f"fusion-only: {n_train} trainable parameters", flush=True)

    batch_rows = args.batch_rows or saved["batch_rows"]
    micro_rows = saved["micro_rows"]
    micros = batch_rows // micro_rows
    T = saved["seq_len"]
    data_train = TokenData.load(args.data_dir, "train", T)
    data_val = TokenData.load(args.data_dir, "val", T)
    model.train()
    model.grad_checkpoint = True
    model.refresh_shadows()

    trace = {"args": vars(args) | {"snapshot": str(args.snapshot), "mtp_weight": saved["mtp_weight"]}, "start_step": start_step, "steps": [], "evals": []}
    metrics = evaluate(model, data_val, args.eval_rows, micro_rows, device, mtp_weight=saved["mtp_weight"])
    trace["evals"].append({"step": 0, **metrics})
    print("eval step 0: " + " ".join(f"{name}={value:.4f}" for name, value in metrics.items())
          + f" gap={metrics['val_fused'] - metrics['val']:+.4f}", flush=True)

    t_start = time.time()
    for i in range(1, args.steps + 1):
        step = start_step + i  # fresh rows: the schedule never reached these
        scale = args.lr_scale * min(1.0, i / max(args.warmup, 1))
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = group["stable_lr"] * scale
        step_loss = step_ntp = step_mtp = pass1 = 0.0
        t0 = time.time()
        for micro in range(micros):
            first_row = (step - 1) * saved["batch_rows"] + micro * micro_rows
            rows = data_train.batch(first_row, micro_rows, device)
            prefix = jitter = None
            if args.passes > 1:
                prefix, jitter = micro_draws(run, step, first_row, args.passes, rows.shape[0], cfg.dim, device)
            with analysis.autocast(device):
                outs = multipass(model, rows, args.passes, prefix_lens=prefix, jitter=jitter)
                loss_result = multipass_loss(model, rows, outs, z_coef=saved["zloss"], mtp_weight=saved["mtp_weight"])
                loss, losses = loss_result.total, loss_result.ntp
            (loss / micros).backward()
            step_loss += loss.item() / micros
            step_ntp += combine_pass_losses(loss_result.ntp).item() / micros
            if cfg.mtp:
                step_mtp += combine_pass_losses(loss_result.mtp).item() / micros
            pass1 += losses[0].item() / micros
        gnorm = clip_gradients([p for p in model.parameters() if p.requires_grad])
        for optimizer in optimizers:
            optimizer.step()
        model.refresh_shadows()
        model.zero_grad(set_to_none=True)
        feedback = step_ntp - pass1 if args.passes > 1 else None
        rec = {"step": i, "loss": step_loss, "ntp": step_ntp, "pass1": pass1, "feedback": feedback, "gnorm": gnorm,
               "lr_scale": scale, "sec": time.time() - t0}
        if cfg.mtp:
            rec["mtp"] = step_mtp
        trace["steps"].append(rec)
        feedback_text = f"{feedback:.4f}" if feedback is not None else "unavailable"
        mtp_text = f"mtp={step_mtp:.4f} " if cfg.mtp else ""
        print(f"step {i:4d} loss={step_loss:.4f} ntp={step_ntp:.4f} pass1={pass1:.4f} fb={feedback_text} {mtp_text}gnorm={gnorm:.3f} "
              f"lr={scale:.3f}x {rec['sec']:.1f}s", flush=True)
        if i % args.eval_every == 0 or i == args.steps:
            metrics = evaluate(model, data_val, args.eval_rows, micro_rows, device, mtp_weight=saved["mtp_weight"])
            trace["evals"].append({"step": i, **metrics})
            print(f"eval step {i}: " + " ".join(f"{name}={value:.4f}" for name, value in metrics.items())
                  + f" gap={metrics['val_fused'] - metrics['val']:+.4f}  [{(time.time() - t_start) / 60:.1f} min]", flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(trace, indent=2, default=str) + "\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trace, indent=2, default=str) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

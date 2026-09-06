"""Short paired stability check for kernel changes, without modifying any run."""

from __future__ import annotations

import argparse
import gc
import json
import time
from importlib.metadata import version
from pathlib import Path

import cut_cross_entropy
import fla
import torch
from kernel_qualification import revision

import delta_feedback_experiment
from delta_feedback_experiment.data import TokenData
from delta_feedback_experiment.model import DFModel, arm_config, iterate_fused
from delta_feedback_experiment.optim import build_optimizers
from delta_feedback_experiment.train import (
    CudaEvalRunner,
    CudaGraphTrainer,
    GraphSpec,
    build_parser,
    build_schedule,
    clip_gradients,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--data-dir", default="/data/df/tokens")
    parser.add_argument("--updates", type=int, default=12)
    options = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(1)
    args = build_parser().parse_args(["kernel-stability", "--arm", "df"])
    payload = torch.load(options.snapshot, map_location="cpu", weights_only=False)
    saved = payload["args"]
    fields = (
        "vocab_size",
        "dim",
        "layers",
        "heads",
        "kv_heads",
        "head_dim",
        "intermediate",
        "pkda_heads",
        "pkda_head_dim",
        "pkda_conv_size",
    )
    for field in (*fields, "seq_len"):
        setattr(args, field, saved[field])
    args.arm = saved["arm"]
    model = DFModel(
        arm_config(
            saved["arm"],
            max_seq_len=args.seq_len + 1,
            **{field: getattr(args, field) for field in fields},
        )
    )
    model.load_state_dict(payload["state"])
    del payload
    gc.collect()
    model.cuda().train()
    optimizers = build_optimizers(
        model, lr_normuonh=args.lr_normuonh, lr_nadam=args.lr_nadam
    )
    train = TokenData.load(options.data_dir, "train", args.seq_len)
    validation = TokenData.load(options.data_dir, "val", args.seq_len)
    torch.cuda.reset_peak_memory_stats()
    prepared_at = time.perf_counter()
    runner = CudaGraphTrainer(model, optimizers, args, build_schedule(args))
    evaluation = CudaEvalRunner(model, args, runner.pool)
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "snapshot": str(options.snapshot),
                "runtime": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "triton": version("triton"),
                    "gpu": torch.cuda.get_device_name(),
                },
                "capture": {
                    "prepare_seconds": time.perf_counter() - prepared_at,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                },
                "revisions": {
                    name: revision(Path(module.__file__).resolve().parents[1])
                    for name, module in (
                        ("experiment", delta_feedback_experiment),
                        ("fla", fla),
                        ("cce", cut_cross_entropy),
                    )
                },
                "inputs": {
                    "data_dir": options.data_dir,
                    "first_row": 100000,
                    "first_randomness_step": 9000,
                    "data_seed": args.data_seed,
                    "jitter": args.jitter,
                    "zloss": args.zloss,
                    "batch_rows": args.batch_rows,
                    "micro_rows": args.micro_rows,
                    "seq_len": args.seq_len,
                },
                "updates": options.updates,
                "optimizer_state": "fresh, identical in baseline and candidate",
                "learning_rates": [args.lr_normuonh, args.lr_nadam],
                "pass_pattern": [1, 1, 1, 2, 2, 3],
                "initial_validation": evaluation.run(validation),
            }
        ),
        flush=True,
    )
    for index in range(options.updates):
        passes = (1, 1, 1, 2, 2, 3)[index % 6]
        state = runner.begin(GraphSpec(passes, False), args.zloss)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for micro in range(args.batch_rows // args.micro_rows):
            first = 100000 + index * args.batch_rows + micro * args.micro_rows
            runner.replay(
                state, train.batch(first, args.micro_rows), 9000 + index, first
            )
        runner.prepare_optimizer(state)
        norm = clip_gradients(model.parameters())
        for optimizer in optimizers:
            optimizer.step()
        model.refresh_shadows()
        runner.zero_grad()
        torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "update": index + 1,
                    "passes": passes,
                    "loss": state.loss_sum.item(),
                    "pass1_loss": state.pass1_sum.item(),
                    "gradient_norm": norm,
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    final_validation = evaluation.run(validation)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        contraction = iterate_fused(model, validation.batch(0, 2, "cuda"), n_iters=8)
    print(
        json.dumps({"final_validation": final_validation, "contraction": contraction}),
        flush=True,
    )


if __name__ == "__main__":
    main()

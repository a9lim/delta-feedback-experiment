"""Fresh-model paired CUDA qualification; synthetic rows do not test model quality.

Run this same file from each checkout with PYTHONPATH pointing at that checkout.
The full-family option also captures evaluation in the shared graph pool.
``--checkpoint-policy blanket`` isolates selective storage from other changes
by using the former whole-block checkpoint arrangement in this process only.
``cell-final`` retains each cell's final compiled block and checkpoints the
others, preserving the same compiled block boundaries in every condition.
"""

import argparse
import importlib.util
import json
import platform
import statistics
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import torch

from delta_feedback_experiment.model import DeltaModel, condition_config
from delta_feedback_experiment.optim import build_optimizers
from delta_feedback_experiment.train import (
    CudaEvalRunner,
    CudaGraphTrainer,
    GraphSpec,
    automatic_checkpoint,
    build_schedule,
    clip_gradients,
    model_fields,
    parse_run_args,
)


def package_revision(package):
    origin = importlib.util.find_spec(package).origin
    directory = Path(origin).resolve().parent
    result = {"source": str(directory)}
    for key, command in (
        ("commit", ["rev-parse", "HEAD"]),
        ("status", ["status", "--porcelain"]),
    ):
        process = subprocess.run(
            ["git", "-C", str(directory), *command],
            capture_output=True,
            text=True,
            check=False,
        )
        result[key] = process.stdout.strip() if process.returncode == 0 else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="1:1,1:4,2:4,3:8")
    parser.add_argument(
        "--batch-modes",
        help="optional comma-separated k:r subset for full batches and gradient saves; other selected modes receive raw replay timing only",
    )
    parser.add_argument("--full-family", action="store_true")
    parser.add_argument(
        "--checkpoint-policy",
        choices=("current", "blanket", "cell-final"),
        default="current",
        help="diagnostic storage control; blanket checkpoints each compiled block, cell-final retains the last block of each routing cell",
    )
    parser.add_argument("--updates", type=int, default=0)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--gradients", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    torch.manual_seed(1)
    torch.set_float32_matmul_precision("high")
    args = parse_run_args(["round8-qualification", "--condition", "arfl"])
    model = (
        DeltaModel(condition_config(args.condition, **model_fields(args)))
        .cuda()
        .train()
    )
    if options.checkpoint_policy != "current":
        from delta_feedback_experiment import model as model_module

        def run_with_storage_policy(compiled, *block_args):
            block = block_args[0]
            # Block owns its layer index, while the model owns cell geometry.
            # Selecting by position also bounds retained blocks without PKDA.
            cell_size = model.cfg.routing_block_size
            if (
                options.checkpoint_policy == "cell-final"
                and block.layer % cell_size == cell_size - 1
            ):
                return compiled(*block_args)
            return torch.utils.checkpoint.checkpoint(
                compiled,
                *block_args,
                use_reentrant=False,
                preserve_rng_state=False,
            )

        def attention_with_storage(*block_args):
            return run_with_storage_policy(model_module._compiled_block, *block_args)

        def pkda_with_storage(*block_args):
            return run_with_storage_policy(
                model_module._compiled_pkda_block, *block_args
            )

        model_module._compiled_selective_block = attention_with_storage
        model_module._compiled_selective_pkda_block = pkda_with_storage
    modes = [tuple(map(int, mode.split(":"))) for mode in options.modes.split(",")]
    batch_modes = (
        {tuple(map(int, mode.split(":"))) for mode in options.batch_modes.split(",")}
        if options.batch_modes
        else set(modes)
    )
    if not batch_modes.issubset(modes):
        parser.error("--batch-modes must be a subset of --modes")
    specs = [
        GraphSpec(k, automatic_checkpoint(model, k, r, args, torch.device("cuda")), r)
        for k, r in modes
    ]
    if not options.full_family:
        CudaGraphTrainer._reachable_specs = lambda self, schedule: specs
    else:
        original_capture = CudaGraphTrainer._capture

        def capture_with_progress(self, state, pool):
            print(
                json.dumps(
                    {
                        "event": "before_train_capture",
                        "k": state.spec.n_passes,
                        "r": state.spec.iterations,
                        "checkpoint": state.spec.checkpoint,
                        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
                        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
                    }
                ),
                flush=True,
            )
            return original_capture(self, state, pool)

        CudaGraphTrainer._capture = capture_with_progress
    optimizers = build_optimizers(
        model, lr_normuonh=args.lr_normuonh, lr_nadam=args.lr_nadam
    )
    started = time.perf_counter()
    runner = CudaGraphTrainer(model, optimizers, args, build_schedule(args))
    if options.full_family:
        print(
            json.dumps(
                {
                    "event": "before_eval_capture",
                    "allocated_gib": torch.cuda.memory_allocated() / 2**30,
                    "reserved_gib": torch.cuda.memory_reserved() / 2**30,
                }
            ),
            flush=True,
        )
    evaluation = (
        CudaEvalRunner(model, args, runner.pool) if options.full_family else None
    )
    result = {
        "capture_seconds": time.perf_counter() - started,
        "graph_count": len(runner.states),
        "evaluation_captured": evaluation is not None,
        "scope": "fresh initialization; fixed synthetic rows; runtime and numerical qualification",
        "modes": [],
    }
    result["provenance"] = {
        "revisions": {
            package: package_revision(package)
            for package in ("delta_feedback_experiment", "fla", "cut_cross_entropy")
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": version("triton"),
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
        },
        "geometry": model_fields(args),
        "condition": args.condition,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "torch_seed": 1,
        "synthetic_rows_seed": 31,
        "seq_len": args.seq_len,
        "batch_rows": args.batch_rows,
        "micro_rows": args.micro_rows,
        "head_flush_every": args.head_flush_every,
        "jitter": args.jitter,
        "z_coef": 0.0,
        "checkpoint_policy": options.checkpoint_policy,
        "batches_per_mode": options.batches,
        "updates_per_mode": options.updates,
        "batch_modes": sorted(batch_modes),
        "update_scope": "optimizer and model state carry across modes when updates are enabled",
    }
    print(
        json.dumps({key: value for key, value in result.items() if key != "modes"}),
        flush=True,
    )
    rows = torch.randint(
        args.vocab_size,
        (args.batch_rows, args.seq_len + 1),
        generator=torch.Generator().manual_seed(31),
    ).cuda()
    for spec in specs:
        state = runner.begin(spec)
        runner.replay(state, rows[:1], 1, 0)
        torch.cuda.synchronize()
        samples = []
        for index in range(10):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            state.graph.replay()
            end.record()
            end.synchronize()
            if index >= 2:
                samples.append(start.elapsed_time(end))
        record = {
            "k": spec.n_passes,
            "r": spec.iterations,
            "checkpoint": spec.checkpoint,
            "raw_median_ms": statistics.median(samples),
            "raw_samples_ms": samples,
            "batches": [],
        }
        batch_count = (
            options.batches + options.updates
            if (spec.n_passes, spec.iterations) in batch_modes
            else 0
        )
        for batch in range(batch_count):
            runner.zero_grad()
            runner.begin(spec)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for micro in range(args.batch_rows):
                runner.replay(state, rows[micro : micro + 1], batch + 1, micro)
            runner.prepare_optimizer(state)
            torch.cuda.synchronize()
            replay_seconds = time.perf_counter() - started
            gradient_serialization_seconds = 0.0
            if options.gradients is not None and batch == 0:
                serialization_started = time.perf_counter()
                options.gradients.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        name: p.grad.cpu()
                        for name, p in model.named_parameters()
                        if p.grad is not None
                    },
                    options.gradients / f"k{spec.n_passes}-r{spec.iterations}.pt",
                )
                gradient_serialization_seconds = (
                    time.perf_counter() - serialization_started
                )
            finish_started = time.perf_counter()
            norm = clip_gradients(model.parameters())
            if batch >= options.batches:
                for optimizer in optimizers:
                    optimizer.step()
                model.refresh_shadows()
            torch.cuda.synchronize()
            record["batches"].append(
                {
                    "replay_seconds": replay_seconds,
                    "total_seconds": replay_seconds
                    + time.perf_counter()
                    - finish_started,
                    "gradient_serialization_seconds": gradient_serialization_seconds,
                    "loss": state.loss_sum.item(),
                    "grad_norm": float(norm),
                }
            )
        result["modes"].append(record)
        result.update(
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

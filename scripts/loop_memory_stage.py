"""Staged memory and time measurement of the loop's training modes on Jobe.

    python scripts/loop_memory_stage.py --out data/summary/loop-stage-DATE.json \
        --condition arfl [train flags, e.g. --three-pass 1 --feedback-start 0]

Stage 1 runs eager one-, two-, and three-pass microbatches at r = 1, forward
and backward, under the trainer's activation policy; stage 2 repeats them at
the iteration cap; stage 3 captures the trainer's whole (pass count, r) graph
family plus the evaluation graphs, then replays every graph and projects the
schedule's replay time from its realized draws. Allocated and reserved peaks
are reported separately, and an out-of-memory mode is recorded rather than
fatal. Nothing here trains or writes a snapshot.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from delta_feedback_experiment.model import (
    DeltaModel,
    condition_config,
    multipass,
    multipass_loss,
)
from delta_feedback_experiment.optim import build_optimizers
from delta_feedback_experiment.train import (
    EXACT_FIELDS,
    CudaEvalRunner,
    CudaGraphTrainer,
    automatic_checkpoint,
    build_parser,
    build_schedule,
    draw_iterations,
    draw_passes,
    micro_draws,
)


def gib(value: int) -> float:
    return round(value / 2**30, 3)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stages", default="1,2,3")
    parser.add_argument("--replays", type=int, default=5)
    own, rest = parser.parse_known_args()
    if "--condition" not in rest:
        rest = ["--condition", "arfl", *rest]
    args = build_parser().parse_args(["loop-stage", *rest])
    stages = set(own.stages.split(","))
    if not torch.cuda.is_available():
        raise SystemExit("this measurement needs the CUDA surface")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    schedule = build_schedule(args)

    def fresh() -> DeltaModel:
        torch.manual_seed(args.seed)
        model = (
            DeltaModel(
                condition_config(
                    args.condition,
                    vocab_size=args.vocab_size,
                    dim=args.dim,
                    layers=args.layers,
                    heads=args.heads,
                    kv_heads=args.kv_heads,
                    head_dim=args.head_dim,
                    intermediate=args.intermediate,
                    pkda_heads=args.pkda_heads,
                    pkda_head_dim=args.pkda_head_dim,
                    pkda_conv_size=args.pkda_conv_size,
                    max_seq_len=args.seq_len + 1,
                    loop_iterations=args.loop_iterations,
                    loop_max_iterations=args.loop_max_iterations,
                )
            )
            .cuda()
            .train()
        )
        model.refresh_shadows()
        return model

    rows = torch.randint(
        0,
        args.vocab_size,
        (args.micro_rows, args.seq_len + 1),
        generator=torch.Generator().manual_seed(31),
    ).cuda()
    record: dict = {
        "args": {field: getattr(args, field) for field in EXACT_FIELDS},
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "stages": [],
    }

    def save() -> None:
        Path(own.out).parent.mkdir(parents=True, exist_ok=True)
        Path(own.out).write_text(json.dumps(record, indent=2))

    def eager(k: int, r: int) -> None:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = fresh()
        model.grad_checkpoint = automatic_checkpoint(model, k, r, args, device)
        entry = {"stage": "eager", "k": k, "r": r, "checkpoint": model.grad_checkpoint}
        prefix = jitter = None
        if k > 1:
            prefix, jitter = micro_draws(
                args, 1, 0, k, args.micro_rows, args.dim, device
            )
        try:
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs = multipass(
                    model, rows, k, prefix_lens=prefix, jitter=jitter, iterations=r
                )
                loss, _ = multipass_loss(model, rows, outs)
            loss.backward()
            torch.cuda.synchronize()
            entry |= {
                "loss": loss.item(),
                "seconds": round(time.monotonic() - started, 3),
                "allocated_gib": gib(torch.cuda.max_memory_allocated()),
                "reserved_gib": gib(torch.cuda.max_memory_reserved()),
            }
            del outs, loss
        except torch.OutOfMemoryError as exc:
            entry["oom"] = str(exc).splitlines()[0][:200]
        finally:
            del model, prefix, jitter
            gc.collect()
            torch.cuda.empty_cache()
        record["stages"].append(entry)
        print(entry, flush=True)
        save()

    if "1" in stages:
        for k in (1, 2, 3):
            eager(k, 1)
    if "2" in stages:
        for k in (1, 2, 3):
            eager(k, args.loop_max_iterations)
    if "3" not in stages:
        save()
        return

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = fresh()
    optimizers = build_optimizers(
        model, lr_normuonh=args.lr_normuonh, lr_nadam=args.lr_nadam
    )
    started = time.monotonic()
    try:
        runner = CudaGraphTrainer(model, optimizers, args, schedule)
        eval_runner = CudaEvalRunner(model, args, runner.pool)
    except torch.OutOfMemoryError as exc:
        record["stages"].append(
            {
                "stage": "capture",
                "oom": str(exc).splitlines()[0][:200],
                "allocated_gib": gib(torch.cuda.max_memory_allocated()),
                "reserved_gib": gib(torch.cuda.max_memory_reserved()),
            }
        )
        save()
        print(record["stages"][-1], flush=True)
        return
    torch.cuda.synchronize()
    entry = {
        "stage": "capture",
        "seconds": round(time.monotonic() - started, 1),
        "train_graphs": len(runner.states),
        "eval_graphs": len(eval_runner.states),
        "allocated_gib": gib(torch.cuda.max_memory_allocated()),
        "reserved_gib": gib(torch.cuda.max_memory_reserved()),
        "replays": [],
    }
    print({key: value for key, value in entry.items() if key != "replays"}, flush=True)
    by_spec: dict[tuple[int, int], float] = {}
    for index, (spec, state) in enumerate(runner.states.items()):
        samples = []
        for replay in range(own.replays):
            runner.zero_grad()
            runner.begin(spec)
            torch.cuda.synchronize()
            tick = time.monotonic()
            runner.replay(state, rows, index + 1, replay * args.micro_rows)
            torch.cuda.synchronize()
            samples.append(time.monotonic() - tick)
        median = statistics.median(samples[1:]) if len(samples) > 1 else samples[0]
        by_spec[spec.n_passes, spec.iterations] = median
        replay_record = {
            "k": spec.n_passes,
            "r": spec.iterations,
            "checkpoint": spec.checkpoint,
            "ms": round(median * 1000, 1),
            "loss": state.loss_sum.item(),
        }
        entry["replays"].append(replay_record)
        print(replay_record, flush=True)
    runner.zero_grad()
    micros = args.batch_rows // args.micro_rows
    projected = 0.0
    for step in range(1, schedule.total + 1):
        k = draw_passes(args, step, schedule.total) if model.cfg.feedback_active else 1
        r = draw_iterations(args, step, model.cfg.loop)
        projected += by_spec[k, r] * micros
    entry["projected_replay_hours"] = round(projected / 3600, 2)
    entry["note"] = (
        "replay only: the optimizer, clipping, shadow refresh, evaluation, and "
        "snapshots add to every step"
    )
    record["stages"].append(entry)
    save()
    print(f"projected replay time for the schedule: {entry['projected_replay_hours']} h")
    print(f"wrote {own.out}")


if __name__ == "__main__":
    main()

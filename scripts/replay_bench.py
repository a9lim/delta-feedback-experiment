"""Time every captured training graph's replay, warm, and trace one replay
by kernel class: the baseline the Hopper levers are scored against.

    python scripts/replay_bench.py --data-root /data/delta [--tag gh200-base]
        [--condition fv] [--scale screen] [--specs 1:1,2:2,3:2] [--trace]
        [--replay-rows 2] [--attention-backend cudnn|flash]
        [--cce-config bf16-base] [--expert-tiles default|ada]
        [--train-steps 4]  # optional optimizer check; requires one captured spec
        [--init runs/TAG.pt.STEP]  # trained weights and selection biases
        [--warm 3] [--repeat 10] [-- --precision bf16 --micro-rows 2 ...]

Builds the trainer exactly as ``delta train`` does (same planner, same
widths, same compiled graphs, recurrence roll from step 0 so every graph
is reachable), then for each captured graph replays one production-width
micro-batch ``--warm`` times untimed and ``--repeat`` times timed. Reports
milliseconds per replay, per row, and per row-column, the replays a
128-row step needs, and the step time each graph implies; weights the
graphs by how often the schedule rolls them for one mean step figure.
The loss sum of one replay is a numerics fingerprint for comparisons at
the same width and weights (different widths use different corpus rows).
``--init`` loads a snapshot's model state (weights and expert-selection
biases, not optimizer state) before the trainer is built: an untrained
head filters nothing in the CCE backward and untrained routers spread
tokens evenly, so kernel timings need trained weights. Any condition with
the snapshot's parameter set loads (``v`` weights run ``fv``). ``--trace`` profiles one replay per graph
and sums kernel time by class (GEMM, FLA recurrence, CCE head, experts,
attention, pointwise, ...) with the top kernels by time. Writes
``logs/replay-bench/<tag>.json``. Run from the experiment directory.
CCE/expert overrides affect only this benchmark process; omitted controls
use the production configuration; --cce-config applies where the head runs
CCE (off Hopper).
``--train-steps N`` then performs N full-batch optimizer updates from the
same initialization, using production schedules and keyed row draws with
one fixed captured graph. This paired check does not exercise the full
recurrence schedule, evaluation, or checkpoint lifecycle.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = [
    ("cce_head", r"cce|linear_cross_entropy|_lse|_indexed"),
    ("experts", r"_grouped_mm|_grouped_dw|_swiglu|_combine_dx|dact_swiglu|moe_"),
    ("attention", r"flash|fmha|sdpa|FlashAttn|cudnn|fused_attn"),
    ("fla_recurrence", r"chunk_|fwd_|bwd_|wy_|atk|kda|precond|solve_tril|intra|inter|dhu|prepare|cumsum_kernel|gate_bwd|gate_chunk"),
    ("conv_norm", r"causal_conv1d|conv|layer_norm_gated|rms_norm"),
    ("routers", r"_route_|_pack_control"),
    ("gemm", r"cutlass|gemm|Cijk|xmma|sm89|sm90|ampere|hopper|nvjet|cublas|gemv|splitK|scaled_mm|e4m3"),
    ("inductor_template", r"triton_tem_|triton_mm|triton_bmm"),
    ("inductor_pointwise", r"triton_poi_|triton_red_|triton_per_|triton_spl_"),
    ("memset_fill", r"Memset|fill_|Fill|vectorized_fill|zero"),
    ("copy_cast", r"copy|Copy|direct_copy|CatArrayBatched|cat"),
    ("sort_topk_scan", r"sort|topk|Topk|radix|scan|cumsum|bitonic|segmented"),
    ("aten_elementwise", r"elementwise_kernel|vectorized_elementwise|unrolled_elementwise|index_|gather|scatter|reduce_kernel|softmax"),
]


def categorize(name: str) -> str:
    for category, pattern in CATEGORIES:
        if re.search(pattern, name):
            return category
    return "other"


def git_rev(path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


def main() -> None:
    from cce_bench import CANDIDATES as CCE_CANDIDATES
    from cce_bench import select as select_cce

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", default="replay-bench")
    parser.add_argument("--condition", default="fv")
    parser.add_argument("--scale", default="screen")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source", default="dclm-100b")
    parser.add_argument("--specs", default=None, help="k:r list; default every graph the schedule reaches")
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=0,
                        help="full-batch optimizer steps after timing; requires exactly one captured graph (default 0)")
    parser.add_argument("--trace", action="store_true", help="profile one replay per graph by kernel class")
    parser.add_argument("--replay-rows", type=int, help="force this replay width; reject graphs that cannot fit it")
    parser.add_argument("--attention-backend", choices=("default", "cudnn", "flash"), default="default",
                        help="force a single attention backend for an A/B comparison")
    parser.add_argument("--cce-config", choices=tuple(CCE_CANDIDATES), default=None,
                        help="use this fixed CCE benchmark configuration; omitted uses production")
    parser.add_argument("--expert-tiles", choices=("default", "ada"), default="default",
                        help="use production expert tiles or restore Ada dW tiles for a paired comparison")
    parser.add_argument("--init", default=None,
                        help="load this snapshot's model state (weights and selection biases) first")
    parser.add_argument("--out", default=None, help="default logs/replay-bench/<tag>.json")
    parser.add_argument("extra", nargs="*", help="further delta train flags after --")
    opts = parser.parse_args()
    if opts.warm < 0 or opts.repeat < 1:
        parser.error("--warm must be nonnegative and --repeat must be positive")
    if opts.train_steps < 0:
        parser.error("--train-steps must be nonnegative")

    import torch
    from torch.nn.attention import SDPBackend
    from torch.profiler import ProfilerActivity, profile

    from delta_feedback_experiment import attention, distributed, moe_kernels

    if opts.attention_backend != "default":
        attention.FUSED_BACKENDS = [{"cudnn": SDPBackend.CUDNN_ATTENTION,
                                    "flash": SDPBackend.FLASH_ATTENTION}[opts.attention_backend]]
    if opts.cce_config is not None:
        select_cce(CCE_CANDIDATES[opts.cce_config])
    if opts.expert_tiles == "ada":
        moe_kernels.TILE_DW_GATE_HOPPER = moe_kernels.TILE_DW_GATE
        moe_kernels.TILE_DW_DOWN_HOPPER = moe_kernels.TILE_DW_DOWN
    from delta_feedback_experiment.data import TokenData
    from delta_feedback_experiment.model import DeltaModel, condition_config
    from delta_feedback_experiment.optim import apply_schedule, build_optimizers
    from delta_feedback_experiment.sites import ParameterSites
    from delta_feedback_experiment.train import (
        CudaGraphTrainer,
        GraphSpec,
        build_schedule,
        model_fields,
        parse_run_args,
        pick_device,
        plan_replay,
        read_checkpoint,
        step_shape,
    )

    argv = [
        opts.tag, "--condition", opts.condition, "--scale", opts.scale,
        "--data-root", opts.data_root, "--source", opts.source,
        "--recurrence-start", "0", *opts.extra,
    ]
    args = parse_run_args(argv)
    topology = distributed.Topology.from_environment()
    if opts.replay_rows is not None and (
        opts.replay_rows < args.micro_rows
        or opts.replay_rows % args.micro_rows
        or (args.batch_rows // topology.world) % opts.replay_rows
    ):
        parser.error("--replay-rows must divide rank rows and be a positive multiple of --micro-rows")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = pick_device(args.device, topology)
    distributed.initialize(topology, device)
    data = TokenData.load(Path(args.data_root) / args.source, "train", args.seq_len)
    model = DeltaModel(condition_config(args.condition, **model_fields(args)))
    if opts.init is not None:
        payload = read_checkpoint(opts.init)
        # A geometry that differs fails here on the first mismatched shape.
        model.load_state_dict(payload["state"])
        del payload
    model = model.to(device)
    sites = ParameterSites(model, topology, fp8=args.precision == "fp8")
    optimizers = build_optimizers(
        model, lr_normuonh=args.lr_normuonh, lr_nadam=args.lr_nadam, owned=sites.owned
    )
    schedule = build_schedule(args)
    if opts.train_steps > schedule.total:
        parser.error("--train-steps cannot exceed the training schedule length")
    if opts.train_steps * args.batch_rows > data.rows:
        parser.error("--train-steps needs more rows than the training stream contains")
    model.train()

    wanted = None
    if opts.specs:
        wanted = [GraphSpec(int(k), int(r)) for k, r in (p.split(":") for p in opts.specs.split(","))]

    class BenchTrainer(CudaGraphTrainer):
        def _reachable_specs(self, schedule):
            specs = super()._reachable_specs(schedule)
            if wanted and any(spec not in specs for spec in wanted):
                raise ValueError(f"requested graphs {wanted} are not all reachable: {specs}")
            specs = [s for s in specs if s in wanted] if wanted else specs
            if opts.train_steps and len(specs) != 1:
                raise ValueError("--train-steps requires exactly one captured graph; select it with --specs k:r")
            return specs

        def _plan(self, spec, calibration, budget_bytes):
            if opts.replay_rows is None:
                return super()._plan(spec, calibration, budget_bytes)
            plan = plan_replay(self.model.cfg, self.args, spec, calibration,
                               budget_bytes, opts.replay_rows)
            if plan.rows_per_replay != opts.replay_rows:
                raise ValueError(f"{spec}: requested {opts.replay_rows} rows do not fit; planner chose {plan.rows_per_replay}")
            return plan

    trainer = BenchTrainer(model, optimizers, sites, args, schedule, topology)
    rolled = Counter(
        GraphSpec(*step_shape(args, step, schedule.total, model.cfg))
        for step in range(1, schedule.total + 1)
    )
    rows = trainer.batch_stager.stage(data, 0)
    torch.cuda.synchronize()

    result = {
        "tag": opts.tag,
        "condition": args.condition,
        "scale": args.scale,
        "geometry": model_fields(args),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "precision": args.precision,
        "cce_config": opts.cce_config,
        "expert_tiles": opts.expert_tiles,
        "attention_backend": opts.attention_backend,
        "requested_replay_rows": opts.replay_rows,
        "init": opts.init,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "revisions": {
            "experiment": git_rev("."),
            "root": git_rev(".."),
            "flash-linear-attention": git_rev("../vendor/flash-linear-attention"),
            "cut-cross-entropy": git_rev("../vendor/ml-cross-entropy"),
        },
        "rank_rows": trainer.rank_rows,
        "seq_len": args.seq_len,
        "static_gib": round(trainer.static_bytes / 2**30, 2),
        "activation_budget_gib": round(trainer.budget_bytes / 2**30, 2),
        "graphs": {},
    }
    weighted_ms = 0.0
    weighted_steps = 0
    for spec, state in trainer.states.items():
        key = f"k={spec.n_passes} r={spec.iterations}"
        width = state.plan.rows_per_replay
        columns = spec.n_passes * spec.iterations
        trainer.zero_grad()
        trainer.begin(spec, 0.0)
        trainer.replay(state, rows[:width], 1, 0)
        torch.cuda.synchronize()
        fingerprint = float(state.loss_sum.item())
        for _ in range(opts.warm):
            trainer.replay(state, rows[:width], 1, 0)
        torch.cuda.synchronize()
        samples = []
        for _ in range(opts.repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            trainer.replay(state, rows[:width], 1, 0)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        median = statistics.median(samples)
        replays = trainer.rank_rows // width
        entry = {
            "rows_per_replay": width,
            "columns": columns,
            "saved": "lean" if state.plan.lean else "full",
            "checkpoint_blocks": state.plan.checkpoint_blocks,
            "eligible_blocks": state.plan.eligible_blocks,
            "estimated_gib": round(state.plan.estimated_gib, 2),
            "ms_per_replay": {"median": round(median, 2), "min": round(min(samples), 2), "mean": round(statistics.mean(samples), 2), "n": len(samples)},
            "ms_per_row": round(median / width, 2),
            "ms_per_row_column": round(median / width / columns, 2),
            "replays_per_step": replays,
            "step_ms": round(median * replays, 1),
            "tokens_per_s": round(width * args.seq_len / median * 1000),
            "schedule_share": round(rolled.get(spec, 0) / schedule.total, 4),
            "loss_fingerprint": fingerprint,
            "normalized_loss_fingerprint": fingerprint * replays,
        }
        if opts.trace:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                state.graph.replay()
                torch.cuda.synchronize()
            events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            by_class = defaultdict(lambda: [0, 0.0])
            by_name = defaultdict(lambda: [0, 0.0])
            total = 0.0
            for e in events:
                us = float(getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0.0)))
                total += us
                by_class[categorize(e.name)][0] += 1
                by_class[categorize(e.name)][1] += us
                by_name[e.name][0] += 1
                by_name[e.name][1] += us
            entry["trace"] = {
                "kernel_ms": round(total / 1000, 2),
                "kernels": len(events),
                "by_class": {
                    c: {"n": v[0], "ms": round(v[1] / 1000, 2), "share": round(v[1] / total, 3) if total else 0}
                    for c, v in sorted(by_class.items(), key=lambda kv: -kv[1][1])
                },
                "top_kernels": [
                    {"name": n[:120], "n": v[0], "ms": round(v[1] / 1000, 3)}
                    for n, v in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:40]
                ],
            }
        result["graphs"][key] = entry
        if spec in rolled:
            weighted_ms += entry["step_ms"] * rolled[spec]
            weighted_steps += rolled[spec]
        print(
            f"{key}: {width} rows x {columns} col, {entry['saved']}, recompute "
            f"{entry['checkpoint_blocks']}/{entry['eligible_blocks']}, "
            f"{median:.1f} ms/replay ({entry['ms_per_row_column']:.2f} ms/row-col), "
            f"{replays} replays -> {entry['step_ms']:.0f} ms/step, share {entry['schedule_share']:.3f}, "
            f"loss {fingerprint:.6g}",
            flush=True,
        )
        if opts.trace:
            top = list(entry["trace"]["by_class"].items())[:6]
            print("  " + ", ".join(f"{c} {v['ms']} ms ({v['share']:.0%})" for c, v in top), flush=True)
    if weighted_steps:
        result["mean_step_ms"] = round(weighted_ms / weighted_steps, 1)
        result["mean_step_share_covered"] = round(weighted_steps / schedule.total, 4)
        print(f"schedule-weighted mean step {result['mean_step_ms']:.0f} ms over {result['mean_step_share_covered']:.1%} of the schedule", flush=True)
    out = Path(opts.out) if opts.out else Path("logs/replay-bench") / f"{opts.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if opts.train_steps:
        spec = next(iter(trainer.states))
        result["training"] = {
            "scope": "fixed-graph paired optimizer check",
            "fixed_spec": {"n_passes": spec.n_passes, "iterations": spec.iterations},
            "requested_steps": opts.train_steps,
            "batch_rows": args.batch_rows,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "steps": [],
        }
        out.write_text(json.dumps(result, indent=2) + "\n")
        # Warmup, timing, and profiling accumulate into the same FP32 sinks.
        trainer.zero_grad()
        torch.cuda.synchronize()
        for step in range(1, opts.train_steps + 1):
            started = time.perf_counter()
            learning_rates = apply_schedule(optimizers, schedule, step)
            phase = schedule.phase(step)[0]
            z_coef = args.zloss if phase == "cooldown" else 0.0
            state = trainer.begin(spec, z_coef)
            trainer.replay_batch(state, data, step, (step - 1) * args.batch_rows)
            sites.reduce_gradients()
            floats = torch.stack([
                state.loss_sum, state.pass1_sum, state.ntp_sum,
                state.mtp_sum, state.expert_balance_sum,
            ])
            expert_counts = state.expert_counts.clone()
            distributed.all_reduce_(floats)
            distributed.all_reduce_(expert_counts)
            loss, pass1, ntp, mtp, expert_balance = floats.tolist()
            trainer.prepare_optimizer(state)
            grad_norm = sites.gradient_norm()
            if not all(math.isfinite(value) for value in (
                loss, pass1, ntp, mtp, expert_balance, grad_norm, *learning_rates.values()
            )):
                raise RuntimeError(f"optimizer check step {step} has non-finite metrics or learning rates")
            with trainer.pool_scope():
                for optimizer in optimizers:
                    optimizer.step()
                model.update_expert_bias(expert_counts)
            sites.gather_weights()
            model.refresh_shadows()
            sites.requantize()
            trainer.zero_grad()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            result["training"]["steps"].append({
                "step": step, "phase": phase, "loss": loss, "pass1": pass1,
                "ntp": ntp, "mtp": mtp, "expert_balance": expert_balance,
                "grad_norm": grad_norm, "lr": learning_rates, "seconds": seconds,
            })
            out.write_text(json.dumps(result, indent=2) + "\n")
            print(
                f"optimizer check {step}/{opts.train_steps} k={spec.n_passes} r={spec.iterations}: "
                f"loss {loss:.6f}, ntp {ntp:.6f}, mtp {mtp:.6f}, "
                f"grad_norm {grad_norm:.6g}, lr {learning_rates}, {seconds:.2f}s",
                flush=True,
            )
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

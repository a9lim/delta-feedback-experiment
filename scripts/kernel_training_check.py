"""Short paired stability check for kernel changes, without modifying any run."""

from __future__ import annotations

import argparse
import contextlib
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
    parser.add_argument("--label", default="current")
    parser.add_argument("--gemm-backends")
    parser.add_argument("--attention", choices=("plain", "hints", "prescale"))
    parser.add_argument("--input-mode", choices=("micro", "staged"), default="micro")
    parser.add_argument("--profile-optimizer", type=Path)
    options = parser.parse_args()
    if options.gemm_backends:
        torch._inductor.config.max_autotune_gemm_backends = options.gemm_backends
    if options.attention:
        import delta_feedback_experiment.attention as attention
        from torch.nn.attention.flex_attention import flex_attention

        kernel_options = {"BACKEND": "TRITON"}
        if options.attention != "plain":
            kernel_options.update(ROWS_GUARANTEED_SAFE=True, BLOCKS_ARE_CONTIGUOUS=True)
        if options.attention == "prescale":
            kernel_options["PRESCALE_QK"] = True

        def causal(query, key, value, block_mask):
            return flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                enable_gqa=True,
                kernel_options=kernel_options,
            )

        attention._compiled_causal_attention = torch.compile(
            causal,
            fullgraph=True,
            dynamic=False,
            mode=delta_feedback_experiment.INDUCTOR_MODE,
        )
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
                "label": options.label,
                "variants": {
                    "attention": options.attention,
                    "input_mode": options.input_mode,
                    "gemm_backends": torch._inductor.config.max_autotune_gemm_backends,
                },
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
        events = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
        events[0].record()
        if options.input_mode == "staged":
            runner.replay_batch(
                state, train, 9000 + index, 100000 + index * args.batch_rows
            )
        else:
            for micro in range(args.batch_rows // args.micro_rows):
                first = 100000 + index * args.batch_rows + micro * args.micro_rows
                runner.replay(
                    state, train.batch(first, args.micro_rows), 9000 + index, first
                )
        events[1].record()
        runner.prepare_optimizer(state)
        profiling = (
            options.profile_optimizer is not None and index == options.updates - 1
        )
        profile = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True,
            )
            if profiling
            else contextlib.nullcontext()
        )
        with profile:
            with torch.profiler.record_function("clip"):
                norm = clip_gradients(model.parameters())
            events[2].record()
            for slot, optimizer in enumerate(optimizers):
                with torch.profiler.record_function(type(optimizer).__name__):
                    optimizer.step()
                events[3 + slot].record()
            with torch.profiler.record_function("refresh_shadows"):
                model.refresh_shadows()
            events[5].record()
            with torch.profiler.record_function("zero_grad"):
                runner.zero_grad()
            events[6].record()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if profiling:
            profile.export_chrome_trace(str(options.profile_optimizer))
            print(
                profile.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=30
                ),
                flush=True,
            )
        print(
            json.dumps(
                {
                    "update": index + 1,
                    "passes": passes,
                    "loss": state.loss_sum.item(),
                    "pass1_loss": state.pass1_sum.item(),
                    "gradient_norm": norm,
                    "seconds": elapsed,
                    "profiled": profiling,
                    "stages_ms": dict(
                        zip(
                            (
                                "forward_backward",
                                "clip",
                                "normuonh",
                                "nadam",
                                "shadows",
                                "zero_grad",
                            ),
                            (
                                left.elapsed_time(right)
                                for left, right in zip(events, events[1:])
                            ),
                        )
                    ),
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
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

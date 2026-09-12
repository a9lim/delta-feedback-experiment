"""Small coordinate diagnostic for the expert learning-rate scaling rule.

Compare production optimizer groups with an otherwise identical control whose
expert input/output groups use the base NorMuonH rate. One learned MoE bank is
trained for at most four steps against the fixed teacher ``sin(x)``; the same
normalized batch is reused on every step and in both arms. Preset residual and
expert widths are divided by eight, while routed/selected counts and the real
muP reference width stay unchanged. This is a coordinate-dynamics diagnostic,
not evidence of language-model quality or hyperparameter transfer.

CPU uses portable FP32 operations. CUDA uses the compiled production MoE,
BF16 operands, FP32 masters and gradient sinks, and the compiled production
NorMuonH update. It does not capture the full trainer's CUDA graphs.

Usage:
    python scripts/expert_scaling_check.py
    python scripts/expert_scaling_check.py --device cuda --output /tmp/expert-check.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from delta_feedback_experiment import INDUCTOR_MODE
from delta_feedback_experiment.model import (
    EXPERT_BALANCE_COEF,
    MUP_BASE_ACTIVE_EXPERTS,
    MUP_BASE_DIM,
    DeltaModel,
    ModelConfig,
)
from delta_feedback_experiment.moe import EXPERT_BIAS_RATE, MixtureOfExperts
from delta_feedback_experiment.optim import (
    DEFAULT_NADAM_LR,
    DEFAULT_NORMUONH_LR,
    build_optimizers,
)
from delta_feedback_experiment.train import GRAD_CLIP_NORM, SCALES, clip_gradients

WIDTH_DIVISOR = 8
DATA_SEED = 20260912
BATCH_ROWS = 2
SEQUENCE_LENGTH = 16


class SingleBank(nn.Module):
    """Expose the production parameter names without constructing a trunk."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.block = nn.Module()
        self.block.mlp = MixtureOfExperts(
            cfg.dim,
            cfg.expert_intermediate,
            cfg.num_routed_experts,
            cfg.experts_per_token,
        )
        # These authoritative initializers use only cfg and named modules /
        # parameters. The .block.mlp prefix gives every matrix the same owner
        # as in DeltaModel, including the NAdam-owned learned router.
        self.apply(lambda module: DeltaModel._init_weights(self, module))
        DeltaModel._scale_initialization(self)

    def forward(self, x: Tensor):
        return self.block.mlp(x, want_weights=True)


def rms(tensor: Tensor) -> float:
    return tensor.detach().float().square().mean().sqrt().item()


@torch.no_grad()
def metrics(
    model: SingleBank, x: Tensor, result, target: Tensor, previous: Tensor | None
) -> dict[str, float]:
    output, auxiliary, weights, _counts = result
    logits = torch.nn.functional.linear(x.float(), model.block.mlp.router.weight)
    affinities = logits.sigmoid()
    probabilities = affinities / affinities.sum(dim=-1, keepdim=True)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)
    selected_entropy = -(weights * weights.clamp_min(1e-30).log()).sum(dim=-1)
    return {
        "input_rms": rms(x),
        "bank_output_rms": rms(output),
        "bank_output_change_rms": 0.0
        if previous is None
        else rms(output.float() - previous.float()),
        "teacher_rms": rms(target),
        "mse": torch.nn.functional.mse_loss(output.float(), target).item(),
        "sequence_balance_auxiliary": auxiliary.item(),
        "router_logit_rms": rms(logits),
        "router_probability_entropy_nats": entropy.mean().item(),
        "router_probability_entropy_fraction": entropy.mean().item()
        / math.log(model.cfg.num_routed_experts),
        "selected_probability_entropy_nats": selected_entropy.mean().item(),
        "expert_bias_abs_max": model.block.mlp.expert_bias.abs().max().item(),
    }


def run_arm(
    cfg: ModelConfig,
    seed: int,
    arm: str,
    batch: Tensor,
    device: torch.device,
    steps: int,
) -> dict:
    # Constructor and authoritative initializers both consume CPU RNG. Reset
    # them together so the two rate arms start with identical parameters.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SingleBank(cfg).to(device).train()
    optimizers = build_optimizers(model)
    if arm == "base_expert_rates":
        for group in optimizers[0].param_groups:
            if group["rate_name"].startswith("normuonh_expert_"):
                group["lr"] = group["stable_lr"] = DEFAULT_NORMUONH_LR
    rates = {
        group["rate_name"]: group["lr"]
        for optimizer in optimizers
        for group in optimizer.param_groups
        if group["params"]
    }
    parameters = tuple(model.parameters())
    refresh = []
    sinks = {}
    if device.type == "cuda":
        sinks = {parameter: torch.zeros_like(parameter) for parameter in parameters}
        _bound, refresh = model.block.mlp.bind_gradient_sinks(sinks)
        for parameter, sink in sinks.items():
            parameter.grad = sink
        forward = torch.compile(
            model, fullgraph=True, dynamic=False, mode=INDUCTOR_MODE
        )
    else:
        forward = model
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    x = batch.to(device=device, dtype=dtype).requires_grad_()
    target = x.detach().float().sin()

    def evaluate():
        precision = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with precision:
            return forward(x)

    with torch.no_grad():
        result = evaluate()
        history = [{"step": 0, **metrics(model, x, result, target, None)}]
        previous = result[0].detach().clone()
    for step in range(1, steps + 1):
        if sinks:
            for sink in sinks.values():
                sink.zero_()
        else:
            model.zero_grad(set_to_none=True)
        x.grad = None
        output, auxiliary, _weights, counts = evaluate()
        loss = (
            torch.nn.functional.mse_loss(output.float(), target)
            + EXPERT_BALANCE_COEF * auxiliary
        )
        loss.backward()
        gradient_norm = clip_gradients(parameters)
        for optimizer in optimizers:
            optimizer.step()
        model.block.mlp.update_bias(counts.detach())
        with torch.no_grad():
            for shadow, parameter in refresh:
                shadow.copy_(parameter)
            parameter_norms = torch.stack(
                [parameter.norm() for parameter in parameters]
            )
            if not torch.isfinite(parameter_norms).all().item():
                raise FloatingPointError(
                    f"nonfinite parameters: {cfg.dim=} {seed=} {arm=}"
                )
            result = evaluate()
            history.append(
                {
                    "step": step,
                    "training_loss_before_update": loss.item(),
                    "gradient_norm_before_clip": gradient_norm,
                    **metrics(model, x, result, target, previous),
                }
            )
            previous = result[0].detach().clone()
    # Reject nonfinite measurements, without imposing any preferred scale trend.
    json.dumps(history, allow_nan=False)
    return {"seed": seed, "arm": arm, "learning_rates": rates, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--scales", nargs="+", choices=tuple(SCALES), default=list(SCALES)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--steps", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if len(args.seeds) > 2:
        parser.error("use at most two seeds for this bounded diagnostic")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requires an available CUDA device")
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(DATA_SEED)
    raw = torch.randn(
        BATCH_ROWS,
        SEQUENCE_LENGTH,
        max(preset["dim"] for preset in SCALES.values()) // WIDTH_DIVISOR,
        generator=generator,
    )
    report = {
        "purpose": "single-bank coordinate dynamics under paired expert learning rates",
        "limitations": [
            "Reduced geometry and a fixed synthetic regression batch; no full model or language data.",
            "A finite trace is not a hyperparameter-transfer or training-quality result.",
            "Real production muP reference retained; reduced widths are geometry overrides.",
            "Output changes include learned routing and post-step bias-controller changes.",
            "CUDA compiles the production bank and optimizer but does not capture trainer graphs.",
        ],
        "settings": {
            "device": str(device),
            "torch_version": torch.__version__,
            "width_divisor": WIDTH_DIVISOR,
            "mup_base_dim": MUP_BASE_DIM,
            "mup_base_active_experts": MUP_BASE_ACTIVE_EXPERTS,
            "data_seed": DATA_SEED,
            "batch_rows": BATCH_ROWS,
            "sequence_length": SEQUENCE_LENGTH,
            "teacher": "sin(x), coordinatewise on the fixed execution-dtype input",
            "input": "common Gaussian prefixes, normalized to unit RMS per token",
            "steps": args.steps,
            "schedule": "constant stable rates; no warmup or decay",
            "seeds": args.seeds,
            "cpu_threads": args.threads,
            "master_dtype": "float32",
            "activation_dtype": "bfloat16" if device.type == "cuda" else "float32",
            "compiled": device.type == "cuda",
            "base_normuonh_lr": DEFAULT_NORMUONH_LR,
            "base_nadam_lr": DEFAULT_NADAM_LR,
            "gradient_clip_norm": GRAD_CLIP_NORM,
            "expert_balance_coefficient": EXPERT_BALANCE_COEF,
            "expert_bias_rate": EXPERT_BIAS_RATE,
        },
        "scales": [],
    }
    for name in args.scales:
        preset = SCALES[name]
        cfg = ModelConfig(
            dim=preset["dim"] // WIDTH_DIVISOR,
            expert_intermediate=preset["expert_intermediate"] // WIDTH_DIVISOR,
            num_routed_experts=preset["num_routed_experts"],
            experts_per_token=preset["experts_per_token"],
        )
        batch = raw[..., : cfg.dim].clone()
        batch /= batch.square().mean(dim=-1, keepdim=True).sqrt()
        report["scales"].append(
            {
                "name": name,
                "dim": cfg.dim,
                "expert_intermediate": cfg.expert_intermediate,
                "num_routed_experts": cfg.num_routed_experts,
                "experts_per_token": cfg.experts_per_token,
                "runs": [
                    run_arm(cfg, seed, arm, batch, device, args.steps)
                    for seed in args.seeds
                    for arm in ("production_rates", "base_expert_rates")
                ],
            }
        )
    report["elapsed_seconds"] = time.perf_counter() - started
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)


if __name__ == "__main__":
    main()

"""The experiment optimizer stack: NorMuonH for ordinary hidden matrices and
Adam for parameters whose norm carries semantic information, under one WSD
learning-rate multiplier.

NorMuonH combines NorMuon's Nesterov momentum, Newton-Schulz
orthogonalization, and neuron-wise second-moment normalization with the
Hyperball constraint (arXiv:2606.16899). Each constrained matrix keeps its
initial FP32 Frobenius radius, the NorMuon direction is normalized to unit
Frobenius norm, and every trial step is projected exactly back to the
initial-radius sphere. No optimizer group uses weight decay.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
from torch import Tensor

NS_COEFFS = (3.4445, -4.7750, 2.0315)
"""Quintic Newton-Schulz coefficients (Muon's standard choice)."""

DEFAULT_NORMUONH_LR = 2e-2
"""Stable dimensionless NorMuonH relative step for fresh runs."""

DEFAULT_ADAM_LR = 5e-4
"""Stable Adam learning rate for fresh runs."""


def orthogonalize(matrix: Tensor, steps: int = 5) -> Tensor:
    """Approximately orthogonalize a matrix via Newton-Schulz iteration.

    Operates on the smaller side (transposing when rows > cols) so the
    Gram matrix stays as small as possible; returns the original
    orientation.
    """
    a, b, c = NS_COEFFS
    transposed = matrix.shape[0] > matrix.shape[1]
    x = matrix.mT if transposed else matrix
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.mT if transposed else x


def _normuonh_batch(
    momentum: Tensor,
    row_moment: Tensor,
    parameters: Tensor,
    radii: Tensor,
    gradient: Tensor,
    lr: Tensor,
    momentum_beta: float,
    beta2: float,
    eps: float,
    ns_steps: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Batched Nesterov NorMuon direction and exact Hyperball update."""
    a, b, c = NS_COEFFS
    momentum = torch.lerp(momentum, gradient, 1 - momentum_beta)
    direction = torch.lerp(gradient, momentum, momentum_beta)
    transposed = direction.shape[-2] > direction.shape[-1]
    x = direction.mT if transposed else direction
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(ns_steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    update = x.mT if transposed else x
    row_moment = torch.lerp(
        row_moment, update.square().mean(dim=-1, keepdim=True), 1 - beta2
    )
    update = update / (row_moment.sqrt() + eps)
    update = update / (update.norm(dim=(-2, -1), keepdim=True).clamp_min(eps))

    radii = radii.reshape(-1, 1, 1)
    trial = parameters - lr * radii * update
    projected = radii * trial / (trial.norm(dim=(-2, -1), keepdim=True).clamp_min(eps))
    return momentum, row_moment, projected


_compiled_normuonh_batch = torch.compile(
    _normuonh_batch,
    fullgraph=True,
    dynamic=True,
    mode="max-autotune-no-cudagraphs",
)


class NorMuonH(torch.optim.Optimizer):
    """Nesterov NorMuon directions on each initial Frobenius sphere."""

    def __init__(
        self,
        params,
        lr: float = DEFAULT_NORMUONH_LR,
        momentum: float = 0.95,
        beta2: float = 0.95,
        eps: float = 1e-8,
        ns_steps: int = 5,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "beta2": beta2,
            "eps": eps,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)
        self._initial_radii: dict[torch.nn.Parameter, Tensor] = {}
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError(
                        f"NorMuonH takes 2D matrices only, got shape "
                        f"{tuple(parameter.shape)}"
                    )
                radius = parameter.detach().norm()
                radius_value = radius.item()
                if not math.isfinite(radius_value) or radius_value <= 0:
                    raise ValueError(
                        "NorMuonH requires a finite nonzero initial Frobenius "
                        f"radius, got {radius_value} for shape {tuple(parameter.shape)}"
                    )
                self._initial_radii[parameter] = radius

    @torch.no_grad()
    def warmup(self, active: set[torch.nn.Parameter] | None = None) -> None:
        """Compile every CUDA shape bucket without touching optimizer state."""
        for group in self.param_groups:
            buckets = defaultdict(list)
            for parameter in group["params"]:
                if active is not None and parameter not in active:
                    continue
                buckets[(parameter.device, parameter.dtype, parameter.shape)].append(
                    parameter
                )
            for (device, dtype, shape), parameters in buckets.items():
                if device.type != "cuda":
                    continue
                batch_shape = (len(parameters), *shape)
                matrices = torch.full(
                    batch_shape,
                    1 / math.sqrt(shape[0] * shape[1]),
                    device=device,
                    dtype=dtype,
                )
                rows = torch.zeros(
                    len(parameters), shape[0], 1, device=device, dtype=dtype
                )
                radii = torch.ones(len(parameters), device=device, dtype=dtype)
                scalar = torch.zeros((), device=device)
                _compiled_normuonh_batch(
                    matrices,
                    rows,
                    matrices,
                    radii,
                    matrices,
                    scalar,
                    group["momentum"],
                    group["beta2"],
                    group["eps"],
                    group["ns_steps"],
                )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            buckets = defaultdict(list)
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["momentum"] = torch.zeros_like(parameter)
                    state["row_moment"] = torch.zeros(
                        parameter.shape[0],
                        1,
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                    state["radius"] = self._initial_radii[parameter].clone()
                buckets[(parameter.device, parameter.dtype, parameter.shape)].append(
                    parameter
                )

            for (device, _dtype, _shape), parameters in buckets.items():
                gradients = torch.stack([parameter.grad for parameter in parameters])
                parameter_batch = torch.stack(parameters)
                momenta = torch.stack(
                    [self.state[parameter]["momentum"] for parameter in parameters]
                )
                row_moments = torch.stack(
                    [self.state[parameter]["row_moment"] for parameter in parameters]
                )
                radii = torch.stack(
                    [self.state[parameter]["radius"] for parameter in parameters]
                )
                lr = torch.scalar_tensor(group["lr"], device=device)
                update_fn = (
                    _compiled_normuonh_batch
                    if device.type == "cuda"
                    else _normuonh_batch
                )
                momenta, row_moments, projected = update_fn(
                    momenta,
                    row_moments,
                    parameter_batch,
                    radii,
                    gradients,
                    lr,
                    group["momentum"],
                    group["beta2"],
                    group["eps"],
                    group["ns_steps"],
                )
                torch._foreach_copy_(
                    [self.state[p]["momentum"] for p in parameters],
                    list(momenta.unbind()),
                )
                torch._foreach_copy_(
                    [self.state[p]["row_moment"] for p in parameters],
                    list(row_moments.unbind()),
                )
                torch._foreach_copy_(parameters, list(projected.unbind()))
        return loss


def split_parameters(model: torch.nn.Module) -> tuple[list, list]:
    """Partition trainable parameters into the sole NorMuonH and Adam groups.

    Ordinary hidden 2D weights get NorMuonH. The tied embedding/unembedding,
    global-attention gates, the FBT token gate, and PKDA's packed controls,
    decay expansion, and output-gate expansion are explicit matrix exceptions;
    they join norms, depthwise convolutions, routing parameters, and vectors in
    Adam. PKDA Q/K/V/output projections and the FBT value projection use
    NorMuonH.
    """
    matrices, rest = [], []
    pkda_adam = (
        ".attn.control_proj.",
        ".attn.decay_up.",
        ".attn.output_gate_up.",
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        adam_matrix = (
            "embed_tokens" in name
            or name.startswith("attention_gates.")
            or name == "fuse_gate.weight"
            or any(marker in name for marker in pkda_adam)
        )
        if parameter.ndim == 2 and not adam_matrix:
            matrices.append(parameter)
        else:
            rest.append(parameter)
    return matrices, rest


def build_optimizers(
    model: torch.nn.Module,
    *,
    lr_h: float = DEFAULT_NORMUONH_LR,
    lr_adam: float = DEFAULT_ADAM_LR,
    adam_betas: tuple[float, float] = (0.9, 0.95),
) -> list[torch.optim.Optimizer]:
    """Build the authoritative NorMuonH/Adam stack with stable WSD rates."""
    matrices, rest = split_parameters(model)
    normuonh = NorMuonH(matrices, lr=lr_h)
    use_fused_adam = bool(rest) and rest[0].is_cuda
    adam = torch.optim.Adam(
        rest,
        lr=lr_adam,
        betas=adam_betas,
        eps=1e-8,
        fused=use_fused_adam,
        capturable=use_fused_adam,
    )
    for group in normuonh.param_groups:
        group["stable_lr"] = lr_h
    for group in adam.param_groups:
        group["stable_lr"] = lr_adam
    return [normuonh, adam]


class OptimizerPair:
    """Checkpoint-facing facade over the NorMuonH/Adam stack.

    The shared checkpoints module speaks to one optimizer object; this
    bundles both state dicts under a single stable schema.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = list(optimizers)

    def state_dict(self) -> dict:
        return {"stack": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state: dict) -> None:
        for optimizer, saved in zip(self.optimizers, state["stack"], strict=True):
            optimizer.load_state_dict(saved)


def apply_schedule(
    optimizers: list[torch.optim.Optimizer],
    schedule,
    step: int,
) -> float:
    """Set both optimizer learning rates from the shared WSD multiplier.

    Returns the dimensionless NorMuonH relative step for telemetry.
    """
    lead = None
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            rate = schedule.rate_at(step, group["stable_lr"])
            group["lr"] = rate
            if lead is None:
                lead = rate
    return lead

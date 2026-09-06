"""The experiment optimizer stack: NorMuonH for ordinary hidden matrices and
NAdam for parameters whose norm carries semantic information, under one WSD
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

from .parameter_groups import is_normuonh_parameter

NS_COEFFS = (3.4445, -4.7750, 2.0315)
"""Quintic Newton-Schulz coefficients (Muon's standard choice)."""

DEFAULT_NORMUONH_LR = 6e-3
"""Stable dimensionless NorMuonH relative step for fresh runs."""

DEFAULT_NADAM_LR = 3e-4
"""Stable learning rate for every NAdam-owned parameter."""

DEFAULT_NADAM_BETAS = (0.9, 0.95)
"""First- and second-moment coefficients for NAdam."""

NADAM_MOMENTUM_DECAY = 4e-3
"""PyTorch/Dozat time-varying Nesterov momentum schedule coefficient."""


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


def _normuonh_bucket_step(
    parameters: list[Tensor],
    gradients: list[Tensor],
    momenta: list[Tensor],
    row_moments: list[Tensor],
    radii: list[Tensor],
    lr: Tensor,
    momentum_beta: float,
    beta2: float,
    eps: float,
    ns_steps: int,
) -> None:
    """Keep packing and state writebacks inside the compiled bucket boundary.

    State remains ordinary per-parameter tensors for checkpoints. Inductor can
    fuse the stack producers and update epilogues without retaining a second
    packed copy of the model or optimizer between updates.
    """
    new_momenta, new_rows, projected = _normuonh_batch(
        torch.stack(momenta),
        torch.stack(row_moments),
        torch.stack(parameters),
        torch.stack(radii),
        torch.stack(gradients),
        lr,
        momentum_beta,
        beta2,
        eps,
        ns_steps,
    )
    torch._foreach_copy_(momenta, list(new_momenta.unbind()))
    torch._foreach_copy_(row_moments, list(new_rows.unbind()))
    torch._foreach_copy_(parameters, list(projected.unbind()))


_compiled_normuonh_bucket_step = torch.compile(
    _normuonh_bucket_step,
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
                matrices = [
                    torch.nn.Parameter(
                        torch.full(
                            shape,
                            1 / math.sqrt(shape[0] * shape[1]),
                            device=device,
                            dtype=dtype,
                        ),
                        requires_grad=parameter.requires_grad,
                    )
                    for parameter in parameters
                ]
                gradients = [torch.zeros_like(matrix) for matrix in matrices]
                momenta = [torch.zeros_like(matrix) for matrix in matrices]
                rows = [
                    torch.zeros(shape[0], 1, device=device, dtype=dtype)
                    for _ in parameters
                ]
                radii = [torch.ones((), device=device, dtype=dtype) for _ in parameters]
                scalar = torch.zeros((), device=device)
                _compiled_normuonh_bucket_step(
                    matrices,
                    gradients,
                    momenta,
                    rows,
                    radii,
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
                lr = torch.scalar_tensor(group["lr"], device=device)
                update_fn = (
                    _compiled_normuonh_bucket_step
                    if device.type == "cuda"
                    else _normuonh_bucket_step
                )
                update_fn(
                    parameters,
                    [parameter.grad for parameter in parameters],
                    [self.state[parameter]["momentum"] for parameter in parameters],
                    [self.state[parameter]["row_moment"] for parameter in parameters],
                    [self.state[parameter]["radius"] for parameter in parameters],
                    lr,
                    group["momentum"],
                    group["beta2"],
                    group["eps"],
                    group["ns_steps"],
                )
        return loss


def split_parameters(model: torch.nn.Module) -> tuple[list, list]:
    """Partition trainable parameters into NorMuonH and NAdam groups.

    Ordinary hidden 2D weights get NorMuonH. The tied embedding/unembedding,
    global-attention gates, the FBT token gate, and PKDA's packed controls,
    decay expansion, and output-gate expansion join norms, depthwise
    convolutions, routing parameters, and vectors in the NAdam group. PKDA
    Q/K/V/output projections and the FBT value projection use NorMuonH.
    """
    matrices, nadam = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if is_normuonh_parameter(name, parameter):
            matrices.append(parameter)
        else:
            nadam.append(parameter)
    return matrices, nadam


def build_optimizers(
    model: torch.nn.Module,
    *,
    lr_normuonh: float = DEFAULT_NORMUONH_LR,
    lr_nadam: float = DEFAULT_NADAM_LR,
    nadam_betas: tuple[float, float] = DEFAULT_NADAM_BETAS,
) -> list[torch.optim.Optimizer]:
    """Build the authoritative NorMuonH/NAdam stack with stable WSD rates."""
    matrices, nadam_parameters = split_parameters(model)
    normuonh = NorMuonH(matrices, lr=lr_normuonh)
    use_foreach_nadam = bool(nadam_parameters) and nadam_parameters[0].is_cuda
    nadam = torch.optim.NAdam(
        nadam_parameters,
        lr=lr_nadam,
        betas=nadam_betas,
        eps=1e-8,
        momentum_decay=NADAM_MOMENTUM_DECAY,
        foreach=use_foreach_nadam,
    )
    for group in normuonh.param_groups:
        group["rate_name"] = "normuonh"
        group["stable_lr"] = lr_normuonh
    nadam.param_groups[0]["rate_name"] = "nadam"
    nadam.param_groups[0]["stable_lr"] = lr_nadam
    return [normuonh, nadam]


class OptimizerPair:
    """Checkpoint-facing facade over the NorMuonH/NAdam stack.

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
) -> dict[str, float]:
    """Set all parameter-group learning rates from the shared WSD multiplier.

    Returns each scheduled rate for telemetry.
    """
    rates = {}
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            rate = schedule.rate_at(step, group["stable_lr"])
            group["lr"] = rate
            rates[group["rate_name"]] = rate
    return rates

"""The experiment optimizer stack: NorMuonH for ordinary hidden matrices and
NAdam for parameters whose norm carries semantic information, under one WSD
learning-rate multiplier.

NorMuonH uses its base relative step for ordinary hidden matrices. Expert
gate/up maps use sqrt(1536 / dim), and expert down maps use sqrt(8 / (k+1)),
where k is the selected routed count; both shared and MTP experts follow
these rules. NAdam splits in two: matrices whose fan-in is the residual width
run at the base rate times the muP width ratio ``MUP_BASE_DIM / dim``, every other NAdam
parameter at the base rate. ``lr_nadam`` is therefore the rate at the
flagship width, and the tied readout carries the same ratio as a logit
multiplier inside the model.

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

from .parameter_groups import (
    is_normuonh_parameter,
    is_width_scaled_parameter,
    normuonh_rate_name,
)

NS_COEFFS = (3.4445, -4.7750, 2.0315)
"""Quintic Newton-Schulz coefficients (Muon's standard choice)."""

DEFAULT_NORMUONH_LR = 6e-3
"""Stable dimensionless NorMuonH relative step for fresh runs."""

DEFAULT_NADAM_LR = 3e-4
"""Stable NAdam learning rate at the muP reference width."""

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


def _normuonh_bucket_values(
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
) -> tuple[Tensor, Tensor, Tensor]:
    """Compile packing and update arithmetic without mutating their inputs."""
    return _normuonh_batch(
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


_compiled_normuonh_bucket_values = torch.compile(
    _normuonh_bucket_values,
    fullgraph=True,
    dynamic=True,
    mode="max-autotune-no-cudagraphs",
)


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
    # Materialize all results before any writeback. Fusing input momentum
    # mutation into the arithmetic can reread the updated momentum in a
    # singleton bucket, changing the Nesterov direction after the first step.
    values = (
        _compiled_normuonh_bucket_values
        if parameters[0].is_cuda else _normuonh_bucket_values
    )
    new_momenta, new_rows, projected = values(
        parameters, gradients, momenta, row_moments, radii, lr,
        momentum_beta, beta2, eps, ns_steps,
    )
    torch._foreach_copy_(momenta, list(new_momenta.unbind()))
    torch._foreach_copy_(row_moments, list(new_rows.unbind()))
    torch._foreach_copy_(parameters, list(projected.unbind()))


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
        max_bucket_elements: int | None = None,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "beta2": beta2,
            "eps": eps,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)
        self.max_bucket_elements = max_bucket_elements
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

    def _batches(self, parameters: list[Tensor]):
        """Bound expert packing temporaries without splitting any matrix."""
        size = len(parameters)
        if self.max_bucket_elements is not None:
            size = max(1, self.max_bucket_elements // parameters[0].numel())
        for start in range(0, len(parameters), size):
            yield parameters[start : start + size]

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
            for (device, dtype, shape), bucket in buckets.items():
                if device.type != "cuda":
                    continue
                for parameters in self._batches(bucket):
                    self._warm_bucket(parameters, device, dtype, shape, group)

    def _warm_bucket(self, parameters, device, dtype, shape, group) -> None:
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
            torch.zeros(shape[0], 1, device=device, dtype=dtype) for _ in parameters
        ]
        radii = [torch.ones((), device=device, dtype=dtype) for _ in parameters]
        scalar = torch.zeros((), device=device)
        _normuonh_bucket_step(
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

            for (device, _dtype, _shape), bucket in buckets.items():
                lr = torch.scalar_tensor(group["lr"], device=device)
                for parameters in self._batches(bucket):
                    _normuonh_bucket_step(
                        parameters,
                        [parameter.grad for parameter in parameters],
                        [self.state[parameter]["momentum"] for parameter in parameters],
                        [
                            self.state[parameter]["row_moment"]
                            for parameter in parameters
                        ],
                        [self.state[parameter]["radius"] for parameter in parameters],
                        lr,
                        group["momentum"],
                        group["beta2"],
                        group["eps"],
                        group["ns_steps"],
                    )
        return loss


def split_parameters(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    """Partition all trainable parameters into five disjoint scheduled groups.

    NorMuonH separates ordinary hidden matrices from expert gate/up and down
    projections. NAdam separates residual-width controls/routers from its
    base-rate embeddings, fixed-head-width expansions, and vectors.
    """
    groups = {
        name: [] for name in (
            "normuonh", "normuonh_expert_in", "normuonh_expert_out",
            "nadam", "nadam_width",
        )
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if is_normuonh_parameter(name, parameter):
            groups[normuonh_rate_name(name)].append(parameter)
        elif is_width_scaled_parameter(name, parameter):
            groups["nadam_width"].append(parameter)
        else:
            groups["nadam"].append(parameter)
    return groups


def build_optimizers(
    model: torch.nn.Module,
    *,
    lr_normuonh: float = DEFAULT_NORMUONH_LR,
    lr_nadam: float = DEFAULT_NADAM_LR,
    nadam_betas: tuple[float, float] = DEFAULT_NADAM_BETAS,
) -> list[torch.optim.Optimizer]:
    """Build the authoritative NorMuonH/NAdam stack with stable WSD rates.

    Expert input/output factors modify the final NorMuonH relative step,
    after gradient normalization. Initialization and Frobenius radii retain
    their fan-in contract. The two expert factors remain independent under
    geometry overrides, even though they coincide at the four presets.
    """
    parameters = split_parameters(model)
    rates = {
        "normuonh": lr_normuonh,
        "normuonh_expert_in": lr_normuonh * model.cfg.expert_in_lr_scale,
        "normuonh_expert_out": lr_normuonh * model.cfg.expert_out_lr_scale,
        "nadam": lr_nadam,
        "nadam_width": lr_nadam * model.cfg.mup_ratio,
    }

    def group(name):
        return {
            "params": parameters[name],
            "lr": rates[name],
            "rate_name": name,
            "stable_lr": rates[name],
        }

    normuonh = NorMuonH(
        [group(name) for name in ("normuonh", "normuonh_expert_in", "normuonh_expert_out")],
        lr=lr_normuonh,
        max_bucket_elements=32 * 1024 * 1024,
    )
    nadam_parameters = parameters["nadam"]
    use_foreach_nadam = bool(nadam_parameters) and nadam_parameters[0].is_cuda
    nadam = torch.optim.NAdam(
        [group("nadam"), group("nadam_width")],
        lr=lr_nadam,
        betas=nadam_betas,
        eps=1e-8,
        momentum_decay=NADAM_MOMENTUM_DECAY,
        foreach=use_foreach_nadam,
    )
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
            if isinstance(optimizer, torch.optim.NAdam):
                # PyTorch's generic load casts mu_product onto each parameter's
                # device. Non-capturable NAdam creates both scalar counters on
                # CPU; restore that same placement for exact CUDA resumption.
                for group in optimizer.param_groups:
                    if not group["capturable"]:
                        for parameter in group["params"]:
                            values = optimizer.state.get(parameter, {})
                            for key in ("step", "mu_product"):
                                if key in values:
                                    values[key] = values[key].cpu()


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

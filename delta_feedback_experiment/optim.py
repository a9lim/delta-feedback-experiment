"""The experiment optimizer stack: NorMuonH for ordinary hidden matrices and
NAdam for parameters whose norm carries semantic information, under one WSD
learning-rate multiplier.

NorMuonH uses an estimated RMS-to-RMS operator step for ordinary matrices.
Expert gate/up and down maps additionally use sqrt(8 / (k+1)), where k is
the selected routed count; shared and MTP experts follow the same rule.
NAdam splits in two: matrices whose fan-in is the residual width
run at the base rate times the muP width ratio ``MUP_BASE_DIM / dim``, every other NAdam
parameter at the base rate. ``lr_nadam`` is therefore the rate at the
flagship width, and the tied readout carries the same ratio as a logit
multiplier inside the model.

NorMuonH combines NorMuon's Nesterov momentum, Newton-Schulz
orthogonalization, and neuron-wise second-moment normalization with the
Hyperball constraint (arXiv:2606.16899). Each constrained matrix keeps its
initial FP32 Frobenius radius. After row adaptation, the tangent direction
is scaled by an estimated spectral norm and sqrt(fan_out / fan_in), then
the trial step is projected back onto the initial-radius sphere. Three
power iterations use a checkpointed right vector and an energy-row restart.
This spectral/tangent extension is a scaling candidate, not a strict bound
on the final displacement or an established Hyperball result.
No optimizer group uses weight decay.

On CUDA the Newton-Schulz iterations run in BF16, as reference Muon
implementations do; the direction returns to FP32 before row adaptation, the
spectral/tangent step, and the retraction. CPU and MPS keep FP32 throughout.
Each shape bucket holds its momenta, row moments, radii, and right vectors in
one packed tensor per state kind, with every parameter's state entries as
views into them, so a step packs only parameters and gradients.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .parameter_groups import (
    is_normuonh_parameter,
    is_width_scaled_parameter,
    normuonh_rate_name,
)

NS_COEFFS = (3.4445, -4.7750, 2.0315)
"""Quintic Newton-Schulz coefficients (Muon's standard choice)."""

NS_CUDA_DTYPE = torch.bfloat16
"""Newton-Schulz working precision on CUDA; every other device keeps FP32."""

DEFAULT_NORMUONH_LR = 6e-3
"""Stable estimated RMS-to-RMS trial-step budget for fresh runs."""

SPECTRAL_POWER_STEPS = 3
"""Paired power iterations per matrix update; no extra model forward passes."""

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


def _spectral_norm(
    matrix: Tensor, vector: Tensor, eps: float, steps: int = SPECTRAL_POWER_STEPS,
) -> tuple[Tensor, Tensor]:
    """Estimate batched spectral norms with a deterministic warm/restart choice.

    The strongest row supplies a non-null right-vector restart whenever the
    matrix is nonzero, including a new rank-one direction orthogonal to the
    saved vector. Choose whichever start has the larger measured image, then
    perform power iteration. No RNG is consumed; a zero matrix returns zero.
    The estimate is a lower bound in exact arithmetic, not a certificate.
    """
    row = matrix.square().sum(dim=-1).argmax(dim=-1)
    restart = matrix.gather(
        -2, row[:, None, None].expand(-1, 1, matrix.shape[-1])
    ).squeeze(-2)
    restart = restart / restart.norm(dim=-1, keepdim=True).clamp_min(eps)
    warm_image = (matrix @ vector.unsqueeze(-1)).squeeze(-1)
    restart_image = (matrix @ restart.unsqueeze(-1)).squeeze(-1)
    use_warm = warm_image.square().sum(dim=-1, keepdim=True) > (
        restart_image.square().sum(dim=-1, keepdim=True)
    )
    image = torch.where(use_warm, warm_image, restart_image)
    for iteration in range(steps):
        if iteration:
            image = (matrix @ vector.unsqueeze(-1)).squeeze(-1)
        left = image / image.norm(dim=-1, keepdim=True).clamp_min(eps)
        vector = (matrix.mT @ left.unsqueeze(-1)).squeeze(-1)
        vector = vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)
    image = (matrix @ vector.unsqueeze(-1)).squeeze(-1)
    return image.norm(dim=-1), vector


def _spectral_tangent_step(
    parameters: Tensor, update: Tensor, radii: Tensor, vector: Tensor,
    lr: Tensor, eps: float,
) -> tuple[Tensor, Tensor]:
    """Normalize a tangent trial step, then retract to the fixed weight sphere."""
    update = update / update.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    weight_sq = parameters.square().sum(dim=(-2, -1), keepdim=True)
    radial = (parameters * update).sum(dim=(-2, -1), keepdim=True) / (
        weight_sq.clamp_min(eps)
    )
    tangent = update - radial * parameters
    tangent_norm = tangent.norm(dim=(-2, -1), keepdim=True)
    # A radial direction has zero tangent mathematically. Do not amplify its
    # FP32 cancellation residue into a full spectral step.
    moving = tangent_norm > 32 * torch.finfo(parameters.dtype).eps
    tangent = torch.where(
        moving, tangent / tangent_norm.clamp_min(eps), torch.zeros_like(tangent)
    )
    sigma, vector = _spectral_norm(tangent, vector, eps)
    factor = math.sqrt(parameters.shape[-2] / parameters.shape[-1])
    trial = parameters - lr * factor * tangent / sigma[:, None, None].clamp_min(eps)
    projected = radii[:, None, None] * trial / trial.norm(
        dim=(-2, -1), keepdim=True
    ).clamp_min(eps)
    # Initialization, zero-rate schedule endpoints, and zero/radial updates
    # preserve weights byte-for-byte instead of renormalizing them again.
    return torch.where(moving & (lr != 0), projected, parameters), vector


def _normuonh_batch(
    momentum: Tensor,
    row_moment: Tensor,
    parameters: Tensor,
    radii: Tensor,
    spectral_vectors: Tensor,
    gradient: Tensor,
    lr: Tensor,
    momentum_beta: float,
    beta2: float,
    eps: float,
    ns_steps: int,
    ns_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Batched NorMuon direction and spectral tangent update on a fixed sphere.

    The Newton-Schulz iterations run in ``ns_dtype``; everything from the row
    second moment onward is back in the parameter dtype.
    """
    a, b, c = NS_COEFFS
    momentum = torch.lerp(momentum, gradient, 1 - momentum_beta)
    direction = torch.lerp(gradient, momentum, momentum_beta)
    transposed = direction.shape[-2] > direction.shape[-1]
    x = direction.mT if transposed else direction
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    x = x.to(ns_dtype)
    for _ in range(ns_steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    x = x.to(direction.dtype)
    update = x.mT if transposed else x
    row_moment = torch.lerp(
        row_moment, update.square().mean(dim=-1, keepdim=True), 1 - beta2
    )
    update = update / (row_moment.sqrt() + eps)
    projected, spectral_vectors = _spectral_tangent_step(
        parameters, update, radii, spectral_vectors, lr, eps
    )
    return momentum, row_moment, projected, spectral_vectors


def _normuonh_bucket_values(
    parameters: list[Tensor],
    gradients: list[Tensor],
    momentum: Tensor,
    row_moment: Tensor,
    radii: Tensor,
    spectral_vectors: Tensor,
    lr: Tensor,
    momentum_beta: float,
    beta2: float,
    eps: float,
    ns_steps: int,
    ns_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compile packing and update arithmetic without mutating their inputs.

    State arrives already packed; only the parameters and gradients, which are
    separate tensors owned by the model and the gradient buffers, are stacked.
    """
    return _normuonh_batch(
        momentum,
        row_moment,
        torch.stack(parameters),
        radii,
        spectral_vectors,
        torch.stack(gradients),
        lr,
        momentum_beta,
        beta2,
        eps,
        ns_steps,
        ns_dtype,
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
    momentum: Tensor,
    row_moment: Tensor,
    radii: Tensor,
    spectral_vectors: Tensor,
    lr: Tensor,
    momentum_beta: float,
    beta2: float,
    eps: float,
    ns_steps: int,
) -> None:
    # Materialize all results before any writeback. Fusing input momentum
    # mutation into the arithmetic can reread the updated momentum in a
    # singleton bucket, changing the Nesterov direction after the first step.
    cuda = parameters[0].is_cuda
    values = _compiled_normuonh_bucket_values if cuda else _normuonh_bucket_values
    new_momentum, new_rows, projected, new_vectors = values(
        parameters, gradients, momentum, row_moment, radii, spectral_vectors, lr,
        momentum_beta, beta2, eps, ns_steps,
        NS_CUDA_DTYPE if cuda else parameters[0].dtype,
    )
    momentum.copy_(new_momentum)
    row_moment.copy_(new_rows)
    spectral_vectors.copy_(new_vectors)
    torch._foreach_copy_(parameters, list(projected.unbind()))


@dataclass
class _ShapeBucket:
    """One compiled bucket: fixed members, packed state, cached row indices."""

    group_index: int
    params: tuple[torch.nn.Parameter, ...]
    momentum: Tensor | None = None
    row_moment: Tensor | None = None
    radius: Tensor | None = None
    spectral_vector: Tensor | None = None
    indices: dict[tuple[int, ...], Tensor] = field(default_factory=dict)


class NorMuonH(torch.optim.Optimizer):
    """Spectrally scaled NorMuon tangent directions on each initial sphere."""

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
        self._buckets = self._build_buckets()

    def _build_buckets(self) -> list[_ShapeBucket]:
        """Fix bucket membership once: device, dtype, shape, packing bound."""
        buckets = []
        for index, group in enumerate(self.param_groups):
            shapes = defaultdict(list)
            for parameter in group["params"]:
                shapes[(parameter.device, parameter.dtype, parameter.shape)].append(
                    parameter
                )
            for members in shapes.values():
                for parameters in self._batches(members):
                    buckets.append(_ShapeBucket(index, tuple(parameters)))
        return buckets

    def _batches(self, parameters: list[Tensor]):
        """Bound expert packing temporaries without splitting any matrix."""
        size = len(parameters)
        if self.max_bucket_elements is not None:
            size = max(1, self.max_bucket_elements // parameters[0].numel())
        for start in range(0, len(parameters), size):
            yield parameters[start : start + size]

    def _allocate(self, bucket: _ShapeBucket) -> None:
        """Pack one bucket's state into a tensor per kind, radii included."""
        if bucket.momentum is not None:
            return
        sample = bucket.params[0]
        count = len(bucket.params)
        rows, cols = sample.shape
        kwargs = {"device": sample.device, "dtype": sample.dtype}
        bucket.momentum = torch.zeros(count, rows, cols, **kwargs)
        bucket.row_moment = torch.zeros(count, rows, 1, **kwargs)
        bucket.spectral_vector = torch.zeros(count, cols, **kwargs)
        bucket.radius = torch.stack(
            [self._initial_radii[parameter] for parameter in bucket.params]
        )

    def _views(self, bucket: _ShapeBucket, row: int) -> dict[str, Tensor]:
        return {
            "momentum": bucket.momentum[row],
            "row_moment": bucket.row_moment[row],
            "radius": bucket.radius[row],
            "spectral_vector": bucket.spectral_vector[row],
        }

    def _rows(self, bucket: _ShapeBucket, present: list[int]) -> Tensor:
        """Cache the gather/scatter index of each recurring active subset."""
        key = tuple(present)
        index = bucket.indices.get(key)
        if index is None:
            index = torch.tensor(
                present, device=bucket.params[0].device, dtype=torch.long
            )
            bucket.indices[key] = index
        return index

    @torch.no_grad()
    def warmup(self, active: set[torch.nn.Parameter] | None = None) -> None:
        """Compile every CUDA shape bucket without touching optimizer state."""
        for bucket in self._buckets:
            sample = bucket.params[0]
            if not sample.is_cuda:
                continue
            count = sum(
                1
                for parameter in bucket.params
                if active is None or parameter in active
            )
            if count:
                self._warm_bucket(bucket, count)

    def _warm_bucket(self, bucket: _ShapeBucket, count: int) -> None:
        sample = bucket.params[0]
        group = self.param_groups[bucket.group_index]
        rows, cols = sample.shape
        kwargs = {"device": sample.device, "dtype": sample.dtype}
        matrices = [
            torch.nn.Parameter(
                torch.full(sample.shape, 1 / math.sqrt(rows * cols), **kwargs),
                requires_grad=parameter.requires_grad,
            )
            for parameter in bucket.params[:count]
        ]
        _normuonh_bucket_step(
            matrices,
            [torch.zeros_like(matrix) for matrix in matrices],
            torch.zeros(count, rows, cols, **kwargs),
            torch.zeros(count, rows, 1, **kwargs),
            torch.ones(count, **kwargs),
            torch.zeros(count, cols, **kwargs),
            torch.zeros((), device=sample.device),
            group["momentum"],
            group["beta2"],
            group["eps"],
            group["ns_steps"],
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        rates: dict[tuple[int, torch.device], Tensor] = {}
        for bucket in self._buckets:
            present = [
                row
                for row, parameter in enumerate(bucket.params)
                if parameter.grad is not None
            ]
            if not present:
                continue
            self._allocate(bucket)
            parameters = [bucket.params[row] for row in present]
            for row, parameter in zip(present, parameters):
                state = self.state[parameter]
                if not state:
                    state.update(self._views(bucket, row))
            group = self.param_groups[bucket.group_index]
            rate = (bucket.group_index, bucket.params[0].device)
            if rate not in rates:
                rates[rate] = torch.scalar_tensor(group["lr"], device=rate[1])
            arguments = (
                rates[rate],
                group["momentum"],
                group["beta2"],
                group["eps"],
                group["ns_steps"],
            )
            gradients = [parameter.grad for parameter in parameters]
            if len(present) == len(bucket.params):
                _normuonh_bucket_step(
                    parameters,
                    gradients,
                    bucket.momentum,
                    bucket.row_moment,
                    bucket.radius,
                    bucket.spectral_vector,
                    *arguments,
                )
                continue
            # A bucket whose members were not all reached this step gathers the
            # active rows, so absent parameters keep both weights and state.
            index = self._rows(bucket, present)
            momentum = bucket.momentum.index_select(0, index)
            row_moment = bucket.row_moment.index_select(0, index)
            spectral_vector = bucket.spectral_vector.index_select(0, index)
            _normuonh_bucket_step(
                parameters,
                gradients,
                momentum,
                row_moment,
                bucket.radius.index_select(0, index),
                spectral_vector,
                *arguments,
            )
            bucket.momentum.index_copy_(0, index, momentum)
            bucket.row_moment.index_copy_(0, index, row_moment)
            bucket.spectral_vector.index_copy_(0, index, spectral_vector)
        return loss

    def load_state_dict(self, state_dict) -> None:
        """Copy a restored state into the packed buckets, keeping the views.

        The generic load rebuilds ``self.state`` from the saved tensors; each
        restored value is copied into its bucket row and replaced by the view
        again, so resume lands in the same storage a fresh run allocates.
        """
        super().load_state_dict(state_dict)
        for bucket in self._buckets:
            # Reallocate so rows the checkpoint does not mention start from a
            # fresh run's zeros and initial radius rather than stale values.
            bucket.momentum = bucket.row_moment = None
            bucket.radius = bucket.spectral_vector = None
            restored = [
                (row, self.state[parameter])
                for row, parameter in enumerate(bucket.params)
                if self.state.get(parameter)
            ]
            if not restored:
                continue
            self._allocate(bucket)
            for row, state in restored:
                for name, view in self._views(bucket, row).items():
                    view.copy_(state[name])
                    state[name] = view


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

    Spectral normalization handles each matrix's fan-in/fan-out. Both expert
    groups additionally scale their operator-step budget by active count:
    coherently aligned branch changes add as sqrt(k+1) after the bank's
    forward normalization. Initialization and radii retain their fan-in rule.
    """
    parameters = split_parameters(model)
    rates = {
        "normuonh": lr_normuonh,
        "normuonh_expert_in": lr_normuonh * model.cfg.expert_lr_scale,
        "normuonh_expert_out": lr_normuonh * model.cfg.expert_lr_scale,
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

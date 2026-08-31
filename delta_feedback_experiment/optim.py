"""The experiment optimizer stack: NorMuon for ordinary hidden matrices and
Adam for scale-sensitive mixer gates plus non-matrix parameters under WSD
with cooldown weight-decay scaling.

NorMuon (arXiv:2510.05491): EMA momentum, Newton-Schulz
orthogonalization, then neuron-wise (row) second-moment normalization
and a global rescale to Frobenius norm 0.2·sqrt(m·n) — the row moments
shape *relative* row magnitudes while the rescale fixes the overall update
RMS at 0.2, so no bias correction is needed. The registered defaults are
learning rate 1e-2 and weight decay 0.01.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
from torch import Tensor

NS_COEFFS = (3.4445, -4.7750, 2.0315)
"""Quintic Newton-Schulz coefficients (Muon's standard choice)."""


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


def _normuon_batch(
    momentum: Tensor,
    row_moment: Tensor,
    gradient: Tensor,
    lr: Tensor,
    weight_decay: Tensor,
    momentum_beta: float,
    beta2: float,
    eps: float,
    ns_steps: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Batched NorMuon update for a bucket of identically shaped weights."""
    a, b, c = NS_COEFFS
    momentum = torch.lerp(momentum, gradient, 1 - momentum_beta)
    transposed = momentum.shape[-2] > momentum.shape[-1]
    x = momentum.mT if transposed else momentum
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(ns_steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    update = x.mT if transposed else x
    row_moment = torch.lerp(
        row_moment, update.square().mean(dim=-1, keepdim=True), 1 - beta2
    )
    update = update / (row_moment.sqrt() + eps)
    scale = 0.2 * math.sqrt(update.shape[-2] * update.shape[-1])
    scale = scale / (update.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    delta = update * scale * -lr
    decay = 1 - lr * weight_decay
    return momentum, row_moment, delta, decay


_compiled_normuon_batch = torch.compile(
    _normuon_batch,
    fullgraph=True,
    dynamic=True,
    mode="max-autotune-no-cudagraphs",
)


class NorMuon(torch.optim.Optimizer):
    """Neuron-wise normalized Muon for 2D hidden weight matrices."""

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0.95,
        beta2: float = 0.95,
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        ns_steps: int = 5,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "eps": eps,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError(
                        f"NorMuon takes 2D matrices only, got shape "
                        f"{tuple(parameter.shape)}"
                    )

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
                matrices = torch.zeros(batch_shape, device=device, dtype=dtype)
                rows = torch.zeros(
                    len(parameters), shape[0], 1, device=device, dtype=dtype
                )
                scalar = torch.zeros((), device=device)
                _compiled_normuon_batch(
                    matrices,
                    rows,
                    matrices,
                    scalar,
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
                buckets[(parameter.device, parameter.dtype, parameter.shape)].append(
                    parameter
                )

            for (device, _dtype, _shape), parameters in buckets.items():
                gradients = torch.stack([parameter.grad for parameter in parameters])
                momenta = torch.stack(
                    [self.state[parameter]["momentum"] for parameter in parameters]
                )
                row_moments = torch.stack(
                    [self.state[parameter]["row_moment"] for parameter in parameters]
                )
                lr = torch.scalar_tensor(group["lr"], device=device)
                weight_decay = torch.scalar_tensor(group["weight_decay"], device=device)
                update_fn = (
                    _compiled_normuon_batch if device.type == "cuda" else _normuon_batch
                )
                momenta, row_moments, deltas, decay = update_fn(
                    momenta,
                    row_moments,
                    gradients,
                    lr,
                    weight_decay,
                    group["momentum"],
                    group["beta2"],
                    group["eps"],
                    group["ns_steps"],
                )
                momentum_list = list(momenta.unbind())
                row_moment_list = list(row_moments.unbind())
                delta_list = list(deltas.unbind())
                torch._foreach_copy_(
                    [self.state[p]["momentum"] for p in parameters], momentum_list
                )
                torch._foreach_copy_(
                    [self.state[p]["row_moment"] for p in parameters], row_moment_list
                )
                torch._foreach_mul_(parameters, decay)
                torch._foreach_add_(parameters, delta_list)
        return loss


def split_parameters(model: torch.nn.Module) -> tuple[list, list]:
    """(NorMuon matrices, Adam rest) per the FBT/NorMuon convention.

    Ordinary hidden 2D weights get NorMuon.  The tied
    embedding/unembedding, global-attention gates, and PKDA's decay, update,
    preconditioner, and output-gate projections are explicit matrix exceptions;
    they join norms, depthwise convolutions, routing parameters, and vectors in
    Adam. PKDA Q/K/V/output projections and FBT fusion retain NorMuon.
    """
    matrices, rest = [], []
    pkda_adam = (
        ".attn.decay_",
        ".attn.beta_proj.",
        ".attn.precond_",
        ".attn.output_gate_",
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        adam_matrix = (
            "embed_tokens" in name
            or name.startswith("attention_gates.")
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
    lr_muon: float = 1e-2,
    wd_muon: float = 0.01,
    lr_adam: float = 5e-4,
    adam_betas: tuple[float, float] = (0.9, 0.95),
) -> list[torch.optim.Optimizer]:
    """The two-optimizer stack, stable rates recorded on each group.

    Each group carries ``stable_lr`` (and NorMuon ``stable_wd``) so the
    trainer can reshape lr — and, in cooldown, weight decay (AdamC) —
    as pure functions of the schedule step.
    """
    matrices, rest = split_parameters(model)
    muon = NorMuon(matrices, lr=lr_muon, weight_decay=wd_muon)
    use_fused_adam = bool(rest) and rest[0].is_cuda
    adam = torch.optim.Adam(
        rest,
        lr=lr_adam,
        betas=adam_betas,
        eps=1e-8,
        fused=use_fused_adam,
        capturable=use_fused_adam,
    )
    for group in muon.param_groups:
        group["stable_lr"] = lr_muon
        group["stable_wd"] = wd_muon
    for group in adam.param_groups:
        group["stable_lr"] = lr_adam
    return [muon, adam]


class OptimizerPair:
    """Checkpoint-facing facade over the NorMuon+Adam stack.

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
    """Set per-group lr (and AdamC-decayed wd in cooldown) at one step.

    Returns the NorMuon lr for telemetry. Weight decay follows the learning
    rate down during cooldown only; warmup keeps the full stable decay.
    """
    in_cooldown = schedule.phase(step)[0] == "cooldown"
    lead = None
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            rate = schedule.rate_at(step, group["stable_lr"])
            group["lr"] = rate
            if lead is None:
                lead = rate
            if "stable_wd" in group:
                factor = rate / group["stable_lr"] if in_cooldown else 1.0
                group["weight_decay"] = group["stable_wd"] * factor
    return lead

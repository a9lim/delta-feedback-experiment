"""The FBT-binding optimizer stack: NorMuon for hidden matrices, Adam
for everything else, under a WSD schedule with AdamC-style weight-decay
decay in cooldown.

NorMuon (arXiv:2510.05491): EMA momentum, Newton-Schulz
orthogonalization, then neuron-wise (row) second-moment normalization
and a global rescale to Frobenius norm 0.2·sqrt(m·n) — the row moments
shape *relative* row magnitudes while the rescale fixes the overall
update RMS at 0.2, so no bias correction is needed.  FBT's published
hyperparameters (lr 1e-2, wd 0.01) are the binding defaults here, not
the NorMuon paper's own.
"""

from __future__ import annotations

import math

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
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["momentum"] = torch.zeros_like(parameter)
                    state["row_moment"] = torch.zeros(
                        parameter.shape[0], 1,
                        device=parameter.device, dtype=parameter.dtype,
                    )
                momentum = state["momentum"]
                momentum.lerp_(gradient, 1 - group["momentum"])
                update = orthogonalize(momentum, group["ns_steps"])
                row_moment = state["row_moment"]
                row_moment.lerp_(
                    update.square().mean(dim=1, keepdim=True), 1 - group["beta2"]
                )
                update = update / (row_moment.sqrt() + group["eps"])
                scale = 0.2 * math.sqrt(parameter.numel()) / (update.norm() + 1e-7)
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"] * scale)
        return loss


def split_parameters(model: torch.nn.Module) -> tuple[list, list]:
    """(NorMuon matrices, Adam rest) per the FBT/NorMuon convention.

    Hidden 2D weights get NorMuon; the tied embedding/unembedding, norms,
    routing queries, null sources, and every other vector go to Adam.
    """
    matrices, rest = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 2 and "embed_tokens" not in name:
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
    adam = torch.optim.Adam(rest, lr=lr_adam, betas=adam_betas, eps=1e-8)
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

    Returns the NorMuon lr for telemetry.  Weight decay follows the
    learning rate down during cooldown only (Defazio's AdamC, as FBT
    applies it); warmup keeps the full stable decay.
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

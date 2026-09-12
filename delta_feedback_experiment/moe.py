"""One shared and top-k routed SwiGLU experts of explicit intermediate width.

Routing is token-local and dropless. CUDA groups the fixed ``k * tokens``
assignments by expert for sparse GEMMs; the portable implementation executes
the same selected rows with ordinary differentiable PyTorch operations.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .cuda_kernels import sink_linear
from .moe_kernels import sparse_experts

EXPERT_BIAS_RATE = 0.001


def validate_expert_geometry(intermediate: int, routed: int, selected: int) -> None:
    """Reject empty banks and impossible selections before model construction."""
    if intermediate < 1:
        raise ValueError("expert intermediate width must be positive")
    if not 1 <= selected <= routed:
        raise ValueError("experts must satisfy 1 <= experts_per_token <= num_routed_experts")


class _DispatchGather(torch.autograd.Function):
    """Rows of the dispatched matrix in token order, through a known permutation.

    ``rows[inverse]`` gathers each token's slot rows; because ``assignments``
    is the inverse permutation, the adjoint is the gather through it, so
    neither direction scatters or adds atomically.
    """

    @staticmethod
    def forward(ctx, rows: Tensor, inverse: Tensor, assignments: Tensor) -> Tensor:
        ctx.save_for_backward(assignments)
        return rows.index_select(0, inverse)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None, None]:
        (assignments,) = ctx.saved_tensors
        return gradient.index_select(0, assignments), None, None


class Expert(nn.Module):
    """A full-residual-width SwiGLU with its own matrix optimizer state."""

    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(dim, 2 * intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)
        self.gate_up_sink: Tensor | None = None
        self.down_sink: Tensor | None = None
        self.gate_up_shadow: Tensor | None = None
        self.down_shadow: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        gate, up = sink_linear(
            x, (self.gate_up_proj.weight,), (self.gate_up_sink,), self.gate_up_shadow
        ).chunk(2, dim=-1)
        return sink_linear(
            F.silu(gate) * up,
            (self.down_proj.weight,),
            (self.down_sink,),
            self.down_shadow,
        )


class MixtureOfExperts(nn.Module):
    """One shared expert plus a normalized top-k routed mixture.

    The branch is ``(shared + k * sum(selected_probability * expert)) / sqrt(k + 1)``.
    At uniform routing the k+1 independent fan-in-normalized expert outputs
    therefore have the same initial variance as one dense FFN. The returned
    auxiliary is the unweighted sequence-balancing loss; the trainer owns its
    coefficient and averaging across executed layer invocations. Selection
    uses sigmoid affinity plus bias, while gates use the unbiased affinities.
    Returned dispatch counts drive a separate update after each training step.
    """

    def __init__(
        self,
        dim: int,
        expert_intermediate: int,
        num_routed_experts: int = 15,
        experts_per_token: int = 3,
    ):
        super().__init__()
        validate_expert_geometry(expert_intermediate, num_routed_experts, experts_per_token)
        self.dim = dim
        self.intermediate = expert_intermediate
        self.num_routed_experts = num_routed_experts
        self.experts_per_token = experts_per_token
        self.output_scale = 1 / math.sqrt(experts_per_token + 1)
        self.shared = Expert(dim, self.intermediate)
        self.experts = nn.ModuleList(
            Expert(dim, self.intermediate) for _ in range(self.num_routed_experts)
        )
        self.router = nn.Linear(dim, self.num_routed_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(self.num_routed_experts))
        self._gate_up_shadow: Tensor | None = None
        self._down_shadow: Tensor | None = None

    def forward(
        self, x: Tensor, *, want_weights: bool = False
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
        flat = x.reshape(-1, self.dim)
        router_dtype = torch.float64 if flat.dtype == torch.float64 else torch.float32
        # Routing is a discrete decision: keep the logits in FP32 even when
        # the expert tensor-core GEMMs run under BF16 autocast.
        with torch.autocast(device_type=flat.device.type, enabled=False):
            logits = F.linear(
                flat.to(router_dtype), self.router.weight.to(router_dtype)
            )
        affinities = logits.sigmoid()
        selected = (affinities + self.expert_bias.to(router_dtype)).topk(
            self.experts_per_token, dim=-1
        ).indices
        selected_scores = affinities.gather(1, selected)
        selected_probability = selected_scores / selected_scores.sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(router_dtype).tiny)
        probabilities = affinities / affinities.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(router_dtype).tiny
        )
        # Membership by comparison rather than one_hot: the compiled graph then
        # carries no index-range assertions, and the same [pairs, experts]
        # matrix drives the counts and the dispatch order below.
        experts = torch.arange(self.num_routed_experts, device=flat.device)
        flat_selected = selected.reshape(-1)
        membership = (flat_selected.unsqueeze(1) == experts).to(torch.int32)
        counts = membership.sum(dim=0, dtype=torch.int64)
        # The weak auxiliary controls individual sequences. The step-level
        # controller receives detached counts and never runs in this forward.
        # V3's sequence auxiliary uses affinity-only choices, independently
        # of the bias-controlled dispatch decisions used by the controller.
        auxiliary_selection = (
            affinities.topk(self.experts_per_token, dim=-1).indices.unsqueeze(-1)
            == experts
        ).sum(dim=1)
        length = x.shape[-2] if x.ndim >= 2 else 1
        sequence_probabilities = probabilities.reshape(-1, length, self.num_routed_experts)
        fractions = (
            auxiliary_selection.reshape(-1, length, self.num_routed_experts)
            .to(router_dtype)
            .mean(dim=1)
        ) / self.experts_per_token
        auxiliary = self.num_routed_experts * (
            sequence_probabilities.mean(dim=1) * fractions
        ).sum(dim=-1).mean()

        if flat.is_cuda:
            offsets = F.pad(counts.cumsum(dim=0), (1, 0))
            # Counting sort into expert-major dispatch order, stable in
            # (token, slot): ``inverse`` is each pair's dispatch row and
            # ``assignments`` the pair at each dispatch row. Prefix counts
            # replace a general sort and make the order deterministic.
            ranks = (
                (membership.cumsum(dim=0) - membership)
                .gather(1, flat_selected.unsqueeze(1))
                .squeeze(1)
            )
            inverse = offsets[flat_selected] + ranks
            assignments = torch.empty_like(inverse).scatter_(
                0, inverse, torch.arange(inverse.numel(), device=flat.device)
            )
            routed = sparse_experts(
                flat,
                assignments,
                inverse,
                offsets,
                tuple(expert.gate_up_proj.weight for expert in self.experts),
                tuple(expert.down_proj.weight for expert in self.experts),
                tuple(expert.gate_up_sink for expert in self.experts),
                tuple(expert.down_sink for expert in self.experts),
                self._gate_up_shadow,
                self._down_shadow,
            )
            # Gather each token's slot rows and mix them in FP32: one fused
            # read of the dispatched rows instead of a scatter and a second pass.
            mixed = (
                (
                    _DispatchGather.apply(routed, inverse, assignments)
                    .view(-1, self.experts_per_token, self.dim)
                    .to(router_dtype)
                    * selected_probability.unsqueeze(-1)
                )
                .sum(dim=1)
                .to(flat.dtype)
            )
        else:
            mixed = torch.zeros_like(flat)
            for index, expert in enumerate(self.experts):
                token, slot = torch.where(selected == index)
                values = expert(flat.index_select(0, token))
                weighted = (
                    values.to(router_dtype) * selected_probability[token, slot, None]
                ).to(flat.dtype)
                mixed = mixed.index_add(0, token, weighted)

        output = (
            (self.shared(flat) + self.experts_per_token * mixed) * self.output_scale
        ).reshape_as(x)
        weights = None
        if want_weights:
            weights = (
                torch.zeros_like(probabilities)
                .scatter(1, selected, selected_probability)
                .reshape(*x.shape[:-1], self.num_routed_experts)
            )
        return output, auxiliary, weights, counts

    @torch.no_grad()
    def update_bias(self, counts: Tensor, *, rate: float = EXPERT_BIAS_RATE) -> None:
        """Adjust selection for the next step from completed logical forwards.

        Counts include all microbatches and all uses of this physical bank.
        Comparing integers avoids rounding the fractional per-expert target.
        Evaluation keeps the trained selection biases fixed.
        """
        if not self.training:
            return
        if counts.shape != self.expert_bias.shape or counts.dtype != torch.int64:
            raise ValueError("expert counts must be an int64 vector matching the routed bank")
        direction = (counts.sum() - self.num_routed_experts * counts).sign()
        self.expert_bias.add_(direction.to(self.expert_bias), alpha=rate)

    def bind_gradient_sinks(
        self, sinks: dict[nn.Parameter, Tensor] | None
    ) -> tuple[set[nn.Parameter], list[tuple[Tensor, Tensor]]]:
        """Bind FP32 sinks and retain address-stable, nonpersistent BF16 copies.

        Packed copies contain only BF16 operands. Every master parameter and
        optimizer-facing gradient remains its own two-dimensional matrix.
        Returned refresh pairs join the model's once-per-step shadow refresh.
        """
        lookup = sinks or {}
        bound: set[nn.Parameter] = set()
        refresh: list[tuple[Tensor, Tensor]] = []

        def bind(expert: Expert) -> None:
            for name in ("gate_up", "down"):
                parameter = getattr(expert, f"{name}_proj").weight
                sink = lookup.get(parameter) if parameter.requires_grad else None
                if sink is not None:
                    if (
                        sink.shape != parameter.shape
                        or sink.dtype != torch.float32
                        or sink.device != parameter.device
                        or not sink.is_contiguous()
                    ):
                        raise ValueError(
                            "MoE sinks must match their FP32 matrix parameters"
                        )
                    bound.add(parameter)
                setattr(expert, f"{name}_sink", sink)

        def shadows(
            current: Tensor | None, parameters: tuple[Tensor, ...]
        ) -> Tensor | None:
            if sinks is None or not parameters[0].is_cuda:
                return None
            shape = (len(parameters), *parameters[0].shape)
            fresh = (
                current is None
                or current.shape != shape
                or current.device != parameters[0].device
            )
            if fresh:
                current = torch.empty(
                    shape, device=parameters[0].device, dtype=torch.bfloat16
                )
            for index, parameter in enumerate(parameters):
                segment = current[index]
                if fresh:
                    segment.copy_(parameter.detach())
                refresh.append((segment, parameter.detach()))
            return current

        bind(self.shared)
        for name in ("gate_up", "down"):
            parameter = getattr(self.shared, f"{name}_proj").weight
            existing = getattr(self.shared, f"{name}_shadow")
            packed = None
            if getattr(self.shared, f"{name}_sink") is not None:
                packed = shadows(
                    None if existing is None else existing.unsqueeze(0), (parameter,)
                )
            setattr(
                self.shared, f"{name}_shadow", None if packed is None else packed[0]
            )
        for expert in self.experts:
            bind(expert)
        self._gate_up_shadow = shadows(
            self._gate_up_shadow,
            tuple(expert.gate_up_proj.weight for expert in self.experts),
        )
        self._down_shadow = shadows(
            self._down_shadow,
            tuple(expert.down_proj.weight for expert in self.experts),
        )
        return bound, refresh

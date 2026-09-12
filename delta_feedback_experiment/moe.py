"""One shared and three of fifteen routed quarter-width SwiGLU experts.

Routing is token-local and dropless. CUDA groups the fixed ``3 * tokens``
assignments by expert for sparse GEMMs; the portable implementation executes
the same selected rows with ordinary differentiable PyTorch operations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .cuda_kernels import sink_linear
from .moe_kernels import sparse_experts

NUM_ROUTED_EXPERTS = 15
EXPERTS_PER_TOKEN = 3
EXPERT_WIDTH_DIVISOR = 4


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
    """Quarter-width shared expert plus a normalized top-three routed mixture.

    The branch is ``(shared + 3 * sum(selected_probability * expert)) / 2``.
    At uniform routing the four independent fan-in-normalized expert outputs
    therefore have the same initial variance as one dense FFN. The returned
    auxiliary is the unweighted load-balancing loss; the trainer owns its
    coefficient and averaging across executed layer invocations.
    """

    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        if intermediate < EXPERT_WIDTH_DIVISOR or intermediate % EXPERT_WIDTH_DIVISOR:
            raise ValueError("MoE requires an FFN width divisible by four")
        self.dim = dim
        self.intermediate = intermediate // EXPERT_WIDTH_DIVISOR
        self.shared = Expert(dim, self.intermediate)
        self.experts = nn.ModuleList(
            Expert(dim, self.intermediate) for _ in range(NUM_ROUTED_EXPERTS)
        )
        self.router = nn.Linear(dim, NUM_ROUTED_EXPERTS, bias=False)
        self._gate_up_shadow: Tensor | None = None
        self._down_shadow: Tensor | None = None

    def forward(
        self, x: Tensor, *, want_weights: bool = False
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        flat = x.reshape(-1, self.dim)
        router_dtype = torch.float64 if flat.dtype == torch.float64 else torch.float32
        # Routing is a discrete decision: keep the logits in FP32 even when
        # the expert tensor-core GEMMs run under BF16 autocast.
        with torch.autocast(device_type=flat.device.type, enabled=False):
            logits = F.linear(
                flat.to(router_dtype), self.router.weight.to(router_dtype)
            )
        top_logits, selected = logits.topk(EXPERTS_PER_TOKEN, dim=-1)
        selected_probability = top_logits.softmax(dim=-1)
        probabilities = logits.softmax(dim=-1)
        selection = F.one_hot(selected, NUM_ROUTED_EXPERTS).sum(dim=1)
        fractions = selection.to(router_dtype).mean(dim=0) / EXPERTS_PER_TOKEN
        auxiliary = NUM_ROUTED_EXPERTS * (probabilities.mean(dim=0) * fractions).sum()

        if flat.is_cuda:
            counts = selection.sum(dim=0)
            offsets = F.pad(counts.cumsum(dim=0), (1, 0))
            _, assignments = selected.flatten().sort(stable=True)
            routed = sparse_experts(
                flat,
                assignments,
                offsets,
                tuple(expert.gate_up_proj.weight for expert in self.experts),
                tuple(expert.down_proj.weight for expert in self.experts),
                tuple(expert.gate_up_sink for expert in self.experts),
                tuple(expert.down_sink for expert in self.experts),
                self._gate_up_shadow,
                self._down_shadow,
            )
            ordered = torch.empty_like(routed).index_copy(0, assignments, routed)
            mixed = (
                (
                    ordered.view(-1, EXPERTS_PER_TOKEN, self.dim).to(router_dtype)
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

        output = ((self.shared(flat) + EXPERTS_PER_TOKEN * mixed) / 2).reshape_as(x)
        weights = None
        if want_weights:
            weights = (
                torch.zeros_like(probabilities)
                .scatter(1, selected, selected_probability)
                .reshape(*x.shape[:-1], NUM_ROUTED_EXPERTS)
            )
        return output, auxiliary, weights

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

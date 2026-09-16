"""Parameter sites: one contiguous slab per GEMM operand, and who owns it.

A site is the set of parameters one kernel reads as a single operand: the
three PKDA projections behind one GEMM, a dense layer's QKV projection with
its attention gate, an expert bank's stacked gate/up or down matrices, the
fusion matrix. Each site holds two slabs, a working copy in the dtype the
kernels read and an FP32 gradient sink the backward accumulates into, and
every member parameter is a view of both. The parameters NAdam owns share
one flat FP32 gradient arena instead.

Ownership follows the site. A stacked bank is chunked along its expert
axis, padded to a multiple of the rank count, so every rank owns a
contiguous run of experts and the bank's reduce-scatter and all-gather run
in place. A dense site is owned whole by one rank, assigned greedily by
size after the banks have spread their load evenly. Replicated parameters,
everything NAdam owns, have their gradients all-reduced and their working
copies refreshed from the FP32 masters every rank keeps.

Once the optimizer holds the FP32 masters of the matrices this rank owns,
:meth:`ParameterSites.adopt` re-points every sharded parameter at its
working view and the model releases the replicated FP32 copies. On CUDA the
working dtype is BF16, so a NorMuonH matrix then costs two bytes per element
on every rank plus four for its master on the owner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor, nn

from . import distributed


@dataclass(frozen=True)
class MemberView:
    """One parameter's views of its site's slabs."""

    weight: Tensor | None
    grad: Tensor


@dataclass(frozen=True)
class Binding:
    """What a site's owner module receives: the slabs and the member views.

    ``weight`` and ``grad`` cover the real members only; a stacked slab's
    rank-multiple padding stays outside the views the kernels read.
    """

    weight: Tensor | None
    grad: Tensor
    members: tuple[MemberView, ...]


@dataclass(frozen=True)
class SlabSpec:
    """One site as the model declares it.

    ``kind`` is ``rows`` for members concatenated along the first dimension,
    ``stack`` for equal-shape members stacked along a new leading axis, and
    ``arena`` for a flat FP32 gradient buffer with no working copy. Sharded
    members belong to NorMuonH and precede any replicated member in a row
    slab, so the sharded rows form one contiguous leading range.
    """

    name: str
    members: tuple[nn.Parameter, ...]
    bind: Callable[[Binding | None], None]
    kind: str = "rows"
    sharded: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("rows", "stack", "arena"):
            raise ValueError(f"unknown slab kind {self.kind!r}")
        if not self.members:
            raise ValueError(f"site {self.name} has no members")
        sharded = self.sharded or tuple(self.kind != "arena" for _ in self.members)
        if len(sharded) != len(self.members):
            raise ValueError(f"site {self.name}: one sharded flag per member")
        object.__setattr__(self, "sharded", sharded)
        if self.kind == "stack" and not all(sharded):
            raise ValueError(f"stacked site {self.name} must be wholly sharded")
        if self.kind == "arena" and any(sharded):
            raise ValueError(f"arena {self.name} holds replicated parameters only")
        if self.kind == "rows":
            if not all(m.ndim == 2 for m in self.members):
                raise ValueError(f"row site {self.name} takes matrices only")
            if len({m.shape[1] for m in self.members}) != 1:
                raise ValueError(f"row site {self.name} members must share columns")
            leading = list(sharded)
            if leading != sorted(leading, reverse=True):
                raise ValueError(f"site {self.name}: sharded members come first")
        if self.kind == "stack" and len({tuple(m.shape) for m in self.members}) != 1:
            raise ValueError(f"stacked site {self.name} members must share a shape")

    @property
    def sharded_members(self) -> tuple[nn.Parameter, ...]:
        return tuple(m for m, s in zip(self.members, self.sharded, strict=True) if s)


@dataclass
class Slab:
    """One allocated site."""

    spec: SlabSpec
    weight: Tensor | None
    grad: Tensor
    views: tuple[MemberView, ...]
    chunk: int = 0
    """Stacked: entries per rank along the padded leading axis."""
    sharded_rows: int = 0
    """Rows: leading rows owned by ``owner``; the rest are replicated."""
    owner: int | None = None
    refresh: list[tuple[Tensor, Tensor]] = field(default_factory=list)
    """Replicated members' (working view, FP32 master) pairs."""
    grad_real: Tensor | None = None
    """The gradient slab over the real members, padding excluded."""


class ParameterSites:
    """The site table of one model on one rank."""

    def __init__(self, model: nn.Module, topology: distributed.Topology):
        self.model = model
        self.topology = topology
        self.specs: list[SlabSpec] = list(model.slab_specs())
        self.owner: dict[nn.Parameter, int] = {}
        self.slabs: list[Slab] = []
        self._views: dict[nn.Parameter, MemberView] = {}
        self._adopted = False
        seen: set[nn.Parameter] = set()
        for spec in self.specs:
            for member in spec.members:
                if member in seen:
                    raise ValueError(f"{spec.name}: parameter declared twice")
                seen.add(member)
        declared = {p for p in model.parameters() if p.requires_grad}
        if seen != declared:
            raise ValueError("slab specs must cover every trainable parameter once")
        self._plan()

    # -- ownership -------------------------------------------------------------

    def _plan(self) -> None:
        world = self.topology.world
        loads = [0.0] * world
        dense: list[tuple[int, str, SlabSpec]] = []
        for spec in self.specs:
            if spec.kind == "stack":
                chunk = self.topology.chunk(len(spec.members))
                for index, member in enumerate(spec.members):
                    rank = index // chunk
                    self.owner[member] = rank
                    loads[rank] += member.numel()
            elif spec.sharded_members:
                size = sum(m.numel() for m in spec.sharded_members)
                dense.append((size, spec.name, spec))
        # Largest first onto the lightest rank; ties by name keep every rank's
        # table identical, since all of them compute it from the same model.
        for size, _, spec in sorted(dense, key=lambda item: (-item[0], item[1])):
            rank = min(range(world), key=lambda r: (loads[r], r))
            for member in spec.sharded_members:
                self.owner[member] = rank
            loads[rank] += size

    @property
    def rank(self) -> int:
        return self.topology.rank

    @property
    def sharded(self) -> frozenset[nn.Parameter]:
        return frozenset(self.owner)

    @property
    def owned(self) -> frozenset[nn.Parameter]:
        return frozenset(p for p, rank in self.owner.items() if rank == self.rank)

    @property
    def replicated(self) -> frozenset[nn.Parameter]:
        return frozenset(
            m for spec in self.specs for m, s in zip(spec.members, spec.sharded) if not s
        )

    # -- allocation ------------------------------------------------------------

    @property
    def allocated(self) -> bool:
        return bool(self.slabs)

    def allocate(self) -> None:
        """Allocate every slab, copy the parameters in, and bind the sites."""
        if self.allocated:
            raise RuntimeError("sites are already allocated")
        for spec in self.specs:
            self.slabs.append(self._allocate(spec))
        for slab in self.slabs:
            slab.spec.bind(Binding(self._real(slab.weight, slab), slab.grad_real, slab.views))

    def _allocate(self, spec: SlabSpec) -> Slab:
        sample = spec.members[0]
        device = sample.device
        working = torch.bfloat16 if device.type == "cuda" else torch.float32
        if spec.kind == "arena":
            total = sum(m.numel() for m in spec.members)
            grad = torch.zeros(total, dtype=torch.float32, device=device)
            views, start = [], 0
            for member in spec.members:
                views.append(MemberView(None, grad[start : start + member.numel()].view_as(member)))
                start += member.numel()
            return Slab(spec, None, grad, tuple(views), grad_real=grad)
        if spec.kind == "stack":
            count = len(spec.members)
            padded = self.topology.padded(count)
            shape = (padded, *sample.shape)
            weight = torch.zeros(shape, dtype=working, device=device)
            grad = torch.zeros(shape, dtype=torch.float32, device=device)
            views = []
            for index, member in enumerate(spec.members):
                weight[index].copy_(member.detach())
                views.append(MemberView(weight[index], grad[index]))
            return Slab(
                spec, weight, grad, tuple(views), chunk=self.topology.chunk(count),
                grad_real=grad[:count],
            )
        rows = sum(m.shape[0] for m in spec.members)
        weight = torch.empty((rows, sample.shape[1]), dtype=working, device=device)
        grad = torch.zeros((rows, sample.shape[1]), dtype=torch.float32, device=device)
        views, start, sharded_rows = [], 0, 0
        refresh = []
        for member, is_sharded in zip(spec.members, spec.sharded, strict=True):
            stop = start + member.shape[0]
            weight[start:stop].copy_(member.detach())
            views.append(MemberView(weight[start:stop], grad[start:stop]))
            if is_sharded:
                sharded_rows = stop
            else:
                refresh.append((weight[start:stop], member))
            start = stop
        owner = self.owner.get(spec.sharded_members[0]) if spec.sharded_members else None
        return Slab(
            spec, weight, grad, tuple(views), sharded_rows=sharded_rows, owner=owner,
            refresh=refresh, grad_real=grad,
        )

    @staticmethod
    def _real(weight: Tensor | None, slab: Slab) -> Tensor | None:
        if weight is None:
            return None
        if slab.spec.kind == "stack":
            return weight[: len(slab.spec.members)]
        return weight

    def release(self) -> None:
        """Unbind every site; the model returns to ordinary autograd gradients."""
        for slab in self.slabs:
            slab.spec.bind(None)
        self.slabs.clear()
        self._views.clear()

    # -- working copies --------------------------------------------------------

    def adopt(self) -> None:
        """Make every sharded parameter its working view.

        The optimizer must already hold the FP32 masters of the matrices this
        rank owns: from here on the model has no FP32 copy of any sharded
        matrix, and the working copy is what the optimizer writes and the
        gather refreshes.
        """
        if not self.allocated:
            raise RuntimeError("allocate the sites before adopting working copies")
        for slab in self.slabs:
            for member, view, is_sharded in zip(
                slab.spec.members, slab.views, slab.spec.sharded, strict=True
            ):
                if is_sharded:
                    member.data = view.weight
        self._adopted = True

    @property
    def adopted(self) -> bool:
        return self._adopted

    def refresh_pairs(self) -> list[tuple[Tensor, Tensor]]:
        """Replicated (working view, FP32 master) pairs to refresh after a step."""
        return [pair for slab in self.slabs for pair in slab.refresh]

    # -- gradients -------------------------------------------------------------

    def gradients(self) -> dict[nn.Parameter, Tensor]:
        """Every trainable parameter's FP32 gradient view."""
        return {
            member: view.grad
            for slab in self.slabs
            for member, view in zip(slab.spec.members, slab.views, strict=True)
        }

    def zero_gradients(self) -> None:
        torch._foreach_zero_([slab.grad for slab in self.slabs])

    def reduce_gradients(self) -> None:
        """Sum the step's gradients across ranks onto their owners.

        Stacked slabs reduce-scatter in place, so each rank's chunk holds the
        sum for the experts it owns; a dense site's sharded rows reduce onto
        the owner; replicated rows and the arena all-reduce.
        """
        rank = self.rank
        for slab in self.slabs:
            spec = slab.spec
            if spec.kind == "stack":
                distributed.reduce_scatter_(slab.grad, rank, slab.chunk)
                continue
            if slab.sharded_rows:
                distributed.reduce_(slab.grad[: slab.sharded_rows], slab.owner)
            if slab.sharded_rows < slab.grad.shape[0]:
                distributed.all_reduce_(slab.grad[slab.sharded_rows :])

    def gather_weights(self) -> None:
        """Bring every owner's updated working copy to every rank."""
        rank = self.rank
        for slab in self.slabs:
            spec = slab.spec
            if spec.kind == "stack":
                distributed.all_gather_(slab.weight, rank, slab.chunk)
            elif slab.sharded_rows:
                distributed.broadcast_(slab.weight[: slab.sharded_rows], slab.owner)

    def gradient_norm(self) -> float:
        """L2 norm of the reduced global gradient; non-finite stops the run.

        Nothing is clipped. Each rank squares the reduced entries it owns and
        rank zero adds the replicated ones, so one all-reduce gives every
        rank the same global norm.
        """
        device = self.slabs[0].grad.device
        total = torch.zeros((), dtype=torch.float32, device=device)
        rank = self.rank
        for slab in self.slabs:
            spec = slab.spec
            if spec.kind == "stack":
                start = rank * slab.chunk
                stop = min(start + slab.chunk, len(spec.members))
                if stop > start:
                    total += slab.grad[start:stop].square().sum()
                continue
            if slab.sharded_rows and slab.owner == rank:
                total += slab.grad[: slab.sharded_rows].square().sum()
            if slab.sharded_rows < slab.grad.shape[0] and self.topology.main:
                total += slab.grad[slab.sharded_rows :].square().sum()
        distributed.all_reduce_(total)
        norm = math.sqrt(total.item())
        if not math.isfinite(norm):
            raise RuntimeError(f"the gradient norm is non-finite ({norm})")
        return norm

    # -- checkpoints -----------------------------------------------------------

    def collect(
        self,
        member: nn.Parameter,
        tensor: Tensor | None,
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> Tensor | None:
        """Bring one owner-held tensor of a sharded member to rank zero.

        The owner passes the tensor and every other rank ``None``; the shape
        and dtype are stated by every caller, since each rank can derive them
        from the member alone. Rank zero gets the tensor itself from a local
        owner or a host copy from a remote one; other ranks get ``None``.
        """
        owner = self.owner[member]
        if self.topology.world == 1:
            return tensor
        return distributed.collect(
            tensor, owner, self.topology, shape=shape, dtype=dtype, device=member.device
        )

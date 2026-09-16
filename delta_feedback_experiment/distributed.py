"""Data-parallel ranks and the collectives one training step needs.

One process per device. Every rank runs the same graphs on its own slice of
the step's rows: gradients meet in a reduction that lands each matrix's sum
on the rank that owns it, and updated weights come back through a gather.
World size one is the same code with no communication at all, so the
single-device path and the node path never diverge.

``torchrun`` supplies the rank environment; :func:`initialize` joins the
process group and warms the communicator so its buffers exist before the
trainer reads the device's free memory. Every collective here is in place
on a contiguous tensor or a contiguous leading-dimension chunk of one, which
is what the parameter sites allocate.
"""

from __future__ import annotations

import os
import signal
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor
from transformer_experiments.spool import STOP_GRACE


@dataclass(frozen=True)
class Topology:
    """This process's place among the ranks of one invocation."""

    rank: int = 0
    world: int = 1
    local_rank: int = 0

    @classmethod
    def from_environment(cls) -> Topology:
        """The rank ``torchrun`` assigned, or the lone rank without it."""
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world < 1:
            raise ValueError(f"WORLD_SIZE must be positive, got {world}")
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        if not 0 <= rank < world:
            raise ValueError(f"RANK {rank} is outside WORLD_SIZE {world}")
        return cls(rank, world, local_rank)

    @property
    def main(self) -> bool:
        """Whether this rank logs, evaluates the monitors, and writes snapshots."""
        return self.rank == 0

    def chunk(self, count: int) -> int:
        """Entries per rank of a leading dimension padded to a rank multiple."""
        return -(-count // self.world)

    def padded(self, count: int) -> int:
        return self.chunk(count) * self.world


LAUNCHER = "torch.distributed.run"
LAUNCHER_SHUTDOWN_SECONDS = int(STOP_GRACE) - 10
"""Seconds the launcher gives its ranks to finish their step, snapshot, and
exit after it forwards a stop signal, before it kills them. The launcher's
default is 30, shorter than a deep step with its snapshot; this stays inside
the spool's grace so the launcher reaps its own ranks and the spool never has
to reach past it."""


def relaunch(module: str, argv: list[str], ranks: int) -> None:
    """Replace this process with the launcher when several ranks are asked for.

    Returns at once under the launcher or for one rank; otherwise ``exec``
    never returns. The launcher starts one process per device running
    ``module`` with the same arguments, keeps the process id the spool
    started, forwards a stop signal to every rank, and waits
    :data:`LAUNCHER_SHUTDOWN_SECONDS` for them.
    """
    if ranks <= 1 or "WORLD_SIZE" in os.environ:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(
        sys.executable,
        [
            sys.executable, "-m", LAUNCHER, "--standalone",
            f"--nproc-per-node={ranks}",
            f"--shutdown-timeout={LAUNCHER_SHUTDOWN_SECONDS}",
            "-m", module, *argv,
        ],
    )


def initialize(topology: Topology, device: torch.device) -> None:
    """Join the process group of a multi-rank invocation.

    NCCL serves CUDA ranks and gloo everything else. One trivial all-reduce
    afterwards makes the communicator allocate its buffers now, so the free
    memory the trainer measures for its activation budget already excludes
    them.
    """
    if topology.world == 1:
        return
    if dist.is_initialized():
        return
    backend = "nccl" if device.type == "cuda" else "gloo"
    kwargs = {"device_id": device} if device.type == "cuda" else {}
    dist.init_process_group(
        backend, rank=topology.rank, world_size=topology.world, **kwargs
    )
    all_reduce_(torch.zeros(1, device=device))


def shutdown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def active() -> bool:
    return dist.is_initialized() and dist.get_world_size() > 1


# -- in-place collectives ------------------------------------------------------


def all_reduce_(tensor: Tensor, *, maximum: bool = False) -> Tensor:
    """Sum (or take the maximum of) ``tensor`` across ranks, in place."""
    if active():
        op = dist.ReduceOp.MAX if maximum else dist.ReduceOp.SUM
        dist.all_reduce(tensor, op=op)
    return tensor


def reduce_(tensor: Tensor, owner: int) -> Tensor:
    """Sum ``tensor`` across ranks onto ``owner``, in place."""
    if active():
        dist.reduce(tensor, dst=owner)
    return tensor


def broadcast_(tensor: Tensor, owner: int) -> Tensor:
    if active():
        dist.broadcast(tensor, src=owner)
    return tensor


def reduce_scatter_(slab: Tensor, rank: int, chunk: int) -> Tensor:
    """Sum ``slab`` across ranks, leaving rank ``r`` its ``r``-th leading chunk.

    The output is a view of the input at this rank's own chunk, which is the
    in-place layout NCCL and gloo both accept; the other chunks keep this
    rank's unreduced contribution and are never read again.
    """
    if active():
        own = slab.narrow(0, rank * chunk, chunk)
        dist.reduce_scatter_single(own, slab)
    return slab


def all_gather_(slab: Tensor, rank: int, chunk: int) -> Tensor:
    """Fill every rank's ``slab`` from each rank's own leading chunk, in place."""
    if active():
        own = slab.narrow(0, rank * chunk, chunk)
        dist.all_gather_single(slab, own)
    return slab


def barrier() -> None:
    """Hold every rank here until all of them have arrived."""
    if active():
        dist.barrier()


def agree(flag: bool, device: torch.device) -> bool:
    """Whether any rank raised ``flag``; identical on every rank."""
    if not active():
        return flag
    signal_tensor = torch.tensor([int(flag)], device=device)
    all_reduce_(signal_tensor, maximum=True)
    return bool(signal_tensor.item())


def collect(
    tensor: Tensor | None, owner: int, topology: Topology, *, shape, dtype, device
) -> Tensor | None:
    """Bring one owner-held tensor to rank zero.

    The owner passes its tensor; every other rank passes ``None``. Rank zero
    receives a host copy from a remote owner, or the tensor itself when it is
    the owner, so a local tensor stays where the checkpoint stager can copy
    it asynchronously. Other ranks receive nothing.
    """
    if topology.rank == owner:
        assert tensor is not None
        if owner == 0:
            return tensor
        dist.send(tensor.contiguous(), dst=0)
        return None
    if topology.rank == 0:
        staging = torch.empty(shape, dtype=dtype, device=device)
        dist.recv(staging, src=owner)
        return staging.cpu()
    return None


# -- cooperative interruption -------------------------------------------------


class Interrupt:
    """Turn SIGINT into a flag the step loop reads at a safe point.

    Every rank must leave the loop at the same step, snapshot the same state
    and enter the same collectives, so a signal only raises a flag here and
    :meth:`agreed` lets the ranks settle it together once the step is done.
    A second signal raises ``KeyboardInterrupt`` immediately for an operator
    who does not want to wait for the step.
    """

    def __init__(self) -> None:
        self.requested = False
        self._previous = None

    def _handle(self, signum, frame) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True

    def __enter__(self) -> Interrupt:
        self._previous = signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        signal.signal(signal.SIGINT, self._previous)

    def agreed(self, device: torch.device) -> bool:
        return agree(self.requested, device)

"""One condition's training run under the binding recipe.

The loop is a pure function of (init seed, data seed, step): data rows
are step-addressed slices of the fixed stream, feedback randomness
(pass counts, prefix lengths, jitter) derives from keyed generators
rather than ambient RNG state, and the WSD schedule is the shared
:class:`Schedule` addressed by cumulative step.  That is what makes a
resumed invocation bit-identical to an uninterrupted one and every
condition's batches the same bytes (the paired-comparison contract; paired
runs must share ``data_seed``, ``batch_rows``, and ``micro_rows``).

Operational layer — telemetry records, immutable ``runs``-addressed
snapshots, resume reconciliation — comes from ``transformer_experiments``.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import math
from types import SimpleNamespace
from fractions import Fraction
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from transformer_experiments import checkpoints, runs, telemetry
from transformer_experiments.schedule import Schedule

from .data import TokenData, read_meta
from .model import (
    CONDITION_LETTERS,
    DeltaModel,
    condition_config,
    depth_trace,
    iterate_fused,
    multipass,
    multipass_loss,
    parse_condition,
)
from .optim import (
    DEFAULT_NADAM_LR,
    DEFAULT_NORMUONH_LR,
    OptimizerPair,
    apply_schedule,
    build_optimizers,
)

CONTRACT = checkpoints.CheckpointContract(
    version=26, resumable=frozenset({26}), surface_version=26
)

GRAD_CLIP_NORM = 10.0
"""Global FP32 gradient-norm ceiling shared by every run."""

BATCH_TOKENS = 524_288
"""Predicted tokens per optimizer step at every scale: 2^19, 128 rows of 4,096."""

SCALES: dict[str, dict[str, int]] = {
    "screen": dict(
        dim=768, layers=12, heads=8, kv_heads=4, intermediate=3328, pkda_heads=10,
        seq_len=4096, batch_rows=128, micro_rows=1,
    ),
    "bridge": dict(
        dim=1152, layers=16, heads=12, kv_heads=6, intermediate=4992, pkda_heads=15,
        seq_len=4096, batch_rows=128, micro_rows=1,
    ),
    "flagship": dict(
        dim=1536, layers=24, heads=16, kv_heads=8, intermediate=6656, pkda_heads=20,
        seq_len=4096, batch_rows=128, micro_rows=1,
    ),
}
"""The geometries of ``docs/scaling.md``: the column, and the row length,
rows per step, and single-process microbatch every scale shares, 4,096-token
rows in one-row microbatches with ``BATCH_TOKENS`` predictions per step."""

DEFAULT_TOKENS_PER_PARAM = 25.0
"""The screen recipe's predicted tokens per reference active parameter."""

EXACT_FIELDS = (
    "condition",
    "seed",
    "data_seed",
    "seq_len",
    "batch_rows",
    "micro_rows",
    "steps",
    "warmup_frac",
    "cooldown_frac",
    "feedback_start",
    "three_pass",
    "loop_iterations",
    "loop_max_iterations",
    "lr_normuonh",
    "lr_nadam",
    "jitter",
    "zloss",
    "vocab_size",
    "dim",
    "layers",
    "heads",
    "kv_heads",
    "head_dim",
    "intermediate",
    "pkda_heads",
    "pkda_head_dim",
    "pkda_conv_size",
)
"""State-defining settings: a resume takes these from the checkpoint."""

RUNTIME_FIELDS = (
    "data_dir",
    "out_dir",
    "device",
    "eval_every",
    "snapshot_every",
    "eval_rows",
    "head_flush_every",
)
"""Per-invocation settings: inherited unless retyped."""


def probability(value: str) -> float:
    """An argparse probability in the closed unit interval."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid probability: {value!r}") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(
            f"probability must be finite and in [0, 1], got {value!r}"
        )
    return parsed


def condition(value: str) -> str:
    try:
        return parse_condition(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "delta train", description="Train one condition of the delta family."
    )
    parser.add_argument("tag", type=runs.validate_run_tag, help="run tag")
    parser.add_argument(
        "--condition",
        type=condition,
        default="",
        metavar="LETTERS",
        help=(
            "one letter per change from the plain gated GQA decoder, in any order "
            "(default: none, the plain decoder): "
            + "; ".join(
                f"{letter} = {change}"
                for letter, (_, change) in CONDITION_LETTERS.items()
            )
        ),
    )
    parser.add_argument("--seed", type=int, default=1, help="init seed")
    parser.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="shared randomness stream; identical across paired conditions",
    )
    parser.add_argument("--data-dir", default="data/tokens")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the tag from its latest snapshot",
    )
    parser.add_argument(
        "--continue",
        dest="continue_from",
        metavar="TAG",
        default=None,
        help=(
            "extend a finished run under this tag: restore TAG's last snapshot "
            "that the longer schedule reproduces and train on, every setting "
            "but the schedule length inherited from it"
        ),
    )

    scale = parser.add_argument_group("scale")
    scale.add_argument(
        "--scale",
        choices=tuple(SCALES),
        default="screen",
        help=(
            "geometry and batch preset from docs/scaling.md; a trunk or recipe "
            "flag typed alongside overrides its field (default: screen)"
        ),
    )
    scale.add_argument(
        "--tokens-per-param",
        type=float,
        default=DEFAULT_TOKENS_PER_PARAM,
        metavar="RATIO",
        help=(
            "predicted tokens per active non-embedding parameter of the flat "
            "full stack at this scale; derives --steps, rounded up to whole "
            "steps, unless --steps is given (default: 25; the Prime recipes "
            "use 400)"
        ),
    )

    schedule = parser.add_argument_group("schedule (state-defining)")
    schedule.add_argument(
        "--steps",
        type=runs.parse_step_count,
        default=None,
        help="schedule length; derived from --tokens-per-param unless given",
    )
    schedule.add_argument(
        "--warmup-frac",
        type=probability,
        default=0.02,
        help="fraction of total steps used for linear learning-rate warmup",
    )
    schedule.add_argument(
        "--cooldown-frac",
        type=probability,
        default=0.2,
        help="fraction of total steps used for 1-sqrt learning-rate cooldown",
    )
    schedule.add_argument(
        "--feedback-start",
        type=probability,
        default=0.75,
        help=(
            "fraction of steps before feedback passes begin; every later step "
            "draws two or three passes"
        ),
    )
    schedule.add_argument(
        "--three-pass",
        type=probability,
        default=0.12,
        help="P(k = 3) on each feedback-phase step; otherwise k = 2",
    )

    recipe = parser.add_argument_group("recipe (state-defining)")
    recipe.add_argument(
        "--batch-rows",
        type=int,
        default=128,
        help=(
            "global batch in rows; the scale keeps 524,288 predictions per "
            "step, as does a retyped --seq-len without this flag"
        ),
    )
    recipe.add_argument("--micro-rows", type=int, default=1)
    recipe.add_argument("--seq-len", type=int, default=4096)
    recipe.add_argument(
        "--lr-normuonh",
        type=float,
        default=DEFAULT_NORMUONH_LR,
        help="dimensionless NorMuonH relative step (default: 0.006)",
    )
    recipe.add_argument(
        "--lr-nadam",
        type=float,
        default=DEFAULT_NADAM_LR,
        help=(
            "NAdam learning rate at the muP reference width 1536; the fan-in-D "
            "NAdam matrices run at this times 1536/dim (default: 0.0003)"
        ),
    )
    recipe.add_argument("--jitter", type=float, default=0.02)
    recipe.add_argument("--zloss", type=float, default=1e-5)
    recipe.add_argument(
        "--loop-iterations",
        type=int,
        default=4,
        help=(
            "l: mean core iterations per column, drawn once per step; also the "
            "fixed count evaluation and decoding use"
        ),
    )
    recipe.add_argument(
        "--loop-max-iterations",
        type=int,
        default=8,
        help="l: cap of the per-step iteration draw",
    )

    trunk = parser.add_argument_group("trunk (state-defining)")
    trunk.add_argument("--vocab-size", type=int, default=151936)
    trunk.add_argument("--dim", type=int, default=768)
    trunk.add_argument("--layers", type=int, default=12)
    trunk.add_argument("--heads", type=int, default=8)
    trunk.add_argument("--kv-heads", type=int, default=4)
    trunk.add_argument("--head-dim", type=int, default=96)
    trunk.add_argument("--intermediate", type=int, default=3328)
    trunk.add_argument("--pkda-heads", type=int, default=10)
    trunk.add_argument("--pkda-head-dim", type=int, default=128)
    trunk.add_argument("--pkda-conv-size", type=int, default=4)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--max-steps",
        type=runs.parse_step_count,
        default=None,
        help="cap this invocation's additional steps; never rescales the schedule",
    )
    runtime.add_argument("--device", default=None)
    runtime.add_argument("--eval-every", type=int, default=250)
    runtime.add_argument("--snapshot-every", type=int, default=500)
    runtime.add_argument(
        "--eval-rows",
        type=int,
        default=128,
        help="held-out rows per evaluation, from the slice's head; 128 rows of "
        "4,096 is one optimizer batch of predictions",
    )
    runtime.add_argument(
        "--head-flush-every",
        type=int,
        default=4,
        help="head calls -- one per feedback pass per microbatch -- whose "
        "classifier gradient accumulates in BF16 before it is flushed into "
        "the FP32 embedding sink; 1 is the per-call path exactly, and wider "
        "windows trade the head's gradient precision for the flush's bandwidth",
    )
    return parser


# -- deterministic derived randomness ------------------------------------------


def mix(*parts: int) -> int:
    """A stable 63-bit mix of integers (no salted ``hash``)."""
    value = 0
    for part in parts:
        value = (value ^ (part + 0x9E3779B97F4A7C15)) * 0xBF58476D1CE4E5B9 % (1 << 63)
        value ^= value >> 27
    return value


def feedback_boundary(args, total: int) -> int:
    """Last one-pass step; feedback conditions draw k > 1 on every later step."""
    return round(args.feedback_start * total)


def continuation_step(source, source_schedule: Schedule, args, schedule: Schedule) -> int:
    """The last step of a finished run that a longer schedule reproduces.

    Up to it every step was warmup or heat under both schedules and on the
    same side of both feedback boundaries, so a run restored from its snapshot
    there and trained on under the longer schedule is the longer run from that
    step on, apart from the warmup it inherited. When the feedback boundary
    moves that is the boundary itself, the last one-pass state; otherwise it
    is the cooldown boundary.
    """
    fork = min(source_schedule.heat_end, schedule.heat_end)
    old = feedback_boundary(source, source_schedule.total)
    new = feedback_boundary(args, schedule.total)
    if old != new:
        fork = min(fork, old, new)
    return fork


def draw_passes(args, step: int, total: int) -> int:
    """The step's pass count — shared across every condition with ``f``.

    Before the boundary every step is one pass. After it there are no
    one-pass steps at all: each step draws three passes with probability
    ``three_pass`` and two passes otherwise, so the fused mode is trained
    on every update rather than eroded between them.
    """
    if step <= feedback_boundary(args, total):
        return 1
    generator = torch.Generator().manual_seed(mix(args.data_seed, step, 1))
    draw = torch.rand((), generator=generator).item()
    return 3 if draw < args.three_pass else 2


LOOP_DRAW_SIGMA = 0.5
"""Log-normal spread of the recurrent-depth iteration draw."""


def draw_iterations(args, step: int, loop: bool) -> int:
    """The step's core iteration count under ``l``, shared by every pass and
    microbatch of the step.

    The recurrent-depth log-normal Poisson draw with its rate shifted by one::

        tau ~ Normal(log(r_mean - 1) - sigma^2 / 2, sigma)
        r   = min(1 + Poisson(exp(tau)), r_max)

    so the uncapped mean of ``r`` is ``r_mean``. The draw has its own keyed
    sub-stream, so the pass, prefix, and jitter draws stay identical to the
    condition without ``l``.
    """
    if not loop:
        return 1
    mean, cap = args.loop_iterations, args.loop_max_iterations
    if mean <= 1:
        return 1
    generator = torch.Generator().manual_seed(mix(args.data_seed, 0x6C6F6F70, step))
    tau = torch.normal(
        math.log(mean - 1) - LOOP_DRAW_SIGMA**2 / 2,
        LOOP_DRAW_SIGMA,
        size=(1,),
        generator=generator,
    )
    count = torch.poisson(tau.exp(), generator=generator)
    return min(1 + int(count.item()), cap)


def micro_draws(
    args,
    step: int,
    first_row: int,
    n_passes: int,
    n_rows: int,
    dim: int,
    device,
    *,
    prefix_out: torch.Tensor | None = None,
    jitter_out: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(prefix_lens [k-1, n], jitter [k-1, n, seq_len+1, dim]) for one
    microbatch, keyed by (data seed, step, first global row) — identical
    across conditions for any run sharing the batch geometry."""
    if generator is None:
        generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(mix(args.data_seed, step, first_row))
    columns = args.seq_len + 1
    shape = (n_passes - 1, n_rows)
    if prefix_out is None:
        prefix_out = torch.empty(shape, dtype=torch.long, device=device)
    # Plain-prefix lengths in 1..seq_len-1: position 0 is always plain and
    # every row keeps at least one fused position.
    torch.randint(1, args.seq_len, shape, generator=generator, out=prefix_out)
    jitter_shape = (n_passes - 1, n_rows, columns, dim)
    if jitter_out is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        jitter_out = torch.empty(jitter_shape, dtype=dtype, device=device)
    jitter_out.uniform_(-args.jitter, args.jitter, generator=generator)
    return prefix_out, jitter_out


def automatic_checkpoint(
    model: DeltaModel, n_passes: int, iterations: int, args, device
) -> bool:
    """Measured internal activation policy for the screen."""
    if device.type != "cuda":
        return False
    cfg = model.cfg
    # Ten cell-passes of the screen geometry fit the 24 GiB card raw: the whole
    # one-pass loop family through r = 8 captured at 14.6 GiB allocated, and
    # its r = 8 replay is 154 ms raw against 199 ms recomputing every block.
    # Every deeper mode (two passes above r = 3, three passes above r = 1)
    # still checkpoints; the three-pass family raw ran out of memory. Measured
    # at four 1,024-token rows per microbatch, the same 4,096 tokens as the
    # one-row microbatch that replaced them.
    raw_work = 4096 * 768 * 40
    work = (
        args.micro_rows
        * args.seq_len
        * cfg.dim
        * cfg.executed_layers(iterations)
        * n_passes
    )
    return work > raw_work


@dataclass(frozen=True)
class GraphSpec:
    n_passes: int
    checkpoint: bool
    iterations: int = 1


@dataclass
class CapturedMicro:
    spec: GraphSpec
    rows: torch.Tensor
    prefix: torch.Tensor | None
    jitter: torch.Tensor | None
    z_coef: torch.Tensor
    loss_sum: torch.Tensor
    pass1_sum: torch.Tensor
    graph: torch.cuda.CUDAGraph | None = None
    active: frozenset[torch.nn.Parameter] = frozenset()


class CudaBatchStager:
    """Reuse one pinned host batch and one device batch on the replay stream.

    The copy event protects host storage from being overwritten while DMA is
    still reading it. Device copies and graph replays use the same stream, so
    the next batch cannot overwrite tokens still consumed by the prior one.
    """

    def __init__(self, rows: int, columns: int, device: torch.device):
        self.host = torch.empty(rows, columns, dtype=torch.long, pin_memory=True)
        self.device = torch.empty(rows, columns, dtype=torch.long, device=device)
        self.copied = torch.cuda.Event()
        self.pending = False

    def stage(self, data: TokenData, first_row: int) -> torch.Tensor:
        if self.pending:
            self.copied.synchronize()
        self.host.copy_(data.batch(first_row, self.host.shape[0]))
        self.device.copy_(self.host, non_blocking=True)
        self.copied.record()
        self.pending = True
        return self.device


@contextlib.contextmanager
def _capture_without_gc(graph, pool):
    """Keep cyclic CUDA resource destruction outside stream capture."""
    # PyTorch 2.14 no longer unconditionally collects warm-up cycles before
    # capture. A later Python collection can free their CUDA resources and
    # invalidate the active stream capture, even inside an unrelated kernel.
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.graph(graph, pool=pool):
            yield
    finally:
        if was_enabled:
            gc.enable()


class CudaGraphTrainer:
    """Fixed-address forward/backward graphs for every reachable step mode.

    The model's gradient tensors and every graph input are allocated once.
    Data and keyed CUDA randomness are copied/drawn into those addresses before
    replay.  Graphs share a private pool and never overlap; each captured body
    ends after backward, so no saved activation survives between replays.

    Every persistent FP32 gradient buffer exists before the first forward.  The
    tied embedding and the large projections accumulate into their buffers in
    place from inside backward and hand autograd no gradient, so a buffer that
    warm-up touched marks its parameter active for that mode exactly as an
    autograd gradient marks the remaining vectors and small matrices.

    The head is the one site whose gradient does not reach its FP32 sink on
    every microbatch: every head call lock-adds into one persistent BF16
    classifier buffer, and the buffer is added into the sink and cleared
    whenever another replay would take it past ``head_flush_every`` head calls,
    plus once more before the optimizer reads the step's gradient.
    """

    def __init__(self, model: DeltaModel, optimizers, args, schedule: Schedule):
        self.model = model
        self.optimizers = optimizers
        self.args = args
        self.device = next(model.parameters()).device
        self.micros = args.batch_rows // args.micro_rows
        self.autocast = torch.autocast("cuda", dtype=torch.bfloat16)
        self.generator = torch.Generator(device=self.device)
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.batch_stager = CudaBatchStager(
            args.batch_rows, args.seq_len + 1, self.device
        )
        self.states: dict[GraphSpec, CapturedMicro] = {}
        self.model.refresh_shadows()
        specs = self._reachable_specs(schedule)
        for spec in specs:
            self.states[spec] = self._allocate(spec)
        self._buffers = {p: torch.zeros_like(p) for p in self.parameters}
        self.sink_fed = model.bind_gradient_sinks(self._buffers)

        # The head's classifier gradient lands in one persistent BF16 buffer
        # instead of a fresh 233 MB zero tensor per call, and reaches the FP32
        # sink once every ``head_flush_every`` head calls.  It exists and is
        # zeroed before warm-up traces the head, carries no autograd history,
        # and keeps its address through capture.
        self.head_flush_every = args.head_flush_every
        self._head_pending = 0
        self.head_accum = None
        if model.embed_tokens.grad_sink is not None:
            self.head_accum = torch.zeros_like(model._classifier_shadow)
            model.bind_classifier_accum(self.head_accum)

        # The package sets Dynamo's recompile budget once for the process, so
        # every block and router specialization compiles here and in later
        # eager evaluation alike.
        active_by_spec = {
            spec: self._warm(state) for spec, state in self.states.items()
        }
        model.zero_grad(set_to_none=True)
        union = set().union(*active_by_spec.values())
        self.grad_buffers = {p: self._buffers[p] for p in self.parameters if p in union}
        del self._buffers
        # Rebinding drops the sinks of parameters no reachable mode ever touches.
        self.sink_fed = model.bind_gradient_sinks(self.grad_buffers)
        # A dropped embedding sink takes the head's accumulator with it.
        self.head_accum = model._classifier_accum
        for parameter, gradient in self.grad_buffers.items():
            parameter.grad = gradient

        self._initialize_optimizers()
        self.model.refresh_shadows()
        for active in active_by_spec.values():
            for optimizer in self.optimizers:
                warmup = getattr(optimizer, "warmup", None)
                if warmup is not None:
                    warmup(active)
        self.zero_grad()
        torch.cuda.empty_cache()

        self.pool = torch.cuda.graph_pool_handle()
        for spec, state in self.states.items():
            state.active = frozenset(active_by_spec[spec])
            self._capture(state, self.pool)
        self.zero_grad()
        torch.cuda.synchronize()

    def _reachable_specs(self, schedule: Schedule) -> list[GraphSpec]:
        specs = set()
        for step in range(1, schedule.total + 1):
            n_passes = (
                draw_passes(self.args, step, schedule.total)
                if self.model.cfg.feedback_active
                else 1
            )
            iterations = draw_iterations(self.args, step, self.model.cfg.loop)
            specs.add(
                GraphSpec(
                    n_passes,
                    automatic_checkpoint(
                        self.model, n_passes, iterations, self.args, self.device
                    ),
                    iterations,
                )
            )
        return sorted(specs, key=lambda spec: (spec.n_passes, spec.iterations))

    def _allocate(self, spec: GraphSpec) -> CapturedMicro:
        rows = torch.zeros(
            self.args.micro_rows,
            self.args.seq_len + 1,
            dtype=torch.long,
            device=self.device,
        )
        prefix = jitter = None
        if spec.n_passes > 1:
            prefix = torch.ones(
                spec.n_passes - 1,
                self.args.micro_rows,
                dtype=torch.long,
                device=self.device,
            )
            jitter = torch.zeros(
                spec.n_passes - 1,
                self.args.micro_rows,
                self.args.seq_len + 1,
                self.args.dim,
                dtype=torch.bfloat16,
                device=self.device,
            )
        return CapturedMicro(
            spec,
            rows,
            prefix,
            jitter,
            torch.zeros((), dtype=torch.float32, device=self.device),
            torch.zeros((), dtype=torch.float32, device=self.device),
            torch.zeros((), dtype=torch.float32, device=self.device),
        )

    def _body(self, state: CapturedMicro) -> None:
        self.model.grad_checkpoint = state.spec.checkpoint
        with self.autocast:
            outs = multipass(
                self.model,
                state.rows,
                state.spec.n_passes,
                prefix_lens=state.prefix,
                jitter=state.jitter,
                iterations=state.spec.iterations,
            )
            loss, losses = multipass_loss(
                self.model, state.rows, outs, z_coef=state.z_coef
            )
        (loss / self.micros).backward()
        state.loss_sum.add_(loss.detach() / self.micros)
        state.pass1_sum.add_(losses[0].detach() / self.micros)

    def _drain_head_accum(self) -> None:
        """Add the head's accumulated BF16 gradient into its FP32 sink.

        Eager work between replays, never inside a captured body: the buffer's
        contents are the only thing that crosses, its address does not move,
        and the sink is the same one every other bound site accumulates into.
        """
        self.model.embed_tokens.grad_sink.add_(self.head_accum)
        self.head_accum.zero_()
        self._head_pending = 0

    def flush_head_accum(self) -> None:
        """Drain the head accumulator when a replay has written into it."""
        if self.head_accum is not None and self._head_pending:
            self._drain_head_accum()

    def _warm(self, state: CapturedMicro) -> set[torch.nn.Parameter]:
        for parameter in self.sink_fed:
            self._buffers[parameter].zero_()
        if self.head_accum is not None:
            self.head_accum.zero_()
            self._head_pending = 0
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.model.zero_grad(set_to_none=True)
                self._body(state)
                # Warm-up decides which parameters a mode touches from its
                # buffers, so the head's gradient has to reach the sink here
                # exactly as a replayed step's does.
                if self.head_accum is not None:
                    self._drain_head_accum()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        active = {p for p in self.parameters if p.grad is not None}
        active |= {p for p in self.sink_fed if bool(self._buffers[p].any())}
        return active

    def _capture(self, state: CapturedMicro, pool) -> None:
        for gradient in self.grad_buffers.values():
            gradient.zero_()
        if self.head_accum is not None:
            # Capture records the head's accumulation into this buffer; the
            # zeroing stays outside the body, where every replay leaves it.
            self.head_accum.zero_()
            self._head_pending = 0
        state.loss_sum.zero_()
        state.pass1_sum.zero_()
        # Capturing on a blocking stream cannot inherit unfinished
        # default-stream writes from optimizer/kernel preparation.
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with _capture_without_gc(graph, pool):
            self._body(state)
        state.graph = graph
        state.loss_sum.zero_()
        state.pass1_sum.zero_()

    def _initialize_optimizers(self) -> None:
        """Materialize persistent state before the graph-private pool grows."""
        if any(optimizer.state for optimizer in self.optimizers):
            # Resume already restored state; only warm the compiled NorMuonH body.
            for optimizer in self.optimizers:
                warmup = getattr(optimizer, "warmup", None)
                if warmup is not None:
                    warmup()
            return

        saved = [group["lr"] for opt in self.optimizers for group in opt.param_groups]
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = 0.0
            optimizer.step()
        for optimizer in self.optimizers:
            for values in optimizer.state.values():
                for name, value in values.items():
                    if not isinstance(value, torch.Tensor) or name == "radius":
                        continue
                    if name == "mu_product":
                        value.fill_(1)
                    else:
                        value.zero_()
        for rate, group in zip(
            saved,
            (group for opt in self.optimizers for group in opt.param_groups),
            strict=True,
        ):
            group["lr"] = rate

    def begin(self, spec: GraphSpec, z_coef: float = 0.0) -> CapturedMicro:
        state = self.states[spec]
        state.z_coef.fill_(z_coef)
        state.loss_sum.zero_()
        state.pass1_sum.zero_()
        return state

    def replay(self, state: CapturedMicro, rows: torch.Tensor, step: int, first: int):
        state.rows.copy_(rows, non_blocking=rows.is_cuda)
        if state.spec.n_passes > 1:
            micro_draws(
                self.args,
                step,
                first,
                state.spec.n_passes,
                self.args.micro_rows,
                self.args.dim,
                self.device,
                prefix_out=state.prefix,
                jitter_out=state.jitter,
                generator=self.generator,
            )
        state.graph.replay()
        if self.head_accum is not None:
            # The window counts head calls, not replays: a k-pass microbatch
            # calls the head k times and every call rounds the running BF16
            # sum, so the cadence has to hold the number of contributions
            # between flushes fixed across conditions. Draining before the
            # next replay would overrun keeps that number at or under the
            # window even when it is not a multiple of the pass count.
            self._head_pending += state.spec.n_passes
            if self._head_pending + state.spec.n_passes > self.head_flush_every:
                self._drain_head_accum()

    def replay_batch(
        self, state: CapturedMicro, data: TokenData, step: int, first_row: int
    ) -> None:
        rows = self.batch_stager.stage(data, first_row)
        for offset in range(0, self.args.batch_rows, self.args.micro_rows):
            self.replay(
                state,
                rows[offset : offset + self.args.micro_rows],
                step,
                first_row + offset,
            )
        # A microbatch count that is not a multiple of the flush cadence leaves
        # a remainder; the step's gradient is complete only once it is drained.
        self.flush_head_accum()

    def prepare_optimizer(self, state: CapturedMicro) -> None:
        self.flush_head_accum()
        for parameter in self.parameters:
            parameter.grad = (
                self.grad_buffers.get(parameter) if parameter in state.active else None
            )

    def zero_grad(self) -> None:
        if self.head_accum is not None:
            self.head_accum.zero_()
            self._head_pending = 0
        for parameter in self.parameters:
            gradient = self.grad_buffers.get(parameter)
            parameter.grad = gradient
            if gradient is not None:
                gradient.zero_()


# -- evaluation ----------------------------------------------------------------


@dataclass
class CapturedEval:
    rows: torch.Tensor
    prefix: torch.Tensor | None
    val_sum: torch.Tensor
    fused_sum: torch.Tensor
    graph: torch.cuda.CUDAGraph | None = None


class CudaEvalRunner:
    """No-grad validation graphs sharing the trainer's private memory pool."""

    def __init__(self, model: DeltaModel, args, pool):
        self.model = model
        self.args = args
        self.device = next(model.parameters()).device
        self.autocast = torch.autocast("cuda", dtype=torch.bfloat16)
        remainder = args.eval_rows % args.micro_rows
        sizes = {min(args.eval_rows, args.micro_rows)}
        if remainder:
            sizes.add(remainder)
        self.states = {size: self._capture(size, pool) for size in sorted(sizes)}

    def _body(self, state: CapturedEval) -> None:
        n_passes = 2 if self.model.cfg.feedback_active else 1
        with self.autocast:
            outs = multipass(
                self.model,
                state.rows,
                n_passes,
                prefix_lens=state.prefix,
            )
            _, losses = multipass_loss(self.model, state.rows, outs)
        rows = state.rows.shape[0]
        state.val_sum.add_(losses[0].detach() * rows)
        if n_passes > 1:
            state.fused_sum.add_(losses[1].detach() * rows)

    @torch.no_grad()
    def _capture(self, rows: int, pool) -> CapturedEval:
        state = CapturedEval(
            torch.zeros(
                rows,
                self.args.seq_len + 1,
                dtype=torch.long,
                device=self.device,
            ),
            (
                torch.ones((1, rows), dtype=torch.long, device=self.device)
                if self.model.cfg.feedback_active
                else None
            ),
            torch.zeros((), dtype=torch.float32, device=self.device),
            torch.zeros((), dtype=torch.float32, device=self.device),
        )
        was_training = self.model.training
        self.model.eval()
        self.model.grad_checkpoint = False
        for _ in range(2):
            state.val_sum.zero_()
            state.fused_sum.zero_()
            self._body(state)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        state.val_sum.zero_()
        state.fused_sum.zero_()
        with _capture_without_gc(graph, pool):
            self._body(state)
        state.graph = graph
        state.val_sum.zero_()
        state.fused_sum.zero_()
        self.model.train(was_training)
        return state

    @torch.no_grad()
    def run(self, data_val: TokenData) -> dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        total_val = torch.zeros((), dtype=torch.float32, device=self.device)
        total_fused = torch.zeros_like(total_val)
        for first in range(0, self.args.eval_rows, self.args.micro_rows):
            rows = min(self.args.micro_rows, self.args.eval_rows - first)
            state = self.states[rows]
            state.val_sum.zero_()
            state.fused_sum.zero_()
            state.rows.copy_(data_val.batch(first, rows))
            state.graph.replay()
            total_val.add_(state.val_sum)
            if self.model.cfg.feedback_active:
                total_fused.add_(state.fused_sum)
        result = {"val": (total_val / self.args.eval_rows).item()}
        if self.model.cfg.feedback_active:
            result["val_fused"] = (total_fused / self.args.eval_rows).item()
        self.model.train(was_training)
        return result


def execution_fields(model, graph_runner, eval_graph_runner) -> dict[str, int]:
    """Production CUDA telemetry without assuming a packed mixer layout."""
    has_global_attention = any(not block.is_pkda for block in model.blocks)
    return {
        "flex": int(has_global_attention),
        "cce": 1,
        "cuda_graphs": len(graph_runner.states) + len(eval_graph_runner.states),
        "eval_graphs": len(eval_graph_runner.states),
        "checkpoint_modes": sum(spec.checkpoint for spec in graph_runner.states),
        "head_flush_every": graph_runner.head_flush_every,
    }


@torch.no_grad()
def evaluate(
    model: DeltaModel,
    data_val: TokenData,
    args,
    device,
    graph_runner: CudaEvalRunner | None = None,
) -> dict[str, float]:
    """Paired val losses: pass-1 always; one fused pass on feedback conditions."""
    if graph_runner is not None:
        return graph_runner.run(data_val)
    model.eval()
    sums = {}
    counted = 0
    for first in range(0, args.eval_rows, args.micro_rows):
        rows = data_val.batch(
            first, min(args.micro_rows, args.eval_rows - first), device
        )
        n_passes = 2 if model.cfg.feedback_active else 1
        prefix = torch.ones((1, rows.shape[0]), dtype=torch.long, device=device)
        outs = multipass(
            model, rows, n_passes, prefix_lens=prefix if n_passes > 1 else None
        )
        _, losses = multipass_loss(model, rows, outs)
        sums["val"] = sums.get("val", 0.0) + losses[0].item() * rows.shape[0]
        if n_passes > 1:
            sums["val_fused"] = (
                sums.get("val_fused", 0.0) + losses[1].item() * rows.shape[0]
            )
        counted += rows.shape[0]
    model.train()
    return {key: value / counted for key, value in sums.items()}


@torch.no_grad()
def route_summary(model: DeltaModel, data_val: TokenData, args, device) -> list[dict]:
    """Per-site routing observables from one validation microbatch."""
    if not model.cfg.routing_active:
        return []
    model.eval()
    rows = data_val.batch(0, min(2, args.eval_rows), device)
    if model.cfg.feedback_active:
        prefix = torch.ones((1, rows.shape[0]), dtype=torch.long, device=device)
        out = multipass(model, rows, 2, prefix_lens=prefix, want_weights=True)[-1]
    else:
        out = multipass(model, rows, 1, want_weights=True)[0]
    records = []
    for site, weights in out.route_weights.items():
        w = weights.float()
        mean = w.mean(dim=(1, 2, 3))
        head_mean = w.mean(dim=-1, keepdim=True)
        head_js = (
            (w * (w.clamp_min(1e-12).log() - head_mean.clamp_min(1e-12).log()))
            .sum(dim=0)
            .mean()
        )
        head_js = head_js / math.log(min(w.shape[0], w.shape[-1]))
        record = {
            "site": site,
            "n": weights.shape[0],
            "heads": weights.shape[-1],
            "max": round(w.max(dim=0).values.mean().item(), 4),
            "head_js": round(head_js.item(), 4),
        }
        names = out.route_source_names[site]
        for label in ("null", "prev", "seed"):
            if label in names:
                record[label] = round(mean[names.index(label)].item(), 4)
        if site == "payload":
            router = model.payload_router
        else:
            # A core site under l is tagged by iteration: ``L4i2.attn``.
            layer_kind, sublayer = site.split(".")
            block = model.blocks[int(layer_kind[1:].partition("i")[0])]
            router = getattr(block, f"{sublayer}_router")
        record["null_rms"] = round(router.null.float().square().mean().sqrt().item(), 4)
        records.append(record)
    model.train()
    return records


# -- checkpointing -------------------------------------------------------------


def _write_snapshot(
    path: Path,
    staged: checkpoints.StagedCheckpoint,
    args,
    step: int,
    protected: set[int],
) -> Path:
    checkpoints.write_staged(path, staged)
    telemetry.log("checkpoint", step=step, path=str(path), kind="snapshot")
    existing = runs.snapshots(args.tag, args.out_dir)
    keep = {snapshot_step for snapshot_step, _ in existing[-2:]} | protected
    for snapshot_step, snapshot_path in existing:
        if snapshot_step not in keep:
            snapshot_path.unlink()
    return path


class AsyncSnapshotWriter:
    """Single-flight immutable staging plus background atomic serialization."""

    def __init__(self, args, model, pair, protected: set[int]):
        self.args = args
        self.model = model
        self.pair = pair
        self.protected = protected
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snapshot")
        self.pending: Future[Path] | None = None

    def poll(self, *, wait: bool = False) -> Path | None:
        if self.pending is None or (not wait and not self.pending.done()):
            return None
        result = self.pending.result()
        self.pending = None
        return result

    def submit(self, step: int) -> Path:
        # Snapshot intervals are long; bounding the queue at one keeps pinned
        # host memory and write failures explicit rather than accumulating them.
        self.poll(wait=True)
        path = runs.snapshot_path(self.args.tag, step, self.args.out_dir)
        staged = checkpoints.stage(CONTRACT, self.model, self.pair, self.args, step)
        self.pending = self.executor.submit(
            _write_snapshot,
            path,
            staged,
            self.args,
            step,
            self.protected,
        )
        return path

    def close(self) -> None:
        try:
            self.poll(wait=True)
        finally:
            self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


# -- the run -------------------------------------------------------------------


def model_fields(args) -> dict:
    """The ``ModelConfig`` fields one run's arguments name."""
    return dict(
        vocab_size=args.vocab_size,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        intermediate=args.intermediate,
        pkda_heads=args.pkda_heads,
        pkda_head_dim=args.pkda_head_dim,
        pkda_conv_size=args.pkda_conv_size,
        max_seq_len=args.seq_len + 1,
        loop_iterations=args.loop_iterations,
        loop_max_iterations=args.loop_max_iterations,
    )


def reference_active(args) -> int:
    """Active non-embedding parameters of the flat full stack, ``arf``, at this
    geometry: the denominator of the tokens-per-parameter ratio. Every
    condition at a scale shares it, so paired runs keep one schedule."""
    with torch.device("meta"):
        model = DeltaModel(condition_config("arf", **model_fields(args)))
    total = sum(parameter.numel() for parameter in model.parameters())
    return total - model.embed_tokens.weight.numel()


def schedule_steps(args, tokens_per_param: float | None = None) -> int:
    """Steps that reach ``tokens_per_param`` predicted tokens per reference
    active parameter, rounded up to whole steps so the target is never
    undershot; the ratio defaults to the run's own."""
    ratio = args.tokens_per_param if tokens_per_param is None else tokens_per_param
    target = Fraction(str(ratio)) * reference_active(args)
    return math.ceil(target / (args.batch_rows * args.seq_len))


def resolve_run_args(
    parser: argparse.ArgumentParser, argv: list[str]
) -> tuple[argparse.Namespace, frozenset[str]]:
    """Parse one command line and settle the recipe it names.

    ``--scale`` fills every geometry and batch field the operator left unset;
    a retyped ``--seq-len`` without ``--batch-rows`` keeps ``BATCH_TOKENS``
    predictions per step; and a fresh run's ``--steps`` is derived from
    ``--tokens-per-param`` unless typed. Returns the settled arguments and the
    destinations the operator pinned: what they typed, plus what an explicit
    ``--scale`` or ``--tokens-per-param`` fixed, which a resume validates
    against the checkpoint rather than inherits from it.
    """
    args = parser.parse_args(argv)
    explicit = checkpoints.explicit_destinations(parser, argv)
    pinned = set(explicit)
    for field, value in SCALES[args.scale].items():
        if field not in explicit:
            setattr(args, field, value)
            if "scale" in explicit:
                pinned.add(field)
    if "seq_len" in explicit and "batch_rows" not in explicit:
        if BATCH_TOKENS % args.seq_len:
            parser.error(
                f"--seq-len {args.seq_len} does not divide the {BATCH_TOKENS:,}"
                "-prediction step; give --batch-rows"
            )
        args.batch_rows = BATCH_TOKENS // args.seq_len
        pinned.add("batch_rows")
    if "steps" in explicit and "tokens_per_param" in explicit:
        parser.error("give --steps or --tokens-per-param, not both")
    if args.resume and args.continue_from:
        parser.error("--resume continues this tag; --continue extends another")
    if "steps" not in explicit and not (args.resume or args.continue_from):
        args.steps = schedule_steps(args)
        if "tokens_per_param" in explicit:
            pinned.add("steps")
    return args, frozenset(pinned)


def parse_run_args(argv: list[str]) -> argparse.Namespace:
    """One run's settled arguments; the entry every parser user goes through."""
    return resolve_run_args(build_parser(), argv)[0]


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_schedule(args) -> Schedule:
    """WSD as the shared four-phase Schedule (no preheat), with ratio
    nudges so integer spans land exactly."""
    total = args.steps
    # Warmup is fixed per scale: the fraction applies to the shorter of the
    # run and the 25x recipe at its geometry, so a longer run keeps the 25x
    # run's warmup and is that run's continuation exactly, while a shorter
    # one keeps the fraction. Cooldown and the feedback boundary stay
    # fractions of the run.
    warmup = round(args.warmup_frac * min(total, schedule_steps(args, DEFAULT_TOKENS_PER_PARAM)))
    cooldown = round(args.cooldown_frac * total)
    heat = total - warmup - cooldown
    if heat < 1:
        raise ValueError(f"steps={total} leaves no stable phase")
    schedule = Schedule(
        heat=heat,
        warmup=(warmup + 1e-9) / heat,
        cooldown=(cooldown + 1e-9) / heat,
    )
    assert schedule.warmup_steps == warmup
    assert schedule.cooldown_steps == cooldown
    assert schedule.total == total
    return schedule


def clip_gradients(parameters) -> float:
    """Clip one accumulated global gradient vector and return its pre-clip norm."""
    total_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=GRAD_CLIP_NORM,
        norm_type=2.0,
        error_if_nonfinite=True,
    )
    return total_norm.item()


def train(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    args, pinned = resolve_run_args(parser, argv)
    device = pick_device(args.device)
    if device.type == "cuda":
        # Ada's TF32 tensor cores materially accelerate NorMuonH's FP32 batched
        # Newton-Schulz products; the trunk itself runs BF16 under autocast.
        torch.set_float32_matmul_precision("high")

    start_step = 0
    payload = None
    if args.resume:
        path = runs.latest_snapshot(args.tag, args.out_dir)
        payload = checkpoints.read(path, CONTRACT, map_location="cpu")
        CONTRACT.check_resumable(path, payload["version"])
        saved = payload["args"]
        missing = checkpoints.missing_fields(saved, EXACT_FIELDS)
        if missing:
            raise ValueError(f"{path}: checkpoint lacks settings {missing}")
        conflicts = checkpoints.mismatched_fields(
            saved, args, EXACT_FIELDS, only=pinned
        )
        if conflicts:
            raise ValueError(f"resume conflicts with checkpoint settings: {conflicts}")
        checkpoints.inherit(
            args,
            saved,
            exact_fields=EXACT_FIELDS,
            runtime_fields=RUNTIME_FIELDS,
            explicit=pinned,
        )
        if "tokens_per_param" in pinned and schedule_steps(args) != args.steps:
            raise ValueError(
                f"--tokens-per-param {args.tokens_per_param:g} is "
                f"{schedule_steps(args)} steps at the checkpoint's geometry; "
                f"its schedule is {args.steps}"
            )
    elif args.continue_from:
        source = args.continue_from
        found = runs.snapshots(source, args.out_dir)
        if not found:
            raise FileNotFoundError(f"no snapshot of {source} under {args.out_dir}")
        latest_path = found[-1][1]
        payload = checkpoints.read(latest_path, CONTRACT, map_location="cpu")
        saved = payload["args"]
        missing = checkpoints.missing_fields(saved, EXACT_FIELDS)
        if missing:
            raise ValueError(f"{latest_path}: checkpoint lacks settings {missing}")
        # A continuation is the same run with a longer schedule: every
        # state-defining field but the length carries over, and typing a
        # different one is a different experiment.
        carried = tuple(field for field in EXACT_FIELDS if field != "steps")
        conflicts = checkpoints.mismatched_fields(saved, args, carried, only=pinned)
        if conflicts:
            raise ValueError(
                f"--continue {source} changes {conflicts}; a continuation keeps "
                "every setting but the schedule length"
            )
        checkpoints.inherit(
            args,
            saved,
            exact_fields=carried,
            runtime_fields=RUNTIME_FIELDS,
            explicit=pinned,
        )
        if args.steps is None:
            args.steps = schedule_steps(args)
        source_args = SimpleNamespace(**saved)
        source_schedule = build_schedule(source_args)
        longer = build_schedule(args)
        if longer.total <= source_schedule.total:
            raise ValueError(
                f"--continue {source} needs a longer schedule than its "
                f"{source_schedule.total} steps, not {longer.total}"
            )
        fork = continuation_step(source_args, source_schedule, args, longer)
        eligible = [(step, p) for step, p in found if step <= fork]
        if not eligible:
            raise ValueError(
                f"{source} has no snapshot at or before step {fork}, the last "
                "step the longer schedule reproduces"
            )
        path = eligible[-1][1]
        if path != latest_path:
            payload = checkpoints.read(path, CONTRACT, map_location="cpu")
        CONTRACT.check_resumable(path, payload["version"])

    if args.batch_rows % args.micro_rows:
        raise ValueError("batch-rows must be a multiple of micro-rows")
    if args.head_flush_every < 1:
        raise ValueError("head-flush-every must be at least one head call")
    schedule = build_schedule(args)
    total = schedule.total

    meta = read_meta(args.data_dir)
    if meta["vocab_size"] > args.vocab_size:
        raise ValueError(
            f"data vocab {meta['vocab_size']} exceeds model vocab {args.vocab_size}"
        )
    data_train = TokenData.load(args.data_dir, "train", args.seq_len)
    data_val = TokenData.load(args.data_dir, "val", args.seq_len)
    needed = total * args.batch_rows
    if needed > data_train.rows:
        raise ValueError(f"schedule needs {needed} rows, stream has {data_train.rows}")

    torch.manual_seed(args.seed)
    model = DeltaModel(
        condition_config(args.condition, **model_fields(args))
    ).to(device)
    optimizers = build_optimizers(
        model,
        lr_normuonh=args.lr_normuonh,
        lr_nadam=args.lr_nadam,
    )
    pair = OptimizerPair(optimizers)

    if payload is not None:
        start_step = checkpoints.restore(
            payload, CONTRACT, model, pair, current_optimizer_groups=False
        )
    if args.resume:
        # The spool folds the log at this boundary; it needs the source path.
        telemetry.log(
            "resume", step=telemetry.step_address(start_step, total), path=str(path)
        )
    else:
        telemetry.log(
            "run",
            tag=args.tag,
            params=sum(p.numel() for p in model.parameters()),
            device=str(device),
            grad_clip=GRAD_CLIP_NORM,
            routing_block_size=model.cfg.routing_block_size,
            mup_ratio=model.cfg.mup_ratio,
            **{name: getattr(args, name) for name in EXACT_FIELDS},
        )
        if args.continue_from:
            telemetry.log(
                "continue",
                source=args.continue_from,
                step=telemetry.step_address(start_step, total),
                exact=source_schedule.warmup_steps == schedule.warmup_steps,
                path=str(path),
            )

    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    # Persistent snapshots at the cooldown boundary, the feedback boundary
    # (the last one-pass state), and the end, so cooldown and feedback
    # variants can continue from the exact pre-boundary state.
    protected = {schedule.heat_end, feedback_boundary(args, total), total} - {0}
    end_step = total
    if args.max_steps is not None:
        end_step = min(total, start_step + args.max_steps)
    telemetry.log(
        "schedule",
        warmup_steps=schedule.warmup_steps,
        preheat_steps=schedule.preheat_steps,
        heat_steps=schedule.heat_steps,
        cooldown_steps=schedule.cooldown_steps,
        feedback_boundary=feedback_boundary(args, total),
        start_step=start_step,
        end_step=end_step,
        total_steps=total,
    )
    model.train()
    graph_runner = None
    eval_graph_runner = None
    if device.type == "cuda":
        graph_runner = CudaGraphTrainer(model, optimizers, args, schedule)
        eval_graph_runner = CudaEvalRunner(model, args, graph_runner.pool)
        torch.cuda.reset_peak_memory_stats()
        telemetry.log(
            "execution",
            **execution_fields(model, graph_runner, eval_graph_runner),
        )
    process_start = time.monotonic()
    window_start, window_tokens, window_pass_tokens = process_start, 0, 0
    window_cell_tokens = 0.0
    summary: dict = {}
    interrupted = False
    snapshot_writer = AsyncSnapshotWriter(args, model, pair, protected)

    try:
        for step in range(start_step + 1, end_step + 1):
            snapshot_writer.poll()
            learning_rates = apply_schedule(optimizers, schedule, step)
            phase = schedule.phase(step)[0]
            z_coef = args.zloss if phase == "cooldown" else 0.0
            n_passes = 1
            if model.cfg.feedback_active:
                n_passes = draw_passes(args, step, total)
            iterations = draw_iterations(args, step, model.cfg.loop)
            checkpointing = automatic_checkpoint(
                model, n_passes, iterations, args, device
            )

            micros = args.batch_rows // args.micro_rows
            if graph_runner is not None:
                spec = GraphSpec(n_passes, checkpointing, iterations)
                graph_state = graph_runner.begin(spec, z_coef)
                graph_runner.replay_batch(
                    graph_state, data_train, step, (step - 1) * args.batch_rows
                )
                step_loss = graph_state.loss_sum.item()
                pass1_loss = graph_state.pass1_sum.item()
                graph_runner.prepare_optimizer(graph_state)
            else:
                model.grad_checkpoint = checkpointing
                step_loss = 0.0
                pass1_loss = 0.0
                for micro in range(micros):
                    first_row = (step - 1) * args.batch_rows + micro * args.micro_rows
                    rows = data_train.batch(first_row, args.micro_rows, device)
                    prefix = jitter = None
                    if n_passes > 1:
                        prefix, jitter = micro_draws(
                            args,
                            step,
                            first_row,
                            n_passes,
                            rows.shape[0],
                            args.dim,
                            device,
                        )
                    with autocast:
                        outs = multipass(
                            model,
                            rows,
                            n_passes,
                            prefix_lens=prefix,
                            jitter=jitter,
                            iterations=iterations,
                        )
                        loss, losses = multipass_loss(model, rows, outs, z_coef=z_coef)
                    (loss / micros).backward()
                    step_loss += loss.item() / micros
                    pass1_loss += losses[0].item() / micros

            grad_norm = clip_gradients(model.parameters())
            for optimizer in optimizers:
                optimizer.step()
            model.refresh_shadows()
            if graph_runner is not None:
                graph_runner.zero_grad()
            else:
                model.zero_grad(set_to_none=True)
            # A pass executes ``executed_layers / cell`` cell-equivalents, so
            # cell-tokens beside pass-tokens keep matched data apart from
            # matched compute when the loop makes the column deeper.
            cells = model.cfg.executed_layers(iterations) / model.cfg.routing_block_size
            window_tokens += args.batch_rows * args.seq_len
            window_pass_tokens += n_passes * args.batch_rows * args.seq_len
            window_cell_tokens += cells * n_passes * args.batch_rows * args.seq_len

            elapsed = time.monotonic() - window_start
            fields = {
                "step": telemetry.step_address(step, total),
                "phase": phase,
                "loss": telemetry.format_metric(step_loss),
                "pass1": telemetry.format_metric(pass1_loss),
                "k": n_passes,
            }
            if model.cfg.loop:
                fields["r"] = iterations
            fields |= {
                "lr_normuonh": telemetry.format_metric(learning_rates["normuonh"]),
                "lr_nadam": telemetry.format_metric(learning_rates["nadam"]),
                "lr_nadam_width": telemetry.format_metric(
                    learning_rates["nadam_width"]
                ),
                "gnorm": telemetry.format_metric(grad_norm),
                "tok_s": f"{window_tokens / max(elapsed, 1e-9):.0f}",
                "pass_tok_s": (f"{window_pass_tokens / max(elapsed, 1e-9):.0f}"),
                "cell_tok_s": (f"{window_cell_tokens / max(elapsed, 1e-9):.0f}"),
                "elapsed": f"{time.monotonic() - process_start:.1f}",
            }
            if device.type == "cuda":
                fields["mem"] = f"{torch.cuda.max_memory_allocated() / 2**30:.1f}G"
            telemetry.log("step", **fields)
            window_start, window_tokens, window_pass_tokens = (
                time.monotonic(),
                0,
                0,
            )
            window_cell_tokens = 0.0

            if step % args.eval_every == 0 or step == total:
                address = telemetry.step_address(step, total)
                scores = evaluate(
                    model, data_val, args, device, graph_runner=eval_graph_runner
                )
                telemetry.log(
                    "eval",
                    step=address,
                    **{
                        key: telemetry.format_metric(value)
                        for key, value in scores.items()
                    },
                )
                summary.update(scores)
                for record in route_summary(model, data_val, args, device):
                    telemetry.log("route", step=address, **record)
                if model.cfg.feedback_active:
                    trace = iterate_fused(
                        model, data_val.batch(0, 2, device), n_iters=8
                    )
                    model.train()
                    telemetry.log(
                        "contract",
                        step=address,
                        loss0=telemetry.format_metric(trace[0]["loss"]),
                        loss8=telemetry.format_metric(trace[-1]["loss"]),
                        upd8=telemetry.format_metric(trace[-1]["update_norm"]),
                    )
                if model.cfg.loop:
                    sweep = depth_trace(model, data_val.batch(0, 2, device))
                    model.train()
                    by_depth = {record["iterations"]: record for record in sweep}
                    telemetry.log(
                        "depth",
                        step=address,
                        r_mean=model.cfg.loop_iterations,
                        r_max=model.cfg.loop_max_iterations,
                        loss_one=telemetry.format_metric(by_depth[1]["loss"]),
                        loss_mean=telemetry.format_metric(
                            by_depth[model.cfg.loop_iterations]["loss"]
                        ),
                        loss_max=telemetry.format_metric(sweep[-1]["loss"]),
                        upd_max=telemetry.format_metric(sweep[-1]["update_norm"]),
                    )

            if step % args.snapshot_every == 0 or step in protected:
                snapshot_writer.submit(step)
            summary["step"] = step
            summary["loss"] = step_loss
    except KeyboardInterrupt:
        interrupted = True
        step = summary.get("step", start_step)
        if step > start_step:
            snapshot_writer.submit(step)
        telemetry.log("interrupt", step=telemetry.step_address(step, total))
    except BaseException:
        snapshot_writer.close()
        raise

    if not interrupted:
        last = summary.get("step", start_step)
        if last < total and last > start_step and last % args.snapshot_every:
            snapshot_writer.submit(last)
    snapshot_writer.close()

    if not interrupted:
        last = summary.get("step", start_step)
        if last < total:
            telemetry.log("yield", step=telemetry.step_address(last, total))
        else:
            telemetry.log(
                "done",
                step=telemetry.step_address(total, total),
                **{
                    k: telemetry.format_metric(v)
                    for k, v in summary.items()
                    if k.startswith("val")
                },
            )
    return summary


def main() -> None:
    train()


if __name__ == "__main__":
    main()

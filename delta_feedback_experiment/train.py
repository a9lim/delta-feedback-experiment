"""One condition's training run under the binding recipe.

The loop is a pure function of (init seed, data seed, step): data rows
are step-addressed slices of the fixed stream, recurrence randomness
(the pass and column roll, prefix lengths, jitter) derives from keyed generators
addressed by step and global row rather than ambient RNG state, and the WSD
schedule is the shared :class:`Schedule` addressed by cumulative step.  That
is what makes a resumed invocation bit-identical to an uninterrupted one on
the same replay plan and every condition's batches the same bytes (the
paired-comparison contract; paired runs must share ``data_seed`` and
``batch_rows``).

Data parallelism splits each step's rows across ``--ranks`` processes, one
per device, launched through ``torchrun``. Every rank replays the same graphs
on its own rows; the FP32 gradient sinks reduce onto the ranks that own each
matrix, NorMuonH steps the matrices it owns, and the updated working copies
gather back. One rank is the same code without communication.

Operational layer — telemetry records, immutable ``runs``-addressed
snapshots, resume reconciliation — comes from ``transformer_experiments``.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import math
import sys
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch
from transformer_experiments import checkpoints, runs, telemetry
from transformer_experiments.schedule import Schedule

from . import distributed
from .data import DEFAULT_SOURCE, SOURCES, TokenData, read_meta
from .model import (
    CONDITION_LETTERS,
    DEFAULT_LOOP_ITERATIONS,
    EXPERT_BALANCE_COEF,
    EXPERT_BIAS_RATE,
    LOOP_MAX_ITERATIONS,
    MTP_LOSS_WEIGHT,
    DeltaModel,
    combine_column_losses,
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
    NorMuonH,
    OptimizerPair,
    apply_schedule,
    build_optimizers,
)
from .sites import ParameterSites
from .tokenizer import SYNTHETIC_TOKENIZER_ID, TOKENIZER_ID, VOCAB_SIZE

CONTRACT = checkpoints.CheckpointContract(
    version=42, resumable=frozenset({42}), surface_version=42
)


def read_checkpoint(path: str | Path) -> dict:
    """Read a snapshot whose vocabulary and model state have current meaning."""
    payload = checkpoints.read(path, CONTRACT, map_location="cpu")
    CONTRACT.check_resumable(path, payload["version"])
    if payload["args"].get("tokenizer_id") not in (
        TOKENIZER_ID,
        SYNTHETIC_TOKENIZER_ID,
    ):
        raise ValueError(
            f"{path}: checkpoint tokenizer identity differs from current code"
        )
    return payload


BATCH_TOKENS = 524_288
"""Predicted tokens per optimizer step at every scale: 2^19, 128 rows of 4,096."""

SCALES: dict[str, dict[str, int]] = {
    "screen": {
        "dim": 768,
        "layers": 16,
        "heads": 8,
        "kv_heads": 4,
        "expert_intermediate": 832,
        "num_routed_experts": 15,
        "experts_per_token": 3,
        "pkda_heads": 10,
        "seq_len": 4096,
        "batch_rows": 128,
        "micro_rows": 1,
    },
    "bridge": {
        "dim": 1152,
        "layers": 16,
        "heads": 12,
        "kv_heads": 6,
        "expert_intermediate": 832,
        "num_routed_experts": 23,
        "experts_per_token": 5,
        "pkda_heads": 15,
        "seq_len": 4096,
        "batch_rows": 128,
        "micro_rows": 1,
    },
    "flagship": {
        "dim": 1536,
        "layers": 16,
        "heads": 16,
        "kv_heads": 8,
        "expert_intermediate": 832,
        "num_routed_experts": 31,
        "experts_per_token": 7,
        "pkda_heads": 20,
        "seq_len": 4096,
        "batch_rows": 128,
        "micro_rows": 1,
    },
    "extension": {
        "dim": 2304,
        "layers": 16,
        "heads": 24,
        "kv_heads": 12,
        "expert_intermediate": 832,
        "num_routed_experts": 47,
        "experts_per_token": 11,
        "pkda_heads": 30,
        "seq_len": 4096,
        "batch_rows": 128,
        "micro_rows": 1,
    },
}
"""Geometry and batch presets from ``docs/scaling.md``.
Every scale shares 4,096-token rows in one-row microbatches with
``BATCH_TOKENS`` predictions per step and the same recurrent-depth recipe."""

DEFAULT_TOKENS_PER_PARAM = 25.0
"""The screen recipe's predicted tokens per reference active parameter."""

EXACT_FIELDS = (
    "tokenizer_id",
    "condition",
    "seed",
    "data_seed",
    "seq_len",
    "batch_rows",
    "steps",
    "warmup_frac",
    "cooldown_frac",
    "recurrence_start",
    "three_rate",
    "loop_iterations",
    "lr_normuonh",
    "lr_nadam",
    "jitter",
    "zloss",
    "mtp_weight",
    "vocab_size",
    "dim",
    "layers",
    "heads",
    "kv_heads",
    "head_dim",
    "expert_intermediate",
    "num_routed_experts",
    "experts_per_token",
    "pkda_heads",
    "pkda_head_dim",
    "pkda_conv_size",
)
"""State-defining settings: a resume takes these from the checkpoint."""

RUNTIME_FIELDS = (
    "data_root",
    "source",
    "out_dir",
    "device",
    "eval_every",
    "snapshot_every",
    "eval_rows",
    "micro_rows",
    "checkpoint_margin_gib",
)
"""Per-invocation settings: inherited unless retyped. ``--ranks`` is neither:
it describes this invocation's processes alone and is never inherited."""


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


def nonnegative_finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


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
        default="f",
        metavar="{f,l,fl}",
        help=(
            "feedback (f), looped depth (l), or both (fl); default f: "
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
    parser.add_argument(
        "--data-root",
        default="data",
        help="parent directory of token stores; reads DATA_ROOT/SOURCE (default data)",
    )
    parser.add_argument(
        "--source",
        choices=tuple(SOURCES),
        default=DEFAULT_SOURCE,
        help="token-store source under --data-root (default dclm-100b)",
    )
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
            "geometry and batch preset from docs/scaling.md; "
            "a trunk or recipe flag typed alongside overrides its field "
            "(default: screen)"
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
            "steps, unless --steps is given (default: 25)"
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
        "--recurrence-start",
        type=probability,
        default=0.75,
        help=(
            "fraction of steps before the recurrence roll begins; every earlier "
            "step is one plain column and every later step rolls two or three "
            "passes (f) and two or three columns per pass (l)"
        ),
    )
    schedule.add_argument(
        "--three-rate",
        type=probability,
        default=0.12,
        help=(
            "on each rolled step, P(three passes at two columns) and, "
            "separately, P(three columns at two passes); otherwise two and two"
        ),
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
    recipe.add_argument("--seq-len", type=int, default=4096)
    recipe.add_argument(
        "--lr-normuonh",
        type=float,
        default=DEFAULT_NORMUONH_LR,
        help=(
            "estimated RMS-to-RMS NorMuonH trial-step budget; expert gate/up "
            "and down maps multiply by sqrt(8/(experts-per-token+1)) "
            "(default: 0.006)"
        ),
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
    recipe.add_argument(
        "--jitter", type=float, default=0.02,
        help=(
            "uniform +/- payload jitter half-width in payload units, which are "
            "unit RMS at initialization; one draw per column serves MTP, "
            "feedback, and the loop (default: 0.02)"
        ),
    )
    recipe.add_argument("--zloss", type=float, default=1e-5)
    recipe.add_argument(
        "--mtp-weight", type=nonnegative_finite, default=MTP_LOSS_WEIGHT,
        help="payload-based auxiliary second-token CE and z-loss weight (default: 0.3)",
    )
    recipe.add_argument(
        "--loop-iterations",
        type=int,
        choices=range(1, LOOP_MAX_ITERATIONS + 1),
        default=DEFAULT_LOOP_ITERATIONS,
        help=(
            "l: fixed evaluation/decode columns per position (default: 2); "
            "training draws two or three from the recurrence roll"
        ),
    )

    trunk = parser.add_argument_group("trunk (state-defining)")
    trunk.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.set_defaults(tokenizer_id=TOKENIZER_ID)
    trunk.add_argument("--dim", type=int, default=768)
    trunk.add_argument("--layers", type=int, default=16)
    trunk.add_argument("--heads", type=int, default=8)
    trunk.add_argument("--kv-heads", type=int, default=4)
    trunk.add_argument(
        "--head-dim", type=int, default=192, help="GQA head width (default: 192)"
    )
    trunk.add_argument(
        "--expert-intermediate",
        type=int,
        default=832,
        help="intermediate width of each shared or routed expert",
    )
    trunk.add_argument("--num-routed-experts", type=int, default=15)
    trunk.add_argument("--experts-per-token", type=int, default=3)
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
        "--micro-rows",
        type=int,
        default=1,
        help="smallest rows per replay, and the evaluation microbatch; the "
        "trainer replays the largest divisor of a rank's rows whose "
        "activations fit, never changing the keyed per-row draws (default: 1)",
    )
    runtime.add_argument(
        "--ranks",
        type=int,
        default=1,
        help="data-parallel processes, one per device, launched through "
        "torchrun; each takes batch-rows/ranks rows of every step (default: 1; "
        "never inherited by a resume)",
    )
    runtime.add_argument(
        "--checkpoint-margin-gib",
        type=float,
        default=DEFAULT_CHECKPOINT_MARGIN_GIB,
        help="device memory kept free of retained activations when the trainer "
        "plans rows per replay and recomputed blocks for each graph "
        f"(default: {DEFAULT_CHECKPOINT_MARGIN_GIB}); raise it if warm-up or "
        "capture runs out of memory",
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


def recurrence_boundary(args, total: int) -> int:
    """Last single-column step; every later step rolls passes and columns."""
    return round(args.recurrence_start * total)


def continuation_step(
    source, source_schedule: Schedule, args, schedule: Schedule
) -> int:
    """The last step of a finished run that a longer schedule reproduces.

    Up to it every step was warmup or heat under both schedules and on the
    same side of both recurrence boundaries, so a run restored from its
    snapshot there and trained on under the longer schedule is the longer run
    from that step on, apart from the warmup it inherited. When the boundary
    moves that is the boundary itself, the last single-column state;
    otherwise it is the cooldown boundary.
    """
    fork = min(source_schedule.heat_end, schedule.heat_end)
    old = recurrence_boundary(source, source_schedule.total)
    new = recurrence_boundary(args, schedule.total)
    if old != new:
        fork = min(fork, old, new)
    return fork


def draw_recurrence(args, step: int, total: int) -> tuple[int, int]:
    """The step's (passes, columns per pass) roll, before any condition
    projects it.

    Before the boundary every step is one pass of one column. After it there
    are no single-column steps at all, so each recurrent mode is trained on
    every update rather than eroded between them: one keyed uniform draw
    lands on three passes at two columns with probability ``three_rate``, on
    three columns at two passes with the same probability, and otherwise on
    two and two. ``f`` reads the pass count, ``l`` the column count, and
    ``fl`` both, so the three conditions align step by step, and the pass
    projection is the feedback draw of a condition without ``l``.
    """
    if step <= recurrence_boundary(args, total):
        return 1, 1
    generator = torch.Generator().manual_seed(mix(args.data_seed, step, 1))
    draw = torch.rand((), generator=generator).item()
    if draw < args.three_rate:
        return 3, 2
    if draw < 2 * args.three_rate:
        return 2, 3
    return 2, 2


def step_shape(args, step: int, total: int, cfg) -> tuple[int, int]:
    """The (passes, columns per pass) a condition runs at ``step``: the
    roll's components its letters read, one each for the letters it lacks."""
    passes, iterations = draw_recurrence(args, step, total)
    return (passes if cfg.feedback else 1), (iterations if cfg.loop else 1)


def micro_draws(
    args,
    step: int,
    first_row: int,
    n_passes: int,
    iterations: int,
    n_rows: int,
    dim: int,
    device,
    *,
    prefix_out: torch.Tensor | None = None,
    jitter_out: torch.Tensor | None = None,
    loop_jitter_out: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """(prefix_lens [k-1, n], jitter [k, n, seq_len+1, dim], loop jitter
    [k, r-1, n, seq_len+1, dim] or None at one column) for ``n_rows``
    consecutive rows starting at global row ``first_row``.

    Every row's draws are keyed by (data seed, step, its own global row), so
    they are identical across conditions sharing the batch geometry and
    independent of how many rows a replay holds or which rank replays them.
    Within a row the prefix and the pass jitter, which perturbs each pass's
    last column, are drawn first, so a condition with ``l`` shares them with
    the condition without it; the loop jitter for the earlier columns
    follows. Jitter is drawn directly in payload units, uniform in +/-
    args.jitter.
    """
    if generator is None:
        generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    columns = args.seq_len + 1
    if prefix_out is None:
        prefix_out = torch.empty((n_passes - 1, n_rows), dtype=torch.long, device=device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if jitter_out is None:
        jitter_out = torch.empty(
            (n_passes, n_rows, columns, dim), dtype=dtype, device=device
        )
    if iterations > 1 and loop_jitter_out is None:
        loop_jitter_out = torch.empty(
            (n_passes, iterations - 1, n_rows, columns, dim), dtype=dtype, device=device
        )
    amplitude = args.jitter
    for row in range(n_rows):
        generator.manual_seed(mix(args.data_seed, step, first_row + row))
        # Plain-prefix lengths in 1..seq_len-1: position 0 is always plain and
        # every row keeps at least one fused position.
        torch.randint(
            1, args.seq_len, (n_passes - 1, 1), generator=generator,
            out=prefix_out[:, row : row + 1],
        )
        jitter_out[:, row].uniform_(-amplitude, amplitude, generator=generator)
        if iterations > 1:
            loop_jitter_out[:, :, row].uniform_(
                -amplitude, amplitude, generator=generator
            )
    if iterations == 1:
        return prefix_out, jitter_out, None
    return prefix_out, jitter_out, loop_jitter_out


@dataclass(frozen=True)
class GraphSpec:
    """One captured training graph: its pass count and columns per pass."""

    n_passes: int
    iterations: int = 1


@dataclass(frozen=True)
class ReplayPlan:
    """How one graph executes: rows per replay, whether its recurrences keep
    or rebuild their intermediates, and blocks recomputed in backward."""

    rows_per_replay: int
    checkpoint_blocks: int
    eligible_blocks: int
    estimated_gib: float
    lean: bool = True


@dataclass(frozen=True)
class Calibration:
    """Retained bytes per block invocation of one raw forward with the
    recurrences keeping (``full_block``) or rebuilding (``lean_block``) their
    intermediates, and the bytes one recomputed lean block releases."""

    full_block: float
    lean_block: float
    checkpoint: float


@dataclass
class CapturedMicro:
    spec: GraphSpec
    plan: ReplayPlan
    rows: torch.Tensor
    prefix: torch.Tensor
    jitter: torch.Tensor
    loop_jitter: torch.Tensor | None
    z_coef: torch.Tensor
    loss_sum: torch.Tensor
    pass1_sum: torch.Tensor
    ntp_sum: torch.Tensor
    mtp_sum: torch.Tensor
    expert_balance_sum: torch.Tensor
    expert_counts: torch.Tensor
    graph: torch.cuda.CUDAGraph | None = None
    active: frozenset[torch.nn.Parameter] = frozenset()


DEFAULT_CHECKPOINT_MARGIN_GIB = 2.0
"""Free device memory, measured after the static footprint exists, that the
retained-forward activation budget leaves untouched. Backward workspaces,
checkpoint recomputation, allocator rounding, CUDA context growth from
kernels compiled after the budget is measured, and graph instantiation also
need memory. On the 24 GiB card a 1 GiB margin OOMs in the eager warm-up
backward of the deepest screen fl graph and 3.5 GiB runs; the ``execution``
record's peak allocated bytes show what a run actually needed above its
static footprint and retained activations. The optimizer step and periodic
monitors reuse the graphs' pool (``pool_scope``)."""

def replay_widths(rank_rows: int, micro_rows: int) -> list[int]:
    """Rows per replay a rank may use, largest first: the divisors of its
    rows of the step that are multiples of the smallest replay."""
    if rank_rows % micro_rows:
        raise ValueError("a rank's rows must be a multiple of micro-rows")
    return [
        rows
        for rows in range(rank_rows, 0, -1)
        if rank_rows % rows == 0 and rows % micro_rows == 0
    ]


def input_bytes(args, spec: GraphSpec, rows: int) -> int:
    """Bytes of one captured graph's persistent inputs at ``rows`` per replay:
    the token rows, the prefix lengths, and the pass and loop jitter."""
    columns = args.seq_len + 1
    tokens = rows * columns * 8
    prefix = (spec.n_passes - 1) * rows * 8
    jitter = spec.n_passes * spec.iterations * rows * columns * args.dim * 2
    return tokens + prefix + jitter


def block_invocations(cfg, spec: GraphSpec) -> tuple[int, int]:
    """(all, checkpoint-eligible) block invocations of one logical forward.

    Every column runs the trunk layers plus the auxiliary block. The PKDA
    blocks and the auxiliary block can recompute in backward; the
    global-attention blocks stay retained.
    """
    columns = spec.n_passes * spec.iterations
    return columns * (cfg.layers + 1), columns * (cfg.pkda_layers + 1)


def plan_replay(
    cfg,
    args,
    spec: GraphSpec,
    calibration: Calibration,
    budget_bytes: float,
    rank_rows: int | None = None,
) -> ReplayPlan:
    """Fit one graph into the activation budget.

    The calibrated bytes per block invocation are what a raw forward retains,
    everything outside the blocks included, so a graph needs its block count
    times that: the per-pass extras scale with the blocks and a two-pass
    forward retains exactly twice a one-pass one. The bytes one recomputed
    block releases are calibrated too, because the eligible blocks retain
    more than the average (the global-attention blocks retain less and are
    never recomputed).

    Candidates from the widest replay down: at each width the recurrences
    keeping their intermediates, then rebuilding them; and when even the
    smallest replay does not fit raw, the first blocks of the logical forward
    recompute in backward, as many as the shortfall needs. Width comes first
    because the one-row replay measurably costs occupancy and wider replays
    may pay on a card with more SMs, while keeping the intermediates was
    measured worth 0.4 ms per two-row replay on the 4090: free where it
    fits, never worth a narrower replay. The keyed draws are per row, so the
    width changes no value.
    """
    blocks, eligible = block_invocations(cfg, spec)
    if rank_rows is None:
        rank_rows = args.batch_rows
    variants = ((False, calibration.full_block), (True, calibration.lean_block))

    def needed(rows: int, per_block: float) -> float:
        return (rows / args.micro_rows) * blocks * per_block

    for rows in replay_widths(rank_rows, args.micro_rows):
        for lean, per_block in variants:
            if needed(rows, per_block) <= budget_bytes:
                return ReplayPlan(
                    rows, 0, eligible, needed(rows, per_block) / 2**30, lean
                )
    rows = args.micro_rows
    shortfall = needed(rows, calibration.lean_block) - budget_bytes
    count = min(eligible, math.ceil(shortfall / max(calibration.checkpoint, 1.0)))
    return ReplayPlan(
        rows, count, eligible, needed(rows, calibration.lean_block) / 2**30, True
    )


def plan_graphs(
    args,
    specs: list[GraphSpec],
    budget_for: Callable[[int], float],
    plan: Callable[[GraphSpec, float], ReplayPlan],
) -> tuple[dict[GraphSpec, ReplayPlan], int, float]:
    """Every graph's plan, the bytes their inputs set aside, and the budget.

    ``budget_for(inputs)`` is the activation budget once ``inputs`` bytes of
    persistent graph inputs are set aside, the smallest across ranks. The
    inputs depend on the widths the plans choose and the plans on the budget
    the inputs leave, so this starts from the inputs of the smallest replays,
    plans, and re-plans with what the plans need until no plan's inputs
    exceed what was set aside; widths only shrink between rounds, so the
    loop ends within the number of widths.
    """
    reserved = sum(input_bytes(args, spec, args.micro_rows) for spec in specs)
    while True:
        budget = budget_for(reserved)
        plans = {spec: plan(spec, budget) for spec in specs}
        needed = sum(
            input_bytes(args, spec, chosen.rows_per_replay)
            for spec, chosen in plans.items()
        )
        if needed <= reserved:
            return plans, reserved, budget
        reserved = needed


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
    # Collect warm-up cycles before capture. A later collection can free
    # their CUDA resources and
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


@dataclass
class StepSums:
    """One step's accumulated losses and expert counts on the device."""

    loss_sum: torch.Tensor
    pass1_sum: torch.Tensor
    ntp_sum: torch.Tensor
    mtp_sum: torch.Tensor
    expert_balance_sum: torch.Tensor
    expert_counts: torch.Tensor

    @classmethod
    def zeros(cls, model: DeltaModel, device: torch.device) -> StepSums:
        scalar = lambda: torch.zeros((), dtype=torch.float32, device=device)  # noqa: E731
        return cls(
            scalar(), scalar(), scalar(), scalar(), scalar(),
            torch.zeros(
                (len(model.expert_banks), model.cfg.num_routed_experts),
                dtype=torch.int64,
                device=device,
            ),
        )

    def reset(self) -> None:
        for total in (
            self.loss_sum, self.pass1_sum, self.ntp_sum, self.mtp_sum,
            self.expert_balance_sum, self.expert_counts,
        ):
            total.zero_()


def mode_activity(
    gradients: dict[torch.nn.Parameter, torch.Tensor],
    sharded: frozenset[torch.nn.Parameter],
    banks,
) -> set[torch.nn.Parameter]:
    """The parameters one graph mode updates, decided after its warm-up.

    A sharded matrix is active when warm-up left anything in its sink: the
    projections behind a GEMM either run or do not. Every replicated
    parameter is active in every mode, whatever its warm-up gradient holds:
    autograd reaches all of them on every column, and a reached gradient can
    be exactly zero (a routing site's key-norm gain multiplies its
    zero-initialized query), which must not read as absence. Warm-up rows
    can select only a few experts, and replay data changes those choices,
    so activity is structural for the entire bank.
    """
    active = {parameter for parameter in gradients if parameter not in sharded}
    active.update(p for p in sharded if bool(gradients[p].any()))
    for bank in banks:
        active.update(bank.parameters())
    return active


class Trainer:
    """What both execution engines share: the sites, the gradient table, the
    optimizer materialization, and the rows of the step this rank runs.

    The tied embedding and the large projections accumulate into their sinks
    in place from inside backward and hand autograd no gradient; NorMuonH
    reads those sinks through the gradient table. The replicated parameters
    NAdam owns keep ordinary autograd gradients whose ``.grad`` is a view of
    the FP32 arena.
    """

    def __init__(
        self,
        model: DeltaModel,
        optimizers,
        sites: ParameterSites,
        args,
        topology: distributed.Topology,
    ):
        self.model = model
        self.optimizers = optimizers
        self.sites = sites
        self.args = args
        self.topology = topology
        self.device = next(model.parameters()).device
        self.rank_rows = args.batch_rows // topology.world
        if not sites.allocated:
            sites.allocate()
        self.model.refresh_shadows()
        self.gradients = sites.gradients()
        self.sharded = sites.sharded
        self.replicated = [
            parameter for parameter in self.gradients if parameter not in self.sharded
        ]
        self.normuonh = next(o for o in optimizers if isinstance(o, NorMuonH))
        self.normuonh.bind_gradients(self.gradients)
        for parameter in self.replicated:
            parameter.grad = self.gradients[parameter]
        self._initialize_optimizers()
        self.normuonh.materialize()
        # From here the sharded matrices exist only as working copies plus the
        # owner's FP32 master.
        sites.adopt()

    def _initialize_optimizers(self) -> None:
        """Materialize persistent state before anything transient."""
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

    def rank_first_row(self, first_row: int) -> int:
        """The first of this rank's rows within a step starting at ``first_row``."""
        return first_row + self.topology.rank * self.rank_rows

    def prepare_optimizer(self, active: frozenset[torch.nn.Parameter]) -> None:
        """Hand the optimizers the reduced step gradient of the active parameters.

        An absent parameter keeps its weights and state. Any per-pattern
        optimizer tensor must be allocated here, outside ``pool_scope``: one
        allocated inside would sit in memory the next replay overwrites.
        """
        self.normuonh.bind_gradients(
            {p: g for p, g in self.gradients.items() if p in active and p in self.sharded}
        )
        for parameter in self.replicated:
            parameter.grad = self.gradients[parameter] if parameter in active else None
        for optimizer in self.optimizers:
            prepare = getattr(optimizer, "prepare", None)
            if prepare is not None:
                prepare()

    def zero_grad(self) -> None:
        self.sites.zero_gradients()
        for parameter in self.replicated:
            parameter.grad = self.gradients[parameter]

    @contextlib.contextmanager
    def pool_scope(self):
        yield


class EagerTrainer(Trainer):
    """The portable engine: eager microbatches of ``micro_rows`` rows."""

    def __init__(self, model, optimizers, sites, args, topology):
        super().__init__(model, optimizers, sites, args, topology)
        self.active = frozenset(self.gradients)
        self.sums = StepSums.zeros(model, self.device)

    def begin(self, spec: GraphSpec, z_coef: float = 0.0) -> StepSums:
        self.spec = spec
        self.z_coef = z_coef
        self.sums.reset()
        return self.sums

    def replay_batch(self, state: StepSums, data: TokenData, step: int, first_row: int):
        args = self.args
        first = self.rank_first_row(first_row)
        scale = args.micro_rows / args.batch_rows
        for offset in range(0, self.rank_rows, args.micro_rows):
            rows = data.batch(first + offset, args.micro_rows, self.device)
            prefix, jitter, loop_jitter = micro_draws(
                args, step, first + offset, self.spec.n_passes, self.spec.iterations,
                rows.shape[0], args.dim, self.device,
            )
            outs = multipass(
                self.model, rows, self.spec.n_passes, iterations=self.spec.iterations,
                prefix_lens=prefix, jitter=jitter, loop_jitter=loop_jitter,
            )
            loss_result = multipass_loss(
                self.model, rows, outs, z_coef=self.z_coef, mtp_weight=args.mtp_weight
            )
            loss, losses = loss_result.total, loss_result.ntp
            (loss * scale).backward()
            state.loss_sum.add_(loss.detach() * scale)
            state.pass1_sum.add_(losses[0][0].detach() * scale)
            state.ntp_sum.add_(combine_column_losses(loss_result.ntp).detach() * scale)
            state.mtp_sum.add_(combine_column_losses(loss_result.mtp).detach() * scale)
            state.expert_counts.add_(loss_result.expert_counts)
            state.expert_balance_sum.add_(loss_result.expert_aux_loss.detach() * scale)

    def prepare_optimizer(self, state: StepSums) -> None:  # type: ignore[override]
        super().prepare_optimizer(self.active)


class CudaGraphTrainer(Trainer):
    """Fixed-address forward/backward graphs for every reachable step mode.

    The model's gradient slabs and every graph input are allocated once.
    Data and keyed CUDA randomness are copied/drawn into those addresses before
    replay.  Graphs share a private pool and never overlap; each captured body
    ends after backward, so no saved activation survives between replays.

    Every persistent FP32 gradient slab, the optimizer state, and the owned
    FP32 masters exist before the first forward, and the replicated FP32
    parameters are gone, so the activation budget is measured against the
    real static footprint: the cheapest reachable graph runs one eager
    forward, its retained bytes per block invocation calibrate
    ``plan_replay``, and every graph then gets its rows per replay and its
    recomputed block count. The budget is the smallest across ranks, so every
    rank replays the same plan.

    A slab that warm-up touched marks its parameter active for that mode
    exactly as an autograd gradient marks the replicated vectors and small
    matrices. The head's classifier gradient accumulates straight into the
    tied embedding's FP32 sink on every call, whatever the replay's width.
    """

    def __init__(self, model: DeltaModel, optimizers, sites, args, schedule: Schedule, topology):
        super().__init__(model, optimizers, sites, args, topology)
        self.autocast = torch.autocast("cuda", dtype=torch.bfloat16)
        self.generator = torch.Generator(device=self.device)
        self.batch_stager = CudaBatchStager(
            self.rank_rows, args.seq_len + 1, self.device
        )

        specs = self._reachable_specs(schedule)
        base = min(specs, key=lambda spec: (spec.n_passes, spec.iterations))
        # Compile before the budget is read: the calibration forwards build
        # every block, both recurrence variants, at the smallest replay, and
        # the optimizer warm-up every bucket, so the CUDA context they grow
        # is already outside the memory then measured free. On several ranks
        # the main rank goes first and fills the compile caches the others
        # then read, instead of every rank compiling the same kernels at once.
        if not topology.main:
            distributed.barrier()
        self.calibration = self._calibrate(base)
        self.normuonh.warmup()
        if topology.main:
            distributed.barrier()
        # Measure what the device reports free, with the allocator's cache
        # emptied: that excludes the CUDA context, the communicator, and any
        # other process, which a total-minus-static estimate would count.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.static_bytes = torch.cuda.memory_allocated()
        # Cached but unallocated memory at this point is fragmentation the
        # budget cannot use: the replicated FP32 copies released by adoption
        # sit between live allocations, so only whole free pages return.
        self.cached_bytes = torch.cuda.memory_reserved() - self.static_bytes
        free_bytes = torch.cuda.mem_get_info(self.device)[0]
        margin = args.checkpoint_margin_gib * 2**30

        def budget_for(inputs: int) -> float:
            # Every graph's persistent inputs are allocated once the plans
            # exist and stay live together, so they come off the free memory
            # first; the smallest budget across ranks plans every rank.
            budget = torch.tensor(
                [-(free_bytes - inputs - margin)], dtype=torch.float64, device=self.device
            )
            return -distributed.all_reduce_(budget, maximum=True).item()

        plans, self.inputs_bytes, self.budget_bytes = plan_graphs(
            args,
            specs,
            budget_for,
            lambda spec, budget: self._plan(spec, self.calibration, budget),
        )
        telemetry.log(
            "memory_plan",
            static_gib=round(self.static_bytes / 2**30, 2),
            cached_gib=round(self.cached_bytes / 2**30, 2),
            inputs_gib=round(self.inputs_bytes / 2**30, 2),
            activation_budget_gib=round(self.budget_bytes / 2**30, 2),
            checkpoint_margin_gib=args.checkpoint_margin_gib,
            block_mib=round(self.calibration.lean_block / 2**20, 1),
            block_full_mib=round(self.calibration.full_block / 2**20, 1),
            checkpoint_mib=round(self.calibration.checkpoint / 2**20, 1),
        )
        self.states: dict[GraphSpec, CapturedMicro] = {
            spec: self._allocate(spec, plan) for spec, plan in plans.items()
        }

        # The package sets Dynamo's recompile budget once for the process, so
        # every block and router specialization compiles here and in later
        # eager evaluation alike.
        active_by_spec = {}
        for spec, state in self.states.items():
            telemetry.log(
                "plan",
                k=spec.n_passes,
                r=spec.iterations,
                rows=state.plan.rows_per_replay,
                saved="lean" if state.plan.lean else "full",
                checkpoint_blocks=state.plan.checkpoint_blocks,
                eligible_blocks=state.plan.eligible_blocks,
                estimated_gib=round(state.plan.estimated_gib, 2),
            )
            active_by_spec[spec] = self._warm(state)
        self.zero_grad()
        for active in active_by_spec.values():
            for optimizer in self.optimizers:
                warmup = getattr(optimizer, "warmup", None)
                if warmup is not None:
                    warmup(active)
        torch.cuda.empty_cache()

        # A MemPool object, not a bare handle: ``pool_scope`` needs the
        # object to route eager allocations into this same private pool.
        self.mempool = torch.cuda.MemPool()
        self.pool = self.mempool.id
        for spec, state in self.states.items():
            state.active = frozenset(active_by_spec[spec])
            telemetry.log("capture", k=spec.n_passes, r=spec.iterations)
            self._capture(state, self.pool)
        self.zero_grad()
        torch.cuda.synchronize()
        if not any(
            segment["is_expandable"] for segment in torch.cuda.memory_snapshot()
        ):
            raise RuntimeError(
                "the graph pool needs the allocator's expandable segments: set "
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before the first "
                "CUDA allocation (the package sets it on import unless overridden)"
            )

    @contextlib.contextmanager
    def pool_scope(self):
        """Run eager work inside the graphs' private pool.

        Only work that allocates nothing the scope outlives belongs here: the
        allocator hands out blocks the graph bodies freed, and a replay writes
        its captured addresses without allocating, so a tensor from this scope
        that is still alive at the next replay aliases what that replay
        writes. No ``graph.replay()`` may run inside the scope for the same
        reason, and the collectives stay outside it too.

        The graph bodies free every activation they allocate, but a private
        pool never returns its segments, so eager work between replays would
        otherwise be paid for twice: once in the pool and once in the margin
        the activation budget leaves free. The allocator keeps its free lists
        per stream, so only allocations on the capture stream can reuse the
        bodies' blocks, and only with expandable segments, which coalesce the
        pool into a few segments instead of hundreds the size of one body
        tensor; the two waits keep the eager work ordered against the replay
        stream on both sides.
        """
        stream = torch.cuda.graph.default_capture_stream
        assert stream is not None, "pool_scope runs after the first capture"
        current = torch.cuda.current_stream()
        stream.wait_stream(current)
        try:
            with torch.cuda.stream(stream), torch.cuda.use_mem_pool(self.mempool):
                yield
        finally:
            current.wait_stream(stream)

    def _reachable_specs(self, schedule: Schedule) -> list[GraphSpec]:
        specs = {
            GraphSpec(*step_shape(self.args, step, schedule.total, self.model.cfg))
            for step in range(1, schedule.total + 1)
        }
        return sorted(specs, key=lambda spec: (spec.n_passes, spec.iterations))

    def _plan(
        self, spec: GraphSpec, calibration: Calibration, budget_bytes: float
    ) -> ReplayPlan:
        return plan_replay(
            self.model.cfg, self.args, spec, calibration, budget_bytes, self.rank_rows
        )

    def _inputs(
        self, spec: GraphSpec, rows: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        tokens = torch.zeros(
            rows, self.args.seq_len + 1, dtype=torch.long, device=self.device
        )
        prefix = torch.ones(
            spec.n_passes - 1, rows, dtype=torch.long, device=self.device
        )
        jitter = torch.zeros(
            spec.n_passes,
            rows,
            self.args.seq_len + 1,
            self.args.dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        loop_jitter = None
        if spec.iterations > 1:
            loop_jitter = torch.zeros(
                spec.n_passes,
                spec.iterations - 1,
                rows,
                self.args.seq_len + 1,
                self.args.dim,
                dtype=torch.bfloat16,
                device=self.device,
            )
        return tokens, prefix, jitter, loop_jitter

    def _retained(self, spec: GraphSpec, checkpoint_blocks: int, lean: bool) -> int:
        """Bytes one eager forward of ``spec`` at ``micro_rows`` rows leaves
        allocated once it returns: exactly what its backward would consume."""
        tokens, prefix, jitter, loop_jitter = self._inputs(spec, self.args.micro_rows)
        self.model.checkpoint_blocks = checkpoint_blocks
        self.model.set_recurrence_saving(lean)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        with self.autocast:
            outs = multipass(
                self.model,
                tokens,
                spec.n_passes,
                iterations=spec.iterations,
                prefix_lens=prefix,
                jitter=jitter,
                loop_jitter=loop_jitter,
            )
            result = multipass_loss(
                self.model, tokens, outs, mtp_weight=self.args.mtp_weight
            )
        torch.cuda.synchronize()
        alive = torch.cuda.memory_allocated() - before
        del outs, result
        self.zero_grad()
        torch.cuda.synchronize()
        return alive

    def _calibrate(self, spec: GraphSpec) -> Calibration:
        """The calibration from three eager forwards of the cheapest graph.

        A raw forward divided by its block count is the average a graph
        retains per block, extras included, which the graphs stack per pass
        and per block; the recurrences keeping and rebuilding their
        intermediates give the two averages. The lean forward with its first
        eligible block recomputed releases what every recomputed block
        releases: the PKDA and auxiliary blocks retain more than the average,
        so the difference, not the average, sizes a shortfall in recomputed
        blocks.
        """
        blocks, _ = block_invocations(self.model.cfg, spec)
        full = self._retained(spec, 0, lean=False)
        raw = self._retained(spec, 0, lean=True)
        released = raw - self._retained(spec, 1, lean=True)
        self.model.checkpoint_blocks = 0
        per_block = raw / blocks
        return Calibration(
            full / blocks, per_block, released if released > 0 else per_block
        )

    def _allocate(self, spec: GraphSpec, plan: ReplayPlan) -> CapturedMicro:
        tokens, prefix, jitter, loop_jitter = self._inputs(spec, plan.rows_per_replay)
        sums = StepSums.zeros(self.model, self.device)
        return CapturedMicro(
            spec,
            plan,
            tokens,
            prefix,
            jitter,
            loop_jitter,
            torch.zeros((), dtype=torch.float32, device=self.device),
            sums.loss_sum,
            sums.pass1_sum,
            sums.ntp_sum,
            sums.mtp_sum,
            sums.expert_balance_sum,
            sums.expert_counts,
        )

    def _body(self, state: CapturedMicro) -> None:
        self.model.checkpoint_blocks = state.plan.checkpoint_blocks
        self.model.set_recurrence_saving(state.plan.lean)
        # Each replay's mean loss enters the step in proportion to its rows.
        scale = state.plan.rows_per_replay / self.args.batch_rows
        with self.autocast:
            outs = multipass(
                self.model,
                state.rows,
                state.spec.n_passes,
                iterations=state.spec.iterations,
                prefix_lens=state.prefix,
                jitter=state.jitter,
                loop_jitter=state.loop_jitter,
            )
            loss_result = multipass_loss(
                self.model, state.rows, outs, z_coef=state.z_coef,
                mtp_weight=self.args.mtp_weight,
            )
            loss, losses = loss_result.total, loss_result.ntp
        (loss * scale).backward()
        state.loss_sum.add_(loss.detach() * scale)
        state.pass1_sum.add_(losses[0][0].detach() * scale)
        state.ntp_sum.add_(combine_column_losses(loss_result.ntp).detach() * scale)
        state.mtp_sum.add_(combine_column_losses(loss_result.mtp).detach() * scale)
        # These are outputs of logical forwards, collected outside the
        # checkpointed blocks. Recomputed backwards cannot count again.
        state.expert_counts.add_(loss_result.expert_counts)
        state.expert_balance_sum.add_(loss_result.expert_aux_loss.detach() * scale)

    def _warm(self, state: CapturedMicro) -> set[torch.nn.Parameter]:
        self.zero_grad()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.zero_grad()
                self._body(state)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        return mode_activity(self.gradients, self.sharded, self.model.expert_banks)

    def _capture(self, state: CapturedMicro, pool) -> None:
        self.zero_grad()
        state.loss_sum.zero_()
        state.pass1_sum.zero_()
        state.ntp_sum.zero_()
        state.mtp_sum.zero_()
        state.expert_balance_sum.zero_()
        state.expert_counts.zero_()
        # Capturing on a blocking stream cannot inherit unfinished
        # default-stream writes from optimizer/kernel preparation.
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with _capture_without_gc(graph, pool):
            self._body(state)
        state.graph = graph
        state.loss_sum.zero_()
        state.pass1_sum.zero_()
        state.ntp_sum.zero_()
        state.mtp_sum.zero_()
        state.expert_balance_sum.zero_()
        state.expert_counts.zero_()

    def begin(self, spec: GraphSpec, z_coef: float = 0.0) -> CapturedMicro:
        state = self.states[spec]
        state.z_coef.fill_(z_coef)
        state.loss_sum.zero_()
        state.pass1_sum.zero_()
        state.ntp_sum.zero_()
        state.mtp_sum.zero_()
        state.expert_balance_sum.zero_()
        state.expert_counts.zero_()
        return state

    def replay(self, state: CapturedMicro, rows: torch.Tensor, step: int, first: int):
        state.rows.copy_(rows, non_blocking=rows.is_cuda)
        micro_draws(
            self.args,
            step,
            first,
            state.spec.n_passes,
            state.spec.iterations,
            state.plan.rows_per_replay,
            self.args.dim,
            self.device,
            prefix_out=state.prefix,
            jitter_out=state.jitter,
            loop_jitter_out=state.loop_jitter,
            generator=self.generator,
        )
        state.graph.replay()

    def replay_batch(
        self, state: CapturedMicro, data: TokenData, step: int, first_row: int
    ) -> None:
        first = self.rank_first_row(first_row)
        rows = self.batch_stager.stage(data, first)
        width = state.plan.rows_per_replay
        for offset in range(0, self.rank_rows, width):
            self.replay(state, rows[offset : offset + width], step, first + offset)

    def prepare_optimizer(self, state: CapturedMicro) -> None:  # type: ignore[override]
        super().prepare_optimizer(state.active)


# -- evaluation ----------------------------------------------------------------


@dataclass
class CapturedEval:
    rows: torch.Tensor
    prefix: torch.Tensor | None
    val_sum: torch.Tensor
    fused_sum: torch.Tensor
    mtp_sum: torch.Tensor
    mtp_fused_sum: torch.Tensor
    one_sum: torch.Tensor
    graph: torch.cuda.CUDAGraph | None = None

    def reset(self) -> None:
        for total in (
            self.val_sum, self.fused_sum, self.mtp_sum, self.mtp_fused_sum, self.one_sum
        ):
            total.zero_()


def eval_slice(args, topology: distributed.Topology) -> tuple[int, int]:
    """(first row, rows) of the held-out slice this rank evaluates.

    The rows split evenly across ranks, the last rank taking the remainder;
    the per-row sums all-reduce, so every rank reports the same means.
    """
    per_rank = -(-args.eval_rows // topology.world)
    first = min(topology.rank * per_rank, args.eval_rows)
    return first, max(0, min(per_rank, args.eval_rows - first))


class CudaEvalRunner:
    """No-grad validation graphs sharing the trainer's private memory pool."""

    def __init__(self, model: DeltaModel, args, pool, topology: distributed.Topology):
        self.model = model
        self.args = args
        self.topology = topology
        self.device = next(model.parameters()).device
        self.autocast = torch.autocast("cuda", dtype=torch.bfloat16)
        self.first, self.count = eval_slice(args, topology)
        remainder = self.count % args.micro_rows
        sizes = {min(self.count, args.micro_rows)} - {0}
        if remainder:
            sizes.add(remainder)
        self.states = {size: self._capture(size, pool) for size in sorted(sizes)}

    def _body(self, state: CapturedEval) -> None:
        n_passes = 2 if self.model.cfg.feedback else 1
        with self.autocast:
            outs = multipass(
                self.model,
                state.rows,
                n_passes,
                prefix_lens=state.prefix,
            )
            result = multipass_loss(
                self.model, state.rows, outs, mtp_weight=self.args.mtp_weight
            )
            losses = result.ntp
        rows = state.rows.shape[0]
        # Each pass reads out after its last column, at the evaluation depth;
        # the first pass's first column is the single-column model.
        state.val_sum.add_(losses[0][-1].detach() * rows)
        state.one_sum.add_(losses[0][0].detach() * rows)
        if n_passes > 1:
            state.fused_sum.add_(losses[1][-1].detach() * rows)
        state.mtp_sum.add_(result.mtp[0][-1].detach() * rows)
        if n_passes > 1:
            state.mtp_fused_sum.add_(result.mtp[1][-1].detach() * rows)

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
                if self.model.cfg.feedback
                else None
            ),
            torch.zeros((), dtype=torch.float32, device=self.device),
            torch.zeros((), dtype=torch.float32, device=self.device),
            torch.zeros((), dtype=torch.float32, device=self.device),
            torch.zeros((), dtype=torch.float32, device=self.device),
            torch.zeros((), dtype=torch.float32, device=self.device),
        )
        was_training = self.model.training
        self.model.eval()
        self.model.checkpoint_blocks = 0
        for _ in range(2):
            state.reset()
            self._body(state)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        state.reset()
        with _capture_without_gc(graph, pool):
            self._body(state)
        state.graph = graph
        state.reset()
        self.model.train(was_training)
        return state

    @torch.no_grad()
    def run(self, data_val: TokenData) -> dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        total_val = torch.zeros((), dtype=torch.float32, device=self.device)
        total_fused = torch.zeros_like(total_val)
        total_mtp = torch.zeros_like(total_val)
        total_mtp_fused = torch.zeros_like(total_val)
        total_one = torch.zeros_like(total_val)
        for offset in range(0, self.count, self.args.micro_rows):
            rows = min(self.args.micro_rows, self.count - offset)
            state = self.states[rows]
            state.reset()
            state.rows.copy_(data_val.batch(self.first + offset, rows))
            state.graph.replay()
            total_val.add_(state.val_sum)
            if self.model.cfg.feedback:
                total_fused.add_(state.fused_sum)
            total_mtp.add_(state.mtp_sum)
            total_mtp_fused.add_(state.mtp_fused_sum)
            total_one.add_(state.one_sum)
        totals = torch.stack(
            [total_val, total_fused, total_mtp, total_mtp_fused, total_one]
        )
        distributed.all_reduce_(totals)
        total_val, total_fused, total_mtp, total_mtp_fused, total_one = totals.unbind()
        result = {"val": (total_val / self.args.eval_rows).item()}
        if self.model.cfg.feedback:
            result["val_fused"] = (total_fused / self.args.eval_rows).item()
        result["val_mtp"] = (total_mtp / self.args.eval_rows).item()
        if self.model.cfg.feedback:
            result["val_mtp_fused"] = (total_mtp_fused / self.args.eval_rows).item()
        if self.model.cfg.loop:
            result["val_one"] = (total_one / self.args.eval_rows).item()
        self.model.train(was_training)
        return result


def execution_fields(model, graph_runner, eval_graph_runner) -> dict[str, float]:
    """Production CUDA telemetry without assuming a packed mixer layout.

    ``static_gib`` is this rank's footprint: the working copies, the FP32
    gradient slabs, the replicated NAdam parameters and moments, and the FP32
    masters and NorMuonH state of the matrices this rank owns.
    ``peak_allocated_gib`` is the most memory live at once through
    calibration, warm-up, and capture, so minus the static footprint and the
    deepest graph's retained activations it is the transient the margin has
    to cover; ``reserved_gib`` and ``free_gib`` describe the device once
    capture ends. Read before the peak statistics reset for the step field.
    """
    has_global_attention = any(not block.is_pkda for block in model.blocks)
    return {
        "flash_sdpa": int(has_global_attention),
        "cce": 1,
        "cuda_graphs": len(graph_runner.states) + len(eval_graph_runner.states),
        "eval_graphs": len(eval_graph_runner.states),
        "ranks": graph_runner.topology.world,
        "rank_rows": graph_runner.rank_rows,
        "static_gib": round(graph_runner.static_bytes / 2**30, 2),
        "activation_budget_gib": round(graph_runner.budget_bytes / 2**30, 2),
        "inputs_gib": round(graph_runner.inputs_bytes / 2**30, 2),
        "block_mib": round(graph_runner.calibration.lean_block / 2**20, 1),
        "block_full_mib": round(graph_runner.calibration.full_block / 2**20, 1),
        "checkpoint_mib": round(graph_runner.calibration.checkpoint / 2**20, 1),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
        "free_gib": round(torch.cuda.mem_get_info()[0] / 2**30, 2),
    }


@torch.no_grad()
def evaluate(
    model: DeltaModel,
    data_val: TokenData,
    args,
    device,
    graph_runner: CudaEvalRunner | None = None,
    topology: distributed.Topology = distributed.Topology(),
) -> dict[str, float]:
    """Paired val losses at the evaluation depth: pass 1 always, one fused
    pass on feedback conditions, and the single-column readout under ``l``."""
    if graph_runner is not None:
        return graph_runner.run(data_val)
    model.eval()
    keys = ("val", "val_fused", "val_mtp", "val_mtp_fused", "val_one")
    sums = dict.fromkeys(keys, 0.0)
    start, count = eval_slice(args, topology)
    # CUDA evaluates under the captured graphs' activation precision, whose
    # head reads the BF16 classifier shadow.
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    for offset in range(0, count, args.micro_rows):
        rows = data_val.batch(
            start + offset, min(args.micro_rows, count - offset), device
        )
        n_passes = 2 if model.cfg.feedback else 1
        prefix = torch.ones((1, rows.shape[0]), dtype=torch.long, device=device)
        with autocast:
            outs = multipass(
                model, rows, n_passes, prefix_lens=prefix if n_passes > 1 else None
            )
            result = multipass_loss(model, rows, outs, mtp_weight=args.mtp_weight)
        losses = result.ntp
        weight = rows.shape[0]
        sums["val"] += losses[0][-1].item() * weight
        sums["val_mtp"] += result.mtp[0][-1].item() * weight
        if n_passes > 1:
            sums["val_fused"] += losses[1][-1].item() * weight
            sums["val_mtp_fused"] += result.mtp[1][-1].item() * weight
        if model.cfg.loop:
            sums["val_one"] += losses[0][0].item() * weight
    totals = torch.tensor([sums[key] for key in keys], dtype=torch.float64, device=device)
    distributed.all_reduce_(totals)
    means = dict(zip(keys, (totals / args.eval_rows).tolist(), strict=True))
    model.train()
    wanted = ["val"] + (["val_fused"] if model.cfg.feedback else []) + ["val_mtp"]
    wanted += ["val_mtp_fused"] if model.cfg.feedback else []
    wanted += ["val_one"] if model.cfg.loop else []
    return {key: means[key] for key in wanted}


@torch.no_grad()
def route_summary(model: DeltaModel, data_val: TokenData, args, device) -> list[dict]:
    """Per-site routing observables from one validation microbatch."""
    model.eval()
    rows = data_val.batch(0, min(2, args.eval_rows), device)
    if model.cfg.feedback:
        prefix = torch.ones((1, rows.shape[0]), dtype=torch.long, device=device)
        out = multipass(model, rows, 2, prefix_lens=prefix, want_weights=True)[-1][-1]
    else:
        out = multipass(model, rows, 1, want_weights=True)[0][-1]
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
            layer_kind, sublayer = site.split(".")
            block = model.blocks[int(layer_kind[1:])]
            router = getattr(block, f"{sublayer}_router")
        record["null_rms"] = round(router.null.float().square().mean().sqrt().item(), 4)
        records.append(record)
    model.train()
    return records


@torch.no_grad()
def expert_summary(model: DeltaModel, data_val: TokenData, args, device) -> list[dict]:
    """Actual routed assignment fractions and gate entropy by executed site."""
    was_training = model.training
    model.eval()
    try:
        rows = data_val.batch(0, min(2, args.eval_rows), device)
        with (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        ):
            out = multipass(model, rows, 1, want_weights=True)[0][0]
            # The training geometry: every payload row, next tokens 1..T.
            mtp = model.forward_mtp_fused(out.fused_input, want_weights=True)
        records = []
        weights_by_site = out.expert_weights | {"mtp.experts": mtp.expert_weights}
        for site, weights in weights_by_site.items():
            w = weights.float()
            load = (w > 0).float().mean(dim=(0, 1)) / model.cfg.experts_per_token
            entropy = -(w * w.clamp_min(1e-30).log()).sum(-1).mean()
            record = {
                "site": site,
                "used": int((load > 0).sum()),
                "max_load": round(load.max().item(), 4),
                "min_load": round(load.min().item(), 4),
                "entropy": round(entropy.item(), 4),
            }
            record.update(
                {f"expert{i}": round(value, 4) for i, value in enumerate(load.tolist())}
            )
            bank = (
                model.mtp.block.mlp if site == "mtp.experts"
                else model.blocks[int(site.split(".")[0][1:])].mlp
            )
            record.update({
                f"bias{i}": round(value, 6)
                for i, value in enumerate(bank.expert_bias.tolist())
            })
            records.append(record)
        return records
    finally:
        model.train(was_training)


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


class SnapshotState:
    """The model's checkpoint state with every sharded matrix's FP32 master.

    Rank zero assembles the dict a single process would write: replicated
    parameters and buffers as they are, and each sharded matrix's master from
    its owner, local ones as device views the stager copies asynchronously
    and remote ones as host copies. Other ranks take part in the gather and
    return nothing.
    """

    def __init__(self, model: DeltaModel, sites: ParameterSites, normuonh: NorMuonH):
        self.model = model
        self.sites = sites
        self.normuonh = normuonh

    def state_dict(self) -> dict:
        contents = self.model.state_dict()
        owned = self.sites.owned
        for name, parameter in self.model.named_parameters():
            if parameter not in self.sites.sharded:
                continue
            master = self.normuonh.master_of(parameter) if parameter in owned else None
            gathered = self.sites.collect(
                parameter, master, shape=tuple(parameter.shape), dtype=torch.float32
            )
            if self.sites.topology.main:
                contents[name] = gathered
        return contents if self.sites.topology.main else {}


class SnapshotOptimizer:
    """The optimizer stack's checkpoint state, whole, on rank zero.

    NAdam is replicated and its state dict is the same on every rank.
    NorMuonH's parameter groups are global while each rank's state names the
    matrices it owns, so rank zero gathers every matrix's momentum, row
    moment, radius, and right vector from its owner into one state dict of
    the single-process schema.
    """

    def __init__(self, optimizers, sites: ParameterSites):
        self.optimizers = optimizers
        self.sites = sites

    def state_dict(self) -> dict:
        stack = [
            self._normuonh(optimizer)
            if isinstance(optimizer, NorMuonH)
            else optimizer.state_dict()
            for optimizer in self.optimizers
        ]
        return {"stack": stack} if self.sites.topology.main else {}

    def _normuonh(self, optimizer: NorMuonH) -> dict:
        local = optimizer.state_dict()
        state: dict[int, dict] = {}
        owned = self.sites.owned
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                entry = optimizer.state[parameter] if parameter in owned else None
                gathered = {}
                for name, (shape, dtype) in optimizer.state_shapes(parameter).items():
                    tensor = entry[name] if entry else None
                    value = self.sites.collect(parameter, tensor, shape=shape, dtype=dtype)
                    if self.sites.topology.main:
                        gathered[name] = value
                if self.sites.topology.main:
                    state[optimizer.index_of(parameter)] = gathered
        return {"state": state, "param_groups": local["param_groups"]}


class AsyncSnapshotWriter:
    """Single-flight immutable staging plus background atomic serialization.

    Every rank stages, since the staging gathers the sharded state to rank
    zero, and only rank zero serializes.
    """

    def __init__(self, args, model, sites, optimizers, protected: set[int]):
        self.args = args
        self.topology = sites.topology
        normuonh = next(o for o in optimizers if isinstance(o, NorMuonH))
        self.state = SnapshotState(model, sites, normuonh)
        self.optimizer = SnapshotOptimizer(optimizers, sites)
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
        staged = checkpoints.stage(CONTRACT, self.state, self.optimizer, self.args, step)
        if self.topology.main:
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
    return {
        "vocab_size": args.vocab_size,
        "dim": args.dim,
        "layers": args.layers,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "expert_intermediate": args.expert_intermediate,
        "num_routed_experts": args.num_routed_experts,
        "experts_per_token": args.experts_per_token,
        "pkda_heads": args.pkda_heads,
        "pkda_head_dim": args.pkda_head_dim,
        "pkda_conv_size": args.pkda_conv_size,
        "max_seq_len": args.seq_len + 1,
        "loop_iterations": args.loop_iterations,
    }


def reference_active(args) -> int:
    """Active non-embedding parameters of flat ``f``, including MTP.

    Every condition at a scale shares this denominator and schedule. Count
    the shared expert and configured selected experts per layer, excluding
    idle experts.
    """
    return _reference_active(**model_fields(args))


@lru_cache(maxsize=128)
def _reference_active(**fields: int) -> int:
    """Count once per geometry; parser and schedule calls reuse the result."""
    with torch.device("meta"):
        model = DeltaModel(condition_config("f", **fields))
    total = sum(parameter.numel() for parameter in model.parameters())
    inactive = sum(
        parameter.numel()
        for bank in model.expert_banks
        for expert in bank.experts[bank.experts_per_token :]
        for parameter in expert.parameters()
    )
    return total - model.embed_tokens.weight.numel() - inactive


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

    ``--scale`` fills geometry and batch fields left unset;
    a retyped ``--seq-len`` without ``--batch-rows`` keeps ``BATCH_TOKENS``
    predictions per step; and a fresh run's ``--steps`` is derived from
    ``--tokens-per-param`` unless typed. Returns the settled arguments and the
    destinations the operator pinned: what they typed, plus what an explicit
    ``--scale`` or ``--tokens-per-param`` fixed, which a resume validates
    against the checkpoint rather than inherits from it.
    """
    args = parser.parse_args(argv)
    if args.seq_len < 2:
        parser.error("two-token prediction requires --seq-len at least 2")
    if 2 * args.three_rate > 1:
        parser.error("--three-rate covers two roll outcomes; it must be at most 0.5")
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


def pick_device(
    name: str | None, topology: distributed.Topology = distributed.Topology()
) -> torch.device:
    """The device of this rank: a named one, else CUDA by local rank, MPS, CPU."""
    if name:
        device = torch.device(name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", topology.local_rank)
        elif topology.world > 1 and device.index != topology.local_rank:
            raise ValueError(
                f"--device {name} names one CUDA device, but a {topology.world}-rank "
                "invocation places each rank on its own local device; give a bare "
                "'cuda' or leave --device unset"
            )
        torch.cuda.set_device(device)
    return device


def build_schedule(args) -> Schedule:
    """WSD as the shared four-phase Schedule (no preheat), with ratio
    nudges so integer spans land exactly."""
    total = args.steps
    # Warmup is fixed per scale: the fraction applies to the shorter of the
    # run and the 25x recipe at its geometry, so a longer run keeps the 25x
    # run's warmup and is that run's continuation exactly, while a shorter
    # one keeps the fraction. Cooldown and the feedback boundary stay
    # fractions of the run.
    warmup = round(
        args.warmup_frac * min(total, schedule_steps(args, DEFAULT_TOKENS_PER_PARAM))
    )
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


def train(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    args, pinned = resolve_run_args(parser, argv)
    topology = distributed.Topology.from_environment()
    if args.ranks != topology.world:
        raise ValueError(
            f"--ranks {args.ranks} but this invocation has {topology.world} "
            "process(es); launch through torchrun or run `delta train` directly"
        )
    if not topology.main:
        telemetry.silence()
    tags = (args.tag, args.continue_from) if args.continue_from else (args.tag,)
    # The tag lock belongs to one process; the other ranks share its fate.
    lock = runs.lock_tags(args.out_dir, *tags) if topology.main else contextlib.nullcontext()
    try:
        with lock:
            return _train(args, pinned, topology)
    finally:
        distributed.shutdown()


def _train(
    args: argparse.Namespace, pinned: frozenset[str], topology: distributed.Topology
) -> dict:
    device = pick_device(args.device, topology)
    distributed.initialize(topology, device)
    if device.type == "cuda":
        # TF32 for the FP32 matmuls that remain outside autocast: NorMuonH's
        # spectral power iterations and the FP32 diagnostics. The Newton-Schulz
        # loop runs in BF16 on CUDA and the trunk runs BF16 under autocast.
        torch.set_float32_matmul_precision("high")

    start_step = 0
    payload = None
    if args.resume:
        path = runs.latest_snapshot(args.tag, args.out_dir)
        payload = read_checkpoint(path)
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
        payload = read_checkpoint(latest_path)
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
            payload = read_checkpoint(path)

    if args.micro_rows < 1:
        raise ValueError("micro-rows must be positive")
    if args.batch_rows % (topology.world * args.micro_rows):
        raise ValueError(
            "batch-rows must be a multiple of ranks times micro-rows, so every "
            "rank takes the same whole replays of every step"
        )
    schedule = build_schedule(args)
    total = schedule.total

    data_directory = Path(args.data_root) / args.source
    meta = read_meta(data_directory)
    if payload is not None and args.tokenizer_id != meta["tokenizer_id"]:
        raise ValueError("checkpoint and token store use different tokenizers")
    args.tokenizer_id = meta["tokenizer_id"]
    if args.tokenizer_id == TOKENIZER_ID and args.vocab_size != VOCAB_SIZE:
        raise ValueError(f"NeoX ChatML requires model vocab {VOCAB_SIZE}")
    if meta["vocab_size"] > args.vocab_size:
        raise ValueError(
            f"data vocab {meta['vocab_size']} exceeds model vocab {args.vocab_size}"
        )
    data_train = TokenData.load(data_directory, "train", args.seq_len)
    data_val = TokenData.load(data_directory, "val", args.seq_len)
    needed = total * args.batch_rows
    if needed > data_train.rows:
        raise ValueError(f"schedule needs {needed} rows, stream has {data_train.rows}")

    torch.manual_seed(args.seed)
    model = DeltaModel(condition_config(args.condition, **model_fields(args))).to(
        device
    )
    sites = ParameterSites(model, topology)
    optimizers = build_optimizers(
        model,
        lr_normuonh=args.lr_normuonh,
        lr_nadam=args.lr_nadam,
        owned=sites.owned,
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
            data_root=args.data_root,
            source=args.source,
            params=sum(p.numel() for p in model.parameters()),
            device=str(device),
            ranks=topology.world,
            routing_block_size=model.cfg.routing_block_size,
            mup_ratio=model.cfg.mup_ratio,
            expert_lr_scale=model.cfg.expert_lr_scale,
            expert_shared=1,
            expert_routed=model.cfg.num_routed_experts,
            expert_top_k=model.cfg.experts_per_token,
            expert_width=model.cfg.expert_intermediate,
            expert_balance_coef=EXPERT_BALANCE_COEF,
            expert_bias_rate=EXPERT_BIAS_RATE,
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

    # Persistent snapshots at the cooldown boundary, the recurrence boundary
    # (the last single-column state), and the end, so cooldown and recurrence
    # variants can continue from the exact pre-boundary state.
    protected = {schedule.heat_end, recurrence_boundary(args, total), total} - {0}
    end_step = total
    if args.max_steps is not None:
        end_step = min(total, start_step + args.max_steps)
    telemetry.log(
        "schedule",
        warmup_steps=schedule.warmup_steps,
        preheat_steps=schedule.preheat_steps,
        heat_steps=schedule.heat_steps,
        cooldown_steps=schedule.cooldown_steps,
        recurrence_boundary=recurrence_boundary(args, total),
        start_step=start_step,
        end_step=end_step,
        total_steps=total,
    )
    model.train()
    sites.allocate()
    eval_graph_runner = None
    if device.type == "cuda":
        trainer = CudaGraphTrainer(model, optimizers, sites, args, schedule, topology)
        eval_graph_runner = CudaEvalRunner(model, args, trainer.pool, topology)
        telemetry.log(
            "execution",
            **execution_fields(model, trainer, eval_graph_runner),
        )
        torch.cuda.reset_peak_memory_stats()
    else:
        trainer = EagerTrainer(model, optimizers, sites, args, topology)
    process_start = time.monotonic()
    window_start, window_tokens, window_pass_tokens = process_start, 0, 0
    window_cell_tokens = 0.0
    summary: dict = {}
    interrupted = False
    snapshot_writer = AsyncSnapshotWriter(args, model, sites, optimizers, protected)

    try:
        with distributed.Interrupt() as interrupt:
            for step in range(start_step + 1, end_step + 1):
                snapshot_writer.poll()
                learning_rates = apply_schedule(optimizers, schedule, step)
                phase = schedule.phase(step)[0]
                z_coef = args.zloss if phase == "cooldown" else 0.0
                n_passes, iterations = step_shape(args, step, total, model.cfg)

                spec = GraphSpec(n_passes, iterations)
                state = trainer.begin(spec, z_coef)
                trainer.replay_batch(state, data_train, step, (step - 1) * args.batch_rows)
                sites.reduce_gradients()
                # The step's sums and the stop flag settle in two collectives.
                floats = torch.stack(
                    [
                        state.loss_sum, state.pass1_sum, state.ntp_sum,
                        state.mtp_sum, state.expert_balance_sum,
                    ]
                )
                integers = torch.cat(
                    [
                        state.expert_counts.flatten(),
                        torch.tensor([int(interrupt.requested)], device=device),
                    ]
                )
                distributed.all_reduce_(floats)
                distributed.all_reduce_(integers)
                step_loss, pass1_loss, ntp_loss, mtp_loss, expert_balance = (
                    floats.tolist()
                )
                expert_counts = integers[:-1].view_as(state.expert_counts)
                stop = bool(integers[-1].item())
                trainer.prepare_optimizer(state)
                grad_norm = sites.gradient_norm()
                with trainer.pool_scope():
                    for optimizer in optimizers:
                        optimizer.step()
                    model.update_expert_bias(expert_counts)
                sites.gather_weights()
                model.refresh_shadows()
                trainer.zero_grad()
                # A pass executes ``iterations`` columns of ``routing_blocks``
                # cells, so cell-tokens beside pass-tokens keep matched data apart
                # from matched compute when the loop runs more columns.
                cells = iterations * model.cfg.routing_blocks
                window_tokens += args.batch_rows * args.seq_len
                window_pass_tokens += n_passes * args.batch_rows * args.seq_len
                window_cell_tokens += cells * n_passes * args.batch_rows * args.seq_len

                elapsed = time.monotonic() - window_start
                fields = {
                    "step": telemetry.step_address(step, total),
                    "phase": phase,
                    "loss": telemetry.format_metric(step_loss),
                    "pass1": telemetry.format_metric(pass1_loss),
                    "ntp": telemetry.format_metric(ntp_loss),
                    "k": n_passes,
                }
                fields["mtp"] = telemetry.format_metric(mtp_loss)
                fields["expert_balance"] = telemetry.format_metric(expert_balance)
                # Each physical bank has its own target; every bank executes once
                # per column, repeatedly on looped steps.
                loads = expert_counts.float()
                violation = (
                    model.cfg.num_routed_experts
                    * loads.amax(dim=-1)
                    / loads.sum(dim=-1).clamp_min(1)
                    - 1
                )
                fields["expert_max_violation"] = telemetry.format_metric(
                    violation.max().item()
                )
                fields["expert_bias_max"] = telemetry.format_metric(
                    torch.stack([bank.expert_bias for bank in model.expert_banks])
                    .abs().max().item()
                )
                if model.cfg.loop:
                    fields["r"] = iterations
                fields |= {
                    "lr_normuonh": telemetry.format_metric(learning_rates["normuonh"]),
                    "lr_normuonh_expert_in": telemetry.format_metric(
                        learning_rates["normuonh_expert_in"]
                    ),
                    "lr_normuonh_expert_out": telemetry.format_metric(
                        learning_rates["normuonh_expert_out"]
                    ),
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

                # A stop agreed this step snapshots first: the launcher and the
                # spool give the ranks a bounded grace, and the monitors are
                # not part of the state.
                if (step % args.eval_every == 0 or step == total) and not stop:
                    address = telemetry.step_address(step, total)
                    scores = evaluate(
                        model, data_val, args, device,
                        graph_runner=eval_graph_runner, topology=topology,
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
                    # The monitors run eagerly on one rank and allocate only
                    # transients, so they take the pool too; ``evaluate``
                    # replays graphs and therefore stays outside.
                    if topology.main:
                        with trainer.pool_scope():
                            for record in route_summary(model, data_val, args, device):
                                telemetry.log("route", step=address, **record)
                            for record in expert_summary(model, data_val, args, device):
                                telemetry.log("expert", step=address, **record)
                            if model.cfg.feedback:
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
                                by_depth = {
                                    record["iterations"]: record for record in sweep
                                }
                                telemetry.log(
                                    "depth",
                                    step=address,
                                    r_eval=model.cfg.loop_iterations,
                                    r_max=LOOP_MAX_ITERATIONS,
                                    loss_one=telemetry.format_metric(by_depth[1]["loss"]),
                                    loss_eval=telemetry.format_metric(
                                        by_depth[model.cfg.loop_iterations]["loss"]
                                    ),
                                    loss_max=telemetry.format_metric(sweep[-1]["loss"]),
                                    upd_max=telemetry.format_metric(
                                        sweep[-1]["update_norm"]
                                    ),
                                )

                snapshotted = step % args.snapshot_every == 0 or step in protected
                if snapshotted:
                    snapshot_writer.submit(step)
                summary["step"] = step
                summary["loss"] = step_loss
                if stop:
                    interrupted = True
                    if not snapshotted:
                        snapshot_writer.submit(step)
                    telemetry.log("interrupt", step=telemetry.step_address(step, total))
                    break
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


def main(argv: list[str] | None = None) -> None:
    """Run one invocation, under the launcher when it asks for several ranks.

    Both ``delta train`` and ``python -m delta_feedback_experiment.train``
    enter here: a direct ``--ranks N`` invocation replaces itself with the
    launcher, which runs this module once per device with the same arguments.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    distributed.relaunch(__spec__.name, argv, parse_run_args(argv).ranks)
    train(argv)


if __name__ == "__main__":
    main()

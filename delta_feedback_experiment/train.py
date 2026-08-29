"""One arm's training run under the binding recipe.

The loop is a pure function of (init seed, data seed, step): data rows
are step-addressed slices of the fixed stream, feedback randomness
(pass counts, prefix lengths, jitter) derives from keyed generators
rather than ambient RNG state, and the WSD schedule is the shared
:class:`Schedule` addressed by cumulative step.  That is what makes a
resumed invocation bit-identical to an uninterrupted one and every arm's
batches the same bytes (the paired-comparison contract; arms must share
``data_seed``, ``batch_rows``, and ``micro_rows``).

Operational layer — telemetry records, immutable ``runs``-addressed
snapshots, resume reconciliation — comes from ``transformer_experiments``.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformer_experiments import checkpoints, runs, telemetry
from transformer_experiments.schedule import Schedule

from .data import TokenData, read_meta
from .model import ARMS, DFModel, arm_config, iterate_fused, multipass, multipass_loss
from .optim import OptimizerPair, apply_schedule, build_optimizers

CONTRACT = checkpoints.CheckpointContract(
    version=2, resumable=frozenset({2}), surface_version=2
)

EXACT_FIELDS = (
    "arm",
    "seed",
    "data_seed",
    "seq_len",
    "batch_rows",
    "micro_rows",
    "steps",
    "warmup_steps",
    "cooldown_frac",
    "feedback_start",
    "three_pass",
    "lr_muon",
    "wd_muon",
    "lr_adam",
    "jitter",
    "zloss",
    "vocab_size",
    "dim",
    "layers",
    "heads",
    "kv_heads",
    "head_dim",
    "intermediate",
)
"""State-defining settings: a resume takes these from the checkpoint."""

RUNTIME_FIELDS = (
    "data_dir",
    "out_dir",
    "device",
    "log_every",
    "eval_every",
    "snapshot_every",
    "eval_rows",
)
"""Per-invocation settings: inherited unless retyped."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "df train", description="Train one arm of the DF factorial."
    )
    parser.add_argument("tag", type=runs.validate_run_tag, help="run tag")
    parser.add_argument("--arm", choices=ARMS, default="vanilla")
    parser.add_argument("--seed", type=int, default=1, help="init seed")
    parser.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="shared randomness stream; identical across paired arms",
    )
    parser.add_argument("--data-dir", default="data/tokens")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the tag from its latest snapshot",
    )

    schedule = parser.add_argument_group("schedule (state-defining)")
    schedule.add_argument("--steps", type=runs.parse_step_count, default=6700)
    schedule.add_argument("--warmup-steps", type=int, default=200)
    schedule.add_argument("--cooldown-frac", type=float, default=0.25)
    schedule.add_argument(
        "--feedback-start",
        type=float,
        default=0.75,
        help="fraction of steps before feedback passes begin",
    )
    schedule.add_argument(
        "--three-pass",
        type=float,
        default=0.12,
        help="P(k=3) within the feedback phase (0.12 -> 75/22/3 overall)",
    )

    recipe = parser.add_argument_group("recipe (state-defining)")
    recipe.add_argument(
        "--batch-rows",
        type=int,
        default=292,
        help="global batch in rows (292 x 1025 tokens ~ FBT's 300K)",
    )
    recipe.add_argument("--micro-rows", type=int, default=4)
    recipe.add_argument("--seq-len", type=int, default=1024)
    recipe.add_argument("--lr-muon", type=float, default=1e-2)
    recipe.add_argument("--wd-muon", type=float, default=0.01)
    recipe.add_argument("--lr-adam", type=float, default=5e-4)
    recipe.add_argument("--jitter", type=float, default=0.02)
    recipe.add_argument("--zloss", type=float, default=1e-5)

    trunk = parser.add_argument_group("trunk (state-defining)")
    trunk.add_argument("--vocab-size", type=int, default=151936)
    trunk.add_argument("--dim", type=int, default=768)
    trunk.add_argument("--layers", type=int, default=12)
    trunk.add_argument("--heads", type=int, default=8)
    trunk.add_argument("--kv-heads", type=int, default=4)
    trunk.add_argument("--head-dim", type=int, default=96)
    trunk.add_argument("--intermediate", type=int, default=3072)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--max-steps",
        type=runs.parse_step_count,
        default=None,
        help="cap this invocation's additional steps; never rescales the schedule",
    )
    runtime.add_argument("--device", default=None)
    runtime.add_argument("--log-every", type=int, default=10)
    runtime.add_argument("--eval-every", type=int, default=100)
    runtime.add_argument("--snapshot-every", type=int, default=500)
    runtime.add_argument("--eval-rows", type=int, default=32)
    return parser


# -- deterministic derived randomness ------------------------------------------


def mix(*parts: int) -> int:
    """A stable 63-bit mix of integers (no salted ``hash``)."""
    value = 0
    for part in parts:
        value = (value ^ (part + 0x9E3779B97F4A7C15)) * 0xBF58476D1CE4E5B9 % (1 << 63)
        value ^= value >> 27
    return value


def draw_passes(args, step: int, total: int) -> int:
    """The step's pass count — shared across every feedback-bearing arm."""
    if step <= round(args.feedback_start * total):
        return 1
    generator = torch.Generator().manual_seed(mix(args.data_seed, step, 1))
    return 3 if torch.rand((), generator=generator).item() < args.three_pass else 2


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
    across arms for any run sharing the batch geometry."""
    if generator is None:
        generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(mix(args.data_seed, step, first_row))
    columns = args.seq_len + 1
    shape = (n_passes - 1, n_rows)
    if prefix_out is None:
        prefix_out = torch.empty(shape, dtype=torch.long, device=device)
    torch.randint(1, columns, shape, generator=generator, out=prefix_out)
    jitter_shape = (n_passes - 1, n_rows, columns, dim)
    if jitter_out is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        jitter_out = torch.empty(jitter_shape, dtype=dtype, device=device)
    jitter_out.uniform_(-args.jitter, args.jitter, generator=generator)
    return prefix_out, jitter_out


def automatic_checkpoint(model: DFModel, n_passes: int, args, device) -> bool:
    """Measured internal activation policy; the Jobe screen fits k<=3 raw."""
    if device.type != "cuda":
        return False
    cfg = model.cfg
    screen_work = 4 * 1025 * 768 * 12 * 3
    work = args.micro_rows * (args.seq_len + 1) * cfg.dim * cfg.layers * n_passes
    return work > screen_work


@dataclass(frozen=True)
class GraphSpec:
    n_passes: int
    want_z: bool
    checkpoint: bool


@dataclass
class CapturedMicro:
    spec: GraphSpec
    rows: torch.Tensor
    prefix: torch.Tensor | None
    jitter: torch.Tensor | None
    loss_sum: torch.Tensor
    pass1_sum: torch.Tensor
    graph: torch.cuda.CUDAGraph | None = None
    active: frozenset[torch.nn.Parameter] = frozenset()


class CudaGraphTrainer:
    """Fixed-address forward/backward graphs for every reachable step mode.

    The model's gradient tensors and every graph input are allocated once.
    Data and keyed CUDA randomness are copied/drawn into those addresses before
    replay.  Graphs share a private pool and never overlap; each captured body
    ends after backward, so no saved activation survives between replays.
    """

    def __init__(self, model: DFModel, optimizers, args, schedule: Schedule):
        self.model = model
        self.optimizers = optimizers
        self.args = args
        self.device = next(model.parameters()).device
        self.micros = args.batch_rows // args.micro_rows
        self.autocast = torch.autocast("cuda", dtype=torch.bfloat16)
        self.generator = torch.Generator(device=self.device)
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.states: dict[GraphSpec, CapturedMicro] = {}
        specs = self._reachable_specs(schedule)
        for spec in specs:
            self.states[spec] = self._allocate(spec)

        active_by_spec = {
            spec: self._warm(state) for spec, state in self.states.items()
        }
        model.zero_grad(set_to_none=True)
        union = set().union(*active_by_spec.values())
        self.grad_buffers = {p: torch.zeros_like(p) for p in union}
        for parameter, gradient in self.grad_buffers.items():
            parameter.grad = gradient

        pool = torch.cuda.graph_pool_handle()
        for spec, state in self.states.items():
            state.active = frozenset(active_by_spec[spec])
            self._capture(state, pool)
        self.zero_grad()
        for optimizer in self.optimizers:
            warmup = getattr(optimizer, "warmup", None)
            if warmup is not None:
                warmup()
        torch.cuda.synchronize()

    def _reachable_specs(self, schedule: Schedule) -> list[GraphSpec]:
        specs = set()
        for step in range(1, schedule.total + 1):
            want_z = schedule.phase(step)[0] == "cooldown" and self.args.zloss > 0
            n_passes = (
                draw_passes(self.args, step, schedule.total)
                if self.model.cfg.feedback_active
                else 1
            )
            specs.add(
                GraphSpec(
                    n_passes,
                    want_z,
                    automatic_checkpoint(self.model, n_passes, self.args, self.device),
                )
            )
        return sorted(specs, key=lambda spec: (spec.n_passes, spec.want_z))

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
            )
            z_coef = self.args.zloss if state.spec.want_z else 0.0
            loss, losses = multipass_loss(self.model, state.rows, outs, z_coef=z_coef)
        (loss / self.micros).backward()
        state.loss_sum.add_(loss.detach() / self.micros)
        state.pass1_sum.add_(losses[0].detach() / self.micros)

    def _warm(self, state: CapturedMicro) -> set[torch.nn.Parameter]:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.model.zero_grad(set_to_none=True)
                self._body(state)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        return {p for p in self.parameters if p.grad is not None}

    def _capture(self, state: CapturedMicro, pool) -> None:
        for gradient in self.grad_buffers.values():
            gradient.zero_()
        state.loss_sum.zero_()
        state.pass1_sum.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            self._body(state)
        state.graph = graph
        state.loss_sum.zero_()
        state.pass1_sum.zero_()

    def begin(self, spec: GraphSpec) -> CapturedMicro:
        state = self.states[spec]
        state.loss_sum.zero_()
        state.pass1_sum.zero_()
        return state

    def replay(self, state: CapturedMicro, rows: torch.Tensor, step: int, first: int):
        state.rows.copy_(rows)
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

    def prepare_optimizer(self, state: CapturedMicro) -> None:
        for parameter in self.parameters:
            parameter.grad = (
                self.grad_buffers.get(parameter) if parameter in state.active else None
            )

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            gradient = self.grad_buffers.get(parameter)
            parameter.grad = gradient
            if gradient is not None:
                gradient.zero_()


# -- evaluation ----------------------------------------------------------------


@torch.no_grad()
def evaluate(model: DFModel, data_val: TokenData, args, device) -> dict[str, float]:
    """Paired val losses: pass-1 always; one fused pass on feedback arms."""
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
def route_summary(model: DFModel, data_val: TokenData, args, device) -> list[dict]:
    """Per-site routing observables from one val microbatch (a fused pass
    on feedback arms, so DF-soft's p_prev source is present)."""
    if not model.cfg.routing_active:
        return []
    model.eval()
    rows = data_val.batch(0, min(2, args.eval_rows), device)
    if model.cfg.feedback_active:
        prefix = torch.ones((1, rows.shape[0]), dtype=torch.long, device=device)
        out = multipass(model, rows, 2, prefix_lens=prefix, want_weights=True)[-1]
    else:
        out = multipass(model, rows, 1, want_weights=True)[0]
    soft = model.cfg.soft
    records = []
    for site, weights in out.route_weights.items():
        mean = weights.float().mean(dim=(1, 2))
        record = {
            "site": site,
            "n": weights.shape[0],
            "max": round(weights.float().max(dim=0).values.mean().item(), 4),
        }
        offset = 0
        if soft:
            record["null"] = round(mean[0].item(), 4)
            offset = 1
        if site == "payload":
            records.append(record)
            continue
        if out.n_seeds == 2:  # standing p_prev present (DF-soft fused pass)
            record["prev"] = round(mean[offset].item(), 4)
        record["seed"] = round(mean[offset + out.n_seeds - 1].item(), 4)
        records.append(record)
    model.train()
    return records


# -- checkpointing -------------------------------------------------------------


def save_snapshot(args, model, pair, step: int, protected: set[int]) -> Path:
    path = runs.snapshot_path(args.tag, step, args.out_dir)
    checkpoints.save(path, CONTRACT, model, pair, args, step)
    telemetry.log("checkpoint", step=step, path=str(path), kind="snapshot")
    existing = runs.snapshots(args.tag, args.out_dir)
    keep = {s for s, _ in existing[-2:]} | protected
    for snapshot_step, snapshot_path in existing:
        if snapshot_step not in keep:
            snapshot_path.unlink()
    return path


# -- the run -------------------------------------------------------------------


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
    cooldown = round(args.cooldown_frac * total)
    heat = total - args.warmup_steps - cooldown
    if heat < 1:
        raise ValueError(f"steps={total} leaves no stable phase")
    schedule = Schedule(
        heat=heat,
        warmup=(args.warmup_steps + 1e-9) / heat,
        cooldown=(cooldown + 1e-9) / heat,
    )
    assert schedule.warmup_steps == args.warmup_steps
    assert schedule.cooldown_steps == cooldown
    assert schedule.total == total
    return schedule


def train(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    device = pick_device(args.device)

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
        explicit = checkpoints.explicit_destinations(parser, argv)
        conflicts = checkpoints.mismatched_fields(
            saved, args, EXACT_FIELDS, only=explicit
        )
        if conflicts:
            raise ValueError(f"resume conflicts with checkpoint settings: {conflicts}")
        checkpoints.inherit(
            args,
            saved,
            exact_fields=EXACT_FIELDS,
            runtime_fields=RUNTIME_FIELDS,
            explicit=explicit,
        )

    if args.batch_rows % args.micro_rows:
        raise ValueError("batch-rows must be a multiple of micro-rows")
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
    model = DFModel(
        arm_config(
            args.arm,
            vocab_size=args.vocab_size,
            dim=args.dim,
            layers=args.layers,
            heads=args.heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            intermediate=args.intermediate,
            max_seq_len=args.seq_len + 1,
        )
    ).to(device)
    optimizers = build_optimizers(
        model, lr_muon=args.lr_muon, wd_muon=args.wd_muon, lr_adam=args.lr_adam
    )
    pair = OptimizerPair(optimizers)

    if payload is not None:
        start_step = checkpoints.restore(
            payload, CONTRACT, model, pair, current_optimizer_groups=False
        )
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
            **{name: getattr(args, name) for name in EXACT_FIELDS},
        )

    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    protected = {schedule.heat_end, total}
    end_step = total
    if args.max_steps is not None:
        end_step = min(total, start_step + args.max_steps)
    telemetry.log(
        "schedule",
        warmup_steps=schedule.warmup_steps,
        preheat_steps=schedule.preheat_steps,
        heat_steps=schedule.heat_steps,
        cooldown_steps=schedule.cooldown_steps,
        start_step=start_step,
        end_step=end_step,
        total_steps=total,
    )
    model.train()
    graph_runner = None
    if device.type == "cuda":
        graph_runner = CudaGraphTrainer(model, optimizers, args, schedule)
        torch.cuda.reset_peak_memory_stats()
        telemetry.log(
            "execution",
            flash=int(model.blocks[0].attn.qkv_proj.weight.is_cuda),
            cce=1,
            cuda_graphs=len(graph_runner.states),
            checkpoint_modes=sum(spec.checkpoint for spec in graph_runner.states),
        )
    process_start = time.monotonic()
    window_start, window_tokens, window_pass_tokens = process_start, 0, 0
    summary: dict = {}
    interrupted = False

    try:
        for step in range(start_step + 1, end_step + 1):
            lr = apply_schedule(optimizers, schedule, step)
            phase = schedule.phase(step)[0]
            z_coef = args.zloss if phase == "cooldown" else 0.0
            n_passes = 1
            if model.cfg.feedback_active:
                n_passes = draw_passes(args, step, total)
            checkpointing = automatic_checkpoint(model, n_passes, args, device)

            micros = args.batch_rows // args.micro_rows
            if graph_runner is not None:
                spec = GraphSpec(n_passes, z_coef > 0, checkpointing)
                graph_state = graph_runner.begin(spec)
                for micro in range(micros):
                    first_row = (step - 1) * args.batch_rows + micro * args.micro_rows
                    rows = data_train.batch(first_row, args.micro_rows)
                    graph_runner.replay(graph_state, rows, step, first_row)
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
                        )
                        loss, losses = multipass_loss(model, rows, outs, z_coef=z_coef)
                    (loss / micros).backward()
                    step_loss += loss.item() / micros
                    pass1_loss += losses[0].item() / micros

            logging = step % args.log_every == 0 or step == total
            grad_norm = None
            if logging:
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                norms = torch._foreach_norm(gradients)
                grad_norm = torch.stack(norms).norm().item()
            for optimizer in optimizers:
                optimizer.step()
            if graph_runner is not None:
                graph_runner.zero_grad()
            else:
                model.zero_grad(set_to_none=True)
            window_tokens += args.batch_rows * args.seq_len
            window_pass_tokens += n_passes * args.batch_rows * args.seq_len

            if logging:
                elapsed = time.monotonic() - window_start
                fields = {
                    "step": telemetry.step_address(step, total),
                    "phase": phase,
                    "loss": telemetry.format_metric(step_loss),
                    "pass1": telemetry.format_metric(pass1_loss),
                    "k": n_passes,
                    "lr": telemetry.format_metric(lr),
                    "gnorm": telemetry.format_metric(grad_norm),
                    "tok_s": f"{window_tokens / max(elapsed, 1e-9):.0f}",
                    "pass_tok_s": (f"{window_pass_tokens / max(elapsed, 1e-9):.0f}"),
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

            if step % args.eval_every == 0 or step == total:
                address = telemetry.step_address(step, total)
                scores = evaluate(model, data_val, args, device)
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

            if step % args.snapshot_every == 0 or step in protected:
                save_snapshot(args, model, pair, step, protected)
            summary["step"] = step
            summary["loss"] = step_loss
    except KeyboardInterrupt:
        interrupted = True
        step = summary.get("step", start_step)
        if step > start_step:
            save_snapshot(args, model, pair, step, protected)
        telemetry.log("interrupt", step=telemetry.step_address(step, total))

    if not interrupted:
        last = summary.get("step", start_step)
        if last < total:
            if last > start_step and last % args.snapshot_every:
                save_snapshot(args, model, pair, last, protected)
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

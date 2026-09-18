"""Downstream zero-shot tasks on a run's snapshot: ``delta eval TAG``.

Uses the workspace's ``transformer_experiments.downstream`` tasks (pinned Hub
revisions, harness-identical prompts) with this model's own scorer: one plain
column pass (Standard), a plain pass followed by a fully fused pass with
plain-prefix length 1 (Fused, the mode ``val_fused`` reports), or a plain pass
followed by a fused pass whose plain prefix is each item's context, so only the
scored continuation receives feedback (Soft, the Jacobi form of feedback
decoding: exact for the first continuation token, and for the first ``k``
tokens after ``k`` passes).  ``--passes k`` iterates the feedback pass ``k``
times, each consuming the previous pass's payload, and scores the last pass.
Batches are padded to a few bucket lengths so the compiled blocks see few
shapes.

Every mode the snapshot's condition supports runs by default, against one
loaded model.  Each task emits a ``downstream`` record as it finishes, and each
mode writes ``figures/downstream-TAG/MODE.STEP.json`` (``MODEk.STEP.json`` for
``k > 1`` passes); compare two files with
``python -m transformer_experiments.downstream --compare``.  A ``--limit`` run
is a smoke and writes nothing.  The queue runs this module as the phase after
a finished training schedule.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformer_experiments import downstream, runs, telemetry

from . import analysis
from .tokenizer import TOKENIZER_ID, load_tokenizer, tokenizer_metadata

MODES = ("standard", "fused", "soft")
FIGURES = Path("figures")


class DeltaScorer:
    """Continuation scores from a DeltaModel column pass under the trainer's numerics."""

    def __init__(self, model, mode: str, passes: int = 1):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != "standard" and not model.cfg.feedback:
            raise ValueError(f"{mode} mode needs a snapshot of a condition with f")
        if passes < 1:
            raise ValueError("passes must be positive")
        self.model = model
        self.mode = mode
        self.passes = passes if mode != "standard" else 0
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def score(self, ids, spans):
        model = self.model
        ids = ids.to(self.device)
        with analysis.autocast(self.device):
            e = model.embed_tokens(ids)
            # Under l every pass runs the evaluation column count and scores
            # its last column.
            out = model.forward_iterations(
                model.plain_seed(e), need_payload=self.passes > 0
            )[-1]
            if self.mode == "soft":
                # Plain context, feedback along the scored continuation only.
                prefix = torch.tensor([start for start, _ in spans], device=self.device)
            else:
                prefix = 1
            for i in range(self.passes):
                out = model.forward_iterations(
                    analysis.fused_inputs(model, e, out.payload, prefix),
                    need_payload=i < self.passes - 1,
                )[-1]
            weight = model.embed_tokens.weight
            head = lambda h: F.linear(model.readout_input(h), weight.to(h.dtype))
            return downstream.span_scores(out.h_top, ids, spans, head)


def condition_modes(model) -> tuple[str, ...]:
    """The scoring modes a snapshot's condition supports."""
    return MODES if model.cfg.feedback else MODES[:1]


def result_path(tag: str, step: int, mode: str, passes: int = 1) -> Path:
    """Where one mode's results for one snapshot live under ``figures/``."""
    suffix = f"{mode}{passes}" if mode != "standard" and passes > 1 else mode
    return FIGURES / f"downstream-{tag}" / f"{suffix}.{step}.json"


def task_fields(result: downstream.TaskResult) -> dict[str, str]:
    """One task's headline numbers as telemetry fields."""
    fields = {"n": str(result.n)}
    for name, metric in result.metrics.items():
        key = "ppl" if name == "perplexity" else name
        fields[key] = telemetry.format_metric(metric["mean"])
    return fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "delta eval",
        description="Score a run's snapshot on the downstream zero-shot tasks.",
    )
    parser.add_argument("tag")
    parser.add_argument(
        "--mode",
        dest="modes",
        nargs="+",
        choices=MODES,
        metavar="MODE",
        default=None,
        help=f"scoring modes from {', '.join(MODES)} (default: every mode the "
        "snapshot's condition supports)",
    )
    parser.add_argument(
        "--step", type=int, default=None, help="snapshot step (default: the latest)"
    )
    parser.add_argument(
        "--passes", type=int, default=1, help="feedback passes in fused or soft mode"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(downstream.TASKS),
        metavar="TASK",
        default=list(downstream.DEFAULT_TASKS),
        help=f"tasks from {', '.join(downstream.TASKS)} (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="documents per task; a smoke that writes no results",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--buckets", type=int, nargs="+", default=(128, 256, 512, 1024))
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default="runs", help="snapshot directory")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.step is None:
        path = runs.latest_snapshot(args.tag, args.out_dir)
    else:
        path = runs.snapshot_path(args.tag, args.step, args.out_dir)
        if not path.is_file():
            raise FileNotFoundError(f"no snapshot of {args.tag} at step {args.step}: {path}")

    model, saved = analysis.load_checkpoint(path, args.device)
    if saved["tokenizer_id"] != TOKENIZER_ID:
        raise ValueError("downstream text evaluation requires a NeoX ChatML snapshot")
    scorers = [
        DeltaScorer(model, mode, args.passes)
        for mode in args.modes or condition_modes(model)
    ]
    tokenize = downstream.hf_tokenize(load_tokenizer())
    max_len = min(saved["seq_len"], max(args.buckets))
    buckets = [b for b in args.buckets if b <= max_len]
    address = telemetry.step_address(saved["step"], saved["steps"])
    requests: dict[str, list[downstream.Request]] = {}
    started = time.time()

    def progress(line: str) -> None:
        print(f"  [{time.time() - started:6.0f}s] {line}", flush=True)

    for scorer in scorers:
        print(f"{telemetry.HEADING}{scorer.mode}: {path}", flush=True)
        results = {}
        for name in args.tasks:
            task = downstream.TASKS[name]
            if name not in requests:
                requests[name] = downstream.load_requests(task, args.limit)
            results[name] = downstream.run_task(
                task,
                requests[name],
                tokenize,
                scorer,
                batch_size=args.batch_size,
                max_len=max_len,
                buckets=buckets,
                progress=progress,
            )
            telemetry.log(
                "downstream",
                step=address,
                mode=scorer.mode,
                passes=scorer.passes,
                task=name,
                **task_fields(results[name]),
            )
        print("\n" + downstream.format_table(results), flush=True)
        if args.limit is not None:
            print(f"--limit {args.limit} is a smoke; nothing written", flush=True)
            continue
        out_path = result_path(saved["tag"], saved["step"], scorer.mode, args.passes)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "snapshot": str(path),
            "tag": saved["tag"],
            "condition": saved["condition"],
            "step": saved["step"],
            "mode": scorer.mode,
            "passes": scorer.passes,
            **tokenizer_metadata(),
            "tokenizer_id": TOKENIZER_ID,
            "device": str(scorer.device),
            "batch_size": args.batch_size,
            "buckets": list(args.buckets),
        }
        out_path.write_text(
            json.dumps(downstream.to_json(results, **meta), indent=1) + "\n"
        )
        print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

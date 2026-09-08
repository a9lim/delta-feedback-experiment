"""``df`` — the operator entry point.

Model-lazy: subcommands import what they need, so ``df --help`` costs
nothing and the spool worker drives phases as separate processes.
Durable orchestration (queue, worker, status/watch/stop/clear) is the
shared ``transformer_experiments.spool``; this module owns only the
pipeline: an offline probe, then the training run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformer_experiments import spool

ROOT = Path(__file__).resolve().parents[1]

USAGE = """\
df — delta-feedback experiment operator

  df train TAG [FLAGS]      train one condition directly (see df train --help)
  df tokenize [FLAGS]       build the fixed token stream (once)
  df probe                  run the offline invariant suite
  df queue TAG [FLAGS]      append a training job to the detached spool
  df queue FILE             append jobs from a file (TAG FLAGS per line)
  df status                 print queue and recent-run state
  df watch                  follow milestones until the queue is idle
  df stop TAG|live [--at STEP]
  df stop queue|all
  df clear TAG|all          move an idle tag's artifacts to recovery
"""


def _train_args(job: spool.Job) -> tuple[str, ...]:
    return job.argv[0]


def _resumes(job: spool.Job) -> bool:
    return "--resume" in _train_args(job)


def _validate_job(job: spool.Job) -> None:
    from .train import build_parser

    build_parser().parse_args([job.tag, *_train_args(job)])


PIPELINE = spool.Pipeline(
    slots=("train",),
    validate=_validate_job,
    phases=(
        spool.Phase(
            name="probe",
            module="delta_feedback_experiment.probe",
            argv=lambda job: [],
            log="{tag}.probe.log",
        ),
        spool.Phase(
            name="train run",
            module="delta_feedback_experiment.train",
            argv=lambda job: [job.tag, *_train_args(job)],
            log="{tag}.log",
            resume=_resumes,
            telemetry=True,
            report_exit=True,
        ),
    ),
)

LAYOUT = spool.Layout(
    root=ROOT,
    worker_module="delta_feedback_experiment.cli",
    snapshot_dir=Path("runs"),
)

SPOOL = spool.Spool(LAYOUT, PIPELINE, prog="df")

STOP_PHASE = "train run"


def tokenize_command(argv: list[str]) -> None:
    from .data import (
        CANONICAL_CONFIG,
        CANONICAL_DATASET_REVISION,
        CANONICAL_TARGET_TOKENS,
        CANONICAL_TOKENIZER_REVISION,
        CANONICAL_VAL_TOKENS,
    )

    parser = argparse.ArgumentParser("df tokenize")
    parser.add_argument("--out", default="data/tokens")
    parser.add_argument(
        "--target",
        type=float,
        default=CANONICAL_TARGET_TOKENS,
        help="total tokenization target, including the held-out prefix",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=CANONICAL_VAL_TOKENS,
        help="held-out tokens from the stream head",
    )
    parser.add_argument("--config", default=CANONICAL_CONFIG)
    parser.add_argument("--revision", default=CANONICAL_DATASET_REVISION)
    parser.add_argument("--tokenizer-revision", default=CANONICAL_TOKENIZER_REVISION)
    args = parser.parse_args(argv)
    from .data import tokenize

    tokenize(
        args.out,
        target_tokens=int(args.target),
        val_tokens=int(args.val),
        config=args.config,
        revision=args.revision,
        tokenizer_revision=args.tokenizer_revision,
    )


def probe_command(argv: list[str]) -> None:
    from .probe import main

    main(argv)


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE, end="")
        return
    command, rest = argv[0], argv[1:]
    if command == "_worker":
        SPOOL.worker()
    elif command == "train":
        from .train import train

        train(rest)
    elif command == "tokenize":
        tokenize_command(rest)
    elif command == "probe":
        probe_command(rest)
    elif command == "queue":
        SPOOL.queue_command(rest, resumes=_resumes)
    elif command == "status":
        SPOOL.status()
    elif command == "watch":
        SPOOL.watch()
    elif command == "stop":
        target, at = SPOOL.parse_stop(rest)
        SPOOL.stop(target, at=at, phase=STOP_PHASE)
    elif command == "clear":
        if len(rest) != 1:
            raise SystemExit("usage: df clear TAG|all")
        SPOOL.clear(rest[0])
    else:
        print(USAGE, end="", file=sys.stderr)
        raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main()

"""``delta`` — the operator entry point.

Model-lazy: subcommands import what they need, so ``delta --help`` costs
nothing and the spool worker drives phases as separate processes.
Durable orchestration (queue, worker, status/watch/stop/clear) is the
shared ``transformer_experiments.spool``; this module owns only the
pipeline: an offline probe, then the training run.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

from transformer_experiments import spool

ROOT = Path(__file__).resolve().parents[1]

USAGE = """\
delta — delta-feedback experiment operator

  delta train TAG [FLAGS]  train one condition directly (delta train --help)
  delta tokenize [FLAGS]   build a source token stream prefix (once)
  delta verify DIR         check a token store against its meta and sidecars
  delta probe              run the offline invariant suite
  delta queue TAG [FLAGS]  append a training job to the detached spool
  delta queue FILE         append jobs from a file (TAG FLAGS per line)
  delta queue TAG --continue SRC [FLAGS]
                           extend finished run SRC to a longer schedule under TAG
  delta status             print queue and recent-run state
  delta watch              follow milestones until the queue is idle
  delta stop TAG|live [--at STEP]
  delta stop queue|all
  delta clear TAG|all      move an idle tag's artifacts to recovery
  delta move OLD NEW       rename an idle run and its artifacts
"""


def _train_args(job: spool.Job) -> tuple[str, ...]:
    return job.argv[0]


def _resumes(job: spool.Job) -> bool:
    return "--resume" in _train_args(job)


def _validate_job(job: spool.Job) -> None:
    from .train import parse_run_args

    parse_run_args([job.tag, *_train_args(job)])


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

SPOOL = spool.Spool(LAYOUT, PIPELINE, prog="delta")

STOP_PHASE = "train run"

RUN_ARTIFACTS = (
    *(f"figures/{kind}-{{tag}}" for kind in ("route", "fused", "downstream", "depth")),
    *(
        pattern
        for kind in ("compare", "weights", "curves")
        for pattern in (
            f"figures/{kind}-{{tag}}-vs-*",
            f"figures/{kind}-*-vs-{{tag}}",
            f"figures/{kind}-*-vs-{{tag}}-vs-*",
        )
    ),
)


def move_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        "delta move",
        description=SPOOL.help("move").split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", metavar="OLD")
    parser.add_argument("destination", metavar="NEW")
    parser.add_argument(
        "--out-dir", type=Path, help="snapshot directory (default runs)"
    )
    args = parser.parse_args(argv)
    layout = replace(
        SPOOL.layout,
        snapshot_dir=args.out_dir or SPOOL.layout.snapshot_dir,
        artifact_globs=RUN_ARTIFACTS,
    )
    try:
        spool.Spool(layout, SPOOL.pipeline, prog="delta").move(
            args.source, args.destination
        )
    except (ValueError, OSError) as error:
        parser.exit(1, f"delta move: {error}\n")


def tokenize_command(argv: list[str]) -> None:
    from .data import (
        CANONICAL_SHUFFLE_SEED,
        CANONICAL_TARGET_TOKENS,
        CANONICAL_TOKENIZER_REVISION,
        CANONICAL_TOKENS_PER_DOC,
        CANONICAL_VAL_TOKENS,
        DEFAULT_SOURCE,
        SOURCES,
    )

    parser = argparse.ArgumentParser(
        "delta tokenize",
        description=(
            "Build a reproducible prefix of --source: index "
            "the pinned parquet files, select and tokenize the documents "
            "whose stream position falls below --target / --tokens-per-doc, "
            "then write the held-out slice, the train shards, and the document "
            "sidecars. Resumable; refuses to overwrite a finished store."
        ),
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="parent directory for source stores (default data)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="explicit output directory (default DATA_ROOT/SOURCE)",
    )
    parser.add_argument(
        "--source",
        choices=tuple(SOURCES),
        default=DEFAULT_SOURCE,
        help="pinned dataset and file subset (default dclm-100b, already shuffled)",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="shuffle documents across the source; default off for dclm-100b, "
        "on for other sources; --no-shuffle keeps published file/row order",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="stored tokens to write, held-out slice included (default 57e9, "
        "the screen's 400x schedule; the bridge's needs 167e9)",
    )
    parser.add_argument(
        "--scale",
        choices=("screen", "bridge", "flagship"),
        help="derive --target from this scale's schedule at --tokens-per-param: "
        "its rows of seq_len + 1 tokens plus the held-out slice, rounded up to "
        "the next billion",
    )
    parser.add_argument("--tokens-per-param", type=float)
    parser.add_argument(
        "--continue",
        dest="extend",
        action="store_true",
        help="extend the store at --out to the target in place, appending the "
        "stream's next documents; every other setting must match its meta",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=CANONICAL_VAL_TOKENS,
        help="held-out tokens: the chosen stream's first documents",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CANONICAL_SHUFFLE_SEED,
        help="key of the optional document shuffle; prefix matching also "
        "requires the same source, revision, tokenizer and ordering mode",
    )
    parser.add_argument(
        "--tokens-per-doc",
        type=int,
        default=CANONICAL_TOKENS_PER_DOC,
        help="conservative mean document length used to size the selection; "
        "does not truncate documents; the build fails if the selection runs out",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="files tokenized concurrently; each worker prefetches one next file",
    )
    parser.add_argument(
        "--readers",
        type=int,
        default=8,
        help="concurrent document reads during shuffled assembly (default 8)",
    )
    parser.add_argument(
        "--scratch",
        default=None,
        help="download directory, removed on success (default OUT/scratch)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="override the selected source's pinned revision",
    )
    parser.add_argument("--tokenizer-revision", default=CANONICAL_TOKENIZER_REVISION)
    args = parser.parse_args(argv)
    if (args.scale is None) != (args.tokens_per_param is None):
        parser.error("--scale and --tokens-per-param go together")
    if args.scale is not None:
        if args.target is not None:
            parser.error("--target and --scale name the target two ways")
        target = stream_target(args.scale, args.tokens_per_param, int(args.val))
    else:
        target = int(CANONICAL_TARGET_TOKENS if args.target is None else args.target)
    from .data import tokenize

    tokenize(
        args.out or str(Path(args.data_root) / args.source),
        target_tokens=target,
        val_tokens=int(args.val),
        seed=args.seed,
        tokens_per_doc=args.tokens_per_doc,
        workers=args.workers,
        readers=args.readers,
        scratch=args.scratch,
        source_name=args.source,
        shuffle=args.shuffle,
        revision=args.revision,
        tokenizer_revision=args.tokenizer_revision,
        extend=args.extend,
    )


BILLION = 1_000_000_000


def stream_target(scale: str, tokens_per_param: float, val_tokens: int) -> int:
    """Stored tokens for a scale's schedule at a ratio: its rows of
    ``seq_len + 1`` tokens plus the held-out slice, rounded up to the next
    billion so the store reads as a round figure and leaves headroom."""
    from .train import parse_run_args

    run = parse_run_args(
        ["stream", "--scale", scale, "--tokens-per-param", f"{tokens_per_param:g}"]
    )
    needed = run.steps * run.batch_rows * (run.seq_len + 1) + val_tokens
    return math.ceil(needed / BILLION) * BILLION


def verify_command(argv: list[str]) -> None:
    import json

    parser = argparse.ArgumentParser(
        "delta verify",
        description="Check a token store against its meta and sidecars.",
    )
    parser.add_argument("directory")
    args = parser.parse_args(argv)
    from .data import verify

    print(json.dumps(verify(args.directory), indent=2))


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
    elif command == "verify":
        verify_command(rest)
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
            raise SystemExit("usage: delta clear TAG|all")
        SPOOL.clear(rest[0])
    elif command == "move":
        move_command(rest)
    else:
        print(USAGE, end="", file=sys.stderr)
        raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main()

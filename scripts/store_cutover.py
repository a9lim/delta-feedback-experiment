"""Retire old token stores once nothing on the spool reads them.

    python scripts/store_cutover.py --new /data/delta/dclm-100b \\
        --old /data/delta/tokens --old /data/delta/tokens-smoke [--wait] [--delete]

Verifies the new store, then refuses while an active or pending job's
arguments name an old path. ``--wait`` polls until none does;
``--delete`` removes the old paths, otherwise the script only reports. A run
trained on a deleted store cannot be resumed or ``--continue``d afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from transformer_experiments import checkpoints, runs

from delta_feedback_experiment.cli import SPOOL
from delta_feedback_experiment.data import verify
from delta_feedback_experiment.train import build_parser


def busy(old: list[Path]) -> list[str]:
    """Spool jobs, active or pending, whose arguments name an old store."""
    names = {path.resolve() for path in old}
    reasons = []
    active = SPOOL.layout.active
    candidates = [("active", active)] if active.exists() else []
    candidates += [("pending", path) for path in SPOOL.pending_paths()]
    for state, path in candidates:
        job = json.loads(path.read_text())
        words = [word for vector in job.get("argv", []) for word in vector]
        parser = build_parser()
        args = parser.parse_args([job["tag"], *words])
        explicit = checkpoints.explicit_destinations(parser, words)
        if (args.resume or args.continue_from) and not {
            "data_root",
            "source",
        } <= explicit:
            snapshots = Path(args.out_dir)
            if not snapshots.is_absolute():
                snapshots = SPOOL.layout.root / snapshots
            try:
                snapshot = runs.latest_snapshot(
                    args.continue_from or args.tag, snapshots
                )
                # Only inspect saved settings; mapped tensors need not be read.
                saved = torch.load(
                    snapshot, map_location="cpu", weights_only=False, mmap=True
                )["args"]
                for field in ("data_root", "source"):
                    if field not in explicit:
                        setattr(args, field, saved[field])
            except (FileNotFoundError, KeyError, ValueError, RuntimeError, EOFError):
                reasons.append(f"{state} reader unresolved: {job['tag']}")
                continue
        directory = Path(args.data_root) / args.source
        if not directory.is_absolute():
            directory = SPOOL.layout.root / directory
        if directory.resolve() in names:
            reasons.append(f"{state} reader: {job['tag']}")
    return reasons


def size(path: Path) -> int:
    return sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, files in os.walk(path)
        for name in files
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--new", required=True, type=Path, help="the store to keep")
    parser.add_argument(
        "--old", action="append", default=[], type=Path, help="a store to retire"
    )
    parser.add_argument(
        "--wait", action="store_true", help="poll every five minutes until clear"
    )
    parser.add_argument("--delete", action="store_true", help="remove the old stores")
    args = parser.parse_args(argv)

    summary = verify(args.new)
    print(json.dumps(summary, indent=2))
    old = [path for path in args.old if path.exists()]
    for path in args.old:
        if path not in old:
            print(f"{path}: already absent")
    while True:
        reasons = busy(old)
        if not reasons:
            break
        print(
            "waiting on " + "; ".join(reasons)
            if args.wait
            else "blocked by " + "; ".join(reasons)
        )
        if not args.wait:
            sys.exit(1)
        time.sleep(300)
    for path in old:
        print(
            f"{path}: {size(path) / 1e9:.1f} GB",
            "removed" if args.delete else "would be removed",
        )
        if args.delete:
            shutil.rmtree(path)


if __name__ == "__main__":
    main()

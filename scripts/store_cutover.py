"""Retire old token stores once nothing on the spool reads them.

    python scripts/store_cutover.py --new /data/delta/tokens-350B \\
        --old /data/delta/tokens --old /data/delta/tokens-smoke [--wait] [--delete]

Verifies the new store, then refuses while a job is active or a pending job's
arguments name an old path. ``--wait`` polls until the spool is clear;
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

from delta_feedback_experiment.cli import SPOOL
from delta_feedback_experiment.data import verify


def busy(old: list[Path]) -> list[str]:
    """Spool jobs that stand in the way: the active one, or a pending reader."""
    reasons = []
    active = SPOOL.layout.active
    if active.exists():
        reasons.append(f"active: {json.loads(active.read_text())['tag']}")
    names = {str(path) for path in old}
    for pending in SPOOL.pending_paths():
        job = json.loads(pending.read_text())
        words = [word for vector in job.get("argv", []) for word in vector]
        if any(word in names for word in words):
            reasons.append(f"pending reader: {job['tag']}")
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
        print("waiting on " + "; ".join(reasons) if args.wait else "blocked by " + "; ".join(reasons))
        if not args.wait:
            sys.exit(1)
        time.sleep(300)
    for path in old:
        print(f"{path}: {size(path) / 1e9:.1f} GB", "removed" if args.delete else "would be removed")
        if args.delete:
            shutil.rmtree(path)


if __name__ == "__main__":
    main()

"""``df`` — the operator entry point.

Model-lazy: subcommands import what they need, so ``df --help`` costs
nothing and the spool can drive phases as separate processes.
"""

from __future__ import annotations

import argparse
import sys

USAGE = """\
df — delta-feedback experiment operator

  df train TAG [FLAGS]     train one arm (see df train --help)
  df tokenize [FLAGS]      build the fixed token stream (once)
  df probe                 run the offline invariant suite
"""


def tokenize_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser("df tokenize")
    parser.add_argument("--out", default="data/tokens")
    parser.add_argument("--target", type=float, default=35e9,
                        help="train tokens to write")
    parser.add_argument("--val", type=float, default=30e6,
                        help="held-out tokens from the stream head")
    parser.add_argument("--config", default="sample-100BT")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args(argv)
    from .data import tokenize

    tokenize(
        args.out,
        target_tokens=int(args.target),
        val_tokens=int(args.val),
        config=args.config,
        revision=args.revision,
    )


def probe_command(argv: list[str]) -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", *argv], check=False
    )
    raise SystemExit(result.returncode)


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE, end="")
        return
    command, rest = argv[0], argv[1:]
    if command == "train":
        from .train import train

        train(rest)
    elif command == "tokenize":
        tokenize_command(rest)
    elif command == "probe":
        probe_command(rest)
    else:
        print(USAGE, end="", file=sys.stderr)
        raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main()

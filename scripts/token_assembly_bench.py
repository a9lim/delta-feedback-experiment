"""Read-only gather benchmark over a build's completed parts.

Run before the build removes its parts. Uses distinct seeded document samples
in alternating mode order; does not evict shared caches or write token files.
This measures the gather stage under the host's current cache/I/O load, not
end-to-end build speed. A separate repeated-input check verifies exact bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from delta_feedback_experiment.data import _gather_tokens


def read_bytes() -> int | None:
    path = Path("/proc/self/io")
    if not path.exists():
        return None
    fields = dict(line.split(": ") for line in path.read_text().splitlines())
    return int(fields["read_bytes"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", type=Path)
    parser.add_argument("--docs-per-part", type=int, default=32)
    parser.add_argument("--readers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=310)
    args = parser.parse_args()
    if args.docs_per_part < 1 or args.readers < 1:
        parser.error("docs-per-part and readers must be positive")
    paths = sorted(args.parts.glob("*.index.npy"))
    if not paths:
        parser.error("no completed part indices found")
    indices, maps = [], []
    for path in paths:
        indices.append(np.load(path, mmap_mode="r"))
        tokens = path.with_name(path.name.replace(".index.npy", ".tokens.bin"))
        maps.append(
            np.memmap(tokens, dtype=np.uint32, mode="r")
            if tokens.stat().st_size
            else np.empty(0, dtype=np.uint32)
        )

    def trial(advice: str, readers: int, seed: int, *, parity: bool = False) -> str:
        rng = np.random.default_rng(seed)
        docs = []
        for part, index in enumerate(indices):
            for row in rng.choice(
                len(index), min(args.docs_per_part, len(index)), replace=False
            ):
                record = index[row]
                docs.append(
                    (
                        int(record["position"]),
                        part,
                        int(record["offset"]),
                        int(record["length"]),
                    )
                )
        if not docs:
            raise ValueError("all completed parts are empty")
        docs.sort()
        hint = getattr(mmap, f"MADV_{advice.upper()}", None)
        for array in maps:
            if array.size and hint is not None:
                array._mmap.madvise(hint)
        pieces = [maps[part][offset : offset + size] for _, part, offset, size in docs]
        before = read_bytes()
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=readers) as pool:
            output = _gather_tokens(pieces, pool, readers)
        elapsed = time.perf_counter() - started
        after = read_bytes()
        disk_bytes = None if before is None or after is None else after - before
        digest = hashlib.sha256(output).hexdigest()
        print(
            json.dumps(
                {
                    "advice": advice,
                    "advice_supported": hint is not None,
                    "readers": readers,
                    "seed": seed,
                    "parity_only": parity,
                    "docs": len(docs),
                    "output_bytes": output.nbytes,
                    "seconds": elapsed,
                    "MB_per_second": output.nbytes / elapsed / 1e6,
                    "disk_read_bytes": disk_bytes,
                    "read_amplification": None
                    if disk_bytes is None
                    else disk_bytes / output.nbytes,
                    "sha256": digest,
                }
            ),
            flush=True,
        )
        return digest

    modes = [("normal", 1), ("random", 1), ("random", args.readers)]
    for i, (advice, readers) in enumerate(modes + modes[::-1]):
        trial(advice, readers, args.seed + i)
    optimized = trial("random", args.readers, args.seed + 6, parity=True)
    reference = trial("normal", 1, args.seed + 6, parity=True)
    if optimized != reference:
        raise AssertionError("gather changed token bytes")


if __name__ == "__main__":
    main()

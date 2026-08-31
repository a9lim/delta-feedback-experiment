"""delta-feedback-experiment: MHDB x FBT on a PKDA/GGQA hybrid trunk.

Does widening the depth axis (Multi-Head Delta Block routing) and the
token-time axis (full-bandwidth latent feedback) of a transformer's compute
lattice help complementarily or redundantly? See docs/architecture.md for the
model and docs/design.md for the experiment.
"""

from __future__ import annotations

import os
from pathlib import Path

# Inductor's generated code is PyTorch-, CUDA-, and GPU-specific. Keep this
# project's expensive fixed-shape autotuning cache durable across the probe and
# train subprocesses without pretending it is a source artifact. An explicit
# operator setting still wins when a different cache volume is appropriate.
INDUCTOR_CACHE_DIR = Path(
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        str(Path.home() / ".cache" / "delta-feedback" / "torchinductor"),
    )
)
INDUCTOR_MODE = "max-autotune-no-cudagraphs"

__version__ = "0.4.0"

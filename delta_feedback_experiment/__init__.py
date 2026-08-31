"""delta-feedback-experiment: MHDB x FBT on a PKDA/GGQA hybrid trunk.

Does widening the depth axis (Multi-Head Delta Block routing) and the
token-time axis (full-bandwidth latent feedback) of a transformer's compute
lattice help complementarily or redundantly? See docs/architecture.md for the
model and docs/design.md for the experiment.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Inductor's generated code is PyTorch-, CUDA-, and GPU-specific. Keep this
# project's expensive fixed-shape autotuning cache durable across the probe and
# train subprocesses without pretending it is a source artifact. PyTorch sets
# its own default under /tmp as soon as an earlier torch.compile user imports;
# replace that ephemeral default while retaining any durable operator setting.
_configured_cache = os.environ.get("DF_INDUCTOR_CACHE_DIR") or os.environ.get(
    "TORCHINDUCTOR_CACHE_DIR"
)
_temporary_roots = {
    Path(tempfile.gettempdir()).resolve(),
    Path("/tmp").resolve(),
}
_configured_path = (
    Path(_configured_cache).expanduser().resolve() if _configured_cache else None
)
if _configured_path is None or any(
    _configured_path.is_relative_to(root) for root in _temporary_roots
):
    _configured_path = Path.home() / ".cache" / "delta-feedback" / "torchinductor"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(_configured_path)
INDUCTOR_CACHE_DIR = _configured_path
INDUCTOR_MODE = "max-autotune-no-cudagraphs"

__version__ = "0.4.0"

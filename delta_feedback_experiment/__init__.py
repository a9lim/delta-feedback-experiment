"""delta-feedback-experiment: a nursery for a small recurrent model organism.

One ``DeltaModel`` combines a PKDA/gated-GQA trunk, MoE channel mixers,
MHDB block-delta reads, and sequential two-token prediction. Conditions
select feedback (``f``), tied depth (``l``), or both (``fl``).
See docs/architecture.md for the model and docs/design.md for the recipe.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Inductor's generated code is PyTorch-, CUDA-, and GPU-specific. Keep this
# training autotuning cache durable across subprocesses. PyTorch sets
# its own default under /tmp as soon as an earlier torch.compile user imports;
# replace that ephemeral default while retaining any durable operator setting.
_configured_cache = os.environ.get("DELTA_INDUCTOR_CACHE_DIR") or os.environ.get(
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

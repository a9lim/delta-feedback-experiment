"""Inductor settings shared by every compiled region, and one upstream fix.

``INDUCTOR_MODE`` turns on template autotuning and coordinate-descent tuning
of the generated kernels' block sizes. Split-scan kernels (Inductor's
lowering of a long ``cumsum``, such as the routers' membership ranks over
every token assignment) size their global-memory workspace at codegen for
``rnumel / config.triton.min_split_scan_rblock`` programs and record that
minimum in ``inductor_meta`` under ``min_split_scan_rblock``; the tuner's
block floor and the register-spill halving read ``min_rblock`` instead, so
nothing stops them from trying a smaller ``R0_BLOCK`` whose extra programs
write flags past the workspace. On the 4090 the search never crossed the
bound; on a GH200 it did, and the overrun reached an unmapped page (an MMU
fault under every recipe, reproducible with a 15 x 12,288 ``cumsum`` alone
under ``max_autotune`` plus ``coordinate_descent_tuning``). Recording the
same minimum under the key the tuner reads closes both paths; ``setdefault``
retires the fix once upstream records it.
"""

from __future__ import annotations

from torch._inductor import config as inductor_config
from torch._inductor.codegen.triton_split_scan import TritonSplitScanKernel

INDUCTOR_MODE = "max-autotune-no-cudagraphs"

_upstream_meta = TritonSplitScanKernel.inductor_meta_common.__func__


@classmethod
def _split_scan_meta(cls) -> dict:
    meta = _upstream_meta(cls)
    meta.setdefault("min_rblock", inductor_config.triton.min_split_scan_rblock)
    return meta


TritonSplitScanKernel.inductor_meta_common = _split_scan_meta

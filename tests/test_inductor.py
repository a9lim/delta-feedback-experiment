"""The split-scan tuner floor is recorded under the key the tuner reads."""

import pytest

from torch._inductor import config as inductor_config
from torch._inductor.codegen.triton_split_scan import TritonSplitScanKernel
from torch._inductor.codegen.triton import TritonKernel

import delta_feedback_experiment.inductor  # noqa: F401  (applies the fix)


def test_split_scan_meta_carries_the_workspace_minimum():
    pytest.importorskip("triton")  # the meta hashes the Triton backend
    meta = TritonSplitScanKernel.inductor_meta_common()
    assert meta["min_rblock"] == inductor_config.triton.min_split_scan_rblock
    assert meta["min_split_scan_rblock"] == meta["min_rblock"]
    # Other kernels keep their own floors (or none).
    assert "min_rblock" not in TritonKernel.inductor_meta_common()

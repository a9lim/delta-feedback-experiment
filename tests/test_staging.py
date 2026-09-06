"""Addressing and DMA-lifetime checks for whole-update token staging."""

import pytest
import torch

from delta_feedback_experiment.data import TokenData, write_synthetic
from delta_feedback_experiment.train import CudaBatchStager


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Pinned asynchronous CUDA copy"
)
def test_staging_reuses_storage_without_overwriting_inflight_batches(tmp_path):
    write_synthetic(tmp_path, train_tokens=4096, val_tokens=256, vocab=97)
    data = TokenData.load(tmp_path, "train", 16)
    stager = CudaBatchStager(4, 17, torch.device("cuda"))
    pointers = stager.host.data_ptr(), stager.device.data_ptr()
    assert stager.host.is_pinned()
    copies = []
    # No caller synchronization between uploads. The next stage must protect
    # the pinned host bytes and preserve the earlier device consumer's input.
    for first in (0, 4, 8, 12, 16, 20, 24, 28, 0):
        # Make even these tiny uploads wait behind queued device work so a
        # missing host-reuse fence cannot pass merely because DMA was fast.
        torch.cuda._sleep(10_000_000)
        staged = stager.stage(data, first)
        copies.append((first, staged.clone()))
        assert (stager.host.data_ptr(), staged.data_ptr()) == pointers
    for first, value in copies:
        assert torch.equal(value.cpu(), data.batch(first, 4))

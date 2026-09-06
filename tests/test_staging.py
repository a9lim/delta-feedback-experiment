"""Addressing and DMA-lifetime checks for whole-update token staging."""

from types import SimpleNamespace

import pytest
import torch

from delta_feedback_experiment.data import TokenData, write_synthetic
from delta_feedback_experiment.train import (
    CudaBatchStager,
    CudaGraphTrainer,
    GraphSpec,
    micro_draws,
)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Keyed CUDA graph inputs")
def test_staged_replay_preserves_addressed_rows_and_feedback_draws(tmp_path):
    write_synthetic(tmp_path, train_tokens=4096, val_tokens=256)
    data = TokenData.load(tmp_path, "train", 16)
    runner = object.__new__(CudaGraphTrainer)
    runner.args = SimpleNamespace(
        batch_rows=12, micro_rows=4, seq_len=16, dim=32, data_seed=9, jitter=0.02
    )
    runner.device = torch.device("cuda")
    runner.generator = torch.Generator(device="cuda")
    runner.batch_stager = CudaBatchStager(12, 17, runner.device)
    state = runner._allocate(GraphSpec(3, False))
    observed = []

    class RecordReplay:
        def replay(self):
            observed.append(
                (state.rows.clone(), state.prefix.clone(), state.jitter.clone())
            )

    state.graph = RecordReplay()
    runner.replay_batch(state, data, step=172, first_row=5)
    for first, (rows, prefix, jitter) in zip((5, 9, 13), observed, strict=True):
        expected_prefix, expected_jitter = micro_draws(
            runner.args, 172, first, 3, 4, 32, runner.device
        )
        assert torch.equal(rows.cpu(), data.batch(first, 4))
        assert torch.equal(prefix, expected_prefix)
        assert torch.equal(jitter, expected_jitter)

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
    # A dense model stub needs no expert counts or head accumulator.
    runner.model = SimpleNamespace(cfg=SimpleNamespace(experts=False))
    runner.head_accum = None
    runner.head_flush_every = 1
    runner._head_pending = 0
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Persistent head buffer")
@pytest.mark.parametrize(
    ("passes", "window", "expected_flushes"),
    [
        # The window counts head calls, and a k-pass replay makes k of them,
        # so no window ever holds more than ``window`` BF16 contributions.
        (1, 1, 7),
        (1, 2, 4),
        (1, 3, 3),
        (1, 16, 1),
        (2, 4, 4),
        (3, 4, 7),
        (3, 6, 4),
    ],
)
def test_head_accumulator_reaches_the_sink_once_per_window(
    tmp_path, passes, window, expected_flushes
):
    """Every head call's classifier gradient must land in the FP32 sink
    exactly once, whatever the cadence and whatever remainder it leaves, and
    no flush may carry more contributions than the window allows."""
    write_synthetic(tmp_path, train_tokens=4096, val_tokens=256)
    data = TokenData.load(tmp_path, "train", 16)
    runner = object.__new__(CudaGraphTrainer)
    runner.args = SimpleNamespace(
        batch_rows=28, micro_rows=4, seq_len=16, dim=8, data_seed=9, jitter=0.02
    )
    runner.device = torch.device("cuda")
    runner.generator = torch.Generator(device="cuda")
    runner.batch_stager = CudaBatchStager(28, 17, runner.device)
    runner.head_accum = torch.zeros(5, 3, device="cuda", dtype=torch.bfloat16)
    runner.head_flush_every = window
    runner._head_pending = 0
    sink = torch.zeros(5, 3, device="cuda")
    runner.model = SimpleNamespace(
        cfg=SimpleNamespace(experts=False),
        embed_tokens=SimpleNamespace(grad_sink=sink),
    )
    flushes = []
    state = runner._allocate(GraphSpec(passes, False))

    class AccumulatingReplay:
        def replay(self):
            # Stand in for the head's lock-add, once per pass.
            runner.head_accum.add_(float(passes))

    state.graph = AccumulatingReplay()
    original = runner._drain_head_accum

    def drain():
        flushes.append(int(runner.head_accum[0, 0].item()))
        original()

    runner._drain_head_accum = drain
    runner.replay_batch(state, data, step=3, first_row=0)
    calls = 7 * passes
    assert len(flushes) == expected_flushes
    assert sum(flushes) == calls
    assert max(flushes) <= window
    assert torch.equal(sink, torch.full((5, 3), float(calls), device="cuda"))
    assert not runner.head_accum.any()

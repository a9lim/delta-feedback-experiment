"""Checkpoint-analysis helpers shared by the scripts under ``scripts/``.

Analysis must reproduce the trainer's numbers before it interprets a
checkpoint, so everything here is the trainer's own convention: the snapshot
loader rebuilds the condition from the saved state-defining arguments, evaluation
runs under the same BF16 autocast the captured CUDA evaluation uses (FP32 on
portable devices), per-token cross-entropies read the tied readout through
``final_norm``, and fused inputs are built exactly as ``multipass`` builds
them (shift the payload right, fuse through the FBT entry, keep the plain
prefix).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import DeltaModel, ModelConfig, condition_config, shift_right
from .train import model_fields, pick_device, read_checkpoint


def saved_args(payload: dict) -> dict:
    """A snapshot's settings, plus its ``step``."""
    saved = dict(payload["args"])
    saved["step"] = payload["step"]
    return saved


def config_from_args(saved: dict) -> ModelConfig:
    """The configuration a snapshot's saved arguments describe."""
    return condition_config(
        saved["condition"],
        **model_fields(SimpleNamespace(**saved)),
    )


def load_checkpoint(
    path: str | Path, device: torch.device | str | None = None
) -> tuple[DeltaModel, dict]:
    """Rebuild a snapshot's model for analysis.

    Returns the model in evaluation mode on ``device`` (auto-selected when
    ``None``) with its CUDA classifier shadow prepared, plus the saved
    arguments with the snapshot's cumulative ``step`` added.  The optimizer
    state is dropped; use ``checkpoints.restore`` to continue training.
    """
    payload = read_checkpoint(path)
    saved = saved_args(payload)
    model = DeltaModel(config_from_args(saved))
    model.load_state_dict(payload["state"])
    del payload
    resolved = pick_device(None if device is None else str(device))
    model = model.to(resolved).eval()
    model.refresh_shadows()
    return model, saved


def autocast(device: torch.device | str):
    """The trainer's evaluation numerics: BF16 activations on CUDA only."""
    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def logprob_chunks(
    model: DeltaModel, h_top: Tensor, chunk: int = 256
) -> Iterator[tuple[int, Tensor]]:
    """Yield ``(start, log_softmax)`` over sequence chunks of the readout.

    The full ``[B, T, V]`` log-probabilities do not fit comfortably at screen
    scale, so callers consume one chunk at a time.  The classifier is read in
    the activation dtype, matching the CUDA cross-entropy path.
    """
    weight = model.embed_tokens.weight.to(h_top.dtype)
    for start in range(0, h_top.shape[1], chunk):
        piece = model.readout_input(h_top[:, start : start + chunk])
        yield start, F.linear(piece, weight).float().log_softmax(-1)


@torch.no_grad()
def token_ce(model: DeltaModel, h_top: Tensor, targets: Tensor, chunk: int = 256) -> Tensor:
    """Per-token cross-entropy ``[B, T]`` of a top state against its targets."""
    pieces = []
    for start, logprobs in logprob_chunks(model, h_top, chunk):
        target = targets[:, start : start + chunk]
        pieces.append(-logprobs.gather(-1, target[..., None]).squeeze(-1))
    return torch.cat(pieces, dim=1)


def plain_mask(length: int, prefix: int | Tensor, device) -> Tensor:
    """``[B or 1, T, 1]`` boolean mask of plain-embedding positions."""
    positions = torch.arange(length, device=device)
    if isinstance(prefix, Tensor):
        return (positions[None, :] < prefix[:, None])[..., None]
    return (positions[None, :] < prefix)[..., None]


def fused_inputs(
    model: DeltaModel, e: Tensor, payload: Tensor, prefix: int | Tensor = 1
) -> Tensor:
    """The next pass's column input: plain prefix, FBT-fused suffix.

    ``payload`` is the preceding pass's payload at the same positions; it is
    shifted one column right (position 0 receives zero) before fusion, as in
    ``multipass``.  ``prefix`` is a plain-prefix length or a per-row tensor.
    """
    fused = model.fuse(shift_right(payload), e)
    return torch.where(plain_mask(e.shape[1], prefix, e.device), e, fused)


def position_bins(length: int) -> list[tuple[int, int]]:
    """Inclusive position intervals, widening from the first token to the row end."""
    bins = []
    start, stop = 0, 1
    while start < length:
        bins.append((start, min(stop, length) - 1))
        start = stop
        stop *= 4 if stop < 256 else 2
    return bins

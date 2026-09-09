"""Selective storage for compiled training blocks that exceed the raw budget."""

from functools import partial

import torch
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)


def activation_policy(ctx, op, *args, **kwargs):
    """Keep bounded projections and attention; reconstruct expanded MLP values.

    PKDA exposes its full auxiliary tensor family as one operator, so saving
    that operation would retain its entire recurrent history. Recomputing it
    and pointwise intermediates bounds memory in the longest loop modes.
    """
    if op == torch.ops.aten._scaled_dot_product_flash_attention.default:
        return CheckpointPolicy.MUST_SAVE
    if op in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
        left, right = args[-2:]
        # The fused gate/up projection expands D to 2I (>8D here). Keeping
        # those outputs across 120 executed layers exceeds the graph pool;
        # the packed PKDA QKV output (5D) still fits this sixfold bound.
        if right.shape[-1] <= 6 * left.shape[-1]:
            return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


checkpoint_context = partial(create_selective_checkpoint_contexts, activation_policy)

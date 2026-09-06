"""Compiled FlexAttention for causal GQA and prefix-cache decoding.

The CUDA backend ships with PyTorch; no external flash-attn extension is
required. Masks depend only on sequence geometry and are shared across layers
and passes. Cache writes remain explicit in the model.
"""

from functools import lru_cache

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from . import INDUCTOR_MODE


def _causal_mask(batch, head, query, key):
    return query >= key


@lru_cache(maxsize=64)
def _cached_causal_mask(length: int, device: torch.device):
    return create_block_mask(
        _causal_mask, None, None, length, length, device=device
    )


@torch.compiler.assume_constant_result
def causal_block_mask(length: int, device: torch.device):
    """Build a broadcast mask once, outside captured training work.

    Keep the constant-result function separate from the LRU wrapper: Dynamo
    deliberately unwraps LRU functions and otherwise traces mask construction.
    """
    return _cached_causal_mask(length, device)


def _causal_attention(query, key, value):
    return flex_attention(
        query,
        key,
        value,
        block_mask=causal_block_mask(query.shape[-2], query.device),
        enable_gqa=True,
        kernel_options={"BACKEND": "TRITON"},
    )


causal_attention = torch.compile(
    _causal_attention, fullgraph=True, dynamic=False, mode=INDUCTOR_MODE
)


def _prefix_attention(query, key, value):
    # A single new query sees every key in the already-sliced valid prefix.
    # A local q_idx >= kv_idx mask would incorrectly hide all but key zero.
    return flex_attention(query, key, value, enable_gqa=True)


prefix_attention = torch.compile(
    _prefix_attention, fullgraph=True, dynamic=True, mode=INDUCTOR_MODE
)

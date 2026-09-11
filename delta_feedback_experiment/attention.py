"""Native PyTorch Flash SDPA for causal GQA, FlexAttention for cached prefixes.

Both kernels ship with PyTorch. Training selects Flash explicitly inside its
compiled region; FP32 diagnostics use dense math. A single decode query
still reads its complete valid prefix.
"""

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention

from . import INDUCTOR_MODE

ROPE_THETA = 10_000.0


def rotary_qk(query, key, offset: int = 0):
    """Rotate every adjacent feature pair after Q/K normalization.

    Inputs are [B,H,T,D], with possibly different Q/K head counts. Positions
    count tokens, never feedback passes or core iterations. Construct phases
    and rotate in FP32 even under autocast; only the outputs return to the
    activation dtype. No mutable tables or checkpoint state are needed.
    """
    dim = query.shape[-1]
    frequency = ROPE_THETA ** (
        -torch.arange(0, dim, 2, device=query.device, dtype=torch.float32) / dim
    )
    position = (
        torch.arange(query.shape[-2], device=query.device, dtype=torch.float32) + offset
    )
    phase = position[:, None] * frequency[None, :]
    cosine, sine = phase.cos(), phase.sin()

    def rotate(value):
        even, odd = value.float()[..., 0::2], value.float()[..., 1::2]
        return (
            torch.stack(
                (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
            )
            .flatten(-2)
            .to(value.dtype)
        )

    return rotate(query), rotate(key)


def _causal_attention(query, key, value):
    # Flash accepts half precision; the probe also executes a small FP32
    # reference model, whose dense math path must remain explicit.
    backend = (
        SDPBackend.FLASH_ATTENTION
        if query.dtype in (torch.float16, torch.bfloat16)
        else SDPBackend.MATH
    )
    # Dynamo records this scoped backend selection, including restoration, in
    # an enclosing fullgraph block. Training cannot silently select math or
    # another SDPA backend, and the caller's process-wide preferences survive.
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(
            query, key, value, is_causal=True, enable_gqa=True
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

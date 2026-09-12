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


def _causal_attention(query, key, value):
    # Flash accepts half precision; FP32 analysis uses the explicit math path.
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

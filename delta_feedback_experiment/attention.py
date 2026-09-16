"""Native PyTorch fused SDPA for causal GQA, FlexAttention for cached prefixes.

Every kernel ships with PyTorch. Training takes the fused kernels in a fixed
priority inside its compiled region: Flash attention first, with cuDNN as
a fallback where it serves the geometry; FP32
diagnostics use dense math. A single decode query still reads its complete
valid prefix.
"""

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention

from .inductor import INDUCTOR_MODE

FUSED_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]


def _causal_attention(query, key, value):
    # Torch 2.14's cuDNN backend rejects backward at head width 192 on
    # the GH200. Flash serves the production training geometry;
    # FP32 analysis uses the explicit math path.
    backends = (
        FUSED_BACKENDS
        if query.dtype in (torch.float16, torch.bfloat16)
        else [SDPBackend.MATH]
    )
    # Dynamo records this scoped backend selection, including restoration, in
    # an enclosing fullgraph block. Training cannot silently select math, and
    # the caller's enabled backends are restored. A single forced backend
    # needs no priority change (including the FP32 diagnostic math path).
    with sdpa_kernel(backends, set_priority=len(backends) > 1):
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

"""Small independent GQA value and gradient oracle."""

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from delta_feedback_experiment.attention import _causal_attention


@pytest.mark.parametrize("head_dim", [192, 256])
def test_causal_gqa_matches_repeated_heads_and_restores_backend(head_dim):
    inputs = tuple(
        torch.randn(1, heads, 7, head_dim, requires_grad=True) for heads in (4, 2, 2)
    )
    refs = tuple(value.detach().clone().requires_grad_() for value in inputs)
    priority = torch._C._get_sdp_priority_order()
    with sdpa_kernel(SDPBackend.MATH):
        actual = _causal_attention(*inputs)
        expected = F.scaled_dot_product_attention(
            refs[0],
            refs[1].repeat_interleave(2, dim=1),
            refs[2].repeat_interleave(2, dim=1),
            is_causal=True,
        )
        assert torch.backends.cuda.math_sdp_enabled()
        assert not torch.backends.cuda.flash_sdp_enabled()
        assert torch._C._get_sdp_priority_order() == priority
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    for got, wanted in zip(
        torch.autograd.grad(actual, inputs, upstream),
        torch.autograd.grad(expected, refs, upstream),
        strict=True,
    ):
        torch.testing.assert_close(got, wanted)

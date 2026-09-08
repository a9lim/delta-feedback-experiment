"""FlexAttention qualification against independent dense math and cache semantics.

CUDA tests use the screen's non-power-of-two head dimension and real GQA.
They belong to the serial Jobe gate; CPU checks retain cache-policy coverage.
"""

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from delta_feedback_experiment.attention import causal_attention, prefix_attention
from delta_feedback_experiment.model import KVCache, condition_config

CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlexAttention requires CUDA qualification"
)


def _math_attention(query, key, value, *, causal):
    # Explicit repetition and the math backend make this independent of fused
    # GQA selection and its accumulation precision.
    groups = query.shape[1] // key.shape[1]
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            query.float(),
            key.float().repeat_interleave(groups, dim=1),
            value.float().repeat_interleave(groups, dim=1),
            is_causal=causal,
        )


def _inputs(length, heads=8, kv_heads=4):
    torch.manual_seed(42)
    # Projections and position-major caches reach attention in this strided
    # layout, rather than contiguous [B,H,T,D] allocations.
    return tuple(
        torch.randn(2, length, count, 96, device="cuda", dtype=torch.bfloat16)
        .transpose(1, 2)
        .requires_grad_()
        for count in (heads, kv_heads, kv_heads)
    )


@CUDA_ONLY
@pytest.mark.parametrize("length", [129, 257, 1025])
def test_causal_gqa_bf16_forward_and_gradients_match_math(length):
    query, key, value = _inputs(length)
    reference_inputs = tuple(
        x.detach().float().requires_grad_() for x in (query, key, value)
    )
    actual = causal_attention(query, key, value)
    expected = _math_attention(*reference_inputs, causal=True)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=3e-3)

    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream.float())
    for tensor, reference in zip((query, key, value), reference_inputs, strict=True):
        assert torch.isfinite(tensor.grad).all()
        # On Ada, both Flex and built-in Flash SDPA have 0.23--0.32% gradient
        # L2 error against FP32 math. Cancellation near zero reaches 0.0076
        # absolute error, so combine an absolute floor with a tight L2 bound.
        relative = (tensor.grad.float() - reference.grad).norm() / reference.grad.norm()
        assert relative < 5e-3
        torch.testing.assert_close(
            tensor.grad.float(), reference.grad, rtol=3e-2, atol=1e-2
        )


@CUDA_ONLY
@torch.no_grad()
def test_causal_gqa_cannot_read_future_across_block_boundary():
    query, key, value = _inputs(257, heads=4, kv_heads=2)
    original = causal_attention(query, key, value)
    cut = 129
    changed_key, changed_value = key.clone(), value.clone()
    changed_key[:, :, cut:] = torch.randn_like(changed_key[:, :, cut:]) * 4
    changed_value[:, :, cut:] += 32
    changed = causal_attention(query, changed_key, changed_value)
    torch.testing.assert_close(
        changed[:, :, :cut], original[:, :, :cut], rtol=0, atol=0
    )
    assert (changed[:, :, cut:] - original[:, :, cut:]).abs().max() > 1


def _cache(device):
    cfg = condition_config(
        "",
        vocab_size=97,
        dim=384,
        layers=1,
        heads=4,
        kv_heads=2,
        head_dim=96,
        intermediate=512,
        max_seq_len=300,
    )
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    cache = KVCache(cfg, batch=2, device=device, dtype=dtype)
    # Poison all unoccupied storage, so accidentally attending beyond the
    # written prefix is observable even when an output happens to be near zero.
    cache.k.fill_(float("nan"))
    cache.v.fill_(float("nan"))
    return cache


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA_ONLY)])
@torch.no_grad()
def test_prefix_cache_reads_every_valid_key_and_excludes_unused_tail(device):
    cache = _cache(device)
    dtype = cache.k.dtype
    query = torch.zeros(2, 4, 1, 96, device=device, dtype=dtype)
    # Successive prefixes exercise dynamic decoding at both sides of a kernel
    # block boundary. The last valid value controls the analytically known mean.
    for length in (1, 127, 128, 129, 257):
        count = length - cache.pos
        key = torch.zeros(2, count, 2, 96, device=device, dtype=dtype)
        value = torch.zeros_like(key)
        value[:, -1] = 16
        k_prefix, v_prefix = cache.update(0, key, value)
        cache.advance(count)
        key, value = k_prefix.transpose(1, 2), v_prefix.transpose(1, 2)
        if device == "cuda":
            actual = prefix_attention(query, key, value)
        else:
            actual = _math_attention(query, key, value, causal=False)
        expected = value.float().mean(dim=2, keepdim=True).repeat_interleave(2, dim=1)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=1e-3)
        # Tail poison is still present: finite output cannot result from merely
        # overwriting the entire allocation with valid values.
        assert torch.isnan(cache.v[:, :, length:]).all()


@CUDA_ONLY
def test_checkpoint_keeps_each_outstanding_forward_attention_geometry():
    from delta_feedback_experiment.model import DeltaModel

    torch.manual_seed(314)
    cfg = condition_config(
        "",
        vocab_size=97,
        dim=192,
        layers=1,
        heads=2,
        kv_heads=1,
        head_dim=96,
        intermediate=64,
        max_seq_len=257,
    )
    checkpointed = DeltaModel(cfg).cuda().train()
    reference = DeltaModel(cfg).cuda().train()
    reference.load_state_dict(checkpointed.state_dict())
    checkpointed.grad_checkpoint = True
    reference.grad_checkpoint = False

    actual_inputs, reference_inputs, actual_outputs, reference_outputs = [], [], [], []
    upstreams = []
    # Backward for the long graph happens after a shorter forward has run on
    # the same module. A mutable "latest mask" would truncate recomputation.
    for length in (257, 129):
        x = torch.randn(1, length, cfg.dim, device="cuda", dtype=torch.bfloat16)
        actual_inputs.append(x.detach().requires_grad_())
        reference_inputs.append(x.detach().clone().requires_grad_())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual_outputs.append(checkpointed.forward_column(actual_inputs[-1]).h_top)
            reference_outputs.append(
                reference.forward_column(reference_inputs[-1]).h_top
            )
        torch.testing.assert_close(
            actual_outputs[-1], reference_outputs[-1], rtol=0, atol=0
        )
        upstreams.append(torch.randn_like(actual_outputs[-1]))

    for index, upstream in enumerate(upstreams):
        actual_outputs[index].backward(upstream)
        reference_outputs[index].backward(upstream)
        torch.testing.assert_close(
            actual_inputs[index].grad,
            reference_inputs[index].grad,
            rtol=2e-5,
            atol=2e-6,
        )
        for (name, parameter), (ref_name, ref_parameter) in zip(
            checkpointed.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert name == ref_name
            if ref_parameter.grad is None:
                assert parameter.grad is None, name
            else:
                torch.testing.assert_close(
                    parameter.grad,
                    ref_parameter.grad,
                    rtol=2e-5,
                    atol=2e-6,
                    msg=lambda message, name=name: f"{name}: {message}",
                )

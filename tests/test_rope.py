"""Layer selection, independent rotary math, and the compiled CUDA attention path."""

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from delta_feedback_experiment import INDUCTOR_MODE
from delta_feedback_experiment.attention import rotary_qk
from delta_feedback_experiment.model import DeltaModel, condition_config
from delta_feedback_experiment.train import _capture_without_gc

CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def complex_rotation(value, offset=0):
    """Independent FP64 complex multiplication, including its autograd path."""
    dim = value.shape[-1]
    positions = torch.arange(value.shape[-2], device=value.device).double() + offset
    frequencies = torch.tensor(
        [10000.0 ** (-2 * pair / dim) for pair in range(dim // 2)],
        device=value.device,
        dtype=torch.float64,
    )
    phase = positions[:, None] * frequencies
    pairs = torch.view_as_complex(value.double().reshape(*value.shape[:-1], -1, 2))
    rotated = pairs * torch.polar(torch.ones_like(phase), phase)
    return torch.view_as_real(rotated).flatten(-2).to(value.dtype)


@pytest.mark.parametrize("condition", ["", "r", "f", "rf", "l", "rfl", "a", "arfl"])
def test_rope_layers_align_with_pkda_at_every_cell(condition):
    cfg = condition_config(
        condition,
        vocab_size=97,
        dim=32,
        layers=16,
        heads=2,
        kv_heads=2,
        head_dim=16,
        intermediate=64,
        pkda_heads=2,
        pkda_head_dim=16,
    )
    model = DeltaModel(cfg)
    for layer, block in enumerate(model.blocks):
        if "a" in condition and layer % 4 != 3:
            assert block.is_pkda
        else:
            assert not block.is_pkda
            assert block.attn.rope == ("a" not in condition and layer % 4 != 3)
    assert not any("rope" in name or "rotary" in name for name in model.state_dict())


@pytest.mark.parametrize("offset", [0, 255, 4096])
def test_rotary_values_gradients_and_relative_positions(offset):
    torch.manual_seed(13)
    inputs = tuple(
        torch.randn(2, heads, 17, 96, requires_grad=True) for heads in (4, 2)
    )
    references = tuple(value.detach().clone().requires_grad_() for value in inputs)
    actual = rotary_qk(*inputs, offset)
    expected = tuple(complex_rotation(value, offset) for value in references)
    upstreams = tuple(torch.randn_like(value) for value in actual)
    gradients = torch.autograd.grad(actual, inputs, upstreams)
    reference_gradients = torch.autograd.grad(expected, references, upstreams)
    for value, reference, grad, reference_grad in zip(
        actual, expected, gradients, reference_gradients, strict=True
    ):
        # At large positions FP32 phases round before sin/cos; FP64 does not.
        torch.testing.assert_close(value, reference, rtol=4e-4, atol=8e-4)
        torch.testing.assert_close(grad, reference_grad, rtol=4e-4, atol=8e-4)
    base_q, base_k = rotary_qk(*inputs)
    q, k = actual
    torch.testing.assert_close(q.square().sum(-1), inputs[0].square().sum(-1))
    torch.testing.assert_close(k.square().sum(-1), inputs[1].square().sum(-1))
    # A shared position offset leaves relative attention scores invariant.
    torch.testing.assert_close(
        q @ k.repeat_interleave(2, dim=1).transpose(-1, -2),
        base_q @ base_k.repeat_interleave(2, dim=1).transpose(-1, -2),
        rtol=5e-4,
        atol=4e-3,
    )


def test_rope_rejects_an_unpaired_head_coordinate():
    with pytest.raises(ValueError, match="positive even"):
        DeltaModel(condition_config("", vocab_size=97, dim=32, head_dim=15))


@pytest.mark.parametrize("layer", [0, 3])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA_ONLY)])
def test_gqa_normalizes_then_rotates_before_attention_and_captures(layer, device):
    torch.manual_seed(29)
    cuda = device == "cuda"
    # CUDA qualifies a complete production-width attention layer at T=4096,
    # including projection, learned Q/K scales, RoPE, Flash, gating, and dW/dX.
    dim, heads, kv_heads, length = (768, 8, 4, 4096) if cuda else (192, 2, 1, 17)
    cfg = condition_config(
        "",
        vocab_size=97,
        dim=dim,
        layers=4,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=96,
        intermediate=64,
        max_seq_len=length,
    )
    model = DeltaModel(cfg).to(device)
    attention = model.blocks[layer].attn
    gate = model.attention_gates[layer].weight
    with torch.no_grad():
        # Unequal gains catch a mistaken rotation-before-RMSNorm ordering.
        attention.q_norm.weight.uniform_(0.5, 1.5)
        attention.k_norm.weight.uniform_(0.5, 1.5)
    x = torch.randn(1, length, dim, device=device, requires_grad=True)
    upstream = torch.randn_like(x)
    inputs = (x, *attention.parameters(), gate)

    def step():
        out = attention(x, gate, None, layer)
        return out, torch.autograd.grad(out, inputs, upstream)

    if cuda:
        attention.forward = torch.compile(
            attention.forward, fullgraph=True, dynamic=False, mode=INDUCTOR_MODE
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                step()
            graph = torch.cuda.CUDAGraph()
            with _capture_without_gc(graph, None):
                actual, gradients = step()
            # Replays must use fresh inputs, rather than stale rotated values.
            with torch.no_grad():
                x.normal_()
            graph.replay()
        torch.cuda.synchronize()
    else:
        actual, gradients = step()

    def reference():
        q, k, v = F.linear(x, attention.qkv_proj.weight).split(
            (attention.q_size, attention.kv_size, attention.kv_size), dim=-1
        )
        q = attention.q_norm(q.view(1, length, heads, 96).transpose(1, 2))
        k = attention.k_norm(k.view(1, length, kv_heads, 96).transpose(1, 2))
        v = v.view(1, length, kv_heads, 96).transpose(1, 2)
        if layer != 3:
            q, k = complex_rotation(q), complex_rotation(k)
        # Flash here bounds the change to RoPE/projections, without attributing
        # fused-vs-math accumulation differences to the new positional encoding.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION if cuda else SDPBackend.MATH):
            z = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        z = z.transpose(1, 2).reshape(1, length, -1)
        return F.linear(z * F.linear(x, gate).sigmoid(), attention.o_proj.weight)

    with torch.autocast(device, dtype=torch.bfloat16, enabled=cuda):
        expected = reference()
        expected_gradients = torch.autograd.grad(expected, inputs, upstream)
    for value, target in zip(
        (actual, *gradients), (expected, *expected_gradients), strict=True
    ):
        assert torch.isfinite(value).all()
        if cuda:
            relative = (value.float() - target.float()).norm() / target.float().norm()
            assert relative < 0.025, relative.item()
        else:
            torch.testing.assert_close(value, target, rtol=2e-4, atol=2e-5)

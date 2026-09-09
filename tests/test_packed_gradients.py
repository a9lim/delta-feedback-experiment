"""Projection gradient slabs retain parameter ownership and accumulate once."""

from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F

from delta_feedback_experiment.activation import checkpoint_context
from delta_feedback_experiment.cuda_kernels import sink_linear
from delta_feedback_experiment.model import DeltaModel, condition_config


def tiny(condition):
    return DeltaModel(
        condition_config(
            condition,
            vocab_size=97,
            dim=32,
            layers=12 if "l" in condition else 4,
            heads=2,
            kv_heads=2,
            head_dim=16,
            intermediate=64,
            pkda_heads=2,
            pkda_head_dim=16,
            pkda_conv_size=4,
            max_seq_len=16,
        )
    )


@pytest.mark.parametrize("condition", ["", "a", "arfl"])
def test_gradient_slabs_cover_parameters_without_overlap(condition):
    model = tiny(condition)
    buffers = model.allocate_gradient_buffers()
    assert set(buffers) == {p for p in model.parameters() if p.requires_grad}
    intervals = {}
    for parameter, buffer in buffers.items():
        assert buffer.shape == parameter.shape
        assert buffer.dtype == torch.float32
        assert buffer.is_contiguous()
        storage = buffer.untyped_storage()
        key = storage.data_ptr()
        intervals.setdefault(key, []).append(
            (buffer.storage_offset(), buffer.storage_offset() + buffer.numel())
        )
    # Every allocation is covered exactly once, including the packed slabs.
    for spans in intervals.values():
        spans.sort()
        assert spans[0][0] == 0
        assert all(left[1] == right[0] for left, right in pairwise(spans))
    allocated = {
        buffer.untyped_storage().data_ptr(): buffer.untyped_storage().nbytes()
        for buffer in buffers.values()
    }
    assert sum(allocated.values()) == sum(p.numel() * 4 for p in buffers)

    model.bind_gradient_sinks(buffers)
    first_binding = [block.attn.packed_qkv_sink for block in model.blocks]
    for block, packed in zip(model.blocks, first_binding, strict=True):
        assert packed is not None
        if block.is_pkda:
            parameters = (
                block.attn.q_proj.weight,
                block.attn.k_proj.weight,
                block.attn.v_proj.weight,
            )
        else:
            parameters = (
                block.attn.qkv_proj.weight,
                model.attention_gates[block.global_gate_index].weight,
            )
        start = 0
        for index, parameter in enumerate(parameters):
            buffers[parameter].fill_(index + 1)
            rows = parameter.shape[0]
            torch.testing.assert_close(packed[start : start + rows], buffers[parameter])
            start += rows
    # The trainer rebinds after warming. Preserve objects as well as addresses.
    model.bind_gradient_sinks(buffers)
    assert all(
        block.attn.packed_qkv_sink is packed
        for block, packed in zip(model.blocks, first_binding, strict=True)
    )
    model.bind_gradient_sinks(None)
    assert all(block.attn.packed_qkv_sink is None for block in model.blocks)


def test_independent_or_partial_gradient_buffers_do_not_pack():
    model = tiny("a")
    buffers = {p: torch.zeros_like(p) for p in model.parameters()}
    model.bind_gradient_sinks(buffers)
    assert all(block.attn.packed_qkv_sink is None for block in model.blocks)
    buffers = model.allocate_gradient_buffers()
    del buffers[model.blocks[0].attn.k_proj.weight]
    model.bind_gradient_sinks(buffers)
    assert model.blocks[0].attn.packed_qkv_sink is None
    assert model.blocks[1].attn.packed_qkv_sink is not None


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_packed_linear_repeated_backward_preserves_segment_gradients(
    compiled, checkpointed
):
    torch.manual_seed(89)
    weights = tuple(torch.nn.Parameter(torch.randn(rows, 8)) for rows in (6, 3, 5))
    # Nonzero initial contents and a nonzero storage offset catch overwrite
    # and mistaken backing-allocation addressing in beta=1 accumulation.
    backing = torch.full((18, 8), 0.25)
    packed = backing[2:16]
    sinks = packed.split((6, 3, 5))
    shadow = torch.cat([weight.detach() for weight in weights])
    reference_weights = tuple(
        weight.detach().clone().requires_grad_() for weight in weights
    )

    def projection(x):
        output = sink_linear(x, weights, sinks, shadow, packed_sink=packed)
        return output.sin() * torch.sigmoid(output) if checkpointed else output

    def forward(x):
        if checkpointed:
            return torch.utils.checkpoint.checkpoint(
                projection,
                x,
                use_reentrant=False,
                preserve_rng_state=False,
                context_fn=checkpoint_context,
            )
        return projection(x)

    run = (
        torch.compile(forward, backend="aot_eager", fullgraph=True)
        if compiled
        else forward
    )
    for _ in range(3):
        x = torch.randn(2, 5, 8, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        output = run(x)
        expected = F.linear(reference_x, torch.cat(reference_weights))
        if checkpointed:
            expected = expected.sin() * torch.sigmoid(expected)
        cotangent = torch.randn_like(output)
        output.backward(cotangent)
        expected.backward(cotangent)
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(x.grad, reference_x.grad)
        for weight, buffer, reference_weight in zip(
            weights, sinks, reference_weights, strict=True
        ):
            assert weight.grad is None
            torch.testing.assert_close(buffer, 0.25 + reference_weight.grad)
    torch.testing.assert_close(backing[:2], torch.full_like(backing[:2], 0.25))
    torch.testing.assert_close(backing[16:], torch.full_like(backing[16:], 0.25))

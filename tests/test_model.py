"""One tiny model per distinct causal, recurrence, and numerical contract."""

from dataclasses import fields

import pytest
import torch

from delta_feedback_experiment.model import (
    CONDITION_LETTERS,
    DeltaModel,
    KVCache,
    ModelConfig,
    condition_config,
    multipass,
    multipass_loss,
    parse_condition,
)
from delta_feedback_experiment.moe import MixtureOfExperts
from delta_feedback_experiment.pkda import PreconditionedKDA

TINY = {
    "vocab_size": 31,
    "dim": 16,
    "layers": 4,
    "heads": 2,
    "kv_heads": 2,
    "head_dim": 8,
    "intermediate": 32,
    "pkda_heads": 2,
    "pkda_head_dim": 8,
    "pkda_conv_size": 4,
    "max_seq_len": 12,
    "loop_iterations": 2,
    "loop_max_iterations": 2,
}


def tiny(condition="f", **overrides):
    torch.manual_seed(7)
    geometry = TINY | ({"layers": 12} if "l" in condition else {}) | overrides
    return DeltaModel(condition_config(condition, **geometry)).eval()


def tokens():
    return torch.randint(0, 31, (1, 6), generator=torch.Generator().manual_seed(11))


def test_current_conditions_and_permanent_structure():
    assert tuple(CONDITION_LETTERS) == ("f", "l")
    assert parse_condition("lf") == "fl"
    assert ModelConfig().condition == "f"
    assert not {"hybrid", "block_routing", "experts", "mtp"} & {
        field.name for field in fields(ModelConfig)
    }
    for invalid in ("", "a", "r", "m", "e", "arf", "ff", "x"):
        with pytest.raises(ValueError):
            parse_condition(invalid)
    with pytest.raises(ValueError):
        ModelConfig(feedback=False, loop=False)
    with pytest.raises(ValueError, match="three"):
        condition_config("l", **TINY)
    with pytest.raises(ValueError):
        condition_config("f", **(TINY | {"intermediate": 30}))
    model = tiny()
    assert [isinstance(block.attn, PreconditionedKDA) for block in model.blocks] == [
        True,
        True,
        True,
        False,
    ]
    assert all(isinstance(block.mlp, MixtureOfExperts) for block in model.blocks)
    assert all(
        block.attn_router is not None and block.mlp_router is not None
        for block in model.blocks
    )
    assert model.mtp is not None
    assert isinstance(model.mtp.block.attn, PreconditionedKDA)
    assert isinstance(model.mtp.block.mlp, MixtureOfExperts)
    assert model.mtp.block.attn_router is None and model.mtp.block.mlp_router is None
    assert all(model.mtp.block.attn is not block.attn for block in model.blocks)
    assert model.expert_banks == (
        *(block.mlp for block in model.blocks), model.mtp.block.mlp
    )
    loop = tiny("l").cfg
    assert not loop.feedback and loop.core_layers == range(4, 8)
    assert loop.executed_layers(2) == 16
    with pytest.raises(ValueError):
        loop.resolve_iterations(3)


def test_payload_and_auxiliary_initialization_pair_across_all_conditions():
    states = [tiny(condition, layers=12).state_dict() for condition in ("f", "l", "fl")]
    common = states[0].keys() & states[1].keys() & states[2].keys()
    assert "payload_router.query" in common and "payload_norm.weight" in common
    assert any(name.startswith("mtp.block.attn.") for name in common)
    assert any(name.startswith("mtp.block.mlp.experts.") for name in common)
    for name in common:
        for state in states[1:]:
            torch.testing.assert_close(states[0][name], state[name], atol=0, rtol=0, msg=name)


def test_routing_matches_normalized_math_and_telescopes():
    model = tiny("fl")
    router = model.blocks[2].mlp_router
    sources = [torch.randn(1, 3, 16) for _ in range(3)]
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "query" in name:
                parameter.normal_()
    routed, weights = router(sources, True)
    values = torch.stack([router.null.expand_as(sources[0]), *sources])
    logits = torch.einsum(
        "hd,nbthd->nbth",
        router.query.view(2, 8),
        router.key_norm(values).view(4, 1, 3, 2, 8),
    )
    expected_weights = logits.softmax(0)
    expected = torch.einsum(
        "nbth,nbthd->bthd",
        expected_weights,
        values.view(4, 1, 3, 2, 8),
    ).reshape_as(routed)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(routed, expected)
    out = model.forward_column(model.embed_tokens(tokens()), want_weights=True)
    torch.testing.assert_close(
        out.h_top, out.sources[0] + torch.stack(out.sources[1:]).sum(0)
    )
    torch.testing.assert_close(out.core_state - out.core_entry, out.sources[2])
    assert out.route_source_names["L4i1.attn"] == ("null", "seed", "block0", "partial1")
    assert out.route_source_names["payload"] == (
        "null",
        "seed",
        "block0",
        "block1",
        "block2",
    )


def test_feedback_and_loop_are_token_causal():
    model = tiny("fl")
    original = tokens()
    edited = original.clone()
    edited[:, 3] = (edited[:, 3] + 1) % model.cfg.vocab_size
    prefix = torch.ones((1, 1), dtype=torch.long)
    embeddings = []

    def retain_embedding(module, inputs, output):
        output.retain_grad()
        embeddings.append(output)

    hook = model.embed_tokens.register_forward_hook(retain_embedding)
    before = multipass(model, original, 2, prefix_lens=prefix)
    hook.remove()
    after = multipass(model, edited, 2, prefix_lens=prefix)
    for left, right in zip(before, after, strict=True):
        torch.testing.assert_close(
            left.h_top[:, :3], right.h_top[:, :3], atol=5e-5, rtol=1e-5
        )
        assert not torch.allclose(left.h_top[:, 3:], right.h_top[:, 3:])
    # Sparse expert batches can round FP32 GEMMs differently after the edit;
    # the exact zero derivative also checks that no future dependency exists.
    before[-1].h_top[:, :3].square().sum().backward()
    assert torch.count_nonzero(embeddings[0].grad[:, 3:]) == 0
    plain = model.forward_column(model.embed_tokens(original[:, :-1])).h_top
    torch.testing.assert_close(before[0].h_top, plain)


def test_loop_once_pairs_with_flat_values_and_gradients():
    flat, loop = tiny("f", layers=12).train(), tiny("fl").train()
    assert flat.state_dict().keys() == loop.state_dict().keys()
    losses = []
    for model in (flat, loop):
        outs = multipass(
            model,
            tokens(),
            2,
            iterations=1,
            prefix_lens=torch.ones(1, 1, dtype=torch.long),
        )
        loss = multipass_loss(model, tokens(), outs).total
        losses.append(loss.detach())
        loss.backward()
    torch.testing.assert_close(*losses, atol=0, rtol=0)
    for (name, left), (_, right) in zip(
        flat.named_parameters(), loop.named_parameters(), strict=True
    ):
        torch.testing.assert_close(left, right, atol=0, rtol=0, msg=name)
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0, msg=name)


def test_checkpointing_preserves_feedback_loop_and_auxiliary_gradients():
    models = [tiny("fl").train(), tiny("fl").train()]
    models[1].grad_checkpoint = True
    losses, counts = [], []
    for model in models:
        outs = multipass(
            model, tokens(), 2, prefix_lens=torch.ones(1, 1, dtype=torch.long)
        )
        result = multipass_loss(model, tokens(), outs, z_coef=0.017)
        logical_counts = result.expert_counts.clone()
        biases = [bank.expert_bias.clone() for bank in model.expert_banks]
        result.total.backward()
        losses.append(result.total.detach())
        counts.append(logical_counts)
        # Checkpoint recomputation is backward work, not another dispatch for
        # the next-step controller. It must neither count twice nor move bias.
        torch.testing.assert_close(result.expert_counts, logical_counts, atol=0, rtol=0)
        for bank, before in zip(model.expert_banks, biases, strict=True):
            torch.testing.assert_close(bank.expert_bias, before, atol=0, rtol=0)
        trunk_count = 3 * (tokens().shape[1] - 1) * 2
        auxiliary_count = 3 * (tokens().shape[1] - 2) * 2
        assert logical_counts.sum(-1).tolist() == (
            [trunk_count] * 4 + [2 * trunk_count] * 4 + [trunk_count] * 4
            + [auxiliary_count]
        )
        for name in (
            "fuse_value.weight",
            "fuse_gate.weight",
            "payload_router.query",
            "embed_tokens.weight",
            "attention_gates.0.weight",
        ):
            gradient = dict(model.named_parameters())[name].grad
            assert gradient is not None and gradient.dtype == torch.float32, name
            assert gradient.abs().sum() > 0 and torch.isfinite(gradient).all(), name
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.mtp.parameters()
        )
    torch.testing.assert_close(*losses, atol=0, rtol=0)
    torch.testing.assert_close(*counts, atol=0, rtol=0)
    for (name, left), (_, right) in zip(
        models[0].named_parameters(), models[1].named_parameters(), strict=True
    ):
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(
                left.grad, right.grad, atol=1e-6, rtol=1e-5, msg=name
            )


@pytest.mark.parametrize("condition", ["l", "fl"])
@torch.no_grad()
def test_cached_decode_matches_full_recomputation(condition):
    model = tiny(condition)
    toks = tokens()
    embedded = model.embed_tokens(toks)
    cache = KVCache(model.cfg, batch=1, device="cpu", dtype=torch.float32)
    prefill = model.forward_column(embedded[:, :3], cache=cache)
    payload = prefill.payload[:, -1:] if model.cfg.feedback else None
    reference_rows = embedded[:, :3]
    for position in range(3, toks.shape[1]):
        out = model.step(toks[:, position : position + 1], payload, cache)
        new = embedded[:, position : position + 1]
        if model.cfg.feedback:
            reference_payload = model.forward_column(reference_rows).payload[:, -1:]
            new = model.fuse(reference_payload, new)
        reference_rows = torch.cat([reference_rows, new], dim=1)
        reference = model.forward_column(reference_rows)
        torch.testing.assert_close(
            out.h_top, reference.h_top[:, -1:], atol=3e-5, rtol=1e-5
        )
        payload = out.payload[:, -1:] if model.cfg.feedback else None
    assert cache.k.shape[0] == 4  # prelude + two core iterations + coda
    assert len(cache.pkda_states) == 12

"""Numerical and causal contracts for the teacher-forced second-token head."""

import pytest
import torch
import torch.nn.functional as F

from delta_feedback_experiment.model import (
    DeltaModel,
    KVCache,
    condition_config,
    multipass,
    multipass_loss,
)
from delta_feedback_experiment.optim import split_parameters

GEOMETRY = {
    "vocab_size": 41,
    "dim": 32,
    "layers": 4,
    "heads": 4,
    "kv_heads": 2,
    "head_dim": 16,
    "intermediate": 64,
    "pkda_heads": 2,
    "pkda_head_dim": 16,
    "pkda_conv_size": 4,
    "max_seq_len": 16,
}


def tiny(condition="m", *, seed=13):
    torch.manual_seed(seed)
    return DeltaModel(condition_config(condition, **GEOMETRY)).eval()


def tokens(length=8):
    generator = torch.Generator().manual_seed(29)
    return torch.randint(0, GEOMETRY["vocab_size"], (2, length), generator=generator)


def passes(model, toks, count):
    prefixes = torch.full((count - 1, toks.shape[0]), 2, dtype=torch.long)
    return multipass(model, toks, count, prefix_lens=prefixes)


def explicit_head_losses(model, hidden, targets):
    """Independent materialized-logit reference for both readout objectives."""
    normalized = hidden.float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(-1, keepdim=True) + model.cfg.norm_eps
    )
    normalized = normalized * model.final_norm.weight.float() * model.cfg.mup_ratio
    logits = F.linear(normalized, model.embed_tokens.weight.float())
    ce = F.cross_entropy(logits.flatten(0, 1), targets.reshape(-1))
    z = torch.logsumexp(logits, dim=-1).square().mean()
    return ce, z


def feedback_sum(values):
    return (
        values[0] + sum(values[1:]) / (len(values) - 1)
        if len(values) > 1
        else values[0]
    )


@pytest.mark.parametrize("condition", ["", "arf", "erf"])
def test_mtp_toggle_preserves_paired_trunk_initialization_and_rng(condition):
    ordinary = tiny(condition)
    ordinary_rng = torch.random.get_rng_state().clone()
    auxiliary = tiny(condition + "m")
    auxiliary_rng = torch.random.get_rng_state()

    assert ordinary.mtp is None and auxiliary.mtp is not None
    assert not ordinary.cfg.mtp and auxiliary.cfg.mtp
    assert torch.equal(ordinary_rng, auxiliary_rng)
    auxiliary_state = auxiliary.state_dict()
    for name, value in ordinary.state_dict().items():
        assert torch.equal(value, auxiliary_state[name]), name
    assert set(auxiliary_state) - set(ordinary.state_dict())
    assert all(
        name.startswith("mtp.")
        for name in set(auxiliary_state) - set(ordinary.state_dict())
    )


@pytest.mark.parametrize("condition", ["", "arf"])
@torch.no_grad()
def test_mtp_toggle_preserves_standard_prefill_and_cached_decode(condition):
    toks = tokens(length=7)
    results = []
    for model in (tiny(condition), tiny(condition + "m")):
        cache = KVCache(model.cfg, batch=2, device=toks.device, dtype=torch.float32)
        first = model.forward_column(model.embed_tokens(toks[:, :3]), cache=cache)
        hidden = [first.h_top]
        for position in range(3, toks.shape[1]):
            hidden.append(
                model.step(toks[:, position : position + 1], None, cache).h_top
            )
        decoded = torch.cat(hidden, dim=1)
        full = model.forward_column(model.embed_tokens(toks)).h_top
        torch.testing.assert_close(decoded, full, atol=3e-5, rtol=1e-5)
        results.append(model.logits(decoded))
    torch.testing.assert_close(*results, atol=0, rtol=0)


@pytest.mark.parametrize("condition", ["m", "arfm"])
@torch.no_grad()
def test_second_token_prediction_cannot_see_its_target(condition):
    """Editing x[t] can first change the auxiliary prediction of x[t+1]."""
    model = tiny(condition)
    original = tokens()
    changed = original.clone()
    edited_position = 4
    changed[:, edited_position] = (
        changed[:, edited_position] + 7
    ) % model.cfg.vocab_size
    count = 3 if model.cfg.feedback else 1
    original_out = passes(model, original, count)
    changed_out = passes(model, changed, count)

    for before, after in zip(original_out, changed_out, strict=True):
        original_hidden = model.mtp_hidden(before.h_top[:, :-1], original[:, 1:-1])
        changed_hidden = model.mtp_hidden(after.h_top[:, :-1], changed[:, 1:-1])
        # Auxiliary index t-2 predicts x[t]; neither x[t] nor later tokens may
        # influence it, even through an earlier feedback pass.
        torch.testing.assert_close(
            original_hidden[:, : edited_position - 1],
            changed_hidden[:, : edited_position - 1],
            atol=1e-6,
            rtol=1e-6,
        )
        assert not torch.allclose(
            original_hidden[:, edited_position - 1],
            changed_hidden[:, edited_position - 1],
        )


def test_next_token_conditioning_and_hidden_stream_are_differentiable():
    model = tiny()
    supplied = tokens(length=5)
    hidden = torch.randn(2, 5, model.cfg.dim, requires_grad=True)
    result = model.mtp_hidden(hidden, supplied)
    assert result.shape == hidden.shape
    weights = torch.linspace(-1, 1, result.numel()).reshape_as(result)
    (result * weights).sum().backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    assert model.embed_tokens.weight.grad is not None
    assert model.embed_tokens.weight.grad[supplied.unique()].abs().sum() > 0
    # The supplied next token goes through the same embedding used by the
    # trunk; the auxiliary module must not allocate its own vocabulary table.
    assert not any(
        isinstance(module, torch.nn.Embedding) for module in model.mtp.modules()
    )


def test_auxiliary_objective_trains_trunk_feedback_and_shared_readout():
    model = tiny("arfm").train()
    toks = tokens()
    outs = passes(model, toks, 3)
    loss = multipass_loss(model, toks, outs)
    feedback_sum(loss.mtp).backward()

    for name, parameter in model.named_parameters():
        if name.startswith(("blocks.", "mtp.")) or name in {
            "embed_tokens.weight",
            "final_norm.weight",
            "fuse_value.weight",
            "fuse_gate.weight",
        }:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    assert model.embed_tokens.weight.grad.abs().sum() > 0
    assert model.final_norm.weight.grad.abs().sum() > 0
    assert model.fuse_value.weight.grad.abs().sum() > 0
    assert sum(p.grad.abs().sum() for p in model.blocks.parameters()) > 0


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("z_coef", [0.0, 0.017])
def test_mtp_loss_matches_materialized_logits_and_feedback_weighting(count, z_coef):
    model = tiny("fm")
    toks = tokens(length=6)
    outs = passes(model, toks, count)
    coefficient = 0.23
    actual = multipass_loss(model, toks, outs, mtp_weight=coefficient, z_coef=z_coef)
    assert len(actual.ntp) == count and len(actual.mtp) == count
    ntp, mtp, ntp_z, mtp_z = [], [], [], []
    for index, out in enumerate(outs):
        ce, z = explicit_head_losses(model, out.h_top, toks[:, 1:])
        second_hidden = model.mtp_hidden(out.h_top[:, :-1], toks[:, 1:-1])
        second_ce, second_z = explicit_head_losses(model, second_hidden, toks[:, 2:])
        ntp.append(ce)
        ntp_z.append(z)
        mtp.append(second_ce)
        mtp_z.append(second_z)
        torch.testing.assert_close(actual.ntp[index], ce)
        torch.testing.assert_close(actual.mtp[index], second_ce)
    expected = (
        feedback_sum(ntp)
        + z_coef * feedback_sum(ntp_z)
        + coefficient * (feedback_sum(mtp) + z_coef * feedback_sum(mtp_z))
    )
    torch.testing.assert_close(actual.total, expected)


def test_mtp_weight_zero_keeps_ntp_loss_and_auxiliary_metrics():
    model = tiny("fm")
    toks = tokens()
    outs = passes(model, toks, 3)
    actual = multipass_loss(model, toks, outs, mtp_weight=0.0)
    torch.testing.assert_close(actual.total, feedback_sum(actual.ntp))
    assert len(actual.mtp) == 3
    assert all(torch.isfinite(value) for value in actual.mtp)


def test_mtp_checkpointing_preserves_loss_and_all_parameter_gradients():
    toks = tokens(length=6)
    models = [tiny("arfm").train(), tiny("arfm").train()]
    models[1].grad_checkpoint = True
    losses = []
    for model in models:
        loss = multipass_loss(model, toks, passes(model, toks, 2), z_coef=0.017)
        losses.append(loss.total.detach())
        loss.total.backward()
    torch.testing.assert_close(*losses, atol=0, rtol=0)
    for (name, ordinary), (checkpoint_name, checkpointed) in zip(
        models[0].named_parameters(), models[1].named_parameters(), strict=True
    ):
        assert name == checkpoint_name
        assert (ordinary.grad is None) == (checkpointed.grad is None), name
        if ordinary.grad is not None:
            torch.testing.assert_close(
                ordinary.grad, checkpointed.grad, atol=1e-6, rtol=1e-5, msg=name
            )


def test_mtp_parameters_follow_hidden_matrix_and_gate_optimizer_rules():
    model = tiny()
    matrices, base, width = (
        {id(p) for p in group} for group in split_parameters(model)
    )
    assert len(matrices | base | width) == len(list(model.parameters()))
    for name, parameter in model.mtp.named_parameters():
        if name == "attention_gate.weight":
            assert id(parameter) in width
        elif parameter.ndim == 2:
            assert id(parameter) in matrices, name
        else:
            assert id(parameter) in base, name


@pytest.mark.parametrize("length", [0, 1, 2])
def test_mtp_loss_rejects_rows_without_a_second_token_target(length):
    model = tiny()
    # Use a real column output so rejection concerns row alignment, not an
    # unrelated empty-output failure or an attention kernel's empty input.
    valid = tokens(length=3)
    outs = passes(model, valid, 1)
    with pytest.raises(ValueError):
        multipass_loss(model, valid[:, :length], outs)


@pytest.mark.parametrize("length", [3, 8])
def test_mtp_loss_uses_only_real_second_token_targets(length):
    model = tiny()
    toks = tokens(length=length)
    outs = passes(model, toks, 1)
    hidden = model.mtp_hidden(outs[0].h_top[:, :-1], toks[:, 1:-1])
    assert hidden.shape == (toks.shape[0], length - 2, model.cfg.dim)
    reference, _ = explicit_head_losses(model, hidden, toks[:, 2:])
    actual = multipass_loss(model, toks, outs)
    torch.testing.assert_close(actual.mtp[0], reference)
    assert torch.isfinite(actual.total)


@pytest.mark.parametrize("shape", [(2, 0), (2, 3), (1, 4), (2, 4, 1)])
def test_mtp_hidden_rejects_misaligned_conditioning(shape):
    model = tiny()
    hidden = torch.zeros(2, 4, model.cfg.dim)
    with pytest.raises(ValueError):
        model.mtp_hidden(hidden, torch.zeros(shape, dtype=torch.long))

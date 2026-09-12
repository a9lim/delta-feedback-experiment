"""Numerical and causal contracts for the teacher-forced second-token head."""

import torch
import torch.nn.functional as F

from delta_feedback_experiment.model import (
    EXPERT_BALANCE_COEF,
    DeltaModel,
    condition_config,
    multipass,
    multipass_loss,
)

GEOMETRY = {
    "vocab_size": 41,
    "dim": 16,
    "layers": 4,
    "heads": 2,
    "kv_heads": 2,
    "head_dim": 8,
    "intermediate": 32,
    "pkda_heads": 2,
    "pkda_head_dim": 8,
    "pkda_conv_size": 4,
    "max_seq_len": 16,
}


def tiny(condition="f", *, seed=13):
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


@torch.no_grad()
def test_second_token_prediction_cannot_see_its_target():
    """Editing x[t] can first change the auxiliary prediction of x[t+1]."""
    model = tiny()
    original = tokens()
    changed = original.clone()
    edited_position = 4
    changed[:, edited_position] = (
        changed[:, edited_position] + 7
    ) % model.cfg.vocab_size
    count = 2
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


def test_mtp_loss_matches_materialized_logits_and_feedback_weighting():
    count, z_coef = 2, 0.017
    model = tiny("f")
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
        + EXPERT_BALANCE_COEF
        * torch.stack([out.expert_aux_loss for out in outs]).mean()
    )
    torch.testing.assert_close(actual.total, expected)

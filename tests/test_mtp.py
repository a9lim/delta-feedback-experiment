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
    "expert_intermediate": 8,
    "num_routed_experts": 3,
    "experts_per_token": 2,
    "pkda_heads": 2,
    "pkda_head_dim": 8,
    "pkda_conv_size": 4,
    "max_seq_len": 16,
    "loop_iterations": 2,
    "loop_max_iterations": 2,
}


def tiny(condition="f", *, seed=13):
    torch.manual_seed(seed)
    geometry = GEOMETRY | ({"layers": 16} if "l" in condition else {})
    return DeltaModel(condition_config(condition, **geometry)).eval()


def tokens(length=8):
    generator = torch.Generator().manual_seed(29)
    return torch.randint(0, GEOMETRY["vocab_size"], (2, length), generator=generator)


def passes(model, toks, count):
    prefixes = torch.full((count - 1, toks.shape[0]), 2, dtype=torch.long)
    return multipass(model, toks, count, prefix_lens=prefixes)


def explicit_head_rows(model, hidden, targets):
    """Independent materialized-logit reference: per-row NLL and squared LSE."""
    normalized = hidden.float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(-1, keepdim=True) + model.cfg.norm_eps
    )
    normalized = normalized * model.final_norm.weight.float() * model.cfg.mup_ratio
    logits = F.linear(normalized, model.embed_tokens.weight.float())
    nll = F.cross_entropy(
        logits.flatten(0, 1), targets.reshape(-1), reduction="none"
    ).view_as(targets)
    return nll, torch.logsumexp(logits, dim=-1).square()


def explicit_head_losses(model, hidden, targets):
    """The main head's objective: every row supervised."""
    nll, z = explicit_head_rows(model, hidden, targets)
    return nll.mean(), z.mean()


def explicit_mtp_losses(model, hidden, toks):
    """The auxiliary head's objective: the padded last row carries no weight."""
    targets = F.pad(toks[:, 2:], (0, 1))
    nll, z = explicit_head_rows(model, hidden, targets)
    return nll[:, :-1].mean(), z[:, :-1].mean()


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
    # Keep the selection margin fixed so changing a future token cannot
    # alter the sparse GEMM batch shape used for earlier tokens.
    for bank in model.expert_banks:
        bank.expert_bias[-bank.experts_per_token :] = 2
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
        original_hidden = model.forward_mtp(before.payload, original[:, 1:]).hidden
        changed_hidden = model.forward_mtp(after.payload, changed[:, 1:]).hidden
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


def test_next_token_conditioning_and_payload_are_differentiable():
    model = tiny()
    supplied = tokens(length=5)
    payload = torch.randn(2, 5, model.cfg.dim, requires_grad=True)
    result = model.forward_mtp(payload, supplied).hidden
    assert result.shape == payload.shape
    weights = torch.linspace(-1, 1, result.numel()).reshape_as(result)
    (result * weights).sum().backward()
    assert payload.grad is not None and payload.grad.abs().sum() > 0
    assert model.embed_tokens.weight.grad is not None
    assert model.embed_tokens.weight.grad[supplied.unique()].abs().sum() > 0
    # The supplied next token goes through the same embedding used by the
    # trunk; the auxiliary module must not allocate its own vocabulary table.
    assert not any(
        isinstance(module, torch.nn.Embedding) for module in model.mtp.modules()
    )


@torch.no_grad()
def test_the_padded_auxiliary_row_leaves_the_supervised_rows_unchanged():
    """Appending the unsupervised last row cannot move any earlier row."""
    model = tiny()
    toks = tokens(length=6)
    out = passes(model, toks, 1)[0]
    full = model.forward_mtp(out.payload, toks[:, 1:]).hidden
    cropped = model.forward_mtp(out.payload[:, :-1], toks[:, 1:-1]).hidden
    torch.testing.assert_close(full[:, :-1], cropped, atol=2e-6, rtol=1e-5)


def test_mtp_trains_the_payload_writer_on_every_supervised_pass():
    count = 2
    model = tiny("fl")
    toks = tokens(length=5)
    outs = passes(model, toks, count)
    for out in outs:
        assert out.payload is not None
        auxiliary = model.forward_mtp(out.payload, toks[:, 1:])
        loss, _ = explicit_mtp_losses(model, auxiliary.hidden, toks)
        gradients = torch.autograd.grad(
            loss,
            (
                out.payload,
                model.payload_router.query,
                model.payload_norm.weight,
                model.blocks[0].attn.q_proj.weight,
            ),
            retain_graph=True,
        )
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert all(gradient.abs().sum() > 0 for gradient in gradients)
        # The final payload position has no second-token target.
        assert torch.count_nonzero(gradients[0][:, -1]) == 0


@torch.no_grad()
def test_auxiliary_recurrent_state_is_independent_between_rows_and_calls():
    model = tiny("fl")
    supplied = tokens(length=5)
    payload = torch.randn(2, 5, model.cfg.dim)
    before = model.forward_mtp(payload, supplied)
    model.forward_mtp(-payload, supplied.flip(1))
    passes(model, tokens(), 2)
    after = model.forward_mtp(payload, supplied)
    torch.testing.assert_close(before.hidden, after.hidden, atol=0, rtol=0)
    # Another row cannot seed this row's convolution, preconditioner, or KDA
    # memory. Running each row alone must reproduce the batched predictor.
    for row in range(payload.shape[0]):
        isolated = model.forward_mtp(payload[row : row + 1], supplied[row : row + 1])
        torch.testing.assert_close(
            isolated.hidden, before.hidden[row : row + 1], atol=2e-6, rtol=1e-5
        )


def test_mtp_loss_matches_materialized_logits_and_feedback_weighting():
    count, z_coef, coefficient = 3, 0.017, 0.23
    model = tiny("fl")
    toks = tokens(length=6)
    outs = passes(model, toks, count)
    actual = multipass_loss(model, toks, outs, mtp_weight=coefficient, z_coef=z_coef)
    assert len(actual.ntp) == count and len(actual.mtp) == count
    ntp, mtp, ntp_z, mtp_z = [], [], [], []
    expert_aux, expert_counts = [], []
    for index, out in enumerate(outs):
        ce, z = explicit_head_losses(model, out.h_top, toks[:, 1:])
        auxiliary = model.forward_mtp(out.payload, toks[:, 1:], want_weights=True)
        second_ce, second_z = explicit_mtp_losses(model, auxiliary.hidden, toks)
        invocations = model.cfg.executed_layers(out.iterations)
        expert_aux.append(
            (out.expert_aux_loss * invocations + auxiliary.expert_aux_loss)
            / (invocations + 1)
        )
        # Check dispatch accounting against actual selected expert weights,
        # independently of the loss implementation's count aggregation.
        selected_counts = auxiliary.expert_weights.count_nonzero(dim=(0, 1))
        torch.testing.assert_close(auxiliary.expert_counts, selected_counts)
        assert auxiliary.expert_counts.sum() == (
            toks.shape[0] * (toks.shape[1] - 1) * model.cfg.experts_per_token
        )
        expert_counts.append(
            torch.cat((out.expert_counts, selected_counts.unsqueeze(0)))
        )
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
        + EXPERT_BALANCE_COEF * torch.stack(expert_aux).mean()
    )
    torch.testing.assert_close(actual.total, expected)
    torch.testing.assert_close(actual.expert_aux_loss, torch.stack(expert_aux).mean())
    assert actual.expert_counts.shape == (
        model.cfg.layers + 1,
        model.cfg.num_routed_experts,
    )
    torch.testing.assert_close(actual.expert_counts, torch.stack(expert_counts).sum(0))

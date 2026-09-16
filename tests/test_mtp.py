"""Numerical and causal contracts for the teacher-forced second-token head."""

import pytest
import torch
import torch.nn.functional as F

from delta_feedback_experiment.model import (
    EMBEDDING_LOOKUP_SCALE,
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
}


def tiny(condition="f", *, seed=13):
    torch.manual_seed(seed)
    return DeltaModel(condition_config(condition, **GEOMETRY)).eval()


def tokens(length=8):
    generator = torch.Generator().manual_seed(29)
    return torch.randint(0, GEOMETRY["vocab_size"], (2, length), generator=generator)


def passes(model, toks, count, *, jitter=None, loop_jitter=None, iterations=1):
    """``[pass][column]`` outputs of ``count`` passes with a fixed prefix."""
    prefixes = torch.full((count - 1, toks.shape[0]), 2, dtype=torch.long)
    return multipass(
        model,
        toks,
        count,
        iterations=iterations,
        prefix_lens=prefixes,
        jitter=jitter,
        loop_jitter=loop_jitter,
    )


def columns(outs):
    """Every column of a multipass result in execution order."""
    return [out for pass_outs in outs for out in pass_outs]


def jitter_for(model, toks, count, iterations=1):
    """(pass jitter [k, B, T+1, D], loop jitter [k, r-1, B, T+1, D] or None)."""
    generator = torch.Generator().manual_seed(37)
    jitter = torch.empty(count, *toks.shape, model.cfg.dim).uniform_(
        -0.02, 0.02, generator=generator
    )
    if iterations == 1:
        return jitter, None
    loop_jitter = torch.empty(
        count, iterations - 1, *toks.shape, model.cfg.dim
    ).uniform_(-0.02, 0.02, generator=generator)
    return jitter, loop_jitter


def explicit_embedding(model, toks):
    """Independent lookup: the tied row times the fixed lookup scale."""
    return F.embedding(toks, model.embed_tokens.weight) * EMBEDDING_LOOKUP_SCALE


def explicit_fusion(model, payload, token_embedding):
    """Independent concat equation: scaled token lookup and prepared payload."""
    return F.linear(
        torch.cat((token_embedding, payload), dim=-1), model.fuse_proj.weight
    )


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


def grid_sum(values, count, iterations):
    """FBT's combine on both axes of a pass-major flat list: passes within
    each column, then columns."""
    per_column = [
        feedback_sum([values[p * iterations + c] for p in range(count)])
        for c in range(iterations)
    ]
    return feedback_sum(per_column)


@torch.no_grad()
def test_fusion_preserves_independent_token_and_payload_contributions():
    """Both inputs contribute; payload amplitude and additive jitter survive."""
    model = tiny()
    payload, embedding = torch.randn(2, 4, model.cfg.dim), torch.randn(
        2, 4, model.cfg.dim
    )
    fused = model.fuse(payload, embedding)
    token_only = model.fuse(torch.zeros_like(payload), embedding)
    payload_only = model.fuse(payload, torch.zeros_like(embedding))
    assert torch.all(token_only.norm(dim=-1) > 0)
    assert torch.all(payload_only.norm(dim=-1) > 0)
    torch.testing.assert_close(fused, explicit_fusion(model, payload, embedding))
    torch.testing.assert_close(fused, token_only + payload_only)
    torch.testing.assert_close(
        model.fuse(3 * payload, embedding), token_only + 3 * payload_only
    )
    jitter = torch.randn_like(payload) * 0.2
    torch.testing.assert_close(
        model.fuse(payload + jitter, embedding) - fused,
        F.linear(jitter, model.fuse_proj.weight[:, model.cfg.dim :]),
        atol=1e-6,
        rtol=1e-5,
    )
    model.fuse_proj.weight.mul_(2)
    torch.testing.assert_close(model.fuse(payload, embedding), 2 * fused)


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
    jitter, _ = jitter_for(model, original, count)
    original_out = columns(passes(model, original, count, jitter=jitter))
    changed_out = columns(passes(model, changed, count, jitter=jitter))

    for before, after in zip(original_out, changed_out, strict=True):
        original_hidden = model.forward_mtp_fused(before.fused_input).hidden
        changed_hidden = model.forward_mtp_fused(after.fused_input).hidden
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
    assert set(model.mtp._modules) == {"block"}


@torch.no_grad()
def test_the_padded_auxiliary_row_leaves_the_supervised_rows_unchanged():
    """Appending the unsupervised last row cannot move any earlier row."""
    model = tiny()
    toks = tokens(length=6)
    out = passes(model, toks, 1)[0][0]
    full = model.forward_mtp(out.payload, toks[:, 1:]).hidden
    cropped = model.forward_mtp(out.payload[:, :-1], toks[:, 1:-1]).hidden
    torch.testing.assert_close(full[:, :-1], cropped, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("condition, count", [("f", 1), ("l", 1), ("fl", 2)])
def test_mtp_trains_shared_fusion_and_payload_on_every_column(condition, count):
    model = tiny(condition)
    toks = tokens(length=5)
    iterations = 2 if model.cfg.loop else 1
    jitter, loop_jitter = jitter_for(model, toks, count, iterations)
    outs = columns(
        passes(
            model, toks, count, iterations=iterations, jitter=jitter,
            loop_jitter=loop_jitter,
        )
    )
    assert len(outs) == count * iterations
    for out in outs:
        assert out.payload is not None
        auxiliary = model.forward_mtp_fused(out.fused_input)
        loss, _ = explicit_mtp_losses(model, auxiliary.hidden, toks)
        gradients = torch.autograd.grad(
            loss,
            (
                out.payload,
                model.payload_router.query,
                model.payload_norm.weight,
                model.blocks[0].attn.q_proj.weight,
                model.fuse_proj.weight,
                model.embed_tokens.weight,
            ),
            retain_graph=True,
        )
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert all(gradient.abs().sum() > 0 for gradient in gradients)
        assert all(
            gradient.abs().sum() > 0
            for gradient in gradients[4].split(model.cfg.dim, dim=1)
        )
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


def test_reused_fusion_matches_duplicated_values_gradients_and_head_losses():
    """Independent consumers recompute the jittered concat equation; sharing that
    input and raw token lookup must preserve both objectives' derivatives."""
    count, z_coef, coefficient = 3, 0.017, 0.23
    model, reference = tiny("fl"), tiny("fl")
    for candidate in (model, reference):
        for bank in candidate.expert_banks:
            bank.expert_bias[-bank.experts_per_token :] = 2
        with torch.no_grad():
            candidate.blank_payload.copy_(torch.linspace(-1, 1, candidate.cfg.dim))
    toks = tokens(length=6)
    iterations = 2
    jitter, loop_jitter = jitter_for(model, toks, count, iterations)
    prefixes = torch.tensor([[1, 3], [4, 2]])
    outs = multipass(
        model, toks, count, iterations=iterations, prefix_lens=prefixes,
        jitter=jitter, loop_jitter=loop_jitter,
    )
    actual = multipass_loss(model, toks, outs, mtp_weight=coefficient, z_coef=z_coef)
    assert [len(pass_outs) for pass_outs in actual.ntp] == [iterations] * count
    assert [len(pass_outs) for pass_outs in actual.mtp] == [iterations] * count

    # Feedback, the loop, and MTP deliberately perform their own embedding
    # lookups and fusion. They share parameters, but have no shared prepared
    # tensor. The plain seed is the same concat equation with the blank
    # payload; a column's jitter row is the pass draw at a pass's last column
    # and the loop draw before it.
    e = explicit_embedding(reference, toks[:, :-1])
    seed = explicit_fusion(reference, reference.blank_payload.expand_as(e), e)
    length = e.shape[1]

    def draw(index, column):
        source = jitter[index] if column + 1 == iterations else loop_jitter[index, column]
        return source[:, :length]

    reference_outs = []
    x = seed
    for index in range(count):
        pass_outs = []
        for column in range(iterations):
            pass_outs.append(reference.forward_column(x))
            payload = pass_outs[-1].payload + draw(index, column)
            if column + 1 < iterations:
                # The loop re-enters with the position's own raw embedding.
                x = explicit_fusion(reference, payload, e)
        reference_outs.append(pass_outs)
        if index + 1 < count:
            # Keep the production order (fuse, then shift): changing the rows
            # presented to the matrix product adds FP32 roundoff that the deep
            # recurrence amplifies. The two consumers still recompute separately.
            unshifted = explicit_fusion(
                reference, payload, explicit_embedding(reference, toks[:, 1:])
            )
            fused = torch.cat((torch.zeros_like(unshifted[:, :1]), unshifted[:, :-1]), 1)
            plain = torch.arange(length)[None, :] < prefixes[index, :, None]
            x = torch.where(plain[..., None], seed, fused)

    ntp, mtp, ntp_z, mtp_z = [], [], [], []
    expert_aux, expert_counts = [], []
    pairs = [
        (index, column, out, expected_out)
        for index, (pass_outs, expected_pass) in enumerate(
            zip(outs, reference_outs, strict=True)
        )
        for column, (out, expected_out) in enumerate(
            zip(pass_outs, expected_pass, strict=True)
        )
    ]
    for index, column, out, expected_out in pairs:
        torch.testing.assert_close(out.h_top, expected_out.h_top)
        torch.testing.assert_close(out.payload, expected_out.payload)
        ce, z = explicit_head_losses(reference, expected_out.h_top, toks[:, 1:])
        fused = explicit_fusion(
            reference,
            expected_out.payload + draw(index, column),
            explicit_embedding(reference, toks[:, 1:]),
        )
        torch.testing.assert_close(out.fused_input, fused)
        hidden, _, _, _, aux, weights, counts = reference.mtp.block(
            fused, None, None, None, True
        )
        second_ce, second_z = explicit_mtp_losses(reference, hidden, toks)
        invocations = reference.cfg.layers
        expert_aux.append(
            (expected_out.expert_aux_loss * invocations + aux)
            / (invocations + 1)
        )
        # Check dispatch accounting against actual selected expert weights,
        # independently of the loss implementation's count aggregation.
        selected_counts = weights.count_nonzero(dim=(0, 1))
        torch.testing.assert_close(counts, selected_counts)
        assert counts.sum() == (
            toks.shape[0] * (toks.shape[1] - 1) * model.cfg.experts_per_token
        )
        expert_counts.append(
            torch.cat((expected_out.expert_counts, selected_counts.unsqueeze(0)))
        )
        ntp.append(ce)
        ntp_z.append(z)
        mtp.append(second_ce)
        mtp_z.append(second_z)
        torch.testing.assert_close(actual.ntp[index][column], ce)
        torch.testing.assert_close(actual.mtp[index][column], second_ce)
    # First-plus-mean along passes, then along columns, for both heads and
    # their z-losses.
    expected = (
        grid_sum(ntp, count, iterations)
        + z_coef * grid_sum(ntp_z, count, iterations)
        + coefficient * (
            grid_sum(mtp, count, iterations)
            + z_coef * grid_sum(mtp_z, count, iterations)
        )
        + EXPERT_BALANCE_COEF * torch.stack(expert_aux).mean()
    )
    torch.testing.assert_close(actual.total, expected)
    torch.testing.assert_close(actual.expert_aux_loss, torch.stack(expert_aux).mean())
    assert actual.expert_counts.shape == (
        model.cfg.layers + 1,
        model.cfg.num_routed_experts,
    )
    torch.testing.assert_close(actual.expert_counts, torch.stack(expert_counts).sum(0))
    actual.total.backward()
    expected.backward()
    for (name, parameter), (_, expected_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert (parameter.grad is None) == (expected_parameter.grad is None), name
        if parameter.grad is not None:
            # Reusing the lookup and fusion changes FP32 gradient summation
            # order through the repeated feedback passes and looped columns. The
            # small raw seeds amplify that drift in early PKDA decay gradients
            # even with identical forward states. Bound aggregate drift and
            # individual coordinates relative to the tensor's RMS.
            error = parameter.grad - expected_parameter.grad
            scale = expected_parameter.grad.square().mean().sqrt()
            assert error.square().mean().sqrt() < 2e-6 + 5e-5 * scale, name
            assert error.abs().max() < 2e-6 + 2e-4 * scale, name


@pytest.mark.parametrize("condition, iterations", [("f", 1), ("fl", 2)])
def test_training_reuses_one_lookup_and_one_scaled_payload_per_column(
    monkeypatch, condition, iterations
):
    """One raw lookup serves every column; each column writes one scaled
    payload and fuses it once for MTP, plus once more to re-enter the loop
    when another column follows; pass 1 adds the one blank-payload fusion of
    its seed."""
    import delta_feedback_experiment.model as implementation

    model = tiny(condition)
    toks, count = tokens(length=5), 3
    calls = {"embedding": 0, "payload_norm": 0, "fusion": 0}
    auxiliary_inputs = []
    written_payloads = []
    sink_linear = implementation.sink_linear

    def count_linear(x, weights, *args, **kwargs):
        if weights[0] is model.fuse_proj.weight:
            calls["fusion"] += 1
        return sink_linear(x, weights, *args, **kwargs)

    def count_embedding(module, args, result):
        assert result.shape == (*toks.shape, model.cfg.dim)
        calls["embedding"] += 1

    def count_payload_norm(module, args, result):
        calls["payload_norm"] += 1
        written_payloads.append(result)

    monkeypatch.setattr(implementation, "sink_linear", count_linear)
    embedding_hook = model.embed_tokens.register_forward_hook(count_embedding)
    payload_norm_hook = model.payload_norm.register_forward_hook(count_payload_norm)
    mtp_hook = model.mtp.register_forward_pre_hook(
        lambda module, args: auxiliary_inputs.append(args[0])
    )
    try:
        jitter, loop_jitter = jitter_for(model, toks, count, iterations)
        outs = passes(
            model, toks, count, iterations=iterations, jitter=jitter,
            loop_jitter=loop_jitter,
        )
        multipass_loss(model, toks, outs)
    finally:
        embedding_hook.remove()
        payload_norm_hook.remove()
        mtp_hook.remove()
    executed = count * iterations
    assert calls == {
        "embedding": 1,
        "payload_norm": executed,
        "fusion": 1 + executed + count * (iterations - 1),
    }
    assert all(
        payload is out.payload
        for payload, out in zip(written_payloads, columns(outs), strict=True)
    )
    assert len(auxiliary_inputs) == executed
    assert all(
        auxiliary is out.fused_input
        for auxiliary, out in zip(auxiliary_inputs, columns(outs), strict=True)
    )

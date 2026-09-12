"""Independent value/gradient oracles for expert selection and balancing."""

import torch
import torch.nn.functional as F

from delta_feedback_experiment.moe import MixtureOfExperts


def explicit_expert(expert, x):
    gate, up = F.linear(x, expert.gate_up_proj.weight).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, expert.down_proj.weight)


def explicit_mixture(moe, x):
    """Dense oracle evaluates all experts before selecting the routed sum."""
    scores = F.linear(x, moe.router.weight).sigmoid()
    indices = (scores + moe.expert_bias).topk(3, dim=-1).indices
    values = scores.gather(-1, indices)
    weights = torch.zeros_like(scores).scatter(
        -1, indices, values / values.sum(dim=-1, keepdim=True)
    )
    routed = torch.stack([explicit_expert(expert, x) for expert in moe.experts], dim=-2)
    y = (
        explicit_expert(moe.shared, x) + 3 * (weights.unsqueeze(-1) * routed).sum(-2)
    ) / 2
    selected = torch.zeros_like(scores, dtype=torch.int64).scatter(-1, indices, 1)
    normalized_scores = scores / scores.sum(dim=-1, keepdim=True)
    unbiased_selection = torch.zeros_like(scores).scatter(
        -1, scores.topk(3, dim=-1).indices, 1.0
    )
    fraction = unbiased_selection.mean(dim=1).detach() / 3
    aux = 15 * (normalized_scores.mean(dim=1) * fraction).sum(dim=-1).mean()
    return y, aux, weights, selected.sum(dim=(0, 1))


def test_sparse_mixture_matches_dense_oracle_and_gradients():
    torch.manual_seed(12)
    moe = MixtureOfExperts(8, 16).double()
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    actual = moe(x, want_weights=True)
    expected = explicit_mixture(moe, x)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-11)
    variables = (x, *moe.parameters())
    probe = torch.randn_like(actual[0])
    actual_grad = torch.autograd.grad(
        (actual[0] * probe).sum() + 0.07 * actual[1], variables, allow_unused=True
    )
    expected_grad = torch.autograd.grad(
        (expected[0] * probe).sum() + 0.07 * expected[1], variables
    )
    for parameter, got, want in zip(variables, actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(
            torch.zeros_like(parameter) if got is None else got,
            want,
            rtol=1e-9,
            atol=1e-10,
        )


def test_selection_bias_does_not_enter_mixture_weights():
    moe = MixtureOfExperts(4, 8)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[:, 0].copy_(torch.linspace(-2, 2, 15))
    x = torch.ones(1, 3, 4)
    _, original_aux, original_weights, original_counts = moe(x, want_weights=True)
    assert torch.count_nonzero(original_weights[..., :12]) == 0
    with torch.no_grad():
        moe.expert_bias[:3] = 2
    output, aux, weights, counts = moe(x, want_weights=True)
    scores = torch.linspace(-2, 2, 15).sigmoid()[:3]
    torch.testing.assert_close(weights[0, 0, :3], scores / scores.sum())
    assert torch.count_nonzero(weights[..., 3:]) == 0
    assert counts.tolist() == [3] * 3 + [0] * 12
    assert not torch.equal(counts, original_counts)
    torch.testing.assert_close(aux, original_aux, rtol=0, atol=0)
    expected, expected_aux, _, _ = explicit_mixture(moe, x)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(aux, expected_aux)
    with torch.no_grad():
        moe.expert_bias[:3] = 4
    unchanged = moe(x, want_weights=True)
    for before, after in zip((output, aux, weights, counts), unchanged, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_sequence_auxiliary_does_not_pool_independent_sequences():
    moe = MixtureOfExperts(4, 8)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[:, 0].copy_(torch.linspace(-4, 4, 15))
    x = torch.ones(2, 4, 4, requires_grad=True)
    with torch.no_grad():
        x[1].neg_()
    _, aux, weights, _ = moe(x, want_weights=True)
    first = moe(x[:1])[1]
    second = moe(x[1:])[1]
    torch.testing.assert_close(aux, (first + second) / 2)
    scores = F.linear(x, moe.router.weight).sigmoid()
    normalized = scores / scores.sum(-1, keepdim=True)
    pooled_load = (weights > 0).float().mean(dim=(0, 1)) / 3
    pooled = 15 * (normalized.mean(dim=(0, 1)) * pooled_load).sum()
    assert not torch.isclose(aux, pooled)
    joint_gradient = torch.autograd.grad(aux, x, retain_graph=True)[0][0]
    isolated_gradient = torch.autograd.grad(first, x)[0][0]
    torch.testing.assert_close(joint_gradient, isolated_gradient / 2)


def test_bias_update_uses_global_counts_and_has_no_optimizer_gradient():
    moe = MixtureOfExperts(4, 8)
    assert "expert_bias" in dict(moe.named_buffers())
    assert "expert_bias" not in dict(moe.named_parameters())
    assert not moe.expert_bias.requires_grad
    counts = torch.tensor([27, 15, 3] + [0] * 12, dtype=torch.int64)
    direction = torch.tensor([-1, -1, 0] + [1] * 12, dtype=torch.float32)
    moe.update_bias(counts)
    torch.testing.assert_close(moe.expert_bias, 0.001 * direction)
    moe.update_bias(counts * 100, rate=0.002)
    torch.testing.assert_close(moe.expert_bias, 0.003 * direction)
    assert moe.expert_bias.grad_fn is None and moe.expert_bias.grad is None
    before = moe.expert_bias.clone()
    moe.update_bias(torch.zeros_like(counts))
    moe.update_bias(torch.ones_like(counts) * 3)
    torch.testing.assert_close(moe.expert_bias, before, rtol=0, atol=0)
    moe.eval()
    moe.update_bias(counts)
    torch.testing.assert_close(moe.expert_bias, before, rtol=0, atol=0)

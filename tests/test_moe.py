"""Independent value/gradient oracles for expert selection and balancing."""

import copy
import math

import pytest
import torch
import torch.nn.functional as F

from delta_feedback_experiment.moe import MixtureOfExperts

GEOMETRIES = [(15, 3), (23, 5), (31, 7)]


def explicit_expert(expert, x):
    gate, up = F.linear(x, expert.gate_up_proj.weight).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, expert.down_proj.weight)


def explicit_mixture(moe, x):
    """Dense oracle evaluates all experts before selecting the routed sum."""
    count, selected_count = moe.num_routed_experts, moe.experts_per_token
    scores = F.linear(x, moe.router.weight).sigmoid()
    indices = (scores + moe.expert_bias).topk(selected_count, dim=-1).indices
    values = scores.gather(-1, indices)
    weights = torch.zeros_like(scores).scatter(
        -1, indices, values / values.sum(dim=-1, keepdim=True)
    )
    routed = torch.stack([explicit_expert(expert, x) for expert in moe.experts], dim=-2)
    y = (
        explicit_expert(moe.shared, x)
        + selected_count * (weights.unsqueeze(-1) * routed).sum(-2)
    ) / math.sqrt(selected_count + 1)
    selected = torch.zeros_like(scores, dtype=torch.int64).scatter(-1, indices, 1)
    normalized_scores = scores / scores.sum(dim=-1, keepdim=True)
    unbiased_selection = torch.zeros_like(scores).scatter(
        -1, scores.topk(selected_count, dim=-1).indices, 1.0
    )
    fraction = unbiased_selection.mean(dim=1).detach() / selected_count
    aux = count * (normalized_scores.mean(dim=1) * fraction).sum(dim=-1).mean()
    return y, aux, weights, selected.sum(dim=(0, 1))


@pytest.mark.parametrize("count,selected_count", GEOMETRIES)
def test_sparse_mixture_matches_dense_oracle_and_gradients(count, selected_count):
    torch.manual_seed(12)
    moe = MixtureOfExperts(8, 4, count, selected_count).double()
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


@pytest.mark.parametrize("count,selected_count", GEOMETRIES)
def test_selection_bias_does_not_enter_mixture_weights(count, selected_count):
    moe = MixtureOfExperts(4, 2, count, selected_count)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[:, 0].copy_(torch.linspace(-2, 2, count))
    x = torch.ones(1, 3, 4)
    _, original_aux, original_weights, original_counts = moe(x, want_weights=True)
    assert torch.count_nonzero(original_weights[..., : count - selected_count]) == 0
    assert original_counts[-1] == 3
    with torch.no_grad():
        moe.expert_bias[:selected_count] = 2
    output, aux, weights, counts = moe(x, want_weights=True)
    scores = torch.linspace(-2, 2, count).sigmoid()[:selected_count]
    torch.testing.assert_close(weights[0, 0, :selected_count], scores / scores.sum())
    assert torch.count_nonzero(weights[..., selected_count:]) == 0
    assert counts.tolist() == [3] * selected_count + [0] * (count - selected_count)
    assert not torch.equal(counts, original_counts)
    torch.testing.assert_close(aux, original_aux, rtol=0, atol=0)
    expected, expected_aux, _, _ = explicit_mixture(moe, x)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(aux, expected_aux)
    with torch.no_grad():
        moe.expert_bias[:selected_count] = 4
    unchanged = moe(x, want_weights=True)
    for before, after in zip((output, aux, weights, counts), unchanged, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize("count,selected_count", GEOMETRIES)
def test_sequence_auxiliary_does_not_pool_independent_sequences(count, selected_count):
    moe = MixtureOfExperts(4, 2, count, selected_count)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[:, 0].copy_(torch.linspace(-4, 4, count))
    x = torch.ones(2, 4, 4, requires_grad=True)
    with torch.no_grad():
        x[1].neg_()
    _, aux, weights, _ = moe(x, want_weights=True)
    first = moe(x[:1])[1]
    second = moe(x[1:])[1]
    torch.testing.assert_close(aux, (first + second) / 2)
    scores = F.linear(x, moe.router.weight).sigmoid()
    normalized = scores / scores.sum(-1, keepdim=True)
    pooled_load = (weights > 0).float().mean(dim=(0, 1)) / selected_count
    pooled = count * (normalized.mean(dim=(0, 1)) * pooled_load).sum()
    assert not torch.isclose(aux, pooled)
    joint_gradient = torch.autograd.grad(aux, x, retain_graph=True)[0][0]
    isolated_gradient = torch.autograd.grad(first, x)[0][0]
    torch.testing.assert_close(joint_gradient, isolated_gradient / 2)


@pytest.mark.parametrize("count,selected_count", GEOMETRIES)
def test_bias_update_uses_global_counts_and_has_no_optimizer_gradient(
    count, selected_count
):
    moe = MixtureOfExperts(4, 2, count, selected_count)
    assert "expert_bias" in dict(moe.named_buffers())
    assert "expert_bias" not in dict(moe.named_parameters())
    assert not moe.expert_bias.requires_grad
    counts = torch.tensor(
        [2 * count - 3, count, 3] + [0] * (count - 3), dtype=torch.int64
    )
    direction = torch.tensor([-1, -1, 0] + [1] * (count - 3), dtype=torch.float32)
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


@pytest.mark.parametrize("count,selected_count", GEOMETRIES)
def test_equal_affinities_normalize_all_active_branches(count, selected_count):
    moe = MixtureOfExperts(4, 2, count, selected_count).double()
    with torch.no_grad():
        moe.router.weight.zero_()
        for expert in moe.experts:
            expert.load_state_dict(moe.shared.state_dict())
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    output, auxiliary, weights, counts = moe(x, want_weights=True)
    torch.testing.assert_close(
        output, math.sqrt(selected_count + 1) * explicit_expert(moe.shared, x)
    )
    torch.testing.assert_close(auxiliary, auxiliary.new_tensor(1.0))
    torch.testing.assert_close(
        weights[weights > 0],
        weights.new_full((6 * selected_count,), 1 / selected_count),
    )
    assert counts.sum() == 6 * selected_count


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA MoE kernels")
@pytest.mark.parametrize("count,selected_count", GEOMETRIES)
def test_cuda_sparse_mixture_matches_dense_values_and_gradients(count, selected_count):
    torch.manual_seed(17)
    moe = MixtureOfExperts(32, 24, count, selected_count).cuda()
    # Leave most experts empty, including expert zero, and exercise the highest
    # index. Freeze different selected gate/down matrices to test autograd's
    # dynamic parameter slices independently of the dispatch mask.
    with torch.no_grad():
        moe.expert_bias[-selected_count:] = 2
    moe.experts[-selected_count].gate_up_proj.weight.requires_grad_(False)
    moe.experts[-selected_count + 1].down_proj.weight.requires_grad_(False)
    x = torch.randn(2, 19, 32, device="cuda", requires_grad=True)
    actual = moe(x, want_weights=True)
    expected = explicit_mixture(moe, x)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-6)
    assert actual[3].tolist() == [0] * (count - selected_count) + [38] * selected_count
    variables = (
        x,
        *(parameter for parameter in moe.parameters() if parameter.requires_grad),
    )
    cotangent = torch.randn_like(actual[0])
    actual_gradients = torch.autograd.grad(
        (actual[0] * cotangent).sum() + 0.07 * actual[1], variables
    )
    expected_gradients = torch.autograd.grad(
        (expected[0] * cotangent).sum() + 0.07 * expected[1], variables
    )
    for got, want in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(got, want, rtol=3e-4, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA MoE kernels")
def test_cuda_compiled_mixture_accumulates_mixed_packed_sinks_and_autograd():
    torch.manual_seed(29)
    count, selected_count, dim, width = 31, 7, 32, 24
    moe = MixtureOfExperts(dim, width, count, selected_count).cuda()
    with torch.no_grad():
        moe.expert_bias[-selected_count:] = 2
    moe.experts[24].gate_up_proj.weight.requires_grad_(False)
    moe.experts[25].down_proj.weight.requires_grad_(False)
    reference = copy.deepcopy(moe)
    # Per-matrix slices have nonzero storage offsets. Guard experts on both
    # ends catch addressing the backing allocation instead of the slice.
    gate_backing = torch.full((count + 2, 2 * width, dim), 0.25, device="cuda")
    down_backing = torch.full((count + 2, dim, width), 0.25, device="cuda")
    sinks = {}
    for index, expert in enumerate(moe.experts):
        if index % 2 == 0:
            sinks[expert.gate_up_proj.weight] = gate_backing[index + 1]
        if index % 2 == 1 or index == count - 1:
            sinks[expert.down_proj.weight] = down_backing[index + 1]
    bound, refresh = moe.bind_gradient_sinks(sinks)
    assert bound == {parameter for parameter in sinks if parameter.requires_grad}
    assert moe._gate_up_shadow.shape == (count, 2 * width, dim)
    assert moe._down_shadow.shape == (count, dim, width)
    assert len(refresh) == 2 * count
    gate_shadow, down_shadow = moe._gate_up_shadow, moe._down_shadow
    moe.bind_gradient_sinks(sinks)
    assert moe._gate_up_shadow is gate_shadow and moe._down_shadow is down_shadow
    compiled = torch.compile(moe, fullgraph=True)
    reference_parameters = dict(reference.named_parameters())
    for _ in range(2):
        x = torch.randn(2, 19, dim, device="cuda", requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        actual = compiled(x, want_weights=True)
        expected = explicit_mixture(reference, reference_x)
        cotangent = torch.randn_like(actual[0])
        ((actual[0] * cotangent).sum() + 0.07 * actual[1]).backward()
        ((expected[0] * cotangent).sum() + 0.07 * expected[1]).backward()
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-6)
        torch.testing.assert_close(x.grad, reference_x.grad, rtol=3e-4, atol=2e-5)
        for name, parameter in moe.named_parameters():
            reference_gradient = reference_parameters[name].grad
            if not parameter.requires_grad:
                assert parameter.grad is None and reference_gradient is None
                if parameter in sinks:
                    torch.testing.assert_close(
                        sinks[parameter], torch.full_like(parameter, 0.25)
                    )
            elif parameter in bound:
                assert parameter.grad is None
                torch.testing.assert_close(
                    sinks[parameter], 0.25 + reference_gradient, rtol=3e-4, atol=2e-5
                )
            else:
                torch.testing.assert_close(
                    parameter.grad, reference_gradient, rtol=3e-4, atol=2e-5
                )
        for backing in (gate_backing, down_backing):
            torch.testing.assert_close(backing[0], torch.full_like(backing[0], 0.25))
            torch.testing.assert_close(backing[-1], torch.full_like(backing[-1], 0.25))
    moe.bind_gradient_sinks(None)
    assert moe._gate_up_shadow is None and moe._down_shadow is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA MoE kernels")
@pytest.mark.parametrize(
    "dim,count,selected_count",
    [(768, 15, 3), (1152, 23, 5), (1536, 31, 7)],
    ids=["screen", "bridge", "flagship"],
)
def test_cuda_bf16_preset_bank_matches_dense_fp32_oracle(dim, count, selected_count):
    """Qualify real projection shapes, without allocating a full model.

    The reference uses ordinary dense FP32 operations on the same quantized
    input. The production bank uses BF16 shadows/activations and FP32 gradient
    sinks. Relative L2 comparisons allow BF16 rounding without relaxing the
    separate FP32 CUDA oracle or obscuring errors in small router gradients.
    """
    torch.manual_seed(307)
    moe = MixtureOfExperts(dim, 832, count, selected_count).cuda()
    with torch.no_grad():
        moe.expert_bias[-selected_count:] = 2
    reference = copy.deepcopy(moe)
    sinks = {
        parameter: torch.zeros_like(parameter)
        for name, parameter in moe.named_parameters()
        if name != "router.weight"
    }
    bound, _ = moe.bind_gradient_sinks(sinks)
    assert bound == set(sinks)
    assert moe._gate_up_shadow.dtype == torch.bfloat16
    assert moe._down_shadow.dtype == torch.bfloat16
    x = torch.randn(
        1, 65, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    reference_x = x.detach().float().requires_grad_()
    actual = moe(x, want_weights=True)
    expected = explicit_mixture(reference, reference_x)
    assert actual[0].dtype == torch.bfloat16
    assert actual[3].tolist() == [0] * (count - selected_count) + [65] * selected_count
    for got, want in zip(actual[1:], expected[1:], strict=True):
        torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)
    cotangent = torch.randn_like(reference_x)
    ((actual[0].float() * cotangent).sum() + 0.07 * actual[1]).backward()
    ((expected[0] * cotangent).sum() + 0.07 * expected[1]).backward()

    comparisons = {
        "output": [(actual[0], expected[0])],
        "input_gradient": [(x.grad, reference_x.grad)],
        "router_gradient": [(moe.router.weight.grad, reference.router.weight.grad)],
        "shared_gradients": [],
        "routed_gate_gradients": [],
        "routed_down_gradients": [],
    }
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in moe.named_parameters():
        if parameter not in sinks:
            continue
        assert parameter.grad is None
        assert sinks[parameter].dtype == torch.float32
        expected_gradient = reference_parameters[name].grad
        if name.startswith("shared."):
            group = "shared_gradients"
        elif ".gate_up_proj." in name:
            group = "routed_gate_gradients"
        else:
            group = "routed_down_gradients"
        comparisons[group].append((sinks[parameter], expected_gradient))
        if name.startswith("experts."):
            index = int(name.split(".")[1])
            if index < count - selected_count:
                assert torch.count_nonzero(sinks[parameter]) == 0
                assert torch.count_nonzero(expected_gradient) == 0

    errors = {}
    for name, pairs in comparisons.items():
        squared_error = sum(
            (got.float() - want.float()).square().sum() for got, want in pairs
        )
        squared_reference = sum(want.float().square().sum() for _, want in pairs)
        assert squared_reference > 0
        relative_error = (squared_error / squared_reference).sqrt().item()
        errors[name] = relative_error
        assert relative_error < 0.03, (name, relative_error)
    print(
        f"BF16 MoE {dim}/832 {selected_count}/{count} relative L2: "
        + ", ".join(f"{name}={error:.5f}" for name, error in errors.items())
    )

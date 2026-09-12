"""Portable contracts for the shared-plus-routed expert condition."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from delta_feedback_experiment.model import (
    DeltaModel,
    KVCache,
    condition_config,
    multipass,
    multipass_loss,
    parse_condition,
)
from delta_feedback_experiment.moe import MixtureOfExperts
from delta_feedback_experiment.parameter_groups import (
    is_normuonh_parameter,
    is_width_scaled_parameter,
)

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
    "max_seq_len": 16,
}


def tiny(condition, *, seed=7, **overrides):
    geometry = (
        TINY
        | (
            {"layers": 12, "loop_iterations": 2, "loop_max_iterations": 3}
            if "l" in condition
            else {}
        )
        | overrides
    )
    torch.manual_seed(seed)
    return DeltaModel(condition_config(condition, **geometry))


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


def test_expert_geometry_and_condition():
    assert parse_condition("lfera") == "aerfl"
    assert condition_config("e", **TINY).experts
    assert not condition_config("", **TINY).experts
    with pytest.raises(ValueError, match="divisible|multiple"):
        condition_config("e", **(TINY | {"intermediate": 30}))
    with pytest.raises(ValueError, match="divisible|multiple"):
        MixtureOfExperts(8, 30)
    moe = MixtureOfExperts(8, 16)
    assert len(moe.experts) == 15
    assert moe.router.weight.shape == (15, 8)
    for expert in [moe.shared, *moe.experts]:
        assert expert.gate_up_proj.weight.shape == (8, 8)
        assert expert.down_proj.weight.shape == (8, 4)
    assert sum(p.numel() for p in moe.parameters()) == 4 * (3 * 8 * 16) + 15 * 8


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


def test_expert_input_and_router_gradcheck():
    torch.manual_seed(18)
    moe = MixtureOfExperts(4, 8).double()
    x = torch.randn(1, 2, 4, dtype=torch.float64, requires_grad=True)
    router = moe.router.weight.detach().clone().requires_grad_()

    def evaluate(inputs, router_weight):
        output, aux, _, _ = torch.func.functional_call(
            moe, {"router.weight": router_weight}, (inputs,)
        )
        return output, aux

    assert torch.autograd.gradcheck(evaluate, (x, router), fast_mode=True)


def test_top_three_normalization_and_selected_expert_gradients():
    moe = MixtureOfExperts(4, 8)
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[:, 0].copy_(torch.linspace(-1, 1, 15))
    x = torch.ones(2, 4, 4)
    output, auxiliary, weights, counts = moe(x, want_weights=True)
    assert weights.shape == (2, 4, 15)
    assert torch.all((weights != 0).sum(-1) == 3)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4))
    assert torch.count_nonzero(weights[..., :12]) == 0
    assert counts.dtype == torch.int64
    assert counts.tolist() == [0] * 12 + [8] * 3
    output.square().sum().backward(retain_graph=True)
    assert moe.shared.down_proj.weight.grad.norm() > 0
    for index, expert in enumerate(moe.experts):
        for parameter in expert.parameters():
            if index < 12:
                assert (
                    parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
                )
            else:
                assert parameter.grad is not None and parameter.grad.norm() > 0
    auxiliary_gradient = torch.autograd.grad(auxiliary, moe.router.weight)[0]
    assert auxiliary_gradient[:12].norm() > 0


def test_forward_routing_is_token_local_even_when_load_changes():
    torch.manual_seed(24)
    moe = MixtureOfExperts(8, 16)
    x = torch.randn(2, 5, 8)
    output, _, weights, _ = moe(x, want_weights=True)
    short, _, short_weights, _ = moe(x[:1, :2], want_weights=True)
    torch.testing.assert_close(short, output[:1, :2])
    torch.testing.assert_close(short_weights, weights[:1, :2])
    assert moe(x)[2] is None


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


@pytest.mark.parametrize("training", [True, False])
def test_fixed_bias_forward_is_pure(training):
    moe = MixtureOfExperts(4, 8).train(training)
    with torch.no_grad():
        moe.expert_bias.copy_(torch.linspace(-0.03, 0.04, 15))
    before = {name: value.clone() for name, value in moe.state_dict().items()}
    x = torch.randn(2, 3, 4, requires_grad=True)
    output, aux, _, counts = moe(x)
    saved_counts = counts.clone()
    (output.square().mean() + 1e-4 * aux).backward()
    moe(x)
    for name, value in moe.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    torch.testing.assert_close(counts, saved_counts, rtol=0, atol=0)
    assert counts.sum() == 3 * 2 * 3
    assert not counts.requires_grad


@pytest.mark.parametrize("condition", ["e", "erf", "erfl"])
def test_expert_model_is_token_causal(condition):
    model = tiny(condition).eval()
    tokens = torch.randint(0, 31, (1, 7))
    changed = tokens.clone()
    changed[:, 4:] = (changed[:, 4:] + 3) % 31
    passes = 2 if "f" in condition else 1
    kwargs = (
        {"prefix_lens": torch.ones(passes - 1, 1, dtype=torch.long)}
        if passes > 1
        else {}
    )
    original = multipass(model, tokens, passes, want_weights=True, **kwargs)
    perturbed = multipass(model, changed, passes, want_weights=True, **kwargs)
    for before, after in zip(original, perturbed, strict=True):
        torch.testing.assert_close(before.h_top[:, :4], after.h_top[:, :4])
        for site, weights in before.expert_weights.items():
            torch.testing.assert_close(
                weights[:, :4], after.expert_weights[site][:, :4]
            )


@pytest.mark.parametrize("condition", ["e", "erfl"])
def test_cached_expert_forward_matches_full_sequence(condition):
    model = tiny(condition).eval()
    tokens = torch.randint(0, 31, (1, 6))
    with torch.no_grad():
        full = model.forward_column(model.embed_tokens(tokens), want_weights=True)
        cache = KVCache(model.cfg, batch=1, device="cpu", dtype=torch.float32)
        columns = [
            model.step(tokens[:, t : t + 1], None, cache, want_weights=True)
            for t in range(tokens.shape[1])
        ]
    torch.testing.assert_close(
        torch.cat([out.h_top for out in columns], dim=1),
        full.h_top,
        # Tokenwise and full-row CPU GEMMs differ by a few FP32 ulps;
        # this recurrent fixture reaches 2.5e-6 on Jobe's CPU backend.
        atol=4e-6,
        rtol=2e-5,
    )
    for site, weights in full.expert_weights.items():
        cached_weights = torch.cat([out.expert_weights[site] for out in columns], dim=1)
        assert torch.equal(cached_weights.ne(0), weights.ne(0))
        torch.testing.assert_close(
            cached_weights,
            weights,
            atol=2e-6,
            rtol=2e-5,
        )


def test_auxiliary_loss_is_mean_over_executed_layers():
    model = tiny("erfl")
    losses = []
    hooks = [
        block.mlp.register_forward_hook(
            lambda _module, _args, output: losses.append(output[1])
        )
        for block in model.blocks
    ]
    try:
        out = model.forward_column(
            torch.randn(1, 5, 16), iterations=3, want_weights=True
        )
    finally:
        for hook in hooks:
            hook.remove()
    assert len(losses) == 20
    assert len(out.expert_weights) == 20
    torch.testing.assert_close(out.expert_aux_loss, torch.stack(losses).mean())
    plain = tiny("").forward_column(torch.randn(1, 5, 16), want_weights=True)
    assert plain.expert_aux_loss is None and plain.expert_weights == {}
    assert plain.expert_counts is None


def test_column_counts_aggregate_each_physical_expert_bank():
    model = tiny("erfl")
    observed = [[] for _ in model.blocks]
    hooks = [
        block.mlp.register_forward_hook(
            lambda _module, _args, output, index=index: observed[index].append(
                output[3]
            )
        )
        for index, block in enumerate(model.blocks)
    ]
    try:
        out = model.forward_column(torch.randn(2, 5, 16), iterations=3)
    finally:
        for hook in hooks:
            hook.remove()
    expected = torch.stack([torch.stack(calls).sum(0) for calls in observed])
    assert out.expert_counts.shape == (12, 15)
    assert out.expert_counts.dtype == torch.int64
    torch.testing.assert_close(out.expert_counts, expected, rtol=0, atol=0)
    assert out.expert_counts.sum(dim=-1).tolist() == [30] * 4 + [90] * 4 + [30] * 4
    before = torch.stack([block.mlp.expert_bias.clone() for block in model.blocks])
    model.update_expert_bias(out.expert_counts)
    direction = (expected.sum(-1, keepdim=True) - 15 * expected).sign()
    after = torch.stack([block.mlp.expert_bias for block in model.blocks])
    torch.testing.assert_close(after, before + 0.001 * direction)
    model.eval()
    model.update_expert_bias(out.expert_counts)
    torch.testing.assert_close(
        torch.stack([block.mlp.expert_bias for block in model.blocks]),
        after,
        rtol=0,
        atol=0,
    )


def test_expert_bias_roundtrip_is_required_current_snapshot_state():
    model = tiny("erfl")
    for index, block in enumerate(model.blocks):
        with torch.no_grad():
            block.mlp.expert_bias.copy_(torch.linspace(-0.04, 0.03, 15) * (index + 1))
    state = {name: value.clone() for name, value in model.state_dict().items()}
    assert not any("expert_counts" in name for name in state)
    restored = tiny("erfl", seed=19)
    restored.load_state_dict(state, strict=True)
    x = torch.randn(1, 4, 16)
    original_output = model.forward_column(x, want_weights=True)
    restored_output = restored.forward_column(x, want_weights=True)
    torch.testing.assert_close(
        original_output.h_top, restored_output.h_top, rtol=0, atol=0
    )
    torch.testing.assert_close(
        original_output.expert_counts, restored_output.expert_counts, rtol=0, atol=0
    )
    for site, weights in original_output.expert_weights.items():
        torch.testing.assert_close(
            weights, restored_output.expert_weights[site], rtol=0, atol=0
        )
    missing_bias = {
        name: value for name, value in state.items() if not name.endswith("expert_bias")
    }
    with pytest.raises(RuntimeError, match="expert_bias"):
        restored.load_state_dict(missing_bias, strict=True)
    dense = tiny("")
    assert not any("expert_bias" in name for name in dense.state_dict())
    tiny("", seed=19).load_state_dict(dense.state_dict(), strict=True)


@pytest.mark.parametrize("passes", [1, 2, 3])
def test_expert_loss_averages_passes_without_polluting_ce(monkeypatch, passes):
    model = tiny("ef")
    out = model.forward_column(torch.randn(1, 3, 16))
    auxiliaries = [
        torch.tensor(float(value), requires_grad=True) for value in (2, 5, 11)[:passes]
    ]
    outs = [replace(out, expert_aux_loss=auxiliary) for auxiliary in auxiliaries]
    monkeypatch.setattr(
        "delta_feedback_experiment.model.sequence_ce",
        lambda *args: (torch.tensor(3.0), torch.tensor(7.0)),
    )
    loss_result = multipass_loss(
        model, torch.zeros(1, 4, dtype=torch.long), outs, z_coef=0.2
    )
    loss, per_pass = loss_result.total, loss_result.ntp
    weight = 1 if passes == 1 else 2
    torch.testing.assert_close(
        loss,
        torch.tensor(weight * (3 + 0.2 * 7)) + 1e-4 * torch.stack(auxiliaries).mean(),
    )
    assert [value.item() for value in per_pass] == [3.0] * passes
    for gradient in torch.autograd.grad(loss, auxiliaries):
        torch.testing.assert_close(gradient, torch.tensor(1e-4 / passes))


@pytest.mark.parametrize("trunk", ["", "a"])
def test_experts_preserve_dense_trunk_pairing(trunk):
    dense = dict(tiny(trunk + "rf").named_parameters())
    sparse = dict(tiny(trunk + "erf").named_parameters())
    shared = set(dense) & set(sparse)
    assert shared
    assert not any(".mlp." in name for name in shared)
    for name in shared:
        assert torch.equal(dense[name], sparse[name]), name


@pytest.mark.parametrize("condition", ["", "arf"])
def test_expert_initialization_preserves_ambient_cpu_rng(condition):
    torch.manual_seed(37)
    DeltaModel(condition_config(condition, **TINY))
    dense_rng = torch.random.get_rng_state()
    torch.manual_seed(37)
    DeltaModel(condition_config(condition + "e", **TINY))
    assert torch.equal(torch.random.get_rng_state(), dense_rng)


def test_expert_private_initialization_does_not_seed_cuda(monkeypatch):
    torch.manual_seed(37)
    calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", calls.append)
    DeltaModel(condition_config("e", **TINY))
    assert calls == []


def test_expert_parameters_pair_across_other_letters_and_loop_once():
    flat = tiny("erf", layers=12)
    looped = tiny("erfl")
    shared_only = dict(tiny("e", layers=12).named_parameters())
    looped_parameters = dict(looped.named_parameters())
    for name, parameter in flat.named_parameters():
        assert torch.equal(parameter, looped_parameters[name]), name
        if ".mlp." in name:
            assert torch.equal(parameter, shared_only[name]), name
    x = torch.randn(1, 4, 16)
    flat_output = flat.forward_column(x, want_weights=True)
    loop_output = looped.forward_column(x, iterations=1, want_weights=True)
    torch.testing.assert_close(flat_output.h_top, loop_output.h_top, rtol=0, atol=0)
    torch.testing.assert_close(
        flat_output.expert_aux_loss, loop_output.expert_aux_loss, rtol=0, atol=0
    )


def test_expert_checkpoint_recomputation_preserves_gradients():
    direct = tiny("erfl")
    checkpointed = tiny("erfl")
    checkpointed.grad_checkpoint = True
    x = torch.randn(1, 5, 16)
    direct_output = direct.forward_column(x)
    checkpoint_output = checkpointed.forward_column(x)
    for out in (direct_output, checkpoint_output):
        (out.h_top.square().mean() + 1e-4 * out.expert_aux_loss).backward()
    torch.testing.assert_close(direct_output.h_top, checkpoint_output.h_top)
    torch.testing.assert_close(
        direct_output.expert_aux_loss, checkpoint_output.expert_aux_loss
    )
    torch.testing.assert_close(
        direct_output.expert_counts, checkpoint_output.expert_counts, rtol=0, atol=0
    )
    assert checkpoint_output.expert_counts.sum() == 3 * 5 * 16
    for block in checkpointed.blocks:
        assert torch.count_nonzero(block.mlp.expert_bias) == 0
    for (name, p), (_, q) in zip(
        direct.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(
                p.grad, q.grad, msg=lambda message, name=name: f"{name}: {message}"
            )


def test_expert_optimizer_ownership():
    model = tiny("e")
    for name, parameter in model.named_parameters():
        if ".mlp.router." in name:
            assert not is_normuonh_parameter(name, parameter)
            assert is_width_scaled_parameter(name, parameter)
        elif ".mlp." in name:
            assert is_normuonh_parameter(name, parameter)
            assert not is_width_scaled_parameter(name, parameter)


@pytest.mark.parametrize(
    "scale,expected",
    [("screen", 455_637_288), ("bridge", 1_302_973_776), ("flagship", 3_388_167_456)],
)
def test_expert_scale_parameter_counts(scale, expected):
    from delta_feedback_experiment.train import SCALES

    overrides = {
        key: value
        for key, value in SCALES[scale].items()
        if key not in {"seq_len", "batch_rows", "micro_rows"}
    }
    with torch.device("meta"):
        flat = DeltaModel(condition_config("aerf", **overrides))
        looped = DeltaModel(condition_config("aerfl", **overrides))
    assert sum(p.numel() for p in flat.parameters()) == expected
    assert sum(p.numel() for p in looped.parameters()) == expected

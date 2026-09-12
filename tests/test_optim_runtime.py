"""NorMuonH bucket updates preserve state across sparse gradients and resume."""

import math
from copy import deepcopy

import pytest
import torch
from transformer_experiments.schedule import Schedule

from delta_feedback_experiment.model import DeltaModel, condition_config
from delta_feedback_experiment.optim import (
    NADAM_MOMENTUM_DECAY,
    NorMuonH,
    OptimizerPair,
    apply_schedule,
    build_optimizers,
    split_parameters,
)


def tiny_optimizer_model(**overrides):
    geometry = {
        "vocab_size": 31,
        "dim": 32,
        "layers": 4,
        "heads": 2,
        "kv_heads": 2,
        "head_dim": 16,
        "expert_intermediate": 8,
        "num_routed_experts": 7,
        "experts_per_token": 5,
        "pkda_heads": 2,
        "pkda_head_dim": 8,
        "max_seq_len": 8,
    }
    return DeltaModel(condition_config("f", **(geometry | overrides)))


def expected_stable_rates(cfg, normuonh, nadam):
    return {
        "normuonh": normuonh,
        "normuonh_expert_in": normuonh * math.sqrt(1536 / cfg.dim),
        "normuonh_expert_out": normuonh * math.sqrt(8 / (cfg.experts_per_token + 1)),
        "nadam": nadam,
        "nadam_width": nadam * 1536 / cfg.dim,
    }


def test_bucket_optimizer_preserves_absent_gradients_and_resume():
    """Changing active sets must neither update missing state nor break resume."""
    torch.manual_seed(31)
    parameters = [
        torch.nn.Parameter(torch.randn(*shape)) for shape in ((7, 5), (7, 5), (5, 7))
    ]
    optimizer = NorMuonH(parameters, lr=0.02)
    initial = [parameter.detach().clone() for parameter in parameters]
    parameters[0].grad = torch.randn_like(parameters[0])
    optimizer.step()
    assert set(optimizer.state) == {parameters[0]}
    for index in (1, 2):
        assert torch.equal(parameters[index], initial[index])

    restored_parameters = [
        torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters
    ]
    restored = NorMuonH(restored_parameters, lr=0.02)
    restored.load_state_dict(deepcopy(optimizer.state_dict()))
    untouched = deepcopy(optimizer.state[parameters[0]])
    saved_weight = parameters[0].detach().clone()

    for index, (parameter, other) in enumerate(zip(parameters, restored_parameters)):
        gradient = None if index == 0 else torch.randn_like(parameter)
        parameter.grad = gradient
        other.grad = None if gradient is None else gradient.clone()
    optimizer.step()
    restored.step()

    assert torch.equal(parameters[0], saved_weight)
    for name, value in untouched.items():
        assert torch.equal(optimizer.state[parameters[0]][name], value)
    assert set(optimizer.state_dict()) == {"state", "param_groups"}
    for parameter, other in zip(parameters, restored_parameters):
        assert torch.equal(parameter, other)
        assert set(optimizer.state[parameter]) == {"momentum", "row_moment", "radius"}
        for name, value in optimizer.state[parameter].items():
            assert torch.equal(value, restored.state[other][name])


def test_bounded_expert_buckets_match_independent_matrix_updates():
    torch.manual_seed(87)
    parameters = [torch.nn.Parameter(torch.randn(7, 5)) for _ in range(7)]
    references = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    optimizer = NorMuonH(parameters, lr=0.02, max_bucket_elements=70)
    separate = [NorMuonH([p], lr=0.02) for p in references]
    assert [len(batch) for batch in optimizer._batches(parameters)] == [2, 2, 2, 1]
    for _ in range(3):
        for parameter, reference in zip(parameters, references):
            gradient = torch.randn_like(parameter)
            parameter.grad = gradient
            reference.grad = gradient.clone()
        optimizer.step()
        for reference, single in zip(references, separate):
            single.step()
        for parameter, reference in zip(parameters, references):
            torch.testing.assert_close(parameter, reference)


@pytest.mark.parametrize(
    "overrides",
    [{}, {"dim": 64}, {"experts_per_token": 3}, {"expert_intermediate": 12}],
    ids=["tiny_bridge", "residual_width", "active_experts", "expert_width"],
)
def test_five_rate_groups_partition_trunk_mtp_and_frozen_parameters(overrides):
    model = tiny_optimizer_model(**overrides)
    frozen = {
        model.fuse_value.weight,
        model.blocks[0].mlp.experts[0].gate_up_proj.weight,
        model.mtp.block.mlp.shared.down_proj.weight,
        model.mtp.block.mlp.router.weight,
        model.final_norm.weight,
    }
    for parameter in frozen:
        parameter.requires_grad_(False)

    # Derive ownership from physical modules, independently of name predicates.
    ordinary = {model.fuse_value.weight, model.mtp.projection.weight}
    width = {model.fuse_gate.weight}
    width.update(gate.weight for gate in model.attention_gates)
    expert_in, expert_out = set(), set()
    for block in (*model.blocks, model.mtp.block):
        attention = block.attn
        ordinary.add(attention.o_proj.weight)
        if block.is_pkda:
            ordinary.update(
                projection.weight
                for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
            )
            width.add(attention.control_proj.weight)
        else:
            ordinary.add(attention.qkv_proj.weight)
        width.add(block.mlp.router.weight)
        for expert in (block.mlp.shared, *block.mlp.experts):
            expert_in.add(expert.gate_up_proj.weight)
            expert_out.add(expert.down_proj.weight)
    trainable = {
        parameter for parameter in model.parameters() if parameter.requires_grad
    }
    expected = {
        "normuonh": ordinary - frozen,
        "normuonh_expert_in": expert_in - frozen,
        "normuonh_expert_out": expert_out - frozen,
        "nadam": trainable - ordinary - expert_in - expert_out - width,
        "nadam_width": width - frozen,
    }
    partition = split_parameters(model)
    assert {name: set(parameters) for name, parameters in partition.items()} == expected
    flattened = [
        parameter for parameters in partition.values() for parameter in parameters
    ]
    assert len(flattened) == len(set(flattened)) == len(trainable)
    assert set(flattened) == trainable

    stable = expected_stable_rates(model.cfg, normuonh=0.021, nadam=0.00017)
    assert model.cfg.expert_in_lr_scale == pytest.approx(
        math.sqrt(1536 / model.cfg.dim)
    )
    assert model.cfg.expert_out_lr_scale == pytest.approx(
        math.sqrt(8 / (model.cfg.experts_per_token + 1))
    )
    optimizers = build_optimizers(model, lr_normuonh=0.021, lr_nadam=0.00017)
    assert isinstance(optimizers[0], NorMuonH)
    assert isinstance(optimizers[1], torch.optim.NAdam)
    groups = {
        group["rate_name"]: group
        for optimizer in optimizers
        for group in optimizer.param_groups
    }
    assert set(groups) == set(stable)
    assert {group["rate_name"] for group in optimizers[0].param_groups} == {
        "normuonh", "normuonh_expert_in", "normuonh_expert_out"
    }
    for name, group in groups.items():
        assert set(group["params"]) == expected[name]
        assert group["stable_lr"] == pytest.approx(stable[name])
        assert group["lr"] == pytest.approx(stable[name])


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires compiled CUDA optimizer"
            ),
        ),
    ],
)
def test_five_scheduled_rates_match_independent_updates_and_resume(device):
    torch.manual_seed(113)
    model = tiny_optimizer_model().to(device)
    selected = {
        "normuonh": [
            "blocks.0.attn.q_proj.weight", "mtp.projection.weight"
        ],
        "normuonh_expert_in": [
            "blocks.0.mlp.shared.gate_up_proj.weight",
            "blocks.0.mlp.experts.0.gate_up_proj.weight",
            "mtp.block.mlp.experts.6.gate_up_proj.weight",
        ],
        "normuonh_expert_out": [
            "blocks.0.mlp.shared.down_proj.weight",
            "blocks.0.mlp.experts.0.down_proj.weight",
            "mtp.block.mlp.experts.6.down_proj.weight",
        ],
        "nadam": ["embed_tokens.weight", "mtp.payload_norm.weight"],
        "nadam_width": [
            "blocks.0.mlp.router.weight", "mtp.block.mlp.router.weight"
        ],
    }
    owners = {
        name: rate_name for rate_name, names in selected.items() for name in names
    }
    parameters = dict(model.named_parameters())
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in owners)
    frozen_weights = {
        name: parameter.detach().clone()
        for name, parameter in parameters.items()
        if name not in owners
    }
    normuonh_rate, nadam_rate, betas = 0.021, 0.00017, (0.8, 0.91)
    stable = expected_stable_rates(model.cfg, normuonh_rate, nadam_rate)
    stack = build_optimizers(
        model, lr_normuonh=normuonh_rate, lr_nadam=nadam_rate, nadam_betas=betas
    )
    # The ordinary Q projection and expert gate/up matrices deliberately share
    # a shape. Packing may never combine them across their different rates.
    assert parameters[selected["normuonh"][0]].shape == parameters[
        selected["normuonh_expert_in"][0]
    ].shape
    references, separate = {}, {}
    for name, rate_name in owners.items():
        reference = torch.nn.Parameter(parameters[name].detach().cpu().clone())
        references[name] = reference
        if rate_name.startswith("normuonh"):
            separate[name] = NorMuonH([reference], lr=stable[rate_name])
        else:
            separate[name] = torch.optim.NAdam(
                [reference], lr=stable[rate_name], betas=betas,
                momentum_decay=NADAM_MOMENTUM_DECAY, eps=1e-8, foreach=False,
            )

    schedule = Schedule(heat=4, warmup=0.5, cooldown=0.5)
    multipliers = (0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1 - math.sqrt(0.5), 0.0)
    deferred = "blocks.0.mlp.experts.0.gate_up_proj.weight"
    restored_stack, restored_parameters = None, None
    for step, multiplier in enumerate(multipliers, 1):
        expected_rates = {name: rate * multiplier for name, rate in stable.items()}
        assert apply_schedule(stack, schedule, step) == pytest.approx(expected_rates)
        if restored_stack is not None:
            assert apply_schedule(restored_stack, schedule, step) == pytest.approx(
                expected_rates
            )
        previous = {
            name: parameters[name].detach().clone() for name in owners
        }
        missing_state = {}
        for index, (name, rate_name) in enumerate(owners.items()):
            parameter, reference = parameters[name], references[name]
            active = (index + step) % 3 != 0 and not (name == deferred and step <= 2)
            gradient = torch.randn_like(reference) if active else None
            parameter.grad = None if gradient is None else gradient.to(device)
            reference.grad = None if gradient is None else gradient.clone()
            independent = separate[name]
            independent.param_groups[0]["lr"] = expected_rates[rate_name]
            if restored_parameters is not None:
                restored_parameters[name].grad = (
                    None if gradient is None else gradient.to(device).clone()
                )
            if not active:
                optimizer = stack[0 if rate_name.startswith("normuonh") else 1]
                missing_state[name] = deepcopy(optimizer.state.get(parameter))
        for optimizer in stack:
            optimizer.step()
        for optimizer in separate.values():
            optimizer.step()
        if restored_stack is not None:
            for optimizer in restored_stack:
                optimizer.step()

        for name, rate_name in owners.items():
            parameter, reference = parameters[name], references[name]
            optimizer_index = 0 if rate_name.startswith("normuonh") else 1
            state = stack[optimizer_index].state.get(parameter)
            torch.testing.assert_close(
                parameter.cpu(), reference, rtol=5e-4, atol=5e-6, msg=name
            )
            expected_state = separate[name].state.get(reference)
            assert (state is None) == (expected_state is None)
            if state is not None:
                assert set(state) == set(expected_state)
                for key, value in state.items():
                    torch.testing.assert_close(
                        value.cpu(), expected_state[key], rtol=5e-4, atol=5e-6,
                        msg=f"{name}: {key}",
                    )
            if name in missing_state:
                torch.testing.assert_close(parameter, previous[name], rtol=0, atol=0)
                before = missing_state[name]
                assert (state is None) == (before is None)
                if state is not None:
                    for key, value in state.items():
                        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            if restored_stack is not None:
                other = restored_parameters[name]
                torch.testing.assert_close(parameter, other, rtol=0, atol=0)
                restored_state = restored_stack[optimizer_index].state.get(other)
                assert (state is None) == (restored_state is None)
                if state is not None:
                    for key, value in state.items():
                        torch.testing.assert_close(
                            value, restored_state[key], rtol=0, atol=0
                        )
        for name, original in frozen_weights.items():
            torch.testing.assert_close(parameters[name], original, rtol=0, atol=0)

        if step == 2:
            assert parameters[deferred] not in stack[0].state
            restored_model = deepcopy(model)
            restored_parameters = dict(restored_model.named_parameters())
            restored_stack = build_optimizers(
                restored_model, lr_normuonh=0.09, lr_nadam=0.005
            )
            OptimizerPair(restored_stack).load_state_dict(
                deepcopy(OptimizerPair(stack).state_dict())
            )
            restored_rates = {
                group["rate_name"]: group["stable_lr"]
                for optimizer in restored_stack
                for group in optimizer.param_groups
            }
            assert restored_rates == pytest.approx(stable)

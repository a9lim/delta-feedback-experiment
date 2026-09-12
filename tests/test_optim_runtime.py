"""NorMuonH bucket updates preserve state across sparse gradients and resume."""

import math
from copy import deepcopy

import pytest
import torch

from delta_feedback_experiment.model import DeltaModel, condition_config
from delta_feedback_experiment.optim import (
    NorMuonH,
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
        "normuonh_expert_in": normuonh * math.sqrt(8 / (cfg.experts_per_token + 1)),
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
        assert set(optimizer.state[parameter]) == {
            "momentum",
            "row_moment",
            "radius",
            "spectral_vector",
        }
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


def test_rate_groups_partition_trunk_mtp_and_frozen_parameters():
    model = tiny_optimizer_model()
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
    assert model.cfg.expert_lr_scale == pytest.approx(
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
        "normuonh",
        "normuonh_expert_in",
        "normuonh_expert_out",
    }
    for name, group in groups.items():
        assert set(group["params"]) == expected[name]
        assert group["stable_lr"] == pytest.approx(stable[name])
        assert group["lr"] == pytest.approx(stable[name])

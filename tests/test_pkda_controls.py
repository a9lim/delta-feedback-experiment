"""Compiled packed controls retain the unpadded projection's values and gradients."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch._inductor.utils import fresh_cache

from delta_feedback_experiment import INDUCTOR_MODE
from delta_feedback_experiment.pkda import PreconditionedKDA


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Inductor")
@pytest.mark.parametrize("training", [False, True])
def test_compiled_screen_controls_match_unpadded_reference(training, caplog):
    torch.manual_seed(59)
    attention = PreconditionedKDA(768, num_heads=10, head_dim=128).cuda()
    reference = copy.deepcopy(attention)
    attention.control_shadow = attention.control_proj.weight.detach().bfloat16()
    attention.decay_up_shadow = attention.decay_up.weight.detach().bfloat16()
    x = torch.randn(1, 4096, 768, device="cuda", dtype=torch.bfloat16)
    x.requires_grad_(training)
    reference_x = x.detach().clone().requires_grad_(training)

    def parameters(module):
        return (
            module.control_proj.weight,
            module.decay_up.weight,
            module.A_log_precond,
            module.dt_bias_precond,
        )

    # Compile each mode so a cached graph cannot hide a scheduling conflict.
    with fresh_cache():
        compiled = torch.compile(attention._controls, mode=INDUCTOR_MODE, fullgraph=True)
        with torch.set_grad_enabled(training), torch.autocast("cuda", torch.bfloat16):
            actual = compiled(x)
            decay, beta, precond_decay, precond_beta, gate = F.linear(
                reference_x, reference.control_proj.weight
            ).split(reference.control_splits, dim=-1)
            expected = (
                F.linear(decay, reference.decay_up.weight).reshape(1, 4096, 10, 128),
                beta.sigmoid(),
                -reference.A_log_precond.float().exp()
                * F.softplus(precond_decay.float() + reference.dt_bias_precond),
                precond_beta.sigmoid(),
                gate,
            )
            if training:
                cotangents = tuple(torch.randn_like(value) for value in expected)
                gradients = torch.autograd.grad(
                    actual, (x, *parameters(attention)), cotangents
                )
                reference_gradients = torch.autograd.grad(
                    expected, (reference_x, *parameters(reference)), cotangents
                )
                actual = (*actual, *gradients)
                expected = (*expected, *reference_gradients)
    for value, target in zip(actual, expected, strict=True):
        error = (value.float() - target.float()).norm() / target.float().norm()
        assert error < 0.01, error.item()
    assert not any("layout conflict" in record.message.lower() for record in caplog.records)

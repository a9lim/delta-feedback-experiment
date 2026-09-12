"""Spectral scaling controls rank/shape effects and survives degenerate directions."""

import math

import pytest
import torch

from delta_feedback_experiment.optim import (
    NorMuonH,
    _spectral_norm,
    _spectral_tangent_step,
)


def test_power_iteration_recovers_a_new_direction_orthogonal_to_its_cache():
    # The old singular vector lies in the nullspace of the new rank-one map.
    matrix = torch.zeros(2, 16, 24)
    matrix[0, 3, 7] = 2
    matrix[1, 8, 11] = -3
    cache = torch.zeros(2, 24)
    cache[:, 0] = 1
    before = torch.random.get_rng_state()
    sigma, vector = _spectral_norm(matrix, cache, 1e-8)
    torch.testing.assert_close(sigma, torch.tensor([2., 3.]))
    torch.testing.assert_close(vector.norm(dim=-1), torch.ones(2))
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.count_nonzero(cache[:, 1:]) == 0  # inputs were not mutated


@pytest.mark.parametrize("shape", [(48, 48), (64, 32), (32, 64)])
def test_power_estimate_against_exact_singular_values(shape):
    generator = torch.Generator().manual_seed(121)
    matrix = torch.randn(4, *shape, generator=generator)
    exact = torch.linalg.matrix_norm(matrix, ord=2)
    vector = torch.zeros(4, shape[1])
    previous = torch.zeros(4)
    for _ in range(20):
        estimate, vector = _spectral_norm(matrix, vector, 1e-8)
        assert torch.all(estimate <= exact * 1.00001)
        assert torch.all(estimate >= previous * 0.99999)
        # The diagnostic, rather than this finite sample, characterizes the
        # error on production shapes. No universal accuracy bound is claimed.
        assert torch.all(estimate >= exact * 0.8)
        previous = estimate
    # A cached estimate converges on a stationary matrix; three iterations
    # alone do not guarantee this accuracy on a newly rotated direction.
    assert torch.all(estimate >= exact * 0.99)


@pytest.mark.parametrize("width", [32, 64, 128])
def test_rank_one_optimizer_step_has_width_invariant_operator_scale(width):
    generator = torch.Generator().manual_seed(432)
    weight = torch.nn.Parameter(
        torch.randn(width, width, generator=generator) / math.sqrt(width)
    )
    start = weight.detach().clone()
    radius = weight.norm().detach()
    weight.grad = torch.outer(
        torch.randn(width, generator=generator),
        torch.randn(width, generator=generator),
    )
    lr = 0.006
    optimizer = NorMuonH([weight], lr=lr)
    optimizer.step()
    actual = torch.linalg.matrix_norm(weight - start, ord=2).item()
    assert actual / lr == pytest.approx(1, rel=0.02)
    torch.testing.assert_close(weight.norm(), radius)


@pytest.mark.parametrize("shape", [(32, 32), (64, 32), (32, 64)])
def test_tangent_step_uses_rectangular_rms_geometry(shape):
    rows, cols = shape
    weight = torch.zeros(1, rows, cols)
    weight[0, 0, 0] = math.sqrt(rows)
    direction = torch.zeros_like(weight)
    direction[0, 1, 1] = 1  # exactly tangent, with known spectral norm
    lr = 0.006
    radius = weight.norm(dim=(-2, -1))
    result, _ = _spectral_tangent_step(
        weight, direction, radius, torch.zeros(1, cols), torch.tensor(lr), 1e-8
    )
    actual = torch.linalg.matrix_norm(result - weight, ord=2).item()
    assert math.sqrt(cols / rows) * actual / lr == pytest.approx(1, rel=1e-4)
    torch.testing.assert_close(result.norm(dim=(-2, -1)), radius)


def test_zero_radial_and_zero_rate_steps_preserve_weights_exactly():
    generator = torch.Generator().manual_seed(91)
    weight = torch.randn(3, 16, 24, generator=generator)
    radius = weight.norm(dim=(-2, -1))
    direction = weight.clone()
    direction[0].zero_()
    direction[1].neg_()
    for update, lr in [(direction, 0.006), (torch.ones_like(weight), 0.)]:
        result, vector = _spectral_tangent_step(
            weight, update, radius, torch.zeros(3, 24), torch.tensor(lr), 1e-8
        )
        assert torch.equal(result, weight)
        assert torch.isfinite(vector).all()


def test_zero_gradient_does_not_destroy_a_subsequent_nonzero_update():
    parameter = torch.nn.Parameter(torch.eye(16))
    optimizer = NorMuonH([parameter])
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert torch.equal(parameter, torch.eye(16))
    assert optimizer.state[parameter]["spectral_vector"].count_nonzero() == 0
    parameter.grad[0, 1] = 1
    optimizer.step()
    assert parameter[0, 1] != 0
    assert torch.isfinite(parameter).all()

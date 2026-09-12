"""Small CUDA MoE regressions; full screen memory is qualified by delta probe."""

import pytest
import torch

from delta_feedback_experiment.moe_probe import (
    _bias_step_parity,
    _frozen_kernel_parity,
    _kernel_parity,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize(
    ("dim", "intermediate", "tokens", "skewed"),
    ((48, 140, 37, False), (48, 140, 67, True), (128, 256, 1, True)),
)
def test_sparse_expert_cuda_values_gradients_and_persistent_sinks(
    dim, intermediate, tokens, skewed
):
    _kernel_parity(dim, intermediate, tokens, skewed=skewed)


@pytest.mark.parametrize("partial", (False, True))
def test_frozen_sparse_experts_preserve_input_gradient(partial):
    _frozen_kernel_parity(partial=partial)


def test_biased_sigmoid_routing_and_sequence_auxiliary_match_cuda_reference():
    _kernel_parity(48, 140, 19, batch=3, biased=True)


def test_cuda_bias_updates_only_at_steps_and_restores_from_snapshot():
    _bias_step_parity()

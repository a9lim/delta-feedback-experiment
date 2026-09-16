"""The 384-wide MHDB groups agree with portable routing, including backward.

Run explicitly on the CUDA worker; the portable default suite skips these
cases when CUDA is unavailable. No model compilation or training data needed.
"""

from copy import deepcopy

import pytest
import torch

from delta_feedback_experiment.cuda_kernels import MAX_ROUTE_SOURCES
from delta_feedback_experiment.model import ModelConfig, Router


@pytest.fixture(scope="module")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("MHDB routing oracle requires CUDA")
    pytest.importorskip("triton")
    return torch.device("cuda")


@pytest.fixture
def routing_case(cuda_device):
    def make(num_heads, n_sources=3, dtype=torch.float32):
        dim = num_heads * 384
        generator = torch.Generator().manual_seed(191 + num_heads + n_sources)
        reference = Router(ModelConfig(dim=dim, kv_heads=num_heads))
        with torch.no_grad():
            reference.query.copy_(torch.randn(dim, generator=generator) * 0.05)
            reference.null.copy_(torch.randn(dim, generator=generator) * 0.5)
            reference.key_norm.weight.copy_(
                torch.rand(dim, generator=generator) + 0.5
            )
        actual = deepcopy(reference).to(cuda_device)
        # Seventeen tokens exercise the last partial eight-token backward
        # program; unequal feature energies expose a per-head RMS mistake.
        feature_scale = torch.linspace(0.3, 1.7, dim)
        sources = [
            (torch.randn(1, 17, dim, generator=generator) * feature_scale)
            .to(dtype).requires_grad_()
            for _ in range(n_sources)
        ]
        cuda_sources = [
            source.detach().to(cuda_device).requires_grad_() for source in sources
        ]
        upstream = torch.randn(1, 17, dim, generator=generator).to(dtype)
        expected, expected_weights = reference(sources, want_weights=True)
        expected.backward(upstream)
        return (
            reference, actual, sources, cuda_sources, upstream.to(cuda_device),
            expected, expected_weights,
        )

    return make


def assert_agrees(actual, expected, *, bf16=False, name="tensor"):
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    assert torch.isfinite(actual).all(), name
    if not bf16:
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5, msg=name)
        return
    # Portable BF16 rounds products and weights before reduction; Triton
    # folds them in FP32. Bound aggregate drift and spikes against RMS scale.
    difference = actual - expected
    relative_l2 = difference.norm() / expected.norm().clamp_min(1e-6)
    relative_peak = difference.abs().max() / expected.square().mean().sqrt().clamp_min(1e-6)
    assert relative_l2 < 0.02, (name, "relative_l2", relative_l2.item())
    assert relative_peak < 0.15, (name, "relative_peak", relative_peak.item())


@pytest.mark.parametrize("num_heads,n_sources", [
    (2, 3), (3, 3), (4, 3), (6, 3), (4, MAX_ROUTE_SOURCES - 1),
])
def test_384_wide_routing_matches_portable_forward_and_gradients(
    routing_case, num_heads, n_sources
):
    reference, actual, sources, cuda_sources, upstream, expected, expected_weights = (
        routing_case(num_heads, n_sources)
    )
    output, weights = actual(cuda_sources, want_weights=True)
    output.backward(upstream)
    assert_agrees(output, expected, name="output")
    assert_agrees(weights, expected_weights, name="weights")
    for (name, parameter), ref_parameter in zip(
        actual.named_parameters(), reference.parameters(), strict=True
    ):
        assert parameter.grad is not None
        assert ref_parameter.grad.abs().sum() > 0
        assert_agrees(parameter.grad, ref_parameter.grad, name=name)
    for index, (source, ref_source) in enumerate(zip(cuda_sources, sources, strict=True)):
        assert source.grad is not None
        assert_agrees(source.grad, ref_source.grad, name=f"source {index}")


def test_384_wide_bf16_banked_routing_replays_without_stale_gradients(routing_case):
    from delta_feedback_experiment.train import _capture_without_gc

    reference, actual, sources, cuda_sources, upstream, expected, expected_weights = (
        routing_case(3, dtype=torch.bfloat16)
    )
    # Bridge's three groups exercise a padded head as well as the third tile.
    # Two source sinks start nonzero, so an overwrite cannot pass as an add.
    initial = [torch.full_like(source, 0.03125) for source in cuda_sources[:2]]
    accumulators = [value.clone() for value in initial]
    leaves = [*actual.parameters(), *cuda_sources[2:]]
    for leaf in leaves:
        leaf.grad = torch.zeros_like(leaf)

    def run():
        for leaf in leaves:
            leaf.grad.zero_()
        for accumulator, value in zip(accumulators, initial, strict=True):
            accumulator.copy_(value)
        output, weights = actual(
            cuda_sources, want_weights=True, accumulators=tuple(accumulators)
        )
        output.backward(upstream)
        return output, weights

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with _capture_without_gc(graph, pool=None):
        output, weights = run()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert_agrees(output, expected, bf16=True, name="output")
        assert_agrees(weights, expected_weights, bf16=True, name="weights")
        for (name, parameter), ref_parameter in zip(
            actual.named_parameters(), reference.parameters(), strict=True
        ):
            assert_agrees(parameter.grad, ref_parameter.grad, bf16=True, name=name)
        for index, (source, ref_source) in enumerate(zip(cuda_sources, sources, strict=True)):
            if index < len(accumulators):
                assert source.grad is None
                expected_sink = initial[index].cpu() + ref_source.grad
                assert_agrees(accumulators[index], expected_sink, bf16=True, name=f"sink {index}")
            else:
                assert_agrees(source.grad, ref_source.grad, bf16=True, name=f"source {index}")

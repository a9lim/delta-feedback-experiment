"""The dense cuBLAS head matches FP32 autograd, with and without a sink.

Run on a CUDA worker; the portable default suite skips these cases.
"""

import pytest
import torch
import torch.nn.functional as F

from delta_feedback_experiment.head import dense_head, weighted_dense_head


@pytest.fixture(scope="module")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("the dense head requires CUDA")
    pytest.importorskip("triton")
    return torch.device("cuda")


def _case(device, rows=300, vocab=5003, dim=64):
    generator = torch.Generator().manual_seed(23)
    embeddings = torch.randn(rows, dim, generator=generator) * 0.5
    classifier = torch.randn(vocab, dim, generator=generator) * 0.5
    targets = torch.randint(0, vocab, (rows,), generator=generator)
    grad_nll = torch.rand(rows, generator=generator) / rows
    grad_lse = torch.rand(rows, generator=generator) * 1e-3
    return tuple(
        t.to(device)
        for t in (embeddings, classifier, targets, grad_nll, grad_lse)
    )


def _reference(embeddings, classifier, targets, grad_nll, grad_lse):
    e = embeddings.bfloat16().float().requires_grad_(True)
    c = classifier.bfloat16().float().requires_grad_(True)
    logits = e @ c.t()
    lse = torch.logsumexp(logits, -1)
    nll = F.cross_entropy(logits, targets, reduction="none")
    ((nll * grad_nll).sum() + (lse * grad_lse).sum()).backward()
    return nll.detach(), lse.detach(), e.grad, c.grad


@pytest.mark.parametrize("with_sink", [True, False])
def test_dense_head_matches_fp32_autograd(cuda_device, with_sink):
    embeddings, classifier, targets, grad_nll, grad_lse = _case(cuda_device)
    nll_ref, lse_ref, de_ref, dc_ref = _reference(
        embeddings, classifier, targets, grad_nll, grad_lse
    )
    rows = embeddings.bfloat16().requires_grad_(True)
    weights = classifier.bfloat16()
    sink = None
    if with_sink:
        sink = torch.full(weights.shape, 0.25, device=cuda_device)
    else:
        weights.requires_grad_(True)
    # A chunk that does not divide the rows exercises the ragged last chunk.
    nll, lse = dense_head(rows, weights, targets, sink=sink, chunk=128)
    ((nll * grad_nll).sum() + (lse * grad_lse).sum()).backward()

    torch.testing.assert_close(nll, nll_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(rows.grad.float(), de_ref, rtol=2e-2, atol=2e-5)
    if with_sink:
        assert weights.grad is None
        torch.testing.assert_close(sink - 0.25, dc_ref, rtol=2e-2, atol=2e-6)
    else:
        torch.testing.assert_close(weights.grad.float(), dc_ref, rtol=2e-2, atol=2e-6)


@pytest.mark.parametrize("with_sink", [True, False])
def test_weighted_head_applies_its_gradient_in_forward(cuda_device, with_sink):
    embeddings, classifier, targets, weight_nll, weight_z = _case(cuda_device)
    scale = 0.25

    e = embeddings.bfloat16().float().requires_grad_(True)
    c = classifier.bfloat16().float().requires_grad_(True)
    logits = e @ c.t()
    lse_ref = torch.logsumexp(logits, -1)
    nll_ref = F.cross_entropy(logits, targets, reduction="none")
    total_ref = (weight_nll * nll_ref).sum() + (weight_z * lse_ref.square()).sum()
    (total_ref * scale).backward()

    rows = embeddings.bfloat16().requires_grad_(True)
    weights = classifier.bfloat16()
    sink = None
    if with_sink:
        sink = torch.zeros(weights.shape, device=cuda_device)
    else:
        weights.requires_grad_(True)
    total, nll, lse = weighted_dense_head(
        rows, weights, targets, weight_nll, weight_z, grad_scale=scale, sink=sink, chunk=128
    )
    assert not nll.requires_grad and not lse.requires_grad
    (total * scale).backward()

    torch.testing.assert_close(total, total_ref.detach(), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(nll, nll_ref.detach(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(lse, lse_ref.detach(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(rows.grad.float(), e.grad, rtol=2e-2, atol=2e-6)
    classifier_grad = sink if with_sink else weights.grad.float()
    torch.testing.assert_close(classifier_grad, c.grad, rtol=2e-2, atol=2e-7)


def test_dense_head_without_gradients(cuda_device):
    embeddings, classifier, targets, _, _ = _case(cuda_device)
    nll_ref, lse_ref, _, _ = _reference(
        embeddings, classifier, targets, torch.zeros_like(targets, dtype=torch.float),
        torch.zeros_like(targets, dtype=torch.float),
    )
    with torch.no_grad():
        nll, lse = dense_head(embeddings.bfloat16(), classifier.bfloat16(), targets)
    torch.testing.assert_close(nll, nll_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-4, atol=1e-4)

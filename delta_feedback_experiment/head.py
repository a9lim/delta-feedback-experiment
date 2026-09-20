"""The tied readout's cross-entropy as cuBLAS logits in row chunks, on Hopper.

Each chunk of readout rows multiplies the BF16 classifier on the tensor cores
into FP32 logits, which a fused Triton pass reduces to each row's
log-partition and target logit. Nothing is filtered and nothing is
lock-added: every GEMM is a whole cuBLAS product, the logits and both
gradient accumulations stay FP32, and the materialized chunk is the price.
On Hopper that price is small against CCE's Triton kernels, which reach a
quarter of the tensor cores there; on Ada CCE ran at peak and its filter won.

Training uses ``weighted_dense_head``. The objective is linear in every
row's NLL and squared log-partition with weights the loss fixes before the
forward runs (column coefficients, row means, the MTP weight, the z-loss
coefficient, the replay's share of the step), so the forward already knows
the gradient each logit receives: one pass over a chunk's logits yields the
row statistics and the BF16 logit gradient, the two gradient GEMMs follow at
once (the classifier's straight into the tied embedding's FP32 sink), and
the backward hands back the saved readout-row gradient. That saves the
recomputed logits and a second pass over them. ``dense_head`` returns the
per-row values with an ordinary backward, for evaluation and any caller
that differentiates them some other way.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except (ImportError, OSError):  # pragma: no cover - portable installs
    triton = None
    tl = None

LOGIT_CHUNK_ROWS = 4096
"""Readout rows per materialized logit chunk: 824 MB of FP32 logits plus
412 MB of BF16 gradient at the 50,304-row vocabulary, transient workspace."""

LSE_BLOCK = 2048
"""One statistics pass per row: a program per row keeps HBM busiest."""

ROW_BLOCK = 4096
ROW_WARPS = 16
"""Two passes per row (statistics, then gradient): one persistent program per
SM, so the second pass reads the row back from L2 (0.89 vs 1.14 ms per
4,096-row chunk on the H100 PCIe)."""

GRAD_BLOCK = 4096


def dense_head_device(device: torch.device) -> bool:
    """Whether the CUDA head on ``device`` runs here rather than through CCE.

    Hopper: cuBLAS reaches 330-430 TFLOPS on the head's shapes where CCE's
    Triton kernels reach 120-160, so the unfiltered dense head is about twice
    as fast. On Ada CCE ran at the tensor cores' peak and its filter skipped
    ~40% of the backward, which a dense head cannot."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0)


if triton is not None:

    @triton.jit
    def _row_lse(base, V: tl.constexpr, BLOCK: tl.constexpr):
        columns = tl.arange(0, BLOCK)
        running_max = tl.full((BLOCK,), float("-inf"), tl.float32)
        running_sum = tl.zeros((BLOCK,), tl.float32)
        for start in range(0, V, BLOCK):
            offsets = start + columns
            x = tl.load(base + offsets, offsets < V, other=float("-inf"))
            new_max = tl.maximum(running_max, x)
            # Lanes that have seen only padding keep a zero sum.
            seen = new_max != float("-inf")
            running_sum = tl.where(
                seen,
                running_sum * tl.exp(running_max - new_max) + tl.exp(x - new_max),
                0.0,
            )
            running_max = new_max
        row_max = tl.max(running_max, axis=0)
        return row_max + tl.log(tl.sum(running_sum * tl.exp(running_max - row_max), axis=0))

    @triton.jit
    def _row_lse_kernel(
        logits, targets, lse, target_logit, V: tl.constexpr, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0).to(tl.int64)
        base = logits + row * V
        tl.store(lse + row, _row_lse(base, V, BLOCK))
        tl.store(target_logit + row, tl.load(base + tl.load(targets + row)))

    @triton.jit
    def _logit_grad_kernel(
        logits, lse, targets, grad_nll, grad_lse, out,
        V: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """``out = softmax * (g_nll + g_lse) - onehot(target) * g_nll`` in BF16."""
        row = tl.program_id(0).to(tl.int64)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < V
        x = tl.load(logits + row * V + offsets, mask, other=float("-inf"))
        g_nll = tl.load(grad_nll + row)
        gradient = tl.exp(x - tl.load(lse + row)) * (g_nll + tl.load(grad_lse + row))
        gradient = tl.where(offsets == tl.load(targets + row), gradient - g_nll, gradient)
        tl.store(out + row * V + offsets, gradient.to(tl.bfloat16), mask)

    @triton.jit
    def _weighted_row_kernel(
        logits, targets, weight_nll, weight_z, lse, nll, out,
        scale, ROWS, V: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Each row's statistics and the BF16 gradient of
        ``scale * (w_nll * nll + w_z * lse^2)`` over its logits.

        A persistent program per SM walks its rows one at a time, so the
        gradient pass re-reads a row the statistics pass has just brought
        into L2 instead of from HBM."""
        columns = tl.arange(0, BLOCK)
        for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
            row = row.to(tl.int64)
            base = logits + row * V
            row_lse = _row_lse(base, V, BLOCK)
            target = tl.load(targets + row)
            tl.store(lse + row, row_lse)
            tl.store(nll + row, row_lse - tl.load(base + target))
            g_nll = tl.load(weight_nll + row) * scale
            g_total = g_nll + 2.0 * tl.load(weight_z + row) * row_lse * scale
            for start in range(0, V, BLOCK):
                offsets = start + columns
                mask = offsets < V
                x = tl.load(base + offsets, mask, other=float("-inf"))
                gradient = tl.exp(x - row_lse) * g_total
                gradient = tl.where(offsets == target, gradient - g_nll, gradient)
                tl.store(out + row * V + offsets, gradient.to(tl.bfloat16), mask)


def _row_programs(device: torch.device, rows: int) -> int:
    return min(rows, torch.cuda.get_device_properties(device).multi_processor_count)


def _chunk_logits(rows: Tensor, classifier: Tensor) -> Tensor:
    return torch.mm(rows, classifier.t(), out_dtype=torch.float32)


def _sink_alias(sink: Tensor) -> Tensor:
    """The sink's memory under a transient tensor, so accumulating into it
    leaves the sink's own version counter alone (as ``dw_accum`` does)."""
    alias = torch.empty(0, dtype=sink.dtype, device=sink.device)
    alias.set_(sink.untyped_storage(), sink.storage_offset(), sink.shape, sink.stride())
    return alias


def _statistics(rows: Tensor, classifier: Tensor, targets: Tensor, chunk: int):
    count, vocab = rows.shape[0], classifier.shape[0]
    lse = torch.empty(count, device=rows.device, dtype=torch.float32)
    target_logit = torch.empty_like(lse)
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        logits = _chunk_logits(rows[start:stop], classifier)
        _row_lse_kernel[(stop - start,)](
            logits, targets[start:stop], lse[start:stop], target_logit[start:stop],
            vocab, LSE_BLOCK, num_warps=8,
        )
        del logits
    return lse - target_logit, lse


def _gradient_gemms(gradient, classifier, chunk_rows, grad_rows, target):
    torch.mm(gradient, classifier, out=grad_rows)
    torch.addmm(
        target, gradient.t(), chunk_rows, beta=1.0, alpha=1.0,
        out_dtype=torch.float32, out=target,
    )


class _DenseHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, classifier, targets, sink, chunk):
        nll, lse = _statistics(rows, classifier, targets, chunk)
        ctx.save_for_backward(rows, classifier, targets, lse)
        ctx.sink = sink
        ctx.chunk = chunk
        return nll, lse

    @staticmethod
    def backward(ctx, grad_nll, grad_lse):
        rows, classifier, targets, lse = ctx.saved_tensors
        count, vocab = rows.shape[0], classifier.shape[0]
        zeros = torch.zeros_like(lse)
        grad_nll = zeros if grad_nll is None else grad_nll.float().contiguous()
        grad_lse = zeros if grad_lse is None else grad_lse.float().contiguous()
        grad_rows = torch.empty_like(rows)
        sink = ctx.sink
        target = _sink_alias(sink) if sink is not None else torch.zeros(
            classifier.shape, device=classifier.device, dtype=torch.float32
        )
        for start in range(0, count, ctx.chunk):
            stop = min(start + ctx.chunk, count)
            logits = _chunk_logits(rows[start:stop], classifier)
            gradient = torch.empty(logits.shape, device=logits.device, dtype=torch.bfloat16)
            _logit_grad_kernel[(stop - start, triton.cdiv(vocab, GRAD_BLOCK))](
                logits, lse[start:stop], targets[start:stop], grad_nll[start:stop],
                grad_lse[start:stop], gradient, vocab, GRAD_BLOCK, num_warps=8,
            )
            del logits
            _gradient_gemms(gradient, classifier, rows[start:stop], grad_rows[start:stop], target)
            del gradient
        grad_classifier = None if sink is not None else target.to(classifier.dtype)
        return grad_rows, grad_classifier, None, None, None


def dense_head(
    rows: Tensor,
    classifier: Tensor,
    targets: Tensor,
    *,
    sink: Tensor | None = None,
    chunk: int = LOGIT_CHUNK_ROWS,
) -> tuple[Tensor, Tensor]:
    """Per-row NLL and log-partition of ``rows [N, D] @ classifier [V, D]^T``.

    ``rows`` and ``classifier`` are BF16 CUDA tensors, ``targets [N]`` int64.
    With ``sink`` the classifier's gradient accumulates into that FP32
    ``[V, D]`` buffer and the classifier receives no autograd gradient.
    """
    rows = rows.contiguous()
    targets = targets.reshape(-1).contiguous()
    if targets.shape[0] != rows.shape[0]:
        raise ValueError("the head needs one target per readout row")
    if not torch.is_grad_enabled() or not (rows.requires_grad or classifier.requires_grad):
        with torch.no_grad():
            return _statistics(rows, classifier, targets, chunk)
    return _DenseHead.apply(rows, classifier, targets, sink, chunk)


class _WeightedDenseHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, classifier, targets, weight_nll, weight_z, sink, grad_scale, chunk):
        count, vocab = rows.shape[0], classifier.shape[0]
        nll = torch.empty(count, device=rows.device, dtype=torch.float32)
        lse = torch.empty_like(nll)
        grad_rows = torch.empty_like(rows)
        target = _sink_alias(sink) if sink is not None else torch.zeros(
            classifier.shape, device=classifier.device, dtype=torch.float32
        )
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            logits = _chunk_logits(rows[start:stop], classifier)
            gradient = torch.empty(logits.shape, device=logits.device, dtype=torch.bfloat16)
            _weighted_row_kernel[(_row_programs(rows.device, stop - start),)](
                logits, targets[start:stop], weight_nll[start:stop], weight_z[start:stop],
                lse[start:stop], nll[start:stop], gradient, grad_scale,
                stop - start, vocab, ROW_BLOCK, num_warps=ROW_WARPS,
            )
            del logits
            _gradient_gemms(gradient, classifier, rows[start:stop], grad_rows[start:stop], target)
            del gradient
        ctx.save_for_backward(grad_rows)
        ctx.grad_classifier = None if sink is not None else target.to(classifier.dtype)
        ctx.grad_scale = grad_scale
        ctx.mark_non_differentiable(nll, lse)
        loss = (weight_nll * nll).sum() + (weight_z * lse.square()).sum()
        return loss, nll, lse

    @staticmethod
    def backward(ctx, grad_loss, _grad_nll, _grad_lse):
        (grad_rows,) = ctx.saved_tensors
        # The forward already applied this gradient; any other would need the
        # classifier's accumulation redone.
        torch._assert_async(grad_loss == ctx.grad_scale)
        return grad_rows, ctx.grad_classifier, None, None, None, None, None, None


def weighted_dense_head(
    rows: Tensor,
    classifier: Tensor,
    targets: Tensor,
    weight_nll: Tensor,
    weight_z: Tensor,
    *,
    grad_scale: float,
    sink: Tensor | None = None,
    chunk: int = LOGIT_CHUNK_ROWS,
) -> tuple[Tensor, Tensor, Tensor]:
    """``(sum(w_nll * nll + w_z * lse^2), nll, lse)`` over the rows.

    Both gradients are computed in the forward for an incoming gradient of
    exactly ``grad_scale`` on the returned sum, which the backward checks on
    the device: the readout rows' gradient is saved and the classifier's
    lands in ``sink`` (or, without one, becomes its autograd gradient).
    ``nll`` and ``lse`` are detached per-row statistics. The weights are
    FP32 ``[N]`` tensors.
    """
    rows = rows.contiguous()
    targets = targets.reshape(-1).contiguous()
    if not targets.shape[0] == rows.shape[0] == weight_nll.shape[0] == weight_z.shape[0]:
        raise ValueError("the head needs one target and one weight pair per readout row")
    return _WeightedDenseHead.apply(
        rows, classifier, targets, weight_nll.float().contiguous(),
        weight_z.float().contiguous(), sink, float(grad_scale), chunk,
    )

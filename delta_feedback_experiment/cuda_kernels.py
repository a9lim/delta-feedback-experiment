"""Bespoke CUDA kernels for the MHDB x FBT hot path.

The portable semantic implementations remain in :mod:`model`.  This module is
optional at import time and exposes only kernels whose fixed screen geometry
benefits materially from owning the reduction and memory traffic directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

try:  # Triton is part of Jobe's pinned torch stack, not the Mac test runtime.
    import triton
    import triton.language as tl
except (ImportError, OSError):  # pragma: no cover - exercised on Jobe
    triton = None
    tl = None


MAX_ROUTE_SOURCES = 27


if triton is not None:

    @triton.jit
    def _route_pointer(
        index: tl.constexpr,
        s0,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        s8,
        s9,
        s10,
        s11,
        s12,
        s13,
        s14,
        s15,
        s16,
        s17,
        s18,
        s19,
        s20,
        s21,
        s22,
        s23,
        s24,
        s25,
        s26,
    ):
        """Select one pointer without materializing a source-value bank."""
        if index == 0:
            return s0
        if index == 1:
            return s1
        if index == 2:
            return s2
        if index == 3:
            return s3
        if index == 4:
            return s4
        if index == 5:
            return s5
        if index == 6:
            return s6
        if index == 7:
            return s7
        if index == 8:
            return s8
        if index == 9:
            return s9
        if index == 10:
            return s10
        if index == 11:
            return s11
        if index == 12:
            return s12
        if index == 13:
            return s13
        if index == 14:
            return s14
        if index == 15:
            return s15
        if index == 16:
            return s16
        if index == 17:
            return s17
        if index == 18:
            return s18
        if index == 19:
            return s19
        if index == 20:
            return s20
        if index == 21:
            return s21
        if index == 22:
            return s22
        if index == 23:
            return s23
        if index == 24:
            return s24
        if index == 25:
            return s25
        return s26

    @triton.jit
    def _route_forward_kernel(
        s0,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        s8,
        s9,
        s10,
        s11,
        s12,
        s13,
        s14,
        s15,
        s16,
        s17,
        s18,
        s19,
        s20,
        s21,
        s22,
        s23,
        s24,
        s25,
        s26,
        projected,
        present,
        logits,
        inv_rms,
        routed,
        bt: tl.constexpr,
        dim: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        eps: tl.constexpr,
        n_sources: tl.constexpr,
        has_present: tl.constexpr,
        block_h: tl.constexpr,
        block_k: tl.constexpr,
    ):
        """Full-width RMS keys, per-group source softmax, and the value mix in
        one pass over the bank: every source tile is read exactly once per
        token, with the softmax folded in online.  Source 0 is the site's
        width-``dim`` null and is read at a fixed address."""
        token = tl.program_id(0)
        h_offsets = tl.arange(0, block_h)
        k_offsets = tl.arange(0, block_k)
        head_mask = h_offsets < num_heads
        mask = head_mask[:, None] & (k_offsets[None, :] < head_dim)
        offsets = h_offsets[:, None] * head_dim + k_offsets[None, :]
        p = tl.load(projected + offsets, mask=mask, other=0.0).to(tl.float32)
        running_max = tl.full((block_h,), -float("inf"), tl.float32)
        running_sum = tl.zeros((block_h,), tl.float32)
        mixed = tl.zeros((block_h, block_k), tl.float32)
        for index in tl.static_range(n_sources):
            source = _route_pointer(
                index,
                s0,
                s1,
                s2,
                s3,
                s4,
                s5,
                s6,
                s7,
                s8,
                s9,
                s10,
                s11,
                s12,
                s13,
                s14,
                s15,
                s16,
                s17,
                s18,
                s19,
                s20,
                s21,
                s22,
                s23,
                s24,
                s25,
                s26,
            )
            base = 0 if index == 0 else token * dim
            value = tl.load(source + base + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            inverse = tl.rsqrt(
                tl.sum(tl.sum(value * value, axis=1), axis=0) / dim + eps
            )
            score = tl.sum(value * p, axis=1) * inverse
            if has_present:
                exists = tl.load(present + index * bt + token)
                score = tl.where(exists, score, -float("inf"))
            tl.store(
                logits + index * bt * num_heads + token * num_heads + h_offsets,
                score,
                mask=head_mask,
            )
            tl.store(inv_rms + index * bt + token, inverse)
            new_max = tl.maximum(running_max, score)
            rescale = tl.where(
                new_max == -float("inf"), 1.0, tl.exp(running_max - new_max)
            )
            term = tl.where(score == -float("inf"), 0.0, tl.exp(score - new_max))
            running_sum = running_sum * rescale + term
            mixed = mixed * rescale[:, None] + term[:, None] * value
            running_max = new_max
        mixed = mixed / running_sum[:, None]
        tl.store(
            routed + token * dim + offsets,
            mixed.to(routed.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _route_softmax_kernel(
        logits,
        weights,
        bt: tl.constexpr,
        num_heads: tl.constexpr,
        n_sources: tl.constexpr,
        block_n: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        offsets = tl.arange(0, block_n)
        mask = offsets < n_sources
        addresses = offsets * bt * num_heads + token * num_heads + head
        values = tl.load(logits + addresses, mask=mask, other=-float("inf"))
        values = values - tl.max(values, axis=0)
        numerators = tl.exp(values)
        result = numerators / tl.sum(numerators, axis=0)
        tl.store(weights + addresses, result, mask=mask)

    @triton.jit
    def _route_backward_kernel(
        s0,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        s8,
        s9,
        s10,
        s11,
        s12,
        s13,
        s14,
        s15,
        s16,
        s17,
        s18,
        s19,
        s20,
        s21,
        s22,
        s23,
        s24,
        s25,
        s26,
        g0,
        g1,
        g2,
        g3,
        g4,
        g5,
        g6,
        g7,
        g8,
        g9,
        g10,
        g11,
        g12,
        g13,
        g14,
        g15,
        g16,
        g17,
        g18,
        g19,
        g20,
        g21,
        g22,
        g23,
        g24,
        g25,
        g26,
        projected,
        grad_routed,
        weights,
        inv_rms,
        partials,
        bt: tl.constexpr,
        dim: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        n_sources: tl.constexpr,
        tokens_per_program: tl.constexpr,
        block_h: tl.constexpr,
        block_k: tl.constexpr,
    ):
        """Analytic MHDB backward over ``tokens_per_program`` tokens.

        Per-token source gradients are stored directly; the query and null
        gradients, which are sums over every token, stay in FP32 registers and
        leave as one ``[dim]`` partial per program.
        """
        pid = tl.program_id(0)
        h_offsets = tl.arange(0, block_h)
        k_offsets = tl.arange(0, block_k)
        head_mask = h_offsets < num_heads
        mask = head_mask[:, None] & (k_offsets[None, :] < head_dim)
        offsets = h_offsets[:, None] * head_dim + k_offsets[None, :]
        p = tl.load(projected + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_p = tl.zeros((block_h, block_k), tl.float32)
        grad_null = tl.zeros((block_h, block_k), tl.float32)
        for step in range(tokens_per_program):
            token = pid * tokens_per_program + step
            valid = token < bt
            token_mask = mask & valid
            head_valid = head_mask & valid
            upstream = tl.load(
                grad_routed + token * dim + offsets, mask=token_mask, other=0.0
            ).to(tl.float32)
            centered = tl.zeros((block_h,), tl.float32)
            for index in tl.static_range(n_sources):
                source = _route_pointer(
                    index,
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    s6,
                    s7,
                    s8,
                    s9,
                    s10,
                    s11,
                    s12,
                    s13,
                    s14,
                    s15,
                    s16,
                    s17,
                    s18,
                    s19,
                    s20,
                    s21,
                    s22,
                    s23,
                    s24,
                    s25,
                    s26,
                )
                base = 0 if index == 0 else token * dim
                value = tl.load(source + base + offsets, mask=token_mask, other=0.0).to(
                    tl.float32
                )
                weight = tl.load(
                    weights + index * bt * num_heads + token * num_heads + h_offsets,
                    mask=head_valid,
                    other=0.0,
                )
                centered += weight * tl.sum(upstream * value, axis=1)

            for index in tl.static_range(n_sources):
                source = _route_pointer(
                    index,
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    s6,
                    s7,
                    s8,
                    s9,
                    s10,
                    s11,
                    s12,
                    s13,
                    s14,
                    s15,
                    s16,
                    s17,
                    s18,
                    s19,
                    s20,
                    s21,
                    s22,
                    s23,
                    s24,
                    s25,
                    s26,
                )
                base = 0 if index == 0 else token * dim
                value = tl.load(source + base + offsets, mask=token_mask, other=0.0).to(
                    tl.float32
                )
                weight = tl.load(
                    weights + index * bt * num_heads + token * num_heads + h_offsets,
                    mask=head_valid,
                    other=0.0,
                )
                inverse = tl.load(inv_rms + index * bt + tl.minimum(token, bt - 1))
                route_beta = weight * (tl.sum(upstream * value, axis=1) - centered)
                dkp = route_beta[:, None] * p
                # RMS statistics are shared across the full hidden width, so the
                # norm-backward correction couples all routing heads.
                norm_dot = tl.sum(tl.sum(dkp * value, axis=1), axis=0)
                source_grad = (
                    weight[:, None] * upstream
                    + inverse * dkp
                    - (inverse * inverse * inverse / dim) * norm_dot * value
                )
                if index == 0:
                    grad_null += source_grad
                else:
                    grad_source = _route_pointer(
                        index - 1,
                        g0,
                        g1,
                        g2,
                        g3,
                        g4,
                        g5,
                        g6,
                        g7,
                        g8,
                        g9,
                        g10,
                        g11,
                        g12,
                        g13,
                        g14,
                        g15,
                        g16,
                        g17,
                        g18,
                        g19,
                        g20,
                        g21,
                        g22,
                        g23,
                        g24,
                        g25,
                        g26,
                    )
                    tl.store(
                        grad_source + token * dim + offsets,
                        source_grad.to(grad_source.dtype.element_ty),
                        mask=token_mask,
                    )
                grad_p += route_beta[:, None] * value * inverse
        tl.store(partials + pid * dim + offsets, grad_p, mask=mask)
        tl.store(
            partials + (tl.num_programs(0) + pid) * dim + offsets, grad_null, mask=mask
        )


ROUTE_TOKENS_PER_PROGRAM = 4
"""Tokens folded into one backward program; sets the query/null partial count."""


def _padded_sources(sources: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    if not 2 <= len(sources) <= MAX_ROUTE_SOURCES:
        raise ValueError(f"bespoke router needs 2..{MAX_ROUTE_SOURCES} sources")
    return sources + (sources[-1],) * (MAX_ROUTE_SOURCES - len(sources))


def _route_launch(num_heads: int, head_dim: int) -> tuple[int, int, int]:
    block_h = triton.next_power_of_2(num_heads)
    block_k = triton.next_power_of_2(head_dim)
    tile = block_h * block_k
    if tile > 8192:
        raise RuntimeError(
            f"MHDB routing tile {block_h}x{block_k} is too large "
            f"(H={num_heads}, D/H={head_dim})"
        )
    num_warps = 8 if tile >= 4096 else (4 if tile >= 1024 else 2)
    return block_h, block_k, num_warps


def _check_route_dims(dim: int, num_heads: int) -> int:
    if num_heads < 2 or dim % num_heads:
        raise RuntimeError(
            f"MHDB requires at least two heads dividing hidden size; "
            f"got D={dim}, H={num_heads}"
        )
    return dim // num_heads


def _route_forward_impl(
    projected: Tensor,
    present: Tensor,
    null: Tensor,
    sources: list[Tensor],
    eps: float,
    num_heads: int,
) -> tuple[Tensor, Tensor, Tensor]:
    bank = (null, *sources)
    n_sources = len(bank)
    batch, length, dim = sources[0].shape
    bt = batch * length
    head_dim = _check_route_dims(dim, num_heads)
    block_h, block_k, num_warps = _route_launch(num_heads, head_dim)
    padded = _padded_sources(bank)
    logits = torch.empty(
        (n_sources, bt, num_heads), device=projected.device, dtype=torch.float32
    )
    inv_rms = torch.empty((n_sources, bt), device=projected.device, dtype=torch.float32)
    weights = torch.empty_like(logits)
    routed = torch.empty_like(sources[0], memory_format=torch.contiguous_format)
    has_present = present.numel() > 0
    present_arg = present if has_present else sources[0]
    _route_forward_kernel[(bt,)](
        *padded,
        projected,
        present_arg,
        logits,
        inv_rms,
        routed,
        bt=bt,
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        eps=eps,
        n_sources=n_sources,
        has_present=has_present,
        block_h=block_h,
        block_k=block_k,
        num_warps=num_warps,
    )
    _route_softmax_kernel[(bt, num_heads)](
        logits,
        weights,
        bt=bt,
        num_heads=num_heads,
        n_sources=n_sources,
        block_n=triton.next_power_of_2(n_sources),
        num_warps=1,
    )
    return (
        routed,
        weights.view(n_sources, batch, length, num_heads),
        inv_rms.view(n_sources, batch, length),
    )


def _route_backward_impl(
    projected: Tensor,
    grad_routed: Tensor,
    weights: Tensor,
    inv_rms: Tensor,
    null: Tensor,
    sources: list[Tensor],
    num_heads: int,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    grad_routed = grad_routed.contiguous()
    bank = (null, *sources)
    n_sources = len(bank)
    batch, length, dim = sources[0].shape
    bt = batch * length
    head_dim = _check_route_dims(dim, num_heads)
    block_h, block_k, num_warps = _route_launch(num_heads, head_dim)
    flat_weights = weights.view(n_sources, bt, num_heads)
    padded = _padded_sources(bank)
    source_grads = [
        torch.empty(source.shape, device=source.device, dtype=source.dtype)
        for source in sources
    ]
    padded_grads = tuple(source_grads) + (source_grads[-1],) * (
        MAX_ROUTE_SOURCES - len(source_grads)
    )
    programs = triton.cdiv(bt, ROUTE_TOKENS_PER_PROGRAM)
    partials = torch.empty(
        (2, programs, dim), device=projected.device, dtype=torch.float32
    )
    _route_backward_kernel[(programs,)](
        *padded,
        *padded_grads,
        projected,
        grad_routed,
        flat_weights,
        inv_rms,
        partials,
        bt=bt,
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        n_sources=n_sources,
        tokens_per_program=ROUTE_TOKENS_PER_PROGRAM,
        block_h=block_h,
        block_k=block_k,
        num_warps=num_warps,
    )
    summed = partials.sum(dim=1)
    return summed[0].to(projected.dtype), summed[1].to(null.dtype), source_grads


if triton is not None:

    @triton.jit
    def _pack_control_gradients_kernel(
        grad0,
        grad1,
        grad2,
        grad3,
        grad4,
        packed,
        n_elements,
        size0: tl.constexpr,
        size1: tl.constexpr,
        size2: tl.constexpr,
        size3: tl.constexpr,
        size4: tl.constexpr,
        total: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < n_elements
        column = offsets % total
        row = offsets // total

        edge1 = size0
        edge2 = edge1 + size1
        edge3 = edge2 + size2
        edge4 = edge3 + size3

        value0 = tl.load(
            grad0 + row * size0 + column,
            mask=valid & (column < edge1),
            other=0.0,
        )
        value1 = tl.load(
            grad1 + row * size1 + column - edge1,
            mask=valid & (column >= edge1) & (column < edge2),
            other=0.0,
        )
        value2 = tl.load(
            grad2 + row * size2 + column - edge2,
            mask=valid & (column >= edge2) & (column < edge3),
            other=0.0,
        )
        value3 = tl.load(
            grad3 + row * size3 + column - edge3,
            mask=valid & (column >= edge3) & (column < edge4),
            other=0.0,
        )
        value4 = tl.load(
            grad4 + row * size4 + column - edge4,
            mask=valid & (column >= edge4),
            other=0.0,
        )
        value = tl.where(
            column < edge1,
            value0,
            tl.where(
                column < edge2,
                value1,
                tl.where(
                    column < edge3, value2, tl.where(column < edge4, value3, value4)
                ),
            ),
        )
        tl.store(packed + offsets, value, mask=valid)


def _pack_control_gradients_impl(
    grad0: Tensor,
    grad1: Tensor,
    grad2: Tensor,
    grad3: Tensor,
    grad4: Tensor,
) -> Tensor:
    gradients = (grad0, grad1, grad2, grad3, grad4)
    prefix = grad0.shape[:-1]
    if any(gradient.shape[:-1] != prefix for gradient in gradients[1:]):
        raise ValueError("control gradients must share their leading shape")
    if any(not gradient.is_contiguous() for gradient in gradients):
        raise ValueError("control gradients must be contiguous")
    total = sum(gradient.shape[-1] for gradient in gradients)
    packed = torch.empty((*prefix, total), device=grad0.device, dtype=grad0.dtype)
    sizes = tuple(gradient.shape[-1] for gradient in gradients)
    _pack_control_gradients_kernel[(triton.cdiv(packed.numel(), 256),)](
        *gradients,
        packed,
        packed.numel(),
        *sizes,
        total,
        BLOCK=256,
        num_warps=4,
    )
    return packed


# -- in-place weight-gradient accumulation -------------------------------------


@torch.library.custom_op(
    "delta_feedback::dw_accum", mutates_args=(), device_types="cpu"
)
def dw_accum(grad_output: Tensor, activations: Tensor, sink: Tensor) -> Tensor:
    """sink += grad_output^T @ activations, accumulated in FP32 in place.

    ``grad_output`` is ``[tokens, rows]`` and ``activations`` ``[tokens, cols]``;
    ``sink`` is the persistent FP32 ``[rows, cols]`` weight gradient.  The CPU
    kernel is the literal reference; CUDA hands cuBLAS the BF16 operands with
    the FP32 sink as both the ``beta=1`` addend and the output, so a weight
    gradient is never materialized separately from its accumulator.  Measured
    on the 4090 against a Triton tensor-core kernel with the same
    read-modify-write epilogue, cuBLAS was 18% faster over the trunk shapes
    with equal or lower error, and it captures into CUDA graphs.

    The op deliberately declares no mutation.  Inside a compiled block the sink
    is a saved tensor of the compiled autograd node, and a declared mutation
    would bump its version between the backward passes that share it; nothing
    in any graph reads a sink, so the accumulation is invisible to autograd
    by design.  The returned zero scalar is a fence: callers fold it into the
    input gradient so the accumulation can never be dead-code-eliminated.
    """
    sink.addmm_(grad_output.float().mT, activations.float())
    return sink.new_zeros(())


@dw_accum.register_fake
def _dw_accum_fake(grad_output: Tensor, activations: Tensor, sink: Tensor) -> Tensor:
    return sink.new_zeros(())


@dw_accum.register_kernel("cuda")
def _dw_accum_cuda(grad_output: Tensor, activations: Tensor, sink: Tensor) -> Tensor:
    if sink.dtype != torch.float32:
        raise TypeError("dw_accum sinks are FP32 gradient buffers")
    # Accumulate through a transient alias of the sink's memory: an in-place
    # ``out=`` on the sink itself would bump the version counter that the
    # compiled blocks' saved-tensor checks read between the passes sharing it.
    target = torch.empty(0, dtype=sink.dtype, device=sink.device)
    target.set_(
        sink.untyped_storage(), sink.storage_offset(), sink.shape, sink.stride()
    )
    torch.addmm(
        target,
        grad_output.mT,
        activations,
        beta=1.0,
        alpha=1.0,
        out_dtype=torch.float32,
        out=target,
    )
    return sink.new_zeros(())


class _SinkLinear(torch.autograd.Function):
    """``F.linear`` over concatenated weights whose weight gradients accumulate
    straight into persistent FP32 buffers.

    Autograd's default path materializes each microbatch's weight gradient,
    widens it, and adds it into ``.grad`` as separate kernels.  Here the
    backward's tensor-core reduction ends in a read-modify-write of the sink,
    so the only weight-gradient traffic is the accumulator itself, and the
    parameters receive no autograd gradient at all.  Row segments keep
    separate parameters (Q/K/V, or QKV plus the attention gate) behind one GEMM.

    ``operand`` is the activation-dtype copy of the concatenated weights that
    the GEMM actually reads: the trainer's persistent shadow when one is bound,
    otherwise a fresh cast.  The FP32 parameters stay inputs so the output's
    autograd requirement follows them exactly as it would through ``F.linear``.
    """

    @staticmethod
    def forward(
        ctx, x: Tensor, splits: tuple[int, ...], operand: Tensor, *tensors: Tensor
    ) -> Tensor:
        count = len(splits)
        ctx.save_for_backward(x, operand)
        # Sinks are mutated by every backward that reaches them, so they are
        # held as plain attributes rather than version-checked saved tensors.
        ctx.sinks = tensors[count:]
        ctx.splits = splits
        return F.linear(x, operand)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        x, weight = ctx.saved_tensors
        sinks = ctx.sinks
        grad_x = gradient @ weight
        flat_gradient = gradient.reshape(-1, gradient.shape[-1])
        flat_x = x.reshape(-1, x.shape[-1])
        start = 0
        fence = None
        for size, sink in zip(ctx.splits, sinks, strict=True):
            token = dw_accum(flat_gradient[:, start : start + size], flat_x, sink)
            fence = token if fence is None else fence + token
            start += size
        # The fence is zero; the dependency keeps every accumulation alive.
        grad_x = grad_x + fence.to(grad_x.dtype)
        return (grad_x, None, None) + (None,) * (2 * len(ctx.splits))


def sink_linear(
    x: Tensor,
    weights: tuple[Tensor, ...],
    sinks: tuple[Tensor | None, ...],
    shadow: Tensor | None = None,
) -> Tensor:
    """Linear over concatenated weights; bound sinks accumulate dW in place.

    ``shadow`` is the trainer-owned activation-dtype copy of the concatenated
    weights, refreshed once per optimizer step, so no replay re-casts or
    re-concatenates FP32 parameters.  Without one (portable paths, analysis,
    cached decoding) the operand is cast per call.
    """
    if shadow is not None and shadow.dtype == x.dtype:
        operand = shadow
    else:
        operand = (weights[0] if len(weights) == 1 else torch.cat(weights, dim=0)).to(
            x.dtype
        )
    if not torch.is_grad_enabled() or any(sink is None for sink in sinks):
        return F.linear(x, operand)
    return _SinkLinear.apply(
        x, tuple(w.shape[0] for w in weights), operand, *weights, *sinks
    )


if triton is not None:

    @torch.library.custom_op(
        "delta_feedback::pack_control_gradients",
        mutates_args=(),
        device_types="cuda",
    )
    def _pack_control_gradients_op(
        grad0: Tensor,
        grad1: Tensor,
        grad2: Tensor,
        grad3: Tensor,
        grad4: Tensor,
    ) -> Tensor:
        return _pack_control_gradients_impl(grad0, grad1, grad2, grad3, grad4)

    @_pack_control_gradients_op.register_fake
    def _pack_control_gradients_fake(
        grad0: Tensor,
        grad1: Tensor,
        grad2: Tensor,
        grad3: Tensor,
        grad4: Tensor,
    ) -> Tensor:
        total = sum(
            gradient.shape[-1] for gradient in (grad0, grad1, grad2, grad3, grad4)
        )
        return torch.empty(
            (*grad0.shape[:-1], total), device=grad0.device, dtype=grad0.dtype
        )
else:  # pragma: no cover - the Mac never enters the CUDA packer.
    _pack_control_gradients_op = None


def pack_control_gradients(gradients: tuple[Tensor, ...]) -> Tensor:
    """Assemble the five control gradients without split backward's cat."""
    if len(gradients) != 5:
        raise ValueError("PKDA has exactly five packed control gradients")
    if triton is None or not gradients[0].is_cuda:
        return torch.cat(gradients, dim=-1)
    return _pack_control_gradients_op(*gradients)


if triton is not None:

    @torch.library.custom_op(
        "delta_feedback::route_forward", mutates_args=(), device_types="cuda"
    )
    def _route_forward_op(
        projected: Tensor,
        present: Tensor,
        null: Tensor,
        sources: list[Tensor],
        eps: float,
        num_heads: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return _route_forward_impl(projected, present, null, sources, eps, num_heads)

    @_route_forward_op.register_fake
    def _route_forward_fake(
        projected: Tensor,
        present: Tensor,
        null: Tensor,
        sources: list[Tensor],
        eps: float,
        num_heads: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del present, null, eps
        batch, length, _ = sources[0].shape
        n_sources = len(sources) + 1
        routed = torch.empty_like(sources[0])
        weights = torch.empty(
            (n_sources, batch, length, num_heads),
            device=projected.device,
            dtype=torch.float32,
        )
        inv_rms = torch.empty(
            (n_sources, batch, length), device=projected.device, dtype=torch.float32
        )
        return routed, weights, inv_rms

    @torch.library.custom_op(
        "delta_feedback::route_backward", mutates_args=(), device_types="cuda"
    )
    def _route_backward_op(
        projected: Tensor,
        grad_routed: Tensor,
        weights: Tensor,
        inv_rms: Tensor,
        null: Tensor,
        sources: list[Tensor],
        num_heads: int,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        return _route_backward_impl(
            projected, grad_routed, weights, inv_rms, null, sources, num_heads
        )

    @_route_backward_op.register_fake
    def _route_backward_fake(
        projected: Tensor,
        grad_routed: Tensor,
        weights: Tensor,
        inv_rms: Tensor,
        null: Tensor,
        sources: list[Tensor],
        num_heads: int,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        del grad_routed, weights, inv_rms, num_heads
        return (
            torch.empty_like(projected),
            torch.empty_like(null),
            [torch.empty_like(source) for source in sources],
        )

    def _route_setup_context(ctx, inputs, output) -> None:
        projected, _present, null, sources, _eps, num_heads = inputs
        _routed, weights, inv_rms = output
        ctx.save_for_backward(projected, null, weights, inv_rms, *sources)
        ctx.num_heads = num_heads
        ctx.mark_non_differentiable(weights, inv_rms)

    def _route_autograd_backward(ctx, grad_routed, _grad_weights, _grad_inv_rms):
        projected, null, weights, inv_rms, *sources = ctx.saved_tensors
        grad_projected, grad_null, source_grads = _route_backward_op(
            projected,
            grad_routed,
            weights,
            inv_rms,
            null,
            list(sources),
            ctx.num_heads,
        )
        return grad_projected, None, grad_null, source_grads, None, None

    _route_forward_op.register_autograd(
        _route_autograd_backward, setup_context=_route_setup_context
    )
else:  # pragma: no cover - the Mac never enters the CUDA route.
    _route_forward_op = None


def bespoke_route(
    projected: Tensor,
    present: Tensor | None,
    null: Tensor,
    eps: float,
    num_heads: int,
    sources: tuple[Tensor, ...],
) -> tuple[Tensor, Tensor]:
    """Full-width RMS keys plus per-head depth softmaxes in fused Triton.

    ``null`` is the site's width-``dim`` null value; it is source 0 of the
    returned weights and never materialized per token.
    """
    if triton is None or not projected.is_cuda:
        raise RuntimeError("bespoke_route requires Triton CUDA")
    present_arg = (
        present
        if present is not None
        else torch.empty(0, device=projected.device, dtype=torch.bool)
    )
    routed, weights, _inv_rms = _route_forward_op(
        projected, present_arg, null.contiguous(), list(sources), eps, num_heads
    )
    return routed, weights


__all__ = [
    "MAX_ROUTE_SOURCES",
    "ROUTE_TOKENS_PER_PROGRAM",
    "bespoke_route",
    "dw_accum",
    "pack_control_gradients",
    "sink_linear",
    "triton",
]

"""Bespoke CUDA kernels for the DF hot path.

The portable semantic implementations remain in :mod:`model`.  This module is
optional at import time and exposes only kernels whose fixed screen geometry
benefits materially from owning the reduction and memory traffic directly.
"""

from __future__ import annotations

import torch
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
    def _route_logits_kernel(
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
        bt: tl.constexpr,
        dim: tl.constexpr,
        eps: tl.constexpr,
        n_sources: tl.constexpr,
        has_present: tl.constexpr,
        null_first: tl.constexpr,
        block_d: tl.constexpr,
    ):
        token = tl.program_id(0)
        offsets = tl.arange(0, block_d)
        mask = offsets < dim
        p = tl.load(projected + offsets, mask=mask, other=0.0).to(tl.float32)
        for index in range(n_sources):
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
            base = 0 if null_first and index == 0 else token * dim
            value = tl.load(source + base + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            inverse = tl.rsqrt(tl.sum(value * value, axis=0) / dim + eps)
            score = tl.sum(value * p, axis=0) * inverse
            if has_present:
                exists = tl.load(present + index * bt + token)
                score = tl.where(exists, score, -float("inf"))
            tl.store(logits + index * bt + token, score)
            tl.store(inv_rms + index * bt + token, inverse)

    @triton.jit
    def _route_softmax_kernel(
        logits,
        weights,
        bt: tl.constexpr,
        n_sources: tl.constexpr,
        block_n: tl.constexpr,
    ):
        token = tl.program_id(0)
        offsets = tl.arange(0, block_n)
        mask = offsets < n_sources
        values = tl.load(logits + offsets * bt + token, mask=mask, other=-float("inf"))
        values = values - tl.max(values, axis=0)
        numerators = tl.exp(values)
        result = numerators / tl.sum(numerators, axis=0)
        tl.store(weights + offsets * bt + token, result, mask=mask)

    @triton.jit
    def _route_mix_kernel(
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
        weights,
        routed,
        bt: tl.constexpr,
        dim: tl.constexpr,
        n_sources: tl.constexpr,
        null_first: tl.constexpr,
        block_d: tl.constexpr,
    ):
        token = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * block_d + tl.arange(0, block_d)
        mask = offsets < dim
        total = tl.zeros((block_d,), tl.float32)
        for index in range(n_sources):
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
            base = 0 if null_first and index == 0 else token * dim
            value = tl.load(source + base + offsets, mask=mask, other=0.0)
            weight = tl.load(weights + index * bt + token)
            total += value * weight
        tl.store(routed + token * dim + offsets, total, mask=mask)

    @triton.jit
    def _route_beta_kernel(
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
        grad_routed,
        weights,
        beta,
        bt: tl.constexpr,
        dim: tl.constexpr,
        n_sources: tl.constexpr,
        null_first: tl.constexpr,
        block_d: tl.constexpr,
        block_n: tl.constexpr,
    ):
        token = tl.program_id(0)
        d_offsets = tl.arange(0, block_d)
        d_mask = d_offsets < dim
        grad = tl.load(
            grad_routed + token * dim + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32)
        n_offsets = tl.arange(0, block_n)
        n_mask = n_offsets < n_sources
        products = tl.zeros((block_n,), tl.float32)
        for index in range(n_sources):
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
            base = 0 if null_first and index == 0 else token * dim
            value = tl.load(source + base + d_offsets, mask=d_mask, other=0.0).to(
                tl.float32
            )
            dot = tl.sum(grad * value, axis=0)
            products = tl.where(n_offsets == index, dot, products)
        route_weights = tl.load(
            weights + n_offsets * bt + token, mask=n_mask, other=0.0
        )
        centered = products - tl.sum(route_weights * products, axis=0)
        tl.store(
            beta + n_offsets * bt + token,
            route_weights * centered,
            mask=n_mask,
        )

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
        beta,
        grad_projected,
        bt: tl.constexpr,
        dim: tl.constexpr,
        n_sources: tl.constexpr,
        null_first: tl.constexpr,
        block_d: tl.constexpr,
    ):
        token = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * block_d + tl.arange(0, block_d)
        mask = offsets < dim
        p = tl.load(projected + offsets, mask=mask, other=0.0).to(tl.float32)
        upstream = tl.load(
            grad_routed + token * dim + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        grad_p = tl.zeros((block_d,), tl.float32)
        for index in range(n_sources):
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
            grad_source = _route_pointer(
                index,
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
            source_base = 0 if null_first and index == 0 else token * dim
            value = tl.load(source + source_base + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            weight = tl.load(weights + index * bt + token)
            route_beta = tl.load(beta + index * bt + token)
            inverse = tl.load(inv_rms + index * bt + token)
            # ``logits`` carries -inf for absent sources into softmax. Rebuild
            # the finite pre-mask score here so beta=0 produces an exact zero
            # gradient rather than the indeterminate 0 * inf.
            score = tl.sum(value * p, axis=0) * inverse
            source_grad = weight * upstream + route_beta * (
                p * inverse - score * inverse * inverse * value / dim
            )
            tl.store(grad_source + token * dim + offsets, source_grad, mask=mask)
            grad_p += route_beta * value * inverse
        tl.store(grad_projected + token * dim + offsets, grad_p, mask=mask)


def _padded_sources(sources: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    if not 2 <= len(sources) <= MAX_ROUTE_SOURCES:
        raise ValueError(f"bespoke router needs 2..{MAX_ROUTE_SOURCES} sources")
    return sources + (sources[-1],) * (MAX_ROUTE_SOURCES - len(sources))


def _route_forward_impl(
    projected: Tensor,
    present: Tensor,
    sources: list[Tensor],
    null_first: bool,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    n_sources = len(sources)
    batch, length, dim = sources[0].shape
    bt = batch * length
    padded = _padded_sources(tuple(sources))
    logits = torch.empty((n_sources, bt), device=projected.device, dtype=torch.float32)
    inv_rms = torch.empty_like(logits)
    weights = torch.empty_like(logits)
    routed = torch.empty_like(sources[0], memory_format=torch.contiguous_format)
    block_d = triton.next_power_of_2(dim)
    has_present = present.numel() > 0
    present_arg = present if has_present else sources[0]
    _route_logits_kernel[(bt,)](
        *padded,
        projected,
        present_arg,
        logits,
        inv_rms,
        bt=bt,
        dim=dim,
        eps=eps,
        n_sources=n_sources,
        has_present=has_present,
        null_first=null_first,
        block_d=block_d,
        num_warps=8,
    )
    _route_softmax_kernel[(bt,)](
        logits,
        weights,
        bt=bt,
        n_sources=n_sources,
        block_n=triton.next_power_of_2(n_sources),
        num_warps=1,
    )
    mix_block = 256
    _route_mix_kernel[(bt, triton.cdiv(dim, mix_block))](
        *padded,
        weights,
        routed,
        bt=bt,
        dim=dim,
        n_sources=n_sources,
        null_first=null_first,
        block_d=mix_block,
        num_warps=4,
    )
    return routed, weights.view(n_sources, batch, length), inv_rms


def _route_backward_impl(
    projected: Tensor,
    grad_routed: Tensor,
    weights: Tensor,
    inv_rms: Tensor,
    sources: list[Tensor],
    null_first: bool,
) -> tuple[Tensor, list[Tensor]]:
    grad_routed = grad_routed.contiguous()
    batch, length, dim = sources[0].shape
    bt = batch * length
    n_sources = len(sources)
    flat_weights = weights.view(n_sources, bt)
    padded = _padded_sources(tuple(sources))
    beta = torch.empty_like(flat_weights)
    block_d = triton.next_power_of_2(dim)
    _route_beta_kernel[(bt,)](
        *padded,
        grad_routed,
        flat_weights,
        beta,
        bt=bt,
        dim=dim,
        n_sources=n_sources,
        null_first=null_first,
        block_d=block_d,
        block_n=triton.next_power_of_2(n_sources),
        num_warps=8,
    )
    source_grads = [
        torch.empty(source.shape, device=source.device, dtype=source.dtype)
        for source in sources
    ]
    padded_grads = tuple(source_grads) + (source_grads[-1],) * (
        MAX_ROUTE_SOURCES - n_sources
    )
    grad_projected_tokens = torch.empty_like(
        sources[0], memory_format=torch.contiguous_format
    )
    grad_block = 256
    _route_backward_kernel[(bt, triton.cdiv(dim, grad_block))](
        *padded,
        *padded_grads,
        projected,
        grad_routed,
        flat_weights,
        inv_rms,
        beta,
        grad_projected_tokens,
        bt=bt,
        dim=dim,
        n_sources=n_sources,
        null_first=null_first,
        block_d=grad_block,
        num_warps=4,
    )
    grad_projected = grad_projected_tokens.sum(dim=(0, 1)).to(projected.dtype)
    return grad_projected, source_grads


if triton is not None:

    @torch.library.custom_op(
        "delta_feedback::route_forward", mutates_args=(), device_types="cuda"
    )
    def _route_forward_op(
        projected: Tensor,
        present: Tensor,
        sources: list[Tensor],
        null_first: bool,
        eps: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return _route_forward_impl(projected, present, sources, null_first, eps)

    @_route_forward_op.register_fake
    def _route_forward_fake(
        projected: Tensor,
        present: Tensor,
        sources: list[Tensor],
        null_first: bool,
        eps: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del present, null_first, eps
        batch, length, _ = sources[0].shape
        aux_shape = (len(sources), batch, length)
        routed = torch.empty_like(sources[0])
        weights = torch.empty(aux_shape, device=projected.device, dtype=torch.float32)
        inv_rms = torch.empty(aux_shape, device=projected.device, dtype=torch.float32)
        return routed, weights, inv_rms

    @torch.library.custom_op(
        "delta_feedback::route_backward", mutates_args=(), device_types="cuda"
    )
    def _route_backward_op(
        projected: Tensor,
        grad_routed: Tensor,
        weights: Tensor,
        inv_rms: Tensor,
        sources: list[Tensor],
        null_first: bool,
    ) -> tuple[Tensor, list[Tensor]]:
        return _route_backward_impl(
            projected, grad_routed, weights, inv_rms, sources, null_first
        )

    @_route_backward_op.register_fake
    def _route_backward_fake(
        projected: Tensor,
        grad_routed: Tensor,
        weights: Tensor,
        inv_rms: Tensor,
        sources: list[Tensor],
        null_first: bool,
    ) -> tuple[Tensor, list[Tensor]]:
        del grad_routed, weights, inv_rms, null_first
        return torch.empty_like(projected), [
            torch.empty_like(source) for source in sources
        ]

    def _route_setup_context(ctx, inputs, output) -> None:
        projected, _present, sources, null_first, _eps = inputs
        _routed, weights, inv_rms = output
        ctx.save_for_backward(projected, weights, inv_rms, *sources)
        ctx.n_sources = len(sources)
        ctx.null_first = null_first
        ctx.mark_non_differentiable(weights, inv_rms)

    def _route_autograd_backward(ctx, grad_routed, _grad_weights, _grad_inv_rms):
        projected, weights, inv_rms, *sources = ctx.saved_tensors
        grad_projected, source_grads = _route_backward_op(
            projected,
            grad_routed,
            weights,
            inv_rms,
            list(sources),
            ctx.null_first,
        )
        return grad_projected, None, source_grads, None, None

    _route_forward_op.register_autograd(
        _route_autograd_backward, setup_context=_route_setup_context
    )
else:  # pragma: no cover - the Mac never enters the CUDA route.
    _route_forward_op = None


def bespoke_route(
    projected: Tensor,
    present: Tensor | None,
    null_first: bool,
    eps: float,
    sources: tuple[Tensor, ...],
) -> tuple[Tensor, Tensor]:
    """RMS-key softmax route with fixed-capacity Triton forward/backward."""
    if triton is None or not projected.is_cuda:
        raise RuntimeError("bespoke_route requires Triton CUDA")
    present_arg = (
        present
        if present is not None
        else torch.empty(0, device=projected.device, dtype=torch.bool)
    )
    routed, weights, _inv_rms = _route_forward_op(
        projected, present_arg, list(sources), null_first, eps
    )
    return routed, weights


__all__ = ["MAX_ROUTE_SOURCES", "bespoke_route", "triton"]

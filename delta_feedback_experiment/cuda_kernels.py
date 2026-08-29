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
        sources = (
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
        for index in range(n_sources):
            base = 0 if null_first and index == 0 else token * dim
            value = tl.load(sources[index] + base + offsets, mask=mask, other=0.0).to(
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
        sources = (
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
        for index in range(n_sources):
            base = 0 if null_first and index == 0 else token * dim
            value = tl.load(sources[index] + base + offsets, mask=mask, other=0.0)
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
        sources = (
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
        for index in range(n_sources):
            base = 0 if null_first and index == 0 else token * dim
            value = tl.load(
                sources[index] + base + d_offsets, mask=d_mask, other=0.0
            ).to(tl.float32)
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
        logits,
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
        sources = (
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
        gradients = (
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
        for index in range(n_sources):
            source_base = 0 if null_first and index == 0 else token * dim
            value = tl.load(
                sources[index] + source_base + offsets, mask=mask, other=0.0
            ).to(tl.float32)
            weight = tl.load(weights + index * bt + token)
            route_beta = tl.load(beta + index * bt + token)
            inverse = tl.load(inv_rms + index * bt + token)
            score = tl.load(logits + index * bt + token)
            source_grad = weight * upstream + route_beta * (
                p * inverse - score * inverse * inverse * value / dim
            )
            tl.store(gradients[index] + token * dim + offsets, source_grad, mask=mask)
            grad_p += route_beta * value * inverse
        tl.store(grad_projected + token * dim + offsets, grad_p, mask=mask)


def _padded_sources(sources: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    if not 2 <= len(sources) <= MAX_ROUTE_SOURCES:
        raise ValueError(f"bespoke router needs 2..{MAX_ROUTE_SOURCES} sources")
    return sources + (sources[-1],) * (MAX_ROUTE_SOURCES - len(sources))


class _BespokeRoute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        projected: Tensor,
        present: Tensor | None,
        null_first: bool,
        eps: float,
        *sources: Tensor,
    ):
        if triton is None:  # pragma: no cover - guarded by the public wrapper
            raise RuntimeError("Triton is unavailable")
        n_sources = len(sources)
        batch, length, dim = sources[0].shape
        bt = batch * length
        padded = _padded_sources(tuple(sources))
        logits = torch.empty(
            (n_sources, bt), device=projected.device, dtype=torch.float32
        )
        inv_rms = torch.empty_like(logits)
        weights = torch.empty_like(logits)
        routed = torch.empty_like(sources[0], memory_format=torch.contiguous_format)
        block_d = triton.next_power_of_2(dim)
        present_arg = present if present is not None else sources[0]
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
            has_present=present is not None,
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
        ctx.save_for_backward(projected, weights, logits, inv_rms, *sources)
        ctx.null_first = null_first
        ctx.shape = (bt, dim)
        return routed, weights.view(n_sources, batch, length)

    @staticmethod
    def backward(ctx, grad_routed: Tensor, _grad_weights: Tensor | None):
        projected, weights, logits, inv_rms, *sources = ctx.saved_tensors
        bt, dim = ctx.shape
        n_sources = len(sources)
        padded = _padded_sources(tuple(sources))
        beta = torch.empty_like(weights)
        block_d = triton.next_power_of_2(dim)
        _route_beta_kernel[(bt,)](
            *padded,
            grad_routed,
            weights,
            beta,
            bt=bt,
            dim=dim,
            n_sources=n_sources,
            null_first=ctx.null_first,
            block_d=block_d,
            block_n=triton.next_power_of_2(n_sources),
            num_warps=8,
        )
        source_grads = tuple(
            torch.empty(source.shape, device=source.device, dtype=source.dtype)
            for source in sources
        )
        padded_grads = source_grads + (source_grads[-1],) * (
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
            weights,
            logits,
            inv_rms,
            beta,
            grad_projected_tokens,
            bt=bt,
            dim=dim,
            n_sources=n_sources,
            null_first=ctx.null_first,
            block_d=grad_block,
            num_warps=4,
        )
        grad_projected = grad_projected_tokens.sum(dim=(0, 1)).to(projected.dtype)
        return grad_projected, None, None, None, *source_grads


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
    return _BespokeRoute.apply(projected, present, null_first, eps, *sources)


__all__ = ["MAX_ROUTE_SOURCES", "bespoke_route", "triton"]

"""Bespoke CUDA kernels for the MHDB x FBT hot path.

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
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        eps: tl.constexpr,
        n_sources: tl.constexpr,
        has_present: tl.constexpr,
        null_first: tl.constexpr,
        block_h: tl.constexpr,
        block_k: tl.constexpr,
    ):
        token = tl.program_id(0)
        h_offsets = tl.arange(0, block_h)
        k_offsets = tl.arange(0, block_k)
        head_mask = h_offsets < num_heads
        mask = head_mask[:, None] & (k_offsets[None, :] < head_dim)
        offsets = h_offsets[:, None] * head_dim + k_offsets[None, :]
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
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        n_sources: tl.constexpr,
        null_first: tl.constexpr,
        block_k: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        k_offsets = tl.arange(0, block_k)
        offsets = head * head_dim + k_offsets
        mask = k_offsets < head_dim
        total = tl.zeros((block_k,), tl.float32)
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
            weight = tl.load(
                weights + index * bt * num_heads + token * num_heads + head
            )
            total += value * weight
        tl.store(routed + token * dim + offsets, total, mask=mask)

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
        grad_projected,
        bt: tl.constexpr,
        dim: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        n_sources: tl.constexpr,
        null_first: tl.constexpr,
        block_h: tl.constexpr,
        block_k: tl.constexpr,
    ):
        token = tl.program_id(0)
        h_offsets = tl.arange(0, block_h)
        k_offsets = tl.arange(0, block_k)
        head_mask = h_offsets < num_heads
        mask = head_mask[:, None] & (k_offsets[None, :] < head_dim)
        offsets = h_offsets[:, None] * head_dim + k_offsets[None, :]
        p = tl.load(projected + offsets, mask=mask, other=0.0).to(tl.float32)
        upstream = tl.load(
            grad_routed + token * dim + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        centered = tl.zeros((block_h,), tl.float32)
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
            source_base = 0 if null_first and index == 0 else token * dim
            value = tl.load(source + source_base + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            weight = tl.load(
                weights + index * bt * num_heads + token * num_heads + h_offsets,
                mask=head_mask,
                other=0.0,
            )
            centered += weight * tl.sum(upstream * value, axis=1)

        grad_p = tl.zeros((block_h, block_k), tl.float32)
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
            weight = tl.load(
                weights + index * bt * num_heads + token * num_heads + h_offsets,
                mask=head_mask,
                other=0.0,
            )
            inverse = tl.load(inv_rms + index * bt + token)
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
            tl.store(grad_source + token * dim + offsets, source_grad, mask=mask)
            grad_p += route_beta[:, None] * value * inverse
        tl.store(grad_projected + token * dim + offsets, grad_p, mask=mask)


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
    sources: list[Tensor],
    null_first: bool,
    eps: float,
    num_heads: int,
) -> tuple[Tensor, Tensor, Tensor]:
    n_sources = len(sources)
    batch, length, dim = sources[0].shape
    bt = batch * length
    head_dim = _check_route_dims(dim, num_heads)
    block_h, block_k, num_warps = _route_launch(num_heads, head_dim)
    padded = _padded_sources(tuple(sources))
    logits = torch.empty(
        (n_sources, bt, num_heads), device=projected.device, dtype=torch.float32
    )
    inv_rms = torch.empty((n_sources, bt), device=projected.device, dtype=torch.float32)
    weights = torch.empty_like(logits)
    routed = torch.empty_like(sources[0], memory_format=torch.contiguous_format)
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
        num_heads=num_heads,
        head_dim=head_dim,
        eps=eps,
        n_sources=n_sources,
        has_present=has_present,
        null_first=null_first,
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
    _route_mix_kernel[(bt, num_heads)](
        *padded,
        weights,
        routed,
        bt=bt,
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        n_sources=n_sources,
        null_first=null_first,
        block_k=block_k,
        num_warps=max(1, num_warps // 2),
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
    sources: list[Tensor],
    null_first: bool,
    num_heads: int,
) -> tuple[Tensor, list[Tensor]]:
    grad_routed = grad_routed.contiguous()
    batch, length, dim = sources[0].shape
    bt = batch * length
    n_sources = len(sources)
    head_dim = _check_route_dims(dim, num_heads)
    block_h, block_k, num_warps = _route_launch(num_heads, head_dim)
    flat_weights = weights.view(n_sources, bt, num_heads)
    padded = _padded_sources(tuple(sources))
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
    _route_backward_kernel[(bt,)](
        *padded,
        *padded_grads,
        projected,
        grad_routed,
        flat_weights,
        inv_rms,
        grad_projected_tokens,
        bt=bt,
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        n_sources=n_sources,
        null_first=null_first,
        block_h=block_h,
        block_k=block_k,
        num_warps=num_warps,
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
        num_heads: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return _route_forward_impl(
            projected, present, sources, null_first, eps, num_heads
        )

    @_route_forward_op.register_fake
    def _route_forward_fake(
        projected: Tensor,
        present: Tensor,
        sources: list[Tensor],
        null_first: bool,
        eps: float,
        num_heads: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del present, null_first, eps
        batch, length, _ = sources[0].shape
        routed = torch.empty_like(sources[0])
        weights = torch.empty(
            (len(sources), batch, length, num_heads),
            device=projected.device,
            dtype=torch.float32,
        )
        inv_rms = torch.empty(
            (len(sources), batch, length),
            device=projected.device,
            dtype=torch.float32,
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
        sources: list[Tensor],
        null_first: bool,
        num_heads: int,
    ) -> tuple[Tensor, list[Tensor]]:
        return _route_backward_impl(
            projected,
            grad_routed,
            weights,
            inv_rms,
            sources,
            null_first,
            num_heads,
        )

    @_route_backward_op.register_fake
    def _route_backward_fake(
        projected: Tensor,
        grad_routed: Tensor,
        weights: Tensor,
        inv_rms: Tensor,
        sources: list[Tensor],
        null_first: bool,
        num_heads: int,
    ) -> tuple[Tensor, list[Tensor]]:
        del grad_routed, weights, inv_rms, null_first, num_heads
        return torch.empty_like(projected), [
            torch.empty_like(source) for source in sources
        ]

    def _route_setup_context(ctx, inputs, output) -> None:
        projected, _present, sources, null_first, _eps, num_heads = inputs
        _routed, weights, inv_rms = output
        ctx.save_for_backward(projected, weights, inv_rms, *sources)
        ctx.n_sources = len(sources)
        ctx.null_first = null_first
        ctx.num_heads = num_heads
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
            ctx.num_heads,
        )
        return grad_projected, None, source_grads, None, None, None

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
    num_heads: int,
    sources: tuple[Tensor, ...],
) -> tuple[Tensor, Tensor]:
    """Full-width RMS keys plus per-head depth softmaxes in fused Triton."""
    if triton is None or not projected.is_cuda:
        raise RuntimeError("bespoke_route requires Triton CUDA")
    present_arg = (
        present
        if present is not None
        else torch.empty(0, device=projected.device, dtype=torch.bool)
    )
    routed, weights, _inv_rms = _route_forward_op(
        projected, present_arg, list(sources), null_first, eps, num_heads
    )
    return routed, weights


__all__ = ["MAX_ROUTE_SOURCES", "bespoke_route", "triton"]

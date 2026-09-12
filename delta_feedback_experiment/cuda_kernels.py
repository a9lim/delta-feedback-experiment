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

ROUTE_TILE_LANES = 64
"""Lanes in one head-width sub-tile of the routing kernels."""

MAX_ROUTE_TILES = 4
"""Sub-tiles the routing kernels unroll; wider heads take wider sub-tiles."""


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
    def _route_tile_masks(
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_h: tl.constexpr,
        block_k: tl.constexpr,
    ):
        """Head offsets plus the offsets and masks of the width sub-tiles.

        The head width is walked in ``block_k``-lane sub-tiles instead of being
        padded to the next power of two: a 192-wide head is three full 64-lane
        tiles rather than one 256-lane tile with a quarter of its lanes idle.
        Only a width that is not a multiple of ``block_k`` masks anything, and
        only in its last tile.
        """
        h_offsets = tl.arange(0, block_h)
        columns = tl.arange(0, block_k)[None, :]
        head_mask = h_offsets < num_heads
        rows = h_offsets[:, None] * head_dim
        offsets0 = rows + columns
        offsets1 = offsets0 + block_k
        offsets2 = offsets1 + block_k
        offsets3 = offsets2 + block_k
        mask0 = head_mask[:, None] & (columns < head_dim)
        mask1 = head_mask[:, None] & (columns + block_k < head_dim)
        mask2 = head_mask[:, None] & (columns + 2 * block_k < head_dim)
        mask3 = head_mask[:, None] & (columns + 3 * block_k < head_dim)
        return (
            h_offsets,
            head_mask,
            offsets0,
            offsets1,
            offsets2,
            offsets3,
            mask0,
            mask1,
            mask2,
            mask3,
        )

    @triton.jit
    def _route_load(
        pointer,
        offsets0,
        offsets1,
        offsets2,
        offsets3,
        mask0,
        mask1,
        mask2,
        mask3,
        block_h: tl.constexpr,
        block_k: tl.constexpr,
        tiles: tl.constexpr,
    ):
        """One head-width read as up to four FP32 sub-tiles."""
        value0 = tl.load(pointer + offsets0, mask=mask0, other=0.0).to(tl.float32)
        value1 = tl.zeros((block_h, block_k), tl.float32)
        value2 = value1
        value3 = value1
        if tiles > 1:
            value1 = tl.load(pointer + offsets1, mask=mask1, other=0.0).to(tl.float32)
        if tiles > 2:
            value2 = tl.load(pointer + offsets2, mask=mask2, other=0.0).to(tl.float32)
        if tiles > 3:
            value3 = tl.load(pointer + offsets3, mask=mask3, other=0.0).to(tl.float32)
        return value0, value1, value2, value3

    @triton.jit
    def _route_store(
        pointer,
        offsets0,
        offsets1,
        offsets2,
        offsets3,
        mask0,
        mask1,
        mask2,
        mask3,
        value0,
        value1,
        value2,
        value3,
        tiles: tl.constexpr,
    ):
        """Write the sub-tiles back in the pointer's own dtype."""
        dtype = pointer.dtype.element_ty
        tl.store(pointer + offsets0, value0.to(dtype), mask=mask0)
        if tiles > 1:
            tl.store(pointer + offsets1, value1.to(dtype), mask=mask1)
        if tiles > 2:
            tl.store(pointer + offsets2, value2.to(dtype), mask=mask2)
        if tiles > 3:
            tl.store(pointer + offsets3, value3.to(dtype), mask=mask3)

    @triton.jit
    def _route_fold(
        a0,
        a1,
        a2,
        a3,
        b0,
        b1,
        b2,
        b3,
        tiles: tl.constexpr,
    ):
        """``sum_t a_t * b_t`` elementwise, so the caller reduces once."""
        folded = a0 * b0
        if tiles > 1:
            folded += a1 * b1
        if tiles > 2:
            folded += a2 * b2
        if tiles > 3:
            folded += a3 * b3
        return folded

    @triton.jit
    def _route_scale(a0, a1, a2, a3, scale, tiles: tl.constexpr):
        """Scale every sub-tile by one per-head factor."""
        factor = scale[:, None]
        c0 = a0 * factor
        c1 = a1
        c2 = a2
        c3 = a3
        if tiles > 1:
            c1 = a1 * factor
        if tiles > 2:
            c2 = a2 * factor
        if tiles > 3:
            c3 = a3 * factor
        return c0, c1, c2, c3

    @triton.jit
    def _route_add(a0, a1, a2, a3, b0, b1, b2, b3, tiles: tl.constexpr):
        """Sub-tile-wise addition."""
        c0 = a0 + b0
        c1 = a1
        c2 = a2
        c3 = a3
        if tiles > 1:
            c1 = a1 + b1
        if tiles > 2:
            c2 = a2 + b2
        if tiles > 3:
            c3 = a3 + b3
        return c0, c1, c2, c3

    @triton.jit
    def _route_mix(
        a0,
        a1,
        a2,
        a3,
        rescale,
        v0,
        v1,
        v2,
        v3,
        term,
        tiles: tl.constexpr,
    ):
        """``a_t * rescale + term * v_t``: the online mixture update."""
        scale = rescale[:, None]
        weight = term[:, None]
        c0 = a0 * scale + weight * v0
        c1 = a1
        c2 = a2
        c3 = a3
        if tiles > 1:
            c1 = a1 * scale + weight * v1
        if tiles > 2:
            c2 = a2 * scale + weight * v2
        if tiles > 3:
            c3 = a3 * scale + weight * v3
        return c0, c1, c2, c3

    @triton.jit
    def _route_source_grad(
        u0,
        u1,
        u2,
        u3,
        weight,
        p0,
        p1,
        p2,
        p3,
        alpha,
        v0,
        v1,
        v2,
        v3,
        coefficient,
        tiles: tl.constexpr,
    ):
        """``weight * upstream + alpha * query - coefficient * value``."""
        mass = weight[:, None]
        pull = alpha[:, None]
        c0 = mass * u0 + pull * p0 - coefficient * v0
        c1 = u1
        c2 = u2
        c3 = u3
        if tiles > 1:
            c1 = mass * u1 + pull * p1 - coefficient * v1
        if tiles > 2:
            c2 = mass * u2 + pull * p2 - coefficient * v2
        if tiles > 3:
            c3 = mass * u3 + pull * p3 - coefficient * v3
        return c0, c1, c2, c3

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
        weights,
        inv_rms,
        routed,
        bt: tl.constexpr,
        dim: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        eps: tl.constexpr,
        n_sources: tl.constexpr,
        block_h: tl.constexpr,
        block_k: tl.constexpr,
        block_n: tl.constexpr,
        tiles: tl.constexpr,
    ):
        """Full-width RMS keys, per-group source softmax, and the value mix in
        one pass over the bank: every source tile is read exactly once per
        token, with the softmax folded in online.  The per-source scores stay in
        registers, so the normalized weights leave with the mixture and no
        second kernel re-reads a logit buffer.  Source 0 is the site's
        width-``dim`` null and is read at a fixed address."""
        token = tl.program_id(0)
        (
            h_offsets,
            head_mask,
            offsets0,
            offsets1,
            offsets2,
            offsets3,
            mask0,
            mask1,
            mask2,
            mask3,
        ) = _route_tile_masks(num_heads, head_dim, block_h, block_k)
        p0, p1, p2, p3 = _route_load(
            projected,
            offsets0,
            offsets1,
            offsets2,
            offsets3,
            mask0,
            mask1,
            mask2,
            mask3,
            block_h,
            block_k,
            tiles,
        )
        n_offsets = tl.arange(0, block_n)
        source_mask = n_offsets < n_sources
        scores = tl.full((block_h, block_n), -float("inf"), tl.float32)
        inverses = tl.zeros((block_n,), tl.float32)
        running_max = tl.full((block_h,), -float("inf"), tl.float32)
        running_sum = tl.zeros((block_h,), tl.float32)
        a0 = tl.zeros((block_h, block_k), tl.float32)
        a1 = a0
        a2 = a0
        a3 = a0
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
            v0, v1, v2, v3 = _route_load(
                source + base,
                offsets0,
                offsets1,
                offsets2,
                offsets3,
                mask0,
                mask1,
                mask2,
                mask3,
                block_h,
                block_k,
                tiles,
            )
            squares = _route_fold(v0, v1, v2, v3, v0, v1, v2, v3, tiles)
            inverse = tl.rsqrt(tl.sum(tl.sum(squares, axis=1), axis=0) / dim + eps)
            products = _route_fold(v0, v1, v2, v3, p0, p1, p2, p3, tiles)
            score = tl.sum(products, axis=1) * inverse
            scores = tl.where(n_offsets[None, :] == index, score[:, None], scores)
            inverses = tl.where(n_offsets == index, inverse, inverses)
            new_max = tl.maximum(running_max, score)
            rescale = tl.where(
                new_max == -float("inf"), 1.0, tl.exp(running_max - new_max)
            )
            term = tl.where(score == -float("inf"), 0.0, tl.exp(score - new_max))
            running_sum = running_sum * rescale + term
            a0, a1, a2, a3 = _route_mix(
                a0, a1, a2, a3, rescale, v0, v1, v2, v3, term, tiles
            )
            running_max = new_max
        normalizer = 1.0 / running_sum
        a0, a1, a2, a3 = _route_scale(a0, a1, a2, a3, normalizer, tiles)
        _route_store(
            routed + token * dim,
            offsets0,
            offsets1,
            offsets2,
            offsets3,
            mask0,
            mask1,
            mask2,
            mask3,
            a0,
            a1,
            a2,
            a3,
            tiles,
        )
        # The running max and sum that normalized the mixture normalize the
        # scores, so the stored weights are exactly the ones the mix used.
        probabilities = tl.exp(scores - running_max[:, None]) * normalizer[:, None]
        tl.store(
            weights
            + n_offsets[None, :] * bt * num_heads
            + token * num_heads
            + h_offsets[:, None],
            probabilities,
            mask=head_mask[:, None] & source_mask[None, :],
        )
        tl.store(inv_rms + n_offsets * bt + token, inverses, mask=source_mask)

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
        n_banked: tl.constexpr,
        tokens_per_program: tl.constexpr,
        block_h: tl.constexpr,
        block_k: tl.constexpr,
        tiles: tl.constexpr,
    ):
        """Analytic MHDB backward over ``tokens_per_program`` tokens.

        Per-token source gradients are stored directly; the query and null
        gradients, which are sums over every token, stay in FP32 registers and
        leave as one ``[dim]`` partial per program.  As in the forward the head
        width is walked in unpadded sub-tiles.

        The first ``n_banked`` sources are banked: their destination is the
        source's own accumulator, which this program reads and rewrites for the
        tokens it owns.  No two programs share a token, so the read-modify-write
        needs no atomics, and the caller then has one gradient per banked source
        instead of one per reader for autograd to sum.
        """
        pid = tl.program_id(0)
        (
            h_offsets,
            head_mask,
            offsets0,
            offsets1,
            offsets2,
            offsets3,
            mask0,
            mask1,
            mask2,
            mask3,
        ) = _route_tile_masks(num_heads, head_dim, block_h, block_k)
        p0, p1, p2, p3 = _route_load(
            projected,
            offsets0,
            offsets1,
            offsets2,
            offsets3,
            mask0,
            mask1,
            mask2,
            mask3,
            block_h,
            block_k,
            tiles,
        )
        zero = tl.zeros((block_h, block_k), tl.float32)
        grad_p0 = zero
        grad_p1 = zero
        grad_p2 = zero
        grad_p3 = zero
        grad_null0 = zero
        grad_null1 = zero
        grad_null2 = zero
        grad_null3 = zero
        for step in range(tokens_per_program):
            token = pid * tokens_per_program + step
            valid = token < bt
            live0 = mask0 & valid
            live1 = mask1 & valid
            live2 = mask2 & valid
            live3 = mask3 & valid
            head_valid = head_mask & valid
            u0, u1, u2, u3 = _route_load(
                grad_routed + token * dim,
                offsets0,
                offsets1,
                offsets2,
                offsets3,
                live0,
                live1,
                live2,
                live3,
                block_h,
                block_k,
                tiles,
            )
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
                v0, v1, v2, v3 = _route_load(
                    source + base,
                    offsets0,
                    offsets1,
                    offsets2,
                    offsets3,
                    live0,
                    live1,
                    live2,
                    live3,
                    block_h,
                    block_k,
                    tiles,
                )
                weight = tl.load(
                    weights + index * bt * num_heads + token * num_heads + h_offsets,
                    mask=head_valid,
                    other=0.0,
                )
                upstream_dot = tl.sum(
                    _route_fold(u0, u1, u2, u3, v0, v1, v2, v3, tiles), axis=1
                )
                centered += weight * upstream_dot

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
                v0, v1, v2, v3 = _route_load(
                    source + base,
                    offsets0,
                    offsets1,
                    offsets2,
                    offsets3,
                    live0,
                    live1,
                    live2,
                    live3,
                    block_h,
                    block_k,
                    tiles,
                )
                weight = tl.load(
                    weights + index * bt * num_heads + token * num_heads + h_offsets,
                    mask=head_valid,
                    other=0.0,
                )
                inverse = tl.load(inv_rms + index * bt + tl.minimum(token, bt - 1))
                upstream_dot = tl.sum(
                    _route_fold(u0, u1, u2, u3, v0, v1, v2, v3, tiles), axis=1
                )
                query_dot = tl.sum(
                    _route_fold(p0, p1, p2, p3, v0, v1, v2, v3, tiles), axis=1
                )
                route_beta = weight * (upstream_dot - centered)
                # RMS statistics are shared across the full hidden width, so the
                # norm-backward correction couples all routing heads.
                norm_dot = tl.sum(route_beta * query_dot, axis=0)
                c0, c1, c2, c3 = _route_source_grad(
                    u0,
                    u1,
                    u2,
                    u3,
                    weight,
                    p0,
                    p1,
                    p2,
                    p3,
                    inverse * route_beta,
                    v0,
                    v1,
                    v2,
                    v3,
                    (inverse * inverse * inverse / dim) * norm_dot,
                    tiles,
                )
                if index == 0:
                    grad_null0, grad_null1, grad_null2, grad_null3 = _route_add(
                        grad_null0,
                        grad_null1,
                        grad_null2,
                        grad_null3,
                        c0,
                        c1,
                        c2,
                        c3,
                        tiles,
                    )
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
                    if index <= n_banked:
                        b0, b1, b2, b3 = _route_load(
                            grad_source + token * dim,
                            offsets0,
                            offsets1,
                            offsets2,
                            offsets3,
                            live0,
                            live1,
                            live2,
                            live3,
                            block_h,
                            block_k,
                            tiles,
                        )
                        c0, c1, c2, c3 = _route_add(
                            c0, c1, c2, c3, b0, b1, b2, b3, tiles
                        )
                    _route_store(
                        grad_source + token * dim,
                        offsets0,
                        offsets1,
                        offsets2,
                        offsets3,
                        live0,
                        live1,
                        live2,
                        live3,
                        c0,
                        c1,
                        c2,
                        c3,
                        tiles,
                    )
                q0, q1, q2, q3 = _route_scale(
                    v0, v1, v2, v3, route_beta * inverse, tiles
                )
                grad_p0, grad_p1, grad_p2, grad_p3 = _route_add(
                    grad_p0, grad_p1, grad_p2, grad_p3, q0, q1, q2, q3, tiles
                )
        _route_store(
            partials + pid * dim,
            offsets0,
            offsets1,
            offsets2,
            offsets3,
            mask0,
            mask1,
            mask2,
            mask3,
            grad_p0,
            grad_p1,
            grad_p2,
            grad_p3,
            tiles,
        )
        _route_store(
            partials + (tl.num_programs(0) + pid) * dim,
            offsets0,
            offsets1,
            offsets2,
            offsets3,
            mask0,
            mask1,
            mask2,
            mask3,
            grad_null0,
            grad_null1,
            grad_null2,
            grad_null3,
            tiles,
        )


ROUTE_TOKENS_PER_PROGRAM = 8
"""Tokens per backward program; sets the query/null partial count."""


def _padded_sources(sources: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    if not 2 <= len(sources) <= MAX_ROUTE_SOURCES:
        raise ValueError(f"bespoke router needs 2..{MAX_ROUTE_SOURCES} sources")
    return sources + (sources[-1],) * (MAX_ROUTE_SOURCES - len(sources))


def _route_launch(num_heads: int, head_dim: int) -> tuple[int, int, int, int]:
    """``(block_h, block_k, tiles, num_warps)`` for one routing site.

    The head width is covered by ``tiles`` sub-tiles of ``block_k`` lanes
    instead of one tile padded to the next power of two, so the common widths
    (192 at every trained scale) carry no idle lanes.  A width wider than
    ``ROUTE_TILE_LANES * MAX_ROUTE_TILES`` widens the sub-tile rather than
    adding tiles, which keeps the kernels' unrolled tile count fixed.
    """
    block_h = 1 << (num_heads - 1).bit_length()
    block_k = min(ROUTE_TILE_LANES, 1 << (head_dim - 1).bit_length())
    tiles = -(-head_dim // block_k)
    while tiles > MAX_ROUTE_TILES:
        block_k *= 2
        tiles = -(-head_dim // block_k)
    lanes = block_h * block_k * tiles
    if lanes > 8192:
        raise RuntimeError(
            f"MHDB routing tile {block_h}x{block_k}x{tiles} is too large "
            f"(H={num_heads}, D/H={head_dim})"
        )
    num_warps = 8 if lanes >= 4096 else (4 if lanes >= 1024 else 2)
    return block_h, block_k, tiles, num_warps


def _check_route_dims(dim: int, num_heads: int) -> int:
    if num_heads < 2 or dim % num_heads:
        raise RuntimeError(
            f"MHDB requires at least two heads dividing hidden size; "
            f"got D={dim}, H={num_heads}"
        )
    return dim // num_heads


def _route_forward_impl(
    projected: Tensor,
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
    block_h, block_k, tiles, num_warps = _route_launch(num_heads, head_dim)
    padded = _padded_sources(bank)
    inv_rms = torch.empty((n_sources, bt), device=projected.device, dtype=torch.float32)
    weights = torch.empty(
        (n_sources, bt, num_heads), device=projected.device, dtype=torch.float32
    )
    routed = torch.empty_like(sources[0], memory_format=torch.contiguous_format)
    _route_forward_kernel[(bt,)](
        *padded,
        projected,
        weights,
        inv_rms,
        routed,
        bt=bt,
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        eps=eps,
        n_sources=n_sources,
        block_h=block_h,
        block_k=block_k,
        block_n=triton.next_power_of_2(n_sources),
        tiles=tiles,
        num_warps=num_warps,
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
    accumulators: list[Tensor],
    num_heads: int,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    grad_routed = grad_routed.contiguous()
    bank = (null, *sources)
    n_sources = len(bank)
    n_banked = len(accumulators)
    batch, length, dim = sources[0].shape
    bt = batch * length
    head_dim = _check_route_dims(dim, num_heads)
    block_h, block_k, tiles, num_warps = _route_launch(num_heads, head_dim)
    flat_weights = weights.view(n_sources, bt, num_heads)
    padded = _padded_sources(bank)
    # A banked source's destination is its own accumulator, which the kernel
    # adds into; only the rest need a gradient tensor of their own.
    source_grads = [
        torch.empty(source.shape, device=source.device, dtype=source.dtype)
        for source in sources[n_banked:]
    ]
    destinations = [*accumulators, *source_grads]
    padded_grads = tuple(destinations) + (destinations[-1],) * (
        MAX_ROUTE_SOURCES - len(destinations)
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
        n_banked=n_banked,
        tokens_per_program=ROUTE_TOKENS_PER_PROGRAM,
        block_h=block_h,
        block_k=block_k,
        tiles=tiles,
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
    gradient is never materialized separately from its accumulator.

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


class ShadowOperand(torch.autograd.Function):
    """Read an address-stable activation-dtype shadow of an FP32 master.

    The forward hands the caller the shadow as the GEMM operand; the backward
    either adds the operand gradient into a persistent FP32 ``sink`` and hands
    autograd nothing (the tied classifier), or returns it widened to FP32 as
    the master's ordinary autograd gradient (the small NAdam-owned matrices).
    Either way no replay casts the master.
    """

    @staticmethod
    def forward(ctx, master: Tensor, shadow: Tensor, sink: Tensor | None) -> Tensor:
        ctx.sink = sink
        return shadow

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor | None, None, None]:
        if ctx.sink is not None:
            ctx.sink.add_(gradient)
            return None, None, None
        return gradient.float(), None, None


def shadowed_weight(
    master: Tensor, shadow: Tensor | None, dtype: torch.dtype
) -> Tensor:
    """The GEMM operand for ``master``: its bound shadow when one matches
    ``dtype`` and autograd is live, otherwise the master itself (autocast or the
    caller then casts as before)."""
    if shadow is not None and shadow.dtype == dtype and torch.is_grad_enabled():
        return ShadowOperand.apply(master, shadow, None)
    return master


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
        ctx,
        x: Tensor,
        splits: tuple[int, ...],
        operand: Tensor,
        packed_sink: Tensor | None,
        *tensors: Tensor,
    ) -> Tensor:
        count = len(splits)
        ctx.save_for_backward(x, operand)
        # Sinks are mutated by every backward that reaches them, so they are
        # held as plain attributes rather than version-checked saved tensors.
        ctx.sinks = tensors[count:]
        ctx.packed_sink = packed_sink
        ctx.splits = splits
        return F.linear(x, operand)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        x, weight = ctx.saved_tensors
        sinks = ctx.sinks
        grad_x = gradient @ weight
        flat_gradient = gradient.reshape(-1, gradient.shape[-1])
        flat_x = x.reshape(-1, x.shape[-1])
        if ctx.packed_sink is not None:
            # The optimizer still sees disjoint per-parameter views, while one
            # GEMM accumulates all row segments into their shared backing.
            fence = dw_accum(flat_gradient, flat_x, ctx.packed_sink)
        else:
            start = 0
            fence = None
            for size, sink in zip(ctx.splits, sinks, strict=True):
                token = dw_accum(flat_gradient[:, start : start + size], flat_x, sink)
                fence = token if fence is None else fence + token
                start += size
        # The fence is zero; the dependency keeps every accumulation alive.
        grad_x = grad_x + fence.to(grad_x.dtype)
        return (grad_x, None, None, None) + (None,) * (2 * len(ctx.splits))


def sink_linear(
    x: Tensor,
    weights: tuple[Tensor, ...],
    sinks: tuple[Tensor | None, ...],
    shadow: Tensor | None = None,
    *,
    packed_sink: Tensor | None = None,
) -> Tensor:
    """Linear over concatenated weights; bound sinks accumulate dW in place.

    ``shadow`` is the trainer-owned activation-dtype copy of the concatenated
    weights, refreshed once per optimizer step, so no replay re-casts or
    re-concatenates FP32 parameters.  Without one (portable paths, analysis,
    cached decoding) the operand is cast per call.

    A bound ``packed_sink`` spans the disjoint row views in ``sinks`` and lets
    their weight gradients accumulate with one GEMM. The model validates the
    storage relationship at binding time, outside compiled execution.
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
        x, tuple(w.shape[0] for w in weights), operand, packed_sink, *weights, *sinks
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
        null: Tensor,
        sources: list[Tensor],
        accumulators: list[Tensor],
        eps: float,
        num_heads: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # ``accumulators`` is the banked sources' gradient destination. The
        # forward never reads it; it is an input so that the site's backward
        # receives the same buffers the source's bank will hand to autograd.
        del accumulators
        return _route_forward_impl(projected, null, sources, eps, num_heads)

    @_route_forward_op.register_fake
    def _route_forward_fake(
        projected: Tensor,
        null: Tensor,
        sources: list[Tensor],
        accumulators: list[Tensor],
        eps: float,
        num_heads: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del null, accumulators, eps
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
        accumulators: list[Tensor],
        num_heads: int,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """MHDB backward; banked sources are accumulated into in place.

        Like ``dw_accum`` this declares no mutation while writing the
        accumulators, and for the same reason: each site's forward saves them,
        and a declared mutation would bump the version counter that every other
        site's saved copy is checked against.  The two differ in what keeps the
        call alive.  ``dw_accum`` returns a zero token its caller folds into a
        live gradient; here the query and null gradients are live outputs of
        every site, because a router's query and null are always trained.
        ``Router.forward`` asserts that, so freezing them fails there rather
        than silently dropping banked gradients.  ``torch.library.opcheck``'s
        schema test rejects both; the accumulators are external graph inputs
        held by the source's bank, so no buffer reuse can reach them.
        """
        return _route_backward_impl(
            projected,
            grad_routed,
            weights,
            inv_rms,
            null,
            sources,
            accumulators,
            num_heads,
        )

    @_route_backward_op.register_fake
    def _route_backward_fake(
        projected: Tensor,
        grad_routed: Tensor,
        weights: Tensor,
        inv_rms: Tensor,
        null: Tensor,
        sources: list[Tensor],
        accumulators: list[Tensor],
        num_heads: int,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        del grad_routed, weights, inv_rms, num_heads
        return (
            torch.empty_like(projected),
            torch.empty_like(null),
            [torch.empty_like(source) for source in sources[len(accumulators) :]],
        )

    def _route_setup_context(ctx, inputs, output) -> None:
        projected, null, sources, accumulators, _eps, num_heads = inputs
        _routed, weights, inv_rms = output
        # The accumulators are mutated by every reader between this forward and
        # this backward, through an operator that declares no mutation for the
        # reason ``dw_accum`` documents, so their saved versions stay valid.
        ctx.save_for_backward(
            projected, null, weights, inv_rms, *sources, *accumulators
        )
        ctx.num_heads = num_heads
        ctx.n_sources = len(sources)
        ctx.mark_non_differentiable(weights, inv_rms)

    def _route_autograd_backward(ctx, grad_routed, _grad_weights, _grad_inv_rms):
        projected, null, weights, inv_rms, *rest = ctx.saved_tensors
        sources = list(rest[: ctx.n_sources])
        accumulators = list(rest[ctx.n_sources :])
        grad_projected, grad_null, source_grads = _route_backward_op(
            projected,
            grad_routed,
            weights,
            inv_rms,
            null,
            sources,
            accumulators,
            ctx.num_heads,
        )
        # A banked source took its gradient in place; autograd gets none for
        # it. The list inputs take a list of gradients of their own length.
        gradients = [None] * len(accumulators) + list(source_grads)
        return (
            grad_projected,
            grad_null,
            gradients,
            [None] * len(accumulators),
            None,
            None,
        )

    _route_forward_op.register_autograd(
        _route_autograd_backward, setup_context=_route_setup_context
    )
else:  # pragma: no cover - the Mac never enters the CUDA route.
    _route_forward_op = None


def bespoke_route(
    projected: Tensor,
    null: Tensor,
    eps: float,
    num_heads: int,
    sources: tuple[Tensor, ...],
    accumulators: tuple[Tensor, ...] = (),
) -> tuple[Tensor, Tensor]:
    """Full-width RMS keys plus per-head depth softmaxes in fused Triton.

    ``null`` is the site's width-``dim`` null value; it is source 0 of the
    returned weights and never materialized per token.  ``accumulators`` are the
    banked destinations of the leading sources: the backward adds each of their
    gradients into its accumulator and returns none for it, so a source read by
    many sites costs one gradient tensor rather than one per site.
    """
    if triton is None or not projected.is_cuda:
        raise RuntimeError("bespoke_route requires Triton CUDA")
    if len(accumulators) > len(sources):
        raise ValueError("banked accumulators must be a prefix of the sources")
    routed, weights, _inv_rms = _route_forward_op(
        projected,
        null.contiguous(),
        list(sources),
        list(accumulators),
        eps,
        num_heads,
    )
    return routed, weights


__all__ = [
    "MAX_ROUTE_SOURCES",
    "ROUTE_TOKENS_PER_PROGRAM",
    "ShadowOperand",
    "bespoke_route",
    "dw_accum",
    "pack_control_gradients",
    "shadowed_weight",
    "sink_linear",
    "triton",
]

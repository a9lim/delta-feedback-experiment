"""Dropless, fixed-shape grouped CUDA GEMMs for the routed expert bank.

The launch envelope covers every possible expert load, while device-side
offsets skip empty tiles. Storage is exactly the selected assignments per token;
there is no capacity factor, CPU routing decision, or dense all-expert FFN.

Each GEMM has its own tile configuration, with wider weight-gradient tiles
on Hopper. The backward fuses the SwiGLU derivative into the epilogue of the
activation-gradient GEMM and recomputes the SwiGLU activation there, so the forward
saves only the pre-activation for backward.

Under the FP8 recipe the two forward GEMMs and the two activation-gradient
GEMMs read FP8 operands with per-row scales: the block input and the
incoming gradient are quantized per token by the caller, the SwiGLU output
by the kernel that computes it, and the gate/up gradient by the epilogue
that produces it, with one scale per row of each column tile, which the
input-gradient GEMM applies block by block along its reduction. The bank's
FP8 copies supply the weights, the transposed copy wherever the reduction
runs over the weight's output axis, so every operand is contiguous along the
reduction. The two weight-gradient GEMMs keep BF16 operands into the FP32
sinks.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .cuda_kernels import FP8_MAX, FP8_SCALE_FLOOR, quantize_rows

try:
    import triton
    import triton.language as tl
except (ImportError, OSError):
    triton = None
    tl = None


# (BM, BN, BK, num_warps, num_stages) per GEMM, measured on Ada at the screen
# geometry (12,288 assignments, D=768, h=832, 15 experts).
TILE_GATE_UP = (64, 128, 32, 4, 3)
TILE_DOWN = (64, 64, 32, 4, 3)
TILE_DACT = (64, 64, 32, 4, 3)
TILE_DX = (128, 64, 32, 4, 3)
TILE_DW_GATE = (64, 64, 64, 4, 3)
TILE_DW_DOWN = (64, 64, 64, 4, 3)
# GH200 benefits from wider output/reduction tiles in both BF16 dW GEMMs.
# The activation GEMMs and their FP8 scale boundaries retain their own tiles.
TILE_DW_GATE_HOPPER = (64, 128, 128, 4, 3)
TILE_DW_DOWN_HOPPER = (64, 128, 128, 4, 3)
SWIGLU_BLOCK = 1024

# The FP8 GEMMs read half the bytes per reduction step, so their tiles reach
# twice as far along it. The gate/up gradient's scale tile is the width of
# the activation-gradient epilogue that writes it.
TILE_GATE_UP_FP8 = (64, 128, 64, 4, 3)
TILE_DOWN_FP8 = (64, 64, 64, 4, 3)
TILE_DACT_FP8 = (64, 64, 64, 4, 3)
TILE_DX_FP8 = (128, 64, 64, 4, 3)


def _weight_gradient_tiles(device: torch.device):
    # Select the tensor's device, without creating a CUDA context on import or
    # accidentally consulting device zero before a distributed rank is placed.
    if torch.cuda.get_device_capability(device) == (9, 0):
        return TILE_DW_GATE_HOPPER, TILE_DW_DOWN_HOPPER
    return TILE_DW_GATE, TILE_DW_DOWN


if triton is not None:

    @triton.jit
    def _grouped_mm(
        x,
        weight,
        assignments,
        offsets,
        output,
        SELECTED: tl.constexpr,
        K: tl.constexpr,
        O: tl.constexpr,
        GATHER: tl.constexpr,
        TRANSPOSE: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        expert = tl.program_id(2)
        start = tl.load(offsets + expert)
        end = tl.load(offsets + expert + 1)
        first = tl.program_id(0) * BM
        if start + first < end:
            rows = start + first + tl.arange(0, BM)
            columns = tl.program_id(1) * BN + tl.arange(0, BN)
            inner = tl.arange(0, BK)
            source_rows = rows
            if GATHER:
                source_rows = (
                    tl.load(assignments + rows, rows < end, other=0) // SELECTED
                )
            acc = tl.zeros((BM, BN), tl.float32)
            for block in range(tl.cdiv(K, BK)):
                k = block * BK + inner
                a = tl.load(
                    x + source_rows[:, None] * K + k[None, :],
                    (rows[:, None] < end) & (k[None, :] < K),
                    other=0,
                )
                if TRANSPOSE:
                    w_offsets = expert * O * K + k[:, None] * O + columns[None, :]
                else:
                    w_offsets = expert * O * K + columns[None, :] * K + k[:, None]
                b = tl.load(
                    weight + w_offsets,
                    (k[:, None] < K) & (columns[None, :] < O),
                    other=0,
                )
                acc = tl.dot(a, b, acc, input_precision="tf32x3")
            tl.store(
                output + rows[:, None] * O + columns[None, :],
                acc,
                (rows[:, None] < end) & (columns[None, :] < O),
            )

    @triton.jit
    def _grouped_mm_dact_swiglu(
        grad,
        weight,
        gate_up,
        offsets,
        dgate_up,
        activated,
        K: tl.constexpr,
        O: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        """``dact = grad @ W_down`` for one expert's rows with the SwiGLU
        backward in the epilogue: writes both halves of ``dgate_up`` and
        recomputes ``activated`` for the down-projection weight gradient."""
        expert = tl.program_id(2)
        start = tl.load(offsets + expert)
        end = tl.load(offsets + expert + 1)
        first = tl.program_id(0) * BM
        if start + first < end:
            rows = start + first + tl.arange(0, BM)
            columns = tl.program_id(1) * BN + tl.arange(0, BN)
            inner = tl.arange(0, BK)
            acc = tl.zeros((BM, BN), tl.float32)
            for block in range(tl.cdiv(K, BK)):
                k = block * BK + inner
                a = tl.load(
                    grad + rows[:, None] * K + k[None, :],
                    (rows[:, None] < end) & (k[None, :] < K),
                    other=0,
                )
                w_offsets = expert * O * K + k[:, None] * O + columns[None, :]
                b = tl.load(
                    weight + w_offsets,
                    (k[:, None] < K) & (columns[None, :] < O),
                    other=0,
                )
                acc = tl.dot(a, b, acc, input_precision="tf32x3")
            mask = (rows[:, None] < end) & (columns[None, :] < O)
            gate_address = gate_up + rows[:, None] * (2 * O) + columns[None, :]
            gate = tl.load(gate_address, mask, other=0).to(tl.float32)
            up = tl.load(gate_address + O, mask, other=0).to(tl.float32)
            sigmoid = tl.sigmoid(gate)
            tl.store(
                dgate_up + rows[:, None] * (2 * O) + columns[None, :],
                acc * up * sigmoid * (1 + gate * (1 - sigmoid)),
                mask,
            )
            tl.store(
                dgate_up + rows[:, None] * (2 * O) + O + columns[None, :],
                acc * gate * sigmoid,
                mask,
            )
            tl.store(
                activated + rows[:, None] * O + columns[None, :],
                gate * sigmoid * up,
                mask,
            )

    @triton.jit
    def _swiglu(
        gate_up, activated, COUNT: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row, column = index // H, index % H
        gate = tl.load(gate_up + row * (2 * H) + column, index < COUNT, other=0).to(
            tl.float32
        )
        up = tl.load(gate_up + row * (2 * H) + H + column, index < COUNT, other=0).to(
            tl.float32
        )
        value = gate * tl.sigmoid(gate) * up
        tl.store(activated + index, value, index < COUNT)

    @triton.jit
    def _grouped_dw(
        x,
        grad,
        assignments,
        offsets,
        sinks,
        K: tl.constexpr,
        O: tl.constexpr,
        SELECTED: tl.constexpr,
        EXPERTS: tl.constexpr,
        GATHER: tl.constexpr,
        ACTIVE: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        expert = tl.program_id(2)
        # Keep the flags alongside the pointer tuple: a scalar bit mask would
        # constrain the bank size to Triton's integer width.
        sink = sinks[0]
        active = tl.full((), ACTIVE[0], tl.int1)
        for index in tl.static_range(1, EXPERTS):
            if expert == index:
                sink = sinks[index]
                active = tl.full((), ACTIVE[index], tl.int1)
        if active:
            start = tl.load(offsets + expert)
            end = tl.load(offsets + expert + 1)
            # An expert without assignments contributes nothing: skip the
            # read-modify-write of its sink entirely.
            if end > start:
                rows = tl.program_id(0) * BM + tl.arange(0, BM)
                columns = tl.program_id(1) * BN + tl.arange(0, BN)
                inner = tl.arange(0, BK)
                acc = tl.zeros((BM, BN), tl.float32)
                for block in range(tl.cdiv(end - start, BK)):
                    tokens = start + block * BK + inner
                    source_rows = tokens
                    if GATHER:
                        source_rows = (
                            tl.load(assignments + tokens, tokens < end, other=0)
                            // SELECTED
                        )
                    a = tl.load(
                        grad + tokens[None, :] * O + rows[:, None],
                        (tokens[None, :] < end) & (rows[:, None] < O),
                        other=0,
                    )
                    b = tl.load(
                        x + source_rows[:, None] * K + columns[None, :],
                        (tokens[:, None] < end) & (columns[None, :] < K),
                        other=0,
                    )
                    acc = tl.dot(a, b, acc, input_precision="tf32x3")
                addresses = sink + rows[:, None] * K + columns[None, :]
                mask = (rows[:, None] < O) & (columns[None, :] < K)
                previous = tl.load(addresses, mask, other=0)
                tl.store(addresses, previous + acc, mask)

    @triton.jit
    def _grouped_mm_fp8(
        x,
        x_scale,
        weight,
        w_scale,
        assignments,
        offsets,
        output,
        SELECTED: tl.constexpr,
        K: tl.constexpr,
        O: tl.constexpr,
        GATHER: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        """``output = (x * x_scale) @ (weight[e] * w_scale[e])^T`` per expert on
        FP8 operands, ``weight`` ``[E, O, K]`` contiguous along ``K``."""
        expert = tl.program_id(2)
        start = tl.load(offsets + expert)
        end = tl.load(offsets + expert + 1)
        first = tl.program_id(0) * BM
        if start + first < end:
            rows = start + first + tl.arange(0, BM)
            columns = tl.program_id(1) * BN + tl.arange(0, BN)
            inner = tl.arange(0, BK)
            source_rows = rows
            if GATHER:
                source_rows = (
                    tl.load(assignments + rows, rows < end, other=0) // SELECTED
                )
            acc = tl.zeros((BM, BN), tl.float32)
            for block in range(tl.cdiv(K, BK)):
                k = block * BK + inner
                a = tl.load(
                    x + source_rows[:, None] * K + k[None, :],
                    (rows[:, None] < end) & (k[None, :] < K),
                    other=0.0,
                )
                b = tl.load(
                    weight + expert * O * K + columns[None, :] * K + k[:, None],
                    (k[:, None] < K) & (columns[None, :] < O),
                    other=0.0,
                )
                acc = tl.dot(a, b, acc)
            row_scale = tl.load(x_scale + source_rows, rows < end, other=0.0)
            column_scale = tl.load(w_scale + expert * O + columns, columns < O, other=0.0)
            acc = acc * row_scale[:, None] * column_scale[None, :]
            tl.store(
                output + rows[:, None] * O + columns[None, :],
                acc,
                (rows[:, None] < end) & (columns[None, :] < O),
            )

    @triton.jit
    def _grouped_mm_dact_swiglu_fp8(
        grad,
        grad_scale,
        weight,
        w_scale,
        gate_up,
        offsets,
        dgate_up,
        dgate_up_fp8,
        dgate_up_scale,
        activated,
        K: tl.constexpr,
        O: tl.constexpr,
        SCALE_TILES: tl.constexpr,
        FP8_MAX_VALUE: tl.constexpr,
        SCALE_FLOOR: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        """``dact = grad @ W_down`` on FP8 operands (``weight`` is the bank's
        transposed down copy ``[E, h, D]``) with the SwiGLU backward in the
        epilogue: writes both halves of ``dgate_up`` in BF16 for the weight
        gradient and in FP8 for the input gradient, one scale per row of this
        column tile covering both halves, and recomputes ``activated``."""
        expert = tl.program_id(2)
        start = tl.load(offsets + expert)
        end = tl.load(offsets + expert + 1)
        first = tl.program_id(0) * BM
        if start + first < end:
            rows = start + first + tl.arange(0, BM)
            columns = tl.program_id(1) * BN + tl.arange(0, BN)
            inner = tl.arange(0, BK)
            acc = tl.zeros((BM, BN), tl.float32)
            for block in range(tl.cdiv(K, BK)):
                k = block * BK + inner
                a = tl.load(
                    grad + rows[:, None] * K + k[None, :],
                    (rows[:, None] < end) & (k[None, :] < K),
                    other=0.0,
                )
                b = tl.load(
                    weight + expert * O * K + columns[None, :] * K + k[:, None],
                    (k[:, None] < K) & (columns[None, :] < O),
                    other=0.0,
                )
                acc = tl.dot(a, b, acc)
            row_scale = tl.load(grad_scale + rows, rows < end, other=0.0)
            column_scale = tl.load(w_scale + expert * O + columns, columns < O, other=0.0)
            acc = acc * row_scale[:, None] * column_scale[None, :]
            mask = (rows[:, None] < end) & (columns[None, :] < O)
            gate_address = gate_up + rows[:, None] * (2 * O) + columns[None, :]
            gate = tl.load(gate_address, mask, other=0).to(tl.float32)
            up = tl.load(gate_address + O, mask, other=0).to(tl.float32)
            sigmoid = tl.sigmoid(gate)
            dgate = acc * up * sigmoid * (1 + gate * (1 - sigmoid))
            dup = acc * gate * sigmoid
            gate_output = dgate_up + rows[:, None] * (2 * O) + columns[None, :]
            tl.store(gate_output, dgate, mask)
            tl.store(gate_output + O, dup, mask)
            tl.store(
                activated + rows[:, None] * O + columns[None, :],
                gate * sigmoid * up,
                mask,
            )
            amax = tl.maximum(
                tl.max(tl.abs(dgate), axis=1), tl.max(tl.abs(dup), axis=1)
            )
            scale = tl.maximum(amax / FP8_MAX_VALUE, SCALE_FLOOR)
            gate_quantized = dgate_up_fp8 + rows[:, None] * (2 * O) + columns[None, :]
            tl.store(gate_quantized, (dgate / scale[:, None]).to(tl.float8e4nv), mask)
            tl.store(gate_quantized + O, (dup / scale[:, None]).to(tl.float8e4nv), mask)
            tl.store(
                dgate_up_scale + rows * SCALE_TILES + tl.program_id(1), scale, rows < end
            )

    @triton.jit
    def _grouped_mm_tilescaled_fp8(
        x,
        x_scale,
        weight,
        w_scale,
        offsets,
        output,
        H: tl.constexpr,
        O: tl.constexpr,
        SCALE_TILES: tl.constexpr,
        SCALE_WIDTH: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        """``output = x @ W_gate_up`` on FP8 operands: ``x`` is the gate/up
        gradient ``[rows, 2H]`` whose scales are per row and per
        ``SCALE_WIDTH``-column tile of each half, applied to each reduction
        block as it lands; ``weight`` is the bank's transposed gate/up copy
        ``[E, O, 2H]`` with a scale per output row."""
        expert = tl.program_id(2)
        start = tl.load(offsets + expert)
        end = tl.load(offsets + expert + 1)
        first = tl.program_id(0) * BM
        if start + first < end:
            rows = start + first + tl.arange(0, BM)
            columns = tl.program_id(1) * BN + tl.arange(0, BN)
            inner = tl.arange(0, BK)
            acc = tl.zeros((BM, BN), tl.float32)
            for half in tl.static_range(2):
                for block in range(tl.cdiv(H, BK)):
                    local = block * BK + inner
                    k = half * H + local
                    a = tl.load(
                        x + rows[:, None] * (2 * H) + k[None, :],
                        (rows[:, None] < end) & (local[None, :] < H),
                        other=0.0,
                    )
                    b = tl.load(
                        weight + expert * O * (2 * H) + columns[None, :] * (2 * H) + k[:, None],
                        (local[:, None] < H) & (columns[None, :] < O),
                        other=0.0,
                    )
                    scale = tl.load(
                        x_scale + rows * SCALE_TILES + (block * BK) // SCALE_WIDTH,
                        rows < end,
                        other=0.0,
                    )
                    acc += tl.dot(a, b) * scale[:, None]
            column_scale = tl.load(w_scale + expert * O + columns, columns < O, other=0.0)
            acc = acc * column_scale[None, :]
            tl.store(
                output + rows[:, None] * O + columns[None, :],
                acc,
                (rows[:, None] < end) & (columns[None, :] < O),
            )

    @triton.jit
    def _swiglu_fp8(
        gate_up,
        activated,
        scale,
        H: tl.constexpr,
        FP8_MAX_VALUE: tl.constexpr,
        SCALE_FLOOR: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One row of the SwiGLU output, quantized with its own scale."""
        row = tl.program_id(0)
        column = tl.arange(0, BLOCK)
        mask = column < H
        gate = tl.load(gate_up + row * (2 * H) + column, mask, other=0).to(tl.float32)
        up = tl.load(gate_up + row * (2 * H) + H + column, mask, other=0).to(tl.float32)
        value = gate * tl.sigmoid(gate) * up
        row_scale = tl.maximum(tl.max(tl.abs(value), axis=0) / FP8_MAX_VALUE, SCALE_FLOOR)
        tl.store(activated + row * H + column, (value / row_scale).to(tl.float8e4nv), mask)
        tl.store(scale + row, row_scale)

    @triton.jit
    def _combine_dx(
        dx,
        inverse,
        output,
        COUNT: tl.constexpr,
        D: tl.constexpr,
        SELECTED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        token, column = index // D, index % D
        value = tl.full((BLOCK,), 0, tl.float32)
        for slot in tl.static_range(SELECTED):
            row = tl.load(inverse + token * SELECTED + slot, index < COUNT, other=0)
            value += tl.load(dx + row * D + column, index < COUNT, other=0).to(
                tl.float32
            )
        tl.store(output + index, value, index < COUNT)


def _mm(
    x: Tensor,
    weight: Tensor,
    assignments: Tensor,
    offsets: Tensor,
    tokens: int,
    output_width: int,
    tile: tuple[int, int, int, int, int],
    *,
    gather: bool = False,
    transpose: bool = False,
) -> Tensor:
    selected = assignments.numel() // tokens
    experts = weight.shape[0]
    block_m, block_n, block_k, warps, stages = tile
    result = x.new_empty((assignments.numel(), output_width))
    _grouped_mm[
        (triton.cdiv(tokens, block_m), triton.cdiv(output_width, block_n), experts)
    ](
        x,
        weight,
        assignments,
        offsets,
        result,
        selected,
        x.shape[-1],
        output_width,
        gather,
        transpose,
        block_m,
        block_n,
        block_k,
        num_warps=warps,
        num_stages=stages,
    )
    return result


@torch.library.custom_op(
    "delta_feedback::moe_forward", mutates_args=(), device_types="cuda"
)
def _forward(
    x: Tensor,
    gate_weight: Tensor,
    down_weight: Tensor,
    assignments: Tensor,
    offsets: Tensor,
) -> tuple[Tensor, Tensor]:
    tokens, dim = x.shape
    width = down_weight.shape[-1]
    gate_up = _mm(
        x, gate_weight, assignments, offsets, tokens, 2 * width, TILE_GATE_UP, gather=True
    )
    activated = x.new_empty((assignments.numel(), width))
    _swiglu[(triton.cdiv(activated.numel(), SWIGLU_BLOCK),)](
        gate_up,
        activated,
        activated.numel(),
        width,
        SWIGLU_BLOCK,
    )
    output = _mm(activated, down_weight, assignments, offsets, tokens, dim, TILE_DOWN)
    return output, gate_up


@_forward.register_fake
def _forward_fake(
    x: Tensor,
    gate_weight: Tensor,
    down_weight: Tensor,
    assignments: Tensor,
    offsets: Tensor,
) -> tuple[Tensor, Tensor]:
    dim = x.shape[-1]
    width = down_weight.shape[-1]
    return (
        x.new_empty((assignments.numel(), dim)),
        x.new_empty((assignments.numel(), 2 * width)),
    )


@torch.library.custom_op(
    "delta_feedback::moe_backward", mutates_args=(), device_types="cuda"
)
def _backward(
    gradient: Tensor,
    x: Tensor,
    gate_up: Tensor,
    gate_weight: Tensor,
    down_weight: Tensor,
    assignments: Tensor,
    inverse: Tensor,
    offsets: Tensor,
    gate_sinks: list[Tensor],
    down_sinks: list[Tensor],
    gate_bound: list[bool],
    down_bound: list[bool],
    gate_required: list[bool],
    down_required: list[bool],
) -> tuple[Tensor, list[Tensor]]:
    # As with dw_accum, sinks are write-only side effects whose versions must
    # remain stable across recurrent backward invocations. The returned dX is
    # the graph dependency keeping these FP32 accumulations alive.
    tokens, dim = x.shape
    selected = assignments.numel() // tokens
    experts = gate_weight.shape[0]
    width = gate_up.shape[-1] // 2
    fresh = []

    def complete(sinks, bound, required, operand):
        existing = iter(sinks)
        result = []
        for index, (is_bound, needed) in enumerate(zip(bound, required, strict=True)):
            if not needed:
                result.append(None)
            elif is_bound:
                result.append(next(existing))
            else:
                gradient = torch.zeros_like(operand[index], dtype=torch.float32)
                fresh.append(gradient)
                result.append(gradient)
        dummy = next((sink for sink in result if sink is not None), None)
        return [dummy if sink is None else sink for sink in result]

    gate_sinks = complete(gate_sinks, gate_bound, gate_required, gate_weight)
    down_sinks = complete(down_sinks, down_bound, down_required, down_weight)
    gradient = gradient.contiguous()
    dgate_up = torch.empty_like(gate_up)
    activated = x.new_empty((assignments.numel(), width))
    block_m, block_n, block_k, warps, stages = TILE_DACT
    _grouped_mm_dact_swiglu[
        (triton.cdiv(tokens, block_m), triton.cdiv(width, block_n), experts)
    ](
        gradient,
        down_weight,
        gate_up,
        offsets,
        dgate_up,
        activated,
        dim,
        width,
        block_m,
        block_n,
        block_k,
        num_warps=warps,
        num_stages=stages,
    )
    dx_assignments = _mm(
        dgate_up, gate_weight, assignments, offsets, tokens, dim, TILE_DX, transpose=True
    )
    gate_tile, down_tile = _weight_gradient_tiles(x.device)
    if any(down_required):
        block_m, block_n, block_k, warps, stages = down_tile
        _grouped_dw[(triton.cdiv(dim, block_m), triton.cdiv(width, block_n), experts)](
            activated,
            gradient,
            assignments,
            offsets,
            tuple(down_sinks),
            width,
            dim,
            selected,
            experts,
            False,
            tuple(down_required),
            block_m,
            block_n,
            block_k,
            num_warps=warps,
            num_stages=stages,
        )
    if any(gate_required):
        block_m, block_n, block_k, warps, stages = gate_tile
        _grouped_dw[
            (triton.cdiv(2 * width, block_m), triton.cdiv(dim, block_n), experts)
        ](
            x,
            dgate_up,
            assignments,
            offsets,
            tuple(gate_sinks),
            dim,
            2 * width,
            selected,
            experts,
            True,
            tuple(gate_required),
            block_m,
            block_n,
            block_k,
            num_warps=warps,
            num_stages=stages,
        )
    dx = torch.empty_like(x)
    _combine_dx[(triton.cdiv(dx.numel(), 256),)](
        dx_assignments,
        inverse,
        dx,
        dx.numel(),
        dim,
        selected,
        256,
    )
    return dx, fresh


@_backward.register_fake
def _backward_fake(
    gradient: Tensor,
    x: Tensor,
    gate_up: Tensor,
    gate_weight: Tensor,
    down_weight: Tensor,
    assignments: Tensor,
    inverse: Tensor,
    offsets: Tensor,
    gate_sinks: list[Tensor],
    down_sinks: list[Tensor],
    gate_bound: list[bool],
    down_bound: list[bool],
    gate_required: list[bool],
    down_required: list[bool],
) -> tuple[Tensor, list[Tensor]]:
    gradients = [
        gate_weight.new_empty(gate_weight.shape[1:], dtype=torch.float32)
        for index, bound in enumerate(gate_bound)
        if not bound and gate_required[index]
    ]
    gradients.extend(
        down_weight.new_empty(down_weight.shape[1:], dtype=torch.float32)
        for index, bound in enumerate(down_bound)
        if not bound and down_required[index]
    )
    return torch.empty_like(x), gradients


def _mm_fp8(
    x: Tensor,
    x_scale: Tensor,
    weight: Tensor,
    w_scale: Tensor,
    assignments: Tensor,
    offsets: Tensor,
    tokens: int,
    output_width: int,
    tile: tuple[int, int, int, int, int],
    *,
    gather: bool = False,
) -> Tensor:
    selected = assignments.numel() // tokens
    experts = weight.shape[0]
    block_m, block_n, block_k, warps, stages = tile
    result = torch.empty(
        (assignments.numel(), output_width), dtype=torch.bfloat16, device=x.device
    )
    _grouped_mm_fp8[
        (triton.cdiv(tokens, block_m), triton.cdiv(output_width, block_n), experts)
    ](
        x,
        x_scale,
        weight,
        w_scale,
        assignments,
        offsets,
        result,
        selected,
        x.shape[-1],
        output_width,
        gather,
        block_m,
        block_n,
        block_k,
        num_warps=warps,
        num_stages=stages,
    )
    return result


@torch.library.custom_op(
    "delta_feedback::moe_forward_fp8", mutates_args=(), device_types="cuda"
)
def _forward_fp8(
    x: Tensor,
    x_scale: Tensor,
    gate_weight: Tensor,
    gate_scale: Tensor,
    down_weight: Tensor,
    down_scale: Tensor,
    assignments: Tensor,
    offsets: Tensor,
) -> tuple[Tensor, Tensor]:
    """The routed forward on FP8 operands: ``x`` is the block input quantized
    per token, the weights the bank's FP8 copies with per-row scales. Returns
    the BF16 output and the BF16 pre-activation the backward reads."""
    tokens, dim = x.shape
    width = down_weight.shape[-1]
    gate_up = _mm_fp8(
        x, x_scale, gate_weight, gate_scale, assignments, offsets, tokens,
        2 * width, TILE_GATE_UP_FP8, gather=True,
    )
    activated = torch.empty(
        (assignments.numel(), width), dtype=torch.float8_e4m3fn, device=x.device
    )
    activated_scale = torch.empty(
        (assignments.numel(),), dtype=torch.float32, device=x.device
    )
    _swiglu_fp8[(assignments.numel(),)](
        gate_up,
        activated,
        activated_scale,
        width,
        FP8_MAX,
        FP8_SCALE_FLOOR,
        triton.next_power_of_2(width),
    )
    output = _mm_fp8(
        activated, activated_scale, down_weight, down_scale, assignments, offsets,
        tokens, dim, TILE_DOWN_FP8,
    )
    return output, gate_up


@_forward_fp8.register_fake
def _forward_fp8_fake(
    x: Tensor,
    x_scale: Tensor,
    gate_weight: Tensor,
    gate_scale: Tensor,
    down_weight: Tensor,
    down_scale: Tensor,
    assignments: Tensor,
    offsets: Tensor,
) -> tuple[Tensor, Tensor]:
    dim = x.shape[-1]
    width = down_weight.shape[-1]
    return (
        torch.empty((assignments.numel(), dim), dtype=torch.bfloat16, device=x.device),
        torch.empty(
            (assignments.numel(), 2 * width), dtype=torch.bfloat16, device=x.device
        ),
    )


@torch.library.custom_op(
    "delta_feedback::moe_backward_fp8", mutates_args=(), device_types="cuda"
)
def _backward_fp8(
    gradient: Tensor,
    gradient_fp8: Tensor,
    gradient_scale: Tensor,
    x: Tensor,
    gate_up: Tensor,
    gate_transposed: Tensor,
    gate_transposed_scale: Tensor,
    down_transposed: Tensor,
    down_transposed_scale: Tensor,
    assignments: Tensor,
    inverse: Tensor,
    offsets: Tensor,
    gate_sinks: list[Tensor],
    down_sinks: list[Tensor],
    gate_bound: list[bool],
    down_bound: list[bool],
    gate_required: list[bool],
    down_required: list[bool],
) -> tuple[Tensor, list[Tensor]]:
    """The routed backward with the activation-gradient GEMMs on FP8 operands.

    ``gradient`` arrives in BF16 for the down weight gradient and quantized
    per row for ``dact``; the transposed FP8 copies of both banks are the
    weight operands, contiguous along each reduction. The weight gradients
    take BF16 operands into the FP32 sinks exactly as the BF16 recipe does.
    """
    tokens, dim = x.shape
    selected = assignments.numel() // tokens
    experts = gate_transposed.shape[0]
    width = gate_up.shape[-1] // 2
    fresh = []

    def complete(sinks, bound, required, operand):
        existing = iter(sinks)
        result = []
        for index, (is_bound, needed) in enumerate(zip(bound, required, strict=True)):
            if not needed:
                result.append(None)
            elif is_bound:
                result.append(next(existing))
            else:
                gradient = torch.zeros(
                    operand.shape[2], operand.shape[1], dtype=torch.float32,
                    device=operand.device,
                )
                fresh.append(gradient)
                result.append(gradient)
        dummy = next((sink for sink in result if sink is not None), None)
        return [dummy if sink is None else sink for sink in result]

    gate_sinks = complete(gate_sinks, gate_bound, gate_required, gate_transposed)
    down_sinks = complete(down_sinks, down_bound, down_required, down_transposed)
    gradient = gradient.contiguous()
    dgate_up = torch.empty_like(gate_up)
    dgate_up_fp8 = torch.empty_like(gate_up, dtype=torch.float8_e4m3fn)
    activated = x.new_empty((assignments.numel(), width))
    block_m, block_n, block_k, warps, stages = TILE_DACT_FP8
    scale_tiles = triton.cdiv(width, block_n)
    dgate_up_scale = torch.empty(
        (assignments.numel(), scale_tiles), dtype=torch.float32, device=x.device
    )
    _grouped_mm_dact_swiglu_fp8[
        (triton.cdiv(tokens, block_m), scale_tiles, experts)
    ](
        gradient_fp8,
        gradient_scale,
        down_transposed,
        down_transposed_scale,
        gate_up,
        offsets,
        dgate_up,
        dgate_up_fp8,
        dgate_up_scale,
        activated,
        dim,
        width,
        scale_tiles,
        FP8_MAX,
        FP8_SCALE_FLOOR,
        block_m,
        block_n,
        block_k,
        num_warps=warps,
        num_stages=stages,
    )
    scale_width = block_n
    block_m, block_n, block_k, warps, stages = TILE_DX_FP8
    if scale_width % block_k:
        raise ValueError("the input-gradient reduction block must divide the scale tile")
    dx_assignments = torch.empty(
        (assignments.numel(), dim), dtype=torch.bfloat16, device=x.device
    )
    _grouped_mm_tilescaled_fp8[
        (triton.cdiv(tokens, block_m), triton.cdiv(dim, block_n), experts)
    ](
        dgate_up_fp8,
        dgate_up_scale,
        gate_transposed,
        gate_transposed_scale,
        offsets,
        dx_assignments,
        width,
        dim,
        scale_tiles,
        scale_width,
        block_m,
        block_n,
        block_k,
        num_warps=warps,
        num_stages=stages,
    )
    gate_tile, down_tile = _weight_gradient_tiles(x.device)
    if any(down_required):
        block_m, block_n, block_k, warps, stages = down_tile
        _grouped_dw[(triton.cdiv(dim, block_m), triton.cdiv(width, block_n), experts)](
            activated,
            gradient,
            assignments,
            offsets,
            tuple(down_sinks),
            width,
            dim,
            selected,
            experts,
            False,
            tuple(down_required),
            block_m,
            block_n,
            block_k,
            num_warps=warps,
            num_stages=stages,
        )
    if any(gate_required):
        block_m, block_n, block_k, warps, stages = gate_tile
        _grouped_dw[
            (triton.cdiv(2 * width, block_m), triton.cdiv(dim, block_n), experts)
        ](
            x,
            dgate_up,
            assignments,
            offsets,
            tuple(gate_sinks),
            dim,
            2 * width,
            selected,
            experts,
            True,
            tuple(gate_required),
            block_m,
            block_n,
            block_k,
            num_warps=warps,
            num_stages=stages,
        )
    dx = torch.empty_like(x)
    _combine_dx[(triton.cdiv(dx.numel(), 256),)](
        dx_assignments,
        inverse,
        dx,
        dx.numel(),
        dim,
        selected,
        256,
    )
    return dx, fresh


@_backward_fp8.register_fake
def _backward_fp8_fake(
    gradient: Tensor,
    gradient_fp8: Tensor,
    gradient_scale: Tensor,
    x: Tensor,
    gate_up: Tensor,
    gate_transposed: Tensor,
    gate_transposed_scale: Tensor,
    down_transposed: Tensor,
    down_transposed_scale: Tensor,
    assignments: Tensor,
    inverse: Tensor,
    offsets: Tensor,
    gate_sinks: list[Tensor],
    down_sinks: list[Tensor],
    gate_bound: list[bool],
    down_bound: list[bool],
    gate_required: list[bool],
    down_required: list[bool],
) -> tuple[Tensor, list[Tensor]]:
    gradients = [
        torch.empty(
            (gate_transposed.shape[2], gate_transposed.shape[1]),
            dtype=torch.float32, device=x.device,
        )
        for index, bound in enumerate(gate_bound)
        if not bound and gate_required[index]
    ]
    gradients.extend(
        torch.empty(
            (down_transposed.shape[2], down_transposed.shape[1]),
            dtype=torch.float32, device=x.device,
        )
        for index, bound in enumerate(down_bound)
        if not bound and down_required[index]
    )
    return torch.empty_like(x), gradients


class _SparseExperts(torch.autograd.Function):
    """The routed bank under the BF16 recipe: BF16 working operands."""

    @staticmethod
    def forward(
        ctx,
        x,
        assignments,
        inverse,
        offsets,
        gate_operand,
        down_operand,
        gate_sinks,
        down_sinks,
        *parameters,
    ):
        output, gate_up = _forward(x, gate_operand, down_operand, assignments, offsets)
        ctx.save_for_backward(
            x, gate_up, gate_operand, down_operand, assignments, inverse, offsets
        )
        ctx.gate_sinks = gate_sinks
        ctx.down_sinks = down_sinks
        return output

    @staticmethod
    def backward(ctx, gradient):
        x, gate_up, gate_operand, down_operand, assignments, inverse, offsets = (
            ctx.saved_tensors
        )
        experts = gate_operand.shape[0]
        required, bound, sinks = _sink_plan(ctx, experts)
        dx, fresh = _backward(
            gradient,
            x,
            gate_up,
            gate_operand,
            down_operand,
            assignments,
            inverse,
            offsets,
            *sinks,
            *bound,
            *required,
        )
        return (dx, None, None, None, None, None, None, None, *_spread(fresh, required, bound))


class _SparseExpertsFp8(torch.autograd.Function):
    """The routed bank under the FP8 recipe.

    The block input arrives quantized per token beside its BF16 self, which
    the gate/up weight gradient still reads; the incoming gradient is
    quantized per row in the backward. The bank's FP8 copies supply the
    weights: the direct copies to the forward, the transposed copies to the
    activation gradients.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        x_fp8,
        x_scale,
        assignments,
        inverse,
        offsets,
        gate_weight,
        gate_scale,
        gate_transposed,
        gate_transposed_scale,
        down_weight,
        down_scale,
        down_transposed,
        down_transposed_scale,
        gate_sinks,
        down_sinks,
        *parameters,
    ):
        output, gate_up = _forward_fp8(
            x_fp8, x_scale, gate_weight, gate_scale, down_weight, down_scale,
            assignments, offsets,
        )
        ctx.save_for_backward(
            x,
            gate_up,
            gate_transposed,
            gate_transposed_scale,
            down_transposed,
            down_transposed_scale,
            assignments,
            inverse,
            offsets,
        )
        ctx.gate_sinks = gate_sinks
        ctx.down_sinks = down_sinks
        return output

    @staticmethod
    def backward(ctx, gradient):
        (
            x,
            gate_up,
            gate_transposed,
            gate_transposed_scale,
            down_transposed,
            down_transposed_scale,
            assignments,
            inverse,
            offsets,
        ) = ctx.saved_tensors
        experts = gate_transposed.shape[0]
        required, bound, sinks = _sink_plan(ctx, experts, first_parameter=16)
        gradient_fp8, gradient_scale = quantize_rows(gradient)
        dx, fresh = _backward_fp8(
            gradient,
            gradient_fp8,
            gradient_scale,
            x,
            gate_up,
            gate_transposed,
            gate_transposed_scale,
            down_transposed,
            down_transposed_scale,
            assignments,
            inverse,
            offsets,
            *sinks,
            *bound,
            *required,
        )
        return (dx, *(None,) * 15, *_spread(fresh, required, bound))


def _sink_plan(ctx, experts: int, *, first_parameter: int = 8):
    """Which expert matrices need a gradient this backward, which of those
    have a bound sink, and the bound sinks in order, for both banks."""
    gate_required = list(ctx.needs_input_grad[first_parameter : first_parameter + experts])
    down_required = list(
        ctx.needs_input_grad[first_parameter + experts : first_parameter + 2 * experts]
    )
    gate_bound = [
        sink is not None and needed
        for sink, needed in zip(ctx.gate_sinks, gate_required, strict=True)
    ]
    down_bound = [
        sink is not None and needed
        for sink, needed in zip(ctx.down_sinks, down_required, strict=True)
    ]
    gate_sinks = [
        sink for sink, bound in zip(ctx.gate_sinks, gate_bound, strict=True) if bound
    ]
    down_sinks = [
        sink for sink, bound in zip(ctx.down_sinks, down_bound, strict=True) if bound
    ]
    return (
        (gate_required, down_required),
        (gate_bound, down_bound),
        (gate_sinks, down_sinks),
    )


def _spread(fresh, required, bound):
    """The autograd gradients of the expert parameters: a fresh tensor for
    each matrix that needed one without a bound sink, None elsewhere."""
    remaining = iter(fresh)
    return tuple(
        next(remaining) if needed and not is_bound else None
        for needed, is_bound in zip(
            required[0] + required[1], bound[0] + bound[1], strict=True
        )
    )


def sparse_experts(
    x: Tensor,
    assignments: Tensor,
    inverse: Tensor,
    offsets: Tensor,
    gate_parameters: tuple[Tensor, ...],
    down_parameters: tuple[Tensor, ...],
    gate_sinks: tuple[Tensor | None, ...],
    down_sinks: tuple[Tensor | None, ...],
    gate_shadow: Tensor | None,
    down_shadow: Tensor | None,
    gate_fp8=None,
    down_fp8=None,
) -> Tensor:
    """Execute only selected expert rows and accumulate selected dW in FP32.

    ``assignments`` lists ``(token, slot)`` pairs in expert-major dispatch order
    and ``inverse`` is its inverse permutation, the dispatch row of each pair;
    the backward sums each token's slot rows through ``inverse``.

    ``gate_fp8`` and ``down_fp8`` are the bank's FP8 copies over the routed
    experts. A BF16 training forward with both bound runs on FP8; forwards
    without gradients (evaluation) and every other dtype read the shadows.
    """
    if triton is None:
        raise RuntimeError("CUDA MoE requires Triton from the declared CUDA runtime")
    if (
        gate_fp8 is not None
        and down_fp8 is not None
        and x.dtype == torch.bfloat16
        and torch.is_grad_enabled()
    ):
        x_fp8, x_scale = quantize_rows(x)
        return _SparseExpertsFp8.apply(
            x,
            x_fp8,
            x_scale,
            assignments,
            inverse,
            offsets,
            gate_fp8.weight,
            gate_fp8.scale,
            gate_fp8.transposed,
            gate_fp8.transposed_scale,
            down_fp8.weight,
            down_fp8.scale,
            down_fp8.transposed,
            down_fp8.transposed_scale,
            gate_sinks,
            down_sinks,
            *gate_parameters,
            *down_parameters,
        )
    gate_operand = gate_shadow
    down_operand = down_shadow
    if gate_operand is None or gate_operand.dtype != x.dtype:
        gate_operand = torch.stack(
            [parameter.detach().to(x.dtype) for parameter in gate_parameters]
        )
    if down_operand is None or down_operand.dtype != x.dtype:
        down_operand = torch.stack(
            [parameter.detach().to(x.dtype) for parameter in down_parameters]
        )
    return _SparseExperts.apply(
        x,
        assignments,
        inverse,
        offsets,
        gate_operand,
        down_operand,
        gate_sinks,
        down_sinks,
        *gate_parameters,
        *down_parameters,
    )

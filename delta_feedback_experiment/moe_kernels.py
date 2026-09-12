"""Dropless, fixed-shape grouped CUDA GEMMs for the routed expert bank.

The launch envelope covers every possible expert load, while device-side
offsets skip empty tiles. Storage is exactly three assignments per token;
there is no capacity factor, CPU routing decision, or dense all-expert FFN.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except (ImportError, OSError):
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _grouped_mm(
        x,
        weight,
        assignments,
        offsets,
        output,
        N: tl.constexpr,
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
                source_rows = tl.load(assignments + rows, rows < end, other=0) // 3
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
    def _swiglu_backward(
        grad, gate_up, output, COUNT: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row, column = index // H, index % H
        gate = tl.load(gate_up + row * (2 * H) + column, index < COUNT, other=0).to(
            tl.float32
        )
        up = tl.load(gate_up + row * (2 * H) + H + column, index < COUNT, other=0).to(
            tl.float32
        )
        gradient = tl.load(grad + index, index < COUNT, other=0).to(tl.float32)
        sigmoid = tl.sigmoid(gate)
        tl.store(
            output + row * (2 * H) + column,
            gradient * up * sigmoid * (1 + gate * (1 - sigmoid)),
            index < COUNT,
        )
        tl.store(
            output + row * (2 * H) + H + column,
            gradient * gate * sigmoid,
            index < COUNT,
        )

    @triton.jit
    def _grouped_dw(
        x,
        grad,
        assignments,
        offsets,
        sinks,
        K: tl.constexpr,
        O: tl.constexpr,
        GATHER: tl.constexpr,
        ACTIVE: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        expert = tl.program_id(2)
        if (ACTIVE >> expert) & 1:
            sink = sinks[0]
            for index in tl.static_range(1, 15):
                if expert == index:
                    sink = sinks[index]
            start = tl.load(offsets + expert)
            end = tl.load(offsets + expert + 1)
            rows = tl.program_id(0) * BM + tl.arange(0, BM)
            columns = tl.program_id(1) * BN + tl.arange(0, BN)
            inner = tl.arange(0, BK)
            acc = tl.zeros((BM, BN), tl.float32)
            for block in range(tl.cdiv(end - start, BK)):
                tokens = start + block * BK + inner
                source_rows = tokens
                if GATHER:
                    source_rows = (
                        tl.load(assignments + tokens, tokens < end, other=0) // 3
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
    def _invert_assignments(
        assignments,
        inverse,
        COUNT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        assignment = tl.load(assignments + index, index < COUNT, other=0)
        tl.store(inverse + assignment, index, index < COUNT)

    @triton.jit
    def _combine_dx(
        dx,
        inverse,
        output,
        COUNT: tl.constexpr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        token, column = index // D, index % D
        value = tl.full((BLOCK,), 0, tl.float32)
        for slot in tl.static_range(3):
            row = tl.load(inverse + token * 3 + slot, index < COUNT, other=0)
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
    *,
    gather: bool = False,
    transpose: bool = False,
) -> Tensor:
    result = x.new_empty((3 * tokens, output_width))
    _grouped_mm[(triton.cdiv(tokens, 32), triton.cdiv(output_width, 64), 15)](
        x,
        weight,
        assignments,
        offsets,
        result,
        tokens,
        x.shape[-1],
        output_width,
        gather,
        transpose,
        32,
        64,
        32,
        num_warps=4,
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
) -> tuple[Tensor, Tensor, Tensor]:
    tokens, dim = x.shape
    width = down_weight.shape[-1]
    gate_up = _mm(x, gate_weight, assignments, offsets, tokens, 2 * width, gather=True)
    activated = x.new_empty((3 * tokens, width))
    _swiglu[(triton.cdiv(activated.numel(), 256),)](
        gate_up,
        activated,
        activated.numel(),
        width,
        256,
    )
    output = _mm(activated, down_weight, assignments, offsets, tokens, dim)
    return output, gate_up, activated


@_forward.register_fake
def _forward_fake(
    x: Tensor,
    gate_weight: Tensor,
    down_weight: Tensor,
    assignments: Tensor,
    offsets: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    tokens, dim = x.shape
    width = down_weight.shape[-1]
    return (
        x.new_empty((3 * tokens, dim)),
        x.new_empty((3 * tokens, 2 * width)),
        x.new_empty((3 * tokens, width)),
    )


@torch.library.custom_op(
    "delta_feedback::moe_backward", mutates_args=(), device_types="cuda"
)
def _backward(
    gradient: Tensor,
    x: Tensor,
    gate_up: Tensor,
    activated: Tensor,
    gate_weight: Tensor,
    down_weight: Tensor,
    assignments: Tensor,
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
    width = activated.shape[-1]
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
    dactivated = _mm(
        gradient, down_weight, assignments, offsets, tokens, width, transpose=True
    )
    dgate_up = torch.empty_like(gate_up)
    _swiglu_backward[(triton.cdiv(activated.numel(), 256),)](
        dactivated,
        gate_up,
        dgate_up,
        activated.numel(),
        width,
        256,
    )
    dx_assignments = _mm(
        dgate_up, gate_weight, assignments, offsets, tokens, dim, transpose=True
    )
    if any(down_required):
        _grouped_dw[(triton.cdiv(dim, 32), triton.cdiv(width, 64), 15)](
            activated,
            gradient,
            assignments,
            offsets,
            tuple(down_sinks),
            width,
            dim,
            False,
            sum(int(required) << index for index, required in enumerate(down_required)),
            32,
            64,
            32,
            num_warps=4,
        )
    if any(gate_required):
        _grouped_dw[(triton.cdiv(2 * width, 32), triton.cdiv(dim, 64), 15)](
            x,
            dgate_up,
            assignments,
            offsets,
            tuple(gate_sinks),
            dim,
            2 * width,
            True,
            sum(int(required) << index for index, required in enumerate(gate_required)),
            32,
            64,
            32,
            num_warps=4,
        )
    inverse = torch.empty_like(assignments)
    _invert_assignments[(triton.cdiv(assignments.numel(), 256),)](
        assignments,
        inverse,
        assignments.numel(),
        256,
    )
    dx = torch.empty_like(x)
    _combine_dx[(triton.cdiv(dx.numel(), 256),)](
        dx_assignments,
        inverse,
        dx,
        dx.numel(),
        dim,
        256,
    )
    return dx, fresh


@_backward.register_fake
def _backward_fake(
    gradient: Tensor,
    x: Tensor,
    gate_up: Tensor,
    activated: Tensor,
    gate_weight: Tensor,
    down_weight: Tensor,
    assignments: Tensor,
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


class _SparseExperts(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        assignments,
        offsets,
        gate_operand,
        down_operand,
        gate_sinks,
        down_sinks,
        *parameters,
    ):
        output, gate_up, activated = _forward(
            x, gate_operand, down_operand, assignments, offsets
        )
        ctx.save_for_backward(
            x, gate_up, activated, gate_operand, down_operand, assignments, offsets
        )
        ctx.gate_sinks = gate_sinks
        ctx.down_sinks = down_sinks
        return output

    @staticmethod
    def backward(ctx, gradient):
        x, gate_up, activated, gate_operand, down_operand, assignments, offsets = (
            ctx.saved_tensors
        )
        gate_required = list(ctx.needs_input_grad[7:22])
        down_required = list(ctx.needs_input_grad[22:37])
        gate_bound = [
            sink is not None and needed
            for sink, needed in zip(ctx.gate_sinks, gate_required, strict=True)
        ]
        down_bound = [
            sink is not None and needed
            for sink, needed in zip(ctx.down_sinks, down_required, strict=True)
        ]
        dx, fresh = _backward(
            gradient,
            x,
            gate_up,
            activated,
            gate_operand,
            down_operand,
            assignments,
            offsets,
            [
                sink
                for sink, bound in zip(ctx.gate_sinks, gate_bound, strict=True)
                if bound
            ],
            [
                sink
                for sink, bound in zip(ctx.down_sinks, down_bound, strict=True)
                if bound
            ],
            gate_bound,
            down_bound,
            gate_required,
            down_required,
        )
        remaining = iter(fresh)
        gradients = tuple(
            next(remaining) if needed and not bound else None
            for needed, bound in zip(
                gate_required + down_required, gate_bound + down_bound, strict=True
            )
        )
        return (dx, None, None, None, None, None, None, *gradients)


def sparse_experts(
    x: Tensor,
    assignments: Tensor,
    offsets: Tensor,
    gate_parameters: tuple[Tensor, ...],
    down_parameters: tuple[Tensor, ...],
    gate_sinks: tuple[Tensor | None, ...],
    down_sinks: tuple[Tensor | None, ...],
    gate_shadow: Tensor | None,
    down_shadow: Tensor | None,
) -> Tensor:
    """Execute only selected expert rows and accumulate selected dW in FP32."""
    if triton is None:
        raise RuntimeError("CUDA MoE requires Triton from the declared CUDA runtime")
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
        offsets,
        gate_operand,
        down_operand,
        gate_sinks,
        down_sinks,
        *gate_parameters,
        *down_parameters,
    )

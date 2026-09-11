"""The FLA kernels a PKDA layer calls, as operators Dynamo can keep in a graph.

The three dense PKDA kernels are custom operators with exact fake
implementations, so Dynamo can compile the whole block as one graph.

Each wrapper covers exactly the dense training and prefill configuration: no
cache state, no final state, no variable-length batching, the gate computed
inside the kernel, and Q/K normalized upstream by the convolution.  Cached
decoding keeps calling FLA directly.  Every forward asserts the metadata its
fake implementation promises, so a change on the kernel side fails loudly here
instead of miscompiling.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:  # Pinned CUDA-only dependency, exactly as in :mod:`pkda`.
    from fla.modules.conv.triton import causal_conv1d_bwd, causal_conv1d_fwd
    from fla.modules.fused_norm_gate import layer_norm_gated_bwd, layer_norm_gated_fwd
    from fla.ops.kda.gate import kda_gate_bwd, kda_gate_chunk_cumsum
    from fla.ops.precond_kda.chunk import chunk_precond_kda_bwd, chunk_precond_kda_fwd
    from fla.ops.utils.constant import RCP_LN2
except (ImportError, OSError):  # pragma: no cover - exercised on Jobe
    causal_conv1d_bwd = causal_conv1d_fwd = None
    layer_norm_gated_bwd = layer_norm_gated_fwd = None
    kda_gate_bwd = kda_gate_chunk_cumsum = None
    chunk_precond_kda_bwd = chunk_precond_kda_fwd = None
    RCP_LN2 = None

CHUNK_SIZE = 64
"""The chunk length FLA's preconditioned KDA forward and backward share."""

PKDA_INTERMEDIATES = (
    "Aqk",
    "Akk",
    "w",
    "kg",
    "v_new",
    "h",
    "k_precond",
    "ac_atk",
    "a_atk",
    "sa_atk",
    "qg",
    "g_cum",
)
"""What the recurrence hands its own backward, in the order it is carried.

The fork's forward stores the gated query ``qg`` from its output kernel and
keeps ``g_cum``, the FP32 chunk-local gate cumsum, so the backward relaunches
neither; ``u`` is never read by a backward that keeps these and is dropped.
"""


def fla_ops_available() -> bool:
    return chunk_precond_kda_fwd is not None


def _intermediate_meta(
    q: Tensor, v: Tensor
) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    """Shape and dtype of every saved intermediate, from the input geometry."""
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    chunks = -(-length // CHUNK_SIZE)
    activation = q.dtype
    return (
        ((batch, length, heads, CHUNK_SIZE), activation),  # Aqk
        ((batch, length, heads, CHUNK_SIZE), activation),  # Akk
        ((batch, length, heads, key_dim), activation),  # w
        ((batch, length, heads, key_dim), activation),  # kg
        ((batch, length, heads, value_dim), activation),  # v_new
        ((batch, chunks, heads, key_dim, value_dim), activation),  # h
        ((batch, length, heads, key_dim), activation),  # k_precond
        ((batch, chunks, heads, key_dim), torch.float32),  # ac_atk
        ((batch, chunks, heads, key_dim), torch.float32),  # a_atk
        ((batch, chunks, heads), torch.float32),  # sa_atk
        ((batch, length, heads, key_dim), activation),  # qg
        ((batch, length, heads, key_dim), torch.float32),  # g_cum
    )


def _check_intermediates(tensors: list[Tensor], q: Tensor, v: Tensor) -> None:
    expected = _intermediate_meta(q, v)
    if len(tensors) != len(expected):
        raise RuntimeError(
            f"FLA's preconditioned KDA forward returned {len(tensors)} "
            f"intermediates, not {len(expected)}"
        )
    for name, tensor, (shape, dtype) in zip(
        PKDA_INTERMEDIATES, tensors, expected, strict=True
    ):
        ok = (
            tensor is not None
            and tuple(tensor.shape) == shape
            and tensor.dtype == dtype
            and tensor.device == q.device
            and tensor.is_contiguous()
        )
        if not ok:
            actual = (
                None
                if tensor is None
                else (tuple(tensor.shape), tensor.dtype, tensor.device,
                      tensor.is_contiguous())
            )
            raise RuntimeError(
                f"FLA's preconditioned KDA intermediate {name} is {actual}, not "
                f"{(shape, dtype, q.device, True)}; the fake implementation is stale"
            )


# -- the three preconditioned-KDA kernels --------------------------------------


@torch.library.custom_op(
    "delta_feedback::pkda_qkv_conv", mutates_args=(), device_types="cuda"
)
def pkda_qkv_conv(
    x: Tensor, weight: Tensor, head_dim: int, norm_channels: int, eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    """One SiLU short convolution over concatenated Q/K/V with the per-head L2
    normalization fused in, writing the three projections as separate slabs.

    ``x`` is made contiguous here and again in the backward, which is what the
    layer already hands over and what makes the kernel's standard-layout path
    exact on both sides. FLA's own wrapper instead records the layout in its
    context; the wrapper cannot, because it saves the operator's inputs.
    """
    x = x.contiguous()
    width = weight.shape[0] // 3
    outputs, final_state = causal_conv1d_fwd(
        x=x,
        weight=weight.contiguous(),
        bias=None,
        residual=None,
        initial_state=None,
        output_final_state=False,
        activation="silu",
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        chunk_indices=None,
        BT=CHUNK_SIZE,
        layout_fallback=False,
        l2norm_head_dim=head_dim,
        l2norm_channels=norm_channels,
        l2norm_eps=eps,
        split_outputs=(width, width, width),
    )
    if final_state is not None or len(outputs) != 3:
        raise RuntimeError("the fused QKV convolution changed its return shape")
    return outputs[0], outputs[1], outputs[2]


@pkda_qkv_conv.register_fake
def _pkda_qkv_conv_fake(
    x: Tensor, weight: Tensor, head_dim: int, norm_channels: int, eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    del head_dim, norm_channels, eps
    batch, length, _ = x.shape
    width = weight.shape[0] // 3
    return tuple(x.new_empty(batch, length, width) for _ in range(3))


@torch.library.custom_op(
    "delta_feedback::pkda_qkv_conv_backward", mutates_args=(), device_types="cuda"
)
def _pkda_qkv_conv_backward(
    x: Tensor,
    weight: Tensor,
    dq: Tensor,
    dk: Tensor,
    dv: Tensor,
    head_dim: int,
    norm_channels: int,
    eps: float,
) -> tuple[Tensor, Tensor]:
    width = weight.shape[0] // 3
    dx, dw, _db, _dr, _dh0 = causal_conv1d_bwd(
        x=x.contiguous(),
        dy=[dq.contiguous(), dk.contiguous(), dv.contiguous()],
        dht=None,
        weight=weight.contiguous(),
        bias=None,
        residual=None,
        initial_state=None,
        activation="silu",
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        chunk_indices=None,
        BT=CHUNK_SIZE,
        layout_fallback=False,
        l2norm_head_dim=head_dim,
        l2norm_channels=norm_channels,
        l2norm_eps=eps,
        split_outputs=(width, width, width),
    )
    return dx, dw


@_pkda_qkv_conv_backward.register_fake
def _pkda_qkv_conv_backward_fake(
    x: Tensor,
    weight: Tensor,
    dq: Tensor,
    dk: Tensor,
    dv: Tensor,
    head_dim: int,
    norm_channels: int,
    eps: float,
) -> tuple[Tensor, Tensor]:
    del dq, dk, dv, head_dim, norm_channels, eps
    return torch.empty_like(x), torch.empty_like(weight)


def _conv_setup_context(ctx, inputs, output) -> None:
    x, weight, head_dim, norm_channels, eps = inputs
    ctx.save_for_backward(x, weight)
    ctx.geometry = (head_dim, norm_channels, eps)


def _conv_backward(ctx, dq, dk, dv):
    x, weight = ctx.saved_tensors
    head_dim, norm_channels, eps = ctx.geometry
    dx, dw = _pkda_qkv_conv_backward(
        x, weight, dq, dk, dv, head_dim, norm_channels, eps
    )
    return dx, dw, None, None, None


pkda_qkv_conv.register_autograd(_conv_backward, setup_context=_conv_setup_context)


@torch.library.custom_op(
    "delta_feedback::pkda_recurrence", mutates_args=(), device_types="cuda"
)
def pkda_recurrence(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    g_atk: Tensor,
    beta_atk: Tensor,
    beta: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    log_atk_scale: Tensor,
    scale: float,
    squash_x: float,
    squash_eps: float,
) -> tuple[Tensor, list[Tensor]]:
    """The chunked preconditioned-KDA recurrence and what its backward reuses.

    ``g`` is the raw decay projection: the log-space gate and its chunk-local
    cumulative sum happen inside the kernel, exactly as ``use_gate_in_kernel``
    does, and the backward recomputes them the same way.  The returned
    intermediates are the ones FLA's own ``disable_recompute`` path keeps.

    Every tensor input is made contiguous here and again in the backward, which
    is what FLA's ``input_guard`` does for its own entry point and what keeps
    the two sides reading one layout.
    """
    q, k, v, g = q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous()
    g_atk, beta_atk, beta = (
        g_atk.contiguous(),
        beta_atk.contiguous(),
        beta.contiguous(),
    )
    cumulative = kda_gate_chunk_cumsum(
        g=g,
        A_log=A_log,
        chunk_size=CHUNK_SIZE,
        scale=RCP_LN2,
        dt_bias=dt_bias,
        cu_seqlens=None,
        chunk_indices=None,
        lower_bound=None,
    )
    (
        o,
        Aqk,
        Akk,
        final_state,
        at,
        w,
        u,
        kg,
        v_new,
        h,
        k_precond,
        ac_atk,
        a_atk,
        sa_atk,
        qg,
    ) = chunk_precond_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=cumulative,
        g_atk=g_atk,
        beta_atk=beta_atk,
        beta=beta,
        scale=scale,
        initial_state=None,
        initial_A_state=None,
        output_final_state=False,
        chunk_size=CHUNK_SIZE,
        cu_seqlens=None,
        cp_context=None,
        chunk_indices=None,
        solve_tril_precision=None,
        safe_gate=False,
        x=squash_x,
        eps=squash_eps,
        log_atk_scale=log_atk_scale,
        transpose_state_layout=False,
        output_qg=True,
    )
    if final_state is not None or at is not None:
        raise RuntimeError("the stateless PKDA recurrence returned a final state")
    del u
    saved = [Aqk, Akk, w, kg, v_new, h, k_precond, ac_atk, a_atk, sa_atk, qg, cumulative]
    _check_intermediates(saved, q, v)
    return o.to(q.dtype), saved


@pkda_recurrence.register_fake
def _pkda_recurrence_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    g_atk: Tensor,
    beta_atk: Tensor,
    beta: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    log_atk_scale: Tensor,
    scale: float,
    squash_x: float,
    squash_eps: float,
) -> tuple[Tensor, list[Tensor]]:
    del k, g, g_atk, beta_atk, beta, A_log, dt_bias, log_atk_scale
    del scale, squash_x, squash_eps
    output = torch.empty(
        (*q.shape[:3], v.shape[-1]), device=q.device, dtype=q.dtype
    )
    saved = [
        torch.empty(shape, device=q.device, dtype=dtype)
        for shape, dtype in _intermediate_meta(q, v)
    ]
    return output, saved


@torch.library.custom_op(
    "delta_feedback::pkda_recurrence_backward", mutates_args=(), device_types="cuda"
)
def _pkda_recurrence_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    g_atk: Tensor,
    beta_atk: Tensor,
    beta: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    log_atk_scale: Tensor,
    saved: list[Tensor],
    do: Tensor,
    scale: float,
    squash_x: float,
    squash_eps: float,
    autocast_dtype: torch.dtype | None,
) -> list[Tensor]:
    """FLA's preconditioned-KDA backward over the saved intermediates.

    ``autocast_dtype`` reproduces FLA's ``custom_bwd``: its backward runs under
    the autocast state its forward saw, which the surrounding training loop
    leaves disabled by the time ``backward`` is called.
    """
    Aqk, Akk, w, kg, v_new, h, k_precond, ac_atk, a_atk, sa_atk, qg, cumulative = saved
    q, k, v, g = q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous()
    g_atk, beta_atk, beta = (
        g_atk.contiguous(),
        beta_atk.contiguous(),
        beta.contiguous(),
    )
    with torch.autocast(
        "cuda",
        enabled=autocast_dtype is not None,
        dtype=autocast_dtype or torch.bfloat16,
    ):
        # The gate backward absorbs the chunk-local reverse cumsum whenever its
        # row tiles cannot straddle a chunk boundary, exactly as FLA decides it.
        defer_dg_cumsum = q.shape[1] % CHUNK_SIZE == 0
        dq, dk, dv, dg, dg_atk, dbeta_atk, dbeta, d_log_atk_scale, _, _ = (
            chunk_precond_kda_bwd(
                q=q,
                k=k,
                v=v,
                g=cumulative,
                g_atk=g_atk,
                beta_atk=beta_atk,
                beta=beta,
                Aqk=Aqk,
                Akk=Akk,
                scale=scale,
                initial_state=None,
                initial_A_state=None,
                do=do.contiguous(),
                dht=None,
                cu_seqlens=None,
                cp_context=None,
                chunk_indices=None,
                chunk_size=CHUNK_SIZE,
                x=squash_x,
                eps=squash_eps,
                log_atk_scale=log_atk_scale,
                transpose_state_layout=False,
                safe_gate=False,
                disable_recompute=True,
                defer_dg_cumsum=defer_dg_cumsum,
                w=w,
                kg=kg,
                v_new=v_new,
                h=h,
                dat=None,
                k_precond=k_precond,
                ac_atk=ac_atk,
                a_atk=a_atk,
                sa_atk=sa_atk,
                qg=qg,
            )
        )
        dg, dA_log, ddt_bias = kda_gate_bwd(
            g=g,
            A_log=A_log,
            dt_bias=dt_bias,
            dyg=dg,
            lower_bound=None,
            reverse_cumsum_chunk_size=CHUNK_SIZE if defer_dg_cumsum else None,
        )
    return [
        dq.to(q),
        dk.to(k),
        dv.to(v),
        dg.to(g),
        dg_atk.to(g_atk),
        dbeta_atk.to(beta_atk),
        dbeta.to(beta),
        dA_log.to(A_log),
        ddt_bias.to(dt_bias),
        d_log_atk_scale.to(log_atk_scale),
    ]


@_pkda_recurrence_backward.register_fake
def _pkda_recurrence_backward_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    g_atk: Tensor,
    beta_atk: Tensor,
    beta: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    log_atk_scale: Tensor,
    saved: list[Tensor],
    do: Tensor,
    scale: float,
    squash_x: float,
    squash_eps: float,
    autocast_dtype: torch.dtype | None,
) -> list[Tensor]:
    del saved, do, scale, squash_x, squash_eps, autocast_dtype
    return [
        torch.empty_like(tensor)
        for tensor in (
            q,
            k,
            v,
            g,
            g_atk,
            beta_atk,
            beta,
            A_log,
            dt_bias,
            log_atk_scale,
        )
    ]


def _recurrence_setup_context(ctx, inputs, output) -> None:
    q, k, v, g, g_atk, beta_atk, beta, A_log, dt_bias, log_atk_scale, *scalars = (
        inputs
    )
    _o, saved = output
    ctx.save_for_backward(
        q, k, v, g, g_atk, beta_atk, beta, A_log, dt_bias, log_atk_scale, *saved
    )
    ctx.scalars = tuple(scalars)
    ctx.autocast_dtype = (
        torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else None
    )
    ctx.mark_non_differentiable(*saved)


def _recurrence_backward(ctx, do, _dsaved):
    inputs = ctx.saved_tensors[:10]
    saved = list(ctx.saved_tensors[10:])
    scale, squash_x, squash_eps = ctx.scalars
    gradients = _pkda_recurrence_backward(
        *inputs, saved, do, scale, squash_x, squash_eps, ctx.autocast_dtype
    )
    return (*gradients, None, None, None)


pkda_recurrence.register_autograd(
    _recurrence_backward, setup_context=_recurrence_setup_context
)


@torch.library.custom_op(
    "delta_feedback::pkda_norm_gate", mutates_args=(), device_types="cuda"
)
def pkda_norm_gate(
    x: Tensor, gate: Tensor, weight: Tensor, eps: float
) -> tuple[Tensor, Tensor]:
    """FLA's fused RMSNorm with a sigmoid output gate; ``rstd`` feeds backward."""
    flat = x.contiguous().reshape(-1, x.shape[-1])
    y, mean, rstd, _residual = layer_norm_gated_fwd(
        x=flat,
        g=gate.contiguous().reshape(-1, gate.shape[-1]),
        weight=weight,
        bias=None,
        activation="sigmoid",
        eps=eps,
        residual=None,
        residual_dtype=None,
        is_rms_norm=True,
    )
    if mean is not None:
        raise RuntimeError("the RMS-normed gate returned a mean")
    return y.reshape(x.shape), rstd


@pkda_norm_gate.register_fake
def _pkda_norm_gate_fake(
    x: Tensor, gate: Tensor, weight: Tensor, eps: float
) -> tuple[Tensor, Tensor]:
    del gate, weight, eps
    rows = x.numel() // x.shape[-1]
    return (
        torch.empty_like(x),
        torch.empty((rows,), device=x.device, dtype=torch.float32),
    )


@torch.library.custom_op(
    "delta_feedback::pkda_norm_gate_backward", mutates_args=(), device_types="cuda"
)
def _pkda_norm_gate_backward(
    x: Tensor, gate: Tensor, weight: Tensor, rstd: Tensor, dy: Tensor, eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    shape = x.shape
    dx, dg, dw, _db, _dres = layer_norm_gated_bwd(
        dy=dy.contiguous().reshape(-1, shape[-1]),
        x=x.contiguous().reshape(-1, shape[-1]),
        g=gate.contiguous().reshape(-1, shape[-1]),
        weight=weight,
        bias=None,
        activation="sigmoid",
        eps=eps,
        mean=None,
        rstd=rstd,
        dresidual=None,
        has_residual=False,
        is_rms_norm=True,
        x_dtype=x.dtype,
    )
    return dx.reshape(shape), dg.reshape(shape), dw


@_pkda_norm_gate_backward.register_fake
def _pkda_norm_gate_backward_fake(
    x: Tensor, gate: Tensor, weight: Tensor, rstd: Tensor, dy: Tensor, eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    del rstd, dy, eps
    return torch.empty_like(x), torch.empty_like(gate), torch.empty_like(weight)


def _norm_gate_setup_context(ctx, inputs, output) -> None:
    x, gate, weight, eps = inputs
    _y, rstd = output
    ctx.save_for_backward(x, gate, weight, rstd)
    ctx.eps = eps
    ctx.mark_non_differentiable(rstd)


def _norm_gate_backward(ctx, dy, _drstd):
    x, gate, weight, rstd = ctx.saved_tensors
    dx, dg, dw = _pkda_norm_gate_backward(x, gate, weight, rstd, dy, ctx.eps)
    return dx, dg, dw, None


pkda_norm_gate.register_autograd(
    _norm_gate_backward, setup_context=_norm_gate_setup_context
)


__all__ = [
    "CHUNK_SIZE",
    "PKDA_INTERMEDIATES",
    "fla_ops_available",
    "pkda_norm_gate",
    "pkda_qkv_conv",
    "pkda_recurrence",
]

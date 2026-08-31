"""Preconditioned KDA with a portable recurrence and the upstream CUDA kernel.

The parameterization follows FLA's ``PrecondKDA`` layer, while the module
boundary stays native to this project so initialization, optimizer partition,
cache ownership, and activation checkpointing remain explicit. CPU and MPS use
the literal recurrent equations. CUDA requires FLA's chunk/fused-recurrent
operators; it never substitutes the quadratic Python recurrence for training.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:  # Pinned CUDA-only dependency; portable tests use the recurrence below.
    from fla.modules.fused_norm_gate import rms_norm_gated
    from fla.ops.precond_kda import (
        chunk_precond_kda,
        fused_recurrent_precond_kda,
    )
except (ImportError, OSError):  # pragma: no cover - exercised on Jobe
    rms_norm_gated = None
    chunk_precond_kda = None
    fused_recurrent_precond_kda = None


def pkda_cuda_available() -> bool:
    return (
        chunk_precond_kda is not None
        and fused_recurrent_precond_kda is not None
        and rms_norm_gated is not None
    )


class PreconditionedKDA(nn.Module):
    """KDA with the stable diagonal apply-to-key preconditioner.

    Inputs and outputs are ``[B,T,D]``. The recurrent matrix and diagonal
    preconditioner states are always accumulated in FP32; projected activations
    follow the surrounding autocast dtype. ``state``/``a_state`` and the three
    convolution histories are supplied only by autoregressive cache execution.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int = 8,
        head_dim: int = 128,
        conv_size: int = 4,
        norm_eps: float = 1e-5,
        squash_x: float = 1.5,
        squash_eps: float = 1e-6,
    ):
        super().__init__()
        if num_heads < 1 or head_dim < 1:
            raise ValueError("PKDA head count and dimension must be positive")
        if conv_size < 1:
            raise ValueError("PKDA convolution width must be positive")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.projection_size = num_heads * head_dim
        self.conv_size = conv_size
        self.norm_eps = norm_eps
        self.squash_x = squash_x
        self.squash_eps = squash_eps

        self.q_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.projection_size, bias=False)

        self.q_conv = nn.Conv1d(
            self.projection_size,
            self.projection_size,
            conv_size,
            groups=self.projection_size,
            bias=False,
        )
        self.k_conv = nn.Conv1d(
            self.projection_size,
            self.projection_size,
            conv_size,
            groups=self.projection_size,
            bias=False,
        )
        self.v_conv = nn.Conv1d(
            self.projection_size,
            self.projection_size,
            conv_size,
            groups=self.projection_size,
            bias=False,
        )

        # These five small hidden-width projections share one Adam-side GEMM.
        # Their non-overlapping row slices retain independent parameters and
        # exactly the unpacked parameter count.
        self.control_splits = (
            head_dim,
            num_heads,
            num_heads,
            num_heads,
            head_dim,
        )
        self.control_proj = nn.Linear(hidden_size, sum(self.control_splits), bias=False)

        # KDA's channel-wise decay is rank-head_dim, as in the released layer.
        self.decay_up = nn.Linear(head_dim, self.projection_size, bias=False)
        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(
            torch.zeros(self.projection_size, dtype=torch.float32)
        )

        # Independent scalar decay and gain for the diagonal preconditioner.
        self.A_log_precond = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
        self.dt_bias_precond = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
        self.log_precond_center = nn.Parameter(
            torch.full((num_heads,), -0.2, dtype=torch.float32)
        )

        self.output_gate_up = nn.Linear(head_dim, self.projection_size, bias=True)
        self.output_norm = nn.Parameter(torch.ones(head_dim))
        self.o_proj = nn.Linear(self.projection_size, hidden_size, bias=False)
        self.reset_recurrence_parameters()

    @torch.no_grad()
    def reset_recurrence_parameters(self) -> None:
        self.A_log.copy_(torch.empty_like(self.A_log).uniform_(1, 16).log())
        self.A_log_precond.copy_(
            torch.empty_like(self.A_log_precond).uniform_(1, 16).log()
        )
        dt = (
            torch.empty_like(self.dt_bias_precond)
            .uniform_(math.log(0.001), math.log(0.1))
            .exp()
        )
        self.dt_bias_precond.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _causal_conv(
        self,
        x: Tensor,
        conv: nn.Conv1d,
        state: Tensor | None,
        output_final_state: bool,
    ) -> tuple[Tensor, Tensor | None]:
        channels = x.shape[-1]
        history = self.conv_size - 1
        x_t = x.transpose(1, 2)
        if history:
            if state is None:
                state = x_t.new_zeros(x.shape[0], channels, history)
            combined = torch.cat((state, x_t), dim=-1)
        else:
            combined = x_t
        out = F.conv1d(
            combined,
            conv.weight,
            bias=None,
            groups=channels,
        ).transpose(1, 2)
        final = combined[..., -history:] if output_final_state and history else None
        return F.silu(out), final

    def _project(
        self,
        x: Tensor,
        conv_state: tuple[Tensor, Tensor, Tensor] | None,
        output_final_state: bool,
    ) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor, Tensor] | None]:
        states = conv_state or (None, None, None)
        q, q_state = self._causal_conv(
            self.q_proj(x), self.q_conv, states[0], output_final_state
        )
        k, k_state = self._causal_conv(
            self.k_proj(x), self.k_conv, states[1], output_final_state
        )
        v, v_state = self._causal_conv(
            self.v_proj(x), self.v_conv, states[2], output_final_state
        )
        shape = (*x.shape[:2], self.num_heads, self.head_dim)
        final = (q_state, k_state, v_state) if output_final_state else None
        return q.reshape(shape), k.reshape(shape), v.reshape(shape), final

    def _controls(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        (
            decay_hidden,
            beta_logits,
            precond_decay_logits,
            precond_beta_logits,
            output_gate_hidden,
        ) = self.control_proj(x).split(self.control_splits, dim=-1)
        raw_decay = self.decay_up(decay_hidden).reshape(
            *x.shape[:2], self.num_heads, self.head_dim
        )
        beta = torch.sigmoid(beta_logits)
        precond_decay = -self.A_log_precond.float().exp() * F.softplus(
            precond_decay_logits.float() + self.dt_bias_precond
        )
        precond_beta = torch.sigmoid(precond_beta_logits)
        return (
            raw_decay,
            beta,
            precond_decay,
            precond_beta,
            output_gate_hidden,
        )

    def _portable_recurrence(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        raw_decay: Tensor,
        beta: Tensor,
        precond_decay: Tensor,
        precond_beta: Tensor,
        state: Tensor | None,
        a_state: Tensor | None,
        output_final_state: bool,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        dtype = v.dtype
        # FLA's in-kernel l2norm is x / sqrt(sum(x^2) + 1e-6), followed by
        # the operator's 1/sqrt(K) query scale. Keep the portable recurrence
        # literal so the CUDA gate can compare both values and gradients.
        q = q.float()
        k = k.float()
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        q = q * self.head_dim**-0.5
        v = v.float()
        decay = -self.A_log.float().exp()[None, None, :, None] * F.softplus(
            raw_decay.float() + self.dt_bias.view(1, 1, self.num_heads, self.head_dim)
        )
        beta = beta.float()
        precond_decay = precond_decay.float()
        precond_beta = precond_beta.float()
        if state is None:
            state = q.new_zeros(
                q.shape[0], self.num_heads, self.head_dim, self.head_dim
            )
        if a_state is None:
            a_state = q.new_zeros(q.shape[0], self.num_heads, self.head_dim)
        outputs = []
        log_x = math.log(self.squash_x)
        center = self.log_precond_center.float()[None, :, None]
        for index in range(q.shape[1]):
            q_i, k_i, v_i = q[:, index], k[:, index], v[:, index]
            a_state = (
                precond_decay[:, index].exp().unsqueeze(-1) * a_state
                + precond_beta[:, index].unsqueeze(-1) * k_i.square()
            )
            deviation = torch.log(a_state + self.squash_eps) - center
            squashed = deviation / (1 + deviation.abs())
            write_key = k_i * torch.exp(-log_x * squashed)
            state = state * decay[:, index].exp().unsqueeze(-1)
            residual = v_i - torch.einsum("bhkv,bhk->bhv", state, k_i)
            state = state + torch.einsum(
                "bhk,bhv->bhkv",
                beta[:, index].unsqueeze(-1) * write_key,
                residual,
            )
            outputs.append(torch.einsum("bhk,bhkv->bhv", q_i, state))
        output = torch.stack(outputs, dim=1).to(dtype)
        if not output_final_state:
            return output, None, None
        return output, state, a_state

    def _cuda_recurrence(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        raw_decay: Tensor,
        beta: Tensor,
        precond_decay: Tensor,
        precond_beta: Tensor,
        state: Tensor | None,
        a_state: Tensor | None,
        output_final_state: bool,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if not pkda_cuda_available():
            raise RuntimeError(
                "CUDA PKDA requires the pinned flash-linear-attention build"
            )
        kwargs = {
            "q": q,
            "k": k,
            "v": v,
            "g": raw_decay,
            "g_atk": precond_decay,
            "beta_atk": precond_beta,
            "beta": beta,
            "A_log": self.A_log,
            "dt_bias": self.dt_bias,
            "initial_state": state,
            "initial_A_state": a_state,
            "output_final_state": output_final_state,
            "use_gate_in_kernel": True,
            "x": self.squash_x,
            "eps": self.squash_eps,
            "log_atk_scale": self.log_precond_center,
        }
        if output_final_state and q.shape[1] <= 64 and not self.training:
            return fused_recurrent_precond_kda(**kwargs)
        return chunk_precond_kda(**kwargs)

    def forward(
        self,
        x: Tensor,
        *,
        state: Tensor | None = None,
        a_state: Tensor | None = None,
        conv_state: tuple[Tensor, Tensor, Tensor] | None = None,
        output_final_state: bool = False,
    ) -> tuple[
        Tensor,
        Tensor | None,
        Tensor | None,
        tuple[Tensor, Tensor, Tensor] | None,
    ]:
        q, k, v, final_conv = self._project(x, conv_state, output_final_state)
        (
            raw_decay,
            beta,
            precond_decay,
            precond_beta,
            output_gate_hidden,
        ) = self._controls(x)
        if x.is_cuda:
            output, state, a_state = self._cuda_recurrence(
                q,
                k,
                v,
                raw_decay,
                beta,
                precond_decay,
                precond_beta,
                state,
                a_state,
                output_final_state,
            )
        else:
            output, state, a_state = self._portable_recurrence(
                q,
                k,
                v,
                raw_decay,
                beta,
                precond_decay,
                precond_beta,
                state,
                a_state,
                output_final_state,
            )
        gate_logits = self.output_gate_up(output_gate_hidden).reshape(
            *x.shape[:2], self.num_heads, self.head_dim
        )
        if output.is_cuda:
            if rms_norm_gated is None:
                raise RuntimeError(
                    "CUDA PKDA output requires FLA's fused RMSNorm-gate operator"
                )
            mixed = rms_norm_gated(
                output,
                gate_logits,
                self.output_norm,
                None,
                "sigmoid",
                eps=self.norm_eps,
            )
        else:
            normalized = output.float() * torch.rsqrt(
                output.float().square().mean(-1, keepdim=True) + self.norm_eps
            )
            normalized = normalized * self.output_norm.float()
            mixed = normalized.to(output.dtype) * torch.sigmoid(gate_logits)
        return (
            self.o_proj(mixed.flatten(-2)),
            state,
            a_state,
            final_conv,
        )

"""The delta model family: one trunk, one letter per change from the plain decoder.

The architecture contract is ``docs/architecture.md``. Everything here is
condition-agnostic model semantics: a condition is a string of letters from
:data:`CONDITION_LETTERS`, each switching on one :class:`ModelConfig` flag
over the plain twelve-layer RoPE GQA decoder (the empty condition). ``a``
replaces the trunk with the PKDA/gated-GQA hybrid, ``r`` adds MHDB reads,
``f`` adds FBT feedback, and ``l`` names the tied-depth loop that is specified
and not built.
Randomness (jitter draws, prefix lengths, pass counts) enters as *data* —
the trainer owns the shared streams that keep paired conditions
architecturally identical in everything but the flags.

Semantics worth naming because they are easy to get subtly wrong:

- Multi-head block-delta routing is a *transient read*: each KV-sized channel
  group has its own softmax over sources; the concatenated convex mixtures
  enriches one sublayer's pre-norm input and is never accumulated into
  the residual stream, so the stream stays the clean telescoping sum
  ``seed + sum(block_deltas) = h_top``.
- Every router prepends its own learnable zero-initialized null.  The remaining
  within-column bank is the actual input seed, completed four-layer block
  deltas, and at most one current-block partial delta.  Those non-null sources
  exactly decompose the current residual stream.
- The payload router uses the same multi-head primitive over the seed and all
  completed block deltas, additively on top of the top state.  At zero-query
  initialization its uniform non-null mixture is therefore collinear with the
  top state, so the following RMSNorm makes enrichment functionally inert up
  to its epsilon.
- Sublayer branch outputs are scaled ``1/sqrt(2L)``, so deltas — and hence
  routing sources — are the scaled outputs.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import INDUCTOR_MODE
from .attention import causal_attention, causal_block_mask, prefix_attention
from .cuda_kernels import ShadowOperand, sink_linear
from .parameter_groups import is_normuonh_parameter
from .pkda import PreconditionedKDA

try:  # Triton is deliberately a CUDA-only optimization dependency.
    from .cuda_kernels import bespoke_route
    from .cuda_kernels import triton as route_triton
except (ImportError, OSError):  # pragma: no cover - portable fallback
    bespoke_route = None
    route_triton = None

try:
    from cut_cross_entropy import linear_cross_entropy
    from cut_cross_entropy.cce import CCEParams, linear_cross_entropy_apply
    from cut_cross_entropy.utils import _handle_eps, compute_z_loss
except (ImportError, OSError):  # pragma: no cover - exercised on Jobe
    linear_cross_entropy = None
    CCEParams = None
    linear_cross_entropy_apply = None
    _handle_eps = None
    compute_z_loss = None

CONDITION_LETTERS: dict[str, tuple[str, str]] = {
    "a": (
        "hybrid",
        "Kimi Delta Attention: the [PKDA, PKDA, PKDA, gated global GQA] trunk",
    ),
    "r": (
        "block_routing",
        "MHDB residual reads of the seed and block deltas before every sublayer",
    ),
    "f": (
        "feedback",
        "full-bandwidth feedback: the FBT entry and a payload for the next column",
    ),
    "l": ("loop", "Huginn loop: the tied-depth core, specified and not built"),
}
"""Letter -> (``ModelConfig`` flag, one-line change), in canonical order."""


def parse_condition(text: str) -> str:
    """Canonicalize a condition string.

    Each letter is one change from the plain decoder; letters may arrive in any
    order and come back in :data:`CONDITION_LETTERS` order. The empty string is
    the plain twelve-layer RoPE GQA decoder.
    """
    unknown = sorted(set(text) - set(CONDITION_LETTERS))
    if unknown:
        raise ValueError(
            f"unknown condition letters {''.join(unknown)!r} in {text!r}; "
            f"expected letters from {''.join(CONDITION_LETTERS)!r}"
        )
    if len(set(text)) != len(text):
        raise ValueError(f"repeated letter in condition {text!r}")
    return "".join(letter for letter in CONDITION_LETTERS if letter in text)


BASE_NORMAL_INIT_STD = 0.02
"""Sampling scale retained for embeddings and NAdam-owned dense matrices."""


@dataclass(frozen=True)
class ModelConfig:
    """Trunk geometry plus one flag per condition letter.

    Defaults are the screen geometry and the plain RoPE GQA trunk; ``hybrid``
    (``a``) is the 3:1 PKDA/gated-global-GQA hybrid. Routing heads are not an
    independent knob: every routed condition uses one contiguous feature group
    per KV head. The groups do not align to mixer projections.
    """

    vocab_size: int = 151936
    dim: int = 768
    layers: int = 12
    heads: int = 8
    kv_heads: int = 4
    head_dim: int = 96
    intermediate: int = 3328
    pkda_heads: int = 10
    pkda_head_dim: int = 128
    pkda_conv_size: int = 4
    max_seq_len: int = 1024
    rope_theta: float = 1e6
    norm_eps: float = 1e-6
    routing_block_size: int = 4
    """Exact MHDB cell width in transformer layers."""

    hybrid: bool = False
    """``a``: [PKDA, PKDA, PKDA, gated global GQA] cells instead of RoPE GQA."""

    block_routing: bool = False
    """``r``: multi-head block-delta routing before every sublayer."""

    feedback: bool = False
    """``f``: FBT gated entry plus a payload for the next column."""

    loop: bool = False
    """``l``: the tied-depth core of ``docs/depth-architecture.md``; not built."""

    def __post_init__(self) -> None:
        if self.routing_block_size < 1:
            raise ValueError("routing block size must be positive")
        if self.pkda_heads < 1 or self.pkda_head_dim < 1:
            raise ValueError("PKDA head count and dimension must be positive")
        if self.pkda_conv_size < 1:
            raise ValueError("PKDA convolution width must be positive")

    @property
    def condition(self) -> str:
        """The canonical letters of this configuration."""
        return "".join(
            letter
            for letter, (flag, _) in CONDITION_LETTERS.items()
            if getattr(self, flag)
        )

    @property
    def routing_active(self) -> bool:
        return self.block_routing

    @property
    def gated_attention(self) -> bool:
        """Every global attention layer in the hybrid trunk is gated."""
        return self.hybrid

    @property
    def routing_heads(self) -> int:
        """The authoritative MHDB scaling rule: H equals KV-head count."""
        return self.kv_heads

    @property
    def routing_blocks(self) -> int:
        """Number of completed block deltas emitted by a full column."""
        return (self.layers + self.routing_block_size - 1) // self.routing_block_size

    @property
    def feedback_active(self) -> bool:
        return self.feedback

    @property
    def gated_entry(self) -> bool:
        """Whether the payload enters through the mandatory GLU fuse."""
        return self.feedback

    def is_pkda_layer(self, layer: int) -> bool:
        return self.hybrid and layer % 4 != 3

    @property
    def global_attention_layers(self) -> tuple[int, ...]:
        if not self.hybrid:
            return tuple(range(self.layers))
        return tuple(
            layer for layer in range(self.layers) if not self.is_pkda_layer(layer)
        )


def condition_config(condition: str, **overrides) -> ModelConfig:
    """The configuration a condition string names; overrides adjust geometry."""
    letters = parse_condition(condition)
    flags = {flag: letter in letters for letter, (flag, _) in CONDITION_LETTERS.items()}
    return replace(ModelConfig(**overrides), **flags)


class RMSNorm(nn.Module):
    """Learnable RMSNorm computed in fp32, returned in the input dtype."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class _EmbeddingSink(torch.autograd.Function):
    """Token lookup whose backward scatters into a persistent FP32 gradient.

    ``nn.Embedding``'s backward builds a dense zero ``[V, D]`` gradient, scatters
    the row gradients into it, and leaves autograd to add the whole tensor into
    the accumulated gradient: three passes over 467 MB at screen scale for
    4,096 touched rows.  Given the trainer's persistent buffer, the same
    contribution is one ``index_add_`` of the touched rows.
    """

    @staticmethod
    def forward(ctx, tokens: Tensor, weight: Tensor, sink: Tensor) -> Tensor:
        ctx.save_for_backward(tokens)
        ctx.sink = sink
        return F.embedding(tokens, weight)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[None, None, None]:
        (tokens,) = ctx.saved_tensors
        ctx.sink.index_add_(
            0, tokens.reshape(-1), gradient.reshape(-1, gradient.shape[-1]).float()
        )
        return None, None, None


class ResidualEmbedding(nn.Embedding):
    """FP32 tied weights, BF16 CUDA activations inside the training autocast.

    ``nn.Embedding`` is not autocast-aware.  Without this explicit boundary its
    FP32 output silently promotes every residual, payload, and routed source.
    Outside CUDA autocast (CPU tests and explicit FP32 analysis) it remains an
    ordinary embedding.

    ``grad_sink`` is the trainer-owned persistent FP32 gradient buffer of the
    tied weight.  When it is set, both the lookup and the tied classifier
    accumulate straight into it and return no autograd gradient; the buffer is
    the ``.grad`` the clip and optimizer read.  It is never a parameter or a
    checkpoint entry.
    """

    grad_sink: Tensor | None = None

    def forward(self, tokens: Tensor) -> Tensor:
        sink = self.grad_sink
        if sink is not None and self.weight.is_cuda and torch.is_grad_enabled():
            out = _EmbeddingSink.apply(tokens, self.weight, sink)
        else:
            out = super().forward(tokens)
        if out.is_cuda and torch.is_autocast_enabled("cuda"):
            return out.to(torch.bfloat16)
        return out


# -- rotary positions ----------------------------------------------------------


def rope_tables(cfg: ModelConfig, device, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    """(cos, sin) tables [max_seq_len, head_dim], HF half-rotation convention."""
    half = cfg.head_dim // 2
    inv_freq = cfg.rope_theta ** (-torch.arange(0, half, device=device).float() / half)
    angles = torch.outer(torch.arange(cfg.max_seq_len, device=device).float(), inv_freq)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate q or k [B, H, T, hd] by position tables [T, hd]."""
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


# -- kv cache ------------------------------------------------------------------


class KVCache:
    """Hybrid decoding cache with dense KV and fixed-size PKDA states.

    Only global-attention layers receive KV slots. PKDA layers retain their
    FP32 matrix/preconditioner states and three short-convolution histories.
    The caller advances the shared position once per column.
    """

    def __init__(self, cfg: ModelConfig, batch: int, device, dtype):
        self.cfg = cfg
        self.global_slots = {
            layer: slot for slot, layer in enumerate(cfg.global_attention_layers)
        }
        # Position-major storage keeps each incoming column's write contiguous.
        shape = (
            len(self.global_slots),
            batch,
            cfg.max_seq_len,
            cfg.kv_heads,
            cfg.head_dim,
        )
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pkda_states: dict[
            int,
            tuple[
                Tensor,
                Tensor,
                tuple[Tensor, Tensor, Tensor],
            ],
        ] = {}
        self.pos = 0

    def attention_tensors(self, layer: int) -> tuple[Tensor, Tensor]:
        slot = self.global_slots[layer]
        return self.k[slot], self.v[slot]

    def update(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write k/v [B,T,Hkv,D]; return the full prefix in that layout."""
        slot = self.global_slots[layer]
        length = k.shape[1]
        self.k[slot, :, self.pos : self.pos + length] = k
        self.v[slot, :, self.pos : self.pos + length] = v
        return (
            self.k[slot, :, : self.pos + length],
            self.v[slot, :, : self.pos + length],
        )

    def pkda_state(
        self, layer: int
    ) -> tuple[
        Tensor | None,
        Tensor | None,
        tuple[Tensor, Tensor, Tensor] | None,
    ]:
        return self.pkda_states.get(layer, (None, None, None))

    def update_pkda(
        self,
        layer: int,
        state: Tensor,
        a_state: Tensor,
        conv_state: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        self.pkda_states[layer] = (state, a_state, conv_state)

    def advance(self, length: int) -> None:
        self.pos += length

    def reset(self) -> None:
        self.pos = 0
        self.pkda_states.clear()


# -- trunk modules -------------------------------------------------------------


class Attention(nn.Module):
    """Dense GQA: RoPE on the plain trunk, NoPE plus sigmoid gate under ``a``."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.use_rope = not cfg.hybrid
        self.q_size = cfg.heads * cfg.head_dim
        self.kv_size = cfg.kv_heads * cfg.head_dim
        self.qkv_proj = nn.Linear(cfg.dim, self.q_size + 2 * self.kv_size, bias=False)
        self.o_proj = nn.Linear(cfg.heads * cfg.head_dim, cfg.dim, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.qkv_sink: Tensor | None = None
        self.gate_sink: Tensor | None = None
        self.o_sink: Tensor | None = None
        self.qkv_shadow: Tensor | None = None
        self.o_shadow: Tensor | None = None

    def forward(
        self,
        x: Tensor,
        cos: Tensor | None,
        sin: Tensor | None,
        gate_weight: Tensor | None,
        cache: KVCache | None,
        layer: int,
        attention_mask=None,
    ) -> Tensor:
        batch, length, _ = x.shape
        cfg = self.cfg
        if gate_weight is not None:
            # One GEMM produces Q/K/V and the gate logits; the gate keeps its
            # own parameter, optimizer group, and checkpoint name.
            q, k, v, gate_logits = sink_linear(
                x,
                (self.qkv_proj.weight, gate_weight),
                (self.qkv_sink, self.gate_sink),
                self.qkv_shadow,
            ).split((self.q_size, self.kv_size, self.kv_size, self.q_size), dim=-1)
        else:
            gate_logits = None
            q, k, v = sink_linear(
                x, (self.qkv_proj.weight,), (self.qkv_sink,), self.qkv_shadow
            ).split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = q.view(batch, length, cfg.heads, cfg.head_dim).transpose(1, 2)
        k = k.view(batch, length, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(batch, length, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.use_rope:
            if cos is None or sin is None:
                raise ValueError("RoPE GQA requires rotary position tables")
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        if cache is not None and length > 1 and cache.pos != 0:
            raise ValueError("multi-column append to a non-empty cache")
        causal = cache is None or length > 1
        if cache is not None:
            k_prefix, v_prefix = cache.update(
                layer, k.transpose(1, 2), v.transpose(1, 2)
            )
            k, v = k_prefix.transpose(1, 2), v_prefix.transpose(1, 2)
        if q.is_cuda:
            if causal:
                out = causal_attention(q, k, v, attention_mask).transpose(1, 2)
            else:
                out = prefix_attention(q, k, v).transpose(1, 2)
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=causal,
                enable_gqa=cfg.heads != cfg.kv_heads,
            ).transpose(1, 2)
        out = out.reshape(batch, length, cfg.heads * cfg.head_dim)
        if gate_logits is not None:
            out = out * torch.sigmoid(gate_logits)
        return sink_linear(out, (self.o_proj.weight,), (self.o_sink,), self.o_shadow)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(cfg.dim, 2 * cfg.intermediate, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate, cfg.dim, bias=False)
        self.gate_up_sink: Tensor | None = None
        self.down_sink: Tensor | None = None
        self.gate_up_shadow: Tensor | None = None
        self.down_shadow: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        gate, up = sink_linear(
            x, (self.gate_up_proj.weight,), (self.gate_up_sink,), self.gate_up_shadow
        ).chunk(2, dim=-1)
        return sink_linear(
            F.silu(gate) * up,
            (self.down_proj.weight,),
            (self.down_sink,),
            self.down_shadow,
        )


def _check_route_geometry(dim: int, num_heads: int) -> int:
    if num_heads < 2:
        raise ValueError("MHDB requires at least two routing heads")
    if dim % num_heads:
        raise ValueError(
            f"model dimension {dim} must be divisible by {num_heads} routing heads"
        )
    return dim // num_heads


def _route_algebra(
    values: Tensor,
    query: Tensor,
    key_weight: Tensor,
    eps: float,
    present: Tensor | None,
    num_heads: int,
) -> tuple[Tensor, Tensor]:
    """MHDB algebra over a pre-stacked source bank.

    RMS statistics remain full-width, while each contiguous channel group has
    its own source-axis softmax.  Values are mixed only inside their group;
    there is no output projection and no parameter increase over one query.
    """
    n_sources, batch, length, dim = values.shape
    head_dim = _check_route_geometry(dim, num_heads)
    projected = (query.float() * key_weight.float()).to(values.dtype)
    value_heads = values.reshape(n_sources, batch, length, num_heads, head_dim)
    query_heads = projected.reshape(num_heads, head_dim)
    dots = (value_heads * query_heads).float().sum(dim=-1)
    inv_rms = torch.rsqrt(values.float().square().mean(dim=-1) + eps)
    logits = dots * inv_rms.unsqueeze(-1)
    if present is not None:
        logits = logits.masked_fill(~present.unsqueeze(-1), float("-inf"))
    weights = logits.softmax(dim=0)
    routed = (
        (weights.to(values.dtype).unsqueeze(-1) * value_heads)
        .sum(dim=0)
        .reshape(batch, length, dim)
    )
    return routed, weights


def _route_sources(
    query: Tensor,
    key_weight: Tensor,
    eps: float,
    present: Tensor | None,
    num_heads: int,
    *sources: Tensor,
) -> tuple[Tensor, Tensor]:
    """Static MHDB pointer-list router: never copies a value bank."""
    batch, length, dim = sources[0].shape
    head_dim = _check_route_geometry(dim, num_heads)
    projected = (query.float() * key_weight.float()).to(sources[0].dtype)
    query_heads = projected.reshape(num_heads, head_dim)
    logits = torch.stack(
        [
            (source.reshape(batch, length, num_heads, head_dim) * query_heads)
            .float()
            .sum(dim=-1)
            * torch.rsqrt(source.float().square().mean(dim=-1) + eps).unsqueeze(-1)
            for source in sources
        ]
    )
    if present is not None:
        logits = logits.masked_fill(~present.unsqueeze(-1), float("-inf"))
    weights = logits.softmax(dim=0)
    routed = weights[0].to(sources[0].dtype).unsqueeze(-1) * sources[0].reshape(
        batch, length, num_heads, head_dim
    )
    for index in range(1, len(sources)):
        routed = routed + (
            weights[index].to(sources[index].dtype).unsqueeze(-1)
            * sources[index].reshape(batch, length, num_heads, head_dim)
        )
    return routed.reshape(batch, length, dim), weights


_compiled_route_sources = torch.compile(
    _route_sources,
    fullgraph=True,
    dynamic=True,
    mode="max-autotune-no-cudagraphs",
)


class Router(nn.Module):
    """One MHDB site: per-group softmaxes, RMS-normed keys, raw values.

    A learnable zero-init null vector is always prepended.  Its key is
    rmsnorm(0)=0, so its logit is exactly zero at init and routing mass on it
    initially adds nothing. The primitive retains an optional source-presence
    mask for kernel parity, although every screen source is present.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        _check_route_geometry(cfg.dim, cfg.routing_heads)
        self.num_heads = cfg.routing_heads
        self.query = nn.Parameter(torch.zeros(cfg.dim))
        self.key_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.null = nn.Parameter(torch.zeros(cfg.dim))

    def forward(
        self,
        sources: list[Tensor],
        masks: list[Tensor | None],
        want_weights: bool,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Return a routed addition and weights shaped ``[N,B,T,H]``."""
        if not sources:
            return None, None
        null = self.null.to(sources[0].dtype)
        masks = [None] + masks
        present = None
        if any(mask is not None for mask in masks):
            present = torch.stack(
                [
                    mask
                    if mask is not None
                    else torch.ones_like(sources[0][..., 0], dtype=torch.bool)
                    for mask in masks
                ]
            )
        if sources[0].is_cuda and route_triton is not None:
            projected = (self.query.float() * self.key_norm.weight.float()).to(
                sources[0].dtype
            )
            routed, weights = bespoke_route(
                projected,
                present,
                null,
                self.key_norm.eps,
                self.num_heads,
                tuple(sources),
            )
        elif sources[0].is_cuda:
            routed, weights = _compiled_route_sources(
                self.query,
                self.key_norm.weight,
                self.key_norm.eps,
                present,
                self.num_heads,
                null.expand_as(sources[0]),
                *sources,
            )
        else:
            routed, weights = _route_algebra(
                torch.stack([null.expand_as(sources[0]), *sources]),
                self.query,
                self.key_norm.weight,
                self.key_norm.eps,
                present,
                self.num_heads,
            )
        return routed, (weights.detach() if want_weights else None)


class Block(nn.Module):
    """One layer: (routed read →) attention, (routed read →) MLP.

    The routed read enriches the sublayer's pre-norm input only; the
    residual stream accumulates just the scaled branch outputs. The caller
    combines them into a completed four-layer delta; within a cell, the MLP
    read replaces the incoming partial with that partial plus its attention
    delta. The forward is pure — sources in, branch deltas out, no list
    mutation — so it can sit under activation checkpointing, whose backward
    recomputes the forward.
    """

    def __init__(self, cfg: ModelConfig, layer: int):
        super().__init__()
        self.layer = layer
        self.is_pkda = cfg.is_pkda_layer(layer)
        self.global_gate_index = (
            cfg.global_attention_layers.index(layer)
            if cfg.hybrid and not self.is_pkda
            else None
        )
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = (
            PreconditionedKDA(
                cfg.dim,
                num_heads=cfg.pkda_heads,
                head_dim=cfg.pkda_head_dim,
                conv_size=cfg.pkda_conv_size,
                norm_eps=cfg.norm_eps,
            )
            if self.is_pkda
            else Attention(cfg)
        )
        self.mlp = SwiGLU(cfg)
        self.branch_scale = 1.0 / math.sqrt(2 * cfg.layers)
        self.has_prior_partial = layer % cfg.routing_block_size != 0
        if cfg.routing_active:
            self.attn_router = Router(cfg)
            self.mlp_router = Router(cfg)
        else:
            self.attn_router = None
            self.mlp_router = None

    def _read(self, h, router, sources, want_weights):
        if router is None or not sources:
            return h, None
        masks: list[Tensor | None] = [None] * len(sources)
        routed, weights = router(list(sources), masks, want_weights)
        return (h if routed is None else h + routed), weights

    def forward(
        self,
        h: Tensor,
        block_start: Tensor | None,
        cos: Tensor | None,
        sin: Tensor | None,
        cache: KVCache | None,
        gate_weight: Tensor | None,
        want_weights: bool,
        *sources: Tensor,
        attention_mask=None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        """Returns (h, attn delta, mlp delta, cell delta, attn weights, mlp
        weights). ``block_start`` is the residual at the cell's entry, or None
        when this block is that entry; the returned cell delta is the next
        sublayer's partial source or, at the cell boundary, the completed
        block delta, computed here so it lands inside the compiled block
        instead of as an eager subtraction. (Passing ``h`` twice would make
        Dynamo guard the inputs against aliasing, and its recompile-reason
        logging then evaluates those guards across block instances.)"""
        start = h if block_start is None else block_start
        x, w_attn = self._read(h, self.attn_router, sources, want_weights)
        normalized = self.attn_norm(x)
        if self.is_pkda:
            state = a_state = conv_state = None
            if cache is not None:
                state, a_state, conv_state = cache.pkda_state(self.layer)
            mixed, state, a_state, conv_state = self.attn(
                normalized,
                state=state,
                a_state=a_state,
                conv_state=conv_state,
                output_final_state=cache is not None,
            )
            if cache is not None:
                cache.update_pkda(self.layer, state, a_state, conv_state)
        else:
            mixed = self.attn(
                normalized, cos, sin, gate_weight, cache, self.layer, attention_mask
            )
        a = self.branch_scale * mixed
        h = h + a
        if self.mlp_router is not None and self.has_prior_partial:
            mlp_sources = (*sources[:-1], sources[-1] + a)
        else:
            mlp_sources = (*sources, a)
        x, w_mlp = self._read(h, self.mlp_router, mlp_sources, want_weights)
        m = self.branch_scale * self.mlp(self.mlp_norm(x))
        h = h + m
        return h, a, m, h - start, w_attn, w_mlp


def _block_for_checkpoint(block, h, block_start, cos, sin, gate_weight, mask, *sources):
    """Pure checkpoint wrapper; immutable mask geometry is a per-call input."""
    h, a, m, delta, _, _ = block(
        h,
        block_start,
        cos,
        sin,
        None,
        gate_weight,
        False,
        *sources,
        attention_mask=mask,
    )
    return h, a, m, delta


# Each block family compiles through its own code object. Dynamo keys its
# cache on the code object and, when it recompiles, evaluates every earlier
# entry's guards against the current call to log the reason; one family's
# guards name attributes the other family's mixer does not have.
def _attention_block(block, h, block_start, cos, sin, gate_weight, mask, *sources):
    return _block_for_checkpoint(
        block, h, block_start, cos, sin, gate_weight, mask, *sources
    )


def _pkda_block(block, h, block_start, cos, sin, gate_weight, mask, *sources):
    return _block_for_checkpoint(
        block, h, block_start, cos, sin, gate_weight, mask, *sources
    )


_compiled_block = torch.compile(
    _attention_block,
    fullgraph=True,
    dynamic=False,
    # The graph shapes are fixed by the screen geometry. The probe performs
    # the expensive search once and later processes reuse its durable cache;
    # the outer trainer, rather than Inductor, owns CUDA capture.
    mode=INDUCTOR_MODE,
)

_compiled_pkda_block = torch.compile(
    _pkda_block,
    # FLA's opaque recurrence remains its own kernel boundary. Inductor graph
    # breaks around it and fuses the projections, controls, norm/gate, MLP,
    # residual updates, and routing on either side.
    fullgraph=False,
    dynamic=False,
    mode=INDUCTOR_MODE,
)


# -- the model -----------------------------------------------------------------


@dataclass
class ColumnOutput:
    """One full-stack pass over a column range."""

    h_top: Tensor
    """Top-of-stack residual stream [B, T, D], pre final norm."""

    payload: Tensor | None
    """What rides to the next column (conditions with ``f``), [B, T, D]."""

    route_weights: dict[str, Tensor]
    """Site → per-head softmax weights [N, B, T, H], when requested."""

    route_source_names: dict[str, tuple[str, ...]]
    """Site → source-axis labels matching ``route_weights`` exactly."""

    sources: list[Tensor] | None
    """Final stored bank: the seed followed by completed block deltas.
    The stream reconstructs from that seed plus the block deltas."""

    source_names: tuple[str, ...]
    """Names corresponding one-for-one with ``sources``."""

    n_seeds: int
    """How many leading entries of ``sources`` are seeds, not deltas."""


_ClassifierShadow = ShadowOperand
"""The tied classifier reads its BF16 shadow through the shared operand.

With a persistent sink the BF16 classifier gradient is added into it in one
fused pass instead of being widened to a 467 MB FP32 temporary that autograd
then adds a second time.
"""


class DeltaModel(nn.Module):
    """The full model; parents are deletions per the config flags."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.loop:
            raise NotImplementedError(
                "the l condition (tied-depth loop) is specified in "
                "docs/depth-architecture.md and not built"
            )
        self.cfg = cfg
        factor_seed = torch.initial_seed()
        self.embed_tokens = ResidualEmbedding(cfg.vocab_size, cfg.dim)
        self.register_buffer("_classifier_shadow", None, persistent=False)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.layers))
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        # Capture the exact common-trunk initialization boundary before
        # constructing any factor-specific random matrices. Restoring it below
        # keeps every shared parameter byte-identical across conditions.
        common_init_state = torch.random.get_rng_state()
        self.attention_gates = nn.ModuleList(
            nn.Linear(cfg.dim, cfg.heads * cfg.head_dim, bias=False)
            for _ in range(
                len(cfg.global_attention_layers) if cfg.gated_attention else 0
            )
        )
        self.fuse_value_sink: Tensor | None = None
        self.fuse_gate_sink: Tensor | None = None
        self.fuse_value_shadow: Tensor | None = None
        self.fuse_gate_shadow: Tensor | None = None
        self._shadow_refresh: list[tuple[Tensor, Tensor]] = []
        if cfg.gated_entry:
            self.fuse_value = nn.Linear(cfg.dim, cfg.dim, bias=False)
            self.fuse_gate = nn.Linear(cfg.dim, cfg.dim, bias=False)
            self.gate_norm = RMSNorm(cfg.dim, cfg.norm_eps)
            self.entry_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        if cfg.feedback_active:
            self.payload_norm = RMSNorm(cfg.dim, cfg.norm_eps)
            if cfg.routing_active:
                self.payload_router = Router(cfg)
        self._rope: tuple[Tensor, Tensor] | None = None
        self.grad_checkpoint = False
        """Runtime switch: checkpoint each block during training forwards."""
        torch.random.set_rng_state(common_init_state)
        self.embed_tokens.apply(self._init_weights)
        self.blocks.apply(self._init_weights)
        self.final_norm.apply(self._init_weights)
        self._init_factor_linears(
            self.attention_gates, factor_seed ^ 0x4152434849544543
        )
        if cfg.gated_entry:
            self._init_factor_linears(
                (self.fuse_value, self.fuse_gate),
                factor_seed ^ 0x524543555252454E,
            )
        self._scale_normuonh_initialization()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=BASE_NORMAL_INIT_STD)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, Router):
            nn.init.zeros_(module.query)
            nn.init.zeros_(module.null)

    @staticmethod
    def _init_factor_linears(linears: Iterable[nn.Linear], seed: int) -> None:
        """Initialize one factor from its own seed-stable random stream.

        Letter-private modules must not shift common initialization, and the
        same letter must initialize identically in every condition that has it.
        Models are constructed on CPU (or meta for accounting) before moving to
        an execution device, so one local CPU generator is authoritative.
        """
        generator = torch.Generator().manual_seed(seed % ((1 << 63) - 1))
        for linear in linears:
            nn.init.normal_(
                linear.weight, std=BASE_NORMAL_INIT_STD, generator=generator
            )

    @torch.no_grad()
    def _scale_normuonh_initialization(self) -> None:
        """Give each constrained matrix the Hyperball paper's fan-in scale.

        Scaling the already sampled normal values is distributionally identical
        to drawing with the target standard deviation. It also preserves the
        common and factor-private paired random streams exactly.
        """
        for name, parameter in self.named_parameters():
            if is_normuonh_parameter(name, parameter):
                target_std = 1 / math.sqrt(parameter.shape[1])
                parameter.mul_(target_std / BASE_NORMAL_INIT_STD)

    # -- pieces ----------------------------------------------------------------

    def rope(self, device, start: int, length: int) -> tuple[Tensor, Tensor]:
        if self._rope is None or self._rope[0].device != device:
            self._rope = rope_tables(self.cfg, device)
        cos, sin = self._rope
        return cos[start : start + length], sin[start : start + length]

    def fuse(self, payload: Tensor, e: Tensor) -> Tensor:
        """FBT entry: u = rmsnorm(W_U p ⊙ σ(W_G rmsnorm(e))) (Appendix C)."""
        gate = torch.sigmoid(
            sink_linear(
                self.gate_norm(e),
                (self.fuse_gate.weight,),
                (self.fuse_gate_sink,),
                self.fuse_gate_shadow,
            )
        )
        value = sink_linear(
            payload,
            (self.fuse_value.weight,),
            (self.fuse_value_sink,),
            self.fuse_value_shadow,
        )
        return self.entry_norm(value * gate)

    # -- persistent gradient sinks -------------------------------------------

    def bind_gradient_sinks(
        self, sinks: dict[nn.Parameter, Tensor] | None
    ) -> set[nn.Parameter]:
        """Attach trainer-owned FP32 gradient buffers to every sink-capable site.

        The tied embedding, every large projection, and the fusion matrices
        then accumulate their gradients in place and return no autograd
        gradient; ``None`` restores ordinary autograd accumulation.  Returns
        the parameters that were bound, so the trainer can treat a touched
        buffer as that parameter's activity for the captured mode.

        On CUDA every bound site also receives an address-stable BF16 shadow of
        its (concatenated) weights.  ``refresh_shadows`` rewrites the shadows
        from the FP32 masters once per optimizer update, so no replay casts or
        concatenates parameters; the shadows are neither parameters nor
        checkpoint state.
        """
        lookup = sinks or {}
        bound: set[nn.Parameter] = set()
        refresh: list[tuple[Tensor, Tensor]] = []

        def sink(parameter: nn.Parameter) -> Tensor | None:
            buffer = lookup.get(parameter)
            if buffer is not None:
                bound.add(parameter)
            return buffer

        def shadow(current: Tensor | None, *weights: Tensor) -> Tensor | None:
            # Rebinding keeps an existing shadow: the compiled blocks were
            # traced against these exact tensors, and an address must not
            # change between warm-up and capture.
            if sinks is None or not weights[0].is_cuda:
                return None
            rows = sum(weight.shape[0] for weight in weights)
            shape = (rows, weights[0].shape[1])
            fresh = current is None or current.shape != shape
            if fresh:
                current = torch.empty(
                    shape, device=weights[0].device, dtype=torch.bfloat16
                )
            start = 0
            for weight in weights:
                segment = current[start : start + weight.shape[0]]
                if fresh:
                    # A shadow is valid from the moment it exists; warm-up may
                    # trace the blocks before the trainer's first refresh. The
                    # copy must leave no autograd history on the shadow: a
                    # recorded CopySlices node lives on this (uncaptured)
                    # stream and would break the captured backward.
                    segment.copy_(weight.detach())
                refresh.append((segment, weight.detach()))
                start += weight.shape[0]
            return current

        # The portable head reads the tied weight through autograd, so only the
        # CUDA path (CCE plus the bespoke lookup) owns an embedding sink.
        self.embed_tokens.grad_sink = (
            sink(self.embed_tokens.weight) if self.embed_tokens.weight.is_cuda else None
        )
        for block in self.blocks:
            block.mlp.gate_up_sink = sink(block.mlp.gate_up_proj.weight)
            block.mlp.down_sink = sink(block.mlp.down_proj.weight)
            block.mlp.gate_up_shadow = shadow(
                block.mlp.gate_up_shadow, block.mlp.gate_up_proj.weight
            )
            block.mlp.down_shadow = shadow(
                block.mlp.down_shadow, block.mlp.down_proj.weight
            )
            attn = block.attn
            if block.is_pkda:
                attn.q_sink = sink(attn.q_proj.weight)
                attn.k_sink = sink(attn.k_proj.weight)
                attn.v_sink = sink(attn.v_proj.weight)
                attn.o_sink = sink(attn.o_proj.weight)
                attn.qkv_shadow = shadow(
                    attn.qkv_shadow,
                    attn.q_proj.weight,
                    attn.k_proj.weight,
                    attn.v_proj.weight,
                )
                attn.o_shadow = shadow(attn.o_shadow, attn.o_proj.weight)
                # The NAdam-owned control matrices keep ordinary autograd
                # gradients but read shadows too, so no replay casts them.
                attn.control_shadow = shadow(
                    attn.control_shadow, attn.control_proj.weight
                )
                attn.decay_up_shadow = shadow(
                    attn.decay_up_shadow, attn.decay_up.weight
                )
                attn.output_gate_shadow = shadow(
                    attn.output_gate_shadow, attn.output_gate_up.weight
                )
            else:
                attn.qkv_sink = sink(attn.qkv_proj.weight)
                attn.o_sink = sink(attn.o_proj.weight)
                attn.o_shadow = shadow(attn.o_shadow, attn.o_proj.weight)
                if block.global_gate_index is not None:
                    gate = self.attention_gates[block.global_gate_index].weight
                    attn.gate_sink = sink(gate)
                    attn.qkv_shadow = shadow(
                        attn.qkv_shadow, attn.qkv_proj.weight, gate
                    )
                else:
                    attn.gate_sink = None
                    attn.qkv_shadow = shadow(attn.qkv_shadow, attn.qkv_proj.weight)
        if self.cfg.gated_entry:
            self.fuse_value_sink = sink(self.fuse_value.weight)
            self.fuse_gate_sink = sink(self.fuse_gate.weight)
            self.fuse_value_shadow = shadow(
                self.fuse_value_shadow, self.fuse_value.weight
            )
            self.fuse_gate_shadow = shadow(self.fuse_gate_shadow, self.fuse_gate.weight)
        self._shadow_refresh = refresh
        return bound

    def logits(self, h_top: Tensor) -> Tensor:
        return F.linear(self.final_norm(h_top), self.embed_tokens.weight)

    @torch.no_grad()
    def refresh_shadows(self) -> None:
        """Refresh every CUDA-only BF16 operand copy from its FP32 master.

        That is the classifier readout of the tied embedding plus the trunk
        shadows bound with the gradient sinks. None of them is ``state_dict``
        content. Their addresses are allocated once before graph capture and
        remain stable across optimizer updates; only their contents change.
        """
        master = self.embed_tokens.weight
        if not master.is_cuda:
            self._classifier_shadow = None
            return
        shadow = self._classifier_shadow
        if (
            shadow is None
            or shadow.shape != master.shape
            or shadow.device != master.device
        ):
            shadow = torch.empty_like(master, dtype=torch.bfloat16)
            self._classifier_shadow = shadow
        shadow.copy_(master)
        if self._shadow_refresh:
            torch._foreach_copy_(
                [copy for copy, _ in self._shadow_refresh],
                [weight for _, weight in self._shadow_refresh],
            )

    def classifier_for_loss(self) -> Tensor:
        """Return the graph-stable BF16 CCE operand linked to the FP32 master."""
        master = self.embed_tokens.weight
        if not master.is_cuda:
            return master
        if self._classifier_shadow is None:
            raise RuntimeError("CUDA classifier shadow was not prepared")
        return _ClassifierShadow.apply(
            master, self._classifier_shadow, self.embed_tokens.grad_sink
        )

    # -- one column pass -------------------------------------------------------

    def forward_column(
        self,
        x: Tensor,
        *,
        cache: KVCache | None = None,
        want_weights: bool = False,
        need_payload: bool = True,
    ) -> ColumnOutput:
        """Run the stack once over inputs x [B, T, D].

        ``x`` is the actual column input: plain embeddings on pass 1 and
        Standard decoding, or the fused FBT input on feedback passes. With a
        cache, positions start at ``cache.pos`` and every mixer cache advances
        by ``T`` exactly once.
        """
        cfg = self.cfg
        start = cache.pos if cache is not None else 0
        cos, sin = (
            (None, None) if cfg.hybrid else self.rope(x.device, start, x.shape[1])
        )
        # Mask construction belongs outside fullgraph block compilation. The
        # same immutable geometry is reused by all global layers and passes.
        attention_mask = (
            causal_block_mask(x.shape[1], x.device)
            if x.is_cuda and cache is None
            else None
        )

        sources: list[Tensor] | None = None
        source_names: list[str] = []
        if cfg.routing_active:
            sources = [x]
            source_names.append("seed")
        seeds = len(sources) if sources is not None else 0

        h = x
        weights_out: dict[str, Tensor] = {}
        route_source_names: dict[str, tuple[str, ...]] = {}
        checkpointing = (
            self.grad_checkpoint
            and self.training
            and torch.is_grad_enabled()
            and cache is None
            and not want_weights
        )
        block_start = h
        # The residual's distance from the cell entry, produced by each block
        # for the next sublayer's partial source and the cell's completed delta.
        delta: Tensor | None = None
        for block in self.blocks:
            block_index = block.layer // cfg.routing_block_size
            block_offset = block.layer % cfg.routing_block_size
            if block_offset == 0:
                block_start = h
            # A cell-entry block measures its delta from its own input.
            entry = None if block_offset == 0 else block_start
            passed_sources = list(sources) if sources is not None else []
            passed_names = list(source_names)
            if sources is not None and block_offset:
                passed_sources.append(delta)
                passed_names.append(f"partial{block_index}")
            passed = tuple(passed_sources)
            gate_weight = (
                self.attention_gates[block.global_gate_index].weight
                if block.global_gate_index is not None
                else None
            )
            block_fn = _compiled_pkda_block if block.is_pkda else _compiled_block
            mask = None if block.is_pkda else attention_mask
            if checkpointing:
                h, _a, _m, delta = torch.utils.checkpoint.checkpoint(
                    block_fn if h.is_cuda else _block_for_checkpoint,
                    block,
                    h,
                    entry,
                    cos,
                    sin,
                    gate_weight,
                    mask,
                    *passed,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            elif h.is_cuda and cache is None and not want_weights:
                h, _a, _m, delta = block_fn(
                    block, h, entry, cos, sin, gate_weight, mask, *passed
                )
            else:
                h, _a, _m, delta, w_attn, w_mlp = block(
                    h,
                    entry,
                    cos,
                    sin,
                    cache,
                    gate_weight,
                    want_weights,
                    *passed,
                    attention_mask=mask,
                )
                if w_attn is not None:
                    site = f"L{block.layer}.attn"
                    weights_out[site] = w_attn
                    route_source_names[site] = ("null", *passed_names)
                if w_mlp is not None:
                    site = f"L{block.layer}.mlp"
                    weights_out[site] = w_mlp
                    mlp_names = (
                        passed_names
                        if block.has_prior_partial
                        else [*passed_names, f"partial{block_index}"]
                    )
                    route_source_names[site] = ("null", *mlp_names)
            if sources is not None and (
                block_offset == cfg.routing_block_size - 1
                or block.layer == cfg.layers - 1
            ):
                sources.append(delta)
                source_names.append(f"block{block_index}")
        if cache is not None:
            cache.advance(x.shape[1])

        payload = None
        if cfg.feedback_active and need_payload:
            enriched = h
            if cfg.routing_active:
                payload_sources = [sources[seeds - 1], *sources[seeds:]]
                routed, weights = self.payload_router(
                    payload_sources, [None] * len(payload_sources), want_weights
                )
                if weights is not None:
                    weights_out["payload"] = weights
                    route_source_names["payload"] = (
                        "null",
                        "seed",
                        *source_names[seeds:],
                    )
                if routed is not None:
                    enriched = h + routed
            payload = self.payload_norm(enriched)

        return ColumnOutput(
            h_top=h,
            payload=payload,
            route_weights=weights_out,
            route_source_names=route_source_names,
            sources=sources,
            source_names=tuple(source_names),
            n_seeds=seeds,
        )

    # -- sequential decoding ---------------------------------------------------

    def step(
        self,
        tokens: Tensor,
        payload: Tensor | None,
        cache: KVCache,
        *,
        want_weights: bool = False,
    ) -> ColumnOutput:
        """Decode one column: tokens [B, 1], payload [B, 1, D] from the
        previous column (None for Standard decoding and conditions without
        ``f``)."""
        e = self.embed_tokens(tokens)
        if payload is not None and not self.cfg.feedback_active:
            raise ValueError("a condition without f cannot consume a payload")
        x = self.fuse(payload, e) if payload is not None else e
        return self.forward_column(x, cache=cache, want_weights=want_weights)


# -- multi-pass training forward ----------------------------------------------


def shift_right(x: Tensor) -> Tensor:
    """Shift [B, T, D] one position rightward; position 0 becomes zero."""
    return torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)


def multipass(
    model: DeltaModel,
    tokens: Tensor,
    n_passes: int,
    *,
    prefix_lens: Tensor | None = None,
    jitter: Tensor | None = None,
    want_weights: bool = False,
) -> list[ColumnOutput]:
    """The condition-agnostic Jacobi multi-pass forward.

    tokens [B, T+1] is one stored row: the model executes its first T
    positions, which are exactly the positions that predict tokens 1..T.
    The final stored token is only ever a target; computing a column for
    it would be causally dead work.  prefix_lens [n_passes-1, B] holds
    values in 1..T-1 (the plain-embedding prefix per feedback pass; position
    0 is always plain and position T-1 is always fused); jitter [n_passes-1, B, T+1, D] is drawn at the
    stored-row width and its first T columns are added to the carried
    payload before shifting, so the keyed draw is independent of how many
    positions execute.  Both are pre-drawn by the caller — the shared
    randomness contract lives in the trainer, not here.  Conditions
    without ``f`` simply take n_passes=1.
    """
    cfg = model.cfg
    if n_passes > 1 and not cfg.feedback_active:
        raise ValueError("multi-pass batches require a condition with f")
    e = model.embed_tokens(tokens[:, :-1])
    out = model.forward_column(
        e, want_weights=want_weights, need_payload=n_passes > 1 or want_weights
    )
    outs = [out]
    if n_passes == 1:
        return outs

    length = e.shape[1]
    positions = torch.arange(length, device=tokens.device)
    for i in range(n_passes - 1):
        p = outs[-1].payload
        if jitter is not None:
            p = p + jitter[i][:, :length]
        p_shifted = shift_right(p)
        plain = positions[None, :] < prefix_lens[i][:, None]  # [B, T]
        fused = model.fuse(p_shifted, e)
        out = model.forward_column(
            torch.where(plain[..., None], e, fused),
            want_weights=want_weights,
            need_payload=i < n_passes - 2 or want_weights,
        )
        outs.append(out)
    return outs


def _head_losses(
    h_chunk: Tensor,
    target_chunk: Tensor,
    norm_weight: Tensor,
    classifier: Tensor,
    norm_eps: float,
) -> tuple[Tensor, Tensor]:
    dtype = h_chunk.dtype
    normalized = h_chunk.float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(dim=-1, keepdim=True) + norm_eps
    )
    normalized = (normalized * norm_weight.float()).to(dtype)
    logits = F.linear(normalized, classifier).float()
    ce = F.cross_entropy(logits.flatten(0, 1), target_chunk.flatten(), reduction="sum")
    return ce, logits.logsumexp(dim=-1).square().sum()


_compiled_head_losses = torch.compile(
    _head_losses,
    fullgraph=True,
    dynamic=False,
    mode="max-autotune-no-cudagraphs",
)


@torch.no_grad()
def batch_vocab_order(embeddings: Tensor, classifier: Tensor) -> Tensor:
    """This batch's mean-logit ordering of the vocabulary, an int32 permutation.

    Cut cross-entropy drops gradient tiles whose probabilities all fall below
    its epsilon and, when both halves tile the vocabulary the same way, skips
    them before recomputing their logits. Upstream orders the vocabulary by
    the batch's mean logit, measured inside its forward; the mean logit is
    linear in the embeddings, so one classifier product with the mean
    embedding gives it before the forward runs, inside the captured graph,
    with no state.

    The order is ascending, as upstream's ``argsort`` is, and that direction
    is load-bearing: the backward accumulates each row's embedding gradient
    across the vocabulary tiles in BF16 through locks, roughly in tile order,
    so the tiles with the smallest contributions have to arrive first. Every
    other order that was tried (descending mean logit, held-out frequency, a
    per-step average) dropped or computed the same tiles yet rounded the
    tail away against an already large running sum, a coherent error that
    the step's microbatch averaging did not cancel: the step gradient came
    out 20 to 25% too large.
    """
    mean = embeddings.reshape(-1, embeddings.shape[-1]).float().mean(0, keepdim=True)
    logit_avg = torch.addmm(
        torch.zeros(1, classifier.shape[0], device=classifier.device),
        mean.to(classifier.dtype),
        classifier.mT,
        out_dtype=torch.float32,
    )
    return torch.argsort(logit_avg[0], stable=True).to(torch.int32)


def _fixed_cce_z(
    embeddings: Tensor,
    classifier: Tensor,
    targets: Tensor,
    vocab_ordering: Tensor | None = None,
    *,
    skip_early: bool = True,
    tile_flags: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Current CCE with capture-safe preprocessing and differentiable LSE.

    Every training target is a real vocabulary id, so CCE's public
    ``ignore_index`` discovery would always produce ``valids=None``. Construct
    that exact pinned-C CCE request directly: its data-dependent ``nonzero`` is
    illegal inside CUDA graph capture, while its forward and new differentiable
    LSE backward are otherwise the authoritative implementation.

    With ``vocab_ordering`` both halves tile the classifier through that
    permutation and the backward skips every tile the gradient filter would
    drop before recomputing its logits. ``skip_early`` and ``tile_flags`` are
    the fork's diagnostics: the probe forces the late filter alone and checks
    that both paths compute the identical tile set.
    """
    if (
        CCEParams is None
        or linear_cross_entropy_apply is None
        or _handle_eps is None
        or compute_z_loss is None
    ):
        raise RuntimeError("cut-cross-entropy is unavailable")
    embeddings = embeddings.contiguous().flatten(0, -2)
    targets = targets.contiguous().flatten()
    if targets.data_ptr() % 16:
        targets = F.pad(targets, (0, 1))[:-1]
    params = CCEParams(
        targets=targets,
        valids=None,
        softcap=None,
        reduction="mean",
        filter_eps=_handle_eps("auto", embeddings.dtype),
        shift=0,
        batch_shape=targets.shape,
        accum_e_fp32=False,
        accum_c_fp32=False,
        filter_e_grad=True,
        filter_c_grad=True,
        vocab_parallel_options=None,
        return_lse=True,
        vocab_ordering=vocab_ordering,
        skip_early=skip_early,
        tile_flags=tile_flags,
    )
    ce, lse = linear_cross_entropy_apply(
        embeddings,
        classifier,
        None,
        params,
    )
    assert lse is not None
    return ce, compute_z_loss(lse)


def sequence_ce(
    model: DeltaModel,
    h_top: Tensor,
    targets: Tensor,
    *,
    chunk: int = 1024,
) -> tuple[Tensor, Tensor]:
    """(mean CE, mean z²) over [B, T] targets, chunked along the sequence.

    The vocab-sized logits (151936 wide at screen scale) dominate
    activation memory, so the head runs under activation checkpointing
    one sequence-chunk at a time — live logits are bounded to a single
    chunk in both forward and backward.
    """
    count = targets.numel()
    if h_top.is_cuda and linear_cross_entropy is not None:
        # CCE fuses tied unembedding and CE, never materializing [B,T,V].
        # Its high-threshold gradient filter is an intentional throughput-
        # first numerical divergence of the authoritative CUDA recipe.
        normalized = model.final_norm(h_top)
        # The ordering is a scheduling hint for the backward, so evaluation
        # (no backward) tiles the classifier in place.
        ordering = (
            batch_vocab_order(normalized, model._classifier_shadow)
            if torch.is_grad_enabled()
            else None
        )
        return _fixed_cce_z(normalized, model.classifier_for_loss(), targets, ordering)

    ce_sum = h_top.new_zeros((), dtype=torch.float32)
    z_sum = h_top.new_zeros((), dtype=torch.float32)
    recompute = torch.is_grad_enabled() and h_top.requires_grad
    for start in range(0, targets.shape[1], chunk):
        h_piece = h_top[:, start : start + chunk]
        t_piece = targets[:, start : start + chunk]
        if recompute:
            ce, z = torch.utils.checkpoint.checkpoint(
                _compiled_head_losses if h_top.is_cuda else _head_losses,
                h_piece,
                t_piece,
                model.final_norm.weight,
                model.embed_tokens.weight,
                model.final_norm.eps,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            head = _compiled_head_losses if h_top.is_cuda else _head_losses
            ce, z = head(
                h_piece,
                t_piece,
                model.final_norm.weight,
                model.embed_tokens.weight,
                model.final_norm.eps,
            )
        ce_sum = ce_sum + ce
        z_sum = z_sum + z
    return ce_sum / count, z_sum / count


def multipass_loss(
    model: DeltaModel,
    tokens: Tensor,
    outs: list[ColumnOutput],
    *,
    z_coef: float | Tensor = 0.0,
) -> tuple[Tensor, list[Tensor]]:
    """FBT Eq. 12 with λ=1: pass-1 NTP plus the mean over feedback passes,
    plus (in cooldown) the z-loss under the same per-pass weighting.

    Returns (total, per-pass CE losses); per-pass loss 0 is the
    Standard-mode tracking metric.
    """
    targets = tokens[:, 1:]
    losses, z_terms = [], []
    for out in outs:
        ce, z = sequence_ce(model, out.h_top, targets)
        losses.append(ce)
        z_terms.append(z)

    def combine(values: list[Tensor]) -> Tensor:
        if len(values) == 1:
            return values[0]
        return values[0] + torch.stack(values[1:]).mean()

    total = combine(losses)
    total = total + z_coef * combine(z_terms)
    return total, losses


@torch.no_grad()
def iterate_fused(
    model: DeltaModel, tokens: Tensor, n_iters: int
) -> list[dict[str, float]]:
    """The contraction diagnostic: iterated fully-fused prefill passes.

    Repeatedly applies the feedback map (prefix length 1 — everything
    fused) and reports per-iteration validation loss and the update size
    ||h(k) − h(k−1)|| (FBT Fig. 3).  Decaying update norms and flat loss
    are the contraction signature; oscillation or rising loss means the
    map diverges under self-composition.
    """
    cfg = model.cfg
    if not cfg.feedback_active:
        raise ValueError("the contraction diagnostic needs a condition with f")
    # This is part of the standing training monitor, so CUDA must follow the
    # same BF16 activation path as captured training and evaluation, which
    # preserves the numerical contract and reuses the captured block
    # specializations instead of compiling a second set.
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if tokens.is_cuda
        else contextlib.nullcontext()
    )
    with autocast:
        e = model.embed_tokens(tokens[:, :-1])
        batch, length = e.shape[:2]
        positions = torch.arange(length, device=tokens.device)
        plain = (positions[None, :] < 1).expand(batch, -1)

        out = model.forward_column(e)
        records = []
        for _ in range(n_iters):
            previous = out.h_top
            p_shifted = shift_right(out.payload)
            fused = model.fuse(p_shifted, e)
            out = model.forward_column(torch.where(plain[..., None], e, fused))
            loss, _ = sequence_ce(model, out.h_top, tokens[:, 1:])
            delta = (out.h_top - previous).float().norm(dim=-1).mean()
            records.append({"loss": loss.item(), "update_norm": delta.item()})
    return records

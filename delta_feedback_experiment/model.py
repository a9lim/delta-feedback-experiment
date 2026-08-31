"""The DF model family: one trunk with five exact arm configurations.

The architecture contract is ``docs/design.md``. Everything here is
arm-agnostic model semantics: the five arms are one class under
:class:`ModelConfig` flags. The four factorial cells share the hybrid
PKDA/gated-GQA trunk; the axes are MHDB and FBT. ``vanilla`` is the pure-GQA
external trunk control.
Randomness (jitter draws, prefix lengths, pass counts) enters as *data* —
the trainer owns the shared streams that keep paired arms architecturally
identical in everything but the flags.

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

from .pkda import PreconditionedKDA

try:  # Triton is deliberately a CUDA-only optimization dependency.
    from .cuda_kernels import bespoke_route
    from .cuda_kernels import triton as route_triton
except (ImportError, OSError):  # pragma: no cover - portable fallback
    bespoke_route = None
    route_triton = None

try:  # CUDA-only wheels; CPU/MPS retain the exact portable fallbacks.
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
except (ImportError, OSError):  # pragma: no cover - exercised on Jobe
    flash_attn_func = None
    flash_attn_with_kvcache = None

try:
    from cut_cross_entropy import linear_cross_entropy
    from cut_cross_entropy.cce import (
        sort_logit_avg,
    )
    from cut_cross_entropy.cce_backward import cce_backward_kernel
    from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
    from cut_cross_entropy.indexed_dot import indexed_neg_dot_forward_kernel
    from cut_cross_entropy.utils import _handle_eps
except (ImportError, OSError):  # pragma: no cover - exercised on Jobe
    linear_cross_entropy = None
    sort_logit_avg = None
    cce_backward_kernel = None
    cce_lse_forward_kernel = None
    indexed_neg_dot_forward_kernel = None
    _handle_eps = None

ARMS = ("vanilla", "base", "mhdb", "fbt", "df")


@dataclass(frozen=True)
class ModelConfig:
    """Trunk geometry plus the two-axis arm flags.

    Defaults are the registered screen geometry. ``vanilla`` uses the legacy
    pure-GQA trunk; every other arm uses the 3:1 PKDA/gated-global-GQA hybrid.
    Routing heads are not an independent knob: every routed arm uses one
    contiguous feature group per KV head. The groups do not align to mixer
    projections.
    """

    vocab_size: int = 151936
    dim: int = 768
    layers: int = 12
    heads: int = 8
    kv_heads: int = 4
    head_dim: int = 96
    intermediate: int = 3072
    pkda_heads: int = 8
    pkda_head_dim: int = 128
    pkda_conv_size: int = 4
    max_seq_len: int = 1024
    rope_theta: float = 1e6
    norm_eps: float = 1e-6
    routing_block_size: int = 4
    """Exact MHDB cell width in transformer layers."""

    hybrid: bool = False
    """Use [PKDA, PKDA, PKDA, gated global GQA] cells."""

    block_routing: bool = False
    """MHDB axis: multi-head block-delta routing before every sublayer."""

    feedback: bool = False
    """FBT axis: gated entry plus a payload for the next column."""

    def __post_init__(self) -> None:
        if self.routing_block_size < 1:
            raise ValueError("routing block size must be positive")
        if self.pkda_heads < 1 or self.pkda_head_dim < 1:
            raise ValueError("PKDA head count and dimension must be positive")
        if self.pkda_conv_size < 1:
            raise ValueError("PKDA convolution width must be positive")

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


def arm_config(arm: str, **overrides) -> ModelConfig:
    """The named arm's configuration; overrides adjust instantiated geometry."""
    flags = {
        "vanilla": {},
        "base": {"hybrid": True},
        "mhdb": {"hybrid": True, "block_routing": True},
        "fbt": {"hybrid": True, "feedback": True},
        "df": {"hybrid": True, "block_routing": True, "feedback": True},
    }
    if arm not in flags:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    return replace(ModelConfig(**overrides), **flags[arm])


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


class ResidualEmbedding(nn.Embedding):
    """FP32 tied weights, BF16 CUDA activations inside the training autocast.

    ``nn.Embedding`` is not autocast-aware.  Without this explicit boundary its
    FP32 output silently promotes every residual, payload, and routed source.
    Outside CUDA autocast (CPU tests and explicit FP32 analysis) it remains an
    ordinary embedding.
    """

    def forward(self, tokens: Tensor) -> Tensor:
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
        # FlashAttention's native cache layout: [B, S, Hkv, D].
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
    """Dense GQA: RoPE for vanilla, NoPE plus sigmoid gate in the hybrid."""

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

    def forward(
        self,
        x: Tensor,
        cos: Tensor | None,
        sin: Tensor | None,
        gate_weight: Tensor | None,
        cache: KVCache | None,
        layer: int,
    ) -> Tensor:
        batch, length, _ = x.shape
        cfg = self.cfg
        q, k, v = self.qkv_proj(x).split(
            (self.q_size, self.kv_size, self.kv_size), dim=-1
        )
        q = q.view(batch, length, cfg.heads, cfg.head_dim).transpose(1, 2)
        k = k.view(batch, length, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(batch, length, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.use_rope:
            if cos is None or sin is None:
                raise ValueError("vanilla GQA requires rotary position tables")
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        q_flash = q.transpose(1, 2)
        k_flash = k.transpose(1, 2)
        v_flash = v.transpose(1, 2)
        use_flash = q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)

        if cache is not None and length > 1 and cache.pos != 0:
            raise ValueError("multi-column append to a non-empty cache")
        if cache is not None and use_flash and flash_attn_with_kvcache is not None:
            cache_k, cache_v = cache.attention_tensors(layer)
            out = flash_attn_with_kvcache(
                q_flash,
                cache_k,
                cache_v,
                k=k_flash,
                v=v_flash,
                cache_seqlens=cache.pos,
                causal=True,
            )
        elif cache is None and use_flash and flash_attn_func is not None:
            # FlashAttention natively accepts Hq/Hkv GQA without expanding K/V.
            out = flash_attn_func(q_flash, k_flash, v_flash, causal=True)
        else:
            causal = True
            if cache is not None:
                k_flash, v_flash = cache.update(layer, k_flash, v_flash)
                causal = length > 1
            out = F.scaled_dot_product_attention(
                q,
                k_flash.transpose(1, 2),
                v_flash.transpose(1, 2),
                is_causal=causal,
                enable_gqa=cfg.heads != cfg.kv_heads,
            ).transpose(1, 2)
        out = out.reshape(batch, length, cfg.heads * cfg.head_dim)
        if gate_weight is not None:
            out = out * torch.sigmoid(F.linear(x, gate_weight))
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(cfg.dim, 2 * cfg.intermediate, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate, cfg.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


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
    mask for kernel parity, although every registered screen source is present.
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
        null = self.null.to(sources[0].dtype).expand_as(sources[0])
        sources = [null] + sources
        masks = [None] + masks
        if len(sources) < 2:
            return None, None
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
                True,
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
                *sources,
            )
        else:
            routed, weights = _route_algebra(
                torch.stack(sources),
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
        cos: Tensor | None,
        sin: Tensor | None,
        cache: KVCache | None,
        gate_weight: Tensor | None,
        want_weights: bool,
        *sources: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        """Returns (h, attn delta, mlp delta, attn weights, mlp weights)."""
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
            mixed = self.attn(normalized, cos, sin, gate_weight, cache, self.layer)
        a = self.branch_scale * mixed
        h = h + a
        if self.mlp_router is not None and self.has_prior_partial:
            mlp_sources = (*sources[:-1], sources[-1] + a)
        else:
            mlp_sources = (*sources, a)
        x, w_mlp = self._read(h, self.mlp_router, mlp_sources, want_weights)
        m = self.branch_scale * self.mlp(self.mlp_norm(x))
        h = h + m
        return h, a, m, w_attn, w_mlp


def _block_for_checkpoint(block, h, cos, sin, gate_weight, *sources):
    """Tensor-only wrapper for activation checkpointing (no cache, no
    weights) — the recomputation must be free of side effects."""
    h, a, m, _, _ = block(h, cos, sin, None, gate_weight, False, *sources)
    return h, a, m


_compiled_block = torch.compile(
    _block_for_checkpoint,
    fullgraph=True,
    dynamic=True,
    # The complete block has many lifted GEMMs and source-count variants.
    # Default Inductor still fuses every surrounding pointwise epilogue while
    # avoiding a >10 minute exhaustive GEMM search already covered well by
    # cuBLAS/FlashAttention on Ada. The outer trainer owns CUDA capture.
    mode="default",
)

_compiled_pkda_block = torch.compile(
    _block_for_checkpoint,
    # FLA's opaque recurrence remains its own kernel boundary. Inductor graph
    # breaks around it and fuses the projections, controls, norm/gate, MLP,
    # residual updates, and routing on either side.
    fullgraph=False,
    dynamic=True,
    mode="default",
)


# -- the model -----------------------------------------------------------------


@dataclass
class ColumnOutput:
    """One full-stack pass over a column range."""

    h_top: Tensor
    """Top-of-stack residual stream [B, T, D], pre final norm."""

    payload: Tensor | None
    """What rides to the next column (feedback arms), [B, T, D]."""

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


class DFModel(nn.Module):
    """The full model; parents are deletions per the config flags."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        factor_seed = torch.initial_seed()
        self.embed_tokens = ResidualEmbedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.layers))
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        # Capture the exact common-trunk initialization boundary before
        # constructing any factor-specific random matrices. Restoring it below
        # keeps every shared parameter byte-identical across arms.
        common_init_state = torch.random.get_rng_state()
        self.attention_gates = nn.ModuleList(
            nn.Linear(cfg.dim, cfg.heads * cfg.head_dim, bias=False)
            for _ in range(
                len(cfg.global_attention_layers) if cfg.gated_attention else 0
            )
        )
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

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, Router):
            nn.init.zeros_(module.query)
            nn.init.zeros_(module.null)

    @staticmethod
    def _init_factor_linears(linears: Iterable[nn.Linear], seed: int) -> None:
        """Initialize one factor from its own seed-stable random stream.

        Conditional factor modules must not shift common initialization, and
        the same factor must initialize identically in its parent and DF cell.
        Models are constructed on CPU (or meta for accounting) before moving to
        an execution device, so one local CPU generator is authoritative.
        """
        generator = torch.Generator().manual_seed(seed % ((1 << 63) - 1))
        for linear in linears:
            nn.init.normal_(linear.weight, std=0.02, generator=generator)

    # -- pieces ----------------------------------------------------------------

    def rope(self, device, start: int, length: int) -> tuple[Tensor, Tensor]:
        if self._rope is None or self._rope[0].device != device:
            self._rope = rope_tables(self.cfg, device)
        cos, sin = self._rope
        return cos[start : start + length], sin[start : start + length]

    def fuse(self, payload: Tensor, e: Tensor) -> Tensor:
        """FBT entry: u = rmsnorm(W_U p ⊙ σ(W_G rmsnorm(e))) (Appendix C)."""
        gate = torch.sigmoid(self.fuse_gate(self.gate_norm(e)))
        return self.entry_norm(self.fuse_value(payload) * gate)

    def logits(self, h_top: Tensor) -> Tensor:
        return F.linear(self.final_norm(h_top), self.embed_tokens.weight)

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
        for block in self.blocks:
            block_index = block.layer // cfg.routing_block_size
            block_offset = block.layer % cfg.routing_block_size
            if block_offset == 0:
                block_start = h
            passed_sources = list(sources) if sources is not None else []
            passed_names = list(source_names)
            if sources is not None and block_offset:
                passed_sources.append(h - block_start)
                passed_names.append(f"partial{block_index}")
            passed = tuple(passed_sources)
            gate_weight = (
                self.attention_gates[block.global_gate_index].weight
                if block.global_gate_index is not None
                else None
            )
            block_fn = _compiled_pkda_block if block.is_pkda else _compiled_block
            if checkpointing:
                h, _a, _m = torch.utils.checkpoint.checkpoint(
                    block_fn if h.is_cuda else _block_for_checkpoint,
                    block,
                    h,
                    cos,
                    sin,
                    gate_weight,
                    *passed,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            elif h.is_cuda and cache is None and not want_weights:
                h, _a, _m = block_fn(block, h, cos, sin, gate_weight, *passed)
            else:
                h, _a, _m, w_attn, w_mlp = block(
                    h,
                    cos,
                    sin,
                    cache,
                    gate_weight,
                    want_weights,
                    *passed,
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
                sources.append(h - block_start)
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
        previous column (None for Standard decoding and non-feedback arms)."""
        e = self.embed_tokens(tokens)
        if payload is not None and not self.cfg.feedback_active:
            raise ValueError("a feedback-free arm cannot consume a payload")
        x = self.fuse(payload, e) if payload is not None else e
        return self.forward_column(x, cache=cache, want_weights=want_weights)


# -- multi-pass training forward ----------------------------------------------


def shift_right(x: Tensor) -> Tensor:
    """Shift [B, T, D] one position rightward; position 0 becomes zero."""
    return torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)


def multipass(
    model: DFModel,
    tokens: Tensor,
    n_passes: int,
    *,
    prefix_lens: Tensor | None = None,
    jitter: Tensor | None = None,
    want_weights: bool = False,
) -> list[ColumnOutput]:
    """The arm-agnostic Jacobi multi-pass forward.

    tokens [B, T]; prefix_lens [n_passes-1, B] with values in 1..T (the
    plain-embedding prefix per feedback pass; position 0 is always
    plain); jitter [n_passes-1, B, T, D] added to the carried payload
    before shifting.  Both are pre-drawn by the caller — the shared
    randomness contract lives in the trainer, not here.  Non-feedback
    arms simply take n_passes=1.
    """
    cfg = model.cfg
    if n_passes > 1 and not cfg.feedback_active:
        raise ValueError("multi-pass batches require a feedback-bearing arm")
    e = model.embed_tokens(tokens)
    out = model.forward_column(
        e, want_weights=want_weights, need_payload=n_passes > 1 or want_weights
    )
    outs = [out]
    if n_passes == 1:
        return outs

    length = tokens.shape[1]
    positions = torch.arange(length, device=tokens.device)
    for i in range(n_passes - 1):
        p = outs[-1].payload
        if jitter is not None:
            p = p + jitter[i]
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


class _LinearCrossEntropyZFunction(torch.autograd.Function):
    """CCE's tiled linear CE plus exact mean-square log-partition loss.

    The pinned CCE forward already computes one FP32 log-sum-exp per token.
    Preserve it for the scalar z-loss and reuse CCE's probability-tile backward
    for ``2 * z * lse * softmax(logits)``.  The classifier-sized logits are
    never materialized.  CE retains the authoritative high-threshold gradient
    filter; the z term is deliberately unfiltered so its derivative is exact.
    """

    @staticmethod
    def forward(ctx, embeddings: Tensor, classifier: Tensor, targets: Tensor):
        filter_eps = _handle_eps("high", embeddings.dtype)
        return_logit_avg = (
            embeddings.requires_grad or classifier.requires_grad
        ) and filter_eps is not None
        result = cce_lse_forward_kernel(
            embeddings,
            classifier,
            None,
            softcap=None,
            return_logit_avg=return_logit_avg,
        )
        if return_logit_avg:
            lse, logit_avg = result
        else:
            lse, logit_avg = result, None
        neg_dot = indexed_neg_dot_forward_kernel(
            embeddings,
            classifier,
            targets,
            False,
            None,
            None,
            lse.dtype,
        )
        ce = neg_dot.add_(lse).mean()
        z = lse.square().mean()
        ctx.save_for_backward(embeddings, classifier, lse, targets, logit_avg)
        ctx.filter_eps = filter_eps
        return ce, z

    @staticmethod
    def backward(ctx, grad_ce: Tensor, grad_z: Tensor):
        embeddings, classifier, lse, targets, logit_avg = ctx.saved_tensors
        ordering = sort_logit_avg(logit_avg) if logit_avg is not None else None
        scale = 1.0 / lse.numel()
        # One vocabulary sweep: scaling (p-y) by ce + 2*z*lse gives the
        # desired probability gradient but over-scales the target subtraction.
        # Repair that single indexed classifier column below.
        z_scale = grad_z * (2.0 * lse)
        de, dc = cce_backward_kernel(
            grad_ce + z_scale,
            embeddings,
            classifier,
            lse,
            None,
            None,
            ctx.filter_eps,
            targets=targets,
            shift=False,
            vocab_ordering=ordering,
            grad_scale=scale,
        )
        correction = z_scale * scale
        de.add_(
            classifier.index_select(0, targets).to(de.dtype)
            * correction.to(de.dtype).unsqueeze(1)
        )
        dc.index_add_(
            0,
            targets,
            embeddings.to(dc.dtype) * correction.to(dc.dtype).unsqueeze(1),
        )
        return de, dc, None


def _fixed_cce_z(
    embeddings: Tensor, classifier: Tensor, targets: Tensor
) -> tuple[Tensor, Tensor]:
    """Dense capture-safe CCE and z-loss with a BF16 tied classifier operand."""
    embeddings = embeddings.contiguous().flatten(0, -2)
    targets = targets.contiguous().flatten()
    return _LinearCrossEntropyZFunction.apply(
        embeddings, classifier.to(embeddings.dtype), targets
    )


def sequence_ce(
    model: DFModel,
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
        return _fixed_cce_z(normalized, model.embed_tokens.weight, targets)

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
    model: DFModel,
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
        ce, z = sequence_ce(model, out.h_top[:, :-1], targets)
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
    model: DFModel, tokens: Tensor, n_iters: int
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
        raise ValueError("the contraction diagnostic needs a feedback-bearing arm")
    # This is part of the standing training monitor, so CUDA must follow the
    # same BF16 activation path as captured training and evaluation.  Besides
    # preserving the numerical contract, keeping autocast state identical
    # avoids an unprepared whole-block Dynamo specialization at every routed
    # source count after the capture-time recompile limit has been restored.
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if tokens.is_cuda
        else contextlib.nullcontext()
    )
    with autocast:
        batch, length = tokens.shape
        e = model.embed_tokens(tokens)
        positions = torch.arange(length, device=tokens.device)
        plain = (positions[None, :] < 1).expand(batch, -1)

        out = model.forward_column(e)
        records = []
        for _ in range(n_iters):
            previous = out.h_top
            p_shifted = shift_right(out.payload)
            fused = model.fuse(p_shifted, e)
            out = model.forward_column(torch.where(plain[..., None], e, fused))
            loss, _ = sequence_ce(model, out.h_top[:, :-1], tokens[:, 1:])
            delta = (out.h_top - previous).float().norm(dim=-1).mean()
            records.append({"loss": loss.item(), "update_norm": delta.item()})
    return records

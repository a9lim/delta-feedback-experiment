"""The DF model family: one trunk, arms as deletions.

The architecture contract is docs/design.md (Architecture); parity
references are DAR's released code
(references/delta-attention-residuals-code/) for routing semantics and
FBT arXiv:2608.08888 Appendices A/C for the trunk conventions and the
multi-pass entry.  Everything here is arm-agnostic model semantics: the
five arms are one class under :class:`ModelConfig` flags, and the parents
are exact deletions of DF.  Randomness (jitter draws, prefix lengths,
pass counts) enters as *data* — the trainer owns the shared streams that
keep paired arms architecturally identical in everything but the flags.

Semantics worth naming because they are easy to get subtly wrong:

- Depth routing is a *transient read*: the routed convex combination
  enriches one sublayer's pre-norm input and is never accumulated into
  the residual stream, so the stream stays the clean telescoping sum
  seed + Σdeltas = h_top (paper Fig. 3 and released code agree).
- The source list seeds with the column's actual input (complete
  decomposition), and a routing site is a no-op until it can see two
  sources — the singleton seed would route weight 1 onto itself.
- The payload router reads deltas only (never the seed), additively on
  top of the top state; DF-soft's routers all carry a learnable
  zero-init null source (DAR's fine-tuning mechanism, verbatim).
- Sublayer branch outputs are scaled 1/sqrt(2L) (FBT depth scaling, our
  pinned formula), so deltas — and hence routing sources — are the
  scaled outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ARMS = ("vanilla", "dar", "fbt", "df", "df_soft")


@dataclass(frozen=True)
class ModelConfig:
    """Trunk geometry plus the two-axis arm flags.

    Screen defaults are the pinned 220M trunk (design: Sub-flagship
    geometry): DAR's d=768/L=12 with their script-default head split.
    """

    vocab_size: int = 151936
    dim: int = 768
    layers: int = 12
    heads: int = 8
    kv_heads: int = 4
    head_dim: int = 96
    intermediate: int = 3072
    max_seq_len: int = 1024
    rope_theta: float = 1e6
    norm_eps: float = 1e-6

    depth_routing: bool = False
    """DAR axis: routing sites before every sublayer."""

    feedback: bool = False
    """FBT axis: gated entry plus a payload for the next column."""

    soft: bool = False
    """DF-soft: both machineries as free choices — plain entry, standing
    [p_prev, e] sources, a null source in every router."""

    @property
    def routing_active(self) -> bool:
        return self.depth_routing or self.soft

    @property
    def feedback_active(self) -> bool:
        return self.feedback or self.soft

    @property
    def gated_entry(self) -> bool:
        """Whether the payload enters through the mandatory GLU fuse."""
        return self.feedback and not self.soft


def arm_config(arm: str, **overrides) -> ModelConfig:
    """The named arm's configuration; overrides adjust trunk geometry only."""
    flags = {
        "vanilla": {},
        "dar": {"depth_routing": True},
        "fbt": {"feedback": True},
        "df": {"depth_routing": True, "feedback": True},
        "df_soft": {"soft": True},
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
    """Preallocated per-layer key/value cache for sequential decoding.

    All layers write at the same position range; the caller advances the
    shared position once per column via :meth:`advance`.
    """

    def __init__(self, cfg: ModelConfig, batch: int, device, dtype):
        shape = (cfg.layers, batch, cfg.kv_heads, cfg.max_seq_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0

    def update(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write this column's k/v [B, kvH, T, hd]; return the full prefix view."""
        length = k.shape[2]
        self.k[layer, :, :, self.pos : self.pos + length] = k
        self.v[layer, :, :, self.pos : self.pos + length] = v
        return (
            self.k[layer, :, :, : self.pos + length],
            self.v[layer, :, :, : self.pos + length],
        )

    def advance(self, length: int) -> None:
        self.pos += length

    def reset(self) -> None:
        self.pos = 0


# -- trunk modules -------------------------------------------------------------


class Attention(nn.Module):
    """Qwen3-style GQA with per-head QK RMSNorm and rotary positions."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.dim, cfg.heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.heads * cfg.head_dim, cfg.dim, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: KVCache | None,
        layer: int,
    ) -> Tensor:
        batch, length, _ = x.shape
        cfg = self.cfg
        q = self.q_proj(x).view(batch, length, cfg.heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)

        causal = True
        if cache is not None:
            # SDPA's is_causal aligns top-left, which is only correct for a
            # fresh prefill; appended chunks must be single columns.
            if length > 1 and cache.pos != 0:
                raise ValueError("multi-column append to a non-empty cache")
            k, v = cache.update(layer, k, v)
            # A single decoded column attends to the whole cached prefix.
            causal = length > 1

        groups = cfg.heads // cfg.kv_heads
        if groups > 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).reshape(batch, length, cfg.heads * cfg.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.intermediate, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.intermediate, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate, cfg.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    """One DAR routing site: zero-init query, RMS-normed keys, raw values.

    With a null source (DF-soft), a learnable zero-init vector is
    prepended — its key is rmsnorm(0)=0, so its logit is exactly zero at
    init and routing mass on it adds (initially) nothing.  A per-source
    presence mask (True = present at that position) supports DF-soft's
    prefix mixin, where the p_prev source exists only at fused positions.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(cfg.dim))
        self.key_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.null = nn.Parameter(torch.zeros(cfg.dim)) if cfg.soft else None

    def forward(
        self,
        sources: list[Tensor],
        masks: list[Tensor | None],
        want_weights: bool,
    ) -> tuple[Tensor | None, Tensor | None]:
        """(routed addition [B,T,D] or None if inactive, weights [N,B,T] or None)."""
        if self.null is not None:
            sources = [self.null.expand_as(sources[0])] + sources
            masks = [None] + masks
        if len(sources) < 2:
            return None, None
        values = torch.stack(sources)  # [N, B, T, D]
        logits = torch.einsum("d,nbtd->nbt", self.query, self.key_norm(values))
        if any(mask is not None for mask in masks):
            present = torch.stack(
                [
                    mask
                    if mask is not None
                    else torch.ones_like(logits[0], dtype=torch.bool)
                    for mask in masks
                ]
            )
            logits = logits.masked_fill(~present, float("-inf"))
        weights = logits.softmax(dim=0)
        routed = torch.einsum("nbt,nbtd->btd", weights, values)
        return routed, (weights.detach() if want_weights else None)


class Block(nn.Module):
    """One layer: (routed read →) attention, (routed read →) MLP.

    The routed read enriches the sublayer's pre-norm input only; the
    residual stream accumulates just the scaled branch outputs, which are
    also the deltas appended to the source list.
    """

    def __init__(self, cfg: ModelConfig, layer: int):
        super().__init__()
        self.layer = layer
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp = SwiGLU(cfg)
        self.branch_scale = 1.0 / math.sqrt(2 * cfg.layers)
        if cfg.routing_active:
            self.attn_router = Router(cfg)
            self.mlp_router = Router(cfg)
        else:
            self.attn_router = None
            self.mlp_router = None

    def _read(self, h, router, sources, masks, want_weights, weights_out, site):
        if router is None:
            return h
        routed, weights = router(sources, masks, want_weights)
        if weights is not None:
            weights_out[site] = weights
        return h if routed is None else h + routed

    def forward(
        self,
        h: Tensor,
        sources: list[Tensor] | None,
        masks: list[Tensor | None] | None,
        cos: Tensor,
        sin: Tensor,
        cache: KVCache | None,
        want_weights: bool,
        weights_out: dict[str, Tensor],
    ) -> Tensor:
        x = self._read(
            h, self.attn_router, sources, masks, want_weights, weights_out,
            f"L{self.layer}.attn",
        )
        a = self.branch_scale * self.attn(self.attn_norm(x), cos, sin, cache, self.layer)
        h = h + a
        if sources is not None:
            sources.append(a)
            masks.append(None)

        x = self._read(
            h, self.mlp_router, sources, masks, want_weights, weights_out,
            f"L{self.layer}.mlp",
        )
        m = self.branch_scale * self.mlp(self.mlp_norm(x))
        h = h + m
        if sources is not None:
            sources.append(m)
            masks.append(None)
        return h


# -- the model -----------------------------------------------------------------


@dataclass
class ColumnOutput:
    """One full-stack pass over a column range."""

    h_top: Tensor
    """Top-of-stack residual stream [B, T, D], pre final norm."""

    payload: Tensor | None
    """What rides to the next column (feedback arms), [B, T, D]."""

    route_weights: dict[str, Tensor]
    """Site → softmax weights [N, B, T]; populated when requested."""

    sources: list[Tensor] | None
    """The routed source list (seeds then deltas); None off the routing
    arms.  References into the forward graph, so effectively free — the
    stream at any depth reconstructs as seed + cumsum(deltas)."""

    n_seeds: int
    """How many leading entries of ``sources`` are seeds, not deltas."""


class DFModel(nn.Module):
    """The full model; parents are deletions per the config flags."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.layers))
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
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
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, Router):
            nn.init.zeros_(module.query)
            if module.null is not None:
                nn.init.zeros_(module.null)

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
        p_source: Tensor | None = None,
        p_presence: Tensor | None = None,
        cache: KVCache | None = None,
        want_weights: bool = False,
    ) -> ColumnOutput:
        """Run the stack once over inputs x [B, T, D].

        x is whatever the columns' input actually is: plain embeddings on
        pass 1 and Standard decoding, the fused u for spine feedback
        passes, always plain e for DF-soft.  p_source is DF-soft's
        standing previous-column payload (already shifted); p_presence
        [B, T] marks the positions where it exists (prefix-mixin rows are
        absent).  With a cache, positions start at cache.pos and the
        cache is advanced by T.
        """
        cfg = self.cfg
        start = cache.pos if cache is not None else 0
        cos, sin = self.rope(x.device, start, x.shape[1])

        sources: list[Tensor] | None = None
        masks: list[Tensor | None] | None = None
        if cfg.routing_active:
            sources, masks = [], []
            if p_source is not None:
                sources.append(p_source)
                masks.append(p_presence)
            sources.append(x)
            masks.append(None)
        seeds = len(sources) if sources is not None else 0

        h = x
        weights_out: dict[str, Tensor] = {}
        for block in self.blocks:
            h = block(h, sources, masks, cos, sin, cache, want_weights, weights_out)
        if cache is not None:
            cache.advance(x.shape[1])

        payload = None
        if cfg.feedback_active:
            enriched = h
            if cfg.routing_active:
                deltas = sources[seeds:]
                routed, weights = self.payload_router(
                    deltas, [None] * len(deltas), want_weights
                )
                if weights is not None:
                    weights_out["payload"] = weights
                if routed is not None:
                    enriched = h + routed
            payload = self.payload_norm(enriched)

        return ColumnOutput(
            h_top=h,
            payload=payload,
            route_weights=weights_out,
            sources=sources,
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
        if payload is not None and self.cfg.gated_entry:
            x, p_source = self.fuse(payload, e), None
        elif payload is not None:
            x, p_source = e, payload
        else:
            x, p_source = e, None
        return self.forward_column(
            x, p_source=p_source, cache=cache, want_weights=want_weights
        )


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
    """FBT's Jacobi multi-pass forward (Appendix C), arm-agnostic.

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
    out = model.forward_column(e, want_weights=want_weights)
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
        if cfg.soft:
            out = model.forward_column(
                e, p_source=p_shifted, p_presence=~plain, want_weights=want_weights
            )
        else:
            fused = model.fuse(p_shifted, e)
            out = model.forward_column(
                torch.where(plain[..., None], e, fused), want_weights=want_weights
            )
        outs.append(out)
    return outs


def multipass_loss(
    model: DFModel, tokens: Tensor, outs: list[ColumnOutput]
) -> tuple[Tensor, list[Tensor]]:
    """FBT Eq. 12 with λ=1: pass-1 NTP plus the mean over feedback passes.

    Returns (total, per-pass losses); per-pass loss 0 is the Standard-mode
    tracking metric.
    """
    targets = tokens[:, 1:]
    losses = []
    for out in outs:
        logits = model.logits(out.h_top[:, :-1])
        losses.append(
            F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        )
    total = losses[0]
    if len(losses) > 1:
        total = total + torch.stack(losses[1:]).mean()
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
    batch, length = tokens.shape
    e = model.embed_tokens(tokens)
    positions = torch.arange(length, device=tokens.device)
    plain = (positions[None, :] < 1).expand(batch, -1)

    out = model.forward_column(e)
    records = []
    for _ in range(n_iters):
        previous = out.h_top
        p_shifted = shift_right(out.payload)
        if cfg.soft:
            out = model.forward_column(e, p_source=p_shifted, p_presence=~plain)
        else:
            fused = model.fuse(p_shifted, e)
            out = model.forward_column(torch.where(plain[..., None], e, fused))
        loss = F.cross_entropy(
            model.logits(out.h_top[:, :-1]).flatten(0, 1).float(),
            tokens[:, 1:].flatten(),
        )
        delta = (out.h_top - previous).float().norm(dim=-1).mean()
        records.append({"loss": loss.item(), "update_norm": delta.item()})
    return records

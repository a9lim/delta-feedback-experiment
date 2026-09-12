"""The Delta model: a routed PKDA/MoE trunk with two recurrence choices.

The architecture contract is ``docs/architecture.md``. PKDA occupies three
layers of each four-layer cell, with NoPE gated GQA in the fourth, MHDB reads
before every sublayer, MoE channel mixers, and sequential two-token prediction.
``f`` adds FBT feedback; ``l`` ties the middle cells into a core iterated per
column. At least one recurrence must be selected.
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
from dataclasses import dataclass

import torch
import torch._dynamo.config as _dynamo_config
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

from . import INDUCTOR_MODE
from .attention import causal_attention, prefix_attention
from .cuda_kernels import ShadowOperand, bespoke_route, sink_linear
from .moe import EXPERT_BIAS_RATE, MixtureOfExperts, validate_expert_geometry
from .parameter_groups import is_normuonh_parameter, is_width_scaled_parameter
from .pkda import PreconditionedKDA
from .tokenizer import VOCAB_SIZE

# Model blocks specialize by layer, route count, mode, and gradient state.
# Configure their finite compiler budget only when loading the model.
_dynamo_config.recompile_limit = max(_dynamo_config.recompile_limit, 256)
_dynamo_config.accumulated_recompile_limit = max(
    _dynamo_config.accumulated_recompile_limit, 4096
)

try:
    from cut_cross_entropy.cce import CCEParams, linear_cross_entropy_apply
    from cut_cross_entropy.utils import _handle_eps
except (ImportError, OSError):  # pragma: no cover - exercised on Jobe
    CCEParams = None
    linear_cross_entropy_apply = None
    _handle_eps = None

CONDITION_LETTERS: dict[str, tuple[str, str]] = {
    "f": (
        "feedback",
        "full-bandwidth feedback: consume the preceding column's payload at the FBT entry",
    ),
    "l": (
        "loop",
        "Huginn loop: the middle cells become one tied core iterated per column",
    ),
}
"""Letter -> (``ModelConfig`` flag, one-line change), in canonical order."""


def parse_condition(text: str) -> str:
    """Accept ``f``, ``l``, or both in either order; reject an empty condition."""
    if not text:
        raise ValueError("condition must contain f, l, or both")
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
"""Base Gaussian standard deviation for NAdam-owned matrices.

Embeddings and fixed-head-width expansions use this directly. Gate and control
matrices with fan-in ``D`` multiply it by ``sqrt(MUP_BASE_DIM / D)``. Adjust
this constant to tune both families; NorMuonH's fan-in scale is independent.
"""

MUP_BASE_DIM = 1536
"""The fixed muP reference width: the flagship column of ``docs/scaling.md``.

NorMuonH matrices use spectral tangent steps with their own fan-in/fan-out
factor. Both expert projections also use the active-count scale below.
NAdam's per-coordinate step is approximately its learning rate,
so a matrix with fan-in
``D`` moves its output by up to ``rate * D`` per step. The fan-in-``D`` NAdam
matrices therefore run at ``lr_nadam * MUP_BASE_DIM / dim`` and initialize at
``BASE_NORMAL_INIT_STD * sqrt(MUP_BASE_DIM / dim)``. The tied readout
multiplies its logits by the width ratio, so the rate tuned at the
flagship defines the rates at every other geometry, and the flagship itself
is the plain parametrization. Extension keeps this reference and uses 2/3.
"""

MUP_BASE_ACTIVE_EXPERTS = 8
"""Flagship's one shared plus seven selected experts, for expert branch rates."""

EXPERT_BALANCE_COEF = 0.0001
"""Weak sequence-balance coefficient, averaged over executed layers and passes."""

MTP_LOSS_WEIGHT = 0.3
"""Default weight of the one-token-ahead auxiliary prediction objective."""


@dataclass(frozen=True)
class ModelConfig:
    """Permanent PKDA/MoE/MHDB/MTP geometry plus two recurrence flags.

    Routing uses one contiguous feature group per KV head. These groups do
    not align to mixer projections. The default condition is feedback (f).
    """

    vocab_size: int = VOCAB_SIZE
    dim: int = 768
    layers: int = 16
    heads: int = 8
    kv_heads: int = 4
    head_dim: int = 96
    expert_intermediate: int = 832
    num_routed_experts: int = 15
    experts_per_token: int = 3
    pkda_heads: int = 10
    pkda_head_dim: int = 128
    pkda_conv_size: int = 4
    max_seq_len: int = 4096
    norm_eps: float = 1e-6
    routing_block_size: int = 4
    """Exact MHDB cell width in transformer layers."""

    feedback: bool = True
    """``f``: consume the preceding column's payload through the FBT gated entry."""

    loop: bool = False
    """``l``: the cells between the first and last become one tied core that
    runs ``iterations`` times per column, each core cell keeping its own block
    delta across iterations (``docs/architecture.md``)."""

    loop_iterations: int = 4
    """``l``: mean of the per-step iteration draw, and the fixed count that
    evaluation and decoding use."""

    loop_max_iterations: int = 8
    """``l``: cap of the per-step iteration draw."""

    def __post_init__(self) -> None:
        if not (self.feedback or self.loop):
            raise ValueError("condition must enable feedback (f), looping (l), or both")
        validate_expert_geometry(
            self.expert_intermediate, self.num_routed_experts, self.experts_per_token
        )
        if self.layers < 1:
            raise ValueError("model needs at least one layer")
        if self.routing_block_size < 1:
            raise ValueError("routing block size must be positive")
        if self.loop_iterations < 1 or self.loop_max_iterations < self.loop_iterations:
            raise ValueError("loop iterations must satisfy 1 <= mean <= max")
        if self.loop and (
            self.layers % self.routing_block_size
            or self.layers < 3 * self.routing_block_size
        ):
            raise ValueError(
                "l needs whole cells and at least three of them: "
                "a prelude, a core, and a coda"
            )
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
    def mup_ratio(self) -> float:
        """The muP width ratio ``MUP_BASE_DIM / dim``; one at the flagship."""
        return MUP_BASE_DIM / self.dim

    @property
    def expert_lr_scale(self) -> float:
        """Operator-step multiplier for shared + selected expert aggregation.

        It applies to gate/up and down projections; matrix aspect ratio is
        already handled by the optimizer's spectral normalization.
        """
        return math.sqrt(MUP_BASE_ACTIVE_EXPERTS / (self.experts_per_token + 1))

    @property
    def routing_blocks(self) -> int:
        """Number of completed block deltas emitted by a full column: one per
        cell, looped or not."""
        return (self.layers + self.routing_block_size - 1) // self.routing_block_size

    @property
    def core_layers(self) -> range:
        """``l``: the tied core, every layer between the first and last cells."""
        return range(self.routing_block_size, self.layers - self.routing_block_size)

    def is_core_layer(self, layer: int) -> bool:
        return self.loop and layer in self.core_layers

    def executed_layers(self, iterations: int) -> int:
        """Layers one column pass evaluates at ``iterations`` core iterations."""
        if not self.loop:
            return self.layers
        return 2 * self.routing_block_size + iterations * len(self.core_layers)

    def resolve_iterations(self, iterations: int | None) -> int:
        """The core iteration count a column runs; ``None`` is the default."""
        if not self.loop:
            if iterations not in (None, 1):
                raise ValueError("a condition without l runs the core once")
            return 1
        if iterations is None:
            return self.loop_iterations
        if not 1 <= iterations <= self.loop_max_iterations:
            raise ValueError(
                f"iterations {iterations} outside 1..{self.loop_max_iterations}"
            )
        return iterations

    def is_pkda_layer(self, layer: int) -> bool:
        return layer % 4 != 3

    @property
    def global_attention_layers(self) -> tuple[int, ...]:
        """Dense NoPE gated GQA layers: the fourth layer of each cell."""
        return tuple(
            layer for layer in range(self.layers) if not self.is_pkda_layer(layer)
        )


def condition_config(condition: str, **overrides) -> ModelConfig:
    """The configuration a condition string names; overrides adjust geometry."""
    letters = parse_condition(condition)
    flags = {flag: letter in letters for letter, (flag, _) in CONDITION_LETTERS.items()}
    return ModelConfig(**(overrides | flags))


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


# -- kv cache ------------------------------------------------------------------


class KVCache:
    """Hybrid decoding cache with dense KV and fixed-size PKDA states.

    Only global-attention layers receive KV slots. PKDA layers retain their
    FP32 matrix/preconditioner states and three short-convolution histories.
    The caller advances the shared position once per column. Under ``l`` a
    core layer owns one track per iteration, because iteration ``i`` of a
    column mixes over the earlier columns' iteration-``i`` writes;
    ``forward_column`` selects the track through ``iteration`` before each
    core iteration and every track shares the column position.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        batch: int,
        device,
        dtype,
        iterations: int | None = None,
    ):
        self.cfg = cfg
        self.iterations = cfg.resolve_iterations(iterations)
        self.iteration = 0
        self.global_slots: dict[tuple[int, int], int] = {}
        for layer in cfg.global_attention_layers:
            tracks = self.iterations if cfg.is_core_layer(layer) else 1
            for track in range(tracks):
                self.global_slots[layer, track] = len(self.global_slots)
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
            tuple[int, int],
            tuple[
                Tensor,
                Tensor,
                tuple[Tensor, Tensor, Tensor],
            ],
        ] = {}
        self.pos = 0

    def _track(self, layer: int) -> tuple[int, int]:
        return layer, (self.iteration if self.cfg.is_core_layer(layer) else 0)

    def update(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write k/v [B,T,Hkv,D]; return the full prefix in that layout."""
        slot = self.global_slots[self._track(layer)]
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
        return self.pkda_states.get(self._track(layer), (None, None, None))

    def update_pkda(
        self,
        layer: int,
        state: Tensor,
        a_state: Tensor,
        conv_state: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        self.pkda_states[self._track(layer)] = (state, a_state, conv_state)

    def advance(self, length: int) -> None:
        self.pos += length


# -- trunk modules -------------------------------------------------------------


class Attention(nn.Module):
    """Dense causal NoPE gated GQA."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.q_size = cfg.heads * cfg.head_dim
        self.kv_size = cfg.kv_heads * cfg.head_dim
        self.qkv_proj = nn.Linear(cfg.dim, self.q_size + 2 * self.kv_size, bias=False)
        self.o_proj = nn.Linear(cfg.heads * cfg.head_dim, cfg.dim, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.qkv_sink: Tensor | None = None
        self.gate_sink: Tensor | None = None
        self.packed_qkv_sink: Tensor | None = None
        self.o_sink: Tensor | None = None
        self.qkv_shadow: Tensor | None = None
        self.o_shadow: Tensor | None = None

    def forward(
        self,
        x: Tensor,
        gate_weight: Tensor,
        cache: KVCache | None,
        layer: int,
    ) -> Tensor:
        batch, length, _ = x.shape
        cfg = self.cfg
        # One GEMM produces Q/K/V and the gate logits; the gate keeps its own
        # parameter, optimizer group, and checkpoint name.
        q, k, v, gate_logits = sink_linear(
            x,
            (self.qkv_proj.weight, gate_weight),
            (self.qkv_sink, self.gate_sink),
            self.qkv_shadow,
            packed_sink=self.packed_qkv_sink,
        ).split((self.q_size, self.kv_size, self.kv_size, self.q_size), dim=-1)
        q = q.view(batch, length, cfg.heads, cfg.head_dim).transpose(1, 2)
        k = k.view(batch, length, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(batch, length, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
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
                out = causal_attention(q, k, v).transpose(1, 2)
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
        out = out * torch.sigmoid(gate_logits)
        return sink_linear(out, (self.o_proj.weight,), (self.o_sink,), self.o_shadow)


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
    weights = logits.softmax(dim=0)
    routed = (
        (weights.to(values.dtype).unsqueeze(-1) * value_heads)
        .sum(dim=0)
        .reshape(batch, length, dim)
    )
    return routed, weights


class _BankedSource(torch.autograd.Function):
    """Give a routing source one gradient accumulator for all of its readers.

    Each source is read by multiple routing sites. The source is handed out
    through an alias whose readers add their contribution straight into
    ``accumulator`` and return no gradient, and this backward passes the
    finished accumulator on as the source's gradient.

    The residual stream rides through as ``carrier`` so the node is on the
    stream's own path: its backward therefore runs, and it runs only once every
    reader of the alias has, which is exactly when the accumulator is complete.
    A reader that is not banked still returns an ordinary gradient, which arrives here as ``grad_alias`` and is
    added; correctness never depends on who banks.
    """

    @staticmethod
    def forward(
        ctx, carrier: Tensor, source: Tensor, accumulator: Tensor
    ) -> tuple[Tensor, Tensor]:
        ctx.set_materialize_grads(False)
        ctx.accumulator = accumulator
        return carrier.view_as(carrier), source.view_as(source)

    @staticmethod
    def backward(ctx, grad_carrier, grad_alias):
        gathered = ctx.accumulator
        if grad_alias is not None:
            gathered = gathered + grad_alias
        return grad_carrier, gathered, None


def bank_source(carrier: Tensor, source: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(carrier, the alias later sites read, the alias's gradient accumulator).

    The accumulator is allocated where the source is born, so under capture it
    is one address in the graph's own pool and one memset per replay.
    """
    accumulator = torch.zeros_like(source)
    carrier, alias = _BankedSource.apply(carrier, source, accumulator)
    return carrier, alias, accumulator


class Router(nn.Module):
    """One MHDB site: per-group softmaxes, RMS-normed keys, raw values.

    A learnable zero-init null vector is always prepended.  Its key is
    rmsnorm(0)=0, so its logit is exactly zero at init and routing mass on it
    initially adds nothing.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        _check_route_geometry(cfg.dim, cfg.kv_heads)
        self.num_heads = cfg.kv_heads
        self.query = nn.Parameter(torch.zeros(cfg.dim))
        self.key_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.null = nn.Parameter(torch.zeros(cfg.dim))

    def forward(
        self,
        sources: list[Tensor],
        want_weights: bool,
        accumulators: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor | None, Tensor | None]:
        """Return a routed addition and weights shaped ``[N,B,T,H]``.

        ``accumulators`` are the gradient accumulators of the leading, banked
        sources; the Triton backward adds into them instead of returning a
        gradient per site.  Every other path ignores them and returns ordinary
        gradients, which the bank then adds.
        """
        if not sources:
            return None, None
        null = self.null.to(sources[0].dtype)
        if sources[0].is_cuda:
            if accumulators and not (
                self.query.requires_grad and self.null.requires_grad
            ):
                # The banked accumulation rides on this site's query and null
                # gradients being live outputs; see the routing backward.
                raise RuntimeError(
                    "a banked routing site needs a trainable query and null"
                )
            projected = (self.query.float() * self.key_norm.weight.float()).to(
                sources[0].dtype
            )
            routed, weights = bespoke_route(
                projected,
                null,
                self.key_norm.eps,
                self.num_heads,
                tuple(sources),
                tuple(accumulators),
            )
        else:
            routed, weights = _route_algebra(
                torch.stack([null.expand_as(sources[0]), *sources]),
                self.query,
                self.key_norm.weight,
                self.key_norm.eps,
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

    def __init__(self, cfg: ModelConfig, layer: int, *, auxiliary: bool = False):
        super().__init__()
        self.layer = layer
        self.is_pkda = auxiliary or cfg.is_pkda_layer(layer)
        self.global_gate_index = (
            None if self.is_pkda or auxiliary else cfg.global_attention_layers.index(layer)
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
        self.mlp = MixtureOfExperts(
            cfg.dim, cfg.expert_intermediate, cfg.num_routed_experts, cfg.experts_per_token
        )
        self.branch_scale = 1.0 / math.sqrt(2 * cfg.layers)
        if not auxiliary:
            self.attn_router = Router(cfg)
            self.mlp_router = Router(cfg)
        else:
            self.attn_router = None
            self.mlp_router = None

    def _read(self, h, router, sources, accumulators, want_weights):
        if router is None or not sources:
            return h, None
        routed, weights = router(list(sources), want_weights, accumulators)
        return (h if routed is None else h + routed), weights

    def forward(
        self,
        h: Tensor,
        block_start: Tensor | None,
        cache: KVCache | None,
        gate_weight: Tensor | None,
        want_weights: bool,
        *sources: Tensor,
        accumulators: tuple[Tensor, ...] = (),
    ) -> tuple[
        Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None,
        Tensor | None,
    ]:
        """Return residual, delta, diagnostic weights, expert loss and counts. The branch
        outputs stay inside: nothing reads them, and a compiled block that
        returned them would both write them and take autograd's materialized
        zero gradient back. ``block_start`` is the residual at the cell's
        entry, or None when this block opens the cell; with an entry the last
        source is the cell's partial so far. ``accumulators`` belong to the
        leading banked sources, which are the same at both of this block's read
        sites: the partial the MLP read replaces, and the attention delta the
        opening block appends, are always last and never banked. The returned cell
        delta is the next sublayer's partial source or, at the cell boundary,
        the completed block delta, computed here so it lands inside the
        compiled block instead of as an eager subtraction. (Passing ``h`` twice
        would make Dynamo guard the inputs against aliasing, and its
        recompile-reason logging then evaluates those guards across block
        instances.)"""
        start = h if block_start is None else block_start
        x, w_attn = self._read(h, self.attn_router, sources, accumulators, want_weights)
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
            mixed = self.attn(normalized, gate_weight, cache, self.layer)
        a = self.branch_scale * mixed
        h = h + a
        if self.mlp_router is not None and block_start is not None:
            mlp_sources = (*sources[:-1], sources[-1] + a)
        else:
            mlp_sources = (*sources, a)
        x, w_mlp = self._read(
            h, self.mlp_router, mlp_sources, accumulators, want_weights
        )
        normalized = self.mlp_norm(x)
        mixed, aux, expert_weights, expert_counts = self.mlp(
            normalized, want_weights=want_weights
        )
        m = self.branch_scale * mixed
        h = h + m
        return h, h - start, w_attn, w_mlp, aux, expert_weights, expert_counts


@dataclass
class MTPOutput:
    """Auxiliary states and expert statistics from one logical forward."""

    hidden: Tensor
    expert_aux_loss: Tensor
    expert_counts: Tensor
    expert_weights: Tensor | None


class MultiTokenPrediction(nn.Module):
    """One sequential prediction depth; embedding and readout belong to the model.

    Position t consumes the payload at t and the embedding of token t+1.
    Its own causal PKDA recurrence over those pairs cannot see target t+2.
    Every invocation starts with fresh state; no trunk cache is consumed.
    The auxiliary PKDA/MoE block has the trunk's geometry and residual scaling
    without MHDB reads or a tied loop.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.payload_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.embedding_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.projection = nn.Linear(2 * cfg.dim, cfg.dim, bias=False)
        self.projection_sink: Tensor | None = None
        self.projection_shadow: Tensor | None = None
        self.block = Block(cfg, 0, auxiliary=True)

    def forward(
        self, payload: Tensor, next_embedding: Tensor, want_weights: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        joined = torch.cat(
            (self.payload_norm(payload), self.embedding_norm(next_embedding)), dim=-1
        )
        x = sink_linear(
            joined, (self.projection.weight,), (self.projection_sink,),
            self.projection_shadow,
        )
        h, _, _, _, aux, weights, counts = self.block(
            x, None, None, None, want_weights
        )
        return h, aux, counts, weights


def _mtp_forward(module, payload, next_embedding, want_weights):
    return module(payload, next_embedding, want_weights)


_compiled_mtp = torch.compile(
    _mtp_forward, fullgraph=True, dynamic=False, mode=INDUCTOR_MODE
)


def _payload_epilogue(norm, h, routed):
    """The column's payload: the routed read added to the top state, normalized."""
    return norm(h + routed)


_compiled_payload_epilogue = torch.compile(
    _payload_epilogue, fullgraph=True, dynamic=False, mode=INDUCTOR_MODE
)


def _block_for_checkpoint(block, h, block_start, gate_weight, banked, *tensors):
    """Pure checkpoint wrapper with a positional source-bank interface.

    ``banked`` counts the accumulators that follow the sources in ``tensors``:
    the flat signature keeps activation checkpointing and Dynamo's specialized
    call convention positional.
    """
    sources = tensors[: len(tensors) - banked]
    h, delta, _, _, aux, _, counts = block(
        h,
        block_start,
        None,
        gate_weight,
        False,
        *sources,
        accumulators=tensors[len(sources) :],
    )
    return h, delta, aux, counts


# Each block family compiles through its own code object. Dynamo keys its
# cache on the code object and, when it recompiles, evaluates every earlier
# entry's guards against the current call to log the reason; one family's
# guards name attributes the other family's mixer does not have.
def _attention_block(block, h, block_start, gate_weight, banked, *tensors):
    return _block_for_checkpoint(block, h, block_start, gate_weight, banked, *tensors)


def _pkda_block(block, h, block_start, gate_weight, banked, *tensors):
    return _block_for_checkpoint(block, h, block_start, gate_weight, banked, *tensors)


def _retain_block_activations(layer: int, cell_size: int) -> bool:
    """Retain the final block of each cell when the full column exceeds memory.

    Each cell retains its final global-attention block.
    """
    return layer % cell_size == cell_size - 1


_compiled_block = torch.compile(
    _attention_block,
    fullgraph=True,
    dynamic=False,
    # Each training geometry reuses its durable compilation cache. The outer
    # trainer owns CUDA capture.
    mode=INDUCTOR_MODE,
)

_compiled_pkda_block = torch.compile(
    _pkda_block,
    # FLA's kernels are wrapped as custom operators in ``fla_ops``, so the
    # recurrence stays its own kernel boundary without breaking the graph:
    # the projections, controls, norm/gate, MLP, residual updates, and both
    # routed reads compile and fuse as one graph, and every input reaches
    # autograd through exactly one backward.
    fullgraph=True,
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
    """Shared predictive payload [B,T,D], consumed by MTP and, with f, feedback.
    None only when the caller explicitly skips the payload read."""

    route_weights: dict[str, Tensor]
    """Site → per-head softmax weights [N, B, T, H], when requested."""

    route_source_names: dict[str, tuple[str, ...]]
    """Site → source-axis labels matching ``route_weights`` exactly."""

    sources: list[Tensor]
    """Final stored bank: the seed followed by completed block deltas.
    The stream reconstructs from that seed plus the block deltas."""

    source_names: tuple[str, ...]
    """Names corresponding one-for-one with ``sources``."""

    n_seeds: int
    """How many leading entries of ``sources`` are seeds, not deltas."""

    iterations: int
    """Core iterations this column ran; 1 without ``l``."""

    core_entry: Tensor | None
    """``l``: the residual at core entry, the prelude output [B, T, D]."""

    core_state: Tensor | None
    """``l``: the residual at core exit after the last iteration [B, T, D]."""

    expert_aux_loss: Tensor
    """Mean load-balance loss over executed layer invocations."""

    expert_weights: dict[str, Tensor]
    """Site -> sparse normalized top-k routing weights [B, T, routed experts]."""

    expert_counts: Tensor
    """Assignment counts [physical layers, routed experts], summed over core uses."""

    next_embedding: Tensor | None = None
    """[B, T, D] embeddings of the stored row's positions 1..T: the auxiliary
    head's next-token input, taken from the same lookup as the column seed.
    ``multipass`` attaches it to every pass; a bare column leaves it None."""


class DeltaModel(nn.Module):
    """The permanent trunk with feedback and/or tied-depth recurrence."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        factor_seed = torch.initial_seed()
        self.embed_tokens = ResidualEmbedding(cfg.vocab_size, cfg.dim)
        self.register_buffer("_classifier_shadow", None, persistent=False)
        self.register_buffer("_classifier_accum", None, persistent=False)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.layers))
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.payload_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.payload_router = Router(cfg)
        # Capture the exact common-trunk initialization boundary before
        # constructing the attention gates and the letter-private matrices,
        # which draw from their own streams. Restoring it below keeps every
        # shared parameter byte-identical across conditions.
        common_init_state = torch.random.get_rng_state()
        self.attention_gates = nn.ModuleList(
            nn.Linear(cfg.dim, cfg.heads * cfg.head_dim, bias=False)
            for _ in cfg.global_attention_layers
        )
        self.fuse_value_sink: Tensor | None = None
        self.fuse_gate_sink: Tensor | None = None
        self.fuse_value_shadow: Tensor | None = None
        self.fuse_gate_shadow: Tensor | None = None
        self._shadow_refresh: list[tuple[Tensor, Tensor]] = []
        if cfg.feedback:
            self.fuse_value = nn.Linear(cfg.dim, cfg.dim, bias=False)
            self.fuse_gate = nn.Linear(cfg.dim, cfg.dim, bias=False)
            self.gate_norm = RMSNorm(cfg.dim, cfg.norm_eps)
            self.entry_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.grad_checkpoint = False
        """Runtime switch: retain cell-final activations, checkpoint other blocks."""
        self.bank_sources = True
        """Runtime switch: give each routed source one gradient accumulator.

        Off, every site returns its own gradient for every source and autograd
        sums them. This portable path is the reference for checking banked
        accumulation."""
        torch.random.set_rng_state(common_init_state)
        self.embed_tokens.apply(self._init_weights)
        self.blocks.apply(self._init_weights)
        self.final_norm.apply(self._init_weights)
        self.payload_norm.apply(self._init_weights)
        self.payload_router.apply(self._init_weights)
        self._init_factor_linears(
            self.attention_gates, factor_seed ^ 0x4152434849544543
        )
        if cfg.feedback:
            self._init_factor_linears(
                (self.fuse_value, self.fuse_gate),
                factor_seed ^ 0x524543555252454E,
            )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                (factor_seed ^ 0x4D54505F44455054) % ((1 << 63) - 1)
            )
            self.mtp = MultiTokenPrediction(cfg)
            self.mtp.apply(self._init_weights)
        self._scale_initialization()

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
        """Initialize one module family from its own seed-stable random stream.

        The attention gates and the letter-private FBT matrices never advance
        the common trunk stream, so a module initializes identically in every
        condition that has it. Models are constructed on CPU (or meta for
        accounting) before moving to an execution device, so one local CPU
        generator is authoritative.
        """
        generator = torch.Generator().manual_seed(seed % ((1 << 63) - 1))
        for linear in linears:
            nn.init.normal_(
                linear.weight, std=BASE_NORMAL_INIT_STD, generator=generator
            )

    @torch.no_grad()
    def _scale_initialization(self) -> None:
        """Apply NorMuonH fan-in and NAdam gate/control width scaling.

        Scaling the already sampled normal values is distributionally identical
        to drawing with the target standard deviation. It also preserves the
        common and factor-private paired random streams exactly.
        """
        width_scale = math.sqrt(self.cfg.mup_ratio)
        for name, parameter in self.named_parameters():
            if is_normuonh_parameter(name, parameter):
                target_std = 1 / math.sqrt(parameter.shape[1])
                parameter.mul_(target_std / BASE_NORMAL_INIT_STD)
            elif is_width_scaled_parameter(name, parameter):
                parameter.mul_(width_scale)

    # -- pieces ----------------------------------------------------------------

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

    def forward_mtp(
        self, payload: Tensor, next_tokens: Tensor, *, want_weights: bool = False
    ) -> MTPOutput:
        """Predict from payloads and supplied next tokens, returning expert stats."""
        if next_tokens.shape != payload.shape[:2]:
            raise ValueError("MTP needs next tokens aligned with the payloads")
        return self.forward_mtp_embedded(
            payload, self.embed_tokens(next_tokens), want_weights=want_weights
        )

    def forward_mtp_embedded(
        self, payload: Tensor, embedding: Tensor, *, want_weights: bool = False
    ) -> MTPOutput:
        """The same prediction from already-looked-up next-token embeddings.

        The training objective embeds the whole stored row once and hands both
        heads their slice of it, so the auxiliary head costs no second lookup
        and no second scatter into the embedding gradient.
        """
        if (
            payload.ndim != 3
            or embedding.shape[:2] != payload.shape[:2]
            or not payload.shape[1]
        ):
            raise ValueError("MTP needs aligned nonempty payloads and embeddings")
        fn = _compiled_mtp if payload.is_cuda else _mtp_forward
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            result = torch.utils.checkpoint.checkpoint(
                fn, self.mtp, payload, embedding, want_weights,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            result = fn(self.mtp, payload, embedding, want_weights)
        return MTPOutput(*result)

    @property
    def expert_banks(self) -> tuple[MixtureOfExperts, ...]:
        """Physical expert banks in count order: trunk layers, then MTP."""
        return (*(block.mlp for block in self.blocks), self.mtp.block.mlp)

    def _parameter_blocks(self):
        """Physical trunk and auxiliary blocks with their separately owned gates."""
        for block in self.blocks:
            gate = (
                None if block.is_pkda
                else self.attention_gates[block.global_gate_index].weight
            )
            yield block, gate
        yield self.mtp.block, None

    # -- persistent gradient sinks -------------------------------------------

    def allocate_gradient_buffers(self) -> dict[nn.Parameter, Tensor]:
        """Allocate FP32 gradients, packing projections that share a GEMM.

        Q/K/V and dense QKV/gate retain separate parameters and optimizer
        state. Their gradients are disjoint contiguous row views of one
        backing allocation, so the backward can accumulate the whole GEMM
        without materializing or combining per-parameter gradients.
        """
        buffers: dict[nn.Parameter, Tensor] = {}
        for block, gate in self._parameter_blocks():
            attn = block.attn
            if block.is_pkda:
                parameters = (
                    attn.q_proj.weight,
                    attn.k_proj.weight,
                    attn.v_proj.weight,
                )
            else:
                parameters = (
                    attn.qkv_proj.weight,
                    gate,
                )
            if not all(parameter.requires_grad for parameter in parameters):
                continue
            slab = torch.zeros(
                (
                    sum(parameter.shape[0] for parameter in parameters),
                    parameters[0].shape[1],
                ),
                device=parameters[0].device,
                dtype=torch.float32,
            )
            start = 0
            for parameter in parameters:
                rows = parameter.shape[0]
                buffers[parameter] = slab[start : start + rows]
                start += rows
        for parameter in self.parameters():
            if parameter.requires_grad and parameter not in buffers:
                buffers[parameter] = torch.zeros_like(parameter, dtype=torch.float32)
        return buffers

    @torch.no_grad()
    def update_expert_bias(
        self, counts: Tensor, *, rate: float = EXPERT_BIAS_RATE
    ) -> None:
        """Update each physical expert bank once after a complete training step."""
        if not self.training:
            return
        if counts.shape != (len(self.expert_banks), self.cfg.num_routed_experts):
            raise ValueError(
                "expert counts must have shape [trunk layers + MTP, routed experts]"
            )
        for bank, bank_counts in zip(self.expert_banks, counts, strict=True):
            bank.update_bias(bank_counts, rate=rate)

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

        Buffers from ``allocate_gradient_buffers`` also expose a packed sink
        for each concatenated projection. Ordinary independently allocated
        buffers remain valid; their backward accumulates each segment alone.
        """
        lookup = sinks or {}
        bound: set[nn.Parameter] = set()
        refresh: list[tuple[Tensor, Tensor]] = []

        def sink(parameter: nn.Parameter) -> Tensor | None:
            buffer = lookup.get(parameter)
            if buffer is not None:
                bound.add(parameter)
            return buffer

        def packed_sink(
            current: Tensor | None, *parameters: nn.Parameter
        ) -> Tensor | None:
            parts = [lookup.get(parameter) for parameter in parameters]
            if any(part is None for part in parts):
                return None
            first = parts[0]
            columns = parameters[0].shape[1]
            offset = first.storage_offset()
            base_address = first.untyped_storage().data_ptr()
            for parameter, part in zip(parameters, parts, strict=True):
                if (
                    part.shape != parameter.shape
                    or part.dtype != torch.float32
                    or part.device != first.device
                    or not part.is_contiguous()
                    or part.untyped_storage().data_ptr() != base_address
                    or part.storage_offset() != offset
                ):
                    return None
                offset += part.numel()
            shape = (sum(parameter.shape[0] for parameter in parameters), columns)
            if (
                current is not None
                and current.shape == shape
                and current.dtype == first.dtype
                and current.device == first.device
                and current.untyped_storage().data_ptr() == base_address
                and current.storage_offset() == first.storage_offset()
            ):
                return current
            return first.as_strided(shape, (columns, 1))

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
        if self.embed_tokens.grad_sink is None:
            # The classifier accumulator drains into that sink; without one it
            # would strand the head's gradient.
            self._classifier_accum = None
        for block, gate in self._parameter_blocks():
            expert_bound, expert_refresh = block.mlp.bind_gradient_sinks(sinks)
            bound.update(expert_bound)
            refresh.extend(expert_refresh)
            attn = block.attn
            if block.is_pkda:
                attn.q_sink = sink(attn.q_proj.weight)
                attn.k_sink = sink(attn.k_proj.weight)
                attn.v_sink = sink(attn.v_proj.weight)
                attn.packed_qkv_sink = packed_sink(
                    attn.packed_qkv_sink,
                    attn.q_proj.weight,
                    attn.k_proj.weight,
                    attn.v_proj.weight,
                )
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
                attn.gate_sink = sink(gate)
                attn.packed_qkv_sink = packed_sink(
                    attn.packed_qkv_sink, attn.qkv_proj.weight, gate
                )
                attn.o_sink = sink(attn.o_proj.weight)
                attn.qkv_shadow = shadow(attn.qkv_shadow, attn.qkv_proj.weight, gate)
                attn.o_shadow = shadow(attn.o_shadow, attn.o_proj.weight)
        if self.cfg.feedback:
            self.fuse_value_sink = sink(self.fuse_value.weight)
            self.fuse_gate_sink = sink(self.fuse_gate.weight)
            self.fuse_value_shadow = shadow(
                self.fuse_value_shadow, self.fuse_value.weight
            )
            self.fuse_gate_shadow = shadow(self.fuse_gate_shadow, self.fuse_gate.weight)
        self.mtp.projection_sink = sink(self.mtp.projection.weight)
        self.mtp.projection_shadow = shadow(
            self.mtp.projection_shadow, self.mtp.projection.weight
        )
        self._shadow_refresh = refresh
        return bound

    def readout_input(self, h_top: Tensor) -> Tensor:
        """``final_norm(h_top)`` under the muP readout multiplier.

        The tied embedding is an NAdam parameter whose readout role has fan-in
        ``D``, so its logits carry the width ratio here; a width-scaled rate
        would also move its lookup role, whose fan-in is one.
        """
        normalized = self.final_norm(h_top)
        ratio = self.cfg.mup_ratio
        return normalized if ratio == 1.0 else normalized * ratio

    def logits(self, h_top: Tensor) -> Tensor:
        return F.linear(self.readout_input(h_top), self.embed_tokens.weight)

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

    def bind_classifier_accum(self, buffer: Tensor | None) -> None:
        """Bind the trainer's persistent BF16 classifier-gradient accumulator.

        With one bound the head's backward adds its dC straight into this
        buffer over several microbatches, and the trainer flushes it into the
        FP32 embedding sink and clears it on its own cadence; without one every
        head call widens its own dC into the sink. The buffer is neither a
        parameter nor checkpoint state, and it is only meaningful alongside a
        bound embedding sink, which is where its contents eventually land.
        """
        if buffer is not None:
            master = self.embed_tokens.weight
            if self.embed_tokens.grad_sink is None:
                raise RuntimeError("the classifier accumulator needs a bound sink")
            if buffer.shape != master.shape or buffer.dtype != torch.bfloat16:
                raise ValueError("the classifier accumulator must be the BF16 operand")
        self._classifier_accum = buffer

    def classifier_for_loss(self) -> tuple[Tensor, Tensor | None]:
        """The head's classifier operand and the buffer its gradient lands in.

        One decision, so the operand and the destination cannot disagree. On
        CUDA the operand is always the graph-stable BF16 shadow: with an
        accumulator bound and gradients live the head hands it over plainly and
        names the buffer, because CCE's backward writes the classifier gradient
        there itself; otherwise the shadow is read through the shared operand,
        whose backward widens that gradient into the FP32 sink. The portable
        path reads the FP32 master through ordinary autograd.
        """
        master = self.embed_tokens.weight
        if not master.is_cuda:
            return master, None
        if self._classifier_shadow is None:
            raise RuntimeError("CUDA classifier shadow was not prepared")
        if self._classifier_accum is not None and torch.is_grad_enabled():
            return self._classifier_shadow, self._classifier_accum
        return (
            ShadowOperand.apply(
                master, self._classifier_shadow, self.embed_tokens.grad_sink
            ),
            None,
        )

    # -- one column pass -------------------------------------------------------

    def _run_block(
        self,
        block: Block,
        h: Tensor,
        entry: Tensor | None,
        passed: list[Tensor],
        banked: tuple[Tensor, ...],
        *,
        cache: KVCache | None,
        want_weights: bool,
        checkpointing: bool,
    ) -> tuple[
        Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None,
        Tensor | None,
    ]:
        """One block: residual, delta, MHDB weights, expert loss, weights and counts.
        Diagnostic weights are None unless requested.
        ``banked`` are the accumulators of the leading sources in ``passed``."""
        gate_weight = (
            self.attention_gates[block.global_gate_index].weight
            if block.global_gate_index is not None
            else None
        )
        block_fn = _compiled_pkda_block if block.is_pkda else _compiled_block
        if checkpointing and not _retain_block_activations(
            block.layer, self.cfg.routing_block_size
        ):
            # Keep checkpointing outside compilation: both stored and
            # recomputed blocks use the same compiled numerical boundaries.
            h, delta, aux, counts = torch.utils.checkpoint.checkpoint(
                block_fn if h.is_cuda else _block_for_checkpoint,
                block,
                h,
                entry,
                gate_weight,
                len(banked),
                *passed,
                *banked,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            return h, delta, None, None, aux, None, counts
        if h.is_cuda and cache is None and not want_weights:
            h, delta, aux, counts = block_fn(
                block, h, entry, gate_weight, len(banked), *passed, *banked
            )
            return h, delta, None, None, aux, None, counts
        return block(
            h,
            entry,
            cache,
            gate_weight,
            want_weights,
            *passed,
            accumulators=banked,
        )

    def _run_cell(
        self,
        blocks: Iterable[Block],
        h: Tensor,
        cell: int,
        sources: list[Tensor],
        names: list[str],
        accumulators: list[Tensor],
        *,
        entry: Tensor | None = None,
        partial: Tensor | None = None,
        label: str = "",
        weights_out: dict[str, Tensor],
        route_source_names: dict[str, tuple[str, ...]],
        expert_losses: list[Tensor],
        expert_weights_out: dict[str, Tensor],
        expert_counts_out: dict[int, Tensor],
        **runtime,
    ) -> tuple[Tensor, Tensor]:
        """Run ``blocks`` as routing cell ``cell``; return (h, cell delta).

        Without ``entry`` the first block opens the cell and measures from its
        own input. With it the blocks continue a cell already under way:
        ``entry`` is that cell's entry residual and ``partial`` the progress
        made so far, the first block's partial source. ``label`` tags the
        recorded route sites so a tied core's iterations stay distinct.
        """
        cell_start = h if entry is None else entry
        banked = tuple(accumulators)
        for index, block in enumerate(blocks):
            opens = entry is None and index == 0
            passed = list(sources)
            passed_names = list(names)
            if not opens:
                passed.append(partial)
                passed_names.append(f"partial{cell}")
            h, partial, w_attn, w_mlp, aux, w_expert, counts = self._run_block(
                block, h, None if opens else cell_start, passed, banked, **runtime
            )
            expert_losses.append(aux)
            if w_expert is not None:
                expert_weights_out[f"L{block.layer}{label}.experts"] = w_expert
            previous = expert_counts_out.get(block.layer)
            expert_counts_out[block.layer] = (
                counts if previous is None else previous + counts
            )
            if w_attn is not None:
                site = f"L{block.layer}{label}.attn"
                weights_out[site] = w_attn
                route_source_names[site] = ("null", *passed_names)
            if w_mlp is not None:
                site = f"L{block.layer}{label}.mlp"
                weights_out[site] = w_mlp
                mlp_names = [*passed_names, f"partial{cell}"] if opens else passed_names
                route_source_names[site] = ("null", *mlp_names)
        return h, partial

    def forward_column(
        self,
        x: Tensor,
        *,
        cache: KVCache | None = None,
        want_weights: bool = False,
        need_payload: bool = True,
        iterations: int | None = None,
    ) -> ColumnOutput:
        """Run the stack once over inputs x [B, T, D].

        ``x`` is the actual column input: plain embeddings on pass 1 and
        Standard decoding, or the fused FBT input on feedback passes. With a
        cache, positions start at ``cache.pos`` and every mixer cache advances
        by ``T`` exactly once. ``iterations`` is the core iteration count
        under ``l``; the default is the configured mean, or the count the
        cache was allocated for.
        """
        cfg = self.cfg
        if cache is not None:
            if iterations is not None and iterations != cache.iterations:
                raise ValueError(
                    f"the cache holds {cache.iterations} iteration tracks, "
                    f"not {iterations}"
                )
            iterations = cache.iterations
        iterations = cfg.resolve_iterations(iterations)

        # Bank source gradients during CUDA backward; portable execution
        # accumulates ordinary autograd gradients.
        banking = (
            self.bank_sources
            and x.is_cuda
            and torch.is_grad_enabled()
        )
        payload_reads = need_payload

        h = x
        sources: list[Tensor]
        source_names: list[str] = []
        accumulators: list[Tensor] = []
        if banking:
            h, seed, accumulator = bank_source(h, x)
            sources = [seed]
            accumulators.append(accumulator)
        else:
            sources = [x]
        source_names.append("seed")
        seeds = 1

        weights_out: dict[str, Tensor] = {}
        route_source_names: dict[str, tuple[str, ...]] = {}
        expert_losses: list[Tensor] = []
        expert_weights_out: dict[str, Tensor] = {}
        expert_counts_out: dict[int, Tensor] = {}
        checkpointing = (
            self.grad_checkpoint
            and self.training
            and torch.is_grad_enabled()
            and cache is None
            and not want_weights
        )
        runtime = {
            "cache": cache,
            "want_weights": want_weights,
            "checkpointing": checkpointing,
            "weights_out": weights_out,
            "route_source_names": route_source_names,
            "expert_losses": expert_losses,
            "expert_weights_out": expert_weights_out,
            "expert_counts_out": expert_counts_out,
        }

        def complete(carrier: Tensor, delta: Tensor, cell: int) -> Tensor:
            """Store a completed block delta as a source; return the carrier.

            A delta the rest of this column never reads — the last cell's,
            without a payload to write — is stored raw: banking it would give
            the block an all-zero gradient contribution to accumulate.
            """
            readers = cell < cfg.routing_blocks - 1 or payload_reads
            if banking and readers:
                carrier, alias, accumulator = bank_source(carrier, delta)
                sources.append(alias)
                accumulators.append(accumulator)
            else:
                sources.append(delta)
            source_names.append(f"block{cell}")
            return carrier

        size = cfg.routing_block_size
        core_entry = core_state = None
        if cfg.loop:
            # Prelude, tied core, coda. The core is every cell between them,
            # run in order ``iterations`` times. Each core cell keeps its own
            # block delta, accumulated across iterations, and every core site
            # reads every core cell's delta so far, its own as the live
            # partial. An iteration boundary is a boundary in weights, not in
            # state, and at one iteration the column is the unlooped column.
            prelude = self.blocks[:size]
            coda = self.blocks[cfg.layers - size :]
            core = [
                self.blocks[start : start + size]
                for start in range(size, cfg.layers - size, size)
            ]
            h, delta = self._run_cell(
                prelude, h, 0, sources, source_names, accumulators, **runtime
            )
            h = complete(h, delta, 0)
            core_entry = h
            origins: list[Tensor | None] = [None] * len(core)
            deltas: list[Tensor | None] = [None] * len(core)
            exits: list[Tensor | None] = [None] * len(core)
            last = iterations - 1
            for iteration in range(iterations):
                if cache is not None:
                    cache.iteration = iteration
                for index, blocks in enumerate(core):
                    cell = 1 + index
                    if iteration == 0:
                        origin, entry, partial = h, None, None
                    else:
                        # The cell's delta is measured from an origin pinned at
                        # its first entry and advanced by whatever the other
                        # core cells added between its visits. With one core
                        # cell nothing runs between visits and the origin stays
                        # the prelude output.
                        origin = origins[index]
                        if h is not exits[index]:
                            origin = origin + (h - exits[index])
                        entry, partial = origin, deltas[index]
                    passed, passed_names = sources, source_names
                    # The other core cells' deltas so far: the cells before
                    # this one from this iteration, the cells after it from
                    # the previous one. A cell completed on the last
                    # iteration is already in the bank.
                    extra = [
                        (f"block{1 + other}", deltas[other])
                        for other in range(len(core))
                        if other != index
                        and deltas[other] is not None
                        and not (iteration == last and other < index)
                    ]
                    passed = [*sources, *(tensor for _, tensor in extra)]
                    passed_names = [*source_names, *(name for name, _ in extra)]
                    h, delta = self._run_cell(
                        blocks,
                        h,
                        cell,
                        passed,
                        passed_names,
                        accumulators,
                        entry=entry,
                        partial=partial,
                        label=f"i{iteration}",
                        **runtime,
                    )
                    origins[index], deltas[index], exits[index] = origin, delta, h
                    if iteration == last:
                        if index == len(core) - 1:
                            core_state = h
                        h = complete(h, delta, cell)
            if cache is not None:
                cache.iteration = 0
            h, delta = self._run_cell(
                coda, h, 1 + len(core), sources, source_names, accumulators, **runtime
            )
            h = complete(h, delta, 1 + len(core))
        else:
            for cell in range(cfg.routing_blocks):
                blocks = self.blocks[cell * size : (cell + 1) * size]
                h, delta = self._run_cell(
                    blocks, h, cell, sources, source_names, accumulators, **runtime
                )
                h = complete(h, delta, cell)
        if cache is not None:
            cache.advance(x.shape[1])

        payload = None
        if need_payload:
            payload_sources = [sources[seeds - 1], *sources[seeds:]]
            routed, weights = self.payload_router(
                payload_sources,
                want_weights,
                tuple(accumulators),
            )
            if weights is not None:
                weights_out["payload"] = weights
                route_source_names["payload"] = (
                    "null",
                    "seed",
                    *source_names[seeds:],
                )
            if routed is None:
                payload = self.payload_norm(h)
            else:
                epilogue = (
                    _compiled_payload_epilogue if h.is_cuda else _payload_epilogue
                )
                payload = epilogue(self.payload_norm, h, routed)

        return ColumnOutput(
            h_top=h,
            payload=payload,
            route_weights=weights_out,
            route_source_names=route_source_names,
            sources=sources,
            source_names=tuple(source_names),
            n_seeds=seeds,
            iterations=iterations,
            core_entry=core_entry,
            core_state=core_state,
            expert_aux_loss=torch.stack(expert_losses).mean(),
            expert_weights=expert_weights_out,
            expert_counts=torch.stack([
                expert_counts_out[layer] for layer in range(len(self.blocks))
            ]),
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
        if payload is not None and not self.cfg.feedback:
            raise ValueError("a condition without f cannot consume a payload")
        x = self.fuse(payload, e) if payload is not None else e
        return self.forward_column(x, cache=cache, want_weights=want_weights)


# -- multi-pass training forward ----------------------------------------------


def shift_right(x: Tensor) -> Tensor:
    """Shift [B, T, D] one position rightward; position 0 becomes zero."""
    return torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)


def _entry_body(model, payload, e, positions, prefix):
    plain = positions[None, :] < prefix[:, None]  # [B, T]
    fused = model.fuse(shift_right(payload), e)
    return torch.where(plain[..., None], e, fused)


# The jittered and plain entries compile through their own code objects: the
# draw is present on training passes and absent on evaluation and monitor
# passes, and one code object per call shape keeps Dynamo from re-guarding.
def _feedback_entry(model, payload, e, positions, prefix):
    """The next pass's column input: plain prefix, FBT-fused suffix."""
    return _entry_body(model, payload, e, positions, prefix)


def _jittered_feedback_entry(model, payload, jitter, e, positions, prefix):
    """The same entry with this pass's keyed payload jitter added first."""
    return _entry_body(model, payload + jitter, e, positions, prefix)


_compiled_feedback_entry = torch.compile(
    _feedback_entry, fullgraph=True, dynamic=False, mode=INDUCTOR_MODE
)

_compiled_jittered_entry = torch.compile(
    _jittered_feedback_entry, fullgraph=True, dynamic=False, mode=INDUCTOR_MODE
)


def multipass(
    model: DeltaModel,
    tokens: Tensor,
    n_passes: int,
    *,
    prefix_lens: Tensor | None = None,
    jitter: Tensor | None = None,
    want_weights: bool = False,
    iterations: int | None = None,
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
    without ``f`` simply take n_passes=1.  ``iterations`` is the step's core
    iteration count under ``l``, shared by every pass.
    """
    cfg = model.cfg
    if n_passes > 1 and not cfg.feedback:
        raise ValueError("multi-pass batches require a condition with f")
    # One lookup of the whole stored row feeds both heads: the column seed is
    # positions 0..T-1 and the auxiliary head's next-token input is 1..T, so
    # the second lookup and its scatter into the embedding gradient are gone.
    # The seed is copied dense because the routing bank allocates its gradient
    # accumulator with ``zeros_like`` and the routing and FLA kernels read
    # packed rows; the auxiliary input stays a view, normalized and
    # concatenated inside the compiled auxiliary block.
    e_all = model.embed_tokens(tokens)
    e = e_all[:, :-1].contiguous()
    next_embedding = e_all[:, 1:]
    out = model.forward_column(
        e,
        want_weights=want_weights,
        need_payload=True,
        iterations=iterations,
    )
    out.next_embedding = next_embedding
    outs = [out]
    if n_passes == 1:
        return outs

    length = e.shape[1]
    positions = torch.arange(length, device=tokens.device)
    compiled = e.is_cuda
    for i in range(n_passes - 1):
        if jitter is None:
            entry = _compiled_feedback_entry if compiled else _feedback_entry
            x = entry(model, outs[-1].payload, e, positions, prefix_lens[i])
        else:
            entry = _compiled_jittered_entry if compiled else _jittered_feedback_entry
            x = entry(
                model,
                outs[-1].payload,
                jitter[i][:, :length],
                e,
                positions,
                prefix_lens[i],
            )
        out = model.forward_column(
            x,
            want_weights=want_weights,
            need_payload=True,
            iterations=iterations,
        )
        out.next_embedding = next_embedding
        outs.append(out)
    return outs


def _head_losses(
    h_chunk: Tensor,
    target_chunk: Tensor,
    norm_weight: Tensor,
    classifier: Tensor,
    norm_eps: float,
    readout_scale: float,
) -> tuple[Tensor, Tensor]:
    """Per-row NLL and log-partition [B, chunk] of one materialized readout."""
    dtype = h_chunk.dtype
    normalized = h_chunk.float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(dim=-1, keepdim=True) + norm_eps
    )
    normalized = (normalized * (norm_weight.float() * readout_scale)).to(dtype)
    logits = F.linear(normalized, classifier).float()
    nll = F.cross_entropy(
        logits.flatten(0, 1), target_chunk.flatten(), reduction="none"
    )
    return nll.view_as(target_chunk), logits.logsumexp(dim=-1)


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
    so the tiles with the smallest contributions arrive first to limit
    rounding error in the accumulated gradient.
    """
    mean = embeddings.reshape(-1, embeddings.shape[-1]).float().mean(0, keepdim=True)
    logit_avg = torch.addmm(
        torch.zeros(1, classifier.shape[0], device=classifier.device),
        mean.to(classifier.dtype),
        classifier.mT,
        out_dtype=torch.float32,
    )
    return torch.argsort(logit_avg[0], stable=True).to(torch.int32)


def _fixed_cce_rows(
    embeddings: Tensor,
    classifier: Tensor,
    targets: Tensor,
    vocab_ordering: Tensor | None = None,
    *,
    c_grad_accum: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Current CCE with capture-safe preprocessing and differentiable LSE.

    The request is unreduced, so one call serves several heads with different
    per-row weights: the caller weights the returned rows and the fork's
    backward takes the resulting per-row upstream gradient.

    Every training target is a real vocabulary id, so CCE's public
    ``ignore_index`` discovery would always produce ``valids=None``. Construct
    that exact pinned-C CCE request directly: its data-dependent ``nonzero`` is
    illegal inside CUDA graph capture, while its forward and differentiable
    LSE backward are otherwise the authoritative implementation.

    With ``vocab_ordering`` both halves tile the classifier through that
    permutation and the backward skips every tile the gradient filter would
    drop before recomputing its logits.

    ``c_grad_accum`` is a caller-owned persistent BF16 ``[V, D]`` buffer. With
    one the backward lock-adds this call's classifier gradient straight into it
    instead of allocating and returning a fresh gradient tensor, and the
    classifier receives no autograd gradient at all; the caller flushes the
    buffer into its FP32 sink and clears it on its own cadence.
    """
    if CCEParams is None or linear_cross_entropy_apply is None or _handle_eps is None:
        raise RuntimeError("cut-cross-entropy is unavailable")
    embeddings = embeddings.contiguous().flatten(0, -2)
    targets = targets.contiguous().flatten()
    if targets.data_ptr() % 16:
        targets = F.pad(targets, (0, 1))[:-1]
    params = CCEParams(
        targets=targets,
        valids=None,
        softcap=None,
        reduction="none",
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
        skip_early=True,
        tile_flags=None,
        c_grad_accum=c_grad_accum,
    )
    nll, lse = linear_cross_entropy_apply(
        embeddings,
        classifier,
        None,
        params,
    )
    assert lse is not None
    return nll, lse


def _readout_pair(model: DeltaModel, main: Tensor, auxiliary: Tensor) -> Tensor:
    """Both heads' readout rows as one [2*B*T, D] cross-entropy input."""
    return torch.cat(
        (model.readout_input(main), model.readout_input(auxiliary))
    ).flatten(0, -2)


_compiled_readout_pair = torch.compile(
    _readout_pair, fullgraph=True, dynamic=False, mode=INDUCTOR_MODE
)


def head_row_losses(
    model: DeltaModel,
    hiddens: tuple[Tensor, ...],
    targets: tuple[Tensor, ...],
    *,
    chunk: int = 1024,
) -> tuple[Tensor, Tensor]:
    """Per-row NLL and log-partition [heads, B, T] through the tied readout.

    The heads share one vocabulary pass: CUDA concatenates their readout rows
    into a single cut cross-entropy request, so the classifier is tiled, its
    gradient filtered, and its accumulated gradient written once per pass
    rather than once per head. CCE fuses tied unembedding and CE, never
    materializing [B,T,V]; its high-threshold gradient filter is an
    intentional throughput-first numerical divergence of the authoritative
    CUDA recipe. CPU/MPS checkpoint sequence chunks per head to bound the
    materialized vocabulary logits in forward and backward.

    The rows come back unweighted, so a row a head does not supervise simply
    takes no weight; the caller owns every reduction.
    """
    if hiddens[0].is_cuda:
        normalized = (
            _compiled_readout_pair(model, *hiddens)
            if len(hiddens) == 2
            else torch.cat([model.readout_input(h) for h in hiddens]).flatten(0, -2)
        )
        # The ordering is a scheduling hint for the backward, so evaluation
        # (no backward) tiles the classifier in place.
        ordering = (
            batch_vocab_order(normalized, model._classifier_shadow)
            if torch.is_grad_enabled()
            else None
        )
        # The call uses the bound accumulator or the per-call FP32 sink.
        # Evaluation has no backward to accumulate.
        classifier, accum = model.classifier_for_loss()
        nll, lse = _fixed_cce_rows(
            normalized, classifier, torch.cat(targets), ordering, c_grad_accum=accum
        )
        shape = (len(hiddens), *targets[0].shape)
        return nll.view(shape), lse.view(shape)

    rows, partitions = [], []
    for hidden, target in zip(hiddens, targets, strict=True):
        nll_pieces, lse_pieces = [], []
        recompute = torch.is_grad_enabled() and hidden.requires_grad
        for start in range(0, target.shape[1], chunk):
            arguments = (
                hidden[:, start : start + chunk],
                target[:, start : start + chunk],
                model.final_norm.weight,
                model.embed_tokens.weight,
                model.final_norm.eps,
                model.cfg.mup_ratio,
            )
            if recompute:
                nll, lse = torch.utils.checkpoint.checkpoint(
                    _head_losses,
                    *arguments,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                nll, lse = _head_losses(*arguments)
            nll_pieces.append(nll)
            lse_pieces.append(lse)
        rows.append(torch.cat(nll_pieces, dim=1))
        partitions.append(torch.cat(lse_pieces, dim=1))
    return torch.stack(rows), torch.stack(partitions)


def sequence_ce(
    model: DeltaModel,
    h_top: Tensor,
    targets: Tensor,
    *,
    chunk: int = 1024,
) -> tuple[Tensor, Tensor]:
    """Mean CE and z² of one head over [B, T] targets, for the monitors."""
    nll, lse = head_row_losses(model, (h_top,), (targets,), chunk=chunk)
    return nll.mean(), lse.square().mean()


def combine_pass_losses(values: list[Tensor]) -> Tensor:
    """Pass 1 plus the mean over feedback passes (FBT Eq. 12, lambda=1)."""
    if len(values) == 1:
        return values[0]
    return values[0] + torch.stack(values[1:]).mean()


@dataclass
class LossOutput:
    """Training objective and unregularized CE for each pass and prediction depth."""

    total: Tensor
    ntp: list[Tensor]
    mtp: list[Tensor]
    expert_aux_loss: Tensor
    """Mean over passes of the mean over trunk and auxiliary layer invocations."""
    expert_counts: Tensor
    """Assignments summed over passes, [trunk layers + MTP, routed experts]."""


def multipass_loss(
    model: DeltaModel,
    tokens: Tensor,
    outs: list[ColumnOutput],
    *,
    z_coef: float | Tensor = 0.0,
    mtp_weight: float = MTP_LOSS_WEIGHT,
) -> LossOutput:
    """FBT Eq. 12 with λ=1: pass-1 NTP plus the mean over feedback passes,
    plus (in cooldown) the z-loss under the same per-pass weighting.

    MTP adds one sequential second-token predictor per executed pass. Its CE
    and z-loss have the same pass weighting, scaled by ``mtp_weight``. Each
    head is normalized over its own real targets; the auxiliary head runs over
    every executed position, and its last row, whose second token is past the
    end of the stored row, takes a dummy target and no weight. Both heads'
    rows go through one vocabulary pass per model pass. ``ntp[0]`` is the
    Standard-mode tracking metric.
    """
    if not outs:
        raise ValueError("loss needs at least one model pass")
    if not math.isfinite(mtp_weight) or mtp_weight < 0:
        raise ValueError("mtp_weight must be finite and nonnegative")
    if tokens.shape[1] < 3:
        raise ValueError("MTP needs at least three stored tokens")
    targets = tokens[:, 1:]
    mtp_targets = F.pad(tokens[:, 2:], (0, 1))
    losses, z_terms = [], []
    mtp_losses, mtp_z = [], []
    expert_losses, expert_counts = [], []
    for out in outs:
        if out.payload is None:
            raise ValueError("MTP loss needs a payload from every model pass")
        if out.next_embedding is None:
            raise ValueError("MTP loss needs the row's next-token embeddings")
        mtp = model.forward_mtp_embedded(out.payload, out.next_embedding)
        nll, lse = head_row_losses(
            model, (out.h_top, mtp.hidden), (targets, mtp_targets)
        )
        losses.append(nll[0].mean())
        z_terms.append(lse[0].square().mean())
        # The padded row is the auxiliary head's causally last one, so leaving
        # it out of both means leaves every earlier row untouched.
        mtp_losses.append(nll[1][:, :-1].mean())
        mtp_z.append(lse[1][:, :-1].square().mean())
        invocations = model.cfg.executed_layers(out.iterations)
        expert_losses.append(
            (out.expert_aux_loss * invocations + mtp.expert_aux_loss)
            / (invocations + 1)
        )
        expert_counts.append(torch.cat((out.expert_counts, mtp.expert_counts[None])))

    total = combine_pass_losses(losses)
    total = total + z_coef * combine_pass_losses(z_terms)
    total = total + mtp_weight * (
        combine_pass_losses(mtp_losses) + z_coef * combine_pass_losses(mtp_z)
    )
    expert_aux_loss = torch.stack(expert_losses).mean()
    total = total + EXPERT_BALANCE_COEF * expert_aux_loss
    return LossOutput(
        total, losses, mtp_losses, expert_aux_loss, torch.stack(expert_counts).sum(0)
    )


def _monitor_autocast(tokens: Tensor):
    # The monitors are part of the standing training loop, so CUDA follows
    # the same BF16 activation path as captured training and evaluation, which
    # preserves the numerical contract and reuses the captured block
    # specializations instead of compiling a second set.
    if tokens.is_cuda:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def iterate_fused(
    model: DeltaModel,
    tokens: Tensor,
    n_iters: int,
    iterations: int | None = None,
) -> list[dict[str, float]]:
    """The contraction diagnostic: iterated fully-fused prefill passes.

    Repeatedly applies the feedback map (prefix length 1 — everything
    fused) and reports per-iteration validation loss and the update size
    ||h(k) − h(k−1)|| (FBT Fig. 3).  Decaying update norms and flat loss
    are the contraction signature; oscillation or rising loss means the
    map diverges under self-composition.  Every pass runs ``iterations``
    core iterations under ``l`` (default: the configured mean).
    """
    cfg = model.cfg
    if not cfg.feedback:
        raise ValueError("the contraction diagnostic needs a condition with f")
    with _monitor_autocast(tokens):
        e = model.embed_tokens(tokens[:, :-1])
        batch, length = e.shape[:2]
        positions = torch.arange(length, device=tokens.device)
        plain = (positions[None, :] < 1).expand(batch, -1)

        out = model.forward_column(e, iterations=iterations)
        records = []
        for _ in range(n_iters):
            previous = out.h_top
            p_shifted = shift_right(out.payload)
            fused = model.fuse(p_shifted, e)
            out = model.forward_column(
                torch.where(plain[..., None], e, fused), iterations=iterations
            )
            loss, _ = sequence_ce(model, out.h_top, tokens[:, 1:])
            delta = (out.h_top - previous).float().norm(dim=-1).mean()
            records.append({"loss": loss.item(), "update_norm": delta.item()})
    return records


@torch.no_grad()
def depth_trace(
    model: DeltaModel,
    tokens: Tensor,
    iterations: int | None = None,
    *,
    fused: bool = False,
) -> list[dict[str, float]]:
    """The loop's fixed-``r`` sweep: loss and core update size per iteration.

    Runs the column at every iteration count from 1 to ``iterations``
    (default: the configured cap) and reports the held-out loss with the coda
    applied after that many iterations, plus ``mean_token ||core_state(r) -
    core_state(r-1)||_2``, the size of iteration ``r``'s update; at ``r = 1``
    the update is measured from the core entry.  Under same-depth mixing the
    state after iteration ``i`` does not depend on how many iterations follow,
    so on plain positions the sweep is one trajectory read out after every
    iteration.  With ``fused`` the column is a second pass with plain-prefix
    length 1 whose payload comes from a first pass at the same count.
    """
    cfg = model.cfg
    if not cfg.loop:
        raise ValueError("the depth trace needs a condition with l")
    if fused and not cfg.feedback:
        raise ValueError("a fused depth trace needs a condition with f")
    if iterations is None:
        iterations = cfg.loop_max_iterations
    with _monitor_autocast(tokens):
        e = model.embed_tokens(tokens[:, :-1])
        targets = tokens[:, 1:]
        batch, length = e.shape[:2]
        positions = torch.arange(length, device=tokens.device)
        plain = (positions[None, :] < 1).expand(batch, -1)
        records = []
        previous = None
        for depth in range(1, iterations + 1):
            x = e
            if fused:
                first = model.forward_column(e, iterations=depth)
                x = torch.where(
                    plain[..., None], e, model.fuse(shift_right(first.payload), e)
                )
            out = model.forward_column(x, need_payload=False, iterations=depth)
            loss, _ = sequence_ce(model, out.h_top, targets)
            reference = out.core_entry if previous is None else previous
            update = (out.core_state - reference).float().norm(dim=-1).mean()
            previous = out.core_state
            records.append(
                {
                    "iterations": depth,
                    "loss": loss.item(),
                    "update_norm": update.item(),
                }
            )
    return records

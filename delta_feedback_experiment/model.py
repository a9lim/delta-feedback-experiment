"""The Delta model: a routed PKDA/MoE trunk with two recurrence choices.

The architecture contract is ``docs/architecture.md``. PKDA occupies three
layers of each four-layer cell, with NoPE gated GQA in the fourth, MHDB reads
before every sublayer, MoE channel mixers, and sequential two-token prediction.
``f`` adds FBT feedback; ``v`` re-enters the whole column through the same
fusion, iterated per column, fusing its own payload with the position's own
token again; ``n`` selects neither recurrence, the plain column every
condition trains before its recurrence boundary.
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

from .inductor import INDUCTOR_MODE
from .attention import causal_attention, prefix_attention
from .cuda_kernels import (
    Fp8Weights,
    ShadowOperand,
    bespoke_route,
    sink_linear,
)
from .head import dense_head, dense_head_device, weighted_dense_head
from .moe import EXPERT_BIAS_RATE, MixtureOfExperts, validate_expert_geometry
from .parameter_groups import is_normuonh_parameter, is_width_scaled_parameter
from .pkda import PreconditionedKDA
from .sites import Binding, SlabSpec
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
        (
            "full-bandwidth feedback: fuse the preceding column's payload "
            "with the raw token embedding by concatenation"
        ),
    ),
    "v": (
        "loop",
        (
            "looped depth on a virtual repeat of the token: the whole column "
            "re-enters through the shared fusion of its own payload with the "
            "position's own token embedding, iterated per column"
        ),
    ),
}
"""Letter -> (``ModelConfig`` flag, one-line change), in canonical order."""

NULL_CONDITION = "n"
"""The name of the condition with none of the letters.

Every update and evaluation is one plain column seeded by the shared fusion
of the token with the blank payload, which is every condition's update before
its recurrence boundary, so ``n`` trains as ``f`` whose roll never begins."""

NULL_CHANGE = (
    "no recurrence: every column is plain, fusing the raw token embedding "
    "with the blank payload"
)
"""``n``'s one-line change, in the form of ``CONDITION_LETTERS``."""


def parse_condition(text: str) -> str:
    """Accept ``n`` alone, or ``f``, ``v``, or both in either order; reject
    an empty condition."""
    if text == NULL_CONDITION:
        return text
    if not text:
        raise ValueError("condition must be n, or contain f, v, or both")
    if NULL_CONDITION in text:
        raise ValueError(
            f"condition {text!r} is not n alone; n names the condition "
            "without any letters and takes none beside it"
        )
    unknown = sorted(set(text) - set(CONDITION_LETTERS))
    if unknown:
        raise ValueError(
            f"unknown condition letters {''.join(unknown)!r} in {text!r}; "
            f"expected n or letters from {''.join(CONDITION_LETTERS)!r}"
        )
    if len(set(text)) != len(text):
        raise ValueError(f"repeated letter in condition {text!r}")
    return "".join(letter for letter in CONDITION_LETTERS if letter in text)


BASE_NORMAL_INIT_STD = 0.02
"""Base Gaussian standard deviation.

Embeddings and fixed-head-width expansions use this directly. Gate and control
matrices with fan-in ``D`` multiply it by ``sqrt(MUP_BASE_DIM / D)``. Adjust
this constant to tune both families; NorMuonH's fan-in scale is independent.
Token lookups multiply the tied table by ``EMBEDDING_LOOKUP_SCALE``, its
inverse, so the fusion sees unit-RMS tokens at initialization while the
classifier reads the raw table.
"""

EMBEDDING_LOOKUP_SCALE = 1 / BASE_NORMAL_INIT_STD
"""Fixed multiplier on every token lookup.

The tied table initializes at ``BASE_NORMAL_INIT_STD`` for the classifier's
sake, so the lookup carries the inverse: a token enters the shared fusion at
unit RMS, the payload writer's RMSNorm produces unit RMS, and the fan-in
scaled fusion seeds the residual stream at unit RMS. A seed at embedding
scale instead (0.02) left the early pre-norms dividing by a tiny residual,
which amplified the gradient about 160x through a column and 8x per
re-entry hop at screen scale, so looped and feedback columns swamped the
plain column's gradient. A multiplier rather than a larger initialization
keeps the table at the scale its NAdam rate was tuned for.
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

LOOP_MAX_ITERATIONS = 3
"""Most columns one pass runs at a position: the recurrence roll draws two or
three, and the fixed evaluation/decode count stays within this range."""

DEFAULT_LOOP_ITERATIONS = 2
"""Fixed evaluation/decode iteration count; training draws from the roll."""


@dataclass(frozen=True)
class ModelConfig:
    """Permanent PKDA/MoE/MHDB/MTP geometry plus the recurrence flags.

    Routing uses one contiguous feature group per KV head. These groups do
    not align to mixer projections. The default condition is feedback (f).
    """

    vocab_size: int = VOCAB_SIZE
    dim: int = 768
    layers: int = 16
    heads: int = 4
    kv_heads: int = 2
    head_dim: int = 256
    expert_intermediate: int = 832
    num_routed_experts: int = 15
    experts_per_token: int = 3
    pkda_heads: int = 8
    pkda_head_dim: int = 128
    pkda_conv_size: int = 4
    max_seq_len: int = 4096
    norm_eps: float = 1e-6
    routing_block_size: int = 4
    """Exact MHDB cell width in transformer layers."""

    feedback: bool = True
    """``f``: consume the preceding column's payload through shared concat fusion."""

    loop: bool = False
    """``v``: every column runs the whole stack ``iterations`` times, each
    later column seeded by the shared fusion of the preceding column's payload
    with the position's own token embedding, which is a feedback entry onto a
    virtual repeat of the token (``docs/architecture.md``). It adds no
    parameter, so every condition shares one parameter set."""

    loop_iterations: int = DEFAULT_LOOP_ITERATIONS
    """``v``: fixed evaluation/decode count in 1..3. Training draws from
    the recurrence roll, independently of this count."""

    def __post_init__(self) -> None:
        validate_expert_geometry(
            self.expert_intermediate, self.num_routed_experts, self.experts_per_token
        )
        if self.layers < 1:
            raise ValueError("model needs at least one layer")
        if self.routing_block_size < 1:
            raise ValueError("routing block size must be positive")
        if not isinstance(self.loop_iterations, int) or self.loop_iterations < 1:
            raise ValueError("loop evaluation/decode iterations must be a positive integer")
        if self.loop and self.loop_iterations > LOOP_MAX_ITERATIONS:
            raise ValueError(
                f"loop evaluation/decode iterations must be an integer in 1..{LOOP_MAX_ITERATIONS}"
            )
        if self.pkda_heads < 1 or self.pkda_head_dim < 1:
            raise ValueError("PKDA head count and dimension must be positive")
        if self.pkda_conv_size < 1:
            raise ValueError("PKDA convolution width must be positive")

    @property
    def condition(self) -> str:
        """The canonical letters of this configuration, ``n`` for none."""
        letters = "".join(
            letter
            for letter, (flag, _) in CONDITION_LETTERS.items()
            if getattr(self, flag)
        )
        return letters or NULL_CONDITION

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
    def pkda_layers(self) -> int:
        """PKDA layers per column: the blocks that can recompute in backward,
        while the global-attention blocks stay retained."""
        return sum(self.is_pkda_layer(layer) for layer in range(self.layers))

    def resolve_iterations(self, iterations: int | None) -> int:
        """The columns one pass runs at a position; ``None`` is the default."""
        if not self.loop:
            if iterations not in (None, 1):
                raise ValueError("a condition without v runs one column per pass")
            return 1
        if iterations is None:
            return self.loop_iterations
        if not isinstance(iterations, int) or not 1 <= iterations <= LOOP_MAX_ITERATIONS:
            raise ValueError(
                f"iterations {iterations} outside integer range 1..{LOOP_MAX_ITERATIONS}"
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
    """Learned RMSNorm, applied in FP32 before the cast back."""

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
    """Token inputs at unit scale from FP32 tied weights with accumulated gradients.

    The lookup is the tied row times ``EMBEDDING_LOOKUP_SCALE``, unnormalized,
    so token magnitudes stay learnable while the residual stream starts at
    unit RMS; the classifier reads ``weight`` raw. ``nn.Embedding`` is not
    autocast-aware.  Without this explicit boundary its FP32 output silently
    promotes every residual, payload, and routed source. CUDA autocast casts
    lookup activations to BF16. Plain seeds, feedback, the loop, and MTP all
    consume this lookup.

    ``grad_sink`` is the trainer-owned persistent FP32 gradient buffer of the
    tied weight.  When it is set, both the lookup and the tied classifier
    accumulate straight into it and return no autograd gradient; the buffer is
    the ``.grad`` the norm and optimizer read.  It is never a parameter or a
    checkpoint entry.
    """

    grad_sink: Tensor | None = None

    def forward(self, tokens: Tensor) -> Tensor:
        sink = self.grad_sink
        if sink is not None and self.weight.is_cuda and torch.is_grad_enabled():
            out = _EmbeddingSink.apply(tokens, self.weight, sink)
        else:
            out = super().forward(tokens)
        out = out * EMBEDDING_LOOKUP_SCALE
        if out.is_cuda and torch.is_autocast_enabled("cuda"):
            out = out.to(torch.bfloat16)
        return out


# -- kv cache ------------------------------------------------------------------


class KVCache:
    """Hybrid decoding cache with dense KV and fixed-size PKDA states.

    Only global-attention layers receive KV slots. PKDA layers retain their
    FP32 matrix/preconditioner states and three short-convolution histories.
    ``forward_iterations`` advances the shared position once per token
    position. Under ``v`` every layer owns one track per iteration, because
    column ``i`` at a position mixes over the earlier positions' column-``i``
    writes; the driver selects the track through ``iteration`` before each
    column and every track shares the position.
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
            for track in range(self.iterations):
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
        return layer, self.iteration

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
        self.qkv_fp8: Fp8Weights | None = None
        self.o_fp8: Fp8Weights | None = None

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
            fp8=self.qkv_fp8,
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
        return sink_linear(
            out, (self.o_proj.weight,), (self.o_sink,), self.o_shadow, fp8=self.o_fp8
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
    """One auxiliary layer over the model's shared concat-fused inputs.

    Position t consumes the payload at t and the embedding of token t+1.
    Its own causal PKDA recurrence over those pairs cannot see target t+2.
    Every invocation starts with fresh state; no trunk cache is consumed.
    The auxiliary PKDA/MoE block has the trunk's geometry and residual scaling
    without MHDB reads or a tied loop.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.block = Block(cfg, 0, auxiliary=True)

    def forward(
        self, fused_input: Tensor, want_weights: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        h, _, _, _, aux, weights, counts = self.block(
            fused_input, None, None, None, want_weights
        )
        return h, aux, counts, weights


ACTIVATION_MEMORY_BUDGET = 0.9
"""Inductor's activation memory budget for the compiled blocks.

The partitioner has coarse tiers. At 0.9 it recomputes only cheap fused
tensors in backward (the shared expert's SwiGLU output, flattened residual
views) and keeps every custom-operator result, which frees 437 MiB per pass
at one 4,096-token row for half a percent of block time; at 0.85 and below it
re-executes the routed expert forward, which costs 8.6%. The trainer's
whole-block recompute remains the stage past this one.
"""


def _activation_budget():
    """The budget annotation while a block compiles; nothing when it runs eagerly."""
    if torch.compiler.is_compiling():
        return torch.autograd.graph.region_activation_memory_budget(
            ACTIVATION_MEMORY_BUDGET
        )
    return contextlib.nullcontext()


def _mtp_forward(module, fused_input, want_weights):
    with _activation_budget():
        return module(fused_input, want_weights)


_compiled_mtp = torch.compile(
    _mtp_forward, fullgraph=True, dynamic=False, mode=INDUCTOR_MODE
)


def _payload_epilogue(norm, h, routed):
    """Write the routed read plus top state through the payload norm."""
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
    with _activation_budget():
        return _block_for_checkpoint(block, h, block_start, gate_weight, banked, *tensors)


def _pkda_block(block, h, block_start, gate_weight, banked, *tensors):
    with _activation_budget():
        return _block_for_checkpoint(block, h, block_start, gate_weight, banked, *tensors)


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

    expert_aux_loss: Tensor
    """Mean load-balance loss over this column's layer invocations."""

    expert_weights: dict[str, Tensor]
    """Site -> sparse normalized top-k routing weights [B, T, routed experts]."""

    expert_counts: Tensor
    """Assignment counts [physical layers, routed experts] of this column."""

    fused_input: Tensor | None = None
    """[B,T,D] shared fusion of this column's jittered payload and next tokens.
    MTP reads it directly; after a pass's last column the next feedback pass
    shifts it right and restores its plain prefix. ``multipass`` attaches it;
    a bare column leaves it None."""


class DeltaModel(nn.Module):
    """The permanent trunk with feedback and/or looped-depth recurrence."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        factor_seed = torch.initial_seed()
        self.embed_tokens = ResidualEmbedding(cfg.vocab_size, cfg.dim)
        self.register_buffer("_classifier_shadow", None, persistent=False)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.layers))
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.payload_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.payload_router = Router(cfg)
        # Capture the exact common-trunk initialization boundary before
        # constructing the attention gates and shared fusion projection,
        # which draw from their own streams. Restoring it below keeps every
        # shared parameter byte-identical across conditions.
        common_init_state = torch.random.get_rng_state()
        self.attention_gates = nn.ModuleList(
            nn.Linear(cfg.dim, cfg.heads * cfg.head_dim, bias=False)
            for _ in cfg.global_attention_layers
        )
        self.fuse_proj_sink: Tensor | None = None
        self.fuse_proj_shadow: Tensor | None = None
        self.fuse_proj_fp8: Fp8Weights | None = None
        self._shadow_refresh: dict[str, tuple[Tensor, Tensor]] = {}
        """Replicated working copies to refresh from their FP32 masters after
        every optimizer update, keyed by the site that bound them."""
        self.fuse_proj = nn.Linear(2 * cfg.dim, cfg.dim, bias=False)
        self.blank_payload = nn.Parameter(torch.zeros(cfg.dim))
        """Learned stand-in payload, in payload units, for positions
        with no incoming payload: pass 1, plain prefixes, and Standard
        decoding. Every column seed therefore passes through ``fuse_proj``,
        and a feedback position differs from a plain one only by
        ``W_p (p_(t-1) - p_0)``. Zero-initialized, consuming no draws."""
        self.checkpoint_blocks = 0
        """Runtime switch: how many PKDA and auxiliary block invocations of
        each logical forward, in execution order, recompute in backward
        instead of retaining their activations. ``multipass`` starts the
        count; the global-attention blocks are always retained."""
        self._checkpoint_left = 0
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
        self._init_factor_linears(
            (self.fuse_proj,),
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

        The attention gates and the shared fusion projection never advance
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

    def fuse(self, payload: Tensor, token_embedding: Tensor) -> Tensor:
        """Shared entry W [token; payload], with no further norm.

        Both inputs are unit RMS at initialization: the lookup carries the
        fixed ``EMBEDDING_LOOKUP_SCALE`` and the writer's RMSNorm gain starts
        at one, so the fan-in scaled projection seeds the residual stream at
        unit RMS. Any jitter is already in payload units. Token magnitudes
        and learned payload gains survive the linear fusion. Every column
        seed is one of these products; ``plain_seed`` supplies the blank
        payload where no payload arrives, and ``loop_seed`` fuses a looped
        column's preceding payload with the same token again.
        """
        return sink_linear(
            torch.cat((token_embedding, payload), dim=-1),
            (self.fuse_proj.weight,),
            (self.fuse_proj_sink,),
            self.fuse_proj_shadow,
            fp8=self.fuse_proj_fp8,
        )

    def plain_seed(self, token_embedding: Tensor) -> Tensor:
        """The seed of a position with no incoming payload: the shared fusion
        of its raw token embedding with the learned blank payload."""
        blank = self.blank_payload.to(token_embedding.dtype).expand_as(token_embedding)
        return self.fuse(blank, token_embedding)

    def loop_seed(self, payload: Tensor, token_embedding: Tensor) -> Tensor:
        """The seed of a looped column's re-entry: the shared fusion of the
        preceding column's payload with the position's own raw
        ``token_embedding``.

        It is the feedback entry of a chain that repeats every token once, so
        with the blank payload it is ``plain_seed`` and a later column
        repeats the first.
        """
        return self.fuse(payload, token_embedding)

    def forward_mtp(
        self, payload: Tensor, next_tokens: Tensor, *, want_weights: bool = False
    ) -> MTPOutput:
        """Predict from payloads and supplied next tokens, returning expert stats."""
        if next_tokens.shape != payload.shape[:2]:
            raise ValueError("MTP needs next tokens aligned with the payloads")
        return self.forward_mtp_fused(
            self.fuse(payload, self.embed_tokens(next_tokens)), want_weights=want_weights
        )

    def forward_mtp_fused(
        self, fused_input: Tensor, *, want_weights: bool = False
    ) -> MTPOutput:
        """Run the independent auxiliary layer on the shared fused input."""
        if (
            fused_input.ndim != 3
            or fused_input.shape[-1] != self.cfg.dim
            or not fused_input.shape[1]
        ):
            raise ValueError("MTP needs nonempty fused inputs of shape [B,T,D]")
        fn = _compiled_mtp if fused_input.is_cuda else _mtp_forward
        if self._checkpoint_left > 0 and self.training and torch.is_grad_enabled():
            self._checkpoint_left -= 1
            result = torch.utils.checkpoint.checkpoint(
                fn, self.mtp, fused_input, want_weights,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            result = fn(self.mtp, fused_input, want_weights)
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

    # -- parameter sites -------------------------------------------------------

    def slab_specs(self) -> list[SlabSpec]:
        """Every trainable parameter's site, in a fixed order.

        The trainer allocates one working slab and one FP32 gradient slab per
        site and binds them here. Every large projection then reads its
        working copy and accumulates its weight gradient in place through the
        backward, the expert banks as one stacked operand each, and the
        parameters NAdam owns share a flat gradient arena while keeping their
        FP32 masters. Binding ``None`` restores ordinary autograd gradients
        and per-call operand casts. Q/K/V and dense QKV/gate keep separate
        parameters and optimizer state behind one packed GEMM.
        """
        if any(not parameter.requires_grad for parameter in self.parameters()):
            raise ValueError("parameter sites cover trainable models only")
        specs: list[SlabSpec] = []
        for block, gate in self._parameter_blocks():
            attn = block.attn
            prefix = "mtp" if block is self.mtp.block else f"L{block.layer}"
            if block.is_pkda:
                specs.append(
                    SlabSpec(
                        f"{prefix}.qkv",
                        (attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight),
                        self._bind_pkda_qkv(attn),
                    )
                )
            else:
                specs.append(
                    SlabSpec(
                        f"{prefix}.qkv",
                        (attn.qkv_proj.weight, gate),
                        self._bind_attention_qkv(f"{prefix}.qkv", attn, gate),
                        sharded=(True, False),
                    )
                )
            specs.append(SlabSpec(f"{prefix}.o", (attn.o_proj.weight,), self._bind_o(attn)))
            for name in ("gate_up", "down"):
                specs.append(block.mlp.site_spec(f"{prefix}.{name}", name))
        specs.append(SlabSpec("fuse", (self.fuse_proj.weight,), self._bind_fuse))
        covered = {member for spec in specs for member in spec.members}
        rest = tuple(p for p in self.parameters() if p not in covered)
        specs.append(SlabSpec("nadam", rest, self._bind_arena(rest), kind="arena"))
        return specs

    def _bind_pkda_qkv(self, attn):
        def bind(binding: Binding | None) -> None:
            if binding is None:
                attn.q_sink = attn.k_sink = attn.v_sink = None
                attn.packed_qkv_sink = attn.qkv_shadow = attn.qkv_fp8 = None
                return
            attn.q_sink, attn.k_sink, attn.v_sink = (m.grad for m in binding.members)
            attn.packed_qkv_sink = binding.grad
            attn.qkv_shadow = binding.weight
            attn.qkv_fp8 = binding.fp8

        return bind

    def _bind_attention_qkv(self, name: str, attn, gate: nn.Parameter):
        def bind(binding: Binding | None) -> None:
            if binding is None:
                attn.qkv_sink = attn.gate_sink = None
                attn.packed_qkv_sink = attn.qkv_shadow = attn.qkv_fp8 = None
                self._shadow_refresh.pop(name, None)
                return
            attn.qkv_sink, attn.gate_sink = (m.grad for m in binding.members)
            attn.packed_qkv_sink = binding.grad
            attn.qkv_shadow = binding.weight
            attn.qkv_fp8 = binding.fp8
            # The NAdam-owned gate keeps its FP32 master; its rows of the
            # packed operand are a replicated working copy.
            self._shadow_refresh[name] = (binding.members[1].weight, gate.detach())

        return bind

    @staticmethod
    def _bind_o(attn):
        def bind(binding: Binding | None) -> None:
            if binding is None:
                attn.o_sink = attn.o_shadow = attn.o_fp8 = None
                return
            attn.o_sink = binding.members[0].grad
            attn.o_shadow = binding.weight
            attn.o_fp8 = binding.fp8

        return bind

    def _bind_fuse(self, binding: Binding | None) -> None:
        if binding is None:
            self.fuse_proj_sink = self.fuse_proj_shadow = self.fuse_proj_fp8 = None
            return
        self.fuse_proj_sink = binding.members[0].grad
        self.fuse_proj_shadow = binding.weight
        self.fuse_proj_fp8 = binding.fp8

    def _bind_arena(self, members: tuple[nn.Parameter, ...]):
        """The replicated parameters: an FP32 gradient view each, and on CUDA
        a BF16 working copy of every NAdam matrix the mixers read, refreshed
        from its master after each update. The tied embedding's sink is also
        where the head's classifier gradient accumulates."""
        controls = [
            block.attn for block, _ in self._parameter_blocks() if block.is_pkda
        ]

        def bind(binding: Binding | None) -> None:
            if binding is None:
                self.embed_tokens.grad_sink = None
                for attn in controls:
                    attn.control_shadow = attn.decay_up_shadow = None
                    attn.output_gate_shadow = None
                for key in [k for k in self._shadow_refresh if k.startswith("nadam.")]:
                    del self._shadow_refresh[key]
                return
            views = dict(zip(members, binding.members, strict=True))
            embedding = self.embed_tokens.weight
            self.embed_tokens.grad_sink = (
                views[embedding].grad if embedding.is_cuda else None
            )
            for index, attn in enumerate(controls):
                for name, matrix in (
                    ("control", attn.control_proj.weight),
                    ("decay_up", attn.decay_up.weight),
                    ("output_gate", attn.output_gate_up.weight),
                ):
                    current = getattr(attn, f"{name}_shadow")
                    key = f"nadam.{index}.{name}"
                    if not matrix.is_cuda:
                        setattr(attn, f"{name}_shadow", None)
                        continue
                    if current is None or current.shape != matrix.shape:
                        current = matrix.detach().to(torch.bfloat16)
                        setattr(attn, f"{name}_shadow", current)
                    self._shadow_refresh[key] = (current, matrix.detach())

        return bind

    def set_recurrence_saving(self, lean: bool) -> None:
        """Whether every PKDA recurrence rebuilds its WY representation and
        chunk states in backward (lean) or keeps them; the trainer's plan
        decides per graph, from what fits."""
        for block, _ in self._parameter_blocks():
            if block.is_pkda:
                block.attn.lean_recurrence = lean

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
        """Refresh every replicated working copy from its FP32 master.

        That is the CUDA classifier readout of the tied embedding plus every
        NAdam-owned matrix a bound site reads through a working copy, on any
        device. None of them is ``state_dict`` content. Their addresses are
        allocated once before graph capture and remain stable across
        optimizer updates; only their contents change. Sharded matrices are
        not refreshed: their working copies are what the optimizer writes and
        the gather fills.
        """
        master = self.embed_tokens.weight
        if not master.is_cuda:
            self._classifier_shadow = None
        else:
            shadow = self._classifier_shadow
            if (
                shadow is None
                or shadow.shape != master.shape
                or shadow.device != master.device
            ):
                shadow = torch.empty_like(master, dtype=torch.bfloat16)
                self._classifier_shadow = shadow
            shadow.copy_(master)
        # The packed sites hold replicated rows on every device: a dense
        # layer's gate rides in its QKV operand as a copy of the NAdam master.
        if self._shadow_refresh:
            pairs = list(self._shadow_refresh.values())
            torch._foreach_copy_(
                [copy for copy, _ in pairs], [weight for _, weight in pairs]
            )

    def classifier_for_loss(self) -> tuple[Tensor, Tensor | None]:
        """The head's classifier operand and the buffer its gradient lands in.

        One decision, so the operand and the destination cannot disagree. On
        CUDA the operand is always the graph-stable BF16 shadow. With the
        tied embedding's FP32 sink bound and gradients live, the head hands
        the shadow over plainly and names the sink: the head's classifier
        gradient accumulates straight into it in FP32, whatever the number of
        rows in the call, and the classifier receives no autograd gradient.
        Without a sink the shadow is read through the shared operand, whose
        backward widens the gradient into the master's autograd gradient. The
        portable path reads the FP32 master through ordinary autograd.
        """
        master = self.embed_tokens.weight
        if not master.is_cuda:
            return master, None
        if self._classifier_shadow is None:
            raise RuntimeError("CUDA classifier shadow was not prepared")
        sink = self.embed_tokens.grad_sink
        if sink is not None and torch.is_grad_enabled():
            return self._classifier_shadow, sink
        return ShadowOperand.apply(master, self._classifier_shadow, None), None

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
        if checkpointing and block.is_pkda and self._checkpoint_left > 0:
            # Keep checkpointing outside compilation: both stored and
            # recomputed blocks use the same compiled numerical boundaries.
            self._checkpoint_left -= 1
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
        weights_out: dict[str, Tensor],
        route_source_names: dict[str, tuple[str, ...]],
        expert_losses: list[Tensor],
        expert_weights_out: dict[str, Tensor],
        expert_counts_out: dict[int, Tensor],
        **runtime,
    ) -> tuple[Tensor, Tensor]:
        """Run ``blocks`` as routing cell ``cell``; return (h, cell delta).

        The first block opens the cell and measures from its own input; the
        later blocks read the cell's progress so far as their partial source.
        """
        cell_start = h
        banked = tuple(accumulators)
        partial: Tensor | None = None
        for index, block in enumerate(blocks):
            opens = index == 0
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
                expert_weights_out[f"L{block.layer}.experts"] = w_expert
            expert_counts_out[block.layer] = counts
            if w_attn is not None:
                site = f"L{block.layer}.attn"
                weights_out[site] = w_attn
                route_source_names[site] = ("null", *passed_names)
            if w_mlp is not None:
                site = f"L{block.layer}.mlp"
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
    ) -> ColumnOutput:
        """Run the stack once over inputs x [B, T, D].

        ``x`` is the actual column input: the shared fusion of raw embeddings
        with the blank payload (``plain_seed``) on pass 1, plain prefixes, and
        Standard decoding, or with the preceding position's payload on
        feedback positions; on a looped column, the fusion of the preceding
        column's own payload with the position's own token (``loop_seed``).
        With a cache, positions start at ``cache.pos`` on the cache's
        current iteration track; ``forward_iterations`` owns the track
        selection and advances the position once per token position.
        """
        cfg = self.cfg

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
        # Blocks may recompute while the logical forward's budget lasts.
        checkpointing = (
            self.training
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
        for cell in range(cfg.routing_blocks):
            blocks = self.blocks[cell * size : (cell + 1) * size]
            h, delta = self._run_cell(
                blocks, h, cell, sources, source_names, accumulators, **runtime
            )
            h = complete(h, delta, cell)

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
            expert_aux_loss=torch.stack(expert_losses).mean(),
            expert_weights=expert_weights_out,
            expert_counts=torch.stack([
                expert_counts_out[layer] for layer in range(len(self.blocks))
            ]),
        )

    # -- looped depth and sequential decoding ----------------------------------

    def forward_iterations(
        self,
        x: Tensor,
        token_embedding: Tensor,
        *,
        iterations: int | None = None,
        cache: KVCache | None = None,
        loop_jitter: Tensor | None = None,
        want_weights: bool = False,
        need_payload: bool = True,
    ) -> list[ColumnOutput]:
        """Run every column of one position range and return them in order.

        ``x`` [B, T, D] seeds the first column and ``token_embedding``
        [B, T, D] is the raw lookup of the same positions' tokens. Each later
        column is seeded by the shared fusion of the preceding column's
        payload with ``token_embedding`` (``loop_seed``), after adding that
        column's row of ``loop_jitter`` [iterations-1, B, T, D] in payload
        units when given. ``iterations`` is the column count under ``v``: the
        configured evaluation/decode count, or the count the cache was
        allocated for. Every column but possibly the last writes a payload.
        With a cache, column ``i`` mixes on track ``i`` and the shared
        position advances once, after the last column.
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
        if loop_jitter is not None and loop_jitter.shape != (iterations - 1, *x.shape):
            raise ValueError("loop jitter must have shape [iterations-1,B,T,D]")
        compiled = x.is_cuda
        outs: list[ColumnOutput] = []
        for i in range(iterations):
            if cache is not None:
                cache.iteration = i
            last = i + 1 == iterations
            out = self.forward_column(
                x,
                cache=cache,
                want_weights=want_weights,
                need_payload=need_payload or not last,
            )
            outs.append(out)
            if not last:
                if loop_jitter is None:
                    reentry = _compiled_loop_seed if compiled else _loop_seed
                    x = reentry(self, out.payload, token_embedding)
                else:
                    reentry = (
                        _compiled_jittered_loop_seed
                        if compiled
                        else _jittered_loop_seed
                    )
                    x = reentry(self, out.payload, loop_jitter[i], token_embedding)
        if cache is not None:
            cache.iteration = 0
            cache.advance(x.shape[1])
        return outs

    def step(
        self,
        tokens: Tensor,
        payload: Tensor | None,
        cache: KVCache,
        *,
        want_weights: bool = False,
    ) -> ColumnOutput:
        """Decode one position: tokens [B, 1] and the preceding position's
        payload [B, 1, D] (None for Standard decoding and conditions without
        ``f``). Runs the cache's column count and returns the last column."""
        e = self.embed_tokens(tokens)
        if payload is not None and not self.cfg.feedback:
            raise ValueError("a condition without f cannot consume a payload")
        x = self.fuse(payload, e) if payload is not None else self.plain_seed(e)
        return self.forward_iterations(
            x, e, cache=cache, want_weights=want_weights
        )[-1]


# -- multi-pass training forward ----------------------------------------------


def shift_right(x: Tensor) -> Tensor:
    """Shift [B, T, D] one position rightward; position 0 becomes zero."""
    return torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)


def _plain_seed(model, token_embedding):
    return model.plain_seed(token_embedding)


def _fused_input(model, payload, token_embedding):
    return model.fuse(payload, token_embedding)


def _jittered_fused_input(model, payload, jitter, token_embedding):
    return model.fuse(payload + jitter, token_embedding)


def _loop_seed(model, payload, token_embedding):
    return model.loop_seed(payload, token_embedding)


def _jittered_loop_seed(model, payload, jitter, token_embedding):
    return model.loop_seed(payload + jitter, token_embedding)


def _feedback_entry(fused_input, seed, positions, prefix):
    """Reuse the preceding pass's fused input, restoring this pass's prefix
    to the blank-fused plain seed."""
    plain = positions[None, :] < prefix[:, None]
    return torch.where(plain[..., None], seed, shift_right(fused_input))


_compiled_plain_seed = torch.compile(
    _plain_seed, fullgraph=True, dynamic=False, mode=INDUCTOR_MODE
)

_compiled_fused_input = torch.compile(
    _fused_input, fullgraph=True, dynamic=False, mode=INDUCTOR_MODE
)

_compiled_jittered_fused_input = torch.compile(
    _jittered_fused_input, fullgraph=True, dynamic=False, mode=INDUCTOR_MODE
)


_compiled_loop_seed = torch.compile(
    _loop_seed, fullgraph=True, dynamic=False, mode=INDUCTOR_MODE
)
_compiled_jittered_loop_seed = torch.compile(
    _jittered_loop_seed, fullgraph=True, dynamic=False, mode=INDUCTOR_MODE
)
_compiled_feedback_entry = torch.compile(
    _feedback_entry, fullgraph=True, dynamic=False, mode=INDUCTOR_MODE
)

def multipass(
    model: DeltaModel,
    tokens: Tensor,
    n_passes: int,
    *,
    iterations: int | None = None,
    prefix_lens: Tensor | None = None,
    jitter: Tensor | None = None,
    loop_jitter: Tensor | None = None,
    want_weights: bool = False,
) -> list[list[ColumnOutput]]:
    """The condition-agnostic Jacobi multi-pass forward: ``[pass][column]``.

    tokens [B, T+1] is one stored row: the model executes its first T
    positions, which are exactly the positions that predict tokens 1..T.
    Pass 1 seeds every position through the shared fusion with the blank
    payload; feedback passes replace the suffix past each row's prefix.
    The final stored token is a main-head target and MTP input, but needs no
    trunk column.  Under ``v`` every pass runs ``iterations`` columns at each
    position, each later column seeded by the fusion of the preceding
    column's jittered payload with the position's own token embedding.
    prefix_lens [n_passes-1, B] holds values in 1..T-1 (the plain-embedding
    prefix per feedback pass; position 0 is always plain and position T-1 is
    always fused). jitter [n_passes, B, T+1, D] is drawn at the stored-row
    width in payload units for each pass's last column, and
    loop_jitter [n_passes, iterations-1, B, T+1, D] for the columns before
    it. A column's first T jitter rows are added to its payload once, before
    both fusions: the MTP input with the next tokens, and either the next
    column's seed or, after a pass's last column, the next pass's entry after
    shifting and prefix selection.  Both are pre-drawn by the caller — the
    shared randomness contract lives in the trainer, not here.  Conditions
    without ``f`` simply take n_passes=1.
    """
    cfg = model.cfg
    iterations = cfg.resolve_iterations(iterations)
    if n_passes < 1:
        raise ValueError("multipass needs at least one pass")
    if n_passes > 1 and not cfg.feedback:
        raise ValueError("multi-pass batches require a condition with f")
    if jitter is not None and jitter.shape != (
        n_passes, *tokens.shape, cfg.dim
    ):
        raise ValueError("jitter must have shape [passes,B,T+1,D]")
    if (loop_jitter is not None) != (jitter is not None and iterations > 1):
        raise ValueError(
            "loop jitter accompanies pass jitter exactly when a pass runs more "
            "than one column"
        )
    if loop_jitter is not None and loop_jitter.shape != (
        n_passes, iterations - 1, *tokens.shape, cfg.dim
    ):
        raise ValueError("loop jitter must have shape [passes,iterations-1,B,T+1,D]")
    # One logical forward: its trunk columns here and the auxiliary blocks in
    # ``multipass_loss`` share this recomputation budget in execution order.
    model._checkpoint_left = model.checkpoint_blocks
    # One raw lookup of the whole stored row feeds both heads: the plain
    # seed fuses positions 0..T-1 with the blank payload and the auxiliary
    # head's next-token input is 1..T, so the second lookup and its gradient
    # scatter are gone. The blank fusion is a fresh packed tensor, as every
    # feedback entry is, so pass 1 shares the compiled blocks and their
    # rounding with the later passes: the routing bank allocates its gradient
    # accumulator with ``zeros_like``, the routing and FLA kernels read packed
    # rows, and Dynamo guards on strides.
    e_all = model.embed_tokens(tokens)
    compiled = e_all.is_cuda
    plain = _compiled_plain_seed if compiled else _plain_seed
    own_embedding = e_all[:, :-1]
    seed = plain(model, own_embedding)
    length = seed.shape[1]
    positions = torch.arange(length, device=tokens.device)
    token_embedding = e_all[:, 1:]
    entry = _compiled_feedback_entry if compiled else _feedback_entry
    outs: list[list[ColumnOutput]] = []
    x = seed
    for p in range(n_passes):
        if p:
            x = entry(
                outs[-1][-1].fused_input, seed, positions, prefix_lens[p - 1]
            )
        columns = model.forward_iterations(
            x,
            own_embedding,
            iterations=iterations,
            loop_jitter=(
                None if loop_jitter is None else loop_jitter[p][:, :, :length]
            ),
            want_weights=want_weights,
        )
        for i, out in enumerate(columns):
            if jitter is None:
                fusion = _compiled_fused_input if compiled else _fused_input
                out.fused_input = fusion(model, out.payload, token_embedding)
            else:
                fusion = _compiled_jittered_fused_input if compiled else _jittered_fused_input
                draw = jitter[p] if i + 1 == iterations else loop_jitter[p, i]
                out.fused_input = fusion(
                    model, out.payload, draw[:, :length], token_embedding
                )
        outs.append(columns)
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
    into a single request, so the classifier is read and its accumulated
    gradient written once per pass rather than once per head. Hopper runs
    the dense cuBLAS head (``head.py``), unfiltered with FP32 logits; other
    CUDA devices run cut cross-entropy, which never materializes [B,T,V] and
    whose high-threshold gradient filter is an intentional throughput-first
    numerical divergence. CPU/MPS checkpoint sequence chunks per head to
    bound the materialized vocabulary logits in forward and backward.
    Training calls ``weighted_head_loss`` instead, which on Hopper applies
    the objective's gradient during the forward.

    The rows come back unweighted, so a row a head does not supervise simply
    takes no weight; the caller owns every reduction.
    """
    if hiddens[0].is_cuda:
        normalized = (
            _compiled_readout_pair(model, *hiddens)
            if len(hiddens) == 2
            else torch.cat([model.readout_input(h) for h in hiddens]).flatten(0, -2)
        )
        if dense_head_device(normalized.device):
            classifier, accum = model.classifier_for_loss()
            nll, lse = dense_head(normalized, classifier, torch.cat(targets), sink=accum)
            shape = (len(hiddens), *targets[0].shape)
            return nll.view(shape), lse.view(shape)
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
            normalized,
            classifier,
            torch.cat(targets),
            ordering,
            c_grad_accum=accum,
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


def _first_plus_mean(values: list[Tensor]) -> Tensor:
    """FBT Eq. 12 at lambda=1 along one axis: the first plus the mean of the rest."""
    if len(values) == 1:
        return values[0]
    return values[0] + torch.stack(values[1:]).mean()


def first_plus_mean_weights(count: int) -> list[float]:
    """Each element's coefficient under ``_first_plus_mean`` of ``count``
    values: one for the first, ``1/(count-1)`` for each later one."""
    return [1.0] + [1.0 / (count - 1)] * (count - 1) if count > 1 else [1.0]


def combine_column_losses(values: list[list[Tensor]]) -> Tensor:
    """FBT's first-plus-mean combine applied along both recurrence axes.

    ``values`` is ``[pass][column]``. Each column's series is combined along
    passes, the first pass plus the mean over later passes, and the resulting
    per-column series is combined the same way along columns. The four
    blocks of a rolled step therefore each carry unit weight, spread evenly
    inside the block: the plain first column; the loop alone (pass 1, later
    columns); feedback alone (later passes, column 1); and both together.
    With one axis absent this is the other axis's own combine, so the ``f``
    and ``v`` objectives are its marginals.
    """
    per_column = [
        _first_plus_mean([pass_values[column] for pass_values in values])
        for column in range(len(values[0]))
    ]
    return _first_plus_mean(per_column)


@dataclass
class LossOutput:
    """Training objective and unregularized CE for each column and prediction depth."""

    total: Tensor
    ntp: list[list[Tensor]]
    """Main-head CE ``[pass][column]``; ``ntp[0][0]`` is the plain column."""
    mtp: list[list[Tensor]]
    """Auxiliary-head CE ``[pass][column]``."""
    expert_aux_loss: Tensor
    """Mean over columns of the mean over trunk and auxiliary layer invocations."""
    expert_counts: Tensor
    """Assignments summed over columns, [trunk layers + MTP, routed experts]."""


def weighted_head_loss(
    model: DeltaModel,
    hiddens: tuple[Tensor, ...],
    targets: tuple[Tensor, ...],
    weight_nll: Tensor,
    weight_z: Tensor,
    *,
    grad_scale: float | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """``(sum(w_nll * nll + w_z * lse^2), nll, lse)`` through the tied readout.

    The weights are FP32 over the heads' rows in order, ``[heads * B * T]``;
    ``nll`` and ``lse`` come back ``[heads, B, T]``. A trainer that will
    back-propagate exactly ``grad_scale`` into the sum passes it; on Hopper
    that call runs the dense head, which applies the objective's gradient
    during its forward and returns detached statistics. Every other call
    forms the same sum over ``head_row_losses`` with ordinary autograd.
    """
    shape = (len(hiddens), *targets[0].shape)
    main = hiddens[0]
    if (
        grad_scale is not None
        and main.is_cuda
        and torch.is_grad_enabled()
        and main.requires_grad
        and dense_head_device(main.device)
    ):
        normalized = (
            _compiled_readout_pair(model, *hiddens)
            if len(hiddens) == 2
            else torch.cat([model.readout_input(h) for h in hiddens]).flatten(0, -2)
        )
        classifier, accum = model.classifier_for_loss()
        total, nll, lse = weighted_dense_head(
            normalized, classifier, torch.cat(targets).flatten(), weight_nll, weight_z,
            grad_scale=grad_scale, sink=accum,
        )
        return total, nll.view(shape), lse.view(shape)
    nll, lse = head_row_losses(model, hiddens, targets)
    total = (weight_nll * nll.flatten()).sum() + (weight_z * lse.flatten().square()).sum()
    return total, nll, lse


def multipass_loss(
    model: DeltaModel,
    tokens: Tensor,
    outs: list[list[ColumnOutput]],
    *,
    z_coef: float | Tensor = 0.0,
    mtp_weight: float = MTP_LOSS_WEIGHT,
    grad_scale: float | None = None,
) -> LossOutput:
    """FBT Eq. 12 with λ=1 along passes and again along columns, plus (in
    cooldown) the z-loss under the same weighting: ``combine_column_losses``
    gives the plain column, the loop alone, feedback alone, and both together
    one unit each.

    Every column is supervised. MTP adds one sequential second-token
    predictor per column. Its CE and z-loss have the same weighting,
    scaled by ``mtp_weight``. Each head is normalized over its own real
    targets; the auxiliary head runs over every executed position, and its
    last row, whose second token is past the end of the stored row, takes a
    dummy target and no weight. Both heads' rows go through one vocabulary
    pass per column. ``ntp[0][0]`` is the Standard-mode tracking metric.

    The objective is assembled row by row: each column's head call receives
    every row's weight (its column's combine coefficient, the row mean, the
    MTP weight, the z-loss coefficient), which lets the Hopper head apply
    its gradient during the forward. A trainer opts into that by passing
    ``grad_scale``, the gradient it back-propagates into ``total`` (a
    replay's share of the step); the per-column CE series then come back
    detached there. Without it every output stays differentiable.
    """
    if not outs or not outs[0]:
        raise ValueError("loss needs at least one model column")
    if not math.isfinite(mtp_weight) or mtp_weight < 0:
        raise ValueError("mtp_weight must be finite and nonnegative")
    if tokens.shape[1] < 3:
        raise ValueError("MTP needs at least three stored tokens")
    targets = tokens[:, 1:]
    mtp_targets = F.pad(tokens[:, 2:], (0, 1))
    batch, length = targets.shape
    # Row weights of one unit-coefficient column: each head's mean over its
    # real targets. The padded row is the auxiliary head's causally last
    # one, so leaving it out of both means leaves every earlier row untouched.
    row_weights = torch.full(
        (2, batch, length), 1.0 / (batch * length), device=tokens.device
    )
    row_weights[1] = mtp_weight / (batch * (length - 1))
    row_weights[1, :, -1] = 0.0
    row_weights = row_weights.flatten()
    pass_coefficients = first_plus_mean_weights(len(outs))
    column_coefficients = first_plus_mean_weights(len(outs[0]))
    losses: list[list[Tensor]] = []
    mtp_losses: list[list[Tensor]] = []
    head_totals, expert_losses, expert_counts = [], [], []
    invocations = model.cfg.layers
    for pass_coefficient, columns in zip(pass_coefficients, outs, strict=True):
        losses.append([])
        mtp_losses.append([])
        for column_coefficient, out in zip(column_coefficients, columns, strict=True):
            if out.payload is None:
                raise ValueError("MTP loss needs a payload from every column")
            if out.fused_input is None:
                raise ValueError("MTP loss needs the column's shared fused input")
            mtp = model.forward_mtp_fused(out.fused_input)
            weight_nll = row_weights * (pass_coefficient * column_coefficient)
            head_total, nll, lse = weighted_head_loss(
                model, (out.h_top, mtp.hidden), (targets, mtp_targets),
                weight_nll, weight_nll * z_coef, grad_scale=grad_scale,
            )
            head_totals.append(head_total)
            losses[-1].append(nll[0].mean())
            mtp_losses[-1].append(nll[1][:, :-1].mean())
            expert_losses.append(
                (out.expert_aux_loss * invocations + mtp.expert_aux_loss)
                / (invocations + 1)
            )
            expert_counts.append(
                torch.cat((out.expert_counts, mtp.expert_counts[None]))
            )

    expert_aux_loss = torch.stack(expert_losses).mean()
    total = torch.stack(head_totals).sum() + EXPERT_BALANCE_COEF * expert_aux_loss
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
    columns per position under ``v`` (default: the evaluation
    count), and the map composes the last column's payload.
    """
    cfg = model.cfg
    if not cfg.feedback:
        raise ValueError("the contraction diagnostic needs a condition with f")
    with _monitor_autocast(tokens):
        e = model.embed_tokens(tokens[:, :-1])
        seed = model.plain_seed(e)
        batch, length = e.shape[:2]
        positions = torch.arange(length, device=tokens.device)
        plain = (positions[None, :] < 1).expand(batch, -1)

        out = model.forward_iterations(seed, e, iterations=iterations)[-1]
        records = []
        for _ in range(n_iters):
            previous = out.h_top
            p_shifted = shift_right(out.payload)
            fused = model.fuse(p_shifted, e)
            out = model.forward_iterations(
                torch.where(plain[..., None], seed, fused), e, iterations=iterations
            )[-1]
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
    """The loop's per-column readout: loss and update size at every depth.

    Runs ``iterations`` columns (default: the maximum) and reads out after
    each: the held-out loss of that column's top state, plus ``mean_token
    ||h_top(r) - h_top(r-1)||_2``, the size of column ``r``'s update; at ``r
    = 1`` the update is measured from the seed. A column's state does not
    depend on the columns after it, so one run reports every depth up to the
    count. With ``fused`` the run is a second pass with plain-prefix length 1
    whose payload comes from a first pass at the same count.
    """
    cfg = model.cfg
    if not cfg.loop:
        raise ValueError("the depth trace needs a condition with v")
    if fused and not cfg.feedback:
        raise ValueError("a fused depth trace needs a condition with f")
    if iterations is None:
        iterations = LOOP_MAX_ITERATIONS
    cfg.resolve_iterations(iterations)
    with _monitor_autocast(tokens):
        e = model.embed_tokens(tokens[:, :-1])
        seed = model.plain_seed(e)
        targets = tokens[:, 1:]
        batch, length = e.shape[:2]
        positions = torch.arange(length, device=tokens.device)
        plain = (positions[None, :] < 1).expand(batch, -1)
        x = seed
        if fused:
            first = model.forward_iterations(seed, e, iterations=iterations)[-1]
            x = torch.where(
                plain[..., None], seed, model.fuse(shift_right(first.payload), e)
            )
        columns = model.forward_iterations(
            x, e, iterations=iterations, need_payload=False
        )
        records = []
        previous = x
        for depth, out in enumerate(columns, start=1):
            loss, _ = sequence_ce(model, out.h_top, targets)
            update = (out.h_top - previous).float().norm(dim=-1).mean()
            previous = out.h_top
            records.append(
                {
                    "iterations": depth,
                    "loss": loss.item(),
                    "update_norm": update.item(),
                }
            )
    return records

"""Offline invariants for the delta model family (CPU, tiny config).

These are the architecture contract's checkable claims: transient-read routing
keeps the stream telescoping, zero-init routing is uniform, the payload
init-matches the bare top state, the Jacobi passes are token-causal, the
prefix mixin degenerates to pass 1, and cached sequential decoding equals
the exact recurrence computed by full recomputation.
"""

import itertools
import math

import pytest
import torch

from delta_feedback_experiment.model import (
    BASE_NORMAL_INIT_STD,
    CONDITION_LETTERS,
    DeltaModel,
    KVCache,
    condition_config,
    depth_trace,
    iterate_fused,
    multipass,
    multipass_loss,
    parse_condition,
    shift_right,
)
from delta_feedback_experiment.pkda import PreconditionedKDA, _PackedControlSplit

TINY = {
    "vocab_size": 97,
    "dim": 32,
    "layers": 4,
    "heads": 2,
    "kv_heads": 2,
    "head_dim": 16,
    "intermediate": 64,
    "pkda_heads": 2,
    "pkda_head_dim": 16,
    "pkda_conv_size": 4,
    "max_seq_len": 32,
}


TINY_LOOP = TINY | {"layers": 12, "loop_iterations": 2, "loop_max_iterations": 3}
"""The loop needs three whole cells: prelude, core, coda."""

BUILDABLE = ("", "a", "r", "f", "ar", "af", "rf", "arf")
"""The subsets of ``arf`` in canonical order; every one builds at ``TINY``."""

LOOPED = ("l", "al", "rl", "fl", "arl", "afl", "rfl", "arfl")
"""The same subsets with ``l``; every one builds at ``TINY_LOOP``."""


def geometry(condition: str) -> dict:
    return TINY_LOOP if "l" in condition else TINY


def tiny(condition: str, seed: int = 0, **overrides) -> DeltaModel:
    torch.manual_seed(seed)
    return DeltaModel(condition_config(condition, **(geometry(condition) | overrides))).eval()


def tokens(batch=2, length=16, seed=1, vocab=97):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, length), generator=generator)


def forward(model, toks, **kwargs):
    return model.forward_column(model.embed_tokens(toks), **kwargs)


# -- structure -----------------------------------------------------------------


def test_condition_grammar():
    """One letter per change from the plain decoder; any order in, canonical
    order out; the configuration renders its own letters back."""
    assert tuple(CONDITION_LETTERS) == ("a", "r", "f", "l")
    assert parse_condition("") == ""
    assert parse_condition("fra") == "arf"
    assert parse_condition("lfra") == "arfl"
    for text in ("dar", "mhdb", "df", "vanilla", "aa", "arfx"):
        with pytest.raises(ValueError, match="condition"):
            parse_condition(text)
    for count in range(5):
        for letters in itertools.combinations("lfra", count):
            text = "".join(letters)
            config = condition_config(text, **geometry(text))
            assert config.condition == parse_condition(text)
    looped = condition_config("arfl", **TINY_LOOP)
    assert looped.loop and looped.hybrid
    assert looped.block_routing and looped.feedback
    assert looped.core_layers == range(4, 8)
    assert looped.routing_blocks == 3
    assert looped.executed_layers(1) == 12 and looped.executed_layers(3) == 20
    deeper = condition_config("arfl", **(TINY_LOOP | {"layers": 20}))
    assert deeper.core_layers == range(4, 16) and deeper.routing_blocks == 5
    assert deeper.executed_layers(1) == 20 and deeper.executed_layers(3) == 44
    assert not condition_config("arf", **TINY).is_core_layer(1)


def test_loop_geometry_and_iteration_bounds():
    """The loop needs whole cells and at least three of them; iteration counts
    live in 1..max, and a condition without l runs the core exactly once."""
    with pytest.raises(ValueError, match="three"):
        condition_config("arfl", **TINY)
    with pytest.raises(ValueError, match="three"):
        condition_config("l", **(TINY | {"layers": 9}))
    with pytest.raises(ValueError, match="iterations"):
        condition_config("l", **(TINY_LOOP | {"loop_max_iterations": 1}))
    with pytest.raises(ValueError, match="iterations"):
        condition_config("l", **(TINY_LOOP | {"loop_iterations": 0}))
    looped = condition_config("arfl", **TINY_LOOP)
    assert looped.resolve_iterations(None) == 2
    assert looped.resolve_iterations(3) == 3
    for bad in (0, 4):
        with pytest.raises(ValueError, match="outside"):
            looped.resolve_iterations(bad)
    plain = condition_config("arf", **TINY)
    assert plain.resolve_iterations(None) == 1 and plain.resolve_iterations(1) == 1
    with pytest.raises(ValueError, match="without l"):
        plain.resolve_iterations(2)
    model = tiny("arf")
    with pytest.raises(ValueError, match="without l"):
        forward(model, tokens(), iterations=2)


def test_condition_flags():
    plain = condition_config("", **TINY)
    assert not plain.block_routing
    assert not plain.feedback
    assert not plain.hybrid
    # The same three positions per cell use RoPE without a and PKDA with a.
    assert plain.global_attention_layers == tuple(range(TINY["layers"]))
    assert [plain.is_rope_layer(layer) for layer in range(8)] == [
        True,
        True,
        True,
        False,
    ] * 2
    hybrid = condition_config("a", **TINY)
    assert hybrid.hybrid
    assert not hybrid.block_routing and not hybrid.feedback
    assert hybrid.is_pkda_layer(0) and not hybrid.is_pkda_layer(3)
    assert hybrid.global_attention_layers == (3,)
    assert not any(hybrid.is_rope_layer(layer) for layer in range(8))
    routed = condition_config("ar", **TINY)
    assert routed.block_routing
    assert not routed.feedback
    assert routed.kv_heads == TINY["kv_heads"]
    assert routed.routing_block_size == 4
    fed = condition_config("af", **TINY)
    assert fed.feedback
    assert not fed.block_routing
    full = condition_config("arf", **TINY)
    assert full.feedback
    assert full.block_routing
    # Letters compose independently of the trunk letter.
    plain_routed_fed = condition_config("rf", **TINY)
    assert plain_routed_fed.block_routing and plain_routed_fed.feedback
    assert not plain_routed_fed.hybrid

    screen = condition_config("a")
    assert screen.intermediate * 6 == screen.dim * 26
    assert screen.pkda_heads == 10
    assert screen.pkda_heads * screen.pkda_head_dim * 3 == screen.dim * 5
    assert screen.kv_heads == 4
    assert screen.norm_eps == 1e-6
    assert DeltaModel(hybrid).blocks[0].attn.norm_eps == hybrid.norm_eps


def test_letters_only_add_parameters():
    """On either trunk, adding letters adds parameters: a condition's names are
    a strict subset of every condition that has its letters and more."""
    names = {
        condition: {name for name, _ in tiny(condition).named_parameters()}
        for condition in BUILDABLE
    }
    for small in BUILDABLE:
        for large in BUILDABLE:
            same_trunk = ("a" in small) == ("a" in large)
            if small != large and set(small) < set(large) and same_trunk:
                assert names[small] < names[large], (small, large)
    # The union of r and f misses exactly the payload router, which needs both.
    for trunk in ("", "a"):
        assert names[trunk + "rf"] - (names[trunk + "r"] | names[trunk + "f"]) == {
            "payload_router.query",
            "payload_router.key_norm.weight",
            "payload_router.null",
        }
    for condition in ("r", "ar", "rf", "arf"):
        assert any("null" in name for name in names[condition])
    # l adds no parameter: the core is the same cell, tied across iterations.
    for condition in BUILDABLE:
        assert {name for name, _ in tiny(condition + "l").named_parameters()} == {
            name
            for name, _ in tiny(condition, layers=12).named_parameters()
        }, condition
    assert not any("blocks.0.attn.q_proj" in name for name in names[""])
    # Every dense attention layer owns a gate: all four on the plain trunk,
    # the cell's fourth layer under a.
    assert sum("attention_gates" in name for name in names["rf"]) == TINY["layers"]
    assert sum("attention_gates" in name for name in names["arf"]) == 1


def test_screen_param_count():
    expected = {
        "": (158_979_072, 120_345_600),
        "r": (159_034_368, 120_400_896),
        "f": (160_161_024, 121_527_552),
        "rf": (160_218_624, 121_585_152),
        "a": (178_221_864, 139_588_392),
        "ar": (178_277_160, 139_643_688),
        "af": (179_403_816, 140_770_344),
        "arf": (179_461_416, 140_827_944),
        "arfl": (179_461_416, 140_827_944),
    }
    with torch.device("meta"):
        for condition, (total, active_non_embedding) in expected.items():
            model = DeltaModel(condition_config(condition))
            count = sum(parameter.numel() for parameter in model.parameters())
            assert count == total
            assert count - model.embed_tokens.weight.numel() == active_non_embedding


def test_large_projections_are_persistently_packed():
    model = tiny("arf")
    pkda = model.blocks[0].attn
    attention = model.blocks[3].attn
    mlp = model.blocks[0].mlp
    assert pkda.q_proj.weight.shape == (32, 32)
    assert pkda.decay_up.weight.shape == (32, 16)
    assert attention.qkv_proj.weight.shape == (96, 32)
    assert not hasattr(attention, "q_proj")
    assert model.attention_gates[0].weight.shape == (32, 32)
    assert mlp.gate_up_proj.weight.shape == (128, 32)
    assert not hasattr(mlp, "gate_proj")


def test_pkda_preconditioner_initialization_and_bound():
    torch.manual_seed(4)
    pkda = PreconditionedKDA(32, num_heads=2, head_dim=16)
    assert torch.all((pkda.A_log.exp() >= 1) & (pkda.A_log.exp() <= 16))
    assert torch.all((pkda.A_log_precond.exp() >= 1) & (pkda.A_log_precond.exp() <= 16))
    assert torch.equal(pkda.log_precond_center, torch.full((2,), -0.2))
    assert pkda.control_splits == (16, 2, 2, 2, 16)
    controls = pkda.control_proj.weight.split(pkda.control_splits)
    assert [part.shape for part in controls] == [
        (16, 32),
        (2, 32),
        (2, 32),
        (2, 32),
        (16, 32),
    ]
    assert [part.storage_offset() for part in controls] == [0, 512, 576, 640, 704]

    diagonal_state = torch.logspace(-12, 12, 100).reshape(1, 1, -1)
    deviation = torch.log(diagonal_state + pkda.squash_eps) + 0.2
    preconditioner = torch.exp(
        -math.log(pkda.squash_x) * deviation / (1 + deviation.abs())
    )
    assert preconditioner.min() >= 1 / pkda.squash_x
    assert preconditioner.max() <= pkda.squash_x


def test_packed_control_split_preserves_forward_and_backward():
    splits = (5, 2, 2, 2, 5)
    actual = torch.randn(3, 7, sum(splits), requires_grad=True)
    expected = actual.detach().clone().requires_grad_()
    cotangents = tuple(torch.randn(3, 7, size) for size in splits)

    actual_parts = _PackedControlSplit.apply(actual, splits)
    expected_parts = expected.split(splits, dim=-1)
    assert all(
        torch.equal(got, want)
        for got, want in zip(actual_parts, expected_parts, strict=True)
    )
    torch.autograd.backward(actual_parts, cotangents)
    torch.autograd.backward(expected_parts, cotangents)
    assert torch.equal(actual.grad, expected.grad)


def test_initialization_pairs_across_conditions_by_letter():
    """Conditions on the same trunk share every common parameter
    byte-identically: the trunk from the common stream, and each letter's
    private modules from that letter's own seeded stream, which never
    advances the common one."""
    states = {
        condition: tiny(condition, seed=17).state_dict() for condition in BUILDABLE
    }
    for left in BUILDABLE:
        for right in BUILDABLE:
            if ("a" in left) != ("a" in right):
                continue
            for name in states[left].keys() & states[right].keys():
                assert torch.equal(states[left][name], states[right][name]), (
                    left,
                    right,
                    name,
                )
    # The loop pairs with its own unlooped condition at the loop geometry.
    for condition in ("al", "arl", "arfl", "rfl"):
        looped = tiny(condition, seed=17).state_dict()
        flat = tiny(condition.replace("l", ""), seed=17, layers=12).state_dict()
        assert looped.keys() == flat.keys()
        for name in looped:
            assert torch.equal(looped[name], flat[name]), (condition, name)
    for left in BUILDABLE:
        for right in BUILDABLE:
            if ("a" in left) != ("a" in right):
                continue
            for name in states[left].keys() & states[right].keys():
                assert torch.equal(states[left][name], states[right][name]), (
                    left,
                    right,
                    name,
                )
    # The two trunks consume the common stream differently, so names they
    # both have do not pair across the trunk letter.
    assert not torch.equal(
        states[""]["blocks.0.mlp.gate_up_proj.weight"],
        states["a"]["blocks.0.mlp.gate_up_proj.weight"],
    )


def test_normuonh_matrices_use_inverse_sqrt_fan_in_initialization():
    model = tiny("arf", seed=23)

    def rms(weight):
        return weight.detach().square().mean().sqrt().item()

    # NorMuonH matrices use the paper's fan-in scale, including both projection
    # orientations and the factor-private FBT value map.
    assert rms(model.blocks[0].attn.q_proj.weight) == pytest.approx(
        1 / math.sqrt(32), rel=0.08
    )
    assert rms(model.blocks[0].mlp.down_proj.weight) == pytest.approx(
        1 / math.sqrt(64), rel=0.08
    )
    assert rms(model.fuse_value.weight) == pytest.approx(1 / math.sqrt(32), rel=0.08)

    # NAdam matrices use the base scale or its gate/control width adjustment.
    assert rms(model.embed_tokens.weight) == pytest.approx(
        BASE_NORMAL_INIT_STD, rel=0.08
    )
    width_std = BASE_NORMAL_INIT_STD * math.sqrt(model.cfg.mup_ratio)
    assert rms(model.attention_gates[0].weight) == pytest.approx(width_std, rel=0.08)
    assert rms(model.fuse_gate.weight) == pytest.approx(width_std, rel=0.08)
    assert rms(model.blocks[0].attn.control_proj.weight) == pytest.approx(
        width_std, rel=0.08
    )


@pytest.mark.parametrize("dim", [768, 1152, 1536])
@torch.no_grad()
def test_gate_control_initialization_preserves_logit_variance_across_widths(dim):
    """Unit-RMS inputs see the same direct and two-stage control variance at
    every production width; embeddings and head-width expansions stay fixed."""
    from delta_feedback_experiment.model import MUP_BASE_DIM

    model = tiny("arf", seed=23, dim=dim, pkda_head_dim=128)
    attn = model.blocks[0].attn

    def output_rms(weight):
        # Expected output RMS for an independent isotropic unit-RMS input.
        return weight.square().sum(-1).mean().sqrt().item()

    expected = BASE_NORMAL_INIT_STD * math.sqrt(MUP_BASE_DIM)
    for weight in (
        model.attention_gates[0].weight,
        model.fuse_gate.weight,
        *attn.control_proj.weight.split(attn.control_splits),
    ):
        assert output_rms(weight) == pytest.approx(expected, rel=0.08)

    controls = attn.control_proj.weight.split(attn.control_splits)
    for down, up in (
        (controls[0], attn.decay_up.weight),
        (controls[4], attn.output_gate_up.weight),
    ):
        assert output_rms(up @ down) == pytest.approx(
            expected * BASE_NORMAL_INIT_STD * math.sqrt(128), rel=0.08
        )

    for weight in (
        model.embed_tokens.weight,
        attn.decay_up.weight,
        attn.output_gate_up.weight,
    ):
        assert weight.square().mean().sqrt().item() == pytest.approx(
            BASE_NORMAL_INIT_STD, rel=0.08
        )


def test_base_init_constant_tunes_nadam_matrices_only(monkeypatch):
    from delta_feedback_experiment import model as model_module
    from delta_feedback_experiment.parameter_groups import is_normuonh_parameter

    baseline = tiny("arf", seed=23)
    baseline_rng = torch.random.get_rng_state()
    monkeypatch.setattr(model_module, "BASE_NORMAL_INIT_STD", 2 * BASE_NORMAL_INIT_STD)
    tuned = tiny("arf", seed=23)
    assert torch.equal(torch.random.get_rng_state(), baseline_rng)
    baseline_parameters = dict(baseline.named_parameters())
    for name, parameter in tuned.named_parameters():
        expected = baseline_parameters[name]
        if parameter.ndim == 2 and not is_normuonh_parameter(name, parameter):
            expected = expected * 2
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0, msg=name)


def test_readout_and_width_group_carry_the_mup_ratio():
    """The muP reference width is the flagship, so its ratio is one and the
    screen's is two; the readout multiplies its logits by the ratio, and the
    width-scaled NAdam group is exactly the NAdam matrices with fan-in D."""
    from delta_feedback_experiment.model import MUP_BASE_DIM, ModelConfig
    from delta_feedback_experiment.optim import DEFAULT_NADAM_LR, build_optimizers
    from delta_feedback_experiment.parameter_groups import (
        is_normuonh_parameter,
        is_width_scaled_parameter,
    )
    from delta_feedback_experiment.train import SCALES

    assert MUP_BASE_DIM == SCALES["flagship"]["dim"] == 1536
    assert ModelConfig(dim=SCALES["flagship"]["dim"]).mup_ratio == 1.0
    assert ModelConfig(dim=SCALES["bridge"]["dim"]).mup_ratio == pytest.approx(4 / 3)
    assert ModelConfig(dim=SCALES["screen"]["dim"]).mup_ratio == 2.0

    model = tiny("arf")
    h = torch.randn(2, 5, TINY["dim"])
    plain = torch.nn.functional.linear(model.final_norm(h), model.embed_tokens.weight)
    assert model.cfg.mup_ratio == 48
    torch.testing.assert_close(model.readout_input(h), model.final_norm(h) * 48)
    torch.testing.assert_close(model.logits(h), plain * 48)

    # At the reference width the parametrization is the plain one.
    reference = tiny("arf", dim=1536)
    h = torch.randn(2, 5, 1536)
    assert reference.cfg.mup_ratio == 1.0
    assert reference.readout_input(h) is not None
    torch.testing.assert_close(reference.readout_input(h), reference.final_norm(h))
    _, nadam = build_optimizers(reference)
    assert [group["lr"] for group in nadam.param_groups] == [DEFAULT_NADAM_LR] * 2

    for condition in BUILDABLE + LOOPED:
        model = tiny(condition)
        width = [
            name
            for name, p in model.named_parameters()
            if is_width_scaled_parameter(name, p)
        ]
        assert width, condition
        for name, p in model.named_parameters():
            scaled = is_width_scaled_parameter(name, p)
            assert not (scaled and is_normuonh_parameter(name, p)), name
            if scaled:
                assert p.ndim == 2 and p.shape[1] == TINY["dim"], name
            elif p.ndim == 2 and not is_normuonh_parameter(name, p):
                # The other NAdam matrices read one token or one head width.
                assert name == "embed_tokens.weight" or p.shape[1] == TINY["pkda_head_dim"], name


def test_classifier_shadow_is_derived_and_preserves_master_gradients():
    from delta_feedback_experiment.model import _ClassifierShadow

    model = tiny("arf")
    assert model._classifier_shadow is None
    operand, accumulator = model.classifier_for_loss()
    assert operand is model.embed_tokens.weight and accumulator is None
    assert not any("classifier_shadow" in name for name in model.state_dict())

    master = torch.randn(11, 7, dtype=torch.float32, requires_grad=True)
    shadow = master.detach().to(torch.bfloat16)
    operand = _ClassifierShadow.apply(master, shadow, None)
    operand.float().sum().backward()
    assert master.grad.dtype == torch.float32
    assert torch.equal(master.grad, torch.ones_like(master))
    assert shadow.grad is None


def test_sink_linear_accumulates_segment_gradients_in_place():
    """One GEMM over concatenated weights; each segment's dW lands in its own
    persistent FP32 buffer and the parameters see no autograd gradient."""
    from delta_feedback_experiment.cuda_kernels import sink_linear

    torch.manual_seed(3)
    x = torch.randn(2, 5, 8, requires_grad=True)
    weights = tuple(torch.nn.Parameter(torch.randn(rows, 8)) for rows in (6, 3))
    sinks = tuple(torch.full_like(w, 0.25) for w in weights)
    reference = torch.nn.functional.linear(x, torch.cat(weights))
    out = sink_linear(x, weights, sinks)
    assert torch.equal(out, reference)
    cotangent = torch.randn_like(out)
    out.backward(cotangent)
    x_ref = x.detach().clone().requires_grad_()
    w_ref = tuple(w.detach().clone().requires_grad_() for w in weights)
    torch.nn.functional.linear(x_ref, torch.cat(w_ref)).backward(cotangent)
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6)
    for weight, sink, ref in zip(weights, sinks, w_ref, strict=True):
        assert weight.grad is None
        assert torch.allclose(sink, 0.25 + ref.grad, atol=1e-5)
    # An unbound sink or disabled autograd falls back to the plain linear.
    assert torch.equal(sink_linear(x, weights, (sinks[0], None)), reference)
    with torch.no_grad():
        assert torch.equal(sink_linear(x, weights, sinks), reference)


def test_gradient_sinks_bind_every_large_projection_and_unbind_cleanly():
    model = tiny("arf").train()
    buffers = {p: torch.zeros_like(p) for p in model.parameters()}
    bound = model.bind_gradient_sinks(buffers)
    names = {name for name, p in model.named_parameters() if p in bound}
    assert "embed_tokens.weight" not in names  # CUDA-only: CCE owns that path
    assert "blocks.0.attn.q_proj.weight" in names
    assert "blocks.3.attn.qkv_proj.weight" in names
    assert "attention_gates.0.weight" in names
    assert "blocks.1.mlp.down_proj.weight" in names
    assert "fuse_value.weight" in names and "fuse_gate.weight" in names
    assert not any(".norm" in name or "router" in name for name in names)
    toks = tokens()
    prefix = torch.ones((1, toks.shape[0]), dtype=torch.long)
    outs = multipass(model, toks, 2, prefix_lens=prefix)
    total, _ = multipass_loss(model, toks, outs)
    total.backward()
    for name, p in model.named_parameters():
        if p in bound:
            assert p.grad is None, name
            assert buffers[p].abs().sum() > 0, name
        else:
            assert p.grad is not None, name
    assert model.bind_gradient_sinks(None) == set()
    model.zero_grad(set_to_none=True)
    outs = multipass(model, toks, 2, prefix_lens=prefix)
    multipass_loss(model, toks, outs)[0].backward()
    assert model.blocks[0].attn.q_proj.weight.grad is not None


def test_tied_embedding_sink_accumulates_both_gradient_paths_in_place():
    """With a trainer-owned FP32 buffer, the lookup scatters rows and the
    classifier adds its BF16 gradient directly; autograd sees no gradient."""
    from delta_feedback_experiment.model import _ClassifierShadow, _EmbeddingSink

    model = tiny("arf").train()
    weight = model.embed_tokens.weight
    toks = tokens()
    sink = torch.zeros_like(weight)

    e = _EmbeddingSink.apply(toks, weight, sink)
    (e * 0.5).sum().backward()
    assert weight.grad is None
    reference = torch.zeros_like(weight).index_add_(
        0, toks.reshape(-1), torch.full((toks.numel(), weight.shape[1]), 0.5)
    )
    assert torch.equal(sink, reference)

    shadow = weight.detach().to(torch.bfloat16)
    _ClassifierShadow.apply(weight, shadow, sink).float().sum().backward()
    assert weight.grad is None
    assert torch.equal(sink, reference + 1)

    model.embed_tokens(toks).sum().backward()
    assert weight.grad is not None


def test_zero_gqa_gate_halves_the_ungated_attention_branch():
    """``o = W_o(sigmoid(W_g x) * GQA(q, k, v))``: a zero gate is a factor 1/2,
    on the plain trunk's fourth (NoPE) layer here."""
    model = tiny("")
    cfg = model.cfg
    attention = model.blocks[3].attn
    gate_weight = model.attention_gates[3].weight
    x = torch.randn(2, 7, TINY["dim"])

    def heads(t, n):
        return t.view(2, 7, n, cfg.head_dim).transpose(1, 2)

    with torch.no_grad():
        gate_weight.zero_()
        gated = attention(x, gate_weight, None, 3)
        q, k, v = attention.qkv_proj(x).split(
            (attention.q_size, attention.kv_size, attention.kv_size), dim=-1
        )
        z = torch.nn.functional.scaled_dot_product_attention(
            attention.q_norm(heads(q, cfg.heads)),
            attention.k_norm(heads(k, cfg.kv_heads)),
            heads(v, cfg.kv_heads),
            is_causal=True,
            enable_gqa=cfg.heads != cfg.kv_heads,
        )
        ungated = attention.o_proj(z.transpose(1, 2).reshape(2, 7, -1))
    assert torch.allclose(gated, 0.5 * ungated, atol=1e-6)


# -- routing semantics ---------------------------------------------------------


def test_zero_init_routing_uniform():
    """Zero-init queries route uniformly, including over every null."""
    for condition in ("ar", "arf"):
        out = forward(tiny(condition), tokens(), want_weights=True)
        assert out.route_source_names["L0.attn"] == ("null", "seed")
        for site, weights in out.route_weights.items():
            n = weights.shape[0]
            assert weights.shape[-1] == TINY["kv_heads"]
            assert torch.allclose(weights, torch.full_like(weights, 1.0 / n)), (
                f"{condition} {site}"
            )


def test_algebraic_router_matches_normalized_reference_in_fp32():
    model = tiny("ar")
    router = model.blocks[2].mlp_router
    with torch.no_grad():
        router.query.normal_()
        router.key_norm.weight.uniform_(0.5, 1.5)
    sources = [torch.randn(2, 7, 32) for _ in range(5)]
    routed, weights = router(sources, True)
    values = torch.stack([router.null.expand_as(sources[0]), *sources])
    h = model.cfg.kv_heads
    k = model.cfg.dim // h
    logits = torch.einsum(
        "hk,nbthk->nbth",
        router.query.view(h, k),
        router.key_norm(values).view(6, 2, 7, h, k),
    )
    reference_weights = logits.softmax(dim=0)
    reference = torch.einsum(
        "nbth,nbthk->bthk", reference_weights, values.view(6, 2, 7, h, k)
    ).reshape(2, 7, 32)
    assert torch.allclose(weights, reference_weights, rtol=2e-5, atol=2e-6)
    assert torch.allclose(routed, reference, rtol=2e-5, atol=2e-6)


def test_routing_heads_can_select_different_sources():
    model = tiny("ar")
    router = model.blocks[2].mlp_router
    source0 = torch.zeros(1, 1, TINY["dim"])
    source1 = torch.zeros_like(source0)
    source0[..., :16] = 1
    source1[..., 16:] = 1
    with torch.no_grad():
        router.query.fill_(8)
    routed, weights = router([source0, source1], True)
    assert weights[:, 0, 0, 0].argmax().item() == 1
    assert weights[:, 0, 0, 1].argmax().item() == 2
    assert routed[..., :16].mean() > 0.99
    assert routed[..., 16:].mean() > 0.99


def test_null_value_follows_activation_dtype_with_fp32_gradient():
    model = tiny("ar").train()
    router = model.blocks[0].attn_router
    source = torch.randn(1, 3, TINY["dim"], dtype=torch.bfloat16)
    routed, _ = router([source], False)
    assert routed.dtype == torch.bfloat16
    routed.float().sum().backward()
    assert router.null.grad is not None
    assert router.null.grad.dtype == torch.float32


def test_single_head_routing_and_unknown_letters_are_rejected():
    with pytest.raises(ValueError, match="unknown condition letters 'd'"):
        condition_config("dar", **TINY)
    with pytest.raises(ValueError, match="unknown condition letters 'bdhm'"):
        condition_config("mhdb", **TINY)
    with pytest.raises(ValueError, match="repeated letter"):
        condition_config("arra", **TINY)
    cfg = condition_config("ar", **(TINY | {"kv_heads": 1}))
    with pytest.raises(ValueError, match="at least two"):
        DeltaModel(cfg)
    with pytest.raises(ValueError, match="block size"):
        condition_config("ar", **(TINY | {"routing_block_size": 0}))


def test_telescoping():
    """The stream is exactly seed + Σdeltas at the top — transient-read
    routing never leaks into the residual stream."""
    for condition in ("ar", "arf"):
        model = tiny(condition)
        # Sharpen every query so routing is far from uniform — the identity
        # must hold because of *semantics*, not because routing is ~0.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "query" in name:
                    parameter.normal_(std=1.0)
        out = forward(model, tokens())
        stream_seed = out.sources[out.n_seeds - 1]  # x is always the last seed
        deltas = out.sources[out.n_seeds :]
        rebuilt = stream_seed + torch.stack(deltas).sum(0)
        assert torch.allclose(out.h_top, rebuilt, atol=1e-5), condition


def test_block_sources_and_current_partial_labels():
    model = tiny("arf")
    out = forward(model, tokens(), want_weights=True)
    assert out.source_names == ("seed", "block0")
    assert out.route_source_names["L0.attn"] == ("null", "seed")
    assert out.route_source_names["L0.mlp"] == ("null", "seed", "partial0")
    assert out.route_source_names["L2.attn"] == ("null", "seed", "partial0")
    assert out.route_source_names["payload"] == ("null", "seed", "block0")


def test_four_layer_block_boundaries():
    config = TINY | {"layers": 9}
    torch.manual_seed(0)
    model = DeltaModel(condition_config("ar", **config)).eval()
    out = model.forward_column(model.embed_tokens(tokens()), want_weights=True)
    assert out.source_names == ("seed", "block0", "block1", "block2")
    assert out.route_source_names["L4.attn"] == ("null", "seed", "block0")
    assert out.route_source_names["L5.attn"] == (
        "null",
        "seed",
        "block0",
        "partial1",
    )
    assert out.route_source_names["L8.attn"] == (
        "null",
        "seed",
        "block0",
        "block1",
    )
    rebuilt = out.sources[0] + torch.stack(out.sources[1:]).sum(0)
    assert torch.allclose(out.h_top, rebuilt, atol=1e-5)


def test_payload_init_matches_bare_top_state():
    """The null-plus-complete decomposition is collinear with h at init."""
    model = tiny("arf")
    out = forward(model, tokens())
    n_sources = 1 + (len(out.sources) - out.n_seeds + 1)
    expected = model.payload_norm(out.h_top * (1 + 1 / n_sources))
    assert torch.allclose(out.payload, expected, atol=1e-5)
    assert torch.allclose(
        out.payload,
        model.payload_norm(out.h_top),
        rtol=3e-3,
        atol=3e-3,
    )

    model = tiny("af")
    out = forward(model, tokens())
    assert torch.allclose(out.payload, model.payload_norm(out.h_top), atol=1e-6)


# -- multi-pass ----------------------------------------------------------------


def test_multipass_k1_is_plain_forward():
    model = tiny("arf")
    toks = tokens()
    single = multipass(model, toks, 1)
    plain = forward(model, toks[:, :-1])
    assert torch.equal(single[0].h_top, plain.h_top)
    assert single[0].h_top.shape[1] == toks.shape[1] - 1
    assert single[0].payload is None  # no consumer exists after the final pass


def test_multipass_matches_the_causally_equivalent_full_row_forward():
    """Executing only the T input positions of a T+1 row is the same
    function: the dropped final column never fed a loss or a payload."""
    for condition in ("arf", "af", "a"):
        model = tiny(condition)
        toks = tokens()
        prefix = torch.ones((2, toks.shape[0]), dtype=torch.long) * 3
        jitter = torch.randn((2, *toks.shape, TINY["dim"])) * 0.02
        n_passes = 3 if model.cfg.feedback else 1
        outs = multipass(
            model,
            toks,
            n_passes,
            prefix_lens=prefix if n_passes > 1 else None,
            jitter=jitter if n_passes > 1 else None,
        )
        total, losses = multipass_loss(model, toks, outs)
        assert len(losses) == n_passes and torch.isfinite(total)

        # Literal full-row reference: run every stored token and drop the last.
        e = model.embed_tokens(toks)
        ref = [model.forward_column(e, need_payload=n_passes > 1)]
        length = toks.shape[1]
        positions = torch.arange(length)
        for i in range(n_passes - 1):
            p = shift_right(ref[-1].payload + jitter[i])
            plain = positions[None, :] < prefix[i][:, None]
            ref.append(
                model.forward_column(
                    torch.where(plain[..., None], e, model.fuse(p, e)),
                    need_payload=i < n_passes - 2,
                )
            )
        for out, full in zip(outs, ref, strict=True):
            assert torch.allclose(out.h_top, full.h_top[:, :-1], atol=1e-5), condition
        ref_total, _ = multipass_loss(
            model, toks, [type(o)(**{**vars(o), "h_top": o.h_top[:, :-1]}) for o in ref]
        )
        assert torch.allclose(total, ref_total, atol=1e-5), condition


def test_all_plain_prefix_degenerates_to_pass1():
    """A feedback pass whose prefix covers the whole sequence reproduces
    pass 1 exactly — the mixin's boundary semantics."""
    for condition in ("arf", "af"):
        model = tiny(condition)
        toks = tokens()
        length = toks.shape[1]
        prefix = torch.full((1, toks.shape[0]), length)
        outs = multipass(model, toks, 2, prefix_lens=prefix)
        assert torch.allclose(outs[1].h_top, outs[0].h_top, atol=1e-6), condition


def test_multipass_token_causality():
    """Changing token t leaves every pass's outputs at positions < t
    unchanged: the Jacobi shift preserves causality."""
    for condition in ("arf",):
        model = tiny(condition)
        toks = tokens()
        length = toks.shape[1]
        prefix = torch.ones((2, toks.shape[0]), dtype=torch.long)
        outs = multipass(model, toks, 3, prefix_lens=prefix)
        edited = toks.clone()
        edited[:, length // 2] = (edited[:, length // 2] + 1) % TINY["vocab_size"]
        outs_edited = multipass(model, edited, 3, prefix_lens=prefix)
        for out, out_edited in zip(outs, outs_edited, strict=True):
            assert torch.allclose(
                out.h_top[:, : length // 2],
                out_edited.h_top[:, : length // 2],
                atol=1e-5,
            ), condition


def test_multipass_loss_shape():
    model = tiny("arf")
    toks = tokens()
    prefix = torch.ones((2, toks.shape[0]), dtype=torch.long)
    outs = multipass(model, toks, 3, prefix_lens=prefix)
    total, losses = multipass_loss(model, toks, outs)
    assert len(losses) == 3
    expected = losses[0] + (losses[1] + losses[2]) / 2
    assert torch.allclose(total, expected)


def test_multipass_loss_accepts_a_device_side_z_coefficient():
    model = tiny("arf")
    toks = tokens()
    prefix = torch.ones((1, toks.shape[0]), dtype=torch.long)
    outs = multipass(model, toks, 2, prefix_lens=prefix)
    float_total, _ = multipass_loss(model, toks, outs, z_coef=1e-5)
    tensor_total, _ = multipass_loss(model, toks, outs, z_coef=torch.tensor(1e-5))
    assert torch.equal(float_total, tensor_total)


def test_multipass_rejects_conditions_without_f():
    with pytest.raises(ValueError):
        multipass(
            tiny(""),
            tokens(),
            2,
            prefix_lens=torch.ones((1, 2), dtype=torch.long),
        )


def test_gradients_reach_the_feedback_machinery():
    model = tiny("arf").train()
    toks = tokens()
    prefix = torch.ones((2, toks.shape[0]), dtype=torch.long)
    jitter = torch.zeros((2, *toks.shape, TINY["dim"]))
    outs = multipass(model, toks, 3, prefix_lens=prefix, jitter=jitter)
    total, _ = multipass_loss(model, toks, outs)
    total.backward()
    for name in (
        "attention_gates.0.weight",
        "fuse_value.weight",
        "fuse_gate.weight",
        "payload_router.query",
    ):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and gradient.abs().sum() > 0, name


# -- decoding ------------------------------------------------------------------


def reference_decode(model, toks, prompt_len):
    """The exact recurrence of FBT Eq. 8 by full recomputation.

    Grows the input matrix one column at a time: column t's input uses
    the payload of column t−1 computed in its own context, and causality
    makes each full recomputation agree with the incremental cached one.
    Returns h_top for columns prompt_len..length−1.
    """
    _, length = toks.shape
    e = model.embed_tokens(toks)
    rows = e[:, :prompt_len]

    for t in range(prompt_len, length):
        p_last = model.forward_column(rows).payload[:, -1:]
        rows = torch.cat([rows, model.fuse(p_last, e[:, t : t + 1])], dim=1)
    return model.forward_column(rows).h_top[:, prompt_len:]


def test_cached_feedback_decode_matches_exact_recurrence():
    """Sequential cached stepping equals the recurrence computed by full
    recomputation — the train/decode parity invariant, per feedback condition."""
    for condition in ("f", "rf", "af", "arf"):
        model = tiny(condition)
        cfg = model.cfg
        toks = tokens(batch=2, length=12)
        prompt_len = 5

        cache = KVCache(cfg, batch=2, device=toks.device, dtype=torch.float32)
        prefill = model.forward_column(
            model.embed_tokens(toks[:, :prompt_len]), cache=cache
        )
        stepped = []
        payload = prefill.payload[:, -1:]
        for t in range(prompt_len, toks.shape[1]):
            out = model.step(toks[:, t : t + 1], payload, cache)
            stepped.append(out.h_top)
            payload = out.payload
        stepped = torch.cat(stepped, dim=1)

        reference = reference_decode(model, toks, prompt_len)
        # Parallel and single-column attention accumulate FP32 products in
        # different orders; keep the bound tight relative to O(1) activations.
        assert torch.allclose(stepped, reference, atol=5e-5), condition


def test_cached_standard_decode_matches_full_forward():
    """Standard decoding (no feedback) through the cache equals one full
    parallel forward, on every condition."""
    for condition in BUILDABLE:
        model = tiny(condition)
        toks = tokens(batch=2, length=10)
        full = forward(model, toks)

        cache = KVCache(model.cfg, batch=2, device=toks.device, dtype=torch.float32)
        prefill = model.forward_column(model.embed_tokens(toks[:, :4]), cache=cache)
        pieces = [prefill.h_top]
        for t in range(4, toks.shape[1]):
            pieces.append(model.step(toks[:, t : t + 1], None, cache).h_top)
        stepped = torch.cat(pieces, dim=1)
        assert torch.allclose(stepped, full.h_top, atol=3e-5), condition


def test_hybrid_cache_owns_only_global_kv_and_fixed_pkda_states():
    model = tiny("a")
    toks = tokens(batch=2, length=5)
    cache = KVCache(model.cfg, batch=2, device=toks.device, dtype=torch.float32)
    model.forward_column(model.embed_tokens(toks), cache=cache)
    assert cache.k.shape[:2] == (1, 2)
    assert set(cache.pkda_states) == {(0, 0), (1, 0), (2, 0)}
    for state, a_state, conv_state in cache.pkda_states.values():
        assert state.shape == (2, 2, 16, 16) and state.dtype == torch.float32
        assert a_state.shape == (2, 2, 16) and a_state.dtype == torch.float32
        assert all(part.shape == (2, 32, 3) for part in conv_state)


# -- the loop ------------------------------------------------------------------


def test_loop_at_one_iteration_is_the_unlooped_condition():
    """At r = 1 the column executes the same layers with the same parameters
    and the same banks: values, routes, losses, and gradients coincide with
    the condition without l, at three cells and with a multi-cell core."""
    cases = [(condition, 12) for condition in LOOPED]
    cases += [("arl", 16), ("arfl", 20)]
    for condition, layers in cases:
        looped = tiny(condition, layers=layers).train()
        flat = tiny(condition.replace("l", ""), layers=layers).train()
        toks = tokens()
        n_passes = 2 if looped.cfg.feedback else 1
        prefix = torch.ones((1, toks.shape[0]), dtype=torch.long) * 3
        kwargs = {"prefix_lens": prefix} if n_passes > 1 else {}
        outs_l = multipass(
            looped, toks, n_passes, iterations=1, want_weights=True, **kwargs
        )
        outs_f = multipass(flat, toks, n_passes, want_weights=True, **kwargs)
        for out_l, out_f in zip(outs_l, outs_f, strict=True):
            assert torch.equal(out_l.h_top, out_f.h_top), condition
            assert out_l.source_names == out_f.source_names
            if out_l.payload is not None:
                assert torch.equal(out_l.payload, out_f.payload)
            # The core's sites carry an iteration tag; everything else matches.
            translated = {
                site.replace("i0.", "."): weights
                for site, weights in out_l.route_weights.items()
            }
            assert translated.keys() == out_f.route_weights.keys(), condition
            for site, weights in translated.items():
                assert torch.equal(weights, out_f.route_weights[site]), (
                    condition,
                    site,
                )
            translated_names = {
                site.replace("i0.", "."): names
                for site, names in out_l.route_source_names.items()
            }
            assert translated_names == out_f.route_source_names, condition
        total_l, _ = multipass_loss(looped, toks, outs_l)
        total_f, _ = multipass_loss(flat, toks, outs_f)
        assert torch.equal(total_l, total_f), condition
        total_l.backward()
        total_f.backward()
        for (name, p_l), (_, p_f) in zip(
            looped.named_parameters(), flat.named_parameters(), strict=True
        ):
            assert (p_l.grad is None) == (p_f.grad is None), (condition, name)
            if p_l.grad is not None:
                assert torch.equal(p_l.grad, p_f.grad), (condition, name)


def test_loop_banks_sites_and_telescoping():
    """The core is one cell: its partial is measured from the prelude output
    across iterations, absent only at the attention entry of iteration 1, and
    the residual still telescopes over seed and the three cell deltas."""
    model = tiny("arfl")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "query" in name:
                parameter.normal_(std=1.0)
    out = forward(model, tokens(), want_weights=True, iterations=3)
    assert out.iterations == 3
    assert out.source_names == ("seed", "block0", "block1", "block2")
    names = out.route_source_names
    assert names["L3.mlp"] == ("null", "seed", "partial0")
    assert names["L4i0.attn"] == ("null", "seed", "block0")
    assert names["L4i0.mlp"] == ("null", "seed", "block0", "partial1")
    assert names["L4i1.attn"] == ("null", "seed", "block0", "partial1")
    assert names["L4i2.attn"] == ("null", "seed", "block0", "partial1")
    assert names["L7i2.mlp"] == ("null", "seed", "block0", "partial1")
    assert names["L8.attn"] == ("null", "seed", "block0", "block1")
    assert names["L11.mlp"] == ("null", "seed", "block0", "block1", "partial2")
    assert names["payload"] == ("null", "seed", "block0", "block1", "block2")
    assert not any("L4.attn" == site or "L8i" in site for site in names)
    rebuilt = out.sources[0] + torch.stack(out.sources[1:]).sum(0)
    assert torch.allclose(out.h_top, rebuilt, atol=1e-5)
    assert torch.allclose(out.core_state - out.core_entry, out.sources[2], atol=1e-5)
    assert torch.allclose(out.core_entry, out.sources[0] + out.sources[1], atol=1e-5)


def test_loop_core_cells_keep_their_own_deltas():
    """With more than one core cell each cell's block delta accumulates across
    iterations under its own name. A core site reads the cells before it from
    this iteration and the cells after it from the previous one, its own as
    the live partial; on the last iteration a finished cell is in the bank.
    The residual telescopes over the seed and one delta per cell."""
    model = tiny("arfl", layers=16)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "query" in name:
                parameter.normal_(std=1.0)
    out = forward(model, tokens(), want_weights=True, iterations=3)
    assert out.source_names == ("seed", "block0", "block1", "block2", "block3")
    names = out.route_source_names
    assert names["L4i0.attn"] == ("null", "seed", "block0")
    assert names["L4i0.mlp"] == ("null", "seed", "block0", "partial1")
    assert names["L8i0.attn"] == ("null", "seed", "block0", "block1")
    assert names["L8i0.mlp"] == ("null", "seed", "block0", "block1", "partial2")
    assert names["L4i1.attn"] == ("null", "seed", "block0", "block2", "partial1")
    assert names["L8i1.attn"] == ("null", "seed", "block0", "block1", "partial2")
    assert names["L4i2.attn"] == ("null", "seed", "block0", "block2", "partial1")
    assert names["L8i2.mlp"] == ("null", "seed", "block0", "block1", "partial2")
    assert names["L12.attn"] == ("null", "seed", "block0", "block1", "block2")
    assert names["L15.mlp"] == ("null", "seed", "block0", "block1", "block2", "partial3")
    assert names["payload"] == ("null", "seed", "block0", "block1", "block2", "block3")
    rebuilt = out.sources[0] + torch.stack(out.sources[1:]).sum(0)
    assert torch.allclose(out.h_top, rebuilt, atol=1e-5)
    core_delta = out.sources[2] + out.sources[3]
    assert torch.allclose(out.core_state - out.core_entry, core_delta, atol=1e-5)
    # Each cell's delta is its own: the first core cell's delta at one
    # iteration is what it wrote on iteration 0, which a deeper run then
    # continues rather than replaces.
    one = forward(model, tokens(), want_weights=True, iterations=1)
    assert torch.equal(one.core_entry, out.core_entry)
    assert not torch.allclose(one.sources[2], out.sources[2])
    assert torch.equal(one.route_weights["L4i0.attn"], out.route_weights["L4i0.attn"])
    assert torch.equal(one.route_weights["L8i0.mlp"], out.route_weights["L8i0.mlp"])


def test_loop_same_depth_iterations_are_a_prefix():
    """Iteration i of a column reads earlier positions' iteration-i writes, so
    the state after i iterations does not depend on how many follow: a deeper
    run repeats the shallower run's iterations exactly."""
    model = tiny("arfl")
    toks = tokens()
    two = forward(model, toks, want_weights=True, iterations=2)
    three = forward(model, toks, want_weights=True, iterations=3)
    assert torch.equal(two.core_entry, three.core_entry)
    for site, weights in two.route_weights.items():
        if "i0." in site or "i1." in site:
            assert torch.equal(weights, three.route_weights[site]), site
    assert not torch.allclose(two.core_state, three.core_state)
    assert not torch.allclose(two.h_top, three.h_top)


def test_loop_gradients_sum_across_iterations():
    model = tiny("arfl").train()
    toks = tokens()
    prefix = torch.ones((1, toks.shape[0]), dtype=torch.long)
    core_name = "blocks.5.mlp.down_proj.weight"

    def gradient(iterations):
        model.zero_grad(set_to_none=True)
        outs = multipass(model, toks, 2, prefix_lens=prefix, iterations=iterations)
        total, _ = multipass_loss(model, toks, outs)
        total.backward()
        return dict(model.named_parameters())[core_name].grad.clone()

    one, three = gradient(1), gradient(3)
    assert one.abs().sum() > 0 and three.abs().sum() > 0
    assert not torch.allclose(one, three)


def test_loop_cached_decode_matches_full_forward():
    """Standard decoding through per-iteration core tracks equals one full
    parallel forward at the same iteration count."""
    for condition in LOOPED:
        for iterations in (1, 3):
            model = tiny(condition)
            toks = tokens(batch=2, length=10)
            full = forward(model, toks, iterations=iterations)
            cache = KVCache(
                model.cfg, batch=2, device=toks.device, dtype=torch.float32,
                iterations=iterations,
            )
            prefill = model.forward_column(model.embed_tokens(toks[:, :4]), cache=cache)
            pieces = [prefill.h_top]
            for t in range(4, toks.shape[1]):
                pieces.append(model.step(toks[:, t : t + 1], None, cache).h_top)
            stepped = torch.cat(pieces, dim=1)
            # Parallel and single-column mixers accumulate FP32 products in
            # different orders, and the loop executes up to twenty layers, so
            # the bound is relative rather than the four-layer absolute one.
            relative = (stepped - full.h_top).norm() / full.h_top.norm()
            assert relative < 1e-4, (condition, iterations, relative)
            assert cache.iteration == 0
    model = tiny("arfl")
    cache = KVCache(model.cfg, batch=2, device="cpu", dtype=torch.float32, iterations=3)
    assert len(cache.global_slots) == 1 + 3 + 1
    with pytest.raises(ValueError, match="tracks"):
        model.forward_column(model.embed_tokens(tokens()), cache=cache, iterations=2)


@pytest.mark.parametrize("condition", ["fl", "rfl", "afl", "arfl"])
def test_loop_cached_feedback_decode_matches_exact_recurrence(condition):
    model = tiny(condition)
    toks = tokens(batch=2, length=12)
    prompt_len = 5
    cache = KVCache(model.cfg, batch=2, device=toks.device, dtype=torch.float32)
    assert cache.iterations == 2
    prefill = model.forward_column(model.embed_tokens(toks[:, :prompt_len]), cache=cache)
    stepped = []
    payload = prefill.payload[:, -1:]
    for t in range(prompt_len, toks.shape[1]):
        out = model.step(toks[:, t : t + 1], payload, cache)
        stepped.append(out.h_top)
        payload = out.payload
    stepped = torch.cat(stepped, dim=1)
    reference = reference_decode(model, toks, prompt_len)
    assert (stepped - reference).norm() / reference.norm() < 1e-4


def test_depth_trace_runs_and_is_one_trajectory():
    model = tiny("arfl")
    toks = tokens(batch=2, length=12)
    records = depth_trace(model, toks)
    assert [record["iterations"] for record in records] == [1, 2, 3]
    assert all(
        math.isfinite(record["loss"]) and math.isfinite(record["update_norm"])
        for record in records
    )
    fused = depth_trace(model, toks, 2, fused=True)
    assert len(fused) == 2 and all(math.isfinite(r["loss"]) for r in fused)
    with pytest.raises(ValueError, match="needs a condition with l"):
        depth_trace(tiny("arf"), toks)
    with pytest.raises(ValueError, match="with f"):
        depth_trace(tiny("arl"), toks, fused=True)


def test_contraction_diagnostic_runs():
    model = tiny("arf")
    records = iterate_fused(model, tokens(batch=2, length=12), n_iters=3)
    assert len(records) == 3
    assert all(
        math.isfinite(r["loss"]) and math.isfinite(r["update_norm"]) for r in records
    )


def test_shift_right():
    x = torch.arange(6.0).reshape(1, 3, 2)
    shifted = shift_right(x)
    assert torch.equal(shifted[0, 0], torch.zeros(2))
    assert torch.equal(shifted[0, 1:], x[0, :-1])

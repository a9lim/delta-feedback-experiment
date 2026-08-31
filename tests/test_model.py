"""Offline invariants for the DF model family (CPU, tiny config).

These are the design contract's checkable claims: transient-read routing
keeps the stream telescoping, zero-init routing is uniform, the payload
init-matches the bare top state, the Jacobi passes are token-causal, the
prefix mixin degenerates to pass 1, and cached sequential decoding equals
the exact recurrence computed by full recomputation.
"""

import math

import pytest
import torch

from delta_feedback_experiment.model import (
    ARMS,
    DFModel,
    KVCache,
    arm_config,
    iterate_fused,
    multipass,
    multipass_loss,
    shift_right,
)

TINY = {
    "vocab_size": 97,
    "dim": 32,
    "layers": 3,
    "heads": 2,
    "kv_heads": 2,
    "head_dim": 16,
    "intermediate": 64,
    "max_seq_len": 32,
}


def tiny(arm: str, seed: int = 0) -> DFModel:
    torch.manual_seed(seed)
    return DFModel(arm_config(arm, **TINY)).eval()


def tokens(batch=2, length=16, seed=1, vocab=97):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, length), generator=generator)


def forward(model, toks, **kwargs):
    return model.forward_column(model.embed_tokens(toks), **kwargs)


# -- structure -----------------------------------------------------------------


def test_arm_flags():
    vanilla = arm_config("vanilla", **TINY)
    assert not vanilla.routing_active
    assert not vanilla.feedback_active
    assert not vanilla.gated_attention
    mhdb = arm_config("mhdb", **TINY)
    assert mhdb.routing_active
    assert not mhdb.feedback_active
    assert mhdb.gated_attention
    assert mhdb.routing_heads == TINY["kv_heads"]
    assert mhdb.routing_block_size == 4
    fbt = arm_config("fbt", **TINY)
    assert fbt.gated_entry
    assert not fbt.routing_active
    assert not fbt.gated_attention
    df = arm_config("df", **TINY)
    assert df.gated_entry
    assert df.routing_active
    assert df.gated_attention
    soft = arm_config("df_soft", **TINY)
    assert soft.routing_active and soft.feedback_active and not soft.gated_entry
    assert soft.gated_attention


def test_parents_are_deletions():
    """Every parent's parameter set is a strict subset of DF's."""
    names = {arm: {name for name, _ in tiny(arm).named_parameters()} for arm in ARMS}
    assert names["vanilla"] < names["mhdb"] < names["df"]
    assert names["vanilla"] < names["fbt"] < names["df"]
    # The union misses exactly the payload router, which needs both axes.
    assert names["df"] - (names["mhdb"] | names["fbt"]) == {
        "payload_router.query",
        "payload_router.key_norm.weight",
        "payload_router.null",
    }
    # Every routed arm has per-site nulls; DF-soft omits the GLU entry.
    for arm in ("mhdb", "df", "df_soft"):
        assert any("null" in name for name in names[arm])
    assert not any("fuse_" in name for name in names["df_soft"])


def test_screen_param_count():
    expected = {
        "vanilla": 222_876_672,
        "mhdb": 230_009_856,
        "fbt": 224_058_624,
        "df": 231_194_112,
        "df_soft": 230_012_928,
    }
    with torch.device("meta"):
        for arm, count in expected.items():
            model = DFModel(arm_config(arm))
            assert sum(parameter.numel() for parameter in model.parameters()) == count


def test_large_projections_are_persistently_packed():
    model = tiny("df")
    attention = model.blocks[0].attn
    mlp = model.blocks[0].mlp
    assert attention.qkv_proj.weight.shape == (96, 32)
    assert not hasattr(attention, "q_proj")
    assert model.attention_gates[0].weight.shape == (32, 32)
    assert mlp.gate_up_proj.weight.shape == (128, 32)
    assert not hasattr(mlp, "gate_proj")


def test_factorial_initialization_is_paired_by_semantic_factor():
    models = {arm: tiny(arm, seed=17) for arm in ARMS}
    states = {arm: model.state_dict() for arm, model in models.items()}

    # Every parameter in vanilla is common and must be byte-identical in every
    # other cell. Conditional modules cannot advance the common RNG stream.
    for name, value in states["vanilla"].items():
        for arm in ARMS[1:]:
            assert torch.equal(value, states[arm][name]), (arm, name)

    # Each intervention package is also paired between its parent and DF.
    for layer in range(TINY["layers"]):
        name = f"attention_gates.{layer}.weight"
        assert torch.equal(states["mhdb"][name], states["df"][name])
        assert torch.equal(states["mhdb"][name], states["df_soft"][name])
        for kind in ("attn", "mlp"):
            null = f"blocks.{layer}.{kind}_router.null"
            assert torch.equal(states["mhdb"][null], states["df"][null])
            assert torch.equal(states["mhdb"][null], states["df_soft"][null])
    for name in ("fuse_value.weight", "fuse_gate.weight"):
        assert torch.equal(states["fbt"][name], states["df"][name])


def test_zero_gqa_gate_halves_the_ungated_attention_branch():
    model = tiny("mhdb")
    attention = model.blocks[0].attn
    gate_weight = model.attention_gates[0].weight
    x = torch.randn(2, 7, TINY["dim"])
    cos, sin = model.rope(x.device, 0, x.shape[1])
    with torch.no_grad():
        gate_weight.zero_()
        ungated = attention(x, cos, sin, None, None, 0)
        gated = attention(x, cos, sin, gate_weight, None, 0)
    assert torch.allclose(gated, 0.5 * ungated, atol=1e-6)


# -- routing semantics ---------------------------------------------------------


def test_zero_init_routing_uniform():
    """Zero-init queries route uniformly, including over every null."""
    for arm in ("mhdb", "df", "df_soft"):
        out = forward(tiny(arm), tokens(), want_weights=True)
        assert out.route_source_names["L0.attn"] == ("null", "seed")
        for site, weights in out.route_weights.items():
            n = weights.shape[0]
            assert weights.shape[-1] == TINY["kv_heads"]
            assert torch.allclose(weights, torch.full_like(weights, 1.0 / n)), (
                f"{arm} {site}"
            )


def test_algebraic_router_matches_normalized_reference_in_fp32():
    model = tiny("mhdb")
    router = model.blocks[2].mlp_router
    with torch.no_grad():
        router.query.normal_()
        router.key_norm.weight.uniform_(0.5, 1.5)
    sources = [torch.randn(2, 7, 32) for _ in range(5)]
    routed, weights = router(sources, [None] * len(sources), True)
    values = torch.stack([router.null.expand_as(sources[0]), *sources])
    h = model.cfg.routing_heads
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
    model = tiny("mhdb")
    router = model.blocks[2].mlp_router
    source0 = torch.zeros(1, 1, TINY["dim"])
    source1 = torch.zeros_like(source0)
    source0[..., :16] = 1
    source1[..., 16:] = 1
    with torch.no_grad():
        router.query.fill_(8)
    routed, weights = router([source0, source1], [None, None], True)
    assert weights[:, 0, 0, 0].argmax().item() == 1
    assert weights[:, 0, 0, 1].argmax().item() == 2
    assert routed[..., :16].mean() > 0.99
    assert routed[..., 16:].mean() > 0.99


def test_single_head_routing_and_unknown_arm_are_rejected():
    with pytest.raises(ValueError, match="unknown arm"):
        arm_config("dar", **TINY)
    with pytest.raises(ValueError, match="unknown arm"):
        arm_config("mhdar", **TINY)
    cfg = arm_config("mhdb", **(TINY | {"kv_heads": 1}))
    with pytest.raises(ValueError, match="at least two"):
        DFModel(cfg)
    with pytest.raises(ValueError, match="block size"):
        arm_config("mhdb", **(TINY | {"routing_block_size": 0}))


def test_telescoping():
    """The stream is exactly seed + Σdeltas at the top — transient-read
    routing never leaks into the residual stream."""
    for arm in ("mhdb", "df", "df_soft"):
        model = tiny(arm)
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
        assert torch.allclose(out.h_top, rebuilt, atol=1e-5), arm


def test_block_sources_and_current_partial_labels():
    model = tiny("df")
    out = forward(model, tokens(), want_weights=True)
    assert out.source_names == ("seed", "block0")
    assert out.route_source_names["L0.attn"] == ("null", "seed")
    assert out.route_source_names["L0.mlp"] == ("null", "seed", "partial0")
    assert out.route_source_names["L2.attn"] == ("null", "seed", "partial0")
    assert out.route_source_names["payload"] == ("null", "seed", "block0")


def test_four_layer_block_boundaries():
    config = TINY | {"layers": 9}
    torch.manual_seed(0)
    model = DFModel(arm_config("mhdb", **config)).eval()
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
    for arm in ("df", "df_soft"):
        model = tiny(arm)
        out = forward(model, tokens())
        n_sources = 1 + (len(out.sources) - out.n_seeds + 1)
        expected = model.payload_norm(out.h_top * (1 + 1 / n_sources))
        assert torch.allclose(out.payload, expected, atol=1e-5), arm
        assert torch.allclose(
            out.payload,
            model.payload_norm(out.h_top),
            rtol=3e-3,
            atol=3e-3,
        ), arm

    model = tiny("fbt")
    out = forward(model, tokens())
    assert torch.allclose(out.payload, model.payload_norm(out.h_top), atol=1e-6)


# -- multi-pass ----------------------------------------------------------------


def test_multipass_k1_is_plain_forward():
    model = tiny("df")
    toks = tokens()
    single = multipass(model, toks, 1)
    plain = forward(model, toks)
    assert torch.equal(single[0].h_top, plain.h_top)
    assert single[0].payload is None  # no consumer exists after the final pass


def test_all_plain_prefix_degenerates_to_pass1():
    """A feedback pass whose prefix covers the whole sequence reproduces
    pass 1 exactly — the mixin's boundary semantics, both entry forms."""
    for arm in ("df", "fbt", "df_soft"):
        model = tiny(arm)
        toks = tokens()
        length = toks.shape[1]
        prefix = torch.full((1, toks.shape[0]), length)
        outs = multipass(model, toks, 2, prefix_lens=prefix)
        assert torch.allclose(outs[1].h_top, outs[0].h_top, atol=1e-6), arm


def test_multipass_token_causality():
    """Changing token t leaves every pass's outputs at positions < t
    unchanged: the Jacobi shift preserves causality."""
    for arm in ("df", "df_soft"):
        model = tiny(arm)
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
            ), arm


def test_multipass_loss_shape():
    model = tiny("df")
    toks = tokens()
    prefix = torch.ones((2, toks.shape[0]), dtype=torch.long)
    outs = multipass(model, toks, 3, prefix_lens=prefix)
    total, losses = multipass_loss(model, toks, outs)
    assert len(losses) == 3
    expected = losses[0] + (losses[1] + losses[2]) / 2
    assert torch.allclose(total, expected)


def test_multipass_rejects_feedback_free_arms():
    with pytest.raises(ValueError):
        multipass(
            tiny("vanilla"),
            tokens(),
            2,
            prefix_lens=torch.ones((1, 2), dtype=torch.long),
        )


def test_gradients_reach_the_feedback_machinery():
    model = tiny("df").train()
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
    cfg = model.cfg
    batch, length = toks.shape
    e = model.embed_tokens(toks)
    rows = e[:, :prompt_len]
    payloads = None  # for the soft arm: p_prev source rows, zeros at prompt

    def run(rows, payloads):
        presence = None
        if payloads is not None:
            presence = (
                torch.arange(rows.shape[1])[None, :].expand(batch, -1) >= prompt_len
            )
        return model.forward_column(rows, p_source=payloads, p_presence=presence)

    for t in range(prompt_len, length):
        p_last = run(rows, payloads).payload[:, -1:]  # column t−1's payload
        if cfg.gated_entry:
            rows = torch.cat([rows, model.fuse(p_last, e[:, t : t + 1])], dim=1)
        else:
            rows = torch.cat([rows, e[:, t : t + 1]], dim=1)
            if payloads is None:
                prompt_zeros = torch.zeros(batch, prompt_len, cfg.dim)
                payloads = torch.cat([prompt_zeros, p_last], dim=1)
            else:
                payloads = torch.cat([payloads, p_last], dim=1)
    return run(rows, payloads).h_top[:, prompt_len:]


def test_cached_soft_decode_matches_exact_recurrence():
    """Sequential cached stepping equals the recurrence computed by full
    recomputation — the train/decode parity invariant, per feedback arm."""
    for arm in ("fbt", "df", "df_soft"):
        model = tiny(arm)
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
        assert torch.allclose(stepped, reference, atol=3e-5), arm


def test_cached_standard_decode_matches_full_forward():
    """Standard decoding (no feedback) through the cache equals one full
    parallel forward, on every arm."""
    for arm in ARMS:
        model = tiny(arm)
        toks = tokens(batch=2, length=10)
        full = forward(model, toks)

        cache = KVCache(model.cfg, batch=2, device=toks.device, dtype=torch.float32)
        prefill = model.forward_column(model.embed_tokens(toks[:, :4]), cache=cache)
        pieces = [prefill.h_top]
        for t in range(4, toks.shape[1]):
            pieces.append(model.step(toks[:, t : t + 1], None, cache).h_top)
        assert torch.allclose(torch.cat(pieces, dim=1), full.h_top, atol=3e-5), arm


def test_contraction_diagnostic_runs():
    model = tiny("df")
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

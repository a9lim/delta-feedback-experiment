"""One tiny model per distinct causal, recurrence, and numerical contract."""

import pytest
import torch
import torch.nn.functional as F

from delta_feedback_experiment import analysis
from delta_feedback_experiment.cuda_kernels import MAX_ROUTE_TILES, _route_launch
from delta_feedback_experiment.model import (
    EMBEDDING_LOOKUP_SCALE,
    DeltaModel,
    KVCache,
    ModelConfig,
    combine_column_losses,
    condition_config,
    multipass,
    multipass_loss,
    parse_condition,
)
from delta_feedback_experiment.optim import split_parameters

TINY = {
    "vocab_size": 31,
    "dim": 16,
    "layers": 4,
    "heads": 2,
    "kv_heads": 2,
    "head_dim": 16,
    "expert_intermediate": 8,
    "num_routed_experts": 3,
    "experts_per_token": 2,
    "pkda_heads": 2,
    "pkda_head_dim": 8,
    "pkda_conv_size": 4,
    "max_seq_len": 12,
    "loop_iterations": 2,
}


def tiny(condition="f", **overrides):
    torch.manual_seed(7)
    return DeltaModel(condition_config(condition, **(TINY | overrides))).eval()


def tokens():
    return torch.randint(0, 31, (1, 6), generator=torch.Generator().manual_seed(11))


def test_condition_letters_name_one_feedback_and_one_loop_choice():
    for text, canonical in (("f", "f"), ("lf", "fl"), ("vf", "fv"), ("v", "v")):
        assert parse_condition(text) == canonical
        assert condition_config(text).condition == canonical
    looping = {c: condition_config(c) for c in ("l", "v", "fl", "fv")}
    assert all(cfg.loop for cfg in looping.values())
    assert not condition_config("f").loop
    assert [cfg.loop_token for cfg in looping.values()] == [False, True, False, True]
    # ``l`` and ``v`` fill the same re-entry token slot.
    for text in ("lv", "flv"):
        with pytest.raises(ValueError, match="both l and v"):
            parse_condition(text)
    with pytest.raises(ValueError, match="not both"):
        ModelConfig(loop_blank=True, loop_token=True)
    for text in ("", "x", "ff"):
        with pytest.raises(ValueError):
            parse_condition(text)


def test_multipass_seed_has_standard_strides():
    """Pass 1, the feedback passes, and the looped columns must reach the
    compiled blocks through one stride pattern, or Dynamo compiles pass 1 its
    own copy of every block and the plain prefix of a later pass stops being
    bitwise pass 1."""
    model = tiny("fl")
    outs = multipass(model, tokens(), 2, prefix_lens=torch.tensor([[2]]))
    seed = outs[0][0].sources[0]
    _, length, dim = seed.shape
    assert seed.stride() == (length * dim, dim, 1)
    assert outs[0][1].sources[0].stride() == seed.stride()
    assert outs[1][0].sources[0].stride() == seed.stride()


def test_plain_and_standard_inputs_fuse_scaled_lookups_with_the_blank():
    """A lookup is the tied row times the fixed lookup scale, unnormalized,
    and the classifier reads the raw table."""
    model = tiny()
    toks = tokens()
    with torch.no_grad():
        model.embed_tokens.weight.mul_(
            torch.linspace(0.5, 2, model.cfg.vocab_size)[:, None]
        )
        model.blank_payload.copy_(torch.linspace(-1, 1, model.cfg.dim))
    expected = F.embedding(toks, model.embed_tokens.weight) * EMBEDDING_LOOKUP_SCALE
    torch.testing.assert_close(model.embed_tokens(toks), expected, atol=0, rtol=0)
    # Token magnitudes reach the seed through the fusion, unnormalized.
    seeds = F.linear(
        torch.cat((expected, model.blank_payload.expand_as(expected)), dim=-1),
        model.fuse_proj.weight,
    )
    outs = multipass(model, toks, 2, prefix_lens=torch.tensor([[2]]))
    torch.testing.assert_close(outs[0][0].sources[0], seeds[:, :-1])
    torch.testing.assert_close(outs[1][0].sources[0][:, :2], seeds[:, :2])
    cache = KVCache(model.cfg, batch=1, device="cpu", dtype=torch.float32)
    standard = model.step(toks[:, :1], None, cache)
    torch.testing.assert_close(standard.sources[0], seeds[:, :1])

    # The plain input path contributes directly to the shared embedding table.
    gradient = torch.autograd.grad(
        outs[0][0].h_top.square().mean(), model.embed_tokens.weight
    )[0]
    assert torch.isfinite(gradient).all()
    assert gradient[toks[:, :-1].unique()].abs().sum() > 0
    # The classifier independently reads the raw table, without the lookup scale.
    hidden = outs[0][0].h_top.detach()
    expected_logits = F.linear(model.readout_input(hidden), model.embed_tokens.weight)
    torch.testing.assert_close(model.logits(hidden), expected_logits, atol=0, rtol=0)
    with torch.no_grad():
        model.embed_tokens.weight.mul_(2)
    torch.testing.assert_close(model.embed_tokens(toks), 2 * expected, atol=0, rtol=0)
    torch.testing.assert_close(model.logits(hidden), 2 * expected_logits, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_payload_norm_precedes_output_cast_and_preserves_learned_gain_gradients(dtype):
    model = tiny()
    with torch.no_grad():
        model.payload_norm.weight.uniform_(0.5, 1.5)
    hidden = torch.randn(2, 5, model.cfg.dim).to(dtype).requires_grad_()
    reference = hidden.detach().clone().requires_grad_()
    gain = model.payload_norm.weight.detach().clone().requires_grad_()
    actual = model.payload_norm(hidden)
    # One upcast, as in the module: two casts would round two large partial
    # gradients to BF16 separately before summing them.
    upcast = reference.float()
    normalized = upcast * torch.rsqrt(
        upcast.pow(2).mean(-1, keepdim=True) + model.cfg.norm_eps
    )
    expected = (normalized * gain).to(dtype)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    cotangent = torch.randn_like(actual)
    actual.backward(cotangent)
    expected.backward(cotangent)
    torch.testing.assert_close(hidden.grad, reference.grad)
    torch.testing.assert_close(model.payload_norm.weight.grad, gain.grad)


@torch.no_grad()
def separate_expert_selection(model):
    """Keep causal/cache comparisons away from discrete selection boundaries.

    Expert values, weights, and gradients still depend on the input. Router
    scores, selection, and gradients have independent tests in test_moe.py.
    """
    for bank in model.expert_banks:
        bank.expert_bias[-bank.experts_per_token :] = 2


def test_payload_and_auxiliary_initialization_pair_across_all_conditions():
    conditions = ("f", "l", "fl", "v", "fv")
    states = [tiny(condition, layers=16).state_dict() for condition in conditions]
    common = set.intersection(*(set(state) for state in states))
    assert "payload_router.query" in common and "payload_norm.weight" in common
    assert {name for name in common if name.startswith("embed_tokens.")} == {
        "embed_tokens.weight"
    }
    assert {name for name in common if name.startswith("fuse_")} == {"fuse_proj.weight"}
    assert "blank_payload" in common
    assert ["blank_embedding" in state for state in states] == [
        "l" in condition for condition in conditions
    ]
    # ``v`` re-enters with the token itself, so it is ``f``'s parameter set.
    assert states[0].keys() == states[3].keys() == states[4].keys()
    projection = states[0]["fuse_proj.weight"]
    assert projection.shape == (TINY["dim"], 2 * TINY["dim"])
    torch.testing.assert_close(
        projection.std(), torch.tensor((2 * TINY["dim"]) ** -0.5), atol=0, rtol=0.12
    )
    assert any(name.startswith("mtp.block.attn.") for name in common)
    assert any(name.startswith("mtp.block.mlp.experts.") for name in common)
    for name in common:
        for state in states[1:]:
            torch.testing.assert_close(
                states[0][name], state[name], atol=0, rtol=0, msg=name
            )


def test_routing_matches_normalized_math_and_telescopes():
    model = tiny("fl")
    router = model.blocks[2].mlp_router
    sources = [torch.randn(1, 3, 16) for _ in range(3)]
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "query" in name:
                parameter.normal_()
    routed, weights = router(sources, True)
    values = torch.stack([router.null.expand_as(sources[0]), *sources])
    logits = torch.einsum(
        "hd,nbthd->nbth",
        router.query.view(2, 8),
        router.key_norm(values).view(4, 1, 3, 2, 8),
    )
    expected_weights = logits.softmax(0)
    expected = torch.einsum(
        "nbth,nbthd->bthd",
        expected_weights,
        values.view(4, 1, 3, 2, 8),
    ).reshape_as(routed)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(routed, expected)
    out = model.forward_column(model.embed_tokens(tokens()), want_weights=True)
    torch.testing.assert_close(
        out.h_top, out.sources[0] + torch.stack(out.sources[1:]).sum(0)
    )


def test_routing_sub_tiles_cover_the_head_width():
    """The routing kernels walk a head width in sub-tiles instead of padding it
    to the next power of two, so 384 -- the width at every registered scale --
    is three full 128-lane tiles with no idle lanes."""
    for num_heads in (2, 3, 4, 6):
        block_h, block_k, tiles, _ = _route_launch(num_heads, 384)
        assert block_h >= num_heads
        assert (block_k, tiles) == (128, 3)
        assert block_k * tiles == 384
    for num_heads, head_dim in ((4, 16), (2, 32), (2, 96), (4, 192), (2, 1024)):
        _, block_k, tiles, _ = _route_launch(num_heads, head_dim)
        assert tiles <= MAX_ROUTE_TILES
        assert block_k * (tiles - 1) < head_dim <= block_k * tiles


def test_feedback_and_loop_are_token_causal():
    model = tiny("fl")
    separate_expert_selection(model)
    original = tokens()
    edited = original.clone()
    edited[:, 3] = (edited[:, 3] + 1) % model.cfg.vocab_size
    prefix = torch.ones((1, 1), dtype=torch.long)
    embeddings = []

    def retain_embedding(module, inputs, output):
        output.retain_grad()
        embeddings.append(output)

    hook = model.embed_tokens.register_forward_hook(retain_embedding)
    before = multipass(model, original, 2, prefix_lens=prefix)
    hook.remove()
    after = multipass(model, edited, 2, prefix_lens=prefix)
    assert [len(columns) for columns in before] == [2, 2]
    flat_before = [out for columns in before for out in columns]
    flat_after = [out for columns in after for out in columns]
    for left, right in zip(flat_before, flat_after, strict=True):
        torch.testing.assert_close(
            left.h_top[:, :3], right.h_top[:, :3], atol=5e-5, rtol=1e-5
        )
        assert not torch.allclose(left.h_top[:, 3:], right.h_top[:, 3:])
    # Sparse expert batches can round FP32 GEMMs differently after the edit;
    # the exact zero derivative also checks that no future dependency exists.
    before[-1][-1].h_top[:, :3].square().sum().backward()
    assert torch.count_nonzero(embeddings[0].grad[:, 3:]) == 0
    plain = model.forward_column(model.plain_seed(model.embed_tokens(original[:, :-1]))).h_top
    torch.testing.assert_close(before[0][0].h_top, plain)


def test_blank_payload_seeds_every_plain_position():
    """Every seed is one fusion product: a plain position fuses its
    embedding with the learned blank payload, so feeding the blank as if it
    were feedback reproduces the plain pass exactly."""
    model = tiny()
    with torch.no_grad():
        model.blank_payload.copy_(torch.linspace(-1, 1, model.cfg.dim))
    toks = tokens()
    e = model.embed_tokens(toks[:, :-1])
    seed = model.plain_seed(e)
    blank = model.blank_payload.expand_as(e)
    torch.testing.assert_close(seed, model.fuse(blank, e), atol=0, rtol=0)
    assert torch.equal(analysis.fused_inputs(model, e, blank, 1), seed)
    outs = multipass(model, toks, 2, prefix_lens=torch.ones(1, 1, dtype=torch.long))
    assert torch.equal(outs[0][0].sources[0], seed)
    torch.testing.assert_close(outs[0][0].h_top, model.forward_column(seed).h_top)
    # Position 0 of the feedback pass is plain: the same blank-fused seed.
    assert torch.equal(outs[1][0].sources[0][:, :1], seed[:, :1])
    assert not torch.equal(outs[1][0].sources[0][:, 1:], seed[:, 1:])
    # A plain vector parameter: zero at construction, NAdam-owned, and
    # trained by every plain position.
    fresh = tiny()
    assert torch.equal(fresh.blank_payload, torch.zeros(fresh.cfg.dim))
    assert any(p is fresh.blank_payload for p in split_parameters(fresh)["nadam"])
    multipass_loss(fresh, toks, multipass(fresh, toks, 1)).total.backward()
    gradient = fresh.blank_payload.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_blank_embedding_marks_the_loop_reentry():
    """The token is news only to a position's first column. A later column
    fuses its payload with the learned blank embedding, so its seed carries
    no token term and differs from the feedback seed of the same payload
    even when the next token repeats."""
    model = tiny("fl")
    toks = tokens()
    toks[:, 3] = toks[:, 2]
    e = model.embed_tokens(toks[:, :-1])
    first, second = multipass(model, toks, 1, iterations=2)[0]
    token_term = model.fuse(torch.zeros_like(e), e)
    payload_term = model.fuse(first.payload, torch.zeros_like(e))
    # Zero at construction: the re-entry seed is the payload term alone.
    torch.testing.assert_close(second.sources[0], payload_term)
    torch.testing.assert_close(model.loop_seed(first.payload, e), payload_term)
    # Position 3 repeats position 2's token, so feeding position 2's payload
    # forward fuses it with the very embedding the old re-entry used.
    feedback_seed = model.fuse(first.payload[:, 2], e[:, 3])
    torch.testing.assert_close(feedback_seed, (payload_term + token_term)[:, 2])
    assert not torch.allclose(feedback_seed, second.sources[0][:, 2])
    # Once learned, the blank is a constant marker on every re-entry seed.
    with torch.no_grad():
        model.blank_embedding.copy_(torch.linspace(1, -1, model.cfg.dim))
    marker = model.fuse(torch.zeros(model.cfg.dim), model.blank_embedding)
    torch.testing.assert_close(
        model.loop_seed(first.payload, e), payload_term + marker
    )
    # A plain vector parameter of ``l`` alone: zero at construction,
    # NAdam-owned, trained by every re-entry and untouched by a single column.
    assert tiny("f").blank_embedding is None
    assert tiny("v").blank_embedding is None
    fresh = tiny("l")
    assert torch.equal(fresh.blank_embedding, torch.zeros(fresh.cfg.dim))
    assert any(p is fresh.blank_embedding for p in split_parameters(fresh)["nadam"])
    multipass_loss(fresh, toks, multipass(fresh, toks, 1, iterations=1)).total.backward()
    assert fresh.blank_embedding.grad is None
    multipass_loss(fresh, toks, multipass(fresh, toks, 1, iterations=2)).total.backward()
    gradient = fresh.blank_embedding.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_token_reentry_is_feedback_onto_a_repeated_token():
    """Under ``v`` a later column fuses its payload with the position's own
    token again. That is the feedback seed of a chain that repeats every
    token once, and with the blank payload it is the plain seed, so a later
    column whose payload is removed is the first column."""
    model = tiny("fv")
    with torch.no_grad():
        model.blank_payload.copy_(torch.linspace(-1, 1, model.cfg.dim))
    toks = tokens()
    toks[:, 3] = toks[:, 2]
    e = model.embed_tokens(toks[:, :-1])
    first, second = multipass(model, toks, 1, iterations=2)[0]
    torch.testing.assert_close(
        second.sources[0], model.fuse(first.payload, e), atol=0, rtol=0
    )
    # Position 3 repeats position 2's token, so position 2's re-entry seed is
    # the seed feedback gives position 3 from the same payload.
    feedback_seed = model.fuse(first.payload[:, 2], e[:, 3])
    torch.testing.assert_close(second.sources[0][:, 2], feedback_seed)
    # The null of the loop payload is in distribution: it is pass 1's seed.
    blank = model.blank_payload.expand_as(e)
    null_seed = model.loop_seed(blank, e)
    torch.testing.assert_close(null_seed, model.plain_seed(e), atol=0, rtol=0)
    torch.testing.assert_close(
        model.forward_column(null_seed).h_top, first.h_top, atol=0, rtol=0
    )
    assert not torch.allclose(second.h_top, first.h_top)
    # The re-entry is a second reader of the token rows.
    fresh = tiny("v").train()
    gradients = []
    for iterations in (1, 2):
        fresh.zero_grad()
        outs = multipass(fresh, toks, 1, iterations=iterations)
        multipass_loss(fresh, toks, outs).ntp[0][-1].backward()
        gradients.append(fresh.embed_tokens.weight.grad.clone())
    assert not torch.allclose(*gradients)


@pytest.mark.parametrize("condition", ["fl", "fv"])
def test_looped_columns_reenter_through_the_shared_fusion(condition):
    """A pass's later columns are seeded by the shared fusion of the preceding
    column's jittered payload with the blank embedding under ``l`` or the
    position's own token under ``v``; every column is read out; a column never
    depends on the columns after it; and the first column is the
    single-column model."""
    model = tiny(condition)
    if model.cfg.loop_blank:
        with torch.no_grad():
            model.blank_embedding.copy_(torch.linspace(1, -1, model.cfg.dim))
    toks = tokens()
    e = model.embed_tokens(toks)
    length = toks.shape[1] - 1
    generator = torch.Generator().manual_seed(5)
    jitter = torch.empty(1, *toks.shape, model.cfg.dim).uniform_(
        -0.02, 0.02, generator=generator
    )
    loop_jitter = torch.empty(1, 1, *toks.shape, model.cfg.dim).uniform_(
        -0.02, 0.02, generator=generator
    )
    outs = multipass(
        model, toks, 1, iterations=2, jitter=jitter, loop_jitter=loop_jitter
    )
    assert [len(columns) for columns in outs] == [2]
    first, second = outs[0]
    torch.testing.assert_close(
        first.h_top, model.forward_column(model.plain_seed(e[:, :-1])).h_top
    )
    jittered = first.payload + loop_jitter[0, 0][:, :length]
    token_side = (
        model.blank_embedding.expand_as(jittered)
        if model.cfg.loop_blank
        else e[:, :-1]
    )
    reentry = model.fuse(jittered, token_side)
    torch.testing.assert_close(second.sources[0], reentry)
    torch.testing.assert_close(second.h_top, model.forward_column(reentry).h_top)
    # One jitter draw per column serves both consumers of its payload: the
    # MTP fusion with the next tokens and the loop re-entry.
    torch.testing.assert_close(
        first.fused_input,
        model.fuse(first.payload + loop_jitter[0, 0][:, :length], e[:, 1:]),
    )
    torch.testing.assert_close(
        second.fused_input, model.fuse(second.payload + jitter[0][:, :length], e[:, 1:])
    )
    single = multipass(model, toks, 1, iterations=1, jitter=jitter)
    torch.testing.assert_close(single[0][0].h_top, first.h_top, atol=0, rtol=0)
    # The loss reads every column through FBT's first-plus-mean rule on both
    # axes: passes within each column, then columns.
    result = multipass_loss(model, toks, outs)
    assert [len(columns) for columns in result.ntp] == [2]
    assert [len(columns) for columns in result.mtp] == [2]
    one, two, four, eight = (torch.tensor(value) for value in (1.0, 2.0, 4.0, 8.0))
    assert combine_column_losses([[one]]).item() == 1.0
    assert combine_column_losses([[one, two]]).item() == 3.0  # l alone
    assert combine_column_losses([[one], [four]]).item() == 5.0  # f alone
    assert combine_column_losses([[one, two], [four, eight]]).item() == 15.0
    grid = [[torch.tensor(float(10 * p + c)) for c in (1, 2, 3)] for p in (1, 2, 3)]
    expected = 11 + (12 + 13) / 2 + (21 + 31) / 2 + (22 + 23 + 32 + 33) / 4
    assert combine_column_losses(grid).item() == pytest.approx(expected)
    with pytest.raises(ValueError, match="loop jitter"):
        multipass(model, toks, 1, iterations=2, jitter=jitter)
    with pytest.raises(ValueError, match="loop jitter"):
        multipass(model, toks, 1, iterations=1, jitter=jitter, loop_jitter=loop_jitter)


@pytest.mark.parametrize(
    "condition,added", [("fl", {"blank_embedding"}), ("fv", set())], ids=["fl", "fv"]
)
def test_loop_once_pairs_with_flat_values_and_gradients(condition, added):
    geometry = {"layers": 16}
    flat, loop = tiny("f", **geometry).train(), tiny(condition, **geometry).train()
    # The blank embedding is the only parameter a loop adds, and only under
    # ``l``; a single column never re-enters, so it stays out of the graph.
    assert loop.state_dict().keys() - flat.state_dict().keys() == added
    assert flat.state_dict().keys() <= loop.state_dict().keys()
    losses = []
    for model in (flat, loop):
        outs = multipass(
            model,
            tokens(),
            2,
            iterations=1,
            prefix_lens=torch.ones(1, 1, dtype=torch.long),
        )
        loss = multipass_loss(model, tokens(), outs).total
        losses.append(loss.detach())
        loss.backward()
    torch.testing.assert_close(*losses, atol=0, rtol=0)
    assert loop.blank_embedding is None or loop.blank_embedding.grad is None
    shared = [
        item for item in loop.named_parameters() if item[0] != "blank_embedding"
    ]
    for (name, left), (_, right) in zip(flat.named_parameters(), shared, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0, msg=name)
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0, msg=name)


def test_checkpointing_preserves_feedback_loop_and_auxiliary_gradients():
    models = [tiny("fl").train(), tiny("fl").train()]
    models[1].checkpoint_blocks = 10**6
    losses, counts = [], []
    for model in models:
        outs = multipass(
            model, tokens(), 2, prefix_lens=torch.ones(1, 1, dtype=torch.long)
        )
        result = multipass_loss(model, tokens(), outs, z_coef=0.017)
        logical_counts = result.expert_counts.clone()
        biases = [bank.expert_bias.clone() for bank in model.expert_banks]
        result.total.backward()
        losses.append(result.total.detach())
        counts.append(logical_counts)
        # Checkpoint recomputation is backward work, not another dispatch for
        # the next-step controller. It must neither count twice nor move bias.
        torch.testing.assert_close(result.expert_counts, logical_counts, atol=0, rtol=0)
        for bank, before in zip(model.expert_banks, biases, strict=True):
            torch.testing.assert_close(bank.expert_bias, before, atol=0, rtol=0)
        # Two passes of two columns: every trunk bank and the auxiliary bank
        # dispatch once per column.
        columns = 2 * model.cfg.loop_iterations
        count = columns * (tokens().shape[1] - 1) * 2
        assert logical_counts.sum(-1).tolist() == [count] * (model.cfg.layers + 1)
        for name in (
            "fuse_proj.weight",
            "payload_norm.weight",
            "payload_router.query",
            "embed_tokens.weight",
            "attention_gates.0.weight",
        ):
            gradient = dict(model.named_parameters())[name].grad
            assert gradient is not None and gradient.dtype == torch.float32, name
            assert gradient.abs().sum() > 0 and torch.isfinite(gradient).all(), name
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.mtp.parameters()
        )
    torch.testing.assert_close(*losses, atol=0, rtol=0)
    torch.testing.assert_close(*counts, atol=0, rtol=0)
    for (name, left), (_, right) in zip(
        models[0].named_parameters(), models[1].named_parameters(), strict=True
    ):
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(
                left.grad, right.grad, atol=1e-6, rtol=1e-5, msg=name
            )


@torch.no_grad()
@pytest.mark.parametrize("condition", ["fl", "l", "fv", "v"])
def test_cached_decode_matches_full_recomputation(condition):
    model = tiny(condition)
    separate_expert_selection(model)
    toks = tokens()
    embedded = model.embed_tokens(toks)
    seeds = model.plain_seed(embedded)
    cache = KVCache(model.cfg, batch=1, device="cpu", dtype=torch.float32)
    prefill = model.forward_iterations(seeds[:, :3], embedded[:, :3], cache=cache)[-1]
    assert cache.pos == 3
    payload = prefill.payload[:, -1:] if model.cfg.feedback else None
    reference_rows = seeds[:, :3]
    for position in range(3, toks.shape[1]):
        out = model.step(toks[:, position : position + 1], payload, cache)
        assert cache.pos == position + 1
        new = seeds[:, position : position + 1]
        if model.cfg.feedback:
            reference_payload = model.forward_iterations(
                reference_rows, embedded[:, :position]
            )[-1].payload[:, -1:]
            new = model.fuse(reference_payload, embedded[:, position : position + 1])
        reference_rows = torch.cat([reference_rows, new], dim=1)
        reference = model.forward_iterations(
            reference_rows, embedded[:, : position + 1]
        )[-1]
        # Two looped columns compound full-row versus single-row projection
        # rounding. Bound both aggregate drift and spikes against state scale,
        # including coordinates where the reference happens to cross zero.
        expected = reference.h_top[:, -1:]
        error = out.h_top - expected
        relative_l2 = error.norm() / expected.norm().clamp_min(1e-6)
        relative_peak = error.abs().max() / expected.square().mean().sqrt().clamp_min(
            1e-6
        )
        assert relative_l2 < 1e-4, relative_l2.item()
        assert relative_peak < 2e-4, relative_peak.item()
        payload = out.payload[:, -1:] if model.cfg.feedback else None
    # Every layer owns one track per column: one GQA layer and three PKDA
    # layers at two columns.
    assert cache.k.shape[0] == 2
    assert len(cache.pkda_states) == 6


def test_partial_checkpointing_counts_blocks_and_preserves_gradients():
    """The recomputation budget applies to the first PKDA and auxiliary block
    invocations of a logical forward, in execution order, and changes nothing
    numerically."""
    raw, partial = tiny("fl").train(), tiny("fl").train()
    partial.checkpoint_blocks = 3
    results = []
    for model in (raw, partial):
        outs = multipass(
            model, tokens(), 2, prefix_lens=torch.ones(1, 1, dtype=torch.long)
        )
        result = multipass_loss(model, tokens(), outs, z_coef=0.017)
        result.total.backward()
        results.append(result.total.detach())
    torch.testing.assert_close(*results, atol=0, rtol=0)
    assert partial.checkpoint_blocks - partial._checkpoint_left == 3
    assert raw._checkpoint_left == 0
    for (name, left), (_, right) in zip(
        raw.named_parameters(), partial.named_parameters(), strict=True
    ):
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0, msg=name)
    # A budget beyond the eligible invocations recomputes every one of them.
    everything = tiny("fl").train()
    everything.checkpoint_blocks = 10**6
    outs = multipass(
        everything, tokens(), 2, prefix_lens=torch.ones(1, 1, dtype=torch.long)
    )
    multipass_loss(everything, tokens(), outs)
    columns = 2 * everything.cfg.loop_iterations
    eligible = columns * (everything.cfg.pkda_layers + 1)
    assert everything.checkpoint_blocks - everything._checkpoint_left == eligible


def test_unit_scale_entry_keeps_reentry_contracting():
    """Both fusion inputs are unit RMS at initialization, so the seed enters
    the residual stream at unit scale and a re-entry hop does not amplify
    gradients: the second column's loss reaches the first column's top with
    weight of order one relative to the first column's own loss (0.8 at
    screen geometry, around one on this small one). With the seed at
    embedding scale the early pre-norms divided by a tiny residual and each
    hop multiplied the gradient about tenfold."""
    model = tiny(
        "fl", dim=128, head_dim=64, pkda_head_dim=64, layers=8
    ).train()
    toks = torch.randint(0, 31, (1, 12), generator=torch.Generator().manual_seed(12))

    def rms(x):
        return x.float().square().mean().sqrt().item()

    assert 0.5 < rms(model.embed_tokens(toks)) < 2
    outs = multipass(model, toks, 1, iterations=2)
    loss = multipass_loss(model, toks, outs)
    first, second = outs[0]
    assert 0.5 < rms(first.payload) < 2
    assert 0.3 < rms(second.sources[0]) < 3
    own = torch.autograd.grad(loss.ntp[0][0], first.h_top, retain_graph=True)[0]
    hop = torch.autograd.grad(loss.ntp[0][1], first.h_top)[0]
    assert hop.norm() < 2 * own.norm()

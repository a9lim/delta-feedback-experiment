"""CUDA numerical and production-geometry qualification for condition ``e``.

The sparse reference is deliberately ordinary PyTorch over selected rows. Its
operands have the CUDA kernel's BF16 rounding, so these checks measure dispatch,
derivatives, and accumulation rather than a change in training precision.
"""

from __future__ import annotations

import copy
import gc
import math
import time

import torch
import torch.nn.functional as F
from torch import Tensor

from .moe import MixtureOfExperts


def _relative(actual: Tensor, expected: Tensor, *, floor: float = 1e-12) -> float:
    actual, expected = actual.detach().float(), expected.detach().float()
    return float((actual - expected).norm() / expected.norm().clamp_min(floor))


def _close(label: str, actual: Tensor, expected: Tensor, bound: float) -> float:
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"nonfinite MoE {label}")
    relative = _relative(actual, expected)
    if relative > bound:
        raise AssertionError(f"MoE {label} relative L2 drift {relative:.5g} > {bound}")
    return relative


def _reference(
    module: MixtureOfExperts, x: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Literal dropless top-three execution with the CUDA operand precision."""
    flat = x.reshape(-1, module.dim)
    with torch.autocast(device_type=flat.device.type, enabled=False):
        logits = F.linear(flat.float(), module.router.weight.float())
    top, selected = logits.topk(3, dim=-1)
    selected_weights = top.softmax(dim=-1)
    probabilities = logits.softmax(dim=-1)
    fractions = F.one_hot(selected, 15).sum(dim=1).float().mean(dim=0) / 3
    auxiliary = 15 * (probabilities.mean(dim=0) * fractions).sum()

    # Scatter by assignment slot before the FP32 mixture reduction: summing
    # separately rounded weighted experts would introduce a different forward.
    ordered = flat.new_zeros((3 * flat.shape[0], module.dim))
    for index, expert in enumerate(module.experts):
        token, slot = torch.where(selected == index)
        inputs = flat.index_select(0, token)
        gate, up = F.linear(
            inputs, expert.gate_up_proj.weight.to(flat.dtype)
        ).chunk(2, dim=-1)
        activated = (F.silu(gate.float()) * up.float()).to(flat.dtype)
        values = F.linear(activated, expert.down_proj.weight.to(flat.dtype))
        ordered = ordered.index_copy(0, 3 * token + slot, values)
    mixed = (
        ordered.view(-1, 3, module.dim).float() * selected_weights.unsqueeze(-1)
    ).sum(dim=1).to(flat.dtype)
    gate, up = F.linear(
        flat, module.shared.gate_up_proj.weight.to(flat.dtype)
    ).chunk(2, dim=-1)
    shared = F.linear(
        F.silu(gate) * up, module.shared.down_proj.weight.to(flat.dtype)
    )
    output = ((shared + 3 * mixed) / 2).reshape_as(x)
    weights = torch.zeros_like(probabilities).scatter(1, selected, selected_weights)
    return output, auxiliary, weights.reshape(*x.shape[:-1], 15)


def _kernel_parity(
    dim: int, intermediate: int, tokens: int, *, skewed: bool = False
) -> dict[str, float]:
    """Compare output, router/input/dW, and repeated persistent FP32 sinks."""
    torch.manual_seed(713 + tokens)
    actual = MixtureOfExperts(dim, intermediate).cuda()
    x = torch.randn(1, tokens, dim, device="cuda", dtype=torch.bfloat16)
    if skewed:
        # Every token chooses experts 0, 1, and 2, leaving twelve empty experts.
        # Each selected expert must accommodate the complete row without drops.
        x.fill_(0.25)
        with torch.no_grad():
            actual.router.weight.zero_()
            actual.router.weight[0].fill_(0.75 / dim)
            actual.router.weight[1].fill_(0.50 / dim)
            actual.router.weight[2].fill_(0.25 / dim)
            actual.router.weight[3:].fill_(-0.25 / dim)
    reference = copy.deepcopy(actual)
    actual_x = x.detach().clone().requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    cotangent = torch.randn_like(x)
    weight_cotangent = torch.randn(1, tokens, 15, device="cuda")

    def objective(result):
        output, auxiliary, weights = result
        return (
            (output.float() * cotangent.float()).sum() / tokens
            + 0.01 * auxiliary
            + 0.01 * (weights * weight_cotangent).sum() / tokens
        )

    actual_result = actual(actual_x, want_weights=True)
    reference_result = _reference(reference, reference_x)
    objective(actual_result).backward()
    objective(reference_result).backward()
    errors = {
        "output": _close("output", actual_result[0], reference_result[0], 0.015),
        "auxiliary": _close("balance loss", actual_result[1], reference_result[1], 1e-6),
        "weights": _close("routing weights", actual_result[2], reference_result[2], 1e-6),
        "input": _close("input gradient", actual_x.grad, reference_x.grad, 0.035),
    }
    if skewed:
        active = actual_result[2].ne(0)
        if not active[..., :3].all() or active[..., 3:].any():
            raise AssertionError("MoE skew fixture did not select only the first three")
    expected_gradients = {}
    actual_gradients = {}
    for (name, parameter), (ref_name, ref_parameter) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == ref_name
        if parameter.grad is None or ref_parameter.grad is None:
            raise AssertionError(f"MoE ordinary backward lost {name}")
        expected_gradients[name] = ref_parameter.grad.clone()
        actual_gradients[name] = parameter.grad.clone()
        errors[name] = _close(name, parameter.grad, ref_parameter.grad, 0.04)

    del actual_result, reference_result
    actual.zero_grad(set_to_none=True)
    actual_x.grad = None
    sinks = {parameter: torch.zeros_like(parameter) for parameter in actual.parameters()}
    bound, refresh = actual.bind_gradient_sinks(sinks)
    addresses = {parameter: sinks[parameter].data_ptr() for parameter in bound}
    shadow_addresses = [shadow.data_ptr() for shadow, _ in refresh]
    if len(bound) != 32:
        raise AssertionError("MoE must bind both matrices of all sixteen experts")
    for _ in range(2):
        objective(actual(actual_x, want_weights=True)).backward()
    _close(
        "accumulated input gradient", actual_x.grad, 2 * reference_x.grad, 0.035
    )
    _close(
        "accumulated router gradient",
        actual.router.weight.grad,
        2 * reference.router.weight.grad,
        0.04,
    )
    for name, parameter in actual.named_parameters():
        if parameter not in bound:
            continue
        if parameter.grad is not None or sinks[parameter].data_ptr() != addresses[parameter]:
            raise AssertionError(f"MoE persistent sink lost its storage contract: {name}")
        if sinks[parameter].dtype != torch.float32:
            raise AssertionError(f"MoE sink is not FP32: {name}")
        errors[f"sink.{name}"] = _close(
            f"accumulated sink {name}", sinks[parameter], 2 * expected_gradients[name], 0.04
        )
        # The second call must add into the existing FP32 sink. This compares
        # against the CUDA ordinary backward itself, independent of reference
        # GEMM rounding (the shared expert's sink avoids BF16 dW rounding too).
        _close(f"two writes {name}", sinks[parameter], 2 * actual_gradients[name], 0.02)
    torch._foreach_copy_([s for s, _ in refresh], [p for _, p in refresh])
    if [shadow.data_ptr() for shadow, _ in refresh] != shadow_addresses:
        raise AssertionError("MoE shadow refresh changed storage")
    if any("shadow" in name or "sink" in name for name in actual.state_dict()):
        raise AssertionError("MoE derived CUDA buffers escaped into checkpoint state")
    return errors


def _frozen_kernel_parity(*, partial: bool = False) -> None:
    """Frozen expert matrices retain dX and omit only their own dW."""
    torch.manual_seed(83)
    actual = MixtureOfExperts(48, 140).cuda()
    actual.experts.requires_grad_(False)
    if partial:
        actual.experts[0].down_proj.weight.requires_grad_(True)
    reference = copy.deepcopy(actual)
    x = torch.randn(1, 19, 48, device="cuda", dtype=torch.bfloat16).requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    output, auxiliary, _ = actual(x)
    ref_output, ref_auxiliary, _ = _reference(reference, ref_x)
    (output.float().square().mean() + 0.01 * auxiliary).backward()
    (ref_output.float().square().mean() + 0.01 * ref_auxiliary).backward()
    _close("frozen bank input gradient", x.grad, ref_x.grad, 0.035)
    for (name, parameter), (_, ref_parameter) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        if parameter.requires_grad:
            if parameter.grad is None or ref_parameter.grad is None:
                raise AssertionError(f"partial frozen bank lost gradient: {name}")
            _close(f"partial frozen bank {name}", parameter.grad, ref_parameter.grad, 0.04)
        elif parameter.grad is not None:
            raise AssertionError(f"frozen expert unexpectedly received dW: {name}")


def _compiled_replay() -> None:
    """Compile the sparse route and replay changing expert assignments."""
    from torch._dynamo.utils import counters

    from .train import _capture_without_gc

    torch.manual_seed(29)
    module = MixtureOfExperts(48, 140).cuda()
    sinks = {parameter: torch.zeros_like(parameter) for parameter in module.parameters()}
    bound, refresh = module.bind_gradient_sinks(sinks)
    for parameter in module.parameters():
        parameter.grad = sinks[parameter]
    x = torch.randn(1, 37, 48, device="cuda", dtype=torch.bfloat16).requires_grad_()
    x.grad = torch.zeros_like(x)
    compiled = torch.compile(module, fullgraph=True, mode="max-autotune-no-cudagraphs")

    def clear():
        for gradient in sinks.values():
            gradient.zero_()
        x.grad.zero_()

    def body():
        output, auxiliary, _ = compiled(x)
        loss = output.float().square().mean() + 0.01 * auxiliary
        loss.backward()
        return loss

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            clear()
            body()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    clear()
    graph = torch.cuda.CUDAGraph()
    with _capture_without_gc(graph, torch.cuda.graph_pool_handle()):
        captured_loss = body()
    pointer = {parameter: gradient.data_ptr() for parameter, gradient in sinks.items()}
    unique_graphs = counters["stats"]["unique_graphs"]
    for seed in (41, 43):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.no_grad():
            x.copy_(torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=generator))
        clear()
        graph.replay()
        torch.cuda.synchronize()
        if not math.isfinite(captured_loss.item()):
            raise AssertionError("MoE captured sparse loss is nonfinite")
        if not torch.isfinite(x.grad).all() or not x.grad.any():
            raise AssertionError("MoE captured sparse input gradient is missing")
        for parameter in bound:
            if sinks[parameter].data_ptr() != pointer[parameter]:
                raise AssertionError("MoE captured sink address changed")
            if not torch.isfinite(sinks[parameter]).all():
                raise AssertionError("MoE captured expert gradient is nonfinite")
        if not module.router.weight.grad.any():
            raise AssertionError("MoE captured router received no gradient")
        # Compare the replay to an eager forward at precisely its new inputs;
        # this catches a graph whose dispatch was frozen during preparation.
        with torch.no_grad():
            output, auxiliary, _ = module(x)
            expected = output.float().square().mean() + 0.01 * auxiliary
        _close("changing-dispatch graph loss", captured_loss, expected, 0.015)
    if counters["stats"]["unique_graphs"] != unique_graphs:
        raise AssertionError("MoE graph replay compiled a new graph")
    if len(refresh) != 32:
        raise AssertionError("MoE shadow surface omits an expert projection")


def _decode_parity() -> dict[str, float]:
    """Sparse BF16 inference stays token-causal through cached recurrence."""
    from .model import DeltaModel, KVCache, condition_config

    errors = {}
    for condition, layers in (("e", 4), ("aerfl", 12)):
        torch.manual_seed(101)
        config = condition_config(
            condition,
            vocab_size=257,
            dim=128,
            layers=layers,
            heads=4,
            kv_heads=2,
            head_dim=32,
            intermediate=256,
            pkda_heads=2,
            pkda_head_dim=128,
            max_seq_len=32,
            loop_iterations=2,
            loop_max_iterations=2,
        )
        model = DeltaModel(config).cuda().eval()
        tokens = torch.randint(0, config.vocab_size, (1, 13), device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            embeddings = model.embed_tokens(tokens)
            whole = model.forward_column(embeddings, want_weights=True)
            changed = tokens.clone()
            changed[:, 7:] = (changed[:, 7:] + 1) % config.vocab_size
            future = model.forward_column(model.embed_tokens(changed), want_weights=True)
            errors[f"{condition}.causal"] = _close(
                f"{condition} causal prefix", future.h_top[:, :7], whole.h_top[:, :7], 1e-6
            )
            for site, weights in whole.expert_weights.items():
                if not torch.equal(weights[:, :7], future.expert_weights[site][:, :7]):
                    raise AssertionError(f"future tokens altered expert choice at {site}")

            cache = KVCache(config, 1, "cuda", torch.bfloat16)
            prefill = model.forward_column(embeddings[:, :5], cache=cache)
            pieces = [prefill.h_top]
            for column in range(5, tokens.shape[1]):
                pieces.append(model.step(tokens[:, column : column + 1], None, cache).h_top)
            errors[f"{condition}.cache"] = _close(
                f"{condition} cached decode", torch.cat(pieces, dim=1), whole.h_top, 0.08
            )
            if config.feedback:
                cache = KVCache(config, 1, "cuda", torch.bfloat16)
                prefill = model.forward_column(embeddings[:, :5], cache=cache)
                payload = prefill.payload[:, -1:]
                pieces = []
                reference_rows = embeddings[:, :5]
                for column in range(5, tokens.shape[1]):
                    result = model.step(tokens[:, column : column + 1], payload, cache)
                    pieces.append(result.h_top)
                    payload = result.payload
                    previous = model.forward_column(reference_rows).payload[:, -1:]
                    reference_rows = torch.cat(
                        [reference_rows, model.fuse(previous, embeddings[:, column : column + 1])],
                        dim=1,
                    )
                reference = model.forward_column(reference_rows).h_top[:, 5:]
                errors[f"{condition}.feedback"] = _close(
                    f"{condition} cached feedback", torch.cat(pieces, dim=1), reference, 0.10
                )
    return errors


@torch.no_grad()
def _decode_diagnostics() -> dict:
    """Separate hard-router sensitivity from cache semantics on the failed seed.

    The fixed-route control intervenes only on top-k indices. It still uses
    the CUDA sparse expert kernels and current-input mixing probabilities.
    This is a diagnostic, not a different deployed inference path.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    from .model import DeltaModel, KVCache, condition_config

    class _FixedRoutes(TorchDispatchMode):
        def __init__(self, selections):
            super().__init__()
            self.selections = selections
            self.column = 0
            self.calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.topk.default and args[0].shape[-1] == 15:
                logits = args[0]
                chosen = self.selections[self.calls % len(self.selections)]
                selected = chosen[0, self.column : self.column + logits.shape[0]]
                self.calls += 1
                return logits.gather(-1, selected), selected
            return func(*args, **(kwargs or {}))

    def cached(model, tokens, dtype, fixed=None):
        cache = KVCache(model.cfg, 1, tokens.device, dtype)
        if fixed is not None:
            fixed.column = 0
        outputs = [
            model.forward_column(
                model.embed_tokens(tokens[:, :5]), cache=cache, want_weights=True
            )
        ]
        for column in range(5, tokens.shape[1]):
            if fixed is not None:
                fixed.column = column
            outputs.append(
                model.step(tokens[:, column : column + 1], None, cache, want_weights=True)
            )
        return (
            torch.cat([out.h_top for out in outputs], dim=1),
            {
                site: torch.cat([out.expert_weights[site] for out in outputs], dim=1)
                for site in outputs[0].expert_weights
            },
        )

    results = {}
    token_fixture = None
    for condition, device, bf16 in (
        ("aerfl", "cuda", True),
        ("arfl", "cuda", True),
        ("aerfl", "cuda", False),
        ("aerfl", "cpu", False),
    ):
        torch.manual_seed(101)
        config = condition_config(
            condition,
            vocab_size=257,
            dim=128,
            layers=12,
            heads=4,
            kv_heads=2,
            head_dim=32,
            intermediate=256,
            pkda_heads=2,
            pkda_head_dim=128,
            max_seq_len=32,
            loop_iterations=2,
            loop_max_iterations=2,
        )
        model = DeltaModel(config).to(device).eval()
        if token_fixture is None:
            token_fixture = torch.randint(0, 257, (1, 13), device="cuda").cpu()
        tokens = token_fixture.to(device)
        recorded = []

        def record(module, inputs, captured=recorded):
            captured.append(inputs[0].detach())

        handles = [
            block.mlp.register_forward_pre_hook(record)
            for block in model.blocks
            if isinstance(block.mlp, MixtureOfExperts)
        ]
        dtype = torch.bfloat16 if bf16 else torch.float32
        label = f"{condition}.{device}.{'bf16' if bf16 else 'fp32'}"
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16):
            whole = model.forward_column(model.embed_tokens(tokens), want_weights=True)
            full_inputs = tuple(recorded)
            recorded.clear()
            incremental, incremental_weights = cached(model, tokens, dtype)
            result = {"relative": _relative(incremental, whole.h_top)}
            sites = list(whole.expert_weights)
            swaps = []
            for index, site in enumerate(sites):
                expected = whole.expert_weights[site]
                actual = incremental_weights[site]
                changed = expected.ne(0).ne(actual.ne(0)).any(dim=-1)
                if not changed.any():
                    continue
                position = int(torch.where(changed[0])[0][0])
                cached_inputs = torch.cat(recorded[index::len(sites)], dim=1)
                full_input = full_inputs[index][0, position]
                cached_input = cached_inputs[0, position]
                # The executed order supplies each module across core repeats.
                execution = list(range(4)) + list(range(4, 8)) * 2 + list(range(8, 12))
                router = model.blocks[execution[index]].mlp.router
                with torch.autocast(device_type=device, enabled=False):
                    full_logits = F.linear(full_input.float(), router.weight.float())
                    cached_logits = F.linear(cached_input.float(), router.weight.float())
                full_top = full_logits.topk(4).values
                cached_top = cached_logits.topk(4).values
                swaps.append(
                    {
                        "site": site,
                        "swapped_tokens": int(changed.sum()),
                        "first_position": position,
                        "full_experts": expected[0, position].nonzero().flatten().tolist(),
                        "cached_experts": actual[0, position].nonzero().flatten().tolist(),
                        "full_margin": float(full_top[2] - full_top[3]),
                        "cached_margin": float(cached_top[2] - cached_top[3]),
                        "max_logit_delta": float((full_logits - cached_logits).abs().max()),
                        "input_relative": _relative(cached_input, full_input),
                    }
                )
            result["route_disagreements"] = swaps
            print(f"MoE decode diagnostic {label}: {result}", flush=True)
            if sites and bf16:
                fixed = _FixedRoutes([whole.expert_weights[site].topk(3).indices for site in sites])
                originals = [
                    (block.mlp, block.mlp.forward)
                    for block in model.blocks
                    if isinstance(block.mlp, MixtureOfExperts)
                ]
                # Limit dispatch interception to the MoE itself. Attention
                # retains its compiled CUDA path and never sees this mode.
                for module, original in originals:
                    def routed_forward(*args, forward=original, route_mode=fixed, **kwargs):
                        with route_mode:
                            return forward(*args, **kwargs)

                    module.forward = routed_forward
                try:
                    pinned, _ = cached(model, tokens, dtype, fixed=fixed)
                finally:
                    for module, original in originals:
                        module.forward = original
                result["fixed_route_relative"] = _relative(pinned, whole.h_top)
                print(
                    f"MoE decode diagnostic {label}: "
                    f"fixed_route_relative={result['fixed_route_relative']:.6g}",
                    flush=True,
                )
        for handle in handles:
            handle.remove()
        results[label] = result
    return results


def _screen_capture(
    condition: str, modes: tuple[tuple[int, int], ...], *, all_modes: bool = False
) -> str:
    """Capture the selected or complete family, then replay representative modes."""
    from torch._dynamo.utils import counters

    from .model import DeltaModel, condition_config
    from .optim import build_optimizers
    from .train import (
        CudaEvalRunner,
        CudaGraphTrainer,
        GraphSpec,
        automatic_checkpoint,
        build_schedule,
        clip_gradients,
        parse_run_args,
    )

    args = parse_run_args(["moe-cuda-probe", "--condition", condition])
    cfg = condition_config(
        condition,
        vocab_size=args.vocab_size,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        intermediate=args.intermediate,
        max_seq_len=args.seq_len + 1,
        loop_iterations=args.loop_iterations,
        loop_max_iterations=args.loop_max_iterations,
    )
    torch.manual_seed(args.seed)
    model = DeltaModel(cfg).cuda().train()
    optimizers = build_optimizers(model)

    class _SelectedModes(CudaGraphTrainer):
        def _reachable_specs(self, schedule):
            return [
                GraphSpec(
                    passes,
                    automatic_checkpoint(model, passes, iterations, args, self.device),
                    iterations,
                )
                for passes, iterations in modes
            ]

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer = CudaGraphTrainer if all_modes else _SelectedModes
    runner = trainer(model, optimizers, args, build_schedule(args))
    eval_runner = CudaEvalRunner(model, args, runner.pool)
    if all_modes and len(runner.states) != 24:
        raise AssertionError("MoE loop qualification did not retain all 24 training modes")
    def replay_eval():
        for state in eval_runner.states.values():
            state.rows.copy_(torch.randint_like(state.rows, high=args.vocab_size))
            state.val_sum.zero_()
            state.fused_sum.zero_()
            state.graph.replay()
            torch.cuda.synchronize()
            if not math.isfinite(state.val_sum.item()) or not math.isfinite(state.fused_sum.item()):
                raise AssertionError("MoE resident evaluation graph is nonfinite")

    replay_eval()
    prepared = time.monotonic() - started
    addresses = {parameter: sink.data_ptr() for parameter, sink in runner.grad_buffers.items()}
    shadows = [
        buffer
        for block in model.blocks
        for buffer in (block.mlp._gate_up_shadow, block.mlp._down_shadow)
    ]
    shadow_addresses = [buffer.data_ptr() for buffer in shadows]
    experts = {
        parameter
        for block in model.blocks
        for expert in block.mlp.experts
        for parameter in expert.parameters()
    }
    compiled = counters["stats"]["unique_graphs"]
    records = []
    for spec, state in runner.states.items():
        if not experts <= state.active:
            raise AssertionError("CUDA preparation pruned experts absent from warm-up routing")
        if (spec.n_passes, spec.iterations) not in modes:
            continue
        old_router = model.blocks[0].mlp.router.weight.detach().clone()
        old_shared = model.blocks[0].mlp.shared.down_proj.weight.detach().clone()
        old_routed = model.blocks[0].mlp.experts[0].down_proj.weight.detach().clone()
        generator = torch.Generator().manual_seed(59 + spec.n_passes + spec.iterations)
        runner.zero_grad()
        runner.begin(spec, args.zloss)
        started = time.monotonic()
        # Two distinct rows test accumulation as well as changing dispatch.
        for index in range(2):
            rows = torch.randint(
                0, args.vocab_size, (1, args.seq_len + 1), generator=generator
            )
            runner.replay(state, rows, 17 + index, index)
        runner.prepare_optimizer(state)
        torch.cuda.synchronize()
        elapsed = (time.monotonic() - started) / 2
        gradient_norm = clip_gradients(model.parameters())
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise AssertionError(f"MoE screen {condition} {spec} invalid gradient norm")
        if not math.isfinite(state.loss_sum.item()):
            raise AssertionError(f"MoE screen {condition} {spec} nonfinite loss")
        balance = state.expert_balance_sum.item() * runner.micros / 2
        if not math.isfinite(balance) or balance <= 0:
            raise AssertionError(f"MoE screen {condition} {spec} lost balance objective")
        for parameter in experts:
            gradient = parameter.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise AssertionError("MoE screen expert gradient is absent or nonfinite")
            if gradient.data_ptr() != addresses[parameter]:
                raise AssertionError("MoE screen expert gradient changed persistent storage")
        for optimizer in optimizers:
            optimizer.step()
        model.refresh_shadows()
        if torch.equal(model.blocks[0].mlp.router.weight, old_router):
            raise AssertionError("MoE screen router did not update")
        if torch.equal(model.blocks[0].mlp.shared.down_proj.weight, old_shared):
            raise AssertionError("MoE screen shared expert did not update")
        if torch.equal(model.blocks[0].mlp.experts[0].down_proj.weight, old_routed):
            raise AssertionError("MoE screen routed expert did not update")
        if [buffer.data_ptr() for buffer in shadows] != shadow_addresses:
            raise AssertionError("MoE screen expert shadow addresses changed after update")
        # Replay once using refreshed expert values, without retracing.
        runner.zero_grad()
        runner.begin(spec, 0.0)
        runner.replay(state, rows, 20, 0)
        runner.prepare_optimizer(state)
        torch.cuda.synchronize()
        if not math.isfinite(state.loss_sum.item()):
            raise AssertionError("MoE screen post-update replay is nonfinite")
        records.append(
            f"k={spec.n_passes}/r={spec.iterations}/ckpt={int(spec.checkpoint)}:"
            f"{1000 * elapsed:.1f}ms/balance={balance:.3f}"
        )
    # The live schedule chooses modes out of capture order. Return to the
    # first/raw graph after the largest mode, then evaluate with its gradients
    # and the whole training family still resident in the shared pool.
    first = next(spec for spec in runner.states if (spec.n_passes, spec.iterations) in modes)
    state = runner.begin(first)
    runner.zero_grad()
    runner.replay(state, rows, 23, 0)
    runner.prepare_optimizer(state)
    torch.cuda.synchronize()
    if not math.isfinite(state.loss_sum.item()):
        raise AssertionError("MoE raw replay after the cap graph is nonfinite")
    gradient_norm = clip_gradients(model.parameters())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise AssertionError("MoE raw replay after the cap graph lost gradients")
    replay_eval()
    if counters["stats"]["unique_graphs"] != compiled:
        raise AssertionError("MoE screen replay escaped CUDA preparation")
    # CUDA graph executable/driver storage is not all visible to the PyTorch
    # allocator. This device-wide snapshot also includes the display/context.
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return (
        f"{condition}: prepare={prepared:.1f}s "
        f"allocated={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB "
        f"reserved={torch.cuda.max_memory_reserved() / 2**30:.2f}GiB "
        f"device_resident={(total_bytes - free_bytes) / 2**30:.2f}GiB "
        f"resident_graphs={len(runner.states)}+{len(eval_runner.states)}eval "
        + " ".join(records)
    )


def cuda_moe_gate() -> None:
    """Qualify sparse derivatives, replay, decode, screen, and recurrence cap."""
    if not torch.cuda.is_available():
        raise RuntimeError("MoE CUDA gate requires a CUDA device")
    torch.set_float32_matmul_precision("high")

    def release():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    release()
    errors = []
    for dim, intermediate, tokens, skewed in (
        (48, 140, 37, False),
        (48, 140, 67, True),
        (128, 256, 1, True),
        (768, 3328, 67, False),
    ):
        errors.append(max(_kernel_parity(dim, intermediate, tokens, skewed=skewed).values()))
    _frozen_kernel_parity()
    _frozen_kernel_parity(partial=True)
    _compiled_replay()
    decode = _decode_parity()
    release()
    print(
        "moe CUDA numerics | "
        f"max_relative={max(errors):.5f} | "
        + " | ".join(f"{name}={value:.5f}" for name, value in decode.items()),
        flush=True,
    )
    # Each scope returns before collection, so separate full models never
    # coexist. The loop retains its complete 24-mode family plus evaluation;
    # replay covers raw/checkpoint boundaries and the worst arithmetic mode.
    for condition, modes in (
        ("aerf", ((1, 1), (2, 1), (3, 1))),
        ("aerfl", ((1, 1), (1, 3), (1, 4), (2, 4), (3, 8))),
    ):
        record = _screen_capture(condition, modes, all_modes=condition == "aerfl")
        print(f"moe CUDA screen | {record}", flush=True)
        release()

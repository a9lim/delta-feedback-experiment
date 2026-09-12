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
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Literal dropless top-three execution with the CUDA operand precision."""
    flat = x.reshape(-1, module.dim)
    with torch.autocast(device_type=flat.device.type, enabled=False):
        logits = F.linear(flat.float(), module.router.weight.float())
    scores = logits.sigmoid()
    selected = (scores + module.expert_bias).topk(3, dim=-1).indices
    selected_scores = scores.gather(1, selected)
    selected_weights = selected_scores / selected_scores.sum(dim=-1, keepdim=True)
    probabilities = scores / scores.sum(dim=-1, keepdim=True)
    selection = F.one_hot(selected, 15).sum(dim=1)
    length = x.shape[-2]
    auxiliary_selection = F.one_hot(scores.topk(3, dim=-1).indices, 15).sum(dim=1)
    fractions = auxiliary_selection.reshape(-1, length, 15).float().mean(dim=1) / 3
    sequence_probabilities = probabilities.reshape(-1, length, 15).mean(dim=1)
    auxiliary = 15 * (sequence_probabilities * fractions).sum(dim=-1).mean()
    counts = selection.sum(dim=0)

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
    return output, auxiliary, weights.reshape(*x.shape[:-1], 15), counts


def _kernel_parity(
    dim: int, intermediate: int, tokens: int, *, skewed: bool = False,
    batch: int = 1, biased: bool = False,
) -> dict[str, float]:
    """Compare output, router/input/dW, and repeated persistent FP32 sinks."""
    torch.manual_seed(713 + tokens)
    actual = MixtureOfExperts(dim, intermediate).cuda()
    x = torch.randn(batch, tokens, dim, device="cuda", dtype=torch.bfloat16)
    if biased:
        actual.expert_bias.copy_(torch.linspace(-0.25, 0.25, 15, device="cuda"))
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
    weight_cotangent = torch.randn(batch, tokens, 15, device="cuda")
    bias_before = actual.expert_bias.clone()

    def objective(result):
        output, auxiliary, weights, _ = result
        return (
            (output.float() * cotangent.float()).sum() / (batch * tokens)
            + 1e-4 * auxiliary
            + 0.01 * (weights * weight_cotangent).sum() / (batch * tokens)
        )

    actual_result = actual(actual_x, want_weights=True)
    reference_result = _reference(reference, reference_x)
    objective(actual_result).backward()
    objective(reference_result).backward()
    if actual_result[3].dtype != torch.int64 or actual_result[3].requires_grad:
        raise AssertionError("MoE usage counts must be detached int64")
    if not torch.equal(actual_result[3], reference_result[3]):
        raise AssertionError("MoE selected expert counts drifted from the reference")
    if actual_result[3].sum().item() != 3 * batch * tokens:
        raise AssertionError("MoE count total lost or duplicated token assignments")
    if not torch.equal(actual.expert_bias, bias_before):
        raise AssertionError("MoE forward/backward mutated the selection bias")
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
    if not torch.equal(actual.expert_bias, bias_before):
        raise AssertionError("MoE accumulated backward mutated the selection bias")
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
    output, auxiliary, _, _ = actual(x)
    ref_output, ref_auxiliary, _, _ = _reference(reference, ref_x)
    (output.float().square().mean() + 1e-4 * auxiliary).backward()
    (ref_output.float().square().mean() + 1e-4 * ref_auxiliary).backward()
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


def _bias_step_parity() -> None:
    """Step-only bias updates change selections, retain unbiased gates, and resume."""
    torch.manual_seed(47)
    module = MixtureOfExperts(48, 140).cuda().train()
    with torch.no_grad():
        module.router.weight.zero_()
    x = torch.randn(2, 17, 48, device="cuda", dtype=torch.bfloat16)
    before = module.expert_bias.clone()
    pointer = module.expert_bias.data_ptr()
    if before.any() or before.dtype != torch.float32:
        raise AssertionError("MoE selection bias must start as a zero FP32 buffer")
    output, auxiliary, weights, counts = module(x, want_weights=True)
    (output.float().square().mean() + 1e-4 * auxiliary).backward()
    if not torch.equal(before, module.expert_bias):
        raise AssertionError("MoE forward or backward updated routing bias")
    if counts.sum().item() != 3 * 2 * 17:
        raise AssertionError("MoE batch counts do not include every sequence")
    direction = (counts.sum() - 15 * counts).sign().float()
    module.update_bias(counts)
    if not torch.equal(module.expert_bias, before + 0.001 * direction):
        raise AssertionError("MoE bias update is not the exact integer load direction")
    if module.expert_bias.data_ptr() != pointer:
        raise AssertionError("MoE bias update replaced its persistent buffer")
    with torch.no_grad():
        changed, changed_auxiliary, changed_weights, changed_counts = module(x, want_weights=True)
    if not torch.equal(auxiliary, changed_auxiliary):
        raise AssertionError("selection bias changed the unbiased sequence auxiliary")
    if torch.equal(weights.ne(0), changed_weights.ne(0)):
        raise AssertionError("MoE bias update did not change the tied-score selections")
    if not torch.allclose(
        changed_weights[changed_weights.ne(0)],
        torch.full_like(changed_weights[changed_weights.ne(0)], 1 / 3),
        atol=0, rtol=0,
    ):
        raise AssertionError("selection bias leaked into the normalized expert gates")
    module.eval()
    saved_bias = module.expert_bias.clone()
    module.update_bias(changed_counts)
    with torch.no_grad():
        module(x)
    if not torch.equal(module.expert_bias, saved_bias):
        raise AssertionError("MoE evaluation mutated routing bias")
    if "expert_bias" in dict(module.named_parameters()):
        raise AssertionError("MoE bias entered the trainable parameter surface")
    snapshot = {name: value.clone() for name, value in module.state_dict().items()}
    if "expert_bias" not in snapshot:
        raise AssertionError("MoE snapshot omitted the routing bias")
    restored = MixtureOfExperts(48, 140).cuda().eval()
    restored.load_state_dict(snapshot)
    with torch.no_grad():
        resumed, _, resumed_weights, resumed_counts = restored(x, want_weights=True)
    if not torch.equal(changed_weights, resumed_weights) or not torch.equal(
        changed_counts, resumed_counts
    ):
        raise AssertionError("MoE snapshot did not restore exact routing state")
    _close("restored biased output", resumed, changed, 1e-6)


def _compiled_replay() -> None:
    """Capture counts and gradients, then change the routing bias between steps."""
    from torch._dynamo.utils import counters

    from .train import _capture_without_gc

    torch.manual_seed(29)
    module = MixtureOfExperts(48, 140).cuda().train()
    with torch.no_grad():
        # Tiny differences keep input-dependent initial choices, while one
        # default bias update is guaranteed to move the overused boundary.
        module.router.weight.mul_(1e-4)
    sinks = {parameter: torch.zeros_like(parameter) for parameter in module.parameters()}
    bound, refresh = module.bind_gradient_sinks(sinks)
    for parameter in module.parameters():
        parameter.grad = sinks[parameter]
    x = torch.randn(1, 37, 48, device="cuda", dtype=torch.bfloat16).requires_grad_()
    x.grad = torch.zeros_like(x)
    usage = torch.zeros(15, device="cuda", dtype=torch.int64)
    usage_pointer = usage.data_ptr()
    bias_pointer = module.expert_bias.data_ptr()
    prepared_bias = module.expert_bias.clone()
    compiled = torch.compile(module, fullgraph=True, mode="max-autotune-no-cudagraphs")

    def clear():
        for gradient in sinks.values():
            gradient.zero_()
        x.grad.zero_()
        usage.zero_()

    def body():
        output, auxiliary, _, counts = compiled(x)
        loss = output.float().square().mean() + 1e-4 * auxiliary
        loss.backward()
        usage.add_(counts)
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
    if not torch.equal(module.expert_bias, prepared_bias):
        raise AssertionError("MoE warm-up or capture mutated the routing bias")
    pointer = {parameter: gradient.data_ptr() for parameter, gradient in sinks.items()}
    unique_graphs = counters["stats"]["unique_graphs"]

    def check_replay():
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
        if usage.data_ptr() != usage_pointer or usage.sum().item() != 3 * x.shape[1]:
            raise AssertionError("MoE captured logical counts were lost or duplicated")
        with torch.no_grad():
            output, auxiliary, weights, counts = _reference(module, x)
            expected = output.float().square().mean() + 1e-4 * auxiliary
        _close("changing-bias graph loss", captured_loss, expected, 0.015)
        if not torch.equal(usage, counts):
            raise AssertionError("MoE captured counts disagree with biased sigmoid routing")
        return weights

    for index, seed in enumerate((41, 43)):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.no_grad():
            x.copy_(torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=generator))
        bias_before = module.expert_bias.clone()
        clear()
        graph.replay()
        torch.cuda.synchronize()
        old_weights = check_replay()
        if not torch.equal(module.expert_bias, bias_before):
            raise AssertionError("MoE replay or backward updated bias inside a step")
        direction = (usage.sum() - 15 * usage).sign().float()
        module.update_bias(usage)
        if not torch.equal(module.expert_bias, bias_before + 0.001 * direction):
            raise AssertionError("MoE step bias did not follow complete captured counts")
        if module.expert_bias.data_ptr() != bias_pointer:
            raise AssertionError("MoE step replaced the captured bias buffer")
        updated_bias = module.expert_bias.clone()
        clear()
        graph.replay()
        torch.cuda.synchronize()
        new_weights = check_replay()
        if not torch.equal(module.expert_bias, updated_bias):
            raise AssertionError("MoE subsequent replay mutated the updated bias")
        if index == 0 and torch.equal(old_weights.ne(0), new_weights.ne(0)):
            raise AssertionError("MoE captured route did not respond to the step bias update")

    snapshot = {name: value.clone() for name, value in module.state_dict().items()}
    with torch.no_grad():
        module.expert_bias.add_(0.25)
    module.load_state_dict(snapshot)
    if module.expert_bias.data_ptr() != bias_pointer or not torch.equal(
        module.expert_bias, snapshot["expert_bias"]
    ):
        raise AssertionError("MoE state restore did not preserve the captured bias buffer")
    clear()
    graph.replay()
    torch.cuda.synchronize()
    check_replay()
    if counters["stats"]["unique_graphs"] != unique_graphs:
        raise AssertionError("MoE graph replay or bias update compiled a new graph")
    if len(refresh) != 32:
        raise AssertionError("MoE shadow surface omits an expert projection")


@torch.no_grad()
def _decode_parity() -> dict[str, float]:
    """Qualify cache semantics separately from finite-precision hard routing.

    CUDA's full-row and cached mixer paths can move a near-tied third/fourth
    expert boundary in either BF16 or FP32. Report ordinary trajectories;
    hold fixed decisions to the component rounding bound and prove the full
    standard/feedback cache semantics with the same parameters and tokens in
    the portable FP32 equations.
    """
    from contextlib import contextmanager

    from torch.utils._python_dispatch import TorchDispatchMode

    from .model import DeltaModel, KVCache, condition_config

    class _FixedRoutes(TorchDispatchMode):
        def __init__(self, selections):
            super().__init__()
            self.selections = selections
            self.column = 0
            self.calls = 0
            self.topk_calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.topk.default and args[0].shape[-1] == 15:
                self.topk_calls += 1
                if self.topk_calls == 1:
                    # The first top-k controls biased dispatch. The second
                    # belongs to the unbiased sequence auxiliary and stays
                    # on its actual input-dependent branch.
                    scores = args[0]
                    chosen = self.selections[self.calls % len(self.selections)]
                    selected = chosen[0, self.column : self.column + scores.shape[0]]
                    self.calls += 1
                    return scores.gather(-1, selected), selected
            return func(*args, **(kwargs or {}))

    @contextmanager
    def fixed_routes(model, weights):
        fixed = _FixedRoutes([value.topk(3).indices for value in weights.values()])
        originals = [(block.mlp, block.mlp.forward) for block in model.blocks]
        # Intervene only on the discrete expert decision, keeping its current
        # logits, mixture probabilities, and sparse CUDA arithmetic. Compiled
        # attention must remain outside this diagnostic dispatch mode.
        for module, original in originals:
            def routed_forward(*args, forward=original, route_mode=fixed, **kwargs):
                route_mode.topk_calls = 0
                with route_mode:
                    result = forward(*args, **kwargs)
                if route_mode.topk_calls != 2:
                    raise AssertionError("MoE route control expected dispatch and auxiliary top-k")
                return result

            module.forward = routed_forward
        try:
            yield fixed
        finally:
            for module, original in originals:
                module.forward = original

    def standard(model, tokens, dtype, fixed=None):
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

    def feedback(model, tokens, dtype):
        embeddings = model.embed_tokens(tokens)
        cache = KVCache(model.cfg, 1, tokens.device, dtype)
        prefill = model.forward_column(embeddings[:, :5], cache=cache, want_weights=True)
        payload = prefill.payload[:, -1:]
        pieces = []
        reference_rows = embeddings[:, :5]
        for column in range(5, tokens.shape[1]):
            result = model.step(
                tokens[:, column : column + 1], payload, cache, want_weights=True
            )
            pieces.append(result.h_top)
            payload = result.payload
            previous = model.forward_column(reference_rows, want_weights=True).payload[:, -1:]
            reference_rows = torch.cat(
                [reference_rows, model.fuse(previous, embeddings[:, column : column + 1])],
                dim=1,
            )
        reference = model.forward_column(reference_rows, want_weights=True).h_top[:, 5:]
        return torch.cat(pieces, dim=1), reference

    def finite_relative(label, actual, expected):
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError(f"MoE {label} produced nonfinite outputs")
        return _relative(actual, expected)

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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            whole = model.forward_column(model.embed_tokens(tokens), want_weights=True)
            changed = tokens.clone()
            changed[:, 7:] = (changed[:, 7:] + 1) % config.vocab_size
            future = model.forward_column(model.embed_tokens(changed), want_weights=True)
            errors[f"{condition}.causal"] = _close(
                f"{condition} causal prefix", future.h_top[:, :7], whole.h_top[:, :7], 1e-6
            )
            for site, weights in whole.expert_weights.items():
                if not torch.equal(weights[:, :7], future.expert_weights[site][:, :7]):
                    raise AssertionError(f"future tokens altered expert choice at {site}")
            incremental, incremental_weights = standard(model, tokens, torch.bfloat16)
            errors[f"{condition}.bf16_raw"] = finite_relative(
                f"{condition} standard", incremental, whole.h_top
            )
            errors[f"{condition}.swapped_sites_tokens"] = sum(
                int(weights.ne(0).ne(incremental_weights[site].ne(0)).any(dim=-1).sum())
                for site, weights in whole.expert_weights.items()
            )
            with fixed_routes(model, whole.expert_weights) as fixed:
                pinned, pinned_weights = standard(model, tokens, torch.bfloat16, fixed)
            for site, weights in whole.expert_weights.items():
                if not torch.equal(weights.ne(0), pinned_weights[site].ne(0)):
                    raise AssertionError(f"route-conditioned cache control changed {site}")
            errors[f"{condition}.bf16_fixed"] = _close(
                f"{condition} route-conditioned BF16 cache", pinned, whole.h_top, 0.08
            )
            if config.feedback:
                actual, expected = feedback(model, tokens, torch.bfloat16)
                errors[f"{condition}.feedback_bf16_raw"] = finite_relative(
                    f"{condition} feedback", actual, expected
                )

        # CUDA FP32 still exercises the actual sparse kernels and cache tracks.
        with torch.autocast("cuda", enabled=False):
            whole = model.forward_column(model.embed_tokens(tokens), want_weights=True)
            incremental, incremental_weights = standard(model, tokens, torch.float32)
            raw = finite_relative(f"{condition} CUDA FP32 cache", incremental, whole.h_top)
            disagreements = {
                site: weights.ne(0).ne(incremental_weights[site].ne(0)).any(dim=-1)
                for site, weights in whole.expert_weights.items()
            }
            swapped = sum(int(changed.sum()) for changed in disagreements.values())
            with fixed_routes(model, whole.expert_weights) as fixed:
                pinned, pinned_weights = standard(model, tokens, torch.float32, fixed)
            for site, weights in whole.expert_weights.items():
                if not torch.equal(weights.ne(0), pinned_weights[site].ne(0)):
                    raise AssertionError(f"FP32 route-conditioned cache control changed {site}")
            conditional = _close(
                f"{condition} route-conditioned CUDA FP32 cache", pinned, whole.h_top, 0.01
            )
            errors[f"{condition}.cuda_fp32_raw"] = raw
            errors[f"{condition}.cuda_fp32_swapped_sites_tokens"] = swapped
            errors[f"{condition}.cuda_fp32_fixed"] = conditional
            if not swapped:
                # When the choices agree, the ordinary path must satisfy the
                # same bound without the diagnostic intervention.
                _close(f"{condition} CUDA FP32 cache", incremental, whole.h_top, 0.01)
        portable = copy.deepcopy(model).cpu().eval()
        cpu_tokens = tokens.cpu()
        whole = portable.forward_column(portable.embed_tokens(cpu_tokens), want_weights=True)
        incremental, _ = standard(portable, cpu_tokens, torch.float32)
        errors[f"{condition}.cpu_fp32"] = _close(
            f"{condition} portable FP32 cache", incremental, whole.h_top, 1e-4
        )
        if config.feedback:
            actual, expected = feedback(portable, cpu_tokens, torch.float32)
            errors[f"{condition}.feedback_cpu_fp32"] = _close(
                f"{condition} portable FP32 feedback cache", actual, expected, 1e-4
            )
    return errors


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
    bias_addresses = [block.mlp.expert_bias.data_ptr() for block in model.blocks]

    def biases():
        return torch.stack([block.mlp.expert_bias for block in model.blocks])

    def check_bias(expected, label):
        if not torch.equal(biases(), expected):
            raise AssertionError(f"MoE screen bias mutated during {label}")
        if [block.mlp.expert_bias.data_ptr() for block in model.blocks] != bias_addresses:
            raise AssertionError(f"MoE screen bias storage changed during {label}")

    initial_bias = biases().clone()

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
    check_bias(initial_bias, "training/evaluation warm-up and capture")
    count_addresses = {}
    for spec, state in runner.states.items():
        counts = state.expert_counts
        if counts is None or counts.dtype != torch.int64 or counts.shape != (cfg.layers, 15):
            raise AssertionError("MoE screen lacks physical-bank int64 count buffers")
        if counts.any():
            raise AssertionError("MoE warm-up or capture leaked into training counts")
        count_addresses[spec] = counts.data_ptr()

    def check_counts(state, micros):
        counts = state.expert_counts
        if counts.data_ptr() != count_addresses[state.spec]:
            raise AssertionError("MoE screen replaced a captured count buffer")
        repetitions = torch.tensor(
            [state.spec.iterations if cfg.is_core_layer(layer) else 1 for layer in range(cfg.layers)],
            device="cuda", dtype=torch.int64,
        )
        expected = repetitions * (3 * args.seq_len * args.micro_rows * state.spec.n_passes * micros)
        if not torch.equal(counts.sum(dim=1), expected):
            raise AssertionError(
                "MoE screen counts lost or duplicated logical forward assignments "
                f"for {state.spec}: {counts.sum(dim=1).tolist()} versus {expected.tolist()}"
            )
    if all_modes and len(runner.states) != 24:
        raise AssertionError("MoE loop qualification did not retain all 24 training modes")
    def replay_eval():
        unchanged_bias = biases().clone()
        unchanged_counts = {spec: state.expert_counts.clone() for spec, state in runner.states.items()}
        for state in eval_runner.states.values():
            state.rows.copy_(torch.randint_like(state.rows, high=args.vocab_size))
            state.val_sum.zero_()
            state.fused_sum.zero_()
            state.graph.replay()
            torch.cuda.synchronize()
            if not math.isfinite(state.val_sum.item()) or not math.isfinite(state.fused_sum.item()):
                raise AssertionError("MoE resident evaluation graph is nonfinite")
        check_bias(unchanged_bias, "evaluation replay")
        for spec, counts in unchanged_counts.items():
            if not torch.equal(runner.states[spec].expert_counts, counts):
                raise AssertionError("MoE evaluation replay changed training usage counts")

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
        step_bias = biases().clone()
        generator = torch.Generator().manual_seed(59 + spec.n_passes + spec.iterations)
        runner.zero_grad()
        runner.begin(spec, args.zloss)
        check_counts(state, 0)
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
        check_counts(state, 2)
        check_bias(step_bias, "forward/backward and checkpoint recomputation")
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
        check_bias(step_bias, "parameter optimizer updates")
        counts = state.expert_counts
        direction = (counts.sum(dim=1, keepdim=True) - 15 * counts).sign().float()
        model.update_expert_bias(counts)
        expected_bias = step_bias + 0.001 * direction
        model.refresh_shadows()
        check_bias(expected_bias, "one explicit step-boundary bias update")
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
        check_counts(state, 1)
        check_bias(expected_bias, "post-update replay")
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
    replay_bias = biases().clone()
    state = runner.begin(first)
    runner.zero_grad()
    runner.replay(state, rows, 23, 0)
    runner.prepare_optimizer(state)
    torch.cuda.synchronize()
    check_counts(state, 1)
    check_bias(replay_bias, "raw replay after the cap graph")
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
    errors.append(max(_kernel_parity(48, 140, 19, batch=3, biased=True).values()))
    _frozen_kernel_parity()
    _frozen_kernel_parity(partial=True)
    _bias_step_parity()
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

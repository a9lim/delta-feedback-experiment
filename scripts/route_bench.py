"""Standalone MHDB routing bench: forward and backward at production shapes.

Times ``_route_forward_kernel`` and ``_route_backward_kernel`` directly with
CUDA events, over the screen-scale site profile (33 sites per column: eight
each with a bank of 2, 3, 4 and 5 sources, plus the payload writer's 6).
Launch parameters are overridable so Hopper can be swept without editing the
module.
"""

from __future__ import annotations

import argparse
import itertools
import json

import torch
import triton

from delta_feedback_experiment import cuda_kernels as ck

DIM = 768
HEADS = 2
SEQ = 4096
EPS = 1e-6

# (sources in the bank including the null, number of such sites per column)
SITE_PROFILE = ((2, 8), (3, 8), (4, 8), (5, 8), (6, 1))


def make_case(rows: int, n_bank: int, dtype=torch.bfloat16):
    """Trained-like operands for one routing site with ``n_bank`` sources."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(17 + n_bank)
    n_real = n_bank - 1
    bt = rows * SEQ
    projected = (torch.randn(DIM, generator=gen, device=dev) * 0.05).to(dtype)
    null = (torch.randn(DIM, generator=gen, device=dev) * 0.5).to(dtype)
    sources = [
        (torch.randn(rows, SEQ, DIM, generator=gen, device=dev)).to(dtype)
        for _ in range(n_real)
    ]
    grad_routed = torch.randn(rows, SEQ, DIM, generator=gen, device=dev).to(dtype)
    accumulators = [torch.zeros_like(s) for s in sources]
    weights = torch.rand(n_bank, bt, HEADS, device=dev, dtype=torch.float32)
    weights /= weights.sum(0, keepdim=True)
    inv_rms = torch.rand(n_bank, bt, device=dev, dtype=torch.float32) + 0.5
    routed = torch.empty_like(sources[0])
    return dict(
        rows=rows,
        bt=bt,
        n_bank=n_bank,
        projected=projected,
        null=null,
        sources=sources,
        accumulators=accumulators,
        grad_routed=grad_routed,
        weights=weights,
        inv_rms=inv_rms,
        routed=routed,
    )


def launch_geometry(block_k: int | None, warps: int | None, backward=False):
    block_h = 1 << (HEADS - 1).bit_length()
    if block_k is None:
        block_h, bk, tiles, *picks = ck._route_launch(HEADS, DIM // HEADS, torch.device("cuda"))
        nw = picks[1] if backward and len(picks) > 1 else picks[0]
        return block_h, bk, tiles, (warps or nw)
    head_dim = DIM // HEADS
    tiles = -(-head_dim // block_k)
    lanes = block_h * block_k * tiles
    default = 8 if lanes >= 4096 else (4 if lanes >= 1024 else 2)
    return block_h, block_k, tiles, (warps or default)


def forward_call(case, block_k, warps, stages, tpp):
    block_h, bk, tiles, nw = launch_geometry(block_k, warps)
    bank = (case["null"], *case["sources"])
    padded = ck._padded_sources(bank)
    n_sources = case["n_bank"]
    bt = case["bt"]
    extra = {}
    if "tokens_per_program" in ck._route_forward_kernel.arg_names:
        extra["tokens_per_program"] = tpp
        grid = (triton.cdiv(bt, tpp),)
    else:
        grid = (bt,)
    ck._route_forward_kernel[grid](
        *padded,
        case["projected"],
        case["weights"],
        case["inv_rms"],
        case["routed"],
        bt=bt,
        dim=DIM,
        num_heads=HEADS,
        head_dim=DIM // HEADS,
        eps=EPS,
        n_sources=n_sources,
        block_h=block_h,
        block_k=bk,
        block_n=triton.next_power_of_2(n_sources),
        tiles=tiles,
        num_warps=nw,
        num_stages=stages,
        **extra,
    )


def backward_call(case, block_k, warps, stages, tpp):
    block_h, bk, tiles, nw = launch_geometry(block_k, warps, backward=True)
    bank = (case["null"], *case["sources"])
    padded = ck._padded_sources(bank)
    destinations = list(case["accumulators"])
    padded_grads = tuple(destinations) + (destinations[-1],) * (
        ck.MAX_ROUTE_SOURCES - len(destinations)
    )
    bt = case["bt"]
    programs = triton.cdiv(bt, tpp)
    if case.get("partials") is None or case["partials"].shape[1] != programs:
        case["partials"] = torch.empty(
            (2, programs, DIM), device="cuda", dtype=torch.float32
        )
    middle: list = [case["projected"], case["grad_routed"]]
    if "routed" in ck._route_backward_kernel.arg_names:
        middle.append(case["routed"])
    middle += [case["weights"], case["inv_rms"], case["partials"]]
    ck._route_backward_kernel[(programs,)](
        *padded,
        *padded_grads,
        *middle,
        bt=bt,
        dim=DIM,
        num_heads=HEADS,
        head_dim=DIM // HEADS,
        n_sources=case["n_bank"],
        n_banked=len(case["accumulators"]),
        tokens_per_program=tpp,
        block_h=block_h,
        block_k=bk,
        tiles=tiles,
        num_warps=nw,
        num_stages=stages,
    )
    case["partials"].sum(dim=1)


def time_ms(fn, iters=8, warmup=3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bytes_forward(n_bank, bt):
    return bt * DIM * 2 * (n_bank - 1 + 1)


def bytes_backward(n_bank, bt, has_routed):
    passes = 1 if has_routed else 2
    n_real = n_bank - 1
    per = passes * n_real + 1 + (1 if has_routed else 0) + 2 * n_real
    return bt * DIM * 2 * per


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[4])
    parser.add_argument("--block-k", type=int, nargs="+", default=[0])
    parser.add_argument("--warps", type=int, nargs="+", default=[0])
    parser.add_argument("--stages", type=int, nargs="+", default=[1])
    parser.add_argument("--tpp", type=int, nargs="+", default=[8])
    parser.add_argument("--fwd-tpp", type=int, nargs="+", default=[1])
    parser.add_argument("--banks", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    print("module:", ck.__file__)
    has_routed = "routed" in ck._route_backward_kernel.arg_names
    fwd_tpp_ok = "tokens_per_program" in ck._route_forward_kernel.arg_names
    print("backward reads routed:", has_routed, "| forward tpp knob:", fwd_tpp_ok)

    counts = dict(SITE_PROFILE)
    records = []
    for rows in args.rows:
        cases = {n: make_case(rows, n) for n in args.banks}
        for block_k, warps, stages in itertools.product(
            args.block_k, args.warps, args.stages
        ):
            bk = block_k or None
            wp = warps or None
            geom = launch_geometry(bk, wp)
            for ftpp in args.fwd_tpp:
                fwd = {}
                for n in args.banks:
                    case = cases[n]
                    fwd[n] = time_ms(
                        lambda c=case, f=ftpp: forward_call(c, bk, wp, stages, f),
                        iters=args.iters,
                    )
                total = sum(fwd[n] * counts[n] for n in args.banks if n in counts)
                eff = {
                    n: bytes_forward(n, cases[n]["bt"]) / (fwd[n] * 1e-3) / 1e12
                    for n in args.banks
                }
                rec = dict(
                    kind="forward", rows=rows, geom=geom, stages=stages, tpp=ftpp,
                    per_bank={n: round(fwd[n], 4) for n in args.banks},
                    tbps={n: round(eff[n], 2) for n in args.banks},
                    column_ms=round(total, 3),
                )
                records.append(rec)
                print(json.dumps(rec))
            for tpp in args.tpp:
                bwd = {}
                for n in args.banks:
                    case = cases[n]
                    bwd[n] = time_ms(
                        lambda c=case, t=tpp: backward_call(c, bk, wp, stages, t),
                        iters=args.iters,
                    )
                total = sum(bwd[n] * counts[n] for n in args.banks if n in counts)
                eff = {
                    n: bytes_backward(n, cases[n]["bt"], has_routed)
                    / (bwd[n] * 1e-3)
                    / 1e12
                    for n in args.banks
                }
                rec = dict(
                    kind="backward", rows=rows, geom=geom, stages=stages, tpp=tpp,
                    per_bank={n: round(bwd[n], 4) for n in args.banks},
                    tbps={n: round(eff[n], 2) for n in args.banks},
                    column_ms=round(total, 3),
                )
                records.append(rec)
                print(json.dumps(rec))
        del cases
        torch.cuda.empty_cache()

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(records, handle, indent=1)


if __name__ == "__main__":
    main()

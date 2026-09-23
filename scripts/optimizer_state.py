"""Read the optimizer state a run's snapshots carry: how much of the gradient
is step-to-step bounce, whether momentum holds a persistent direction, and
which tensors are still moving ballistically.

    python scripts/optimizer_state.py runs/TAG.pt.5485 runs/TAG.pt.7314 \\
        runs/TAG.pt.9000 runs/TAG.pt.9142 --log logs/TAG.log

Snapshots are read in step order on the CPU. A momentum buffer fed
uncorrelated gradients holds a known share of its input energy, the iid
floor; energy below it means the gradient flips sign between steps, energy
above it means a persistent direction. Consecutive snapshots whose weights
barely rotate (late cooldown) have independent noise, so their momentum
overlap is the persistent share directly. With the run log, the logged
gradient norm splits the energy between the two optimizers.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from delta_feedback_experiment.analysis import config_from_args, saved_args
from delta_feedback_experiment.model import DeltaModel
from delta_feedback_experiment.optim import OPTIMIZER_GROUPS, split_parameters
from delta_feedback_experiment.train import read_checkpoint

torch.set_grad_enabled(False)


def state_names(model: torch.nn.Module) -> list[list[str]]:
    """The parameter name behind every optimizer state index: a state dict
    numbers its entries through the scheduled groups in order."""
    names = {parameter: name for name, parameter in model.named_parameters()}
    groups = split_parameters(model)
    return [[names[p] for group in stack for p in groups[group]] for stack in OPTIMIZER_GROUPS]


def category(name: str) -> str:
    """One label per tensor role: block and expert indices collapse."""
    name = re.sub(r"^blocks\.\d+\.", "L.", name)
    return re.sub(r"\.\d+\.", ".#.", name)


def flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().flatten()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = flat(a), flat(b)
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def rotation(a: torch.Tensor, b: torch.Tensor) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine(a, b)))))


def logged(path: Path, keys: tuple[str, ...]) -> dict[int, dict[str, float]]:
    out = {}
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.startswith(b"step "):
                continue
            fields = dict(part.strip().split("=", 1) for part in raw.decode("utf8", "replace").split("|")[1:] if "=" in part)
            out[int(fields["step"].split("/")[0])] = {k: float(fields[k]) for k in keys if k in fields}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("snapshots", type=Path, nargs="+", help="two or more snapshots of one run")
    parser.add_argument("--log", type=Path, help="the run's log, for the gradient-norm split and learning-rate path")
    parser.add_argument("--top", type=int, default=24, help="NAdam categories to list, largest first")
    args = parser.parse_args()
    payloads = sorted((read_checkpoint(p) for p in args.snapshots), key=lambda p: p["step"])
    if len(payloads) < 2:
        parser.error("give at least two snapshots")
    steps = [p["step"] for p in payloads]
    with torch.device("meta"):
        model = DeltaModel(config_from_args(saved_args(payloads[0])))
    muon, nadam = state_names(model)
    nadam_groups = payloads[0]["optimizer"]["stack"][1]["param_groups"]
    nadam_lr = [g["stable_lr"] for g in nadam_groups for _ in g["params"]]
    nadam_base = nadam_groups[0]["stable_lr"]
    log = logged(args.log, ("gnorm", "lr_normuonh", "lr_nadam")) if args.log else {}
    pairs = [f"{a}→{b}" for a, b in itertools.pairwise(steps)]
    print(f"{payloads[0]['args']['tag']}: snapshots at {steps}")
    if log:
        print("  learning-rate multiplier at each:", [round(log[s]["lr_normuonh"] / payloads[0]["optimizer"]["stack"][0]["param_groups"][0]["stable_lr"], 3) if s in log else None for s in steps])

    # NorMuonH: rotation of each matrix between snapshots, momentum overlap, energy.
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    energy = np.zeros(len(payloads))
    for index, name in enumerate(muon):
        weights = [p["state"][name] for p in payloads]
        momenta = [p["optimizer"]["stack"][0]["state"][index]["momentum"] for p in payloads]
        row = rows[category(name)]
        row["n"].append(weights[0].numel())
        for k in range(len(payloads) - 1):
            row[f"rot{k}"].append(rotation(weights[k], weights[k + 1]))
            row[f"mcos{k}"].append(cosine(momenta[k], momenta[k + 1]))
        for k, m in enumerate(momenta):
            energy[k] += float(flat(m).square().sum())
    print("\nNorMuonH matrices: mean rotation (deg) between consecutive snapshots, then momentum overlap cos(M_k, M_k+1)")
    print(f"{'category':46s}{'#':>5s} " + " ".join(f"{p:>12s}" for p in pairs) + " | " + " ".join(f"{p:>12s}" for p in pairs))
    for name, row in sorted(rows.items(), key=lambda kv: -sum(kv[1]["n"])):
        print(
            f"{name:46s}{len(row['n']):5d} " + " ".join(f"{np.mean(row[f'rot{k}']):12.2f}" for k in range(len(pairs)))
            + " | " + " ".join(f"{np.mean(row[f'mcos{k}']):12.3f}" for k in range(len(pairs)))
        )
    beta_m = payloads[0]["optimizer"]["stack"][0]["param_groups"][0]["momentum"]
    print(f"  Σ|M|² per snapshot: {[f'{e:.5f}' for e in energy]}")

    # NAdam: EMA energy ratio against the iid floor, update size, momentum
    # correlation, and displacement against the learning-rate path.
    nrows = collections.defaultdict(lambda: collections.defaultdict(list))
    v_total = np.zeros(len(payloads))
    beta1 = payloads[0]["optimizer"]["stack"][1]["param_groups"][0]["betas"][0]
    for index, name in enumerate(nadam):
        states = [p["optimizer"]["stack"][1]["state"][index] for p in payloads]
        weights = [p["state"][name] for p in payloads]
        row = nrows[category(name)]
        row["n"].append(weights[0].numel())
        for k, s in enumerate(states):
            m, v = flat(s["exp_avg"]), flat(s["exp_avg_sq"])
            row[f"ratio{k}"].append(float(m.square().sum() / (v.sum() + 1e-30)))
            row[f"upd{k}"].append(float((m / (v.sqrt() + 1e-8)).square().mean().sqrt()))
            v_total[k] += float(v.sum())
        for k in range(len(payloads) - 1):
            row[f"mcorr{k}"].append(cosine(states[k]["exp_avg"], states[k + 1]["exp_avg"]))
            moved = float((flat(weights[k + 1]) - flat(weights[k])).square().mean().sqrt())
            if log:
                rate = sum(log[s]["lr_nadam"] / nadam_base * nadam_lr[index] for s in range(steps[k] + 1, steps[k + 1] + 1) if s in log)
            else:
                rate = nadam_lr[index] * (steps[k + 1] - steps[k])
            path = rate * 0.5 * (row[f"upd{k}"][-1] + row[f"upd{k + 1}"][-1])
            row[f"persist{k}"].append(moved / (path + 1e-30))
    floor = (1 - beta1) / (1 + beta1)
    print(f"\nNAdam: Σm²/Σv per snapshot (iid floor {floor:.4f}, persistent signal → 1); rms update m/√v; momentum corr and displacement/path per pair; top {args.top} by size")
    print(f"{'category':46s}{'#':>5s}{'numel':>10s} " + " ".join(f"{'ratio':>7s}" for _ in steps) + " " + " ".join(f"{'upd':>5s}" for _ in steps) + " | " + " ".join(f"{'mcorr':>7s}{'persist':>8s}" for _ in pairs))
    for name, row in sorted(nrows.items(), key=lambda kv: -sum(kv[1]["n"]))[: args.top]:
        print(
            f"{name:46s}{len(row['n']):5d}{sum(row['n']):10d} " + " ".join(f"{np.mean(row[f'ratio{k}']):7.3f}" for k in range(len(steps)))
            + " " + " ".join(f"{np.mean(row[f'upd{k}']):5.2f}" for k in range(len(steps)))
            + " | " + " ".join(f"{np.mean(row[f'mcorr{k}']):7.3f}{np.mean(row[f'persist{k}']):8.3f}" for k in range(len(pairs)))
        )

    # Energy split between the optimizers and the implied gradient autocorrelation.
    if log:
        def ema_g2(step: int) -> float:
            window = [(beta_m ** k, log[step - k]["gnorm"] ** 2) for k in range(200) if step - k in log]
            return sum(w * g for w, g in window) / sum(w for w, _ in window)
        g2 = np.array([ema_g2(s) for s in steps])
        muon_g2 = g2 - v_total
        print(f"\nlogged EMA gnorm²:        {[f'{x:.4f}' for x in g2]}")
        print(f"NAdam Σv (its |g|²):      {[f'{x:.4f}' for x in v_total]}  share {[f'{x:.2f}' for x in v_total / g2]}")
        print(f"NorMuonH |G|² (the rest): {[f'{x:.4f}' for x in muon_g2]}")
        ratio = energy / np.maximum(muon_g2, 1e-9)
        muon_floor = (1 - beta_m) / (1 + beta_m)
        print(f"NorMuonH Σ|M|²/|G|²:      {[f'{x:.4f}' for x in ratio]}  (iid floor {muon_floor:.4f})")

        def lag1(r: float, beta: float) -> float:
            x = max(r, 1e-9) * (1 + beta) / (1 - beta)
            return (x - 1) / (beta * (x + 1))
        emb = nrows.get("embed_tokens.weight")
        if emb:
            print(f"implied lag-1 gradient autocorrelation (AR(1) reading): embedding {[round(lag1(emb[f'ratio{k}'][0], beta1), 2) for k in range(len(steps))]}, NorMuonH {[round(float(lag1(r, beta_m)), 2) for r in ratio]}")

    # Scale drift of the tensors that set the logit temperature.
    print("\nfinal_norm gain mean, embedding row-norm rms over its 0.02·√D init:")
    for p in payloads:
        E = p["state"]["embed_tokens.weight"].float()
        print(f"  step {p['step']}: {p['state']['final_norm.weight'].mean():.3f}, {E.norm(dim=1).square().mean().sqrt() / (0.02 * E.shape[1] ** 0.5):.3f}")


if __name__ == "__main__":
    main()

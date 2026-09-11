"""Weight-space divergence of two paired checkpoints from their shared initialization.

Paired conditions initialize their shared trunk byte-identically and
letter-private modules pairwise from the same seed (design.md), so for every parameter both
snapshots share, the script measures how far each run moved from ``W0``, how
far apart the two ended, and how aligned their total updates are.  NorMuonH
matrices live on a fixed Frobenius sphere, so their movement is purely angular
and the reconstruction of ``W0`` is checked against their invariant radii.
Run this script from the initialization source revision that trained both
snapshots, including its base initialization constant. The radius check alone
cannot detect a different NAdam gate/control initialization.

Usage:
    python scripts/weight_divergence.py runs/A.pt.STEP runs/B.pt.STEP
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch

from delta_feedback_experiment.analysis import config_from_args
from delta_feedback_experiment.model import DeltaModel
from delta_feedback_experiment.parameter_groups import is_normuonh_parameter
from delta_feedback_experiment.train import read_checkpoint

MATRIX_FAMILIES = ("attn_matrix", "mlp_matrix", "pkda_control", "router", "norms", "attn_gate", "embedding", "final_norm")


def family(name: str) -> str:
    if name == "embed_tokens.weight":
        return "embedding"
    if name == "final_norm.weight":
        return "final_norm"
    if name.startswith("attention_gates."):
        return "attn_gate"
    if name == "fuse_value.weight":
        return "fuse_value"
    if name == "fuse_gate.weight":
        return "fuse_gate"
    if name in ("gate_norm.weight", "entry_norm.weight", "payload_norm.weight"):
        return "fbt_norms"
    if name.startswith("payload_router."):
        return "payload_router"
    if "_router." in name:
        return "router"
    if re.search(r"\.attn\.(q|k|v|o|qkv)_proj\.weight$", name):
        return "attn_matrix"
    if re.search(r"\.mlp\.(gate_up|down)_proj\.weight$", name):
        return "mlp_matrix"
    if re.search(r"\.attn\.(control_proj|decay_up|output_gate_up)\.", name):
        return "pkda_control"
    if "conv" in name:
        return "pkda_conv"
    if "norm" in name:
        return "norms"
    return "pkda_vectors"


def layer_of(name: str) -> int | None:
    m = re.match(r"blocks\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float((a * b).sum() / (na * nb))


def angle(c: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, c)))) if not math.isnan(c) else float("nan")


def initial_state(saved: dict) -> tuple[dict, set, set]:
    torch.manual_seed(saved["seed"])
    model = DeltaModel(config_from_args(saved))
    state = {k: v.detach().float() for k, v in model.state_dict().items()}
    normuonh = {n for n, p in model.named_parameters() if is_normuonh_parameter(n, p)}
    trainable = {n for n, _ in model.named_parameters()}
    return state, normuonh, trainable


def group_stats(w0: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor) -> dict:
    n0 = w0.norm().item()
    rel = lambda x: x / n0 if n0 > 0 else float("nan")
    return {
        "numel": int(w0.numel()), "norm0": n0,
        "move_a": rel((wa - w0).norm().item()), "move_b": rel((wb - w0).norm().item()), "gap": rel((wa - wb).norm().item()),
        "abs_move_a": (wa - w0).norm().item(), "abs_move_b": (wb - w0).norm().item(), "abs_gap": (wa - wb).norm().item(),
        "cos_a_b": cos(wa, wb), "cos_upd": cos(wa - w0, wb - w0),
        "angle_deg_a_0": angle(cos(wa, w0)), "angle_deg_b_0": angle(cos(wb, w0)), "angle_deg_a_b": angle(cos(wa, wb)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("a", type=Path, help="first snapshot")
    parser.add_argument("b", type=Path, help="second snapshot")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    pa = read_checkpoint(args.a)
    pb = read_checkpoint(args.b)
    saved_a, saved_b = pa["args"], pb["args"]
    if saved_a["seed"] != saved_b["seed"]:
        raise SystemExit("the snapshots were initialized from different seeds; nothing is paired")
    sa, sb = pa["state"], pb["state"]
    del pa["optimizer"], pb["optimizer"]
    tag_a, tag_b = saved_a["tag"], saved_b["tag"]
    out_dir = args.out_dir or Path("figures") / f"weights-{tag_a}-vs-{tag_b}"
    out_dir.mkdir(parents=True, exist_ok=True)

    s0a, normuonh_a, trainable_a = initial_state(saved_a)
    s0b, normuonh_b, trainable_b = initial_state(saved_b)
    shared = sorted(trainable_a & trainable_b)
    # The paired-initialization contract: every shared parameter starts identical.
    mismatched = [n for n in shared if not torch.equal(s0a[n], s0b[n])]
    if mismatched:
        raise SystemExit(f"shared parameters do not pair at init: {mismatched[:5]}")
    worst = 0.0
    for n, st, s0, normuonh in ((tag_a, sa, s0a, normuonh_a), (tag_b, sb, s0b, normuonh_b)):
        for name in normuonh:
            r0, r = s0[name].norm().item(), st[name].float().norm().item()
            worst = max(worst, abs(r0 - r) / r)
    print(f"NorMuonH radius reconstruction: worst relative mismatch {worst:.2e}")

    per_param, groups, fam_groups = {}, defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for n in shared:
        w0, wa, wb = s0a[n].reshape(-1), sa[n].float().reshape(-1), sb[n].float().reshape(-1)
        fam, layer = family(n), layer_of(n)
        per_param[n] = {"family": fam, "layer": layer, "normuonh": n in normuonh_a, **group_stats(w0, wa, wb)}
        for store, key in ((groups, (fam, layer)), (fam_groups, fam)):
            store[key]["w0"].append(w0)
            store[key]["wa"].append(wa)
            store[key]["wb"].append(wb)
    grouped = {}
    for (fam, layer), g in groups.items():
        key = f"{fam}@L{layer}" if layer is not None else fam
        grouped[key] = {"family": fam, "layer": layer, **group_stats(*(torch.cat(g[k]) for k in ("w0", "wa", "wb")))}
    families = {fam: group_stats(*(torch.cat(g[k]) for k in ("w0", "wa", "wb"))) for fam, g in fam_groups.items()}
    unshared = {}
    for tag, st, s0, trainable in ((tag_a, sa, s0a, trainable_a), (tag_b, sb, s0b, trainable_b)):
        for n in sorted(trainable - set(shared)):
            w0, w = s0[n].reshape(-1), st[n].float().reshape(-1)
            n0 = w0.norm().item()
            unshared[f"{tag}:{n}"] = {"family": family(n), "norm0": n0, "norm": w.norm().item(), "cos_to_init": cos(w, w0),
                                     "move": (w - w0).norm().item() / n0 if n0 > 0 else float("nan")}
    report = {"a": str(args.a), "b": str(args.b), "labels": {"a": tag_a, "b": tag_b}, "radius_check_worst_rel": worst,
              "families": families, "grouped": grouped, "unshared": unshared, "per_param": per_param}
    out_path = out_dir / "weight_divergence.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n== shared families (relative to ||W0||; a={tag_a}, b={tag_b}; cos_upd = alignment of the two total updates) ==")
    print(f"  {'family':<14}{'numel':>12}{'move_a':>9}{'move_b':>9}{'gap':>9}{'gap/move':>9}{'cos_a_b':>9}{'cos_upd':>9}")
    for fam, e in sorted(families.items(), key=lambda t: -t[1]["numel"]):
        ratio = e["gap"] / max(e["move_a"], 1e-12) if e["move_a"] == e["move_a"] else float("nan")
        print(f"  {fam:<14}{e['numel']:>12}{e['move_a']:>9.4f}{e['move_b']:>9.4f}{e['gap']:>9.4f}{ratio:>9.3f}{e['cos_a_b']:>9.4f}{e['cos_upd']:>9.4f}")
    print("\n== per layer ==")
    print(f"  {'layer':<6}{'family':<14}{'move_a':>9}{'move_b':>9}{'gap':>9}{'cos_upd':>9}{'ang_a0':>8}{'ang_b0':>8}{'ang_ab':>8}")
    for key, e in sorted(grouped.items(), key=lambda t: (t[1]["layer"] if t[1]["layer"] is not None else -1, t[1]["family"])):
        if e["family"] not in MATRIX_FAMILIES:
            continue
        print(f"  {e['layer']!s:<6}{e['family']:<14}{e['move_a']:>9.4f}{e['move_b']:>9.4f}{e['gap']:>9.4f}{e['cos_upd']:>9.4f}{e['angle_deg_a_0']:>8.2f}{e['angle_deg_b_0']:>8.2f}{e['angle_deg_a_b']:>8.2f}")
    print("\n== unshared parameters ==")
    for n, e in unshared.items():
        print(f"  {n:<44} norm0={e['norm0']:.3f} norm={e['norm']:.3f} cos_to_init={e['cos_to_init']:.4f} move={e['move']:.4f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

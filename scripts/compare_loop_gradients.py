"""Compare saved full-batch gradients from paired loop qualification runs.

    python scripts/compare_loop_gradients.py BASELINE_DIR CANDIDATE_DIR --output /tmp/gradient-comparison.json

Pairs kX-rY.pt files by name. FP64 chunked reductions avoid large temporary
copies. Per-parameter relative errors are diagnostic: tiny baseline gradients
can have large relative changes while barely affecting the full update.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch

MODE_FILE = re.compile(r"k(\d+)-r(\d+)\.pt")


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def measures(sums: dict) -> dict:
    if sums["baseline_nonfinite"] or sums["candidate_nonfinite"]:
        return {
            "finite": False,
            "baseline_nonfinite": sums["baseline_nonfinite"],
            "candidate_nonfinite": sums["candidate_nonfinite"],
            "baseline_l2": None,
            "candidate_l2": None,
            "error_l2": None,
            "relative_l2": None,
            "norm_ratio": None,
            "cosine": None,
            "max_abs": None,
        }
    baseline_norm = math.sqrt(sums["baseline_square"])
    candidate_norm = math.sqrt(sums["candidate_square"])
    error_norm = math.sqrt(sums["error_square"])
    return {
        "finite": True,
        "baseline_nonfinite": 0,
        "candidate_nonfinite": 0,
        "baseline_l2": baseline_norm,
        "candidate_l2": candidate_norm,
        "error_l2": error_norm,
        "relative_l2": safe_ratio(error_norm, baseline_norm),
        "norm_ratio": safe_ratio(candidate_norm, baseline_norm),
        "cosine": safe_ratio(sums["dot"], baseline_norm * candidate_norm),
        "max_abs": sums["max_abs"],
    }


def new_sums() -> dict:
    return {
        "baseline_square": 0.0,
        "candidate_square": 0.0,
        "error_square": 0.0,
        "dot": 0.0,
        "max_abs": 0.0,
        "baseline_nonfinite": 0,
        "candidate_nonfinite": 0,
    }


def tensor_sums(baseline: torch.Tensor, candidate: torch.Tensor, chunk: int) -> dict:
    baseline = baseline.reshape(-1)
    candidate = candidate.reshape(-1)
    sums = new_sums()
    for offset in range(0, baseline.numel(), chunk):
        a = baseline[offset : offset + chunk].to(torch.float64)
        b = candidate[offset : offset + chunk].to(torch.float64)
        bad_a = int((~torch.isfinite(a)).sum())
        bad_b = int((~torch.isfinite(b)).sum())
        sums["baseline_nonfinite"] += bad_a
        sums["candidate_nonfinite"] += bad_b
        if bad_a or bad_b:
            continue
        difference = b - a
        sums["baseline_square"] += float(a.square().sum())
        sums["candidate_square"] += float(b.square().sum())
        sums["error_square"] += float(difference.square().sum())
        sums["dot"] += float((a * b).sum())
        if difference.numel():
            sums["max_abs"] = max(sums["max_abs"], float(difference.abs().max()))
    return sums


def load_gradients(path: Path) -> dict[str, torch.Tensor]:
    gradients = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(gradients, dict) or any(
        not isinstance(name, str)
        or not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.layout != torch.strided
        for name, value in gradients.items()
    ):
        raise ValueError(f"{path}: expected a dict of names to dense floating tensors")
    if not gradients:
        raise ValueError(f"{path}: gradient dictionary is empty")
    return gradients


def compare_pair(baseline_path: Path, candidate_path: Path, options) -> dict:
    baseline, candidate = load_gradients(baseline_path), load_gradients(candidate_path)
    missing = sorted(baseline.keys() - candidate.keys())
    extra = sorted(candidate.keys() - baseline.keys())
    mismatches = []
    dtype_differences = []
    totals = new_sums()
    parameters = []
    for name in sorted(baseline.keys() & candidate.keys()):
        a, b = baseline[name], candidate[name]
        if a.shape != b.shape:
            mismatches.append(
                {"name": name, "baseline": list(a.shape), "candidate": list(b.shape)}
            )
            continue
        if a.dtype != b.dtype:
            dtype_differences.append(
                {"name": name, "baseline": str(a.dtype), "candidate": str(b.dtype)}
            )
        sums = tensor_sums(a, b, options.chunk_elements)
        for key in totals:
            if key == "max_abs":
                totals[key] = max(totals[key], sums[key])
            else:
                totals[key] += sums[key]
        parameters.append(
            {
                "name": name,
                "shape": list(a.shape),
                "elements": a.numel(),
                **measures(sums),
            }
        )
    structural_match = not (missing or extra or mismatches)
    global_measures = measures(totals)
    global_norm = global_measures["baseline_l2"]
    global_error = global_measures["error_l2"]
    for row in parameters:
        baseline_norm, error_norm = row["baseline_l2"], row["error_l2"]
        row["small_baseline_gradient"] = (
            baseline_norm <= options.small_gradient_fraction * global_norm
            if baseline_norm is not None and global_norm is not None
            else None
        )
        row["baseline_zero_candidate_nonzero"] = baseline_norm == 0 and row[
            "candidate_l2"
        ] not in (0, None)
        row["error_energy_fraction"] = (
            (error_norm / global_error) ** 2
            if error_norm is not None and global_error
            else None
        )
    parameters.sort(
        key=lambda row: row["error_l2"] if row["error_l2"] is not None else math.inf,
        reverse=True,
    )
    relative_order = sorted(
        parameters,
        key=lambda row: (
            row["relative_l2"]
            if row["relative_l2"] is not None
            else (
                math.inf
                if row["baseline_zero_candidate_nonzero"] or not row["finite"]
                else -1
            )
        ),
        reverse=True,
    )
    relative_error = global_measures["relative_l2"]
    catastrophic = (
        relative_error is not None
        and relative_error >= options.catastrophic_relative_error
    ) or (global_norm == 0 and global_measures["candidate_l2"] not in (None, 0))
    return {
        "mode": baseline_path.stem,
        "baseline_file": str(baseline_path.resolve()),
        "candidate_file": str(candidate_path.resolve()),
        "structural_match": structural_match,
        "missing_parameters": missing,
        "extra_parameters": extra,
        "shape_mismatches": mismatches,
        "dtype_differences": dtype_differences,
        "compared_parameter_count": len(parameters),
        "compared_elements": sum(row["elements"] for row in parameters),
        "global_scope": "all gradients"
        if structural_match
        else "matching parameters only; incomplete comparison",
        "global": global_measures,
        "catastrophic_mismatch": catastrophic,
        "passed_basic_checks": structural_match
        and global_measures["finite"]
        and not catastrophic,
        "top_by_error_l2": parameters[: options.top],
        "top_by_relative_l2": relative_order[: options.top],
        "parameters_by_error_l2": parameters,
    }


def mode_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"Not a gradient directory: {directory}")
    return {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and MODE_FILE.fullmatch(path.name)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--chunk-elements", type=int, default=1_048_576)
    parser.add_argument("--small-gradient-fraction", type=float, default=1e-4)
    parser.add_argument(
        "--catastrophic-relative-error",
        type=float,
        default=1.0,
        help="basic failure boundary; default only rejects full-gradient error at least as large as the baseline",
    )
    options = parser.parse_args()
    if (
        options.top < 1
        or options.chunk_elements < 1
        or options.small_gradient_fraction < 0
        or options.catastrophic_relative_error <= 0
    ):
        parser.error(
            "counts and catastrophic threshold must be positive; small-gradient fraction must be nonnegative"
        )
    baseline, candidate = mode_files(options.baseline), mode_files(options.candidate)
    common = baseline.keys() & candidate.keys()
    report = {
        "baseline_directory": str(options.baseline.resolve()),
        "candidate_directory": str(options.candidate.resolve()),
        "missing_candidate_modes": sorted(baseline.keys() - candidate.keys()),
        "extra_candidate_modes": sorted(candidate.keys() - baseline.keys()),
        "thresholds": {
            "small_gradient_fraction_of_global_norm": options.small_gradient_fraction,
            "catastrophic_global_relative_l2": options.catastrophic_relative_error,
        },
        "interpretation": [
            "This checks engineering gradient agreement, not training quality or equivalent learning trajectories.",
            "Snapshots must come from matched weights, rows, randomness, and accumulation schedules; tensor files alone cannot verify that provenance.",
            "Sort absolute error and global error-energy contribution first; large relative errors on tiny gradients need not materially change the update.",
            "Relative error and norm ratio are null for a zero baseline norm; cosine is null if either norm is zero.",
            "Basic checks reject missing/extra modes or parameters, shape mismatch, nonfinite gradients, and catastrophic global mismatch; no 1% agreement gate is imposed.",
        ],
        "modes": [],
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    for name in sorted(
        common, key=lambda value: tuple(map(int, MODE_FILE.fullmatch(value).groups()))
    ):
        comparison = compare_pair(baseline[name], candidate[name], options)
        report["modes"].append(comparison)
        options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "mode": comparison["mode"],
                    "global": comparison["global"],
                    "passed_basic_checks": comparison["passed_basic_checks"],
                    "top_by_error_l2": comparison["top_by_error_l2"][:3],
                },
                allow_nan=False,
            ),
            flush=True,
        )
    report["passed_basic_checks"] = (
        bool(common)
        and not (report["missing_candidate_modes"] or report["extra_candidate_modes"])
        and all(row["passed_basic_checks"] for row in report["modes"])
    )
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["passed_basic_checks"]:
        raise SystemExit("Gradient comparison failed basic checks; see saved JSON.")


if __name__ == "__main__":
    main()

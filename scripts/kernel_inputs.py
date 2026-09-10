"""Record trained PKDA backward and CCE head operands for kernel benchmarks."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path

import torch

import delta_feedback_experiment.model as model_module
from delta_feedback_experiment import analysis
from delta_feedback_experiment.data import TokenData


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-dir", default="/data/delta/dclm-100b")
    parser.add_argument("--head-only", action="store_true")
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    payload = torch.load(options.snapshot, map_location="cpu", weights_only=False)
    saved = analysis.saved_args(payload)
    model = model_module.DeltaModel(analysis.config_from_args(saved))
    model.load_state_dict(payload["state"])
    del payload
    model.cuda().train()
    model.refresh_shadows()
    # Eager wrappers expose the real operator arguments without tracing fake
    # tensors or adding recording work to the measured CUDA graph.
    model_module._compiled_pkda_block = model_module._pkda_block
    model_module._compiled_block = model_module._attention_block
    model_module._compiled_route_sources = model_module._route_sources
    chunk = importlib.import_module("fla.ops.precond_kda.chunk")
    original = chunk.chunk_precond_kda_bwd_intra
    signature = inspect.signature(original)
    index = 0

    def capture(*args, **kwargs):
        nonlocal index
        values = signature.bind(*args, **kwargs).arguments
        if index in (0, 4, 8):
            values = {
                name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for name, value in values.items()
            }
            destination = options.output / f"intra-{index}.pt"
            torch.save(values, destination)
            print(
                json.dumps({"backward_index": index, "path": str(destination)}),
                flush=True,
            )
        index += 1
        return original(*args, **kwargs)

    chunk.chunk_precond_kda_bwd_intra = capture
    data = TokenData.load(options.data_dir, "train", saved["seq_len"])
    rows = data.batch(100000, 4, "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model_module.multipass(model, rows, 1)
        head = model.readout_input(outputs[0].h_top)
        torch.save(
            {
                "e": head.detach().cpu(),
                "c": model._classifier_shadow.detach().cpu(),
                "targets": rows[:, 1:].contiguous().cpu(),
                "metadata": {
                    "snapshot": str(options.snapshot),
                    "first_row": 100000,
                    "pass": 1,
                },
            },
            options.output / "head.pt",
        )
        del head
        if options.head_only:
            print(json.dumps({"head": str(options.output / "head.pt")}), flush=True)
            return
        loss, _ = model_module.multipass_loss(model, rows, outputs, z_coef=1e-5)
    loss.backward()
    torch.cuda.synchronize()
    print(json.dumps({"loss": loss.item(), "pkda_calls": index}), flush=True)


if __name__ == "__main__":
    main()

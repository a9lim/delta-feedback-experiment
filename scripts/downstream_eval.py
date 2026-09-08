"""Downstream zero-shot tasks on a snapshot, in Standard or Fused mode.

Uses the workspace's ``transformer_experiments.downstream`` tasks (pinned Hub
revisions, harness-identical prompts) with this model's own scorer: one plain
column pass (Standard) or a plain pass followed by a fully fused pass with
plain-prefix length 1 (Fused), the same modes ``val`` and ``val_fused`` report.
Batches are padded to a few bucket lengths so the compiled blocks see few
shapes.  Writes ``downstream_<mode>.json`` under ``figures/downstream-<tag>/``;
compare two runs with ``python -m transformer_experiments.downstream --compare``.

Usage:
    python scripts/downstream_eval.py runs/TAG.pt.STEP --mode standard
    python scripts/downstream_eval.py runs/TAG.pt.STEP --mode fused --tasks hellaswag lambada_openai
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformer_experiments import downstream

from delta_feedback_experiment import analysis
from delta_feedback_experiment.data import CANONICAL_TOKENIZER, CANONICAL_TOKENIZER_REVISION

MODES = ("standard", "fused")


class DFScorer:
    """Continuation scores from a DFModel column pass under the trainer's numerics."""

    def __init__(self, model, mode: str):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "fused" and not model.cfg.feedback_active:
            raise ValueError("fused mode needs an fbt or df snapshot")
        self.model = model
        self.mode = mode
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def score(self, ids, spans):
        model = self.model
        ids = ids.to(self.device)
        with analysis.autocast(self.device):
            e = model.embed_tokens(ids)
            out = model.forward_column(e, need_payload=self.mode == "fused")
            if self.mode == "fused":
                out = model.forward_column(analysis.fused_inputs(model, e, out.payload, 1), need_payload=False)
            weight = model.embed_tokens.weight
            head = lambda h: F.linear(model.final_norm(h), weight.to(h.dtype))
            return downstream.span_scores(out.h_top, ids, spans, head)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--mode", choices=MODES, default="standard")
    parser.add_argument("--tasks", nargs="*", default=list(downstream.DEFAULT_TASKS))
    parser.add_argument("--limit", type=int, default=None, help="documents per task (smoke tests)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--buckets", type=int, nargs="*", default=(128, 256, 512, 1024))
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    model, saved = analysis.load_checkpoint(args.snapshot, args.device)
    max_len = min(saved["seq_len"], max(args.buckets))
    tokenizer = AutoTokenizer.from_pretrained(CANONICAL_TOKENIZER, revision=CANONICAL_TOKENIZER_REVISION)
    tag = saved.get("tag", args.snapshot.stem)
    out_dir = args.out_dir or Path("figures") / f"downstream-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer = DFScorer(model, args.mode)
    started = time.time()
    results = downstream.run(
        args.tasks,
        downstream.hf_tokenize(tokenizer),
        scorer,
        limit=args.limit,
        batch_size=args.batch_size,
        max_len=max_len,
        buckets=[b for b in args.buckets if b <= max_len],
        progress=lambda line: print(f"  [{time.time() - started:6.0f}s] {line}", flush=True),
    )
    print("\n" + downstream.format_table(results))
    out_path = out_dir / f"downstream_{args.mode}.json"
    meta = {
        "snapshot": str(args.snapshot), "tag": tag, "arm": saved["arm"], "step": saved["step"], "mode": args.mode,
        "tokenizer": CANONICAL_TOKENIZER, "tokenizer_revision": CANONICAL_TOKENIZER_REVISION,
        "device": str(scorer.device), "limit": args.limit, "batch_size": args.batch_size, "buckets": list(args.buckets),
    }
    out_path.write_text(json.dumps(downstream.to_json(results, **meta), indent=1) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

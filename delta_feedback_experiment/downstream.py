"""Downstream zero-shot tasks on a run's snapshot: ``delta eval TAG``.

Uses the workspace's ``transformer_experiments.downstream`` tasks (pinned Hub
revisions, harness-identical prompts) with this model's own scorer: one plain
column pass (Standard), a plain pass followed by a fully fused pass with
plain-prefix length 1 (Fused, the mode ``val_fused`` reports), or a plain pass
followed by a fused pass whose plain prefix is each item's context, so only the
scored continuation receives feedback (Soft, the Jacobi form of feedback
decoding: exact for the first continuation token, and for the first ``k``
tokens after ``k`` passes).  ``--passes k`` iterates the feedback pass ``k``
times, each consuming the previous pass's payload, and scores the last pass.
Batches are padded to a few bucket lengths so the compiled blocks see few
shapes.

Every mode the snapshot's condition supports runs by default, against one
loaded model.  Each task emits a ``downstream`` record as it finishes, and each
mode writes ``figures/downstream-TAG/MODE.STEP.json`` (``MODEk.STEP.json`` for
``k > 1`` passes); compare two files with
``python -m transformer_experiments.downstream --compare``.  A ``--tasks``
subset updates those tasks in the file and keeps the rest; a ``--limit`` run
is a smoke and writes nothing.  The queue runs this module as the phase after
a finished training schedule.

``--baseline MODEL`` adds a yardstick: a published Hub causal LM scored on the
same documents by the workspace's reference scorer in FP32, the numerics its
published numbers use.  Its results do not depend on the run, so they are
scored once into ``figures/baseline/ORG/NAME.json`` and reused; each mode then
prints its paired comparison and one ``downstream`` record carrying ``against``
and the pooled accuracy difference.  Accuracy, whole-continuation
log-probability, and LAMBADA's per-word perplexity compare across tokenizers.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformer_experiments import downstream, runs, telemetry

from . import analysis
from .tokenizer import TOKENIZER_ID, load_tokenizer, tokenizer_metadata

MODES = ("standard", "fused", "soft")
FIGURES = Path("figures")
HUB_ID = re.compile(r"[A-Za-z0-9][\w.-]*(?:/[A-Za-z0-9][\w.-]*)?")


class DeltaScorer:
    """Continuation scores from a DeltaModel column pass under the trainer's numerics."""

    def __init__(self, model, mode: str, passes: int = 1):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != "standard" and not model.cfg.feedback:
            raise ValueError(f"{mode} mode needs a snapshot of a condition with f")
        if passes < 1:
            raise ValueError("passes must be positive")
        self.model = model
        self.mode = mode
        self.passes = passes if mode != "standard" else 0
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def score(self, ids, spans):
        model = self.model
        ids = ids.to(self.device)
        with analysis.autocast(self.device):
            e = model.embed_tokens(ids)
            # Under v every pass runs the evaluation column count and
            # scores its last column.
            out = model.forward_iterations(
                model.plain_seed(e), e, need_payload=self.passes > 0
            )[-1]
            if self.mode == "soft":
                # Plain context, feedback along the scored continuation only.
                prefix = torch.tensor([start for start, _ in spans], device=self.device)
            else:
                prefix = 1
            for i in range(self.passes):
                out = model.forward_iterations(
                    analysis.fused_inputs(model, e, out.payload, prefix),
                    e,
                    need_payload=i < self.passes - 1,
                )[-1]
            weight = model.embed_tokens.weight
            head = lambda h: F.linear(model.readout_input(h), weight.to(h.dtype))
            return downstream.span_scores(out.h_top, ids, spans, head)


def condition_modes(model) -> tuple[str, ...]:
    """The scoring modes a snapshot's condition supports."""
    return MODES if model.cfg.feedback else MODES[:1]


def result_path(tag: str, step: int, mode: str, passes: int = 1) -> Path:
    """Where one mode's results for one snapshot live under ``figures/``."""
    suffix = f"{mode}{passes}" if mode != "standard" and passes > 1 else mode
    return FIGURES / f"downstream-{tag}" / f"{suffix}.{step}.json"


def stored_results(path: Path, max_len: int) -> tuple[dict[str, downstream.TaskResult], dict]:
    """A result file's still-current tasks, in suite order, and its meta.

    A task is current while the pinned dataset revision and ``max_len`` it was
    scored under are the ones in force; scores depend on nothing else a rerun
    could change, so a partial rerun updates its tasks and keeps the rest.
    """
    if not path.is_file():
        return {}, {}
    payload = json.loads(path.read_text())
    found = downstream.from_json(payload)
    return {
        name: found[name]
        for name, task in downstream.TASKS.items()
        if name in found
        and found[name].meta.get("revision") == task.revision
        and found[name].meta.get("max_len") == max_len
    }, payload["meta"]


def write_results(path: Path, results: dict[str, downstream.TaskResult], meta: dict) -> None:
    ordered = {name: results[name] for name in downstream.TASKS if name in results}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(downstream.to_json(ordered, **meta), indent=1) + "\n")
    print(f"wrote {path}", flush=True)


def baseline_path(model_id: str) -> Path:
    """Where a published model's results live: ``figures/baseline/ORG/NAME.json``."""
    if not HUB_ID.fullmatch(model_id):
        raise ValueError(f"--baseline takes Hub model ids such as ORG/NAME, got {model_id!r}")
    return FIGURES / "baseline" / f"{model_id}.json"


def hub_id(text: str) -> str:
    try:
        baseline_path(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    return text


def load_baseline(model_id: str, device: torch.device):
    """A published causal LM under the workspace's reference scorer, in FP32.

    Returns its tokenizer callable, scorer, padding id, and resolved Hub commit.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    return (
        downstream.hf_tokenize(tokenizer),
        downstream.HFScorer(model, device),
        tokenizer.pad_token_id or 0,
        getattr(model.config, "_commit_hash", None),
    )


def baseline_results(
    model_id: str,
    tasks: list[str],
    requests: dict[str, list[downstream.Request]],
    device: torch.device,
    *,
    limit: int | None,
    max_len: int,
    **run_kwargs,
) -> dict[str, downstream.TaskResult]:
    """A published model's results on ``tasks``, scored once and then reused.

    They do not depend on the run, so they are kept beside no tag. A stored
    task is reused while its pinned dataset revision and ``max_len`` match;
    only missing tasks load the model, and a model whose Hub commit has moved
    is rescored whole. A ``--limit`` smoke neither reads nor writes the store.
    """
    path = baseline_path(model_id)
    stored, meta = stored_results(path, max_len) if limit is None else ({}, {})
    commit = meta.get("commit")
    missing = [name for name in tasks if name not in stored]
    if missing:
        tokenize, scorer, pad_id, resolved = load_baseline(model_id, device)
        if resolved != commit:
            stored, missing, commit = {}, list(tasks), resolved
        for name in missing:
            stored[name] = downstream.run_task(
                downstream.TASKS[name],
                requests[name],
                tokenize,
                scorer,
                max_len=max_len,
                pad_id=pad_id,
                **run_kwargs,
            )
        if limit is None:
            meta = {"model": model_id, "commit": commit, "dtype": "float32"}
            write_results(path, stored, meta)
    return {name: stored[name] for name in tasks}


def task_fields(result: downstream.TaskResult) -> dict[str, str]:
    """One task's headline numbers as telemetry fields."""
    fields = {"n": str(result.n)}
    for name, metric in result.metrics.items():
        key = "ppl" if name == "perplexity" else name
        fields[key] = telemetry.format_metric(metric["mean"])
    return fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "delta eval",
        description="Score a run's snapshot on the downstream zero-shot tasks.",
    )
    parser.add_argument("tag")
    parser.add_argument(
        "--mode",
        dest="modes",
        nargs="+",
        choices=MODES,
        metavar="MODE",
        default=None,
        help=f"scoring modes from {', '.join(MODES)} (default: every mode the "
        "snapshot's condition supports)",
    )
    parser.add_argument(
        "--step", type=int, default=None, help="snapshot step (default: the latest)"
    )
    parser.add_argument(
        "--passes", type=int, default=1, help="feedback passes in fused or soft mode"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(downstream.TASKS),
        metavar="TASK",
        default=list(downstream.DEFAULT_TASKS),
        help=f"tasks from {', '.join(downstream.TASKS)} (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="documents per task; a smoke that writes no results",
    )
    parser.add_argument(
        "--baseline",
        dest="baselines",
        nargs="+",
        type=hub_id,
        metavar="MODEL",
        default=[],
        help="published Hub models to compare against, paired by document; each "
        "is scored once in FP32 and kept under figures/baseline/",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--buckets", type=int, nargs="+", default=(128, 256, 512, 1024))
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default="runs", help="snapshot directory")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.step is None:
        path = runs.latest_snapshot(args.tag, args.out_dir)
    else:
        path = runs.snapshot_path(args.tag, args.step, args.out_dir)
        if not path.is_file():
            raise FileNotFoundError(f"no snapshot of {args.tag} at step {args.step}: {path}")

    model, saved = analysis.load_checkpoint(path, args.device)
    if saved["tokenizer_id"] != TOKENIZER_ID:
        raise ValueError("downstream text evaluation requires a NeoX ChatML snapshot")
    scorers = [
        DeltaScorer(model, mode, args.passes)
        for mode in args.modes or condition_modes(model)
    ]
    tokenize = downstream.hf_tokenize(load_tokenizer())
    max_len = min(saved["seq_len"], max(args.buckets))
    buckets = [b for b in args.buckets if b <= max_len]
    address = telemetry.step_address(saved["step"], saved["steps"])
    requests: dict[str, list[downstream.Request]] = {}
    started = time.time()

    def progress(line: str) -> None:
        print(f"  [{time.time() - started:6.0f}s] {line}", flush=True)

    scored: dict[str, dict[str, downstream.TaskResult]] = {}
    for scorer in scorers:
        print(f"{telemetry.HEADING}{scorer.mode}: {path}", flush=True)
        results = scored[scorer.mode] = {}
        for name in args.tasks:
            task = downstream.TASKS[name]
            if name not in requests:
                requests[name] = downstream.load_requests(task, args.limit)
            results[name] = downstream.run_task(
                task,
                requests[name],
                tokenize,
                scorer,
                batch_size=args.batch_size,
                max_len=max_len,
                buckets=buckets,
                progress=progress,
            )
            telemetry.log(
                "downstream",
                step=address,
                mode=scorer.mode,
                passes=scorer.passes,
                task=name,
                **task_fields(results[name]),
            )
        print("\n" + downstream.format_table(results), flush=True)
        if args.limit is not None:
            print(f"--limit {args.limit} is a smoke; nothing written", flush=True)
            continue
        out_path = result_path(saved["tag"], saved["step"], scorer.mode, args.passes)
        meta = {
            "snapshot": str(path),
            "tag": saved["tag"],
            "condition": saved["condition"],
            "step": saved["step"],
            "mode": scorer.mode,
            "passes": scorer.passes,
            **tokenizer_metadata(),
            "tokenizer_id": TOKENIZER_ID,
            "device": str(scorer.device),
            "batch_size": args.batch_size,
            "buckets": list(args.buckets),
        }
        write_results(out_path, stored_results(out_path, max_len)[0] | results, meta)

    if not args.baselines:
        return
    # Baselines follow the snapshot's own results, and take its memory.
    device, passes = scorers[0].device, {s.mode: s.passes for s in scorers}
    del model, scorers, scorer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for model_id in args.baselines:
        print(f"{telemetry.HEADING}baseline: {model_id}", flush=True)
        reference = baseline_results(
            model_id,
            args.tasks,
            requests,
            device,
            limit=args.limit,
            max_len=max_len,
            batch_size=args.batch_size,
            buckets=buckets,
            progress=progress,
        )
        print("\n" + downstream.format_table(reference), flush=True)
        label = model_id.rsplit("/", 1)[-1]
        for mode, results in scored.items():
            comparison = downstream.compare(reference, results)
            print("\n" + downstream.format_comparison(comparison, label, mode), flush=True)
            pool = downstream.pooled(comparison)
            telemetry.log(
                "downstream",
                step=address,
                mode=mode,
                passes=passes[mode],
                against=model_id,
                diff=telemetry.format_metric(pool["diff"]),
                se=telemetry.format_metric(pool["se"]),
                z=f"{pool['z']:+.1f}",
                positive=f"{pool['positive']}/{pool['count']}",
            )


if __name__ == "__main__":
    main()

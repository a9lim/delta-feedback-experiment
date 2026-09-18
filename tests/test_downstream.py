"""Downstream scoring modes, the ``delta eval`` lifecycle, and its queue phase."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from transformer_experiments import checkpoints, spool
from transformer_experiments import downstream as tasks

from delta_feedback_experiment import analysis, cli, downstream
from delta_feedback_experiment.model import DeltaModel, condition_config
from delta_feedback_experiment.optim import OptimizerPair, build_optimizers
from delta_feedback_experiment.tokenizer import SYNTHETIC_TOKENIZER_ID, TOKENIZER_ID
from delta_feedback_experiment.train import CONTRACT

TINY = {
    "vocab_size": 97,
    "dim": 16,
    "layers": 4,
    "heads": 2,
    "kv_heads": 2,
    "head_dim": 8,
    "expert_intermediate": 8,
    "num_routed_experts": 3,
    "experts_per_token": 2,
    "pkda_heads": 2,
    "pkda_head_dim": 8,
    "pkda_conv_size": 4,
    "loop_iterations": 2,
}


def tiny(condition="f", seed=3):
    torch.manual_seed(seed)
    return DeltaModel(condition_config(condition, max_seq_len=9, **TINY)).eval()


def snapshot(root, condition="f", tokenizer_id=TOKENIZER_ID):
    model = tiny(condition)
    args = SimpleNamespace(
        condition=condition,
        seed=3,
        seq_len=8,
        steps=5,
        tag="tiny",
        tokenizer_id=tokenizer_id,
        **TINY,
    )
    pair = OptimizerPair(build_optimizers(model))
    path = root / "runs" / "tiny.pt.5"
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoints.write_payload(path, checkpoints.payload(CONTRACT, model, pair, args, 5))
    return path


def direct(model, h_top, ids, spans):
    ce = analysis.token_ce(model, h_top[:, :-1], ids[:, 1:])
    return [-ce[row, start - 1 : end - 1].sum().item() for row, (start, end) in enumerate(spans)]


IDS = torch.randint(1, 97, (2, 8), generator=torch.Generator().manual_seed(4))


def test_standard_and_fused_scores_match_the_analysis_passes():
    model = tiny()
    spans = [(3, 6), (5, 8)]
    with torch.no_grad():
        e = model.embed_tokens(IDS)
        plain = model.forward_iterations(model.plain_seed(e), need_payload=True)[-1]
        fused = model.forward_iterations(
            analysis.fused_inputs(model, e, plain.payload, 1), need_payload=False
        )[-1]
    for mode, out in (("standard", plain), ("fused", fused)):
        got = downstream.DeltaScorer(model, mode).score(IDS, spans)
        want = direct(model, out.h_top, IDS, spans)
        assert [score for score, _ in got] == pytest.approx(want, rel=1e-5), mode
    assert direct(model, plain.h_top, IDS, spans) != pytest.approx(
        direct(model, fused.h_top, IDS, spans), rel=1e-5
    )


def test_soft_feedback_starts_after_the_first_continuation_token():
    model = tiny()
    standard = downstream.DeltaScorer(model, "standard")
    soft = downstream.DeltaScorer(model, "soft")
    scores = lambda scorer, spans: [score for score, _ in scorer.score(IDS, spans)]
    first = [(3, 4), (5, 6)]
    assert scores(soft, first) == pytest.approx(scores(standard, first), rel=1e-5)
    longer = [(3, 6), (5, 8)]
    assert scores(soft, longer) != pytest.approx(scores(standard, longer), rel=1e-5)


def test_modes_follow_the_condition():
    assert downstream.condition_modes(tiny("f")) == downstream.MODES
    looped = tiny("l")
    assert downstream.condition_modes(looped) == ("standard",)
    with pytest.raises(ValueError, match="condition with f"):
        downstream.DeltaScorer(looped, "fused")


@pytest.fixture
def suite(tmp_path, monkeypatch):
    """A two-document task and a character tokenizer, under a scratch root."""
    monkeypatch.chdir(tmp_path)
    request = tasks.Request(("ab", "ab"), ("c", "d"), 0, ("c", "d"))
    monkeypatch.setattr(
        tasks, "load_requests", lambda task, limit=None: [request, request][:limit]
    )
    encode = lambda text, add_special_tokens: SimpleNamespace(
        input_ids=[1 + ord(c) % 96 for c in text]
    )
    monkeypatch.setattr(downstream, "load_tokenizer", lambda: encode)
    return tmp_path


def test_eval_scores_every_mode_of_the_latest_snapshot(suite, capsys):
    snapshot(suite)
    downstream.main(["tiny", "--tasks", "piqa", "--device", "cpu"])
    written = sorted(p.name for p in (suite / "figures/downstream-tiny").iterdir())
    assert written == ["fused.5.json", "soft.5.json", "standard.5.json"]
    payload = json.loads((suite / "figures/downstream-tiny/soft.5.json").read_text())
    assert payload["meta"]["mode"] == "soft" and payload["meta"]["step"] == 5
    assert tasks.from_json(payload)["piqa"].n == 2
    records = [
        record
        for line in capsys.readouterr().out.splitlines()
        if (record := spool.telemetry.parse_record(line)) and record["event"] == "downstream"
    ]
    assert [record["mode"] for record in records] == ["standard", "fused", "soft"]
    assert all(record["step"] == "5/5" and record["n"] == "2" for record in records)


def test_eval_limit_is_a_smoke_and_passes_name_their_file(suite):
    snapshot(suite)
    common = ["tiny", "--tasks", "piqa", "--device", "cpu", "--mode", "fused"]
    downstream.main([*common, "--limit", "1"])
    assert not (suite / "figures").exists()
    downstream.main([*common, "--passes", "2", "--step", "5"])
    assert [p.name for p in (suite / "figures/downstream-tiny").iterdir()] == [
        "fused2.5.json"
    ]


def test_eval_refuses_missing_steps_and_synthetic_vocabularies(suite):
    snapshot(suite, tokenizer_id=SYNTHETIC_TOKENIZER_ID)
    with pytest.raises(FileNotFoundError, match="step 4"):
        downstream.main(["tiny", "--step", "4"])
    with pytest.raises(ValueError, match="NeoX"):
        downstream.main(["tiny", "--device", "cpu"])


class Published:
    """Stands in for a Hub model: scores by span length, counts its loads."""

    def __init__(self, monkeypatch, commit="c1"):
        self.commit, self.loads, self.scored = commit, 0, []
        monkeypatch.setattr(downstream, "load_baseline", self.load)

    def load(self, model_id, device):
        self.loads += 1
        tokenize = lambda text: [1 + ord(c) % 96 for c in text]
        return tokenize, self, 0, self.commit

    def score(self, ids, spans):
        self.scored.append(len(spans))
        return [(-float(end - start + row), row % 2 == 0) for row, (start, end) in enumerate(spans)]


EVAL = ["tiny", "--device", "cpu", "--mode", "standard", "--baseline", "org/ref"]


def test_baseline_is_scored_once_and_compared_by_document(suite, monkeypatch, capsys):
    snapshot(suite)
    published = Published(monkeypatch)
    downstream.main([*EVAL, "--tasks", "piqa"])
    stored = suite / "figures/baseline/org/ref.json"
    payload = json.loads(stored.read_text())
    assert payload["meta"] == {"model": "org/ref", "commit": "c1", "dtype": "float32"}
    assert list(payload["tasks"]) == ["piqa"] and published.loads == 1
    out = capsys.readouterr().out
    assert "pooled" in out
    against = [
        record
        for line in out.splitlines()
        if (record := spool.telemetry.parse_record(line)) and "against" in record
    ]
    assert [(r["mode"], r["against"], r["step"]) for r in against] == [
        ("standard", "org/ref", "5/5")
    ]

    downstream.main([*EVAL, "--tasks", "piqa"])
    assert published.loads == 1  # the stored task needs no model
    downstream.main([*EVAL, "--tasks", "piqa", "sciq"])
    assert published.loads == 2 and len(published.scored) == 2  # only sciq is new
    assert list(json.loads(stored.read_text())["tasks"]) == ["piqa", "sciq"]

    published.commit = "c2"
    downstream.main([*EVAL, "--tasks", "piqa", "arc_easy"])
    payload = json.loads(stored.read_text())
    assert payload["meta"]["commit"] == "c2"  # a moved model is rescored whole
    assert list(payload["tasks"]) == ["piqa", "arc_easy"]


def test_baseline_smoke_writes_nothing_and_ids_stay_under_figures(suite, monkeypatch):
    snapshot(suite)
    published = Published(monkeypatch)
    downstream.main([*EVAL, "--tasks", "piqa", "--limit", "1"])
    assert published.loads == 1 and not (suite / "figures").exists()
    for bad in ("/abs/model", "../escape", "org/name/extra", ".hidden"):
        with pytest.raises(SystemExit):
            downstream.build_parser().parse_args(["tiny", "--baseline", bad])


@pytest.fixture
def operator(tmp_path, monkeypatch):
    operator = spool.Spool(replace(cli.LAYOUT, root=tmp_path), cli.PIPELINE, prog="delta")
    operator.ensure_dirs()
    monkeypatch.setattr(cli, "SPOOL", operator)
    return operator


def test_queue_evaluates_only_a_finished_schedule(operator):
    job = operator.split_tokens(["run", "--out-dir", "elsewhere"], separator="--")
    log = operator.layout.logs / "run.log"
    assert cli._eval_skip(job) is not None
    log.write_text("step | step=3/5\nyield | step=3/5\n")
    assert "not finished" in cli._eval_skip(job)
    log.write_text("yield | step=3/5\nresume | step=3/5\ndone | step=5/5\n")
    assert cli._eval_skip(job) is None
    log.write_text("done | step=5/5\ninterrupt | step=5/5\n")
    assert cli._eval_skip(job) is not None
    assert cli._eval_argv(job) == ["run", "--out-dir", "elsewhere"]


def test_queue_eval_slot_carries_flags_or_opts_out(operator):
    job = operator.parse_line("run --seed 2 | --mode standard --tasks piqa")
    cli._validate_job(job)
    assert cli._eval_argv(job) == [
        "run", "--out-dir", "runs", "--mode", "standard", "--tasks", "piqa",
    ]
    skipped = operator.split_tokens(["run", "--", cli.EVAL_SKIP], separator="--")
    cli._validate_job(skipped)
    (operator.layout.logs / "run.log").write_text("done | step=5/5\n")
    assert cli._eval_skip(skipped) == "skipped by --skip"
    with pytest.raises(ValueError, match="alone"):
        cli._validate_job(operator.parse_line("run | --skip --mode soft"))
    with pytest.raises(SystemExit):
        cli._validate_job(operator.parse_line("run | --mode plain"))
    assert operator.pipeline.logs("run") == ["run.eval.log", "run.log"]

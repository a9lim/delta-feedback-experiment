"""Run renaming preserves trajectories and refuses collisions or live writers."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformer_experiments import checkpoints, runs, spool

from delta_feedback_experiment import cli
from delta_feedback_experiment.train import CONTRACT


@pytest.fixture
def operator(tmp_path, monkeypatch):
    layout = replace(cli.LAYOUT, root=tmp_path)
    operator = spool.Spool(layout, cli.PIPELINE, prog="delta")
    operator.ensure_dirs()
    monkeypatch.setattr(cli, "SPOOL", operator)
    return operator


def write(root, name, content="artifact"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_move_keeps_checkpoint_bytes_and_updates_run_and_analysis_identity(operator):
    root = operator.layout.root
    old, new = "screen-delta-r-s1", "screen-delta-r-s1-nope"
    source = root / "runs" / f"{old}.pt.2858"
    torch.save(
        {
            "version": CONTRACT.version,
            "args": {"tag": old},
            "state": {"weight": torch.ones(2)},
            "step": 2858,
        },
        source,
    )
    original = source.read_bytes()
    log = write(
        root,
        f"logs/{old}.log",
        f"run | tag={old}\ncheckpoint | path=runs/{old}.pt.2858\n"
        f"checkpoint | path=runs/{old}.pt.100\n",
    )
    mtime = log.stat().st_mtime_ns
    write(root, f"logs/{old}.probe.log", f"probe for {old}\n")
    write(
        root,
        "logs/continued.log",
        f"continue | source={old} | path=runs/{old}.pt.100\n",
    )
    write(root, "logs/queue-STATUS", f"2000-01-01 {old} STOPPED\n")
    write(root, "logs/queue.log", f"historical output {old}\n")
    dirs = [
        f"route-{old}",
        f"fused-{old}",
        f"depth-{old}",
        f"downstream-{old}",
        f"compare-{old}-vs-other",
        f"weights-other-vs-{old}",
        f"curves-other-vs-{old}-vs-third",
        f"curves-{old}-vs-{old}-vs-{old}-vs-{old}-vs-other",
        f"compare-{old}-vs-{old}",
    ]
    for directory in dirs:
        write(
            root,
            f"figures/{directory}/report.json",
            json.dumps(
                {
                    "tag": old,
                    "checkpoint": str(source),
                    f"{old}:weight": 1,
                    "output": f"figures/{directory}/plot.png",
                }
            ),
        )
        write(root, f"figures/{directory}/plot.png", "unchanged image")
    untouched = [
        f"runs/{old}-extra.pt.5",
        f"runs/{old}.pt.other.pt.5",
        f"logs/{old}-extra.log",
        f"figures/fused-{old}-extra/report.json",
    ]
    for name in untouched:
        write(root, name)

    cli.move_command([old, new])

    renamed = runs.latest_snapshot(new, root / "runs")
    assert renamed.read_bytes() == original
    assert checkpoints.read(renamed, CONTRACT, map_location="cpu")["args"]["tag"] == new
    assert runs.snapshots(old, root / "runs") == []
    assert not source.exists()
    assert (root / "logs" / f"{new}.log").stat().st_mtime_ns == mtime
    assert f"path=runs/{new}.pt.100" in (root / "logs" / f"{new}.log").read_text()
    assert f"source={new}" in (root / "logs/continued.log").read_text()
    assert f"{new} STOPPED" in operator.layout.marker.read_text()
    assert operator.layout.log.read_text() == f"historical output {old}\n"
    for directory in dirs:
        target = root / "figures" / directory.replace(old, new)
        report = json.loads((target / "report.json").read_text())
        assert report["tag"] == new
        assert report["checkpoint"] == str(renamed)
        assert report[f"{new}:weight"] == 1
        assert report["output"] == str(target.relative_to(root) / "plot.png")
        assert (target / "plot.png").read_text() == "unchanged image"
        assert sorted(p.name for p in target.iterdir()) == ["plot.png", "report.json"]
    assert all((root / name).read_text() == "artifact" for name in untouched)


@pytest.mark.parametrize(
    "occupied",
    ["runs/new.pt.999", "logs/new.probe.log", "figures/route-new/report.json"],
)
def test_move_preflights_destination_before_changing_any_artifact(operator, occupied):
    root = operator.layout.root
    source = write(root, "logs/old.log", "run | tag=old\n")
    write(root, "runs/old.pt.10")
    write(root, occupied)
    with pytest.raises(SystemExit):
        cli.move_command(["old", "new"])
    assert source.read_text() == "run | tag=old\n"
    assert (root / "runs/old.pt.10").exists()


@pytest.mark.parametrize("tag", ["old", "new"])
@pytest.mark.parametrize("state", ["pending", "active"])
def test_move_refuses_managed_tags(operator, tag, state):
    write(operator.layout.root, "runs/old.pt.10")
    payload = spool.Job(tag, ((),)).to_json()
    path = (
        operator.layout.active
        if state == "active"
        else operator.layout.queue_dir / "1.json"
    )
    operator.write_json(path, payload)
    with pytest.raises(SystemExit):
        cli.move_command(["old", "new"])
    assert path.exists()
    assert (operator.layout.snapshots / "old.pt.10").exists()


def test_move_refuses_pending_continuations(operator):
    write(operator.layout.root, "runs/old.pt.10")
    operator.write_json(
        operator.layout.queue_dir / "1.json",
        spool.Job("later", (("--continue", "old"),)).to_json(),
    )
    with pytest.raises(SystemExit):
        cli.move_command(["old", "new"])
    assert operator.pending_jobs()[0].argv == (("--continue", "old"),)


def test_move_refuses_a_direct_training_lock(operator):
    write(operator.layout.root, "runs/old.pt.10")
    with runs.lock_tags(operator.layout.snapshots, "old"), pytest.raises(SystemExit):
        cli.move_command(["old", "new"])
    cli.move_command(["old", "new"])


def test_move_allows_an_unrelated_active_job_and_leaves_its_log_alone(operator):
    root = operator.layout.root
    write(root, "runs/old.pt.10")
    write(root, "logs/old.log", "run | tag=old\n")
    active_log = write(root, "logs/live-run.log", "run | tag=live-run\n")
    inode = active_log.stat().st_ino
    operator.write_json(operator.layout.active, spool.Job("live-run", ((),)).to_json())
    cli.move_command(["old", "new"])
    assert active_log.stat().st_ino == inode
    assert operator.active_payload()["tag"] == "live-run"


def test_move_rolls_back_text_and_renames_after_an_io_failure(operator, monkeypatch):
    root = operator.layout.root
    names = ["logs/old.log", "runs/old.pt.10", "figures/fused-old/report.json"]
    for name in names:
        write(root, name, '{"tag":"old"}\n')
    original_rename = Path.rename

    def fail_snapshot(path, target):
        if path.name == "old.pt.10":
            raise OSError("injected rename failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_snapshot)
    with pytest.raises(SystemExit):
        cli.move_command(["old", "new"])
    assert all((root / name).read_text() == '{"tag":"old"}\n' for name in names)
    assert sorted(p.name for p in (root / "figures/fused-old").iterdir()) == [
        "report.json"
    ]
    assert not [p for p in root.rglob("*new*") if p.suffix != ".lock"]


def test_short_tag_does_not_rewrite_conditions_or_data_sources(operator):
    root = operator.layout.root
    write(root, "runs/a.pt.10")
    write(root, "logs/a.log", "run | tag=a | condition=a | source=a\n")
    write(
        root,
        "figures/route-a/report.json",
        json.dumps(
            {
                "tag": "a",
                "condition": "a",
                "source": "a",
                "labels": {"a": "a pass 1"},
                "checkpoint": "runs/a.pt.10",
            }
        ),
    )
    cli.move_command(["a", "new"])
    assert (
        root / "logs/new.log"
    ).read_text() == "run | tag=new | condition=a | source=a\n"
    report = json.loads((root / "figures/route-new/report.json").read_text())
    assert report["condition"] == report["source"] == "a"
    assert report["tag"] == "new"
    assert report["labels"] == {"a": "new pass 1"}


def test_move_preserves_longer_snapshot_shaped_tag_references(operator):
    root = operator.layout.root
    write(root, "runs/old.pt.10")
    write(root, "logs/other.log", "checkpoint | path=runs/old.pt.10.pt.99\n")
    cli.move_command(["old", "new"])
    assert (
        root / "logs/other.log"
    ).read_text() == "checkpoint | path=runs/old.pt.10.pt.99\n"


def test_training_curves_labels_and_run_keys_are_renamed(operator):
    root = operator.layout.root
    write(root, "runs/a.pt.10")
    write(
        root,
        "figures/curves-a-vs-b/training_curves.json",
        json.dumps(
            {
                "reference": "a",
                "runs": {"a": {"condition": "a"}, "b": {"condition": "a"}},
            }
        ),
    )
    cli.move_command(["a", "new"])
    report = json.loads(
        (root / "figures/curves-new-vs-b/training_curves.json").read_text()
    )
    assert report == {
        "reference": "new",
        "runs": {"new": {"condition": "a"}, "b": {"condition": "a"}},
    }


def test_move_refuses_ambiguous_comparison_tag_boundaries(operator):
    root = operator.layout.root
    write(root, "runs/old.pt.10")
    write(root, "runs/other-vs-old.pt.10")
    report = write(root, "figures/compare-other-vs-old-vs-third/report.json")
    with pytest.raises(SystemExit):
        cli.move_command(["old", "new"])
    assert report.read_text() == "artifact"


@pytest.mark.parametrize(
    "args", [["absent", "new"], ["old", "old"], ["../old", "new"], ["old", "all"]]
)
def test_move_reports_invalid_requests(operator, args, capsys):
    with pytest.raises(SystemExit):
        cli.move_command(args)
    assert "delta move:" in capsys.readouterr().err


def test_move_supports_external_snapshot_directory(operator, tmp_path):
    external = tmp_path / "external"
    write(external, "old.pt.20")
    cli.move_command(["old", "new", "--out-dir", str(external)])
    assert (external / "new.pt.20").read_text() == "artifact"


def test_snapshot_lookup_does_not_match_a_longer_dotted_tag(tmp_path):
    write(tmp_path, "old.pt.10")
    write(tmp_path, "old.pt.other.pt.99")
    assert runs.snapshots("old", tmp_path) == [(10, tmp_path / "old.pt.10")]

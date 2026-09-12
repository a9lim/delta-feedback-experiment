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


def test_move_preserves_snapshot_and_updates_artifact_identity(operator):
    root = operator.layout.root
    source = root / "runs/old.pt.2"
    torch.save(
        {
            "version": CONTRACT.version,
            "args": {"tag": "old"},
            "state": {"weight": torch.ones(2)},
            "step": 2,
        },
        source,
    )
    original = source.read_bytes()
    write(root, "logs/old.log", "run | tag=old\ncheckpoint | path=runs/old.pt.2\n")
    write(root, "logs/continued.log", "continue | source=old | path=runs/old.pt.2\n")
    write(
        root,
        "figures/route-old/report.json",
        json.dumps({"tag": "old", "checkpoint": str(source)}),
    )
    write(root, "figures/route-old/plot.png", "image")
    write(root, "runs/old-extra.pt.2", "unrelated")

    cli.move_command(["old", "new"])

    renamed = runs.latest_snapshot("new", root / "runs")
    assert renamed.read_bytes() == original
    assert (
        checkpoints.read(renamed, CONTRACT, map_location="cpu")["args"]["tag"] == "new"
    )
    assert not source.exists()
    assert (root / "logs/new.log").read_text() == (
        "run | tag=new\ncheckpoint | path=runs/new.pt.2\n"
    )
    assert "source=new" in (root / "logs/continued.log").read_text()
    assert json.loads((root / "figures/route-new/report.json").read_text()) == {
        "tag": "new",
        "checkpoint": str(renamed),
    }
    assert (root / "figures/route-new/plot.png").read_text() == "image"
    assert (root / "runs/old-extra.pt.2").read_text() == "unrelated"


@pytest.mark.parametrize(
    "occupied",
    ["runs/new.pt.999", "logs/new.log", "figures/route-new/report.json"],
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


def test_move_refuses_a_direct_training_lock(operator):
    write(operator.layout.root, "runs/old.pt.10")
    with runs.lock_tags(operator.layout.snapshots, "old"), pytest.raises(SystemExit):
        cli.move_command(["old", "new"])
    cli.move_command(["old", "new"])


def test_move_rolls_back_text_and_renames_after_an_io_failure(operator, monkeypatch):
    root = operator.layout.root
    names = ["logs/old.log", "runs/old.pt.10", "figures/route-old/report.json"]
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
    assert sorted(p.name for p in (root / "figures/route-old").iterdir()) == [
        "report.json"
    ]
    assert not [p for p in root.rglob("*new*") if p.suffix != ".lock"]

"""A store cannot be retired while queued trainer flags still address it."""

import json
from types import SimpleNamespace

import pytest
import torch

from scripts import store_cutover


@pytest.fixture
def spool(tmp_path, monkeypatch):
    active = tmp_path / "active.json"
    pending = tmp_path / "pending.json"
    value = SimpleNamespace(
        layout=SimpleNamespace(root=tmp_path, active=active),
        pending_paths=lambda: [pending] if pending.exists() else [],
    )
    monkeypatch.setattr(store_cutover, "SPOOL", value)
    return active, pending


def test_named_store_readers_cover_relative_and_absolute_roots(tmp_path, spool):
    active, pending = spool
    active.write_text(
        json.dumps({"tag": "a", "argv": [["--data-root", "data", "--source", "dclm"]]})
    )
    pending.write_text(
        json.dumps(
            {
                "tag": "b",
                "argv": [[f"--data-root={tmp_path / 'data'}", "--source=dclm-100b"]],
            }
        )
    )
    assert store_cutover.busy([tmp_path / "data/dclm"]) == ["active reader: a"]
    assert store_cutover.busy([tmp_path / "data/dclm-100b"]) == ["pending reader: b"]
    assert store_cutover.busy([tmp_path / "data/fineweb-edu"]) == []


@pytest.mark.parametrize("mode", [["--resume"], ["--continue", "a"]])
@pytest.mark.parametrize(
    "override", [[], ["--data-root", "moved"], ["--source", "dclm-100b"]]
)
def test_reader_inherits_each_location_field_from_checkpoint(
    tmp_path, spool, mode, override
):
    active, _ = spool
    snapshots = tmp_path / "runs"
    snapshots.mkdir()
    torch.save({"args": {"data_root": "saved", "source": "dclm"}}, snapshots / "a.pt.4")
    active.write_text(json.dumps({"tag": "a", "argv": [[*mode, *override]]}))
    root = "moved" if "--data-root" in override else "saved"
    source = "dclm-100b" if "--source" in override else "dclm"
    assert store_cutover.busy([tmp_path / root / source]) == ["active reader: a"]
    assert store_cutover.busy([tmp_path / "unrelated/dclm"]) == []


def test_unknown_resume_location_blocks_retirement(tmp_path, spool):
    active, _ = spool
    active.write_text(json.dumps({"tag": "missing", "argv": [["--resume"]]}))
    assert store_cutover.busy([tmp_path / "data/dclm"]) == [
        "active reader unresolved: missing"
    ]

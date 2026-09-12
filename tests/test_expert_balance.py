"""The trainer updates persistent expert biases from completed logical steps."""

import torch

from delta_feedback_experiment import train as trainer
from delta_feedback_experiment.data import DEFAULT_SOURCE, write_synthetic
from delta_feedback_experiment.model import DeltaModel
from delta_feedback_experiment.optim import NorMuonH


def arguments(tmp_path, tag):
    return [
        tag,
        "--data-root",
        str(tmp_path / "data"),
        "--source",
        DEFAULT_SOURCE,
        "--out-dir",
        str(tmp_path / "runs"),
        "--condition",
        "efl",
        "--vocab-size",
        "31",
        "--dim",
        "16",
        "--layers",
        "12",
        "--heads",
        "2",
        "--kv-heads",
        "2",
        "--head-dim",
        "8",
        "--intermediate",
        "32",
        "--pkda-heads",
        "2",
        "--pkda-head-dim",
        "8",
        "--seq-len",
        "4",
        "--batch-rows",
        "4",
        "--micro-rows",
        "2",
        "--loop-iterations",
        "2",
        "--loop-max-iterations",
        "3",
        "--steps",
        "2",
        "--warmup-frac",
        "0",
        "--cooldown-frac",
        "0",
        "--feedback-start",
        "0",
        "--eval-every",
        "2",
        "--eval-rows",
        "2",
        "--snapshot-every",
        "1",
        "--seed",
        "7",
        "--data-seed",
        "11",
        "--device",
        "cpu",
    ]


def setup(tmp_path, monkeypatch):
    write_synthetic(
        tmp_path / "data" / DEFAULT_SOURCE, train_tokens=100, val_tokens=40, vocab=31
    )
    monkeypatch.setattr(trainer, "draw_passes", lambda args, step, total: step + 1)
    monkeypatch.setattr(trainer, "draw_iterations", lambda args, step, loop: step + 1)


def test_trainer_updates_bias_once_after_optimizers_with_complete_counts(
    tmp_path, monkeypatch
):
    setup(tmp_path, monkeypatch)
    pending, updates, events = [], [], []
    forward = DeltaModel.forward_column
    update_bias = DeltaModel.update_expert_bias
    matrix_step = NorMuonH.step
    nadam_step = torch.optim.NAdam.step

    def observe_forward(model, *args, **kwargs):
        before = torch.stack([block.mlp.expert_bias.clone() for block in model.blocks])
        out = forward(model, *args, **kwargs)
        if model.training and torch.is_grad_enabled():
            torch.testing.assert_close(
                torch.stack([block.mlp.expert_bias for block in model.blocks]),
                before,
                rtol=0,
                atol=0,
            )
            pending.append(out.expert_counts.clone())
        return out

    def observe_matrix_step(optimizer, *args, **kwargs):
        result = matrix_step(optimizer, *args, **kwargs)
        events.append("matrix")
        return result

    def observe_nadam_step(optimizer, *args, **kwargs):
        result = nadam_step(optimizer, *args, **kwargs)
        events.append("nadam")
        return result

    def observe_bias_update(model, counts, **kwargs):
        step = len(updates) + 1
        assert events == ["matrix", "nadam", "bias"] * (step - 1) + ["matrix", "nadam"]
        assert len(pending) == 2 * (step + 1)
        torch.testing.assert_close(counts, torch.stack(pending).sum(0), rtol=0, atol=0)
        per_layer = 3 * 4 * 4 * (step + 1)
        assert counts.sum(-1).tolist() == (
            [per_layer] * 4 + [per_layer * (step + 1)] * 4 + [per_layer] * 4
        )
        before = torch.stack([block.mlp.expert_bias.clone() for block in model.blocks])
        update_bias(model, counts, **kwargs)
        after = torch.stack([block.mlp.expert_bias.clone() for block in model.blocks])
        expected = before + 0.001 * (counts.sum(-1, keepdim=True) - 15 * counts).sign()
        torch.testing.assert_close(after, expected, rtol=0, atol=0)
        pending.clear()
        updates.append(after)
        events.append("bias")

    monkeypatch.setattr(DeltaModel, "forward_column", observe_forward)
    monkeypatch.setattr(NorMuonH, "step", observe_matrix_step)
    monkeypatch.setattr(torch.optim.NAdam, "step", observe_nadam_step)
    monkeypatch.setattr(DeltaModel, "update_expert_bias", observe_bias_update)
    summary = trainer.train(arguments(tmp_path, "counts"))
    assert summary["step"] == 2
    assert len(updates) == 2 and not pending
    snapshot = trainer.read_checkpoint(tmp_path / "runs" / "counts.pt.2")
    state = snapshot[trainer.CONTRACT.state_key]
    for layer in range(12):
        torch.testing.assert_close(
            state[f"blocks.{layer}.mlp.expert_bias"], updates[-1][layer], rtol=0, atol=0
        )


def test_resumed_expert_state_matches_uninterrupted_buffers_and_parameters(
    tmp_path, monkeypatch
):
    setup(tmp_path, monkeypatch)
    trainer.train(arguments(tmp_path, "full"))
    trainer.train(arguments(tmp_path, "resumed") + ["--max-steps", "1"])
    trainer.train(arguments(tmp_path, "resumed") + ["--resume"])
    full = trainer.read_checkpoint(tmp_path / "runs" / "full.pt.2")
    resumed = trainer.read_checkpoint(tmp_path / "runs" / "resumed.pt.2")
    full_state = full[trainer.CONTRACT.state_key]
    resumed_state = resumed[trainer.CONTRACT.state_key]
    assert full_state.keys() == resumed_state.keys()
    bias_names = [name for name in full_state if name.endswith("expert_bias")]
    assert len(bias_names) == 12
    assert any(torch.count_nonzero(full_state[name]) for name in bias_names)
    for name, value in full_state.items():
        torch.testing.assert_close(
            value,
            resumed_state[name],
            rtol=0,
            atol=0,
            msg=lambda message, name=name: f"{name}: {message}",
        )

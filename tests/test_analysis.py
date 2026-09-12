"""The analysis helpers reproduce the trainer's own numbers on a snapshot."""

from types import SimpleNamespace

import pytest
import torch
from transformer_experiments import checkpoints

from delta_feedback_experiment import analysis
from delta_feedback_experiment.model import (
    DeltaModel,
    condition_config,
    multipass,
    multipass_loss,
    sequence_ce,
)
from delta_feedback_experiment.optim import OptimizerPair, build_optimizers
from delta_feedback_experiment.tokenizer import SYNTHETIC_TOKENIZER_ID
from delta_feedback_experiment.train import CONTRACT

TINY = {
    "vocab_size": 97,
    "dim": 16,
    "layers": 4,
    "heads": 2,
    "kv_heads": 2,
    "head_dim": 8,
    "intermediate": 32,
    "pkda_heads": 2,
    "pkda_head_dim": 8,
    "pkda_conv_size": 4,
    "loop_iterations": 4,
    "loop_max_iterations": 8,
}


def snapshot(tmp_path, condition="f", seed=3):
    torch.manual_seed(seed)
    model = DeltaModel(condition_config(condition, max_seq_len=17, **TINY))
    args = SimpleNamespace(
        condition=condition,
        seed=seed,
        seq_len=16,
        tag="tiny",
        tokenizer_id=SYNTHETIC_TOKENIZER_ID,
        **TINY,
    )
    pair = OptimizerPair(build_optimizers(model))
    path = tmp_path / "tiny.pt.5"
    checkpoints.write_payload(path, checkpoints.payload(CONTRACT, model, pair, args, 5))
    return model.eval(), path


def test_load_checkpoint_rebuilds_the_saved_model(tmp_path):
    condition = "f"
    model, path = snapshot(tmp_path, condition)
    loaded, saved = analysis.load_checkpoint(path, "cpu")
    assert saved["condition"] == condition and saved["step"] == 5
    assert saved["seq_len"] == 16
    assert not loaded.training
    for (name, a), (_, b) in zip(
        model.state_dict().items(), loaded.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name
    tokens = torch.randint(0, 97, (2, 17), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        want = model.forward_column(model.embed_tokens(tokens[:, :-1])).h_top
        got = loaded.forward_column(loaded.embed_tokens(tokens[:, :-1])).h_top
    assert torch.equal(want, got)


def test_token_ce_matches_sequence_ce(tmp_path):
    model, _ = snapshot(tmp_path)
    tokens = torch.randint(0, 97, (2, 17), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        out = model.forward_column(model.embed_tokens(tokens[:, :-1]))
        per_token = analysis.token_ce(model, out.h_top, tokens[:, 1:], chunk=5)
        mean, _ = sequence_ce(model, out.h_top, tokens[:, 1:])
    assert per_token.shape == (2, 16)
    assert per_token.mean().item() == pytest.approx(mean.item(), rel=1e-5)


def test_fused_inputs_match_multipass(tmp_path):
    model, _ = snapshot(tmp_path)
    tokens = torch.randint(0, 97, (3, 17), generator=torch.Generator().manual_seed(2))
    prefix = torch.tensor([[1, 5, 15]])
    with torch.no_grad():
        outs = multipass(model, tokens, 2, prefix_lens=prefix)
        losses = multipass_loss(model, tokens, outs).ntp
        e = model.embed_tokens(tokens[:, :-1])
        first = model.forward_column(e, need_payload=True)
        x = analysis.fused_inputs(model, e, first.payload, prefix[0])
        second = model.forward_column(x, need_payload=False)
        fused_ce = analysis.token_ce(model, second.h_top, tokens[:, 1:]).mean()
    assert torch.equal(second.h_top, outs[1].h_top)
    assert fused_ce.item() == pytest.approx(losses[1].item(), rel=1e-5)
    assert analysis.plain_mask(16, 1, "cpu").squeeze(-1).sum() == 1

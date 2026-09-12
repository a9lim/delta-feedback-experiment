"""Text identity, chat structure, and stored vocabulary validation."""

import json

import pytest
import torch

from delta_feedback_experiment import tokenizer as profile
from delta_feedback_experiment.data import META, TokenData, read_meta, write_synthetic
from delta_feedback_experiment.train import CONTRACT, read_checkpoint


def test_chat_template_preserves_roles_and_whitespace():
    from jinja2 import Template

    messages = [
        {"role": "a9lim", "content": "  first\nsecond\t"},
        {"role": "critic_17", "content": ""},
        {"role": "a9lim", "content": "again"},
    ]
    expected = (
        "<|im_start|>a9lim\n  first\nsecond\t<|im_end|>\n"
        "<|im_start|>critic_17\n<|im_end|>\n"
        "<|im_start|>a9lim\nagain<|im_end|>\n"
    )
    template = Template(profile.CHAT_TEMPLATE)
    assert template.render(messages=messages) == expected
    assert template.render(messages=messages, add_generation_prompt=True) == (
        expected + "<|im_start|>self\n"
    )
    assert (
        template.render(
            messages=messages, add_generation_prompt=True, next_role="critic_17"
        )
        == expected + "<|im_start|>critic_17\n"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"tokenizer_id": "different-tokenizer"},
        {"chat_template": "discard roles"},
    ],
)
def test_store_readers_reject_incompatible_token_identity(tmp_path, change):
    write_synthetic(tmp_path, train_tokens=100, val_tokens=100)
    meta = read_meta(tmp_path)
    meta.update(profile.tokenizer_metadata(), tokenizer_id=profile.TOKENIZER_ID)
    (tmp_path / META).write_text(json.dumps(meta))
    TokenData.load(tmp_path, "val", 8)
    meta.update(change)
    (tmp_path / META).write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="tokenizer"):
        TokenData.load(tmp_path, "val", 8)


def test_checkpoint_requires_tokenizer_identity(tmp_path):
    path = tmp_path / "test.pt.1"
    payload = {
        "version": CONTRACT.version,
        "state": {},
        "args": {"tokenizer_id": "different-tokenizer"},
    }
    torch.save(payload, path)
    with pytest.raises(ValueError, match="tokenizer identity"):
        read_checkpoint(path)


def test_checkpoint_requires_current_version(tmp_path):
    version = CONTRACT.version + 1
    path = tmp_path / "test.pt.1"
    torch.save({"version": version, "state": {}, "args": {}}, path)
    with pytest.raises(ValueError, match="checkpoint version"):
        read_checkpoint(path)

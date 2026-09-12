"""Text identity, chat structure, and stored vocabulary validation."""

import json

import pytest
import torch

from delta_feedback_experiment import tokenizer as profile
from delta_feedback_experiment.data import META, TokenData, read_meta, write_synthetic
from delta_feedback_experiment.train import CONTRACT, read_checkpoint


@pytest.fixture(scope="module")
def tokenizer():
    # The offline suite never downloads. Qualify the pinned vocabulary after
    # loading it once on each machine; skip these text checks if it is absent.
    from transformers import AutoTokenizer

    original = AutoTokenizer.from_pretrained
    with pytest.MonkeyPatch.context() as patch:

        def cached(*args, **kwargs):
            return original(*args, **kwargs, local_files_only=True)

        patch.setattr(AutoTokenizer, "from_pretrained", cached)
        try:
            return profile.load_tokenizer()
        except OSError:
            pytest.skip("pinned NeoX tokenizer is not cached")


def test_chatml_preserves_arbitrary_and_repeated_roles(tokenizer):
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
    assert tokenizer.apply_chat_template(messages, tokenize=False) == expected
    ids = tokenizer.apply_chat_template(messages, return_dict=False)
    assert tokenizer.decode(ids, skip_special_tokens=False) == expected
    assert ids.count(profile.CHATML_START_ID) == 3
    assert ids.count(profile.CHATML_END_ID) == 3
    for role in ("self", "claude", "a9lim"):
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, next_role=role
        )
        assert rendered == expected + f"<|im_start|>{role}\n"
    assert (
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        == expected + "<|im_start|>self\n"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"tokenizer_id": "different-tokenizer"},
        {"tokenizer_revision": "another-revision"},
        {"chatml_end_id": 0},
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


@pytest.mark.parametrize("version", [CONTRACT.version - 1, CONTRACT.version + 1])
def test_checkpoint_requires_current_version(tmp_path, version):
    path = tmp_path / "test.pt.1"
    torch.save({"version": version, "state": {}, "args": {}}, path)
    with pytest.raises(ValueError, match="checkpoint version"):
        read_checkpoint(path)

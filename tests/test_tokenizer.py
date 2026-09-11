"""Text identity, chat structure, and vocabulary cutover boundaries."""

import json
import unicodedata

import pytest
import torch

from delta_feedback_experiment import tokenizer as profile
from delta_feedback_experiment.data import META, TokenData, read_meta, write_synthetic
from delta_feedback_experiment.train import parse_run_args, read_checkpoint


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


def test_vocabulary_and_specials_survive_serialization(tokenizer, tmp_path):
    from transformers import AutoTokenizer

    tokenizer.save_pretrained(tmp_path)
    restored = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    assert len(restored) == profile.TOKENIZER_VOCAB_SIZE == 50279
    assert restored.eos_token_id == 0
    assert restored.chat_template == profile.CHAT_TEMPLATE
    for text, token_id in ((profile.CHATML_START, 50277), (profile.CHATML_END, 50278)):
        assert restored.encode(text, add_special_tokens=False) == [token_id]
        assert token_id in restored.all_special_ids
        assert restored.decode([token_id], skip_special_tokens=True) == ""
    assert parse_run_args(["test"]).vocab_size == profile.VOCAB_SIZE == 50304


@pytest.mark.parametrize(
    "text", ["  hello\n\tworld", "a\x00b\x01c", "e\u0301 café", "\U00040000 🦦 中文"]
)
def test_text_roundtrip_obeys_neox_nfc_normalization(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.decode(ids) == unicodedata.normalize("NFC", text)
    assert 0 not in ids


@pytest.mark.parametrize(
    "change",
    [
        {"tokenizer_id": "old-qwen"},
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
    payload = {"version": 28, "state": {}, "args": {"tokenizer_id": "old-qwen"}}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="tokenizer identity"):
        read_checkpoint(path)


def test_corpus_build_uses_raw_text_and_document_eos(tokenizer, tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from delta_feedback_experiment import data

    source = tmp_path / "source" / "filtered"
    source.mkdir(parents=True)
    text = "A short English document. " * 12
    pq.write_table(pa.table({"text": [text] * 40}), source / "part.parquet")
    monkeypatch.setattr(data, "load_tokenizer", lambda: tokenizer)
    out = tmp_path / "store"
    meta = data.tokenize(
        out,
        source=data.LocalSource(source.parent),
        target_tokens=500,
        val_tokens=100,
        tokens_per_doc=1,
        check_packages=False,
    )
    assert meta["tokenizer_id"] == profile.TOKENIZER_ID
    assert meta["tokenizer_vocab_size"] == 50279
    assert meta["vocab_size"] == 50304
    assert meta["eos_id"] == 0
    data.verify(out)
    val = data.TokenData.load(out, "val", 1)
    ids = val.read(0, val.total_tokens).tolist()
    assert ids == tokenizer.encode(text, add_special_tokens=False) + [0]
    assert profile.CHATML_START_ID not in ids and profile.CHATML_END_ID not in ids

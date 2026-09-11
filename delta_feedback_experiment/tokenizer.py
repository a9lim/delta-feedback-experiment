"""The experiment's pinned text vocabulary and role-preserving ChatML format."""

from __future__ import annotations

import hashlib
import json

CANONICAL_TOKENIZER = "EleutherAI/gpt-neox-20b"
CANONICAL_TOKENIZER_REVISION = "c292233c833e336628618a88a648727eb3dff0a7"
EOS_ID = 0
CHATML_START = "<|im_start|>"
CHATML_END = "<|im_end|>"
CHATML_START_ID = 50277
CHATML_END_ID = 50278
TOKENIZER_VOCAB_SIZE = 50279
VOCAB_SIZE = 50304
"""Tied embedding/readout rows, padded to a multiple of 128."""

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>' + (next_role | default('self')) + '\n' }}"
    "{% endif %}"
)
"""Text messages retain every role and turn; no identity or alternation is imposed."""


def tokenizer_metadata() -> dict:
    """Complete identity of the pinned tokenizer plus our local additions."""
    return {
        "tokenizer": CANONICAL_TOKENIZER,
        "tokenizer_revision": CANONICAL_TOKENIZER_REVISION,
        "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
        "vocab_size": VOCAB_SIZE,
        "eos_id": EOS_ID,
        "chatml_start_id": CHATML_START_ID,
        "chatml_end_id": CHATML_END_ID,
        "chat_template": CHAT_TEMPLATE,
    }


TOKENIZER_ID = hashlib.sha256(
    json.dumps(tokenizer_metadata(), sort_keys=True).encode()
).hexdigest()
SYNTHETIC_TOKENIZER_ID = "synthetic-v1"
"""Explicit identity reserved for generated numerical test fixtures."""


def load_tokenizer():
    """Load vocabulary only, then register distinct ChatML turn delimiters.

    Corpus documents use the original EOS. ChatML is applied only when a
    caller explicitly uses ``apply_chat_template``; role names are plain text.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        CANONICAL_TOKENIZER, revision=CANONICAL_TOKENIZER_REVISION
    )
    if len(tokenizer) != CHATML_START_ID or tokenizer.eos_token_id != EOS_ID:
        raise ValueError("the pinned NeoX base vocabulary has changed")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [CHATML_START, CHATML_END]}
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    if (
        len(tokenizer) != TOKENIZER_VOCAB_SIZE
        or tokenizer.convert_tokens_to_ids(CHATML_START) != CHATML_START_ID
        or tokenizer.convert_tokens_to_ids(CHATML_END) != CHATML_END_ID
    ):
        raise ValueError("NeoX ChatML token IDs do not match the experiment contract")
    return tokenizer

"""Shared parameter ownership predicates for initialization and optimization."""

from torch import Tensor

_NADAM_MATRIX_MARKERS = (
    ".attn.control_proj.",
    ".attn.decay_up.",
    ".attn.output_gate_up.",
)


def is_normuonh_parameter(name: str, parameter: Tensor) -> bool:
    """Whether a trainable parameter belongs to the NorMuonH matrix group."""
    if not parameter.requires_grad or name == "embed_tokens.weight":
        return False
    nadam_matrix = (
        name.startswith("attention_gates.")
        or name == "fuse_gate.weight"
        or any(marker in name for marker in _NADAM_MATRIX_MARKERS)
    )
    return parameter.ndim == 2 and not nadam_matrix

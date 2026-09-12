"""Shared parameter ownership predicates for initialization and optimization."""

from torch import Tensor

_NADAM_MATRIX_MARKERS = (
    ".mlp.router.",
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


def normuonh_rate_name(name: str) -> str:
    """Separate expert input/output rates, including shared and MTP experts."""
    if ".mlp.shared." in name or ".mlp.experts." in name:
        if name.endswith(".gate_up_proj.weight"):
            return "normuonh_expert_in"
        if name.endswith(".down_proj.weight"):
            return "normuonh_expert_out"
    return "normuonh"


def is_width_scaled_parameter(name: str, parameter: Tensor) -> bool:
    """Whether an NAdam-owned matrix reads the full residual width.

    The GGQA gate matrices, the FBT token gate, and PKDA's packed control
    projection have fan-in ``D``, so their NAdam rate carries the muP width
    ratio and their initialization standard deviation its square root.
    Every other NAdam parameter has a fan-in that does not change
    across geometries: the tied embedding's lookup, the router's fixed group
    width, PKDA's head-width expansions, the depthwise convolutions, and the
    vectors. The tied embedding's readout role has fan-in ``D`` too and takes
    the ratio as a logit multiplier instead, which leaves its lookup rate
    alone.
    """
    if not parameter.requires_grad or parameter.ndim != 2:
        return False
    return (
        name.startswith("attention_gates.")
        or name == "fuse_gate.weight"
        or ".attn.control_proj." in name
        or ".mlp.router." in name
    )

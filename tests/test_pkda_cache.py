"""Cached PKDA projections preserve the full-row convolution precision."""

import pytest
import torch

from delta_feedback_experiment.model import DeltaModel, condition_config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PKDA convolution precision requires CUDA"
)


@pytest.mark.parametrize(
    "conv_size,prefix,batch", [(1, 1, 2), (2, 65, 1), (4, 1, 2), (4, 5, 2), (4, 65, 2)]
)
@torch.no_grad()
def test_cached_pkda_qkv_matches_full_row_and_projection_history(
    conv_size, prefix, batch
):
    torch.manual_seed(0)
    cfg = condition_config(
        "a",
        vocab_size=97,
        dim=128,
        layers=4,
        heads=4,
        kv_heads=2,
        head_dim=32,
        intermediate=256,
        pkda_heads=2,
        pkda_head_dim=128,
        pkda_conv_size=conv_size,
    )
    attn = DeltaModel(cfg).cuda().eval().blocks[0].attn
    x = torch.randn(batch, prefix + 3, cfg.dim, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        full = attn._project(x, None, False)[:3]
        pieces = [[] for _ in full]
        weights = torch.cat(
            [
                projection.weight
                for projection in (attn.q_proj, attn.k_proj, attn.v_proj)
            ]
        )
        raw_pieces = []
        history = None
        for start, end in [
            (0, prefix),
            *[(i, i + 1) for i in range(prefix, x.shape[1])],
        ]:
            q, k, v, history = attn._project(x[:, start:end], history, True)
            for track, value in zip(pieces, (q, k, v), strict=True):
                track.append(value)
            if conv_size > 1:
                # The cache stores projected inputs, before convolution/SiLU.
                # Match the GEMM partition: changing its row count can round a
                # BF16 projection differently even before convolution.
                raw_pieces.append(torch.nn.functional.linear(x[:, start:end], weights))
                raw_full = torch.cat(raw_pieces, dim=1)
                for stored, raw in zip(
                    history,
                    raw_full.split(attn.projection_size, dim=-1),
                    strict=True,
                ):
                    expected = torch.nn.functional.pad(
                        raw.transpose(1, 2), (conv_size - 1, 0)
                    )[:, :, -(conv_size - 1) :]
                    torch.testing.assert_close(stored, expected, rtol=0, atol=0)
                    assert (
                        stored.untyped_storage().nbytes()
                        == stored.numel() * stored.element_size()
                    )
            else:
                assert history is None

    for expected, parts in zip(full, pieces, strict=True):
        actual = torch.cat(parts, dim=1)
        error = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert error < 1e-4, error.item()

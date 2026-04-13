"""Tests for src/models/layers/schema_encoder.py and its integration with
EdgeTypedAttention via the type_emb_override plumbing."""

from __future__ import annotations

import os
import sys

import pytest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.layers import poincare_ops as po  # noqa: E402
from src.models.layers.edge_attention import EdgeTypedAttention  # noqa: E402
from src.models.layers.hyp_message_pass import HyperbolicMessagePassing  # noqa: E402
from src.models.layers.schema_encoder import SchemaEncoder  # noqa: E402


def _rand_ball(n, d, c=1.0, scale=0.3, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, d, generator=g, dtype=dtype) * scale
    return po.project(v, c)


# --------------------------------------------------------------------------- #
# shape and I/O correctness
# --------------------------------------------------------------------------- #
def test_edge_only_shape():
    enc = SchemaEncoder(edge_feat_dim=6, type_dim=8).double()
    desc = torch.randn(5, 6, dtype=torch.float64)
    edge_emb, node_emb = enc(desc)
    assert edge_emb.shape == (5, 8)
    assert node_emb is None


def test_edge_and_node_shape():
    enc = SchemaEncoder(edge_feat_dim=6, type_dim=8, node_feat_dim=4).double()
    e_desc = torch.randn(3, 6, dtype=torch.float64)
    n_desc = torch.randn(7, 4, dtype=torch.float64)
    edge_emb, node_emb = enc(e_desc, n_desc)
    assert edge_emb.shape == (3, 8)
    assert node_emb.shape == (7, 8)


def test_node_descriptor_without_node_mlp_errors():
    enc = SchemaEncoder(edge_feat_dim=4, type_dim=8).double()
    with pytest.raises(ValueError):
        enc(torch.randn(2, 4, dtype=torch.float64),
            torch.randn(2, 4, dtype=torch.float64))


# --------------------------------------------------------------------------- #
# schema-portability contract: output depends only on descriptor content, not
# on type-index position. Two types with identical descriptors must get
# identical embeddings.
# --------------------------------------------------------------------------- #
def test_identical_descriptors_produce_identical_embeddings():
    torch.manual_seed(0)
    enc = SchemaEncoder(edge_feat_dim=6, type_dim=8).double()
    d = torch.randn(6, dtype=torch.float64)
    # Types 0 and 3 have the same descriptor.
    desc = torch.stack([d, torch.randn(6, dtype=torch.float64),
                        torch.randn(6, dtype=torch.float64), d,
                        torch.randn(6, dtype=torch.float64)])
    edge_emb, _ = enc(desc)
    assert torch.allclose(edge_emb[0], edge_emb[3], atol=1e-12)


def test_different_descriptors_produce_different_embeddings():
    torch.manual_seed(1)
    enc = SchemaEncoder(edge_feat_dim=6, type_dim=8).double()
    desc = torch.randn(4, 6, dtype=torch.float64)
    edge_emb, _ = enc(desc)
    # Pairwise: no two rows should be identical.
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(edge_emb[i], edge_emb[j], atol=1e-6)


# --------------------------------------------------------------------------- #
# gradient flow
# --------------------------------------------------------------------------- #
def test_gradient_flow():
    torch.manual_seed(2)
    enc = SchemaEncoder(edge_feat_dim=6, type_dim=8, node_feat_dim=4).double()
    e_desc = torch.randn(5, 6, dtype=torch.float64, requires_grad=True)
    n_desc = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    edge_emb, node_emb = enc(e_desc, n_desc)
    (edge_emb.sum() + node_emb.sum()).backward()
    assert torch.isfinite(e_desc.grad).all() and (e_desc.grad.abs().sum() > 0)
    assert torch.isfinite(n_desc.grad).all() and (n_desc.grad.abs().sum() > 0)
    for p in enc.parameters():
        assert torch.isfinite(p.grad).all()


# --------------------------------------------------------------------------- #
# integration: schema-encoded embeddings feed EdgeTypedAttention via override
# and produce a valid attention output, which then feeds message passing.
# --------------------------------------------------------------------------- #
def test_end_to_end_with_override():
    torch.manual_seed(3)
    N, d, T, E = 15, 8, 4, 40
    c = 1.0
    edge_feat_dim = 6
    type_dim = 8

    x = _rand_ball(N, d, c=c)
    edge_index = torch.randint(0, N, (2, E))
    edge_type = torch.randint(0, T, (E,))
    edge_desc = torch.randn(T, edge_feat_dim, dtype=torch.float64)

    enc = SchemaEncoder(edge_feat_dim, type_dim).double()
    attn = EdgeTypedAttention(
        node_dim=d, num_edge_types=T, type_dim=type_dim, c=c
    ).double()
    mp = HyperbolicMessagePassing(d, d, c=c).double()

    edge_type_emb, _ = enc(edge_desc)
    alpha = attn(x, edge_index, edge_type, type_emb_override=edge_type_emb)
    out = mp(x, edge_index, edge_weight=alpha)

    assert alpha.shape == (E,)
    assert torch.isfinite(alpha).all()
    # softmax-normalized per receiver
    for r in torch.unique(edge_index[1]).tolist():
        mask = edge_index[1] == r
        assert torch.isclose(alpha[mask].sum(),
                             torch.tensor(1.0, dtype=alpha.dtype), atol=1e-10)
    assert out.shape == (N, d)
    assert torch.isfinite(out).all()


def test_override_shape_mismatch_errors():
    attn = EdgeTypedAttention(node_dim=8, num_edge_types=3, type_dim=8).double()
    x = _rand_ball(4, 8, c=1.0)
    edge_index = torch.tensor([[0, 1], [2, 3]])
    edge_type = torch.tensor([0, 1])
    wrong = torch.randn(3, 4, dtype=torch.float64)  # type_dim mismatch
    with pytest.raises(ValueError):
        attn(x, edge_index, edge_type, type_emb_override=wrong)


# --------------------------------------------------------------------------- #
# schema-portability in practice: same encoder, two different schemas
# (different T, different descriptors), both produce valid attention.
# This is the CLAUDE.md Commitment #3 contract — one trained model, two schemas.
# --------------------------------------------------------------------------- #
def test_cross_schema_reuse():
    torch.manual_seed(4)
    d, edge_feat_dim, type_dim = 8, 6, 8
    c = 1.0

    enc = SchemaEncoder(edge_feat_dim, type_dim).double()
    attn = EdgeTypedAttention(
        node_dim=d, num_edge_types=16, type_dim=type_dim, c=c
    ).double()  # num_edge_types is an upper bound; not used under override

    for T in (3, 7):
        N, E = 12, 30
        x = _rand_ball(N, d, c=c, seed=T)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, T, (E,))
        edge_desc = torch.randn(T, edge_feat_dim, dtype=torch.float64)

        edge_type_emb, _ = enc(edge_desc)
        alpha = attn(x, edge_index, edge_type, type_emb_override=edge_type_emb)
        assert torch.isfinite(alpha).all()
        assert alpha.shape == (E,)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

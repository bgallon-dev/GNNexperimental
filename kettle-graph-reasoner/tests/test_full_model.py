"""End-to-end assembly test for KettleGraphReasoner."""

from __future__ import annotations

import os
import sys

import pytest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.hyperbolic_gnn import KettleGraphReasoner  # noqa: E402
from src.modelsv2.hyperbolic_gnnV2 import (  # noqa: E402
    KettleGraphReasoner as KGRv2,
)
from src.modelsv2.layers import poincare_ops as Pv2  # noqa: E402


def _make_batch(N=20, E=60, node_feat=12, edge_feat=6, query=16, T=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    node_features = torch.randn(N, node_feat, generator=g, dtype=torch.float64)
    edge_index = torch.randint(0, N, (2, E), generator=g)
    edge_type = torch.randint(0, T, (E,), generator=g)
    edge_descriptor = torch.randn(T, edge_feat, generator=g, dtype=torch.float64)
    query_vec = torch.randn(query, generator=g, dtype=torch.float64)
    return node_features, edge_index, edge_type, edge_descriptor, query_vec


def _build(**overrides):
    cfg = dict(
        node_feat_dim=12,
        edge_feat_dim=6,
        query_dim=16,
        hidden_dim=32,
        num_layers=3,
        type_dim=8,
    )
    cfg.update(overrides)
    return KettleGraphReasoner(**cfg).double()


def test_log_depth_returns_per_round():
    torch.manual_seed(0)
    nf, ei, et, ed, q = _make_batch()

    # Default-off: per_round_embeddings should be None.
    out_off = _build()(nf, ei, et, ed, q)
    assert out_off.per_round_embeddings is None

    # Flag on: list of length num_layers, each (N, hidden_dim) in-graph.
    model = _build(log_depth=True, num_layers=4)
    out_on = model(nf, ei, et, ed, q)
    assert out_on.per_round_embeddings is not None
    assert len(out_on.per_round_embeddings) == 4
    for h_r in out_on.per_round_embeddings:
        assert h_r.shape == (nf.size(0), model.hidden_dim)
        assert h_r.requires_grad  # gradient must reach each round
    # Last round equals node_embeddings (same underlying tensor).
    assert torch.equal(out_on.per_round_embeddings[-1], out_on.node_embeddings)


def test_forward_shapes_and_ranges():
    torch.manual_seed(0)
    nf, ei, et, ed, q = _make_batch()
    model = _build()
    out = model(nf, ei, et, ed, q)

    assert out.node_scores.shape == (nf.size(0),)
    assert out.edge_scores.shape == (ei.size(1),)
    assert out.node_embeddings.shape == (nf.size(0), model.hidden_dim)
    assert out.edge_type_embeddings.shape == (ed.size(0), model.type_dim)

    for s in (out.node_scores, out.edge_scores):
        assert torch.isfinite(s).all()
        assert (s >= 0).all() and (s <= 1).all()


def test_gradient_flows_through_everything():
    torch.manual_seed(1)
    nf, ei, et, ed, q = _make_batch()
    nf = nf.clone().requires_grad_(True)
    ed = ed.clone().requires_grad_(True)
    q = q.clone().requires_grad_(True)
    model = _build()

    out = model(nf, ei, et, ed, q)
    loss = out.node_scores.mean() + out.edge_scores.mean()
    loss.backward()

    assert torch.isfinite(nf.grad).all() and (nf.grad.abs().sum() > 0)
    assert torch.isfinite(ed.grad).all() and (ed.grad.abs().sum() > 0)
    assert torch.isfinite(q.grad).all() and (q.grad.abs().sum() > 0)
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_query_conditioning_changes_output():
    """Swapping the query vector (same graph, same schema) must change
    node/edge scores — otherwise the model is not query-conditioned."""
    torch.manual_seed(2)
    nf, ei, et, ed, q1 = _make_batch()
    _, _, _, _, q2 = _make_batch(seed=99)
    model = _build()
    model.eval()

    out1 = model(nf, ei, et, ed, q1)
    out2 = model(nf, ei, et, ed, q2)

    assert not torch.allclose(out1.node_scores, out2.node_scores, atol=1e-4)
    assert not torch.allclose(out1.edge_scores, out2.edge_scores, atol=1e-4)


def test_schema_swap_without_retraining():
    """Same model weights, two different schemas (different T and
    descriptors) — forward must succeed and produce valid scores for both.
    This exercises Commitment #3 end-to-end."""
    torch.manual_seed(3)
    model = _build()
    model.eval()

    for T in (3, 9):
        nf, ei, et, ed, q = _make_batch(T=T, seed=T)
        out = model(nf, ei, et, ed, q)
        assert torch.isfinite(out.node_scores).all()
        assert torch.isfinite(out.edge_scores).all()


def test_parameter_count_within_budget():
    """Sanity-check the 500K–2M parameter envelope from CLAUDE.md.
    At hidden=32 / L=3 we expect well under 500K (we have headroom to
    scale up). At hidden=128 we should still be safely under 2M."""
    small = _build(hidden_dim=32, num_layers=3)
    assert small.parameter_count() < 500_000
    large = _build(hidden_dim=128, num_layers=4)
    assert large.parameter_count() < 2_000_000


def _build_v2(**overrides):
    cfg = dict(
        node_feat_dim=12,
        edge_feat_dim=6,
        query_dim=16,
        hidden_dim=32,
        num_layers=3,
        type_dim=8,
    )
    cfg.update(overrides)
    return KGRv2(**cfg).double()


def test_v2_depth_attn_forward_and_ball_constraint():
    torch.manual_seed(0)
    nf, ei, et, ed, q = _make_batch()
    model = _build_v2(depth_attn=True)
    out = model(nf, ei, et, ed, q)

    # Ball constraint on attended embedding: ||h|| < 1/sqrt(c).
    c = float(model.c)
    max_norm = out.node_embeddings.norm(dim=-1).max().item()
    assert max_norm < 1.0 / (c**0.5)
    assert torch.isfinite(out.node_scores).all()
    assert torch.isfinite(out.edge_scores).all()


def test_v2_depth_attn_zero_init_uniform_softmax():
    """Zero-init pseudo-queries → uniform depth weights at step 0."""
    torch.manual_seed(0)
    nf, ei, et, ed, q = _make_batch()
    model = _build_v2(depth_attn=True)

    # Run forward up to the snapshots, then exercise depth_attention.
    # Easiest: hand-build three random tangent snapshots and check that
    # with zero queries the softmax is exactly 1/L.
    D = model.hidden_dim
    snaps = [torch.randn(nf.size(0), D, dtype=torch.float64) for _ in range(3)]
    # Cast the module to double so keys/queries match.
    da = model.depth_attention.double()
    stack = torch.stack(snaps, dim=0)
    keys = da._norm_keys(stack)
    logits = torch.einsum("d,lnd->ln", da.depth_queries[-1], keys)
    alpha = logits.softmax(dim=0)
    assert torch.allclose(
        alpha, torch.full_like(alpha, 1.0 / len(snaps)), atol=1e-6
    )


def test_v2_depth_attn_parameter_delta():
    """DepthAttention adds a bounded, small number of params."""
    base = _build_v2(depth_attn=False)
    with_attn = _build_v2(depth_attn=True)
    delta = with_attn.parameter_count() - base.parameter_count()
    # L*D queries + D RMSNorm scale = (num_layers + 1) * hidden_dim.
    expected = (with_attn.num_layers + 1) * with_attn.hidden_dim
    assert delta == expected


def test_v2_intra_stack_runs():
    torch.manual_seed(1)
    nf, ei, et, ed, q = _make_batch()
    model = _build_v2(depth_attn=True, depth_attn_intra_stack=True)
    out = model(nf, ei, et, ed, q)
    assert torch.isfinite(out.node_scores).all()
    assert out.node_embeddings.norm(dim=-1).max().item() < 1.0 / (float(model.c) ** 0.5)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

"""Baseline parity tests: Euclidean GAT must accept the same inputs as
the experimental model and return the same KGROutput shape."""

from __future__ import annotations

import os
import sys

import pytest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.euclidean_baseline import EuclideanBaseline  # noqa: E402
from src.models.hyperbolic_gnn import KettleGraphReasoner  # noqa: E402


def _batch(N=20, E=60, nf=12, ef=6, q=16, T=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        node_features=torch.randn(N, nf, generator=g, dtype=torch.float64),
        edge_index=torch.randint(0, N, (2, E), generator=g),
        edge_type=torch.randint(0, T, (E,), generator=g),
        edge_descriptor=torch.randn(T, ef, generator=g, dtype=torch.float64),
        query=torch.randn(q, generator=g, dtype=torch.float64),
    )


def test_api_parity_with_experimental_model():
    torch.manual_seed(0)
    cfg = dict(node_feat_dim=12, edge_feat_dim=6, query_dim=16,
               hidden_dim=32, num_layers=3, type_dim=8)
    kgr = KettleGraphReasoner(**cfg).double()
    base = EuclideanBaseline(**cfg).double()
    b = _batch()

    out_k = kgr(**b)
    out_b = base(**b)

    assert out_k.node_scores.shape == out_b.node_scores.shape
    assert out_k.edge_scores.shape == out_b.edge_scores.shape


def test_forward_ranges_and_finite():
    torch.manual_seed(1)
    base = EuclideanBaseline(node_feat_dim=12, edge_feat_dim=6, query_dim=16,
                             hidden_dim=32, num_layers=3).double()
    out = base(**_batch())
    assert torch.isfinite(out.node_scores).all()
    assert (out.node_scores >= 0).all() and (out.node_scores <= 1).all()
    assert torch.isfinite(out.edge_scores).all()
    assert (out.edge_scores >= 0).all() and (out.edge_scores <= 1).all()


def test_gradient_flow():
    torch.manual_seed(2)
    base = EuclideanBaseline(node_feat_dim=12, edge_feat_dim=6, query_dim=16,
                             hidden_dim=32, num_layers=3).double()
    b = _batch()
    b["node_features"] = b["node_features"].requires_grad_(True)
    b["query"] = b["query"].requires_grad_(True)
    out = base(**b)
    (out.node_scores.mean() + out.edge_scores.mean()).backward()

    assert torch.isfinite(b["node_features"].grad).all()
    assert torch.isfinite(b["query"].grad).all()
    for name, p in base.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all()


def test_query_conditioning():
    torch.manual_seed(3)
    base = EuclideanBaseline(node_feat_dim=12, edge_feat_dim=6, query_dim=16,
                             hidden_dim=32, num_layers=3).double()
    base.eval()
    b1 = _batch(seed=10)
    b2 = dict(b1); b2["query"] = torch.randn(16, dtype=torch.float64)
    out1 = base(**b1)
    out2 = base(**b2)
    assert not torch.allclose(out1.node_scores, out2.node_scores, atol=1e-4)


def test_edge_type_is_ignored():
    """Baseline is homogeneous — permuting edge_type must not change output."""
    torch.manual_seed(4)
    base = EuclideanBaseline(node_feat_dim=12, edge_feat_dim=6, query_dim=16,
                             hidden_dim=32, num_layers=3).double()
    base.eval()
    b = _batch(seed=20)
    out1 = base(**b)
    b2 = dict(b); b2["edge_type"] = torch.randperm(b["edge_type"].numel()) % 5
    out2 = base(**b2)
    assert torch.allclose(out1.node_scores, out2.node_scores, atol=1e-10)
    assert torch.allclose(out1.edge_scores, out2.edge_scores, atol=1e-10)


def test_parameter_counts_comparable():
    """At matched hidden_dim and num_layers the two models should be in the
    same order of magnitude — a fair A/B, not orders apart."""
    cfg = dict(node_feat_dim=12, edge_feat_dim=6, query_dim=16,
               hidden_dim=64, num_layers=3, type_dim=8)
    kgr = KettleGraphReasoner(**cfg)
    base = EuclideanBaseline(**cfg)
    r = kgr.parameter_count() / base.parameter_count()
    assert 0.5 < r < 2.0, f"param ratio {r} outside [0.5, 2.0]"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

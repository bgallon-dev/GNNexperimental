"""Tests for src/models/layers/edge_attention.py."""

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


def _rand_ball(n: int, d: int, c: float = 1.0, scale: float = 0.3,
               dtype=torch.float64, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, d, generator=g, dtype=dtype) * scale
    return po.project(v, c)


def test_softmax_normalized_per_receiver():
    torch.manual_seed(0)
    N, d, T = 10, 8, 4
    c = 1.0
    x = _rand_ball(N, d, c=c)
    # Hand-constructed: each node has at least one incoming edge; some have >1.
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 5, 5, 5]]
    )
    edge_type = torch.randint(0, T, (edge_index.size(1),))
    attn = EdgeTypedAttention(d, num_edge_types=T, c=c).double()

    alpha = attn(x, edge_index, edge_type)

    assert alpha.shape == (edge_index.size(1),)
    assert torch.isfinite(alpha).all()
    assert (alpha >= 0).all() and (alpha <= 1 + 1e-10).all()

    # Sum to 1 per receiver (for receivers that appear in dst).
    dst = edge_index[1]
    for r in torch.unique(dst).tolist():
        mask = dst == r
        assert torch.isclose(alpha[mask].sum(), torch.tensor(1.0, dtype=alpha.dtype),
                             atol=1e-10)


def test_edge_type_matters():
    torch.manual_seed(1)
    N, d, T = 6, 8, 3
    c = 1.0
    x = _rand_ball(N, d, c=c)
    edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]])  # 3 incoming to node 3
    attn = EdgeTypedAttention(d, num_edge_types=T, c=c).double()

    et_a = torch.tensor([0, 1, 2])
    et_b = torch.tensor([2, 1, 0])
    a_a = attn(x, edge_index, et_a)
    a_b = attn(x, edge_index, et_b)
    # Different type assignments should produce different attention.
    assert not torch.allclose(a_a, a_b, atol=1e-6)


def test_gradient_flow():
    torch.manual_seed(2)
    N, d, T, E = 15, 8, 4, 40
    c = 1.0
    x = _rand_ball(N, d, c=c).requires_grad_(True)
    edge_index = torch.randint(0, N, (2, E))
    edge_type = torch.randint(0, T, (E,))
    attn = EdgeTypedAttention(d, num_edge_types=T, c=c).double()

    alpha = attn(x, edge_index, edge_type)
    alpha.sum().backward()

    assert torch.isfinite(attn.W_q.weight.grad).all()
    assert torch.isfinite(attn.type_emb.weight.grad).all()
    assert torch.isfinite(x.grad).all()
    assert (attn.a.grad.abs().sum() > 0).item()


def test_integrates_with_message_passing():
    """Full smoke test: attention scores feed straight into message passing."""
    torch.manual_seed(3)
    N, d, T, E = 20, 8, 4, 60
    c = 1.0
    x = _rand_ball(N, d, c=c)
    edge_index = torch.randint(0, N, (2, E))
    edge_type = torch.randint(0, T, (E,))

    attn = EdgeTypedAttention(d, num_edge_types=T, c=c).double()
    mp = HyperbolicMessagePassing(d, d, c=c).double()

    alpha = attn(x, edge_index, edge_type)
    out = mp(x, edge_index, edge_weight=alpha)

    assert out.shape == (N, d)
    assert torch.isfinite(out).all()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

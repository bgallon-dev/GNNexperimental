"""Tests for src/models/layers/hyp_message_pass.py.

Targets the exact failure modes the layer can exhibit:
- finiteness and ball-interior invariants on random graphs,
- gradient flow through logmap/expmap (the _Artanh boundary-safe path),
- edge-weight routing correctness,
- curvature threading,
- isolated-node (degree-zero receiver) behavior,
- zero-aggregate drift,
- near-boundary numerical stress (forward + backward).
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.layers import poincare_ops as po  # noqa: E402
from src.models.layers.hyp_message_pass import HyperbolicMessagePassing  # noqa: E402


def _rand_ball(n: int, d: int, c: float = 1.0, scale: float = 0.3,
               dtype=torch.float64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, d, generator=g, dtype=dtype) * scale
    return po.project(v, c)


def _inside_ball(x: torch.Tensor, c: float) -> bool:
    eps = po.BALL_EPS[x.dtype] if x.dtype in po.BALL_EPS else 1e-5
    maxnorm = (1 - eps) / math.sqrt(c)
    return bool((x.norm(dim=-1) <= maxnorm + 1e-6).all())


# --------------------------------------------------------------------------- #
# 1. shape + finiteness on a random graph
# --------------------------------------------------------------------------- #
def test_forward_shape_finite_and_in_ball():
    torch.manual_seed(0)
    N, d_in, d_out, E = 50, 16, 16, 200
    c = 1.0
    x = _rand_ball(N, d_in, c=c, dtype=torch.float64)
    edge_index = torch.randint(0, N, (2, E))
    layer = HyperbolicMessagePassing(d_in, d_out, c=c).double()

    out = layer(x, edge_index)

    assert out.shape == (N, d_out)
    assert torch.isfinite(out).all()
    assert _inside_ball(out, c)


# --------------------------------------------------------------------------- #
# 2. gradient flow (boundary-safe _Artanh path must not NaN)
# --------------------------------------------------------------------------- #
def test_gradient_flow_is_finite():
    torch.manual_seed(1)
    N, d, E = 30, 8, 100
    c = 1.0
    x = _rand_ball(N, d, c=c, dtype=torch.float64).requires_grad_(True)
    edge_index = torch.randint(0, N, (2, E))
    layer = HyperbolicMessagePassing(d, d, c=c).double()

    out = layer(x, edge_index)
    loss = out.pow(2).sum()
    loss.backward()

    assert torch.isfinite(layer.weight.grad).all()
    assert (layer.weight.grad.abs().sum() > 0).item()
    assert torch.isfinite(layer.bias.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert (x.grad.abs().sum() > 0).item()


# --------------------------------------------------------------------------- #
# 3. edge-weight correctness: weight=0 on an edge drops it
# --------------------------------------------------------------------------- #
def test_edge_weight_zero_drops_edge():
    torch.manual_seed(2)
    d = 4
    c = 1.0
    x = _rand_ball(3, d, c=c, dtype=torch.float64)
    # Two incoming edges to node 2: from 0 and from 1.
    edge_index = torch.tensor([[0, 1], [2, 2]])
    layer = HyperbolicMessagePassing(d, d, c=c).double()

    w_full = torch.tensor([1.0, 0.0], dtype=torch.float64)
    out_full = layer(x, edge_index, edge_weight=w_full)

    # Equivalent graph: only the first edge exists, with weight 1.
    edge_index_single = torch.tensor([[0], [2]])
    w_single = torch.tensor([1.0], dtype=torch.float64)
    out_single = layer(x, edge_index_single, edge_weight=w_single)

    assert torch.allclose(out_full[2], out_single[2], atol=1e-10)


# --------------------------------------------------------------------------- #
# 4. curvature is threaded: c=0.5 and c=1.0 produce different finite outputs
# --------------------------------------------------------------------------- #
def test_curvature_threading():
    torch.manual_seed(3)
    N, d, E = 20, 8, 60
    x_euc = torch.randn(N, d, dtype=torch.float64) * 0.25
    edge_index = torch.randint(0, N, (2, E))

    x05 = po.project(x_euc, 0.5)
    x10 = po.project(x_euc, 1.0)

    layer05 = HyperbolicMessagePassing(d, d, c=0.5).double()
    layer10 = HyperbolicMessagePassing(d, d, c=1.0).double()
    # Share weights so only `c` differs.
    layer10.weight.data.copy_(layer05.weight.data)
    layer10.bias.data.copy_(layer05.bias.data)

    out05 = layer05(x05, edge_index)
    out10 = layer10(x10, edge_index)
    assert torch.isfinite(out05).all() and torch.isfinite(out10).all()
    assert not torch.allclose(out05, out10, atol=1e-4)


# --------------------------------------------------------------------------- #
# 5. isolated node: degree-zero receiver goes through self + linear transform
# --------------------------------------------------------------------------- #
def test_isolated_node_equals_self_transform():
    torch.manual_seed(4)
    d = 4
    c = 1.0
    x = _rand_ball(3, d, c=c, dtype=torch.float64)
    # Node 2 has no incoming edges. Nodes 0,1 wire arbitrarily.
    edge_index = torch.tensor([[0, 1], [1, 0]])
    layer = HyperbolicMessagePassing(d, d, c=c, activation="identity",
                                     use_bias=False).double()

    out = layer(x, edge_index)

    # Manual computation for isolated receiver: agg=0 → expmap(0, x_i)=x_i →
    # mobius_matvec → project.
    expected = po.project(po.mobius_matvec(layer.weight, x[2:3], c), c)
    assert torch.allclose(out[2:3], expected, atol=1e-10)


# --------------------------------------------------------------------------- #
# 6. zero tangent vector expmap does not drift from basepoint
# --------------------------------------------------------------------------- #
def test_expmap_zero_returns_basepoint():
    # Defensive check on the ops layer that the message-pass layer relies on.
    c = 1.0
    x = _rand_ball(5, 6, c=c, dtype=torch.float64)
    zero = torch.zeros_like(x)
    y = po.expmap(zero, x, c)
    assert torch.allclose(y, x, atol=1e-12)


# --------------------------------------------------------------------------- #
# 7. near-boundary stress: forward + backward stay finite at norm ≈ 0.99/√c
# --------------------------------------------------------------------------- #
def test_near_boundary_stability():
    torch.manual_seed(5)
    N, d, E = 20, 8, 60
    c = 1.0
    # Points at ~0.99 of the ball radius.
    v = torch.randn(N, d, dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True) * (0.99 / math.sqrt(c))
    x = po.project(v, c).requires_grad_(True)
    edge_index = torch.randint(0, N, (2, E))
    layer = HyperbolicMessagePassing(d, d, c=c).double()

    out = layer(x, edge_index)
    loss = out.pow(2).sum()
    loss.backward()

    assert torch.isfinite(out).all()
    assert _inside_ball(out.detach(), c)
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.isfinite(x.grad).all()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

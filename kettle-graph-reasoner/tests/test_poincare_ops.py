"""Tests for src/models/layers/poincare_ops.py.

- fp64 for correctness / algebraic identities.
- fp32 for stability tests (the real training regime).
- geoopt acts as the reference implementation via the k = -c sign mapping.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# Add the project root (the parent of this tests/ directory) to sys.path so
# `src.models.layers.poincare_ops` imports regardless of pytest invocation cwd.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from geoopt.manifolds.stereographic import math as gmath  # noqa: E402

from src.models.layers import poincare_ops as po  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _to_geoopt_k(c: float, dtype=torch.float64) -> torch.Tensor:
    """Our c > 0 → geoopt's signed k = -c for the Poincaré ball."""
    return torch.tensor(-c, dtype=dtype)


def _rand_ball(shape, c: float, dtype=torch.float64, max_frac: float = 0.9, seed: int = 0):
    """Random points strictly inside the ball of radius 1/√c."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g, dtype=dtype)
    x_norm = x.norm(dim=-1, keepdim=True)
    scale = torch.rand(*shape[:-1], 1, generator=g, dtype=dtype) * max_frac / (c ** 0.5)
    return x / x_norm.clamp_min(1e-15) * scale


def _rand_tangent(shape, dtype=torch.float64, scale: float = 0.3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=dtype) * scale


CURVATURES = [0.5, 1.0, 2.0]
ATOL_CORR = 1e-6
ATOL_ROUND = 1e-8


# --------------------------------------------------------------------------- #
# correctness: geoopt cross-check at multiple curvatures
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("c", CURVATURES)
def test_mobius_add_matches_geoopt(c):
    x = _rand_ball((8, 16), c, seed=1)
    y = _rand_ball((8, 16), c, seed=2)
    k = _to_geoopt_k(c)
    ours = po.mobius_add(x, y, c)
    ref = gmath.mobius_add(x, y, k=k)
    # project(ref) so the comparison is against the same invariant our op upholds
    ref = gmath.project(ref, k=k)
    assert torch.allclose(ours, ref, atol=ATOL_CORR, rtol=ATOL_CORR)


@pytest.mark.parametrize("c", CURVATURES)
def test_expmap_matches_geoopt(c):
    x = _rand_ball((8, 16), c, seed=3)
    v = _rand_tangent((8, 16), seed=4)
    k = _to_geoopt_k(c)
    ours = po.expmap(v, x, c)
    ref = gmath.project(gmath.expmap(x, v, k=k), k=k)
    assert torch.allclose(ours, ref, atol=ATOL_CORR, rtol=ATOL_CORR)


@pytest.mark.parametrize("c", CURVATURES)
def test_logmap_matches_geoopt(c):
    x = _rand_ball((8, 16), c, seed=5)
    y = _rand_ball((8, 16), c, seed=6)
    k = _to_geoopt_k(c)
    ours = po.logmap(y, x, c)
    ref = gmath.logmap(x, y, k=k)
    assert torch.allclose(ours, ref, atol=ATOL_CORR, rtol=ATOL_CORR)


@pytest.mark.parametrize("c", CURVATURES)
def test_dist_matches_geoopt(c):
    x = _rand_ball((8, 16), c, seed=7)
    y = _rand_ball((8, 16), c, seed=8)
    k = _to_geoopt_k(c)
    ours = po.dist(x, y, c)
    ref = gmath.dist(x, y, k=k)
    assert torch.allclose(ours, ref, atol=ATOL_CORR, rtol=ATOL_CORR)


@pytest.mark.parametrize("c", CURVATURES)
def test_parallel_transport_matches_geoopt(c):
    x = _rand_ball((8, 16), c, seed=9)
    y = _rand_ball((8, 16), c, seed=10)
    v = _rand_tangent((8, 16), seed=11)
    k = _to_geoopt_k(c)
    ours = po.parallel_transport(x, y, v, c)
    ref = gmath.parallel_transport(x, y, v, k=k)
    assert torch.allclose(ours, ref, atol=ATOL_CORR, rtol=ATOL_CORR)


@pytest.mark.parametrize("c", CURVATURES)
def test_expmap0_matches_geoopt(c):
    v = _rand_tangent((8, 16), seed=12)
    k = _to_geoopt_k(c)
    ours = po.expmap0(v, c)
    ref = gmath.project(gmath.expmap0(v, k=k), k=k)
    assert torch.allclose(ours, ref, atol=ATOL_CORR, rtol=ATOL_CORR)


@pytest.mark.parametrize("c", CURVATURES)
def test_logmap0_matches_geoopt(c):
    y = _rand_ball((8, 16), c, seed=13)
    k = _to_geoopt_k(c)
    ours = po.logmap0(y, c)
    ref = gmath.logmap0(y, k=k)
    assert torch.allclose(ours, ref, atol=ATOL_CORR, rtol=ATOL_CORR)


# --------------------------------------------------------------------------- #
# algebraic identities
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("c", CURVATURES)
def test_mobius_add_inverse(c):
    x = _rand_ball((8, 16), c, seed=20)
    out = po.mobius_add(x, -x, c)
    assert torch.allclose(out, torch.zeros_like(out), atol=ATOL_ROUND)


@pytest.mark.parametrize("c", CURVATURES)
def test_mobius_add_identity(c):
    x = _rand_ball((8, 16), c, seed=21)
    z = torch.zeros_like(x)
    assert torch.allclose(po.mobius_add(z, x, c), x, atol=ATOL_ROUND)
    assert torch.allclose(po.mobius_add(x, z, c), x, atol=ATOL_ROUND)


@pytest.mark.parametrize("c", CURVATURES)
def test_log_exp_round_trip_at_x(c):
    x = _rand_ball((8, 16), c, seed=22)
    v = _rand_tangent((8, 16), seed=23, scale=0.2)
    y = po.expmap(v, x, c)
    v_rec = po.logmap(y, x, c)
    assert torch.allclose(v_rec, v, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("c", CURVATURES)
def test_exp_log_round_trip_at_x(c):
    x = _rand_ball((8, 16), c, seed=24)
    y = _rand_ball((8, 16), c, seed=25)
    v = po.logmap(y, x, c)
    y_rec = po.expmap(v, x, c)
    assert torch.allclose(y_rec, y, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("c", CURVATURES)
def test_dist_zero_self(c):
    x = _rand_ball((8, 16), c, seed=26)
    out = po.dist(x, x, c)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)


@pytest.mark.parametrize("c", CURVATURES)
def test_dist_symmetric(c):
    x = _rand_ball((8, 16), c, seed=27)
    y = _rand_ball((8, 16), c, seed=28)
    assert torch.allclose(po.dist(x, y, c), po.dist(y, x, c), atol=ATOL_ROUND)


@pytest.mark.parametrize("c", CURVATURES)
def test_expmap0_agrees_with_expmap_at_zero(c):
    v = _rand_tangent((8, 16), seed=29)
    zeros = torch.zeros_like(v)
    assert torch.allclose(po.expmap0(v, c), po.expmap(v, zeros, c), atol=ATOL_ROUND)


@pytest.mark.parametrize("c", CURVATURES)
def test_logmap0_agrees_with_logmap_at_zero(c):
    y = _rand_ball((8, 16), c, seed=30)
    zeros = torch.zeros_like(y)
    assert torch.allclose(po.logmap0(y, c), po.logmap(y, zeros, c), atol=ATOL_ROUND)


# --------------------------------------------------------------------------- #
# numerical stability (the real reason this layer exists)
# --------------------------------------------------------------------------- #

def test_project_clamps_far_exterior():
    x = torch.randn(32, 16, dtype=torch.float32)
    x = x / x.norm(dim=-1, keepdim=True) * 10.0  # grossly outside the ball
    c = 1.0
    projected = po.project(x, c)
    norms = projected.norm(dim=-1)
    assert torch.isfinite(projected).all()
    assert (norms < 1.0 / (c ** 0.5)).all()


def test_dist_finite_at_boundary_fp32():
    c = 1.0
    # two distinct points projected to near-boundary norm
    x = torch.randn(32, 16, dtype=torch.float32)
    y = torch.randn(32, 16, dtype=torch.float32)
    x = po.project(x / x.norm(dim=-1, keepdim=True) * 10.0, c)
    y = po.project(y / y.norm(dim=-1, keepdim=True) * 10.0, c)
    d = po.dist(x, y, c)
    assert torch.isfinite(d).all()
    # distance should be large but finite; assert reasonable magnitude (≤ 50)
    assert d.max() < 50.0


def test_mobius_add_finite_at_boundary_fp32():
    c = 1.0
    x = torch.randn(32, 16, dtype=torch.float32)
    y = torch.randn(32, 16, dtype=torch.float32)
    x = po.project(x / x.norm(dim=-1, keepdim=True) * 10.0, c)
    y = po.project(y / y.norm(dim=-1, keepdim=True) * 10.0, c)
    out = po.mobius_add(x, y, c)
    assert torch.isfinite(out).all()
    assert (out.norm(dim=-1) < 1.0 / (c ** 0.5)).all()


def test_dist_grad_finite_at_origin():
    c = 1.0
    x = torch.zeros(4, 8, dtype=torch.float64, requires_grad=True)
    y = _rand_ball((4, 8), c, seed=40)
    d = po.dist(x, y, c).sum()
    d.backward()
    assert torch.isfinite(x.grad).all()


def test_logmap_grad_finite_at_identical_points():
    c = 1.0
    base = _rand_ball((4, 8), c, seed=41)
    x = base.clone().requires_grad_(True)
    y = base.clone()  # y == x
    v = po.logmap(y, x, c).sum()
    v.backward()
    assert torch.isfinite(x.grad).all()


def test_expmap_logmap_chain_grad_finite_fp32():
    c = 1.0
    x = _rand_ball((4, 8), c, dtype=torch.float32, seed=42)
    y = _rand_ball((4, 8), c, dtype=torch.float32, seed=43).requires_grad_(True)
    out = po.expmap(po.logmap(y, x, c), x, c).sum()
    out.backward()
    assert torch.isfinite(y.grad).all()


# --------------------------------------------------------------------------- #
# differentiability (gradcheck on small interior inputs)
# --------------------------------------------------------------------------- #

def _small_interior(shape, seed, c=1.0):
    x = _rand_ball(shape, c, dtype=torch.float64, max_frac=0.5, seed=seed)
    return x.detach().clone().requires_grad_(True)


def test_gradcheck_mobius_add():
    c = 1.0
    x = _small_interior((2, 4), seed=50, c=c)
    y = _small_interior((2, 4), seed=51, c=c)
    assert torch.autograd.gradcheck(lambda a, b: po.mobius_add(a, b, c), (x, y), atol=1e-4)


def test_gradcheck_expmap():
    c = 1.0
    x = _small_interior((2, 4), seed=52, c=c)
    v = torch.randn(2, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(53)) * 0.1
    v.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda vv, xx: po.expmap(vv, xx, c), (v, x), atol=1e-4)


def test_gradcheck_logmap():
    c = 1.0
    x = _small_interior((2, 4), seed=54, c=c)
    y = _small_interior((2, 4), seed=55, c=c)
    assert torch.autograd.gradcheck(lambda yy, xx: po.logmap(yy, xx, c), (y, x), atol=1e-4)


def test_gradcheck_dist():
    c = 1.0
    x = _small_interior((2, 4), seed=56, c=c)
    y = _small_interior((2, 4), seed=57, c=c)
    assert torch.autograd.gradcheck(lambda a, b: po.dist(a, b, c), (x, y), atol=1e-4)

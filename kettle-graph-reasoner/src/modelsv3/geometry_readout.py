r"""Geometry read-out primitives — new scoring families over frozen balls.

The 2026-07 geometry-readout probe program (Docs/GEOMETRY_READOUT_PROBES_PLAN.md)
asks what geometric structure the trunk already encodes that no shipped op
reads. This module holds the training-free primitives those probes share.
Everything here is fit-or-zero-training by construction: no gradients flow,
no encoder is touched.

    busemann_score(xi, x, c)          horosphere depth toward ideal point xi
                                      (the hyperbolic 'linear functional' —
                                      cone/half-space relevance, d params)
    geodesic_points(a, b, c, T)       T points on the TRUE geodesic a->b
                                      (expmap_a, not the tangent-at-origin
                                      _midpoint approximation)
    dist_to_geodesic_segment(a, b, x) min_t dist(x, gamma(t)) — 'lies between
                                      A and B' as a segment, not a midpoint
    karcher_mean(points, weights, c)  weighted Frechet mean on the ball —
                                      multi-anchor query composition
    hyperbolic_dispersion(x, c)       mean pairwise dist + radius spread of a
                                      candidate set (routing/abstain signal)
    geodesic_descent(...)             graph-edge walk greedily decreasing
                                      hyperbolic distance to a target point

Conventions match ``distance_scoring.py``: points live on the radius
``1/sqrt(c)`` Poincare ball, ``c`` is the curvature magnitude, batch dim
first. All functions accept (N, D) tensors and are pure torch.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch
from torch import Tensor

from ..modelsv2.layers import poincare_ops as P

Curvature = Union[float, Tensor]

_EPS = 1e-9


def _as_2d(x: Tensor) -> Tensor:
    return x.unsqueeze(0) if x.dim() == 1 else x


# ---------------------------------------------------------------------------
# Busemann function — scoring family probe (P2)
# ---------------------------------------------------------------------------

def busemann(xi: Tensor, x: Tensor, c: Curvature = 1.0) -> Tensor:
    r"""Busemann function ``B_xi(x)`` toward the ideal point ``xi``.

    On the unit Poincare ball the closed form is

        B_xi(y) = log( ||xi - y||^2 / (1 - ||y||^2) )

    with ``xi`` on the boundary sphere. For curvature ``c`` we rescale to
    the unit ball (``y = sqrt(c) x``) and divide by ``sqrt(c)`` so values
    are comparable with ``P.dist`` at the same curvature.

    B_xi(0) = 0; B decreases toward ``xi`` (more negative = deeper in the
    horoball = further 'downstream' in direction xi) and increases away.
    Level sets are horospheres — the hyperbolic analog of the affine
    hyperplanes a Euclidean inner-product scorer uses. This is the d-param
    'linear functional' scoring family: direction, not point.

    Parameters
    ----------
    xi : (D,) unit vector (ideal boundary point). Normalized defensively.
    x  : (N, D) ball points.
    Returns (N,) Busemann values (LOWER = deeper toward xi).
    """
    c_t = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    sqrt_c = c_t.clamp_min(1e-15).sqrt()
    xi = xi / xi.norm().clamp_min(_EPS)
    y = _as_2d(x) * sqrt_c
    y2 = y.pow(2).sum(dim=-1)
    num = (xi.unsqueeze(0) - y).pow(2).sum(dim=-1).clamp_min(_EPS)
    den = (1.0 - y2).clamp_min(_EPS)
    return torch.log(num / den) / sqrt_c


def busemann_score(xi: Tensor, x: Tensor, c: Curvature = 1.0) -> Tensor:
    """Relevance = ``-B_xi`` (higher = deeper toward the ideal point).
    Drop-in shaped like ``score_from_embeddings`` but conditioned on a
    DIRECTION instead of a ball point."""
    return -busemann(xi, x, c)


def ideal_point_from_query(query_point: Tensor) -> Tensor:
    """Zero-training xi: push the query point radially to the boundary.
    The probe's first rung — reuses the trained QueryToBall output as a
    direction. Falls back to e_1 for a query at the exact origin."""
    n = query_point.norm()
    if float(n) < _EPS:
        xi = torch.zeros_like(query_point)
        xi[0] = 1.0
        return xi
    return query_point / n


# ---------------------------------------------------------------------------
# True geodesic + segment distance (P3)
# ---------------------------------------------------------------------------

def geodesic_points(a: Tensor, b: Tensor, c: Curvature = 1.0,
                    n_points: int = 33) -> Tensor:
    r"""``n_points`` samples of the TRUE geodesic ``gamma(t)`` from ``a``
    to ``b``: ``gamma(t) = expmap_a(t * logmap_a(b))``. Unlike
    ``eval_retrieval_midpoint._midpoint`` (tangent-at-origin average) this
    is the actual constant-speed geodesic of the ball."""
    a2, b2 = _as_2d(a), _as_2d(b)
    v = P.logmap(b2, a2, c)  # tangent at a pointing to b
    ts = torch.linspace(0.0, 1.0, n_points, dtype=a2.dtype, device=a2.device)
    pts = [P.expmap(t * v, a2, c) for t in ts]
    return P.project(torch.cat(pts, dim=0), c)


def geodesic_point(a: Tensor, b: Tensor, t: float,
                   c: Curvature = 1.0) -> Tensor:
    """Single point ``gamma(t)`` on the true geodesic (t=0.5 = the exact
    geodesic midpoint)."""
    a2, b2 = _as_2d(a), _as_2d(b)
    v = P.logmap(b2, a2, c)
    return P.project(P.expmap(t * v, a2, c), c).squeeze(0)


def dist_to_geodesic_segment(a: Tensor, b: Tensor, x: Tensor,
                             c: Curvature = 1.0,
                             n_points: int = 33) -> Tensor:
    """``min_t dist(x_i, gamma(t))`` for every row of ``x`` — 'how far off
    the a->b chain is this node'. Sampled minimum (n_points along the
    segment); adequate for ranking probes since dist(x, gamma(t)) is
    unimodal in t on a geodesic.

    Returns (N,) distances; score with the negative."""
    gam = geodesic_points(a, b, c, n_points)          # (T, D)
    x2 = _as_2d(x)                                    # (N, D)
    N, T = x2.size(0), gam.size(0)
    xx = x2.unsqueeze(1).expand(N, T, -1).reshape(N * T, -1)
    gg = gam.unsqueeze(0).expand(N, T, -1).reshape(N * T, -1)
    d = P.dist(xx, gg, c, keepdim=False).reshape(N, T)
    return d.min(dim=1).values


# ---------------------------------------------------------------------------
# Weighted Karcher (Frechet) mean — multi-anchor composition (P4)
# ---------------------------------------------------------------------------

def karcher_mean(points: Tensor, weights: Optional[Tensor] = None,
                 c: Curvature = 1.0, n_iters: int = 50,
                 tol: float = 1e-6) -> Tensor:
    """Weighted Frechet mean of ``points`` (K, D) on the ball via the
    standard Riemannian fixed-point iteration:

        m <- expmap_m( sum_i w_i * logmap_m(x_i) )

    initialized at the tangent-at-origin weighted mean (the K=2
    equal-weight case converges to the exact geodesic midpoint). Weights
    are normalized to sum to 1. Training-free; generalizes
    ``retrieval_ops.bridge`` from 2 anchors to k weighted anchors."""
    pts = _as_2d(points)
    K = pts.size(0)
    if weights is None:
        w = torch.full((K,), 1.0 / K, dtype=pts.dtype, device=pts.device)
    else:
        w = weights.to(pts) / weights.to(pts).sum().clamp_min(_EPS)
    m = P.project(P.expmap0(
        (P.logmap0(pts, c) * w.unsqueeze(1)).sum(dim=0, keepdim=True), c), c)
    for _ in range(n_iters):
        v = (P.logmap(pts, m.expand_as(pts), c) * w.unsqueeze(1)).sum(
            dim=0, keepdim=True)
        if float(v.norm()) < tol:
            break
        m = P.project(P.expmap(v, m, c), c)
    return m.squeeze(0)


# ---------------------------------------------------------------------------
# Candidate-set dispersion — routing / abstain signal (P5)
# ---------------------------------------------------------------------------

def hyperbolic_dispersion(x: Tensor, c: Curvature = 1.0,
                          max_pairs: int = 2000,
                          generator: Optional[torch.Generator] = None) -> dict:
    """Geometry statistics of a candidate set: mean/max pairwise
    hyperbolic distance (subsampled beyond ``max_pairs`` pairs), radius
    mean/std, and eccentricity of the set around its Karcher mean. These
    are the per-query routing features P5 tests against blend-vs-mixture
    wins."""
    x2 = _as_2d(x)
    N = x2.size(0)
    out: dict = {"n": int(N)}
    r = x2.norm(dim=-1)
    out["radius_mean"] = float(r.mean())
    out["radius_std"] = float(r.std(unbiased=False))
    if N < 2:
        out.update(pairwise_mean=0.0, pairwise_max=0.0, eccentricity=0.0)
        return out
    iu, ju = torch.triu_indices(N, N, offset=1)
    if iu.numel() > max_pairs:
        sel = torch.randperm(iu.numel(), generator=generator)[:max_pairs]
        iu, ju = iu[sel], ju[sel]
    d = P.dist(x2[iu], x2[ju], c, keepdim=False)
    out["pairwise_mean"] = float(d.mean())
    out["pairwise_max"] = float(d.max())
    m = karcher_mean(x2, c=c)
    dm = P.dist(x2, m.unsqueeze(0).expand_as(x2), c, keepdim=False)
    out["eccentricity"] = float(dm.mean())
    return out


# ---------------------------------------------------------------------------
# Geodesic-guided descent — nonlocal retrieval probe (P6)
# ---------------------------------------------------------------------------

def geodesic_descent(
    neighbors: Callable[[int], Sequence[int]],
    node_emb: Tensor,
    start_nodes: Sequence[int],
    target_point: Tensor,
    c: Curvature = 1.0,
    beam: int = 8,
    max_steps: int = 12,
    patience: int = 2,
) -> list[int]:
    """Walk GRAPH edges, greedily decreasing hyperbolic distance to
    ``target_point``; return every node visited, ordered by distance
    (closest first). This is manifold-guided graph search — it transits
    intermediates by following edges (unlike the refuted additive relay,
    which summed affinities with no structural constraint).

    ``neighbors(i)`` returns adjacent node indices; ``node_emb`` is the
    (N, D) frozen ball; ``start_nodes`` seeds the frontier (typically the
    query anchor +/- its kNN); ``beam`` nodes survive per step; the walk
    stops after ``max_steps`` or ``patience`` consecutive steps without
    improving the best distance seen."""
    tgt = target_point.unsqueeze(0)

    def d_of(rows: Sequence[int]) -> Tensor:
        e = node_emb[torch.as_tensor(list(rows), dtype=torch.long)]
        return P.dist(e, tgt.expand_as(e), c, keepdim=False)

    visited: dict[int, float] = {}
    frontier = list(dict.fromkeys(int(s) for s in start_nodes))
    for row, dv in zip(frontier, d_of(frontier)):
        visited[row] = float(dv)
    best = min(visited.values())
    stall = 0
    for _ in range(max_steps):
        cand = {v for u in frontier for v in neighbors(u) if v not in visited}
        if not cand:
            break
        cand = list(cand)
        dc = d_of(cand)
        for row, dv in zip(cand, dc):
            visited[int(row)] = float(dv)
        order = torch.argsort(dc)
        frontier = [int(cand[i]) for i in order[:beam]]
        step_best = float(dc.min())
        if step_best < best - 1e-9:
            best, stall = step_best, 0
        else:
            stall += 1
            if stall >= patience:
                break
    return sorted(visited, key=visited.__getitem__)

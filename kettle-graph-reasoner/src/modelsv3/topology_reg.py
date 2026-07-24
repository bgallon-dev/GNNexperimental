r"""v3.1 Phase 4 — Stage-C topology-preservation losses.

Stage C lets stage-B query gradients touch ONLY the top encoder layer.
This is in tension with the v3 query-agnostic-encoder commitment, so it
is opt-in, gated, and clamped by two penalties measured against the
FROZEN baseline encoder's embeddings (computed once per graph, detached
— the baseline is the structural ground truth Stage C must not destroy):

  edge_preserve_loss   = mean over graph edges of
                         (d(u,v; now) - d(u,v; base))^2
                         keeps connected nodes the same distance apart.
  radius_stability_loss = mean over nodes of
                         (||logmap0(emb_now)|| - ||logmap0(emb_base)||)^2
                         keeps the hyperbolic radius profile stable
                         (the Stage-C analogue of the stage-A radial-reg
                         floor; collapse mitigation must not regress).

Both are geometry-aware (hyperbolic ``P.dist`` / ``logmap0`` radius) and
fall back to L2 for the Euclidean baseline so the same call site works.
"""

from __future__ import annotations

from typing import Union

import torch
from torch import Tensor

from ..modelsv2.layers import poincare_ops as P

Curvature = Union[float, Tensor]


def _edge_dist(emb: Tensor, edge_index: Tensor, c: Curvature,
               euclidean: bool) -> Tensor:
    """Per-edge distance d(src, dst). Returns (E,)."""
    src = emb.index_select(0, edge_index[0])
    dst = emb.index_select(0, edge_index[1])
    if euclidean:
        return (src - dst).norm(dim=-1, p=2)
    return P.dist(src, dst, c, keepdim=False)


def _radius(emb: Tensor, c: Curvature, euclidean: bool) -> Tensor:
    """Per-node radius. Euclidean: ||emb||. Hyperbolic: ||logmap0(emb)||
    = geodesic distance from the origin (the true hyperbolic radius,
    same notion the manifold-index export records)."""
    if euclidean:
        return emb.norm(dim=-1, p=2)
    return P.logmap0(emb, c).norm(dim=-1, p=2)


def edge_preserve_loss(
    emb_now: Tensor,
    emb_base: Tensor,
    edge_index: Tensor,
    c: Curvature = 1.0,
    euclidean: bool = False,
) -> Tensor:
    r"""``mean_e (d_e^now - d_e^base)^2`` over graph edges. ``emb_base``
    must be detached (frozen baseline encoder output)."""
    if edge_index.numel() == 0:
        return emb_now.new_zeros(())
    d_now = _edge_dist(emb_now, edge_index, c, euclidean)
    d_base = _edge_dist(emb_base, edge_index, c, euclidean).detach()
    return ((d_now - d_base) ** 2).mean()


def radius_stability_loss(
    emb_now: Tensor,
    emb_base: Tensor,
    c: Curvature = 1.0,
    euclidean: bool = False,
) -> Tensor:
    r"""``mean_i (r_i^now - r_i^base)^2`` over nodes. ``emb_base`` must
    be detached."""
    r_now = _radius(emb_now, c, euclidean)
    r_base = _radius(emb_base, c, euclidean).detach()
    return ((r_now - r_base) ** 2).mean()

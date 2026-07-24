r"""Distance-based scoring for KGR v3 — eval-time bridge to metrics.py.

Given trained node embeddings and a query-point embedding, produces a
per-node score tensor of shape ``(N,)`` suitable for
``MetricAccumulator.add(scores, labels, task_type)``. The metric module
only uses scores for ranking (``torch.argsort`` / ``torch.topk``), so
raw real-valued ``-dist`` works directly — no sigmoid, no normalization.

Both hyperbolic and Euclidean variants are provided so the
Euclidean-v3 baseline can share the scoring path.
"""

from __future__ import annotations

from typing import Union

import torch
from torch import Tensor

from ..modelsv2.layers import poincare_ops as P

Curvature = Union[float, Tensor]


def score_from_embeddings(
    node_embeddings: Tensor,
    query_point: Tensor,
    c: Curvature = 1.0,
    euclidean: bool = False,
) -> Tensor:
    r"""Per-node relevance = ``-dist(node, query)``.

    Higher score = more relevant (closer to the query point).

    Parameters
    ----------
    node_embeddings : (N, D)
        Poincaré-ball (or Euclidean, if ``euclidean=True``) embeddings
        from the v3 graph encoder.
    query_point : (D,) or (1, D)
        Single-query embedding from ``QueryToBall``. One query per
        scoring call; for multiple queries loop at the caller.
    c : float or Tensor
        Curvature. Ignored when ``euclidean=True``.
    euclidean : bool
        If True, use L2 distance instead of hyperbolic ``dist_p``.

    Returns
    -------
    scores : (N,)
        Real-valued, can be negative. Pass directly to
        ``MetricAccumulator.add``.
    """
    if query_point.dim() == 2:
        if query_point.size(0) != 1:
            raise ValueError(
                f"query_point must be a single point (D,) or (1, D); "
                f"got shape {tuple(query_point.shape)}"
            )
        query_point = query_point.squeeze(0)
    if query_point.dim() != 1:
        raise ValueError(
            f"query_point must be 1-D (D,); got shape {tuple(query_point.shape)}"
        )
    N = node_embeddings.size(0)
    q_exp = query_point.unsqueeze(0).expand(N, -1)
    if euclidean:
        dist = (node_embeddings - q_exp).norm(dim=-1, p=2)
    else:
        dist = P.dist(node_embeddings, q_exp, c, keepdim=False)
    return -dist

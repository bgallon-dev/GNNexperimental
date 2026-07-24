r"""Shared chunked Poincare-distance core for the v3.x eval scripts.

The shipped eval scripts used to materialize a full ``(N, N)`` distance
matrix via ``P.dist(emb.unsqueeze(1), emb.unsqueeze(0), c)``, which
broadcasts an ``(N, N, D)`` intermediate (~13 GB at N=5000, D=128, and
several such intermediates inside the Mobius ops) -> OOM at ~2-3k
nodes, long before the O(edges) encoder strains.

This module promotes the already-validated chunked core from
``profile_node_scale.py`` into one place so every eval script can use
it. **Bit-exactness contract:** at ``N <= EXACT_PAIR_NODE_CAP`` (every
node size any reported v3.1 / scaling result was produced at -
synthetic 50-500, real 400, the 24/100/250/500-graph ladders) the
chunked path returns results **identical** to the old full-matrix path:

  * ``chunked_topk`` scores each row against all N with the *same*
    ``P.dist`` broadcast, the *same* self-``inf`` mask, and the *same*
    ``torch.topk(..., largest=False)`` (sorted) -> identical indices and
    tie-break. It only ever holds a ``(block, N, D)`` intermediate.
  * ``exact_pair_dists`` appends ``d[r, gi+1:]`` per global row, i.e.
    row-major upper triangle == the ``np.triu_indices(N, 1)`` order the
    distribution consumers (median / percentile / ``<tau`` / spearman)
    already iterate; those stats are order-invariant anyway.
  * ``silhouette_row_block`` yields exact full rows (just blocked), so
    silhouette's arbitrary per-row subset indexing is unchanged.

Sampling (``sampled_pair_dists``) triggers only **above** the cap, i.e.
beyond any size that has ever been reported. ``pairwise_distance_matrix``
keeps the old exact full-matrix behaviour at/below the cap and raises a
loud (ASCII) error above it instead of silently OOM-ing.
"""

from __future__ import annotations

from typing import Iterator, Union

import numpy as np
import torch
from torch import Tensor

from ..modelsv2.layers import poincare_ops as P

Curvature = Union[float, Tensor]

# Above this node count, exact all-pairs passes are replaced by sampling
# (collapse/percentile/spearman) — beyond every reported result size.
EXACT_PAIR_NODE_CAP = 1500
MAX_SAMPLED_PAIRS = 400_000


def block(n: int) -> int:
    """Row-block size keeping the ``block x N x D`` intermediate
    ~<= 4M elements. Verbatim from profile_node_scale._block."""
    return int(max(16, min(256, 4_000_000 // max(n, 1))))


def chunked_topk(emb: Tensor, k: int, c: Curvature, euclidean: bool) -> np.ndarray:
    """Top-k nearest (excluding self) per node, computed row-block-wise.

    Returns ``(N, k)`` int64 — never materializes the full ``(N, N)``
    matrix. Bit-identical to
    ``D=full; D.fill_diagonal_(inf); topk(D, k, largest=False).indices``.
    Verbatim from profile_node_scale._chunked_topk.
    """
    N = emb.size(0)
    k = int(min(k, N - 1))
    B = block(N)
    out = torch.empty((N, k), dtype=torch.long)
    for s in range(0, N, B):
        e = min(s + B, N)
        u = emb[s:e].unsqueeze(1)            # (b,1,D)
        v = emb.unsqueeze(0)                 # (1,N,D)
        d = (torch.cdist(emb[s:e], emb, p=2) if euclidean
             else P.dist(u, v, c, keepdim=False))   # (b,N)
        idx = torch.arange(s, e)
        d[torch.arange(e - s), idx] = float("inf")   # mask self
        out[s:e] = torch.topk(d, k, largest=False).indices
    return out.numpy()


def sampled_pair_dists(emb: Tensor, c: Curvature, euclidean: bool,
                       n_pairs: int, rng) -> np.ndarray:
    """Distances over a random sample of node pairs (distribution stats
    when N is too large for an exact all-pairs pass). Verbatim from
    profile_node_scale._sampled_pair_dists."""
    N = emb.size(0)
    i = rng.integers(0, N, size=n_pairs)
    j = rng.integers(0, N, size=n_pairs)
    m = i != j
    i, j = i[m], j[m]
    a = emb[i]
    b = emb[j]
    if euclidean:
        return (a - b).norm(dim=-1).numpy()
    return P.dist(a, b, c, keepdim=False).numpy()


def exact_pair_dists(emb: Tensor, c: Curvature, euclidean: bool) -> np.ndarray:
    """All upper-triangle pair distances, row-chunked (exact; small N).

    Concatenation order is row-major upper triangle == the order
    ``np.triu_indices(N, 1)`` produces, so any order-invariant stat
    (median/percentile/frac-below-tau/spearman) matches the old
    full-matrix off-diagonal exactly. Verbatim from
    profile_node_scale._exact_pair_dists.
    """
    N = emb.size(0)
    B = block(N)
    chunks = []
    for s in range(0, N, B):
        e = min(s + B, N)
        u = emb[s:e].unsqueeze(1)
        v = emb.unsqueeze(0)
        d = (torch.cdist(emb[s:e], emb, p=2) if euclidean
             else P.dist(u, v, c, keepdim=False)).numpy()
        for r in range(e - s):
            gi = s + r
            chunks.append(d[r, gi + 1:])
    return np.concatenate(chunks) if chunks else np.zeros(0)


def offdiag_pair_dists(emb: Tensor, c: Curvature, euclidean: bool,
                       rng, cap: int = EXACT_PAIR_NODE_CAP,
                       n_sampled: int = MAX_SAMPLED_PAIRS) -> tuple[np.ndarray, str]:
    """Off-diagonal pair distances for distribution stats: exact upper
    triangle at ``N <= cap`` (bit-exact vs the old full-matrix
    off-diagonal), random-sampled above it. Returns ``(dists, mode)``
    where ``mode`` in {"exact", "sampled(<n>)"} for transparency."""
    if emb.size(0) <= cap:
        return exact_pair_dists(emb, c, euclidean), "exact"
    return (sampled_pair_dists(emb, c, euclidean, n_sampled, rng),
            f"sampled({n_sampled})")


def pairwise_distance_matrix(emb: Tensor, c: Curvature,
                             euclidean: bool) -> Tensor:
    """Full ``(N, N)`` distance matrix — the EXACT old behaviour of
    ``intrinsic_eval._pairwise_distance_matrix``.

    Kept only for the small-N consumers that genuinely need the whole
    matrix (e.g. ``retrieval_ops.graph_far_geometry_near`` which returns
    specific ``(i, j)`` pairs). Above ``EXACT_PAIR_NODE_CAP`` it raises
    a loud error instead of silently OOM-ing — large-N callers must use
    ``chunked_topk`` / ``offdiag_pair_dists`` / ``silhouette_row_block``.
    """
    N = emb.size(0)
    if N > EXACT_PAIR_NODE_CAP:
        raise RuntimeError(
            f"pairwise_distance_matrix called with N={N} > "
            f"EXACT_PAIR_NODE_CAP={EXACT_PAIR_NODE_CAP}: a full (N,N,D) "
            f"matrix would allocate ~{N*N*emb.size(1)*4/1e9:.1f} GB per "
            f"intermediate. Use chunked_topk / offdiag_pair_dists / "
            f"silhouette_row_block instead (see distance_ops docstring)."
        )
    if euclidean:
        return torch.cdist(emb, emb, p=2)
    u = emb.unsqueeze(1)  # (N, 1, D)
    v = emb.unsqueeze(0)  # (1, N, D)
    return P.dist(u, v, c, keepdim=False)  # (N, N)


def silhouette_row_block(
    emb: Tensor, c: Curvature, euclidean: bool
) -> Iterator[tuple[slice, np.ndarray]]:
    """Yield ``(row_slice, D_block)`` where ``D_block`` is the exact
    distance of rows ``[s:e]`` to all N nodes, row-blocked. Lets
    ``silhouette_score`` keep its arbitrary per-row subset indexing
    (``D[i, members].mean()``) bit-exactly with bounded memory.

    ``D_block`` rows are global-index-aligned: ``D_block[i - s, j]`` ==
    ``dist(node i, node j)`` for ``i in range(s, e)``, ``j in range(N)``
    — identical values to the old full ``D[i, j]``.
    """
    N = emb.size(0)
    B = block(N)
    for s in range(0, N, B):
        e = min(s + B, N)
        if euclidean:
            d = torch.cdist(emb[s:e], emb, p=2)
        else:
            u = emb[s:e].unsqueeze(1)
            v = emb.unsqueeze(0)
            d = P.dist(u, v, c, keepdim=False)
        yield slice(s, e), d.detach().cpu().numpy()

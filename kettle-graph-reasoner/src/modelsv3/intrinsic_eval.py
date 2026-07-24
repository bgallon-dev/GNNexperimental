r"""Intrinsic embedding-quality metrics for KGR v3.

Independent of the task head — tests whether the embedding itself is
useful, not whether a downstream head happens to work. Per the v3 plan,
if downstream nDCG@10 looks fine but intrinsic metrics are random, the
head is papering over a pathological embedding.

Three metrics:

1. ``silhouette_score`` — Poincaré-aware silhouette on node-type labels.
   Implemented from scratch (no sklearn dependency). Measures cluster
   compactness vs. separation.
2. ``nn_edge_precision_at_k`` — for each node, what fraction of top-k
   nearest embeddings are its graph out-neighbors. Tests whether the
   embedding preserves the edge structure used as a positive signal.
3. ``nn_label_purity_at_k`` — for each node, what fraction of top-k
   nearest embeddings share its node-type label. Tests whether the
   embedding respects categorical structure not directly wired into
   the feature input.

All metrics support both hyperbolic (``euclidean=False``, default) and
Euclidean (``euclidean=True``) embeddings so the same harness runs
against both arms.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from torch import Tensor

from ..modelsv2.layers import poincare_ops as P
from .distance_ops import chunked_topk, silhouette_row_block
from .distance_ops import pairwise_distance_matrix as _pairwise_distance_matrix

Curvature = Union[float, Tensor]

# _pairwise_distance_matrix is now the shared, cap-guarded full-matrix
# helper (re-exported for any external importer of this name). The
# O(N^2) consumers below were rewired to chunked_topk /
# silhouette_row_block, which are bit-identical at N <= the cap.
__all__ = ["silhouette_score", "nn_edge_precision_at_k",
           "nn_label_purity_at_k", "_pairwise_distance_matrix"]


def silhouette_score(
    embeddings: Tensor,
    labels: Tensor,
    c: Curvature = 1.0,
    euclidean: bool = False,
) -> dict:
    r"""Silhouette score — mean of ``(b_i - a_i) / max(a_i, b_i)``
    across nodes, where ``a_i`` is the mean distance from node ``i`` to
    other members of its cluster and ``b_i`` is the mean distance to
    the nearest other cluster.

    Labels must be an integer-typed ``(N,)`` tensor. Nodes whose label
    occurs only once (singleton clusters) are skipped; those whose
    nearest other cluster is empty are also skipped. If all nodes are
    skipped, the score is reported as ``NaN``.

    Returns a dict with keys ``mean``, ``n_evaluated``, ``n_clusters``
    so callers can see how much of the graph actually contributed.
    """
    if embeddings.dim() != 2:
        raise ValueError(f"embeddings must be (N, D); got shape {tuple(embeddings.shape)}")
    if labels.dim() != 1 or labels.size(0) != embeddings.size(0):
        raise ValueError("labels must be (N,) aligned with embeddings")

    N = embeddings.size(0)
    y = labels.detach().cpu().numpy().astype(np.int64)
    unique_labels = np.unique(y[y >= 0])

    # Row-blocked exact distances (bit-identical to the old full D[i, :],
    # just never materializing the whole (N, N) matrix). Iteration order
    # over i is unchanged (contiguous ascending blocks), so s_vals is
    # appended in the same order -> identical mean.
    s_vals: list[float] = []
    for sl, D_block in silhouette_row_block(embeddings, c, euclidean):
        for i in range(sl.start, sl.stop):
            li = y[i]
            if li < 0:
                continue
            own_members = np.nonzero((y == li) & (np.arange(N) != i))[0]
            if own_members.size == 0:
                continue  # singleton cluster — silhouette undefined
            row = D_block[i - sl.start]
            a_i = float(row[own_members].mean())
            b_i = float("inf")
            for lj in unique_labels:
                if lj == li:
                    continue
                other_members = np.nonzero(y == lj)[0]
                if other_members.size == 0:
                    continue
                mean_dist = float(row[other_members].mean())
                if mean_dist < b_i:
                    b_i = mean_dist
            if not np.isfinite(b_i):
                continue
            denom = max(a_i, b_i)
            if denom < 1e-12:
                continue
            s_vals.append((b_i - a_i) / denom)

    if not s_vals:
        return {"mean": float("nan"), "n_evaluated": 0, "n_clusters": int(unique_labels.size)}
    return {
        "mean": float(np.mean(s_vals)),
        "n_evaluated": len(s_vals),
        "n_clusters": int(unique_labels.size),
    }


def nn_edge_precision_at_k(
    embeddings: Tensor,
    edge_index: Tensor,
    k: int = 5,
    c: Curvature = 1.0,
    euclidean: bool = False,
) -> dict:
    r"""For each node ``i``, precision@k = fraction of the top-k closest
    embeddings (excluding self) that are outgoing neighbors of ``i``
    in ``edge_index``.

    Intuition: if the encoder preserves the graph structure used as
    positive signal in stage A, this is high. Near-random embedding
    → precision ~ ``avg_out_degree / N``.

    Nodes with zero out-neighbors contribute 0 and are counted in the
    denominator — the metric answers "how well does the embedding
    recover edge structure across the graph."
    """
    if embeddings.dim() != 2:
        raise ValueError(f"embeddings must be (N, D); got shape {tuple(embeddings.shape)}")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be (2, E)")
    N = embeddings.size(0)
    k = int(min(k, N - 1))

    # Row-chunked top-k, self-masked — bit-identical to
    # D=full; D.fill_diagonal_(inf); topk(D,k,largest=False).indices.
    topk_np = chunked_topk(embeddings, k, c, euclidean)  # (N, k) int64

    # Build out-neighbor sets per source node.
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    neighbors: list[set[int]] = [set() for _ in range(N)]
    for s, d in zip(src, dst):
        neighbors[int(s)].add(int(d))
    hits = np.zeros(N, dtype=np.float64)
    for i in range(N):
        nbrs = neighbors[i]
        if not nbrs:
            continue
        hits[i] = sum(1 for j in topk_np[i] if int(j) in nbrs) / k

    mean_avg_deg = float(np.mean([len(s) for s in neighbors]))
    random_baseline = min(1.0, mean_avg_deg / max(N - 1, 1))
    return {
        "mean_precision": float(hits.mean()),
        "k": k,
        "random_baseline": random_baseline,
        "mean_out_degree": mean_avg_deg,
    }


def nn_label_purity_at_k(
    embeddings: Tensor,
    labels: Tensor,
    k: int = 5,
    c: Curvature = 1.0,
    euclidean: bool = False,
) -> dict:
    r"""For each node ``i``, purity@k = fraction of the top-k closest
    embeddings (excluding self) that share ``labels[i]``. Random
    baseline is ``(cluster_size - 1) / (N - 1)``; a useful embedding
    scores well above that.

    ``labels`` here are typically the integer node-type labels recovered
    from ``x[:, 0:12].argmax(...)``. ``-1`` (unlabeled) nodes are
    skipped.
    """
    if embeddings.dim() != 2:
        raise ValueError(f"embeddings must be (N, D); got shape {tuple(embeddings.shape)}")
    if labels.dim() != 1 or labels.size(0) != embeddings.size(0):
        raise ValueError("labels must be (N,) aligned with embeddings")

    N = embeddings.size(0)
    k = int(min(k, N - 1))

    topk = chunked_topk(embeddings, k, c, euclidean)  # (N, k), self excluded
    y = labels.detach().cpu().numpy().astype(np.int64)

    evaluated: list[float] = []
    for i in range(N):
        li = y[i]
        if li < 0:
            continue
        shared = sum(1 for j in topk[i] if int(y[int(j)]) == int(li))
        evaluated.append(shared / k)

    # Random baseline: expected share of same-label among k uniform draws.
    _, counts = np.unique(y[y >= 0], return_counts=True)
    total = int(counts.sum())
    if total > 1:
        random_baseline = float(np.sum(counts * (counts - 1)) / (total * (total - 1)))
    else:
        random_baseline = 0.0
    return {
        "mean_purity": float(np.mean(evaluated)) if evaluated else float("nan"),
        "k": k,
        "n_evaluated": len(evaluated),
        "random_baseline": random_baseline,
    }


def try_umap_coords(
    embeddings: Tensor,
    euclidean: bool = False,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 0,
) -> np.ndarray | None:
    r"""Optional UMAP projection for manual inspection. Returns ``None``
    if umap-learn isn't installed — the caller decides whether to
    continue without the dump.

    Hyperbolic embeddings are projected via ``logmap0`` to a Euclidean
    tangent view before UMAP, since UMAP assumes a metric space and
    Poincaré ``dist_p`` there would require a precomputed distance
    matrix path. The tangent approximation is informative for clustering
    structure near origin — the exact geometry near the boundary is lost.
    """
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError:
        return None
    coords_tensor = (
        embeddings
        if euclidean
        else P.logmap0(embeddings, 1.0)  # curvature agnostic for viz
    )
    coords = coords_tensor.detach().cpu().numpy().astype(np.float32)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    return reducer.fit_transform(coords)

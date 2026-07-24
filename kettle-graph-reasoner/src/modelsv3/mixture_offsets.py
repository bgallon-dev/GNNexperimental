r"""K-point mixture query head, prototype-fit (no gradients).

The pool-retrieval plumbing fix that followed the multi-modality
discovery (runs/probe_true_point_ceiling): same-case answers form 2-3
clusters at task-consistent positions in the ANCHOR'S GYRO-FRAME.
Fitting K offsets per task by kmeans over training positives (a
prototype/generative fit) transfers to held-out cases; discriminative
gradient training of the same geometry does NOT (0.03 — the
sampled-negative shift kills query-geometry heads, again).

Measured (tutorstructure, frozen v1.0, held-out nonlocal pool ndcg@10):
0.097 all-tasks (vs blend 0.095, anchor_emb 0.074, bfs 0.070); 0.169 on
multi-positive tasks; IMPORT_DEPENDENCY 0.284 (vs anchor_emb 0.000).
Single-offset control collapses to 0.015-0.030 — K>1 is load-bearing.
Scoring = K ANN lookups; index property preserved.

Usage:
    V = fit_mixture_offsets(emb, train_cases_by_task, c)   # task -> (K,d)
    sc = mixture_score(emb, anchor_row, cand_rows, V[task], c)
"""

from __future__ import annotations

import torch

from ..modelsv2.layers import poincare_ops as P
from .distance_scoring import score_from_embeddings

K_DEFAULT = 3


def kmeans(x: torch.Tensor, k: int, iters: int = 15) -> torch.Tensor:
    k = min(k, x.shape[0])
    cents = x[torch.randperm(x.shape[0])[:k]].clone()
    for _ in range(iters):
        asg = torch.cdist(x, cents).argmin(1)
        for j in range(k):
            m = asg == j
            if m.any():
                cents[j] = x[m].mean(0)
    return cents


def anchor_frame_offsets(emb: torch.Tensor, anchor: int,
                         rows: torch.Tensor, c) -> torch.Tensor:
    """Tangent offsets of ``rows`` in ``anchor``'s gyro-frame."""
    return P.logmap0(P.mobius_add(-emb[anchor], emb[rows], c), c)


def fit_mixture_offsets(emb: torch.Tensor,
                        cases_by_task: dict[str, list[tuple[int, list[int]]]],
                        c, k: int = K_DEFAULT) -> dict[str, torch.Tensor]:
    """``cases_by_task``: task -> [(anchor_row, positive_rows), ...].
    Returns task -> (k, d) tangent offsets. Zero gradients; seconds."""
    out = {}
    for task, cases in cases_by_task.items():
        cents = []
        for anchor, pos in cases:
            if not pos:
                continue
            offs = anchor_frame_offsets(
                emb, anchor, torch.tensor(sorted(set(pos))), c)
            cents.append(kmeans(offs, k))
        if cents:
            out[task] = kmeans(torch.cat(cents, dim=0), k)
    return out


def mixture_query_points(emb: torch.Tensor, anchor: int,
                         V: torch.Tensor, c) -> torch.Tensor:
    """(k, d) ball points: anchor gyro-translated by each offset."""
    return P.mobius_add(emb[anchor].unsqueeze(0), P.expmap0(V, c), c)


def mixture_score(emb: torch.Tensor, anchor: int, cand_rows: torch.Tensor,
                  V: torch.Tensor, c) -> torch.Tensor:
    """max over K query points of -dist(candidate, point)."""
    qps = mixture_query_points(emb, anchor, V, c)
    return torch.stack(
        [score_from_embeddings(emb[cand_rows], q, c=c) for q in qps]
    ).max(0).values

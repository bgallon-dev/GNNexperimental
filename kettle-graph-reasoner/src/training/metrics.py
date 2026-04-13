r"""Retrieval metrics on per-node relevance scores.

Labels are continuous in [0, 1]. For precision/recall we threshold at
0.5 to form a binary relevance set; nDCG uses the continuous labels
directly. All metrics are computed per (graph, task) and then averaged.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor


def _topk_indices(scores: Tensor, k: int) -> Tensor:
    k = min(k, scores.numel())
    return torch.topk(scores, k=k, largest=True).indices


def precision_at_k(scores: Tensor, labels: Tensor, k: int, threshold: float = 0.5) -> float:
    if scores.numel() == 0:
        return 0.0
    topk = _topk_indices(scores, k)
    relevant = (labels[topk] >= threshold).float()
    return relevant.mean().item()


def recall_at_k(scores: Tensor, labels: Tensor, k: int, threshold: float = 0.5) -> float:
    total_relevant = (labels >= threshold).sum().item()
    if total_relevant == 0:
        return 0.0
    topk = _topk_indices(scores, k)
    hit = (labels[topk] >= threshold).sum().item()
    return hit / total_relevant


def ndcg_at_k(scores: Tensor, labels: Tensor, k: int) -> float:
    if scores.numel() == 0:
        return 0.0
    k = min(k, scores.numel())
    order = torch.argsort(scores, descending=True)[:k]
    gains = labels[order]
    discounts = torch.tensor(
        [1.0 / math.log2(i + 2) for i in range(k)],
        dtype=gains.dtype,
        device=gains.device,
    )
    dcg = (gains * discounts).sum().item()

    ideal, _ = torch.sort(labels, descending=True)
    ideal = ideal[:k]
    idcg = (ideal * discounts[: ideal.numel()]).sum().item()
    if idcg <= 0:
        return 0.0
    return dcg / idcg


class MetricAccumulator:
    """Aggregate per-task metrics, optionally broken out by task_type."""

    def __init__(self, ks: Iterable[int] = (5, 10, 20)) -> None:
        self.ks = tuple(ks)
        self._rows: list[tuple[int, dict[str, float]]] = []

    def add(self, scores: Tensor, labels: Tensor, task_type: int) -> None:
        row: dict[str, float] = {}
        for k in self.ks:
            row[f"p@{k}"] = precision_at_k(scores, labels, k)
            row[f"r@{k}"] = recall_at_k(scores, labels, k)
            row[f"ndcg@{k}"] = ndcg_at_k(scores, labels, k)
        self._rows.append((task_type, row))

    def summary(self) -> dict:
        if not self._rows:
            return {}
        keys = list(self._rows[0][1].keys())
        n = len(self._rows)
        out: dict = {"overall": {k: sum(r[k] for _, r in self._rows) / n for k in keys}}
        by_type: dict[int, list[dict[str, float]]] = {}
        for t, r in self._rows:
            by_type.setdefault(t, []).append(r)
        out["by_task_type"] = {
            t: {k: sum(r[k] for r in rows) / len(rows) for k in keys}
            for t, rows in sorted(by_type.items())
        }
        return out

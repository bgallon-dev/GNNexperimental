r"""Code-graph ranking metrics (fold-aware, for leave-one-repo-out CV).

Reuses the tested ranking math in ``src/training/metrics.py``
(``ndcg_at_k``, ``recall_at_k``) and adds the two metrics that file
doesn't have: MRR and the positive-vs-hard-negative score margin.
Rows are accumulated per (fold, task, split, mode); the headline CV
number is the macro-average over folds (each held-out repo weighted
equally), with per-cell detail kept for the report.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..training.metrics import ndcg_at_k, recall_at_k


def mrr(scores: Tensor, labels: Tensor, threshold: float = 0.5) -> float:
    if scores.numel() == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    rel = labels[order] >= threshold
    hit = torch.nonzero(rel, as_tuple=False)
    if hit.numel() == 0:
        return 0.0
    return 1.0 / float(hit[0].item() + 1)


def pos_hardneg_margin(
    scores: Tensor, labels: Tensor, hardneg_mask: Tensor
) -> float:
    """mean(score | positive) - mean(score | hard-negative). Model-only
    diagnostic; not meaningful for the distance-scale baselines."""
    pos = labels >= 0.5
    if pos.sum() == 0 or hardneg_mask.sum() == 0:
        return 0.0
    return float(scores[pos].mean().item() - scores[hardneg_mask].mean().item())


class CodeMetricAccumulator:
    """Keyed by (fold, task, split, mode); each ``add`` is one case."""

    KEYS = ("ndcg@10", "mrr", "r@10", "r@50", "pos_hardneg_margin")

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str, str, str], list[dict]] = {}

    def add(
        self,
        fold: str,
        task: str,
        split: str,
        mode: str,
        scores: Tensor,
        labels: Tensor,
        hardneg_mask: Tensor,
        locality: str = "na",
    ) -> None:
        scores = scores.detach().cpu().float()
        labels = labels.detach().cpu().float()
        hardneg_mask = hardneg_mask.detach().cpu().bool()
        row = {
            "ndcg@10": ndcg_at_k(scores, labels, 10),
            "mrr": mrr(scores, labels),
            "r@10": recall_at_k(scores, labels, 10),
            "r@50": recall_at_k(scores, labels, 50),
            "pos_hardneg_margin": pos_hardneg_margin(
                scores, labels, hardneg_mask
            ),
            # anchor-adjacency of the case's nearest positive; drives the
            # de-localized breakdown below. "na" for cases with no scorable
            # positive (kept out of the local/nonlocal split).
            "locality": locality,
        }
        self._rows.setdefault((fold, task, split, mode), []).append(row)

    @staticmethod
    def _avg(rows: list[dict]) -> dict:
        return {
            k: sum(r[k] for r in rows) / len(rows)
            for k in CodeMetricAccumulator.KEYS
        } | {"n": len(rows)}

    @staticmethod
    def _macro(cells: list[dict]) -> dict:
        # mean over folds of each per-fold cell mean (equal-weighted folds)
        out = {
            k: sum(c[k] for c in cells) / len(cells)
            for k in CodeMetricAccumulator.KEYS
        }
        out["folds"] = len(cells)
        out["n"] = sum(c["n"] for c in cells)
        return out

    def summary(self) -> dict:
        if not self._rows:
            return {}
        by_cell = {
            f"{f}|{t}|{s}|{m}": self._avg(rows)
            for (f, t, s, m), rows in sorted(self._rows.items())
        }
        # per-fold (task,split,mode) and (split,mode) means
        fold_tsm: dict[tuple, dict] = {}
        fold_sm: dict[tuple, list[dict]] = {}
        for (f, t, s, m), rows in self._rows.items():
            fold_tsm[(f, t, s, m)] = self._avg(rows)
            fold_sm.setdefault((f, s, m), []).extend(rows)
        fold_sm_avg = {k: self._avg(v) for k, v in fold_sm.items()}

        cv_tsm: dict[str, list[dict]] = {}
        for (f, t, s, m), cell in fold_tsm.items():
            cv_tsm.setdefault(f"{t}|{s}|{m}", []).append(cell)
        cv_sm: dict[str, list[dict]] = {}
        for (f, s, m), cell in fold_sm_avg.items():
            cv_sm.setdefault(f"{s}|{m}", []).append(cell)

        # De-localized breakdown: bucket every case by the anchor-adjacency
        # of its nearest positive ("local" = hop<=1, "nonlocal" = hop>=2)
        # before pooling. Anchor-BFS is near-oracle by construction on local
        # cases, so a fair "beats heuristic" number reads off the nonlocal
        # bucket. Aggregation mirrors cv_by_split_mode (per-fold cell mean,
        # then equal-weighted macro over folds).
        fold_sml: dict[tuple, list[dict]] = {}
        for (f, t, s, m), rows in self._rows.items():
            for r in rows:
                fold_sml.setdefault(
                    (f, s, m, r.get("locality", "na")), []
                ).append(r)
        fold_sml_avg = {k: self._avg(v) for k, v in fold_sml.items()}
        cv_sml: dict[str, list[dict]] = {}
        for (f, s, m, loc), cell in fold_sml_avg.items():
            cv_sml.setdefault(f"{s}|{m}|{loc}", []).append(cell)

        return {
            "cv_by_split_mode": {
                k: self._macro(v) for k, v in sorted(cv_sm.items())
            },
            "cv_by_task_split_mode": {
                k: self._macro(v) for k, v in sorted(cv_tsm.items())
            },
            "cv_by_locality_mode": {
                k: self._macro(v) for k, v in sorted(cv_sml.items())
            },
            "by_fold_task_split_mode": by_cell,
        }


class ClassificationMetricAccumulator:
    """Per-(fold, task, split) classification accumulator. Mirrors the
    shape of ``CodeMetricAccumulator.summary`` so the harness report can
    stack ranking + classification blocks side by side.

    Records ``(pred, label, n_labels)`` per case and computes accuracy +
    macro-F1 over those at summary time. CV aggregation = mean over
    per-fold cell means (equal-weighted folds), same as ranking."""

    KEYS = ("accuracy", "macro_f1")

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str, str], list[tuple[int, int, int]]] = {}

    def add(self, fold: str, task: str, split: str,
            pred: int, label: int, n_labels: int) -> None:
        self._rows.setdefault((fold, task, split), []).append(
            (int(pred), int(label), int(n_labels))
        )

    @staticmethod
    def _cell_metrics(rows: list[tuple[int, int, int]]) -> dict:
        if not rows:
            return {k: 0.0 for k in ClassificationMetricAccumulator.KEYS} | {"n": 0}
        preds = [r[0] for r in rows]
        labels = [r[1] for r in rows]
        n_labels = rows[0][2]
        # Accuracy
        n = len(rows)
        acc = sum(1 for p, l in zip(preds, labels) if p == l) / n
        # Macro-F1: per-class F1 averaged. Undefined classes (no preds
        # AND no labels) get F1=0 and DO count in the macro mean to
        # punish silent class collapse.
        f1s = []
        for c in range(n_labels):
            tp = sum(1 for p, l in zip(preds, labels) if p == c and l == c)
            fp = sum(1 for p, l in zip(preds, labels) if p == c and l != c)
            fn = sum(1 for p, l in zip(preds, labels) if p != c and l == c)
            denom = 2 * tp + fp + fn
            f1s.append(2 * tp / denom if denom > 0 else 0.0)
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        return {"accuracy": acc, "macro_f1": macro_f1, "n": n}

    @staticmethod
    def _macro(cells: list[dict]) -> dict:
        out = {
            k: sum(c[k] for c in cells) / len(cells)
            for k in ClassificationMetricAccumulator.KEYS
        }
        out["folds"] = len(cells)
        out["n"] = sum(c["n"] for c in cells)
        return out

    def summary(self) -> dict:
        if not self._rows:
            return {}
        by_cell = {
            f"{f}|{t}|{s}": self._cell_metrics(rows)
            for (f, t, s), rows in sorted(self._rows.items())
        }
        fold_ts: dict[tuple, dict] = {}
        fold_s: dict[tuple, list[tuple[int, int, int]]] = {}
        for (f, t, s), rows in self._rows.items():
            fold_ts[(f, t, s)] = self._cell_metrics(rows)
            fold_s.setdefault((f, s), []).extend(rows)
        fold_s_avg = {k: self._cell_metrics(v) for k, v in fold_s.items()}

        cv_ts: dict[str, list[dict]] = {}
        for (f, t, s), cell in fold_ts.items():
            cv_ts.setdefault(f"{t}|{s}", []).append(cell)
        cv_s: dict[str, list[dict]] = {}
        for (f, s), cell in fold_s_avg.items():
            cv_s.setdefault(s, []).append(cell)

        return {
            "cv_by_split": {k: self._macro(v) for k, v in sorted(cv_s.items())},
            "cv_by_task_split": {
                k: self._macro(v) for k, v in sorted(cv_ts.items())
            },
            "by_fold_task_split": by_cell,
        }

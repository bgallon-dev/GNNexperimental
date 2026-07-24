r"""T4 node-strategy evaluation harness (pre-registered; plan T4).

Written and frozen BEFORE any graded fixture exists -- the analysis code
cannot have been tuned to the data. When graded fixtures arrive this runs
push-button; until then it is exercised only on synthetic labels in
tests.

Pre-registered rules implemented here (Docs/EVIDENCE_WORKSPACE_PLAN.md):
- Metric: nDCG@10 per question, graded relevance (gain 2^g - 1).
- Two observable lanes per question:
    ``core``               -- the full ranking judged against all grades
    ``nonlocal_discovery`` -- ranking and judgments filtered to hop >= 2
- Cells: 4 families x 2 lanes. A cell with fewer than 8 held-out cases is
  ``insufficient_evidence`` and ships BFS.
- Otherwise: paired per-question deltas (candidate - bfs);
  10,000 deterministic paired bootstrap resamples (seed from config,
  default 1729) -> 95% percentile CI; paired sign-flip permutation test;
  Holm correction across the 8 cells.
- Ship the candidate ONLY when CI lower bound > 0 AND Holm-adjusted
  p < 0.05. Everything else (including inconclusive) ships BFS.
- Aggregate numbers are reported descriptively; they never control
  adoption.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

BOOTSTRAP_SEED = 1729
N_RESAMPLES = 10_000
MIN_CELL_CASES = 8
ALPHA = 0.05
NDCG_K = 10
FAMILIES = ("policy_change", "provenance", "institutional_bridge",
            "contested_claim")
LANES = ("core", "nonlocal_discovery")


# -- metric ---------------------------------------------------------------------

def ndcg_at_k(ranking: Sequence[str], grades: Mapping[str, int],
              k: int = NDCG_K) -> float:
    """Graded nDCG@k. ``grades`` maps public key -> 0..3 (absent = 0).
    Returns 0.0 when there are no positive grades (defined, not NaN)."""
    dcg = sum((2 ** grades.get(key, 0) - 1) / math.log2(i + 2)
              for i, key in enumerate(ranking[:k]))
    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2)
               for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def lane_score(ranking: Sequence[str], graded_nodes: Sequence[Mapping],
               lane: str, k: int = NDCG_K) -> float | None:
    """One question's score in one lane.

    ``ranking`` must already be the LANE ranking (the ranking side filters
    ``nonlocal_discovery`` to hop >= 2 candidates -- observable at serving
    time; ground-truth locality is never used for routing). This function
    only filters the JUDGMENTS: the discovery lane is judged against
    nonlocal positives alone. Returns None when the lane has no judged
    positives (the question contributes no case to that cell)."""
    if lane == "nonlocal_discovery":
        graded = {n["public_key"]: int(n["grade"]) for n in graded_nodes
                  if n.get("hop_locality") == "nonlocal"}
    else:
        graded = {n["public_key"]: int(n["grade"]) for n in graded_nodes}
    if not any(g > 0 for g in graded.values()):
        return None
    return ndcg_at_k(ranking, graded, k)


# -- statistics -------------------------------------------------------------------

def paired_bootstrap_ci(deltas: Sequence[float], *,
                        n_resamples: int = N_RESAMPLES,
                        seed: int = BOOTSTRAP_SEED,
                        level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of the mean paired delta (deterministic seed)."""
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_resamples))
    lo_i = int((1 - level) / 2 * n_resamples)
    hi_i = n_resamples - 1 - lo_i
    return means[lo_i], means[hi_i]


def paired_permutation_pvalue(deltas: Sequence[float], *,
                              n_resamples: int = N_RESAMPLES,
                              seed: int = BOOTSTRAP_SEED) -> float:
    """Two-sided sign-flip permutation test on the mean paired delta."""
    rng = random.Random(seed + 1)          # independent stream
    observed = abs(sum(deltas) / len(deltas))
    hits = 0
    for _ in range(n_resamples):
        m = sum(d if rng.random() < 0.5 else -d for d in deltas) / len(deltas)
        if abs(m) >= observed - 1e-15:
            hits += 1
    return (hits + 1) / (n_resamples + 1)


def holm_adjust(pvals: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped 1)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[key] = running
    return out


# -- the decision table -----------------------------------------------------------

@dataclass
class CellResult:
    family: str
    lane: str
    n_cases: int
    status: str                       # ship_candidate | ship_bfs |
    #                                   insufficient_evidence
    mean_delta: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    p_holm: float | None = None
    per_question: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def evaluate_strategy_table(
        cases: Sequence[Mapping[str, Any]], *,
        candidate: str, baseline: str = "bfs",
        seed: int = BOOTSTRAP_SEED,
        min_cell_cases: int = MIN_CELL_CASES) -> dict[str, Any]:
    """The pre-registered T4 decision procedure.

    ``cases``: one dict per question:
      {question_id, family, graded_nodes: [essential_nodes...],
       rankings: {strategy: {lane: [public_key ranked...]}}}

    Returns the full table: per-cell verdicts + descriptive aggregate.
    """
    deltas_by_cell: dict[tuple[str, str], dict[str, float]] = {
        (f, l): {} for f in FAMILIES for l in LANES}
    for case in cases:
        fam = case["family"]
        for lane in LANES:
            s_cand = lane_score(case["rankings"][candidate][lane],
                                case["graded_nodes"], lane)
            s_base = lane_score(case["rankings"][baseline][lane],
                                case["graded_nodes"], lane)
            if s_cand is None or s_base is None:
                continue
            deltas_by_cell[(fam, lane)][case["question_id"]] = \
                s_cand - s_base

    cells: dict[str, CellResult] = {}
    pvals: dict[str, float] = {}
    for (fam, lane), dq in deltas_by_cell.items():
        name = f"{fam}/{lane}"
        deltas = list(dq.values())
        if len(deltas) < min_cell_cases:
            cells[name] = CellResult(fam, lane, len(deltas),
                                     "insufficient_evidence",
                                     per_question=dq)
            continue
        lo, hi = paired_bootstrap_ci(deltas, seed=seed)
        p = paired_permutation_pvalue(deltas, seed=seed)
        cells[name] = CellResult(
            fam, lane, len(deltas), "pending_holm",
            mean_delta=sum(deltas) / len(deltas),
            ci_low=lo, ci_high=hi, p_value=p, per_question=dq)
        pvals[name] = p

    adjusted = holm_adjust(pvals)
    for name, cell in cells.items():
        if cell.status != "pending_holm":
            continue
        cell.p_holm = adjusted[name]
        ships = (cell.ci_low is not None and cell.ci_low > 0
                 and cell.p_holm < ALPHA)
        cell.status = "ship_candidate" if ships else "ship_bfs"

    all_deltas = [d for c in cells.values() for d in c.per_question.values()]
    return {
        "candidate": candidate,
        "baseline": baseline,
        "bootstrap_seed": seed,
        "n_resamples": N_RESAMPLES,
        "alpha": ALPHA,
        "min_cell_cases": min_cell_cases,
        "cells": {name: c.to_dict() for name, c in sorted(cells.items())},
        "strategy_table": {name: (candidate if c.status == "ship_candidate"
                                  else baseline)
                           for name, c in sorted(cells.items())},
        "aggregate_descriptive_only": {
            "n_cases": len(all_deltas),
            "mean_delta": (sum(all_deltas) / len(all_deltas)
                           if all_deltas else None),
        },
    }

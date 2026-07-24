r"""Cross-seed aggregation of the embedding-collapse investigation.

Takes the 6 ``collapse_diag.json`` files produced by
``investigate_collapse.py`` and answers:

    (1) Within each geometry (hyperbolic / euclidean), how consistent
        are the collapsed pairs across seeds?  Computed as pairwise
        Jaccard similarity of the collapsed-pair sets, per graph,
        then averaged.

    (2) What fraction of nodes participate in at least one collapsed
        pair, aggregated per geometry?

    (3) Cross-geometry: do hyperbolic and Euclidean collapse the same
        pairs?  If so, collapse is input-driven.  If not, it's
        geometry/training-dynamics-driven.

Usage
-----
    py src/modelsv3/compare_collapse.py \\
        --hyp-results runs/v3_hyp_compute_seed{0,1,2}/collapse_diag.json \\
        --euc-results runs/v3_euc_compute_seed{0,1,2}/collapse_diag.json \\
        --out         runs/compare_collapse.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path


def _load(paths: list[str]) -> list[dict]:
    return [json.load(open(p)) for p in paths]


def _pairs_to_tuples(pairs: list[list[int]]) -> set[tuple[int, int]]:
    return {tuple(p) for p in pairs}  # type: ignore[misc]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def _nodes_in_pairs(pairs: set[tuple[int, int]]) -> set[int]:
    out: set[int] = set()
    for i, j in pairs:
        out.add(i)
        out.add(j)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hyp-results", type=str, nargs="+", required=True)
    p.add_argument("--euc-results", type=str, nargs="+", required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    hyp_arms = _load(args.hyp_results)
    euc_arms = _load(args.euc_results)

    # graph_ids that are present in all 6 arms
    graph_ids: set[int] = None  # type: ignore[assignment]
    for arm in hyp_arms + euc_arms:
        gs = {int(k) for k in arm["q5_collapsed_pair_sets"].keys()}
        graph_ids = gs if graph_ids is None else (graph_ids & gs)
    if graph_ids is None:
        graph_ids = set()
    graph_ids_sorted = sorted(graph_ids)

    def pairset(arm: dict, gi: int) -> set[tuple[int, int]]:
        return _pairs_to_tuples(arm["q5_collapsed_pair_sets"][str(gi)])

    # (1) within-geometry Jaccard
    within_geom: dict[str, dict] = {}
    for name, arms in (("hyperbolic", hyp_arms), ("euclidean", euc_arms)):
        per_graph_jaccard: list[float] = []
        per_graph_sizes: list[int] = []
        per_graph_union_sizes: list[int] = []
        for gi in graph_ids_sorted:
            sets_at_gi = [pairset(a, gi) for a in arms]
            sizes_at_gi = [len(s) for s in sets_at_gi]
            per_graph_sizes.extend(sizes_at_gi)
            union_at_gi = set().union(*sets_at_gi) if sets_at_gi else set()
            per_graph_union_sizes.append(len(union_at_gi))
            pair_jaccards: list[float] = []
            for a, b in itertools.combinations(sets_at_gi, 2):
                pair_jaccards.append(_jaccard(a, b))
            if pair_jaccards:
                per_graph_jaccard.append(statistics.mean(pair_jaccards))
        within_geom[name] = {
            "n_graphs": len(per_graph_jaccard),
            "mean_within_seed_jaccard_per_graph": (
                statistics.mean(per_graph_jaccard) if per_graph_jaccard else float("nan")
            ),
            "median_within_seed_jaccard_per_graph": (
                statistics.median(per_graph_jaccard) if per_graph_jaccard else float("nan")
            ),
            "per_graph_jaccard": per_graph_jaccard,
            "per_arm_pair_counts": {
                "mean": statistics.mean(per_graph_sizes) if per_graph_sizes else 0.0,
                "max": max(per_graph_sizes) if per_graph_sizes else 0,
            },
        }

    # (2) node-level collapse participation rate
    participation: dict[str, dict] = {}
    for name, arms in (("hyperbolic", hyp_arms), ("euclidean", euc_arms)):
        per_arm_graph_rates: list[float] = []
        for arm in arms:
            # sum across graphs: total collapsed-participating nodes /
            # total nodes. Graph size taken from per-graph Q1-Q2.
            graph_sizes = {
                int(g["graph_idx"]): g.get("n_nodes", 0)
                for g in arm["per_graph_q1_q2"]
            }
            per_graph = []
            for gi in graph_ids_sorted:
                nodes_collapsed = _nodes_in_pairs(pairset(arm, gi))
                n = graph_sizes.get(gi, 0)
                if n > 0:
                    per_graph.append(len(nodes_collapsed) / n)
            if per_graph:
                per_arm_graph_rates.append(statistics.mean(per_graph))
        participation[name] = {
            "per_arm_mean_participation_rate": per_arm_graph_rates,
            "mean_across_arms": (
                statistics.mean(per_arm_graph_rates) if per_arm_graph_rates else float("nan")
            ),
        }

    # (3) cross-geometry Jaccard
    cross_geom_jaccard: list[float] = []
    for gi in graph_ids_sorted:
        hyp_pairs: set[tuple[int, int]] = set()
        euc_pairs: set[tuple[int, int]] = set()
        for a in hyp_arms:
            hyp_pairs |= pairset(a, gi)
        for a in euc_arms:
            euc_pairs |= pairset(a, gi)
        cross_geom_jaccard.append(_jaccard(hyp_pairs, euc_pairs))
    cross_geom = {
        "per_graph_jaccard_hyp_vs_euc_union": cross_geom_jaccard,
        "mean": (
            statistics.mean(cross_geom_jaccard)
            if cross_geom_jaccard else float("nan")
        ),
    }

    results = {
        "n_graphs_common": len(graph_ids_sorted),
        "within_geometry": within_geom,
        "node_participation_rate": participation,
        "cross_geometry": cross_geom,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    _print_table(results)
    return 0


def _print_table(r: dict) -> None:
    print()
    print("=" * 80)
    print(f"Collapse cross-seed aggregation ({r['n_graphs_common']} common graphs)")
    print("=" * 80)

    print("\nWithin-geometry Jaccard (how consistent are collapsed pairs across seeds?)")
    print("-" * 80)
    for name in ("hyperbolic", "euclidean"):
        w = r["within_geometry"][name]
        print(
            f"  {name:<12}  mean_jaccard={w['mean_within_seed_jaccard_per_graph']:.4f}  "
            f"median={w['median_within_seed_jaccard_per_graph']:.4f}  "
            f"(avg n_pairs/arm={w['per_arm_pair_counts']['mean']:.1f})"
        )

    print("\nNode-level collapse participation rate (fraction of nodes in ≥1 collapsed pair)")
    print("-" * 80)
    for name in ("hyperbolic", "euclidean"):
        p = r["node_participation_rate"][name]
        arms = p["per_arm_mean_participation_rate"]
        print(
            f"  {name:<12}  mean={p['mean_across_arms']:.4f}  "
            f"per_arm={[f'{x:.4f}' for x in arms]}"
        )

    print("\nCross-geometry Jaccard (do hyperbolic and Euclidean collapse the same pairs?)")
    print("-" * 80)
    print(f"  hyp_union vs euc_union  mean={r['cross_geometry']['mean']:.4f}")
    print(f"  per-graph: {[f'{x:.3f}' for x in r['cross_geometry']['per_graph_jaccard_hyp_vs_euc_union']]}")


if __name__ == "__main__":
    sys.exit(main())

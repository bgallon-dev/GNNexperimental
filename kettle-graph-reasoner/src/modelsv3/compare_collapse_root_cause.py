r"""Cross-seed aggregation of the collapse root-cause investigation.

Combines the six ``collapse_root_cause.json`` files produced by
``investigate_collapse_root_cause.py`` and produces:

    (1) Per-geometry Q1 / Q2 / Q3 summaries averaged across seeds.
    (2) Node-level agreement across seeds: for each graph, which
        specific nodes appear in the collapsed set in all 3 seeds?
        If the intersection is large (close to the per-seed size),
        collapse is deterministic and input-driven.

Usage
-----
    py src/modelsv3/compare_collapse_root_cause.py \\
        --hyp-results runs/v3_hyp_compute_seed{0,1,2}/collapse_root_cause.json \\
        --euc-results runs/v3_euc_compute_seed{0,1,2}/collapse_root_cause.json \\
        --out         runs/compare_collapse_root_cause.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _load(paths: list[str]) -> list[dict]:
    return [json.load(open(p)) for p in paths]


def _mean_across_seeds(arms: list[dict], path: list[str | int]) -> float:
    vals = []
    for arm in arms:
        cur = arm
        ok = True
        for p in path:
            if isinstance(p, str):
                if not isinstance(cur, dict) or p not in cur:
                    ok = False
                    break
                cur = cur[p]
            else:
                if not isinstance(cur, list) or len(cur) <= p:
                    ok = False
                    break
                cur = cur[p]
        if ok and isinstance(cur, (int, float)) and cur == cur:
            vals.append(float(cur))
    return statistics.mean(vals) if vals else float("nan")


def _intersection_analysis(arms: list[dict]) -> dict:
    """For each graph, compute the intersection and union of
    collapsed-node-id sets across the 3 seeds.  Returns mean sizes
    (normalised by per-seed mean size to get a stability coefficient)."""
    # Get all graph ids that appear in every arm
    all_graph_ids: set[int] = None  # type: ignore[assignment]
    for arm in arms:
        gs = {int(k) for k in arm["collapsed_node_ids_per_graph"].keys()}
        all_graph_ids = gs if all_graph_ids is None else (all_graph_ids & gs)
    if all_graph_ids is None:
        all_graph_ids = set()

    per_graph_stats: list[dict] = []
    for gi in sorted(all_graph_ids):
        sets = [set(arm["collapsed_node_ids_per_graph"][str(gi)]) for arm in arms]
        sizes = [len(s) for s in sets]
        inter = sets[0].copy()
        union = sets[0].copy()
        for s in sets[1:]:
            inter &= s
            union |= s
        mean_size = statistics.mean(sizes) if sizes else 0
        per_graph_stats.append({
            "graph_idx": gi,
            "per_seed_size_mean": mean_size,
            "intersection_size": len(inter),
            "union_size": len(union),
            "jaccard": (len(inter) / len(union)) if union else 1.0,
            "intersection_over_mean_size": (
                len(inter) / mean_size if mean_size > 0 else float("nan")
            ),
        })

    return {
        "per_graph": per_graph_stats,
        "summary": {
            "mean_intersection_over_mean_size": statistics.mean(
                [g["intersection_over_mean_size"] for g in per_graph_stats
                 if g["intersection_over_mean_size"] == g["intersection_over_mean_size"]]
            ) if per_graph_stats else float("nan"),
            "mean_jaccard": statistics.mean(
                [g["jaccard"] for g in per_graph_stats]
            ) if per_graph_stats else float("nan"),
            "n_graphs": len(per_graph_stats),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hyp-results", type=str, nargs="+", required=True)
    p.add_argument("--euc-results", type=str, nargs="+", required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    hyp_arms = _load(args.hyp_results)
    euc_arms = _load(args.euc_results)

    results: dict = {}
    for name, arms in (("hyperbolic", hyp_arms), ("euclidean", euc_arms)):
        # Q1
        q1 = {
            "norm_collapsed_mean": _mean_across_seeds(
                arms, ["aggregate", "q1_norm_collapsed_mean"]
            ),
            "norm_non_collapsed_mean": _mean_across_seeds(
                arms, ["aggregate", "q1_norm_non_collapsed_mean"]
            ),
            "cohens_d_collapsed_vs_not": _mean_across_seeds(
                arms, ["aggregate", "q1_norm_collapsed_vs_not_cohens_d"]
            ),
        }
        if name == "hyperbolic":
            q1["frac_collapsed_within_r_1e-3"] = _mean_across_seeds(
                arms, ["aggregate", "q1_frac_collapsed_within_r_1e-3"]
            )

        # Q2
        q2_cohens: dict[str, list[float]] = {}
        for arm in arms:
            for k, v in arm["aggregate"].get("q2_cohens_d_mean_across_graphs", {}).items():
                q2_cohens.setdefault(k, []).append(v)
        q2 = {
            "cohens_d_mean": {k: statistics.mean(v) for k, v in q2_cohens.items()},
            "type_entropy_collapsed": _mean_across_seeds(
                arms, ["aggregate", "q2_type_entropy_collapsed"]
            ),
            "type_entropy_non_collapsed": _mean_across_seeds(
                arms, ["aggregate", "q2_type_entropy_non_collapsed"]
            ),
            "layer_entropy_collapsed": _mean_across_seeds(
                arms, ["aggregate", "q2_layer_entropy_collapsed"]
            ),
            "layer_entropy_non_collapsed": _mean_across_seeds(
                arms, ["aggregate", "q2_layer_entropy_non_collapsed"]
            ),
        }

        # Q3
        q3: dict[str, dict] = {}
        for arm in arms:
            for block, d in arm["aggregate"].get("q3_cosine_similarity", {}).items():
                q3.setdefault(block, {"collapsed": [], "random": []})
                if d["collapsed_mean"] == d["collapsed_mean"]:
                    q3[block]["collapsed"].append(d["collapsed_mean"])
                if d["random_mean"] == d["random_mean"]:
                    q3[block]["random"].append(d["random_mean"])
        q3_summary = {
            block: {
                "collapsed_mean": statistics.mean(d["collapsed"]) if d["collapsed"] else float("nan"),
                "random_mean": statistics.mean(d["random"]) if d["random"] else float("nan"),
                "elevation": (
                    (statistics.mean(d["collapsed"]) / statistics.mean(d["random"]))
                    if d["collapsed"] and d["random"] and statistics.mean(d["random"]) != 0
                    else float("nan")
                ),
            }
            for block, d in q3.items()
        }

        # Intersection of collapsed nodes across seeds
        inter = _intersection_analysis(arms)

        results[name] = {
            "q1": q1, "q2": q2, "q3": q3_summary,
            "collapsed_node_intersection_across_seeds": inter,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    _print_report(results)
    return 0


def _print_report(r: dict) -> None:
    print()
    print("=" * 96)
    print("ROOT-CAUSE INVESTIGATION — cross-seed aggregation")
    print("=" * 96)

    for name in ("hyperbolic", "euclidean"):
        data = r[name]
        print()
        print("─" * 96)
        print(name.upper())
        print("─" * 96)

        q1 = data["q1"]
        print(f"\nQ1 — Manifold location")
        print(f"  norm collapsed        = {q1['norm_collapsed_mean']:.4f}")
        print(f"  norm non-collapsed    = {q1['norm_non_collapsed_mean']:.4f}")
        print(f"  Cohen's d             = {q1['cohens_d_collapsed_vs_not']:+.3f}")
        if "frac_collapsed_within_r_1e-3" in q1:
            print(f"  fraction r<1e-3       = {q1['frac_collapsed_within_r_1e-3']:.4f}")

        q2 = data["q2"]
        print(f"\nQ2 — Feature-level Cohen's d (collapsed vs not)")
        for k, v in q2["cohens_d_mean"].items():
            print(f"  {k:<26}  d = {v:+.3f}")
        print(f"\n  Type entropy  collapsed={q2['type_entropy_collapsed']:.3f}  "
              f"non_collapsed={q2['type_entropy_non_collapsed']:.3f}  "
              f"(max possible: 2.485 for 12 types)")
        print(f"  Layer entropy collapsed={q2['layer_entropy_collapsed']:.3f}  "
              f"non_collapsed={q2['layer_entropy_non_collapsed']:.3f}  "
              f"(max possible: 1.386 for 4 layers)")

        q3 = data["q3"]
        print(f"\nQ3 — Input cosine similarity (collapsed vs random pairs)")
        print(f"  {'block':<12}  {'collapsed':>12}  {'random':>12}  {'elevation':>10}")
        for block, d in q3.items():
            cm = d["collapsed_mean"]
            rm = d["random_mean"]
            el = d["elevation"]
            print(f"  {block:<12}  {cm:+12.4f}  {rm:+12.4f}  {el:+10.2f}×")

        inter = data["collapsed_node_intersection_across_seeds"]["summary"]
        print(f"\nNode-level cross-seed intersection (does the same node collapse in every seed?)")
        print(f"  mean intersection / mean per-seed size = {inter['mean_intersection_over_mean_size']:.3f}")
        print(f"  mean Jaccard across 3 seeds            = {inter['mean_jaccard']:.3f}")
        print(f"  (1.0 = perfectly deterministic; 0 = no common node)")


if __name__ == "__main__":
    sys.exit(main())

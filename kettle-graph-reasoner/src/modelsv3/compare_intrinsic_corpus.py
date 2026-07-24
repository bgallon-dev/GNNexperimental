r"""Experiment 2.1 comparison — aggregates intrinsic metrics across
seeds and geometries.

Reads the per-checkpoint ``intrinsic_corpus.json`` files written by
``eval_intrinsic_corpus.py`` and produces:
    1. Per-architecture three-seed summary (mean ± std across seeds
       of the per-graph mean).
    2. Head-to-head per-graph comparison: for each val graph and each
       metric, which geometry produced the better value? Aggregated
       into a "wins" count across the 50 val graphs, averaged over
       seed pairings.
    3. Printed table + JSON artefact.

Usage
-----
    py src/modelsv3/compare_intrinsic_corpus.py \\
        --hyp-results runs/v3_hyp_compute_seed{0,1,2}/intrinsic_corpus.json \\
        --euc-results runs/v3_euc_compute_seed{0,1,2}/intrinsic_corpus.json \\
        --out         runs/compare_intrinsic.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _load(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        with open(p, "r") as f:
            out.append(json.load(f))
    return out


def _seed_means(arms: list[dict], metric_key: str) -> list[float]:
    """Extract the per-seed mean of a metric across graphs."""
    vals = []
    for r in arms:
        s = r["summary"][metric_key]
        if s["mean"] == s["mean"]:  # not NaN
            vals.append(s["mean"])
    return vals


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "n": 1}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "n": len(values),
    }


def _per_graph_map(arm: dict) -> dict[int, dict]:
    """Index per-graph entries by graph_idx."""
    return {int(r["graph_idx"]): r for r in arm["per_graph"]}


def _head_to_head(
    hyp_arms: list[dict], euc_arms: list[dict]
) -> dict:
    """Per-graph head-to-head: for each metric, fraction of graphs
    where hyperbolic > Euclidean. Uses the mean across seeds for each
    arm to reduce noise, then compares per-graph."""
    metrics = [
        ("silhouette_mean", "higher_better"),
        ("edge_prec_mean", "higher_better"),
        ("label_purity_mean", "higher_better"),
    ]

    # Mean of the metric per graph across the 3 seeds for each arm.
    hyp_by_graph: dict[int, dict[str, list[float]]] = {}
    for arm in hyp_arms:
        for g in arm["per_graph"]:
            gi = int(g["graph_idx"])
            rec = hyp_by_graph.setdefault(gi, {})
            for mkey, _ in metrics:
                rec.setdefault(mkey, []).append(g[mkey])
    euc_by_graph: dict[int, dict[str, list[float]]] = {}
    for arm in euc_arms:
        for g in arm["per_graph"]:
            gi = int(g["graph_idx"])
            rec = euc_by_graph.setdefault(gi, {})
            for mkey, _ in metrics:
                rec.setdefault(mkey, []).append(g[mkey])

    common_graphs = sorted(set(hyp_by_graph) & set(euc_by_graph))
    h2h: dict[str, dict] = {}
    per_graph_diffs: dict[int, dict[str, float]] = {
        g: {} for g in common_graphs
    }
    for mkey, _direction in metrics:
        hyp_wins = 0
        euc_wins = 0
        ties = 0
        diffs: list[float] = []
        nan_skipped = 0
        for g in common_graphs:
            h_vals = hyp_by_graph[g][mkey]
            e_vals = euc_by_graph[g][mkey]
            h_vals = [v for v in h_vals if v == v]
            e_vals = [v for v in e_vals if v == v]
            if not h_vals or not e_vals:
                nan_skipped += 1
                continue
            h_mean = statistics.mean(h_vals)
            e_mean = statistics.mean(e_vals)
            diff = h_mean - e_mean
            per_graph_diffs[g][mkey] = diff
            diffs.append(diff)
            if h_mean > e_mean:
                hyp_wins += 1
            elif h_mean < e_mean:
                euc_wins += 1
            else:
                ties += 1
        h2h[mkey] = {
            "hyp_wins": hyp_wins,
            "euc_wins": euc_wins,
            "ties": ties,
            "nan_skipped": nan_skipped,
            "n_graphs_compared": hyp_wins + euc_wins + ties,
            "mean_diff": (statistics.mean(diffs) if diffs else float("nan")),
            "median_diff": (statistics.median(diffs) if diffs else float("nan")),
        }
    return {"metrics": h2h, "per_graph_diffs": per_graph_diffs}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hyp-results", type=str, nargs="+", required=True)
    p.add_argument("--euc-results", type=str, nargs="+", required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    hyp_arms = _load(args.hyp_results)
    euc_arms = _load(args.euc_results)

    metric_keys = ("silhouette", "edge_precision_at_k", "label_purity_at_k")
    agg = {"hyperbolic": {}, "euclidean": {}}
    for arm_name, arms in (("hyperbolic", hyp_arms), ("euclidean", euc_arms)):
        for mkey in metric_keys:
            seed_means = _seed_means(arms, mkey)
            agg[arm_name][mkey] = {
                "per_seed_means": seed_means,
                "across_seeds": _summarize(seed_means),
            }

    h2h = _head_to_head(hyp_arms, euc_arms)

    # Per-architecture random baselines (average across seeds, since
    # they're graph-structure-dependent and vary mildly).
    rb = {}
    for arm_name, arms in (("hyperbolic", hyp_arms), ("euclidean", euc_arms)):
        rb[arm_name] = {
            "edge_prec": statistics.mean(
                r["summary"]["random_baseline_edge_prec_mean"] for r in arms
            ),
            "label_purity": statistics.mean(
                r["summary"]["random_baseline_label_purity_mean"] for r in arms
            ),
        }

    results = {
        "aggregated": agg,
        "head_to_head": h2h,
        "random_baselines": rb,
        "inputs": {
            "hyperbolic": args.hyp_results,
            "euclidean": args.euc_results,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    _print_table(results)
    return 0


def _print_table(results: dict) -> None:
    print()
    print("=" * 88)
    print("EXPERIMENT 2.1 — Intrinsic metrics across val corpus (3 seeds each)")
    print("=" * 88)

    print(f"\n{'metric':<24} {'hyperbolic':>26} {'euclidean':>26}")
    print("-" * 88)
    for mkey_display, mkey in (
        ("silhouette", "silhouette"),
        ("edge_prec@5", "edge_precision_at_k"),
        ("label_purity@5", "label_purity_at_k"),
    ):
        hyp = results["aggregated"]["hyperbolic"][mkey]["across_seeds"]
        euc = results["aggregated"]["euclidean"][mkey]["across_seeds"]
        hyp_str = f"{hyp['mean']:+.4f} ± {hyp['std']:.4f} (n={hyp['n']})"
        euc_str = f"{euc['mean']:+.4f} ± {euc['std']:.4f} (n={euc['n']})"
        print(f"{mkey_display:<24} {hyp_str:>26} {euc_str:>26}")
    print(
        f"{'random edge_prec':<24} "
        f"{results['random_baselines']['hyperbolic']['edge_prec']:>26.4f} "
        f"{results['random_baselines']['euclidean']['edge_prec']:>26.4f}"
    )
    print(
        f"{'random label_purity':<24} "
        f"{results['random_baselines']['hyperbolic']['label_purity']:>26.4f} "
        f"{results['random_baselines']['euclidean']['label_purity']:>26.4f}"
    )

    print("\n" + "-" * 88)
    print("Head-to-head: per-graph wins (hyp_mean > euc_mean across seeds)")
    print("-" * 88)
    print(f"{'metric':<24} {'hyp wins':>10} {'euc wins':>10} {'ties':>8} "
          f"{'mean Δ (hyp - euc)':>22} {'median Δ':>12}")
    for mkey_display, mkey in (
        ("silhouette", "silhouette_mean"),
        ("edge_prec@5", "edge_prec_mean"),
        ("label_purity@5", "label_purity_mean"),
    ):
        h = results["head_to_head"]["metrics"][mkey]
        print(
            f"{mkey_display:<24} {h['hyp_wins']:>10d} {h['euc_wins']:>10d} "
            f"{h['ties']:>8d} {h['mean_diff']:>+22.4f} {h['median_diff']:>+12.4f}"
        )


if __name__ == "__main__":
    sys.exit(main())

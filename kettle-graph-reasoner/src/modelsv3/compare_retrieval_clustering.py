r"""Experiment 2.3c comparison — aggregates clustering ARI results
across 3 hyperbolic + 3 Euclidean seeds.

Usage
-----
    py src/modelsv3/compare_retrieval_clustering.py \\
        --hyp-results runs/v3_hyp_compute_seed{0,1,2}/retrieval_clustering.json \\
        --euc-results runs/v3_euc_compute_seed{0,1,2}/retrieval_clustering.json \\
        --out         runs/compare_retrieval_clustering.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


LABEL_CONFIGS = (("type", 12), ("layer", 4))


def _load(paths: list[str]) -> list[dict]:
    return [json.load(open(p)) for p in paths]


def _summarize(values: list[float]) -> dict:
    clean = [v for v in values if v == v]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(clean) == 1:
        return {"mean": clean[0], "std": 0.0, "n": 1}
    return {"mean": statistics.mean(clean),
            "std": statistics.stdev(clean), "n": len(clean)}


def _per_graph_head_to_head(
    hyp_arms: list[dict], euc_arms: list[dict],
    condition: str, label: str
) -> dict:
    def build(arms: list[dict]) -> dict[int, list[float]]:
        out: dict[int, list[float]] = {}
        for arm in arms:
            for g in arm["per_graph"]:
                gi = int(g["graph_idx"])
                val = g["results"].get(condition, {}).get(label, {}).get("ari_mean")
                if val is not None and val == val:
                    out.setdefault(gi, []).append(float(val))
        return out

    hyp_m = build(hyp_arms)
    euc_m = build(euc_arms)
    common = sorted(set(hyp_m) & set(euc_m))
    hyp_wins = euc_wins = ties = 0
    diffs: list[float] = []
    for gi in common:
        if not hyp_m[gi] or not euc_m[gi]:
            continue
        hm = statistics.mean(hyp_m[gi])
        em = statistics.mean(euc_m[gi])
        diff = hm - em
        diffs.append(diff)
        if hm == em:
            ties += 1
        elif hm > em:
            hyp_wins += 1
        else:
            euc_wins += 1
    return {
        "direction": "higher_better",
        "hyp_wins": hyp_wins,
        "euc_wins": euc_wins,
        "ties": ties,
        "n_compared": len(common),
        "mean_diff_hyp_minus_euc": statistics.mean(diffs) if diffs else float("nan"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hyp-results", type=str, nargs="+", required=True)
    p.add_argument("--euc-results", type=str, nargs="+", required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    hyp_arms = _load(args.hyp_results)
    euc_arms = _load(args.euc_results)

    results: dict = {"aggregated": {}, "head_to_head": {}, "collapse_rates": {}}
    for cond in ("unfiltered", "filtered"):
        results["aggregated"][cond] = {"hyperbolic": {}, "euclidean": {}}
        results["head_to_head"][cond] = {}
        for arm_name, arms in (("hyperbolic", hyp_arms), ("euclidean", euc_arms)):
            for label, _k in LABEL_CONFIGS:
                seed_means = [
                    a["summary"][cond][label]["mean"]
                    for a in arms
                    if label in a["summary"][cond]
                    and a["summary"][cond][label]["n"] > 0
                ]
                results["aggregated"][cond][arm_name][label] = {
                    "per_seed": seed_means,
                    "across_seeds": _summarize(seed_means),
                }
        for label, _k in LABEL_CONFIGS:
            results["head_to_head"][cond][label] = _per_graph_head_to_head(
                hyp_arms, euc_arms, cond, label
            )

    for arm_name, arms in (("hyperbolic", hyp_arms), ("euclidean", euc_arms)):
        rates = [a.get("mean_collapse_rate", float("nan")) for a in arms]
        clean = [r for r in rates if r == r]
        results["collapse_rates"][arm_name] = (
            statistics.mean(clean) if clean else float("nan")
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    _print_table(results)
    return 0


def _print_table(r: dict) -> None:
    print()
    print("=" * 92)
    print("EXPERIMENT 2.3c — Clustering ARI (3 seeds each, k-means on embeddings)")
    print("=" * 92)

    print(f"\nCollapse rate (fraction of nodes in any collapsed pair):")
    for name in ("hyperbolic", "euclidean"):
        print(f"  {name:<12}  {r['collapse_rates'][name]:.4f}")

    for cond in ("unfiltered", "filtered"):
        print()
        print("-" * 92)
        print(f"{cond.upper()}")
        print("-" * 92)
        print(f"{'label':<10} {'k':>3}  {'hyperbolic':>24} {'euclidean':>24}")
        for label, k in LABEL_CONFIGS:
            h = r["aggregated"][cond]["hyperbolic"][label]["across_seeds"]
            e = r["aggregated"][cond]["euclidean"][label]["across_seeds"]
            hs = f"{h['mean']:+.4f} ± {h['std']:.4f}"
            es = f"{e['mean']:+.4f} ± {e['std']:.4f}"
            print(f"{label:<10} {k:>3}  {hs:>24} {es:>24}")

        print(f"\n{'Head-to-head':<22} {'direction':<14} "
              f"{'hyp wins':>10} {'euc wins':>10} {'ties':>6} {'mean Δ':>14}")
        for label, _k in LABEL_CONFIGS:
            h = r["head_to_head"][cond][label]
            print(f"  ARI({label}){'':<10}   {h['direction']:<14} "
                  f"{h['hyp_wins']:>10d} {h['euc_wins']:>10d} "
                  f"{h['ties']:>6d} {h['mean_diff_hyp_minus_euc']:>+14.4f}")


if __name__ == "__main__":
    sys.exit(main())

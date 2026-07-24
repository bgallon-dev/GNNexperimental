r"""Experiment 2.2 comparison — aggregates transfer eval across seeds.

Usage
-----
    py src/modelsv3/compare_transfer.py \\
        --hyp-results runs/v3_transfer_hyp_seed{0,1,2}/transfer_eval.json \\
        --euc-results runs/v3_transfer_euc_seed{0,1,2}/transfer_eval.json \\
        --out         runs/compare_transfer.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


CONDITIONS = ("seen_train", "unseen_train", "val")
METRICS = ("edge_prec@5", "label_purity@5", "silhouette")


def _load(paths: list[str]) -> list[dict]:
    return [json.load(open(p)) for p in paths]


def _seed_mean(arm: dict, cond: str, metric: str) -> float:
    block = arm["summary_per_condition"].get(cond, {}).get(metric, {})
    return block["mean"] if block and block.get("n", 0) > 0 else float("nan")


def _summarize(vals: list[float]) -> dict:
    clean = [v for v in vals if v == v]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(clean) == 1:
        return {"mean": clean[0], "std": 0.0, "n": 1}
    return {"mean": statistics.mean(clean),
            "std": statistics.stdev(clean), "n": len(clean)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hyp-results", type=str, nargs="+", required=True)
    p.add_argument("--euc-results", type=str, nargs="+", required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    hyp_arms = _load(args.hyp_results)
    euc_arms = _load(args.euc_results)

    results: dict = {
        "aggregated": {"hyperbolic": {}, "euclidean": {}},
        "transfer_gap": {},
    }
    for arm_name, arms in (("hyperbolic", hyp_arms), ("euclidean", euc_arms)):
        for cond in CONDITIONS:
            results["aggregated"][arm_name][cond] = {}
            for metric in METRICS:
                seed_means = [_seed_mean(a, cond, metric) for a in arms]
                results["aggregated"][arm_name][cond][metric] = {
                    "per_seed": seed_means,
                    "across_seeds": _summarize(seed_means),
                }
        results["transfer_gap"][arm_name] = {}
        for metric in METRICS:
            gaps = {}
            for cond in ("seen_train", "unseen_train"):
                diffs = []
                for a in arms:
                    s = _seed_mean(a, cond, metric)
                    v = _seed_mean(a, "val", metric)
                    if s == s and v == v:
                        diffs.append(s - v)
                gaps[f"{cond}_minus_val"] = _summarize(diffs)
            results["transfer_gap"][arm_name][metric] = gaps

    results["train_graphs_frac"] = {
        "hyperbolic": [a["train_graphs_frac"] for a in hyp_arms],
        "euclidean": [a["train_graphs_frac"] for a in euc_arms],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    _print_table(results)
    return 0


def _print_table(r: dict) -> None:
    print()
    print("=" * 96)
    print("EXPERIMENT 2.2 — Cross-graph transfer (3 seeds each geometry)")
    fracs_h = r["train_graphs_frac"]["hyperbolic"]
    fracs_e = r["train_graphs_frac"]["euclidean"]
    print(f"train_graphs_frac  hyperbolic={fracs_h}  euclidean={fracs_e}")
    print("=" * 96)

    for metric in METRICS:
        print(f"\n--- {metric} ---")
        print(f"{'condition':<14} {'hyperbolic':>26} {'euclidean':>26}")
        print("-" * 96)
        for cond in CONDITIONS:
            h = r["aggregated"]["hyperbolic"][cond][metric]["across_seeds"]
            e = r["aggregated"]["euclidean"][cond][metric]["across_seeds"]
            hs = f"{h['mean']:+.4f} ± {h['std']:.4f}"
            es = f"{e['mean']:+.4f} ± {e['std']:.4f}"
            print(f"{cond:<14} {hs:>26} {es:>26}")

    print("\n" + "-" * 96)
    print("Transfer gap — (condition − val) per metric")
    print("  Near zero  = generalizes.  Large positive = overfit to condition.")
    print("-" * 96)
    for metric in METRICS:
        print(f"\n{metric}:")
        for arm_name in ("hyperbolic", "euclidean"):
            gaps = r["transfer_gap"][arm_name][metric]
            s_gap = gaps["seen_train_minus_val"]["mean"]
            u_gap = gaps["unseen_train_minus_val"]["mean"]
            print(f"  {arm_name:<12}  (seen − val) = {s_gap:+.4f}   "
                  f"(unseen − val) = {u_gap:+.4f}")


if __name__ == "__main__":
    sys.exit(main())

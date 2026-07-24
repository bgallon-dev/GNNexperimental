r"""Experiment 2.3b comparison — aggregates geodesic midpoint retrieval
results across 3 hyperbolic + 3 Euclidean seeds.

Usage
-----
    py src/modelsv3/compare_retrieval_midpoint.py \\
        --hyp-results runs/v3_hyp_compute_seed{0,1,2}/retrieval_midpoint.json \\
        --euc-results runs/v3_euc_compute_seed{0,1,2}/retrieval_midpoint.json \\
        --out         runs/compare_retrieval_midpoint.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


K_VALUES = (1, 3, 5)


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


def _seed_mean(arm: dict, metric_key: str) -> float:
    block = arm["summary"].get(metric_key, {})
    return block["mean"] if block and block.get("n", 0) > 0 else float("nan")


def _per_graph_head_to_head(
    hyp_arms: list[dict], euc_arms: list[dict], metric_key: str
) -> dict:
    """Per-graph wins: hyp_mean_across_seeds vs euc_mean_across_seeds."""
    def build(arms: list[dict]) -> dict[int, list[float]]:
        out: dict[int, list[float]] = {}
        metric_k_key = metric_key.split("@")[0] + "_at_k" if "@" in metric_key else metric_key
        k_sub = metric_key.split("@")[1] if "@" in metric_key else None
        for arm in arms:
            for g in arm["per_graph"]:
                gi = int(g["graph_idx"])
                if "@" in metric_key:
                    val = g.get("path_hit_rate_at_k", {}).get(f"k={k_sub}", None)
                else:
                    val = g.get(metric_key, None)
                if val is not None and val == val:
                    out.setdefault(gi, []).append(float(val))
        return out

    hyp_m = build(hyp_arms)
    euc_m = build(euc_arms)
    common = sorted(set(hyp_m) & set(euc_m))

    hyp_wins = euc_wins = ties = 0
    diffs: list[float] = []
    # For path_hit_rate, higher is better. For mean_nn_hop_from_path, lower is better.
    lower_better = metric_key == "mean_nn_hop_from_path"
    for gi in common:
        if not hyp_m[gi] or not euc_m[gi]:
            continue
        hm = statistics.mean(hyp_m[gi])
        em = statistics.mean(euc_m[gi])
        diff = hm - em
        diffs.append(diff)
        if hm == em:
            ties += 1
        elif lower_better:
            if hm < em:
                hyp_wins += 1
            else:
                euc_wins += 1
        else:
            if hm > em:
                hyp_wins += 1
            else:
                euc_wins += 1

    return {
        "direction": "lower_better" if lower_better else "higher_better",
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

    results: dict = {"aggregated": {"hyperbolic": {}, "euclidean": {}}, "head_to_head": {}}
    metric_keys = [f"path_hit_rate@{k}" for k in K_VALUES] + [
        "random_baseline", "mean_nn_hop_from_path"
    ]
    for arm_name, arms in (("hyperbolic", hyp_arms), ("euclidean", euc_arms)):
        for mk in metric_keys:
            seed_means = [_seed_mean(a, mk) for a in arms]
            results["aggregated"][arm_name][mk] = {
                "per_seed": seed_means,
                "across_seeds": _summarize(seed_means),
            }
    for mk in metric_keys:
        results["head_to_head"][mk] = _per_graph_head_to_head(hyp_arms, euc_arms, mk)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    _print_table(results)
    return 0


def _print_table(r: dict) -> None:
    print()
    print("=" * 90)
    print("EXPERIMENT 2.3b — Geodesic midpoint retrieval (3 seeds each geometry)")
    print("=" * 90)
    print(f"\n{'metric':<28} {'hyperbolic':>26} {'euclidean':>26}")
    print("-" * 90)
    for mk in [f"path_hit_rate@{k}" for k in K_VALUES] + ["random_baseline", "mean_nn_hop_from_path"]:
        h = r["aggregated"]["hyperbolic"][mk]["across_seeds"]
        e = r["aggregated"]["euclidean"][mk]["across_seeds"]
        hs = f"{h['mean']:.4f} ± {h['std']:.4f}"
        es = f"{e['mean']:.4f} ± {e['std']:.4f}"
        print(f"{mk:<28} {hs:>26} {es:>26}")

    print("\n" + "-" * 90)
    print("Head-to-head per graph (hyp_mean across seeds vs euc_mean across seeds)")
    print("-" * 90)
    print(f"{'metric':<28} {'direction':<14} {'hyp wins':>10} {'euc wins':>10} "
          f"{'ties':>6} {'mean Δ':>14}")
    for mk in [f"path_hit_rate@{k}" for k in K_VALUES] + ["mean_nn_hop_from_path"]:
        h = r["head_to_head"][mk]
        print(f"{mk:<28} {h['direction']:<14} {h['hyp_wins']:>10d} "
              f"{h['euc_wins']:>10d} {h['ties']:>6d} "
              f"{h['mean_diff_hyp_minus_euc']:>+14.4f}")


if __name__ == "__main__":
    sys.exit(main())

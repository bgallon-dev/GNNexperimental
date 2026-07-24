r"""Experiment 2.3a (collapse-corrected) comparison — aggregates
NN retrieval quality with and without collapse filtering across 3
hyperbolic seeds and 3 Euclidean seeds.

Produces two comparison tables (one per condition) plus the head-to-
head per-graph wins count that mirrors the original compare_retrieval_nn
script.

Usage
-----
    py src/modelsv3/compare_retrieval_nn_filtered.py \\
        --hyp-results runs/v3_hyp_compute_seed{0,1,2}/retrieval_nn_filtered.json \\
        --euc-results runs/v3_euc_compute_seed{0,1,2}/retrieval_nn_filtered.json \\
        --out         runs/compare_retrieval_nn_filtered.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


DEFAULT_K = "k=5"
METRICS = (
    ("same_type_frac_mean", "higher_better"),
    ("same_layer_frac_mean", "higher_better"),
    ("hop_dist_mean", "lower_better"),
)


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


def _seed_mean(arm: dict, condition: str, k_key: str, metric: str) -> float:
    """Extract per-seed across-graphs mean for a given (condition, k, metric)."""
    block = (arm["summary_across_graphs"].get(condition, {})
                                        .get(k_key, {})
                                        .get(metric, {}))
    return block["mean"] if block else float("nan")


def _head_to_head(
    hyp_arms: list[dict], euc_arms: list[dict],
    condition: str, k_key: str
) -> dict:
    def build(arms: list[dict]) -> dict[int, dict[str, list[float]]]:
        out: dict[int, dict[str, list[float]]] = {}
        for arm in arms:
            for g in arm["per_graph"]:
                gi = int(g["graph_idx"])
                pk = g["metrics"].get(condition, {}).get(k_key, {})
                rec = out.setdefault(gi, {})
                for mname, _ in METRICS:
                    if mname in pk:
                        rec.setdefault(mname, []).append(pk[mname])
        return out

    hyp_map = build(hyp_arms)
    euc_map = build(euc_arms)
    common = sorted(set(hyp_map) & set(euc_map))
    h2h: dict[str, dict] = {}
    for mname, direction in METRICS:
        hyp_wins = euc_wins = ties = 0
        diffs: list[float] = []
        for gi in common:
            hv = hyp_map[gi].get(mname, [])
            ev = euc_map[gi].get(mname, [])
            if not hv or not ev:
                continue
            hm = statistics.mean(hv)
            em = statistics.mean(ev)
            diff = hm - em
            diffs.append(diff)
            if hm == em:
                ties += 1
            elif direction == "higher_better":
                if hm > em:
                    hyp_wins += 1
                else:
                    euc_wins += 1
            else:
                if hm < em:
                    hyp_wins += 1
                else:
                    euc_wins += 1
        h2h[mname] = {
            "direction": direction,
            "hyp_wins": hyp_wins,
            "euc_wins": euc_wins,
            "ties": ties,
            "n_compared": len(common),
            "mean_diff_hyp_minus_euc": statistics.mean(diffs) if diffs else float("nan"),
            "median_diff_hyp_minus_euc": statistics.median(diffs) if diffs else float("nan"),
        }
    return h2h


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hyp-results", type=str, nargs="+", required=True)
    p.add_argument("--euc-results", type=str, nargs="+", required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--k-key", type=str, default=DEFAULT_K)
    args = p.parse_args()

    hyp_arms = _load(args.hyp_results)
    euc_arms = _load(args.euc_results)

    results: dict = {"k_key": args.k_key, "conditions": {}}

    for condition in ("unfiltered", "filtered"):
        agg: dict[str, dict[str, dict]] = {"hyperbolic": {}, "euclidean": {}}
        for arm_name, arms in (("hyperbolic", hyp_arms), ("euclidean", euc_arms)):
            for mname, _ in METRICS:
                seed_means = [
                    _seed_mean(a, condition, args.k_key, mname) for a in arms
                ]
                agg[arm_name][mname] = {
                    "per_seed_means": seed_means,
                    "across_seeds": _summarize(seed_means),
                }
        h2h = _head_to_head(hyp_arms, euc_arms, condition, args.k_key)
        results["conditions"][condition] = {"aggregated": agg, "head_to_head": h2h}

    # Collapse stats (independent of condition)
    collapse_agg = {"hyperbolic": {}, "euclidean": {}}
    for name, arms in (("hyperbolic", hyp_arms), ("euclidean", euc_arms)):
        fracs = []
        excls = []
        for a in arms:
            cs = a.get("collapse_stats_aggregate", {}).get(args.k_key, {})
            if "mean_frac_seeds_affected" in cs:
                fracs.append(cs["mean_frac_seeds_affected"])
            if "total_exclusions_over_total_slots" in cs:
                excls.append(cs["total_exclusions_over_total_slots"])
        collapse_agg[name] = {
            "mean_frac_seeds_affected": (
                statistics.mean(fracs) if fracs else float("nan")
            ),
            "mean_exclusion_rate": (
                statistics.mean(excls) if excls else float("nan")
            ),
        }
    results["collapse_stats"] = collapse_agg

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    _print_table(results)
    return 0


def _print_table(r: dict) -> None:
    print()
    print("=" * 98)
    print(f"EXPERIMENT 2.3a (collapse-corrected) — NN retrieval, "
          f"{r['k_key']}, 3 seeds each geometry")
    print("=" * 98)

    # Collapse stats first (the context frame).
    print("\nCollapse stats (at tau = 1e-4 × graph median dist):")
    print("-" * 98)
    for name in ("hyperbolic", "euclidean"):
        cs = r["collapse_stats"][name]
        print(
            f"  {name:<12}  frac_seeds_affected_by_filter = "
            f"{cs['mean_frac_seeds_affected']:.4f}  "
            f"exclusion_rate = {cs['mean_exclusion_rate']:.4f}"
        )

    name_map = {
        "same_type_frac_mean": "same_type_frac",
        "same_layer_frac_mean": "same_layer_frac",
        "hop_dist_mean": "hop_dist",
    }

    for condition in ("unfiltered", "filtered"):
        cd = r["conditions"][condition]
        print()
        print("-" * 98)
        print(f"{condition.upper()}")
        print("-" * 98)
        print(f"{'metric':<22} {'hyperbolic':>24} {'euclidean':>24}")
        for mname, _ in METRICS:
            hyp = cd["aggregated"]["hyperbolic"][mname]["across_seeds"]
            euc = cd["aggregated"]["euclidean"][mname]["across_seeds"]
            hs = f"{hyp['mean']:+.4f} ± {hyp['std']:.4f}"
            es = f"{euc['mean']:+.4f} ± {euc['std']:.4f}"
            print(f"{name_map[mname]:<22} {hs:>24} {es:>24}")

        print(f"\n{'Head-to-head per graph':<22} {'direction':<14} "
              f"{'hyp wins':>10} {'euc wins':>10} {'ties':>6} {'mean Δ':>14}")
        for mname, direction in METRICS:
            h = cd["head_to_head"][mname]
            print(
                f"{name_map[mname]:<22} {direction:<14} "
                f"{h['hyp_wins']:>10d} {h['euc_wins']:>10d} "
                f"{h['ties']:>6d} {h['mean_diff_hyp_minus_euc']:>+14.4f}"
            )


if __name__ == "__main__":
    sys.exit(main())

r"""Experiment 2.3a comparison — aggregates NN retrieval quality across
seeds and geometries.

Reads six ``retrieval_nn.json`` files (three hyperbolic seeds + three
Euclidean seeds) produced by ``eval_retrieval_nn.py`` and produces:
    1. Per-geometry three-seed summary (mean ± std across seeds of the
       per-graph mean metric).
    2. Head-to-head per-graph wins: for each val graph, which geometry
       produced better ``same_type_frac``, lower ``hop_dist_mean``, and
       better ``same_layer_frac`` at k=5?

Usage
-----
    py src/modelsv3/compare_retrieval_nn.py \\
        --hyp-results runs/v3_hyp_compute_seed{0,1,2}/retrieval_nn.json \\
        --euc-results runs/v3_euc_compute_seed{0,1,2}/retrieval_nn.json \\
        --out         runs/compare_retrieval_nn.json
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


def _seed_mean(arm: dict, k_key: str, metric: str) -> float:
    block = arm["summary_across_graphs"].get(k_key, {}).get(metric)
    return block["mean"] if block else float("nan")


def _summarize(values: list[float]) -> dict:
    clean = [v for v in values if v == v]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(clean) == 1:
        return {"mean": clean[0], "std": 0.0, "n": 1}
    return {"mean": statistics.mean(clean),
            "std": statistics.stdev(clean), "n": len(clean)}


def _head_to_head(
    hyp_arms: list[dict], euc_arms: list[dict], k_key: str
) -> dict:
    """Per-graph: for each metric, average across seeds within each arm
    and compare per-graph."""
    def build(arms: list[dict]) -> dict[int, dict[str, list[float]]]:
        out: dict[int, dict[str, list[float]]] = {}
        for arm in arms:
            for g in arm["per_graph"]:
                gi = int(g["graph_idx"])
                pk = g["per_k"].get(k_key, {})
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
            else:  # lower_better
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

    agg: dict[str, dict[str, dict]] = {"hyperbolic": {}, "euclidean": {}}
    for arm_name, arms in (("hyperbolic", hyp_arms), ("euclidean", euc_arms)):
        for mname, _ in METRICS:
            seed_means = [_seed_mean(a, args.k_key, mname) for a in arms]
            agg[arm_name][mname] = {
                "per_seed_means": seed_means,
                "across_seeds": _summarize(seed_means),
            }

    h2h = _head_to_head(hyp_arms, euc_arms, args.k_key)
    rb = {
        "hyperbolic": {
            "same_type": statistics.mean(
                a["random_baselines_mean_across_graphs"]["same_type"]
                for a in hyp_arms
            ),
            "same_layer": statistics.mean(
                a["random_baselines_mean_across_graphs"]["same_layer"]
                for a in hyp_arms
            ),
            "hop_dist": statistics.mean(
                a["random_baselines_mean_across_graphs"]["hop_dist"]
                for a in hyp_arms
            ),
        },
        "euclidean": {
            "same_type": statistics.mean(
                a["random_baselines_mean_across_graphs"]["same_type"]
                for a in euc_arms
            ),
            "same_layer": statistics.mean(
                a["random_baselines_mean_across_graphs"]["same_layer"]
                for a in euc_arms
            ),
            "hop_dist": statistics.mean(
                a["random_baselines_mean_across_graphs"]["hop_dist"]
                for a in euc_arms
            ),
        },
    }

    results = {
        "k_key": args.k_key,
        "aggregated": agg,
        "head_to_head": h2h,
        "random_baselines": rb,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    _print_table(results)
    return 0


def _print_table(r: dict) -> None:
    print()
    print("=" * 90)
    print(f"EXPERIMENT 2.3a — NN retrieval quality (3 seeds each, {r['k_key']})")
    print("=" * 90)

    print(f"\n{'metric':<28} {'hyperbolic':>26} {'euclidean':>26}")
    print("-" * 90)
    name_map = {
        "same_type_frac_mean": "same_type_frac",
        "same_layer_frac_mean": "same_layer_frac",
        "hop_dist_mean": "hop_dist",
    }
    for mname, _ in METRICS:
        hyp = r["aggregated"]["hyperbolic"][mname]["across_seeds"]
        euc = r["aggregated"]["euclidean"][mname]["across_seeds"]
        hs = f"{hyp['mean']:+.4f} ± {hyp['std']:.4f}"
        es = f"{euc['mean']:+.4f} ± {euc['std']:.4f}"
        print(f"{name_map[mname]:<28} {hs:>26} {es:>26}")

    rb = r["random_baselines"]
    print(
        f"{'random same_type':<28} "
        f"{rb['hyperbolic']['same_type']:>26.4f} {rb['euclidean']['same_type']:>26.4f}"
    )
    print(
        f"{'random same_layer':<28} "
        f"{rb['hyperbolic']['same_layer']:>26.4f} {rb['euclidean']['same_layer']:>26.4f}"
    )
    print(
        f"{'random hop_dist':<28} "
        f"{rb['hyperbolic']['hop_dist']:>26.4f} {rb['euclidean']['hop_dist']:>26.4f}"
    )

    print("\n" + "-" * 90)
    print("Head-to-head: per-graph wins at k=5 (mean across seeds per geometry)")
    print("-" * 90)
    print(f"{'metric':<28} {'direction':<16} {'hyp wins':>10} {'euc wins':>10} "
          f"{'ties':>6} {'mean Δ':>14}")
    for mname, direction in METRICS:
        h = r["head_to_head"][mname]
        name = name_map[mname]
        print(
            f"{name:<28} {direction:<16} {h['hyp_wins']:>10d} {h['euc_wins']:>10d} "
            f"{h['ties']:>6d} {h['mean_diff_hyp_minus_euc']:>+14.4f}"
        )


if __name__ == "__main__":
    sys.exit(main())

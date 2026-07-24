r"""Architecture sweep analysis.

Reads the outputs of sweep_architecture.py and produces:

 1. Main table — each (hidden_dim, num_layers) cell, 3-seed mean ± std
    for nDCG@10, edge_prec@5, collapse_rate.

 2. Marginal effects — for each parameter (hidden_dim, num_layers),
    the mean of each metric across the other axis. Reveals the
    "main effect" of each parameter independent of the other.

 3. Pareto frontier — which (hidden_dim, num_layers) combinations
    are non-dominated on the three metrics (maximize nDCG, maximize
    edge_prec, minimize collapse).

 4. Variance / stability report — flags configs with unusually high
    seed-to-seed variance as unreliable.

 5. Interaction check — test whether (h=128, L=2) and (h=64, L=4)
    produce similar results (same parameter count order).

Usage
-----
    python -m src.modelsv3.sweep_analyze \\
        --sweep-root runs/sweep_arch_hyp \\
        --out runs/sweep_arch_hyp_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


CELL_PATTERN = re.compile(r"^h(\d+)_l(\d+)_seed(\d+)$")


def _parse_cell(name: str) -> tuple[int, int, int] | None:
    m = CELL_PATTERN.match(name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _safe_load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _extract_ndcg(summary: dict | None) -> float:
    """Pull nDCG@10 from final_val.overall."""
    if summary is None:
        return float("nan")
    return (
        summary.get("final_val", {})
        .get("overall", {})
        .get("ndcg@10", float("nan"))
    )


def _extract_intrinsic(intrinsic: dict | None) -> tuple[float, float]:
    """Pull edge_prec@5 and label_purity@5 (mean across graphs).

    Schema (verified by inspection, 2026-04):
        intrinsic["per_graph"] = [
            {"edge_prec_mean": float, "label_purity_mean": float, ...},
            ...
        ]

    Computes the across-graph mean ourselves. Falls back to a few
    pre-aggregated layouts in case a different eval script writes
    something else."""
    if intrinsic is None:
        return float("nan"), float("nan")

    # Primary path: compute means from per_graph array.
    per_graph = intrinsic.get("per_graph")
    if isinstance(per_graph, list) and per_graph:
        ep_vals = [g["edge_prec_mean"] for g in per_graph
                   if isinstance(g, dict) and isinstance(g.get("edge_prec_mean"), (int, float))]
        lp_vals = [g["label_purity_mean"] for g in per_graph
                   if isinstance(g, dict) and isinstance(g.get("label_purity_mean"), (int, float))]
        ep = sum(ep_vals) / len(ep_vals) if ep_vals else float("nan")
        lp = sum(lp_vals) / len(lp_vals) if lp_vals else float("nan")
        return ep, lp

    # Fallback: pre-aggregated block (not observed in practice).
    for top in ("across_graphs", "summary", "aggregated", "overall"):
        block = intrinsic.get(top)
        if isinstance(block, dict) and "edge_prec@5" in block:
            ep = block.get("edge_prec@5", {})
            lp = block.get("label_purity@5", {})
            return (ep.get("mean", float("nan")) if isinstance(ep, dict) else float("nan"),
                    lp.get("mean", float("nan")) if isinstance(lp, dict) else float("nan"))
    return float("nan"), float("nan")


def _extract_collapse(collapse: dict | None) -> float:
    """Pull the collapse rate from investigate_collapse.py output.

    Schema (verified by inspection, 2026-04):
        collapse["q1_q2_aggregate"]["frac_below_threshold"]["1e-04"]["mean"]

    Returns NaN if the expected path isn't present."""
    if collapse is None:
        return float("nan")
    agg = collapse.get("q1_q2_aggregate")
    if not isinstance(agg, dict):
        return float("nan")
    fbt = agg.get("frac_below_threshold")
    if not isinstance(fbt, dict):
        return float("nan")
    block = fbt.get("1e-04")
    if isinstance(block, dict) and "mean" in block:
        return float(block["mean"])
    if isinstance(block, (int, float)):
        return float(block)
    return float("nan")


def _mean_std(vals: list[float]) -> tuple[float, float, int]:
    clean = [v for v in vals if v == v]
    if not clean:
        return float("nan"), float("nan"), 0
    if len(clean) == 1:
        return clean[0], 0.0, 1
    return statistics.mean(clean), statistics.stdev(clean), len(clean)


def _collect(sweep_root: Path) -> list[dict]:
    """Walk the sweep root, collect one record per cell directory."""
    records: list[dict] = []
    for cell_dir in sorted(sweep_root.iterdir()):
        if not cell_dir.is_dir():
            continue
        parsed = _parse_cell(cell_dir.name)
        if parsed is None:
            continue
        h, L, seed = parsed
        summary = _safe_load(cell_dir / "summary.json")
        intrinsic = _safe_load(cell_dir / "intrinsic_eval.json")
        collapse = _safe_load(cell_dir / "collapse.json")
        ep, lp = _extract_intrinsic(intrinsic)
        records.append({
            "cell": cell_dir.name,
            "hidden_dim": h,
            "num_layers": L,
            "seed": seed,
            "ndcg@10": _extract_ndcg(summary),
            "edge_prec@5": ep,
            "label_purity@5": lp,
            "collapse_rate": _extract_collapse(collapse),
            "complete": summary is not None,
        })
    return records


def _aggregate_by_config(records: list[dict]) -> dict[tuple[int, int], dict]:
    """Group by (hidden_dim, num_layers), summarize across seeds."""
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["hidden_dim"], r["num_layers"])].append(r)

    out: dict[tuple[int, int], dict] = {}
    for cfg, rs in groups.items():
        block: dict[str, Any] = {
            "n_seeds": len(rs),
            "seeds_complete": sum(1 for r in rs if r["complete"]),
        }
        for metric in ("ndcg@10", "edge_prec@5", "label_purity@5", "collapse_rate"):
            vals = [r[metric] for r in rs]
            mean, std, n = _mean_std(vals)
            block[metric] = {"mean": mean, "std": std, "n": n, "per_seed": vals}
        out[cfg] = block
    return out


def _marginal_effects(
    records: list[dict], axis: str, other_axis: str, metrics: tuple[str, ...]
) -> dict[Any, dict[str, dict]]:
    """For each value of `axis`, pool across all values of `other_axis`
    and all seeds. Gives the main effect of `axis` independent of the
    other parameter."""
    groups: dict[Any, list[dict]] = defaultdict(list)
    for r in records:
        groups[r[axis]].append(r)

    out: dict[Any, dict[str, dict]] = {}
    for v, rs in groups.items():
        block: dict[str, Any] = {"n_runs": len(rs)}
        for m in metrics:
            vals = [r[m] for r in rs]
            mean, std, n = _mean_std(vals)
            block[m] = {"mean": mean, "std": std, "n": n}
        out[v] = block
    return out


def _pareto_frontier(
    configs: dict[tuple[int, int], dict]
) -> list[tuple[int, int]]:
    """Identify (h, L) combinations that are not strictly dominated by
    another combination. Objectives: maximize ndcg@10 and edge_prec@5,
    minimize collapse_rate.

    Config A dominates B if A >= B on all maximize objectives, A <= B
    on all minimize objectives, with at least one strict inequality."""
    def vec(cfg: dict) -> tuple[float, float, float] | None:
        n = cfg["ndcg@10"]["mean"]
        e = cfg["edge_prec@5"]["mean"]
        c = cfg["collapse_rate"]["mean"]
        if n != n or e != e or c != c:
            return None
        return n, e, c

    cfg_vecs = {k: vec(v) for k, v in configs.items()}
    cfg_vecs = {k: v for k, v in cfg_vecs.items() if v is not None}

    frontier: list[tuple[int, int]] = []
    for k_a, v_a in cfg_vecs.items():
        dominated = False
        for k_b, v_b in cfg_vecs.items():
            if k_a == k_b:
                continue
            n_a, e_a, c_a = v_a
            n_b, e_b, c_b = v_b
            # b dominates a if b is at least as good on all and better on one
            at_least_as_good = (n_b >= n_a) and (e_b >= e_a) and (c_b <= c_a)
            strictly_better = (n_b > n_a) or (e_b > e_a) or (c_b < c_a)
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(k_a)
    return sorted(frontier)


def _variance_flags(
    configs: dict[tuple[int, int], dict], thresh_multiplier: float = 2.0
) -> dict:
    """Flag configs whose across-seed std is unusually high.

    Computes the median std for each metric across all configs, then
    flags any config whose std exceeds ``thresh_multiplier * median``
    on any metric. High-std configs = unstable training."""
    metrics = ("ndcg@10", "edge_prec@5", "label_purity@5", "collapse_rate")
    # Median std per metric
    per_metric_meds: dict[str, float] = {}
    for m in metrics:
        stds = [configs[k][m]["std"] for k in configs
                if configs[k][m]["std"] == configs[k][m]["std"]]
        per_metric_meds[m] = statistics.median(stds) if stds else float("nan")

    flagged: dict[str, list] = defaultdict(list)
    for k, v in configs.items():
        for m in metrics:
            std = v[m]["std"]
            med = per_metric_meds[m]
            if std != std or med != med or med == 0:
                continue
            if std > thresh_multiplier * med:
                flagged[m].append({
                    "config": f"h{k[0]}_l{k[1]}",
                    "std": std,
                    "median_std": med,
                    "ratio": std / med,
                })
    return {
        "threshold_multiplier": thresh_multiplier,
        "median_std_per_metric": per_metric_meds,
        "flagged_per_metric": dict(flagged),
    }


def _interaction_check(configs: dict[tuple[int, int], dict]) -> dict:
    """Compare configs with comparable 'compute' (h=128, L=2) vs
    (h=64, L=4) vs (h=32, L=8)-if-present. The question: does the
    benefit come from width, depth, or just total compute?"""
    pairs = [
        ((128, 2), (64, 4)),  # larger dim w/ fewer layers vs vice versa
        ((128, 3), (64, 3)),  # pure width effect at equal depth
        ((64, 4), (64, 2)),   # pure depth effect at equal width
    ]
    out: dict[str, dict] = {}
    for a, b in pairs:
        if a not in configs or b not in configs:
            continue
        block = {}
        for m in ("ndcg@10", "edge_prec@5", "collapse_rate"):
            v_a = configs[a][m]["mean"]
            v_b = configs[b][m]["mean"]
            if v_a != v_a or v_b != v_b:
                block[m] = None
                continue
            block[m] = {"a": v_a, "b": v_b, "a_minus_b": v_a - v_b}
        out[f"h{a[0]}_l{a[1]}__vs__h{b[0]}_l{b[1]}"] = block
    return out


def _print_main_table(configs: dict[tuple[int, int], dict]) -> None:
    print()
    print("=" * 96)
    print("MAIN TABLE  (3-seed mean ± std)")
    print("=" * 96)
    header = (f"{'config':<14} {'n':>3}  "
              f"{'nDCG@10':>16}  {'edge_prec@5':>16}  "
              f"{'label_pur@5':>16}  {'collapse':>16}")
    print(header)
    print("-" * 96)
    for (h, L) in sorted(configs.keys()):
        b = configs[(h, L)]
        def fmt(key):
            m = b[key]
            if m["n"] == 0:
                return "—"
            return f"{m['mean']:+.4f} ± {m['std']:.4f}"
        print(f"h{h:<3} l{L:<7} {b['seeds_complete']:>3}  "
              f"{fmt('ndcg@10'):>16}  {fmt('edge_prec@5'):>16}  "
              f"{fmt('label_purity@5'):>16}  {fmt('collapse_rate'):>16}")


def _print_marginal_effects(
    by_hidden: dict, by_layer: dict, metrics: tuple[str, ...]
) -> None:
    print()
    print("=" * 96)
    print("MARGINAL EFFECTS  (mean across seeds + the other axis)")
    print("=" * 96)
    print()
    print("By hidden_dim (pooled over num_layers and seed):")
    print("-" * 96)
    header = f"{'hidden_dim':<12} {'n':>4}  "
    header += "  ".join(f"{m:>16}" for m in metrics)
    print(header)
    for v in sorted(by_hidden.keys()):
        b = by_hidden[v]
        row = f"{v:<12} {b['n_runs']:>4}  "
        row += "  ".join(
            f"{b[m]['mean']:+8.4f} ± {b[m]['std']:.4f}" for m in metrics
        )
        print(row)

    print()
    print("By num_layers (pooled over hidden_dim and seed):")
    print("-" * 96)
    header = f"{'num_layers':<12} {'n':>4}  "
    header += "  ".join(f"{m:>16}" for m in metrics)
    print(header)
    for v in sorted(by_layer.keys()):
        b = by_layer[v]
        row = f"{v:<12} {b['n_runs']:>4}  "
        row += "  ".join(
            f"{b[m]['mean']:+8.4f} ± {b[m]['std']:.4f}" for m in metrics
        )
        print(row)


def _print_pareto(
    frontier: list[tuple[int, int]], configs: dict[tuple[int, int], dict]
) -> None:
    print()
    print("=" * 96)
    print("PARETO FRONTIER  (non-dominated configs; maximize nDCG/edge_prec, minimize collapse)")
    print("=" * 96)
    print(f"{'config':<12}  {'nDCG@10':>12}  {'edge_prec':>12}  {'collapse':>12}")
    for (h, L) in frontier:
        b = configs[(h, L)]
        print(f"h{h}_l{L:<8}  {b['ndcg@10']['mean']:>+12.4f}  "
              f"{b['edge_prec@5']['mean']:>+12.4f}  "
              f"{b['collapse_rate']['mean']:>12.4f}")
    print()
    print(f"{len(frontier)} of {len(configs)} configs are non-dominated.")
    if len(frontier) == 1:
        print("→ A single config dominates all others on every metric.")
    elif len(frontier) == len(configs):
        print("→ No config dominates any other; the sweep shows only tradeoffs.")


def _print_variance(var: dict) -> None:
    print()
    print("=" * 96)
    print(f"VARIANCE FLAGS  (std > {var['threshold_multiplier']}x median across sweep)")
    print("=" * 96)
    if not var["flagged_per_metric"]:
        print("No configs flagged. Across-seed variance is reasonably uniform.")
        return
    for metric, flags in var["flagged_per_metric"].items():
        if not flags:
            continue
        print(f"\n{metric} (median std = {var['median_std_per_metric'][metric]:.4f}):")
        for f in flags:
            print(f"  {f['config']:<12}  std={f['std']:.4f}  "
                  f"({f['ratio']:.1f}x median)  — treat with caution")


def _print_interactions(interactions: dict) -> None:
    if not interactions:
        return
    print()
    print("=" * 96)
    print("INTERACTION CHECKS  (A − B per metric)")
    print("=" * 96)
    print("Tests whether depth and width are interchangeable at matched compute.")
    for comparison, metrics in interactions.items():
        print(f"\n{comparison}:")
        for m, vals in metrics.items():
            if vals is None:
                print(f"  {m}: — (missing data)")
            else:
                print(f"  {m}: A={vals['a']:+.4f}  B={vals['b']:+.4f}  "
                      f"Δ={vals['a_minus_b']:+.4f}")


def analyze_sweep(sweep_root: Path, out_path: Path) -> dict:
    records = _collect(sweep_root)
    if not records:
        print(f"[err] no cell directories found under {sweep_root}")
        return {}

    complete_records = [r for r in records if r["complete"]]
    print(f"[analyze] collected {len(records)} cells, "
          f"{len(complete_records)} complete")

    configs = _aggregate_by_config(complete_records)
    metrics = ("ndcg@10", "edge_prec@5", "collapse_rate")
    by_hidden = _marginal_effects(complete_records, "hidden_dim", "num_layers", metrics)
    by_layer = _marginal_effects(complete_records, "num_layers", "hidden_dim", metrics)
    frontier = _pareto_frontier(configs)
    variance = _variance_flags(configs)
    interactions = _interaction_check(configs)

    report: dict = {
        "sweep_root": str(sweep_root),
        "n_cells_found": len(records),
        "n_cells_complete": len(complete_records),
        "per_config_summary": {
            f"h{h}_l{L}": v for (h, L), v in configs.items()
        },
        "marginal_effects": {
            "by_hidden_dim": by_hidden,
            "by_num_layers": by_layer,
        },
        "pareto_frontier": [f"h{h}_l{L}" for (h, L) in frontier],
        "variance_flags": variance,
        "interaction_checks": interactions,
        "per_cell_records": records,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    _print_main_table(configs)
    _print_marginal_effects(by_hidden, by_layer, metrics)
    _print_pareto(frontier, configs)
    _print_variance(variance)
    _print_interactions(interactions)

    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-root", type=str, required=True,
                    help="Root directory of the sweep (containing h*_l*_seed* subdirs).")
    ap.add_argument("--out", type=str, required=True,
                    help="Where to write the JSON report.")
    args = ap.parse_args()
    analyze_sweep(Path(args.sweep_root), Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

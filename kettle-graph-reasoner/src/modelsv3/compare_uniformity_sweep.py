r"""Phase 2+ comparison — uniformity regularization sweep.

Aggregates intrinsic-corpus-eval results + collapse-diagnostic results
across 3 seeds x 3 W values for one geometry (hyp or euc).

Expected directory layout (before running):

    runs/v3_unif_{geom}_w{W}_seed{S}/
        encoder.pt               (from patched train_v3)
        summary.json             (from patched train_v3)
        intrinsic_eval.json      (from eval_intrinsic_corpus.py)
        collapse.json            (from investigate_collapse.py)

Usage
-----
    python -m src.modelsv3.compare_uniformity_sweep \\
        --geometry hyperbolic \\
        --runs-glob 'runs/v3_unif_hyp_w*_seed*' \\
        --out runs/compare_uniformity_hyp.json

Outputs: three-section table
    1. Per-seed raw numbers for sanity
    2. Across-seed W × metric summary
    3. Success/failure verdict vs baseline (W=0)
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


# Expected filenames inside each run directory
SUMMARY_NAME = "summary.json"
INTRINSIC_NAME = "intrinsic_eval.json"
COLLAPSE_NAME = "collapse.json"


def _parse_w_and_seed(run_dir: Path) -> tuple[float, int]:
    """Extract W and seed from a run directory name of the form
    ``v3_unif_{geom}_w{W}_seed{S}``. Returns (W, seed) as floats/ints.

    The W value supports formats like w0, w0.01, w0p01 (underscores not
    allowed in our convention)."""
    name = run_dir.name
    m = re.search(r"_w([0-9]+(?:\.[0-9]+)?|[0-9]+p[0-9]+)_seed([0-9]+)", name)
    if not m:
        raise ValueError(
            f"Could not parse W/seed from run dir name {name!r}. "
            f"Expected pattern '..._w<W>_seed<S>'."
        )
    w_str = m.group(1).replace("p", ".")
    w = float(w_str)
    seed = int(m.group(2))
    return w, seed


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _aggregate_seeds(per_seed_vals: list[float]) -> dict:
    clean = [v for v in per_seed_vals if v == v]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(clean) == 1:
        return {"mean": clean[0], "std": 0.0, "n": 1}
    return {
        "mean": statistics.mean(clean),
        "std": statistics.stdev(clean),
        "n": len(clean),
    }


def _extract_intrinsic_metrics(intrinsic: dict) -> dict:
    """Pull the 3-number summary from eval_intrinsic_corpus.py output.

    Schema (verified by inspection, 2026-04):
        intrinsic["per_graph"] = [
            {"silhouette_mean": float, "edge_prec_mean": float,
             "label_purity_mean": float, ...},
            ...
        ]

    We compute the across-graph mean ourselves from the per_graph list.
    Keeps a fallback path for the older nested-summary schema in case
    it ever reappears."""
    out = {"silhouette": float("nan"),
           "edge_prec@5": float("nan"),
           "label_purity@5": float("nan")}
    if intrinsic is None:
        return out

    # Primary path: compute across-graph means from per_graph array.
    per_graph = intrinsic.get("per_graph")
    if isinstance(per_graph, list) and per_graph:
        for src_key, out_key in (
            ("silhouette_mean", "silhouette"),
            ("edge_prec_mean", "edge_prec@5"),
            ("label_purity_mean", "label_purity@5"),
        ):
            vals = [g[src_key] for g in per_graph
                    if isinstance(g, dict) and isinstance(g.get(src_key), (int, float))]
            if vals:
                out[out_key] = sum(vals) / len(vals)
        return out

    # Fallback: older nested-summary layout (not observed in practice
    # but kept in case a different eval script writes this shape).
    for top_key in ("across_graphs", "summary", "aggregated", "overall"):
        block = intrinsic.get(top_key)
        if isinstance(block, dict) and isinstance(block.get("silhouette"), dict):
            out["silhouette"] = block["silhouette"].get("mean", float("nan"))
            ep = block.get("edge_prec@5", {})
            lp = block.get("label_purity@5", {})
            out["edge_prec@5"] = ep.get("mean", float("nan")) if isinstance(ep, dict) else float("nan")
            out["label_purity@5"] = lp.get("mean", float("nan")) if isinstance(lp, dict) else float("nan")
            return out
    return out


def _extract_collapse_rate(collapse: dict) -> float:
    """Pull the collapse rate from investigate_collapse.py output.

    Schema (verified by inspection, 2026-04):
        collapse["q1_q2_aggregate"]["frac_below_threshold"]["1e-04"]["mean"]

    This is the fraction of all within-graph node pairs whose distance
    is below 1e-4 * median pairwise distance — i.e. the effective
    collapse rate. 1e-04 is tighter than what the comparison rate is
    usually reported against; we keep it for consistency with the
    figures quoted in the findings doc.

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", type=str, required=True,
                    choices=["hyperbolic", "euclidean"])
    ap.add_argument("--runs-glob", type=str, required=True,
                    help="Glob pattern for run directories, e.g. "
                         "'runs/v3_unif_hyp_w*_seed*'.")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--baseline-w", type=float, default=0.0,
                    help="Reference W for computing deltas. Default 0.0.")
    args = ap.parse_args()

    run_dirs = sorted(Path(p) for p in glob.glob(args.runs_glob)
                      if Path(p).is_dir())
    if not run_dirs:
        print(f"[warn] no run directories matched {args.runs_glob}")
        return 1
    print(f"[2+] found {len(run_dirs)} run directories")

    # Parse each run's W and seed, load its outputs.
    records: list[dict] = []
    for rd in run_dirs:
        try:
            w, seed = _parse_w_and_seed(rd)
        except ValueError as e:
            print(f"[warn] skipping {rd}: {e}")
            continue
        summary = _load_json(rd / SUMMARY_NAME)
        intrinsic = _load_json(rd / INTRINSIC_NAME)
        collapse = _load_json(rd / COLLAPSE_NAME)

        if summary is None:
            print(f"[warn] no summary.json in {rd}, skipping")
            continue
        geom_recorded = summary.get("model")
        if geom_recorded and geom_recorded != args.geometry:
            print(f"[warn] {rd} has model={geom_recorded}, "
                  f"but --geometry={args.geometry}; skipping")
            continue

        metrics = _extract_intrinsic_metrics(intrinsic or {})
        metrics["collapse_rate"] = _extract_collapse_rate(collapse or {})

        records.append({
            "run_dir": str(rd),
            "w": w,
            "seed": seed,
            "metrics": metrics,
            "has_intrinsic": intrinsic is not None,
            "has_collapse": collapse is not None,
        })

    if not records:
        print("[err] no valid records found")
        return 1

    # Group by W for aggregation.
    by_w: dict[float, list[dict]] = defaultdict(list)
    for r in records:
        by_w[r["w"]].append(r)
    sorted_ws = sorted(by_w.keys())

    # Aggregate each metric across seeds for each W.
    metric_names = ("silhouette", "edge_prec@5", "label_purity@5",
                    "collapse_rate")
    aggregated: dict[float, dict[str, dict]] = {}
    for w in sorted_ws:
        aggregated[w] = {}
        for m in metric_names:
            vals = [r["metrics"].get(m, float("nan")) for r in by_w[w]]
            aggregated[w][m] = _aggregate_seeds(vals)
            aggregated[w][m]["per_seed"] = vals

    # Deltas vs baseline.
    baseline_w = args.baseline_w
    deltas: dict[float, dict[str, float]] = {}
    if baseline_w in aggregated:
        base = aggregated[baseline_w]
        for w in sorted_ws:
            if w == baseline_w:
                continue
            deltas[w] = {}
            for m in metric_names:
                b_mean = base[m]["mean"]
                w_mean = aggregated[w][m]["mean"]
                if b_mean == b_mean and w_mean == w_mean:
                    deltas[w][m] = w_mean - b_mean
                else:
                    deltas[w][m] = float("nan")

    # Verdict (heuristic): did collapse drop meaningfully without
    # catastrophic damage to the main retrieval metric?
    verdict = _verdict(aggregated, baseline_w, sorted_ws)

    results = {
        "geometry": args.geometry,
        "baseline_w": baseline_w,
        "sorted_ws": sorted_ws,
        "n_records": len(records),
        "per_seed_records": records,
        "aggregated": aggregated,
        "deltas_vs_baseline": deltas,
        "verdict": verdict,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    _print_table(results, metric_names)
    return 0


def _verdict(
    aggregated: dict[float, dict[str, dict]],
    baseline_w: float,
    sorted_ws: list[float],
) -> dict:
    """Heuristic verdict for each non-baseline W:
        SUCCESS: collapse_rate drops by >= 50% AND edge_prec@5 within
                 2x of baseline (for hyp) or stays near-random (for euc
                 since it was already random)
        PARTIAL: collapse drops but metrics degrade significantly
        NO-OP:   collapse doesn't drop
        FAILURE: metrics degrade AND collapse doesn't drop
    """
    out: dict = {}
    if baseline_w not in aggregated:
        return {"note": f"baseline W={baseline_w} not found; no verdict"}
    base = aggregated[baseline_w]
    base_collapse = base["collapse_rate"]["mean"]
    base_ep = base["edge_prec@5"]["mean"]

    for w in sorted_ws:
        if w == baseline_w:
            continue
        cw = aggregated[w]["collapse_rate"]["mean"]
        epw = aggregated[w]["edge_prec@5"]["mean"]
        collapse_drop_frac = (
            (base_collapse - cw) / base_collapse
            if base_collapse == base_collapse and base_collapse > 0
            else float("nan")
        )
        ep_ratio = (
            epw / base_ep if base_ep == base_ep and base_ep > 0
            else float("nan")
        )
        # Verdict logic
        if collapse_drop_frac != collapse_drop_frac:
            label = "UNKNOWN"
        elif collapse_drop_frac >= 0.5 and (ep_ratio >= 0.5 or ep_ratio != ep_ratio):
            label = "SUCCESS"
        elif collapse_drop_frac >= 0.5:
            label = "PARTIAL (collapse fixed, retrieval hurt)"
        elif collapse_drop_frac < 0.1:
            label = "NO-OP (collapse unchanged)"
        else:
            label = "WEAK"
        out[w] = {
            "label": label,
            "collapse_drop_frac": collapse_drop_frac,
            "edge_prec_ratio_vs_baseline": ep_ratio,
        }
    return out


def _print_table(r: dict, metric_names: tuple) -> None:
    print()
    print("=" * 92)
    print(f"UNIFORMITY SWEEP  geometry={r['geometry']}  "
          f"baseline_W={r['baseline_w']}")
    print("=" * 92)

    # Per-W × metric table
    print(f"\n{'W':>8}  {'silhouette':>18}  {'edge_prec@5':>18}  "
          f"{'label_purity@5':>18}  {'collapse_rate':>18}")
    print("-" * 92)
    for w in r["sorted_ws"]:
        row = r["aggregated"][w]
        def fmt(k):
            s = row[k]
            if s["n"] == 0:
                return "—"
            return f"{s['mean']:+.4f} ± {s['std']:.4f}"
        print(f"{w:>8.3f}  {fmt('silhouette'):>18}  "
              f"{fmt('edge_prec@5'):>18}  {fmt('label_purity@5'):>18}  "
              f"{fmt('collapse_rate'):>18}")

    # Deltas
    if r["deltas_vs_baseline"]:
        print()
        print(f"Deltas vs baseline W={r['baseline_w']}")
        print("-" * 92)
        print(f"{'W':>8}  {'Δ silhouette':>18}  {'Δ edge_prec':>18}  "
              f"{'Δ label_pur':>18}  {'Δ collapse':>18}")
        for w, d in r["deltas_vs_baseline"].items():
            def fmtd(k):
                v = d.get(k, float("nan"))
                return "—" if v != v else f"{v:+.4f}"
            print(f"{w:>8.3f}  {fmtd('silhouette'):>18}  "
                  f"{fmtd('edge_prec@5'):>18}  {fmtd('label_purity@5'):>18}  "
                  f"{fmtd('collapse_rate'):>18}")

    # Verdict
    if r.get("verdict"):
        print()
        print("Verdict")
        print("-" * 92)
        for w, v in r["verdict"].items():
            if isinstance(v, dict) and "label" in v:
                drop = v.get("collapse_drop_frac", float("nan"))
                ratio = v.get("edge_prec_ratio_vs_baseline", float("nan"))
                drop_s = "—" if drop != drop else f"{drop*100:+.0f}%"
                ratio_s = "—" if ratio != ratio else f"{ratio:.2f}x"
                print(f"  W={w:.3f}  {v['label']:<45}  "
                      f"collapse_drop={drop_s}  ep_ratio={ratio_s}")
            else:
                print(f"  W={w}  note={v}")


if __name__ == "__main__":
    sys.exit(main())

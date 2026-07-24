r"""Embedding collapse investigation for v3 encoders.

Answers five diagnostic questions about pairwise-distance structure of
embeddings produced by a trained v3 encoder:

    Q1  Minimum / median / tail-fraction of pairwise distances.
    Q2  Log-distance distribution (skewness, percentile ladder).
    Q3  What do near-duplicate pairs share? Per-feature-block analysis
         under both top-K and threshold-based pair selection.
    Q4  Do 2.3a metrics change when near-duplicate NN are filtered out?
    Q5  Are the collapsed pairs the same nodes across seeds? (Answered
         by the companion aggregator, using sets produced by this script.)

Usage
-----
    py src/modelsv3/investigate_collapse.py \\
        --checkpoint runs/v3_hyp_compute_seed0/encoder.pt \\
        --summary    runs/v3_hyp_compute_seed0/summary.json \\
        --out        runs/v3_hyp_compute_seed0/collapse_diag.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv2.layers import poincare_ops as P  # noqa: E402
from src.modelsv3.distance_ops import (  # noqa: E402
    pairwise_distance_matrix as _pairwise_dist,
)
from src.modelsv3.euclidean_v3 import EuclideanReasonerV3  # noqa: E402
from src.modelsv3.hyperbolic_gnnV3 import KettleGraphReasonerV3  # noqa: E402


NODE_TYPE_SLICE = slice(0, 12)
LAYER_SLICE = slice(12, 16)
STRUCT_SLICE = slice(16, 21)
TEMPORAL_SLICE = slice(21, 24)
IDENTITY_SLICE = slice(24, 32)
FEATURE_BLOCKS = {
    "type": NODE_TYPE_SLICE,
    "layer": LAYER_SLICE,
    "structural": STRUCT_SLICE,
    "temporal": TEMPORAL_SLICE,
    "identity": IDENTITY_SLICE,
}
THRESHOLDS_Q1 = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8)
PERCENTILES_Q2 = (0.01, 0.1, 1.0, 10.0, 50.0, 90.0, 99.0)
Q3_TOPK_CAP = 100
Q3_THRESHOLD_FRAC = 0.01  # pairs below this × median distance
Q4_THRESHOLDS_FRAC = (1e-6, 1e-4, 1e-2)  # multiplied by median per graph


# ---------------------------------------------------------------------------
# model loading (same as 2.3a)
# ---------------------------------------------------------------------------

def _load_encoder(
    checkpoint: Path, summary: Path, dataset: CorpusDataset
) -> tuple[torch.nn.Module, dict]:
    with open(summary, "r") as f:
        s = json.load(f)
    cfg = s["config"]
    # E5: honor the attn_type_table key (absent => legacy True).
    net = (dataset.num_edge_types_max
           if cfg.get("attn_type_table", True) else None)
    if cfg["model"] == "hyperbolic":
        model = KettleGraphReasonerV3(
            node_feat_dim=dataset.node_feat_dim,
            edge_feat_dim=dataset.edge_feat_dim_schema,
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            type_dim=int(cfg["type_dim"]),
            c=float(cfg["curvature"]),
            num_edge_types_max=net,
            node_feat_dim_schema=dataset.node_feat_dim_schema,
            tangent_scale_init=float(cfg.get("tangent_scale", 0.1)),
        )
    elif cfg["model"] == "euclidean":
        model = EuclideanReasonerV3(
            node_feat_dim=dataset.node_feat_dim,
            edge_feat_dim=dataset.edge_feat_dim_schema,
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            type_dim=int(cfg["type_dim"]),
            num_edge_types_max=net,
            node_feat_dim_schema=dataset.node_feat_dim_schema,
        )
    else:
        raise ValueError(f"unknown model kind {cfg['model']!r}")
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def _unique_graph_indices(dataset: CorpusDataset) -> list[int]:
    seen: list[int] = []
    seen_set: set[int] = set()
    for graph_idx, _task_idx in dataset.index:
        gi = int(graph_idx)
        if gi not in seen_set:
            seen_set.add(gi)
            seen.append(gi)
    return seen


# _pairwise_dist is now the cap-guarded full-matrix helper imported
# above. This collapse diagnostic does per-pair feature analysis that
# needs the specific (i,j) pairs, so it keeps the full-matrix path:
# bit-identical at N <= cap (every collapse.json was produced there),
# loud error > cap rather than silent OOM.


def _flat_offdiag(D: np.ndarray) -> np.ndarray:
    """Return the vector of off-diagonal distance entries, upper triangle."""
    N = D.shape[0]
    iu = np.triu_indices(N, k=1)
    return D[iu]


# ---------------------------------------------------------------------------
# Q1 & Q2 — distance distribution statistics
# ---------------------------------------------------------------------------

def _q1_q2(D: np.ndarray) -> dict:
    flat = _flat_offdiag(D)
    if flat.size == 0:
        return {}
    finite = flat[np.isfinite(flat)]
    n_pairs = int(finite.size)
    n_non_finite = int(flat.size - n_pairs)
    if n_pairs == 0:
        return {"n_pairs": 0, "n_non_finite": n_non_finite}

    min_v = float(finite.min())
    median_v = float(np.median(finite))
    max_v = float(finite.max())

    # Q1: threshold fractions
    frac_below = {
        f"{t:.0e}": float(np.sum(finite < t) / n_pairs) for t in THRESHOLDS_Q1
    }

    # Q2: log-distance distribution
    # Shift away from zero for log (pairs at exactly 0 get floor_min).
    floor_min = max(1e-16, min_v * 1e-6) if min_v > 0 else 1e-16
    log_dist = np.log10(np.maximum(finite, floor_min))
    percentiles = {
        f"p{p}": float(np.percentile(log_dist, p)) for p in PERCENTILES_Q2
    }

    # Skewness via Fisher-Pearson moment coefficient
    mean = float(log_dist.mean())
    std = float(log_dist.std())
    if std > 0:
        skewness = float(np.mean(((log_dist - mean) / std) ** 3))
    else:
        skewness = 0.0

    # Fraction in bottom 1% of bulk: where "bulk" = above p10
    p10 = np.percentile(finite, 10)
    below_bulk_low = float(np.sum(finite < 0.01 * p10) / n_pairs) if p10 > 0 else 0.0

    return {
        "n_pairs": n_pairs,
        "n_non_finite": n_non_finite,
        "min": min_v,
        "median": median_v,
        "max": max_v,
        "min_over_median": float(min_v / median_v) if median_v > 0 else float("inf"),
        "frac_below_threshold": frac_below,
        "log10_percentiles": percentiles,
        "log10_skewness": skewness,
        "frac_below_1pct_of_p10": below_bulk_low,
    }


# ---------------------------------------------------------------------------
# Q3 — what do near-duplicate pairs share?
# ---------------------------------------------------------------------------

def _pair_cosine_similarity(
    x_np: np.ndarray, pairs: np.ndarray, feat_slice: slice
) -> np.ndarray:
    """Cosine similarity between node-feature blocks for a set of pairs.
    Returns NaN for pairs where either side has zero norm in the block."""
    a = x_np[pairs[:, 0], feat_slice]
    b = x_np[pairs[:, 1], feat_slice]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = na * nb
    out = np.full(len(pairs), np.nan, dtype=np.float64)
    ok = denom > 0
    out[ok] = np.sum(a[ok] * b[ok], axis=1) / denom[ok]
    return out


def _pair_same_label(
    x_np: np.ndarray, pairs: np.ndarray, feat_slice: slice
) -> np.ndarray:
    """For a one-hot block, whether the two endpoints share the argmax label.
    Returns False where a row has all-zero one-hot."""
    a = x_np[pairs[:, 0], feat_slice]
    b = x_np[pairs[:, 1], feat_slice]
    sa = a.sum(axis=1) > 0
    sb = b.sum(axis=1) > 0
    out = np.zeros(len(pairs), dtype=bool)
    valid = sa & sb
    out[valid] = a[valid].argmax(axis=1) == b[valid].argmax(axis=1)
    return out


def _q3_block_analysis(
    x_np: np.ndarray, D: np.ndarray, median_dist: float
) -> dict:
    """Compare close-pair feature similarity vs all-pair baseline.

    Runs two pair-selection strategies: top-K by distance and threshold-
    based (dist < Q3_THRESHOLD_FRAC × median)."""
    N = D.shape[0]
    iu = np.triu_indices(N, k=1)
    all_pairs = np.stack(iu, axis=1)  # (P, 2)
    all_dists = D[iu]
    finite = np.isfinite(all_dists)
    all_pairs = all_pairs[finite]
    all_dists = all_dists[finite]
    if all_pairs.size == 0:
        return {"n_pairs_total": 0}

    # Top-K closest pairs
    topk_n = min(Q3_TOPK_CAP, max(len(all_dists) // 20, 10))
    order = np.argsort(all_dists)
    topk_idx = order[:topk_n]
    topk_pairs = all_pairs[topk_idx]

    # Threshold-based close pairs
    thresh = Q3_THRESHOLD_FRAC * median_dist
    mask_thresh = all_dists < thresh
    thresh_pairs = all_pairs[mask_thresh]
    thresh_dists = all_dists[mask_thresh]

    def summarize_pair_set(pairs: np.ndarray) -> dict:
        out: dict[str, float | int] = {"n_pairs": int(len(pairs))}
        if len(pairs) == 0:
            return out
        for name, sl in FEATURE_BLOCKS.items():
            if name in ("type", "layer"):
                same = _pair_same_label(x_np, pairs, sl)
                out[f"{name}_same_rate"] = float(same.mean())
            else:
                cos = _pair_cosine_similarity(x_np, pairs, sl)
                cos = cos[~np.isnan(cos)]
                out[f"{name}_mean_cos"] = float(cos.mean()) if cos.size else float("nan")
                out[f"{name}_median_cos"] = float(np.median(cos)) if cos.size else float("nan")
        return out

    all_summary = summarize_pair_set(all_pairs)
    topk_summary = summarize_pair_set(topk_pairs)
    thresh_summary = summarize_pair_set(thresh_pairs)

    # Elevation ratios: (close-pair metric) / (all-pair metric)
    def elevation(close_s: dict, all_s: dict) -> dict:
        out: dict[str, float] = {}
        for key in all_s:
            if key == "n_pairs":
                continue
            num = close_s.get(key, float("nan"))
            denom = all_s.get(key, float("nan"))
            if isinstance(num, float) and isinstance(denom, float):
                if denom == 0 or denom != denom:
                    out[key] = float("nan")
                else:
                    out[key] = num / denom
        return out

    return {
        "n_pairs_total": int(len(all_pairs)),
        "topk": {
            "selection": f"top-{topk_n}",
            "summary": topk_summary,
            "elevation_over_baseline": elevation(topk_summary, all_summary),
        },
        "threshold": {
            "selection": f"dist < {Q3_THRESHOLD_FRAC} × median ({thresh:.3e})",
            "summary": thresh_summary,
            "n_pairs": int(len(thresh_pairs)),
            "min_dist": float(thresh_dists.min()) if len(thresh_dists) else float("nan"),
            "max_dist": float(thresh_dists.max()) if len(thresh_dists) else float("nan"),
            "elevation_over_baseline": elevation(thresh_summary, all_summary),
        },
        "baseline_all_pairs": all_summary,
    }


# ---------------------------------------------------------------------------
# Q4 — do 2.3a metrics change when near-duplicate NN are filtered out?
# ---------------------------------------------------------------------------

def _q4_filtered_metrics(
    D: np.ndarray,
    node_types: np.ndarray,
    node_layers: np.ndarray,
    median_dist: float,
    k: int = 5,
) -> dict:
    """Recompute same_type_frac@k and same_layer_frac@k with distance
    thresholds applied to exclude near-duplicate NN."""
    N = D.shape[0]
    D_work = D.copy()
    np.fill_diagonal(D_work, np.inf)

    def compute_with_filter(tau_frac: float | None) -> dict:
        if tau_frac is None:
            # Unfiltered (matches 2.3a)
            tau = -np.inf
        else:
            tau = tau_frac * median_dist

        same_type = np.zeros(N, dtype=np.float64)
        same_layer = np.zeros(N, dtype=np.float64)
        n_seeds_short = 0
        for i in range(N):
            row = D_work[i].copy()
            row[row < tau] = np.inf  # exclude too-close neighbours
            if np.isinf(row).all():
                continue
            k_actual = min(k, int(np.sum(np.isfinite(row))))
            if k_actual < k:
                n_seeds_short += 1
            nbrs = np.argsort(row)[:k_actual]
            ti = node_types[i]
            li = node_layers[i]
            if ti >= 0 and k_actual > 0:
                same_type[i] = float(np.sum(node_types[nbrs] == ti) / k_actual)
            if li >= 0 and k_actual > 0:
                same_layer[i] = float(np.sum(node_layers[nbrs] == li) / k_actual)
        return {
            "same_type_frac_mean": float(same_type.mean()),
            "same_layer_frac_mean": float(same_layer.mean()),
            "n_seeds_with_fewer_than_k": int(n_seeds_short),
        }

    results = {"unfiltered": compute_with_filter(None)}
    for tf in Q4_THRESHOLDS_FRAC:
        results[f"tau={tf:.0e}x_median"] = compute_with_filter(tf)
    return results


# ---------------------------------------------------------------------------
# Q5 — collect collapsed pair sets per graph for cross-seed comparison
# ---------------------------------------------------------------------------

def _q5_collapsed_pairs(
    D: np.ndarray, median_dist: float, tau_frac: float = 1e-4
) -> list[tuple[int, int]]:
    """Return the set of (i, j) with i < j and dist(i, j) < tau_frac × median."""
    tau = tau_frac * median_dist
    N = D.shape[0]
    iu = np.triu_indices(N, k=1)
    dists = D[iu]
    mask = dists < tau
    pairs = [(int(iu[0][idx]), int(iu[1][idx])) for idx in np.where(mask)[0]]
    return pairs


# ---------------------------------------------------------------------------
# top-level driver
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint: Path,
    summary: Path,
    corpus_dir: str,
    split: str,
    split_seed: int,
    task: int | None,
    out_path: Path,
) -> dict:
    include_tasks = {task} if task is not None else None
    dataset = CorpusDataset(
        corpus_dir=corpus_dir, split=split, split_seed=split_seed,
        include_tasks=include_tasks,
    )
    graph_ids = _unique_graph_indices(dataset)
    print(f"[collapse] dataset: {len(dataset)} samples, {len(graph_ids)} graphs")

    model, cfg = _load_encoder(checkpoint, summary, dataset)
    euclidean = cfg["model"] == "euclidean"
    c_val = getattr(model, "c", torch.tensor(float(cfg["curvature"])))

    per_graph_q1q2: list[dict] = []
    per_graph_q3: list[dict] = []
    per_graph_q4: list[dict] = []
    per_graph_q5: dict[int, list[tuple[int, int]]] = {}

    with torch.no_grad():
        for gi in graph_ids:
            graph = dataset._get_graph(gi)
            x = graph["x"]
            edge_index = graph["edge_index"]
            out = model(
                x, edge_index, graph["edge_type"], graph["edge_descriptor"],
                node_descriptor=graph["node_descriptor"],
            )
            emb = out.node_embeddings.detach().cpu()
            D = _pairwise_dist(emb, c_val, euclidean).detach().cpu().numpy()
            D_np = np.asarray(D)
            x_np = x.detach().cpu().numpy()

            # node labels from input features
            tb = x_np[:, NODE_TYPE_SLICE]
            lb = x_np[:, LAYER_SLICE]
            node_types = np.where(tb.sum(axis=1) > 0, tb.argmax(axis=1), -1)
            node_layers = np.where(lb.sum(axis=1) > 0, lb.argmax(axis=1), -1)

            q1q2 = _q1_q2(D_np)
            q1q2["graph_idx"] = gi
            q1q2["n_nodes"] = int(x.size(0))
            per_graph_q1q2.append(q1q2)

            median_dist = q1q2.get("median", float("nan"))
            if median_dist == median_dist and median_dist > 0:
                q3 = _q3_block_analysis(x_np, D_np, median_dist)
                q3["graph_idx"] = gi
                per_graph_q3.append(q3)

                q4 = _q4_filtered_metrics(D_np, node_types, node_layers, median_dist)
                q4["graph_idx"] = gi
                per_graph_q4.append(q4)

                q5_pairs = _q5_collapsed_pairs(D_np, median_dist, tau_frac=1e-4)
                per_graph_q5[gi] = q5_pairs

    # Aggregate Q1: summary stats of key scalar metrics across graphs
    def _stat_across_graphs(field: str, source: list[dict]) -> dict:
        vals = [d[field] for d in source if field in d and d[field] == d[field]]
        if not vals:
            return {"mean": float("nan"), "n": 0}
        return {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            "median": statistics.median(vals),
            "n": len(vals),
        }

    q1_aggregate = {
        "min_over_median": _stat_across_graphs("min_over_median", per_graph_q1q2),
        "median_dist": _stat_across_graphs("median", per_graph_q1q2),
        "min_dist": _stat_across_graphs("min", per_graph_q1q2),
    }
    # Threshold fractions: mean across graphs
    thresh_keys = [f"{t:.0e}" for t in THRESHOLDS_Q1]
    q1_frac_below = {}
    for tk in thresh_keys:
        vals = [
            d["frac_below_threshold"][tk]
            for d in per_graph_q1q2 if "frac_below_threshold" in d
        ]
        q1_frac_below[tk] = {
            "mean": statistics.mean(vals) if vals else float("nan"),
            "max": max(vals) if vals else float("nan"),
        }

    q2_aggregate = {
        "log10_skewness": _stat_across_graphs("log10_skewness", per_graph_q1q2),
        "frac_below_1pct_of_p10": _stat_across_graphs("frac_below_1pct_of_p10", per_graph_q1q2),
    }

    results = {
        "checkpoint": str(checkpoint),
        "model_kind": cfg["model"],
        "split": split,
        "task": task,
        "n_graphs": len(per_graph_q1q2),
        "q1_q2_aggregate": {
            "scalars": q1_aggregate,
            "frac_below_threshold": q1_frac_below,
            "q2_shape": q2_aggregate,
        },
        "per_graph_q1_q2": per_graph_q1q2,
        "per_graph_q3": per_graph_q3,
        "per_graph_q4": per_graph_q4,
        "q5_collapsed_pair_sets": {
            str(gi): pairs for gi, pairs in per_graph_q5.items()
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    _print_summary(results)
    return results


def _print_summary(r: dict) -> None:
    print()
    print("=" * 88)
    print(f"Collapse investigation — {r['n_graphs']} graphs")
    print(f"checkpoint: {r['checkpoint']}   model: {r['model_kind']}")
    print("=" * 88)

    # Q1
    q1 = r["q1_q2_aggregate"]["scalars"]
    print("\nQ1 — raw distance structure (aggregated across graphs)")
    print(f"  median_dist           mean={q1['median_dist']['mean']:.4e}  "
          f"range=[{q1['median_dist']['min']:.3e}, {q1['median_dist']['max']:.3e}]")
    print(f"  min_dist              mean={q1['min_dist']['mean']:.4e}  "
          f"range=[{q1['min_dist']['min']:.3e}, {q1['min_dist']['max']:.3e}]")
    print(f"  min/median ratio      mean={q1['min_over_median']['mean']:.3e}")

    print("\n  Fraction of pairs below threshold (mean / worst graph):")
    for tk, d in r["q1_q2_aggregate"]["frac_below_threshold"].items():
        print(f"    τ={tk:>8}   mean={d['mean']:.4f}   max={d['max']:.4f}")

    # Q2
    q2 = r["q1_q2_aggregate"]["q2_shape"]
    print("\nQ2 — log-distance shape")
    print(f"  log10(skewness)                mean={q2['log10_skewness']['mean']:+.3f}  "
          f"range=[{q2['log10_skewness']['min']:+.3f}, {q2['log10_skewness']['max']:+.3f}]")
    print(f"  frac below 1% of p10 (bulk)    mean={q2['frac_below_1pct_of_p10']['mean']:.4f}  "
          f"max={q2['frac_below_1pct_of_p10']['max']:.4f}")

    # Q3 — summarise mean elevation ratios across graphs, for both selections
    print("\nQ3 — feature similarity of near-duplicate pairs")
    for sel in ("topk", "threshold"):
        print(f"\n  Selection: {sel}")
        elev_keys: set[str] = set()
        for g in r["per_graph_q3"]:
            if sel in g:
                elev_keys.update(g[sel]["elevation_over_baseline"].keys())
        for key in sorted(elev_keys):
            vals = []
            for g in r["per_graph_q3"]:
                if sel in g:
                    v = g[sel]["elevation_over_baseline"].get(key, float("nan"))
                    if v == v and math.isfinite(v):
                        vals.append(v)
            if not vals:
                continue
            mean = statistics.mean(vals)
            print(f"    {key:<22} mean elevation = {mean:+.2f}×  (n={len(vals)} graphs)")

    # Q4 — aggregate the change in metric across graphs
    print("\nQ4 — 2.3a metrics with close-pair NN filtered out")
    label_map = {"unfiltered": "unfiltered", **{
        f"tau={t:.0e}x_median": f"τ={t:.0e}×med" for t in Q4_THRESHOLDS_FRAC
    }}
    for cond_key, display in label_map.items():
        st_vals, sl_vals = [], []
        for g in r["per_graph_q4"]:
            if cond_key in g:
                st_vals.append(g[cond_key]["same_type_frac_mean"])
                sl_vals.append(g[cond_key]["same_layer_frac_mean"])
        if st_vals:
            print(f"  {display:<18}  same_type={statistics.mean(st_vals):.4f}  "
                  f"same_layer={statistics.mean(sl_vals):.4f}")

    # Q5 — just report collapsed-pair count (cross-seed comparison is done separately)
    n_collapsed = [len(v) for v in r["q5_collapsed_pair_sets"].values()]
    if n_collapsed:
        print("\nQ5 — collapsed pairs (dist < 1e-4 × median) per graph")
        print(f"  per-graph counts: mean={statistics.mean(n_collapsed):.1f}  "
              f"total={sum(n_collapsed)}  range=[{min(n_collapsed)}, {max(n_collapsed)}]")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--summary", type=str, default=None)
    p.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--task", type=int, default=2)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    summary = Path(args.summary) if args.summary else checkpoint.parent / "summary.json"
    out = Path(args.out) if args.out else checkpoint.parent / "collapse_diag.json"

    evaluate_checkpoint(
        checkpoint=checkpoint, summary=summary, corpus_dir=args.corpus,
        split=args.split, split_seed=args.split_seed,
        task=None if args.task < 0 else args.task, out_path=out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

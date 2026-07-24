r"""Experiment 2.3c — k-means clustering ARI against node labels.

For each val graph, run k-means on the node embeddings (with k = 12
for type matching, k = 4 for layer matching) and compute ARI
against the true labels.

Tests whether embeddings are unsupervisedly separable by label.

For hyperbolic, k-means is run in tangent-at-origin space (via
logmap0), which is the standard way to apply Euclidean clustering
methods to Poincaré-ball embeddings.

Two conditions:
    unfiltered  — cluster all nodes
    filtered    — cluster after removing nodes that participate in
                   any collapsed pair (dist < 1e-4 × median_dist)

The difference between these two conditions diagnoses how much of
any apparent clustering structure comes from the collapsed node
cluster vs from genuine structure in the non-collapsed nodes.

Metrics:
    ARI(type, k=12)
    ARI(layer, k=4)

Mean over 10 k-means restarts per (checkpoint, graph, condition).

Usage
-----
    py src/modelsv3/eval_retrieval_clustering.py \\
        --checkpoint runs/v3_hyp_compute_seed0/encoder.pt \\
        --out        runs/v3_hyp_compute_seed0/retrieval_clustering.json
"""

from __future__ import annotations

import argparse
import json
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

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
except ImportError as e:
    print("This script requires scikit-learn. Install via: pip install scikit-learn")
    raise


NODE_TYPE_SLICE = slice(0, 12)
LAYER_SLICE = slice(12, 16)
N_RESTARTS = 10
DEFAULT_TAU_FRAC = 1e-4
LABEL_CONFIGS = (("type", 12), ("layer", 4))


def _load_encoder(
    checkpoint: Path, summary: Path, dataset: CorpusDataset
) -> tuple[torch.nn.Module, dict]:
    with open(summary, "r") as f:
        s = json.load(f)
    cfg = s["config"]
    if cfg["model"] == "hyperbolic":
        model = KettleGraphReasonerV3(
            node_feat_dim=dataset.node_feat_dim,
            edge_feat_dim=dataset.edge_feat_dim_schema,
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            type_dim=int(cfg["type_dim"]),
            c=float(cfg["curvature"]),
            num_edge_types_max=dataset.num_edge_types_max,
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
            num_edge_types_max=dataset.num_edge_types_max,
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
# above (bit-identical at N <= cap; loud error > cap). _collapsed_nodes_
# mask needs the specific <tau (i,j) pairs, so it keeps the full-matrix
# path; this is a small-graph clustering diagnostic, never run at scale.


def _collapsed_nodes_mask(
    emb: Tensor, c_val: float | Tensor, euclidean: bool, tau_frac: float
) -> tuple[np.ndarray, float]:
    """Return boolean mask (N,) True if the node participates in at least
    one pair with dist < tau_frac × median_dist."""
    D = _pairwise_dist(emb, c_val, euclidean).detach().cpu().numpy()
    np.fill_diagonal(D, np.inf)
    finite = D[np.isfinite(D)]
    median_dist = float(np.median(finite)) if finite.size > 0 else float("nan")
    if not (median_dist == median_dist) or median_dist <= 0:
        N = emb.size(0)
        return np.zeros(N, dtype=bool), float("nan")
    tau = tau_frac * median_dist
    N = emb.size(0)
    iu = np.triu_indices(N, k=1)
    dists = D[iu]
    mask_pair = (np.isfinite(dists)) & (dists < tau)
    collapsed = np.zeros(N, dtype=bool)
    for i_pair, j_pair in zip(iu[0][mask_pair], iu[1][mask_pair]):
        collapsed[int(i_pair)] = True
        collapsed[int(j_pair)] = True
    return collapsed, median_dist


def _prepare_cluster_points(
    emb: Tensor, c_val: float | Tensor, euclidean: bool
) -> np.ndarray:
    """Get the (N, d) points to feed to k-means. For hyperbolic, map to
    tangent-at-origin via logmap0 (standard practice)."""
    if euclidean:
        return emb.detach().cpu().numpy()
    tan = P.logmap0(emb, c_val).detach().cpu().numpy()
    return tan


def _run_kmeans_ari(
    points: np.ndarray, labels: np.ndarray, k: int, n_restarts: int
) -> dict:
    """Run k-means n_restarts times with different random states,
    compute ARI against labels for each restart."""
    # Filter out nodes with invalid labels (-1) — they can't be judged.
    valid = labels >= 0
    if valid.sum() < k:
        return {"n": 0, "ari_mean": float("nan"), "ari_std": float("nan")}
    pts_v = points[valid]
    lab_v = labels[valid]
    aris: list[float] = []
    for seed in range(n_restarts):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        preds = km.fit_predict(pts_v)
        aris.append(float(adjusted_rand_score(lab_v, preds)))
    return {
        "n": int(valid.sum()),
        "ari_mean": float(np.mean(aris)),
        "ari_std": float(np.std(aris)),
        "ari_min": float(np.min(aris)),
        "ari_max": float(np.max(aris)),
    }


def _evaluate_graph(
    emb: Tensor,
    x: Tensor,
    c_val: float | Tensor,
    euclidean: bool,
    tau_frac: float,
) -> dict:
    x_np = x.detach().cpu().numpy()
    N = emb.size(0)
    tb = x_np[:, NODE_TYPE_SLICE]
    lb = x_np[:, LAYER_SLICE]
    types = np.where(tb.sum(axis=1) > 0, tb.argmax(axis=1), -1)
    layers = np.where(lb.sum(axis=1) > 0, lb.argmax(axis=1), -1)

    collapsed_mask, median_dist = _collapsed_nodes_mask(
        emb, c_val, euclidean, tau_frac
    )

    points = _prepare_cluster_points(emb, c_val, euclidean)

    results = {"unfiltered": {}, "filtered": {}}
    for label_name, k in LABEL_CONFIGS:
        labels = types if label_name == "type" else layers
        # Unfiltered
        results["unfiltered"][label_name] = _run_kmeans_ari(
            points, labels, k, N_RESTARTS
        )
        # Filtered
        if (~collapsed_mask).sum() >= k:
            results["filtered"][label_name] = _run_kmeans_ari(
                points[~collapsed_mask], labels[~collapsed_mask], k, N_RESTARTS
            )
        else:
            results["filtered"][label_name] = {
                "n": int((~collapsed_mask).sum()),
                "ari_mean": float("nan"),
                "ari_std": float("nan"),
            }

    return {
        "n_nodes": N,
        "n_collapsed_nodes": int(collapsed_mask.sum()),
        "median_dist": median_dist,
        "results": results,
    }


def evaluate_checkpoint(
    checkpoint: Path,
    summary: Path,
    corpus_dir: str,
    split: str,
    split_seed: int,
    task: int | None,
    tau_frac: float,
    out_path: Path,
) -> dict:
    include_tasks = {task} if task is not None else None
    dataset = CorpusDataset(
        corpus_dir=corpus_dir, split=split, split_seed=split_seed,
        include_tasks=include_tasks,
    )
    graph_ids = _unique_graph_indices(dataset)
    print(f"[2.3c] {len(dataset)} samples, {len(graph_ids)} unique graphs")

    model, cfg = _load_encoder(checkpoint, summary, dataset)
    euclidean = cfg["model"] == "euclidean"
    c_val = getattr(model, "c", torch.tensor(float(cfg["curvature"])))

    per_graph: list[dict] = []
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
            result = _evaluate_graph(
                emb=emb, x=x, c_val=c_val, euclidean=euclidean, tau_frac=tau_frac,
            )
            result["graph_idx"] = gi
            per_graph.append(result)

    # Aggregate across graphs.
    def agg_ari(condition: str, label: str) -> dict:
        vals = [
            g["results"][condition][label]["ari_mean"]
            for g in per_graph
            if label in g["results"][condition]
            and g["results"][condition][label]["ari_mean"] == g["results"][condition][label]["ari_mean"]
        ]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "median": statistics.median(vals),
            "min": min(vals), "max": max(vals), "n": len(vals),
        }

    summary_out: dict[str, dict[str, dict]] = {"unfiltered": {}, "filtered": {}}
    for cond in ("unfiltered", "filtered"):
        for label_name, _k in LABEL_CONFIGS:
            summary_out[cond][label_name] = agg_ari(cond, label_name)

    # Collapse rate across graphs
    collapse_rate = statistics.mean(
        [g["n_collapsed_nodes"] / g["n_nodes"] for g in per_graph
         if g["n_nodes"] > 0]
    ) if per_graph else float("nan")

    results = {
        "checkpoint": str(checkpoint),
        "model_kind": cfg["model"],
        "task": task,
        "tau_frac": tau_frac,
        "n_restarts_per_kmeans": N_RESTARTS,
        "n_graphs": len(per_graph),
        "mean_collapse_rate": collapse_rate,
        "per_graph": per_graph,
        "summary": summary_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    _print_summary(results)
    return results


def _print_summary(r: dict) -> None:
    print()
    print("=" * 88)
    print(f"Experiment 2.3c — Clustering ARI ({r['n_graphs']} graphs, "
          f"{r['n_restarts_per_kmeans']} k-means restarts)")
    print(f"checkpoint: {r['checkpoint']}   model: {r['model_kind']}")
    print(f"mean_collapse_rate: {r['mean_collapse_rate']:.4f}")
    print("=" * 88)

    print(f"\n{'condition':<14} {'label':<8} {'k':>3}  {'ARI_mean':>12} "
          f"{'ARI_std':>10} {'range':>18}")
    print("-" * 88)
    for cond in ("unfiltered", "filtered"):
        for label_name, k in LABEL_CONFIGS:
            b = r["summary"][cond][label_name]
            if b["n"] == 0:
                continue
            print(f"{cond:<14} {label_name:<8} {k:>3}  "
                  f"{b['mean']:+12.4f} {b['std']:>10.4f}  "
                  f"[{b['min']:+.4f}, {b['max']:+.4f}]")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--summary", type=str, default=None)
    p.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--task", type=int, default=2)
    p.add_argument("--tau-frac", type=float, default=DEFAULT_TAU_FRAC)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    summary = Path(args.summary) if args.summary else checkpoint.parent / "summary.json"
    out = Path(args.out) if args.out else checkpoint.parent / "retrieval_clustering.json"
    evaluate_checkpoint(
        checkpoint=checkpoint, summary=summary, corpus_dir=args.corpus,
        split=args.split, split_seed=args.split_seed,
        task=None if args.task < 0 else args.task,
        tau_frac=args.tau_frac, out_path=out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

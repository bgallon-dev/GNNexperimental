r"""Experiment 2.3a (collapse-corrected) — NN retrieval quality with a
distance-threshold filter that excludes near-duplicate neighbours.

Rationale.  The collapse investigation showed that ~10% of all pairs
in Euclidean-v3 embeddings and ~1.4% of all pairs in hyperbolic-v3
embeddings are near-duplicates (dist < 1e-4 × median pairwise dist
on that graph).  The original 2.3a metrics counted these collapsed
pairs as top-k NN, inflating same_type/same_layer rates, particularly
for Euclidean.

This script recomputes the three 2.3a metrics under two conditions:

    unfiltered      — matches the original 2.3a output (for verification)
    filtered        — top-k NN with dist < τ × median_dist excluded
                      (τ = 1e-4, tracked via the collapse investigation)

Metrics computed under both conditions:
    same_type_frac@k   — fraction of NN sharing seed's node_type
    same_layer_frac@k  — fraction of NN sharing seed's layer
    hop_dist_mean@k    — mean BFS hop distance to NN

The same-seed NN-replacement strategy (option (a) in the design):
when a candidate NN is excluded by the filter, advance to the next-
closest candidate to keep k constant.  This measures "NN quality
once collapse is removed", not "metrics over partial data".

Also tracks how many seeds had NN altered by filtering, and the
total collapse rate among top-k candidates.

Usage
-----
    py src/modelsv3/eval_retrieval_nn_filtered.py \\
        --checkpoint runs/v3_hyp_compute_seed0/encoder.pt \\
        --summary    runs/v3_hyp_compute_seed0/summary.json \\
        --out        runs/v3_hyp_compute_seed0/retrieval_nn_filtered.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv2.layers import poincare_ops as P  # noqa: E402
from src.modelsv3.distance_ops import chunked_topk  # noqa: E402
from src.modelsv3.distance_ops import (  # noqa: E402
    pairwise_distance_matrix as _pairwise_dist,
)
from src.modelsv3.euclidean_v3 import EuclideanReasonerV3  # noqa: E402
from src.modelsv3.hyperbolic_gnnV3 import KettleGraphReasonerV3  # noqa: E402


NODE_TYPE_SLICE = slice(0, 12)
LAYER_SLICE = slice(12, 16)
K_VALUES = (1, 5, 10)
DEFAULT_TAU_FRAC = 1e-4  # filter threshold as fraction of graph median dist


# ---------------------------------------------------------------------------
# model reconstruction (same pattern as 2.3a)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# BFS hop matrix (reused from 2.3a)
# ---------------------------------------------------------------------------

def _bfs_hop_matrix(edge_index: np.ndarray, N: int) -> np.ndarray:
    adj: list[list[int]] = [[] for _ in range(N)]
    for s, t in zip(edge_index[0], edge_index[1]):
        si, ti = int(s), int(t)
        if si == ti:
            continue
        adj[si].append(ti)
        adj[ti].append(si)
    dist = np.full((N, N), -1, dtype=np.int32)
    for start in range(N):
        dist[start, start] = 0
        q: deque[int] = deque([start])
        while q:
            u = q.popleft()
            d = dist[start, u]
            for v in adj[u]:
                if dist[start, v] == -1:
                    dist[start, v] = d + 1
                    q.append(v)
    return dist


# _pairwise_dist is the cap-guarded full-matrix helper imported above.
# The collapse-correction internals (_compute_graph_metrics_filtered,
# _qualitative_dump) genuinely need full per-row sorted distances for
# the tau-filtered walk; at N <= cap that is the IDENTICAL full matrix
# (bit-exact), and >cap it raises loudly rather than OOM (this is a
# small-graph diagnostic, never run at scale). The headline
# degree-stratified edge_prec below uses chunked_topk so it scales.


# ---------------------------------------------------------------------------
# v3.1 Eval-C: degree-stratified edge precision@k
# ---------------------------------------------------------------------------

EDGE_PREC_K = 5


def _degree_stratified_edge_precision(
    emb: Tensor, edge_index: Tensor, c: float | Tensor, euclidean: bool,
    k: int = EDGE_PREC_K,
) -> dict:
    r"""``nn_edge_precision@k`` (same per-node definition as
    ``intrinsic_eval.nn_edge_precision_at_k``) split by out-degree tercile.

    Tests whether the encoder's structural signal is hub-carried: if
    ``high``-degree nodes drive ``all`` but ``low`` collapses to the
    random baseline, the manifold is not preserving the structure of
    obscure (low-degree) entities — which archival retrieval cannot
    treat as disposable.

    Terciles are rank-based (sort by out-degree, split into 3 contiguous
    groups) so they are robust to the skewed integer degree distribution.
    Per-tercile random baseline = ``mean_out_degree_in_bucket / (N-1)``.
    """
    N = emb.size(0)
    k = int(min(k, N - 1))
    # Row-chunked, self-masked top-k — bit-identical to the old
    # full-matrix topk; scales to large N.
    topk = chunked_topk(emb, k, c, euclidean)  # (N, k) int64

    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    neighbors: list[set[int]] = [set() for _ in range(N)]
    for s, d in zip(src, dst):
        neighbors[int(s)].add(int(d))

    hits = np.zeros(N, dtype=np.float64)
    for i in range(N):
        nbrs = neighbors[i]
        if nbrs:
            hits[i] = sum(1 for j in topk[i] if int(j) in nbrs) / k
    degree = np.array([len(s) for s in neighbors], dtype=np.int64)

    def _bucket(idx: np.ndarray) -> dict:
        if idx.size == 0:
            return {"mean_precision": float("nan"), "mean_out_degree": float("nan"),
                    "random_baseline": float("nan"), "n_nodes": 0}
        md = float(degree[idx].mean())
        return {
            "mean_precision": float(hits[idx].mean()),
            "mean_out_degree": md,
            "random_baseline": min(1.0, md / max(N - 1, 1)),
            "n_nodes": int(idx.size),
        }

    order = np.argsort(degree, kind="stable")
    thirds = np.array_split(order, 3)
    return {
        "k": k,
        "by_degree_tercile": {
            "low": _bucket(thirds[0]),
            "mid": _bucket(thirds[1]),
            "high": _bucket(thirds[2]),
            "all": _bucket(np.arange(N)),
        },
    }


# ---------------------------------------------------------------------------
# top-k with optional filtering (the core of this script)
# ---------------------------------------------------------------------------

def _topk_with_filter(
    D_row: np.ndarray, k: int, tau: float | None
) -> tuple[np.ndarray, int]:
    """Return indices of the k nearest neighbours of a seed, after
    excluding any neighbour with dist < tau.

    When tau is None, no filter is applied — reproduces 2.3a.

    Returns:
        indices: shape (k_actual,) where k_actual <= k.  k_actual may
            be less than k only if the graph has fewer than k valid
            candidates (rare).
        n_collapse_excluded: how many NN candidates were excluded by
            the filter during the top-k selection.  Since we use
            next-closest replacement, this is the number of slots that
            would have been filled by collapsed pairs under the
            unfiltered selection but were replaced.
    """
    # Sort all other nodes by distance to seed.
    order = np.argsort(D_row)
    # Skip self (dist = inf) and any below tau (collapsed).
    valid: list[int] = []
    n_excluded = 0
    for idx in order:
        d = D_row[int(idx)]
        if not np.isfinite(d):
            continue
        if tau is not None and d < tau:
            n_excluded += 1
            continue
        valid.append(int(idx))
        if len(valid) >= k:
            break
    return np.array(valid, dtype=np.int64), n_excluded


def _compute_graph_metrics_filtered(
    emb: Tensor,
    edge_index: Tensor,
    node_types: np.ndarray,
    node_layers: np.ndarray,
    c: float | Tensor,
    euclidean: bool,
    k_values: tuple[int, ...],
    tau_frac: float,
) -> dict:
    """Metrics under both unfiltered and filtered (tau = tau_frac × median)
    selection, plus bookkeeping about how the filter affected the NN set.
    """
    N = emb.size(0)
    D = _pairwise_dist(emb, c, euclidean).detach().cpu().numpy()
    np.fill_diagonal(D, np.inf)
    hop_matrix = _bfs_hop_matrix(edge_index.detach().cpu().numpy(), N)
    reachable = hop_matrix[hop_matrix >= 0]
    diameter = int(reachable.max()) if reachable.size > 0 else 0
    unreachable_sentinel = diameter + 1

    # Graph median pairwise distance for the tau threshold.
    finite_offdiag = D[np.isfinite(D)]
    median_dist = float(np.median(finite_offdiag)) if finite_offdiag.size > 0 else float("nan")
    tau = tau_frac * median_dist if median_dist == median_dist else None

    # Precompute unfiltered topk at max k (trivial; we just sort once).
    max_k = max(k_values)

    results: dict[str, dict] = {"unfiltered": {}, "filtered": {}}
    collapse_stats_per_k: dict[str, dict] = {}

    for k in k_values:
        if k > N - 1:
            results["unfiltered"][f"k={k}"] = {}
            results["filtered"][f"k={k}"] = {}
            continue

        # For each seed, run both unfiltered and filtered topk.
        unf_same_type = np.zeros(N, dtype=np.float64)
        unf_same_layer = np.zeros(N, dtype=np.float64)
        unf_hop_means = np.zeros(N, dtype=np.float64)

        fil_same_type = np.zeros(N, dtype=np.float64)
        fil_same_layer = np.zeros(N, dtype=np.float64)
        fil_hop_means = np.zeros(N, dtype=np.float64)

        n_affected_seeds = 0
        total_collapse_exclusions = 0
        total_collapse_slots = 0  # how many NN slots (k × N) could have been collapsed

        for i in range(N):
            D_row = D[i]
            # unfiltered
            nbrs_unf, _ = _topk_with_filter(D_row, k, tau=None)
            # filtered
            nbrs_fil, n_excl = _topk_with_filter(D_row, k, tau=tau)
            if n_excl > 0:
                n_affected_seeds += 1
                total_collapse_exclusions += n_excl

            ti = node_types[i]
            li = node_layers[i]

            def fill(nbrs: np.ndarray) -> tuple[float, float, float]:
                if len(nbrs) == 0:
                    return 0.0, 0.0, 0.0
                st = float(np.sum(node_types[nbrs] == ti) / len(nbrs)) if ti >= 0 else 0.0
                sl = float(np.sum(node_layers[nbrs] == li) / len(nbrs)) if li >= 0 else 0.0
                hops = []
                for j in nbrs:
                    h = hop_matrix[i, int(j)]
                    hops.append(float(h) if h >= 0 else unreachable_sentinel)
                hm = float(np.mean(hops))
                return st, sl, hm

            unf_same_type[i], unf_same_layer[i], unf_hop_means[i] = fill(nbrs_unf)
            fil_same_type[i], fil_same_layer[i], fil_hop_means[i] = fill(nbrs_fil)

        results["unfiltered"][f"k={k}"] = {
            "same_type_frac_mean": float(unf_same_type.mean()),
            "same_layer_frac_mean": float(unf_same_layer.mean()),
            "hop_dist_mean": float(unf_hop_means.mean()),
        }
        results["filtered"][f"k={k}"] = {
            "same_type_frac_mean": float(fil_same_type.mean()),
            "same_layer_frac_mean": float(fil_same_layer.mean()),
            "hop_dist_mean": float(fil_hop_means.mean()),
        }
        collapse_stats_per_k[f"k={k}"] = {
            "n_seeds_affected_by_filter": int(n_affected_seeds),
            "n_seeds_total": int(N),
            "frac_seeds_affected": float(n_affected_seeds / N),
            "total_NN_slots": int(k * N),
            "total_collapse_exclusions": int(total_collapse_exclusions),
        }

    return {
        "graph_context": {
            "n_nodes": N,
            "median_dist": median_dist,
            "tau_frac": tau_frac,
            "tau_absolute": tau,
            "diameter": diameter,
            "unreachable_sentinel": unreachable_sentinel,
        },
        "metrics": results,
        "collapse_stats": collapse_stats_per_k,
    }


# ---------------------------------------------------------------------------
# qualitative dump — annotated with collapse flags
# ---------------------------------------------------------------------------

def _qualitative_dump(
    emb: Tensor,
    edge_index: Tensor,
    node_types: np.ndarray,
    node_layers: np.ndarray,
    c: float | Tensor,
    euclidean: bool,
    graph_idx: int,
    tau: float,
    k: int = 5,
) -> dict:
    N = emb.size(0)
    D = _pairwise_dist(emb, c, euclidean).detach().cpu().numpy()
    np.fill_diagonal(D, np.inf)
    hop_matrix = _bfs_hop_matrix(edge_index.detach().cpu().numpy(), N)

    # Pick 3 seeds spanning layers (same as 2.3a).
    seed_indices: list[int] = []
    picked_layers: set[int] = set()
    for i in range(N):
        li = int(node_layers[i])
        if li < 0 or li in picked_layers:
            continue
        seed_indices.append(i)
        picked_layers.add(li)
        if len(seed_indices) >= 3:
            break
    for i in range(N):
        if len(seed_indices) >= 3:
            break
        if i not in seed_indices:
            seed_indices.append(i)

    seeds_dump: list[dict] = []
    for seed_i in seed_indices:
        order = np.argsort(D[seed_i])
        # Take the first k distinct from self (skip inf diagonal).
        selected: list[int] = []
        for idx in order:
            d = D[seed_i, int(idx)]
            if not np.isfinite(d):
                continue
            selected.append(int(idx))
            if len(selected) >= k:
                break
        entries = []
        for rank, j in enumerate(selected):
            d = float(D[seed_i, j])
            hop = int(hop_matrix[seed_i, j])
            entries.append({
                "rank": rank + 1,
                "node_idx": j,
                "node_type": int(node_types[j]),
                "layer": int(node_layers[j]),
                "distance": d,
                "hop_from_seed": hop if hop >= 0 else None,
                "flagged_as_collapse": bool(d < tau),
            })
        seeds_dump.append({
            "seed_idx": int(seed_i),
            "seed_type": int(node_types[seed_i]),
            "seed_layer": int(node_layers[seed_i]),
            "top_k_nn": entries,
        })
    return {
        "graph_idx": graph_idx,
        "n_nodes": N,
        "k": k,
        "tau": tau,
        "seeds": seeds_dump,
    }


# ---------------------------------------------------------------------------
# top-level driver
# ---------------------------------------------------------------------------

def _unique_graph_indices(dataset: CorpusDataset) -> list[int]:
    seen: list[int] = []
    seen_set: set[int] = set()
    for graph_idx, _task_idx in dataset.index:
        gi = int(graph_idx)
        if gi not in seen_set:
            seen_set.add(gi)
            seen.append(gi)
    return seen


def _median_size_graph(dataset: CorpusDataset, graph_ids: list[int]) -> int:
    sizes: list[tuple[int, int]] = []
    for gi in graph_ids:
        graph = dataset._get_graph(gi)
        sizes.append((gi, int(graph["x"].size(0))))
    sizes.sort(key=lambda t: t[1])
    return sizes[len(sizes) // 2][0]


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
    print(f"[2.3a-filtered] {len(dataset)} samples, {len(graph_ids)} unique graphs")
    print(f"[2.3a-filtered] tau = {tau_frac} × median_dist per graph")

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
            x_np = x.detach().cpu().numpy()
            tb = x_np[:, NODE_TYPE_SLICE]
            lb = x_np[:, LAYER_SLICE]
            node_types = np.where(tb.sum(axis=1) > 0, tb.argmax(axis=1), -1)
            node_layers = np.where(lb.sum(axis=1) > 0, lb.argmax(axis=1), -1)
            out_block = _compute_graph_metrics_filtered(
                emb=emb, edge_index=edge_index,
                node_types=node_types, node_layers=node_layers,
                c=c_val, euclidean=euclidean,
                k_values=K_VALUES, tau_frac=tau_frac,
            )
            out_block["degree_stratified_edge_prec"] = (
                _degree_stratified_edge_precision(
                    emb=emb, edge_index=edge_index,
                    c=c_val, euclidean=euclidean,
                )
            )
            out_block["graph_idx"] = gi
            per_graph.append(out_block)

    # Aggregate: summary across graphs per condition and per k.
    def agg(condition: str, k_key: str, metric: str) -> dict:
        vals = [
            g["metrics"][condition][k_key][metric]
            for g in per_graph if k_key in g["metrics"][condition]
        ]
        clean = [v for v in vals if v == v]
        if not clean:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {
            "mean": statistics.mean(clean),
            "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
            "median": statistics.median(clean),
            "min": min(clean),
            "max": max(clean),
            "n": len(clean),
        }

    summary_out: dict[str, dict[str, dict[str, dict]]] = {
        "unfiltered": {}, "filtered": {}
    }
    for cond in ("unfiltered", "filtered"):
        for k in K_VALUES:
            k_key = f"k={k}"
            summary_out[cond][k_key] = {
                "same_type_frac_mean": agg(cond, k_key, "same_type_frac_mean"),
                "same_layer_frac_mean": agg(cond, k_key, "same_layer_frac_mean"),
                "hop_dist_mean": agg(cond, k_key, "hop_dist_mean"),
            }

    # Collapse statistics: aggregate across graphs.
    collapse_agg: dict[str, dict] = {}
    for k in K_VALUES:
        k_key = f"k={k}"
        fracs = [g["collapse_stats"][k_key]["frac_seeds_affected"] for g in per_graph]
        exclusions = [g["collapse_stats"][k_key]["total_collapse_exclusions"] for g in per_graph]
        slots = [g["collapse_stats"][k_key]["total_NN_slots"] for g in per_graph]
        collapse_agg[k_key] = {
            "mean_frac_seeds_affected": statistics.mean(fracs) if fracs else float("nan"),
            "total_exclusions_over_total_slots": (
                sum(exclusions) / sum(slots) if sum(slots) else float("nan")
            ),
        }

    # Qualitative dump on median-size graph.
    target_gi = _median_size_graph(dataset, graph_ids)
    with torch.no_grad():
        graph = dataset._get_graph(target_gi)
        x = graph["x"]
        edge_index = graph["edge_index"]
        out = model(
            x, edge_index, graph["edge_type"], graph["edge_descriptor"],
            node_descriptor=graph["node_descriptor"],
        )
        emb = out.node_embeddings.detach().cpu()
        x_np = x.detach().cpu().numpy()
        node_types = np.where(
            x_np[:, NODE_TYPE_SLICE].sum(axis=1) > 0,
            x_np[:, NODE_TYPE_SLICE].argmax(axis=1), -1,
        )
        node_layers = np.where(
            x_np[:, LAYER_SLICE].sum(axis=1) > 0,
            x_np[:, LAYER_SLICE].argmax(axis=1), -1,
        )
        # Recover tau for this specific graph.
        D = _pairwise_dist(emb, c_val, euclidean).detach().cpu().numpy()
        finite = D[np.isfinite(D) & (D > 0)]
        med = float(np.median(finite)) if finite.size > 0 else float("nan")
        tau_here = tau_frac * med if med == med else 0.0
        qual = _qualitative_dump(
            emb=emb, edge_index=edge_index,
            node_types=node_types, node_layers=node_layers,
            c=c_val, euclidean=euclidean, graph_idx=target_gi,
            tau=tau_here, k=5,
        )

    # v3.1 Eval-C: aggregate degree-stratified edge precision across graphs.
    def _deg_agg(bucket: str, field: str) -> dict:
        vals = [
            g["degree_stratified_edge_prec"]["by_degree_tercile"][bucket][field]
            for g in per_graph
            if "degree_stratified_edge_prec" in g
        ]
        clean = [v for v in vals if v == v]
        if not clean:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {
            "mean": statistics.mean(clean),
            "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
            "min": min(clean), "max": max(clean), "n": len(clean),
        }

    edge_prec_by_tercile = {
        bucket: {
            "mean_precision": _deg_agg(bucket, "mean_precision"),
            "mean_out_degree": _deg_agg(bucket, "mean_out_degree"),
            "random_baseline": _deg_agg(bucket, "random_baseline"),
        }
        for bucket in ("low", "mid", "high", "all")
    }

    results = {
        "checkpoint": str(checkpoint),
        "model_kind": cfg["model"],
        "split": split,
        "task": task,
        "tau_frac": tau_frac,
        "n_graphs": len(per_graph),
        "per_graph": per_graph,
        "summary_across_graphs": summary_out,
        "collapse_stats_aggregate": collapse_agg,
        f"edge_prec@{EDGE_PREC_K}_by_degree_tercile": edge_prec_by_tercile,
        "qualitative": qual,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    _print_summary(results)
    return results


def _print_summary(r: dict) -> None:
    print()
    print("=" * 96)
    print(f"Experiment 2.3a (collapse-corrected) — {r['n_graphs']} graphs, "
          f"tau = {r['tau_frac']} × median_dist")
    print(f"checkpoint: {r['checkpoint']}   model: {r['model_kind']}")
    print("=" * 96)

    for k_key in ("k=1", "k=5", "k=10"):
        unf = r["summary_across_graphs"]["unfiltered"].get(k_key, {})
        fil = r["summary_across_graphs"]["filtered"].get(k_key, {})
        if not unf or not fil:
            continue
        stats = r["collapse_stats_aggregate"].get(k_key, {})
        print(f"\n{k_key}:   [frac_seeds_affected_by_filter = "
              f"{stats.get('mean_frac_seeds_affected', float('nan')):.3f},   "
              f"total_exclusion_rate = "
              f"{stats.get('total_exclusions_over_total_slots', float('nan')):.4f}]")
        print(f"  {'metric':<22} {'unfiltered':>18} {'filtered':>18} {'change':>14}")
        for mname in ("same_type_frac_mean", "same_layer_frac_mean", "hop_dist_mean"):
            u = unf[mname]
            f = fil[mname]
            delta = f["mean"] - u["mean"]
            print(f"  {mname:<22} "
                  f"{u['mean']:+.4f} ± {u['std']:.4f}  "
                  f"{f['mean']:+.4f} ± {f['std']:.4f}  "
                  f"{delta:+.4f}")

    dt_key = f"edge_prec@{EDGE_PREC_K}_by_degree_tercile"
    dt = r.get(dt_key)
    if dt:
        print(f"\n{dt_key} (is the structural signal hub-carried?):")
        print(f"  {'tercile':<8}{'edge_prec':>12}{'random':>12}{'out_deg':>10}")
        for bucket in ("low", "mid", "high", "all"):
            b = dt[bucket]
            mp = b["mean_precision"]["mean"]
            rb = b["random_baseline"]["mean"]
            od = b["mean_out_degree"]["mean"]
            print(f"  {bucket:<8}{mp:>12.4f}{rb:>12.4f}{od:>10.3f}")

    q = r["qualitative"]
    print(f"\nQualitative dump on graph {q['graph_idx']} (N={q['n_nodes']}, k={q['k']}, "
          f"tau={q['tau']:.3e}):")
    for s in q["seeds"]:
        print(f"  Seed idx={s['seed_idx']:3d}  type={s['seed_type']}  "
              f"layer={s['seed_layer']}")
        for nn in s["top_k_nn"]:
            hop_str = f"{nn['hop_from_seed']}" if nn["hop_from_seed"] is not None else "unreach"
            flag = "  [COLLAPSE]" if nn["flagged_as_collapse"] else ""
            print(f"    #{nn['rank']}  idx={nn['node_idx']:3d}  type={nn['node_type']}  "
                  f"layer={nn['layer']}  dist={nn['distance']:.3e}  hop={hop_str}{flag}")


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
    out = Path(args.out) if args.out else checkpoint.parent / "retrieval_nn_filtered.json"
    evaluate_checkpoint(
        checkpoint=checkpoint, summary=summary, corpus_dir=args.corpus,
        split=args.split, split_seed=args.split_seed,
        task=None if args.task < 0 else args.task, tau_frac=args.tau_frac,
        out_path=out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

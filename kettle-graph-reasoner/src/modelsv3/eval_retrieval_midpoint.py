r"""Experiment 2.3b — Geodesic midpoint retrieval.

For each val graph, sample node pairs (u, v) with BFS graph distance
>= 3, compute their geodesic midpoint in embedding space, retrieve
top-3 nearest neighbours to that midpoint, and check whether any NN
lies on a shortest BFS path from u to v in the undirected graph.

Midpoint construction:
    hyperbolic: m = expmap0((logmap0(u) + logmap0(v)) / 2)
    euclidean : m = (u + v) / 2

Collapse filter applied: nodes within 1e-4 × median_dist of m are
excluded from the top-k NN candidate set, same τ as 2.3a_filtered.

Metrics:
    path_hit_rate@k  — fraction of (u, v) pairs where at least one
                        of the top-k NN to m lies on some shortest
                        BFS path u → v (undirected, length d = hop(u,v)).
                        Excludes u and v themselves from NN set.
    mean_nn_hop_from_path — for each NN, distance to the nearest node
                            on some shortest u→v path. Reported as a
                            secondary metric.

Predictions (from prior findings):
    hyperbolic : moderate path_hit_rate because graph structure is
                 preserved in the embedding.
    euclidean  : low path_hit_rate because midpoints may fall into
                 the origin attractor, destroying structural meaning.
    random     : (d-1)/N approximately, where d = hop distance, N = nodes.

Usage
-----
    py src/modelsv3/eval_retrieval_midpoint.py \\
        --checkpoint runs/v3_hyp_compute_seed0/encoder.pt \\
        --out        runs/v3_hyp_compute_seed0/retrieval_midpoint.json
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
from src.modelsv3.distance_ops import (  # noqa: E402
    EXACT_PAIR_NODE_CAP,
    MAX_SAMPLED_PAIRS,
    pairwise_distance_matrix,
    sampled_pair_dists,
)
from src.modelsv3.euclidean_v3 import EuclideanReasonerV3  # noqa: E402
from src.modelsv3.hyperbolic_gnnV3 import KettleGraphReasonerV3  # noqa: E402


K_VALUES = (1, 3, 5)
MIN_DIST = 3
DEFAULT_N_PAIRS = 20
DEFAULT_TAU_FRAC = 1e-4


# ---------------------------------------------------------------------------
# loader
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


def _unique_graph_indices(dataset: CorpusDataset) -> list[int]:
    seen: list[int] = []
    seen_set: set[int] = set()
    for graph_idx, _task_idx in dataset.index:
        gi = int(graph_idx)
        if gi not in seen_set:
            seen_set.add(gi)
            seen.append(gi)
    return seen


# ---------------------------------------------------------------------------
# BFS
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


def _nodes_on_any_shortest_path(
    hop_matrix: np.ndarray, u: int, v: int
) -> set[int]:
    """Return the set of nodes w such that hop(u, w) + hop(w, v) == hop(u, v),
    and both legs are reachable."""
    d_uv = hop_matrix[u, v]
    if d_uv < 0:
        return set()
    d_u = hop_matrix[u]
    d_v = hop_matrix[v]
    reachable = (d_u >= 0) & (d_v >= 0)
    on_path = reachable & (d_u + d_v == d_uv)
    return set(int(w) for w in np.where(on_path)[0])


def _graph_distance_error(
    hop_matrix: np.ndarray, u: int, v: int, nodes: list[int]
) -> list[float]:
    r"""v3.1 Eval-B addition: how far each retrieved node ``w`` is from
    being a true graph midpoint of ``(u, v)``.

    For an ideal bridge, ``hop(u, w) == hop(v, w) == hop(u, v) / 2``.
    Error per node: ``|hop(u,w) - d/2| + |hop(v,w) - d/2|`` where
    ``d = hop(u, v)``. Nodes with an unreachable leg are skipped (they
    are not bridges by construction). Returns one error per finite
    ``w`` in ``nodes`` (empty if none reachable)."""
    d_uv = hop_matrix[u, v]
    if d_uv < 0:
        return []
    half = d_uv / 2.0
    errs: list[float] = []
    for w in nodes:
        du = hop_matrix[u, w]
        dv = hop_matrix[v, w]
        if du < 0 or dv < 0:
            continue
        errs.append(abs(float(du) - half) + abs(float(dv) - half))
    return errs


# ---------------------------------------------------------------------------
# midpoint + distance
# ---------------------------------------------------------------------------

def _midpoint(u_emb: Tensor, v_emb: Tensor, c: float | Tensor, euclidean: bool) -> Tensor:
    if euclidean:
        return (u_emb + v_emb) / 2
    # Poincaré ball: tangent-space midpoint via logmap0/expmap0.
    u_tan = P.logmap0(u_emb.unsqueeze(0), c).squeeze(0)
    v_tan = P.logmap0(v_emb.unsqueeze(0), c).squeeze(0)
    m_tan = (u_tan + v_tan) / 2
    m = P.expmap0(m_tan.unsqueeze(0), c).squeeze(0)
    return m


def _dist_to_all(
    m: Tensor, all_emb: Tensor, c: float | Tensor, euclidean: bool
) -> Tensor:
    if euclidean:
        return torch.cdist(m.unsqueeze(0), all_emb, p=2).squeeze(0)
    # Broadcast hyperbolic distance: m (d,), all_emb (N, d) -> (N,)
    m_exp = m.unsqueeze(0).expand_as(all_emb)
    return P.dist(m_exp, all_emb, c, keepdim=False)


# The full pairwise matrix was only used for the tau-median; replaced
# by offdiag_pair_dists (exact upper-triangle <= cap == bit-identical
# median for a symmetric metric; sampled > cap so midpoint scales).
# Per-pair retrieval uses _dist_to_all (point-to-all, already O(N)).


# ---------------------------------------------------------------------------
# pair sampling
# ---------------------------------------------------------------------------

def _sample_pairs(
    hop_matrix: np.ndarray, n_pairs: int, seed: int, min_dist: int = MIN_DIST
) -> list[tuple[int, int]]:
    """Sample (u, v) pairs with hop(u, v) >= min_dist, uniformly at random
    without replacement from the valid set."""
    N = hop_matrix.shape[0]
    rng = np.random.default_rng(seed)
    # Enumerate all eligible pairs, upper triangle to avoid duplicates.
    ii, jj = np.triu_indices(N, k=1)
    dists = hop_matrix[ii, jj]
    eligible = np.where((dists >= min_dist) & (dists < N + 1))[0]
    if len(eligible) == 0:
        return []
    n_sample = min(n_pairs, len(eligible))
    chosen = rng.choice(eligible, size=n_sample, replace=False)
    return [(int(ii[k]), int(jj[k])) for k in chosen]


# ---------------------------------------------------------------------------
# per-graph evaluation
# ---------------------------------------------------------------------------

def _evaluate_graph(
    emb: Tensor,
    edge_index: Tensor,
    c_val: float | Tensor,
    euclidean: bool,
    pair_seed: int,
    n_pairs: int,
    tau_frac: float,
) -> dict:
    N = emb.size(0)
    ei_np = edge_index.detach().cpu().numpy()
    hop_matrix = _bfs_hop_matrix(ei_np, N)

    # tau for collapse filter based on pairwise median dist. At N<=cap
    # this is LITERALLY the old code path (full matrix, fill_diagonal,
    # median over finite off-diagonal) -> bit-identical. P.dist is only
    # symmetric to float32 ULP, so the old median over BOTH triangles
    # must be reproduced exactly; the upper-triangle-only path drifted
    # ~2e-7. Above the cap we sample (midpoint then scales).
    if N <= EXACT_PAIR_NODE_CAP:
        _D = pairwise_distance_matrix(emb, c_val, euclidean).detach().cpu().numpy()
        np.fill_diagonal(_D, np.inf)
        _finite = _D[np.isfinite(_D)]
        median_dist = float(np.median(_finite)) if _finite.size > 0 else float("nan")
    else:
        _od = sampled_pair_dists(
            emb, c_val, euclidean, MAX_SAMPLED_PAIRS,
            np.random.default_rng(pair_seed))
        median_dist = float(np.median(_od)) if _od.size > 0 else float("nan")
    tau = tau_frac * median_dist if median_dist == median_dist else 0.0

    pairs = _sample_pairs(hop_matrix, n_pairs=n_pairs, seed=pair_seed)
    if not pairs:
        return {
            "n_pairs_sampled": 0,
            "per_pair": [],
            "path_hit_rate_at_k": {f"k={k}": float("nan") for k in K_VALUES},
            "random_baseline": float("nan"),
            "n_valid_midpoints": 0,
            "median_dist": median_dist,
        }

    per_pair: list[dict] = []
    hits_at_k: dict[int, int] = {k: 0 for k in K_VALUES}
    # v3.1 Eval-B: graph-distance error of the retrieved bridge nodes.
    bridge_err_min_at_k: dict[int, list[float]] = {k: [] for k in K_VALUES}
    bridge_err_mean_at_k: dict[int, list[float]] = {k: [] for k in K_VALUES}
    mean_hop_from_path: list[float] = []
    # random baseline: mean fraction of nodes on some u→v path, across the
    # sampled pairs. This is the probability a random node lies on a
    # shortest path.
    random_hits = 0.0

    with torch.no_grad():
        for u, v in pairs:
            u_emb = emb[u]
            v_emb = emb[v]
            m = _midpoint(u_emb, v_emb, c_val, euclidean)
            d_all = _dist_to_all(m, emb, c_val, euclidean).detach().cpu().numpy()

            # Exclude u, v, and collapsed (dist < tau) nodes.
            excluded = {u, v}
            # Sort candidates by distance to m.
            order = np.argsort(d_all)
            kept: list[int] = []
            for idx in order:
                idx_i = int(idx)
                if idx_i in excluded:
                    continue
                if d_all[idx_i] < tau:
                    continue
                kept.append(idx_i)
                if len(kept) >= max(K_VALUES):
                    break

            path_nodes = _nodes_on_any_shortest_path(hop_matrix, u, v)
            path_nodes -= {u, v}  # exclude the endpoints from the target set
            random_hits += (len(path_nodes) / max(N - 2, 1))

            nn_hops_to_path: list[int] = []
            for w in kept[: max(K_VALUES)]:
                # distance from w to nearest node on path
                d_to_path = min(
                    (int(hop_matrix[w, p])
                     for p in path_nodes if hop_matrix[w, p] >= 0),
                    default=-1,
                )
                nn_hops_to_path.append(d_to_path)

            if nn_hops_to_path:
                finite_hops = [h for h in nn_hops_to_path if h >= 0]
                mean_hop_from_path.append(
                    float(np.mean(finite_hops)) if finite_hops else float("nan")
                )

            pair_bridge_err: dict[str, dict] = {}
            for k in K_VALUES:
                top_k = kept[:k]
                if any(w in path_nodes for w in top_k):
                    hits_at_k[k] += 1
                errs = _graph_distance_error(hop_matrix, u, v, top_k)
                if errs:
                    e_min = float(min(errs))
                    e_mean = float(np.mean(errs))
                    bridge_err_min_at_k[k].append(e_min)
                    bridge_err_mean_at_k[k].append(e_mean)
                    pair_bridge_err[f"k={k}"] = {"min": e_min, "mean": e_mean}

            per_pair.append({
                "u": u, "v": v,
                "hop_dist": int(hop_matrix[u, v]),
                "n_path_nodes": len(path_nodes),
                "top_k_nn": kept[: max(K_VALUES)],
                "nn_hops_to_path": nn_hops_to_path,
                "bridge_graph_dist_error": pair_bridge_err,
            })

    n_pairs_total = len(pairs)
    path_hit_rate = {
        f"k={k}": hits_at_k[k] / n_pairs_total for k in K_VALUES
    }
    random_baseline = random_hits / n_pairs_total

    return {
        "n_pairs_sampled": n_pairs_total,
        "per_pair": per_pair,
        "path_hit_rate_at_k": path_hit_rate,
        "random_baseline": random_baseline,
        "mean_nn_hop_from_path": (
            float(statistics.mean([v for v in mean_hop_from_path if v == v]))
            if mean_hop_from_path else float("nan")
        ),
        "bridge_graph_dist_error_at_k": {
            f"k={k}": {
                "min_mean": (
                    float(np.mean(bridge_err_min_at_k[k]))
                    if bridge_err_min_at_k[k] else float("nan")
                ),
                "mean_mean": (
                    float(np.mean(bridge_err_mean_at_k[k]))
                    if bridge_err_mean_at_k[k] else float("nan")
                ),
                "n": len(bridge_err_min_at_k[k]),
            }
            for k in K_VALUES
        },
        "median_dist": median_dist,
        "n_nodes": N,
    }


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
    n_pairs: int,
    tau_frac: float,
    out_path: Path,
) -> dict:
    include_tasks = {task} if task is not None else None
    dataset = CorpusDataset(
        corpus_dir=corpus_dir, split=split, split_seed=split_seed,
        include_tasks=include_tasks,
    )
    graph_ids = _unique_graph_indices(dataset)
    print(f"[2.3b] {len(dataset)} samples, {len(graph_ids)} unique graphs")

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
                emb=emb, edge_index=edge_index,
                c_val=c_val, euclidean=euclidean,
                pair_seed=gi, n_pairs=n_pairs, tau_frac=tau_frac,
            )
            result["graph_idx"] = gi
            per_graph.append(result)

    # Aggregate across graphs.
    def agg(key_path: list) -> dict:
        vals = []
        for g in per_graph:
            cur = g
            for p in key_path:
                if isinstance(p, str) and p in cur:
                    cur = cur[p]
                else:
                    cur = None
                    break
            if isinstance(cur, (int, float)) and cur == cur:
                vals.append(float(cur))
        if not vals:
            return {"mean": float("nan"), "n": 0}
        return {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "median": statistics.median(vals),
            "min": min(vals), "max": max(vals), "n": len(vals),
        }

    summary_out = {
        f"path_hit_rate@{k}": agg(["path_hit_rate_at_k", f"k={k}"])
        for k in K_VALUES
    }
    summary_out["mean_nn_hop_from_path"] = agg(["mean_nn_hop_from_path"])
    summary_out["random_baseline"] = agg(["random_baseline"])
    # v3.1 Eval-B: best-bridge graph-distance error per k (lower = the
    # geometric midpoint retrieved a near-true graph midpoint).
    for k in K_VALUES:
        summary_out[f"bridge_graph_dist_error@{k}"] = agg(
            ["bridge_graph_dist_error_at_k", f"k={k}", "min_mean"]
        )

    results = {
        "checkpoint": str(checkpoint),
        "model_kind": cfg["model"],
        "task": task,
        "min_pair_distance": MIN_DIST,
        "n_pairs_per_graph": n_pairs,
        "tau_frac": tau_frac,
        "n_graphs": len(per_graph),
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
    print(f"Experiment 2.3b — Geodesic midpoint retrieval ({r['n_graphs']} graphs)")
    print(f"checkpoint: {r['checkpoint']}   model: {r['model_kind']}")
    print(f"min_pair_distance={r['min_pair_distance']}, n_pairs/graph={r['n_pairs_per_graph']}, "
          f"tau_frac={r['tau_frac']}")
    print("=" * 88)

    s = r["summary"]
    for k in K_VALUES:
        key = f"path_hit_rate@{k}"
        if key not in s or s[key]["n"] == 0:
            continue
        v = s[key]
        print(f"  {key:<24} mean={v['mean']:.4f}  std={v['std']:.4f}  "
              f"range=[{v['min']:.4f}, {v['max']:.4f}]")
    rb = s.get("random_baseline")
    if rb and rb["n"] > 0:
        print(f"  {'random_baseline':<24} mean={rb['mean']:.4f}")
    mnh = s.get("mean_nn_hop_from_path")
    if mnh and mnh["n"] > 0:
        print(f"  {'mean_nn_hop_from_path':<24} mean={mnh['mean']:.3f}  "
              f"(0 = NN is on path, higher = NN is off-path)")
    for k in K_VALUES:
        key = f"bridge_graph_dist_error@{k}"
        b = s.get(key)
        if b and b["n"] > 0:
            print(f"  {key:<24} mean={b['mean']:.4f}  "
                  f"(0 = retrieved a true graph midpoint)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--summary", type=str, default=None)
    p.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--task", type=int, default=2)
    p.add_argument("--n-pairs", type=int, default=DEFAULT_N_PAIRS)
    p.add_argument("--tau-frac", type=float, default=DEFAULT_TAU_FRAC)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    summary = Path(args.summary) if args.summary else checkpoint.parent / "summary.json"
    out = Path(args.out) if args.out else checkpoint.parent / "retrieval_midpoint.json"
    evaluate_checkpoint(
        checkpoint=checkpoint, summary=summary, corpus_dir=args.corpus,
        split=args.split, split_seed=args.split_seed,
        task=None if args.task < 0 else args.task,
        n_pairs=args.n_pairs, tau_frac=args.tau_frac, out_path=out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

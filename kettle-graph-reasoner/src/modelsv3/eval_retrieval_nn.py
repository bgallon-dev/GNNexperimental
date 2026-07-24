r"""Experiment 2.3a — seed-node nearest neighbor retrieval quality.

For each val graph, for each node (as seed), retrieve its top-k nearest
embeddings and measure three orthogonal quality metrics:

    same_type_frac@k  — fraction of NN sharing the seed's node_type
                         (x[:, 0:12].argmax()). Random baseline varies
                         per graph based on type distribution.
    hop_dist_mean@k   — mean BFS hop distance from seed to its NN,
                         treating the graph as undirected. Lower is
                         better — NN are graph-structurally close.
                         Unreachable NN contribute a sentinel value
                         (graph_diameter + 1) and are counted separately.
    same_layer_frac@k — fraction of NN at the same layer as the seed
                         (x[:, 12:16].argmax()). The corpus has 4 layers:
                         source / claim / entity / auxiliary.

Also produces a qualitative dump: for the median-size val graph, pick
3 seeds spanning different layers and dump their top-5 NN per metric.

Usage
-----
    py src/modelsv3/eval_retrieval_nn.py \\
        --checkpoint runs/v3_hyp_compute_seed0/encoder.pt \\
        --summary    runs/v3_hyp_compute_seed0/summary.json \\
        --out        runs/v3_hyp_compute_seed0/retrieval_nn.json
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
    pairwise_distance_matrix as _pairwise_distance_matrix,
)
from src.modelsv3.euclidean_v3 import EuclideanReasonerV3  # noqa: E402
from src.modelsv3.hyperbolic_gnnV3 import KettleGraphReasonerV3  # noqa: E402


NODE_TYPE_SLICE = slice(0, 12)
LAYER_SLICE = slice(12, 16)
K_VALUES = (1, 5, 10)


# ---------------------------------------------------------------------------
# model reconstruction (mirrors 2.1's loader)
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
# BFS for hop-distance matrix
# ---------------------------------------------------------------------------

def _bfs_hop_matrix(edge_index: np.ndarray, N: int) -> np.ndarray:
    """(N, N) int array. ``out[i, j]`` = BFS shortest-path distance from
    i to j treating edges as undirected. Unreachable pairs are ``-1``."""
    # Build undirected adjacency list.
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


# ---------------------------------------------------------------------------
# distance computation (reuses 2.1's approach)
# ---------------------------------------------------------------------------

# _pairwise_distance_matrix is imported (cap-guarded) from distance_ops
# above. It is still used by the single-graph _qualitative_dump (median
# graph, <= cap -> exact). The O(N^2) topk consumer below uses
# chunked_topk instead.


# ---------------------------------------------------------------------------
# per-graph metric computation
# ---------------------------------------------------------------------------

def _compute_graph_metrics(
    emb: Tensor,
    edge_index: Tensor,
    node_types: np.ndarray,
    node_layers: np.ndarray,
    c: float | Tensor,
    euclidean: bool,
    k_values: tuple[int, ...],
) -> dict:
    """Run the full per-graph metric computation. Returns a dict with
    per-k metrics averaged across seed nodes."""
    N = emb.size(0)
    hop_matrix = _bfs_hop_matrix(edge_index.detach().cpu().numpy(), N)
    reachable_diameter = int(hop_matrix[hop_matrix >= 0].max()) if (hop_matrix >= 0).any() else 0
    unreachable_sentinel = reachable_diameter + 1

    results: dict[str, dict[str, float]] = {}
    # Run topk once at max k, subset for smaller k. Row-chunked, self-
    # masked — bit-identical to the old full-matrix topk(max_k).
    max_k = max(k_values)
    max_k = min(max_k, N - 1)
    topk_idx = chunked_topk(emb, max_k, c, euclidean)  # (N, max_k)

    # Per-graph baselines for the metrics (for interpretation context).
    # same_type random: E[share of same-type in random k draws]
    # same_layer random: same formula with layer counts
    def _random_share(labels: np.ndarray) -> float:
        _, counts = np.unique(labels[labels >= 0], return_counts=True)
        total = int(counts.sum())
        if total <= 1:
            return 0.0
        return float(np.sum(counts * (counts - 1)) / (total * (total - 1)))

    type_rand = _random_share(node_types)
    layer_rand = _random_share(node_layers)

    # Random hop_dist baseline: mean BFS distance across reachable pairs,
    # which approximates "what would you get from random retrieval?"
    reachable = hop_matrix[hop_matrix >= 0]
    if reachable.size > 0:
        hop_rand = float(reachable.mean())
    else:
        hop_rand = float("nan")

    for k in k_values:
        if k > N - 1:
            results[f"k={k}"] = {}
            continue
        topk_k = topk_idx[:, :k]
        # same_type
        same_type_hits = np.zeros(N, dtype=np.float64)
        # same_layer
        same_layer_hits = np.zeros(N, dtype=np.float64)
        # hop dist — accumulate per-seed averages
        hop_dists: list[float] = []
        unreachable_pairs = 0
        total_pairs = 0
        for i in range(N):
            nbrs = topk_k[i]
            t_i = node_types[i]
            l_i = node_layers[i]
            if t_i >= 0:
                same_type_hits[i] = float(
                    np.sum(node_types[nbrs] == t_i) / k
                )
            if l_i >= 0:
                same_layer_hits[i] = float(
                    np.sum(node_layers[nbrs] == l_i) / k
                )
            # hop dist
            hop_row = hop_matrix[i, nbrs]
            seed_hop_vals = []
            for h in hop_row:
                total_pairs += 1
                if h < 0:
                    unreachable_pairs += 1
                    seed_hop_vals.append(unreachable_sentinel)
                else:
                    seed_hop_vals.append(int(h))
            hop_dists.append(float(np.mean(seed_hop_vals)))

        results[f"k={k}"] = {
            "same_type_frac_mean": float(same_type_hits.mean()),
            "same_layer_frac_mean": float(same_layer_hits.mean()),
            "hop_dist_mean": float(np.mean(hop_dists)),
            "unreachable_pair_frac": (
                unreachable_pairs / total_pairs if total_pairs else 0.0
            ),
            "n_seeds": N,
        }

    return {
        "per_k": results,
        "graph_context": {
            "n_nodes": N,
            "reachable_diameter": reachable_diameter,
            "unreachable_sentinel_used": unreachable_sentinel,
            "random_same_type_baseline": type_rand,
            "random_same_layer_baseline": layer_rand,
            "random_hop_dist_baseline": hop_rand,
        },
    }


# ---------------------------------------------------------------------------
# qualitative dump — pick 3 seeds from different layers on the median graph
# ---------------------------------------------------------------------------

def _qualitative_dump(
    emb: Tensor,
    edge_index: Tensor,
    node_types: np.ndarray,
    node_layers: np.ndarray,
    c: float | Tensor,
    euclidean: bool,
    graph_idx: int,
    k: int = 5,
) -> dict:
    N = emb.size(0)
    D = _pairwise_distance_matrix(emb, c, euclidean).detach()
    D.fill_diagonal_(float("inf"))
    hop_matrix = _bfs_hop_matrix(edge_index.detach().cpu().numpy(), N)
    topk_idx = torch.topk(D, k=k, largest=False).indices.cpu().numpy()
    topk_dist = torch.topk(D, k=k, largest=False).values.cpu().numpy()

    # Pick one seed per layer if available.
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
    # Fallback: fill with lowest-indexed nodes.
    if len(seed_indices) < 3:
        for i in range(N):
            if i not in seed_indices:
                seed_indices.append(i)
                if len(seed_indices) >= 3:
                    break

    seeds_dump: list[dict] = []
    for seed_i in seed_indices:
        nbrs = topk_idx[seed_i]
        nbr_dists = topk_dist[seed_i]
        nn_entries = []
        for rank, (j, d) in enumerate(zip(nbrs, nbr_dists)):
            hop = int(hop_matrix[seed_i, int(j)])
            nn_entries.append({
                "rank": rank + 1,
                "node_idx": int(j),
                "node_type": int(node_types[int(j)]),
                "layer": int(node_layers[int(j)]),
                "distance": float(d),
                "hop_from_seed": hop if hop >= 0 else None,
            })
        seeds_dump.append({
            "seed_idx": int(seed_i),
            "seed_type": int(node_types[seed_i]),
            "seed_layer": int(node_layers[seed_i]),
            "top_k_nn": nn_entries,
        })

    return {
        "graph_idx": graph_idx,
        "n_nodes": N,
        "k": k,
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


def _median_size_graph(
    dataset: CorpusDataset, graph_ids: list[int]
) -> int:
    """Return the graph id whose node count is closest to the median."""
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
    out_path: Path,
) -> dict:
    include_tasks = {task} if task is not None else None
    dataset = CorpusDataset(
        corpus_dir=corpus_dir,
        split=split,
        split_seed=split_seed,
        include_tasks=include_tasks,
    )
    graph_ids = _unique_graph_indices(dataset)
    print(f"[2.3a] dataset: {len(dataset)} samples, {len(graph_ids)} unique graphs")

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
            # recover per-node type and layer from x
            x_np = x.detach().cpu().numpy()
            type_block = x_np[:, NODE_TYPE_SLICE]
            layer_block = x_np[:, LAYER_SLICE]
            node_types = np.where(
                type_block.sum(axis=1) > 0,
                type_block.argmax(axis=1),
                -1,
            )
            node_layers = np.where(
                layer_block.sum(axis=1) > 0,
                layer_block.argmax(axis=1),
                -1,
            )

            metrics = _compute_graph_metrics(
                emb=emb, edge_index=edge_index,
                node_types=node_types, node_layers=node_layers,
                c=c_val, euclidean=euclidean, k_values=K_VALUES,
            )
            per_graph.append({"graph_idx": gi, **metrics})

    # Aggregate per-k metrics across graphs.
    summary_out: dict[str, dict[str, dict[str, float]]] = {}
    metric_names = (
        "same_type_frac_mean",
        "same_layer_frac_mean",
        "hop_dist_mean",
        "unreachable_pair_frac",
    )
    for k in K_VALUES:
        k_key = f"k={k}"
        block: dict[str, dict[str, float]] = {}
        for mname in metric_names:
            vals = []
            for g in per_graph:
                pk = g["per_k"].get(k_key, {})
                if mname in pk:
                    vals.append(pk[mname])
            if not vals:
                continue
            block[mname] = {
                "mean": float(statistics.mean(vals)),
                "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
                "median": float(statistics.median(vals)),
                "min": float(min(vals)),
                "max": float(max(vals)),
                "n": len(vals),
            }
        summary_out[k_key] = block

    # Random baselines averaged across graphs (for context).
    rand = {
        "same_type": statistics.mean(
            g["graph_context"]["random_same_type_baseline"] for g in per_graph
        ),
        "same_layer": statistics.mean(
            g["graph_context"]["random_same_layer_baseline"] for g in per_graph
        ),
        "hop_dist": statistics.mean(
            g["graph_context"]["random_hop_dist_baseline"] for g in per_graph
            if g["graph_context"]["random_hop_dist_baseline"] == g["graph_context"]["random_hop_dist_baseline"]
        ),
    }

    # Qualitative dump on the median-size val graph.
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
        qual = _qualitative_dump(
            emb=emb, edge_index=edge_index,
            node_types=node_types, node_layers=node_layers,
            c=c_val, euclidean=euclidean, graph_idx=target_gi, k=5,
        )

    results = {
        "checkpoint": str(checkpoint),
        "model_kind": cfg["model"],
        "split": split,
        "task": task,
        "n_graphs_evaluated": len(per_graph),
        "per_graph": per_graph,
        "summary_across_graphs": summary_out,
        "random_baselines_mean_across_graphs": rand,
        "qualitative": qual,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    _print_summary(results)
    return results


def _print_summary(results: dict) -> None:
    print()
    print("=" * 84)
    print(f"Experiment 2.3a — NN retrieval quality ({results['n_graphs_evaluated']} graphs)")
    print(f"checkpoint: {results['checkpoint']}")
    print(f"model: {results['model_kind']}  task: {results['task']}")
    print("=" * 84)

    for k_key in sorted(results["summary_across_graphs"]):
        block = results["summary_across_graphs"][k_key]
        if not block:
            continue
        print(f"\n{k_key}:")
        for mname in ("same_type_frac_mean", "same_layer_frac_mean", "hop_dist_mean"):
            if mname in block:
                b = block[mname]
                print(
                    f"  {mname:<24} mean={b['mean']:+.4f} std={b['std']:.4f} "
                    f"median={b['median']:+.4f} range=[{b['min']:+.4f}, {b['max']:+.4f}]"
                )
        if "unreachable_pair_frac" in block:
            print(f"  {'unreachable_pair_frac':<24} mean={block['unreachable_pair_frac']['mean']:.4f}")

    r = results["random_baselines_mean_across_graphs"]
    print()
    print("Random baselines (averaged across graphs):")
    print(f"  same_type:  {r['same_type']:.4f}")
    print(f"  same_layer: {r['same_layer']:.4f}")
    print(f"  hop_dist:   {r['hop_dist']:.4f}")

    q = results["qualitative"]
    print()
    print(f"Qualitative dump on graph {q['graph_idx']} (N={q['n_nodes']}, k={q['k']}):")
    for s in q["seeds"]:
        print(
            f"  Seed idx={s['seed_idx']:3d}  type={s['seed_type']}  layer={s['seed_layer']}"
        )
        for nn in s["top_k_nn"]:
            hop_str = f"{nn['hop_from_seed']}" if nn["hop_from_seed"] is not None else "unreach"
            print(
                f"    #{nn['rank']}  idx={nn['node_idx']:3d}  type={nn['node_type']}  "
                f"layer={nn['layer']}  dist={nn['distance']:.3f}  hop={hop_str}"
            )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--summary", type=str, default=None)
    p.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--task", type=int, default=2, help="Use -1 for all tasks.")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    summary = Path(args.summary) if args.summary else checkpoint.parent / "summary.json"
    out = Path(args.out) if args.out else checkpoint.parent / "retrieval_nn.json"
    task = None if args.task < 0 else int(args.task)

    evaluate_checkpoint(
        checkpoint=checkpoint, summary=summary,
        corpus_dir=args.corpus, split=args.split, split_seed=args.split_seed,
        task=task, out_path=out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

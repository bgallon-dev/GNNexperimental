r"""v3.1 Phase 1.2 — Eval D: geometry-vs-graph disagreement.

Where does the learned manifold disagree with graph topology? Two
diagnostic directions, over all reachable node pairs per val graph:

  geom_near_graph_far  — pairs in the closest q% by embedding distance
                         that are also in the farthest q% by BFS hop
                         distance. "Geometry over-connects": the
                         manifold pulls graph-distant nodes together.
                         This is the Phase-4 Stage-C structural
                         guardrail metric (must NOT increase).
  graph_near_geom_far  — pairs adjacent/near in the graph but far in
                         embedding space. "Missed structure": the
                         encoder failed to place true neighbors close.

Plus per-graph Spearman(emb_dist, hop_dist): positive = geometry
agrees with graph topology (closer in embedding ==> fewer hops).

These pairs are the candidates an operational system would surface as
"the geometry and the exact graph disagree here" — missing edges, bad
entity resolution, over-compressed motifs. Feeds the Phase-5
``graph_far_geometry_near`` retrieval op.

Reuses ``_pairwise_distance_matrix`` / ``_bfs_hop_matrix`` /
``_load_encoder`` from ``eval_retrieval_nn`` (identical helpers — import
rather than re-define to avoid drift).

Usage
-----
    py -m src.modelsv3.eval_geom_graph_disagreement \
        --checkpoint runs/v3.1-baseline-hyp-h128-l4-seed1/encoder.pt \
        --task 2 \
        --out runs/v3.1-baseline-hyp-h128-l4-seed1/geom_graph_disagreement_baseline.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv3.eval_retrieval_nn import (  # noqa: E402
    _bfs_hop_matrix,
    _load_encoder,
    _pairwise_distance_matrix,
    _unique_graph_indices,
)

Q_VALUES = (5, 10)  # percentile cut-offs for "near"/"far"


# ---------------------------------------------------------------------------
# scipy-free Spearman
# ---------------------------------------------------------------------------

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of each element (ties share the mean rank). Mirrors
    ``scipy.stats.rankdata`` without the dependency."""
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # average tied ranks
    a_sorted = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


# ---------------------------------------------------------------------------
# per-graph
# ---------------------------------------------------------------------------

def _evaluate_graph(emb: torch.Tensor, edge_index: torch.Tensor,
                    c_val, euclidean: bool) -> dict:
    N = emb.size(0)
    D = _pairwise_distance_matrix(emb, c_val, euclidean).detach().cpu().numpy()
    hop = _bfs_hop_matrix(edge_index.detach().cpu().numpy(), N)

    iu, ju = np.triu_indices(N, k=1)
    h = hop[iu, ju]
    reachable = h >= 0
    g = D[iu, ju][reachable].astype(np.float64)
    h = h[reachable].astype(np.float64)
    n_pairs = int(g.size)
    if n_pairs < 2:
        return {
            "n_nodes": N, "n_reachable_pairs": n_pairs,
            "spearman_emb_hop": float("nan"),
            "disagreement": {f"q{q}": {} for q in Q_VALUES},
        }

    spearman = _spearman(g, h)
    disagreement: dict[str, dict] = {}
    for q in Q_VALUES:
        g_lo = float(np.percentile(g, q))           # geometrically near
        g_hi = float(np.percentile(g, 100 - q))     # geometrically far
        h_lo = float(np.percentile(h, q))           # graph near
        h_hi = float(np.percentile(h, 100 - q))     # graph far
        geom_near = g <= g_lo
        geom_far = g >= g_hi
        graph_near = h <= h_lo
        graph_far = h >= h_hi
        disagreement[f"q{q}"] = {
            "geom_near_graph_far_frac": float(np.mean(geom_near & graph_far)),
            "graph_near_geom_far_frac": float(np.mean(graph_near & geom_far)),
            "expected_if_independent": float((q / 100.0) ** 2),
        }
    return {
        "n_nodes": N,
        "n_reachable_pairs": n_pairs,
        "spearman_emb_hop": spearman,
        "disagreement": disagreement,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def evaluate_checkpoint(checkpoint: Path, summary: Path, corpus_dir: str,
                        split: str, split_seed: int, task: int | None,
                        out_path: Path) -> dict:
    include_tasks = {task} if task is not None else None
    dataset = CorpusDataset(
        corpus_dir=corpus_dir, split=split, split_seed=split_seed,
        include_tasks=include_tasks,
    )
    graph_ids = _unique_graph_indices(dataset)
    model, cfg = _load_encoder(checkpoint, summary, dataset)
    euclidean = cfg["model"] == "euclidean"
    c_val = getattr(model, "c", torch.tensor(float(cfg["curvature"])))
    print(f"[evalD] {len(dataset)} samples, {len(graph_ids)} unique graphs  "
          f"model={cfg['model']}")

    per_graph: list[dict] = []
    with torch.no_grad():
        for gi in graph_ids:
            graph = dataset._get_graph(gi)
            out = model(
                graph["x"], graph["edge_index"], graph["edge_type"],
                graph["edge_descriptor"],
                node_descriptor=graph["node_descriptor"],
            )
            emb = out.node_embeddings.detach().cpu()
            r = _evaluate_graph(emb, graph["edge_index"], c_val, euclidean)
            r["graph_idx"] = gi
            per_graph.append(r)

    def _summ(vals: list[float]) -> dict:
        clean = [v for v in vals if v == v]
        if not clean:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {
            "mean": statistics.mean(clean),
            "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
            "min": min(clean), "max": max(clean), "n": len(clean),
        }

    spear = [g["spearman_emb_hop"] for g in per_graph]
    summary_out: dict = {"spearman_emb_hop": _summ(spear)}
    for q in Q_VALUES:
        qk = f"q{q}"
        summary_out[qk] = {
            "geom_near_graph_far_frac": _summ(
                [g["disagreement"][qk].get("geom_near_graph_far_frac", float("nan"))
                 for g in per_graph]
            ),
            "graph_near_geom_far_frac": _summ(
                [g["disagreement"][qk].get("graph_near_geom_far_frac", float("nan"))
                 for g in per_graph]
            ),
        }

    results = {
        "checkpoint": str(checkpoint),
        "model_kind": cfg["model"],
        "split": split,
        "task": task,
        "q_values": list(Q_VALUES),
        "n_graphs": len(per_graph),
        "guardrail_metric": "summary.q5.geom_near_graph_far_frac "
                            "(Phase-4 Stage-C must not increase this)",
        "per_graph": per_graph,
        "summary": summary_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    _print_summary(results)
    return results


def _print_summary(r: dict) -> None:
    s = r["summary"]
    print()
    print("=" * 80)
    print(f"Eval D - geometry vs graph disagreement ({r['n_graphs']} graphs)")
    print(f"checkpoint: {r['checkpoint']}  model: {r['model_kind']}")
    print("=" * 80)
    sp = s["spearman_emb_hop"]
    sign = "agrees" if sp["mean"] > 0 else "DISAGREES"
    print(f"  spearman(emb_dist, hop)  mean={sp['mean']:+.4f}  std={sp['std']:.4f}"
          f"   (geometry {sign} with graph topology; + is good)")
    for q in r["q_values"]:
        qk = f"q{q}"
        gn = s[qk]["geom_near_graph_far_frac"]
        gg = s[qk]["graph_near_geom_far_frac"]
        print(f"  q={q}%:  geom_near_graph_far={gn['mean']:.4f}  "
              f"graph_near_geom_far={gg['mean']:.4f}  "
              f"(indep ~ {(q/100.0)**2:.4f})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--summary", type=str, default=None)
    p.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    p.add_argument("--split", type=str, default="val",
                   choices=["train", "val", "test", "all"])
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--task", type=int, default=2, help="Use -1 for all tasks.")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    summary = Path(args.summary) if args.summary else checkpoint.parent / "summary.json"
    out = Path(args.out) if args.out else checkpoint.parent / "geom_graph_disagreement.json"
    evaluate_checkpoint(
        checkpoint=checkpoint, summary=summary, corpus_dir=args.corpus,
        split=args.split, split_seed=args.split_seed,
        task=None if args.task < 0 else int(args.task), out_path=out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

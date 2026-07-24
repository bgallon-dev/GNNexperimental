r"""Experiment 2.1 — intrinsic metrics at corpus scale.

Loads a trained v3 encoder and evaluates
``silhouette_score / nn_edge_precision@5 / nn_label_purity@5``
from ``src.modelsv3.intrinsic_eval`` on every unique val graph (not
every val *sample* — samples within a graph share structure and would
double-count the encoder's work).

Output: ``results.json`` with per-graph metrics and a summary block
with mean, std, min/max, and percentiles across graphs. Also prints a
compact table to stdout.

Usage
-----
    py src/modelsv3/eval_intrinsic_corpus.py \\
        --checkpoint runs/v3_hyp_compute_seed0/encoder.pt \\
        --summary    runs/v3_hyp_compute_seed0/summary.json \\
        --out        runs/v3_hyp_compute_seed0/intrinsic_corpus.json

The script uses ``summary.json`` next to the checkpoint to recover
``hidden_dim`` / ``num_layers`` / ``model`` — avoids manual flag
duplication and matches the exact training config.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

# Make `src.` importable when invoked from the project root.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv3.contrastive import NODE_TYPE_SLICE  # noqa: E402
from src.modelsv3.euclidean_v3 import EuclideanReasonerV3  # noqa: E402
from src.modelsv3.hyperbolic_gnnV3 import KettleGraphReasonerV3  # noqa: E402
from src.modelsv3.intrinsic_eval import (  # noqa: E402
    nn_edge_precision_at_k,
    nn_label_purity_at_k,
    silhouette_score,
)


def _load_config(summary_path: Path) -> dict:
    """Pull encoder config from training summary.json."""
    with open(summary_path, "r") as f:
        summary = json.load(f)
    cfg = summary.get("config", {})
    required = ("model", "hidden_dim", "num_layers", "type_dim", "curvature")
    for k in required:
        if k not in cfg:
            raise KeyError(
                f"summary.json missing '{k}' in config block. "
                f"Available: {sorted(cfg)}"
            )
    return cfg


def _build_encoder(cfg: dict, dataset: CorpusDataset) -> torch.nn.Module:
    """Construct an encoder matching the training-time shape."""
    model_kind = cfg["model"]
    # E5: post-2026-07-10 checkpoints carry no attention type_emb table;
    # absent key => legacy True (all earlier checkpoints load strict).
    net = (dataset.num_edge_types_max
           if cfg.get("attn_type_table", True) else None)
    if model_kind == "hyperbolic":
        return KettleGraphReasonerV3(
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
    if model_kind == "euclidean":
        return EuclideanReasonerV3(
            node_feat_dim=dataset.node_feat_dim,
            edge_feat_dim=dataset.edge_feat_dim_schema,
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            type_dim=int(cfg["type_dim"]),
            num_edge_types_max=net,
            node_feat_dim_schema=dataset.node_feat_dim_schema,
        )
    raise ValueError(f"unknown model kind {model_kind!r}")


def _unique_graph_indices(dataset: CorpusDataset) -> list[int]:
    """Deduplicate by graph_idx. ``dataset.index`` is a list of
    ``(graph_idx, task_idx)``; many task samples share a graph, so we
    pick one representative per graph."""
    seen: list[int] = []
    seen_set: set[int] = set()
    for graph_idx, _task_idx in dataset.index:
        gi = int(graph_idx)
        if gi in seen_set:
            continue
        seen_set.add(gi)
        seen.append(gi)
    return seen


def _node_type_labels(x: torch.Tensor) -> torch.Tensor:
    """Recover node-type labels from the one-hot block at x[:, 0:12].
    Returns (N,) long; -1 where the block is all-zero."""
    type_block = x[:, NODE_TYPE_SLICE]
    sums = type_block.sum(dim=1)
    labels = torch.where(
        sums > 0,
        type_block.argmax(dim=1),
        torch.full_like(sums, -1, dtype=torch.long),
    )
    return labels


def _summarize(vals: list[float]) -> dict:
    """Mean/std/min/max/percentiles, ignoring NaNs."""
    clean = [v for v in vals if v == v]  # drop NaN
    if not clean:
        return {
            "mean": float("nan"), "std": float("nan"),
            "median": float("nan"),
            "p10": float("nan"), "p90": float("nan"),
            "min": float("nan"), "max": float("nan"),
            "n": 0, "n_nan": len(vals),
        }
    sorted_vals = sorted(clean)
    n = len(clean)
    def pct(q: float) -> float:
        if n == 1:
            return sorted_vals[0]
        idx = q * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac
    std = statistics.stdev(clean) if n > 1 else 0.0
    return {
        "mean": statistics.mean(clean),
        "std": std,
        "median": pct(0.50),
        "p10": pct(0.10),
        "p90": pct(0.90),
        "min": min(clean),
        "max": max(clean),
        "n": n,
        "n_nan": len(vals) - n,
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    summary_path: Path,
    corpus_dir: str,
    split: str,
    split_seed: int,
    include_task: int | None,
    out_path: Path,
    k: int = 5,
    device_str: str = "cpu",
) -> dict:
    device = torch.device(device_str)
    cfg = _load_config(summary_path)

    # Build a dataset that mirrors the training-time configuration as
    # closely as possible. Task filtering must match training so the
    # val set is the same graphs the training pipeline evaluates on.
    include_tasks = {include_task} if include_task is not None else None
    dataset = CorpusDataset(
        corpus_dir=corpus_dir,
        split=split,
        split_seed=split_seed,
        include_tasks=include_tasks,
    )
    print(
        f"[2.1] dataset split={split} len_samples={len(dataset)} "
        f"(filtered to task={include_task})"
    )

    graph_ids = _unique_graph_indices(dataset)
    print(f"[2.1] unique graphs to evaluate: {len(graph_ids)}")

    encoder = _build_encoder(cfg, dataset).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(state)
    encoder.eval()
    euclidean = cfg["model"] == "euclidean"

    # Retrieve curvature tensor once; for hyperbolic the property returns
    # a clamped tensor — use that. For Euclidean it's unused.
    c_val = getattr(encoder, "c", torch.tensor(float(cfg["curvature"])))

    per_graph: list[dict] = []
    with torch.no_grad():
        for gi in graph_ids:
            graph = dataset._get_graph(gi)
            x = graph["x"].to(device)
            edge_index = graph["edge_index"].to(device)
            edge_type = graph["edge_type"].to(device)
            edge_descriptor = graph["edge_descriptor"].to(device)
            node_descriptor = graph["node_descriptor"].to(device)

            out = encoder(
                x, edge_index, edge_type, edge_descriptor,
                node_descriptor=node_descriptor,
            )
            emb = out.node_embeddings.detach().cpu()
            edge_index_cpu = edge_index.detach().cpu()
            labels = _node_type_labels(x.detach().cpu())

            sil = silhouette_score(
                emb, labels, c=c_val, euclidean=euclidean,
            )
            ep = nn_edge_precision_at_k(
                emb, edge_index_cpu, k=k, c=c_val, euclidean=euclidean,
            )
            lp = nn_label_purity_at_k(
                emb, labels, k=k, c=c_val, euclidean=euclidean,
            )
            per_graph.append({
                "graph_idx": int(gi),
                "n_nodes": int(x.size(0)),
                "n_edges": int(edge_index.size(1)),
                "silhouette_mean": sil["mean"],
                "silhouette_n_evaluated": sil["n_evaluated"],
                "silhouette_n_clusters": sil["n_clusters"],
                "edge_prec_mean": ep["mean_precision"],
                "edge_prec_random_baseline": ep["random_baseline"],
                "edge_prec_mean_out_degree": ep["mean_out_degree"],
                "label_purity_mean": lp["mean_purity"],
                "label_purity_random_baseline": lp["random_baseline"],
                "label_purity_n_evaluated": lp["n_evaluated"],
            })

    silhouettes = [r["silhouette_mean"] for r in per_graph]
    edge_precs = [r["edge_prec_mean"] for r in per_graph]
    label_purities = [r["label_purity_mean"] for r in per_graph]
    baselines_edge = [r["edge_prec_random_baseline"] for r in per_graph]
    baselines_label = [r["label_purity_random_baseline"] for r in per_graph]

    results = {
        "checkpoint": str(checkpoint_path),
        "model_kind": cfg["model"],
        "split": split,
        "include_task": include_task,
        "n_graphs_evaluated": len(per_graph),
        "per_graph": per_graph,
        "summary": {
            "silhouette": _summarize(silhouettes),
            "edge_precision_at_k": _summarize(edge_precs),
            "label_purity_at_k": _summarize(label_purities),
            "random_baseline_edge_prec_mean": (
                statistics.mean(baselines_edge) if baselines_edge else float("nan")
            ),
            "random_baseline_label_purity_mean": (
                statistics.mean(baselines_label) if baselines_label else float("nan")
            ),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    _print_summary(results)
    return results


def _print_summary(results: dict) -> None:
    s = results["summary"]
    print()
    print("=" * 72)
    print(f"Intrinsic metrics across {results['n_graphs_evaluated']} val graphs")
    print(f"checkpoint: {results['checkpoint']}")
    print(f"model: {results['model_kind']}   task: {results['include_task']}")
    print("=" * 72)
    for name, key in (
        ("silhouette", "silhouette"),
        ("edge_prec@5", "edge_precision_at_k"),
        ("label_purity@5", "label_purity_at_k"),
    ):
        block = s[key]
        print(
            f"{name:<18} mean={block['mean']:+.4f} std={block['std']:.4f} "
            f"median={block['median']:+.4f} p10={block['p10']:+.4f} "
            f"p90={block['p90']:+.4f}  (n={block['n']}, nan={block['n_nan']})"
        )
    print(
        f"random baseline   edge_prec={s['random_baseline_edge_prec_mean']:.4f}  "
        f"label_purity={s['random_baseline_label_purity_mean']:.4f}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to encoder.pt saved by train_v3.")
    p.add_argument("--summary", type=str, default=None,
                   help="Path to summary.json next to the checkpoint. "
                        "Defaults to sibling file.")
    p.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    p.add_argument("--split", type=str, default="val",
                   choices=["train", "val", "test", "all"])
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--task", type=int, default=2,
                   help="Task filter to match training. Use -1 for all tasks.")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", type=str, default=None,
                   help="Path for output JSON. Defaults to "
                        "<checkpoint_dir>/intrinsic_corpus.json.")
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    summary = Path(args.summary) if args.summary else checkpoint.parent / "summary.json"
    if not summary.exists():
        raise FileNotFoundError(
            f"summary.json not found at {summary}. "
            f"Pass --summary explicitly."
        )
    out = Path(args.out) if args.out else checkpoint.parent / "intrinsic_corpus.json"
    include_task = None if args.task < 0 else int(args.task)

    evaluate_checkpoint(
        checkpoint_path=checkpoint,
        summary_path=summary,
        corpus_dir=args.corpus,
        split=args.split,
        split_seed=args.split_seed,
        include_task=include_task,
        out_path=out,
        k=args.k,
        device_str="cuda" if args.cuda and torch.cuda.is_available() else "cpu",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

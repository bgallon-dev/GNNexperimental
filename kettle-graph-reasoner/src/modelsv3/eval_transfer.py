r"""Experiment 2.2 — Cross-graph transfer evaluation.

For each checkpoint trained with ``--train-graphs-frac < 1.0``,
evaluates intrinsic metrics across three conditions:
    seen_train    — graphs the model was trained on
    unseen_train  — graphs in the train split that were withheld
    val           — graphs in the val split

Uses the same intrinsic metric functions as Experiment 2.1 for
direct comparability.

Usage
-----
    py src/modelsv3/eval_transfer.py \\
        --checkpoint runs/v3_transfer_hyp_seed0/encoder.pt \\
        --out        runs/v3_transfer_hyp_seed0/transfer_eval.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

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


CONDITIONS = ("seen_train", "unseen_train", "val")


def _build_encoder(cfg: dict, dataset: CorpusDataset) -> torch.nn.Module:
    model_kind = cfg["model"]
    if model_kind == "hyperbolic":
        return KettleGraphReasonerV3(
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
    if model_kind == "euclidean":
        return EuclideanReasonerV3(
            node_feat_dim=dataset.node_feat_dim,
            edge_feat_dim=dataset.edge_feat_dim_schema,
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            type_dim=int(cfg["type_dim"]),
            num_edge_types_max=dataset.num_edge_types_max,
            node_feat_dim_schema=dataset.node_feat_dim_schema,
        )
    raise ValueError(f"unknown model kind {model_kind!r}")


def _node_type_labels(x: torch.Tensor) -> torch.Tensor:
    type_block = x[:, NODE_TYPE_SLICE]
    sums = type_block.sum(dim=1)
    return torch.where(
        sums > 0, type_block.argmax(dim=1),
        torch.full_like(sums, -1, dtype=torch.long),
    )


def _eval_graph(
    model: torch.nn.Module, dataset: CorpusDataset, gi: int,
    c_val, euclidean: bool, k: int,
) -> dict:
    graph = dataset._get_graph(gi)
    with torch.no_grad():
        out = model(
            graph["x"], graph["edge_index"], graph["edge_type"],
            graph["edge_descriptor"], node_descriptor=graph["node_descriptor"],
        )
    emb = out.node_embeddings.detach().cpu()
    edge_index_cpu = graph["edge_index"].detach().cpu()
    labels = _node_type_labels(graph["x"].detach().cpu())
    sil = silhouette_score(emb, labels, c=c_val, euclidean=euclidean)
    ep = nn_edge_precision_at_k(emb, edge_index_cpu, k=k, c=c_val, euclidean=euclidean)
    lp = nn_label_purity_at_k(emb, labels, k=k, c=c_val, euclidean=euclidean)
    return {
        "graph_idx": int(gi),
        "n_nodes": int(graph["x"].size(0)),
        "n_edges": int(graph["edge_index"].size(1)),
        "silhouette_mean": sil["mean"],
        "edge_prec_mean": ep["mean_precision"],
        "edge_prec_random_baseline": ep["random_baseline"],
        "label_purity_mean": lp["mean_purity"],
        "label_purity_random_baseline": lp["random_baseline"],
    }


def _unique_graph_indices(dataset: CorpusDataset) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for gi, _ in dataset.index:
        gi_i = int(gi)
        if gi_i not in seen:
            seen.add(gi_i)
            out.append(gi_i)
    return out


def _summarize(vals: list[float]) -> dict:
    clean = [v for v in vals if v == v]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(clean) == 1:
        return {"mean": clean[0], "std": 0.0, "n": 1,
                "median": clean[0], "min": clean[0], "max": clean[0]}
    return {
        "mean": statistics.mean(clean), "std": statistics.stdev(clean),
        "median": statistics.median(clean),
        "min": min(clean), "max": max(clean), "n": len(clean),
    }


def evaluate_checkpoint(
    checkpoint: Path, summary_path: Path, corpus_dir: str,
    split_seed: int, task: int, out_path: Path, k: int = 5,
) -> dict:
    with open(summary_path, "r") as f:
        summary = json.load(f)
    cfg = summary["config"]
    # The patched train_v3.py nests the partition under
    # ``training_graph_partition.{used,held_out}_graph_ids``. Accept a
    # top-level form too for older summaries that used flat keys.
    partition = summary.get("training_graph_partition")
    if partition is not None:
        training_ids = set(int(g) for g in partition["used_graph_ids"])
        held_ids = set(int(g) for g in partition["held_out_graph_ids"])
    elif "training_graph_ids" in summary:
        training_ids = set(int(g) for g in summary["training_graph_ids"])
        held_ids = set(int(g) for g in summary.get("held_out_graph_ids", []))
    else:
        raise KeyError(
            "summary.json is missing the training-graph partition "
            "('training_graph_partition' or 'training_graph_ids'). "
            "Was this checkpoint produced by the patched train_v3.py?"
        )

    train_ds = CorpusDataset(
        corpus_dir=corpus_dir, split="train", split_seed=split_seed,
        include_tasks={task},
    )
    val_ds = CorpusDataset(
        corpus_dir=corpus_dir, split="val", split_seed=split_seed,
        include_tasks={task},
    )
    train_ids_all = set(_unique_graph_indices(train_ds))
    val_ids_all = set(_unique_graph_indices(val_ds))

    seen_ids = sorted(training_ids & train_ids_all)
    unseen_ids = sorted(held_ids & train_ids_all)
    val_ids = sorted(val_ids_all)

    overlap = training_ids & val_ids_all
    if overlap:
        print(f"WARNING: training_graph_ids overlaps val by {len(overlap)} graphs. "
              "Unexpected — check split_seed consistency.")

    print(f"[2.2] conditions: seen_train={len(seen_ids)}  "
          f"unseen_train={len(unseen_ids)}  val={len(val_ids)}")

    model = _build_encoder(cfg, train_ds)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    euclidean = cfg["model"] == "euclidean"
    c_val = getattr(model, "c", torch.tensor(float(cfg["curvature"])))

    def eval_many(gids: list[int], dataset: CorpusDataset) -> list[dict]:
        in_ds = {int(x) for x, _ in dataset.index}
        results = []
        for gi in gids:
            if gi not in in_ds:
                continue
            results.append(_eval_graph(model, dataset, gi, c_val, euclidean, k))
        return results

    per_condition = {
        "seen_train": eval_many(seen_ids, train_ds),
        "unseen_train": eval_many(unseen_ids, train_ds),
        "val": eval_many(val_ids, val_ds),
    }

    agg: dict[str, dict] = {}
    for cond in CONDITIONS:
        graphs = per_condition[cond]
        if not graphs:
            agg[cond] = {
                "n_graphs": 0,
                "silhouette": _summarize([]),
                "edge_prec@5": _summarize([]),
                "label_purity@5": _summarize([]),
            }
            continue
        agg[cond] = {
            "n_graphs": len(graphs),
            "silhouette": _summarize([g["silhouette_mean"] for g in graphs]),
            "edge_prec@5": _summarize([g["edge_prec_mean"] for g in graphs]),
            "label_purity@5": _summarize([g["label_purity_mean"] for g in graphs]),
            "edge_prec_random_baseline": statistics.mean(
                [g["edge_prec_random_baseline"] for g in graphs]
            ),
            "label_purity_random_baseline": statistics.mean(
                [g["label_purity_random_baseline"] for g in graphs]
            ),
        }

    results = {
        "checkpoint": str(checkpoint),
        "model_kind": cfg["model"],
        "train_graphs_frac": float(cfg.get("train_graphs_frac", 1.0)),
        "per_graph": per_condition,
        "summary_per_condition": agg,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    _print_summary(results)
    return results


def _print_summary(r: dict) -> None:
    print()
    print("=" * 92)
    print(f"Experiment 2.2 transfer eval — model={r['model_kind']}  "
          f"train_graphs_frac={r['train_graphs_frac']}")
    print(f"checkpoint: {r['checkpoint']}")
    print("=" * 92)
    print(f"\n{'condition':<14} {'n_graphs':>10} {'edge_prec@5':>18} "
          f"{'label_purity@5':>20} {'silhouette':>18}")
    print("-" * 92)
    for cond in CONDITIONS:
        a = r["summary_per_condition"][cond]
        if a["n_graphs"] == 0:
            print(f"{cond:<14} {0:>10}   (no graphs matched)")
            continue
        ep = a["edge_prec@5"]; lp = a["label_purity@5"]; sl = a["silhouette"]
        print(f"{cond:<14} {a['n_graphs']:>10d} "
              f"{ep['mean']:+.4f} ± {ep['std']:.4f}   "
              f"{lp['mean']:+.4f} ± {lp['std']:.4f}   "
              f"{sl['mean']:+.4f} ± {sl['std']:.4f}")
    print(f"\nRandom baselines:")
    for cond in CONDITIONS:
        a = r["summary_per_condition"][cond]
        if a["n_graphs"] == 0:
            continue
        print(f"  {cond:<14} edge_prec={a['edge_prec_random_baseline']:.4f}  "
              f"label_purity={a['label_purity_random_baseline']:.4f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--summary", type=str, default=None)
    p.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--task", type=int, default=2)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    summary = Path(args.summary) if args.summary else checkpoint.parent / "summary.json"
    out = Path(args.out) if args.out else checkpoint.parent / "transfer_eval.json"
    evaluate_checkpoint(
        checkpoint=checkpoint, summary_path=summary, corpus_dir=args.corpus,
        split_seed=args.split_seed, task=args.task, out_path=out, k=args.k,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

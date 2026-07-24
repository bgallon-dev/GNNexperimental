r"""v3.1 Phase 1.2 — Eval A: candidate recall + oracle rerank.

v3.1's first job is *first-stage retrieval*: pull a candidate set the
v2 reranker can finish. Raw nDCG@10 under-measures that. This script
asks two separate questions per (graph, task) val sample:

  1. recall@C for C in {20, 50, 100} — does the model's top-C contain
     the relevant nodes at all? (binary relevance, label >= 0.5, same
     convention as ``src.training.metrics.recall_at_k``).
  2. oracle-rerank nDCG@K for K in {5, 10, 20} — take the model's
     top-C, re-sort it *perfectly by the true labels*, and score nDCG@K.
     ``oracle_gap@K = oracle_ndcg@K - model_ndcg@K`` is the ceiling a
     downstream reranker could reach on this candidate set. Large gap +
     high recall = "candidate set is good, ranking is the bottleneck"
     (exactly the v3.1 thesis).

Reuses ``score_from_embeddings`` and ``recall_at_k`` / ``ndcg_at_k``
directly — it deliberately does NOT widen ``MetricAccumulator.ks`` (that
would change the schema the sweep quick-check validator reads). C=100
recall lives only in this artifact.

Usage
-----
    py -m src.modelsv3.eval_candidate_recall \
        --checkpoint runs/v3.1-baseline-hyp-h128-l4-seed1/encoder.pt \
        --task 2 \
        --out runs/v3.1-baseline-hyp-h128-l4-seed1/candidate_recall_baseline.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv3.distance_scoring import score_from_embeddings  # noqa: E402
from src.modelsv3.euclidean_v3 import EuclideanReasonerV3  # noqa: E402
from src.modelsv3.hyperbolic_gnnV3 import KettleGraphReasonerV3  # noqa: E402
from src.modelsv3.query_encoder import QueryToBall  # noqa: E402
from src.training.metrics import ndcg_at_k, recall_at_k  # noqa: E402

C_VALUES = (20, 50, 100)
K_VALUES = (5, 10, 20)


# ---------------------------------------------------------------------------
# loaders (encoder mirrors eval_retrieval_nn._load_encoder; query head is
# arch-aware and forward-compatible with the Phase-2 QueryToBall variants)
# ---------------------------------------------------------------------------

def _build_encoder(cfg: dict, dataset: CorpusDataset) -> torch.nn.Module:
    # E5 (Docs/ARCH_EFFICIENCY_PLAN.md): post-2026-07-10 checkpoints carry
    # no per-layer attention type_emb table (dead weight; the schema
    # override always supplies type embeddings). Absent key => legacy True
    # so every earlier checkpoint (incl. frozen v1.0) still loads strict.
    num_edge_types = (dataset.num_edge_types_max
                      if cfg.get("attn_type_table", True) else None)
    if cfg["model"] == "hyperbolic":
        return KettleGraphReasonerV3(
            node_feat_dim=dataset.node_feat_dim,
            edge_feat_dim=dataset.edge_feat_dim_schema,
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            type_dim=int(cfg["type_dim"]),
            c=float(cfg["curvature"]),
            num_edge_types_max=num_edge_types,
            node_feat_dim_schema=dataset.node_feat_dim_schema,
            tangent_scale_init=float(cfg.get("tangent_scale", 0.1)),
        )
    if cfg["model"] == "euclidean":
        return EuclideanReasonerV3(
            node_feat_dim=dataset.node_feat_dim,
            edge_feat_dim=dataset.edge_feat_dim_schema,
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            type_dim=int(cfg["type_dim"]),
            num_edge_types_max=num_edge_types,
            node_feat_dim_schema=dataset.node_feat_dim_schema,
        )
    raise ValueError(f"unknown model kind {cfg['model']!r}")


def build_query_encoder(cfg: dict, dataset: CorpusDataset) -> QueryToBall:
    """Construct ``QueryToBall`` matching the training config.

    Pre-v3.1 checkpoints have no ``query_head_arch`` key; default "qh0".
    ``arch`` / ``norm`` are passed only if the installed ``QueryToBall``
    accepts them, so this loader works both before and after the
    Phase-2 arch-selectable refactor.
    """
    kwargs = dict(
        query_dim=dataset.query_dim,
        hidden_dim=int(cfg["hidden_dim"]),
        c=float(cfg.get("curvature", 1.0)),
        tangent_scale_init=float(cfg.get("tangent_scale", 0.1)),
        euclidean=(cfg["model"] == "euclidean"),
    )
    params = inspect.signature(QueryToBall.__init__).parameters
    if "arch" in params:
        kwargs["arch"] = cfg.get("query_head_arch", "qh0")
    if "norm" in params:
        kwargs["norm"] = cfg.get("query_head_norm", "layernorm")
    return QueryToBall(**kwargs)


def _load(checkpoint: Path, summary: Path, dataset: CorpusDataset):
    with open(summary, "r") as f:
        cfg = json.load(f)["config"]
    encoder = _build_encoder(cfg, dataset)
    encoder.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    encoder.eval()
    qpath = checkpoint.parent / "query_encoder.pt"
    if not qpath.exists():
        raise FileNotFoundError(f"query_encoder.pt not found next to {checkpoint}")
    qenc = build_query_encoder(cfg, dataset)
    qenc.load_state_dict(torch.load(qpath, map_location="cpu"))
    qenc.eval()
    return encoder, qenc, cfg


# ---------------------------------------------------------------------------
# oracle rerank
# ---------------------------------------------------------------------------

def _oracle_ndcg(scores: torch.Tensor, labels: torch.Tensor, c: int, k: int) -> float:
    """nDCG@k of the top-c-by-model candidate set, re-sorted perfectly by
    the true labels. IDCG is over the full graph (same as ndcg_at_k), so
    this is directly comparable to the model's nDCG@k."""
    c = min(c, scores.numel())
    cand = torch.topk(scores, k=c, largest=True).indices
    # Build a synthetic score vector: candidates ranked by true label,
    # everything else pushed below them.
    oracle_scores = torch.full_like(labels, float("-inf"))
    oracle_scores[cand] = labels[cand]
    return ndcg_at_k(oracle_scores, labels, k)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint: Path, summary: Path, corpus_dir: str, split: str,
    split_seed: int, task: int | None, out_path: Path,
) -> dict:
    include_tasks = {task} if task is not None else None
    dataset = CorpusDataset(
        corpus_dir=corpus_dir, split=split, split_seed=split_seed,
        include_tasks=include_tasks,
    )
    encoder, qenc, cfg = _load(checkpoint, summary, dataset)
    euclidean = cfg["model"] == "euclidean"
    c_val = getattr(encoder, "c", torch.tensor(float(cfg.get("curvature", 1.0))))
    print(f"[evalA] {len(dataset)} samples  model={cfg['model']}  "
          f"query_head={cfg.get('query_head_arch', 'qh0')}")

    emb_cache: dict[int, torch.Tensor] = {}
    rows: list[tuple[int, dict[str, float]]] = []
    with torch.no_grad():
        for i in range(len(dataset)):
            gi, _ = dataset.index[i]
            s = dataset[i]
            if gi not in emb_cache:
                out = encoder(
                    s.x, s.edge_index, s.edge_type, s.edge_descriptor,
                    node_descriptor=s.node_descriptor,
                )
                emb_cache[gi] = out.node_embeddings.detach()
            emb = emb_cache[gi]
            q_point = qenc(s.query)
            scores = score_from_embeddings(emb, q_point, c=c_val, euclidean=euclidean)

            row: dict[str, float] = {}
            for c in C_VALUES:
                row[f"recall@{c}"] = recall_at_k(scores, s.labels, c)
            for k in K_VALUES:
                m = ndcg_at_k(scores, s.labels, k)
                row[f"ndcg@{k}"] = m
            for c in C_VALUES:
                for k in K_VALUES:
                    if k <= c:
                        o = _oracle_ndcg(scores, s.labels, c, k)
                        row[f"oracle_ndcg@{k}|C{c}"] = o
                        row[f"oracle_gap@{k}|C{c}"] = o - row[f"ndcg@{k}"]
            rows.append((int(s.task_type), row))

    summary_out = _aggregate(rows)
    results = {
        "checkpoint": str(checkpoint),
        "model_kind": cfg["model"],
        "query_head_arch": cfg.get("query_head_arch", "qh0"),
        "split": split,
        "task": task,
        "n_samples": len(rows),
        "C_values": list(C_VALUES),
        "K_values": list(K_VALUES),
        "summary": summary_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    _print_summary(results)
    return results


def _aggregate(rows: list[tuple[int, dict[str, float]]]) -> dict:
    if not rows:
        return {}
    keys = list(rows[0][1].keys())
    n = len(rows)
    overall = {k: sum(r[k] for _, r in rows) / n for k in keys}
    by_type: dict[int, list[dict[str, float]]] = {}
    for t, r in rows:
        by_type.setdefault(t, []).append(r)
    return {
        "overall": overall,
        "by_task_type": {
            str(t): {k: sum(r[k] for r in rs) / len(rs) for k in keys}
            for t, rs in sorted(by_type.items())
        },
    }


def _print_summary(r: dict) -> None:
    o = r["summary"].get("overall", {})
    print()
    print("=" * 80)
    print(f"Eval A - candidate recall + oracle rerank ({r['n_samples']} samples)")
    print(f"checkpoint: {r['checkpoint']}  model: {r['model_kind']}  "
          f"query_head: {r['query_head_arch']}")
    print("=" * 80)
    print("recall:    " + "  ".join(
        f"@{c}={o.get(f'recall@{c}', float('nan')):.4f}" for c in C_VALUES))
    print("ndcg:      " + "  ".join(
        f"@{k}={o.get(f'ndcg@{k}', float('nan')):.4f}" for k in K_VALUES))
    print("oracle nDCG@10 / gap@10 by candidate-set size C:")
    for c in C_VALUES:
        on = o.get(f"oracle_ndcg@10|C{c}")
        og = o.get(f"oracle_gap@10|C{c}")
        if on is not None:
            print(f"  C={c:<3} oracle_ndcg@10={on:.4f}  gap@10={og:+.4f}")


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
    out = Path(args.out) if args.out else checkpoint.parent / "candidate_recall.json"
    evaluate_checkpoint(
        checkpoint=checkpoint, summary=summary, corpus_dir=args.corpus,
        split=args.split, split_seed=args.split_seed,
        task=None if args.task < 0 else int(args.task), out_path=out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

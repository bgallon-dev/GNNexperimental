r"""v3.1 Phase 5 — v2-scorer hybrid reranker (user-chosen).

Wires the trained v2 query-conditioned scorer
(``src/modelsv2/hyperbolic_gnnV2.KettleGraphReasoner``) in as a
downstream reranker over v3.1's candidate set:

    v3.1 encoder+query head -> top-C nodes by hyperbolic distance
    -> v2.forward(full graph, query) -> reorder those C by node_scores

The query enters *v2 only*; the v3 encoder is never query-conditioned,
so the v3 query-agnostic commitment holds (v2 is a downstream consumer,
same slot as the LLM). v2 runs on the FULL graph (structure-faithful);
only the candidate indices' scores are read.

v2 runs (e.g. runs/compare_task2/hyp_seed_1) store no config block, so
the v2 architecture is inferred from ``best.pt`` state-dict shapes
(hidden_dim, num_layers) with a strict-load fallback over the small
{type_dim, depth_attn} space.

Reports nDCG@10 / MRR@10 for v3.1-alone vs v3.1+v2 vs oracle. If the v2
run/checkpoint is absent this exits non-zero with a clear message —
per the decision tree the hybrid is opt-in and NOT a blocker for P1-P3.

Usage
-----
    py -m src.modelsv3.v2_reranker \
        --v3-run runs/v3.1_qh2_infonce_seed1 \
        --v2-run runs/compare_task2/hyp_seed_1 \
        --task 2 --topc 50 \
        --out runs/v3.1_qh2_infonce_seed1/hybrid.json
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
from src.modelsv2.hyperbolic_gnnV2 import KettleGraphReasoner as V2Model  # noqa: E402
from src.modelsv3.distance_scoring import score_from_embeddings  # noqa: E402
from src.modelsv3.eval_candidate_recall import (  # noqa: E402
    _build_encoder,
    build_query_encoder,
)
from src.training.metrics import ndcg_at_k  # noqa: E402

K = 10
RELEVANCE = 0.5


# ---------------------------------------------------------------------------
# v2 reconstruction (infer arch from best.pt shapes)
# ---------------------------------------------------------------------------

def _load_v2(v2_run: Path, dataset: CorpusDataset) -> V2Model:
    ckpt = v2_run / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"v2 checkpoint not found: {ckpt}. The hybrid is opt-in and "
            f"not a P1-P3 blocker (decision tree)."
        )
    blob = torch.load(ckpt, map_location="cpu")
    # v2 best.pt is {epoch, model_state, cfg, val}; older variants may be
    # a bare state_dict or wrap it under 'state_dict'.
    if isinstance(blob, dict) and "model_state" in blob:
        state = blob["model_state"]
        v2cfg = blob.get("cfg", {})
    elif isinstance(blob, dict) and "state_dict" in blob:
        state = blob["state_dict"]
        v2cfg = blob.get("cfg", {})
    else:
        state, v2cfg = blob, {}

    # A usable v2 *reranker* needs the query-conditioned scoring head.
    # Some compare-harness checkpoints saved only the encoder backbone
    # (no node_score / depth_attention) — reranking with a random head
    # would be meaningless, so decline cleanly. Per the decision tree
    # the hybrid is opt-in and NOT a P1-P3 blocker.
    has_head = any(k.startswith(("node_score", "node_score_hier"))
                   for k in state)
    if not has_head:
        raise RuntimeError(
            f"{ckpt} has no scoring head (keys: "
            f"{sorted(set(k.split('.')[0] for k in state))}). This is an "
            f"encoder-only checkpoint, not a trained v2 query-conditioned "
            f"scorer. Provide a v2 run whose best.pt includes 'node_score.*' "
            f"to run the hybrid. v3.1 P1-P3 do not depend on this."
        )

    hidden_dim = int(v2cfg.get("hidden_dim",
                               state["node_in.weight"].shape[0]))
    num_layers = int(v2cfg.get(
        "num_layers",
        1 + max(int(k.split(".")[1]) for k in state
                if k.startswith("mp_layers.")),
    ))
    # Build the query projection at the CHECKPOINT's query_dim so strict
    # load succeeds. The current corpus emits 18-dim queries; older v2
    # scorers were trained on a 9-dim query schema. If they differ the
    # v2 model cannot consume v3.1's query — caller declines (opt-in).
    ckpt_query_dim = int(state["query_in.weight"].shape[1])
    depth_attn = any(k.startswith("depth_attention") for k in state)
    hier = any(k.startswith("node_score_hier") for k in state)
    base = dict(
        node_feat_dim=dataset.node_feat_dim,
        edge_feat_dim=dataset.edge_feat_dim_schema,
        query_dim=ckpt_query_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_edge_types_max=dataset.num_edge_types_max,
        node_feat_dim_schema=dataset.node_feat_dim_schema,
        depth_attn=depth_attn,
    )
    last_err: Exception | None = None
    for type_dim in (8, 16, 4):
        for hsd in ((dataset.node_feat_dim_schema,) if hier else (0,)):
            try:
                m = V2Model(**base, type_dim=type_dim,
                            hierarchy_subspace_dim=hsd)
                m.load_state_dict(state, strict=True)
                m.eval()
                m._v2_query_dim = ckpt_query_dim  # type: ignore[attr-defined]
                print(f"[v2] rebuilt from cfg: hidden={hidden_dim} "
                      f"layers={num_layers} type_dim={type_dim} "
                      f"depth_attn={depth_attn} hier_subspace={hsd} "
                      f"query_dim={ckpt_query_dim}")
                return m
            except (RuntimeError, KeyError) as e:
                last_err = e
    raise RuntimeError(
        f"could not strict-load v2 best.pt (hidden={hidden_dim}, "
        f"layers={num_layers}, depth_attn={depth_attn}); last: {last_err}"
    )


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _mrr_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    order = torch.argsort(scores, descending=True)[:k]
    for rank, idx in enumerate(order, start=1):
        if labels[idx] >= RELEVANCE:
            return 1.0 / rank
    return 0.0


def _rerank_scores(base_scores: torch.Tensor, cand: torch.Tensor,
                   rerank_vals: torch.Tensor) -> torch.Tensor:
    """A score vector that keeps non-candidates below all candidates and
    orders the candidates by ``rerank_vals`` (aligned with ``cand``)."""
    out = torch.full_like(base_scores, float("-inf"))
    out[cand] = rerank_vals
    return out


def evaluate(v3_run: Path, v2_run: Path, corpus: str, split: str,
             split_seed: int, task: int | None, topc: int,
             out_path: Path) -> dict:
    cfg = json.loads((v3_run / "summary.json").read_text())["config"]
    include_tasks = {task} if task is not None else None
    dataset = CorpusDataset(corpus_dir=corpus, split=split,
                            split_seed=split_seed, include_tasks=include_tasks)

    encoder = _build_encoder(cfg, dataset)
    encoder.load_state_dict(torch.load(v3_run / "encoder.pt", map_location="cpu"))
    encoder.eval()
    qenc = build_query_encoder(cfg, dataset)
    qenc.load_state_dict(
        torch.load(v3_run / "query_encoder.pt", map_location="cpu"))
    qenc.eval()
    euclidean = cfg["model"] == "euclidean"
    c_val = getattr(encoder, "c", torch.tensor(float(cfg.get("curvature", 1.0))))
    v2 = _load_v2(v2_run, dataset)
    v2_qd = getattr(v2, "_v2_query_dim", dataset.query_dim)
    if v2_qd != dataset.query_dim:
        raise RuntimeError(
            f"v2 scorer was trained on a {v2_qd}-dim query schema but the "
            f"current corpus emits {dataset.query_dim}-dim queries - the v2 "
            f"model cannot consume v3.1's query. Provide a v2 run trained on "
            f"the current corpus to run the hybrid. v3.1 P1-P3 do not "
            f"depend on this (hybrid is opt-in)."
        )

    rows: list[dict] = []
    with torch.no_grad():
        for i in range(len(dataset)):
            s = dataset[i]
            out = encoder(s.x, s.edge_index, s.edge_type, s.edge_descriptor,
                          node_descriptor=s.node_descriptor)
            emb = out.node_embeddings.detach()
            qp = qenc(s.query)
            v31 = score_from_embeddings(emb, qp, c=c_val, euclidean=euclidean)

            C = min(topc, v31.numel())
            cand = torch.topk(v31, k=C, largest=True).indices

            # v2 on the FULL graph; read node_scores at the candidates.
            v2_out = v2(s.x, s.edge_index, s.edge_type, s.edge_descriptor,
                        s.query, node_descriptor=s.node_descriptor,
                        task_type=s.task_type)
            v2_scores = v2_out.node_scores.detach()

            hybrid = _rerank_scores(v31, cand, v2_scores[cand])
            oracle = _rerank_scores(v31, cand, s.labels[cand])

            rows.append({
                "task_type": int(s.task_type),
                "v31_ndcg@10": ndcg_at_k(v31, s.labels, K),
                "v31_mrr@10": _mrr_at_k(v31, s.labels, K),
                "hybrid_ndcg@10": ndcg_at_k(hybrid, s.labels, K),
                "hybrid_mrr@10": _mrr_at_k(hybrid, s.labels, K),
                "oracle_ndcg@10": ndcg_at_k(oracle, s.labels, K),
            })

    def _m(key: str) -> float:
        return statistics.mean(r[key] for r in rows) if rows else float("nan")

    summary = {k: _m(k) for k in rows[0]} if rows else {}
    summary.pop("task_type", None)
    hybrid_helps = (
        summary.get("hybrid_ndcg@10", 0) > summary.get("v31_ndcg@10", 0)
    )
    results = {
        "v3_run": str(v3_run), "v2_run": str(v2_run),
        "split": split, "task": task, "topc": topc, "n_samples": len(rows),
        "summary": summary,
        "hybrid_beats_v31_alone": bool(hybrid_helps),
        "ceiling_oracle_ndcg@10": summary.get("oracle_ndcg@10"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    _print(results)
    return results


def _print(r: dict) -> None:
    s = r["summary"]
    print()
    print("=" * 76)
    print(f"v3.1 Phase 5 - v2 hybrid reranker ({r['n_samples']} samples, "
          f"topC={r['topc']})")
    print("=" * 76)
    print(f"  v3.1 alone   ndcg@10={s.get('v31_ndcg@10', float('nan')):.4f}  "
          f"mrr@10={s.get('v31_mrr@10', float('nan')):.4f}")
    print(f"  v3.1 + v2    ndcg@10={s.get('hybrid_ndcg@10', float('nan')):.4f}  "
          f"mrr@10={s.get('hybrid_mrr@10', float('nan')):.4f}")
    print(f"  oracle ceil  ndcg@10={s.get('oracle_ndcg@10', float('nan')):.4f}")
    print(f"\n  hybrid beats v3.1-alone: {r['hybrid_beats_v31_alone']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v3-run", type=str, required=True)
    ap.add_argument("--v2-run", type=str, required=True)
    ap.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    ap.add_argument("--split", type=str, default="val",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--task", type=int, default=2)
    ap.add_argument("--topc", type=int, default=50)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    v3_run = Path(args.v3_run)
    out = Path(args.out) if args.out else v3_run / "hybrid.json"
    try:
        evaluate(v3_run, Path(args.v2_run), args.corpus, args.split,
                 args.split_seed,
                 None if args.task < 0 else int(args.task), args.topc, out)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[v2_reranker] hybrid skipped (opt-in, not a P1-P3 blocker): {e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

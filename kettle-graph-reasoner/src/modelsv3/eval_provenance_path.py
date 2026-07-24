r"""v3.1 Phase 1.2 — Eval E: provenance-path recall.

Kettle's real use case is provenance: does retrieval respect
source-proximal derivation chains, or does it jump across the graph?

For each provenance-flavoured val sample (task 0 = provenance, task 3 =
multi-hop), define:

  target t   = the single highest-label node (the thing being traced).
  relevant R = nodes with label >= 0.5 (binary, same convention as
               src.training.metrics).
  prov-path set P = every node lying on SOME shortest *provenance-only*
               path from a relevant node r to t (provenance edges =
               edge category 0 == EDGE_CAT_PROVENANCE).

Metric: ``prov_path_recall@K`` = fraction of the model's top-K scored
nodes (excluding t) that fall in P, for K in {5, 10, 20}. Random
baseline = |P| / (N - 1). Above baseline ==> the manifold's query
landing follows provenance chains rather than teleporting.

Feature recovery (no corpus regeneration):
  depth = round(x[:, 20] * DEPTH_DIVISOR)   # feature_encoder.py:129
  layer = x[:, 12:16].argmax()              # 0 == LAYER_SOURCE
  provenance edge = edge_descriptor[edge_type, 0:4].argmax() == 0
``x[:, 20]`` is asserted in [0, 1]; ``DEPTH_DIVISOR`` (= BuilderConfig
.max_depth = 5) is recorded in the output so a corpus regen with a
different max_depth is detectable.

Usage
-----
    py -m src.modelsv3.eval_provenance_path \
        --checkpoint runs/v3.1-baseline-hyp-h128-l4-seed1/encoder.pt \
        --task 0 \
        --out runs/v3.1-baseline-hyp-h128-l4-seed1/provenance_path_baseline.json
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
from src.modelsv3.distance_scoring import score_from_embeddings  # noqa: E402
from src.modelsv3.eval_candidate_recall import (  # noqa: E402
    _build_encoder,
    build_query_encoder,
)
from src.modelsv3.eval_retrieval_midpoint import (  # noqa: E402
    _bfs_hop_matrix,
    _nodes_on_any_shortest_path,
)

DEPTH_DIVISOR = 5  # BuilderConfig.max_depth; feature_encoder.py:129
EDGE_CAT_PROVENANCE = 0  # schema_sampler.py:31
PROV_TASKS = (0, 3)  # provenance, multi-hop
K_VALUES = (5, 10, 20)
RELEVANCE_THRESHOLD = 0.5


def _provenance_edge_index(
    edge_index: torch.Tensor, edge_type: torch.Tensor,
    edge_descriptor: torch.Tensor,
) -> np.ndarray:
    """(2, E_prov) numpy array of only the provenance edges."""
    cat = edge_descriptor[:, 0:4].argmax(dim=1)            # (T_edge,)
    edge_cat = cat[edge_type.long()]                        # (E,)
    mask = (edge_cat == EDGE_CAT_PROVENANCE).cpu().numpy()
    ei = edge_index.detach().cpu().numpy()
    return ei[:, mask]


def _load(checkpoint: Path, summary: Path, dataset: CorpusDataset):
    with open(summary, "r") as f:
        cfg = json.load(f)["config"]
    encoder = _build_encoder(cfg, dataset)
    encoder.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    encoder.eval()
    qenc = build_query_encoder(cfg, dataset)
    qenc.load_state_dict(
        torch.load(checkpoint.parent / "query_encoder.pt", map_location="cpu")
    )
    qenc.eval()
    return encoder, qenc, cfg


def evaluate_checkpoint(checkpoint: Path, summary: Path, corpus_dir: str,
                        split: str, split_seed: int, task: int | None,
                        out_path: Path) -> dict:
    include_tasks = {task} if task is not None else None
    dataset = CorpusDataset(
        corpus_dir=corpus_dir, split=split, split_seed=split_seed,
        include_tasks=include_tasks,
    )
    encoder, qenc, cfg = _load(checkpoint, summary, dataset)
    euclidean = cfg["model"] == "euclidean"
    c_val = getattr(encoder, "c", torch.tensor(float(cfg.get("curvature", 1.0))))
    print(f"[evalE] {len(dataset)} samples  model={cfg['model']}  "
          f"prov_tasks={PROV_TASKS}")

    emb_cache: dict[int, torch.Tensor] = {}
    rows: list[tuple[int, dict[str, float]]] = []
    n_skipped = 0
    depth_min, depth_max = 1.0, 0.0
    with torch.no_grad():
        for i in range(len(dataset)):
            s = dataset[i]
            if int(s.task_type) not in PROV_TASKS:
                continue
            gi, _ = dataset.index[i]

            # Assert the depth feature layout still holds.
            d_feat = s.x[:, 20]
            depth_min = min(depth_min, float(d_feat.min()))
            depth_max = max(depth_max, float(d_feat.max()))

            if gi not in emb_cache:
                out = encoder(
                    s.x, s.edge_index, s.edge_type, s.edge_descriptor,
                    node_descriptor=s.node_descriptor,
                )
                emb_cache[gi] = out.node_embeddings.detach()
            emb = emb_cache[gi]
            N = emb.size(0)

            labels = s.labels
            if (labels >= RELEVANCE_THRESHOLD).sum().item() < 2:
                n_skipped += 1
                continue
            t = int(torch.argmax(labels).item())
            R = [int(j) for j in torch.nonzero(
                labels >= RELEVANCE_THRESHOLD, as_tuple=False).flatten()
                if int(j) != t]
            if not R:
                n_skipped += 1
                continue

            prov_ei = _provenance_edge_index(
                s.edge_index, s.edge_type, s.edge_descriptor)
            if prov_ei.shape[1] == 0:
                n_skipped += 1
                continue
            prov_hop = _bfs_hop_matrix(prov_ei, N)

            P: set[int] = set()
            for r in R:
                P |= _nodes_on_any_shortest_path(prov_hop, r, t)
            P.discard(t)
            if not P:
                n_skipped += 1
                continue

            scores = score_from_embeddings(
                emb, qenc(s.query), c=c_val, euclidean=euclidean)
            order = [int(j) for j in torch.argsort(scores, descending=True)
                     if int(j) != t]

            row: dict[str, float] = {}
            for k in K_VALUES:
                topk = order[:k]
                row[f"prov_path_recall@{k}"] = (
                    sum(1 for w in topk if w in P) / max(len(topk), 1)
                )
            row["random_baseline"] = len(P) / max(N - 1, 1)
            row["n_prov_path_nodes"] = float(len(P))
            row["n_relevant"] = float(len(R) + 1)
            rows.append((int(s.task_type), row))

    # Depth-feature sanity (defends against a corpus regen).
    depth_ok = (depth_min >= -1e-6) and (depth_max <= 1.0 + 1e-6)

    summary_out = _aggregate(rows)
    results = {
        "checkpoint": str(checkpoint),
        "model_kind": cfg["model"],
        "split": split,
        "task": task,
        "depth_divisor": DEPTH_DIVISOR,
        "depth_feature_index": 20,
        "depth_feature_range_observed": [depth_min, depth_max],
        "depth_feature_in_unit_range": bool(depth_ok),
        "n_eval_samples": len(rows),
        "n_skipped": n_skipped,
        "K_values": list(K_VALUES),
        "summary": summary_out,
    }
    if not depth_ok:
        results["WARNING"] = (
            "x[:,20] outside [0,1] — depth feature layout may have "
            "changed (corpus regenerated with a different max_depth?). "
            "Re-derive DEPTH_DIVISOR before trusting provenance metrics."
        )
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
    print()
    print("=" * 80)
    print(f"Eval E - provenance-path recall "
          f"({r['n_eval_samples']} samples, {r['n_skipped']} skipped)")
    print(f"checkpoint: {r['checkpoint']}  model: {r['model_kind']}")
    print(f"depth divisor={r['depth_divisor']}  "
          f"x[:,20] range={['%.3f' % v for v in r['depth_feature_range_observed']]}  "
          f"in_unit_range={r['depth_feature_in_unit_range']}")
    print("=" * 80)
    if "WARNING" in r:
        print("  !! " + r["WARNING"])
    o = r["summary"].get("overall", {})
    if not o:
        print("  (no eligible provenance/multi-hop samples for this task "
              "filter; pass --task -1 or --task 0)")
        return
    for k in r["K_values"]:
        print(f"  prov_path_recall@{k:<3} {o.get(f'prov_path_recall@{k}', float('nan')):.4f}")
    print(f"  random_baseline       {o.get('random_baseline', float('nan')):.4f}")
    print(f"  mean |prov_path_set|  {o.get('n_prov_path_nodes', float('nan')):.2f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--summary", type=str, default=None)
    p.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    p.add_argument("--split", type=str, default="val",
                   choices=["train", "val", "test", "all"])
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--task", type=int, default=0,
                   help="Task filter. 0=provenance, 3=multi-hop, -1=all "
                        "(only tasks 0 & 3 are scored regardless).")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    summary = Path(args.summary) if args.summary else checkpoint.parent / "summary.json"
    out = Path(args.out) if args.out else checkpoint.parent / "provenance_path.json"
    evaluate_checkpoint(
        checkpoint=checkpoint, summary=summary, corpus_dir=args.corpus,
        split=args.split, split_seed=args.split_seed,
        task=None if args.task < 0 else int(args.task), out_path=out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

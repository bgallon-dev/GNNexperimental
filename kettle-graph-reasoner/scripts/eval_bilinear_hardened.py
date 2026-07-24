"""Non-probe hardened_250 task-2 transfer eval for a TRAINED Stage-B head.

The headline number: take a real train_v3 run (frozen encoder + a trained
Stage-B head), score the FULL real_domain_eval_hardened_250 task-2 set, report
mean ndcg@10 + the anchor-BFS reference line. This is the deployed-model
analogue of probe_bilinear_head's `bilinear_learned`, but on the head trained
through the REAL `_stage_b` pipeline (not the probe's bespoke loop) and on the
FULL hardened set (the trained head never saw any of it — stricter than the
probe's intra-corpus 1/5 split).

Auto-detects the head from summary.json['config']['stage_b_head']:
  - 'bilinear'  -> BilinearStageBHead from stage_b_head.pt; score = q^T M emb
  - 'qtb'/None  -> QueryToBall from query_encoder.pt; score = -dist (the
                   control arm, scored through the SAME harness)

Reuse only (no new geometry/metric): eval_retrieval_nn._load_encoder,
eval_candidate_recall.build_query_encoder, distance_scoring.score_from_embeddings,
src.training.metrics.ndcg_at_k, probe_scoring_ceiling._bfs_dist, CorpusDataset,
stage_b_bilinear.

    py eval_bilinear_hardened.py --run runs/<cell-dir> [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))        # scripts/

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv3.eval_retrieval_nn import _load_encoder  # noqa: E402
from src.modelsv3.distance_scoring import score_from_embeddings  # noqa: E402
from src.modelsv3.stage_b_bilinear import (  # noqa: E402
    BilinearStageBHead, assert_bilinear_run, assert_qtb_run, _stage_b_head)
from src.training.metrics import ndcg_at_k  # noqa: E402
from probe_scoring_ceiling import _bfs_dist  # noqa: E402 (sibling)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True,
                    help="train_v3 out dir (encoder.pt + summary.json + "
                         "stage_b_head.pt | query_encoder.pt)")
    ap.add_argument("--corpus",
                    default="src/data/corpus/real_domain_eval_hardened_250")
    ap.add_argument("--task", type=int, default=2)
    ap.add_argument("--out", default=None,
                    help="optional JSON path for the driver to consume")
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    run = (root / a.run).resolve() if not Path(a.run).is_absolute() \
        else Path(a.run)
    corpus = str((root / a.corpus).resolve())
    summ = run / "summary.json"
    cfg = json.loads(summ.read_text())["config"]
    head_kind = _stage_b_head(cfg)

    ds = CorpusDataset(corpus_dir=corpus, split="all", split_seed=0,
                       include_tasks={a.task})
    enc, ecfg = _load_encoder(run / "encoder.pt", summ, ds)
    c_val = getattr(enc, "c", torch.tensor(float(ecfg["curvature"])))
    euclidean = ecfg["model"] == "euclidean"

    if head_kind == "bilinear":
        assert_bilinear_run(cfg)
        head = BilinearStageBHead(query_dim=ds.query_dim,
                                  hidden_dim=int(ecfg["hidden_dim"]))
        head.load_state_dict(torch.load(run / "stage_b_head.pt",
                                        map_location="cpu"))
    else:
        assert_qtb_run(cfg)
        from src.modelsv3.eval_candidate_recall import build_query_encoder
        head = build_query_encoder(cfg, ds)
        head.load_state_dict(torch.load(run / "query_encoder.pt",
                                        map_location="cpu"))
    head.eval()
    print(f"[evalH] run={run.name}  head={head_kind}  "
          f"enc=h{ecfg['hidden_dim']}/l{ecfg['num_layers']}  "
          f"corpus={Path(corpus).name}  task={a.task}  "
          f"{len(ds)} samples")

    nd, anc = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            gi, j = ds.index[i]
            s = ds[i]
            out = enc(s.x, s.edge_index, s.edge_type, s.edge_descriptor,
                      node_descriptor=s.node_descriptor)
            emb = out.node_embeddings
            if head_kind == "bilinear":
                scores = head(s.query, emb)
            else:
                qp = head(s.query)
                scores = score_from_embeddings(
                    node_embeddings=emb, query_point=qp, c=c_val,
                    euclidean=euclidean)
            L = s.labels
            nd.append(ndcg_at_k(scores.detach().cpu(), L.detach().cpu(), 10))
            npz = np.load(ds.files[gi])
            ascore = _bfs_dist(npz["x"].shape[0],
                               npz["edge_index"].astype(np.int64),
                               int(npz[f"task_{j}_anchor_row"]))
            npz.close()
            anc.append(ndcg_at_k(torch.tensor(ascore), L.detach().cpu(), 10))

    res = {
        "run": str(run), "stage_b_head": head_kind,
        "corpus": Path(corpus).name, "task": a.task,
        "n_graphs": len(nd),
        "ndcg10_mean": float(np.mean(nd)),
        "ndcg10_std": float(np.std(nd)),
        "anchor_bfs_ndcg10_mean": float(np.mean(anc)),
    }
    print(f"[evalH] ndcg@10 mean={res['ndcg10_mean']:.4f} "
          f"(n={res['n_graphs']})  anchor-BFS={res['anchor_bfs_ndcg10_mean']:.4f}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(f"[evalH] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

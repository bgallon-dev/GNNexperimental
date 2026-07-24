r"""P6 — geodesic-guided descent probe (Docs/GEOMETRY_READOUT_PROBES_PLAN.md).

Attacks the pool=find bottleneck: can walking GRAPH edges greedily toward
the query point reach nonlocal positives (min hop >= 2 from the anchor)
that global kNN(q) misses? Descent transits intermediates structurally —
distinct from the refuted additive relay (no score addition anywhere).

Arms (candidate sets capped at C=50, then ordered by -dist to q; bfs_ball
is ordered by hop):
    knn       global top-C by dist to query point   (deployed first stage)
    descent   geodesic_descent(anchor -> q), beam 8, <=12 steps
    hybrid    descent-visited UNION knn, re-ranked, top-C
    bfs_ball  hop-ordered ball around the anchor    (heuristic reference)

Metrics: recall@50 (positives captured by the candidate set) and ndcg@10
after ordering, nonlocal cases only, paired per case.

    py -m scripts.probe_geodesic_descent --ckpt runs/v2trunk-h32-locked \
        --out runs/geometry_probes/p6_descent/results_h32.json
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.codegraph.harness import _build_encoder
from src.data.corpus_dataset import _build_graph_tensors
from src.modelsv3.distance_scoring import score_from_embeddings
from src.modelsv3.eval_candidate_recall import build_query_encoder
from src.modelsv3.geometry_readout import geodesic_descent
from src.training.metrics import ndcg_at_k

# bfs_ball_qorder is a POST-HOC control (added after the first read, noted
# in the findings): same BFS candidate set as bfs_ball but ordered by
# -dist(q) — separates descent's SET quality from its ORDERING quality.
ARMS = ("knn", "descent", "hybrid", "bfs_ball", "bfs_ball_qorder")
C_SET = 50
K_NDCG = 10
NEG_INF = -1e9


def _bfs(adj, src, n):
    d = np.full(n, np.inf, np.float32)
    d[src] = 0.0
    dq = deque([src])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if d[v] == np.inf:
                d[v] = d[u] + 1.0
                dq.append(v)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/real_domain_eval_all6")
    ap.add_argument("--ckpt", default="runs/v2trunk-h32-locked")
    ap.add_argument("--out", default="runs/geometry_probes/p6_descent/results_h32.json")
    ap.add_argument("--n-graphs", type=int, default=200)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = Path(args.ckpt)
    cfg = json.loads((ckpt / "summary.json").read_text())["config"]
    qenc = build_query_encoder(cfg, SimpleNamespace(query_dim=18))
    qenc.load_state_dict(torch.load(ckpt / "query_encoder.pt",
                                    map_location="cpu"))
    qenc.eval()

    enc = None
    rows = []
    files = sorted(Path(args.corpus).glob("graph_*.npz"))[: args.n_graphs]
    print(f"{len(files)} graphs")
    for f in files:
        z = np.load(f, allow_pickle=True)
        g = _build_graph_tensors(z)
        if enc is None:
            enc, _ = _build_encoder(ckpt, g, device)
        with torch.no_grad():
            emb = enc(
                g["x"].to(device), g["edge_index"].to(device),
                g["edge_type"].to(device), g["edge_descriptor"].to(device),
                node_descriptor=g["node_descriptor"].to(device),
            ).node_embeddings
        n = emb.shape[0]
        adj = [[] for _ in range(n)]
        ei = g["edge_index"].numpy()
        for s, t in zip(ei[0], ei[1]):
            adj[int(s)].append(int(t))
            adj[int(t)].append(int(s))
        c = enc.c
        for i in range(int(z["n_tasks"])):
            key = f"task_{i}_anchor_row"
            if key not in z.files:
                continue
            anchor = int(z[key])
            if anchor < 0:
                continue
            labels = z[f"task_{i}_labels"].astype(np.float32)
            query = torch.from_numpy(
                z[f"task_{i}_query"].astype(np.float32))
            pos = np.where(labels >= 0.5)[0]
            pos = pos[pos != anchor]
            if len(pos) == 0:
                continue
            d_anchor = _bfs(adj, anchor, n)
            min_hop = float(np.nanmin(
                np.where(np.isfinite(d_anchor[pos]), d_anchor[pos], np.nan)))
            if not np.isfinite(min_hop) or min_hop < 2.0:
                continue  # nonlocal cases only

            with torch.no_grad():
                q_point = qenc(query)
            if q_point.dim() == 2:
                q_point = q_point.squeeze(0)
            d_q = -score_from_embeddings(emb, q_point, c=c)
            d_q[anchor] = float("inf")

            knn = torch.topk(-d_q, k=min(C_SET, n - 1)).indices.tolist()
            visited = geodesic_descent(
                adj.__getitem__, emb, [anchor], q_point, c=c,
                beam=args.beam, max_steps=args.max_steps)
            visited = [v for v in visited if v != anchor][:C_SET]
            union = sorted(set(visited) | set(knn),
                           key=lambda r: float(d_q[r]))[:C_SET]
            ball = sorted((r for r in range(n)
                           if r != anchor and np.isfinite(d_anchor[r])),
                          key=lambda r: (d_anchor[r], r))[:C_SET]
            cand_sets = {"knn": knn, "descent": visited,
                         "hybrid": union, "bfs_ball": ball,
                         "bfs_ball_qorder": ball}

            lab_t = torch.from_numpy(labels)
            lab_t = lab_t.clone()
            lab_t[anchor] = 0.0
            row = {"graph": f.name, "task_type": int(z[f"task_{i}_type"]),
                   "min_hop": min_hop, "n_pos": int(len(pos)),
                   "n_visited": len(visited)}
            for arm, cand in cand_sets.items():
                cset = set(cand)
                row[f"{arm}_recall{C_SET}"] = float(
                    np.mean([1.0 if p in cset else 0.0 for p in pos]))
                scores = torch.full((n,), NEG_INF)
                if arm == "bfs_ball":
                    for r in cand:
                        scores[r] = -float(d_anchor[r])
                else:
                    for r in cand:
                        scores[r] = -float(d_q[r])
                row[f"{arm}_ndcg{K_NDCG}"] = ndcg_at_k(scores, lab_t, K_NDCG)
            rows.append(row)

    report = {"config": vars(args), "n_cases": len(rows), "arms": {}}
    print(f"\n=== P6 descent probe: {len(rows)} nonlocal cases ===")
    for arm in ARMS:
        rc = np.array([r[f"{arm}_recall{C_SET}"] for r in rows])
        nd = np.array([r[f"{arm}_ndcg{K_NDCG}"] for r in rows])
        report["arms"][arm] = {
            f"recall@{C_SET}_mean": float(rc.mean()),
            f"ndcg@{K_NDCG}_mean": float(nd.mean()),
        }
        print(f"  {arm:<9} recall@{C_SET}={rc.mean():.4f}  "
              f"ndcg@{K_NDCG}={nd.mean():.4f}")

    for a in ("descent", "hybrid", "bfs_ball", "bfs_ball_qorder"):
        dv = np.array([r[f"{a}_recall{C_SET}"] - r[f"knn_recall{C_SET}"]
                       for r in rows])
        dn = np.array([r[f"{a}_ndcg{K_NDCG}"] - r[f"knn_ndcg{K_NDCG}"]
                       for r in rows])
        report[f"{a}_vs_knn"] = {
            "recall_delta_mean": float(dv.mean()),
            "recall_paired_sem": float(dv.std(ddof=1) / np.sqrt(len(dv))),
            "ndcg_delta_mean": float(dn.mean()),
            "ndcg_paired_sem": float(dn.std(ddof=1) / np.sqrt(len(dn))),
        }
        print(f"{a} - knn: recall {dv.mean():+.4f}±"
              f"{dv.std(ddof=1)/np.sqrt(len(dv)):.4f}  "
              f"ndcg {dn.mean():+.4f}±{dn.std(ddof=1)/np.sqrt(len(dn)):.4f}")
    dv = np.array([r[f"descent_recall{C_SET}"]
                   - r[f"bfs_ball_qorder_recall{C_SET}"] for r in rows])
    dn = np.array([r[f"descent_ndcg{K_NDCG}"]
                   - r[f"bfs_ball_qorder_ndcg{K_NDCG}"] for r in rows])
    report["descent_vs_bfs_qorder"] = {
        "recall_delta_mean": float(dv.mean()),
        "recall_paired_sem": float(dv.std(ddof=1) / np.sqrt(len(dv))),
        "ndcg_delta_mean": float(dn.mean()),
        "ndcg_paired_sem": float(dn.std(ddof=1) / np.sqrt(len(dn))),
    }
    print(f"descent - bfs_ball_qorder: recall {dv.mean():+.4f}±"
          f"{dv.std(ddof=1)/np.sqrt(len(dv)):.4f}  "
          f"ndcg {dn.mean():+.4f}±{dn.std(ddof=1)/np.sqrt(len(dn)):.4f}")
    report["n_visited_mean"] = float(
        np.mean([r["n_visited"] for r in rows])) if rows else 0.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"report: {out}")


if __name__ == "__main__":
    main()

r"""P3 — distance-to-geodesic-segment probe (Docs/GEOMETRY_READOUT_PROBES_PLAN.md).

Does the ball lay graph chains along geodesics? Cases: node pairs (a, b)
with shortest-path length 3-6; ground truth = interior nodes of ANY
shortest a-b path (x is on one iff hops(a,x)+hops(x,b) == hops(a,b));
candidates = all nodes except a, b. Zero training.

Arms:
    segment       -min_t dist(x, gamma(t))       (true geodesic, 33 samples)
    midpoint_geo  -dist(x, gamma(0.5))           (true geodesic midpoint)
    midpoint_tan  -dist(x, tangent-origin mid)   (retrieval_ops.bridge today)
    sum_dist      -(d(a,x) + d(x,b))             (2-point additive family)
    random        floor
    hop_oracle    -(hops(a,x)+hops(x,b))         (graph-truth ceiling)

    py -m scripts.probe_geodesic_segment --ckpt runs/v2trunk-h32-locked \
        --out runs/geometry_probes/p3_segment/results_h32.json
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.codegraph.harness import _build_encoder
from src.data.corpus_dataset import _build_graph_tensors
from src.modelsv2.layers import poincare_ops as P
from src.modelsv3.distance_scoring import score_from_embeddings
from src.modelsv3.geometry_readout import (
    dist_to_geodesic_segment,
    geodesic_point,
)
from src.training.metrics import ndcg_at_k, recall_at_k

ARMS = ("segment", "midpoint_geo", "midpoint_tan", "sum_dist",
        "random", "hop_oracle")
PATH_LEN = (3, 6)
CASES_PER_GRAPH = 2
PAIR_ATTEMPTS = 60


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
    ap.add_argument("--out", default="runs/geometry_probes/p3_segment/results_h32.json")
    ap.add_argument("--n-graphs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    enc = None
    rows = []
    files = sorted(Path(args.corpus).glob("graph_*.npz"))[: args.n_graphs]
    print(f"{len(files)} graphs")
    for f in files:
        z = np.load(f, allow_pickle=True)
        g = _build_graph_tensors(z)
        if enc is None:
            enc, _ = _build_encoder(Path(args.ckpt), g, device)
        with torch.no_grad():
            emb = enc(
                g["x"].to(device), g["edge_index"].to(device),
                g["edge_type"].to(device), g["edge_descriptor"].to(device),
                node_descriptor=g["node_descriptor"].to(device),
            ).node_embeddings
        n = emb.shape[0]
        if n < 12:
            continue
        adj = [[] for _ in range(n)]
        ei = g["edge_index"].numpy()
        for s, t in zip(ei[0], ei[1]):
            adj[int(s)].append(int(t))
            adj[int(t)].append(int(s))
        c = enc.c
        dist_cache: dict[int, np.ndarray] = {}

        def hops(src: int) -> np.ndarray:
            if src not in dist_cache:
                dist_cache[src] = _bfs(adj, src, n)
            return dist_cache[src]

        found = 0
        for _ in range(PAIR_ATTEMPTS):
            if found >= CASES_PER_GRAPH:
                break
            a, b = (int(v) for v in rng.choice(n, size=2, replace=False))
            da = hops(a)
            L = da[b]
            if not np.isfinite(L) or not (PATH_LEN[0] <= L <= PATH_LEN[1]):
                continue
            db = hops(b)
            on_path = (da + db == L)
            on_path[a] = on_path[b] = False
            if on_path.sum() < 1:
                continue
            cand = np.array([r for r in range(n) if r not in (a, b)])
            labels = torch.from_numpy(on_path[cand].astype(np.float32))
            e_cand = emb[torch.from_numpy(cand).long()]
            pa, pb = emb[a], emb[b]

            d_seg = dist_to_geodesic_segment(pa, pb, e_cand, c, n_points=33)
            mid_g = geodesic_point(pa, pb, 0.5, c)
            mid_t = P.expmap0(
                (P.logmap0(pa.unsqueeze(0), c)
                 + P.logmap0(pb.unsqueeze(0), c)) / 2, c).squeeze(0)
            s_a = score_from_embeddings(e_cand, pa, c=c)
            s_b = score_from_embeddings(e_cand, pb, c=c)
            scores = {
                "segment": -d_seg,
                "midpoint_geo": score_from_embeddings(e_cand, mid_g, c=c),
                "midpoint_tan": score_from_embeddings(e_cand, mid_t, c=c),
                "sum_dist": s_a + s_b,
                "random": torch.randn(
                    len(cand),
                    generator=torch.Generator().manual_seed(
                        hash((f.name, a, b)) & 0x7FFFFFFF)),
                "hop_oracle": -torch.from_numpy(
                    (da + db)[cand]).float().nan_to_num(posinf=1e6),
            }
            row = {"graph": f.name, "path_len": float(L),
                   "n_pos": int(on_path.sum()), "n_cand": len(cand)}
            for arm in ARMS:
                row[f"{arm}_ndcg10"] = ndcg_at_k(scores[arm], labels, 10)
                row[f"{arm}_recall10"] = recall_at_k(scores[arm], labels, 10)
            rows.append(row)
            found += 1

    n_graphs_used = len({r["graph"] for r in rows})
    report = {"config": vars(args), "n_cases": len(rows),
              "n_graphs_used": n_graphs_used, "arms": {}}
    print(f"\n=== P3 segment probe: {len(rows)} cases, "
          f"{n_graphs_used} graphs ===")
    for arm in ARMS:
        nd = np.array([r[f"{arm}_ndcg10"] for r in rows])
        rc = np.array([r[f"{arm}_recall10"] for r in rows])
        report["arms"][arm] = {
            "ndcg10_mean": float(nd.mean()),
            "ndcg10_sem": float(nd.std(ddof=1) / np.sqrt(len(nd))),
            "recall10_mean": float(rc.mean()),
        }
        print(f"  {arm:<13} ndcg@10={nd.mean():.4f}±"
              f"{nd.std(ddof=1)/np.sqrt(len(nd)):.4f}  r@10={rc.mean():.4f}")

    mid_best = max(("midpoint_geo", "midpoint_tan"),
                   key=lambda a: report["arms"][a]["ndcg10_mean"])
    seg = np.array([r["segment_ndcg10"] for r in rows])
    mid = np.array([r[f"{mid_best}_ndcg10"] for r in rows])
    orc = np.array([r["hop_oracle_ndcg10"] for r in rows])
    dv = seg - mid
    sem = float(dv.std(ddof=1) / np.sqrt(len(dv)))
    gap = float(orc.mean() - mid.mean())
    closed = float(dv.mean() / gap) if gap > 1e-9 else float("nan")
    report["verdict_inputs"] = {
        "mid_best": mid_best,
        "segment_minus_midbest_mean": float(dv.mean()),
        "paired_sem": sem,
        "gap_mid_to_oracle": gap,
        "frac_gap_closed": closed,
    }
    print(f"\nsegment - {mid_best} = {dv.mean():+.4f} ± {sem:.4f} (paired sem)"
          f"   frac of (mid->oracle) gap closed = {closed:+.3f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"report: {out}")


if __name__ == "__main__":
    main()

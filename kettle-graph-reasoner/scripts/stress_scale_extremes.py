r"""Stress probe: ball-order quality vs neighborhood SIZE (edge of distribution).

On the first N all6 graphs, bucket each task ball by size:
  tiny (<10), small (10-30), mid (30-80), large (>80)
Report emb-order ndcg@10, hop-order ndcg@10, and random floor per bucket
with n. Mirrors scripts/probe_capability_ballrank.py idioms exactly.

Run from kettle-graph-reasoner/:
    PYTHONIOENCODING=utf-8 py -m scripts.stress_scale_extremes \
        --corpus src/data/corpus/real_domain_eval_all6 \
        --n-graphs 80 --out runs/stress_scale_extremes
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
from src.modelsv3.distance_scoring import score_from_embeddings
from src.training.metrics import ndcg_at_k

FAMILY = {0: "provenance", 1: "entity_res", 2: "temporal",
          3: "multihop", 4: "subgraph", 5: "compound"}


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


def _bucket(sz):
    if sz < 10:
        return "tiny(<10)"
    if sz <= 30:
        return "small(10-30)"
    if sz <= 80:
        return "mid(30-80)"
    return "large(>80)"


BUCKETS = ["tiny(<10)", "small(10-30)", "mid(30-80)", "large(>80)"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/real_domain_eval_all6")
    ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--n-graphs", type=int, default=80)
    ap.add_argument("--out", default="runs/stress_scale_extremes")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

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
        adj = [[] for _ in range(n)]
        ei = g["edge_index"].numpy()
        for s, t in zip(ei[0], ei[1]):
            adj[int(s)].append(int(t))
            adj[int(t)].append(int(s))
        for i in range(int(z["n_tasks"])):
            fam = FAMILY.get(int(z[f"task_{i}_type"]), "?")
            anchor = int(z[f"task_{i}_anchor_row"])
            labels = z[f"task_{i}_labels"]
            mh = float(z[f"task_{i}_max_hops"])
            d = _bfs(adj, anchor, n)
            ball = [r for r in range(n)
                    if r != anchor and d[r] <= mh and np.isfinite(d[r])]
            lab_b = torch.from_numpy(labels[ball]).float()
            if len(ball) >= 5 and float(lab_b.max()) > 0 \
                    and float(lab_b.min()) < float(lab_b.max()):
                rows_t = torch.tensor(ball, dtype=torch.long)
                d_e = -score_from_embeddings(emb[rows_t], emb[anchor], c=enc.c)
                hp = torch.from_numpy(d[ball]).float()
                rnd = torch.randn(
                    len(ball),
                    generator=torch.Generator().manual_seed(
                        hash((f.name, i)) & 0x7FFFFFFF))
                rows.append({
                    "family": fam,
                    "n_ball": len(ball),
                    "bucket": _bucket(len(ball)),
                    "emb_ndcg10": ndcg_at_k(-d_e, lab_b, 10),
                    "hop_ndcg10": ndcg_at_k(-hp, lab_b, 10),
                    "rand_ndcg10": ndcg_at_k(rnd, lab_b, 10),
                })

    def _avg(sub, key):
        v = [r[key] for r in sub]
        return sum(v) / len(v) if v else float("nan")

    report = {"config": vars(args), "n_cases_total": len(rows), "by_bucket": {}}
    print("\n=== BALL-ORDER ndcg@10 by SIZE bucket (real all6) ===")
    print(f"{'bucket':<14} {'n':>4} {'emb':>8} {'hop':>8} {'random':>8} "
          f"{'emb-rand':>9}")
    for b in BUCKETS + ["ALL"]:
        sub = rows if b == "ALL" else [r for r in rows if r["bucket"] == b]
        emb = _avg(sub, "emb_ndcg10")
        hop = _avg(sub, "hop_ndcg10")
        rnd = _avg(sub, "rand_ndcg10")
        cell = {"n": len(sub), "emb_ndcg10": emb, "hop_ndcg10": hop,
                "rand_ndcg10": rnd, "emb_minus_rand": emb - rnd if sub else float("nan"),
                "median_ball": (float(np.median([r["n_ball"] for r in sub]))
                                if sub else float("nan"))}
        report["by_bucket"][b] = cell
        if sub:
            print(f"{b:<14} {len(sub):>4} {emb:>8.3f} {hop:>8.3f} {rnd:>8.3f} "
                  f"{emb - rnd:>9.3f}")
        else:
            print(f"{b:<14} {0:>4} {'--':>8} {'--':>8} {'--':>8} {'--':>9}")

    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

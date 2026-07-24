r"""Stress probe (key=robustness): does zero-training ball-ordering survive
graph corruption?

Baseline = emb-order the BFS ball (candidates + labels from the CLEAN graph)
by distance-to-anchor. Then re-embed / re-anchor under three corruptions and
recompute ndcg@10 on the SAME held-fixed candidate set + labels, so numbers
are directly comparable:

 (a) FEATURE NOISE  : add Gaussian noise sigma in {0.05,0.1,0.25} to x, re-embed
 (b) EDGE DROPOUT   : drop {10,25,50}% of edges, re-embed
 (c) ANCHOR CORRUPT : order ball by distance to a RANDOM node hop 1-2 from the
                      true anchor (graph clean, ball/labels unchanged)

Ball, candidate set, and labels are ALWAYS from the clean graph so the task is
identical across arms. random-order floor included.

Run from kettle-graph-reasoner/:
    py -m scripts.stress_robustness --n 50 --out runs/stress_robustness
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


def _embed(enc, x, ei, et, ed, nd, device):
    with torch.no_grad():
        return enc(x.to(device), ei.to(device), et.to(device),
                   ed.to(device), node_descriptor=nd.to(device)).node_embeddings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/real_domain_eval_all6")
    ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="runs/stress_robustness")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    feat_sig = [0.05, 0.1, 0.25]
    drop_p = [0.10, 0.25, 0.50]

    enc = None
    rows = []
    files = sorted(Path(args.corpus).glob("graph_*.npz"))[: args.n]
    print(f"{len(files)} graphs")
    for f in files:
        z = np.load(f, allow_pickle=True)
        g = _build_graph_tensors(z)
        if enc is None:
            enc, _ = _build_encoder(Path(args.ckpt), g, device)
        x, ei, et, ed, nd = (g["x"], g["edge_index"], g["edge_type"],
                             g["edge_descriptor"], g["node_descriptor"])
        emb_clean = _embed(enc, x, ei, et, ed, nd, device)
        n = emb_clean.shape[0]

        # clean adjacency for BFS ball
        adj = [[] for _ in range(n)]
        ein = ei.numpy()
        for s, t in zip(ein[0], ein[1]):
            adj[int(s)].append(int(t))
            adj[int(t)].append(int(s))

        # precompute corrupted embeddings (graph-level, shared across tasks)
        emb_feat = {}
        for sig in feat_sig:
            xn = x + torch.from_numpy(
                rng.normal(0, sig, size=x.shape).astype(np.float32))
            emb_feat[sig] = _embed(enc, xn, ei, et, ed, nd, device)
        emb_drop = {}
        E = ei.shape[1]
        for p in drop_p:
            keep = rng.random(E) >= p
            if keep.sum() == 0:
                keep[0] = True
            ei2 = ei[:, keep]
            et2 = et[keep]
            emb_drop[p] = _embed(enc, x, ei2, et2, ed, nd, device)

        for i in range(int(z["n_tasks"])):
            tk = f"task_{i}_type"
            if tk not in z:
                continue
            fam = FAMILY.get(int(z[tk]), "?")
            anchor = int(z[f"task_{i}_anchor_row"])
            labels = z[f"task_{i}_labels"]
            mh = float(z[f"task_{i}_max_hops"])
            d = _bfs(adj, anchor, n)
            ball = [r for r in range(n)
                    if r != anchor and d[r] <= mh and np.isfinite(d[r])]
            lab_b = torch.from_numpy(labels[ball]).float()
            if not (len(ball) >= 5 and float(lab_b.max()) > 0
                    and float(lab_b.min()) < float(lab_b.max())):
                continue
            rows_t = torch.tensor(ball, dtype=torch.long)

            def emb_score(E_emb, anch):
                # higher = closer = more relevant
                return score_from_embeddings(E_emb[rows_t], E_emb[anch], c=enc.c)

            row = {"family": fam, "n_ball": len(ball)}
            row["baseline"] = ndcg_at_k(emb_score(emb_clean, anchor), lab_b, 10)
            # random floor
            rsc = torch.from_numpy(rng.normal(size=len(ball)).astype(np.float32))
            row["random"] = ndcg_at_k(rsc, lab_b, 10)
            # (a) feature noise
            for sig in feat_sig:
                row[f"feat_{sig}"] = ndcg_at_k(
                    emb_score(emb_feat[sig], anchor), lab_b, 10)
            # (b) edge dropout
            for p in drop_p:
                row[f"drop_{p}"] = ndcg_at_k(
                    emb_score(emb_drop[p], anchor), lab_b, 10)
            # (c) anchor corruption: random node at hop 1-2 from true anchor
            cand = [r for r in range(n) if d[r] in (1.0, 2.0)]
            if cand:
                bad = int(rng.choice(cand))
                row["anchor_h12"] = ndcg_at_k(
                    emb_score(emb_clean, bad), lab_b, 10)
            rows.append(row)

    def _avg(key):
        v = [r[key] for r in rows if key in r]
        return sum(v) / len(v) if v else float("nan")

    metrics = (["baseline", "random"]
               + [f"feat_{s}" for s in feat_sig]
               + [f"drop_{p}" for p in drop_p]
               + ["anchor_h12"])
    base = _avg("baseline")
    summary = {m: _avg(m) for m in metrics}
    rel_drop = {m: (base - summary[m]) / base if base else float("nan")
                for m in metrics}
    report = {"config": vars(args), "n_cases": len(rows),
              "mean_ndcg10": summary, "rel_drop_vs_baseline": rel_drop}
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))

    print(f"\n=== ball-order ndcg@10 robustness (n={len(rows)} tasks) ===")
    print(f"{'arm':<14} {'ndcg@10':>9} {'rel_drop':>9}")
    for m in metrics:
        print(f"{m:<14} {summary[m]:>9.3f} {rel_drop[m]:>9.1%}")
    print(f"\nreport: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

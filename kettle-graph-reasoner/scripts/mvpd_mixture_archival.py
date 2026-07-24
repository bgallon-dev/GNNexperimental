r"""E4 MVP-D: do anchor-gyro-frame mixture offsets replicate on the REAL
archival domain? (pre-registered in Docs/ARCH_EFFICIENCY_PLAN.md, E4)

The mixture line's load-bearing question: are answer-cluster offsets
task-consistent in the anchor's gyro-frame OUTSIDE the code-graph domain
where they were discovered? Zero gradients; graph-level 70/30 split of
real_domain_eval_all6 (fit on train graphs, eval POOL retrieval on
held-out graphs; candidates = every node in the graph except the anchor).

Arms: mixture (K=3 task-family offsets, max-score), anchor_emb (v=0
control — separates "offsets help" from "anchor neighborhood helps"),
anchor_bfs (-hops), random. Buckets: ALL cases + `beyond_ball` (cases
with >=1 positive outside the task's BFS ball — where pool retrieval
has to beat ball assembly).

BAR (MVP-D): mixture > anchor_emb on held-out pool cases (ALL); the
interesting cell is beyond_ball. Kmeans per-case seeded (lens-fix
convention). Live-Neo4j replication is a separate follow-up.

    PYTHONIOENCODING=utf-8 py -m scripts.mvpd_mixture_archival
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.codegraph.harness import _build_encoder
from src.data.corpus_dataset import _build_graph_tensors
from src.modelsv2.layers import poincare_ops as P
from src.modelsv3.distance_scoring import score_from_embeddings
from src.modelsv3.mixture_offsets import kmeans, mixture_query_points
from src.training.metrics import ndcg_at_k

FAMILY = {0: "provenance", 1: "entity_res", 2: "temporal",
          3: "multihop", 4: "subgraph", 5: "compound"}
CKPT = "frozen/kgr-v1.0-2026-07-07/encoder_baseline"
CORPUS = "src/data/corpus/real_domain_eval_all6"
OUT = Path("runs/mvpd_mixture_archival")
K = 3
TRAIN_FRAC = 0.7
SPLIT_SEED = 0


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


def _load(f, enc, device):
    z = np.load(f, allow_pickle=True)
    g = _build_graph_tensors(z)
    with torch.no_grad():
        emb = enc(g["x"].to(device), g["edge_index"].to(device),
                  g["edge_type"].to(device), g["edge_descriptor"].to(device),
                  node_descriptor=g["node_descriptor"].to(device),
                  ).node_embeddings
    n = emb.shape[0]
    adj = [[] for _ in range(n)]
    ei = g["edge_index"].numpy()
    for s, t in zip(ei[0], ei[1]):
        adj[int(s)].append(int(t))
        adj[int(t)].append(int(s))
    return z, g, emb, adj, n


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    files = sorted(Path(CORPUS).glob("graph_*.npz"))
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(files))
    n_train = int(len(files) * TRAIN_FRAC)
    train_f = [files[i] for i in order[:n_train]]
    eval_f = [files[i] for i in order[n_train:]]
    print(f"{len(files)} graphs -> {len(train_f)} fit / {len(eval_f)} eval")

    enc = None
    c = None

    # ---- fit: per-family task-level offsets on train graphs ----
    per_fam: dict[str, list[torch.Tensor]] = {}
    n_cases_fit = 0
    for f in train_f:
        if enc is None:
            z0 = np.load(f, allow_pickle=True)
            enc, _ = _build_encoder(Path(CKPT), _build_graph_tensors(z0),
                                    device)
            c = enc.c
        z, g, emb, adj, n = _load(f, enc, device)
        for i in range(int(z["n_tasks"])):
            fam = FAMILY.get(int(z[f"task_{i}_type"]), "?")
            anchor = int(z[f"task_{i}_anchor_row"])
            pos = [r for r in np.nonzero(z[f"task_{i}_labels"] >= 0.5)[0]
                   if int(r) != anchor]
            if not pos:
                continue
            rows = torch.tensor(sorted(int(r) for r in pos))
            offs = P.logmap0(P.mobius_add(-emb[anchor], emb[rows], c), c)
            gen = torch.Generator().manual_seed(
                hash((f.name, i, "mvpd")) & 0x7FFFFFFF)
            torch.manual_seed(int(gen.initial_seed()) & 0x7FFFFFFF)
            per_fam.setdefault(fam, []).append(kmeans(offs, K))
            n_cases_fit += 1
    V = {}
    for fam, cents in per_fam.items():
        torch.manual_seed(hash((fam, "pool")) & 0x7FFFFFFF)
        V[fam] = kmeans(torch.cat(cents, dim=0), K)
    print(f"fit: {n_cases_fit} cases -> offsets for {sorted(V)} "
          f"(norms {[round(float(v.norm(dim=-1).mean()), 2) for v in V.values()]})")

    # ---- eval: pool retrieval on held-out graphs ----
    rows_out = []
    for f in eval_f:
        z, g, emb, adj, n = _load(f, enc, device)
        for i in range(int(z["n_tasks"])):
            fam = FAMILY.get(int(z[f"task_{i}_type"]), "?")
            if fam not in V:
                continue
            anchor = int(z[f"task_{i}_anchor_row"])
            labels = z[f"task_{i}_labels"]
            mh = float(z[f"task_{i}_max_hops"])
            cand = [r for r in range(n) if r != anchor]
            lab = torch.from_numpy(labels[cand]).float()
            if not (float(lab.max()) > 0 and float(lab.min()) < float(lab.max())):
                continue
            d = _bfs(adj, anchor, n)
            pos_idx = [r for r in np.nonzero(labels >= 0.5)[0]
                       if int(r) != anchor]
            beyond = any((not np.isfinite(d[r])) or d[r] > mh
                         for r in pos_idx)
            rows_t = torch.tensor(cand, dtype=torch.long)
            qps = mixture_query_points(emb, anchor, V[fam], c)
            sc_mix = torch.stack(
                [score_from_embeddings(emb[rows_t], q, c=c) for q in qps]
            ).max(0).values
            sc_a = score_from_embeddings(emb[rows_t], emb[anchor], c=c)
            hp = d[cand]
            hp = np.where(np.isfinite(hp), hp, n + 1.0)
            gen = torch.Generator().manual_seed(
                hash((f.name, i, "rnd")) & 0x7FFFFFFF)
            rows_out.append({
                "family": fam, "beyond_ball": bool(beyond),
                "mixture": ndcg_at_k(sc_mix, lab, 10),
                "anchor_emb": ndcg_at_k(sc_a, lab, 10),
                "anchor_bfs": ndcg_at_k(
                    torch.from_numpy((-hp).astype(np.float32)), lab, 10),
                "random": ndcg_at_k(
                    torch.randn(len(cand), generator=gen), lab, 10),
            })

    arms = ("mixture", "anchor_emb", "anchor_bfs", "random")

    def _avg(sub, a):
        v = [r[a] for r in sub]
        return sum(v) / len(v) if v else float("nan")

    report = {"n_fit_cases": n_cases_fit, "cells": {}}
    print(f"\n=== held-out POOL ndcg@10 (archival all6) ===")
    print(f"{'cell':<28}{'n':>5} " + " ".join(f"{a:>11}" for a in arms))
    fams = sorted({r["family"] for r in rows_out})
    for name, sub in (
        [("ALL", rows_out),
         ("beyond_ball", [r for r in rows_out if r["beyond_ball"]]),
         ("within_ball", [r for r in rows_out if not r["beyond_ball"]])]
        + [(f"fam:{fm}", [r for r in rows_out if r["family"] == fm])
           for fm in fams]
    ):
        cell = {a: _avg(sub, a) for a in arms} | {"n": len(sub)}
        report["cells"][name] = cell
        print(f"{name:<28}{len(sub):>5} "
              + " ".join(f"{cell[a]:>11.3f}" for a in arms))
    mix, ae = report["cells"]["ALL"]["mixture"], \
        report["cells"]["ALL"]["anchor_emb"]
    report["bar_mixture_gt_anchor_emb"] = bool(mix > ae)
    print(f"\nBAR mixture > anchor_emb (ALL): "
          f"{'CLEARED' if mix > ae else 'NOT cleared'} "
          f"({mix:.3f} vs {ae:.3f})")
    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    print(f"report: {OUT / 'results.json'}")


if __name__ == "__main__":
    main()

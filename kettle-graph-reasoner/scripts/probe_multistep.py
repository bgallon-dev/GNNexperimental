r"""Multi-step composition probe: relay ranking + out-of-ball sweep.

Combines the validated single-step primitives into 2-step pipelines, on
the real all6 corpus, targeting the weak families (compound 0.431,
multihop, temporal 0.630 single-ball):

Candidate set per task = ball(anchor) UNION balls(top-3 emb intermediates)
UNION top-50 emb-ranked out-of-ball nodes (the rescue sweep). Arms score
the SAME candidate set:

  ball_only   -d_emb(anchor) inside the original ball; -inf outside
              (the shipped single-step strategy)
  union_emb   -d_emb(anchor) everywhere (sweep, no relay)
  relay_emb   -min(d_emb(anchor), d_emb(intermediate_i)) — 2-step:
              anchor finds intermediates, intermediates vouch for
              their own neighborhoods
  union_hop   -hops(anchor) (BFS strategy on the same set)
  oracle      labels

    py -m scripts.probe_multistep
"""

from __future__ import annotations

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
CKPT = Path("frozen/kgr-v1.0-2026-07-07/encoder_baseline")
OUT = Path("runs/probe_multistep")


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
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    enc = None
    rows = []
    files = sorted(Path("src/data/corpus/real_domain_eval_all6")
                   .glob("graph_*.npz"))
    print(f"{len(files)} graphs")
    for f in files:
        z = np.load(f, allow_pickle=True)
        g = _build_graph_tensors(z)
        if enc is None:
            enc, _ = _build_encoder(CKPT, g, device)
        with torch.no_grad():
            emb = enc(g["x"], g["edge_index"], g["edge_type"],
                      g["edge_descriptor"],
                      node_descriptor=g["node_descriptor"]).node_embeddings
        n = emb.shape[0]
        adj = [[] for _ in range(n)]
        for s, t in zip(*g["edge_index"].numpy()):
            adj[int(s)].append(int(t))
            adj[int(t)].append(int(s))
        d_all = {}

        def hops(src):
            if src not in d_all:
                d_all[src] = _bfs(adj, src, n)
            return d_all[src]

        all_rows = torch.arange(n)
        for i in range(int(z["n_tasks"])):
            fam = FAMILY.get(int(z[f"task_{i}_type"]), "?")
            anchor = int(z[f"task_{i}_anchor_row"])
            labels = torch.from_numpy(z[f"task_{i}_labels"]).float()
            mh = float(z[f"task_{i}_max_hops"])
            d = hops(anchor)
            d_e_all = -score_from_embeddings(emb, emb[anchor], c=enc.c)
            ball = [r for r in range(n) if r != anchor and d[r] <= mh]
            if len(ball) < 5:
                continue
            # top-3 emb intermediates inside the ball
            bt = torch.tensor(ball)
            inter = [ball[j] for j in torch.argsort(
                d_e_all[bt]).tolist()[:3]]
            # candidate union
            cand = set(ball)
            for m in inter:
                dm = hops(m)
                cand.update(r for r in range(n)
                            if r != anchor and dm[r] <= mh)
            outb = [r for r in range(n)
                    if r != anchor and (d[r] > mh or not np.isfinite(d[r]))]
            if outb:
                ot = torch.tensor(outb)
                cand.update(outb[j] for j in
                            torch.argsort(d_e_all[ot]).tolist()[:50])
            cand = sorted(cand)
            lab = labels[torch.tensor(cand)]
            if float(lab.max()) <= 0 or float(lab.min()) == float(lab.max()):
                continue
            ct = torch.tensor(cand)
            de_a = d_e_all[ct]
            relay = de_a.clone()
            relay_add = de_a.clone()   # direct path as the base
            for m in inter:
                d_m = -score_from_embeddings(emb[ct], emb[m], c=enc.c)
                relay = torch.minimum(relay, d_m)
                # additive: total travel anchor->inter->cand; intermediates
                # only help if the whole path is short
                relay_add = torch.minimum(
                    relay_add, float(d_e_all[m]) + d_m)
            in_ball = torch.tensor([r in set(ball) for r in cand])
            ninf = torch.finfo(torch.float32).min
            hp = torch.from_numpy(
                np.where(np.isfinite(d[cand]), d[cand], n + 1.0)
            ).float()
            arms = {
                "ball_only": torch.where(in_ball, -de_a,
                                         torch.full_like(de_a, ninf)),
                "union_emb": -de_a,
                "relay_emb": -relay,
                "relay_add": -relay_add,
                "union_hop": -hp,
                "oracle": lab.clone(),
            }
            row = {"family": fam, "n_cand": len(cand)}
            for a, sc in arms.items():
                row[a] = ndcg_at_k(sc, lab, 10)
            rows.append(row)

    arms = ("ball_only", "union_emb", "relay_emb", "relay_add",
            "union_hop", "oracle")
    print(f"\n=== MULTI-STEP ndcg@10 (expanded candidate set) ===")
    print(f"{'family':<12} {'n':>5} " + " ".join(f"{a:>10}" for a in arms))
    report = {}
    for fam in sorted({r["family"] for r in rows}) + ["ALL"]:
        sub = rows if fam == "ALL" else [r for r in rows
                                         if r["family"] == fam]
        cell = {a: sum(r[a] for r in sub) / len(sub) for a in arms}
        cell["n"] = len(sub)
        report[fam] = cell
        print(f"{fam:<12} {len(sub):>5} "
              + " ".join(f"{cell[a]:>10.3f}" for a in arms))
    (OUT / "multistep_results.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {OUT / 'multistep_results.json'}")


if __name__ == "__main__":
    main()

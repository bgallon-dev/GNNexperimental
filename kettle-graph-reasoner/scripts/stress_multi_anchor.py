r"""Stress probe: compositional / multi-anchor ball ordering (key=multi_anchor).

Only SINGLE-anchor emb-order ball ranking is validated. Here we ask whether a
SECOND (oracle) anchor helps AT ALL on the two families where composition is
plausible: compound (type 5) and subgraph (type 4).

For each such task we synthesize a 2nd anchor = the relevant node (label>=0.5)
FARTHEST IN HOPS from the primary anchor (oracle -> this is a CEILING probe).
We then order the BFS ball (hop<=max_hops around primary anchor) three ways:
  (a) single   : emb-dist to primary anchor          [validated baseline]
  (b) min       : min over {a1,a2} of emb-dist        [union of two balls]
  (c) midpoint : emb-dist to mobius geodesic midpoint of a1,a2
plus random floor and oracle ceiling. Metric = ndcg@10.

The 2nd anchor node itself is EXCLUDED from the scored ball for ALL arms so the
comparison is apples-to-apples (no trivial self-match leak).

CAVEAT: the 2nd anchor is ORACLE (chosen using ground-truth labels). This
bounds the *best case* of a two-anchor scheme; it is NOT a deployable method.

Run from kettle-graph-reasoner/:
    PYTHONIOENCODING=utf-8 py -m scripts.stress_multi_anchor \
        --corpus src/data/corpus/real_domain_eval_all6 \
        --out runs/stress_multi_anchor
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
from src.training.metrics import ndcg_at_k

FAMILY = {4: "subgraph", 5: "compound"}


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


def _midpoint(x, y, c):
    # geodesic midpoint: x (+) 0.5 (x) ((-x) (+) y)
    diff = P.mobius_add(-x, y, c)
    half = P.mobius_scalar_mul(0.5, diff, c)
    return P.mobius_add(x, half, c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/real_domain_eval_all6")
    ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--out", default="runs/stress_multi_anchor")
    ap.add_argument("--n-graphs", type=int, default=200)
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
            ttype = int(z[f"task_{i}_type"])
            if ttype not in FAMILY:
                continue
            fam = FAMILY[ttype]
            anchor = int(z[f"task_{i}_anchor_row"])
            labels = z[f"task_{i}_labels"]
            mh = float(z[f"task_{i}_max_hops"])
            d = _bfs(adj, anchor, n)
            # 2nd anchor = relevant node farthest (in hops) from primary anchor
            rel = [r for r in range(n)
                   if r != anchor and labels[r] >= 0.5 and np.isfinite(d[r])]
            if not rel:
                continue
            a2 = max(rel, key=lambda r: d[r])
            # ball, excluding BOTH anchors for a fair apples-to-apples compare
            ball = [r for r in range(n)
                    if r != anchor and r != a2 and d[r] <= mh
                    and np.isfinite(d[r])]
            lab_b = torch.from_numpy(labels[ball]).float()
            if len(ball) < 5 or float(lab_b.max()) <= 0 \
                    or float(lab_b.min()) >= float(lab_b.max()):
                continue
            rows_t = torch.tensor(ball, dtype=torch.long)
            e_ball = emb[rows_t]
            a1p, a2p = emb[anchor], emb[a2]
            s1 = score_from_embeddings(e_ball, a1p, c=enc.c)     # -dist to a1
            s2 = score_from_embeddings(e_ball, a2p, c=enc.c)     # -dist to a2
            s_min = torch.maximum(s1, s2)                        # min-dist union
            mid = _midpoint(a1p, a2p, enc.c)
            s_mid = score_from_embeddings(e_ball, mid, c=enc.c)
            rnd = torch.randn(
                len(ball),
                generator=torch.Generator().manual_seed(
                    hash((f.name, i)) & 0x7FFFFFFF))
            rows.append({
                "family": fam,
                "n_ball": len(ball),
                "a2_hops": float(d[a2]),
                "single_ndcg10": ndcg_at_k(s1, lab_b, 10),
                "min_ndcg10": ndcg_at_k(s_min, lab_b, 10),
                "midpoint_ndcg10": ndcg_at_k(s_mid, lab_b, 10),
                "random_ndcg10": ndcg_at_k(rnd, lab_b, 10),
                "oracle_ndcg10": ndcg_at_k(lab_b.clone(), lab_b, 10),
            })

    def _avg(sub, key):
        v = [r[key] for r in sub]
        return sum(v) / len(v) if v else float("nan")

    arms = ("single", "min", "midpoint", "random", "oracle")
    fams = sorted({r["family"] for r in rows})
    report = {"config": vars(args), "by_family": {}}
    print("\n=== MULTI-ANCHOR BALL RERANK ndcg@10 (oracle 2nd anchor) ===")
    print(f"{'family':<10} {'n':>4} " + " ".join(f"{a:>9}" for a in arms)
          + f" {'a2_hops':>8}")
    for fam in fams + ["ALL"]:
        sub = rows if fam == "ALL" else [r for r in rows if r["family"] == fam]
        cell = {a: _avg(sub, f"{a}_ndcg10") for a in arms}
        cell["n"] = len(sub)
        cell["a2_hops_mean"] = _avg(sub, "a2_hops")
        report["by_family"][fam] = cell
        print(f"{fam:<10} {len(sub):>4} "
              + " ".join(f"{cell[a]:>9.3f}" for a in arms)
              + f" {cell['a2_hops_mean']:>8.2f}")

    allc = report["by_family"]["ALL"]
    report["deltas_vs_single_ALL"] = {
        "min": allc["min"] - allc["single"],
        "midpoint": allc["midpoint"] - allc["single"],
    }
    print("\ndelta vs single (ALL): "
          f"min={report['deltas_vs_single_ALL']['min']:+.4f} "
          f"midpoint={report['deltas_vs_single_ALL']['midpoint']:+.4f}")
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

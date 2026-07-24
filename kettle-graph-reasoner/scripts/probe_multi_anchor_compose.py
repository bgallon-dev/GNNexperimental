r"""P4 — multi-anchor composition op probe (Docs/GEOMETRY_READOUT_PROBES_PLAN.md).

Replicates the stress_multi_anchor protocol (oracle far-positive 2nd anchor,
BFS ball around the primary anchor, both anchors excluded) and adds the
Karcher-mean challenger arms:

    single       -dist to a1                       (validated baseline)
    union_max    max(-d(a1), -d(a2))               (the verified +0.471 arm)
    karcher      -dist to Karcher mean K=2 equal   (== true geodesic midpoint;
                                                    kept to confirm equality)
    karcher_w    -dist to Karcher mean, weights    (challenger: weight
                 prop. to 1/(1+ball-eccentricity)   toward the anchor whose
                 of each anchor)                    neighborhood is tighter)
    midpoint_tan -dist to tangent-at-origin mid    (what retrieval_ops.bridge
                                                    ships TODAY — not the true
                                                    geodesic midpoint)
    random / oracle floors and ceilings.

Ship rule (pre-registered): union_max lands in retrieval_ops regardless
(replication gate: delta within +-0.10 of +0.471); karcher becomes a
"between" mode ONLY if it beats union_max by > 1 paired std.

Run from kettle-graph-reasoner/:
    py -m scripts.probe_multi_anchor_compose --out runs/geometry_probes/p4_multi_anchor
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
from src.modelsv3.geometry_readout import geodesic_point, karcher_mean
from src.training.metrics import ndcg_at_k

FAMILY = {4: "subgraph", 5: "compound"}
ARMS = ("single", "union_max", "karcher", "karcher_w", "midpoint_tan",
        "random", "oracle")


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


def _midpoint_tan(x, y, c):
    # retrieval_ops.bridge basis: tangent-at-origin average
    return P.expmap0(
        (P.logmap0(x.unsqueeze(0), c) + P.logmap0(y.unsqueeze(0), c)) / 2,
        c).squeeze(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/real_domain_eval_all6")
    ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--out", default="runs/geometry_probes/p4_multi_anchor")
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
            anchor = int(z[f"task_{i}_anchor_row"])
            labels = z[f"task_{i}_labels"]
            mh = float(z[f"task_{i}_max_hops"])
            d = _bfs(adj, anchor, n)
            rel = [r for r in range(n)
                   if r != anchor and labels[r] >= 0.5 and np.isfinite(d[r])]
            if not rel:
                continue
            a2 = max(rel, key=lambda r: d[r])
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
            c = enc.c
            s1 = score_from_embeddings(e_ball, a1p, c=c)
            s2 = score_from_embeddings(e_ball, a2p, c=c)

            anchors = torch.stack([a1p, a2p])
            # eccentricity of each anchor within the ball -> weights
            ecc = torch.stack([(-s1).mean(), (-s2).mean()])
            w = 1.0 / (1.0 + ecc)
            km = karcher_mean(anchors, c=c)
            km_w = karcher_mean(anchors, w, c=c)
            mid_geo = geodesic_point(a1p, a2p, 0.5, c)
            km_vs_mid = float(P.dist(km.unsqueeze(0), mid_geo.unsqueeze(0), c))

            scores = {
                "single": s1,
                "union_max": torch.maximum(s1, s2),
                "karcher": score_from_embeddings(e_ball, km, c=c),
                "karcher_w": score_from_embeddings(e_ball, km_w, c=c),
                "midpoint_tan": score_from_embeddings(
                    e_ball, _midpoint_tan(a1p, a2p, c), c=c),
                "random": torch.randn(
                    len(ball),
                    generator=torch.Generator().manual_seed(
                        hash((f.name, i)) & 0x7FFFFFFF)),
                "oracle": lab_b.clone(),
            }
            row = {"family": FAMILY[ttype], "n_ball": len(ball),
                   "a2_hops": float(d[a2]), "karcher_vs_midgeo_dist": km_vs_mid}
            for a in ARMS:
                row[f"{a}_ndcg10"] = ndcg_at_k(scores[a], lab_b, 10)
            rows.append(row)

    def _avg(sub, key):
        v = [r[key] for r in sub]
        return sum(v) / len(v) if v else float("nan")

    fams = sorted({r["family"] for r in rows})
    report = {"config": vars(args), "n_cases": len(rows), "by_family": {}}
    print("\n=== P4 multi-anchor composition ndcg@10 (oracle 2nd anchor) ===")
    print(f"{'family':<10} {'n':>4} " + " ".join(f"{a:>12}" for a in ARMS))
    for fam in fams + ["ALL"]:
        sub = rows if fam == "ALL" else [r for r in rows if r["family"] == fam]
        cell = {a: _avg(sub, f"{a}_ndcg10") for a in ARMS}
        cell["n"] = len(sub)
        report["by_family"][fam] = cell
        print(f"{fam:<10} {len(sub):>4} "
              + " ".join(f"{cell[a]:>12.3f}" for a in ARMS))

    allr = rows
    deltas = {}
    for a in ("union_max", "karcher", "karcher_w", "midpoint_tan"):
        dv = np.array([r[f"{a}_ndcg10"] - r["single_ndcg10"] for r in allr])
        deltas[a] = {"mean": float(dv.mean()),
                     "std": float(dv.std(ddof=1) / np.sqrt(len(dv)))}
    kv = np.array([r["karcher_w_ndcg10"] - r["union_max_ndcg10"] for r in allr])
    report["deltas_vs_single_ALL"] = deltas
    report["karcher_w_minus_union"] = {
        "mean": float(kv.mean()),
        "paired_sem": float(kv.std(ddof=1) / np.sqrt(len(kv)))}
    report["karcher_eq_midgeo_maxdist"] = float(
        max(r["karcher_vs_midgeo_dist"] for r in allr))
    print("\ndeltas vs single (ALL, paired sem): "
          + "  ".join(f"{a}={d['mean']:+.4f}±{d['std']:.4f}"
                      for a, d in deltas.items()))
    print(f"karcher_w - union_max = {kv.mean():+.4f} "
          f"± {report['karcher_w_minus_union']['paired_sem']:.4f} (paired sem)")
    print(f"max dist(karcher K=2, true geo midpoint) = "
          f"{report['karcher_eq_midgeo_maxdist']:.2e}")
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(f"report: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

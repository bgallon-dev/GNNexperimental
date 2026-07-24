r"""Schema-portability probe: does zero-training ball-rank survive off-tree?

Same lens as probe_capability_ballrank (frozen encoder, order the BFS ball
by emb-distance-to-anchor vs by hop-order), but re-bucketed by DOMAIN
FAMILY (topology) instead of task family, and correlated with per-graph
structural stats. Answers the schema-portability question directly: the
archival reference is emb_order 0.885 > hop_order 0.690 (emb WINS on
tree-like graphs). Where does that hold and where does it break as the
graph stops being tree-like?

Reads a corpus built by build_diverse_domains.py (each npz carries a
`domain_family` tag and `stat_*` fields).

Run from kettle-graph-reasoner/:
    py -m scripts.probe_diverse_domains \
        --corpus src/data/corpus/diverse_domains \
        --out runs/probe_diverse_domains
"""

from __future__ import annotations

import argparse
import json
from collections import deque, defaultdict
from pathlib import Path

import numpy as np
import torch

from src.codegraph.harness import _build_encoder
from src.data.corpus_dataset import _build_graph_tensors
from src.modelsv3.distance_scoring import score_from_embeddings
from src.training.metrics import ndcg_at_k, recall_at_k

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/diverse_domains")
    ap.add_argument("--ckpt",
                    default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--out", default="runs/probe_diverse_domains")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    enc = None
    euc = False
    c_val = 1.0
    rows = []       # per within-ball case
    rescue = []     # per nonlocal case
    files = sorted(Path(args.corpus).glob("graph_*.npz"))
    print(f"{len(files)} graphs")
    for f in files:
        z = np.load(f, allow_pickle=True)
        family = str(z["domain_family"]) if "domain_family" in z else "?"
        stats = {k[5:]: float(z[k]) for k in z.files if k.startswith("stat_")}
        g = _build_graph_tensors(z)
        if enc is None:
            enc, cfg = _build_encoder(Path(args.ckpt), g, device)
            euc = cfg.get("model", "hyperbolic") == "euclidean"
            c_val = getattr(enc, "c", 1.0)
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
                d_e = -score_from_embeddings(emb[rows_t], emb[anchor],
                                             c=c_val, euclidean=euc)
                hp = torch.from_numpy(d[ball]).float()
                de_n = d_e / (float(d_e.max()) + 1e-6)
                arms = {
                    "hop_order": -hp,
                    "emb_order": -d_e,
                    "hop_tb_emb": -(hp + de_n),
                    "random": torch.randn(
                        len(ball),
                        generator=torch.Generator().manual_seed(
                            hash((f.name, i)) & 0x7FFFFFFF)),
                    "oracle": lab_b.clone(),
                }
                row = {"family": family, "task": fam, "n_ball": len(ball),
                       **stats}
                for a, sc in arms.items():
                    row[f"{a}"] = ndcg_at_k(sc, lab_b, 10)
                rows.append(row)
            # nonlocal rescue
            out = [r for r in range(n)
                   if r != anchor and (not np.isfinite(d[r]) or d[r] > mh)]
            if out:
                lab_o = torch.from_numpy((labels[out] >= 0.5)
                                         .astype(np.float32))
                n_rel = int(lab_o.sum())
                if n_rel > 0 and len(out) > 50:
                    rows_o = torch.tensor(out, dtype=torch.long)
                    sc_o = score_from_embeddings(
                        emb[rows_o], emb[anchor], c=c_val, euclidean=euc)
                    rescue.append({
                        "family": family, "n_out": len(out), "n_rel": n_rel,
                        "emb_r50": recall_at_k(sc_o, lab_o, 50),
                        "rand_r50": min(50.0 / len(out), 1.0)})

    def _avg(sub, key):
        v = [r[key] for r in sub if key in r]
        return sum(v) / len(v) if v else float("nan")

    arms = ("hop_order", "emb_order", "hop_tb_emb", "random", "oracle")
    fams = sorted({r["family"] for r in rows})
    report = {"config": vars(args),
              "archival_reference": {"emb_order": 0.885, "hop_order": 0.690},
              "by_family": {}, "rescue": {}, "correlation": {}}

    print("\n=== BALL-RANK ndcg@10 by DOMAIN family (frozen encoder) ===")
    print("(archival reference: emb_order 0.885 > hop_order 0.690)")
    hdr = f"{'domain':16s} {'n':>4} " + " ".join(f"{a:>10}" for a in arms)
    print(hdr + f" {'emb-hop':>8} {'clust':>6} {'cyc_exc':>7} {'delta':>6}")
    order = ["deep_tree", "scale_free", "grid2d", "ring_mesh",
             "bipartite", "dense_community"]
    ordered = [f for f in order if f in fams] + \
              [f for f in fams if f not in order]
    for fam in ordered + ["ALL"]:
        sub = rows if fam == "ALL" else [r for r in rows
                                         if r["family"] == fam]
        if not sub:
            continue
        cell = {a: _avg(sub, a) for a in arms}
        cell["n"] = len(sub)
        cell["emb_minus_hop"] = cell["emb_order"] - cell["hop_order"]
        for s in ("clustering", "cycle_excess", "delta_proxy",
                  "mean_degree", "max_degree", "n_nodes"):
            cell[s] = _avg(sub, s)
        report["by_family"][fam] = cell
        print(f"{fam:16s} {len(sub):>4} "
              + " ".join(f"{cell[a]:>10.3f}" for a in arms)
              + f" {cell['emb_minus_hop']:>+8.3f} "
              f"{cell['clustering']:>6.3f} {cell['cycle_excess']:>7.3f} "
              f"{cell['delta_proxy']:>6.3f}")

    # correlation: does the emb-hop advantage track topology?
    print("\n=== CORRELATION: per-case (emb_order - hop_order) vs structure ===")
    deltas = np.array([r["emb_order"] - r["hop_order"] for r in rows])
    for s in ("clustering", "cycle_excess", "delta_proxy", "mean_degree"):
        xs = np.array([r.get(s, np.nan) for r in rows])
        m = np.isfinite(xs) & np.isfinite(deltas)
        if m.sum() > 2 and xs[m].std() > 0:
            r_pearson = float(np.corrcoef(xs[m], deltas[m])[0, 1])
        else:
            r_pearson = float("nan")
        report["correlation"][s] = r_pearson
        print(f"  corr(emb-hop, {s:14s}) = {r_pearson:+.3f}")

    print("\n=== NONLOCAL RESCUE recall@50 by domain family ===")
    print(f"{'domain':16s} {'n_cases':>7} {'n_rel':>6} {'emb':>8} {'rand':>8}")
    for fam in ordered + ["ALL"]:
        sub = rescue if fam == "ALL" else [r for r in rescue
                                           if r["family"] == fam]
        if not sub:
            continue
        cell = {"n_cases": len(sub),
                "n_rel": sum(r["n_rel"] for r in sub),
                "emb_r50": _avg(sub, "emb_r50"),
                "rand_r50": _avg(sub, "rand_r50")}
        report["rescue"][fam] = cell
        print(f"{fam:16s} {cell['n_cases']:>7} {cell['n_rel']:>6} "
              f"{cell['emb_r50']:>8.3f} {cell['rand_r50']:>8.3f}")

    (out_dir / "diverse_results.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out_dir / 'diverse_results.json'}")


if __name__ == "__main__":
    main()

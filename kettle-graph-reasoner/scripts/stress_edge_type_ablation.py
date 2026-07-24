r"""Stress probe: is edge-typed heterogeneous attention load-bearing? (key=edge_type_ablation)

Baseline emb-order ball ndcg@10 on first 50 real all6 graphs, then COLLAPSE
edge typing and re-embed:
  - edge_type  -> all zeros (single class)
  - edge_descriptor -> every row set identical to row 0 (shape kept valid)
Combined effect: EdgeTypedAttention.type_emb_override.index_select(edge_type)
returns the SAME type embedding for every edge -> attention's per-edge type
term `t = W_t(t_emb)` is constant across edges -> heterogeneous attention
degenerates to homogeneous (scores depend only on q_dst/k_src node content).

Control/floor: random ordering of the ball (mean over 3 seeds per case).

Run from kettle-graph-reasoner/:
    PYTHONIOENCODING=utf-8 py -m scripts.stress_edge_type_ablation
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
CORPUS = "src/data/corpus/real_domain_eval_all6"
CKPT = "frozen/kgr-v1.0-2026-07-07/encoder_baseline"
OUT = "runs/stress_edge_type_ablation"
N_GRAPHS = 50


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
    out_dir = Path(OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    enc = None
    rows = []
    emb_diffs = []
    files = sorted(Path(CORPUS).glob("graph_*.npz"))[:N_GRAPHS]
    print(f"{len(files)} graphs")
    for f in files:
        z = np.load(f, allow_pickle=True)
        g = _build_graph_tensors(z)
        if enc is None:
            enc, _ = _build_encoder(Path(CKPT), g, device)
        x, ei = g["x"], g["edge_index"]
        et, ed, nd = g["edge_type"], g["edge_descriptor"], g["node_descriptor"]
        # BASELINE
        emb_b = _embed(enc, x, ei, et, ed, nd, device)
        # COLLAPSED edge typing
        et_c = torch.zeros_like(et)                       # all edges -> class 0
        ed_c = ed[0:1].repeat(ed.shape[0], 1).clone()     # all rows identical
        emb_c = _embed(enc, x, ei, et_c, ed_c, nd, device)
        emb_diffs.append(float((emb_b - emb_c).norm(dim=-1).mean()))
        n = emb_b.shape[0]
        adj = [[] for _ in range(n)]
        ein = ei.numpy()
        for s, t in zip(ein[0], ein[1]):
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
            if len(ball) < 5:
                continue
            lab_b = torch.from_numpy(labels[ball]).float()
            if not (float(lab_b.max()) > 0 and float(lab_b.min()) < float(lab_b.max())):
                continue
            rows_t = torch.tensor(ball, dtype=torch.long)
            sc_base = score_from_embeddings(emb_b[rows_t], emb_b[anchor], c=enc.c)
            sc_coll = score_from_embeddings(emb_c[rows_t], emb_c[anchor], c=enc.c)
            # random floor: 3 seeds
            rnds = []
            for s_ in range(3):
                g_ = torch.Generator().manual_seed((hash((f.name, i)) & 0xFFFFFF) + s_)
                rnds.append(ndcg_at_k(torch.randn(len(ball), generator=g_), lab_b, 10))
            rows.append({
                "family": fam, "n_ball": len(ball),
                "base_ndcg10": ndcg_at_k(sc_base, lab_b, 10),
                "coll_ndcg10": ndcg_at_k(sc_coll, lab_b, 10),
                "rand_ndcg10": sum(rnds) / len(rnds),
            })

    def _avg(sub, k):
        v = [r[k] for r in sub]
        return sum(v) / len(v) if v else float("nan")

    fams = sorted({r["family"] for r in rows})
    report = {"n_graphs": len(files), "n_cases": len(rows),
              "mean_emb_L2_shift": sum(emb_diffs) / len(emb_diffs),
              "by_family": {}}
    print(f"\nmean per-node emb L2 shift after collapse: "
          f"{report['mean_emb_L2_shift']:.4f}")
    print(f"\n{'family':<12} {'n':>4} {'base':>8} {'collapsed':>10} "
          f"{'delta':>8} {'random':>8}")
    for fam in fams + ["ALL"]:
        sub = rows if fam == "ALL" else [r for r in rows if r["family"] == fam]
        b, c_, r_ = (_avg(sub, "base_ndcg10"), _avg(sub, "coll_ndcg10"),
                     _avg(sub, "rand_ndcg10"))
        report["by_family"][fam] = {"n": len(sub), "base_ndcg10": b,
                                    "coll_ndcg10": c_, "delta": b - c_,
                                    "rand_ndcg10": r_}
        print(f"{fam:<12} {len(sub):>4} {b:>8.4f} {c_:>10.4f} "
              f"{b - c_:>+8.4f} {r_:>8.4f}")
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

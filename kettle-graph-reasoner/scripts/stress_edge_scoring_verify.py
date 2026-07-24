r"""Adversarial verification of key=edge_scoring.

Re-runs the SAME edge-ranking measurement as scripts/stress_edge_scoring.py
but on a DISJOINT slice (graphs [start:end]) and with tightened controls:

  emb_ndcg10   : mean-endpoint -dist(anchor) edge score  (the claim)
  hop_ndcg10   : -min(hop) edge score (pure BFS structure control)
  rand_ndcg10  : random edge order (original control), NEW seed
Diagnostics:
  frac_reledge_incident_anchor : fraction of cases where the (single)
      relevant edge touches the anchor node -> then emb wins trivially
      because dist(anchor,anchor)=0.
  emb_ndcg10_nonincident : emb score restricted to cases where NO relevant
      edge is incident to the anchor (the honest, non-trivial subset).

Run from kettle-graph-reasoner/:
    PYTHONIOENCODING=utf-8 py -m scripts.stress_edge_scoring_verify \
        --start 100 --end 160 --out runs/stress_edge_scoring_verify
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
from src.training.metrics import ndcg_at_k, recall_at_k

FAMILY = {0: "provenance", 4: "subgraph"}


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
    ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--out", default="runs/stress_edge_scoring_verify")
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--end", type=int, default=160)
    ap.add_argument("--seed-salt", type=int, default=987654321)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    enc = None
    rows = []
    files = sorted(Path(args.corpus).glob("graph_*.npz"))[args.start:args.end]
    print(f"{len(files)} graphs (slice [{args.start}:{args.end}])")

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
        ei = g["edge_index"].numpy()
        E = ei.shape[1]
        s_arr, t_arr = ei[0], ei[1]

        adj = [[] for _ in range(n)]
        for s, t in zip(s_arr, t_arr):
            adj[int(s)].append(int(t))
            adj[int(t)].append(int(s))

        for i in range(int(z["n_tasks"])):
            ttype = int(z[f"task_{i}_type"])
            if ttype not in FAMILY:
                continue
            anchor = int(z[f"task_{i}_anchor_row"])
            labels = z[f"task_{i}_labels"]
            rel_node = labels >= 0.5

            edge_lab = (rel_node[s_arr] & rel_node[t_arr]).astype(np.float32)
            n_rel_edges = int(edge_lab.sum())
            if n_rel_edges == 0:
                continue

            # emb edge score = mean of -dist(endpoint, anchor)
            node_sc = score_from_embeddings(emb, emb[anchor], c=enc.c)
            node_sc = node_sc.detach().cpu().numpy()
            edge_sc = 0.5 * (node_sc[s_arr] + node_sc[t_arr])

            # hop control: -mean hop of endpoints (pure BFS structure)
            hd = _bfs(adj, anchor, n)
            hd_f = np.where(np.isfinite(hd), hd, 1e6)
            hop_sc = -(0.5 * (hd_f[s_arr] + hd_f[t_arr]))

            # is the relevant edge incident to the anchor?
            rel_edge_mask = edge_lab > 0.5
            incident = bool(((s_arr[rel_edge_mask] == anchor) |
                             (t_arr[rel_edge_mask] == anchor)).any())

            edge_lab_t = torch.from_numpy(edge_lab)
            emb_sc_t = torch.from_numpy(edge_sc.astype(np.float32))
            hop_sc_t = torch.from_numpy(hop_sc.astype(np.float32))
            rng = torch.Generator().manual_seed(
                (hash((f.name, i)) ^ args.seed_salt) & 0x7FFFFFFF)
            rand_sc_t = torch.randn(E, generator=rng)

            rows.append({
                "family": FAMILY[ttype],
                "E": E,
                "n_rel_edges": n_rel_edges,
                "incident": incident,
                "emb_ndcg10": ndcg_at_k(emb_sc_t, edge_lab_t, 10),
                "emb_recall10": recall_at_k(emb_sc_t, edge_lab_t, 10),
                "hop_ndcg10": ndcg_at_k(hop_sc_t, edge_lab_t, 10),
                "hop_recall10": recall_at_k(hop_sc_t, edge_lab_t, 10),
                "rand_ndcg10": ndcg_at_k(rand_sc_t, edge_lab_t, 10),
                "rand_recall10": recall_at_k(rand_sc_t, edge_lab_t, 10),
            })

    def _avg(sub, key):
        v = [r[key] for r in sub]
        return sum(v) / len(v) if v else float("nan")

    fams = sorted({r["family"] for r in rows})
    report = {"config": vars(args), "by_family": {}}
    print("\n=== EDGE RANKING VERIFY (disjoint slice) ===")
    hdr = (f"{'family':<12} {'n':>4} {'relE':>5} {'incid%':>7} "
           f"{'emb':>7} {'hop':>7} {'rand':>7} {'emb_nonInc':>10}")
    print(hdr)
    for fam in fams + ["ALL"]:
        sub = rows if fam == "ALL" else [r for r in rows if r["family"] == fam]
        non_inc = [r for r in sub if not r["incident"]]
        cell = {
            "n_cases": len(sub),
            "mean_n_rel_edges": _avg(sub, "n_rel_edges"),
            "frac_reledge_incident_anchor":
                (sum(r["incident"] for r in sub) / len(sub)) if sub else float("nan"),
            "emb_ndcg10": _avg(sub, "emb_ndcg10"),
            "hop_ndcg10": _avg(sub, "hop_ndcg10"),
            "rand_ndcg10": _avg(sub, "rand_ndcg10"),
            "emb_recall10": _avg(sub, "emb_recall10"),
            "hop_recall10": _avg(sub, "hop_recall10"),
            "rand_recall10": _avg(sub, "rand_recall10"),
            "n_nonincident": len(non_inc),
            "emb_ndcg10_nonincident": _avg(non_inc, "emb_ndcg10"),
            "hop_ndcg10_nonincident": _avg(non_inc, "hop_ndcg10"),
            "rand_ndcg10_nonincident": _avg(non_inc, "rand_ndcg10"),
        }
        report["by_family"][fam] = cell
        print(f"{fam:<12} {cell['n_cases']:>4} "
              f"{cell['mean_n_rel_edges']:>5.1f} "
              f"{cell['frac_reledge_incident_anchor']:>7.2f} "
              f"{cell['emb_ndcg10']:>7.3f} {cell['hop_ndcg10']:>7.3f} "
              f"{cell['rand_ndcg10']:>7.3f} "
              f"{cell['emb_ndcg10_nonincident']:>10.3f}")
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

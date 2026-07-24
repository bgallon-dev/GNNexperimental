r"""Probe key=edge_scoring — per-edge relevance scoring (never used in spec).

The KGR spec says output is per-NODE AND per-EDGE relevance. Only per-node
has ever been exercised. This probe asks: can the frozen encoder rank
edges by relevance with the simplest possible edge score?

Edge score  = mean over the edge's two endpoints of -dist(endpoint, anchor_emb)
Edge label  = 1.0 if BOTH endpoints have node-label >= 0.5, else 0.0
Families    = provenance (type 0) and subgraph (type 4)
Metrics     = edge-ranking ndcg@10 and recall@10 vs a random edge order.

Also probes whether KGREmbeddingOutput exposes a usable per-edge embedding
(edge_type_embeddings) for an independent geometry-based edge score.

Run from kettle-graph-reasoner/:
    PYTHONIOENCODING=utf-8 py -m scripts.stress_edge_scoring \
        --corpus src/data/corpus/real_domain_eval_all6 \
        --out runs/stress_edge_scoring
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.codegraph.harness import _build_encoder
from src.data.corpus_dataset import _build_graph_tensors
from src.modelsv3.distance_scoring import score_from_embeddings
from src.training.metrics import ndcg_at_k, recall_at_k

FAMILY = {0: "provenance", 4: "subgraph"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/real_domain_eval_all6")
    ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--out", default="runs/stress_edge_scoring")
    ap.add_argument("--n-graphs", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    enc = None
    rows = []
    edge_geom_usable = None
    edge_geom_note = ""
    files = sorted(Path(args.corpus).glob("graph_*.npz"))
    if args.n_graphs > 0:
        files = files[: args.n_graphs]
    print(f"{len(files)} graphs")

    for f in files:
        z = np.load(f, allow_pickle=True)
        g = _build_graph_tensors(z)
        if enc is None:
            enc, _ = _build_encoder(Path(args.ckpt), g, device)
        with torch.no_grad():
            out = enc(
                g["x"].to(device), g["edge_index"].to(device),
                g["edge_type"].to(device), g["edge_descriptor"].to(device),
                node_descriptor=g["node_descriptor"].to(device),
            )
        emb = out.node_embeddings
        n = emb.shape[0]
        ei = g["edge_index"].numpy()
        E = ei.shape[1]
        edge_type = g["edge_type"].numpy()

        # inspect the per-edge-embedding claim once
        if edge_geom_usable is None:
            ete = out.edge_type_embeddings
            # (T, type_dim): one row per edge TYPE, shared by all edges of
            # that type. Not a per-edge object; lives in type_dim space, not
            # the node ball. Cannot measure distance to a node anchor.
            edge_geom_usable = False
            edge_geom_note = (
                f"edge_type_embeddings shape={tuple(ete.shape)} is per-TYPE "
                f"(T={ete.shape[0]} rows), shared across all edges of a type, "
                f"in a {ete.shape[1]}-d space disjoint from the node ball; "
                "no per-edge geometry and no node-anchor comparison possible."
            )

        for i in range(int(z["n_tasks"])):
            ttype = int(z[f"task_{i}_type"])
            if ttype not in FAMILY:
                continue
            anchor = int(z[f"task_{i}_anchor_row"])
            labels = z[f"task_{i}_labels"]
            rel_node = labels >= 0.5

            # edge label = both endpoints relevant
            s = ei[0]
            t = ei[1]
            edge_lab = (rel_node[s] & rel_node[t]).astype(np.float32)
            n_rel_edges = int(edge_lab.sum())
            if n_rel_edges == 0:
                continue  # undefined ranking target

            # edge score = mean of -dist(endpoint, anchor) over both endpoints
            node_sc = score_from_embeddings(emb, emb[anchor], c=enc.c)  # (N,)
            node_sc = node_sc.detach().cpu().numpy()
            edge_sc = 0.5 * (node_sc[s] + node_sc[t])

            edge_lab_t = torch.from_numpy(edge_lab)
            emb_sc_t = torch.from_numpy(edge_sc.astype(np.float32))
            rng = torch.Generator().manual_seed(
                hash((f.name, i)) & 0x7FFFFFFF)
            rand_sc_t = torch.randn(E, generator=rng)

            rows.append({
                "family": FAMILY[ttype],
                "E": E,
                "n_rel_edges": n_rel_edges,
                "emb_ndcg10": ndcg_at_k(emb_sc_t, edge_lab_t, 10),
                "emb_recall10": recall_at_k(emb_sc_t, edge_lab_t, 10),
                "rand_ndcg10": ndcg_at_k(rand_sc_t, edge_lab_t, 10),
                "rand_recall10": recall_at_k(rand_sc_t, edge_lab_t, 10),
            })

    def _avg(sub, key):
        v = [r[key] for r in sub]
        return sum(v) / len(v) if v else float("nan")

    fams = sorted({r["family"] for r in rows})
    report = {"config": vars(args),
              "edge_geom_usable": edge_geom_usable,
              "edge_geom_note": edge_geom_note,
              "by_family": {}}
    print("\n=== EDGE RANKING (emb mean-endpoint -dist vs random) ===")
    hdr = f"{'family':<12} {'n':>4} {'relE':>5} " \
          f"{'emb_ndcg':>9} {'rnd_ndcg':>9} {'emb_rec':>8} {'rnd_rec':>8}"
    print(hdr)
    for fam in fams + ["ALL"]:
        sub = rows if fam == "ALL" else [r for r in rows if r["family"] == fam]
        cell = {
            "n_cases": len(sub),
            "mean_n_rel_edges": _avg(sub, "n_rel_edges"),
            "emb_ndcg10": _avg(sub, "emb_ndcg10"),
            "rand_ndcg10": _avg(sub, "rand_ndcg10"),
            "emb_recall10": _avg(sub, "emb_recall10"),
            "rand_recall10": _avg(sub, "rand_recall10"),
        }
        report["by_family"][fam] = cell
        print(f"{fam:<12} {cell['n_cases']:>4} "
              f"{cell['mean_n_rel_edges']:>5.1f} "
              f"{cell['emb_ndcg10']:>9.3f} {cell['rand_ndcg10']:>9.3f} "
              f"{cell['emb_recall10']:>8.3f} {cell['rand_recall10']:>8.3f}")
    print("\nedge_geom_usable:", edge_geom_usable, "-", edge_geom_note)
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

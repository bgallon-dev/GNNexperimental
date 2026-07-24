r"""Capability sweep: what IS the frozen model good at? (zero/low training)

Regime shift from the deployed-routing eval (which measured pool
retrieval): here we measure the two capabilities the evidence says should
exist, on the real archival all6 corpus, per task family:

1. WITHIN-BALL RERANKING - candidates = the BFS ball around the task
   anchor (hop <= task max_hops), i.e. exactly what a Cypher walk hands
   the context assembler. Can the model ORDER that ball better than hop
   order? Arms:
     hop_order      -hops (the sleeper-finding baseline)
     emb_order      -d_c(emb[anchor], emb[cand])       [zero training]
     hop_tb_emb     hop order, embedding tie-break     [zero training]
     head_synth     frozen synthetic QueryToBall (18-d task query)
     head_real      real-trained temporal head (task-2 validated)
     random / oracle
2. NONLOCAL RESCUE - relevant nodes OUTSIDE the ball (BFS structurally
   cannot see them). Rank out-of-ball nodes by emb-distance-to-anchor:
   recall@50 vs the random expectation. Tests "finds what BFS can't".

Run from kettle-graph-reasoner/:
    py -m scripts.probe_capability_ballrank \
        --corpus src/data/corpus/real_domain_eval_all6 \
        --out runs/probe_capability_ballrank
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.codegraph.harness import _build_encoder
from src.codegraph.metrics_ext import mrr
from src.data.corpus_dataset import _build_graph_tensors
from src.modelsv3.distance_scoring import score_from_embeddings
from src.modelsv3.retrieval_ops import load_query_encoder
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/real_domain_eval_all6")
    ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--real-head", default="frozen/kgr-v1.0-2026-07-07/real_head")
    ap.add_argument("--out", default="runs/probe_capability_ballrank")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    def _try_head(d):
        try:
            return load_query_encoder(d)
        except Exception:
            return None
    head_s = _try_head(args.ckpt)
    head_r = _try_head(args.real_head)
    heads_checked = False

    enc = None
    euc = False
    c_val = 1.0
    rows, rescue = [], []
    files = sorted(Path(args.corpus).glob("graph_*.npz"))
    print(f"{len(files)} graphs")
    for f in files:
        z = np.load(f, allow_pickle=True)
        g = _build_graph_tensors(z)
        if enc is None:
            enc, cfg = _build_encoder(Path(args.ckpt), g, device)
            # E2: euclidean-control checkpoints score with L2 distance.
            euc = cfg.get("model", "hyperbolic") == "euclidean"
            c_val = getattr(enc, "c", 1.0)
            if euc:
                print("[ballrank] euclidean geometry (L2 scoring)")
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
            q = torch.from_numpy(z[f"task_{i}_query"]).float()
            if not heads_checked:
                # E2: a head trained at another width (e.g. the frozen
                # h128 real head vs an h32/h40 trunk) cannot score these
                # embeddings — drop the arm instead of crashing.
                heads_checked = True

                def _dim_ok(h):
                    try:
                        return h(q).shape[-1] == emb.shape[-1]
                    except Exception:
                        return False

                if head_s is not None and not _dim_ok(head_s):
                    print("[ballrank] head_synth dim mismatch; arm dropped")
                    head_s = None
                if head_r is not None and not _dim_ok(head_r):
                    print("[ballrank] head_real dim mismatch; arm dropped")
                    head_r = None
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
                # zero-training ordering variants (no retraining levers):
                # same-type bonus and radius(depth)-tiebreak
                x = z["x"]
                ttypes = x[:, :16].argmax(axis=1)
                same_t = torch.from_numpy(
                    (ttypes[ball] == ttypes[anchor]).astype(np.float32))
                rad = emb.norm(dim=-1)
                arms = {
                    "hop_order": -hp,
                    "emb_order": -d_e,
                    "emb_type": -d_e + 2.0 * same_t,
                    "emb_rad_tb": -d_e - 0.05 * rad[rows_t].cpu(),
                    "hop_tb_emb": -(hp + de_n),   # int hops; emb breaks ties
                    **({"head_synth": score_from_embeddings(
                        emb[rows_t], head_s(q), c=c_val, euclidean=euc)}
                       if head_s else {}),
                    **({"head_real": score_from_embeddings(
                        emb[rows_t], head_r(q), c=c_val, euclidean=euc)}
                       if head_r else {}),
                    "random": torch.randn(
                        len(ball),
                        generator=torch.Generator().manual_seed(
                            hash((f.name, i)) & 0x7FFFFFFF)),
                    "oracle": lab_b.clone(),
                }
                row = {"family": fam, "n_ball": len(ball)}
                for a, sc in arms.items():
                    row[f"{a}_ndcg10"] = ndcg_at_k(sc, lab_b, 10)
                rows.append(row)
            # nonlocal rescue: relevant nodes outside the ball
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
                        "family": fam, "n_out": len(out), "n_rel": n_rel,
                        "emb_r50": recall_at_k(sc_o, lab_o, 50),
                        "rand_r50": min(50.0 / len(out), 1.0),
                    })

    def _avg(sub, key):
        v = [r[key] for r in sub]
        return sum(v) / len(v) if v else float("nan")

    arms = ("hop_order", "emb_order", "emb_type", "emb_rad_tb",
            "hop_tb_emb", "head_synth", "head_real", "random", "oracle")
    fams = sorted({r["family"] for r in rows})
    report = {"config": vars(args), "ballrank": {}, "rescue": {}}
    print("\n=== WITHIN-BALL RERANK ndcg@10 by family (real all6) ===")
    print(f"{'family':<12} {'n':>4} " + " ".join(f"{a:>10}" for a in arms))
    for fam in fams + ["ALL"]:
        sub = rows if fam == "ALL" else [r for r in rows if r["family"] == fam]
        cell = {a: _avg(sub, f"{a}_ndcg10") for a in arms
                if any(f"{a}_ndcg10" in r for r in sub)}
        cell["n"] = len(sub)
        report["ballrank"][fam] = cell
        print(f"{fam:<12} {len(sub):>4} "
              + " ".join(f"{cell[a]:>10.3f}" for a in arms if a in cell))
    print("\n=== NONLOCAL RESCUE recall@50 (relevant OUTSIDE the ball) ===")
    print(f"{'family':<12} {'n_cases':>7} {'n_rel':>6} {'emb':>8} {'random':>8}")
    for fam in sorted({r["family"] for r in rescue}) + ["ALL"]:
        sub = rescue if fam == "ALL" else [r for r in rescue
                                           if r["family"] == fam]
        if not sub:
            continue
        cell = {"n_cases": len(sub),
                "n_rel": sum(r["n_rel"] for r in sub),
                "emb_r50": _avg(sub, "emb_r50"),
                "rand_r50": _avg(sub, "rand_r50")}
        report["rescue"][fam] = cell
        print(f"{fam:<12} {cell['n_cases']:>7} {cell['n_rel']:>6} "
              f"{cell['emb_r50']:>8.3f} {cell['rand_r50']:>8.3f}")
    (out_dir / "capability_results.json").write_text(
        json.dumps(report, indent=2))
    print(f"\nreport: {out_dir / 'capability_results.json'}")


if __name__ == "__main__":
    main()

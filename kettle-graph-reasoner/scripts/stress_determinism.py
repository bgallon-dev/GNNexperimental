r"""Stress probe: determinism + permutation-equivariance of the frozen encoder.

(a) DETERMINISM: embed the same all6 graph twice (rebuild encoder each time),
    max abs embedding diff.
(b) PERMUTATION-EQUIVARIANCE: randomly relabel node ids (permute x rows +
    remap edge_index consistently), re-embed, invert the permutation, and
    check emb-order ball ndcg@10 is identical to the unpermuted run.
    5 graphs x 3 permutations.

Verdict ROBUST if deterministic AND permutation-invariant (both diffs < 1e-4).

Run from kettle-graph-reasoner/:
    py -m scripts.stress_determinism \
        --corpus src/data/corpus/real_domain_eval_all6 \
        --out runs/stress_determinism
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


def _embed(enc, g, device):
    with torch.no_grad():
        return enc(
            g["x"].to(device), g["edge_index"].to(device),
            g["edge_type"].to(device), g["edge_descriptor"].to(device),
            node_descriptor=g["node_descriptor"].to(device),
        ).node_embeddings


def _ball_ndcgs(emb, g, z, c):
    """emb-order ball ndcg@10 for every eligible task in the graph."""
    n = emb.shape[0]
    adj = [[] for _ in range(n)]
    ei = g["edge_index"].numpy()
    for s, t in zip(ei[0], ei[1]):
        adj[int(s)].append(int(t))
        adj[int(t)].append(int(s))
    out = {}
    for i in range(int(z["n_tasks"])):
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
            sc = score_from_embeddings(emb[rows_t], emb[anchor], c=c)
            out[i] = ndcg_at_k(sc, lab_b, 10)
    return out


def _permute_graph(z, perm):
    """Return a new _build_graph_tensors dict with nodes relabeled by perm.

    perm maps NEW row -> OLD row (x_new[i] = x_old[perm[i]]).
    inv maps OLD -> NEW.  edge_index remapped OLD->NEW.
    """
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    d = dict(z)  # shallow copy of npz arrays
    d["x"] = z["x"][perm]
    if "node_descriptor" in z:
        d["node_descriptor"] = z["node_descriptor"][perm]
    ei = z["edge_index"].copy()
    ei = inv[ei]  # remap both endpoints OLD->NEW
    d["edge_index"] = ei
    # per-node task fields must move with the nodes too
    n_tasks = int(z["n_tasks"])
    for i in range(n_tasks):
        ak = f"task_{i}_anchor_row"
        lk = f"task_{i}_labels"
        if ak in z:
            d[ak] = np.asarray(int(inv[int(z[ak])]))       # OLD row -> NEW row
        if lk in z:
            d[lk] = z[lk][perm]                            # labels_new[i]=old[perm[i]]
    return d, inv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/real_domain_eval_all6")
    ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--out", default="runs/stress_determinism")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-graphs", type=int, default=5)
    ap.add_argument("--n-perms", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(0)

    files = sorted(Path(args.corpus).glob("graph_*.npz"))[:args.n_graphs]
    print(f"{len(files)} graphs, {args.n_perms} perms each")

    det_diffs = []           # (a) repeat determinism max abs diff
    perm_emb_diffs = []      # invariance of embeddings (after inverting perm)
    perm_ndcg_diffs = []     # |delta ndcg@10| under permutation
    per_graph = []

    enc = None
    for f in files:
        z = np.load(f, allow_pickle=True)
        g = _build_graph_tensors(z)
        if enc is None:
            enc, _ = _build_encoder(Path(args.ckpt), g, device)
        c = enc.c

        # (a) determinism: rebuild encoder, embed twice
        enc2, _ = _build_encoder(Path(args.ckpt), g, device)
        emb_a = _embed(enc, g, device)
        emb_b = _embed(enc2, g, device)
        det = float((emb_a - emb_b).abs().max())
        det_diffs.append(det)

        base_ndcg = _ball_ndcgs(emb_a, g, z, c)
        n = emb_a.shape[0]

        for p in range(args.n_perms):
            perm = rng.permutation(n)
            zp, inv = _permute_graph(z, perm)
            gp = _build_graph_tensors(zp)
            emb_p = _embed(enc, gp, device)
            # invert: emb_p is indexed by NEW row; map back to OLD via perm
            # OLD row r lives at NEW row inv[r], so emb_back[r] = emb_p[inv[r]]
            emb_back = emb_p[torch.from_numpy(inv).long()]
            ediff = float((emb_back - emb_a).abs().max())
            perm_emb_diffs.append(ediff)
            # ndcg on permuted graph, using permuted z
            perm_ndcg = _ball_ndcgs(emb_p, gp, zp, c)
            # compare per-task ndcg (task indices align; z copied)
            common = set(base_ndcg) & set(perm_ndcg)
            for i in common:
                perm_ndcg_diffs.append(abs(base_ndcg[i] - perm_ndcg[i]))
            per_graph.append({
                "file": f.name, "perm": p, "n": int(n),
                "emb_invert_diff": ediff,
                "n_tasks_compared": len(common),
                "max_ndcg_diff": max(
                    (abs(base_ndcg[i] - perm_ndcg[i]) for i in common),
                    default=0.0),
            })
        print(f"{f.name} n={n} det={det:.2e} "
              f"perm_emb_max={max(perm_emb_diffs[-args.n_perms:]):.2e}")

    report = {
        "config": vars(args),
        "max_det_diff": max(det_diffs) if det_diffs else float("nan"),
        "mean_det_diff": float(np.mean(det_diffs)) if det_diffs else float("nan"),
        "max_perm_emb_invert_diff": max(perm_emb_diffs) if perm_emb_diffs else float("nan"),
        "max_perm_ndcg_diff": max(perm_ndcg_diffs) if perm_ndcg_diffs else 0.0,
        "n_ndcg_compared": len(perm_ndcg_diffs),
        "per_graph": per_graph,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print("\n=== SUMMARY ===")
    print(f"max determinism diff (repeat):    {report['max_det_diff']:.3e}")
    print(f"max perm emb-invert diff:         {report['max_perm_emb_invert_diff']:.3e}")
    print(f"max |delta ndcg@10| under perm:   {report['max_perm_ndcg_diff']:.3e}")
    print(f"ndcg pairs compared:              {report['n_ndcg_compared']}")
    print(f"report: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

r"""Adversarial re-run of stress_determinism on a DISJOINT slice + different seed.

Independent slice: graphs indexed [start:start+n] (default 100..160), disjoint
from the original probe's first 5. Different RNG seed (default 1234).

Adds a tightened control: a "shuffle control" that permutes the CANDIDATE ROWS'
scores before ndcg (should MOVE ndcg) to prove the equivariance ndcg-diff==0 is
non-trivial, i.e. that the metric is actually sensitive.

Run from kettle-graph-reasoner/:
    py -m scripts.stress_determinism_refute --start 100 --n-graphs 8 --seed 1234
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


def _ball_ndcgs(emb, g, z, c, shuffle_rng=None):
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
            if shuffle_rng is not None:
                perm = shuffle_rng.permutation(len(ball))
                sc = sc[torch.from_numpy(perm).long()]
            out[i] = ndcg_at_k(sc, lab_b, 10)
    return out


def _permute_graph(z, perm):
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    d = dict(z)
    d["x"] = z["x"][perm]
    if "node_descriptor" in z:
        d["node_descriptor"] = z["node_descriptor"][perm]
    ei = z["edge_index"].copy()
    ei = inv[ei]
    d["edge_index"] = ei
    n_tasks = int(z["n_tasks"])
    for i in range(n_tasks):
        ak = f"task_{i}_anchor_row"
        lk = f"task_{i}_labels"
        if ak in z:
            d[ak] = np.asarray(int(inv[int(z[ak])]))
        if lk in z:
            d[lk] = z[lk][perm]
    return d, inv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/real_domain_eval_all6")
    ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--out", default="runs/stress_determinism_refute")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--n-graphs", type=int, default=8)
    ap.add_argument("--n-perms", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    all_files = sorted(Path(args.corpus).glob("graph_*.npz"))
    files = all_files[args.start:args.start + args.n_graphs]
    print(f"slice [{args.start}:{args.start+args.n_graphs}] -> {len(files)} graphs, "
          f"seed={args.seed}, {args.n_perms} perms each")

    det_diffs = []
    perm_emb_diffs = []
    perm_ndcg_diffs = []
    shuffle_ctrl_diffs = []   # control: |ndcg(shuffled scores) - ndcg(base)|
    per_graph = []

    enc = None
    for f in files:
        z = np.load(f, allow_pickle=True)
        g = _build_graph_tensors(z)
        if enc is None:
            enc, _ = _build_encoder(Path(args.ckpt), g, device)
        c = enc.c

        enc2, _ = _build_encoder(Path(args.ckpt), g, device)
        emb_a = _embed(enc, g, device)
        emb_b = _embed(enc2, g, device)
        det = float((emb_a - emb_b).abs().max())
        det_diffs.append(det)

        base_ndcg = _ball_ndcgs(emb_a, g, z, c)
        # control: shuffle candidate scores -> ndcg SHOULD change
        ctrl_ndcg = _ball_ndcgs(emb_a, g, z, c, shuffle_rng=np.random.default_rng(args.seed + 7))
        for i in set(base_ndcg) & set(ctrl_ndcg):
            shuffle_ctrl_diffs.append(abs(base_ndcg[i] - ctrl_ndcg[i]))

        n = emb_a.shape[0]
        for p in range(args.n_perms):
            perm = rng.permutation(n)
            zp, inv = _permute_graph(z, perm)
            gp = _build_graph_tensors(zp)
            emb_p = _embed(enc, gp, device)
            emb_back = emb_p[torch.from_numpy(inv).long()]
            ediff = float((emb_back - emb_a).abs().max())
            perm_emb_diffs.append(ediff)
            perm_ndcg = _ball_ndcgs(emb_p, gp, zp, c)
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
        "files": [f.name for f in files],
        "max_det_diff": max(det_diffs) if det_diffs else float("nan"),
        "mean_det_diff": float(np.mean(det_diffs)) if det_diffs else float("nan"),
        "max_perm_emb_invert_diff": max(perm_emb_diffs) if perm_emb_diffs else float("nan"),
        "mean_perm_emb_invert_diff": float(np.mean(perm_emb_diffs)) if perm_emb_diffs else float("nan"),
        "max_perm_ndcg_diff": max(perm_ndcg_diffs) if perm_ndcg_diffs else 0.0,
        "n_ndcg_compared": len(perm_ndcg_diffs),
        "ctrl_shuffle_mean_ndcg_diff": float(np.mean(shuffle_ctrl_diffs)) if shuffle_ctrl_diffs else 0.0,
        "ctrl_shuffle_max_ndcg_diff": max(shuffle_ctrl_diffs) if shuffle_ctrl_diffs else 0.0,
        "ctrl_shuffle_n": len(shuffle_ctrl_diffs),
        "per_graph": per_graph,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print("\n=== SUMMARY (refute) ===")
    print(f"max determinism diff (repeat):    {report['max_det_diff']:.3e}")
    print(f"max perm emb-invert diff:         {report['max_perm_emb_invert_diff']:.3e}")
    print(f"max |delta ndcg@10| under perm:   {report['max_perm_ndcg_diff']:.3e}")
    print(f"ndcg pairs compared:              {report['n_ndcg_compared']}")
    print(f"CONTROL shuffle mean ndcg diff:   {report['ctrl_shuffle_mean_ndcg_diff']:.3e} "
          f"(n={report['ctrl_shuffle_n']}) -- should be >> 0")
    print(f"report: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

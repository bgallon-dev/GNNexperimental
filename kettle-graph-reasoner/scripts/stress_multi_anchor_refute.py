r"""Adversarial re-run of key=multi_anchor on a DISJOINT slice + tightened controls.

Prior claim (all 200 graphs, 30 compound cases): oracle 2nd-anchor min-dist
union lifts ndcg@10 from single=0.231 to min=0.702 (delta +0.471), random 0.23.

REFUTE plan:
  - Re-measure on graphs indexed [start:end) (default 100..200), disjoint from
    the first-100 half. Also emit the first-half (0..100) numbers for contrast.
  - Tightened controls on the SECOND anchor a2:
      a2_oracle_far : relevant node farthest in hops (the ORIGINAL arm)
      a2_rand_rel   : a RANDOM relevant node (still oracle-ish, tests "farthest")
      a2_rand_ball  : a RANDOM ball node (NON-oracle 2nd seed)
    For each we report min-dist-union ndcg@10. If a2_rand_ball ALSO lifts single
    substantially, the "capability" is a trivial two-seed artifact, not
    multi-anchor composition.
  - Different RNG seed offset for the random floor.

Run from kettle-graph-reasoner/:
    PYTHONIOENCODING=utf-8 py -m scripts.stress_multi_anchor_refute \
        --corpus src/data/corpus/real_domain_eval_all6 \
        --start 100 --end 200 --out runs/stress_multi_anchor_refute
"""

from __future__ import annotations

import argparse
import json
import random
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
    diff = P.mobius_add(-x, y, c)
    half = P.mobius_scalar_mul(0.5, diff, c)
    return P.mobius_add(x, half, c)


def _eval_slice(files, ckpt, device, rng_off):
    enc = None
    rows = []
    rng = random.Random(1234 + rng_off)
    for f in files:
        z = np.load(f, allow_pickle=True)
        g = _build_graph_tensors(z)
        if enc is None:
            enc, _ = _build_encoder(Path(ckpt), g, device)
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
            rel = [r for r in range(n)
                   if r != anchor and labels[r] >= 0.5 and np.isfinite(d[r])]
            if not rel:
                continue
            a2 = max(rel, key=lambda r: d[r])          # oracle farthest
            a2_rr = rng.choice(rel)                     # random relevant
            # ball excludes primary anchor and the oracle-far a2 (match original)
            ball = [r for r in range(n)
                    if r != anchor and r != a2 and d[r] <= mh
                    and np.isfinite(d[r])]
            lab_b = torch.from_numpy(labels[ball]).float()
            if len(ball) < 5 or float(lab_b.max()) <= 0 \
                    or float(lab_b.min()) >= float(lab_b.max()):
                continue
            # random NON-oracle 2nd seed drawn from the ball itself
            a2_rb = rng.choice(ball)
            rows_t = torch.tensor(ball, dtype=torch.long)
            e_ball = emb[rows_t]
            a1p = emb[anchor]
            s1 = score_from_embeddings(e_ball, a1p, c=enc.c)
            s2_far = score_from_embeddings(e_ball, emb[a2], c=enc.c)
            s2_rr = score_from_embeddings(e_ball, emb[a2_rr], c=enc.c)
            s2_rb = score_from_embeddings(e_ball, emb[a2_rb], c=enc.c)
            s_min_far = torch.maximum(s1, s2_far)
            s_min_rr = torch.maximum(s1, s2_rr)
            s_min_rb = torch.maximum(s1, s2_rb)
            mid = _midpoint(a1p, emb[a2], enc.c)
            s_mid = score_from_embeddings(e_ball, mid, c=enc.c)
            rnd = torch.randn(
                len(ball),
                generator=torch.Generator().manual_seed(
                    (hash((f.name, i, rng_off)) & 0x7FFFFFFF)))
            rows.append({
                "family": fam,
                "n_ball": len(ball),
                "a2_hops": float(d[a2]),
                "single_ndcg10": ndcg_at_k(s1, lab_b, 10),
                "min_oracle_far_ndcg10": ndcg_at_k(s_min_far, lab_b, 10),
                "min_rand_rel_ndcg10": ndcg_at_k(s_min_rr, lab_b, 10),
                "min_rand_ball_ndcg10": ndcg_at_k(s_min_rb, lab_b, 10),
                "midpoint_ndcg10": ndcg_at_k(s_mid, lab_b, 10),
                "random_ndcg10": ndcg_at_k(rnd, lab_b, 10),
                "oracle_ndcg10": ndcg_at_k(lab_b.clone(), lab_b, 10),
            })
    return rows


def _summ(rows):
    keys = ["single", "min_oracle_far", "min_rand_rel", "min_rand_ball",
            "midpoint", "random", "oracle"]
    out = {"n": len(rows)}
    for k in keys:
        v = [r[f"{k}_ndcg10"] for r in rows]
        out[k] = sum(v) / len(v) if v else float("nan")
    out["a2_hops_mean"] = (sum(r["a2_hops"] for r in rows) / len(rows)
                           if rows else float("nan"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/real_domain_eval_all6")
    ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--out", default="runs/stress_multi_anchor_refute")
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--end", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    all_files = sorted(Path(args.corpus).glob("graph_*.npz"))

    slices = {
        "disjoint_100_200": all_files[args.start:args.end],
        "first_0_100": all_files[0:100],
    }
    report = {"config": vars(args), "slices": {}}
    for name, files in slices.items():
        print(f"\n=== slice {name}: {len(files)} graphs ===")
        rows = _eval_slice(files, args.ckpt, device, rng_off=hash(name) & 0xFF)
        s = _summ(rows)
        report["slices"][name] = s
        print(f"n={s['n']}  a2_hops_mean={s['a2_hops_mean']:.2f}")
        for k in ["single", "min_oracle_far", "min_rand_rel",
                  "min_rand_ball", "midpoint", "random", "oracle"]:
            print(f"  {k:<16} {s[k]:.3f}")
        s["delta_min_oracle_vs_single"] = s["min_oracle_far"] - s["single"]
        s["delta_min_randball_vs_single"] = s["min_rand_ball"] - s["single"]

    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

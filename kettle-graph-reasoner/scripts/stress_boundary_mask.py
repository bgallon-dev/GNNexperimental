r"""Stress probe (key=boundary_mask): subgraph boundary-MASK output.

Ball-order gives a ranking; the product deliverable is a pruned SET.
For each task on the first N all6 graphs:
  ball = BFS nodes within max_hops of anchor (undirected).
  For each candidate, distance = hyperbolic d_c(emb[anchor], emb[cand]).
  Sweep radius as a quantile q in {0.1..0.9} of the ball distance dist:
    predicted subgraph = ball nodes with distance <= quantile(q).
  precision / recall / F1 vs relevant set (label >= 0.5) within the ball.

Load-bearing question: does a SINGLE GLOBAL quantile work?
  - best per-graph F1 (oracle per-task threshold)  -> upper bound
  - F1 at the best GLOBAL q                          -> the deliverable
  - size-matched RANDOM subset F1                    -> control/floor
Verdict CAPABILITY_CONFIRMED if a global threshold gives F1 well above
the size-matched random baseline.

Run from kettle-graph-reasoner/:
    py -m scripts.stress_boundary_mask \
        --corpus src/data/corpus/real_domain_eval_all6 \
        --n-graphs 60 --out runs/stress_boundary_mask
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

FAMILY = {0: "provenance", 1: "entity_res", 2: "temporal",
          3: "multihop", 4: "subgraph", 5: "compound"}

QGRID = [round(0.1 * k, 1) for k in range(1, 10)]  # 0.1 .. 0.9


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


def _prf(pred_mask, rel_mask):
    tp = float((pred_mask & rel_mask).sum())
    p = tp / float(pred_mask.sum()) if pred_mask.sum() > 0 else 0.0
    r = tp / float(rel_mask.sum()) if rel_mask.sum() > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/real_domain_eval_all6")
    ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--n-graphs", type=int, default=60)
    ap.add_argument("--rand-samples", type=int, default=20)
    ap.add_argument("--out", default="runs/stress_boundary_mask")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(0)

    enc = None
    # per-task records: distances (np), rel_mask (np bool), family
    tasks = []
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
            fam = FAMILY.get(int(z[f"task_{i}_type"]), "?")
            anchor = int(z[f"task_{i}_anchor_row"])
            labels = z[f"task_{i}_labels"]
            mh = float(z[f"task_{i}_max_hops"])
            d = _bfs(adj, anchor, n)
            ball = [r for r in range(n)
                    if r != anchor and d[r] <= mh and np.isfinite(d[r])]
            if len(ball) < 5:
                continue
            rel = (labels[ball] >= 0.5)
            # need both relevant and irrelevant in ball to be non-trivial
            if rel.sum() == 0 or rel.sum() == len(ball):
                continue
            rows_t = torch.tensor(ball, dtype=torch.long)
            dist = (-score_from_embeddings(
                emb[rows_t], emb[anchor], c=enc.c)).cpu().numpy()
            tasks.append({"family": fam, "dist": dist,
                          "rel": rel.astype(bool), "n_ball": len(ball),
                          "n_rel": int(rel.sum())})

    print(f"{len(tasks)} usable tasks")

    # Per q: mean F1 across tasks (each task thresholds by its own quantile).
    # Also record predicted set size per task for size-matched random control.
    q_meanf1 = {}
    q_meanprec = {}
    q_randf1 = {}     # size-matched random F1 at this same q
    q_pred_sizes = {}
    per_task_bestf1 = [0.0] * len(tasks)
    for q in QGRID:
        f1s, precs, sizes, rf1s = [], [], [], []
        for ti, t in enumerate(tasks):
            thr = np.quantile(t["dist"], q)
            pred = t["dist"] <= thr
            p, _, f1 = _prf(pred, t["rel"])
            f1s.append(f1)
            precs.append(p)
            k = int(pred.sum())
            sizes.append(k)
            if f1 > per_task_bestf1[ti]:
                per_task_bestf1[ti] = f1
            # size-matched random control at this operating point
            B, rel = t["n_ball"], t["rel"]
            acc = []
            for _ in range(args.rand_samples):
                idx = rng.choice(B, size=min(k, B), replace=False)
                pm = np.zeros(B, dtype=bool)
                pm[idx] = True
                _, _, rf1 = _prf(pm, rel)
                acc.append(rf1)
            rf1s.append(float(np.mean(acc)))
        q_meanf1[q] = float(np.mean(f1s))
        q_meanprec[q] = float(np.mean(precs))
        q_randf1[q] = float(np.mean(rf1s))
        q_pred_sizes[q] = sizes

    best_q = max(q_meanf1, key=q_meanf1.get)
    global_f1 = q_meanf1[best_q]
    oracle_f1 = float(np.mean(per_task_bestf1))

    # Size-matched random control: at best_q, predicted size k per task;
    # sample k random ball nodes, average F1 over rand_samples.
    rand_f1s = []
    for t, k in zip(tasks, q_pred_sizes[best_q]):
        B = t["n_ball"]
        rel = t["rel"]
        acc = []
        for _ in range(args.rand_samples):
            idx = rng.choice(B, size=min(k, B), replace=False)
            pred = np.zeros(B, dtype=bool)
            pred[idx] = True
            _, _, f1 = _prf(pred, rel)
            acc.append(f1)
        rand_f1s.append(float(np.mean(acc)))
    rand_f1 = float(np.mean(rand_f1s))

    # precision/recall at best_q for context
    ps, rs = [], []
    for t in tasks:
        thr = np.quantile(t["dist"], best_q)
        pred = t["dist"] <= thr
        p, r, _ = _prf(pred, t["rel"])
        ps.append(p)
        rs.append(r)
    global_prec = float(np.mean(ps))
    global_rec = float(np.mean(rs))

    # base rate: mean relevant fraction in ball (random precision expectation)
    base_rate = float(np.mean([t["n_rel"] / t["n_ball"] for t in tasks]))

    # per-family global-q F1 (using the single global best_q)
    fam_f1 = {}
    for fam in sorted({t["family"] for t in tasks}):
        sub = [t for t in tasks if t["family"] == fam]
        vals = []
        for t in sub:
            thr = np.quantile(t["dist"], best_q)
            pred = t["dist"] <= thr
            _, _, f1 = _prf(pred, t["rel"])
            vals.append(f1)
        fam_f1[fam] = {"n": len(sub), "global_f1": float(np.mean(vals))}

    # best global q by F1-LIFT over its own size-matched random control
    q_lift = {q: q_meanf1[q] - q_randf1[q] for q in QGRID}
    best_lift_q = max(q_lift, key=q_lift.get)

    report = {
        "config": vars(args),
        "n_tasks": len(tasks),
        "q_meanf1": q_meanf1,
        "q_meanprec": q_meanprec,
        "q_randf1": q_randf1,
        "q_lift": q_lift,
        "best_lift_q": best_lift_q,
        "best_lift_value": q_lift[best_lift_q],
        "best_global_q": best_q,
        "global_f1": global_f1,
        "oracle_pergraph_f1": oracle_f1,
        "random_sizematched_f1": rand_f1,
        "global_precision": global_prec,
        "global_recall": global_rec,
        "ball_base_rate": base_rate,
        "family_global_f1": fam_f1,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))

    print("\n=== q sweep: model F1 vs size-matched random F1 (lift) ===")
    for q in QGRID:
        print(f"  q={q}: F1={q_meanf1[q]:.3f}  prec={q_meanprec[q]:.3f}  "
              f"randF1={q_randf1[q]:.3f}  lift={q_lift[q]:+.3f}")
    print(f"best q by lift = {best_lift_q}  lift={q_lift[best_lift_q]:+.3f}")
    print(f"\nbest global q      = {best_q}")
    print(f"global F1          = {global_f1:.3f}")
    print(f"oracle per-graph F1= {oracle_f1:.3f}")
    print(f"random sizematch F1= {rand_f1:.3f}")
    print(f"global precision   = {global_prec:.3f}  (ball base rate {base_rate:.3f})")
    print(f"global recall      = {global_rec:.3f}")
    print("\nper-family global-q F1:")
    for fam, v in fam_f1.items():
        print(f"  {fam:<12} n={v['n']:<4} F1={v['global_f1']:.3f}")
    print(f"\nreport: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

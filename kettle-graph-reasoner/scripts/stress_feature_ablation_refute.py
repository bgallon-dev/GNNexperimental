r"""REFUTATION of feature_ablation: re-run saliency on a DISJOINT slice.

Prior claim (first 50 all6 graphs, n=424):
  baseline ndcg10 0.8891; zeroing random_id (x[:,24:32]) drops to 0.6397
  (drop 0.2494), all other blocks drop <0.006. => encoder relies mainly on
  the per-node random identity code.

This probe re-runs the SAME measurement on graphs indexed 100..159
(disjoint from the first 50), with a different random-floor seed offset,
and adds a tightened control:
  rand8_ctrl : zero 8 RANDOM columns drawn from [0:24] (the non-random_id
               region), same COUNT of dims as random_id (8), to test whether
               the drop is about the random_id CONTENT or merely about
               zeroing 8 feature dims.

Run from kettle-graph-reasoner/:
    py -m scripts.stress_feature_ablation_refute \
        --corpus src/data/corpus/real_domain_eval_all6 \
        --out runs/stress_feature_ablation_refute
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

FAMILY = {0: "provenance", 1: "entity_res", 2: "temporal",
          3: "multihop", 4: "subgraph", 5: "compound"}

BLOCKS = {
    "type": (0, 16),
    "degree": (16, 19),
    "clustering": (19, 20),
    "depth": (20, 21),
    "temporal": (21, 24),
    "random_id": (24, 32),
}


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/real_domain_eval_all6")
    ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--out", default="runs/stress_feature_ablation_refute")
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--n-graphs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # tightened control: fixed random 8 columns from [0:24]
    ctrl_rng = np.random.RandomState(args.seed)
    ctrl_cols = np.sort(ctrl_rng.choice(np.arange(0, 24), size=8, replace=False))

    variants = (["baseline"] + list(BLOCKS.keys())
                + ["rand8_ctrl", "random"])
    acc = {v: [] for v in variants}
    acc_fam = {v: {} for v in variants}

    enc = None
    all_files = sorted(Path(args.corpus).glob("graph_*.npz"))
    files = all_files[args.start: args.start + args.n_graphs]
    print(f"slice {args.start}..{args.start + len(files)} "
          f"({len(files)} graphs); ctrl_cols={ctrl_cols.tolist()}")
    n_cases = 0
    for f in files:
        z = np.load(f, allow_pickle=True)
        g = _build_graph_tensors(z)
        if enc is None:
            enc, _ = _build_encoder(Path(args.ckpt), g, device)

        embs = {"baseline": _embed(enc, g, device)}
        x_orig = g["x"]
        for name, (lo, hi) in BLOCKS.items():
            x_ab = x_orig.clone()
            x_ab[:, lo:hi] = 0.0
            g_ab = dict(g); g_ab["x"] = x_ab
            embs[name] = _embed(enc, g_ab, device)
        # tightened control: zero the 8 fixed random non-random_id columns
        x_c = x_orig.clone()
        x_c[:, ctrl_cols] = 0.0
        g_c = dict(g); g_c["x"] = x_c
        embs["rand8_ctrl"] = _embed(enc, g_c, device)
        g["x"] = x_orig

        n = embs["baseline"].shape[0]
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
            if not (len(ball) >= 5 and float(lab_b.max()) > 0
                    and float(lab_b.min()) < float(lab_b.max())):
                continue
            rows_t = torch.tensor(ball, dtype=torch.long)
            n_cases += 1
            for v in ["baseline"] + list(BLOCKS.keys()) + ["rand8_ctrl"]:
                emb = embs[v]
                sc = score_from_embeddings(emb[rows_t], emb[anchor], c=enc.c)
                nd = ndcg_at_k(sc, lab_b, 10)
                acc[v].append(nd)
                acc_fam[v].setdefault(fam, []).append(nd)
            rnd = torch.randn(
                len(ball),
                generator=torch.Generator().manual_seed(
                    (hash((f.name, i)) ^ args.seed) & 0x7FFFFFFF))
            nd_r = ndcg_at_k(rnd, lab_b, 10)
            acc["random"].append(nd_r)
            acc_fam["random"].setdefault(fam, []).append(nd_r)

    def _avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    base = _avg(acc["baseline"])
    report = {"config": vars(args), "ctrl_cols": ctrl_cols.tolist(),
              "n_cases": n_cases, "overall": {}, "drop_vs_baseline": {},
              "by_family": {}}
    print(f"\n=== REFUTE FEATURE ABLATION ndcg@10 (n={n_cases}) ===")
    print(f"{'variant':<12} {'ndcg10':>8} {'drop':>8}")
    for v in variants:
        mv = _avg(acc[v])
        report["overall"][v] = mv
        drop = base - mv
        if v not in ("baseline", "random"):
            report["drop_vs_baseline"][v] = drop
        print(f"{v:<12} {mv:>8.4f} "
              + (f"{drop:>8.4f}" if v not in ("baseline", "random") else "     -"))

    ranked = sorted(report["drop_vs_baseline"].items(),
                    key=lambda kv: kv[1], reverse=True)
    report["importance_ranking"] = [k for k, _ in ranked]
    print("\nimportance (largest drop first):",
          " > ".join(report["importance_ranking"]))

    for fam in sorted(acc_fam["baseline"].keys()):
        report["by_family"][fam] = {v: _avg(acc_fam[v].get(fam, []))
                                    for v in variants}

    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

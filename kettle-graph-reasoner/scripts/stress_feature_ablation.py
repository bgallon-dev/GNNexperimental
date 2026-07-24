r"""Feature-block saliency: which x blocks does the frozen encoder rely on?

On first 50 all6 graphs, for each candidate x feature block, ZERO that block
in x, re-embed, recompute emb-order ball ndcg@10. Report drop vs baseline
per block + random floor.

Blocks:
  type       x[:, 0:16]   node-type one-hot
  degree     x[:, 16:19]  degrees(log1p)
  clustering x[:, 19:20]  clustering
  depth      x[:, 20:21]  depth/5
  temporal   x[:, 21:24]  temporal (zeros on code, real on archival)
  random_id  x[:, 24:32]  per-node random identity code

Run from kettle-graph-reasoner/:
    py -m scripts.stress_feature_ablation \
        --corpus src/data/corpus/real_domain_eval_all6 \
        --out runs/stress_feature_ablation
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
    ap.add_argument("--out", default="runs/stress_feature_ablation")
    ap.add_argument("--n-graphs", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    variants = ["baseline"] + list(BLOCKS.keys()) + ["random"]
    # per-variant accumulator: list of ndcg per case, and per-family
    acc = {v: [] for v in variants}
    acc_fam = {v: {} for v in variants}

    enc = None
    files = sorted(Path(args.corpus).glob("graph_*.npz"))[: args.n_graphs]
    print(f"{len(files)} graphs")
    n_cases = 0
    for f in files:
        z = np.load(f, allow_pickle=True)
        g = _build_graph_tensors(z)
        if enc is None:
            enc, _ = _build_encoder(Path(args.ckpt), g, device)

        # precompute embeddings for baseline + each ablated block
        embs = {"baseline": _embed(enc, g, device)}
        x_orig = g["x"]
        for name, (lo, hi) in BLOCKS.items():
            x_ab = x_orig.clone()
            x_ab[:, lo:hi] = 0.0
            g_ab = dict(g)
            g_ab["x"] = x_ab
            embs[name] = _embed(enc, g_ab, device)
        # restore
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
            # emb-order ndcg@10 per variant
            for v in ["baseline"] + list(BLOCKS.keys()):
                emb = embs[v]
                sc = score_from_embeddings(emb[rows_t], emb[anchor], c=enc.c)
                nd = ndcg_at_k(sc, lab_b, 10)
                acc[v].append(nd)
                acc_fam[v].setdefault(fam, []).append(nd)
            # random floor
            rnd = torch.randn(
                len(ball),
                generator=torch.Generator().manual_seed(
                    hash((f.name, i)) & 0x7FFFFFFF))
            nd_r = ndcg_at_k(rnd, lab_b, 10)
            acc["random"].append(nd_r)
            acc_fam["random"].setdefault(fam, []).append(nd_r)

    def _avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    base = _avg(acc["baseline"])
    report = {"config": vars(args), "n_cases": n_cases,
              "overall": {}, "drop_vs_baseline": {}, "by_family": {}}
    print(f"\n=== FEATURE ABLATION emb-order ball ndcg@10 (n={n_cases}) ===")
    print(f"{'variant':<12} {'ndcg10':>8} {'drop':>8}")
    for v in variants:
        mv = _avg(acc[v])
        report["overall"][v] = mv
        drop = base - mv
        if v not in ("baseline", "random"):
            report["drop_vs_baseline"][v] = drop
        print(f"{v:<12} {mv:>8.4f} "
              + (f"{drop:>8.4f}" if v not in ("baseline", "random") else "     -"))

    # importance ranking (largest drop first) over the 6 blocks
    ranked = sorted(report["drop_vs_baseline"].items(),
                    key=lambda kv: kv[1], reverse=True)
    report["importance_ranking"] = [k for k, _ in ranked]
    print("\nimportance (largest drop first):",
          " > ".join(report["importance_ranking"]))

    # per family
    for fam in sorted(acc_fam["baseline"].keys()):
        report["by_family"][fam] = {v: _avg(acc_fam[v].get(fam, []))
                                    for v in variants}

    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

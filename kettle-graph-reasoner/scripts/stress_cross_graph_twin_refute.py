r"""Adversarial refutation of key=cross_graph_twin.

Re-run the SAME measurement (top-1 cross-graph nearest-embedding neighbor
type-match lift) on a DISJOINT slice of all6 graphs (start offset != 0) and
with different seeds. Add a tightened PERMUTATION control: shuffle the pool
type labels; lift must collapse to ~1 or the metric is broken.

Also add a RAW-FEATURE control: nearest neighbor computed in raw x-space
(which contains the type one-hot as input) -> tells us whether the embedding
lift is merely inherited from a shared input feature (mechanism check, not a
refutation of replication).

Run from kettle-graph-reasoner/:
    py -m scripts.stress_cross_graph_twin_refute \
        --start 100 --n-graphs 20 --seed 5 \
        --out runs/stress_cross_graph_twin_refute
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


def measure(pool_emb, pool_type, pool_gid, pool_x, qidx, c, seed):
    rng2 = np.random.default_rng(seed + 1)
    match = 0
    rawmatch = 0
    base_rates = []
    for qi in qidx:
        q_type = int(pool_type[qi])
        q_gid = int(pool_gid[qi])
        cross_mask = pool_gid != q_gid
        cross_types = pool_type[cross_mask]
        base = float((cross_types == q_type).mean())
        base_rates.append(base)
        sc = score_from_embeddings(pool_emb[cross_mask], pool_emb[qi], c=c)
        nn_local = int(torch.argmax(sc).item())
        match += int(int(cross_types[nn_local]) == q_type)
        # raw-feature nearest neighbor (euclidean in x-space) mechanism control
        cross_x = pool_x[cross_mask]
        dx = np.linalg.norm(cross_x - pool_x[qi], axis=1)
        rawmatch += int(int(cross_types[int(dx.argmin())]) == q_type)
    n = len(qidx)
    obs = match / n
    raw_obs = rawmatch / n
    mean_base = float(np.mean(base_rates))
    # random cross-graph neighbor control
    rand_match = 0
    for qi in qidx:
        q_gid = int(pool_gid[qi])
        q_type = int(pool_type[qi])
        cross_idx = np.where(pool_gid != q_gid)[0]
        pick = cross_idx[rng2.integers(len(cross_idx))]
        rand_match += int(int(pool_type[pick]) == q_type)
    rand_obs = rand_match / n
    return {
        "obs": obs, "mean_base": mean_base,
        "lift": obs / mean_base if mean_base > 0 else float("nan"),
        "raw_obs": raw_obs,
        "raw_lift": raw_obs / mean_base if mean_base > 0 else float("nan"),
        "rand_obs": rand_obs,
        "rand_lift": rand_obs / mean_base if mean_base > 0 else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/real_domain_eval_all6")
    ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--out", default="runs/stress_cross_graph_twin_refute")
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--n-graphs", type=int, default=20)
    ap.add_argument("--n-query", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    allfiles = sorted(Path(args.corpus).glob("graph_*.npz"))
    files = allfiles[args.start:args.start + args.n_graphs]
    print(f"{len(files)} graphs, index {args.start}..{args.start + len(files) - 1}")

    enc = None
    all_emb, all_type, all_gid, all_x = [], [], [], []
    for gi, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        g = _build_graph_tensors(z)
        if enc is None:
            enc, _ = _build_encoder(Path(args.ckpt), g, device)
        with torch.no_grad():
            emb = enc(
                g["x"].to(device), g["edge_index"].to(device),
                g["edge_type"].to(device), g["edge_descriptor"].to(device),
                node_descriptor=g["node_descriptor"].to(device),
            ).node_embeddings.cpu()
        x = z["x"]
        all_emb.append(emb)
        all_type.append(x[:, :16].argmax(axis=1).astype(np.int64))
        all_gid.append(np.full(emb.shape[0], gi, dtype=np.int64))
        all_x.append(x.astype(np.float32))

    pool_emb = torch.cat(all_emb, dim=0)
    pool_type = np.concatenate(all_type)
    pool_gid = np.concatenate(all_gid)
    pool_x = np.concatenate(all_x, axis=0)
    M = pool_emb.shape[0]
    c = enc.c
    print(f"pool: {M} nodes, {len(np.unique(pool_type))} distinct top-types")

    results = {"config": vars(args), "pool_nodes": int(M),
               "distinct_types": int(len(np.unique(pool_type))), "seeds": {}}
    for seed in (args.seed, args.seed + 100):
        rng = np.random.default_rng(seed)
        nq = min(args.n_query, M)
        qidx = rng.choice(M, size=nq, replace=False)
        r = measure(pool_emb, pool_type, pool_gid, pool_x, qidx, c, seed)
        # permutation control: shuffle type labels across the pool
        perm = rng.permutation(M)
        r_perm = measure(pool_emb, pool_type[perm], pool_gid, pool_x, qidx, c, seed)
        results["seeds"][str(seed)] = {
            "n_query": int(nq),
            "obs_match_frac": round(r["obs"], 4),
            "mean_base_rate": round(r["mean_base"], 4),
            "lift": round(r["lift"], 4),
            "raw_x_lift": round(r["raw_lift"], 4),
            "control_random_lift": round(r["rand_lift"], 4),
            "control_permuted_type_lift": round(r_perm["lift"], 4),
        }
        print(f"\n--- seed {seed} (n_query {nq}) ---")
        for k, v in results["seeds"][str(seed)].items():
            print(f"  {k:<28}: {v}")

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nreport: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

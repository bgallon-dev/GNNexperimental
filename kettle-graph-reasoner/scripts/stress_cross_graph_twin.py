r"""Stress probe: cross-graph embedding comparability (key=cross_graph_twin).

Expected NEGATIVE. v3 trains intra-graph only, so node embeddings from
different graphs live in per-graph frames. A query node's nearest embedding
neighbor drawn from a DIFFERENT graph should be ~random with respect to node
top-type (argmax x[:, :16]).

Metric: for a sample of query nodes, fraction whose top-1 CROSS-GRAPH neighbor
(nearest node, hyperbolic score, from a different graph) shares the query node's
top-type, vs the base rate of that type in the cross-graph pool (chance).
    lift = observed_match_frac / mean_base_rate
PRE-REGISTERED: lift < 1.3x -> EXPECTED_NEGATIVE_CONFIRMED.
High lift -> SURPRISING (real cross-graph capability).

Run from kettle-graph-reasoner/:
    py -m scripts.stress_cross_graph_twin \
        --corpus src/data/corpus/real_domain_eval_all6 \
        --out runs/stress_cross_graph_twin
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/real_domain_eval_all6")
    ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--out", default="runs/stress_cross_graph_twin")
    ap.add_argument("--n-graphs", type=int, default=20)
    ap.add_argument("--n-query", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    files = sorted(Path(args.corpus).glob("graph_*.npz"))[: args.n_graphs]
    print(f"{len(files)} graphs")

    enc = None
    all_emb = []      # list of [N_g, D] tensors
    all_type = []     # list of [N_g] int arrays
    all_gid = []      # list of [N_g] int arrays
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
        ttype = x[:, :16].argmax(axis=1).astype(np.int64)
        all_emb.append(emb)
        all_type.append(ttype)
        all_gid.append(np.full(emb.shape[0], gi, dtype=np.int64))

    pool_emb = torch.cat(all_emb, dim=0)                 # [M, D]
    pool_type = np.concatenate(all_type)                 # [M]
    pool_gid = np.concatenate(all_gid)                   # [M]
    M = pool_emb.shape[0]
    print(f"pool: {M} nodes, {len(np.unique(pool_type))} distinct top-types")

    # sample query nodes uniformly from the pool
    n_query = min(args.n_query, M)
    qidx = rng.choice(M, size=n_query, replace=False)

    match = 0
    base_rates = []
    type_match_type_base = []  # store per-query (matched?, base_rate)
    c = enc.c
    for qi in qidx:
        q_type = int(pool_type[qi])
        q_gid = int(pool_gid[qi])
        cross_mask = pool_gid != q_gid                   # different graph only
        # base rate: fraction of cross-graph pool sharing q_type
        cross_types = pool_type[cross_mask]
        base = float((cross_types == q_type).mean())
        base_rates.append(base)
        # nearest cross-graph neighbor by hyperbolic score (higher = closer)
        sc = score_from_embeddings(pool_emb[cross_mask], pool_emb[qi], c=c)
        nn_local = int(torch.argmax(sc).item())
        nn_type = int(cross_types[nn_local])
        m = int(nn_type == q_type)
        match += m
        type_match_type_base.append((m, base))

    obs = match / n_query
    mean_base = float(np.mean(base_rates))
    lift = obs / mean_base if mean_base > 0 else float("nan")

    # control: random cross-graph neighbor should match at ~base rate (lift~1)
    rng2 = np.random.default_rng(args.seed + 1)
    rand_match = 0
    for qi in qidx:
        q_type = int(pool_type[qi])
        q_gid = int(pool_gid[qi])
        cross_idx = np.where(pool_gid != q_gid)[0]
        pick = cross_idx[rng2.integers(len(cross_idx))]
        rand_match += int(pool_type[pick] == q_type)
    rand_obs = rand_match / n_query
    rand_lift = rand_obs / mean_base if mean_base > 0 else float("nan")

    report = {
        "config": vars(args),
        "pool_nodes": int(M),
        "n_query": int(n_query),
        "distinct_types_in_pool": int(len(np.unique(pool_type))),
        "observed_top1_type_match_frac": round(obs, 4),
        "mean_base_rate": round(mean_base, 4),
        "lift": round(lift, 4),
        "control_random_neighbor_match_frac": round(rand_obs, 4),
        "control_random_lift": round(rand_lift, 4),
        "n_type_match": int(match),
        "verdict_bar": "lift < 1.3 => EXPECTED_NEGATIVE_CONFIRMED",
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print("\n=== CROSS-GRAPH TWIN (top-1 cross-graph neighbor type match) ===")
    print(f"pool nodes            : {M}")
    print(f"query nodes           : {n_query}")
    print(f"observed match frac   : {obs:.4f}")
    print(f"mean base rate        : {mean_base:.4f}")
    print(f"LIFT                  : {lift:.4f}")
    print(f"control random match  : {rand_obs:.4f}  (lift {rand_lift:.4f})")
    print(f"\nreport: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

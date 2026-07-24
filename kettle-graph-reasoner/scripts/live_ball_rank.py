r"""Live end-to-end ball-ordering demo against the running Neo4j.

The assembler-shaped path, timed per stage: export ONE fresh query
neighborhood from the live graph (reuses the validated
neo4j_eval_export pipeline), embed it with the FROZEN encoder, order the
ball with retrieval_ops.order_ball, print the top-k with real Neo4j node
ids. This is the reference implementation for wiring emb-ordering into
the context assembler (default mode: pure emb-order, ndcg 0.885 vs hop
0.690 — see runs/probe_capability_ballrank).

    py -m scripts.live_ball_rank --seed 123 [--topk 15]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.codegraph.harness import _build_encoder
from src.data.corpus_dataset import _build_graph_tensors
from src.modelsv3.retrieval_ops import order_ball

CKPT = "frozen/kgr-v1.0-2026-07-07/encoder_baseline"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--out", default="runs/live_ball_rank")
    args = ap.parse_args()
    out = Path(args.out)

    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, "neo4j_eval_export.py", "export",
         "--config", "kettle_config.yaml", "--out", f"../{out}",
         "--num-graphs", "1", "--max-nodes", "400",
         "--tasks-per-graph", "1", "--seed", str(args.seed),
         "--sampler", "delocalized", "--n-seeds", "4"],
        cwd="scripts", check=True, capture_output=True)
    t1 = time.perf_counter()

    f = sorted(out.glob("graph_*.npz"))[0]
    z = np.load(f, allow_pickle=True)
    g = _build_graph_tensors(z)
    device = torch.device("cpu")
    enc, _ = _build_encoder(Path(CKPT), g, device)
    t2 = time.perf_counter()
    with torch.no_grad():
        emb = enc(g["x"], g["edge_index"], g["edge_type"],
                  g["edge_descriptor"],
                  node_descriptor=g["node_descriptor"]).node_embeddings
    t3 = time.perf_counter()

    anchor = int(z["task_0_anchor_row"])
    mh = float(z["task_0_max_hops"])
    n = emb.shape[0]
    adj = [[] for _ in range(n)]
    for s_, t_ in zip(*z["edge_index"]):
        adj[int(s_)].append(int(t_))
        adj[int(t_)].append(int(s_))
    d = np.full(n, np.inf)
    d[anchor] = 0
    dq = deque([anchor])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if d[v] == np.inf:
                d[v] = d[u] + 1
                dq.append(v)
    ball = [r for r in range(n) if r != anchor and d[r] <= mh]
    ranked = order_ball(emb, anchor, ball, c=enc.c)
    t4 = time.perf_counter()

    ids = z["neo4j_node_id"]
    print(f"anchor neo4j id: {ids[anchor]}  ball={len(ball)} nodes "
          f"(hop<={mh:.0f} of {n})")
    print(f"top-{args.topk} by emb-order (neo4j_id, hop):")
    for r in ranked[: args.topk]:
        print(f"  {ids[r]}  hop={d[r]:.0f}")
    print(f"\nlatency: export {t1-t0:.1f}s | model load {t2-t1:.1f}s | "
          f"embed({n} nodes) {(t3-t2)*1000:.0f}ms | "
          f"order_ball {(t4-t3)*1000:.0f}ms")
    print("(assembler steady-state cost = embed + order only; "
          "export/load are per-process)")


if __name__ == "__main__":
    main()

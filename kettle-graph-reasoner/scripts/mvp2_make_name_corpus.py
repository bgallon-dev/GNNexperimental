r"""V2 MVP-2a: append pseudo-name feature vectors to a corpus.

Implements the plan's pseudo-name spec (Docs/KGR_V2_PLAN.md), minimal
cut: a GLOBAL topic pool (shared across graphs — the collision structure
that makes lexical evidence transferable), per-graph regions via
multi-source BFS partition, per-region topic draw, node name vec =
normalize(topic + sigma * noise). Appended to x like the landmark block;
the trainer is untouched.

    py -m scripts.mvp2_make_name_corpus --src src/data/corpus/tier1 \
        --dst src/data/corpus/tier1_nm16
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np

NAME_D = 16          # name-vector dims
N_TOPICS = 64        # global topic pool (shared across graphs)
SIGMA = 0.3          # within-topic noise
REGION_SIZE = 25     # ~nodes per region seed


def _regions(adj, n, rng):
    k = max(n // REGION_SIZE, 2)
    seeds = rng.choice(n, min(k, n), replace=False)
    region = np.full(n, -1, np.int64)
    q = deque()
    for i, s in enumerate(seeds):
        region[s] = i
        q.append(int(s))
    while q:
        u = q.popleft()
        for v in adj[u]:
            if region[v] < 0:
                region[v] = region[u]
                q.append(v)
    region[region < 0] = 0      # isolated nodes -> region 0
    return region, len(seeds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    g_rng = np.random.default_rng(777)
    topics = g_rng.standard_normal((N_TOPICS, NAME_D)).astype(np.float32)
    topics /= np.linalg.norm(topics, axis=1, keepdims=True)

    files = sorted(Path(a.src).glob("graph_*.npz"))
    for gi, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        x = d["x"].astype(np.float32)
        n = x.shape[0]
        adj = [[] for _ in range(n)]
        for s_, t_ in zip(*d["edge_index"]):
            adj[int(s_)].append(int(t_))
            adj[int(t_)].append(int(s_))
        rng = np.random.default_rng(9000 + gi)
        region, k = _regions(adj, n, rng)
        topic_of_region = rng.integers(0, N_TOPICS, size=k)
        names = topics[topic_of_region[region]] \
            + SIGMA * rng.standard_normal((n, NAME_D)).astype(np.float32)
        names /= np.linalg.norm(names, axis=1, keepdims=True)
        d["x"] = np.concatenate([x, names.astype(np.float32)], axis=1)
        d["name_topic"] = topic_of_region[region]   # eval bookkeeping
        np.savez_compressed(dst / f.name, **d)
    print(f"wrote {len(files)} graphs to {dst} "
          f"(x: +{NAME_D} name dims, {N_TOPICS} global topics)")


if __name__ == "__main__":
    main()

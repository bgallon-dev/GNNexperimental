r"""V2 MVP-1 step 1: append K=8 landmark-distance features to a corpus.
Landmarks = deterministic random nodes per graph; feature = 1/(1+hops).
Gives the encoder global position inputs (Docs/KGR_V2_PLAN.md delta 3).
    py -m scripts.mvp1_make_landmark_corpus --src src/data/corpus/tier1 --dst src/data/corpus/tier1_lm8"""
import argparse
from collections import deque
from pathlib import Path
import numpy as np

K = 8

def bfs(adj, src, n):
    d = np.full(n, np.inf, np.float32); d[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if d[v] == np.inf: d[v] = d[u] + 1; q.append(v)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    dst = Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.src).glob("graph_*.npz"))
    for gi, f in enumerate(files):
        z = np.load(f, allow_pickle=True); d = {k: z[k] for k in z.files}
        x = d["x"]; n = x.shape[0]
        adj = [[] for _ in range(n)]
        for s, t in zip(*d["edge_index"]):
            adj[int(s)].append(int(t)); adj[int(t)].append(int(s))
        rng = np.random.default_rng(4242 + gi)
        lms = rng.choice(n, min(K, n), replace=False)
        feats = np.zeros((n, K), np.float32)
        for j, lm in enumerate(lms):
            feats[:, j] = 1.0 / (1.0 + bfs(adj, int(lm), n))
        d["x"] = np.concatenate([x.astype(np.float32), feats], axis=1)
        np.savez_compressed(dst / f.name, **d)
    print(f"wrote {len(files)} graphs to {dst} (x: +{K} landmark dims)")

if __name__ == "__main__":
    main()

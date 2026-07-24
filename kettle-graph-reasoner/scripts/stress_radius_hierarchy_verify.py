r"""Adversarial verification (key=radius_hierarchy): re-run the radius-vs-
hierarchy Spearman measurement on a DISJOINT slice (graphs [start:start+n])
with a DIFFERENT control seed. Default to skepticism.

Mirrors scripts/stress_radius_hierarchy.py exactly except:
  - --start offset (default 100) so the slice is disjoint from the first 60
  - --seed for the random control (default 7, not 0)
  - adds a SECOND random control (shuffle of the real radius) as a tightened
    null: a shuffled-radius vs signal rho must also be ~0.

Run from kettle-graph-reasoner/:
    PYTHONIOENCODING=utf-8 py -m scripts.stress_radius_hierarchy_verify \
        --corpus src/data/corpus/real_domain_eval_all6 \
        --start 100 --n 60 --seed 7 \
        --out runs/stress_radius_hierarchy_verify
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


def _spearman(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if len(a) < 3:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _bfs(adj, src, n):
    d = np.full(n, np.inf, np.float64)
    d[src] = 0.0
    dq = deque([src])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if d[v] == np.inf:
                d[v] = d[u] + 1.0
                dq.append(v)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/real_domain_eval_all6")
    ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="runs/stress_radius_hierarchy_verify")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    all_files = sorted(Path(args.corpus).glob("graph_*.npz"))
    files = all_files[args.start: args.start + args.n]
    print(f"{len(files)} graphs (slice [{args.start}:{args.start + args.n}])")

    enc = None
    P = {"rad": [], "layer": [], "degree": [], "depth": [], "rand": [],
         "radshuf": []}
    G = {"layer": [], "degree": [], "depth": [], "rand": [], "radshuf_deg": []}

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
        rad = emb.norm(dim=-1).cpu().numpy().astype(np.float64)
        n = rad.shape[0]

        x = z["x"]
        ttypes = x[:, :16].argmax(axis=1)
        layer_of_type = z["schema_node_layer_assignment"]
        layer = layer_of_type[ttypes].astype(np.float64)
        degree = x[:, 16].astype(np.float64)

        adj = [[] for _ in range(n)]
        ei = g["edge_index"].numpy()
        for s, t in zip(ei[0], ei[1]):
            adj[int(s)].append(int(t))
            adj[int(t)].append(int(s))
        src = int(np.argmax(degree))
        d = _bfs(adj, src, n)
        finite = np.isfinite(d)

        randv = rng.standard_normal(n)
        radshuf = rng.permutation(rad)  # tightened null: shuffle real radius

        P["rad"].append(rad)
        P["layer"].append(layer)
        P["degree"].append(degree)
        P["depth"].append(np.where(finite, d, np.nan))
        P["rand"].append(randv)
        P["radshuf"].append(radshuf)

        G["layer"].append(_spearman(rad, layer))
        G["degree"].append(_spearman(rad, degree))
        if finite.sum() >= 3:
            G["depth"].append(_spearman(rad[finite], d[finite]))
        G["rand"].append(_spearman(rad, randv))
        G["radshuf_deg"].append(_spearman(radshuf, degree))

    rad_all = np.concatenate(P["rad"])
    layer_all = np.concatenate(P["layer"])
    degree_all = np.concatenate(P["degree"])
    depth_all = np.concatenate(P["depth"])
    rand_all = np.concatenate(P["rand"])
    radshuf_all = np.concatenate(P["radshuf"])
    depth_ok = np.isfinite(depth_all)

    def _mean(v):
        v = [x for x in v if not np.isnan(x)]
        return float(np.mean(v)) if v else float("nan")

    agg = {
        "layer": _spearman(rad_all, layer_all),
        "degree": _spearman(rad_all, degree_all),
        "depth": _spearman(rad_all[depth_ok], depth_all[depth_ok]),
        "rand": _spearman(rad_all, rand_all),
        "radshuf_deg": _spearman(radshuf_all, degree_all),
    }
    pergraph = {k: _mean(G[k]) for k in G}

    report = {
        "config": vars(args),
        "n_graphs": len(files),
        "n_nodes_pooled": int(rad_all.shape[0]),
        "rad_mean": float(rad_all.mean()),
        "rad_std": float(rad_all.std()),
        "rad_min": float(rad_all.min()),
        "rad_max": float(rad_all.max()),
        "spearman_pooled": agg,
        "spearman_pergraph_mean": pergraph,
    }
    print("\n=== Spearman rho: radius ||h|| vs structural signal (VERIFY) ===")
    print(f"{'signal':<14} {'pooled':>8} {'pergraph':>9}")
    for k in ("layer", "degree", "depth", "rand"):
        print(f"{k:<14} {agg[k]:>8.3f} {pergraph[k]:>9.3f}")
    print(f"{'radshuf_deg':<14} {agg['radshuf_deg']:>8.3f} "
          f"{pergraph['radshuf_deg']:>9.3f}  (tightened null)")
    print(f"\nradius: mean={rad_all.mean():.4f} std={rad_all.std():.4f} "
          f"min={rad_all.min():.4f} max={rad_all.max():.4f}")
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

r"""Stress probe: geometric health on OOD graph topologies (key=topology_ood).

Build ~60-node synthetic graphs in the tier1/all6 npz schema by LOADING a
real all6 npz template and REWIRING ONLY edge_index + edge_type (single valid
edge class). x is sliced to the first N template nodes; schema_* descriptors
(edge_descriptor 30x13 per-type, node_descriptor 16x4 per-type) are kept
verbatim -- they are per-TYPE lookups, independent of E and N, so the encoder
consumes them unchanged.

Topologies: PATH, CYCLE, COMPLETE, STAR, BIPARTITE, BINTREE (balanced binary
tree), TWO_COMP (two disconnected cliques). Controls: INDIST (the real
template subgraph among the first N nodes -- in-distribution reference) and
RANDOM (Erdos-Renyi p=0.1 -- random-topology floor).

Per topology report |h| mean/max (boundary-saturation: healthy max < ~0.9,
pathological ~0.99) and embedding variance (collapse: var ~ 0). Averaged over
the first N_TEMPLATES real graphs (x varies per template).

Run from kettle-graph-reasoner/:
    py -m scripts.stress_topology_ood \
        --corpus src/data/corpus/real_domain_eval_all6 \
        --out runs/stress_topology_ood
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.codegraph.harness import _build_encoder
from src.data.corpus_dataset import _build_graph_tensors

N = 60
N_TEMPLATES = 5
EDGE_CLASS = 0  # a valid single edge class (schema has 30 types)


def _edges_for(topology: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return edge_index [2, E] (single direction per topological edge)."""
    src, dst = [], []
    if topology == "PATH":
        for i in range(n - 1):
            src.append(i); dst.append(i + 1)
    elif topology == "CYCLE":
        for i in range(n):
            src.append(i); dst.append((i + 1) % n)
    elif topology == "COMPLETE":
        for i in range(n):
            for j in range(i + 1, n):
                src.append(i); dst.append(j)
    elif topology == "STAR":
        for i in range(1, n):
            src.append(0); dst.append(i)
    elif topology == "BIPARTITE":
        h = n // 2
        for i in range(h):
            for j in range(h, n):
                src.append(i); dst.append(j)
    elif topology == "BINTREE":
        for i in range(n):
            for c in (2 * i + 1, 2 * i + 2):
                if c < n:
                    src.append(i); dst.append(c)
    elif topology == "TWO_COMP":
        h = n // 2
        for a, b in ((0, h), (h, n)):
            for i in range(a, b):
                for j in range(i + 1, b):
                    src.append(i); dst.append(j)
    elif topology == "RANDOM":
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.1:
                    src.append(i); dst.append(j)
    else:
        raise ValueError(topology)
    if not src:  # guard (shouldn't happen)
        src, dst = [0], [min(1, n - 1)]
    return np.stack([np.asarray(src), np.asarray(dst)]).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/real_domain_eval_all6")
    ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--out", default="runs/stress_topology_ood")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    files = sorted(Path(args.corpus).glob("graph_*.npz"))[:N_TEMPLATES]
    print(f"{len(files)} templates, N={N} nodes")

    topologies = ["PATH", "CYCLE", "COMPLETE", "STAR", "BIPARTITE",
                  "BINTREE", "TWO_COMP", "INDIST", "RANDOM"]
    acc: dict[str, list[dict]] = {t: [] for t in topologies}

    enc = None
    c_val = None
    for ti, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        g0 = _build_graph_tensors(z)
        if enc is None:
            enc, _ = _build_encoder(Path(args.ckpt), g0, device)
            c_val = float(enc.c)
        n = min(N, g0["x"].shape[0])
        x = g0["x"][:n].clone()
        edge_desc = g0["edge_descriptor"].to(device)
        node_desc = g0["node_descriptor"].to(device)
        rng = np.random.default_rng(1000 + ti)

        for topo in topologies:
            if topo == "INDIST":
                ei_full = g0["edge_index"].numpy()
                mask = (ei_full[0] < n) & (ei_full[1] < n)
                ei = ei_full[:, mask]
                if ei.shape[1] == 0:
                    ei = np.array([[0], [min(1, n - 1)]], dtype=np.int64)
                et_full = g0["edge_type"].numpy()[mask]
                et = torch.from_numpy(et_full.astype(np.int64))
            else:
                ei = _edges_for(topo, n, rng)
                et = torch.full((ei.shape[1],), EDGE_CLASS, dtype=torch.long)
            ei_t = torch.from_numpy(ei).to(device)
            with torch.no_grad():
                emb = enc(
                    x.to(device), ei_t, et.to(device), edge_desc,
                    node_descriptor=node_desc,
                ).node_embeddings
            norms = emb.norm(dim=-1)
            var = float(emb.var(dim=0).mean())  # mean per-dim variance
            acc[topo].append({
                "norm_mean": float(norms.mean()),
                "norm_max": float(norms.max()),
                "var": var,
                "n": n,
                "E": int(ei.shape[1]),
            })

    boundary = 1.0 / np.sqrt(c_val) if c_val > 0 else float("inf")
    report = {"config": vars(args), "N": N, "n_templates": len(files),
              "c": c_val, "boundary": boundary, "per_topology": {}}

    def avg(sub, k):
        return float(np.mean([r[k] for r in sub]))

    print(f"\nc={c_val:.4f}  boundary(1/sqrt(c))={boundary:.4f}")
    print(f"{'topology':<10} {'E':>6} {'norm_mean':>10} {'norm_max':>10} "
          f"{'var':>12}")
    sat_flag = {}
    for topo in topologies:
        sub = acc[topo]
        cell = {"norm_mean": avg(sub, "norm_mean"),
                "norm_max": avg(sub, "norm_max"),
                "var": avg(sub, "var"),
                "E": avg(sub, "E")}
        report["per_topology"][topo] = cell
        # health flags: saturated if max norm within 5% of boundary,
        # collapsed if var below 1e-4
        rel_max = cell["norm_max"] / boundary if boundary != float("inf") else 0
        cell["rel_norm_max"] = rel_max
        cell["saturated"] = bool(rel_max > 0.95)
        cell["collapsed"] = bool(cell["var"] < 1e-4)
        sat_flag[topo] = cell["saturated"] or cell["collapsed"]
        print(f"{topo:<10} {cell['E']:>6.0f} {cell['norm_mean']:>10.4f} "
              f"{cell['norm_max']:>10.4f} {cell['var']:>12.6f}"
              f"  rel_max={rel_max:.3f}"
              + ("  SATURATED" if cell["saturated"] else "")
              + ("  COLLAPSED" if cell["collapsed"] else ""))

    ood = [t for t in topologies if t not in ("INDIST",)]
    failing = [t for t in ood if sat_flag[t]]
    report["failing_topologies"] = failing
    report["verdict"] = "ROBUST" if not failing else "FRAGILE"
    print(f"\nverdict: {report['verdict']}  failing={failing}")

    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(f"report: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

r"""ADVERSARIAL VERIFY of key=topology_ood (out-star collapse).

Re-run the SAME measurement on a DISJOINT slice of all6 templates
(indices 100..160 instead of the first 5) with a DIFFERENT seed base.
Headline under test: on an OUT-STAR topology (edges 0->i, node 0 is a
source hub) all node embeddings collapse to a single point
(out_star_var ~ 4e-15, out_star_unique_rows == 1), while IN-STAR and all
other topologies stay geometrically healthy.

Refute rule: if out_star_var does NOT replicate as ~0 (< 1e-10) and
unique_rows != 1 on the independent slice -> REFUTED. If it clearly holds
-> CONFIRMED. Controls: IN-STAR (should be healthy), RANDOM/INDIST floor.

Run from kettle-graph-reasoner/:
    PYTHONIOENCODING=utf-8 py -m scripts.stress_topology_ood_verify \
        --start 100 --n-templates 6 --out runs/stress_topology_ood_verify
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
EDGE_CLASS = 0


def _edges_for(topology, n, rng):
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
    elif topology == "OUT_STAR":   # center -> leaf (the claimed failure)
        for i in range(1, n):
            src.append(0); dst.append(i)
    elif topology == "IN_STAR":    # leaf -> center (claimed healthy control)
        for i in range(1, n):
            src.append(i); dst.append(0)
    elif topology == "BIDIR_STAR":
        for i in range(1, n):
            src.append(0); dst.append(i)
            src.append(i); dst.append(0)
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
    if not src:
        src, dst = [0], [min(1, n - 1)]
    return np.stack([np.asarray(src), np.asarray(dst)]).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/real_domain_eval_all6")
    ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--out", default="runs/stress_topology_ood_verify")
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--n-templates", type=int, default=6)
    ap.add_argument("--seed-base", type=int, default=7777)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    allf = sorted(Path(args.corpus).glob("graph_*.npz"))
    files = allf[args.start:args.start + args.n_templates]
    print(f"slice {args.start}..{args.start + args.n_templates} "
          f"({len(files)} templates), N={N}")

    topologies = ["PATH", "CYCLE", "COMPLETE", "OUT_STAR", "IN_STAR",
                  "BIDIR_STAR", "BIPARTITE", "BINTREE", "TWO_COMP",
                  "INDIST", "RANDOM"]
    acc = {t: [] for t in topologies}

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
        x_var = float(x.var(dim=0).mean())
        rng = np.random.default_rng(args.seed_base + ti)

        for topo in topologies:
            if topo == "INDIST":
                ei_full = g0["edge_index"].numpy()
                mask = (ei_full[0] < n) & (ei_full[1] < n)
                ei = ei_full[:, mask]
                if ei.shape[1] == 0:
                    ei = np.array([[0], [min(1, n - 1)]], dtype=np.int64)
                et = torch.from_numpy(g0["edge_type"].numpy()[mask].astype(np.int64))
            else:
                ei = _edges_for(topo, n, rng)
                et = torch.full((ei.shape[1],), EDGE_CLASS, dtype=torch.long)
            ei_t = torch.from_numpy(ei).to(device)
            with torch.no_grad():
                emb = enc(x.to(device), ei_t, et.to(device), edge_desc,
                          node_descriptor=node_desc).node_embeddings
            norms = emb.norm(dim=-1)
            var = float(emb.var(dim=0).mean())
            uniq = int(torch.unique(emb.round(decimals=5), dim=0).shape[0])
            acc[topo].append({
                "norm_mean": float(norms.mean()),
                "norm_max": float(norms.max()),
                "var": var,
                "unique_rows": uniq,
                "x_var": x_var,
                "E": int(ei.shape[1]),
            })

    boundary = 1.0 / np.sqrt(c_val) if c_val > 0 else float("inf")

    def avg(sub, k):
        return float(np.mean([r[k] for r in sub]))

    print(f"\nc={c_val:.4f} boundary={boundary:.4f}")
    print(f"{'topology':<12}{'E':>7}{'norm_mean':>10}{'norm_max':>10}"
          f"{'var':>13}{'uniq':>6}")
    report = {"config": vars(args), "N": N, "n_templates": len(files),
              "c": c_val, "boundary": boundary, "per_topology": {}}
    for topo in topologies:
        sub = acc[topo]
        cell = {k: avg(sub, k) for k in ("norm_mean", "norm_max", "var",
                                         "unique_rows", "E")}
        cell["collapsed"] = bool(cell["var"] < 1e-4)
        report["per_topology"][topo] = cell
        print(f"{topo:<12}{cell['E']:>7.0f}{cell['norm_mean']:>10.4f}"
              f"{cell['norm_max']:>10.4f}{cell['var']:>13.3e}"
              f"{cell['unique_rows']:>6.1f}"
              + ("  COLLAPSED" if cell["collapsed"] else ""))

    os = report["per_topology"]["OUT_STAR"]
    ins = report["per_topology"]["IN_STAR"]
    replicates = os["var"] < 1e-10 and round(os["unique_rows"]) == 1
    control_ok = ins["var"] > 1e-5 and round(ins["unique_rows"]) > 1
    report["headline"] = {
        "out_star_var": os["var"],
        "out_star_unique_rows": os["unique_rows"],
        "in_star_var": ins["var"],
        "in_star_unique_rows": ins["unique_rows"],
        "out_star_collapse_replicates": bool(replicates),
        "in_star_control_healthy": bool(control_ok),
    }
    report["verdict"] = "CONFIRMED" if (replicates and control_ok) else "REFUTED"
    print(f"\nout_star collapse replicates: {replicates} "
          f"(var={os['var']:.2e}, uniq={os['unique_rows']:.1f})")
    print(f"in_star control healthy: {control_ok} "
          f"(var={ins['var']:.2e}, uniq={ins['unique_rows']:.1f})")
    print(f"VERDICT: {report['verdict']}")
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(f"report: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

r"""V2 MVP-2a probe: anchor-free FIND across a multi-graph pool.

Pool = all nodes of the last --n-pool tier1_nm16 graphs. For every task
anchor in those graphs, rank the WHOLE pool (cross-graph nodes are all
negatives) for the task's relevant nodes:

  name_cos   cosine over the raw 16-d name block   (text-only ANN baseline)
  emb_nm     name-aware trunk emb-dist to anchor   (the FUSION arm)
  emb_base   frozen v1.0 trunk on the plain 32-d x (structure-only control)

BAR (plan): fusion > text-only on recall@50. Caveat: in-corpus probe
(trunk saw these graphs in Stage-A) — a feature-utility test, not a
generalization claim; treat positives as MVP-2a evidence gating the full
build.

    py -m scripts.mvp2_find_probe --nm-ckpt runs/mvp2-nm16-h128-l4-s0
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
from src.training.metrics import ndcg_at_k, recall_at_k

NAME_D = 16


def _embed(ckpt, z_dict, device, enc_cache={}):
    g = _build_graph_tensors(z_dict)
    key = (str(ckpt), g["x"].shape[1])
    if key not in enc_cache:
        enc_cache[key] = _build_encoder(Path(ckpt), g, device)[0]
    enc = enc_cache[key]
    with torch.no_grad():
        return enc(g["x"], g["edge_index"], g["edge_type"],
                   g["edge_descriptor"],
                   node_descriptor=g["node_descriptor"]).node_embeddings, enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="src/data/corpus/tier1_nm16")
    ap.add_argument("--base-corpus", default="src/data/corpus/tier1")
    ap.add_argument("--nm-ckpt", default="runs/mvp2-nm16-h128-l4-s0")
    ap.add_argument("--base-ckpt",
                    default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--n-pool", type=int, default=20)
    ap.add_argument("--out", default="runs/mvp2_find_probe")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    nm_files = sorted(Path(args.corpus).glob("graph_*.npz"))[-args.n_pool:]
    base_files = sorted(
        Path(args.base_corpus).glob("graph_*.npz"))[-args.n_pool:]

    names, emb_nm, emb_base, offs, tasks = [], [], [], [0], []
    for fn, fb in zip(nm_files, base_files):
        zn = {k: v for k, v in np.load(fn, allow_pickle=True).items()}
        zb = {k: v for k, v in np.load(fb, allow_pickle=True).items()}
        n = zn["x"].shape[0]
        names.append(zn["x"][:, -NAME_D:])
        e_nm, _ = _embed(args.nm_ckpt, zn, device)
        e_b, _ = _embed(args.base_ckpt, zb, device)
        emb_nm.append(e_nm)
        emb_base.append(e_b)
        base = offs[-1]
        for i in range(int(zn["n_tasks"])):
            lab = zn[f"task_{i}_labels"]
            if (lab >= 0.5).sum() >= 1:
                tasks.append({
                    "anchor": base + int(zn[f"task_{i}_anchor_row"]),
                    "pos": base + np.flatnonzero(lab >= 0.5),
                })
        offs.append(base + n)

    NM = torch.from_numpy(np.concatenate(names)).float()
    NM = torch.nn.functional.normalize(NM, dim=-1)
    E_nm = torch.cat(emb_nm)
    E_b = torch.cat(emb_base)
    n_pool = NM.shape[0]
    print(f"pool: {n_pool} nodes across {len(nm_files)} graphs, "
          f"{len(tasks)} anchored find-tasks")

    from src.codegraph.harness import _build_encoder as _be  # noqa
    # curvature for distance scoring: read once from the nm encoder
    _, enc_nm = _embed(args.nm_ckpt,
                       {k: v for k, v in
                        np.load(nm_files[0], allow_pickle=True).items()},
                       device)
    c = enc_nm.c

    res = {a: {"r50": [], "ndcg10": []} for a in
           ("name_cos", "emb_nm", "emb_base")}
    for t in tasks:
        a = t["anchor"]
        lab = torch.zeros(n_pool)
        lab[torch.from_numpy(t["pos"])] = 1.0
        lab[a] = 0.0                      # anchor itself excluded
        arms = {
            "name_cos": NM @ NM[a],
            "emb_nm": score_from_embeddings(E_nm, E_nm[a], c=c),
            "emb_base": score_from_embeddings(E_b, E_b[a], c=c),
        }
        for nm_, sc in arms.items():
            sc = sc.clone()
            sc[a] = torch.finfo(torch.float32).min
            res[nm_]["r50"].append(recall_at_k(sc, lab, 50))
            res[nm_]["ndcg10"].append(ndcg_at_k(sc, lab, 10))

    summary = {}
    print(f"\n=== anchor-free FIND over {n_pool}-node pool ===")
    print(f"{'arm':<10} {'recall@50':>10} {'ndcg@10':>9}")
    for nm_, v in res.items():
        summary[nm_] = {"recall@50": float(np.mean(v["r50"])),
                        "ndcg@10": float(np.mean(v["ndcg10"])),
                        "n": len(v["r50"])}
        print(f"{nm_:<10} {summary[nm_]['recall@50']:>10.3f} "
              f"{summary[nm_]['ndcg@10']:>9.3f}")
    bar = summary["emb_nm"]["recall@50"] > summary["name_cos"]["recall@50"]
    summary["bar_fusion_beats_text"] = bool(bar)
    print(f"\nBAR fusion > text-only (recall@50): "
          f"{'CLEARED' if bar else 'NOT cleared'}")
    (out / "find_results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

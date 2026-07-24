r"""E1 stress probe: is the (untyped) attention stack load-bearing? (key=uniform_attention)

Pre-registered in Docs/ARCH_EFFICIENCY_PLAN.md (E1). The edge-TYPE path of
attention ablates to exactly 0.000 (stress_edge_type_ablation), but the
W_q/W_k projections (131K params, 62% of the encoder) have never been
ablated. Three forward modes on the frozen v1.0 encoder, scored with the
L1 ball-order lens (emb[anchor]-distance ordering of the anchor's BFS
ball, ndcg@10):

  base     - stock forward (trained per-edge alpha)
  uniform  - bypass attention: every MP layer called with
             edge_weight=None, which is uniform 1/|N(i)| by construction
             (HyperbolicMessagePassing step 2). Zero new math.
  shuffled - trained alpha PERMUTED among edges sharing the same receiver
             (3 permutation seeds, mean). Separates "any softmax
             weighting" from "the learned allocation": if uniform hurts
             but shuffled matches base, the win is attention entropy,
             not learned content.

Slices (stress-test convention): primary graphs [0:50], adversarial
verify [100:160]. Random floor: 3 seeds per case.

Pre-registered bars (ALL, both slices):
  |d(uniform, base)| < 0.005  -> attention NOT load-bearing as trained
  d(base - uniform) >= 0.02 on ALL or any family (both slices)
                              -> load-bearing (record where)
  between                     -> INCONCLUSIVE -> escalate to the L2
                                 mixture lens before concluding

Run from kettle-graph-reasoner/:
    PYTHONIOENCODING=utf-8 py -m scripts.stress_uniform_attention [--smoke]
"""
from __future__ import annotations

import json
import sys
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from src.codegraph.harness import _build_encoder
from src.data.corpus_dataset import _build_graph_tensors
from src.modelsv3.distance_scoring import score_from_embeddings
from src.training.metrics import ndcg_at_k

FAMILY = {0: "provenance", 1: "entity_res", 2: "temporal",
          3: "multihop", 4: "subgraph", 5: "compound"}
CORPUS = "src/data/corpus/real_domain_eval_all6"
CKPT = "frozen/kgr-v1.0-2026-07-07/encoder_baseline"
OUT = "runs/stress_uniform_attention"
SLICES = {"primary_0_50": (0, 50), "verify_100_160": (100, 160)}
N_SHUFFLE_SEEDS = 3
BAR_NOT_LOAD_BEARING = 0.005
BAR_LOAD_BEARING = 0.02


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


def _shuffle_within_dst(alpha: torch.Tensor, dst: torch.Tensor,
                        gen: torch.Generator) -> torch.Tensor:
    """Permute alpha among edges that share a receiver. Vectorized: both
    orders below enumerate the same multiset of dst values sorted stably,
    so position i in both has the same receiver; the random pre-sort makes
    the within-group assignment a uniform permutation."""
    e = alpha.shape[0]
    keys = torch.rand(e, generator=gen)
    rand_first = torch.argsort(keys)
    rand_grouped = rand_first[torch.argsort(dst[rand_first], stable=True)]
    base_grouped = torch.argsort(dst, stable=True)
    out = torch.empty_like(alpha)
    out[base_grouped] = alpha[rand_grouped]
    return out


@contextmanager
def _mp_override(enc, mode: str, seed: int = 0):
    """Wrap every MP layer's forward. 'uniform' drops the incoming alpha
    (edge_weight=None -> 1/deg by construction); 'shuffled' permutes it
    within receiver groups. Restores the original forwards on exit."""
    origs = [mp.forward for mp in enc.mp_layers]
    gen = torch.Generator().manual_seed(seed)

    def _wrap(orig):
        def f(x, edge_index, edge_weight=None):
            if mode == "uniform":
                return orig(x, edge_index, edge_weight=None)
            if edge_weight is not None:
                edge_weight = _shuffle_within_dst(
                    edge_weight, edge_index[1], gen)
            return orig(x, edge_index, edge_weight=edge_weight)
        return f

    for mp in enc.mp_layers:
        mp.forward = _wrap(mp.forward)
    try:
        yield
    finally:
        for mp, o in zip(enc.mp_layers, origs):
            mp.forward = o


def _embed(enc, g, device):
    with torch.no_grad():
        return enc(g["x"].to(device), g["edge_index"].to(device),
                   g["edge_type"].to(device), g["edge_descriptor"].to(device),
                   node_descriptor=g["node_descriptor"].to(device),
                   ).node_embeddings


def _run_slice(enc, files, device):
    rows = []
    l2_shift_uniform = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        g = _build_graph_tensors(z)
        emb_base = _embed(enc, g, device)
        with _mp_override(enc, "uniform"):
            emb_unif = _embed(enc, g, device)
        emb_shuf = []
        for s_ in range(N_SHUFFLE_SEEDS):
            with _mp_override(enc, "shuffled", seed=1000 + s_):
                emb_shuf.append(_embed(enc, g, device))
        l2_shift_uniform.append(
            float((emb_base - emb_unif).norm(dim=-1).mean()))

        n = emb_base.shape[0]
        adj = [[] for _ in range(n)]
        ein = g["edge_index"].numpy()
        for s, t in zip(ein[0], ein[1]):
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
            if len(ball) < 5:
                continue
            lab_b = torch.from_numpy(labels[ball]).float()
            if not (float(lab_b.max()) > 0
                    and float(lab_b.min()) < float(lab_b.max())):
                continue
            rows_t = torch.tensor(ball, dtype=torch.long)

            def _score(emb):
                sc = score_from_embeddings(emb[rows_t], emb[anchor], c=enc.c)
                return ndcg_at_k(sc, lab_b, 10)

            rnds = []
            for s_ in range(3):
                g_ = torch.Generator().manual_seed(
                    (hash((f.name, i)) & 0xFFFFFF) + s_)
                rnds.append(ndcg_at_k(
                    torch.randn(len(ball), generator=g_), lab_b, 10))
            rows.append({
                "family": fam, "n_ball": len(ball),
                "base_ndcg10": _score(emb_base),
                "unif_ndcg10": _score(emb_unif),
                "shuf_ndcg10": sum(_score(e) for e in emb_shuf) / len(emb_shuf),
                "rand_ndcg10": sum(rnds) / len(rnds),
            })
    return rows, l2_shift_uniform


def _summarize(rows):
    def _avg(sub, k):
        v = [r[k] for r in sub]
        return sum(v) / len(v) if v else float("nan")

    fams = sorted({r["family"] for r in rows})
    out = {}
    for fam in fams + ["ALL"]:
        sub = rows if fam == "ALL" else [r for r in rows if r["family"] == fam]
        b = _avg(sub, "base_ndcg10")
        u = _avg(sub, "unif_ndcg10")
        s = _avg(sub, "shuf_ndcg10")
        out[fam] = {"n": len(sub), "base_ndcg10": b, "unif_ndcg10": u,
                    "d_uniform": b - u, "shuf_ndcg10": s, "d_shuffled": b - s,
                    "rand_ndcg10": _avg(sub, "rand_ndcg10")}
    return out


def _verdict(summaries):
    """Apply the pre-registered bars to the per-slice ALL/family deltas."""
    d_all = [s["ALL"]["d_uniform"] for s in summaries.values()]
    if all(abs(d) < BAR_NOT_LOAD_BEARING for d in d_all):
        return "NOT_LOAD_BEARING_AS_TRAINED"
    fam_hits = []
    for fam in summaries[next(iter(summaries))]:
        ds = [s[fam]["d_uniform"] for s in summaries.values() if fam in s]
        if len(ds) == len(summaries) and all(d >= BAR_LOAD_BEARING for d in ds):
            fam_hits.append(fam)
    if fam_hits:
        return "LOAD_BEARING:" + ",".join(sorted(fam_hits))
    return "INCONCLUSIVE_ESCALATE_L2"


def main() -> None:
    smoke = "--smoke" in sys.argv
    out_dir = Path(OUT + ("_smoke" if smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    all_files = sorted(Path(CORPUS).glob("graph_*.npz"))
    print(f"corpus: {len(all_files)} graphs; smoke={smoke}")

    enc = None
    report = {"bars": {"not_load_bearing": BAR_NOT_LOAD_BEARING,
                       "load_bearing": BAR_LOAD_BEARING},
              "n_shuffle_seeds": N_SHUFFLE_SEEDS, "slices": {}}
    summaries = {}
    for slice_name, (lo, hi) in SLICES.items():
        files = all_files[lo:hi]
        if smoke:
            files = files[:6]
        if not files:
            print(f"slice {slice_name}: EMPTY, skipped")
            continue
        if enc is None:
            z0 = np.load(files[0], allow_pickle=True)
            enc, _ = _build_encoder(Path(CKPT), _build_graph_tensors(z0),
                                    device)
        rows, l2u = _run_slice(enc, files, device)
        summ = _summarize(rows)
        summaries[slice_name] = summ
        report["slices"][slice_name] = {
            "n_graphs": len(files), "n_cases": len(rows),
            "mean_emb_L2_shift_uniform": sum(l2u) / len(l2u),
            "by_family": summ,
        }
        print(f"\n== slice {slice_name}: {len(files)} graphs, "
              f"{len(rows)} cases; mean emb L2 shift (uniform) "
              f"{sum(l2u) / len(l2u):.4f}")
        print(f"{'family':<12} {'n':>4} {'base':>8} {'uniform':>8} "
              f"{'d_unif':>8} {'shuffled':>9} {'d_shuf':>8} {'random':>8}")
        for fam, v in summ.items():
            print(f"{fam:<12} {v['n']:>4} {v['base_ndcg10']:>8.4f} "
                  f"{v['unif_ndcg10']:>8.4f} {v['d_uniform']:>+8.4f} "
                  f"{v['shuf_ndcg10']:>9.4f} {v['d_shuffled']:>+8.4f} "
                  f"{v['rand_ndcg10']:>8.4f}")

    report["verdict"] = _verdict(summaries) if summaries else "NO_DATA"
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(f"\nVERDICT: {report['verdict']}")
    print(f"report: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

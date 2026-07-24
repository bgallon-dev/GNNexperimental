r"""Pool-retrieval bottleneck probe: index vs query-side decomposition.

The code-graph harness shows within-candidate wins but ~random corpus-wide
(pool) retrieval. The on-record diagnosis blames "Stage-A objective +
query-head pipeline". This probe decomposes that with ZERO training, on the
frozen encoder, by asking: if the query point were placed at the anchor's
own embedding (information parity with anchor-BFS, which gets the anchor's
graph position), how good is pool retrieval?

Arms (pool mode, ranking + abstain families, abstain sentinel excluded):
  anchor_emb  score = -d_c(emb[anchor], emb[cand])   <- the probe
  anchor_bfs  score = -hops(anchor, cand)            (harness baseline)
  random      seeded noise                           (harness baseline)
  oracle_loo  score = -min_p d_c(emb[p], emb[cand]) over positives,
              leave-one-out for positive candidates; cases with >=2
              positives only. Index ceiling: "perfectly placed query".

Pre-registered decision tree:
  1. anchor_emb >> learned head (~0.02 pool ndcg@10) and >= anchor_bfs on
     the nonlocal bucket -> query-side information deficit confirmed; the
     fix is conditioning the Stage-B head on logmap0(emb[anchor]) (the
     current query vec carries only a RANDOM 8-d identity code, which
     cannot generalize to unseen anchors under LORO).
  2. anchor_emb ~ random but oracle_loo high -> index clusters answers,
     but not around the anchor; head must learn task-specific transport.
  3. oracle_loo low -> index degenerate corpus-wide -> Stage-A
     negative-structure lever (cross-region negatives).

Also reports: AUC of P(d(anchor,pos) < d(anchor,poolneg)) in embedding vs
hop space, and a sampled Spearman rho between the two distances (is the
embedding just re-encoding BFS?).

Run from kettle-graph-reasoner/:
    py -m scripts.probe_pool_bottleneck --repo ../tutorstructure_patch \
        --ckpt runs/sweep_arch_hyp/h128_l4_seed1 \
        --out runs/probe_pool_bottleneck
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.codegraph import cases as C
from src.codegraph.harness import _build_encoder, _embed, TASKS
from src.codegraph.ingest import build_npz
from src.codegraph.metrics_ext import mrr
from src.data.corpus_dataset import _build_graph_tensors
from src.modelsv3.distance_scoring import score_from_embeddings
from src.training.metrics import ndcg_at_k, recall_at_k

from collections import deque


def _bfs(adj: list[list[int]], src: int, n: int) -> np.ndarray:
    d = np.full(n, np.inf, np.float32)
    d[src] = 0.0
    dq = deque([src])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if d[v] == np.inf:
                d[v] = d[u] + 1.0
                dq.append(v)
    d[~np.isfinite(d)] = n + 1.0
    return d


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="../tutorstructure_patch")
    ap.add_argument("--ckpt", default="runs/sweep_arch_hyp/h128_l4_seed1")
    ap.add_argument("--out", default="runs/probe_pool_bottleneck")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(args.repo)
    name = repo_dir.name
    device = torch.device(args.device)

    print(f"[1/4] ingest {name} (answer edges ablated, same as harness)")
    required = C.collect_required_edges(repo_dir, TASKS)
    cg = build_npz(repo_dir, out_dir / f"graph_{name}.npz", required)
    with np.load(cg.npz_path) as z:
        g = _build_graph_tensors(z)
        x = z["x"]

    print("[2/4] embed with frozen encoder")
    enc, cfg = _build_encoder(Path(args.ckpt), g, device)
    emb = _embed(enc, g, device)          # (N, hidden) on device
    n_nodes = cg.n_nodes

    cs_all, pools, stats = C.load_repo_cases(repo_dir, cg, x, TASKS, name)
    ranking_cases = [c for c in cs_all
                     if c.task_family in ("ranking", "abstain_ranking")]
    print(f"    {len(ranking_cases)} ranking-family cases, "
          f"{n_nodes} nodes")

    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    ei = g["edge_index"].numpy()
    for s, t in zip(ei[0], ei[1]):
        adj[int(s)].append(int(t))
        adj[int(t)].append(int(s))
    dcache: dict[int, np.ndarray] = {}

    def hops(src: int) -> np.ndarray:
        if src not in dcache:
            dcache[src] = _bfs(adj, src, n_nodes)
        return dcache[src]

    print("[3/4] score all arms (zero training)")
    rows: list[dict] = []
    auc_emb_hits = auc_emb_tot = 0
    auc_hop_hits = auc_hop_tot = 0
    rho_samples: list[float] = []
    rng_auc = np.random.default_rng(args.seed)

    for cs in ranking_cases:
        posset = {r for r in cs.pos_rows if r != C.ABSTAIN_ROW}
        if not posset:
            continue
        pool_rows = pools.get(cs.task, np.empty(0, np.int64)).tolist()
        cand = sorted(set(pool_rows) | posset)
        if len(cand) <= len(posset):
            continue
        cand_t = torch.tensor(cand, dtype=torch.long, device=device)
        cand_emb = emb[cand_t]
        lab = torch.tensor([1.0 if r in posset else 0.0 for r in cand])

        d_q = hops(cs.query_row)
        min_hop = min(float(d_q[r]) for r in posset)
        locality = "local" if min_hop <= 1.0 else "nonlocal"

        # arm: anchor_emb
        qp = emb[cs.query_row]
        sc_emb = score_from_embeddings(cand_emb, qp, c=enc.c).cpu()
        # arm: anchor_bfs
        sc_bfs = torch.from_numpy(
            np.array([-d_q[r] for r in cand], dtype=np.float32))
        # arm: random (same convention as harness)
        seed = int.from_bytes(cs.case_id.encode()[-8:], "little")
        sc_rnd = torch.from_numpy(
            np.random.default_rng(seed)
            .standard_normal(len(cand)).astype(np.float32))
        # arm: oracle_loo (>=2 positives only)
        sc_orc = None
        if len(posset) >= 2:
            pos_list = sorted(posset)
            pos_emb = emb[torch.tensor(pos_list, dtype=torch.long,
                                       device=device)]
            # dist matrix candidates x positives via score (=-dist)
            dmat = torch.stack(
                [-score_from_embeddings(cand_emb, pos_emb[j], c=enc.c)
                 for j in range(len(pos_list))], dim=1).cpu()
            pos_idx_in_cand = {r: i for i, r in enumerate(cand)}
            big = torch.finfo(dmat.dtype).max
            for j, p in enumerate(pos_list):
                i = pos_idx_in_cand.get(p)
                if i is not None:
                    dmat[i, j] = big      # leave-one-out: no self-reference
            sc_orc = -dmat.min(dim=1).values

        row = {"task": cs.task, "locality": locality,
               "n_cand": len(cand), "n_pos": len(posset)}
        for arm, sc in (("anchor_emb", sc_emb), ("anchor_bfs", sc_bfs),
                        ("random", sc_rnd), ("oracle_loo", sc_orc)):
            if sc is None:
                continue
            row[f"{arm}_ndcg10"] = ndcg_at_k(sc, lab, 10)
            row[f"{arm}_mrr"] = mrr(sc, lab)
            row[f"{arm}_r10"] = recall_at_k(sc, lab, 10)
        rows.append(row)

        # AUC + rho diagnostics on a sampled negative set
        negs = [r for r in cand if r not in posset]
        if negs:
            neg_s = rng_auc.choice(negs, size=min(64, len(negs)),
                                   replace=False)
            d_emb_all = (-score_from_embeddings(
                emb, qp, c=enc.c)).cpu().numpy()
            for p in posset:
                for ng in neg_s:
                    auc_emb_tot += 1
                    auc_hop_tot += 1
                    if d_emb_all[p] < d_emb_all[ng]:
                        auc_emb_hits += 1
                    if d_q[p] < d_q[ng]:
                        auc_hop_hits += 1
            samp = np.concatenate([list(posset), neg_s])
            finite = samp[d_q[samp] < n_nodes]
            if len(finite) >= 8:
                rho_samples.append(
                    _spearman(d_emb_all[finite], d_q[finite]))

    print("[4/4] aggregate + report")
    arms = ("anchor_emb", "anchor_bfs", "random", "oracle_loo")

    def _agg(sub: list[dict]) -> dict:
        out: dict = {"n": len(sub)}
        for arm in arms:
            vals = [r[f"{arm}_ndcg10"] for r in sub
                    if f"{arm}_ndcg10" in r]
            if vals:
                out[arm] = {
                    "ndcg@10": sum(vals) / len(vals),
                    "mrr": sum(r[f"{arm}_mrr"] for r in sub
                               if f"{arm}_mrr" in r) / len(vals),
                    "r@10": sum(r[f"{arm}_r10"] for r in sub
                                if f"{arm}_r10" in r) / len(vals),
                    "n": len(vals),
                }
        return out

    report = {
        "config": vars(args),
        "repo_stats": {"n_nodes": n_nodes, "n_cases": len(rows)},
        "overall": _agg(rows),
        "by_locality": {
            loc: _agg([r for r in rows if r["locality"] == loc])
            for loc in ("local", "nonlocal")
        },
        "by_task": {
            t: _agg([r for r in rows if r["task"] == t])
            for t in sorted({r["task"] for r in rows})
        },
        "diagnostics": {
            "auc_emb_pos_closer_than_neg":
                auc_emb_hits / max(auc_emb_tot, 1),
            "auc_hop_pos_closer_than_neg":
                auc_hop_hits / max(auc_hop_tot, 1),
            "spearman_embdist_vs_hopdist_mean":
                float(np.mean(rho_samples)) if rho_samples else None,
            "n_rho_samples": len(rho_samples),
        },
    }
    (out_dir / "probe_results.json").write_text(
        json.dumps(report, indent=2))

    def _tbl(title: str, agg: dict) -> None:
        print(f"\n=== {title} (n={agg['n']}) ===")
        print(f"{'arm':<12} {'ndcg@10':>8} {'mrr':>8} {'r@10':>8} {'n':>6}")
        for arm in arms:
            if arm in agg:
                a = agg[arm]
                print(f"{arm:<12} {a['ndcg@10']:>8.3f} {a['mrr']:>8.3f} "
                      f"{a['r@10']:>8.3f} {a['n']:>6}")

    _tbl("overall", report["overall"])
    for loc in ("local", "nonlocal"):
        _tbl(f"locality={loc}", report["by_locality"][loc])
    d = report["diagnostics"]
    print(f"\nAUC pos-closer-than-neg: emb={d['auc_emb_pos_closer_than_neg']:.3f} "
          f"hop={d['auc_hop_pos_closer_than_neg']:.3f}")
    rho = d["spearman_embdist_vs_hopdist_mean"]
    print(f"spearman(emb-dist, hop-dist) mean over cases: "
          f"{rho:.3f}" if rho is not None else "spearman: n/a")
    print(f"\nreport: {out_dir / 'probe_results.json'}")


if __name__ == "__main__":
    main()

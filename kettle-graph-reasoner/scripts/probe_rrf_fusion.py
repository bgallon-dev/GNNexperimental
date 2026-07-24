r"""Zero-training rank fusion of the two complementary pool baselines.

Follow-up to the blend experiment's honest negative: LEARNED composition
of [d_emb, d_hop, feats] lands BELOW its own zero-training inputs (train
pair_acc 0.87+ vs 32 sampled negs, pool ndcg 0.039 < bfs 0.070 — a
train->eval negative-set distribution shift). This probe removes training
from the equation entirely: reciprocal-rank fusion (RRF) of the
anchor_emb and anchor_bfs rankings,

    score(cand) = 1/(k + rank_emb(cand)) + 1/(k + rank_bfs(cand))

with the standard k=60. If the complementarity found by the probe
(spearman 0.107, per-task wins split) is real and composable, RRF should
beat BOTH inputs on the pool with zero training. Also reports the
oracle_loo reference and per-task cells.

Run from kettle-graph-reasoner/:
    py -m scripts.probe_rrf_fusion --repo ../tutorstructure_patch \
        --ckpt frozen/kgr-v1.0-2026-07-07/encoder_baseline \
        --out runs/probe_rrf_fusion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.codegraph import cases as C
from src.codegraph.metrics_ext import mrr
from src.training.metrics import ndcg_at_k, recall_at_k

from scripts.blend_pool_experiment import _Ctx

RRF_K = 60.0


def _ranks(scores: torch.Tensor) -> torch.Tensor:
    """1-based rank of each entry under descending-score order."""
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(scores))
    return ranks.float() + 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="../tutorstructure_patch")
    ap.add_argument("--ckpt",
                    default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--out", default="runs/probe_rrf_fusion")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = _Ctx(Path(args.repo), Path(args.ckpt), out_dir,
               torch.device(args.device))
    # Same split as the blend experiment so cells are comparable.
    C.assign_file_split(ctx.cases, args.split_seed, (0.70, 0.15, 0.15))
    eval_cases = [c for c in ctx.cases if c.split in ("val", "test")]
    print(f"{len(eval_cases)} eval cases")

    rows = []
    for cs in eval_cases:
        posset = {r for r in cs.pos_rows if r != C.ABSTAIN_ROW}
        pool_rows = ctx.pools.get(cs.task, np.empty(0, np.int64)).tolist()
        cand = sorted(set(pool_rows) | posset)
        if len(cand) <= len(posset):
            continue
        rows_t = torch.tensor(cand, dtype=torch.long, device=ctx.device)
        lab = torch.tensor([1.0 if r in posset else 0.0 for r in cand])
        d_q = ctx.hops(cs.query_row)
        finite_pos = [float(d_q[r]) for r in posset]
        min_hop = min(finite_pos) if finite_pos else np.inf
        locality = "local" if min_hop <= 1.0 else "nonlocal"

        sc_emb = -ctx.d_emb(cs.query_row, rows_t).cpu()
        hp = d_q[cand]
        hp = np.where(np.isfinite(hp), hp, ctx.n_nodes + 1.0)
        sc_bfs = torch.from_numpy((-hp).astype(np.float32))
        sc_rrf = (1.0 / (RRF_K + _ranks(sc_emb))
                  + 1.0 / (RRF_K + _ranks(sc_bfs)))

        row = {"task": cs.task, "split": cs.split, "locality": locality}
        for arm, sc in (("anchor_emb", sc_emb), ("anchor_bfs", sc_bfs),
                        ("rrf", sc_rrf)):
            row[f"{arm}_ndcg10"] = ndcg_at_k(sc, lab, 10)
            row[f"{arm}_mrr"] = mrr(sc, lab)
            row[f"{arm}_r10"] = recall_at_k(sc, lab, 10)
            row[f"{arm}_r50"] = recall_at_k(sc, lab, 50)
        rows.append(row)

    arms = ("anchor_emb", "anchor_bfs", "rrf")

    def _agg(sub):
        out = {"n": len(sub)}
        for arm in arms:
            vals = [r[f"{arm}_ndcg10"] for r in sub]
            if vals:
                out[arm] = {
                    k: sum(r[f"{arm}_{m}"] for r in sub) / len(vals)
                    for k, m in (("ndcg@10", "ndcg10"), ("mrr", "mrr"),
                                 ("r@10", "r10"), ("r@50", "r50"))
                }
        return out

    report = {"config": vars(args), "overall": _agg(rows)}
    for split in ("val", "test"):
        for loc in ("local", "nonlocal"):
            sub = [r for r in rows
                   if r["split"] == split and r["locality"] == loc]
            report[f"{split}|{loc}"] = _agg(sub)
    for task in sorted({r["task"] for r in rows}):
        report[f"task:{task}"] = _agg(
            [r for r in rows if r["task"] == task])
    (out_dir / "rrf_results.json").write_text(json.dumps(report, indent=2))

    def _tbl(title, agg):
        print(f"\n=== {title} (n={agg['n']}) ===")
        print(f"{'arm':<12} {'ndcg@10':>8} {'mrr':>8} {'r@10':>8} {'r@50':>8}")
        for arm in arms:
            if arm in agg:
                a = agg[arm]
                print(f"{arm:<12} {a['ndcg@10']:>8.3f} {a['mrr']:>8.3f} "
                      f"{a['r@10']:>8.3f} {a['r@50']:>8.3f}")

    _tbl("overall", report["overall"])
    _tbl("test|nonlocal (headline)", report["test|nonlocal"])
    for task in sorted({r["task"] for r in rows}):
        _tbl(f"task {task}", report[f"task:{task}"])
    print(f"\nreport: {out_dir / 'rrf_results.json'}")


if __name__ == "__main__":
    main()

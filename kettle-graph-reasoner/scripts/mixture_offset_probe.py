r"""Plumbing step 1: are answer-cluster offsets TASK-CONSISTENT?

The mixture oracle (0.85) knows the labels. The learnable question: do
answer clusters sit at consistent positions RELATIVE TO THE ANCHOR across
cases of a task? Test with zero gradient training:

  TRAIN cases (70%): per case, translate positives into the anchor's
  gyro-frame (mobius_add(-anchor, p)), logmap0 -> tangent offsets; kmeans
  K=3 per case; pool all case-centers per task; kmeans K=3 again ->
  K task-level offset vectors.
  TEST cases (15%+15%): query points = mobius_add(anchor, expmap0(v_k));
  score pool candidates by max_k -dist. ndcg@10, nonlocal bucket.

Arms: offset_mix (the experiment), offset_1 (K=1), anchor_emb (v=0
baseline), plus references: bfs 0.070, blend 0.095, label-kmeans oracle
0.83-0.86. VERDICT RULE: offset_mix > 0.095 -> ship the mixture head
(typed offsets ARE the plumbing fix); offset_mix ~ anchor_emb ->
offsets inconsistent -> relational/motif feature route.

    py -m scripts.mixture_offset_probe
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.codegraph import cases as C
from src.modelsv2.layers import poincare_ops as P
from src.modelsv3.distance_scoring import score_from_embeddings
from src.training.metrics import ndcg_at_k

from scripts.blend_pool_experiment import _Ctx

K = 3
OUT = Path("runs/mixture_offset_probe")


def _kmeans(x: torch.Tensor, k: int, iters: int = 15) -> torch.Tensor:
    k = min(k, x.shape[0])
    idx = torch.randperm(x.shape[0])[:k]
    cents = x[idx].clone()
    for _ in range(iters):
        asg = torch.cdist(x, cents).argmin(1)
        for j in range(k):
            m = asg == j
            if m.any():
                cents[j] = x[m].mean(0)
    return cents


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    ctx = _Ctx(Path("../tutorstructure_patch"),
               Path("frozen/kgr-v1.0-2026-07-07/encoder_baseline"),
               OUT, torch.device("cpu"))
    C.assign_file_split(ctx.cases, 0, (0.70, 0.15, 0.15))
    c = ctx.enc.c
    emb = ctx.emb

    def anchor_frame_offsets(anchor: int, rows: torch.Tensor):
        """tangent offsets of rows in the anchor's gyro-frame."""
        shifted = P.mobius_add(-emb[anchor], emb[rows], c)
        return P.logmap0(shifted, c)

    # fit task-level offsets on train cases
    per_task: dict[str, list] = {}
    for cs in ctx.cases:
        if cs.split != "train":
            continue
        pos = [r for r in cs.pos_rows if r != C.ABSTAIN_ROW]
        if len(pos) < 1:
            continue
        offs = anchor_frame_offsets(cs.query_row,
                                    torch.tensor(sorted(set(pos))))
        per_task.setdefault(cs.task, []).append(_kmeans(offs, K))
    task_offsets = {}
    for task, cents in per_task.items():
        pooled = torch.cat(cents, dim=0)
        task_offsets[task] = _kmeans(pooled, K)
        print(f"  {task}: {len(cents)} train cases, "
              f"offset norms {[f'{v.norm():.2f}' for v in task_offsets[task]]}")

    rows = []
    for cs in ctx.cases:
        if cs.split not in ("val", "test") or cs.task not in task_offsets:
            continue
        posset = {r for r in cs.pos_rows if r != C.ABSTAIN_ROW}
        if not posset:
            continue
        pool = ctx.pools.get(cs.task, np.empty(0, np.int64)).tolist()
        cand = sorted(set(pool) | posset)
        if len(cand) <= len(posset):
            continue
        ct = torch.tensor(cand)
        lab = torch.tensor([1.0 if r in posset else 0.0 for r in cand])
        d_q = ctx.hops(cs.query_row)
        loc = ("local" if min(float(d_q[r]) for r in posset) <= 1
               else "nonlocal")
        a = cs.query_row
        V = task_offsets[cs.task]
        qps = P.mobius_add(emb[a].unsqueeze(0), P.expmap0(V, c), c)
        sc_mix = torch.stack(
            [score_from_embeddings(emb[ct], q, c=c) for q in qps]
        ).max(0).values
        v1 = V.mean(0, keepdim=True)
        qp1 = P.mobius_add(emb[a].unsqueeze(0), P.expmap0(v1, c), c)[0]
        sc_1 = score_from_embeddings(emb[ct], qp1, c=c)
        sc_a = score_from_embeddings(emb[ct], emb[a], c=c)
        rows.append({"task": cs.task, "loc": loc,
                     "offset_mix": ndcg_at_k(sc_mix, lab, 10),
                     "offset_1": ndcg_at_k(sc_1, lab, 10),
                     "anchor_emb": ndcg_at_k(sc_a, lab, 10)})

    arms = ("offset_mix", "offset_1", "anchor_emb")
    report = {}
    print(f"\n=== offset-consistency probe (held-out cases) ndcg@10 ===")
    print(f"{'cell':<38}{'n':>5} " + " ".join(f"{a:>11}" for a in arms))
    for loc in ("nonlocal", "local"):
        sub = [r for r in rows if r["loc"] == loc]
        cell = {a: sum(r[a] for r in sub) / max(len(sub), 1) for a in arms}
        report[loc] = cell | {"n": len(sub)}
        print(f"{'ALL|' + loc:<38}{len(sub):>5} "
              + " ".join(f"{cell[a]:>11.3f}" for a in arms))
    for task in sorted({r["task"] for r in rows}):
        sub = [r for r in rows if r["task"] == task and r["loc"] == "nonlocal"]
        if not sub:
            continue
        cell = {a: sum(r[a] for r in sub) / len(sub) for a in arms}
        report[f"{task}|nonlocal"] = cell | {"n": len(sub)}
        print(f"{task + '|nonlocal':<38}{len(sub):>5} "
              + " ".join(f"{cell[a]:>11.3f}" for a in arms))
    print("\nreferences: bfs 0.070 | blend 0.095 | label-kmeans oracle ~0.85")
    (OUT / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

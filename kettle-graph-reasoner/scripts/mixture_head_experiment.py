r"""Trained K-point mixture head for pool retrieval (the plumbing build).

Follows the offset-consistency probe (kmeans task offsets = 0.169
nonlocal, zero training). Two arms, per task, trained with the validated
hinge + 256-pool-negative recipe:

  task_mix   K=3 learnable tangent offsets V_k per task;
             qp_k = anchor (+) expmap0(v_k); score = max_k -dist
  cond_mix   offsets = V_k + W_k @ logmap0(emb[anchor]) — anchor-
             conditioned residual (the next rung toward the 0.85 oracle)

Bars (test|nonlocal pool ndcg@10): > 0.169 (kmeans-fit offsets);
references: blend 0.095, anchor_emb 0.118, label-kmeans oracle ~0.85.

    py -m scripts.mixture_head_experiment --seeds 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.codegraph import cases as C
from src.modelsv2.layers import poincare_ops as P
from src.modelsv3.distance_scoring import score_from_embeddings
from src.training.metrics import ndcg_at_k

from scripts.blend_pool_experiment import _Ctx, _train_cand, _pairwise_hinge

K = 3
OFF_CLAMP = 3.0


class MixtureHead(nn.Module):
    def __init__(self, d: int, conditioned: bool):
        super().__init__()
        self.V = nn.Parameter(torch.randn(K, d) * 0.1)
        self.cond = conditioned
        if conditioned:
            self.W = nn.Linear(d, K * d)
            nn.init.zeros_(self.W.weight)
            nn.init.zeros_(self.W.bias)

    def offsets(self, anchor_tan: torch.Tensor) -> torch.Tensor:
        v = self.V
        if self.cond:
            v = v + self.W(anchor_tan).view(K, -1)
        n = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return v * (n.clamp_max(OFF_CLAMP) / n)

    def score(self, ctx, anchor: int, cand_emb: torch.Tensor):
        a = ctx.emb[anchor]
        v = self.offsets(P.logmap0(a, ctx.enc.c))
        qps = P.mobius_add(a.unsqueeze(0), P.expmap0(v, ctx.enc.c),
                           ctx.enc.c)
        return torch.stack(
            [score_from_embeddings(cand_emb, q, c=ctx.enc.c) for q in qps]
        ).max(0).values


def train_head(ctx, train_cases, conditioned, seed, epochs, lr):
    torch.manual_seed(seed)
    head = MixtureHead(ctx.emb.shape[1], conditioned)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        tot, acc, nb = 0.0, 0.0, 0
        for idx in rng.permutation(len(train_cases)):
            cs = train_cases[idx]
            cand, posset = _train_cand(ctx, cs, rng, neg_sample=256,
                                       cand_cap=288)
            if not cand:
                continue
            rows = torch.tensor(cand)
            lab = torch.tensor([1.0 if r in posset else 0.0 for r in cand])
            sc = head.score(ctx, cs.query_row, ctx.emb[rows].detach())
            loss, pa = _pairwise_hinge(sc, lab, 0.5)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            acc += pa
            nb += 1
        if ep in (0, epochs - 1):
            print(f"      [ep {ep}] loss={tot/max(nb,1):.4f} "
                  f"pair_acc={acc/max(nb,1):.3f}")
    head.eval()
    return head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--out", default="runs/mixture_head")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ctx = _Ctx(Path("../tutorstructure_patch"),
               Path("frozen/kgr-v1.0-2026-07-07/encoder_baseline"),
               out, torch.device("cpu"))
    C.assign_file_split(ctx.cases, 0, (0.70, 0.15, 0.15))
    tasks = sorted({c.task for c in ctx.cases})
    headline = {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        print(f"[seed {seed}]")
        heads = {}
        for task in tasks:
            tr = [c for c in ctx.cases
                  if c.split == "train" and c.task == task]
            if not tr:
                continue
            print(f"    {task} task_mix (n={len(tr)})")
            heads[(task, False)] = train_head(
                ctx, tr, False, seed, args.epochs, args.lr)
            print(f"    {task} cond_mix")
            heads[(task, True)] = train_head(
                ctx, tr, True, seed, args.epochs, args.lr)
        rows = []
        with torch.no_grad():
            for cs in ctx.cases:
                if cs.split not in ("val", "test"):
                    continue
                posset = {r for r in cs.pos_rows if r != C.ABSTAIN_ROW}
                if not posset:
                    continue
                pool = ctx.pools.get(cs.task,
                                     np.empty(0, np.int64)).tolist()
                cand = sorted(set(pool) | posset)
                if len(cand) <= len(posset):
                    continue
                ct = torch.tensor(cand)
                lab = torch.tensor(
                    [1.0 if r in posset else 0.0 for r in cand])
                d_q = ctx.hops(cs.query_row)
                loc = ("local"
                       if min(float(d_q[r]) for r in posset) <= 1
                       else "nonlocal")
                row = {"loc": loc}
                for nm, cond in (("task_mix", False), ("cond_mix", True)):
                    h = heads.get((cs.task, cond))
                    if h is None:
                        continue
                    sc = h.score(ctx, cs.query_row, ctx.emb[ct])
                    row[nm] = ndcg_at_k(sc, lab, 10)
                row["anchor_emb"] = ndcg_at_k(
                    score_from_embeddings(ctx.emb[ct],
                                          ctx.emb[cs.query_row],
                                          c=ctx.enc.c), lab, 10)
                rows.append(row)
        for loc in ("nonlocal", "local"):
            sub = [r for r in rows if r["loc"] == loc]
            cell = {a: sum(r.get(a, 0.0) for r in sub) / max(len(sub), 1)
                    for a in ("task_mix", "cond_mix", "anchor_emb")}
            if loc == "nonlocal":
                headline.setdefault("task_mix", []).append(
                    cell["task_mix"])
                headline.setdefault("cond_mix", []).append(
                    cell["cond_mix"])
            print(f"  seed {seed} {loc} (n={len(sub)}): "
                  + "  ".join(f"{a}={cell[a]:.3f}"
                              for a in cell))
    summary = {a: {"mean": float(np.mean(v)),
                   "std": float(np.std(v)), "per_seed": v}
               for a, v in headline.items()}
    summary["bars"] = {"kmeans_probe": 0.169, "blend": 0.095,
                       "oracle": 0.85}
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nBAR (> 0.169 kmeans offsets): task_mix "
          f"{summary['task_mix']['mean']:.3f}, cond_mix "
          f"{summary['cond_mix']['mean']:.3f}")


if __name__ == "__main__":
    main()

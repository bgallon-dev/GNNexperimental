r"""True point-family ceiling: optimal per-case query point (Frechet-
mean approx + gradient refinement) vs the oracle_loo lower bound (0.117).
If mean-point >> loo, the point family has headroom (typed offsets /
mixtures); if ~= loo, the family is exhausted and only relational
features or Stage-A changes can break the ceiling.

E2 extension: works on euclidean-control checkpoints too — geometry is
read from the checkpoint config (logmap0/expmap0 become identity, L2
scoring). Hyperbolic path unchanged.

    py -m scripts.probe_true_point_ceiling [--ckpt <run_dir>]"""
from pathlib import Path
import numpy as np, torch, json
from src.codegraph import cases as C
from src.modelsv3.distance_scoring import score_from_embeddings
from src.modelsv2.layers import poincare_ops as P
from src.training.metrics import ndcg_at_k
from scripts.blend_pool_experiment import _Ctx

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
ap.add_argument("--out", default="runs/probe_true_point_ceiling")
ap.add_argument("--repo", default="../tutorstructure_patch")
args = ap.parse_args()
ctx = _Ctx(Path(args.repo), Path(args.ckpt),
           Path(args.out), torch.device("cpu"))
EUC = ctx.euclidean
if EUC:
    print("[ceiling] euclidean geometry (identity maps, L2 scoring)")


def _log0(x):
    return x if EUC else P.logmap0(x, ctx.c)


def _exp0(v):
    return v if EUC else P.expmap0(v, ctx.c)


def _score(e, q):
    return score_from_embeddings(e, q, c=ctx.c, euclidean=EUC)


rows = []
K_MIX = 3
for cs in ctx.cases:
    posset = {r for r in cs.pos_rows if r != C.ABSTAIN_ROW}
    if len(posset) < 2: continue
    pool = ctx.pools.get(cs.task, np.empty(0, np.int64)).tolist()
    cand = sorted(set(pool) | posset)
    if len(cand) <= len(posset): continue
    ct = torch.tensor(cand)
    lab = torch.tensor([1.0 if r in posset else 0.0 for r in cand])
    d_q = ctx.hops(cs.query_row)
    loc = "local" if min(float(d_q[r]) for r in posset) <= 1 else "nonlocal"
    pos_t = torch.tensor(sorted(posset))
    # arm 1: tangent-mean point (Frechet approx; plain mean in euclidean)
    mean_pt = _exp0(_log0(ctx.emb[pos_t]).mean(0))
    sc_mean = _score(ctx.emb[ct], mean_pt)
    # arm 2: gradient-refined optimal point (60 steps, softmax surrogate)
    v = _log0(ctx.emb[pos_t]).mean(0).clone().requires_grad_()
    opt = torch.optim.Adam([v], lr=0.05)
    ce = ctx.emb[ct].detach()
    labd = lab.detach()
    for _ in range(60):
        qp = _exp0(v)
        sc = _score(ce, qp)
        # listwise surrogate: maximize mass of positives in softmax
        logp = torch.log_softmax(sc * 4.0, dim=0)
        loss = -(logp[labd >= 0.5]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        sc_opt = _score(ce, _exp0(v))
    # arm 3: K-point mixture oracle (kmeans-ish on positives, max-score)
    with torch.no_grad():
        ptan = _log0(ctx.emb[pos_t])
        k = min(K_MIX, len(posset))
        # Per-case seeded init: the unseeded variant has +-0.01-0.03
        # SAME-CHECKPOINT aggregate noise (measured 2026-07-10: frozen
        # v1.0 gave 0.8325/0.8581/0.8323 across runs), which swamps the
        # +-0.008 comparison band. Deterministic per-case seeding makes
        # the lens reproducible; numbers from before this change carry
        # that noise and are not band-grade references.
        gen = torch.Generator().manual_seed(
            hash((cs.case_id, "mix_init")) & 0x7FFFFFFF)
        cidx = torch.randperm(len(posset), generator=gen)[:k]
        cents = ptan[cidx].clone()
        for _ in range(10):
            d2 = torch.cdist(ptan, cents)
            asg = d2.argmin(1)
            for j in range(k):
                m = asg == j
                if m.any(): cents[j] = ptan[m].mean(0)
        pts = _exp0(cents)
        sc_mix = torch.stack([_score(ce, p) for p in pts]).max(0).values
    rows.append({"loc": loc,
                 "mean": ndcg_at_k(sc_mean, lab, 10),
                 "opt": ndcg_at_k(sc_opt, lab, 10),
                 "mix": ndcg_at_k(sc_mix, lab, 10)})
out = {}
for loc in ("nonlocal", "local"):
    sub = [r for r in rows if r["loc"] == loc]
    out[loc] = {k: sum(r[k] for r in sub)/max(len(sub),1)
                for k in ("mean","opt","mix")} | {"n": len(sub)}
    print(loc, out[loc])
print("(oracle_loo reference: nonlocal 0.115-0.117)")
Path(args.out).mkdir(parents=True, exist_ok=True)
open(Path(args.out)/"results.json","w").write(json.dumps(out))

r"""P5 — per-query geometric routing probe (Docs/GEOMETRY_READOUT_PROBES_PLAN.md).

The shipped suggester router picks blend vs mixture per TASK on val|nonlocal
(routed pair 0.1002 +- 0.0021 test|nonlocal). Does ball geometry around the
anchor predict the winner per QUERY?

Reuses the exact _Ctx build of blend_pool_experiment (same repo, ckpt,
split seed) so case order matches the persisted rows_seed{0,1,2}.json;
alignment is asserted on (task, split, locality) per row. Features are
training-free geometry stats; stage-1 is a Spearman diagnostic on VAL
rows only, stage-2 fits a 1-feature threshold router on VAL and evaluates
on test|nonlocal — do-no-harm fallback to the per-task choice.

    py -m scripts.probe_geometry_routing --out runs/geometry_probes/p5_routing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.blend_pool_experiment import _Ctx
from src.codegraph import cases as C
from src.modelsv3.geometry_readout import hyperbolic_dispersion

TOPK = 32
FEATURES = ("anchor_radius", "knn_dist_mean", "knn_dist_std",
            "knn_pairwise_mean", "knn_eccentricity", "min_hop_pool")
PAIRS = {"pair": ("blend", "mixture"),
         "pair_typed": ("blend", "mixture_typed")}


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def _spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = _rankdata(x), _rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _case_features(ctx, cs) -> dict[str, float]:
    a = cs.query_row
    pool = ctx.pools.get(cs.task, np.empty(0, np.int64))
    pool = pool[pool != a]
    rows = torch.tensor(pool.tolist(), dtype=torch.long)
    d = ctx.d_emb(a, rows)
    k = min(TOPK, len(pool))
    top = torch.topk(-d, k=k).indices
    d_top = d[top]
    emb_top = ctx.emb[rows[top]]
    disp = hyperbolic_dispersion(
        emb_top, c=ctx.c,
        generator=torch.Generator().manual_seed(int(a)))
    hp = ctx.hops(a)[pool]
    hp_fin = hp[np.isfinite(hp)]
    return {
        "anchor_radius": float(ctx.radius[a]),
        "knn_dist_mean": float(d_top.mean()),
        "knn_dist_std": float(d_top.std(unbiased=False)),
        "knn_pairwise_mean": disp["pairwise_mean"],
        "knn_eccentricity": disp["eccentricity"],
        "min_hop_pool": float(hp_fin.min()) if len(hp_fin) else 99.0,
    }


def _task_route(rows, arms):
    """Shipped per-task router (blend_pool_experiment._route semantics)."""
    default = arms[0]
    choice = {}
    for t in sorted({r["task"] for r in rows}):
        sub = [r for r in rows if r["task"] == t and r["split"] == "val"
               and r["locality"] == "nonlocal"
               and all(f"{a}_ndcg10" in r for a in arms)]
        if len(sub) >= 5:
            means = {a: np.mean([r[f"{a}_ndcg10"] for r in sub]) for a in arms}
            choice[t] = max(means, key=means.get)
        else:
            choice[t] = default
    return choice


def _routed_mean(rows, pick_fn, arms):
    default = arms[0]
    test = [r for r in rows if r["split"] == "test"
            and r["locality"] == "nonlocal"]
    vals = [r.get(f"{pick_fn(r)}_ndcg10", r.get(f"{default}_ndcg10"))
            for r in test]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _fit_query_router(rows, feat, arms):
    """Per task: threshold on ``feat`` with an arm on each side, fit on
    val|nonlocal; keep only if it beats the task's best single arm on
    val (do-no-harm)."""
    per_task = {}
    for t in sorted({r["task"] for r in rows}):
        sub = [r for r in rows if r["task"] == t and r["split"] == "val"
               and r["locality"] == "nonlocal"
               and all(f"{a}_ndcg10" in r for a in arms)]
        if len(sub) < 10:
            per_task[t] = None
            continue
        f = np.array([r["_feat"][feat] for r in sub])
        best_single = max(
            float(np.mean([r[f"{a}_ndcg10"] for r in sub])) for a in arms)
        best = None
        for q in (0.25, 0.5, 0.75):
            thr = float(np.quantile(f, q))
            lo, hi = f <= thr, f > thr
            if lo.sum() < 3 or hi.sum() < 3:
                continue
            for a_lo in arms:
                for a_hi in arms:
                    if a_lo == a_hi:
                        continue
                    m = (np.mean([r[f"{a_lo}_ndcg10"]
                                  for r, s in zip(sub, lo) if s])
                         * lo.mean()
                         + np.mean([r[f"{a_hi}_ndcg10"]
                                    for r, s in zip(sub, hi) if s])
                         * hi.mean())
                    if best is None or m > best[0]:
                        best = (float(m), thr, a_lo, a_hi)
        per_task[t] = (best if best is not None and best[0] > best_single
                       else None)
    return per_task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="../tutorstructure_patch")
    ap.add_argument("--ckpt", default="runs/width-h32-hyp-l4-s0")
    ap.add_argument("--rows-dir", default="runs/blend_h32_suggester")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="runs/geometry_probes/p5_routing")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    print("[1/3] rebuild ctx (must match the h32 suggester run)")
    ctx = _Ctx(Path(args.repo), Path(args.ckpt), out_dir, torch.device("cpu"))
    C.assign_file_split(ctx.cases, args.split_seed, (0.70, 0.15, 0.15))
    eval_cases = [c for c in ctx.cases if c.split in ("val", "test")]

    # replay _eval_arms' skip rule + locality so ordering matches rows files
    kept = []
    for cs in eval_cases:
        posset = {r for r in cs.pos_rows if r != C.ABSTAIN_ROW}
        pool_rows = ctx.pools.get(cs.task, np.empty(0, np.int64)).tolist()
        cand = sorted(set(pool_rows) | posset)
        if len(cand) <= len(posset):
            continue
        d_q = ctx.hops(cs.query_row)
        fin = [float(d_q[r]) for r in posset]
        min_hop = min(fin) if fin else np.inf
        kept.append((cs, "local" if min_hop <= 1.0 else "nonlocal"))

    print(f"[2/3] geometry features for {len(kept)} cases")
    feats = [_case_features(ctx, cs) for cs, _ in kept]

    report = {"config": vars(args), "n_cases": len(kept),
              "stage1": {}, "stage2": {}}
    for seed in seeds:
        rows = json.loads(
            (Path(args.rows_dir) / f"rows_seed{seed}.json").read_text())
        assert len(rows) == len(kept), (len(rows), len(kept))
        for r, (cs, loc), fv in zip(rows, kept, feats):
            assert r["task"] == cs.task and r["split"] == cs.split \
                and r["locality"] == loc, "case alignment broke"
            r["_feat"] = fv

        # ---- stage 1: Spearman diagnostics on VAL rows only
        s1 = {}
        for pname, arms in PAIRS.items():
            val = [r for r in rows if r["split"] == "val"
                   and r["locality"] == "nonlocal"
                   and all(f"{a}_ndcg10" in r for a in arms)]
            delta = [r[f"{arms[0]}_ndcg10"] - r[f"{arms[1]}_ndcg10"]
                     for r in val]
            s1[pname] = {feat: _spearman(
                [r["_feat"][feat] for r in val], delta) for feat in FEATURES}
        report["stage1"][str(seed)] = s1

        # ---- fail-branch abstain check: feature vs RAW per-row ndcg
        abstain = {}
        for arm in ("blend", "mixture_typed"):
            val = [r for r in rows if r["split"] == "val"
                   and r["locality"] == "nonlocal"
                   and f"{arm}_ndcg10" in r]
            abstain[arm] = {feat: _spearman(
                [r["_feat"][feat] for r in val],
                [r[f"{arm}_ndcg10"] for r in val]) for feat in FEATURES}
        report.setdefault("stage1_abstain", {})[str(seed)] = abstain

        # ---- stage 2: per-query threshold router vs per-task router
        s2 = {}
        for pname, arms in PAIRS.items():
            task_choice = _task_route(rows, arms)
            task_routed = _routed_mean(
                rows, lambda r: task_choice[r["task"]], arms)
            best_feat = None
            for feat in FEATURES:
                fit = _fit_query_router(rows, feat, arms)

                def pick(r, fit=fit, tc=task_choice):
                    cfg = fit.get(r["task"])
                    if cfg is None:
                        return tc[r["task"]]
                    _, thr, a_lo, a_hi = cfg
                    return a_lo if r["_feat"][feat] <= thr else a_hi

                m = _routed_mean(rows, pick, arms)
                if best_feat is None or m > best_feat[1]:
                    best_feat = (feat, m)
            s2[pname] = {"task_routed_test": task_routed,
                         "best_feature": best_feat[0],
                         "query_routed_test": best_feat[1]}
        report["stage2"][str(seed)] = s2

    print("[3/3] summary")
    for pname in PAIRS:
        print(f"\n-- {pname} ({' vs '.join(PAIRS[pname])}) --")
        print("stage1 spearman(feature, per-row delta) on val|nonlocal:")
        for feat in FEATURES:
            vals = [report["stage1"][str(s)][pname][feat] for s in seeds]
            sign_ok = len({np.sign(v) for v in vals if abs(v) > 1e-12}) == 1
            print(f"  {feat:<18} " + " ".join(f"{v:+.3f}" for v in vals)
                  + ("  [consistent]" if sign_ok else ""))
        tr = [report["stage2"][str(s)][pname]["task_routed_test"]
              for s in seeds]
        qr = [report["stage2"][str(s)][pname]["query_routed_test"]
              for s in seeds]
        bf = [report["stage2"][str(s)][pname]["best_feature"] for s in seeds]
        print(f"stage2 test|nonlocal: task-routed {np.mean(tr):.4f}"
              f"±{np.std(tr, ddof=1):.4f}  query-routed {np.mean(qr):.4f}"
              f"±{np.std(qr, ddof=1):.4f}  (features: {bf})")
        if pname == "pair":
            print("abstain check: spearman(feature, raw ndcg) on val|nonlocal:")
            for arm in ("blend", "mixture_typed"):
                for feat in FEATURES:
                    vals = [report["stage1_abstain"][str(s)][arm][feat]
                            for s in seeds]
                    if max(abs(v) for v in vals) >= 0.15:
                        print(f"  {arm}/{feat:<18} "
                              + " ".join(f"{v:+.3f}" for v in vals))
        report["stage2"][f"{pname}_summary"] = {
            "task_routed_mean": float(np.mean(tr)),
            "task_routed_std": float(np.std(tr, ddof=1)),
            "query_routed_mean": float(np.mean(qr)),
            "query_routed_std": float(np.std(qr, ddof=1)),
            "best_features": bf,
        }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

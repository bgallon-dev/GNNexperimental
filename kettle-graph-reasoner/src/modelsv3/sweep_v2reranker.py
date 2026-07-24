r"""v3.1.x WS3 — fresh 18-dim v2 reranker + hybrid sweep.

The strongest Phase-1 result was the oracle gap: a perfect rerank of
v3.1's top-50 lifts ndcg@10 ~0.46 -> ~0.93. The old v2 checkpoints are
9-dim-query and unusable; ``train.py`` reads ``dataset.query_dim``
dynamically so a fresh 18-dim v2 needs ZERO training-code change.

General-first (locked decision): train ONE v2 over all tasks
(``train.py`` default ``--include-tasks`` = all; v2's internal BCE/MSE
routes per task), grid a small tiny-by-design {hidden_dim,num_layers},
pick the winner by mean hybrid ndcg@10, refine it over extra seeds.
Per task the existing ``v2_reranker.py`` reports v3.1-alone /
v3.1+v2 / oracle over the v3.1 baseline's top-C. GATE: a task whose
general-v2 closes < ``gap_close_threshold`` of the v3.1->oracle gap is
flagged; if ``auto_pertask`` a per-task v2 is then trained for the
winner config and that task re-measured.

Resumable: a trained v2 (best.pt+summary.json) or an existing hybrid
JSON is reused. Standalone; mirrors the sweep_queryhead skeleton.

Usage
-----
    py -m src.modelsv3.sweep_v2reranker \
        --config src/modelsv3/sweep_config_v2reranker.json
    py -m src.modelsv3.sweep_v2reranker --config ... --smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.modelsv3.sweep_queryhead import _mean_std  # noqa: E402  (reuse)


def _v2_tag(h: int, L: int, seed: int, task: str = "all") -> str:
    t = "all" if task == "all" else f"t{task}"
    return f"v2_scorer_18d_h{h}l{L}_{t}_seed{seed}"


def _train_v2(cfg: dict, h: int, L: int, seed: int, run_dir: Path,
              include_task: str | None) -> dict:
    """Train a fresh 18-dim v2 (general if include_task is None, else
    that single task). Zero code change: train.py reads query_dim."""
    if (run_dir / "best.pt").exists() and (run_dir / "summary.json").exists():
        return {"state": "reused"}
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "src.training.train",
        "--model", cfg.get("model", "hyperbolic"),
        "--corpus", cfg["corpus"],
        "--hidden-dim", str(h),
        "--num-layers", str(L),
        "--epochs", str(cfg["epochs"]),
        "--lr", str(cfg["lr"]),
        "--seed", str(seed),
        "--out", str(run_dir),
    ]
    if include_task is not None:
        cmd += ["--include-tasks", str(include_task)]
    # else: omit --include-tasks -> train.py default "" == ALL tasks
    with open(run_dir / "train.log", "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0 or not (run_dir / "best.pt").exists():
        return {"state": "failed_train", "rc": rc}
    return {"state": "trained"}


def _check_params(run_dir: Path, lo: int, hi: int) -> tuple[bool, int]:
    s = json.loads((run_dir / "summary.json").read_text())
    n = int(s.get("n_params", -1))
    return (lo <= n <= hi), n


def _hybrid(cfg: dict, v3_run: str, v2_run: Path, task: int,
            out_json: Path) -> dict | None:
    if not out_json.exists():
        cmd = [
            sys.executable, "-m", "src.modelsv3.v2_reranker",
            "--v3-run", v3_run,
            "--v2-run", str(v2_run),
            "--corpus", cfg["corpus"],
            "--task", str(task),
            "--topc", str(cfg.get("topc", 50)),
            "--out", str(out_json),
        ]
        with open(out_json.with_suffix(".log"), "w") as f:
            subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if not out_json.exists():
        return None
    r = json.loads(out_json.read_text())
    s = r.get("summary", {})
    v31 = s.get("v31_ndcg@10")
    hyb = s.get("hybrid_ndcg@10")
    orc = s.get("oracle_ndcg@10")
    gap = (orc - v31) if (isinstance(orc, (int, float))
                          and isinstance(v31, (int, float))) else None
    closed = (
        (hyb - v31) / gap if (gap and gap > 1e-9
                              and isinstance(hyb, (int, float))) else None
    )
    return {"v31_ndcg@10": v31, "hybrid_ndcg@10": hyb,
            "oracle_ndcg@10": orc, "gap_closed_frac": closed}


def _eval_v2_over_tasks(cfg: dict, v2_run: Path, out_dir: Path,
                        tag: str) -> dict:
    per_task: dict[str, dict] = {}
    for t in cfg["tasks"]:
        oj = out_dir / f"hybrid_{tag}_task{t}.json"
        r = _hybrid(cfg, cfg["v3_run"], v2_run, int(t), oj)
        if r is not None:
            per_task[str(t)] = r
    hyb = [v["hybrid_ndcg@10"] for v in per_task.values()
           if isinstance(v.get("hybrid_ndcg@10"), (int, float))]
    return {"per_task": per_task,
            "mean_hybrid_ndcg@10": (sum(hyb) / len(hyb)) if hyb else float("nan")}


def run_sweep(config_path: Path, smoke: bool) -> int:
    cfg = json.loads(config_path.read_text())
    out_root = Path(cfg["out_root"])
    v2_root = Path(cfg["v2_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    lo, hi = cfg["n_params_min"], cfg["n_params_max"]
    thr = cfg["gate"]["gap_close_threshold"]

    grid = [(h, L) for h in cfg["grid_hidden_dims"]
            for L in cfg["grid_num_layers"]]
    if smoke:
        grid = grid[:1]
        cfg = {**cfg, "epochs": 1, "tasks": [cfg["tasks"][0]],
               "refine_seeds": []}
        out_root = out_root / "_smoke"
        out_root.mkdir(parents=True, exist_ok=True)

    print(f"[v2-sweep] grid={grid}  epochs={cfg['epochs']}  "
          f"v3_run={cfg['v3_run']}")
    t0 = time.time()
    gs = cfg["grid_seed"]
    grid_results: dict[str, dict] = {}
    for (h, L) in grid:
        tag = f"h{h}l{L}_seed{gs}"
        v2_run = v2_root / _v2_tag(h, L, gs)
        print(f"[v2-sweep] train general v2 {tag} ...")
        st = _train_v2(cfg, h, L, gs, v2_run, include_task=None)
        if st["state"] in ("failed_train",):
            print(f"[v2-sweep]   FAIL train {tag}: {st}")
            grid_results[tag] = {"state": st["state"]}
            continue
        okp, npar = _check_params(v2_run, lo, hi)
        ev = _eval_v2_over_tasks(cfg, v2_run, out_root, tag)
        ev.update({"hidden_dim": h, "num_layers": L, "n_params": npar,
                   "params_in_budget": okp, "v2_run": str(v2_run),
                   "state": "complete"})
        grid_results[tag] = ev
        print(f"[v2-sweep]   {tag}: n_params={npar} in_budget={okp} "
              f"mean_hybrid_ndcg@10={ev['mean_hybrid_ndcg@10']:.4f}")

    # winner = in-budget config with best mean hybrid ndcg@10
    cand = [(k, v) for k, v in grid_results.items()
            if v.get("state") == "complete" and v.get("params_in_budget")]
    winner = max(cand, key=lambda kv: kv[1]["mean_hybrid_ndcg@10"]) \
        if cand else None

    refine: dict = {}
    pertask: dict = {}
    if winner and not smoke:
        wtag, wv = winner
        h, L = wv["hidden_dim"], wv["num_layers"]
        # refine: extra seeds for a noise floor on the chosen config
        seed_rows = [wv]
        for s in cfg.get("refine_seeds", []):
            r2 = v2_root / _v2_tag(h, L, s)
            stt = _train_v2(cfg, h, L, s, r2, include_task=None)
            if stt["state"] != "failed_train":
                seed_rows.append(_eval_v2_over_tasks(
                    cfg, r2, out_root, f"h{h}l{L}_seed{s}"))
        refine = _aggregate_seeds(cfg, seed_rows)

        # gated per-task v2: tasks whose general-v2 closed < threshold
        weak = [t for t, g in refine["by_task"].items()
                if g["gap_closed_frac_mean"] is not None
                and g["gap_closed_frac_mean"] < thr]
        if cfg["gate"].get("auto_pertask") and weak:
            print(f"[v2-sweep] auto_pertask: weak tasks {weak} -> "
                  f"per-task v2 (winner config h{h}l{L})")
            for t in weak:
                r3 = v2_root / _v2_tag(h, L, gs, task=t)
                stt = _train_v2(cfg, h, L, gs, r3, include_task=t)
                if stt["state"] == "failed_train":
                    pertask[t] = {"state": "failed_train"}
                    continue
                oj = out_root / f"hybrid_pertask_t{t}.json"
                pertask[t] = _hybrid(cfg, cfg["v3_run"], r3, int(t), oj) \
                    or {"state": "failed_eval"}

    report = {
        "config": str(config_path),
        "v3_run": cfg["v3_run"],
        "grid": grid_results,
        "winner": winner[0] if winner else None,
        "refine": refine,
        "pertask_v2": pertask,
        "gap_close_threshold": thr,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out_root / "sweep_v2reranker_results.json").write_text(
        json.dumps(report, indent=2))
    _print_report(report, out_root)
    return 0


def _aggregate_seeds(cfg: dict, seed_rows: list[dict]) -> dict:
    by_task: dict[str, dict] = {}
    for t in cfg["tasks"]:
        st = str(t)
        v31 = [r["per_task"][st]["v31_ndcg@10"] for r in seed_rows
               if st in r.get("per_task", {})]
        hyb = [r["per_task"][st]["hybrid_ndcg@10"] for r in seed_rows
               if st in r.get("per_task", {})]
        orc = [r["per_task"][st]["oracle_ndcg@10"] for r in seed_rows
               if st in r.get("per_task", {})]
        gc = [r["per_task"][st]["gap_closed_frac"] for r in seed_rows
              if st in r.get("per_task", {})]
        gcm, _ = _mean_std(gc)
        by_task[st] = {
            "v31_ndcg@10_mean": _mean_std(v31)[0],
            "hybrid_ndcg@10_mean": _mean_std(hyb)[0],
            "hybrid_ndcg@10_std": _mean_std(hyb)[1],
            "oracle_ndcg@10_mean": _mean_std(orc)[0],
            "gap_closed_frac_mean": gcm if gcm == gcm else None,
            "n_seeds": len(hyb),
        }
    return {"by_task": by_task, "n_seeds": len(seed_rows)}


def _print_report(r: dict, out_root: Path) -> None:
    print()
    print("=" * 96)
    print("v3.1.x WS3 - fresh 18-dim v2 reranker hybrid "
          f"(v3 retriever = {r['v3_run']})")
    print("=" * 96)
    for tag, g in r["grid"].items():
        if g.get("state") != "complete":
            print(f"  grid {tag}: {g.get('state')}")
            continue
        print(f"  grid {tag}: n_params={g['n_params']} "
              f"in_budget={g['params_in_budget']} "
              f"mean_hybrid_ndcg@10={g['mean_hybrid_ndcg@10']:.4f}")
    print(f"  WINNER: {r['winner']}")
    rf = r.get("refine", {})
    if rf.get("by_task"):
        print(f"\n  Refined (n_seeds={rf['n_seeds']}) per-task "
              f"v3.1 -> v3.1+v2 -> oracle  [gap closed]:")
        print(f"  {'task':<6}{'v3.1@10':>10}{'+v2@10':>10}"
              f"{'oracle@10':>11}{'gap_closed':>12}{'>=thr':>7}")
        thr = r["gap_close_threshold"]
        for t, b in rf["by_task"].items():
            gc = b["gap_closed_frac_mean"]
            gcs = "n/a" if gc is None else f"{gc:.2f}"
            ok = "" if gc is None else ("YES" if gc >= thr else "no")
            print(f"  {t:<6}{b['v31_ndcg@10_mean']:>10.4f}"
                  f"{b['hybrid_ndcg@10_mean']:>10.4f}"
                  f"{b['oracle_ndcg@10_mean']:>11.4f}"
                  f"{gcs:>12}{ok:>7}")
    if r.get("pertask_v2"):
        print("\n  Gated per-task v2 (weak tasks retrained "
              "--include-tasks T):")
        for t, p in r["pertask_v2"].items():
            if p.get("hybrid_ndcg@10") is not None:
                print(f"    task {t}: hybrid_ndcg@10="
                      f"{p['hybrid_ndcg@10']:.4f}  gap_closed="
                      f"{p.get('gap_closed_frac')}")
            else:
                print(f"    task {t}: {p.get('state')}")
    print("\n  Honest-result note: if hybrid <= v3.1-alone for a task, "
          "that is reported as-is (oracle headroom does not obligate a "
          "lift). results: " + str(out_root /
                                    'sweep_v2reranker_results.json'))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str,
                    default="src/modelsv3/sweep_config_v2reranker.json")
    ap.add_argument("--smoke", action="store_true",
                    help="1 grid config, 1 epoch, 1 task — plumbing only.")
    args = ap.parse_args()
    return run_sweep(Path(args.config), smoke=args.smoke)


if __name__ == "__main__":
    sys.exit(main())

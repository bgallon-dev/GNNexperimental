r"""v3.2 reranker MVP sweep — per (task, seed), gated vs WS3.

Per cell runs ``reranker_v32`` with the retriever = the WS2 per-task
qh1 cell (tasks 0/1/3/4/5) or the v3.1 baseline (temporal task 2,
included to check the no-regression guarantee on the task WS3
regressed). Resumable. Mirrors the established sweep skeleton.

Gate (the v3.2 acceptance):
  * geometry-sensitive tasks {0,3,4,5}: v3.2 mean gap_closed_frac must
    EXCEED WS3's per-task v2 gap_closed_frac (the thing we're trying to
    beat), and
  * regression_vs_retriever == False on EVERY task (the residual's
    structural guarantee must hold empirically, especially temporal).

Usage
-----
    py -m src.modelsv3.sweep_reranker_v32 \
        --config src/modelsv3/sweep_config_reranker_v32.json
    py -m src.modelsv3.sweep_reranker_v32 --config ... --smoke
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


def _retriever_run(cfg: dict, task: int, seed: int) -> Path:
    if task == 2:
        # no WS2 qh1 cell for temporal; the v3.1 baseline IS the task-2
        # retriever (single fixed artifact, used for every seed).
        return Path(cfg["v31_baseline"])
    return Path(cfg["ws2_root"]) / f"task{task}_seed{seed}"


def _cells(cfg: dict) -> list[tuple[int, int]]:
    return [(int(t), int(s)) for t in cfg["tasks"] for s in cfg["seeds"]]


def _ws3_bar(cfg: dict) -> dict:
    """WS3 per-task v2 gap_closed_frac to beat (fallback: WS3 refined
    general-v2 gap)."""
    p = Path(cfg["ws3_results"])
    if not p.exists():
        return {}
    r = json.loads(p.read_text())
    bar: dict[str, float] = {}
    for t, v in (r.get("pertask_v2") or {}).items():
        g = v.get("gap_closed_frac") if isinstance(v, dict) else None
        if isinstance(g, (int, float)):
            bar[str(t)] = float(g)
    for t, v in (r.get("refine", {}).get("by_task") or {}).items():
        if str(t) not in bar:
            g = v.get("gap_closed_frac_mean")
            if isinstance(g, (int, float)):
                bar[str(t)] = float(g)
    return bar


def _run_cell(cfg: dict, task: int, seed: int, retr: Path,
              cell_dir: Path, epochs: int) -> dict:
    cell_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "src.modelsv3.reranker_v32",
        "--retriever-run", str(retr),
        "--task", str(task), "--seed", str(seed),
        "--epochs", str(epochs), "--topc", str(cfg["topc"]),
        "--lr", str(cfg["lr"]), "--temperature", str(cfg["temperature"]),
        "--hidden-dim", str(cfg["hidden_dim"]),
        "--num-layers", str(cfg["num_layers"]),
        "--type-dim", str(cfg["type_dim"]),
        "--combine-mode", str(cfg.get("combine_mode", "v32")),
        "--corpus", cfg["corpus"], "--out", str(cell_dir),
    ]
    with open(cell_dir / "run.log", "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    return {"state": "complete" if (rc == 0 and
            (cell_dir / "hybrid.json").exists()) else "failed", "rc": rc}


def run_sweep(config_path: Path, smoke: bool) -> int:
    cfg = json.loads(config_path.read_text())
    out_root = Path(cfg["out_root"])
    cells = _cells(cfg)
    epochs = cfg["epochs"]
    if smoke:
        cells = [(int(cfg["tasks"][0]), int(cfg["seeds"][0])),
                 (2, int(cfg["seeds"][0]))]  # one geom task + temporal
        epochs = 2
        out_root = out_root / "_smoke"
    out_root.mkdir(parents=True, exist_ok=True)
    ws3 = _ws3_bar(cfg)

    print(f"[rr32-sweep] {len(cells)} cells  epochs={epochs}")
    t0 = time.time()
    results: dict[str, dict] = {}
    for i, (task, seed) in enumerate(cells):
        name = f"task{task}_seed{seed}"
        cd = out_root / name
        retr = _retriever_run(cfg, task, seed)
        if (cd / "hybrid.json").exists():
            print(f"[rr32-sweep] ({i+1}/{len(cells)}) {name} - skip (done)")
        elif not (retr / "summary.json").exists():
            print(f"[rr32-sweep] ({i+1}/{len(cells)}) {name} - SKIP "
                  f"(retriever {retr} missing)")
            results[name] = {"state": "no_retriever", "task": task,
                             "seed": seed}
            continue
        else:
            print(f"[rr32-sweep] ({i+1}/{len(cells)}) {name} "
                  f"- running (retr={retr.name})...")
            st = _run_cell(cfg, task, seed, retr, cd, epochs)
            if st["state"] != "complete":
                print(f"[rr32-sweep]   FAIL {name}: {st}")
                results[name] = {"state": st["state"], "task": task,
                                 "seed": seed}
                continue
        h = json.loads((cd / "hybrid.json").read_text())
        tr = h.get("trained", {})
        results[name] = {
            "state": "complete", "task": task, "seed": seed,
            "v31_ndcg@10": h["summary"]["v31_ndcg@10"],
            "hybrid_ndcg@10": h["summary"]["hybrid_ndcg@10"],  # deployed
            "oracle_ndcg@10": h["summary"]["oracle_ndcg@10"],
            "gap_closed_frac": h.get("gap_closed_frac"),        # deployed
            "regression_vs_retriever": h.get("regression_vs_retriever"),
            "residual_deployed": h.get("residual_deployed"),
            "trained_gap_closed_frac": tr.get("gap_closed_frac"),
            "trained_regression": tr.get("regression_vs_retriever"),
            "learned_scale": h.get("learned_scale"),
            "deployed_scale": h.get("deployed_scale"),
        }

    report = _gate(results, cfg, ws3)
    out = {"config": str(config_path), "ws3_bar": ws3,
           "cells": results, "gate": report,
           "wall_seconds": round(time.time() - t0, 1)}
    (out_root / "sweep_reranker_v32_results.json").write_text(
        json.dumps(out, indent=2))
    _print_report(report, ws3, out_root)
    return 0 if report.get("acceptance_pass") else 2


def _gate(results: dict, cfg: dict, ws3: dict) -> dict:
    geo = set(str(t) for t in cfg["geometry_sensitive"])
    tasks = sorted({v["task"] for v in results.values()
                    if v.get("state") == "complete"})
    by_task: dict[str, dict] = {}
    any_regression = False
    for t in tasks:
        rows = [v for v in results.values()
                if v.get("state") == "complete" and v["task"] == t]
        v31, _ = _mean_std([r["v31_ndcg@10"] for r in rows])
        hyb, hs = _mean_std([r["hybrid_ndcg@10"] for r in rows])
        gc, _ = _mean_std([r["gap_closed_frac"] for r in rows])
        tgc, _ = _mean_std([r.get("trained_gap_closed_frac") for r in rows])
        n_dep = sum(1 for r in rows if r.get("residual_deployed"))
        regr = any(bool(r.get("regression_vs_retriever")) for r in rows)
        any_regression = any_regression or regr
        ws3_g = ws3.get(str(t))
        beats = (ws3_g is not None and gc == gc and gc > ws3_g)
        by_task[str(t)] = {
            "n_seeds": len(rows),
            "v31_ndcg@10_mean": v31,
            "hybrid_ndcg@10_mean": hyb, "hybrid_ndcg@10_std": hs,
            "gap_closed_frac_mean": gc if gc == gc else None,
            "trained_gap_closed_frac_mean": tgc if tgc == tgc else None,
            "n_residual_deployed": n_dep,
            "ws3_pertask_gap_closed": ws3_g,
            "beats_ws3": bool(beats),
            "regression": bool(regr),
            "is_geometry_sensitive": str(t) in geo,
        }
    geo_rows = [b for k, b in by_task.items() if b["is_geometry_sensitive"]]
    geo_beat = [b for b in geo_rows if b["beats_ws3"]]
    return {
        "by_task": by_task,
        "geometry_tasks_beating_ws3": len(geo_beat),
        "geometry_tasks_total": len(geo_rows),
        "any_regression": any_regression,
        "acceptance_pass": (len(geo_beat) == len(geo_rows)
                            and len(geo_rows) > 0
                            and not any_regression),
    }


def _print_report(report: dict, ws3: dict, out_root: Path) -> None:
    print()
    print("=" * 96)
    print("v3.2 reranker (MVP combo) gate  vs  WS3 per-task v2")
    print("=" * 96)
    print(f"  {'task':<6}{'geom':>5}{'retr@10':>10}{'dep@10':>10}"
          f"{'depGap':>8}{'trnGap':>8}{'WS3gap':>8}{'dep?':>6}"
          f"{'beats':>7}{'regr':>6}")
    for t, b in report["by_task"].items():
        gc = b["gap_closed_frac_mean"]
        gcs = "n/a" if gc is None else f"{gc:+.2f}"
        tg = b.get("trained_gap_closed_frac_mean")
        tgs = "n/a" if tg is None else f"{tg:+.2f}"
        w = b["ws3_pertask_gap_closed"]
        ws = "n/a" if w is None else f"{w:+.2f}"
        print(f"  {t:<6}{('Y' if b['is_geometry_sensitive'] else '-'):>5}"
              f"{b['v31_ndcg@10_mean']:>10.4f}"
              f"{b['hybrid_ndcg@10_mean']:>10.4f}"
              f"{gcs:>8}{tgs:>8}{ws:>8}"
              f"{(str(b.get('n_residual_deployed',0))+'/'+str(b['n_seeds'])):>6}"
              f"{('YES' if b['beats_ws3'] else 'no'):>7}"
              f"{('YES' if b['regression'] else 'no'):>6}")
    print(f"\n  geometry-sensitive tasks beating WS3: "
          f"{report['geometry_tasks_beating_ws3']}/"
          f"{report['geometry_tasks_total']}  | "
          f"any regression: {report['any_regression']}  | "
          f"ACCEPTANCE: {report['acceptance_pass']}")
    print("  retr@10 = per-task qh1 retriever alone; +v3.2 = residual "
          "reranker; gap_clr = fraction of (oracle-retr) closed.")
    if not report["acceptance_pass"]:
        print("  Honest note: report the retr/+v3.2/oracle triple as-is. "
              "If a geom task does not beat WS3, the MVP combo did not "
              "help there - state it, do not paper over. Regression on "
              "ANY task contradicts the residual guarantee -> investigate "
              "(scale/standardization), do not ship that cell.")
    print(f"\n  results: {out_root / 'sweep_reranker_v32_results.json'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str,
                    default="src/modelsv3/sweep_config_reranker_v32.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    return run_sweep(Path(args.config), smoke=args.smoke)


if __name__ == "__main__":
    sys.exit(main())

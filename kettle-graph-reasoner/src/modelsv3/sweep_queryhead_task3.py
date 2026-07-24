r"""Task-3 query-head sweep (reframed lever; frozen encoder).

The task-3 escalation FINAL (PHASE1_FINDINGS.md) localised the failure
to ANCHOR LOCALISATION at the retrieval stage: task-3 relevance IS
anchor-relative BFS-hop structure (label~hop r=+0.71) but the qh1
retriever is blind to it and cannot place the anchor; qh1 was the only
head ever tried for task 3. This sweeps qh1 (in-sweep reference) vs
qh2/qh3 over 3 seeds on the LOCKED v3.1 baseline encoder, SHA-asserted,
stage-B only.

Reuses ``sweep_queryhead``'s proven cell mechanics (``_cells``,
``_run_cell``, ``_mean_std``, ``ARCH_PARAM_ORDER``) verbatim. The ONLY
deviation is the frozen-encoder check: ``sweep_queryhead`` asserts
intrinsic ``nn_edge_precision@5`` == the task-2 baseline value, which is
WRONG per-task (task-3 val graph 0 is a different graph -> it
legitimately differs even with an identical frozen encoder; see
``sweep_taskdiversity.py:21-24``). So this uses the task-invariant
guarantee instead: every cell SHA-asserted the locked encoder and
skipped stage A (recorded in ``summary.json['config']``).

Gate: the v3.1 cross-task 0.52 ndcg@10 bar is reported for CONTINUITY
only (task 3 is hard: oracle ~0.749, qh1 ~0.313). The real success
criterion is whether qh2/qh3 beat the qh1 IN-SWEEP incumbent by more
than the combined 3-seed noise (floor 0.01) -- that would raise the
deployed task-3 retriever (absolute headroom 0.324 -> 0.749).

Usage
-----
    py -m src.modelsv3.sweep_queryhead_task3 \
        --config src/modelsv3/sweep_config_queryhead_task3.json
    py -m src.modelsv3.sweep_queryhead_task3 --config ... --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.modelsv3.lock_baseline import gate_threshold, load_manifest  # noqa: E402
from src.modelsv3.sweep_queryhead import (  # noqa: E402  (reuse, verbatim)
    ARCH_PARAM_ORDER,
    _cell_name,
    _cells,
    _mean_std,
    _run_cell,
)


def _done(cell_dir: Path) -> bool:
    return (cell_dir / "summary.json").exists() and (
        cell_dir / "candidate_recall.json").exists()


def _read_cell_metrics(cell_dir: Path) -> dict:
    """sweep_queryhead metrics + the task-invariant frozen-encoder
    evidence from summary.json['config'] (skip_stage_a + asserted sha),
    the way sweep_taskdiversity does it."""
    s = json.loads((cell_dir / "summary.json").read_text())
    fv = s["final_val"]["overall"]
    sc = s.get("config", {})
    cr = json.loads((cell_dir / "candidate_recall.json").read_text())
    cro = cr["summary"]["overall"]
    return {
        "ndcg@10": fv.get("ndcg@10"),
        "ndcg@20": fv.get("ndcg@20"),
        "recall@50": cro.get("recall@50"),
        "recall@100": cro.get("recall@100"),
        "oracle_ndcg@10|C50": cro.get("oracle_ndcg@10|C50"),
        "oracle_gap@10|C50": cro.get("oracle_gap@10|C50"),
        "n_query_params": s.get("n_params_query"),
        "cfg_skip_stage_a": bool(sc.get("skip_stage_a", False)),
        "cfg_assert_sha": sc.get("assert_encoder_sha"),
        "cfg_query_head_arch": sc.get("query_head_arch"),
    }


def run_sweep(config_path: Path, smoke: bool) -> int:
    cfg = json.loads(config_path.read_text())
    manifest = load_manifest(Path(cfg["baseline_dir"]))
    enc_sha = manifest["encoder_sha256"]
    noise = manifest["noise_floor"]
    out_root = Path(cfg["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    cells = _cells(cfg)
    query_epochs = cfg["query_epochs"]
    if smoke:
        # qh1 + qh2 at seed0/layernorm so the incumbent gate is exercised.
        s0 = int(cfg["seeds"][0])
        cells = [("qh1", "layernorm", s0), ("qh2", "layernorm", s0)]
        query_epochs = 1
        out_root = out_root / "_smoke"
        out_root.mkdir(parents=True, exist_ok=True)

    print(f"[qh3-sweep] {len(cells)} cells  task={cfg['task']}  "
          f"encoder={cfg['encoder_path']}")
    print(f"[qh3-sweep] sha-asserted={enc_sha[:12]}...  "
          f"query_epochs={query_epochs}")
    t0 = time.time()
    results: dict[str, dict] = {}
    for i, (arch, norm, seed) in enumerate(cells):
        name = _cell_name(arch, norm, seed)
        cell_dir = out_root / name
        if _done(cell_dir):
            print(f"[qh3-sweep] ({i+1}/{len(cells)}) {name} - skip (done)")
        else:
            print(f"[qh3-sweep] ({i+1}/{len(cells)}) {name} - running...")
            st = _run_cell(cfg, arch, norm, seed, enc_sha, cell_dir,
                           query_epochs)
            if st["state"] != "complete":
                print(f"[qh3-sweep]   FAIL {name}: {st} "
                      f"(see {cell_dir}/*.log)")
                results[name] = {"state": st["state"]}
                continue
        m = _read_cell_metrics(cell_dir)
        m["arch"], m["norm"], m["seed"] = arch, norm, seed
        m["state"] = "complete"
        results[name] = m

    report = _gate(results, cfg, noise, enc_sha)
    out = {
        "config": str(config_path),
        "baseline_dir": cfg["baseline_dir"],
        "encoder_sha256": enc_sha,
        "noise_floor": noise,
        "cells": results,
        "gate": report,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out_root / "sweep_queryhead_task3_results.json").write_text(
        json.dumps(out, indent=2))
    _print_report(report, out_root)
    return 0 if report.get("any_beats_incumbent") else 2


def _gate(results: dict, cfg: dict, noise: dict, enc_sha: str) -> dict:
    spec_target = cfg.get("gate", {}).get("ndcg10_target", 0.52)
    nd_mean = noise["ndcg@10"]["mean"]
    xtask_threshold = max(
        spec_target,
        gate_threshold(noise, "ndcg@10", spec_target - nd_mean))
    floor = float(cfg.get("gate", {}).get("noise_floor_min", 0.01))
    incumbent = cfg.get("gate", {}).get("incumbent_arch", "qh1")

    by_arch: dict[str, dict] = {}
    arch_set = sorted(
        {v["arch"] for v in results.values()
         if v.get("state") == "complete"},
        key=lambda a: ARCH_PARAM_ORDER.index(a)
        if a in ARCH_PARAM_ORDER else 99)
    for arch in arch_set:
        rows = [v for v in results.values()
                if v.get("state") == "complete" and v["arch"] == arch]
        m10, s10 = _mean_std([r["ndcg@10"] for r in rows])
        m20, s20 = _mean_std([r["ndcg@20"] for r in rows])
        r50, _ = _mean_std([r["recall@50"] for r in rows])
        oc, _ = _mean_std([r["oracle_ndcg@10|C50"] for r in rows])
        og, _ = _mean_std([r["oracle_gap@10|C50"] for r in rows])
        # task-invariant frozen-encoder guarantee (correct per-task)
        frozen_ok = all(
            r.get("cfg_skip_stage_a")
            and r.get("cfg_assert_sha") == enc_sha for r in rows)
        by_arch[arch] = {
            "n_cells": len(rows),
            "ndcg@10_mean": m10, "ndcg@10_std": s10,
            "ndcg@20_mean": m20, "ndcg@20_std": s20,
            "recall@50_mean": r50,
            "oracle_ndcg@10_mean": oc,
            "oracle_gap@10_mean": og,
            "n_query_params": rows[0]["n_query_params"] if rows else None,
            "frozen_encoder_ok": frozen_ok,
            "clears_xtask_bar": bool(m10 == m10 and m10 >= xtask_threshold),
        }

    inc = by_arch.get(incumbent)
    inc_m = inc["ndcg@10_mean"] if inc else float("nan")
    inc_s = inc["ndcg@10_std"] if inc else float("nan")
    for arch, b in by_arch.items():
        if arch == incumbent or inc is None:
            b["delta_vs_incumbent"] = (0.0 if arch == incumbent
                                       else float("nan"))
            b["required_margin"] = float("nan")
            b["beats_incumbent"] = False
            continue
        d = b["ndcg@10_mean"] - inc_m
        req = max(math.hypot(b["ndcg@10_std"], inc_s), floor)
        b["delta_vs_incumbent"] = d
        b["required_margin"] = req
        b["beats_incumbent"] = bool(d == d and d > req and
                                    b["frozen_encoder_ok"])

    winners = [a for a in arch_set
               if by_arch[a].get("beats_incumbent")]
    winners.sort(key=lambda a: ARCH_PARAM_ORDER.index(a)
                 if a in ARCH_PARAM_ORDER else 99)
    return {
        "incumbent_arch": incumbent,
        "incumbent_ndcg@10_mean": inc_m,
        "incumbent_ndcg@10_std": inc_s,
        "xtask_threshold_ndcg@10": xtask_threshold,
        "xtask_threshold_basis": "v3.1 cross-task bar (continuity only; "
                                 "task 3 not expected to clear it)",
        "noise_floor_min": floor,
        "by_arch": by_arch,
        "beating_incumbent": winners,
        "selected_arch": winners[0] if winners else None,
        "any_beats_incumbent": bool(winners),
        "frozen_encoder_ok_all": all(
            b["frozen_encoder_ok"] for b in by_arch.values()),
    }


def _print_report(report: dict, out_root: Path) -> None:
    print()
    print("=" * 100)
    print("Task-3 query-head sweep  (qh1 incumbent vs qh2/qh3, frozen "
          "encoder, SHA-asserted)")
    inc = report["incumbent_arch"]
    print(f"incumbent {inc} ndcg@10 = "
          f"{report['incumbent_ndcg@10_mean']:.4f}"
          f"+-{report['incumbent_ndcg@10_std']:.4f}  | beat-by-noise "
          f"floor = {report['noise_floor_min']}  | xtask bar "
          f"{report['xtask_threshold_ndcg@10']:.3f} (continuity only)")
    print("=" * 100)
    print(f"  {'arch':<6}{'params':>9}{'ndcg@10':>17}{'oracle@10':>11}"
          f"{'gap@10':>9}{'recall@50':>11}{'vsInc':>9}{'need>':>8}"
          f"{'frzOK':>7}{'BEATS':>7}")
    for arch, b in report["by_arch"].items():
        d = b.get("delta_vs_incumbent")
        ds = "  ref" if arch == inc else (
            "n/a" if d != d else f"{d:+.4f}")
        rq = b.get("required_margin")
        rs = "-" if (rq != rq) else f"{rq:.3f}"
        print(f"  {arch:<6}{(b['n_query_params'] or 0):>9}"
              f"  {b['ndcg@10_mean']:.4f}+-{b['ndcg@10_std']:.4f}"
              f"  {b['oracle_ndcg@10_mean']:.4f}"
              f"  {b['oracle_gap@10_mean']:+.4f}"
              f"  {b['recall@50_mean']:.4f}"
              f"  {ds:>7}{rs:>8}"
              f"  {str(b['frozen_encoder_ok']):>5}"
              f"  {'YES' if b.get('beats_incumbent') else 'no':>5}")
    sel = report["selected_arch"]
    if sel:
        print(f"\n  RESULT: {sel} beats the {inc} incumbent beyond the "
              f"3-seed noise -> the reframed lever WORKS. Re-point the "
              f"task-3 retriever at {sel} (rebuild the WS2 task-3 cell "
              f"with --query-head-arch {sel}) and re-run the router.")
    else:
        print(f"\n  RESULT: NO arch beats the {inc} incumbent beyond "
              f"noise. Honest negative: a more expressive query head "
              f"does NOT recover task-3 anchor localisation. Report "
              f"(best_mean,std) as-is; task-3 absolute headroom to "
              f"0.749 stands as an open limitation, NOT papered over. "
              f"Do not touch the encoder; do not switch to MSE.")
    if not report["frozen_encoder_ok_all"]:
        print("\n  WARNING: frozen_encoder_ok is False on some arch -> a "
              "cell did NOT skip-stage-a or asserted the wrong SHA. That "
              "is a plumbing BUG (gradient leak), not a tuning result. "
              "Stop and fix before trusting any number above.")
    print(f"\n  results: {out_root / 'sweep_queryhead_task3_results.json'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", type=str,
        default="src/modelsv3/sweep_config_queryhead_task3.json")
    ap.add_argument("--smoke", action="store_true",
                    help="qh1+qh2 at seed0, 1 query epoch (plumbing).")
    args = ap.parse_args()
    return run_sweep(Path(args.config), smoke=args.smoke)


if __name__ == "__main__":
    sys.exit(main())

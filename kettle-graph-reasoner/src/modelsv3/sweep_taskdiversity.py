r"""v3.1.x WS2 — task-diversity sweep driver (frozen encoder).

Trains a per-task ``qh1`` query head (stage B only, pairwise loss) on
ONE shared frozen encoder loaded from the locked baseline, SHA-asserted.
Cell axis = (task, seed) over the geometry-sensitive tasks
0/1/3/4/5 (temporal task 2 is the existing v3.1 baseline, not re-swept).
Synthetic tier1 already contains all task types -> zero new data code.

The thesis: temporal (task 2) is locality-friendly (anchor-BFS is
strong there); graph-native geometry should matter MORE on the
geometry-sensitive tasks. So the report contrasts, per task, qh1
ndcg@10 vs the v3.1 task-2 noise-floor bar AND the oracle ceiling /
ordering headroom (from eval_candidate_recall).

Per cell: ``train_v3 --skip-stage-a --task T --query-head-arch qh1``
then ``eval_candidate_recall --task T`` (+ ``eval_provenance_path`` for
tasks 0/3). Resumable. Standalone (mirrors sweep_queryhead.py; does not
touch any shared harness).

Frozen-encoder guarantee here = SHA-assert + skip-stage-a recorded in
``summary.json['config']`` (task-invariant). The query-head sweep's
cross-task intrinsic-edge-prec equality does NOT apply: per-task
``val_ds[0]`` is a different graph, so its intrinsic value legitimately
differs even with an identical frozen encoder.

Usage
-----
    py -m src.modelsv3.sweep_taskdiversity \
        --config src/modelsv3/sweep_config_taskdiversity.json
    py -m src.modelsv3.sweep_taskdiversity --config ... --smoke
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

from src.modelsv3.lock_baseline import gate_threshold, load_manifest  # noqa: E402
from src.modelsv3.sweep_queryhead import _mean_std  # noqa: E402  (reuse)

PROV_TASKS = (0, 3)  # eval_provenance_path scores only these


def _cells(cfg: dict) -> list[tuple[int, int]]:
    return [(int(t), int(s)) for t in cfg["tasks"] for s in cfg["seeds"]]


def _cell_name(task: int, seed: int) -> str:
    return f"task{task}_seed{seed}"


def _done(cell_dir: Path) -> bool:
    return (cell_dir / "summary.json").exists() and (
        cell_dir / "candidate_recall.json"
    ).exists()


def _run_cell(cfg: dict, task: int, seed: int, enc_sha: str,
              cell_dir: Path, query_epochs: int) -> dict:
    cell_dir.mkdir(parents=True, exist_ok=True)
    train_cmd = [
        sys.executable, "-m", "src.training.train_v3",
        "--task", str(task),
        "--model", cfg.get("model", "hyperbolic"),
        "--corpus", cfg["corpus"],
        "--out", str(cell_dir),
        "--hidden-dim", str(cfg["hidden_dim"]),
        "--num-layers", str(cfg["num_layers"]),
        "--curvature", str(cfg.get("curvature", 1.0)),
        "--seed", str(seed),
        "--skip-stage-a",
        "--load-encoder", cfg["encoder_path"],
        "--assert-encoder-sha", enc_sha,
        "--query-head-arch", cfg.get("query_head_arch", "qh1"),
        "--query-head-norm", cfg.get("query_head_norm", "layernorm"),
        "--query-epochs", str(query_epochs),
        "--lr-query", str(cfg["lr_query"]),
        "--stage-b-loss", cfg.get("stage_b_loss", "pairwise"),
        "--log-every", str(cfg.get("log_every", 200)),
    ]
    with open(cell_dir / "train.log", "w") as f:
        rc = subprocess.call(train_cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        return {"state": "failed_train", "rc": rc}

    eval_cmd = [
        sys.executable, "-m", "src.modelsv3.eval_candidate_recall",
        "--checkpoint", str(cell_dir / "encoder.pt"),
        "--task", str(task),
        "--out", str(cell_dir / "candidate_recall.json"),
    ]
    with open(cell_dir / "eval_candidate_recall.log", "w") as f:
        rc = subprocess.call(eval_cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        return {"state": "failed_eval", "rc": rc}

    if task in PROV_TASKS:
        prov_cmd = [
            sys.executable, "-m", "src.modelsv3.eval_provenance_path",
            "--checkpoint", str(cell_dir / "encoder.pt"),
            "--task", str(task),
            "--out", str(cell_dir / "provenance_path.json"),
        ]
        with open(cell_dir / "eval_provenance_path.log", "w") as f:
            subprocess.call(prov_cmd, stdout=f, stderr=subprocess.STDOUT)
            # provenance is supplementary; a non-zero rc (e.g. no
            # eligible samples) does not fail the cell.
    return {"state": "complete"}


def _read_cell_metrics(cell_dir: Path) -> dict:
    s = json.loads((cell_dir / "summary.json").read_text())
    fv = s["final_val"]["overall"]
    sc = s.get("config", {})
    cr = json.loads((cell_dir / "candidate_recall.json").read_text())
    cro = cr["summary"]["overall"]
    m = {
        "ndcg@10": fv.get("ndcg@10"),
        "ndcg@20": fv.get("ndcg@20"),
        "recall@50": cro.get("recall@50"),
        "recall@100": cro.get("recall@100"),
        "oracle_ndcg@10|C50": cro.get("oracle_ndcg@10|C50"),
        "oracle_gap@10|C50": cro.get("oracle_gap@10|C50"),
        "n_query_params": s.get("n_params_query"),
        # task-invariant frozen-encoder evidence
        "cfg_skip_stage_a": bool(sc.get("skip_stage_a", False)),
        "cfg_assert_sha": sc.get("assert_encoder_sha"),
        "cfg_query_head_arch": sc.get("query_head_arch"),
    }
    pp = cell_dir / "provenance_path.json"
    if pp.exists():
        try:
            ppj = json.loads(pp.read_text())
            m["prov_path_recall@10"] = (
                ppj.get("summary", {}).get("overall", {})
                .get("prov_path_recall@10"))
        except json.JSONDecodeError:
            m["prov_path_recall@10"] = None
    return m


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
        cells = [(int(cfg["tasks"][0]), int(cfg["seeds"][0]))]
        query_epochs = 1
        out_root = out_root / "_smoke"
        out_root.mkdir(parents=True, exist_ok=True)

    print(f"[td-sweep] {len(cells)} cells  encoder={cfg['encoder_path']}")
    print(f"[td-sweep] arch={cfg.get('query_head_arch','qh1')}  "
          f"sha-asserted={enc_sha[:12]}...  query_epochs={query_epochs}")
    t0 = time.time()
    results: dict[str, dict] = {}
    for i, (task, seed) in enumerate(cells):
        name = _cell_name(task, seed)
        cell_dir = out_root / name
        if _done(cell_dir):
            print(f"[td-sweep] ({i+1}/{len(cells)}) {name} - skip (done)")
        else:
            print(f"[td-sweep] ({i+1}/{len(cells)}) {name} - running...")
            st = _run_cell(cfg, task, seed, enc_sha, cell_dir, query_epochs)
            if st["state"] != "complete":
                print(f"[td-sweep]   FAIL {name}: {st} (see {cell_dir}/*.log)")
                results[name] = {"state": st["state"], "task": task,
                                 "seed": seed}
                continue
        m = _read_cell_metrics(cell_dir)
        m["task"], m["seed"], m["state"] = task, seed, "complete"
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
    (out_root / "sweep_taskdiversity_results.json").write_text(
        json.dumps(out, indent=2))
    _print_report(report, out_root)
    return 0 if report.get("acceptance_pass") else 2


def _gate(results: dict, cfg: dict, noise: dict, enc_sha: str) -> dict:
    spec_delta = cfg.get("gate", {}).get("ndcg10_spec_delta", 0.0)
    threshold = gate_threshold(noise, "ndcg@10", spec_delta)
    min_tasks = int(cfg.get("gate", {}).get("min_tasks_pass", 4))

    by_task: dict[str, dict] = {}
    tasks = sorted({v["task"] for v in results.values()
                    if v.get("state") == "complete"})
    for t in tasks:
        rows = [v for v in results.values()
                if v.get("state") == "complete" and v["task"] == t]
        m10, s10 = _mean_std([r["ndcg@10"] for r in rows])
        m20, _ = _mean_std([r["ndcg@20"] for r in rows])
        r50, _ = _mean_std([r["recall@50"] for r in rows])
        og, _ = _mean_std([r["oracle_gap@10|C50"] for r in rows])
        oc, _ = _mean_std([r["oracle_ndcg@10|C50"] for r in rows])
        pp, _ = _mean_std([r.get("prov_path_recall@10") for r in rows])
        # Task-invariant frozen-encoder check: every cell SHA-asserted
        # the locked encoder and skipped stage A (encoder never trained).
        frozen_ok = all(
            r.get("cfg_skip_stage_a") and r.get("cfg_assert_sha") == enc_sha
            for r in rows)
        passed = (m10 >= threshold) and frozen_ok
        by_task[str(t)] = {
            "n_seeds": len(rows),
            "ndcg@10_mean": m10, "ndcg@10_std": s10,
            "ndcg@20_mean": m20,
            "recall@50_mean": r50,
            "oracle_ndcg@10_mean": oc,      # the achievable ceiling
            "oracle_gap@10_mean": og,       # ordering headroom
            "prov_path_recall@10_mean": pp,
            "n_query_params": rows[0]["n_query_params"] if rows else None,
            "frozen_encoder_ok": frozen_ok,
            "pass": bool(passed),
        }
    n_pass = sum(1 for b in by_task.values() if b["pass"])
    return {
        "threshold_ndcg@10": threshold,
        "threshold_basis": "v3.1 task-2 noise floor: baseline_mean + "
                           "max(spec_delta, 1*std)",
        "by_task": by_task,
        "n_tasks_pass": n_pass,
        "n_tasks_total": len(by_task),
        "min_tasks_pass": min_tasks,
        "acceptance_pass": n_pass >= min_tasks
        and all(b["frozen_encoder_ok"] for b in by_task.values()),
    }


def _print_report(report: dict, out_root: Path) -> None:
    print()
    print("=" * 92)
    print("v3.1.x WS2 - task-diversity gate (per-task qh1, frozen encoder)")
    print(f"ndcg@10 bar >= {report['threshold_ndcg@10']:.4f}  "
          f"({report['threshold_basis']})")
    print("=" * 92)
    print(f"  {'task':<6}{'ndcg@10':>17}{'oracle@10':>11}{'gap@10':>9}"
          f"{'recall@50':>11}{'frozenOK':>10}{'PASS':>6}")
    for t, b in report["by_task"].items():
        print(f"  {t:<6}  {b['ndcg@10_mean']:.4f}+-{b['ndcg@10_std']:.4f}"
              f"  {b['oracle_ndcg@10_mean']:.4f}"
              f"  {b['oracle_gap@10_mean']:+.4f}"
              f"  {b['recall@50_mean']:.4f}"
              f"  {str(b['frozen_encoder_ok']):>8}"
              f"  {'YES' if b['pass'] else 'no':>5}")
    print(f"\n  tasks passing the v3.1 noise-floor bar: "
          f"{report['n_tasks_pass']}/{report['n_tasks_total']} "
          f"(acceptance needs >= {report['min_tasks_pass']})")
    print("  Read: oracle@10 = achievable ceiling on qh1's candidate set; "
          "gap@10 = ordering headroom (how much a reranker could add).")
    print("  Thesis: a LARGE gap@10 on geometry-sensitive tasks vs the "
          "locality-friendly temporal task is the WS3 reranker target.")
    if not report["acceptance_pass"]:
        print("\n  ACCEPTANCE NOT MET. Decision tree: for a failing task "
              "walk qh1->qh2->qh3; if none clears, report (best_mean,std) "
              "honestly and flag the weak task. Never touch the encoder; "
              "never MSE. If frozen_encoder_ok is False -> gradient-leak "
              "BUG, stop and fix plumbing.")
    print(f"\n  results: {out_root / 'sweep_taskdiversity_results.json'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str,
                    default="src/modelsv3/sweep_config_taskdiversity.json")
    ap.add_argument("--smoke", action="store_true",
                    help="One cell (first task, first seed, 1 epoch).")
    args = ap.parse_args()
    return run_sweep(Path(args.config), smoke=args.smoke)


if __name__ == "__main__":
    sys.exit(main())

r"""Parallel architecture sweep driver for KGR v3.

Same as sweep_architecture.py, but runs multiple cells concurrently as
separate subprocesses. Each subprocess gets a bounded thread budget
(OMP_NUM_THREADS, MKL_NUM_THREADS) so PyTorch's intra-op threading
doesn't cause 4 parallel processes to fight over all 16 cores.

Defaults (for a 12-16 core workstation):
    --parallel 3      run 3 cells concurrently
    --threads 4       each subprocess uses 4 threads

For a larger CPU (e.g. 24+ cores), try --parallel 4 --threads 6.

Crash-resumable in the same way as the serial version: any cell whose
outputs exist is skipped.

Usage
-----
    python -m src.modelsv3.sweep_architecture_parallel \
        --config src/modelsv3/sweep_config_hyp.json \
        --parallel 3 --threads 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


STATUS_FILENAME = "sweep_status.json"
SUMMARY_FILENAME = "summary.json"
INTRINSIC_FILENAME = "intrinsic_eval.json"
COLLAPSE_FILENAME = "collapse.json"


def _cell_name(hidden_dim: int, num_layers: int, seed: int) -> str:
    return f"h{hidden_dim}_l{num_layers}_seed{seed}"


def _cell_is_complete(cell_dir: Path, need_intrinsic: bool, need_collapse: bool) -> bool:
    if not (cell_dir / SUMMARY_FILENAME).exists():
        return False
    if need_intrinsic and not (cell_dir / INTRINSIC_FILENAME).exists():
        return False
    if need_collapse and not (cell_dir / COLLAPSE_FILENAME).exists():
        return False
    try:
        json.loads((cell_dir / SUMMARY_FILENAME).read_text())
    except json.JSONDecodeError:
        return False
    return True


def _partial_needs(cell_dir: Path, need_intrinsic: bool, need_collapse: bool) -> dict:
    return {
        "train": not (cell_dir / SUMMARY_FILENAME).exists(),
        "intrinsic": need_intrinsic and not (cell_dir / INTRINSIC_FILENAME).exists(),
        "collapse": need_collapse and not (cell_dir / COLLAPSE_FILENAME).exists(),
    }


def _train_cmd(cfg: dict, hidden_dim: int, num_layers: int, seed: int, out_dir: Path) -> list[str]:
    fx = cfg["fixed_hyperparameters"]
    return [
        sys.executable, "-m", "src.training.train_v3",
        "--task", str(cfg["task"]),
        "--model", cfg["geometry"],
        "--corpus", cfg["corpus"],
        "--out", str(out_dir),
        "--hidden-dim", str(hidden_dim),
        "--num-layers", str(num_layers),
        "--seed", str(seed),
        "--contrastive-epochs", str(fx["contrastive_epochs"]),
        "--query-epochs", str(fx["query_epochs"]),
        "--lr", str(fx["lr"]),
        "--anchors-per-step", str(fx["anchors_per_step"]),
        "--log-every", str(fx["log_every"]),
        "--type-dim", str(fx["type_dim"]),
        "--tangent-scale", str(fx["tangent_scale"]),
        "--radial-reg-weight", str(fx["radial_reg_weight"]),
        "--radial-reg-weight-end", str(fx["radial_reg_weight_end"]),
        "--temperature", str(fx["temperature"]),
        "--positive-mix", str(fx["positive_mix"]),
        "--neighbor-exclude-k", str(fx["neighbor_exclude_k"]),
        "--margin", str(fx["margin"]),
        "--stage-b-loss", fx["stage_b_loss"],
        "--curvature", str(fx["curvature"]),
        "--train-graphs-frac", str(fx["train_graphs_frac"]),
        "--uniformity-reg-weight", str(fx.get("uniformity_reg_weight", 0.0)),
        "--uniformity-t", str(fx.get("uniformity_t", 2.0)),
    ]


def _eval_intrinsic_cmd(cell_dir: Path) -> list[str]:
    return [
        sys.executable, "-m", "src.modelsv3.eval_intrinsic_corpus",
        "--checkpoint", str(cell_dir / "encoder.pt"),
        "--out", str(cell_dir / INTRINSIC_FILENAME),
    ]


def _eval_collapse_cmd(cell_dir: Path) -> list[str]:
    return [
        sys.executable, "-m", "src.modelsv3.investigate_collapse",
        "--checkpoint", str(cell_dir / "encoder.pt"),
        "--out", str(cell_dir / COLLAPSE_FILENAME),
    ]


def _thread_env(n_threads: int) -> dict:
    """Environment variables that cap a subprocess's thread count.

    These are read by OpenMP, MKL, and NumExpr — the three layers that
    most commonly cause parallel PyTorch processes to each spawn
    N threads and fight each other for cores."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(n_threads)
    env["MKL_NUM_THREADS"] = str(n_threads)
    env["NUMEXPR_NUM_THREADS"] = str(n_threads)
    env["OPENBLAS_NUM_THREADS"] = str(n_threads)
    return env


def _run_one_cell(args: tuple) -> dict:
    """Worker function: runs the full pipeline for one cell.

    Returns a result dict the driver collects. Runs in a child process
    so stdout/stderr from subprocess.call won't interleave with other
    workers — each cell's logs go to its own files.
    """
    cfg, h, L, seed, out_root_str, n_threads, need_intrinsic, need_collapse = args
    out_root = Path(out_root_str)
    cell = _cell_name(h, L, seed)
    cell_dir = out_root / cell
    cell_dir.mkdir(exist_ok=True, parents=True)

    env = _thread_env(n_threads)
    needs = _partial_needs(cell_dir, need_intrinsic, need_collapse)
    t0 = time.time()
    result: dict[str, Any] = {"cell": cell, "h": h, "L": L, "seed": seed,
                              "started_at": time.strftime("%H:%M:%S"),
                              "stages": {}}

    # Training stage
    if needs["train"]:
        logpath = cell_dir / "train.log"
        with open(logpath, "w") as f:
            rc = subprocess.call(
                _train_cmd(cfg, h, L, seed, cell_dir),
                stdout=f, stderr=subprocess.STDOUT, env=env,
            )
        result["stages"]["train"] = {"rc": rc, "log": str(logpath)}
        if rc != 0:
            result["state"] = "failed_train"
            result["wall_seconds"] = round(time.time() - t0, 1)
            return result

    # Intrinsic eval
    if needs["intrinsic"]:
        logpath = cell_dir / "eval_intrinsic.log"
        with open(logpath, "w") as f:
            rc = subprocess.call(
                _eval_intrinsic_cmd(cell_dir),
                stdout=f, stderr=subprocess.STDOUT, env=env,
            )
        result["stages"]["intrinsic"] = {"rc": rc, "log": str(logpath)}
        if rc != 0:
            result["state"] = "failed_intrinsic"
            result["wall_seconds"] = round(time.time() - t0, 1)
            return result

    # Collapse diagnostic
    if needs["collapse"]:
        logpath = cell_dir / "eval_collapse.log"
        with open(logpath, "w") as f:
            rc = subprocess.call(
                _eval_collapse_cmd(cell_dir),
                stdout=f, stderr=subprocess.STDOUT, env=env,
            )
        result["stages"]["collapse"] = {"rc": rc, "log": str(logpath)}
        if rc != 0:
            result["state"] = "failed_collapse"
            result["wall_seconds"] = round(time.time() - t0, 1)
            return result

    result["state"] = "complete"
    result["wall_seconds"] = round(time.time() - t0, 1)
    return result


def _write_status(out_root: Path, status: dict) -> None:
    tmp = out_root / (STATUS_FILENAME + ".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(out_root / STATUS_FILENAME)


def _eta_string(elapsed_s: float, done: int, total: int, parallel: int) -> str:
    if done == 0:
        return "—"
    per_cell_sequential_equiv = elapsed_s / done
    remaining_sequential = (total - done) * per_cell_sequential_equiv
    effective_parallel = max(1, min(parallel, total - done))
    remaining = remaining_sequential / effective_parallel
    h, m = divmod(int(remaining / 60), 60)
    return f"{h}h{m:02d}m"


def run_sweep_parallel(
    config_path: Path, parallel: int, threads: int
) -> int:
    cfg = json.loads(config_path.read_text())
    grid = cfg["grid"]
    out_root = Path(cfg["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    need_intrinsic = cfg.get("eval", {}).get("run_intrinsic", True)
    need_collapse = cfg.get("eval", {}).get("run_collapse", True)

    all_cells = list(itertools.product(
        grid["hidden_dim"], grid["num_layers"], grid["seed"]
    ))
    total = len(all_cells)

    # Pre-filter: separate the already-done from the to-do.
    todo: list[tuple] = []
    already_complete = 0
    for h, L, s in all_cells:
        cell_dir = out_root / _cell_name(h, L, s)
        if _cell_is_complete(cell_dir, need_intrinsic, need_collapse):
            already_complete += 1
        else:
            todo.append((cfg, h, L, s, str(out_root), threads,
                         need_intrinsic, need_collapse))

    print(f"[driver] sweep grid: {total} cells "
          f"({len(grid['hidden_dim'])} dims x {len(grid['num_layers'])} layers "
          f"x {len(grid['seed'])} seeds)")
    print(f"[driver] output root: {out_root}")
    print(f"[driver] geometry: {cfg['geometry']}  "
          f"uniformity_w={cfg['fixed_hyperparameters'].get('uniformity_reg_weight', 0.0)}")
    print(f"[driver] parallel: {parallel} workers  x  {threads} threads/worker = "
          f"{parallel * threads} cores max")
    print(f"[driver] already complete: {already_complete}/{total}")
    print(f"[driver] to run: {len(todo)}")

    if not todo:
        print("[driver] nothing to do — all cells complete.")
        return 0

    status = {
        "config_path": str(config_path),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parallel": parallel,
        "threads_per_worker": threads,
        "total_cells": total,
        "already_complete": already_complete,
        "completed": 0,
        "skipped": already_complete,
        "failed": [],
        "in_flight": [],
        "eta": None,
        "cells": {},
    }
    _write_status(out_root, status)

    t_start = time.time()
    done_this_run = 0
    with ProcessPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(_run_one_cell, job): job for job in todo}
        for job in todo:
            _cfg, h, L, s, _root, _t, _ni, _nc = job
            status["cells"][_cell_name(h, L, s)] = {"state": "queued"}
        _write_status(out_root, status)

        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as e:
                job = futures[fut]
                _cfg, h, L, s, _r, _t, _ni, _nc = job
                cell = _cell_name(h, L, s)
                print(f"[driver] EXCEPTION in {cell}: {e}")
                status["failed"].append({"cell": cell, "error": str(e)})
                status["cells"][cell] = {"state": "exception", "error": str(e)}
                done_this_run += 1
                _write_status(out_root, status)
                continue

            cell = result["cell"]
            status["cells"][cell] = result
            done_this_run += 1

            if result["state"] == "complete":
                status["completed"] += 1
                elapsed_min = result["wall_seconds"] / 60
                eta = _eta_string(
                    time.time() - t_start, done_this_run,
                    len(todo), parallel,
                )
                status["eta"] = eta
                print(f"[driver] DONE  {cell}  ({elapsed_min:.1f} min)  "
                      f"{done_this_run}/{len(todo)}  eta={eta}")
            else:
                status["failed"].append({
                    "cell": cell, "stage": result["state"], "wall_s": result["wall_seconds"],
                })
                print(f"[driver] FAIL  {cell}  state={result['state']}  "
                      f"(check {out_root / cell}/*.log)")

            _write_status(out_root, status)

    status["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["total_wall_seconds"] = round(time.time() - t_start, 1)
    _write_status(out_root, status)

    print()
    print(f"[driver] sweep finished in {status['total_wall_seconds']/60:.1f} min")
    print(f"[driver]   previously complete: {status['skipped']}")
    print(f"[driver]   completed this run:  {status['completed']}")
    print(f"[driver]   failed:              {len(status['failed'])}")
    if status["failed"]:
        for f in status["failed"]:
            print(f"[driver]     {f}")
    return 0 if not status["failed"] else 2


def _validate_quick_check_outputs(cell_dir: Path) -> tuple[bool, list[str]]:
    """Inspect the outputs of a completed cell and verify they contain
    the fields the analyzer (sweep_analyze.py) expects to read.

    Returns (all_ok, problems). Problems list is human-readable strings
    describing exactly what's missing or malformed.

    This is the distinguishing feature of quick-check over "did the
    script run and produce files" — it catches cases where the script
    succeeds but the schema has drifted and the analyzer would silently
    produce NaN tables."""
    problems: list[str] = []

    # summary.json: need final_val.overall.ndcg@10
    summary_path = cell_dir / "summary.json"
    if not summary_path.exists():
        problems.append(f"{summary_path.name} missing")
    else:
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"{summary_path.name} is not valid JSON: {e}")
            summary = None
        if summary is not None:
            v = summary.get("final_val", {}).get("overall", {}).get("ndcg@10")
            if v is None:
                problems.append(
                    "summary.json missing final_val.overall.ndcg@10 "
                    "— analyzer will return NaN for nDCG"
                )
            elif not isinstance(v, (int, float)):
                problems.append(
                    f"summary.json final_val.overall.ndcg@10 is "
                    f"{type(v).__name__}, expected number"
                )

    # intrinsic_eval.json: the analyzer pulls per_graph[*].{edge_prec_mean,
    # label_purity_mean, silhouette_mean} and averages. Verify the per_graph
    # array exists and has the expected per-graph fields.
    intrinsic_path = cell_dir / "intrinsic_eval.json"
    if not intrinsic_path.exists():
        problems.append(f"{intrinsic_path.name} missing")
    else:
        try:
            intrinsic = json.loads(intrinsic_path.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"{intrinsic_path.name} is not valid JSON: {e}")
            intrinsic = None
        if intrinsic is not None:
            per_graph = intrinsic.get("per_graph")
            if not isinstance(per_graph, list) or not per_graph:
                problems.append(
                    f"intrinsic_eval.json: 'per_graph' is missing or empty. "
                    f"Top-level keys were: {sorted(intrinsic.keys())}"
                )
            else:
                first = per_graph[0]
                missing_fields = [
                    f for f in ("edge_prec_mean", "label_purity_mean",
                                "silhouette_mean")
                    if not isinstance(first, dict) or f not in first
                ]
                if missing_fields:
                    problems.append(
                        f"intrinsic_eval.json: per_graph[0] is missing "
                        f"field(s): {missing_fields}. Present keys: "
                        f"{sorted(first.keys()) if isinstance(first, dict) else first}"
                    )

    # collapse.json: the analyzer pulls q1_q2_aggregate.
    # frac_below_threshold['1e-04'].mean as the collapse rate. Verify
    # the nested path resolves.
    collapse_path = cell_dir / "collapse.json"
    if not collapse_path.exists():
        problems.append(f"{collapse_path.name} missing")
    else:
        try:
            collapse = json.loads(collapse_path.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"{collapse_path.name} is not valid JSON: {e}")
            collapse = None
        if collapse is not None:
            agg = collapse.get("q1_q2_aggregate")
            if not isinstance(agg, dict):
                problems.append(
                    f"collapse.json: 'q1_q2_aggregate' is missing or not a dict. "
                    f"Top-level keys were: {sorted(collapse.keys())}"
                )
            else:
                fbt = agg.get("frac_below_threshold")
                if not isinstance(fbt, dict):
                    problems.append(
                        f"collapse.json: q1_q2_aggregate lacks 'frac_below_threshold'. "
                        f"Present keys: {sorted(agg.keys())}"
                    )
                else:
                    block = fbt.get("1e-04")
                    if not (
                        isinstance(block, dict) and "mean" in block
                    ) and not isinstance(block, (int, float)):
                        problems.append(
                            f"collapse.json: frac_below_threshold lacks the "
                            f"'1e-04' block or it has no 'mean'. Present keys: "
                            f"{sorted(fbt.keys())}"
                        )

    return len(problems) == 0, problems


def run_quick_check(config_path: Path, threads: int = 4) -> int:
    """Run ONE small cell end-to-end, verify it produced valid outputs,
    report PASS/FAIL.

    Purpose: catch plumbing bugs before committing compute to the full
    sweep. Uses the smallest configuration from the grid with a short
    training schedule.

    Takes ~2-4 minutes on a normal workstation.
    """
    cfg = json.loads(config_path.read_text())
    quick_root = Path("runs/quickcheck")
    quick_root.mkdir(parents=True, exist_ok=True)

    # Pick the smallest (h, L) from the grid so training is fastest.
    smallest_h = min(cfg["grid"]["hidden_dim"])
    smallest_L = min(cfg["grid"]["num_layers"])
    seed = cfg["grid"]["seed"][0]

    # Override epochs to 2+2 for speed. Preserve the rest of the config.
    quick_cfg = json.loads(json.dumps(cfg))  # deep copy
    quick_cfg["fixed_hyperparameters"]["contrastive_epochs"] = 2
    quick_cfg["fixed_hyperparameters"]["query_epochs"] = 2
    quick_cfg["fixed_hyperparameters"]["log_every"] = 50

    # IMPORTANT: `_run_one_cell` computes its output directory as
    # `out_root / _cell_name(h, L, seed)` — it does NOT use the
    # display-friendly `quick_` prefix. So the actual write location
    # is `quick_root / _cell_name(...)`, NOT `quick_root / cell_name`
    # (the display name). We pre-compute both and pass them properly.
    display_name = f"quick_h{smallest_h}_l{smallest_L}_seed{seed}"
    actual_cell_dir = quick_root / _cell_name(smallest_h, smallest_L, seed)

    # Clean slate: remove any stale artifacts from prior quick-checks
    # (in both the display dir, which is legacy, and the real write dir).
    for stale_dir in (quick_root / display_name, actual_cell_dir):
        if stale_dir.exists():
            for f in stale_dir.iterdir():
                try:
                    f.unlink()
                except Exception:
                    pass
    actual_cell_dir.mkdir(parents=True, exist_ok=True)

    print(f"[quick-check] config: h={smallest_h}, L={smallest_L}, seed={seed}")
    print(f"[quick-check] epochs: 2 stage-A + 2 stage-B (reduced for speed)")
    print(f"[quick-check] output: {actual_cell_dir}")
    print(f"[quick-check] geometry: {cfg['geometry']}")
    print()

    job = (quick_cfg, smallest_h, smallest_L, seed, str(quick_root),
           threads, True, True)

    t0 = time.time()
    result = _run_one_cell(job)
    wall_s = time.time() - t0

    print()
    print("=" * 72)
    print(f"Quick-check result: {result['state']}  ({wall_s:.0f} s wall)")
    print("=" * 72)

    if result["state"] != "complete":
        print(f"\nFAIL: pipeline stage '{result['state']}' did not succeed.")
        stages = result.get("stages", {})
        for stage_name, info in stages.items():
            rc = info.get("rc", "?")
            log = info.get("log", "—")
            print(f"  {stage_name}: rc={rc}  log={log}")
        print(f"\nCheck the log files in {actual_cell_dir} to diagnose.")
        print(f"Common causes: module import errors, CLI argument name mismatches,")
        print(f"  missing corpus, or Python-env issues.")
        return 1

    # Pipeline ran. Now check the outputs are parseable by the analyzer.
    ok, problems = _validate_quick_check_outputs(actual_cell_dir)

    if ok:
        print("\nPASS: all pipeline stages completed, all outputs validated.")
        print("\nThe outputs contain the fields sweep_analyze.py expects to read:")
        print("  - summary.json: final_val.overall.ndcg@10  ✓")
        print("  - intrinsic_eval.json: edge_prec@5  ✓")
        print("  - collapse.json: collapse_rate  ✓")
        print("\nSafe to launch the full sweep.")
        return 0

    print("\nFAIL: pipeline ran but outputs did not pass schema validation.")
    print("\nProblems:")
    for p in problems:
        print(f"  - {p}")
    print()
    print("The full sweep would run to completion, but sweep_analyze.py would")
    print("silently produce NaN values for the missing metrics.  Fix either:")
    print("  (a) the eval script that produced the JSON, or")
    print("  (b) the field-extraction logic in sweep_analyze.py, to match reality.")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, required=True,
                    help="Path to JSON sweep config.")
    ap.add_argument("--parallel", type=int, default=3,
                    help="Number of cells to run concurrently. Default 3 "
                         "(good for 12-16 core CPU with 4 threads each).")
    ap.add_argument("--threads", type=int, default=4,
                    help="Threads per subprocess (OMP/MKL). Default 4. "
                         "Total cores used = parallel * threads.")
    ap.add_argument("--quick-check", action="store_true",
                    help="Run ONE small cell (smallest h, L; 2+2 epochs) to "
                         "verify the pipeline works before committing hours "
                         "of compute. Writes to runs/quickcheck/. Takes "
                         "roughly 2-4 minutes on a typical workstation.")
    args = ap.parse_args()

    if args.quick_check:
        return run_quick_check(Path(args.config), threads=args.threads)

    if args.parallel * args.threads > 32:
        print(f"[warn] parallel ({args.parallel}) x threads ({args.threads}) = "
              f"{args.parallel * args.threads} total threads. This exceeds most "
              f"workstation core counts; you'll likely see throughput degrade.")

    return run_sweep_parallel(Path(args.config), args.parallel, args.threads)


if __name__ == "__main__":
    sys.exit(main())

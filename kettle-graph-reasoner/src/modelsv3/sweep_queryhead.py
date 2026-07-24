r"""v3.1 Phase 2 — query-head sweep driver (frozen encoder).

Trains only the ``QueryToBall`` head (stage B) on top of ONE frozen
encoder loaded from the locked baseline, SHA-asserted. qh0..qh4 x
seeds (x norm for the variants that use it). No stage-A re-pretrain,
no per-arm encoder confound.

Per cell: ``train_v3 --skip-stage-a --load-encoder ... --query-head-arch
...`` then ``eval_candidate_recall``. Resumable: a cell whose
``summary.json`` + ``candidate_recall.json`` exist is skipped.

Gate (Phase-2 §2.6): the SMALLEST-parameter arch whose mean
``val_ndcg@10`` over seeds >= ``max(0.52, baseline_mean + 1*std)``,
with ``val_ndcg@20`` not regressed beyond noise and the frozen-encoder
assertion intact (intrinsic ``nn_edge_precision@5`` == baseline). The
gate rule lives in ``lock_baseline.gate_threshold``.

Standalone (does NOT touch ``sweep_architecture_parallel.py``) so the
working arch-sweep harness and its quick-check schema validator are
not put at risk — see v3.1 plan risks #5/#6.

Usage
-----
    py -m src.modelsv3.sweep_queryhead \
        --config src/modelsv3/sweep_config_queryhead.json
    py -m src.modelsv3.sweep_queryhead --config ... --smoke   # 1 cell, 1 epoch
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

# Param-count order (from the QueryToBall arch verification): the gate
# picks the SMALLEST passing arch, not the highest-scoring.
ARCH_PARAM_ORDER = ("qh0", "qh1", "qh4", "qh2", "qh3")


def _cells(cfg: dict) -> list[tuple[str, str, int]]:
    """(arch, norm, seed). Norm is only swept for archs that use it;
    others get a single layernorm cell."""
    norm_archs = set(cfg.get("norm_relevant_archs", ["qh2", "qh3"]))
    out: list[tuple[str, str, int]] = []
    for arch in cfg["archs"]:
        norms = cfg["norms"] if arch in norm_archs else ["layernorm"]
        for norm in norms:
            for seed in cfg["seeds"]:
                out.append((arch, norm, int(seed)))
    return out


def _cell_name(arch: str, norm: str, seed: int) -> str:
    return f"{arch}_{norm}_seed{seed}"


def _done(cell_dir: Path) -> bool:
    return (cell_dir / "summary.json").exists() and (
        cell_dir / "candidate_recall.json"
    ).exists()


def _run_cell(cfg: dict, arch: str, norm: str, seed: int,
              enc_sha: str, cell_dir: Path, query_epochs: int) -> dict:
    cell_dir.mkdir(parents=True, exist_ok=True)
    train_cmd = [
        sys.executable, "-m", "src.training.train_v3",
        "--task", str(cfg["task"]),
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
        "--query-head-arch", arch,
        "--query-head-norm", norm,
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
        "--task", str(cfg["task"]),
        "--out", str(cell_dir / "candidate_recall.json"),
    ]
    with open(cell_dir / "eval_candidate_recall.log", "w") as f:
        rc = subprocess.call(eval_cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        return {"state": "failed_eval", "rc": rc}
    return {"state": "complete"}


def _read_cell_metrics(cell_dir: Path) -> dict:
    s = json.loads((cell_dir / "summary.json").read_text())
    fv = s["final_val"]["overall"]
    ep = s.get("intrinsic_val_graph0", {}).get("nn_edge_precision@5", {})
    cr = json.loads((cell_dir / "candidate_recall.json").read_text())
    cro = cr["summary"]["overall"]
    return {
        "ndcg@10": fv.get("ndcg@10"),
        "ndcg@20": fv.get("ndcg@20"),
        "intrinsic_edge_prec@5": ep.get("mean_precision"),
        "recall@50": cro.get("recall@50"),
        "recall@100": cro.get("recall@100"),
        "oracle_gap@10|C50": cro.get("oracle_gap@10|C50"),
        "n_query_params": s.get("n_params_query"),
    }


def _mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var ** 0.5


def run_sweep(config_path: Path, smoke: bool) -> int:
    cfg = json.loads(config_path.read_text())
    manifest = load_manifest(Path(cfg["baseline_dir"]))
    enc_sha = manifest["encoder_sha256"]
    noise = manifest["noise_floor"]
    base_ep5 = manifest["frozen_metrics"]["intrinsic_val_graph0"][
        "nn_edge_precision@5"]["mean_precision"]
    out_root = Path(cfg["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    cells = _cells(cfg)
    query_epochs = cfg["query_epochs"]
    if smoke:
        cells = [("qh2", "layernorm", cfg["seeds"][0])]
        query_epochs = 1
        out_root = out_root / "_smoke"
        out_root.mkdir(parents=True, exist_ok=True)

    print(f"[qh-sweep] {len(cells)} cells  encoder={cfg['encoder_path']}")
    print(f"[qh-sweep] sha-asserted={enc_sha[:12]}...  query_epochs={query_epochs}")
    t0 = time.time()
    results: dict[str, dict] = {}
    for i, (arch, norm, seed) in enumerate(cells):
        name = _cell_name(arch, norm, seed)
        cell_dir = out_root / name
        if _done(cell_dir):
            print(f"[qh-sweep] ({i+1}/{len(cells)}) {name} - skip (done)")
        else:
            print(f"[qh-sweep] ({i+1}/{len(cells)}) {name} - running...")
            st = _run_cell(cfg, arch, norm, seed, enc_sha, cell_dir,
                           query_epochs)
            if st["state"] != "complete":
                print(f"[qh-sweep]   FAIL {name}: {st} "
                      f"(see {cell_dir}/*.log)")
                results[name] = {"state": st["state"]}
                continue
        m = _read_cell_metrics(cell_dir)
        m["arch"], m["norm"], m["seed"] = arch, norm, seed
        m["state"] = "complete"
        results[name] = m

    report = _gate(results, cfg, noise, base_ep5)
    out = {
        "config": str(config_path),
        "baseline_dir": cfg["baseline_dir"],
        "encoder_sha256": enc_sha,
        "noise_floor": noise,
        "baseline_intrinsic_edge_prec@5": base_ep5,
        "cells": results,
        "gate": report,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out_root / "sweep_queryhead_results.json").write_text(
        json.dumps(out, indent=2))
    _print_report(report, out_root)
    return 0 if report.get("any_pass") else 2


def _gate(results: dict, cfg: dict, noise: dict, base_ep5: float) -> dict:
    """Aggregate by arch across seeds; apply the Phase-2 gate."""
    spec_target = cfg.get("gate", {}).get("ndcg10_target", 0.52)
    nd_mean = noise["ndcg@10"]["mean"]
    threshold = max(spec_target,
                    gate_threshold(noise, "ndcg@10", spec_target - nd_mean))
    nd20_floor = noise["ndcg@20"]["mean"] - noise["ndcg@20"]["std"]

    by_arch: dict[str, dict] = {}
    arch_set = sorted({v["arch"] for v in results.values()
                       if v.get("state") == "complete"},
                      key=lambda a: ARCH_PARAM_ORDER.index(a)
                      if a in ARCH_PARAM_ORDER else 99)
    for arch in arch_set:
        rows = [v for v in results.values()
                if v.get("state") == "complete" and v["arch"] == arch]
        nd10 = [r["ndcg@10"] for r in rows]
        nd20 = [r["ndcg@20"] for r in rows]
        ep5 = [r["intrinsic_edge_prec@5"] for r in rows]
        r50 = [r["recall@50"] for r in rows]
        m10, s10 = _mean_std(nd10)
        m20, s20 = _mean_std(nd20)
        # Frozen-encoder correctness: every cell's intrinsic edge_prec@5
        # must equal the baseline (encoder never trained). Any drift is a
        # BUG, not a tuning result.
        frozen_ok = all(
            e is not None and abs(e - base_ep5) < 1e-9 for e in ep5)
        passed = (
            m10 >= threshold
            and m20 >= nd20_floor
            and frozen_ok
        )
        by_arch[arch] = {
            "n_seeds": len(rows),
            "ndcg@10_mean": m10, "ndcg@10_std": s10,
            "ndcg@20_mean": m20, "ndcg@20_std": s20,
            "recall@50_mean": _mean_std(r50)[0],
            "n_query_params": rows[0]["n_query_params"] if rows else None,
            "frozen_encoder_ok": frozen_ok,
            "pass": bool(passed),
        }

    passing = [a for a in arch_set if by_arch[a]["pass"]]
    # smallest-parameter passing arch wins
    passing.sort(key=lambda a: ARCH_PARAM_ORDER.index(a)
                 if a in ARCH_PARAM_ORDER else 99)
    return {
        "threshold_ndcg@10": threshold,
        "ndcg@20_floor": nd20_floor,
        "by_arch": by_arch,
        "passing_archs": passing,
        "selected_arch": passing[0] if passing else None,
        "any_pass": bool(passing),
    }


def _print_report(report: dict, out_root: Path) -> None:
    print()
    print("=" * 84)
    print("v3.1 Phase 2 - query-head sweep gate")
    print(f"ndcg@10 gate >= {report['threshold_ndcg@10']:.4f}  "
          f"ndcg@20 floor >= {report['ndcg@20_floor']:.4f}")
    print("=" * 84)
    print(f"  {'arch':<6}{'params':>9}{'ndcg@10':>18}{'ndcg@20':>14}"
          f"{'recall@50':>11}{'frozenOK':>10}{'PASS':>6}")
    for arch, b in report["by_arch"].items():
        print(f"  {arch:<6}{(b['n_query_params'] or 0):>9}"
              f"  {b['ndcg@10_mean']:.4f}+-{b['ndcg@10_std']:.4f}"
              f"  {b['ndcg@20_mean']:.4f}"
              f"  {b['recall@50_mean']:.4f}"
              f"  {str(b['frozen_encoder_ok']):>8}"
              f"  {'YES' if b['pass'] else 'no':>5}")
    sel = report["selected_arch"]
    if sel:
        print(f"\nSELECTED (smallest passing): {sel}  "
              f"-> use --query-head-arch {sel} for Phase 3.")
    else:
        print("\nNO ARCH CLEARED THE GATE. Decision tree: walk qh1->qh2->qh3; "
              "if none clears, ship best arch and lower the v3.1 ndcg claim "
              "to (best_mean, std). Do NOT touch the encoder.")
    print(f"\nresults: {out_root / 'sweep_queryhead_results.json'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str,
                    default="src/modelsv3/sweep_config_queryhead.json")
    ap.add_argument("--smoke", action="store_true",
                    help="One cell (qh2, 1 query epoch) to validate plumbing.")
    args = ap.parse_args()
    return run_sweep(Path(args.config), smoke=args.smoke)


if __name__ == "__main__":
    sys.exit(main())

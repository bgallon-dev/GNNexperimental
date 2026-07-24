r"""v3.1 Phase 3 — stage-B loss sweep driver (frozen encoder).

Compares the pairwise-hinge stage-B baseline against sampled
multi-positive InfoNCE across negative-pool size and temperature, on
the frozen baseline encoder + the Phase-2-winning query head. No
stage-A re-pretrain.

Phase-3 gate (§3.5): an InfoNCE config PASSES if, vs the pairwise arm:
  - recall@50 AND recall@100 up by more than one combined std, and
  - lower std of stage-B ``rank_accuracy`` over the last
    ``volatility_tail_frac`` of steps (and across seeds), and
  - val ndcg@10 not regressed beyond the baseline noise floor.
Pairwise stays the shipped default; InfoNCE is opt-in.

Resumable. Standalone (does not touch the arch-sweep harness). Reuses
``sweep_queryhead`` helpers + ``lock_baseline.gate_threshold``.

Usage
-----
    py -m src.modelsv3.sweep_stagebloss \
        --config src/modelsv3/sweep_config_stagebloss.json
    py -m src.modelsv3.sweep_stagebloss --config ... --smoke
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
from src.modelsv3.sweep_queryhead import _mean_std, _read_cell_metrics  # noqa: E402


def _resolve_arch(cfg: dict) -> str:
    arch = cfg.get("query_head_arch", "auto")
    if arch != "auto":
        return arch
    fallback = cfg.get("query_head_arch_fallback", "qh2")
    p2 = Path(cfg.get("p2_results", ""))
    if p2.exists():
        try:
            sel = json.loads(p2.read_text()).get("gate", {}).get("selected_arch")
            if sel:
                print(f"[sb-sweep] resolved query_head_arch={sel} from {p2}")
                return sel
        except json.JSONDecodeError:
            pass
    print(f"[sb-sweep] P2 results unusable; falling back to "
          f"query_head_arch={fallback} (decision tree: ship best/qh2)")
    return fallback


def _cells(cfg: dict) -> list[dict]:
    cells: list[dict] = []
    if cfg.get("pairwise_baseline", True):
        for s in cfg["seeds"]:
            cells.append({"loss": "pairwise", "seed": int(s)})
    for ng in cfg["infonce_negatives"]:
        for t in cfg["infonce_temperatures"]:
            for s in cfg["seeds"]:
                cells.append({"loss": "infonce", "neg": int(ng),
                              "temp": float(t), "seed": int(s)})
    return cells


def _cell_name(c: dict) -> str:
    if c["loss"] == "pairwise":
        return f"pairwise_seed{c['seed']}"
    return f"infonce_n{c['neg']}_t{c['temp']}_seed{c['seed']}"


def _done(d: Path) -> bool:
    return (d / "summary.json").exists() and (d / "candidate_recall.json").exists()


def _rank_acc_tail_std(cell_dir: Path, tail_frac: float) -> float:
    """std of stage-B rank_accuracy over the last tail_frac of steps —
    the per-cell volatility the Phase-3 gate watches."""
    hp = cell_dir / "stage_b_history.json"
    if not hp.exists():
        return float("nan")
    hist = json.loads(hp.read_text())
    accs = [h["rank_accuracy"] for h in hist if "rank_accuracy" in h]
    if len(accs) < 2:
        return float("nan")
    tail = accs[max(1, int(len(accs) * (1.0 - tail_frac))):]
    return _mean_std(tail)[1]


def _run_cell(cfg: dict, arch: str, c: dict, enc_sha: str,
              cell_dir: Path, query_epochs: int) -> dict:
    cell_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "src.training.train_v3",
        "--task", str(cfg["task"]),
        "--model", cfg.get("model", "hyperbolic"),
        "--corpus", cfg["corpus"],
        "--out", str(cell_dir),
        "--hidden-dim", str(cfg["hidden_dim"]),
        "--num-layers", str(cfg["num_layers"]),
        "--curvature", str(cfg.get("curvature", 1.0)),
        "--seed", str(c["seed"]),
        "--skip-stage-a",
        "--load-encoder", cfg["encoder_path"],
        "--assert-encoder-sha", enc_sha,
        "--query-head-arch", arch,
        "--query-head-norm", cfg.get("query_head_norm", "layernorm"),
        "--query-epochs", str(query_epochs),
        "--lr-query", str(cfg["lr_query"]),
        "--stage-b-loss", c["loss"],
        "--log-every", str(cfg.get("log_every", 200)),
    ]
    if c["loss"] == "infonce":
        cmd += [
            "--stage-b-negatives", str(c["neg"]),
            "--stage-b-temperature", str(c["temp"]),
            "--stage-b-n-positives", str(cfg.get("infonce_n_positives", 8)),
        ]
    with open(cell_dir / "train.log", "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        return {"state": "failed_train", "rc": rc}
    ev = [
        sys.executable, "-m", "src.modelsv3.eval_candidate_recall",
        "--checkpoint", str(cell_dir / "encoder.pt"),
        "--task", str(cfg["task"]),
        "--out", str(cell_dir / "candidate_recall.json"),
    ]
    with open(cell_dir / "eval_candidate_recall.log", "w") as f:
        rc = subprocess.call(ev, stdout=f, stderr=subprocess.STDOUT)
    return {"state": "complete" if rc == 0 else "failed_eval", "rc": rc}


def run_sweep(config_path: Path, smoke: bool) -> int:
    cfg = json.loads(config_path.read_text())
    manifest = load_manifest(Path(cfg["baseline_dir"]))
    enc_sha = manifest["encoder_sha256"]
    noise = manifest["noise_floor"]
    arch = _resolve_arch(cfg)
    out_root = Path(cfg["out_root"])
    tail_frac = cfg.get("volatility_tail_frac", 0.2)

    cells = _cells(cfg)
    query_epochs = cfg["query_epochs"]
    if smoke:
        cells = [{"loss": "pairwise", "seed": cfg["seeds"][0]},
                 {"loss": "infonce", "neg": 128, "temp": 1.0,
                  "seed": cfg["seeds"][0]}]
        query_epochs = 1
        out_root = out_root / "_smoke"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[sb-sweep] arch={arch}  {len(cells)} cells  "
          f"sha-asserted={enc_sha[:12]}...  query_epochs={query_epochs}")
    t0 = time.time()
    results: dict[str, dict] = {}
    for i, c in enumerate(cells):
        name = _cell_name(c)
        cd = out_root / name
        if _done(cd):
            print(f"[sb-sweep] ({i+1}/{len(cells)}) {name} - skip (done)")
        else:
            print(f"[sb-sweep] ({i+1}/{len(cells)}) {name} - running...")
            st = _run_cell(cfg, arch, c, enc_sha, cd, query_epochs)
            if st["state"] != "complete":
                print(f"[sb-sweep]   FAIL {name}: {st} (see {cd}/*.log)")
                results[name] = {"state": st["state"], **c}
                continue
        m = _read_cell_metrics(cd)
        m.update(c)
        m["rank_acc_tail_std"] = _rank_acc_tail_std(cd, tail_frac)
        m["state"] = "complete"
        results[name] = m

    report = _gate(results, noise)
    out = {
        "config": str(config_path),
        "resolved_query_head_arch": arch,
        "encoder_sha256": enc_sha,
        "noise_floor": noise,
        "cells": results,
        "gate": report,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out_root / "sweep_stagebloss_results.json").write_text(
        json.dumps(out, indent=2))
    _print_report(report, out_root)
    return 0 if report.get("any_infonce_pass") else 2


def _agg(rows: list[dict], key: str) -> tuple[float, float]:
    return _mean_std([r.get(key) for r in rows])


def _gate(results: dict, noise: dict) -> dict:
    complete = [v for v in results.values() if v.get("state") == "complete"]
    pw = [r for r in complete if r["loss"] == "pairwise"]
    if not pw:
        return {"error": "no completed pairwise baseline arm",
                "any_infonce_pass": False}

    pw_r50_m, pw_r50_s = _agg(pw, "recall@50")
    pw_r100_m, pw_r100_s = _agg(pw, "recall@100")
    pw_vol_m, _ = _agg(pw, "rank_acc_tail_std")
    pw_vol_cross = _mean_std([r["recall@50"] for r in pw])[1]
    nd10_floor = noise["ndcg@10"]["mean"] - noise["ndcg@10"]["std"]

    # group infonce by (neg, temp)
    groups: dict[str, list[dict]] = {}
    for r in complete:
        if r["loss"] != "infonce":
            continue
        groups.setdefault(f"n{r['neg']}_t{r['temp']}", []).append(r)

    arms: dict[str, dict] = {}
    for gname, rows in sorted(groups.items()):
        r50_m, r50_s = _agg(rows, "recall@50")
        r100_m, r100_s = _agg(rows, "recall@100")
        nd10_m, _ = _agg(rows, "ndcg@10")
        vol_m, _ = _agg(rows, "rank_acc_tail_std")
        vol_cross = _mean_std([x["recall@50"] for x in rows])[1]
        recall_up = (
            r50_m > pw_r50_m + (pw_r50_s + r50_s)
            and r100_m > pw_r100_m + (pw_r100_s + r100_s)
        )
        less_volatile = (vol_m < pw_vol_m) and (vol_cross <= pw_vol_cross)
        ndcg_ok = nd10_m >= nd10_floor
        arms[gname] = {
            "n_seeds": len(rows),
            "recall@50_mean": r50_m, "recall@100_mean": r100_m,
            "ndcg@10_mean": nd10_m,
            "rank_acc_tail_std_mean": vol_m,
            "recall_above_pairwise": bool(recall_up),
            "less_volatile_than_pairwise": bool(less_volatile),
            "ndcg10_not_regressed": bool(ndcg_ok),
            "pass": bool(recall_up and less_volatile and ndcg_ok),
        }
    passing = [g for g, a in arms.items() if a["pass"]]
    return {
        "pairwise_baseline": {
            "recall@50_mean": pw_r50_m, "recall@100_mean": pw_r100_m,
            "rank_acc_tail_std_mean": pw_vol_m,
        },
        "ndcg@10_floor": nd10_floor,
        "infonce_arms": arms,
        "passing_infonce_arms": passing,
        "any_infonce_pass": bool(passing),
        "decision": (
            f"ship infonce ({passing[0]}) opt-in" if passing
            else "keep pairwise default; infonce reported as negative result "
                 "(do NOT delete pairwise)"
        ),
    }


def _print_report(report: dict, out_root: Path) -> None:
    print()
    print("=" * 90)
    print("v3.1 Phase 3 - stage-B loss gate")
    print("=" * 90)
    if "error" in report:
        print("  ERROR:", report["error"])
        return
    pw = report["pairwise_baseline"]
    print(f"  pairwise baseline: recall@50={pw['recall@50_mean']:.4f} "
          f"recall@100={pw['recall@100_mean']:.4f} "
          f"rank_acc_tail_std={pw['rank_acc_tail_std_mean']:.4f}")
    print(f"  ndcg@10 floor >= {report['ndcg@10_floor']:.4f}")
    print(f"  {'infonce arm':<16}{'r@50':>9}{'r@100':>9}{'ndcg@10':>9}"
          f"{'volStd':>9}{'PASS':>6}")
    for g, a in report["infonce_arms"].items():
        print(f"  {g:<16}{a['recall@50_mean']:>9.4f}{a['recall@100_mean']:>9.4f}"
              f"{a['ndcg@10_mean']:>9.4f}{a['rank_acc_tail_std_mean']:>9.4f}"
              f"{('YES' if a['pass'] else 'no'):>6}")
    print(f"\n  decision: {report['decision']}")
    print(f"  results: {out_root / 'sweep_stagebloss_results.json'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str,
                    default="src/modelsv3/sweep_config_stagebloss.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    return run_sweep(Path(args.config), smoke=args.smoke)


if __name__ == "__main__":
    sys.exit(main())

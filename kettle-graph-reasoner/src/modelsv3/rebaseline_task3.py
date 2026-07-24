r"""Phase 0 - honest task-3 re-baseline (zero training, pure aggregation).

The sweep gate's ``beats_ws3`` compares gap-closed FRACTIONS computed off
DIFFERENT retriever baselines (WS3's per-task v2 ran off a weak 0.194
retriever; v3.2/v3.3 off the strong qh1 0.313 retriever). Absolute
ndcg@10 IS comparable (IDCG is over the full label vector), so this
script re-scores task 3 on one comparable basis:

  R = qh1 retriever ndcg@10 (the basis v3.2/v3.3/router actually use)
  O = oracle ndcg@10 over that retriever's top-50

and reports, per system, absolute ndcg@10, abs_gain = dep - R, and the
SAME-retriever gap_same = (dep - R)/(O - R). WS3's own gap is kept but
explicitly labelled NOT comparable. The corrected success bar for any
NEW task-3 system is stated against the best EXISTING absolute (the
v3.3-blend deployed mean), not the cross-baseline WS3 fraction.

Pure JSON aggregation; no torch; ASCII only (Windows cp1252).

Usage
-----
    py -m src.modelsv3.rebaseline_task3
    py -m src.modelsv3.rebaseline_task3 --v34-results runs/sweep_reranker_v34/sweep_reranker_v32_results.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.modelsv3.sweep_queryhead import _mean_std  # noqa: E402  (reuse)

TASK = 3
FLOOR_MIN = 0.01  # absolute-ndcg margin floor (plan: max(combined_std, 0.01))


def _cells_task3(results_path: Path, key: str) -> list[float]:
    """Per-seed task-3 values from a sweep_reranker_* results json.
    Cells store metrics FLAT at top level (not nested under 'summary')."""
    if not results_path.exists():
        return []
    r = json.loads(results_path.read_text())
    out = []
    for name, v in (r.get("cells") or {}).items():
        if (v.get("state") == "complete" and int(v.get("task", -1)) == TASK
                and v.get(key) is not None):
            out.append(float(v[key]))
    return out


def _ws3_task3(root: Path) -> tuple[list[float], list[float], float]:
    """WS3 per-task v2 task-3: (hybrid per seed, v31 per seed, oracle).
    Uses the winner config from sweep_v2reranker_results.json."""
    res = root / "sweep_v2reranker" / "sweep_v2reranker_results.json"
    cfg = "h96l3"
    if res.exists():
        winner = json.loads(res.read_text()).get("winner", "h96l3_seed0")
        cfg = winner.split("_seed")[0] if "_seed" in winner else "h96l3"
    hyb, v31, orc = [], [], []
    for f in sorted(glob.glob(str(root / "sweep_v2reranker" /
                                  f"hybrid_{cfg}_seed*_task{TASK}.json"))):
        s = json.loads(Path(f).read_text()).get("summary", {})
        if "hybrid_ndcg@10" in s:
            hyb.append(float(s["hybrid_ndcg@10"]))
            v31.append(float(s.get("v31_ndcg@10", float("nan"))))
            orc.append(float(s.get("oracle_ndcg@10", float("nan"))))
    o = next((x for x in orc if x == x), float("nan"))
    return hyb, v31, o


def _gap_same(dep: float, R: float, O: float) -> float:
    return (dep - R) / (O - R) if (O - R) > 1e-9 else float("nan")


def run(runs: Path, v34_results: Path | None) -> dict:
    v32 = runs / "sweep_reranker_v32" / "sweep_reranker_v32_results.json"
    v33 = runs / "sweep_reranker_v33" / "sweep_reranker_v32_results.json"

    retr = _cells_task3(v33, "v31_ndcg@10")          # qh1 retriever, per seed
    orac = _cells_task3(v33, "oracle_ndcg@10")       # oracle over qh1 top-50
    R, R_sd = _mean_std(retr)
    O, _ = _mean_std(orac)

    v32d = _cells_task3(v32, "hybrid_ndcg@10")
    v33d = _cells_task3(v33, "hybrid_ndcg@10")
    ws3_hyb, ws3_v31, ws3_o = _ws3_task3(runs)

    rows: list[dict] = []

    def _add(name: str, vals: list[float], comparable: bool,
             own_R: float | None = None, own_O: float | None = None) -> None:
        m, sd = _mean_std(vals)
        rows.append({
            "system": name,
            "n_seeds": len(vals),
            "abs_ndcg@10_mean": m,
            "abs_ndcg@10_std": sd,
            "abs_gain_vs_qh1": (m - R) if m == m else float("nan"),
            "gap_same_qh1_basis": (_gap_same(m, R, O) if m == m
                                   else float("nan")),
            "comparable_basis": comparable,
            "own_retriever_ndcg@10": own_R,
            "own_oracle_ndcg@10": own_O,
        })

    _add("qh1_retriever", retr, True)
    _add("v3.2-damped", v32d, True)
    _add("v3.3-blend (best existing)", v33d, True)
    # WS3: absolute IS comparable; its gap is NOT (different baseline).
    _add("WS3 per-task v2", ws3_hyb, False,
         own_R=(_mean_std(ws3_v31)[0] if ws3_v31 else None),
         own_O=ws3_o)
    rows.append({"system": "oracle (qh1 top-50)", "n_seeds": len(orac),
                 "abs_ndcg@10_mean": O, "abs_ndcg@10_std": _mean_std(orac)[1],
                 "abs_gain_vs_qh1": O - R, "gap_same_qh1_basis": 1.0,
                 "comparable_basis": True, "own_retriever_ndcg@10": None,
                 "own_oracle_ndcg@10": None})

    best_existing, best_sd = _mean_std(v33d)  # the 0.3241 floor

    new_row = None
    if v34_results and Path(v34_results).exists():
        v34d = _cells_task3(Path(v34_results), "hybrid_ndcg@10")
        if v34d:
            m, sd = _mean_std(v34d)
            combined = max(math.hypot(sd, best_sd), FLOOR_MIN)
            margin = m - best_existing
            new_row = {
                "system": "v3.4 blend+struct (NEW)",
                "n_seeds": len(v34d),
                "abs_ndcg@10_mean": m, "abs_ndcg@10_std": sd,
                "abs_gain_vs_qh1": m - R,
                "gap_same_qh1_basis": _gap_same(m, R, O),
                "vs_best_existing": margin,
                "required_margin": combined,
                "clears_floor": bool(margin > combined),
            }
            _add("v3.4 blend+struct (NEW)", v34d, True)

    report = {
        "task": TASK,
        "comparable_basis": {"qh1_retriever_R": R, "qh1_retriever_R_std":
                             R_sd, "oracle_O": O},
        "rows": rows,
        "best_existing_abs_ndcg@10": best_existing,
        "best_existing_std": best_sd,
        "success_bar": (
            "a NEW task-3 system clears iff mean(deployed) - "
            f"{best_existing:.4f} > max(hypot(std_new, {best_sd:.4f}), "
            f"{FLOOR_MIN}) AND regression_vs_retriever == False"),
        "new_system": new_row,
        "note": ("WS3 per-task v2 'gap_closed_frac' (0.096) is computed "
                 "off its OWN weak 0.194 retriever and is NOT comparable "
                 "to the qh1-basis numbers; WS3's ABSOLUTE ndcg@10 is "
                 "comparable and is below even the qh1 retriever."),
    }
    return report


def _print(rep: dict) -> None:
    b = rep["comparable_basis"]
    print()
    print("=" * 96)
    print(f"Phase 0 - task-{rep['task']} honest re-baseline "
          f"(comparable basis: qh1 R={b['qh1_retriever_R']:.4f}  "
          f"oracle O={b['oracle_O']:.4f})")
    print("=" * 96)
    print(f"  {'system':<28}{'n':>3}{'abs@10':>9}{'+-std':>8}"
          f"{'absGain':>9}{'gapSame':>9}{'cmp?':>6}")
    for r in rep["rows"]:
        m = r["abs_ndcg@10_mean"]
        gs = r["gap_same_qh1_basis"]
        print(f"  {r['system']:<28}{r['n_seeds']:>3}{m:>9.4f}"
              f"{r['abs_ndcg@10_std']:>8.4f}{r['abs_gain_vs_qh1']:>+9.4f}"
              f"{(gs if gs == gs else float('nan')):>9.3f}"
              f"{('Y' if r['comparable_basis'] else 'N*'):>6}")
    print(f"\n  best EXISTING absolute task-3 ndcg@10 = "
          f"{rep['best_existing_abs_ndcg@10']:.4f} (v3.3-blend) -- the "
          f"floor any new system must clear.")
    print(f"  success bar: {rep['success_bar']}")
    if rep.get("new_system"):
        n = rep["new_system"]
        print(f"\n  NEW v3.4 blend+struct: {n['abs_ndcg@10_mean']:.4f} "
              f"(+-{n['abs_ndcg@10_std']:.4f})  vs_best_existing="
              f"{n['vs_best_existing']:+.4f}  required>"
              f"{n['required_margin']:.4f}  -> CLEARS FLOOR: "
              f"{n['clears_floor']}")
    print("  N* = absolute comparable, but its gap/baseline is NOT "
          "(WS3 ran off a different, weaker retriever).")
    print(f"\n  {rep['note']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=str, default="runs")
    ap.add_argument("--v34-results", type=str, default=None,
                    help="optional: fold in the v3.4 sweep results json")
    ap.add_argument("--out", type=str,
                    default="runs/reranker_router/task3_rebaseline.json")
    a = ap.parse_args()
    rep = run(Path(a.runs),
              Path(a.v34_results) if a.v34_results else None)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    _print(rep)
    print(f"\n  results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

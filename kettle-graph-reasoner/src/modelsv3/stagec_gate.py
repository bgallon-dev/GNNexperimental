r"""v3.1 Phase 4 — Stage-C gate check (REJECT-by-default).

Stage C is rejected unless ALL of these hold vs the locked baseline
(and, for ndcg, vs the Phase-3 winner if its summary is supplied):

  1. val ndcg@10 up by > one baseline std (else no benefit).
  2. intrinsic nn_edge_precision@5 (corpus) >= 0.125 AND
     >= baseline_mean - 1 std (structure preserved).
  3. collapse rate (frac pairwise dist < 1e-4 x median) not up beyond
     noise vs the baseline collapse.json.
  4. Eval-B bridge_hit@5 not degraded vs the baseline.
  5. Eval-D geom_near_graph_far (q5) not increased vs the baseline.
  6. (reported) query-agnosticism sanity: --train-frac 0.25 within
     0.03 ndcg@10 of 1.0, WITH Stage C on. Pass the two run dirs to
     check it here; otherwise it is reported as "not evaluated".

Any FAIL -> Stage C rejected; v3.1 ships Phase 2 (+P3). Default
``--freeze-mode full`` is already safe. Runs the guardrail evals on the
Stage-C encoder via the existing eval modules, diffs against the
locked baseline's ``*_baseline.json``, prints the verdict, writes
``stagec_gate.json`` into the Stage-C run dir.

Usage
-----
    py -m src.modelsv3.stagec_gate \
        --stagec-run runs/v3.1_stagec_seed1 \
        --baseline-dir runs/v3.1-baseline-hyp-h128-l4-seed1 \
        [--p3-summary runs/v3.1_qh2_infonce_seed1/summary.json] \
        [--trainfrac-full RUN --trainfrac-quarter RUN]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.modelsv3.lock_baseline import load_manifest  # noqa: E402


def _run(mod: str, ckpt: Path, out: Path, task: int) -> int:
    cmd = [sys.executable, "-m", mod, "--checkpoint", str(ckpt),
           "--task", str(task), "--out", str(out)]
    with open(out.with_suffix(".log"), "w") as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)


def _jload(p: Path):
    return json.loads(Path(p).read_text())


def _collapse_rate(collapse_json: dict) -> float:
    """frac_below_threshold['1e-04'].mean from a collapse.json."""
    agg = collapse_json.get("q1_q2_aggregate", {})
    fbt = agg.get("frac_below_threshold", {})
    b = fbt.get("1e-04")
    if isinstance(b, dict):
        return float(b.get("mean", float("nan")))
    if isinstance(b, (int, float)):
        return float(b)
    return float("nan")


def gate(stagec_run: Path, baseline_dir: Path, task: int,
         p3_summary: Path | None,
         tf_full: Path | None, tf_quarter: Path | None) -> dict:
    man = load_manifest(baseline_dir)
    noise = man["noise_floor"]
    enc = stagec_run / "encoder.pt"
    if not enc.exists():
        raise FileNotFoundError(enc)

    # Reference ndcg@10 to beat: Phase-3 winner if given, else the
    # baseline noise-floor mean.
    if p3_summary and Path(p3_summary).exists():
        ref_ndcg = _jload(p3_summary)["final_val"]["overall"]["ndcg@10"]
        ref_src = str(p3_summary)
    else:
        ref_ndcg = noise["ndcg@10"]["mean"]
        ref_src = "baseline noise_floor mean"
    nd_std = noise["ndcg@10"]["std"]

    # Run guardrail evals on the Stage-C encoder.
    gdir = stagec_run / "stagec_gate"
    gdir.mkdir(parents=True, exist_ok=True)
    _run("src.modelsv3.eval_intrinsic_corpus", enc, gdir / "intrinsic.json", task)
    _run("src.modelsv3.investigate_collapse", enc, gdir / "collapse.json", task)
    _run("src.modelsv3.eval_retrieval_midpoint", enc, gdir / "midpoint.json", task)
    _run("src.modelsv3.eval_geom_graph_disagreement", enc,
         gdir / "disagreement.json", task)
    _run("src.modelsv3.eval_candidate_recall", enc,
         gdir / "candidate_recall.json", task)

    sc_summary = _jload(stagec_run / "summary.json")
    sc_ndcg = sc_summary["final_val"]["overall"]["ndcg@10"]
    sc_intr = _jload(gdir / "intrinsic.json")["summary"]["edge_precision_at_k"]["mean"]
    sc_collapse = _collapse_rate(_jload(gdir / "collapse.json"))
    sc_bridge5 = _jload(gdir / "midpoint.json")["summary"]["path_hit_rate@5"]["mean"]
    sc_geom = _jload(gdir / "disagreement.json")["summary"]["q5"][
        "geom_near_graph_far_frac"]["mean"]

    # Baseline references.
    base_collapse = _collapse_rate(_jload(baseline_dir / "collapse.json"))
    base_bridge5 = _jload(baseline_dir / "retrieval_midpoint_baseline.json")[
        "summary"]["path_hit_rate@5"]["mean"]
    base_geom = _jload(baseline_dir / "geom_graph_disagreement_baseline.json")[
        "summary"]["q5"]["geom_near_graph_far_frac"]["mean"]
    base_ep5_mean = noise["intrinsic_edge_prec@5"]["mean"]
    base_ep5_std = noise["intrinsic_edge_prec@5"]["std"]

    checks: dict[str, dict] = {}
    checks["ndcg10_up"] = {
        "stagec": sc_ndcg, "ref": ref_ndcg, "ref_src": ref_src,
        "threshold": ref_ndcg + nd_std,
        "pass": sc_ndcg > ref_ndcg + nd_std,
    }
    checks["intrinsic_edge_prec@5"] = {
        "stagec": sc_intr, "floor_abs": 0.125,
        "floor_noise": base_ep5_mean - base_ep5_std,
        "pass": sc_intr >= 0.125 and sc_intr >= (base_ep5_mean - base_ep5_std),
    }
    checks["collapse_not_up"] = {
        "stagec": sc_collapse, "baseline": base_collapse,
        "pass": sc_collapse <= base_collapse * 1.25 + 1e-9,
    }
    checks["bridge_hit@5_not_degraded"] = {
        "stagec": sc_bridge5, "baseline": base_bridge5,
        "pass": sc_bridge5 >= base_bridge5 - 1e-9,
    }
    checks["geom_near_graph_far_not_increased"] = {
        "stagec": sc_geom, "baseline": base_geom,
        "pass": sc_geom <= base_geom + 1e-9,
    }

    # Query-agnosticism sanity (reported; needs the two extra runs).
    if tf_full and tf_quarter and Path(tf_full).exists() and Path(tf_quarter).exists():
        f10 = _jload(Path(tf_full) / "summary.json")["final_val"]["overall"]["ndcg@10"]
        q10 = _jload(Path(tf_quarter) / "summary.json")["final_val"]["overall"]["ndcg@10"]
        checks["train_frac_0.25_sanity"] = {
            "full": f10, "quarter": q10, "delta": abs(f10 - q10),
            "pass": abs(f10 - q10) <= 0.03,
        }
    else:
        checks["train_frac_0.25_sanity"] = {
            "pass": None,
            "note": "not evaluated - pass --trainfrac-full and "
                    "--trainfrac-quarter run dirs (Stage C ON) to check",
        }

    hard = [k for k, v in checks.items() if v["pass"] is False]
    accepted = len(hard) == 0
    verdict = {
        "stagec_run": str(stagec_run),
        "reference_ndcg10_source": ref_src,
        "checks": checks,
        "failing_checks": hard,
        "verdict": "ACCEPT" if accepted else "REJECT",
        "action": (
            "Stage C clears all guardrails - may ship opt-in."
            if accepted else
            "Stage C REJECTED. Ship Phase 2 (+P3); default --freeze-mode "
            "full is already safe. Record rejection + numbers."
        ),
    }
    (stagec_run / "stagec_gate.json").write_text(json.dumps(verdict, indent=2))
    _print(verdict)
    return verdict


def _print(v: dict) -> None:
    print()
    print("=" * 84)
    print(f"v3.1 Phase 4 - Stage-C gate: {v['verdict']}")
    print(f"ndcg@10 reference: {v['reference_ndcg10_source']}")
    print("=" * 84)
    for name, c in v["checks"].items():
        p = c["pass"]
        tag = "PASS" if p is True else ("FAIL" if p is False else "N/A ")
        extra = {k: round(x, 4) if isinstance(x, float) else x
                 for k, x in c.items() if k != "pass"}
        print(f"  [{tag}] {name}: {extra}")
    print(f"\n  => {v['verdict']}: {v['action']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stagec-run", type=str, required=True)
    ap.add_argument("--baseline-dir", type=str,
                    default="runs/v3.1-baseline-hyp-h128-l4-seed1")
    ap.add_argument("--task", type=int, default=2)
    ap.add_argument("--p3-summary", type=str, default=None,
                    help="Phase-3 winner summary.json (ndcg@10 reference).")
    ap.add_argument("--trainfrac-full", type=str, default=None)
    ap.add_argument("--trainfrac-quarter", type=str, default=None)
    args = ap.parse_args()
    v = gate(
        Path(args.stagec_run), Path(args.baseline_dir), args.task,
        Path(args.p3_summary) if args.p3_summary else None,
        Path(args.trainfrac_full) if args.trainfrac_full else None,
        Path(args.trainfrac_quarter) if args.trainfrac_quarter else None,
    )
    return 0 if v["verdict"] == "ACCEPT" else 2


if __name__ == "__main__":
    sys.exit(main())

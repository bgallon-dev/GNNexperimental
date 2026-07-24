"""End-to-end factorial A/B: bilinear vs QueryToBall Stage-B head, on the
temporal-aux vs control frozen encoders, through the REAL train_v3 pipeline.

Converts the probe-validated bilinear+temporal-aux result into a trustworthy
multi-seed real-pipeline result with a clean 2x2 {encoder}x{head} attribution.
Only the Stage-B head/loss varies per cell; Stage-A is SKIPPED with a
SHA-pinned frozen encoder (proves "only the head changed"). Each cell runs
the real `python -m src.training.train_v3` entrypoint (real _stage_b + real
_eval) then the non-probe hardened_250 transfer eval.

Matrix (seeds {0,1,2}; h128/l4/tier1/task2/lr_query 3e-4/query-epochs 10):
  E0=tempaux-control (aux OFF), E1=tempaux-w0.5 (aux ON)
  (E0,qtb,pairwise)      pure baseline (neither lever)
  (E0,bilinear,pairwise) head-class alone
  (E1,qtb,pairwise)      aux alone
  (E1,bilinear,pairwise) HEADLINE - both levers stacked
  (E1,bilinear,listwise) objective lever, head fixed

Pre-registered (see ~/.claude/plans/reflective-giggling-sketch.md):
PRIMARY = (E1,bilinear,pairwise) - (E1,qtb,pairwise) on hardened_250 > pooled
std AND mean clears anchor-BFS ~0.714. Honest negatives are valid outcomes.

Reuse: real train_v3 entrypoint; scripts/eval_bilinear_hardened.py;
three_seed_comparison_v3._summarize/_report_pair; lock_baseline.sha256_file.
Resumable (skips cells whose summary.json + hardened.json exist).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.modelsv3.three_seed_comparison_v3 import _summarize, _report_pair  # noqa: E402
from src.modelsv3.lock_baseline import sha256_file  # noqa: E402

E0 = _ROOT / "runs" / "tempaux-control-h128-l4-seed1"   # aux OFF
E1 = _ROOT / "runs" / "tempaux-w0.5-h128-l4-seed1"      # aux ON

# (label, encoder_dir, head, loss)
CELLS = [
    ("E0-qtb-pair", E0, "qtb", "pairwise"),
    ("E0-bilin-pair", E0, "bilinear", "pairwise"),
    ("E1-qtb-pair", E1, "qtb", "pairwise"),
    ("E1-bilin-pair", E1, "bilinear", "pairwise"),   # headline
    ("E1-bilin-list", E1, "bilinear", "listwise"),
]
ANCHOR = 0.714          # hardened_250 task-2 anchor-BFS reference
TIER1_FLOOR = (0.437, 0.021)   # non-aux 3-seed tier1-val ndcg@10 mean,std


def _run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd[-8:]))
    r = subprocess.run(cmd, cwd=str(_ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {' '.join(cmd)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--task", type=int, default=2)
    ap.add_argument("--query-epochs", type=int, default=10)
    ap.add_argument("--corpus", default="src/data/corpus/tier1")
    ap.add_argument("--out", default="runs/ab-bilinear-stageb")
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed, 1 query-epoch, only the 2 E1 pair cells")
    a = ap.parse_args()

    out_root = (_ROOT / a.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = a.seeds
    cells = CELLS
    qe = a.query_epochs
    if a.smoke:
        seeds = [a.seeds[0]]
        qe = 1
        cells = [c for c in CELLS if c[0] in ("E1-qtb-pair",
                                              "E1-bilin-pair")]

    for d in (E0, E1):
        if not (d / "encoder.pt").exists():
            raise SystemExit(f"missing frozen encoder: {d}/encoder.pt")
    sha = {str(E0): sha256_file(E0 / "encoder.pt"),
           str(E1): sha256_file(E1 / "encoder.pt")}
    print(f"[ab] E0 sha={sha[str(E0)][:12]}  E1 sha={sha[str(E1)][:12]}")

    results: dict = {}   # label -> {"tier1":[...], "hardened":[...]}
    for label, enc, head, loss in cells:
        results[label] = {"tier1": [], "hardened": [],
                          "encoder": enc.name, "head": head, "loss": loss}
        for s in seeds:
            cdir = out_root / f"{label}-s{s}"
            hjson = cdir / "hardened.json"
            sjson = cdir / "summary.json"
            if sjson.exists() and hjson.exists():
                print(f"[ab] skip (done): {cdir.name}")
            else:
                print(f"[ab] === {cdir.name} (enc={enc.name} head={head} "
                      f"loss={loss} seed={s}) ===")
                _run([sys.executable, "-m", "src.training.train_v3",
                      "--corpus", a.corpus, "--task", str(a.task),
                      "--model", "hyperbolic", "--hidden-dim", "128",
                      "--num-layers", "4", "--type-dim", "8",
                      "--curvature", "1.0", "--tangent-scale", "0.1",
                      "--skip-stage-a",
                      "--load-encoder", str(enc / "encoder.pt"),
                      "--assert-encoder-sha", sha[str(enc)],
                      "--stage-b-head", head, "--stage-b-loss", loss,
                      "--query-epochs", str(qe), "--lr-query", "3e-4",
                      "--seed", str(s), "--out", str(cdir)])
                _run([sys.executable, "scripts/eval_bilinear_hardened.py",
                      "--run", str(cdir), "--task", str(a.task),
                      "--out", str(hjson)])
            sm = json.loads(sjson.read_text())
            t1 = (sm.get("final_val", {}).get("by_task_type", {})
                  .get(str(a.task), {}).get("ndcg@10"))
            hd = json.loads(hjson.read_text())
            results[label]["tier1"].append(float(t1))
            results[label]["hardened"].append(float(hd["ndcg10_mean"]))
            results[label]["anchor"] = float(hd["anchor_bfs_ndcg10_mean"])

    # ---- aggregate + report ----
    agg = {lb: {"tier1": _summarize(r["tier1"]),
                "hardened": _summarize(r["hardened"]),
                "anchor": r.get("anchor")}
           for lb, r in results.items()}
    print("\n[ab] mean ndcg@10 (n seeds) — tier1-val | hardened_250 t2")
    for lb, _, _, _ in cells:
        A = agg[lb]
        print(f"  {lb:<16} tier1={A['tier1']['mean']:.4f}"
              f"±{A['tier1']['std']:.4f}  "
              f"hard={A['hardened']['mean']:.4f}"
              f"±{A['hardened']['std']:.4f}")
    anc = next((agg[c[0]]["anchor"] for c in cells
                if agg[c[0]]["anchor"] is not None), ANCHOR)

    def pooled(x, y):
        return max(agg[x]["hardened"]["std"], agg[y]["hardened"]["std"])

    print(f"\n[ab] hardened_250 anchor-BFS reference = {anc:.4f}")
    verdict = {}
    if {"E1-bilin-pair", "E1-qtb-pair"} <= set(agg):
        d = (agg["E1-bilin-pair"]["hardened"]["mean"]
             - agg["E1-qtb-pair"]["hardened"]["mean"])
        ps = pooled("E1-bilin-pair", "E1-qtb-pair")
        print("\n[ab] PRIMARY  (E1,bilinear,pair) - (E1,qtb,pair) on hardened:")
        _report_pair("  primary", d, ps)
        m = agg["E1-bilin-pair"]["hardened"]["mean"]
        clears = m > anc
        sig = abs(d) > ps and ps > 0
        if sig and d > 0 and clears:
            verdict["primary"] = "CONFIRMED (>pooled std AND clears anchor)"
        elif d > 0 and not (ps > 0 and abs(d) > ps):
            verdict["primary"] = "UNDERPOWERED (dir>0, <pooled std)"
        elif sig and d > 0 and not clears:
            verdict["primary"] = ("PARTIAL (bilinear>qtb but mean "
                                  f"{m:.4f} < anchor {anc:.4f})")
        else:
            verdict["primary"] = "HONEST NEGATIVE (no bilinear>qtb gain)"
        # 2x2 attribution on hardened
        if {"E0-qtb-pair", "E0-bilin-pair", "E1-qtb-pair",
                "E1-bilin-pair"} <= set(agg):
            h = lambda k: agg[k]["hardened"]["mean"]
            head_E0 = h("E0-bilin-pair") - h("E0-qtb-pair")
            head_E1 = h("E1-bilin-pair") - h("E1-qtb-pair")
            aux_qtb = h("E1-qtb-pair") - h("E0-qtb-pair")
            aux_bil = h("E1-bilin-pair") - h("E0-bilin-pair")
            inter = h("E1-bilin-pair") + h("E0-qtb-pair") \
                - h("E1-qtb-pair") - h("E0-bilin-pair")
            print("\n[ab] 2x2 attribution (hardened mean):")
            print(f"  head effect : E0 {head_E0:+.4f} | E1 {head_E1:+.4f}")
            print(f"  aux  effect : qtb {aux_qtb:+.4f} | bilin {aux_bil:+.4f}")
            print(f"  interaction : {inter:+.4f} "
                  f"(>0 = stacking exceeds additive)")
            verdict["attribution"] = {
                "head_E0": head_E0, "head_E1": head_E1,
                "aux_qtb": aux_qtb, "aux_bil": aux_bil,
                "interaction": inter}
    print(f"\n[ab] VERDICT: {verdict.get('primary','(insufficient cells)')}")
    (out_root / "ab_bilinear_results.json").write_text(json.dumps(
        {"agg": agg, "verdict": verdict, "anchor": anc,
         "tier1_floor": TIER1_FLOOR, "seeds": seeds}, indent=2, default=float))
    print(f"[ab] wrote {out_root/'ab_bilinear_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

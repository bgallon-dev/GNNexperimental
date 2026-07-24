r"""v3.1 — real-eval volume scale-up driver.

Scales the *number* of sampled real query-neighborhood graphs
(24 -> 100 -> 250 -> 500, node count fixed at 400) for BOTH samplers,
to reduce variance and confirm the qh1 query-head gain holds:

  legacy   = anchor_ball  (single-seed BFS; locality-confounded)
  hardened = delocalized   (temporally-stratified multi-seed; the
             meaningful task — removes the locality shortcut)

For each (sampler, N): export the corpus from Neo4j if absent, then run
``eval_real_domain`` (v3.1 qh1 head vs the locked baseline qh0 head,
shared frozen encoder, --split all). Resumable: an existing non-empty
corpus dir is not re-exported; an existing report is not re-evaluated.
Writes ``runs/scale_real_eval_summary.json`` + a ladder table.

Existing assets reused as the N=24 rung:
  legacy   N=24 -> src/data/corpus/real_domain_eval
  hardened N=24 -> src/data/corpus/real_domain_eval_hardened

Usage
-----
    py -m src.modelsv3.scale_real_eval               # full ladder
    py -m src.modelsv3.scale_real_eval --sizes 100 250  # subset
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

SCRIPTS = _ROOT / "scripts"
CORPUS = _ROOT / "src" / "data" / "corpus"
RUNS = _ROOT / "runs"
V31_RUN = "runs/sweep_queryhead/qh1_layernorm_seed1"   # frozen enc + qh1
BASELINE = "runs/v3.1-baseline-hyp-h128-l4-seed1"       # frozen enc + qh0

SAMPLERS = {"legacy": "anchor_ball", "hardened": "delocalized"}


def _corpus_dir(sampler: str, n: int) -> Path:
    if n == 24:
        return CORPUS / ("real_domain_eval" if sampler == "legacy"
                         else "real_domain_eval_hardened")
    return CORPUS / f"real_domain_eval_{sampler}_{n}"


def _report_path(sampler: str, n: int) -> Path:
    return RUNS / f"scale_real_{sampler}_{n}_report.json"


def _has_graphs(d: Path) -> bool:
    return d.is_dir() and any(d.glob("graph_*.npz"))


def _export(sampler: str, n: int, cdir: Path) -> bool:
    print(f"[scale] export {sampler} N={n} -> {cdir.name}", flush=True)
    cmd = [
        sys.executable, "neo4j_eval_export.py", "export",
        "--config", "kettle_config.yaml",
        "--out", str(Path("..") / "src" / "data" / "corpus" / cdir.name),
        "--num-graphs", str(n), "--max-nodes", "400",
        "--tasks-per-graph", "3", "--seed", "0",
        "--sampler", SAMPLERS[sampler],
    ]
    log = RUNS / f"scale_export_{sampler}_{n}.log"
    with open(log, "w") as f:
        rc = subprocess.call(cmd, cwd=str(SCRIPTS), stdout=f,
                             stderr=subprocess.STDOUT)
    ok = rc == 0 and _has_graphs(cdir)
    print(f"[scale]   export rc={rc} ok={ok} (log {log.name})", flush=True)
    return ok


def _eval(cdir: Path, report: Path) -> bool:
    print(f"[scale] eval {cdir.name} -> {report.name}", flush=True)
    cmd = [
        sys.executable, "-m", "src.modelsv3.eval_real_domain",
        "--run", V31_RUN, "--compare-run", BASELINE,
        "--corpus", f"src/data/corpus/{cdir.name}",
        "--task", "2", "--split", "all", "--out", str(report),
    ]
    log = RUNS / f"scale_eval_{report.stem}.log"
    with open(log, "w") as f:
        rc = subprocess.call(cmd, cwd=str(_ROOT), stdout=f,
                             stderr=subprocess.STDOUT)
    ok = report.is_file()
    print(f"[scale]   eval rc={rc} report={'ok' if ok else 'MISSING'}",
          flush=True)
    return ok


_METRICS = ["ndcg@10", "ndcg@20", "recall@50", "recall@100",
            "oracle_gap@10|C50", "intrinsic_edge_prec@5", "bridge_hit@5"]


def _collect(report: Path) -> dict:
    r = json.loads(report.read_text())
    m, c = r["run"]["metrics"], r["compare_run"]["metrics"]
    n_g = r["run"].get("rc", {})  # placeholder
    row = {"n_graphs": None}
    for k in _METRICS:
        row[f"qh1_{k}"] = m.get(k)
        row[f"qh0_{k}"] = c.get(k)
        a, b = m.get(k), c.get(k)
        row[f"d_{k}"] = (a - b) if isinstance(a, (int, float)) \
            and isinstance(b, (int, float)) else None
    return row


def _n_graphs(cdir: Path) -> int:
    return len(list(cdir.glob("graph_*.npz")))


def run(sizes: list[int]) -> int:
    results: dict = {}
    for sampler in ("hardened", "legacy"):
        for n in sizes:
            key = f"{sampler}_{n}"
            cdir = _corpus_dir(sampler, n)
            report = _report_path(sampler, n)
            # N=24 legacy already has a report from the first real-eval run.
            if sampler == "legacy" and n == 24 and \
                    (RUNS / "real_domain_eval_report.json").is_file():
                report = RUNS / "real_domain_eval_report.json"
            # hardened-100 already evaluated in the prior step.
            if sampler == "hardened" and n == 100 and \
                    (RUNS / "real_domain_eval_hardened_100_report.json").is_file():
                report = RUNS / "real_domain_eval_hardened_100_report.json"

            if not _has_graphs(cdir):
                if n == 24:
                    print(f"[scale] SKIP {key}: base corpus {cdir} absent")
                    continue
                if not _export(sampler, n, cdir):
                    print(f"[scale] export FAILED {key}; skipping")
                    continue
            if not report.is_file():
                if not _eval(cdir, report):
                    print(f"[scale] eval FAILED {key}; skipping")
                    continue
            try:
                row = _collect(report)
                row["n_graphs"] = _n_graphs(cdir)
                row["report"] = str(report)
                results[key] = row
            except Exception as e:  # noqa: BLE001
                print(f"[scale] collect FAILED {key}: {e}")

    out = RUNS / "scale_real_eval_summary.json"
    out.write_text(json.dumps(results, indent=2))
    _print(results)
    print(f"\n[scale] summary -> {out}")
    return 0


def _print(results: dict) -> None:
    print()
    print("=" * 100)
    print("v3.1 real-eval volume scale-up  (qh1 vs baseline qh0; "
          "shared FROZEN encoder; --split all)")
    print("=" * 100)
    print(f"  {'sampler/N':<16}{'graphs':>7}{'qh1 ndcg@10':>13}"
          f"{'qh0 ndcg@10':>13}{'delta':>9}{'qh1 r@50':>10}"
          f"{'qh1 r@100':>11}{'edge_p@5':>10}")
    for sampler in ("hardened", "legacy"):
        for key in sorted((k for k in results if k.startswith(sampler)),
                          key=lambda k: int(k.split("_")[1])):
            r = results[key]
            print(f"  {key:<16}{r['n_graphs']:>7}"
                  f"{_f(r.get('qh1_ndcg@10')):>13}"
                  f"{_f(r.get('qh0_ndcg@10')):>13}"
                  f"{_f(r.get('d_ndcg@10'), sign=True):>9}"
                  f"{_f(r.get('qh1_recall@50')):>10}"
                  f"{_f(r.get('qh1_recall@100')):>11}"
                  f"{_f(r.get('qh1_intrinsic_edge_prec@5')):>10}")
    print("\n  Read: delta = qh1 - qh0 ndcg@10 (query-head gain). Structure "
          "metrics (edge_p@5) are identical qh1/qh0 by construction\n"
          "  (shared frozen encoder) — they should be flat across N for a "
          "sampler; their stabilization across N shows variance shrinking.")


def _f(v, sign: bool = False) -> str:
    if not isinstance(v, (int, float)):
        return "  n/a"
    return f"{v:+.4f}" if sign else f"{v:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[24, 100, 250, 500],
                    help="graph-count rungs (node count fixed at 400).")
    args = ap.parse_args()
    return run(args.sizes)


if __name__ == "__main__":
    sys.exit(main())

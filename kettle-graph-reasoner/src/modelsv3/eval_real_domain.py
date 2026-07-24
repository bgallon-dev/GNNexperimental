r"""v3.1 — real-domain evaluation orchestrator.

Runs the full v3.1 eval suite on a real-domain corpus (the
synthetic-schema NPZ export of the real graph) for one v3.1 run, and
optionally side-by-side against a comparison run (e.g. the locked
baseline qh0 vs the v3.1 qh1 head). Real eval corpora are small and
dedicated, so this evaluates with ``--split all`` (every graph), not a
10% val slice.

Sub-evals (each pointed at ``<run>/encoder.pt`` + sibling
``query_encoder.pt`` + ``summary.json``):

  intrinsic            silhouette / edge_prec@5 / label_purity@5
  candidate_recall     ndcg@{10,20} / recall@{50,100} / oracle_gap@10
  retrieval_midpoint   bridge_hit@5 + bridge graph-dist error
  retrieval_nn_filtered  degree-stratified edge_prec@5
  geom_disagreement    spearman(emb,hop) + geom/graph disagreement
  collapse             near-duplicate rate
  provenance_path      (auto-skips if the corpus has no task 0/3)

Writes ``real_domain_eval_report.json`` + a side-by-side table.

Usage
-----
    py -m src.modelsv3.eval_real_domain \
        --run runs/sweep_queryhead/qh1_layernorm_seed1 \
        --compare-run runs/v3.1-baseline-hyp-h128-l4-seed1 \
        --corpus src/data/corpus/real_domain_eval \
        --task 2 --split all \
        --out runs/real_domain_eval_report.json
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


def _run(mod: str, args: list[str], log: Path) -> int:
    with open(log, "w") as f:
        return subprocess.call([sys.executable, "-m", mod] + args,
                               stdout=f, stderr=subprocess.STDOUT)


def _jload(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _g(d, *path, default=None):
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def eval_one(run: Path, corpus: str, split: str, task: int,
             gdir: Path) -> dict:
    gdir.mkdir(parents=True, exist_ok=True)
    enc = str(run / "encoder.pt")
    smry = str(run / "summary.json")
    common = ["--checkpoint", enc, "--summary", smry, "--corpus", corpus,
              "--split", split, "--task", str(task)]

    jobs = {
        "intrinsic": ("src.modelsv3.eval_intrinsic_corpus", common
                      + ["--out", str(gdir / "intrinsic.json")]),
        "candidate_recall": ("src.modelsv3.eval_candidate_recall", common
                             + ["--out", str(gdir / "candidate_recall.json")]),
        "retrieval_midpoint": ("src.modelsv3.eval_retrieval_midpoint", common
                               + ["--out", str(gdir / "midpoint.json")]),
        "retrieval_nn_filtered": ("src.modelsv3.eval_retrieval_nn_filtered",
                                  common + ["--out", str(gdir / "nnf.json")]),
        "geom_disagreement": ("src.modelsv3.eval_geom_graph_disagreement",
                              common + ["--out", str(gdir / "geom.json")]),
        "provenance_path": ("src.modelsv3.eval_provenance_path", common
                            + ["--out", str(gdir / "prov.json")]),
        # investigate_collapse has no --summary arg
        "collapse": ("src.modelsv3.investigate_collapse",
                     ["--checkpoint", enc, "--corpus", corpus,
                      "--split", split, "--task", str(task),
                      "--out", str(gdir / "collapse.json")]),
    }
    # Status is judged by "valid artifact produced", not exit code:
    # some sub-evals (e.g. investigate_collapse) write their JSON and
    # then hit a cosmetic non-ASCII print crash on the Windows console
    # (rc=1) — the metric is still valid. rc is kept for transparency.
    rc: dict[str, int] = {}
    for name, (mod, a) in jobs.items():
        rc[name] = _run(mod, a, gdir / f"{name}.log")
        out_json = Path(a[a.index("--out") + 1])
        ok = _jload(out_json) is not None
        tag = "ok  " if ok else "FAIL"
        note = "" if rc[name] == 0 else f" (rc={rc[name]}, artifact-valid)" \
            if ok else f" (rc={rc[name]})"
        print(f"  [{tag}] {name}{note}")

    intr = _jload(gdir / "intrinsic.json")
    cr = _jload(gdir / "candidate_recall.json")
    mp = _jload(gdir / "midpoint.json")
    nnf = _jload(gdir / "nnf.json")
    gm = _jload(gdir / "geom.json")
    col = _jload(gdir / "collapse.json")
    prov = _jload(gdir / "prov.json")

    cro = _g(cr, "summary", "overall", default={})
    dt = _g(nnf, "edge_prec@5_by_degree_tercile", default={})
    fbt = _g(col, "q1_q2_aggregate", "frac_below_threshold", "1e-04",
             default={})
    collapse_rate = fbt.get("mean") if isinstance(fbt, dict) else fbt

    return {
        "run": str(run),
        "rc": rc,
        "metrics": {
            "intrinsic_edge_prec@5": _g(intr, "summary",
                                        "edge_precision_at_k", "mean"),
            "intrinsic_edge_prec@5_random": _g(
                intr, "summary", "random_baseline_edge_prec_mean"),
            "intrinsic_silhouette": _g(intr, "summary", "silhouette", "mean"),
            "intrinsic_label_purity@5": _g(intr, "summary",
                                           "label_purity_at_k", "mean"),
            "ndcg@10": cro.get("ndcg@10"),
            "ndcg@20": cro.get("ndcg@20"),
            "recall@50": cro.get("recall@50"),
            "recall@100": cro.get("recall@100"),
            "oracle_gap@10|C50": cro.get("oracle_gap@10|C50"),
            "bridge_hit@5": _g(mp, "summary", "path_hit_rate@5", "mean"),
            "bridge_graph_dist_error@5": _g(
                mp, "summary", "bridge_graph_dist_error@5", "mean"),
            "bridge_random_baseline": _g(mp, "summary",
                                         "random_baseline", "mean"),
            "edge_prec@5_low_deg": _g(dt, "low", "mean_precision", "mean"),
            "edge_prec@5_high_deg": _g(dt, "high", "mean_precision", "mean"),
            "edge_prec@5_all_deg": _g(dt, "all", "mean_precision", "mean"),
            "spearman_emb_hop": _g(gm, "summary", "spearman_emb_hop", "mean"),
            "geom_near_graph_far_q5": _g(
                gm, "summary", "q5", "geom_near_graph_far_frac", "mean"),
            "collapse_rate_1e-4": collapse_rate,
            "provenance_path_recall@10": _g(
                prov, "summary", "overall", "prov_path_recall@10"),
            "provenance_n_eval": _g(prov, "n_eval_samples"),
        },
    }


_TABLE_ROWS = [
    ("ndcg@10", "ranking"), ("ndcg@20", "ranking"),
    ("recall@50", "retriever"), ("recall@100", "retriever"),
    ("oracle_gap@10|C50", "headroom"),
    ("intrinsic_edge_prec@5", "structure"),
    ("intrinsic_edge_prec@5_random", "structure"),
    ("intrinsic_silhouette", "structure"),
    ("bridge_hit@5", "bridge"), ("bridge_random_baseline", "bridge"),
    ("edge_prec@5_high_deg", "degree"), ("edge_prec@5_low_deg", "degree"),
    ("spearman_emb_hop", "geom"), ("geom_near_graph_far_q5", "geom"),
    ("collapse_rate_1e-4", "collapse"),
    ("provenance_path_recall@10", "prov"),
]


def _fmt(v) -> str:
    if v is None:
        return "   n/a"
    if isinstance(v, (int, float)):
        return f"{v:7.4f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=str, required=True,
                    help="v3.1 run dir (encoder.pt + query_encoder.pt + "
                         "summary.json).")
    ap.add_argument("--compare-run", type=str, default=None,
                    help="Optional second run for a side-by-side (e.g. the "
                         "locked baseline qh0).")
    ap.add_argument("--corpus", type=str,
                    default="src/data/corpus/real_domain_eval")
    ap.add_argument("--task", type=int, default=2)
    ap.add_argument("--split", type=str, default="all",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--out", type=str,
                    default="runs/real_domain_eval_report.json")
    args = ap.parse_args()

    corpus = args.corpus
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    main_run = Path(args.run)
    print(f"[real-eval] corpus={corpus} split={args.split} task={args.task}")
    print(f"[real-eval] === run: {main_run} ===")
    res_main = eval_one(main_run, corpus, args.split, args.task,
                        main_run / "real_domain_eval")
    report = {"corpus": corpus, "split": args.split, "task": args.task,
              "run": res_main}

    res_cmp = None
    if args.compare_run:
        cmp_run = Path(args.compare_run)
        print(f"[real-eval] === compare-run: {cmp_run} ===")
        res_cmp = eval_one(cmp_run, corpus, args.split, args.task,
                           cmp_run / "real_domain_eval")
        report["compare_run"] = res_cmp

    out.write_text(json.dumps(report, indent=2))

    print()
    print("=" * 88)
    print(f"Real-domain eval  ({corpus}, split={args.split}, "
          f"all graphs)")
    print("=" * 88)
    m = res_main["metrics"]
    if res_cmp is None:
        print(f"  {'metric':<28}{'value':>10}  group")
        for key, grp in _TABLE_ROWS:
            print(f"  {key:<28}{_fmt(m.get(key)):>10}  {grp}")
    else:
        c = res_cmp["metrics"]
        cmp_name = Path(args.compare_run).name
        main_name = main_run.name
        print(f"  {'metric':<28}{main_name[:14]:>15}{cmp_name[:14]:>15}"
              f"{'delta':>10}  group")
        for key, grp in _TABLE_ROWS:
            a, b = m.get(key), c.get(key)
            d = (a - b) if isinstance(a, (int, float)) and isinstance(
                b, (int, float)) else None
            print(f"  {key:<28}{_fmt(a):>15}{_fmt(b):>15}"
                  f"{_fmt(d):>10}  {grp}")
        print(f"\n  ({main_name} = v3.1 head;  {cmp_name} = comparison)")
    print(f"\n  report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

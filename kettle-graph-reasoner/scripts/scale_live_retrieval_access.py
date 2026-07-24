r"""Scale the live Neo4j -> KGR retrieval access path.

This is an orchestration script, not a new model path. It runs the same
three artifacts manually exercised during the live smoke:

  1. export live Neo4j neighborhoods to an NPZ corpus with neo4j_node_id
  2. encode that corpus once into a manifold index
  3. run the retrieval smoke test with optional live Neo4j enrichment

The default is intentionally modest and resumable. Increase ``--num-graphs``
and ``--max-nodes`` as the access layer graduates from smoke to larger evals.

Usage
-----
    py scripts/scale_live_retrieval_access.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"


def _has_graphs(path: Path) -> bool:
    return path.is_dir() and any(path.glob("graph_*.npz"))


def _run(cmd: list[str], *, cwd: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[scale-access] $ {' '.join(cmd)}", flush=True)
    print(f"[scale-access]   log -> {log}", flush=True)
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=str(cwd), stdout=f,
                             stderr=subprocess.STDOUT)
    print(f"[scale-access]   rc={rc}", flush=True)
    return rc


def _export(args, corpus_dir: Path, logs: Path) -> None:
    if _has_graphs(corpus_dir) and not args.force_export:
        print(f"[scale-access] export skipped; corpus exists: {corpus_dir}")
        return
    cmd = [
        sys.executable, "neo4j_eval_export.py", "export",
        "--config", str((_ROOT / args.config).resolve()),
        "--out", str(corpus_dir.resolve()),
        "--num-graphs", str(args.num_graphs),
        "--max-nodes", str(args.max_nodes),
        "--tasks-per-graph", str(args.tasks_per_graph),
        "--seed", str(args.seed),
        "--sampler", args.sampler,
        "--n-seeds", str(args.n_seeds),
    ]
    rc = _run(cmd, cwd=_SCRIPTS, log=logs / "export.log")
    if rc != 0 or not _has_graphs(corpus_dir):
        raise SystemExit("[scale-access] export failed; see export.log")


def _index(args, corpus_dir: Path, index_path: Path, logs: Path) -> None:
    if index_path.exists() and not args.force_index:
        print(f"[scale-access] index skipped; index exists: {index_path}")
        return
    cmd = [
        sys.executable, "-m", "src.modelsv3.export_manifold_index",
        "--run", args.run,
        "--corpus", str(corpus_dir),
        "--split", "all",
        "--task", str(args.task),
        "--out", str(index_path),
        "--assert-sha",
        "--baseline-dir", args.baseline_dir,
    ]
    rc = _run(cmd, cwd=_ROOT, log=logs / "index.log")
    if rc != 0 or not index_path.exists():
        raise SystemExit("[scale-access] index export failed; see index.log")


def _smoke(args, corpus_dir: Path, index_path: Path, logs: Path) -> None:
    if args.skip_smoke:
        return
    cmd = [
        sys.executable, "scripts/smoke_retrieval_workflow.py",
        "--index", str(index_path),
        "--corpus", str(corpus_dir),
        "--split", "all",
        "--task", str(args.task),
        "--examples", str(args.examples),
        "--k", str(args.k),
        "--real-head", args.run,
        "--synthetic-head", args.synthetic_head,
        "--json-out", str(logs / "smoke_results.json"),
        "--reranker-router-results", str(
            (_ROOT / args.reranker_router_results).resolve()),
        "--no-sanity-check",
    ]
    if args.live_neo4j:
        cmd.append("--live-neo4j")
        cmd.extend(["--prop-limit", str(args.prop_limit)])
    rc = _run(cmd, cwd=_ROOT, log=logs / "smoke.log")
    if rc != 0:
        raise SystemExit("[scale-access] smoke failed; see smoke.log")


def _summary(out_dir: Path, corpus_dir: Path, index_path: Path,
             args) -> None:
    n_graphs = len(list(corpus_dir.glob("graph_*.npz")))
    meta_path = index_path.with_name(index_path.stem + "_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    summary = {
        "out_dir": str(out_dir),
        "corpus": str(corpus_dir),
        "index": str(index_path),
        "num_graphs_requested": args.num_graphs,
        "num_graphs_written": n_graphs,
        "max_nodes": args.max_nodes,
        "tasks_per_graph": args.tasks_per_graph,
        "sampler": args.sampler,
        "seed": args.seed,
        "index_nodes": meta.get("n_nodes"),
        "index_graphs": meta.get("n_graphs"),
        "encoder_sha256": meta.get("encoder_sha256"),
        "has_live_neo4j_ids": bool(meta.get("n_nodes")) and _index_has_ids(index_path),
    }
    smoke_json = out_dir / "logs" / "smoke_results.json"
    verdict: dict = {}
    if smoke_json.exists():
        try:
            verdict = json.loads(smoke_json.read_text()).get("verdict", {})
        except (json.JSONDecodeError, OSError):
            verdict = {}
    summary["verdict"] = verdict

    out = out_dir / "scale_live_retrieval_access_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print("=" * 80)
    print("KGR live retrieval access scaled")
    print("=" * 80)
    print(f"corpus:      {corpus_dir} ({n_graphs} graphs)")
    print(f"index:       {index_path} ({meta.get('n_nodes', 'n/a')} nodes)")
    print(f"smoke log:   {out_dir / 'logs' / 'smoke.log'}")
    if verdict:
        syn = verdict.get("synthetic_retriever_ndcg@10")
        real = verdict.get("real_retriever_ndcg@10")
        rr = verdict.get("reranked_ndcg@10")
        orc = verdict.get("oracle_ndcg@10|C50")

        def _f(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"

        print("how well (verdict):")
        print(f"  synthetic head      ndcg@10 = {_f(syn)}")
        print(f"  + real-trained head ndcg@10 = {_f(real)}")
        print(f"  + reranker/router   ndcg@10 = {_f(rr)}"
              + ("" if rr is not None else "  (not apples-to-apples on "
                                           "this corpus -- see smoke log)"))
        print(f"  oracle ceiling      ndcg@10 = {_f(orc)}")
    print(f"summary:     {out}")


def _index_has_ids(index_path: Path) -> bool:
    import numpy as np

    z = np.load(index_path)
    return "neo4j_node_id" in z.files and bool((z["neo4j_node_id"] >= 0).any())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str,
                   default="runs/live_retrieval_access")
    p.add_argument("--config", type=str,
                   default="scripts/kettle_config.yaml")
    p.add_argument("--num-graphs", type=int, default=24)
    p.add_argument("--max-nodes", type=int, default=400)
    p.add_argument("--tasks-per-graph", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampler", type=str, default="delocalized",
                   choices=["delocalized", "anchor_ball"])
    p.add_argument("--n-seeds", type=int, default=4)
    p.add_argument("--task", type=int, default=2)
    p.add_argument("--run", type=str,
                   default="runs/v3.1-real-head-hyp-h128-l4-seed0")
    p.add_argument("--baseline-dir", type=str,
                   default="runs/v3.1-baseline-hyp-h128-l4-seed1")
    p.add_argument("--synthetic-head", type=str,
                   default="runs/v3.1-baseline-hyp-h128-l4-seed1")
    p.add_argument("--examples", type=int, default=5)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--prop-limit", type=int, default=5)
    p.add_argument("--reranker-router-results", type=str,
                   default="runs/reranker_router_real/router_results.json",
                   help="Passed to the smoke for the end-to-end verdict. The "
                        "smoke shows the reranker tier ONLY if its corpus "
                        "matches the corpus the reranker was validated on; "
                        "for a freshly-exported live corpus it is honestly "
                        "marked n/a (the reranker was not trained on it).")
    p.add_argument("--live-neo4j", action="store_true", default=True)
    p.add_argument("--no-live-neo4j", dest="live_neo4j", action="store_false")
    p.add_argument("--skip-smoke", action="store_true")
    p.add_argument("--force-export", action="store_true")
    p.add_argument("--force-index", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    corpus_dir = out_dir / "corpus"
    index_path = out_dir / "manifold_index.npz"
    logs = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    _export(args, corpus_dir, logs)
    _index(args, corpus_dir, index_path, logs)
    _smoke(args, corpus_dir, index_path, logs)
    _summary(out_dir, corpus_dir, index_path, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

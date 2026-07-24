r"""Offline smoke test for the KGR retrieval artifact chain.

This script does not connect to Neo4j and does not run the graph encoder.
It loads an exported manifold index, maps validation queries through one or
more selectable QueryToBall heads, scores the precomputed embeddings, and
checks the result against the known real-val retrieval numbers.

Default path under test:

    manifold_index.npz -> load_query_encoder -> query point -> distance scores

Usage
-----
    py scripts/smoke_retrieval_workflow.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv3.distance_scoring import score_from_embeddings  # noqa: E402
from src.modelsv3.lock_baseline import sha256_file  # noqa: E402
from src.modelsv3.retrieval_ops import (  # noqa: E402
    ManifoldIndex,
    load_index,
    load_query_encoder,
)
from src.training.metrics import ndcg_at_k, recall_at_k  # noqa: E402

K_VALUES = (5, 10, 20)
C_VALUES = (20, 50, 100)


def _check_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _load_head(run_dir: Path):
    _check_file(run_dir / "summary.json", "head summary")
    _check_file(run_dir / "query_encoder.pt", "query encoder")
    return load_query_encoder(run_dir)


def _warn_sha(index: ManifoldIndex, run_dir: Path, label: str) -> None:
    enc_path = run_dir / "encoder.pt"
    if not enc_path.exists():
        print(f"[warn] {label}: encoder.pt missing, cannot SHA-check {run_dir}")
        return
    got = sha256_file(enc_path)
    want = str(index.meta.get("encoder_sha256", ""))
    if want and got != want:
        print(
            f"[warn] {label}: encoder SHA differs from index "
            f"(head={got[:12]} index={want[:12]})"
        )


def _graph_rows(index: ManifoldIndex, graph_idx: int,
                expected_n: int) -> tuple[np.ndarray, dict[int, int]]:
    rows = np.where(index.graph_mask(graph_idx))[0]
    if rows.size == 0:
        raise ValueError(f"graph {graph_idx} is present in the corpus but not "
                         "in the manifold index")
    node_to_row = {int(index.node_idx[r]): int(r) for r in rows}
    missing = [i for i in range(expected_n) if i not in node_to_row]
    if missing:
        preview = ", ".join(str(i) for i in missing[:8])
        raise ValueError(
            f"graph {graph_idx} index is missing {len(missing)} local nodes "
            f"(first missing: {preview})"
        )
    return rows, node_to_row


def _score_graph(index: ManifoldIndex, rows: np.ndarray,
                 query_point: torch.Tensor, expected_n: int) -> torch.Tensor:
    emb = torch.from_numpy(np.ascontiguousarray(index.embedding[rows])).float()
    row_scores = score_from_embeddings(
        emb, query_point, c=index.c, euclidean=index.euclidean)
    scores = torch.full((expected_n,), float("-inf"),
                        dtype=row_scores.dtype)
    node_ids = index.node_idx[rows].astype(np.int64)
    scores[torch.from_numpy(node_ids)] = row_scores.detach().cpu()
    return scores


def _oracle_ndcg(scores: torch.Tensor, labels: torch.Tensor,
                 candidate_k: int, metric_k: int) -> float:
    candidate_k = min(candidate_k, scores.numel())
    cand = torch.topk(scores, k=candidate_k, largest=True).indices
    oracle_scores = torch.full_like(labels, float("-inf"))
    oracle_scores[cand] = labels[cand]
    return ndcg_at_k(oracle_scores, labels, metric_k)


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return float(sum(r[key] for r in rows) / len(rows)) if rows else float("nan")


def _summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = list(rows[0].keys()) if rows else []
    return {k: _mean(rows, k) for k in keys}


def _evaluate_head(
    *,
    name: str,
    query_to_point,
    dataset: CorpusDataset,
    index: ManifoldIndex,
    collect_examples: int,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metric_rows: list[dict[str, float]] = []
    examples: list[dict[str, Any]] = []

    for sample_id in range(len(dataset)):
        graph_idx, _task_idx = dataset.index[sample_id]
        sample = dataset[sample_id]
        labels = sample.labels.detach().cpu().float()
        rows, node_to_row = _graph_rows(index, graph_idx, labels.numel())
        query_point = query_to_point(sample.query)
        scores = _score_graph(index, rows, query_point, labels.numel())

        row: dict[str, float] = {}
        for k in K_VALUES:
            row[f"ndcg@{k}"] = ndcg_at_k(scores, labels, k)
        for c in C_VALUES:
            row[f"recall@{c}"] = recall_at_k(scores, labels, c)
        row["oracle_ndcg@10|C50"] = _oracle_ndcg(scores, labels, 50, 10)
        row["oracle_gap@10|C50"] = row["oracle_ndcg@10|C50"] - row["ndcg@10"]
        metric_rows.append(row)

        if name == "real" and len(examples) < collect_examples:
            k = min(top_k, scores.numel())
            top = torch.topk(scores, k=k, largest=True).indices.tolist()
            ranked = []
            for rank, node_idx in enumerate(top, start=1):
                global_row = node_to_row[int(node_idx)]
                neo4j_id = int(index.neo4j_node_id[global_row])
                ranked.append({
                    "rank": rank,
                    "node_idx": int(node_idx),
                    "neo4j_id": neo4j_id if neo4j_id >= 0 else None,
                    "score": float(scores[node_idx].item()),
                    "label": float(labels[node_idx].item()),
                    "node_type": int(index.node_type[global_row]),
                    "layer": int(index.layer[global_row]),
                    "depth": int(index.depth[global_row]),
                    "collapse_flag": bool(index.collapse_flag[global_row]),
                })
            examples.append({
                "sample_id": sample_id,
                "graph_idx": int(graph_idx),
                "temporal_window": [
                    float(sample.query[6].item()),
                    float(sample.query[7].item()),
                ],
                "hits_at_k": int((labels[top] >= 0.5).sum().item()),
                "top_k": ranked,
            })

    return {
        "name": name,
        "n_samples": len(metric_rows),
        "metrics": _summarize(metric_rows),
    }, examples


def _compact_value(value: Any, limit: int = 80) -> str:
    text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _compact_props(props: dict[str, Any], prop_limit: int) -> dict[str, str]:
    if prop_limit <= 0:
        return {}
    preferred = (
        "name", "title", "year", "entity_id", "refuge_id", "species",
        "taxon_group", "event_type", "activity_type", "observation_id",
        "measurement_id", "paragraph_id", "page_id", "doc_id",
    )
    out: dict[str, str] = {}
    for key in preferred:
        if key in props and props[key] is not None:
            out[key] = _compact_value(props[key])
        if len(out) >= prop_limit:
            return out
    for key in sorted(props):
        if key in out or props[key] is None:
            continue
        out[key] = _compact_value(props[key])
        if len(out) >= prop_limit:
            break
    return out


def _fetch_neo4j_nodes(ids: list[int], prop_limit: int) -> dict[int, dict[str, Any]]:
    from neo4j_eval_export import _driver, _session

    uniq = sorted({int(i) for i in ids if i is not None and int(i) >= 0})
    if not uniq:
        raise ValueError(
            "live Neo4j enrichment requested, but the manifold index has no "
            "neo4j_node_id values. Re-export the corpus with the updated "
            "neo4j_eval_export.py and re-run export_manifold_index.py."
        )

    drv = _driver()
    try:
        drv.verify_connectivity()
        with _session(drv) as s:
            rows = s.run(
                "MATCH (n) WHERE id(n) IN $ids "
                "RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props",
                ids=uniq,
            )
            return {
                int(r["id"]): {
                    "labels": list(r["labels"]),
                    "props": _compact_props(dict(r["props"]), prop_limit),
                }
                for r in rows
            }
    finally:
        drv.close()


def _enrich_examples_live(examples: list[dict[str, Any]], prop_limit: int) -> None:
    if not examples:
        return
    ids = [
        int(row["neo4j_id"])
        for ex in examples
        for row in ex["top_k"]
        if row.get("neo4j_id") is not None
    ]
    details = _fetch_neo4j_nodes(ids, prop_limit)
    for ex in examples:
        for row in ex["top_k"]:
            nid = row.get("neo4j_id")
            if nid is not None and int(nid) in details:
                row["neo4j"] = details[int(nid)]


def _print_head_summary(head: dict[str, Any]) -> None:
    m = head["metrics"]
    print(f"\n{head['name']} head ({head['n_samples']} samples)")
    print("  ndcg:   " + "  ".join(
        f"@{k}={m[f'ndcg@{k}']:.4f}" for k in K_VALUES))
    print("  recall: " + "  ".join(
        f"@{c}={m[f'recall@{c}']:.4f}" for c in C_VALUES))
    print(
        "  oracle: "
        f"ndcg@10|C50={m['oracle_ndcg@10|C50']:.4f}  "
        f"gap={m['oracle_gap@10|C50']:+.4f}"
    )


def _print_examples(examples: list[dict[str, Any]]) -> None:
    if not examples:
        print("\nNo examples requested.")
        return
    print("\nExample rankings from the real head")
    for ex in examples:
        w0, w1 = ex["temporal_window"]
        print(
            f"\n  sample={ex['sample_id']} graph={ex['graph_idx']} "
            f"window=[{w0:.3f}, {w1:.3f}] hits@k={ex['hits_at_k']}"
        )
        print("    rank node score      label type layer depth collapse")
        for r in ex["top_k"]:
            live = ""
            if r.get("neo4j_id") is not None:
                live = f" neo4j={r['neo4j_id']}"
            if "neo4j" in r:
                labels = ":".join(r["neo4j"]["labels"])
                props = ", ".join(
                    f"{k}={v}" for k, v in r["neo4j"]["props"].items())
                live += f" labels={labels}"
                if props:
                    live += f" props={{ {props} }}"
            print(
                f"    {r['rank']:>4} {r['node_idx']:>4} "
                f"{r['score']:>+9.4f} {r['label']:>5.2f} "
                f"{r['node_type']:>4} {r['layer']:>5} {r['depth']:>5} "
                f"{str(r['collapse_flag']):>8}{live}"
            )


def _sanity_checks(results: dict[str, Any], args: argparse.Namespace) -> bool:
    heads = {h["name"]: h for h in results["heads"]}
    ok = True

    n = heads["real"]["n_samples"]
    if n != args.expected_samples:
        print(f"[fail] expected {args.expected_samples} samples, got {n}")
        ok = False

    synth = heads["synthetic"]["metrics"]["ndcg@10"]
    real = heads["real"]["metrics"]["ndcg@10"]
    delta = real - synth
    if abs(synth - args.expected_synthetic_ndcg10) > args.tolerance:
        print(
            f"[fail] synthetic ndcg@10 {synth:.4f} not within "
            f"{args.tolerance:.4f} of {args.expected_synthetic_ndcg10:.4f}"
        )
        ok = False
    if abs(real - args.expected_real_ndcg10) > args.tolerance:
        print(
            f"[fail] real ndcg@10 {real:.4f} not within "
            f"{args.tolerance:.4f} of {args.expected_real_ndcg10:.4f}"
        )
        ok = False
    if delta < args.min_delta:
        print(
            f"[fail] real-synthetic ndcg@10 delta {delta:+.4f} "
            f"< required {args.min_delta:+.4f}"
        )
        ok = False

    if ok:
        print(
            f"\n[pass] sanity checks passed: real-synthetic ndcg@10 "
            f"delta {delta:+.4f}"
        )
    return ok


def _reranker_tier(args: argparse.Namespace) -> dict[str, Any]:
    """Fold the separately-validated reranker/router result into the
    verdict -- but ONLY when this smoke's corpus matches the corpus the
    reranker was validated on. A cross-corpus number would not be
    apples-to-apples (same discipline as rebaseline_task3 / the WS3
    artifact lesson), so it is explicitly withheld and flagged instead.
    """
    rr = Path(args.reranker_router_results)
    same_corpus = (
        Path(args.corpus).expanduser().resolve()
        == Path(args.reranker_corpus).expanduser().resolve()
    )
    if not rr.exists():
        return {"status": "n/a", "reason": f"not found: {rr}"}
    if not same_corpus:
        return {"status": "n/a",
                "reason": (f"corpus mismatch (smoke={args.corpus} != "
                           f"reranker={args.reranker_corpus}); a cross-"
                           f"corpus number is not apples-to-apples")}
    payload = json.loads(rr.read_text())
    rep = payload.get("router", payload)
    bt = (rep.get("by_task") or {}).get(str(args.task))
    if not bt:
        return {"status": "n/a",
                "reason": f"task {args.task} absent in {rr}"}
    return {
        "status": "ok",
        "source": str(rr),
        "chosen_recipe": bt.get("chosen_recipe"),
        "routed_ndcg@10": bt.get("routed_deployed_mean"),
        "routed_std": bt.get("routed_deployed_std"),
        "regression": bool(bt.get("regression")),
        "any_regression": bool(rep.get("any_regression")),
    }


def _print_verdict(head_results: list[dict[str, Any]],
                   tier: dict[str, Any], task: int) -> dict[str, Any]:
    """The consolidated 'how well is this working so far' answer:
    synthetic head -> real-trained head -> +reranker/router -> oracle
    ceiling, with the transfer gain, the fraction of the retriever->
    oracle gap the reranker closes, and the do-no-harm status."""
    by = {h["name"]: h["metrics"] for h in head_results}
    syn = float(by.get("synthetic", {}).get("ndcg@10", float("nan")))
    real = float(by.get("real", {}).get("ndcg@10", float("nan")))
    oracle = float(by.get("real", {}).get("oracle_ndcg@10|C50",
                                          float("nan")))
    print("\n" + "=" * 80)
    print(f"VERDICT - how well is this working so far (task {task})")
    print("=" * 80)
    print(f"  1. synthetic head (shipped, zero-shot)  ndcg@10 = {syn:.4f}")
    print(f"  2. + real-trained head (transfer)       ndcg@10 = {real:.4f}"
          f"   ({real - syn:+.4f} vs 1)")
    verdict: dict[str, Any] = {
        "synthetic_retriever_ndcg@10": syn,
        "real_retriever_ndcg@10": real,
        "oracle_ndcg@10|C50": oracle,
        "transfer_gain": real - syn,
    }
    if tier.get("status") == "ok":
        rr = float(tier["routed_ndcg@10"])
        denom = oracle - real
        gap = (rr - real) / denom if (denom == denom
                                      and abs(denom) > 1e-9) else float("nan")
        dnh = not tier["any_regression"]
        print(f"  3. + reranker/router ({tier['chosen_recipe']})      "
              f"ndcg@10 = {rr:.4f}   ({rr - real:+.4f} vs 2)")
        print(f"  4. oracle ceiling (perfect rerank, C50) ndcg@10 = "
              f"{oracle:.4f}")
        print(f"\n  end-to-end: {syn:.3f} -> {real:.3f} -> {rr:.3f}  "
              f"(ceiling {oracle:.3f})")
        print(f"  reranker closes {gap * 100:.0f}% of the retriever->oracle "
              f"gap | do-no-harm: "
              f"{'OK (0 deployed regression)' if dnh else 'VIOLATED'}")
        verdict.update({
            "reranked_ndcg@10": rr,
            "reranked_recipe": tier["chosen_recipe"],
            "reranker_gap_closed_frac": gap,
            "do_no_harm_ok": dnh,
            "reranker_source": tier["source"],
        })
    else:
        print(f"  3. + reranker/router                    n/a "
              f"({tier.get('reason')})")
        print(f"  4. oracle ceiling (perfect rerank, C50) ndcg@10 = "
              f"{oracle:.4f}")
        print(f"\n  retriever-only end-to-end: {syn:.3f} -> {real:.3f}  "
              f"(ceiling {oracle:.3f})")
        print("  reranker tier withheld (not apples-to-apples here -- see "
              "reason); validate it on this corpus before claiming it.")
        verdict["reranked_ndcg@10"] = None
        verdict["reranker_status"] = tier
    return verdict


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=str, default=(
        "runs/v3.1-real-head-hyp-h128-l4-seed0/"
        "real_val_manifold_index.npz"))
    p.add_argument("--corpus", type=str,
                   default="src/data/corpus/real_domain_eval_ft")
    p.add_argument("--split", type=str, default="val",
                   choices=["train", "val", "test", "all"])
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--task", type=int, default=2,
                   help="Task type to evaluate; use -1 for all tasks.")
    p.add_argument("--real-head", type=str,
                   default="runs/v3.1-real-head-hyp-h128-l4-seed0")
    p.add_argument("--synthetic-head", type=str,
                   default="runs/v3.1-baseline-hyp-h128-l4-seed1")
    p.add_argument("--examples", type=int, default=3)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--json-out", type=str, default=None)
    p.add_argument("--live-neo4j", action="store_true",
                   help="Resolve example top-k rows back to live Neo4j nodes. "
                        "Requires an index exported from a corpus containing "
                        "neo4j_node_id.")
    p.add_argument("--prop-limit", type=int, default=5,
                   help="Max compact Neo4j properties to print per node.")
    p.add_argument("--reranker-router-results", type=str,
                   default="runs/reranker_router_real/router_results.json",
                   help="Separately-validated reranker/router result to fold "
                        "into the end-to-end verdict. Shown ONLY when --corpus "
                        "matches --reranker-corpus (apples-to-apples; no "
                        "cross-corpus number).")
    p.add_argument("--reranker-corpus", type=str,
                   default="src/data/corpus/real_domain_eval_ft",
                   help="Corpus the reranker/router result was validated on; "
                        "the reranker tier is shown only if --corpus is this.")
    p.add_argument("--expected-samples", type=int, default=46)
    p.add_argument("--expected-synthetic-ndcg10", type=float, default=0.2422)
    p.add_argument("--expected-real-ndcg10", type=float, default=0.5575)
    p.add_argument("--tolerance", type=float, default=0.005)
    p.add_argument("--min-delta", type=float, default=0.25)
    p.add_argument("--no-sanity-check", action="store_true")
    args = p.parse_args()

    index_path = Path(args.index)
    meta_path = index_path.with_name(index_path.stem + "_meta.json")
    _check_file(index_path, "manifold index")
    _check_file(meta_path, "manifold index meta")

    index = load_index(index_path)
    include_tasks = None if args.task < 0 else {args.task}
    dataset = CorpusDataset(
        corpus_dir=args.corpus,
        split=args.split,
        split_seed=args.split_seed,
        include_tasks=include_tasks,
    )

    real_head_dir = Path(args.real_head)
    synthetic_head_dir = Path(args.synthetic_head)
    _warn_sha(index, real_head_dir, "real head")
    _warn_sha(index, synthetic_head_dir, "synthetic head")

    heads = [
        ("synthetic", _load_head(synthetic_head_dir)),
        ("real", _load_head(real_head_dir)),
    ]

    print("=" * 80)
    print("KGR offline retrieval smoke test")
    print("=" * 80)
    print(f"index:   {index_path}")
    print(f"corpus:  {args.corpus}")
    print(f"split:   {args.split}  task={args.task}  samples={len(dataset)}")
    print(f"graphs:  {len(set(index.graph_idx.tolist()))}  nodes={len(index.graph_idx)}")

    head_results: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for name, fn in heads:
        result, ex = _evaluate_head(
            name=name,
            query_to_point=fn,
            dataset=dataset,
            index=index,
            collect_examples=max(0, args.examples),
            top_k=max(1, args.k),
        )
        head_results.append(result)
        examples.extend(ex)
        _print_head_summary(result)

    if args.live_neo4j:
        try:
            _enrich_examples_live(examples, max(0, args.prop_limit))
        except ValueError as e:
            raise SystemExit(f"[error] {e}") from e

    _print_examples(examples)

    tier = _reranker_tier(args)
    verdict = _print_verdict(head_results, tier, args.task)

    results = {
        "index": str(index_path),
        "corpus": args.corpus,
        "split": args.split,
        "split_seed": args.split_seed,
        "task": args.task,
        "heads": head_results,
        "examples": examples,
        "reranker_tier": tier,
        "verdict": verdict,
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\njson: {out}")

    if args.no_sanity_check:
        return 0
    return 0 if _sanity_checks(results, args) else 1


if __name__ == "__main__":
    sys.exit(main())

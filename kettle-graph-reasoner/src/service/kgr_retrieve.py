r"""KGR retrieval -- the single product CLI.

    py -m src.service.kgr_retrieve --task 2 --seed-ids 12345,67890 \
        --temporal-window 0.2,0.45 --max-hops 4 --k-hops 2 \
        --max-nodes 400 --top-k 10 [--enrich] [--json-out out.json]

Pulls a bounded subgraph LIVE from the archival `neo4j` DB, encodes it
with the SHA-asserted frozen v3.1 encoder, routes the per-task head +
(gated) reranker, and prints ranked Neo4j node ids + scores + a metadata
block (chosen recipe, head run, encoder SHA, subgraph size, per-stage
latency). Output is structural only -- no language.

Windows `py` launcher; console output is ASCII-only (cp1252) per the
HANDOFF invocation contract. Tasks 1 and 3 are refused with a pointer to
the standing-limitation doc.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
for _p in (str(_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.service.determinism import ensure_pythonhashseed  # noqa: E402

ensure_pythonhashseed()  # deterministic edge-type slot order (re-execs once)

from src.service import IN_SCOPE_TASKS, OUT_OF_SCOPE_TASKS  # noqa: E402


def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _window(s: str | None):
    if not s:
        return None
    a, b = s.split(",")
    return (float(a), float(b))


def _enrich(ids: list[int]) -> dict:
    """Resolve top-k node ids back to live Neo4j labels/props (reuses the
    smoke harness helper -- read-only, default `neo4j` DB)."""
    from smoke_retrieval_workflow import _fetch_neo4j_nodes  # type: ignore

    return _fetch_neo4j_nodes(ids, prop_limit=5)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=int, required=True,
                   help=f"in scope: {IN_SCOPE_TASKS} "
                        f"(1/3 refused -- standing limitations)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--seed-ids", type=str,
                     help="comma-separated Neo4j node ids to grow balls from")
    src.add_argument("--cypher", type=str,
                     help="Cypher returning `id(n) AS id` (node id set)")
    p.add_argument("--temporal-window", type=str, default=None,
                   help="start,end in [0,1] (task 2 / temporal compound)")
    p.add_argument("--max-hops", type=int, default=4,
                   help="query max-hops (normalized into the query vector)")
    p.add_argument("--component-tasks", type=str, default="",
                   help="task 5 only: comma-separated component task ids")
    p.add_argument("--k-hops", type=int, default=2,
                   help="subgraph BFS radius around each seed")
    p.add_argument("--max-nodes", type=int, default=400,
                   help="subgraph node cap (the 200-400 trained regime)")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--candidate-c", type=int, default=50,
                   help="retriever first-stage candidate-set size")
    p.add_argument("--subgraph", type=str, default="domain_only",
                   help="kettle_config subgraph spec to scope to")
    p.add_argument("--routing", type=str, default=None)
    p.add_argument("--schema-map", type=str, default=None)
    p.add_argument("--config", type=str, default=None,
                   help="kettle_config.yaml path (lifecycle + subgraph spec)")
    p.add_argument("--enrich", action="store_true",
                   help="resolve top-k ids to live Neo4j labels/props")
    p.add_argument("--json-out", type=str, default=None)
    args = p.parse_args()

    if args.task in OUT_OF_SCOPE_TASKS:
        print(f"[refused] task {args.task}: {OUT_OF_SCOPE_TASKS[args.task]}")
        return 2
    if args.task not in IN_SCOPE_TASKS:
        print(f"[error] unknown task {args.task}; in scope: {IN_SCOPE_TASKS}")
        return 2

    from src.service.inference_engine import KGRRetriever

    eng = KGRRetriever(
        routing_path=args.routing,
        schema_map_path=args.schema_map,
        subgraph=args.subgraph,
        config_path=args.config,
    )
    try:
        res = eng.retrieve(
            task=args.task,
            seed_ids=_csv_ints(args.seed_ids) if args.seed_ids else None,
            cypher=args.cypher,
            temporal_window=_window(args.temporal_window),
            max_hops=args.max_hops,
            component_tasks=tuple(_csv_ints(args.component_tasks))
            if args.component_tasks else (),
            k_hops=args.k_hops,
            max_nodes=args.max_nodes,
            top_k=args.top_k,
            candidate_c=args.candidate_c,
        )
    finally:
        eng.close()

    enrich = {}
    if args.enrich and res.ranked:
        enrich = _enrich([nid for nid, _ in res.ranked])

    print("=" * 78)
    print(f"KGR retrieval  task={res.task}  recipe={res.recipe}")
    print("=" * 78)
    print(f"  encoder sha   : {res.encoder_sha[:12]}... (SHA-asserted frozen "
          f"v3.1 baseline)")
    print(f"  head run      : {res.head_run}")
    print(f"  subgraph      : {res.n_subgraph_nodes} nodes / "
          f"{res.n_subgraph_edges} edges")
    if res.expected_ndcg is not None:
        print(f"  expected ndcg@10 (real, validated): "
              f"{res.expected_ndcg:.4f}")
    print(f"  routing       : {res.routing_reason}")
    lat = res.latencies_ms
    print("  latency (ms)  : " + "  ".join(
        f"{k}={lat[k]}" for k in
        ("pull", "encode_contract", "encoder_forward", "query_score",
         "rerank", "total") if k in lat))
    print()
    print(f"  {'rank':>4} {'neo4j_id':>10} {'score':>12}  labels / props")
    for i, (nid, sc) in enumerate(res.ranked, start=1):
        extra = ""
        if nid in enrich:
            labels = ":".join(enrich[nid]["labels"])
            props = ", ".join(f"{k}={v}"
                              for k, v in enrich[nid]["props"].items())
            extra = f"  {labels}" + (f"  {{ {props} }}" if props else "")
        print(f"  {i:>4} {nid:>10} {sc:>+12.5f}{extra}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": res.task,
            "recipe": res.recipe,
            "encoder_sha": res.encoder_sha,
            "head_run": res.head_run,
            "n_subgraph_nodes": res.n_subgraph_nodes,
            "n_subgraph_edges": res.n_subgraph_edges,
            "expected_ndcg": res.expected_ndcg,
            "routing_reason": res.routing_reason,
            "latencies_ms": res.latencies_ms,
            "ranked": [
                {"neo4j_id": nid, "score": sc,
                 "neo4j": enrich.get(nid)}
                for nid, sc in res.ranked
            ],
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"\njson: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

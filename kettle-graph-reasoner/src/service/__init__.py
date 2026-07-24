r"""KGR v3 serving layer (productionalization) + Context Service.

A thin, no-model-edit wiring layer that turns the research-validated v3.1
artifact chain into a callable product: a bounded subgraph is pulled live
from the archival Neo4j ``neo4j`` DB, encoded with the SHA-asserted frozen
v3.1 encoder, routed through a per-task ``QueryToBall`` head + reranker, and
returned as ranked Neo4j node IDs + scores + provenance metadata. The
Context Service (``context_service.py``) layers deployable graph-context
ordering for an LLM (order_context multi-anchor + missing-link mode) on
the same chain.

This package is serving/wiring only. It does NOT modify any model code
(``src/modelsv3/*`` stays byte-untouched) and never produces language --
output is structural (node IDs + relevance scores), per the CLAUDE.md
non-negotiables. In scope: tasks 0 (provenance), 2 (temporal),
4 (subgraph), 5 (compound). Out of scope: task 1 (noise floor) and
task 3 (documented standing limitation, Docs/PROJECT_HANDOFF.md sec.6).

Public surface:

    SchemaMap            config-driven Neo4j label/rel -> KGR contract
    encode_subgraph      SubgraphPull -> _build_graph_tensors-shaped dict
    Neo4jSource          live driver/session + bounded subgraph pull
    Routing              per-task head + (gated) reranker recipe
    KGRRetriever         the in-process engine (SHA-asserted)
    KGRContextService    LLM-facing context ordering (order_context)

Single CLI entrypoint: ``python -m src.service.kgr_retrieve``.
"""

from __future__ import annotations

__all__ = [
    "SchemaMap",
    "encode_subgraph",
    "Neo4jSource",
    "SubgraphPull",
    "Routing",
    "KGRRetriever",
    "RetrievalResult",
    "KGRContextService",
    "IN_SCOPE_TASKS",
    "OUT_OF_SCOPE_TASKS",
]

# Task scope is enforced in one place (the CLI and engine both read this).
IN_SCOPE_TASKS = (0, 2, 4, 5)
OUT_OF_SCOPE_TASKS = {
    1: "task 1 (entity-resolution) is ~unsolvable by scoring "
       "(oracle ceiling ~0.26) -- a consequence of the query-agnostic "
       "encoder commitment, not a bug. See Docs/PROJECT_HANDOFF.md sec.4.",
    3: "task 3 (multi-hop) is a documented standing limitation: the "
       "query-head lever was tried and failed (2026-05-17). See "
       "Docs/PROJECT_HANDOFF.md sec.6.",
}


def __getattr__(name: str):  # lazy re-exports (keep import side effects local)
    if name in ("SchemaMap",):
        from .schema_map import SchemaMap

        return SchemaMap
    if name == "encode_subgraph":
        from .tensor_contract import encode_subgraph

        return encode_subgraph
    if name in ("Neo4jSource", "SubgraphPull"):
        from .neo4j_source import Neo4jSource, SubgraphPull

        return {"Neo4jSource": Neo4jSource, "SubgraphPull": SubgraphPull}[name]
    if name == "Routing":
        from .routing import Routing

        return Routing
    if name in ("KGRRetriever", "RetrievalResult"):
        from .inference_engine import KGRRetriever, RetrievalResult

        return {"KGRRetriever": KGRRetriever,
                "RetrievalResult": RetrievalResult}[name]
    if name == "KGRContextService":
        from .context_service import KGRContextService

        return KGRContextService
    raise AttributeError(name)

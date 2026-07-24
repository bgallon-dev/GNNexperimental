"""
Subgraph extraction.

Given a named subgraph spec from config.subgraphs, build the Cypher query
that materializes it (label filters + rel-type filters + lifecycle filter +
optional temporal filter) and return either:

    - an in-memory NetworkX graph (for topology analysis)
    - a streamed JSONL edge list (for GNN training loaders)

Spec fields:
    include_labels: list[str]      -- empty = all labels
    include_rel_types: list[str]   -- empty = all rel types
    exclude_rel_types: list[str]   -- evaluated after include
    require_lifecycle_clean: bool  -- apply the lifecycle predicate
    temporal_filter: dict          -- optional: {property, label, cutoff, comparison}

The temporal_filter requires that the subgraph be defined with respect to
some Year or Period node; for the Kettle schema the convention is that
events are linked to Year via :COVERS_YEAR or :IN_YEAR, so we reach any
entity's "year" by traversing to a Year node.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graph_diagnostics.core import DiagnosticConfig, lifecycle_predicate


@dataclass
class SubgraphStats:
    name: str
    nodes: int
    edges: int
    labels_seen: dict[str, int]
    rel_types_seen: dict[str, int]


def build_node_query(spec: dict, config: DiagnosticConfig) -> tuple[str, dict]:
    """Return a (Cypher, params) pair that yields nodes for the subgraph."""
    clauses: list[str] = []
    params: dict[str, Any] = {}

    include_labels = spec.get("include_labels") or []
    if include_labels:
        # Use label predicate `n:Foo OR n:Bar` rather than MATCH (n:Foo)
        # so we get a single MATCH per query.
        label_clause = " OR ".join(f"n:`{lbl}`" for lbl in include_labels)
        clauses.append(f"({label_clause})")

    if spec.get("require_lifecycle_clean", True):
        clauses.append(lifecycle_predicate(config, var="n"))

    temporal = spec.get("temporal_filter")
    if temporal:
        prop = temporal["property"]
        cutoff = temporal["cutoff"]
        comp = temporal["comparison"]
        ylabel = temporal.get("label", "Year")
        if comp not in ("<", "<=", ">", ">=", "=", "!="):
            raise ValueError(f"Invalid temporal comparison: {comp!r}")
        # Nodes pass the temporal filter if they have ANY path to a Year node
        # whose `year` satisfies the cutoff. We also allow nodes that ARE
        # Year nodes directly. For nodes with no Year connection at all,
        # include them (we can't filter what we can't date).
        clauses.append(
            f"("
            f"(n:`{ylabel}`) OR "
            f"NOT EXISTS {{ MATCH (n)-[*1..3]-(:`{ylabel}`) }} OR "
            f"EXISTS {{ MATCH (n)-[*1..3]-(y:`{ylabel}`) "
            f"WHERE y.`{prop}` {comp} $temporal_cutoff }}"
            f")"
        )
        params["temporal_cutoff"] = cutoff

    where = " AND ".join(clauses) if clauses else "true"
    cypher = f"MATCH (n) WHERE {where} RETURN elementId(n) AS id, labels(n) AS lbls"
    return cypher, params


def build_edge_query(spec: dict, config: DiagnosticConfig) -> tuple[str, dict]:
    """Return a (Cypher, params) pair that yields edges for the subgraph.

    Edges are included only if BOTH endpoints pass the node filter.
    """
    node_clauses_a: list[str] = []
    node_clauses_b: list[str] = []

    include_labels = spec.get("include_labels") or []
    if include_labels:
        la = " OR ".join(f"a:`{lbl}`" for lbl in include_labels)
        lb = " OR ".join(f"b:`{lbl}`" for lbl in include_labels)
        node_clauses_a.append(f"({la})")
        node_clauses_b.append(f"({lb})")

    if spec.get("require_lifecycle_clean", True):
        node_clauses_a.append(lifecycle_predicate(config, var="a"))
        node_clauses_b.append(lifecycle_predicate(config, var="b"))

    where_parts = node_clauses_a + node_clauses_b

    # Rel-type include/exclude.
    include_rels = spec.get("include_rel_types") or []
    exclude_rels = spec.get("exclude_rel_types") or []
    params: dict[str, Any] = {}
    if include_rels:
        rt = "|".join(f"`{t}`" for t in include_rels)
        rel_pattern = f"[r:{rt}]"
    else:
        rel_pattern = "[r]"
    if exclude_rels:
        where_parts.append("NOT type(r) IN $excluded_types")
        params["excluded_types"] = exclude_rels

    # Temporal filter applies to both endpoints.
    temporal = spec.get("temporal_filter")
    if temporal:
        prop = temporal["property"]
        cutoff = temporal["cutoff"]
        comp = temporal["comparison"]
        ylabel = temporal.get("label", "Year")
        for v in ("a", "b"):
            where_parts.append(
                f"("
                f"({v}:`{ylabel}`) OR "
                f"NOT EXISTS {{ MATCH ({v})-[*1..3]-(:`{ylabel}`) }} OR "
                f"EXISTS {{ MATCH ({v})-[*1..3]-(y:`{ylabel}`) "
                f"WHERE y.`{prop}` {comp} $temporal_cutoff }}"
                f")"
            )
        params["temporal_cutoff"] = cutoff

    where = " AND ".join(where_parts) if where_parts else "true"
    cypher = (
        f"MATCH (a)-{rel_pattern}->(b) "
        f"WHERE {where} "
        f"RETURN elementId(a) AS a, elementId(b) AS b, "
        f"       type(r) AS t, labels(a) AS la, labels(b) AS lb"
    )
    return cypher, params


def extract_stats(session, spec: dict, config: DiagnosticConfig,
                  name: str) -> SubgraphStats:
    """Count nodes/edges + label/rel-type distributions for a subgraph.

    Uses server-side Cypher aggregation — returns one row per label/rel-type
    rather than streaming every node/edge through Python.
    """
    node_q, node_params = build_node_query(spec, config)
    edge_q, edge_params = build_edge_query(spec, config)

    # Derive WHERE clause from the node query by stripping the RETURN clause.
    # node_q ends with "RETURN elementId(n) AS id, labels(n) AS lbls"
    node_where_block = node_q.rsplit("RETURN", 1)[0]
    edge_where_block = edge_q.rsplit("RETURN", 1)[0]

    # Server-side label distribution.
    labels_seen: dict[str, int] = {}
    n_nodes = 0
    lbl_agg = (
        f"{node_where_block}"
        f"UNWIND labels(n) AS lbl RETURN lbl, count(*) AS c ORDER BY c DESC"
    )
    for row in session.run(lbl_agg, **node_params):  # pyright: ignore[reportArgumentType]
        labels_seen[row["lbl"]] = row["c"]
        n_nodes += row["c"]
    # n_nodes overcounts multi-label nodes (each label counted separately);
    # get the true node count with a dedicated aggregation.
    node_count_q = f"{node_where_block}RETURN count(n) AS cnt"
    n_nodes = session.run(node_count_q, **node_params).single()["cnt"]  # pyright: ignore[reportArgumentType]

    # Server-side rel-type distribution.
    rel_types_seen: dict[str, int] = {}
    n_edges = 0
    rel_agg = f"{edge_where_block}RETURN type(r) AS t, count(*) AS c ORDER BY c DESC"
    for row in session.run(rel_agg, **edge_params):  # pyright: ignore[reportArgumentType]
        rel_types_seen[row["t"]] = row["c"]
        n_edges += row["c"]

    return SubgraphStats(
        name=name,
        nodes=n_nodes,
        edges=n_edges,
        labels_seen=labels_seen,
        rel_types_seen=rel_types_seen,
    )


def extract_to_networkx(session, spec: dict, config: DiagnosticConfig):
    """Pull the subgraph into a NetworkX Graph for topology analysis."""
    import networkx as nx

    G = nx.Graph()
    node_q, node_params = build_node_query(spec, config)
    for row in session.run(node_q, **node_params):
        G.add_node(row["id"], labels=row["lbls"])

    edge_q, edge_params = build_edge_query(spec, config)
    for row in session.run(edge_q, **edge_params):
        # Only add if both endpoints made it through the node filter.
        if row["a"] in G and row["b"] in G:
            G.add_edge(row["a"], row["b"], rel_type=row["t"])
    return G


def extract_to_jsonl(session, spec: dict, config: DiagnosticConfig,
                     output_path: str | Path) -> tuple[int, int]:
    """Stream the subgraph to JSONL: one node-or-edge record per line.

    Records have a `_type` discriminator ("node" or "edge") so the GNN
    loader can parse them in a single pass.

    Returns (nodes_written, edges_written).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    node_q, node_params = build_node_query(spec, config)
    edge_q, edge_params = build_edge_query(spec, config)

    nodes_written = 0
    edges_written = 0
    seen_nodes: set[str] = set()

    with output_path.open("w", encoding="utf-8") as f:
        for row in session.run(node_q, **node_params):
            f.write(json.dumps({
                "_type": "node", "id": row["id"], "labels": row["lbls"],
            }))
            f.write("\n")
            seen_nodes.add(row["id"])
            nodes_written += 1

        for row in session.run(edge_q, **edge_params):
            if row["a"] in seen_nodes and row["b"] in seen_nodes:
                f.write(json.dumps({
                    "_type": "edge",
                    "source": row["a"], "target": row["b"],
                    "rel_type": row["t"],
                    "source_labels": row["la"], "target_labels": row["lb"],
                }))
                f.write("\n")
                edges_written += 1

    return nodes_written, edges_written

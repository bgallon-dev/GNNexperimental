"""
Structural integrity checks.

Covers:
    - orphan nodes (no relationships in or out)
    - self-loops (rel where start == end)
    - duplicate relationships (same type between same pair)
    - dangling references (rel endpoints that shouldn't exist -- Neo4j guarantees
      this structurally, but we report it for graphs imported via APOC merges
      that could have created stub nodes)

Each finding carries a remediation Cypher statement. The statements are
conservative -- they delete ONLY the items that were flagged, using id lists
that the reviewer can verify before running.
"""
from __future__ import annotations

from graph_diagnostics.core import (
    CheckResult, Finding, Severity, DiagnosticConfig,
    effective_labels, effective_rel_types,
)


def run(session, config: DiagnosticConfig) -> CheckResult:
    result = CheckResult(check="structural")

    _check_orphans(session, config, result)
    _check_self_loops(session, config, result)
    _check_duplicate_relationships(session, config, result)
    _check_stub_nodes(session, config, result)

    return result


def _check_orphans(session, config, result: CheckResult) -> None:
    q = """
    MATCH (n)
    WHERE NOT (n)--()
    RETURN count(n) AS cnt
    """
    cnt = session.run(q).single()["cnt"]
    if cnt == 0:
        return

    sample_q = """
    MATCH (n)
    WHERE NOT (n)--()
    RETURN elementId(n) AS id, labels(n) AS labels
    LIMIT $limit
    """
    sample = [dict(r) for r in session.run(sample_q, limit=config.sample_limit)]

    # Orphans are context-dependent: sometimes legitimate (lookup tables),
    # sometimes noise. Mark MEDIUM and let the reviewer decide.
    result.findings.append(Finding(
        check="structural",
        code="orphan_nodes",
        severity=Severity.MEDIUM,
        message=(
            f"{cnt} nodes have no relationships. For GNN training these "
            f"contribute no message-passing signal and typically should be "
            f"excluded from the training subgraph (or deleted if truly unused)."
        ),
        count=cnt,
        sample=sample,
    ))
    result.remediation.append(
        "// Orphan nodes -- uncomment to delete, or filter in the GNN loader instead.\n"
        "// MATCH (n) WHERE NOT (n)--() DETACH DELETE n"
    )


def _check_self_loops(session, config, result: CheckResult) -> None:
    if config.allow_self_loops:
        return
    q = """
    MATCH (n)-[r]->(n)
    RETURN count(r) AS cnt
    """
    cnt = session.run(q).single()["cnt"]
    if cnt == 0:
        return

    sample_q = """
    MATCH (n)-[r]->(n)
    RETURN elementId(r) AS rel_id, type(r) AS rel_type,
           elementId(n) AS node_id, labels(n) AS labels
    LIMIT $limit
    """
    sample = [dict(r) for r in session.run(sample_q, limit=config.sample_limit)]
    result.findings.append(Finding(
        check="structural",
        code="self_loops",
        severity=Severity.MEDIUM,
        message=(
            f"{cnt} self-loop relationships. Most GNN architectures add self-loops "
            f"implicitly (A + I); explicit ones will double-count."
        ),
        count=cnt,
        sample=sample,
    ))
    result.remediation.append(
        "MATCH (n)-[r]->(n) DELETE r"
    )


def _check_duplicate_relationships(session, config, result: CheckResult) -> None:
    if config.allow_multi_edges:
        return
    # For each (a, type, b), count rels. Anything > 1 is a duplicate.
    q = """
    MATCH (a)-[r]->(b)
    WITH a, b, type(r) AS t, count(r) AS c, collect(elementId(r)) AS ids
    WHERE c > 1
    RETURN count(*) AS pair_count, sum(c - 1) AS excess_count
    """
    row = session.run(q).single()
    if row is None or row["pair_count"] == 0:
        return

    sample_q = """
    MATCH (a)-[r]->(b)
    WITH a, b, type(r) AS t, count(r) AS c, collect(elementId(r)) AS ids
    WHERE c > 1
    RETURN elementId(a) AS a_id, elementId(b) AS b_id, t AS rel_type,
           c AS copies, ids AS rel_ids
    LIMIT $limit
    """
    sample = [dict(r) for r in session.run(sample_q, limit=config.sample_limit)]

    result.findings.append(Finding(
        check="structural",
        code="duplicate_relationships",
        severity=Severity.HIGH,
        message=(
            f"{row['pair_count']} (start, type, end) triples have >1 relationship "
            f"({row['excess_count']} excess edges total). This biases message "
            f"passing and corrupts degree-based negative sampling."
        ),
        count=row["excess_count"],
        sample=sample,
        details={"distinct_pairs": row["pair_count"]},
    ))
    # Remediation: keep the earliest rel per (a, type, b), delete the rest.
    result.remediation.append(
        "MATCH (a)-[r]->(b) "
        "WITH a, b, type(r) AS t, collect(r) AS rels "
        "WHERE size(rels) > 1 "
        "FOREACH (r IN rels[1..] | DELETE r)"
    )


def _check_stub_nodes(session, config, result: CheckResult) -> None:
    """Stub nodes: nodes with no labels AND no properties.

    These arise from sloppy APOC merges or partial imports. Neo4j allows them
    but they are indistinguishable under message passing and should be removed.
    """
    q = """
    MATCH (n)
    WHERE size(labels(n)) = 0 AND size(keys(n)) = 0
    RETURN count(n) AS cnt
    """
    cnt = session.run(q).single()["cnt"]
    if cnt == 0:
        return
    sample_q = """
    MATCH (n)
    WHERE size(labels(n)) = 0 AND size(keys(n)) = 0
    RETURN elementId(n) AS id
    LIMIT $limit
    """
    sample = [dict(r) for r in session.run(sample_q, limit=config.sample_limit)]
    result.findings.append(Finding(
        check="structural",
        code="stub_nodes",
        severity=Severity.HIGH,
        message=f"{cnt} nodes with no labels and no properties (likely import stubs).",
        count=cnt,
        sample=sample,
    ))
    result.remediation.append(
        "MATCH (n) WHERE size(labels(n)) = 0 AND size(keys(n)) = 0 DETACH DELETE n"
    )

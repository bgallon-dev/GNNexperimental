"""
Schema consistency checks.

Covers:
    - multi-labeled nodes where label set is inconsistent across the graph
      (e.g. most :Person nodes also have :Agent, but 14 don't)
    - missing required properties per label
    - type drift: a property key that has mixed value types across nodes of
      the same label (e.g. 'year' is int on 900 nodes and str on 30)
    - non-unique unique keys

Unlike structural, schema findings often want config: what does "required"
mean? The check uses config.required_properties and config.unique_keys when
provided, and performs opportunistic multi-label / type-drift inference
otherwise.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from graph_diagnostics.core import (
    CheckResult, Finding, Severity, DiagnosticConfig, effective_labels,
)


def run(session, config: DiagnosticConfig) -> CheckResult:
    result = CheckResult(check="schema")

    labels = effective_labels(session, config)
    _check_required_properties(session, config, result)
    _check_unique_keys(session, config, result)
    _check_type_drift(session, config, result, labels)
    _check_inconsistent_label_sets(session, config, result, labels)

    return result


def _check_required_properties(session, config, result) -> None:
    for label, required in config.required_properties.items():
        for prop in required:
            q = f"""
            MATCH (n:`{label}`)
            WHERE n.`{prop}` IS NULL
            RETURN count(n) AS cnt
            """
            cnt = session.run(q).single()["cnt"]
            if cnt == 0:
                continue
            sample_q = f"""
            MATCH (n:`{label}`)
            WHERE n.`{prop}` IS NULL
            RETURN elementId(n) AS id
            LIMIT $limit
            """
            sample = [dict(r) for r in session.run(sample_q, limit=config.sample_limit)]
            result.findings.append(Finding(
                check="schema",
                code=f"missing_required_property:{label}.{prop}",
                severity=Severity.HIGH,
                message=(
                    f"{cnt} :{label} nodes are missing required property `{prop}`."
                ),
                count=cnt,
                sample=sample,
            ))
            result.remediation.append(
                f"// Inspect before running -- may need a default or a delete.\n"
                f"// MATCH (n:`{label}`) WHERE n.`{prop}` IS NULL RETURN n"
            )


def _check_unique_keys(session, config, result) -> None:
    for label, key in config.unique_keys.items():
        q = f"""
        MATCH (n:`{label}`)
        WHERE n.`{key}` IS NOT NULL
        WITH n.`{key}` AS k, count(n) AS c, collect(elementId(n)) AS ids
        WHERE c > 1
        RETURN count(*) AS dup_keys, sum(c) AS affected_nodes,
               collect({{key: k, ids: ids}})[0..$limit] AS sample
        """
        row = session.run(q, limit=config.sample_limit).single()
        if not row or row["dup_keys"] == 0:
            continue
        result.findings.append(Finding(
            check="schema",
            code=f"non_unique_key:{label}.{key}",
            severity=Severity.CRITICAL,
            message=(
                f"{row['dup_keys']} distinct values of `{key}` appear on multiple "
                f":{label} nodes ({row['affected_nodes']} total). Declared unique "
                f"key is not unique."
            ),
            count=row["affected_nodes"],
            sample=row["sample"],
        ))
        result.remediation.append(
            f"// Non-unique key {label}.{key}: manual review required. Candidates:\n"
            f"// MATCH (n:`{label}`) WITH n.`{key}` AS k, collect(n) AS ns "
            f"WHERE size(ns) > 1 RETURN k, ns"
        )


def _check_type_drift(session, config, result, labels: list[str]) -> None:
    """For each label, for each property, check if value types vary."""
    for label in labels:
        # Get property keys actually in use for this label.
        keys_q = f"""
        MATCH (n:`{label}`)
        UNWIND keys(n) AS k
        RETURN DISTINCT k AS key
        """
        keys = [r["key"] for r in session.run(keys_q)]
        for key in keys:
            q = f"""
            MATCH (n:`{label}`)
            WHERE n.`{key}` IS NOT NULL
            WITH apoc.meta.type(n.`{key}`) AS t, count(*) AS c
            RETURN collect({{t: t, c: c}}) AS types
            """
            # Fall back to a pure-Cypher type inference if APOC is not installed.
            try:
                row = session.run(q).single()
                types = row["types"] if row else []
            except Exception:
                types = _type_drift_fallback(session, label, key)

            if len(types) <= 1:
                continue
            # If the minority type covers < 1% of nodes, still MEDIUM but noted.
            total = sum(t["c"] for t in types)
            sorted_t = sorted(types, key=lambda x: -x["c"])
            dominant, *rest = sorted_t
            minority_share = sum(t["c"] for t in rest) / total
            severity = Severity.HIGH if minority_share > 0.01 else Severity.MEDIUM
            result.findings.append(Finding(
                check="schema",
                code=f"type_drift:{label}.{key}",
                severity=severity,
                message=(
                    f":{label}.{key} has {len(types)} observed value types "
                    f"(dominant: {dominant['t']} @ {dominant['c']}). "
                    f"Minority share: {minority_share:.2%}."
                ),
                count=total,
                details={"types": types},
            ))
            result.remediation.append(
                f"// Type drift on {label}.{key}: coerce to a single type.\n"
                f"// Example: MATCH (n:`{label}`) WHERE n.`{key}` IS NOT NULL "
                f"SET n.`{key}` = toString(n.`{key}`)"
            )


def _type_drift_fallback(session, label: str, key: str) -> list[dict]:
    """Type-inference without APOC, using Cypher built-ins."""
    q = f"""
    MATCH (n:`{label}`)
    WHERE n.`{key}` IS NOT NULL
    WITH n.`{key}` AS v
    WITH
      CASE
        WHEN v IS :: INTEGER THEN 'INTEGER'
        WHEN v IS :: FLOAT   THEN 'FLOAT'
        WHEN v IS :: STRING  THEN 'STRING'
        WHEN v IS :: BOOLEAN THEN 'BOOLEAN'
        WHEN v IS :: LIST<ANY> THEN 'LIST'
        WHEN v IS :: DATE THEN 'DATE'
        WHEN v IS :: DATETIME THEN 'DATETIME'
        ELSE 'OTHER'
      END AS t
    RETURN t, count(*) AS c
    """
    try:
        return [{"t": r["t"], "c": r["c"]} for r in session.run(q)]
    except Exception:
        return []


def _check_inconsistent_label_sets(session, config, result, labels: list[str]) -> None:
    """Flag labels whose nodes carry inconsistent secondary-label sets.

    Example: 920 :Person nodes are also :Agent, but 14 :Person nodes aren't.
    Reports the dominant label-set pattern and the outliers.
    """
    for label in labels:
        q = f"""
        MATCH (n:`{label}`)
        WITH labels(n) AS lbls
        WITH apoc.coll.sort(lbls) AS lbl_sorted, count(*) AS c
        RETURN collect({{labels: lbl_sorted, c: c}}) AS sets
        """
        try:
            row = session.run(q).single()
        except Exception:
            # No APOC: fall back to pure Cypher.
            q2 = f"""
            MATCH (n:`{label}`)
            WITH labels(n) AS lbls
            RETURN lbls, count(*) AS c
            """
            sets = [{"labels": sorted(r["lbls"]), "c": r["c"]} for r in session.run(q2)]
            # Deduplicate sorted lists manually.
            agg: dict[tuple, int] = defaultdict(int)
            for s in sets:
                agg[tuple(s["labels"])] += s["c"]
            sets = [{"labels": list(k), "c": v} for k, v in agg.items()]
        else:
            sets = row["sets"] if row else []

        if len(sets) <= 1:
            continue
        total = sum(s["c"] for s in sets)
        sets_sorted = sorted(sets, key=lambda x: -x["c"])
        dominant = sets_sorted[0]
        outlier_count = total - dominant["c"]
        if outlier_count == 0 or outlier_count / total > 0.5:
            # If > 50% "outliers", there IS no dominant pattern -- skip.
            continue
        result.findings.append(Finding(
            check="schema",
            code=f"inconsistent_label_sets:{label}",
            severity=Severity.MEDIUM,
            message=(
                f":{label} has {len(sets)} distinct label-set patterns. "
                f"Dominant pattern: {dominant['labels']} ({dominant['c']} / {total}). "
                f"{outlier_count} nodes deviate."
            ),
            count=outlier_count,
            details={"patterns": sets_sorted[:10]},
        ))

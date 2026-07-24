"""
Lifecycle check: quarantine, soft-delete, access-level filtering.

The Kettle schema has built-in review workflow properties:
    - deleted_at         (soft delete)
    - quarantine_status  (held out from training)
    - quarantine_reason  (why)
    - quarantine_timestamp
    - access_level / donor_restricted  (privacy / IP)
    - reclassified_at / from / reason  (schema migration audit trail)

This check's job is NOT to find bugs -- these are deliberate workflow states.
Its job is to:

    1. Report counts per layer so you know what's being excluded.
    2. Emit a canonical lifecycle predicate that downstream checks and the
       subgraph extractor will use to filter training data consistently.
    3. Flag quarantined nodes that are ALSO participating in confirmed
       relationships (e.g. a quarantined Entity with REFERS_TO edges) --
       these are the real bugs, because a confirmed edge to a quarantined
       node means supervision is pointing at excluded data.
"""
from __future__ import annotations

from graph_diagnostics.core import (
    CheckResult, Finding, Severity, DiagnosticConfig, effective_labels,
    lifecycle_predicate,
)


def run(session, config: DiagnosticConfig) -> CheckResult:
    result = CheckResult(check="lifecycle")

    lc = config.lifecycle or {}
    if not lc:
        result.skipped = True
        result.skip_reason = (
            "No lifecycle config set. Add a `lifecycle:` block to the YAML to "
            "enable quarantine / soft-delete / access-level filtering."
        )
        return result

    _report_counts(session, config, result)
    _report_confirmed_edges_to_excluded(session, config, result)
    _report_reclassification_activity(session, config, result)

    return result


def _report_counts(session, config, result) -> None:
    """Per-label count of nodes that WOULD be excluded by lifecycle filters."""
    lc = config.lifecycle

    labels = effective_labels(session, config)
    per_label: list[dict] = []

    for label in labels:
        row = session.run(
            f"MATCH (n:`{label}`) RETURN count(n) AS total"
        ).single()
        total = row["total"]
        if total == 0:
            continue

        excluded_by: dict[str, int] = {}

        if lc.get("exclude_if_deleted"):
            prop = lc.get("deleted_property", "deleted_at")
            r = session.run(
                f"MATCH (n:`{label}`) WHERE n.`{prop}` IS NOT NULL RETURN count(n) AS c"
            ).single()
            if r["c"] > 0:
                excluded_by["deleted"] = r["c"]

        if lc.get("exclude_if_quarantined"):
            prop = lc.get("quarantine_property", "quarantine_status")
            vals = lc.get("quarantine_excluded_values", ["quarantined"])
            r = session.run(
                f"MATCH (n:`{label}`) WHERE n.`{prop}` IN $vals RETURN count(n) AS c",
                vals=vals,
            ).single()
            if r["c"] > 0:
                excluded_by["quarantined"] = r["c"]

        if lc.get("exclude_restricted_access"):
            prop = lc.get("access_property", "access_level")
            vals = lc.get("access_excluded_values", ["restricted"])
            r = session.run(
                f"MATCH (n:`{label}`) WHERE n.`{prop}` IN $vals RETURN count(n) AS c",
                vals=vals,
            ).single()
            if r["c"] > 0:
                excluded_by["access_restricted"] = r["c"]

        if excluded_by:
            per_label.append({
                "label": label,
                "total": total,
                "excluded": sum(excluded_by.values()),
                "excluded_share": sum(excluded_by.values()) / total,
                "by_reason": excluded_by,
            })

    if not per_label:
        result.findings.append(Finding(
            check="lifecycle",
            code="no_exclusions",
            severity=Severity.INFO,
            message=(
                "No nodes match any lifecycle exclusion criterion. Either the "
                "workflow hasn't produced quarantined/deleted nodes yet, or "
                "the property conventions in config don't match the schema."
            ),
        ))
        return

    total_excluded = sum(entry["excluded"] for entry in per_label)
    result.findings.append(Finding(
        check="lifecycle",
        code="exclusion_summary",
        severity=Severity.INFO,
        message=(
            f"{total_excluded} nodes across {len(per_label)} labels match "
            f"lifecycle exclusion criteria and will be filtered from training."
        ),
        count=total_excluded,
        details={"per_label": per_label},
    ))

    # Any single label with > 25% exclusion rate is worth calling out --
    # either the workflow is catching real problems, or the exclusion rules
    # are miscalibrated.
    for entry in per_label:
        if entry["excluded_share"] > 0.25:
            result.findings.append(Finding(
                check="lifecycle",
                code=f"high_exclusion_rate:{entry['label']}",
                severity=Severity.MEDIUM,
                message=(
                    f":{entry['label']} excludes {entry['excluded_share']:.1%} "
                    f"({entry['excluded']} / {entry['total']}) of its nodes. "
                    f"Either the corpus has systemic issues with this label, "
                    f"or the exclusion rules are too aggressive."
                ),
                count=entry["excluded"],
                details=entry,
            ))


def _report_confirmed_edges_to_excluded(session, config, result) -> None:
    """Bug class: confirmed supervision edges pointing at excluded nodes.

    If a Mention -[:REFERS_TO]-> Entity exists but the Entity is quarantined,
    the REFERS_TO edge is training supervision pointing at data we're
    excluding. Either the quarantine is wrong, or the edge is stale.
    """
    lc = config.lifecycle
    pred = lifecycle_predicate(config, var="target")
    # Invert the predicate: we want nodes where lifecycle_predicate is FALSE
    # (i.e. excluded nodes) that are still being pointed at by confirmed edges.
    # Supervision-bearing rel types for the ER task:
    confirmed_rel_types = ["REFERS_TO", "SUPPORTS", "EVIDENCED_BY"]

    rt_list = "|".join(f"`{t}`" for t in confirmed_rel_types)
    q = f"""
    MATCH (source)-[r:{rt_list}]->(target)
    WHERE NOT ({pred})
    RETURN type(r) AS rel_type, labels(target) AS target_labels,
           count(r) AS c
    ORDER BY c DESC
    """
    try:
        rows = [dict(r) for r in session.run(q)]
    except Exception as exc:
        result.findings.append(Finding(
            check="lifecycle",
            code="confirmed_edges_check_failed",
            severity=Severity.LOW,
            message=f"Could not scan for confirmed-edge-to-excluded: {exc}",
        ))
        return

    if not rows:
        return
    total = sum(r["c"] for r in rows)
    result.findings.append(Finding(
        check="lifecycle",
        code="confirmed_edges_to_excluded",
        severity=Severity.HIGH,
        message=(
            f"{total} confirmed supervision edges ({', '.join(confirmed_rel_types)}) "
            f"point at nodes that would be excluded by lifecycle filters. "
            f"Either the edges are stale (the target was quarantined after the "
            f"edge was written) or the quarantine is wrong. Inspect before "
            f"training -- silently dropping these corrupts the supervision signal."
        ),
        count=total,
        details={"by_rel_type_and_target_label": rows},
    ))
    result.remediation.append(
        "// Confirmed edges to excluded nodes -- inspect case by case.\n"
        f"// MATCH (a)-[r:{rt_list}]->(b) WHERE NOT ({pred.replace('target', 'b')}) "
        "RETURN a, r, b LIMIT 50"
    )


def _report_reclassification_activity(session, config, result) -> None:
    """Informational: how active is the reclassification workflow?"""
    q = """
    MATCH (n) WHERE n.reclassified_at IS NOT NULL
    RETURN count(n) AS c, count(DISTINCT n.reclassified_reason) AS reasons
    """
    try:
        row = session.run(q).single()
    except Exception:
        return
    if not row or row["c"] == 0:
        return
    result.findings.append(Finding(
        check="lifecycle",
        code="reclassification_activity",
        severity=Severity.INFO,
        message=(
            f"{row['c']} nodes carry reclassified_at timestamps across "
            f"{row['reasons']} distinct reasons. This is expected workflow "
            f"telemetry -- not a bug -- but worth logging for training runs."
        ),
        count=row["c"],
    ))

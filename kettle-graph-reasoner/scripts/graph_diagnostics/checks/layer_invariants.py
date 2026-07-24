"""
Layer invariants: hierarchical containment and backref integrity.

The Kettle graph has three layers:
    L1 (document):  Document -> Page -> Section -> Paragraph
    L2 (extraction): Paragraph -[:CONTAINS_MENTION]-> Mention, Mention -> Claim
    L3 (domain):    Entity, Place, Refuge, ...

Certain invariants are structural, not probabilistic:
    - every Paragraph MUST have a Section parent
    - every Mention MUST live inside a Paragraph
    - every Claim SHOULD have at least one evidence edge
    - every Observation SHOULD be located somewhere

These are configured as Cypher predicates in config.layer_invariants. Each
invariant runs a COUNT over nodes whose label matches but whose predicate
evaluates to false. Severity is per-invariant.

Why Cypher predicates rather than hardcoded checks? The invariants are
schema-specific; if you change the schema, you edit the YAML, not the code.
"""
from __future__ import annotations

from graph_diagnostics.core import (
    CheckResult, Finding, Severity, DiagnosticConfig, lifecycle_predicate,
)


_SEVERITY_LOOKUP = {
    "info": Severity.INFO,
    "low": Severity.LOW,
    "medium": Severity.MEDIUM,
    "high": Severity.HIGH,
    "critical": Severity.CRITICAL,
}


def run(session, config: DiagnosticConfig) -> CheckResult:
    result = CheckResult(check="layer_invariants")

    if not config.layer_invariants:
        result.skipped = True
        result.skip_reason = (
            "No layer_invariants configured. Add a `layer_invariants:` block "
            "to the YAML."
        )
        return result

    lc_pred = lifecycle_predicate(config, var="n")

    for name, spec in config.layer_invariants.items():
        label = spec["label"]
        predicate = spec["must_satisfy"]
        sev_str = spec.get("severity", "medium").lower()
        severity = _SEVERITY_LOOKUP.get(sev_str, Severity.MEDIUM)

        # Count nodes where predicate is false, among lifecycle-clean nodes.
        # We apply the lifecycle filter so we don't double-report already-
        # excluded nodes as invariant violations.
        q = f"""
        MATCH (n:`{label}`)
        WHERE {lc_pred} AND NOT ({predicate})
        RETURN count(n) AS c
        """
        try:
            row = session.run(q).single()
            cnt = row["c"] if row else 0
        except Exception as exc:
            result.findings.append(Finding(
                check="layer_invariants",
                code=f"{name}:query_error",
                severity=Severity.LOW,
                message=(
                    f"Invariant {name!r} failed to evaluate: {exc}. "
                    f"Check the predicate syntax against your Neo4j version "
                    f"(EXISTS {{ subquery }} requires 5.x)."
                ),
            ))
            continue

        if cnt == 0:
            continue

        sample_q = f"""
        MATCH (n:`{label}`)
        WHERE {lc_pred} AND NOT ({predicate})
        RETURN elementId(n) AS id, labels(n) AS labels
        LIMIT $limit
        """
        sample = [dict(r) for r in session.run(sample_q, limit=config.sample_limit)]

        result.findings.append(Finding(
            check="layer_invariants",
            code=name,
            severity=severity,
            message=(
                f"Invariant {name!r} fails for {cnt} :{label} nodes "
                f"(after lifecycle filtering). Predicate: {predicate}"
            ),
            count=cnt,
            sample=sample,
            details={"label": label, "predicate": predicate},
        ))

        # Remediation: surface the query rather than auto-fix, because the
        # right fix is schema-specific (delete? backfill? relabel?).
        result.remediation.append(
            f"// Invariant violation: {name}\n"
            f"// MATCH (n:`{label}`) WHERE {lc_pred} AND NOT ({predicate}) "
            f"RETURN n LIMIT 50"
        )

    return result

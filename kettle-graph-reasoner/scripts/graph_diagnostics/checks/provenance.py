"""
Provenance / signed-lineage coverage.

Reports which nodes carry the configured provenance properties and which
do not. Default properties: albc_barcode, content_hash, signature.

Two modes:

    1. Advisory (default): just measure coverage across all labels.
    2. Required: if config.provenance_required_labels is non-empty, any node
       of those labels missing provenance is flagged HIGH.

This is a Kettle-specific convention (Aletheia / Wreon / Gemynd stack) --
override config.provenance_property_keys if the graph uses different names.
"""
from __future__ import annotations

from graph_diagnostics.core import (
    CheckResult, Finding, Severity, DiagnosticConfig, effective_labels,
)


def run(session, config: DiagnosticConfig) -> CheckResult:
    result = CheckResult(check="provenance")

    per_label_keys = config.provenance_property_keys_per_label or {}
    default_keys = config.provenance_property_keys

    if not per_label_keys and not default_keys:
        result.skipped = True
        result.skip_reason = (
            "No provenance properties configured. Set "
            "`provenance_property_keys` or `provenance_property_keys_per_label`."
        )
        return result

    labels = effective_labels(session, config)
    required_set = set(config.provenance_required_labels)

    # Per-label coverage.
    for label in labels:
        keys = per_label_keys.get(label, default_keys)
        if not keys:
            # No convention defined for this label -- skip rather than report
            # meaningless zero-coverage findings against irrelevant labels.
            continue

        total = session.run(
            f"MATCH (n:`{label}`) RETURN count(n) AS c"
        ).single()["c"]
        if total == 0:
            continue

        # How many nodes have each key?
        coverage = {}
        for key in keys:
            c = session.run(
                f"MATCH (n:`{label}`) WHERE n.`{key}` IS NOT NULL RETURN count(n) AS c"
            ).single()["c"]
            coverage[key] = {"count": c, "share": c / total}

        # How many nodes have ALL keys?
        all_present_expr = " AND ".join([f"n.`{k}` IS NOT NULL" for k in keys])
        fully_signed = session.run(
            f"MATCH (n:`{label}`) WHERE {all_present_expr} RETURN count(n) AS c"
        ).single()["c"]

        share_signed = fully_signed / total
        is_required = label in required_set

        if is_required and share_signed < 1.0:
            sev = Severity.HIGH
            msg = (
                f":{label} is a provenance-required label but only "
                f"{fully_signed}/{total} ({share_signed:.1%}) nodes carry all "
                f"of {keys}."
            )
        else:
            sev = Severity.INFO
            msg = (
                f":{label} provenance coverage: {fully_signed}/{total} "
                f"({share_signed:.1%}) nodes fully signed."
            )

        result.findings.append(Finding(
            check="provenance",
            code=f"coverage:{label}",
            severity=sev,
            message=msg,
            count=total - fully_signed,
            details={
                "label": label,
                "total_nodes": total,
                "fully_signed": fully_signed,
                "fully_signed_share": share_signed,
                "per_key": coverage,
                "required": is_required,
            },
        ))

        if is_required and share_signed < 1.0:
            missing_cypher = (
                f"// :{label} nodes missing one or more of "
                f"{keys}:\n"
                f"// MATCH (n:`{label}`) WHERE NOT ({all_present_expr}) "
                f"RETURN elementId(n), labels(n), keys(n)"
            )
            result.remediation.append(missing_cypher)

    # Cross-label: for GNN training, if provenance coverage is uneven across
    # labels, model features derived from provenance become confounded with
    # label identity. Flag uneven coverage (> 30% range between labels).
    per_label = [
        f for f in result.findings
        if f.code.startswith("coverage:")
    ]
    if len(per_label) >= 2:
        shares = [f.details["fully_signed_share"] for f in per_label]
        if max(shares) - min(shares) > 0.3:
            result.findings.append(Finding(
                check="provenance",
                code="uneven_coverage",
                severity=Severity.MEDIUM,
                message=(
                    f"Provenance coverage varies widely across labels "
                    f"(min={min(shares):.1%}, max={max(shares):.1%}). If "
                    f"provenance-derived features enter the GNN, they will "
                    f"leak label identity."
                ),
                details={"shares_by_label": {
                    f.details["label"]: f.details["fully_signed_share"]
                    for f in per_label
                }},
            ))

    return result

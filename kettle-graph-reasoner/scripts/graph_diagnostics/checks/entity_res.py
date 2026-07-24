"""
Entity resolution residue.

Catches near-duplicate entities that should probably be merged but weren't.
Three detection methods, configured per label via config.entity_res_rules:

    "exact"      -- group nodes where trim+casefold of the key is identical
    "normalized" -- NFKC + casefold + whitespace collapse + punctuation strip
    "jaro"       -- Jaro-Winkler similarity above a threshold (pairwise within
                    a blocking key; requires `rapidfuzz`).

Config example:

    entity_res_rules:
      Person:
        - {key: name, method: normalized}
        - {key: name, method: jaro, block_on: birth_year}

Blocking keys are important: Jaro-Winkler is O(n^2) and unusable without
them on anything but tiny graphs.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any

from graph_diagnostics.core import (
    CheckResult, Finding, Severity, DiagnosticConfig,
)


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def run(session, config: DiagnosticConfig) -> CheckResult:
    result = CheckResult(check="entity_res")

    if not config.entity_res_rules:
        result.skipped = True
        result.skip_reason = (
            "No entity_res_rules configured. Set config.entity_res_rules "
            "(label -> list of {key, method, [block_on]}) to enable this check."
        )
        return result

    for label, rules in config.entity_res_rules.items():
        for rule in rules:
            _apply_rule(session, config, result, label, rule)

    return result


def _apply_rule(session, config, result, label: str, rule: dict[str, Any]) -> None:
    method = rule.get("method", "exact")
    key = rule["key"]
    block_on = rule.get("block_on")

    if method == "exact":
        _check_exact(session, config, result, label, key, block_on)
    elif method == "normalized":
        _check_normalized(session, config, result, label, key, block_on)
    elif method == "jaro":
        _check_jaro(session, config, result, label, key, block_on)
    else:
        raise ValueError(f"Unknown entity_res method: {method!r}")


def _check_exact(session, config, result, label, key, block_on) -> None:
    # Trivial case: group by lowercase(trim(v)).
    block_expr = f", n.`{block_on}` AS block" if block_on else ""
    block_group = ", block" if block_on else ""
    q = f"""
    MATCH (n:`{label}`)
    WHERE n.`{key}` IS NOT NULL
    WITH toLower(trim(toString(n.`{key}`))) AS k{block_expr},
         collect({{id: elementId(n), raw: n.`{key}`}}) AS nodes
    WITH k{block_group}, nodes
    WHERE size(nodes) > 1
    RETURN k AS canonical{block_group}, nodes, size(nodes) AS c
    ORDER BY c DESC
    LIMIT $limit
    """
    groups = [dict(r) for r in session.run(q, limit=config.sample_limit * 3)]
    if not groups:
        return
    total_affected = sum(g["c"] for g in groups)
    result.findings.append(Finding(
        check="entity_res",
        code=f"exact_duplicates:{label}.{key}",
        severity=Severity.HIGH,
        message=(
            f":{label} has {len(groups)} groups of nodes sharing `{key}` "
            f"(case/whitespace-insensitive); {total_affected} nodes affected."
        ),
        count=total_affected,
        sample=groups[: config.sample_limit],
    ))
    result.remediation.append(
        f"// Entity-res: exact duplicates on {label}.{key}.\n"
        f"// Inspect groups, then use apoc.refactor.mergeNodes on each group.\n"
        f"// MATCH (n:`{label}`) WITH toLower(trim(toString(n.`{key}`))) AS k, "
        f"collect(n) AS ns WHERE size(ns) > 1 "
        f"CALL apoc.refactor.mergeNodes(ns, {{properties:'combine', mergeRels:true}}) "
        f"YIELD node RETURN node"
    )


def _check_normalized(session, config, result, label, key, block_on) -> None:
    # Can't do NFKC in pure Cypher, so pull candidates and group client-side.
    fetch_q = f"""
    MATCH (n:`{label}`)
    WHERE n.`{key}` IS NOT NULL
    RETURN elementId(n) AS id, n.`{key}` AS v
           {', n.`' + block_on + '` AS block' if block_on else ''}
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in session.run(fetch_q):
        norm = _normalize(str(row["v"]))
        block = row.get("block") if block_on else None
        buckets[(norm, block)].append({"id": row["id"], "raw": row["v"]})

    groups = [
        {"canonical": k[0], "block": k[1], "nodes": v, "c": len(v)}
        for k, v in buckets.items()
        if len(v) > 1
    ]
    if not groups:
        return
    groups.sort(key=lambda g: -g["c"])
    total = sum(g["c"] for g in groups)
    result.findings.append(Finding(
        check="entity_res",
        code=f"normalized_duplicates:{label}.{key}",
        severity=Severity.HIGH,
        message=(
            f":{label} has {len(groups)} groups of nodes whose `{key}` is identical "
            f"after Unicode NFKC + casefold + whitespace/punctuation normalization; "
            f"{total} nodes affected."
        ),
        count=total,
        sample=groups[: config.sample_limit],
    ))


def _check_jaro(session, config, result, label, key, block_on) -> None:
    try:
        from rapidfuzz.distance import JaroWinkler
    except ImportError:
        result.findings.append(Finding(
            check="entity_res",
            code=f"jaro_skipped:{label}.{key}",
            severity=Severity.INFO,
            message=(
                "rapidfuzz not installed; Jaro-Winkler check skipped. "
                "Install with: pip install rapidfuzz"
            ),
            count=0,
        ))
        return

    threshold = config.entity_res_jaro_threshold
    fetch_q = f"""
    MATCH (n:`{label}`)
    WHERE n.`{key}` IS NOT NULL
    RETURN elementId(n) AS id, n.`{key}` AS v
           {', n.`' + block_on + '` AS block' if block_on else ''}
    """
    # Bucket by block.
    by_block: dict[Any, list[dict]] = defaultdict(list)
    for row in session.run(fetch_q):
        block = row.get("block") if block_on else None
        by_block[block].append({"id": row["id"], "v": str(row["v"])})

    pairs: list[dict] = []
    for block, items in by_block.items():
        # Guard against O(n^2) blowup.
        if len(items) > 5000:
            # Skip this block and flag it.
            result.findings.append(Finding(
                check="entity_res",
                code=f"jaro_block_too_large:{label}.{key}",
                severity=Severity.LOW,
                message=(
                    f"Block {block!r} has {len(items)} candidates; skipping "
                    f"Jaro-Winkler (would be O(n^2)). Add a finer block_on key."
                ),
                count=len(items),
            ))
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                s = JaroWinkler.similarity(items[i]["v"], items[j]["v"])
                if s >= threshold:
                    pairs.append({
                        "a_id": items[i]["id"],
                        "a": items[i]["v"],
                        "b_id": items[j]["id"],
                        "b": items[j]["v"],
                        "similarity": round(s, 4),
                        "block": block,
                    })

    if not pairs:
        return
    pairs.sort(key=lambda p: -p["similarity"])
    result.findings.append(Finding(
        check="entity_res",
        code=f"jaro_near_duplicates:{label}.{key}",
        severity=Severity.MEDIUM,
        message=(
            f":{label}.{key} -- {len(pairs)} candidate near-duplicate pairs "
            f"with Jaro-Winkler >= {threshold}."
        ),
        count=len(pairs),
        sample=pairs[: config.sample_limit],
    ))
    # No automatic remediation -- reviewer must decide which pairs merge.
    result.remediation.append(
        f"// Jaro-Winkler near-duplicates on {label}.{key} require manual review.\n"
        f"// See findings JSON for the candidate pairs."
    )


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s

"""
Core types and the top-level runner.

A check module exposes a single function with the signature:

    def run(session, config) -> CheckResult

where CheckResult carries a list of Findings and a list of remediation
Cypher statements (as strings). Everything else in this module is plumbing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

from dotenv import find_dotenv, load_dotenv
from neo4j import GraphDatabase, Session


# ---------------------------------------------------------------------------
# Finding / result types
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    INFO = "info"        # observation, no action required
    LOW = "low"          # cosmetic or advisory
    MEDIUM = "medium"    # should fix before training
    HIGH = "high"        # will materially affect GNN results
    CRITICAL = "critical"  # blocks training or corrupts splits


@dataclass
class Finding:
    check: str                           # module name, e.g. "structural"
    code: str                            # short identifier, e.g. "orphan_nodes"
    severity: Severity
    message: str                         # one-line human summary
    count: int = 0                       # how many items are implicated
    sample: list[Any] = field(default_factory=list)  # up to N example ids
    details: dict[str, Any] = field(default_factory=dict)  # arbitrary metadata


@dataclass
class CheckResult:
    check: str
    findings: list[Finding] = field(default_factory=list)
    remediation: list[str] = field(default_factory=list)  # Cypher statements
    skipped: bool = False
    skip_reason: str | None = None
    duration_sec: float = 0.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticConfig:
    """Runtime configuration for the diagnostic suite.

    All fields have sensible defaults so the suite is runnable with zero
    configuration; override per-project via YAML.
    """

    # --- scope ---
    # If empty, auto-discover via db.labels() / db.relationshipTypes().
    node_labels: list[str] = field(default_factory=list)
    relationship_types: list[str] = field(default_factory=list)

    # --- structural ---
    allow_self_loops: bool = False
    allow_multi_edges: bool = False      # multiple rels of same type between same pair

    # --- schema ---
    # Map of label -> list of required property keys.
    required_properties: dict[str, list[str]] = field(default_factory=dict)
    # Map of label -> property key that should uniquely identify the node.
    unique_keys: dict[str, str] = field(default_factory=dict)

    # --- entity resolution ---
    # Label -> (property_key, method). method in {"exact", "normalized", "jaro"}
    # "exact" -- same value after case-fold + strip
    # "normalized" -- Unicode NFKC + case-fold + whitespace collapse
    # "jaro" -- Jaro-Winkler > threshold (requires rapidfuzz, see extras)
    entity_res_rules: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    entity_res_jaro_threshold: float = 0.92

    # --- topology ---
    # The topology check pulls the lifecycle-clean graph ONCE into a numpy
    # cache (graphcache.GraphCache) and derives every per-scope metric from
    # masks over it. Gromov delta is a sampled, advisory 3-bucket verdict.
    gromov_sample_size: int = 1000       # number of 4-tuples for delta sampling
    gromov_max_nodes: int = 20000        # giant-component size that triggers a
                                         # connected snowball sub-sample for
                                         # delta (no longer a skip gate)
    gromov_sample_max_nodes: int = 6000  # node cap for the snowball sub-sample
    gromov_landmarks: int = 200          # # of BFS source nodes; all pairwise
                                         # distances among them feed the 4-tuples
    gromov_timeout_sec: float = 120.0    # wall-clock limit per Gromov scope
    temporal_max_hops: int = 3           # hop radius for Year-reachability
                                         # precompute (replaces per-edge [*1..3])
    degree_histogram_bins: int = 30
    min_component_size_of_interest: int = 2  # smaller than this = isolated

    # --- splits ---
    split_task: str = "link_prediction"  # or "node_classification", "entity_res"
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int = 42

    # --- provenance ---
    provenance_property_keys: list[str] = field(
        default_factory=lambda: ["albc_barcode", "content_hash", "signature"]
    )
    # Per-label override for provenance properties. If a label appears here,
    # its entry replaces `provenance_property_keys` for that label only.
    provenance_property_keys_per_label: dict[str, list[str]] = field(default_factory=dict)
    # Labels that MUST carry provenance. Empty => all labels advisory-only.
    provenance_required_labels: list[str] = field(default_factory=list)

    # --- lifecycle (quarantine / soft-delete / access control) -------------
    # Populated from YAML's `lifecycle:` block. Empty dict = no filtering.
    lifecycle: dict[str, Any] = field(default_factory=dict)

    # --- layer invariants --------------------------------------------------
    # Map of invariant_name -> {label, must_satisfy (Cypher predicate), severity}.
    # Evaluated by the `layer_invariants` check.
    layer_invariants: dict[str, dict[str, Any]] = field(default_factory=dict)

    # --- subgraph definitions ----------------------------------------------
    # Map of subgraph_name -> extraction spec. Consumed by topology (for
    # per-subgraph Gromov delta) and by the subgraph extractor CLI.
    subgraphs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # --- general ---
    sample_limit: int = 20               # max example ids per finding
    enabled_checks: list[str] = field(
        default_factory=lambda: [
            "lifecycle", "structural", "schema", "layer_invariants",
            "entity_res", "topology", "splits", "provenance",
        ]
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DiagnosticConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiagnosticConfig":
        return cls(**data)


# ---------------------------------------------------------------------------
# Driver plumbing
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_ENV_PATH = find_dotenv(filename=".env", usecwd=False, raise_error_if_not_found=False)
if not _ENV_PATH:
    for _parent in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
        _candidate = _parent / ".env"
        if _candidate.is_file():
            _ENV_PATH = str(_candidate)
            break
load_dotenv(_ENV_PATH or None)


def _driver():
    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    return GraphDatabase.driver(
        uri,
        auth=(username, password),
        notifications_min_severity="OFF",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# Import check modules lazily to avoid circular imports; register here.
def _registry() -> dict[str, Callable[[Session, DiagnosticConfig], CheckResult]]:
    from graph_diagnostics.checks import (
        lifecycle, structural, schema, layer_invariants, entity_res,
        topology, splits, provenance,
    )
    return {
        "lifecycle": lifecycle.run,
        "structural": structural.run,
        "schema": schema.run,
        "layer_invariants": layer_invariants.run,
        "entity_res": entity_res.run,
        "topology": topology.run,
        "splits": splits.run,
        "provenance": provenance.run,
    }


def run_check(
    name: str,
    session: Session,
    config: DiagnosticConfig | None = None,
) -> CheckResult:
    """Run a single named check against an already-open session."""
    import time
    config = config or DiagnosticConfig()
    registry = _registry()
    if name not in registry:
        raise KeyError(f"Unknown check {name!r}. Available: {sorted(registry)}")
    t0 = time.perf_counter()
    try:
        result = registry[name](session, config)
    except Exception as exc:
        result = CheckResult(
            check=name,
            skipped=True,
            skip_reason=f"{type(exc).__name__}: {exc}",
        )
    result.duration_sec = time.perf_counter() - t0
    return result


def run_all(
    output_dir: str | Path = "reports",
    config: DiagnosticConfig | None = None,
    config_path: str | Path | None = None,
) -> dict[str, CheckResult]:
    """Run the enabled checks and write report + remediation to disk."""
    if config is None and config_path is not None:
        config = DiagnosticConfig.from_yaml(config_path)
    config = config or DiagnosticConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, CheckResult] = {}

    driver = _driver()
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            for name in config.enabled_checks:
                results[name] = run_check(name, session, config)
    finally:
        driver.close()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = output_dir / f"report_{ts}.md"
    cypher_path = output_dir / f"remediate_{ts}.cypher"
    json_path = output_dir / f"findings_{ts}.json"

    _write_report(report_path, results, config)
    _write_cypher(cypher_path, results)
    _write_json(json_path, results)

    print(f"Report:     {report_path}")
    print(f"Cypher:     {cypher_path}")
    print(f"Findings:   {json_path}")
    return results


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------

def _severity_rank(sev: Severity) -> int:
    return {
        Severity.INFO: 0,
        Severity.LOW: 1,
        Severity.MEDIUM: 2,
        Severity.HIGH: 3,
        Severity.CRITICAL: 4,
    }[sev]


def _write_report(
    path: Path,
    results: dict[str, CheckResult],
    config: DiagnosticConfig,
) -> None:
    lines: list[str] = []
    ts = datetime.now(timezone.utc).isoformat()
    lines.append(f"# Graph Diagnostic Report")
    lines.append(f"_Generated {ts}_\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Check | Findings | Highest severity | Status | Duration (s) |")
    lines.append("|---|---|---|---|---|")
    for name, res in results.items():
        if res.skipped:
            lines.append(f"| {name} | — | — | SKIPPED: {res.skip_reason} | {res.duration_sec:.2f} |")
            continue
        if not res.findings:
            lines.append(f"| {name} | 0 | — | clean | {res.duration_sec:.2f} |")
            continue
        top = max(res.findings, key=lambda f: _severity_rank(f.severity))
        lines.append(
            f"| {name} | {len(res.findings)} | {top.severity.value} "
            f"| see below | {res.duration_sec:.2f} |"
        )
    lines.append("")

    # Detail sections
    for name, res in results.items():
        lines.append(f"## {name}\n")
        if res.skipped:
            lines.append(f"**Skipped.** {res.skip_reason}\n")
            continue
        if not res.findings:
            lines.append("No findings.\n")
            continue
        for f in sorted(res.findings, key=lambda x: -_severity_rank(x.severity)):
            lines.append(f"### `{f.code}` — **{f.severity.value}** ({f.count} items)")
            lines.append(f"{f.message}\n")
            if f.sample:
                lines.append("Sample:")
                lines.append("```")
                for s in f.sample[: config.sample_limit]:
                    lines.append(str(s))
                lines.append("```\n")
            if f.details:
                lines.append("Details:")
                lines.append("```json")
                lines.append(json.dumps(f.details, indent=2, default=str))
                lines.append("```\n")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_cypher(path: Path, results: dict[str, CheckResult]) -> None:
    """Emit an idempotent, commented remediation script."""
    lines: list[str] = []
    lines.append("// ----------------------------------------------------------")
    lines.append("// Graph Diagnostic Remediation Script")
    lines.append(f"// Generated {datetime.now(timezone.utc).isoformat()}")
    lines.append("// Review every statement before executing.")
    lines.append("// Statements are ordered check-by-check; run in order.")
    lines.append("// ----------------------------------------------------------\n")
    for name, res in results.items():
        if not res.remediation:
            continue
        lines.append(f"// ===== {name} =====")
        for stmt in res.remediation:
            # Guarantee each statement is terminated.
            s = stmt.strip()
            if not s.endswith(";"):
                s = s + ";"
            lines.append(s)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, results: dict[str, CheckResult]) -> None:
    serializable = {
        name: {
            "check": res.check,
            "skipped": res.skipped,
            "skip_reason": res.skip_reason,
            "duration_sec": res.duration_sec,
            "remediation": res.remediation,
            "findings": [
                {**asdict(f), "severity": f.severity.value} for f in res.findings
            ],
        }
        for name, res in results.items()
    }
    path.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Small helpers used by check modules
# ---------------------------------------------------------------------------

def discover_labels(session: Session) -> list[str]:
    return [r["label"] for r in session.run("CALL db.labels() YIELD label RETURN label")]


def discover_rel_types(session: Session) -> list[str]:
    return [
        r["relationshipType"]
        for r in session.run(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )
    ]


def effective_labels(session: Session, config: DiagnosticConfig) -> list[str]:
    return config.node_labels or discover_labels(session)


def effective_rel_types(session: Session, config: DiagnosticConfig) -> list[str]:
    return config.relationship_types or discover_rel_types(session)


def sample_ids(rows: Iterable[dict], key: str, limit: int) -> list[Any]:
    out = []
    for row in rows:
        if len(out) >= limit:
            break
        out.append(row.get(key))
    return out


def lifecycle_predicate(config: DiagnosticConfig, var: str = "n") -> str:
    """Render a Cypher predicate that excludes lifecycle-filtered nodes.

    Returns a string suitable for embedding inside a WHERE clause. If no
    lifecycle config is set, returns "true" so the predicate is a no-op
    and can be safely AND-ed into any query.

    Example:
        pred = lifecycle_predicate(cfg, "n")
        cypher = f"MATCH (n:Mention) WHERE {pred} RETURN count(n)"
    """
    lc = config.lifecycle or {}
    if not lc:
        return "true"

    clauses: list[str] = []

    if lc.get("exclude_if_deleted"):
        prop = lc.get("deleted_property", "deleted_at")
        clauses.append(f"{var}.`{prop}` IS NULL")

    if lc.get("exclude_if_quarantined"):
        prop = lc.get("quarantine_property", "quarantine_status")
        excluded = lc.get("quarantine_excluded_values", ["quarantined"])
        vals = ", ".join(repr(v) for v in excluded)
        clauses.append(
            f"({var}.`{prop}` IS NULL OR NOT {var}.`{prop}` IN [{vals}])"
        )

    if lc.get("exclude_restricted_access"):
        prop = lc.get("access_property", "access_level")
        excluded = lc.get("access_excluded_values", ["restricted"])
        vals = ", ".join(repr(v) for v in excluded)
        clauses.append(
            f"({var}.`{prop}` IS NULL OR NOT {var}.`{prop}` IN [{vals}])"
        )

    return " AND ".join(clauses) if clauses else "true"

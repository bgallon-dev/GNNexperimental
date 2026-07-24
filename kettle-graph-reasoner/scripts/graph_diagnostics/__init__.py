"""
graph_diagnostics
=================

Modular diagnostic suite for preparing a Neo4j graph for GNN training.

Design
------
Six independent check modules, each producing a Finding list and a Cypher
remediation list. The runner composes them and emits two artifacts per run:

    <output_dir>/report_<timestamp>.md       -- human-readable findings
    <output_dir>/remediate_<timestamp>.cypher  -- executable Cypher

Nothing is mutated against the live graph. Review the Cypher before running.

Check modules
-------------
    structural       -- orphans, self-loops, duplicate nodes/edges, dangling refs
    schema           -- label consistency, missing required properties, type drift
    entity_res       -- near-duplicate entities via property similarity
    topology         -- degree dist, components, density, Gromov delta (sampled)
    splits           -- train/val/test feasibility, leakage risk
    provenance       -- ALBC / signature coverage

Usage
-----
    from graph_diagnostics import run_all
    run_all(output_dir="reports/", config_path="diag_config.yaml")

Or from the CLI:
    python -m graph_diagnostics run --config diag_config.yaml --output reports/
"""

from graph_diagnostics.core import (
    Finding,
    Severity,
    CheckResult,
    DiagnosticConfig,
    run_all,
    run_check,
)

__all__ = [
    "Finding",
    "Severity",
    "CheckResult",
    "DiagnosticConfig",
    "run_all",
    "run_check",
]

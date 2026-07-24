"""CLI: `python -m graph_diagnostics run --config diag.yaml --output reports/`"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neo4j.exceptions import ServiceUnavailable

from graph_diagnostics.core import DiagnosticConfig, run_all, run_check, _driver


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="graph_diagnostics",
        description="Diagnostic suite for preparing a Neo4j graph for GNN training.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run all enabled checks")
    p_run.add_argument("--config", type=Path, default=None,
                       help="Path to YAML config (optional)")
    p_run.add_argument("--output", type=Path, default=Path("reports"),
                       help="Output directory for report / cypher / json")
    p_run.set_defaults(func=_cmd_run)

    p_one = sub.add_parser("check", help="Run a single named check")
    p_one.add_argument("name", choices=[
        "lifecycle", "structural", "schema", "layer_invariants",
        "entity_res", "topology", "splits", "provenance",
    ])
    p_one.add_argument("--config", type=Path, default=None)
    p_one.set_defaults(func=_cmd_check)

    p_cfg = sub.add_parser("init-config",
                           help="Write a starter YAML config with commentary")
    p_cfg.add_argument("path", type=Path)
    p_cfg.set_defaults(func=_cmd_init_config)

    p_sg = sub.add_parser("subgraph",
                          help="Inspect or export a named subgraph")
    p_sg.add_argument("action", choices=("list", "stats", "export"))
    p_sg.add_argument("--name", help="Subgraph name (for stats/export)")
    p_sg.add_argument("--output", type=Path,
                      help="Output JSONL path (for export)")
    p_sg.add_argument("--config", type=Path, required=True,
                      help="YAML config defining the subgraph")
    p_sg.set_defaults(func=_cmd_subgraph)

    args = parser.parse_args()
    try:
        return args.func(args)
    except ServiceUnavailable as exc:
        uri = __import__("os").getenv("NEO4J_URI", "bolt://localhost:7687")
        print(
            f"error: cannot reach Neo4j at {uri}\n"
            f"       Make sure the database is running and NEO4J_URI in your .env is correct.\n"
            f"       ({exc})",
            file=sys.stderr,
        )
        return 1


def _cmd_run(args) -> int:
    run_all(output_dir=args.output, config_path=args.config)
    return 0


def _cmd_check(args) -> int:
    cfg = DiagnosticConfig.from_yaml(args.config) if args.config else DiagnosticConfig()
    driver = _driver()
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            result = run_check(args.name, session, cfg)
    finally:
        driver.close()

    print(f"== {result.check} ==")
    if result.skipped:
        print(f"SKIPPED: {result.skip_reason}")
        return 0
    if not result.findings:
        print("Clean.")
        return 0
    for f in result.findings:
        print(f"[{f.severity.value}] {f.code}: {f.message} (count={f.count})")
    print(f"\n{len(result.remediation)} remediation statements available.")
    return 0


_STARTER_YAML = '''\
# graph_diagnostics configuration
# Paths are relative to the working directory at run time.

# --- scope ------------------------------------------------------------------
# Leave empty to auto-discover via CALL db.labels() / db.relationshipTypes().
node_labels: []
relationship_types: []

# --- structural -------------------------------------------------------------
allow_self_loops: false
allow_multi_edges: false

# --- schema -----------------------------------------------------------------
# Label -> list of property keys that must be present on every node.
required_properties:
  # Person: [name]
  # Document: [source, ingested_at]

# Label -> property key that should be unique across nodes of that label.
unique_keys:
  # Person: uuid
  # Document: doc_id

# --- entity resolution ------------------------------------------------------
# Label -> list of detection rules.
# Methods: "exact", "normalized", "jaro" (requires rapidfuzz).
# block_on narrows Jaro candidates to pairs sharing that property.
entity_res_rules:
  # Person:
  #   - {key: name, method: normalized}
  #   - {key: name, method: jaro, block_on: birth_year}

entity_res_jaro_threshold: 0.92

# --- topology ---------------------------------------------------------------
gromov_sample_size: 1000
gromov_max_nodes: 20000
degree_histogram_bins: 30
min_component_size_of_interest: 2

# --- splits -----------------------------------------------------------------
# One of: link_prediction, node_classification, entity_res, multi
split_task: multi
split_ratios: [0.8, 0.1, 0.1]
split_seed: 42

# --- provenance (Kettle / Aletheia convention) ------------------------------
provenance_property_keys: [albc_barcode, content_hash, signature]
# Labels for which missing provenance is a HIGH finding (not just advisory).
provenance_required_labels: []

# --- general ----------------------------------------------------------------
sample_limit: 20
enabled_checks:
  - structural
  - schema
  - entity_res
  - topology
  - splits
  - provenance
'''


def _cmd_init_config(args) -> int:
    args.path.parent.mkdir(parents=True, exist_ok=True)
    args.path.write_text(_STARTER_YAML, encoding="utf-8")
    print(f"Wrote starter config to {args.path}")
    return 0


def _cmd_subgraph(args) -> int:
    cfg = DiagnosticConfig.from_yaml(args.config)
    if not cfg.subgraphs:
        print("No subgraphs defined in config.", file=sys.stderr)
        return 2

    if args.action == "list":
        for name, spec in cfg.subgraphs.items():
            print(f"{name}: {spec.get('description', '(no description)')}")
        return 0

    if not args.name:
        print("--name is required for stats/export", file=sys.stderr)
        return 2
    if args.name not in cfg.subgraphs:
        print(f"Unknown subgraph {args.name!r}. Available: "
              f"{sorted(cfg.subgraphs)}", file=sys.stderr)
        return 2

    from graph_diagnostics.subgraphs.extract import (
        extract_stats, extract_to_jsonl,
    )

    spec = cfg.subgraphs[args.name]
    driver = _driver()
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            if args.action == "stats":
                stats = extract_stats(session, spec, cfg, args.name)
                print(f"Subgraph: {stats.name}")
                print(f"Description: {spec.get('description', '')}")
                print(f"Nodes: {stats.nodes:,}")
                print(f"Edges: {stats.edges:,}")
                print(f"Label distribution (top 10):")
                for lbl, c in list(stats.labels_seen.items())[:10]:
                    print(f"  {lbl}: {c:,}")
                print(f"Relationship types (top 10):")
                for rt, c in list(stats.rel_types_seen.items())[:10]:
                    print(f"  {rt}: {c:,}")
                return 0

            if args.action == "export":
                if not args.output:
                    print("--output is required for export", file=sys.stderr)
                    return 2
                n_nodes, n_edges = extract_to_jsonl(
                    session, spec, cfg, args.output
                )
                print(f"Wrote {n_nodes:,} nodes + {n_edges:,} edges "
                      f"to {args.output}")
                return 0
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

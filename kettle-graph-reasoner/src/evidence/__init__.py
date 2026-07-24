"""KGR Evidence Workspace — deterministic evidence-packet engine.

Given a research question and human-confirmed graph anchors, this package
localizes a bounded Neo4j candidate space, orders domain nodes with the frozen
KGR encoder (or BFS, per the family/lane strategy table), attaches exact typed
paths and provenance closure, and compiles immutable JSON + Markdown research
artifacts.

Plan of record: Docs/EVIDENCE_WORKSPACE_PLAN.md (Proposed, 2026-07-11).
Status: scaffold only — T0 (foundation regression gate) must land before
implementation work begins here.

Boundaries carried from the plan:
- Neo4j owns exact topology; KGR only orders an already-localized neighborhood.
- The frozen encoder is never run over an unvalidated ``end_to_end`` projection.
- Source Neo4j is read-only; no endpoint accepts arbitrary Cypher.
- The frozen KGR release (frozen/kgr-v1.0-2026-07-07) remains unchanged.
"""

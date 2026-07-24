# Evidence Workspace (`src/evidence/`)

The answer-completeness layer is specified in
[Docs/EVIDENCE_COVERAGE_REASONER_PLAN.md](../../Docs/EVIDENCE_COVERAGE_REASONER_PLAN.md)
(In progress, 2026-07-15). It is Evidence Workspace T7. The first slice —
T7.0 contracts + the seven frozen answer-shape descriptors + canonical
`coverage.json`, plus the smallest honest T7.1 deterministic slot/gate/verdict
kernel — is implemented under `coverage/` (no learned scorer, no semantic
extraction, no database).

Project home for the deterministic evidence-packet engine. The plan of record
is [Docs/EVIDENCE_WORKSPACE_PLAN.md](../../Docs/EVIDENCE_WORKSPACE_PLAN.md)
(Proposed, 2026-07-11) — read it before adding code here.

## Status

| Track | Scope | Status |
|-------|-------|--------|
| T0 | Foundation regression gate (service hardening) | **PASSED 2026-07-11, fully closed** — suite green; verify P0 PASS; P1 live parity PASS bit-exact (0.0 diff, 3/3 graphs) |
| T1 | Contracts, storage, deterministic runtime, anchor resolver | **SHIPPED 2026-07-11** — gate passed (3 builds, 2 processes, byte-identical; cross-hashseed) |
| T2 | Fixture corpus (historian labels; parallel with T1) | **Tranche A authored 2026-07-12** — 6 dev + 6 pilot, balanced 3/family, corpus-probed; essential-node grading pending |
| T3 | `EvidenceGraphSource` candidate projection + graph obligations | **SHIPPED 2026-07-11** — gate passed hermetic + live (400 nodes 0.5s, closure 1.7s, 8/8 checks; `scripts/evidence_live_smoke.py`) |
| T4 | Node strategy evaluation (family/lane cells) + exact-path ordering | blocked on T2 grading (BFS ships as default meanwhile) |
| T5 | Core packet compiler, CLI, localhost API | **SHIPPED 2026-07-12** — live E2E on dev/q01 (12 core items, validate PASS, reuse + diff verified); `serve` HTTP adapter live-smoked (health/list/files/validate/404) |
| T6 | Source-critical enrichment + optional interpreter | **interpreter SHIPPED 2026-07-12** — packet-only, citation-gated, quarantine on failure; live memo via local LLM. Enrichment awaits curated sidecars |
| T7 | Evidence Coverage Reasoner (answer shapes, slot coverage, abstention, backend-neutral replay) | **T7.0 + smallest T7.1 IMPLEMENTED 2026-07-15** (`coverage/`) — contracts, seven answer shapes, deterministic slot/gate/verdict kernel, canonical `coverage.json` compile; no learned model until historian labels exist. T7.2–T7.5 pending |
| — | Labeling workbench (T2 support) | **SHIPPED 2026-07-12** — `workbench generate/ingest`; all 12 worksheets built (~1,940 rows, seeded-random order) |

## Module layout

T7 status: **T7.0 COMPLETE; T7.1 IN PROGRESS 2026-07-15** — contracts,
shapes, deterministic gates/verdicts, research frontiers, and portable compile
are implemented. Explicit-family selection, rule/lexical nomination, and real
stress-packet sidecars remain; no learned model trains until historian labels
exist.

Modules land with their tracks; do not pre-create empty files.

- `contracts.py` — dataclasses (`ResearchQuestion`, `AnchorResolution`, `CandidateBundle`, `RankedCandidate`, `AnnotationRecord`, `BuildManifest`), validators, `schema_version` (T1 — **shipped**)
- `canonical.py` — canonical UTF-8 JSON serializer, micro-score quantization, determinism-runtime setup (T1 — **shipped**)
- `store.py` — immutable revision store, atomic writer, workspace lock (T1 — **shipped**)
- `resolver.py` — deterministic entity/date anchor resolver (T1 — **shipped**; live-DB smoke pending, tested against fake session)
- `projection.py` — `EvidenceGraphSource`, `default_v1` traversal profile (T3)
- `ranking.py` — family/lane strategy table, KGR/BFS/lexical lanes, exact typed-path ordering (T4)
- `compiler.py` — deterministic packet compiler and Markdown renderer (T5)
- `api.py` — localhost HTTP adapter under `/api/v1` (T5)
- `interpret.py` — optional packet-only interpreter (T6)
- `offline.py` — database-free compilation from persisted question/candidate
  artifacts; BFS/lexical replay plus validated supplied KGR rankings (shipped)
- `coverage/` — answer-shape contracts (`contracts.py`), the strict registry
  loader (`shapes.py`) and seven versioned JSON descriptors (`shapes/*.json`),
  the deterministic slot/gate/verdict kernel (`evaluator.py`), and canonical
  offline `coverage.json` compilation (`compile.py`) (T7.0 + smallest T7.1 —
  **implemented**; runs on portable artifacts, never imports the Neo4j backend)

## Fixed surfaces

- CLI: `python -m src.evidence {anchors,packet,serve} ...` (skeleton in
  `__main__.py`; commands exit 2 until their track lands).
- Runtime artifacts go to `research_workspace/` (gitignored, never committed).
- Evaluation runs go to `runs/evidence_workspace_eval/`; final verdicts go to
  `Docs/EVIDENCE_WORKSPACE_FINDINGS.md`.

## Non-negotiables (from the plan)

- Neo4j owns exact topology; KGR only orders an already-localized neighborhood.
- Never run the frozen encoder over an unvalidated `end_to_end` projection.
- Anchor confirmation is mandatory; unconfirmed/stale anchors fail explicitly.
- Source Neo4j stays read-only; no endpoint accepts arbitrary Cypher.
- The frozen KGR release (`frozen/kgr-v1.0-2026-07-07/`) is never modified.
- The optional interpreter sees only `packet.json`.
- T7 consumes portable artifacts and must not import the Neo4j backend; learned
  scores can nominate assignments but cannot override deterministic hard gates.

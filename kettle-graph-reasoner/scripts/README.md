# graph_diagnostics

Modular, review-before-apply diagnostic suite for preparing a Neo4j graph
for GNN training. Schema-aware via YAML config, layer-aware for three-layer
graphs (document / extraction / domain), and geometry-aware for choosing
between Euclidean and hyperbolic GNN architectures.

## Install

```bash
pip install -r requirements.txt
```

`.env` in the project root:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=replace-me
```

## Run

For the Kettle / Spokane-corpus graph, use the pre-built config:

```bash
python -m graph_diagnostics run --config kettle_config.yaml --output reports/
```

Or generate a generic starter config:

```bash
python -m graph_diagnostics init-config diag.yaml
# edit diag.yaml
python -m graph_diagnostics run --config diag.yaml --output reports/
```

Run a single check for rapid iteration:

```bash
python -m graph_diagnostics check lifecycle --config kettle_config.yaml
python -m graph_diagnostics check topology  --config kettle_config.yaml
```

## Outputs

Each `run` writes three files to the output directory, timestamped:

| File | Purpose |
|---|---|
| `report_<ts>.md` | Human-readable findings, sorted by severity |
| `remediate_<ts>.cypher` | Executable Cypher, grouped by check; review before running |
| `findings_<ts>.json` | Machine-readable findings for downstream tooling |

Structural remediation (self-loops, duplicate rels, stub nodes) is safe to
apply as-is. Schema, entity-resolution, and layer-invariant remediation is
commented out in the output because the right fix is schema-specific.

## Subgraph extraction (KGR staging workflow)

The config defines named subgraphs corresponding to the four KGR training
targets. Inspect them before extracting:

```bash
python -m graph_diagnostics subgraph list --config kettle_config.yaml
python -m graph_diagnostics subgraph stats --config kettle_config.yaml \
    --name mention_entity_bipartite
```

Export a subgraph to JSONL for the GNN loader:

```bash
python -m graph_diagnostics subgraph export --config kettle_config.yaml \
    --name mention_entity_bipartite --output data/mei_bipartite.jsonl
```

The topology check automatically runs Gromov δ on every named subgraph, so
one `run` command gives you the curvature comparison across all four staging
targets in a single pass.

## The eight checks

1. **lifecycle** — quarantine / soft-delete / access-level filtering. Runs
   first so downstream checks exclude already-flagged nodes. Flags
   supervision edges pointing at excluded nodes as a HIGH finding (the
   real bug class: stale `REFERS_TO` edges to quarantined entities).
2. **structural** — orphan nodes, self-loops, duplicate relationships, stub
   nodes (no labels / no props).
3. **schema** — required properties, unique-key violations, per-property
   value-type drift, inconsistent secondary label sets.
4. **layer_invariants** — Cypher-predicate invariants per label: Paragraph
   must have a Section parent, Mention must live inside a Paragraph, Claim
   should have evidence, Observation should be located, Measurement should
   carry a unit. Configurable in YAML.
5. **entity_res** — near-duplicate entities via exact / normalized /
   Jaro-Winkler matching with optional blocking keys.
6. **topology** — degree distribution, hub concentration, weakly-connected
   components, and sampled Gromov δ-hyperbolicity **per named subgraph**
   with δ/diameter verdict.
7. **splits** — per-task train/val/test feasibility: link prediction, node
   classification, generic entity resolution, and the Kettle-specific
   `er_mention_entity` task (`POSSIBLY_REFERS_TO` → `REFERS_TO` promotion)
   with entity-cluster leakage checks and cluster-size distribution.
8. **provenance** — per-label provenance coverage. Different conventions
   per layer: OCR provenance for Document/Page (`file_hash`, `ocr_engine`,
   `ocr_version`), model/run provenance for Mention/Claim (`ner_model`,
   `run_id`, `extraction_confidence`), structural provenance for domain
   entities (SOURCED_FROM edges rather than properties).

## KGR staging methodology

The suite is built around a four-stage methodology for choosing the KGR
training target, rather than a parallel bake-off:

| Stage | Subgraph | Question answered |
|---|---|---|
| 1 | `mention_entity_bipartite` | Does hyperbolic GNN beat Euclidean baseline on the ER task? |
| 2 | `domain_only` | Does the advantage scale to denser semantic relations? |
| 3 | `end_to_end` | Does the architecture degrade gracefully on heterogeneous graphs? |
| 4 | `temporal_pre_1935` / `temporal_post_1935` | What is the time-generalization gap? |

Each stage answers a question the previous one cannot, and each stage's
failure mode is informative. A *rising* δ/diameter at stage 2 would itself
be a publishable finding — "hyperbolic GNN helps on entity resolution
specifically, not graph learning generally" is a more defensible claim than
"hyperbolic GNN is better."

Running the topology check with these subgraphs defined gives you all four
δ measurements *before you train anything*. That is the right order of
operations: measure the geometry, let the geometry pick your primary
experiment, then run the ablations that test whether the curvature story
held up.

## Design notes

- **No mutation.** Every check is read-only. Remediation is emitted as
  Cypher text, commented for review.
- **Lifecycle-first.** The lifecycle predicate is applied consistently
  across structural, schema, layer_invariants, topology, and splits
  checks, so a quarantined node is not re-reported as every other kind of
  bug.
- **Per-layer conventions.** Provenance properties, required properties,
  and unique keys are configured per label. A Document's provenance
  convention (`file_hash`, `archive_ref`) differs from a Mention's
  (`ner_model`, `run_id`, `extraction_confidence`).
- **Subgraph-parametric topology.** Gromov δ is close to meaningless on a
  heterogeneous graph at scale; per-subgraph δ is the useful measurement
  for training-target selection.
- **Degrades gracefully.** GDS → APOC → pure-Cypher → client-side
  fallbacks. Missing `networkx` or `rapidfuzz` disables specific checks
  without failing the run.

## Extending

Add a new check as `graph_diagnostics/checks/<name>.py` exposing
`def run(session, config) -> CheckResult`, then register it in
`core._registry()` and add it to `DiagnosticConfig.enabled_checks`.

Add a new subgraph by editing the `subgraphs:` block in the YAML. No code
changes are needed as long as the include/exclude/temporal filter schema
suffices.

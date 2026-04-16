# `src/models/` — v1 model family

This folder holds the **first-generation** experimental model plus both Euclidean baselines. Its counterpart is [../modelsv2/](../modelsv2/CLAUDE.md), which contains a v2 rewrite of the hyperbolic model only.

## Contents

- [hyperbolic_gnn.py](hyperbolic_gnn.py) — experimental `KettleGraphReasoner` (v1)
- [euclidean_baseline.py](euclidean_baseline.py) — GCN/GAT control
- [euclidean_plus_baseline.py](euclidean_plus_baseline.py) — Euclidean + edge-typed attention control
- [layers/](layers/) — shared primitives (poincaré ops, hyperbolic message passing, edge-typed attention, schema encoder)

## What makes v1 distinctive

Depth aggregation is **concat-based**, not attention-based.

- `concat_depth=False` (default): scoring heads see only the final round's tangent-at-origin view — `logmap0(h_L)`.
- `concat_depth=True`: scoring heads see `[logmap0(h_1) || … || logmap0(h_L)]` — HMT-style depth concatenation. Diagnostic knob: if enabling this lifts multi-hop tasks, depth info is being collapsed by the final-round head and justifies moving to attention-based depth aggregation (v2).
- `log_depth=True` is force-enabled whenever `concat_depth` is on, so per-round ball-space embeddings are returned in `KGROutput.per_round_embeddings`.

Other v1-only specifics:
- `tangent_scale_init` is a constructor argument (default 0.15). v2 hardcodes 0.1.
- When `hierarchy_subspace_dim > 0` **and** `concat_depth=True`, the hierarchy/proximity split is applied **per layer** through the concat (see [hyperbolic_gnn.py:285-296](hyperbolic_gnn.py#L285-L296)).
- Scoring-head input dims scale with `depth_mul = num_layers if concat_depth else 1`.

## Baselines live here, not in v2

The Euclidean and Euclidean+edge-typed baselines are the *control condition* for the research hypothesis. They are deliberately kept alongside v1 and were not duplicated into `modelsv2/` — v2 is a hyperbolic-only iteration on the experimental arm.

## When to touch this folder

- Reproducing original experimental results or running the baselines.
- Ablations against the v2 depth-attention mechanism (v1 with `concat_depth=True` is the natural comparator).
- Do NOT port v2 features back here without a clear reason — v1 is the stable reference.

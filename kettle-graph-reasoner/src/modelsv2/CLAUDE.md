# `src/modelsv2/` — v2 hyperbolic model

Second-generation rewrite of the experimental model. Hyperbolic-only: no baselines live here (those stay in [../models/](../models/CLAUDE.md)). The `layers/` subfolder is **byte-identical** to v1's — only the top-level model changed.

## Contents

- [hyperbolic_gnnV2.py](hyperbolic_gnnV2.py) — `KettleGraphReasoner` v2
- [layers/](layers/) — shared primitives (same files as v1)

## What makes v2 distinctive

Depth aggregation is **learned soft attention over per-layer snapshots**, replacing v1's concat diagnostic.

### `DepthAttention` module ([hyperbolic_gnnV2.py:61-107](hyperbolic_gnnV2.py#L61-L107))

Per-layer pseudo-query softmax attention over tangent-space snapshots from each MP round:

- **RMSNorm on keys** (Technique 3) — prevents magnitude-dominant layers from winning attention by scale. Applied only to the proximity slice `[:, k:]` when `hierarchy_subspace_dim > 0`, so the hierarchy slice `[:, :k]` keeps its depth-as-magnitude signal intact for Task 0.
- **Zero-initialized depth queries** (Technique 4) — one learned query per layer, starts at uniform averaging.
- **Softmax over depth** (Technique 5) — introduces cross-layer competition rather than summation.

### Two attention use sites

1. **Final aggregation** (always on when `depth_attn=True`, default): the scoring head sees an attended tangent mixture across all rounds instead of `logmap0(h_L)`. `node_embeddings` in the output is `expmap0` of the attended tangent, so downstream radial regularization targets the representation the head actually uses.
2. **Intra-stack re-mixing** (`depth_attn_intra_stack=True`, off by default): at each layer `l > 0`, the input to the next attn/MP pair is `expmap0(DepthAttention(snapshots, query_idx=l))` instead of the raw previous `h`. Lets deeper layers reach back into earlier rounds.

### Other v2-only specifics

- `tangent_scale` is hardcoded to `0.1` (v1 exposed `tangent_scale_init`, default 0.15).
- `concat_depth` is **removed** — the depth-concat diagnostic from v1 is superseded by learned depth attention.
- `_RMSNorm` fallback for PyTorch < 2.4, selected via `_make_rmsnorm`.
- Hierarchy/proximity slicing is single-layer (no per-layer concat branch to worry about) — simpler than v1's `concat_depth` path.

## When to touch this folder

- Active experimental work on hyperbolic depth aggregation.
- New ablations: disable `depth_attn`, toggle `depth_attn_intra_stack`, vary RMSNorm placement.
- Keep `layers/` in sync with v1 — if a primitive changes, update both (or promote `layers/` to a shared location).
- For baseline comparisons, reach into [../models/](../models/) — don't add Euclidean variants here.

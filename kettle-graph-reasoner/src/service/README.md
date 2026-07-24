# KGR Context Service

The deployable core that turns the frozen KGR encoder's validated
capability -- anchor-conditioned structural ordering of a graph
neighborhood -- into an API for feeding high-signal context to an LLM.

## Use

```python
from src.service.context_service import KGRContextService
svc = KGRContextService()                       # loads frozen v1.0 encoder
h   = svc.load_graph("path/to/neighborhood.npz")  # embed once
res = svc.order_context(h, anchor_rows=[12], top_k=20, ball_hops=4)
for it in res.items:
    print(it.rank, it.node_id, it.score, it.hop, it.rationale)
```

- `order_context(handle, anchor_rows, top_k, ball_hops)` -- rank context
  by structural relevance. `anchor_rows` may be a list (multi-anchor ->
  min-distance, for compound/union queries). `ball_hops` restricts to the
  Cypher/BFS ball; omit to score the whole neighborhood.
- `suggest_missing_links(handle, anchor_row)` -- code-graph mode: rank
  non-adjacent nodes structurally near the anchor (missing-edge suggester).
- `res.discrimination` -- DESCRIPTIVE score spread only. Measured
  (2026-07-10) NOT to distinguish a correct from a wrong anchor; do not
  gate on it. Anchor correctness must be guaranteed upstream.

## What it does, measured against ground truth (real archival graphs)

- Multi-anchor beats single on COMPOUND: precision@10 0.108 -> 0.377
  (+0.269) using a real 2nd anchor.
- Surfaces the relevant node(s): recall@10 = 1.0 on provenance/subgraph
  (small relevant sets); precision@10 0.75-0.82 on dense families
  (temporal/multihop). precision@10 on small-relevant families is capped
  by |relevant| (~1 node) -- ndcg (0.885-0.999) is the honest quality
  metric there, not precision@10.
- Wrong anchor is visibly harmful (precision@10 0.51 -> 0.16) with NO
  self-detectable warning -- guarantee the anchor upstream.
- Live Neo4j round-trip works end-to-end
  (`scripts/kgr_context_exercise.py`).

## Design is dictated by measured findings

| choice | finding |
|---|---|
| return top-k RANKING, not a distance-thresholded mask | no global threshold selects a clean subgraph (F1 lift ~0) |
| deterministic tie-break on canonical node id | ranking was permutation-unstable on score ties (|dndcg| up to 0.37) |
| multi-anchor via min-distance | compound sets are not one neighborhood; 2 good anchors recover +0.47 |
| anchor is the caller's responsibility (not self-detectable) | a 1-hop anchor error collapses ordering to the random floor; score-spread does NOT flag it |
| structural (schema-agnostic) ordering | encoder rides topology (random-feature MP), not schema semantics |

Ordering quality: ndcg@10 0.885-0.999 on the real archival graph (zero
training). Cost: ~150 ms embed + ~10 ms order for a 400-node ball on CPU.
Details: Docs/STRESS_TEST_2026-07-10.md, Docs/BENCHMARK_2026-07-07.md.

## Live Neo4j

`scripts/live_ball_rank.py` shows the export->encode round-trip against a
running instance; point `load_graph` at the exported npz.

## Live Neo4j Explorer (local web app)

A published claude.ai artifact CANNOT reach Neo4j (sandbox CSP blocks all
network; the model needs Python). So the interactive tool is a LOCAL app:

    py -m src.service.explorer_app        # -> http://127.0.0.1:8765

- Cypher browser (READ-ONLY; write clauses rejected before the DB).
- KGR context: load a live neighborhood once (samples + embeds from
  Neo4j), then click nodes to set anchor(s) -- the model re-ranks
  instantly; multi-select composes anchors by nearest-distance.

Anchor SEARCH: type an entity (name / type / label / id), pick a result,
and the model centres a neighborhood on THAT node (anchor-centered ball
via the exporter graphcache, reused in-process). Verified end-to-end
against the live 327K-node graph (health, cypher, search, center, rank,
write-guard, load, single- and multi-anchor reorder). Same-origin (no
CORS); creds stay server-side via scripts/neo4j_reader (.env).

## LLM integration (the downstream consumer)

The KGR architecture is "structural layer between the graph and an LLM".
The explorer closes that loop: KGR selects + ranks a subgraph, a LOCAL
small LLM answers over it.

- Auto-detects a local OpenAI-compatible server: LM Studio (:1234) or
  Ollama (:11434); no new Python deps (`src/service/llm.py`, urllib).
- `/api/ask` serializes the selected subgraph (NODES + RELATIONS, anchor
  included and marked) and calls the LLM. Modes: `kgr` (model-ranked
  context), `bfs` (hop-order baseline), `none`.
- UI: a question box, "Ask · KGR context", and "Compare KGR vs BFS" —
  the A/B that shows the ranking earn its keep on answer quality.

Verified live (gemma-4-e4b via LM Studio, "what happened in 1980?" on a
Turnbull-Refuge-centred neighborhood): KGR context -> 3 observations
(conf 0.78) + 2 events, cited by id; BFS context -> 1 event. Same
question, same k, same model. Note: gemma-4 is a REASONING model, so the
client uses a generous max_tokens (small budgets yield an empty answer).

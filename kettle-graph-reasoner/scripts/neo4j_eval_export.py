"""
neo4j_eval_export.py
====================

Neo4j real-graph -> tier1-schema NPZ **evaluation** sample exporter.

This is the previously-absent "Neo4j real-graph loader" (CLAUDE.md /
HANDOFF.md list it as not-yet-built). It produces model-ready evaluation
graphs from the live archival graph so the trained KGR encoder + query head
can be scored on a real structural task.

Architectural guardrails (CLAUDE.md):
  * Real graphs are EVALUATION-ONLY. These NPZs are an eval corpus, never
    training data -- the filenames live under a `*_eval` corpus dir and the
    module name says `eval`.
  * The KGR encoder consumes ONE graph at a time and the synthetic corpus
    graphs are ~200-400 nodes (it is a query-neighborhood scorer, not a
    whole-graph model -- HANDOFF.md sec.3). So we do NOT emit the 66.7k-node
    `domain_only` layer as one graph; we sample K connected query
    neighborhoods around Year-linked anchors, each capped at --max-nodes.

What it emits (exact tier1 contract, verified against
src/data/corpus_dataset.py:_build_graph_tensors and feature_encoder.py):

    x                            (N,32)  f32   node features
    neo4j_node_id                (N,)    i64   live Neo4j id, for inspection
    edge_index                   (2,E)   i64
    edge_attr                    (E,30)  f32   [0:25]type [25:29]cat [29]dir
    duplicate_pairs              (0,3)   i64   (unused by task 2)
    seed, schema_seed            scalar  i64
    schema_n_node_types/n_edge_types     i64
    schema_node_layer_assignment (16,)   i64
    schema_edge_category         (30,)   i64
    schema_edge_directed         (30,)   f32
    schema_edge_source_layers    (30,4)  f32
    schema_edge_target_layers    (30,4)  f32
    n_tasks                      scalar  i64
    task_j_type(=2) / _query(18,) / _labels(N,) / _anchor_row /
        _max_hops / _temporal(2,)        TASK_TEMPORAL, mirrors
                                         task_generator.generate_temporal_tasks

Real label -> KGR 4-layer mapping (domain_only is the L3 "domain" layer):
    Year, Period            -> LAYER_AUXILIARY (3)   temporal scaffolding
    everything else          -> LAYER_ENTITY    (2)   domain entities
Real rel-type -> KGR edge category is a documented name heuristic
(`_edge_category`); it only colours the schema descriptor / edge_attr
category dims, never the topology.

Usage
-----
    # export a small real eval sample (DB must be up; .env supplies creds)
    py neo4j_eval_export.py export --config kettle_config.yaml \
        --out ../src/data/corpus/real_domain_eval \
        --num-graphs 16 --max-nodes 400 --tasks-per-graph 3 --seed 0

    # then score it with the shipped checkpoint (run from repo root)
    py scripts/neo4j_eval_export.py score \
        --corpus src/data/corpus/real_domain_eval \
        --checkpoint runs/sweep_arch_hyp/h128_l4_seed1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from dotenv import find_dotenv, load_dotenv
from neo4j import GraphDatabase

from graph_diagnostics.core import DiagnosticConfig, lifecycle_predicate
from graph_diagnostics.graphcache import (
    build_cache, connected_components,
)

# --- KGR constants (mirrored from src/data/schema_sampler.py + corpus_dataset.py)
LAYER_SOURCE = 0
LAYER_CLAIM = 1
LAYER_ENTITY = 2
LAYER_AUXILIARY = 3
NUM_LAYERS = 4
MAX_NODE_TYPES = 16
MAX_EDGE_TYPES = 30
NODE_FEAT_DIM = 32
EDGE_FEAT_DIM = 30
QUERY_FEAT_DIM = 18
NODE_TYPE_DIM_ACTUAL = 12       # feature_encoder.py:41
EDGE_TYPE_DIM = 25              # feature_encoder.py:43
CAT_PROVENANCE, CAT_REFERENCE, CAT_STRUCTURAL, CAT_COOCCURRENCE = 0, 1, 2, 3

# Layer-mapping for the real archival schema. Earlier exports collapsed
# everything-but-Year/Period to LAYER_ENTITY (binary mapping), which left
# task-0 provenance with no LAYER_SOURCE nodes to score. The 4-way mapping
# below matches the synthetic generator's hierarchy (source -> claim ->
# entity -> auxiliary) and is required for the provenance / subgraph
# tasks to have meaningful labels. Real-graph corpora generated AFTER
# this change will have a different schema_node_layer_assignment than the
# pre-change ones; they should be regenerated to match (the encoder was
# trained against the 4-way layout, so this brings real eval closer to
# the training distribution).
_AUX_LABELS = {"Year", "Period"}
_SOURCE_LABELS = {                       # provenance roots (where claims trace to)
    "Document", "Section", "Paragraph", "Page", "Parcel",
    "ExtractionRun",
}
_CLAIM_LABELS = {                        # intermediate assertions / observations
    "Claim", "Mention", "Event", "Observation", "Measurement",
}


def _resolve_env() -> None:
    here = Path(__file__).resolve().parent
    p = find_dotenv(filename=".env", usecwd=False, raise_error_if_not_found=False)
    if not p:
        for parent in (here, *here.parents):
            if (parent / ".env").is_file():
                p = str(parent / ".env")
                break
    load_dotenv(p or None)


def _driver():
    _resolve_env()
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        notifications_min_severity="OFF",
    )


def _session(driver):
    # Match graph_diagnostics: use the server's default database. The data
    # lives in `neo4j` (NEO4J_DATABASE in .env is stale/unused -- the
    # diagnostics tool that successfully pulls 327k nodes ignores it too).
    return driver.session()


# ---------------------------------------------------------------------------
# Real label / rel-type -> KGR layer / category
# ---------------------------------------------------------------------------

def _node_layer(primary_label: str) -> int:
    if primary_label in _AUX_LABELS:
        return LAYER_AUXILIARY
    if primary_label in _SOURCE_LABELS:
        return LAYER_SOURCE
    if primary_label in _CLAIM_LABELS:
        return LAYER_CLAIM
    return LAYER_ENTITY


def _edge_category(rel_type: str) -> int:
    r = rel_type.upper()
    # Temporal scaffolding (Year/Period axis) -> STRUCTURAL. Keyword
    # "DURING" catches OCCURRED_DURING.
    if any(k in r for k in ("YEAR", "PERIOD", "TEMPORAL", "DATE",
                            "SUPERSED", "CORROBORAT", "SCOPE",
                            "DURING", "NEXT")):
        return CAT_STRUCTURAL
    # Provenance: document/containment chains + evidence/derivation +
    # event causation. The original keyword set caught only
    # SOURCED_FROM / EVIDENCED_BY; archival graph also uses
    # HAS_CLAIM (Document->Claim), HAS_PARAGRAPH (Section->Paragraph),
    # CONTAINS_MENTION (Paragraph->Mention), PRODUCED (Event->...),
    # TRIGGERED (Event->Event). All are provenance chains rather than
    # references, so categorize accordingly.
    if any(k in r for k in ("SOURCE", "DERIV", "PROVEN", "EVIDENC",
                            "CONTAINS", "HAS_CLAIM", "HAS_PARAGRAPH",
                            "HAS_SECTION", "HAS_PAGE", "PRODUCED",
                            "TRIGGERED", "EXTRACT")):
        return CAT_PROVENANCE
    if any(k in r for k in ("REFER", "MENTION", "ABOUT", "DESCRIB")):
        return CAT_REFERENCE
    if any(k in r for k in ("CO_", "COOCCUR", "RELATED", "ASSOCIAT",
                            "SAME_AS", "LINKED", "NEAR")):
        return CAT_COOCCURRENCE
    return CAT_STRUCTURAL


# ---------------------------------------------------------------------------
# Query / task encoding -- byte-exact mirror of feature_encoder.encode_query
# and task_generator.generate_temporal_tasks
# ---------------------------------------------------------------------------

def _anchor_identity(neo4j_id: int) -> np.ndarray:
    """8-dim deterministic identity from a Neo4j node id. Used by the
    task-0/3/4/5 query encoders so the model knows *which* node anchors
    the query (task 2 leaves this zero since temporal is anchor-free).

    Mirrors the per-node identity vector packed into ``x[r, to+3:to+11]``
    inside ``_encode_graph`` (line ~480) so the query and the encoded
    anchor node end up in the same identity subspace; both use
    ``np.random.default_rng(neo4j_id & 0xFFFFFFFF).standard_normal(8)``.
    """
    rng = np.random.default_rng(int(neo4j_id) & 0xFFFFFFFF)
    return rng.standard_normal(8).astype(np.float32)


def _encode_query_temporal(window: tuple[float, float], max_hops: int) -> np.ndarray:
    q = np.zeros(QUERY_FEAT_DIM, dtype=np.float32)
    q[2] = 1.0                       # task type 2 flag (slots 0:6)
    q[6] = window[0]
    q[7] = window[1]
    q[8] = max_hops / 10.0           # feature_encoder.py:229
    return q                          # [9]=pad, [10:18]=0 (no anchor identity)


def _encode_query_provenance(anchor_neo4j_id: int, max_hops: int) -> np.ndarray:
    """Task-0 query: provenance trace from a specific anchor entity.

    Slot layout (task_generator.py:generate_provenance_tasks doesn't
    expose query semantics — feature_encoder.encode_query is the spec):
      q[0]=1.0      task-type flag for TASK_PROVENANCE
      q[6:8]=0      no window (provenance is anchor-only)
      q[8]          max_hops / 10
      q[10:18]      anchor identity (8-dim, matches node identity slot)
    """
    q = np.zeros(QUERY_FEAT_DIM, dtype=np.float32)
    q[0] = 1.0
    q[8] = max_hops / 10.0
    q[10:18] = _anchor_identity(anchor_neo4j_id)
    return q


def _encode_query_multihop(anchor_neo4j_id: int, max_hops: int) -> np.ndarray:
    """Task-3 query: multi-hop relevance from an anchor entity."""
    q = np.zeros(QUERY_FEAT_DIM, dtype=np.float32)
    q[3] = 1.0
    q[8] = max_hops / 10.0
    q[10:18] = _anchor_identity(anchor_neo4j_id)
    return q


def _encode_query_subgraph(anchor_neo4j_id: int, window: tuple[float, float],
                           max_hops: int) -> np.ndarray:
    """Task-4 query: composite (entity-anchored + in-window + reachable)."""
    q = np.zeros(QUERY_FEAT_DIM, dtype=np.float32)
    q[4] = 1.0
    q[6] = window[0]
    q[7] = window[1]
    q[8] = max_hops / 10.0
    q[10:18] = _anchor_identity(anchor_neo4j_id)
    return q


def _encode_query_compound(anchor_neo4j_id: int,
                           component_types: tuple[int, int],
                           window: tuple[float, float] | None,
                           max_hops: int) -> np.ndarray:
    """Task-5 query: intersection of two component task labels.

    ``component_types`` is a tuple of two TASK_* ids; we OR their type
    flags into the query so the model knows the composition. If either
    component is temporal we include the window; otherwise zeros.
    """
    q = np.zeros(QUERY_FEAT_DIM, dtype=np.float32)
    q[5] = 1.0                                        # compound flag
    for ct in component_types:                        # OR component flags
        if 0 <= ct < 6:
            q[ct] = 1.0
    if window is not None:
        q[6] = window[0]
        q[7] = window[1]
    q[8] = max_hops / 10.0
    q[10:18] = _anchor_identity(anchor_neo4j_id)
    return q


def _temporal_task(rng, t_start: np.ndarray, t_end: np.ndarray):
    """Mirror task_generator.generate_temporal_tasks (lines 309-369)."""
    w_start = float(rng.uniform(0.0, 0.75))
    width = float(rng.uniform(0.05, 0.25))
    w_end = min(1.0, w_start + width)
    margin = width * 0.2
    n = t_start.shape[0]
    labels = np.zeros(n, dtype=np.float32)
    anchor_row = -1
    for row in range(n):
        os_ = max(t_start[row], w_start)
        oe_ = min(t_end[row], w_end)
        if os_ < oe_:
            labels[row] = 1.0
            if anchor_row < 0:
                anchor_row = row
        elif (t_start[row] < w_end + margin) and (t_end[row] > w_start - margin):
            labels[row] = 0.5
    if anchor_row < 0:
        anchor_row = 0
    return labels, (w_start, w_end), anchor_row


def _build_adj_with_cat(edge_index: np.ndarray, edge_attr: np.ndarray, n: int):
    """Return (adj_undirected, adj_provenance_back, edge_type_freq, deg_total).

    adj_undirected : list[list[(neighbor, edge_type_slot)]] -- symmetric.
    adj_provenance_back : list[list[neighbor]] -- BACKWARD along
        category-PROVENANCE edges (a->b becomes b in adj[a]'s parents
        list -- i.e. follow provenance "upward" from a derived claim
        back to its source). edge_attr cols [25:29] are the category
        one-hot; col 25 is CAT_PROVENANCE.
    edge_type_freq : dict[int, int] frequency of each edge-type slot.
    deg_total : (n,) int total degree per node.
    """
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    prov_back: list[list[int]] = [[] for _ in range(n)]
    edge_type_freq: dict[int, int] = {}
    E = edge_index.shape[1]
    for i in range(E):
        a = int(edge_index[0, i])
        b = int(edge_index[1, i])
        et = int(edge_attr[i, :EDGE_TYPE_DIM].argmax())
        edge_type_freq[et] = edge_type_freq.get(et, 0) + 1
        adj[a].append((b, et))
        adj[b].append((a, et))
        if edge_attr[i, EDGE_TYPE_DIM + CAT_PROVENANCE] >= 0.5:
            # Provenance chains in the archival graph mix both directions
            # (EVIDENCED_BY goes Claim->Document, HAS_CLAIM goes
            # Document->Claim). Treat the provenance subgraph as
            # undirected: from any node we can BFS to anything connected
            # to it by a provenance edge.
            prov_back[a].append(b)
            prov_back[b].append(a)
    deg_total = np.array([len(x) for x in adj], dtype=np.int64)
    return adj, prov_back, edge_type_freq, deg_total


def _provenance_task(rng, node_layer: np.ndarray, prov_back: list[list[int]],
                     anchor_pool_rows: np.ndarray, max_hops: int = 4):
    """Task 0: backward BFS from an Entity anchor along PROVENANCE edges.

    Mirrors src/data/task_generator.py:generate_provenance_tasks. Sources
    (LAYER_SOURCE) get label 1.0; intermediates get 1/(d+1); the anchor
    itself is 1.0. Returns (labels, anchor_row) or (None, -1) if the
    anchor has no reachable LAYER_SOURCE node within ``max_hops``.
    """
    n = node_layer.shape[0]
    if anchor_pool_rows.size == 0:
        return None, -1
    anchor_row = int(anchor_pool_rows[int(rng.integers(anchor_pool_rows.size))])
    labels = np.zeros(n, dtype=np.float32)
    visited: dict[int, int] = {anchor_row: 0}
    queue: list[tuple[int, int]] = [(anchor_row, 0)]
    while queue:
        nid, dist = queue.pop(0)
        if dist >= max_hops:
            continue
        for nb in prov_back[nid]:
            if nb in visited:
                continue
            visited[nb] = dist + 1
            queue.append((nb, dist + 1))
    for nid, dist in visited.items():
        if node_layer[nid] == LAYER_SOURCE:
            labels[nid] = 1.0
        elif dist > 0:
            labels[nid] = 1.0 / (dist + 1)
    labels[anchor_row] = 1.0
    return labels, anchor_row


def _multihop_task(rng, adj: list[list[tuple[int, int]]],
                   edge_type_freq: dict[int, int], deg_total: np.ndarray,
                   anchor_pool_rows: np.ndarray, max_hops: int = 6,
                   alpha: float = 0.85, cutoff: float = 0.15):
    """Task 3: BFS hop-decay relevance from an Entity anchor.

    Recalibrated alpha=0.85 (vs synthetic 0.7) extends the effective hop
    range from 5 to ~11. The Neo4j archival graph has 92% of Entity-
    Entity pairs at hop>=6; the synthetic alpha would zero out almost
    every meaningful candidate.

    Mirrors task_generator.generate_multihop_tasks but works with the
    pre-built adjacency + edge_type_freq from ``_build_adj_with_cat``.
    """
    n = deg_total.shape[0]
    if anchor_pool_rows.size == 0:
        return None, -1
    anchor_row = int(anchor_pool_rows[int(rng.integers(anchor_pool_rows.size))])
    max_freq = max(edge_type_freq.values()) if edge_type_freq else 1
    labels = np.zeros(n, dtype=np.float32)
    visited: dict[int, tuple[int, list[int]]] = {}
    queue: list[tuple[int, int, list[int]]] = [(anchor_row, 0, [])]
    while queue:
        nid, dist, path = queue.pop(0)
        if nid in visited:
            continue
        visited[nid] = (dist, path)
        if dist >= max_hops:
            continue
        for nb, et in adj[nid]:
            if nb not in visited:
                queue.append((nb, dist + 1, path + [et]))
    for nid, (dist, path) in visited.items():
        dist_score = alpha ** dist
        if path:
            rarity = float(np.mean([
                1.0 - (edge_type_freq.get(et, 1) / max_freq) for et in path
            ]))
        else:
            rarity = 0.0
        branch_penalty = 1.0 / max(float(np.log1p(int(deg_total[nid]))), 1.0)
        labels[nid] = dist_score * (1.0 + rarity) * branch_penalty
    mx = float(labels.max())
    if mx > 0:
        labels /= mx
    labels[labels < cutoff] = 0.0
    return labels, anchor_row


def _subgraph_task(rng, adj: list[list[tuple[int, int]]],
                   edge_attr: np.ndarray, edge_index: np.ndarray,
                   t_start_row: np.ndarray, t_end_row: np.ndarray,
                   anchor_pool_rows: np.ndarray, max_hops: int = 3):
    """Task 4: subgraph membership = (in temporal window) AND (within
    ``max_hops`` of anchor along non-temporal edges). Composite of the
    temporal label rule and an entity-reachability mask, binarized.

    Non-temporal reach: BFS over edges whose category is NOT
    CAT_STRUCTURAL (col EDGE_TYPE_DIM+CAT_STRUCTURAL=27). Year/Period
    scaffolding doesn't count as "structural" provenance to the entity.
    """
    n = t_start_row.shape[0]
    if anchor_pool_rows.size == 0:
        return None, -1, None
    anchor_row = int(anchor_pool_rows[int(rng.integers(anchor_pool_rows.size))])

    # Window sampling: identical to temporal-task window.
    w_start = float(rng.uniform(0.0, 0.75))
    width = float(rng.uniform(0.05, 0.25))
    w_end = min(1.0, w_start + width)

    # in_window mask (binary on overlap, no margin -- subgraph is sharper).
    overlap_lo = np.maximum(t_start_row, w_start)
    overlap_hi = np.minimum(t_end_row, w_end)
    in_window = overlap_lo < overlap_hi

    # Non-temporal-edge adjacency from edge_attr.
    E = edge_index.shape[1]
    is_struct = edge_attr[:, EDGE_TYPE_DIM + CAT_STRUCTURAL] >= 0.5
    adj_nt: list[list[int]] = [[] for _ in range(n)]
    for i in range(E):
        if is_struct[i]:
            continue
        a = int(edge_index[0, i]); b = int(edge_index[1, i])
        adj_nt[a].append(b); adj_nt[b].append(a)

    # BFS for reach.
    dist = np.full(n, -1, dtype=np.int64)
    dist[anchor_row] = 0
    fr = [anchor_row]
    while fr:
        nxt: list[int] = []
        for u in fr:
            if dist[u] >= max_hops:
                continue
            for v in adj_nt[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        fr = nxt
    in_reach = dist >= 0

    labels = np.zeros(n, dtype=np.float32)
    labels[in_window & in_reach] = 1.0
    labels[anchor_row] = 1.0                          # anchor always included
    return labels, anchor_row, (w_start, w_end)


def _compound_task(labels_a: np.ndarray, labels_b: np.ndarray,
                   anchor_a: int):
    """Task 5: element-wise intersection. Mirrors
    task_generator.generate_compound_tasks: min(a, b) then binarize at
    >= 0.5 (so a 0.5-temporal-adjacent + 1.0-anchor doesn't dominate).
    """
    inter = np.minimum(labels_a, labels_b)
    labels = (inter >= 0.5).astype(np.float32)
    labels[anchor_a] = 1.0                            # keep anchor in scope
    return labels, anchor_a


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _spec_from_config(config_path: str) -> tuple[DiagnosticConfig, dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = DiagnosticConfig(**data)
    spec = cfg.subgraphs.get("domain_only")
    if spec is None:
        raise SystemExit("kettle_config.yaml has no 'domain_only' subgraph spec")
    return cfg, spec


def cmd_export(args) -> int:
    cfg, spec = _spec_from_config(args.config)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    drv = _driver()
    try:
        drv.verify_connectivity()
        with _session(drv) as s:
            print("pulling domain_only via graphcache...")
            cache = build_cache(s, lifecycle_predicate(cfg, var="n"),
                                progress=lambda m: print("  " + m))

            node_mask = cache.label_mask(spec.get("include_labels") or [])
            edge_mask = cache.edge_mask(
                node_mask,
                spec.get("include_rel_types") or [],
                spec.get("exclude_rel_types") or [],
            )
            indptr, indices = cache.induced_csr(edge_mask)

            # Year-derived temporal position per node (normalized to [0,1]).
            yvals = _pull_year_values(s, cache.id2idx)
            min_y, max_y, has_y = cache.ensure_year_reachability(
                "Year", yvals, cfg.temporal_max_hops,
            )
            if yvals:
                lo, hi = min(yvals.values()), max(yvals.values())
            else:
                lo, hi = 0.0, 1.0
            span = (hi - lo) or 1.0
            t_start = np.where(has_y, (min_y - lo) / span, 0.0).astype(np.float64)
            t_end = np.where(has_y, (max_y - lo) / span, 0.0).astype(np.float64)
            t_start = np.clip(t_start, 0.0, 1.0)
            t_end = np.clip(t_end, 0.0, 1.0)

            # Giant component of domain_only; anchors must be Year-linked so
            # the temporal task is non-degenerate.
            comp = connected_components(indptr, indices, cache.n,
                                        allowed=node_mask)
            lab = comp[node_mask]
            giant = int(np.bincount(lab[lab >= 0]).argmax())
            giant_member = node_mask & (comp == giant)
            anchor_pool = np.flatnonzero(giant_member & has_y)
            if anchor_pool.size == 0:
                raise SystemExit("no Year-linked nodes in domain_only giant "
                                 "component; cannot build temporal eval tasks")

            idx2id = np.empty(cache.n, dtype=np.int64)
            for nid, ix in cache.id2idx.items():
                idx2id[ix] = nid

            # Temporal midpoint per node (normalized) -- used to stratify the
            # de-localized sampler so each graph spans multiple Year regimes.
            mid = np.where(has_y, (t_start + t_end) / 2.0, np.nan)

            print(f"sampling {args.num_graphs} neighborhoods "
                  f"(<= {args.max_nodes} nodes each, sampler={args.sampler})...")
            written = 0
            for gi in range(args.num_graphs):
                if args.sampler == "delocalized":
                    nodes, seed_node = _delocalized_sample(
                        indptr, indices, giant_member, anchor_pool, mid,
                        rng, args.n_seeds, args.max_nodes,
                    )
                else:  # anchor_ball (legacy; locality-confounded)
                    seed_node = int(anchor_pool[rng.integers(anchor_pool.size)])
                    nodes = _bfs_ball(indptr, indices, giant_member,
                                      seed_node, args.max_nodes)
                if nodes.size < 8:
                    continue
                npz = _encode_graph(
                    s, cache, nodes, seed_node, idx2id,
                    t_start, t_end, rng, args.tasks_per_graph, gi,
                )
                fp = out_dir / f"graph_{written:06d}.npz"
                np.savez_compressed(fp, **npz)
                written += 1
                print(f"  graph_{written-1:06d}: N={npz['x'].shape[0]} "
                      f"E={npz['edge_index'].shape[1]} "
                      f"tasks={int(npz['n_tasks'])}")
            print(f"\nwrote {written} eval graphs to {out_dir}")
    finally:
        drv.close()
    return 0


def _pull_year_values(s, id2idx) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in s.run("MATCH (y:`Year`) RETURN id(y) AS id, y.`year` AS v"):
        ix = id2idx.get(row["id"])
        v = row["v"]
        if ix is None or v is None:
            continue
        try:
            out[ix] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _delocalized_sample(indptr, indices, member, anchor_pool, mid,
                        rng, n_seeds, cap):
    """Temporally-stratified multi-seed sample.

    Picks `n_seeds` Year-linked seeds from DISTINCT temporal-midpoint
    quantile bins, grows a small ball around each, and unions them. The
    resulting graph spans several Year regimes, so the in-window nodes for
    any query window are scattered across disjoint sub-regions rather than
    clustered near one center -- this removes the locality shortcut that let
    a trivial "distance from the task anchor" heuristic near-solve the
    anchor-centered eval. The task-label rule itself is unchanged (kept
    identical to task_generator.generate_temporal_tasks for comparability).
    """
    m = mid[anchor_pool]
    ok = np.isfinite(m)
    pool, mv = anchor_pool[ok], m[ok]
    if pool.size == 0:
        seed = int(anchor_pool[rng.integers(anchor_pool.size)])
        return _bfs_ball(indptr, indices, member, seed, cap), seed
    ns = int(max(2, min(n_seeds, pool.size)))
    edges = np.quantile(mv, np.linspace(0, 1, ns + 1))
    per = max(8, cap // ns)
    picked: set[int] = set()
    seeds: list[int] = []
    for b in range(ns):
        lo, hi = edges[b], edges[b + 1]
        inb = pool[(mv >= lo) & (mv <= hi)] if b == ns - 1 else \
            pool[(mv >= lo) & (mv < hi)]
        if inb.size == 0:
            continue
        sd = int(inb[rng.integers(inb.size)])
        seeds.append(sd)
        ball = _bfs_ball(indptr, indices, member, sd, per)
        picked.update(int(x) for x in ball)
        if len(picked) >= cap:
            break
    nodes = np.array(sorted(picked), dtype=np.int64)[:cap]
    return nodes, (seeds[0] if seeds else int(nodes[0]))


def _bfs_ball(indptr, indices, member, seed, cap) -> np.ndarray:
    """Connected BFS ball of <= cap nodes around seed within `member`."""
    picked = {seed}
    frontier = [seed]
    while frontier and len(picked) < cap:
        nxt = []
        for u in frontier:
            for vi in range(indptr[u], indptr[u + 1]):
                v = int(indices[vi])
                if member[v] and v not in picked:
                    picked.add(v)
                    nxt.append(v)
                    if len(picked) >= cap:
                        break
            if len(picked) >= cap:
                break
        frontier = nxt
    return np.array(sorted(picked), dtype=np.int64)


def _encode_graph(s, cache, nodes, seed_node, idx2id,
                  t_start, t_end, rng, tasks_per_graph, gi) -> dict:
    """Encode one sampled neighborhood into the exact tier1 NPZ schema."""
    n = nodes.size
    cidx_to_row = {int(c): r for r, c in enumerate(nodes)}

    # --- primary label + node-type / layer ids ---
    lab_names = cache.label_names
    # frequency of each label across the sampled nodes (for type-slot order)
    freq = np.zeros(len(lab_names), dtype=np.int64)
    per_node_labels: list[list[int]] = []
    for c in nodes:
        lo, hi = cache.lab_indptr[c], cache.lab_indptr[c + 1]
        lids = [int(x) for x in cache.lab_ids[lo:hi]]
        per_node_labels.append(lids)
        for lid in lids:
            freq[lid] += 1
    # type-slot ids: most frequent sampled labels first (top 12 get one-hot)
    present = sorted({lid for ls in per_node_labels for lid in ls},
                     key=lambda l: -freq[l])
    type_id = {lid: i for i, lid in enumerate(present)}

    def primary(lids: list[int]) -> int:
        # prefer a non-generic label; tie-break by rarity across sample
        cand = [l for l in lids if lab_names[l] != "Entity"] or lids
        return min(cand, key=lambda l: freq[l])

    node_type = np.zeros(n, dtype=np.int64)
    node_layer = np.zeros(n, dtype=np.int64)
    prim_label_of_type: dict[int, str] = {}
    for r, lids in enumerate(per_node_labels):
        pl = primary(lids)
        tid = type_id[pl]
        node_type[r] = tid
        node_layer[r] = _node_layer(lab_names[pl])
        prim_label_of_type[tid] = lab_names[pl]

    # --- directed edges among sampled nodes (one bounded query) ---
    ids = [int(idx2id[c]) for c in nodes]
    es, ed, et_names = [], [], []
    for row in s.run(
        "MATCH (a)-[r]->(b) WHERE id(a) IN $ids AND id(b) IN $ids "
        "RETURN id(a) AS a, id(b) AS b, type(r) AS t", ids=ids,
    ):
        ra = cidx_to_row.get(cache.id2idx.get(row["a"], -1))
        rb = cidx_to_row.get(cache.id2idx.get(row["b"], -1))
        if ra is None or rb is None:
            continue
        es.append(ra)
        ed.append(rb)
        et_names.append(row["t"])
    E = len(es)
    edge_index = np.zeros((2, E), dtype=np.int64)
    edge_attr = np.zeros((E, EDGE_FEAT_DIM), dtype=np.float32)

    # edge-type slot ids ordered by frequency (top 25 get one-hot)
    uniq_rt = sorted(set(et_names), key=lambda t: -et_names.count(t))
    rt_id = {t: i for i, t in enumerate(uniq_rt)}

    in_deg = np.zeros(n); out_deg = np.zeros(n)
    nbr: list[set] = [set() for _ in range(n)]
    for i in range(E):
        a, b = es[i], ed[i]
        edge_index[0, i] = a
        edge_index[1, i] = b
        tt = rt_id[et_names[i]]
        if tt < EDGE_TYPE_DIM:
            edge_attr[i, tt] = 1.0
        edge_attr[i, EDGE_TYPE_DIM + _edge_category(et_names[i])] = 1.0
        edge_attr[i, EDGE_TYPE_DIM + 4] = 1.0           # all Neo4j rels directed
        out_deg[a] += 1
        in_deg[b] += 1
        nbr[a].add(b)
        nbr[b].add(a)

    # --- node features (N,32), exact feature_encoder.encode_nodes layout ---
    x = np.zeros((n, NODE_FEAT_DIM), dtype=np.float32)
    total_deg = in_deg + out_deg
    clustering = np.zeros(n)
    for r in range(n):
        ns = list(nbr[r])
        k = len(ns)
        if k < 2:
            continue
        links = sum(
            1 for i in range(k) for j in range(i + 1, k) if ns[j] in nbr[ns[i]]
        )
        clustering[r] = 2.0 * links / (k * (k - 1))
    # depth = BFS distance from the neighborhood seed (proxy for generation depth)
    seed_row = cidx_to_row[seed_node]
    depth = np.full(n, 0.0)
    seen = {seed_row}
    fr = [seed_row]
    d = 0
    while fr:
        d += 1
        nx = []
        for u in fr:
            for w in nbr[u]:
                if w not in seen:
                    seen.add(w)
                    depth[w] = d
                    nx.append(w)
        fr = nx

    for r in range(n):
        if node_type[r] < NODE_TYPE_DIM_ACTUAL:
            x[r, node_type[r]] = 1.0
        x[r, NODE_TYPE_DIM_ACTUAL + node_layer[r]] = 1.0
        so = NODE_TYPE_DIM_ACTUAL + 4
        x[r, so + 0] = np.log1p(total_deg[r])
        x[r, so + 1] = np.log1p(in_deg[r])
        x[r, so + 2] = np.log1p(out_deg[r])
        x[r, so + 3] = clustering[r]
        x[r, so + 4] = depth[r] / 5.0
        to = so + 5
        ts = float(t_start[nodes[r]])
        te = float(t_end[nodes[r]])
        x[r, to + 0] = ts
        x[r, to + 1] = te
        x[r, to + 2] = te - ts
        # deterministic 8-dim identity (seeded by neo4j id) -- mirrors the
        # synthetic random identity_vector; task 2 query never reads it.
        idr = np.random.default_rng(int(idx2id[nodes[r]]) & 0xFFFFFFFF)
        x[r, to + 3 : to + 11] = idr.standard_normal(8).astype(np.float32)

    # --- schema descriptor (exact SchemaDescriptor.to_tensor_dict layout) ---
    n_node_types = len(present)
    nla = np.zeros(MAX_NODE_TYPES, dtype=np.int64)
    for tid in range(min(n_node_types, MAX_NODE_TYPES)):
        nla[tid] = _node_layer(prim_label_of_type.get(tid, "Entity"))

    n_edge_types = len(uniq_rt)
    eca = np.zeros(MAX_EDGE_TYPES, dtype=np.int64)
    edir = np.zeros(MAX_EDGE_TYPES, dtype=np.float32)
    esrc = np.zeros((MAX_EDGE_TYPES, 4), dtype=np.float32)
    etgt = np.zeros((MAX_EDGE_TYPES, 4), dtype=np.float32)
    # observed endpoint layers per rel type -> valid src/tgt layer one-hot
    seen_src: dict[int, set] = {}
    seen_tgt: dict[int, set] = {}
    for i in range(E):
        ti = rt_id[et_names[i]]
        if ti >= MAX_EDGE_TYPES:
            continue
        seen_src.setdefault(ti, set()).add(int(node_layer[es[i]]))
        seen_tgt.setdefault(ti, set()).add(int(node_layer[ed[i]]))
    for t, ti in rt_id.items():
        if ti >= MAX_EDGE_TYPES:
            continue
        eca[ti] = _edge_category(t)
        edir[ti] = 1.0
        for L in seen_src.get(ti, {LAYER_ENTITY}):
            esrc[ti, L] = 1.0
        for L in seen_tgt.get(ti, {LAYER_ENTITY}):
            etgt[ti, L] = 1.0

    # --- task generation: temporal + provenance + multihop + subgraph + compound ---
    tsr = t_start[nodes]
    ter = t_end[nodes]
    ids_arr = np.array(ids, dtype=np.int64)
    out = {
        "x": x,
        "neo4j_node_id": ids_arr,
        "neo4j_seed_node_id": np.array(int(idx2id[seed_node]), dtype=np.int64),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "duplicate_pairs": np.zeros((0, 3), dtype=np.int64),
        "seed": np.array(gi, dtype=np.int64),
        "schema_seed": np.array(0, dtype=np.int64),
        "schema_n_node_types": np.array(n_node_types, dtype=np.int64),
        "schema_n_edge_types": np.array(n_edge_types, dtype=np.int64),
        "schema_node_layer_assignment": nla,
        "schema_edge_category": eca,
        "schema_edge_directed": edir,
        "schema_edge_source_layers": esrc,
        "schema_edge_target_layers": etgt,
    }
    n_tasks = 0

    def _add_task(t_type: int, labels: np.ndarray, arow: int,
                  query: np.ndarray, max_hops: int,
                  window: tuple[float, float] | None = None,
                  component_types: tuple[int, int] | None = None) -> None:
        nonlocal n_tasks
        out[f"task_{n_tasks}_type"] = np.array(t_type, dtype=np.int64)
        out[f"task_{n_tasks}_anchor_row"] = np.array(arow, dtype=np.int64)
        out[f"task_{n_tasks}_labels"] = labels.astype(np.float32)
        out[f"task_{n_tasks}_query"] = query.astype(np.float32)
        out[f"task_{n_tasks}_max_hops"] = np.array(max_hops, dtype=np.int64)
        out[f"task_{n_tasks}_temporal"] = np.array(
            window if window is not None else (0.0, 0.0), dtype=np.float32,
        )
        if component_types is not None:
            out[f"task_{n_tasks}_components"] = np.array(
                component_types, dtype=np.int64,
            )
        n_tasks += 1

    # Adjacency + edge stats reused across the multi-hop / subgraph /
    # provenance generators (one build per graph).
    adj_und, prov_back, edge_type_freq, deg_total = _build_adj_with_cat(
        edge_index, edge_attr, n,
    )

    # Anchor pool: Entity-layer + Claim-layer nodes (both are content-
    # bearing; Claim is the closest real-graph analogue when there's no
    # Entity in the sampled neighborhood). For provenance / subgraph
    # tasks the anchor must additionally have at least one provenance
    # neighbor, since the label is empty otherwise -- so we precompute a
    # prov-anchor pool that filters the content pool by has-prov-edge.
    entity_rows = np.flatnonzero(
        (node_layer == LAYER_ENTITY) | (node_layer == LAYER_CLAIM)
    )
    if entity_rows.size == 0:
        entity_rows = np.flatnonzero(node_layer != LAYER_AUXILIARY)
    has_prov = np.array([len(pb) > 0 for pb in prov_back], dtype=bool)
    prov_anchor_rows = entity_rows[has_prov[entity_rows]]

    # --- Task 2 (TEMPORAL) ---
    for _ in range(max(1, tasks_per_graph)):
        labels_t, window_t, arow_t = _temporal_task(rng, tsr, ter)
        if (labels_t >= 0.5).sum() == 0:      # skip degenerate windows
            continue
        _add_task(
            2, labels_t, arow_t,
            _encode_query_temporal(window_t, 4), 4, window=window_t,
        )

    # --- Task 0 (PROVENANCE) ---
    for _ in range(max(1, tasks_per_graph)):
        labels_p, arow_p = _provenance_task(rng, node_layer, prov_back,
                                            prov_anchor_rows, max_hops=4)
        if labels_p is None or (labels_p > 0).sum() < 2:
            continue
        _add_task(
            0, labels_p, arow_p,
            _encode_query_provenance(int(ids_arr[arow_p]), 4), 4,
        )

    # --- Task 3 (MULTIHOP) ---
    for _ in range(max(1, tasks_per_graph)):
        labels_m, arow_m = _multihop_task(
            rng, adj_und, edge_type_freq, deg_total, entity_rows,
            max_hops=6, alpha=0.85, cutoff=0.15,
        )
        if labels_m is None or (labels_m > 0).sum() < 2:
            continue
        _add_task(
            3, labels_m, arow_m,
            _encode_query_multihop(int(ids_arr[arow_m]), 6), 6,
        )

    # --- Task 4 (SUBGRAPH) ---
    # Anchor needs a non-empty non-temporal neighborhood (otherwise BFS
    # within the subgraph reaches only the anchor itself). Reuse the
    # provenance-anchor pool as a proxy -- any node with provenance
    # connectivity also has non-temporal connectivity.
    subgraph_anchor_rows = (prov_anchor_rows
                            if prov_anchor_rows.size > 0 else entity_rows)
    for _ in range(max(1, tasks_per_graph)):
        labels_s, arow_s, window_s = _subgraph_task(
            rng, adj_und, edge_attr, edge_index, tsr, ter,
            subgraph_anchor_rows, max_hops=3,
        )
        if labels_s is None or (labels_s > 0).sum() < 2 or window_s is None:
            continue
        _add_task(
            4, labels_s, arow_s,
            _encode_query_subgraph(int(ids_arr[arow_s]), window_s, 3), 3,
            window=window_s,
        )

    # --- Task 5 (COMPOUND) — composes already-generated tasks ---
    # Collect (idx_in_out, type, anchor_row, labels) for components.
    components: list[tuple[int, int, int, np.ndarray]] = []
    for j in range(n_tasks):
        tt = int(out[f"task_{j}_type"])
        if tt in (0, 2, 3):                        # composable component set
            components.append((
                j, tt,
                int(out[f"task_{j}_anchor_row"]),
                np.asarray(out[f"task_{j}_labels"]),
            ))
    n_compound_target = max(1, tasks_per_graph // 2)
    pair_specs = [(2, 0), (2, 3), (0, 3)]          # type pairs to try
    for _ in range(n_compound_target):
        # Random pair-type, then sample one component of each type.
        spec = pair_specs[int(rng.integers(len(pair_specs)))]
        a_options = [c for c in components if c[1] == spec[0]]
        b_options = [c for c in components if c[1] == spec[1]]
        if not a_options or not b_options:
            continue
        ca = a_options[int(rng.integers(len(a_options)))]
        cb = b_options[int(rng.integers(len(b_options)))]
        labels_c, anchor_c = _compound_task(ca[3], cb[3], ca[2])
        if (labels_c > 0).sum() < 2:
            continue
        # Re-use the window if either component is temporal.
        if spec[0] == 2:
            win_c = tuple(out[f"task_{ca[0]}_temporal"].tolist())
        elif spec[1] == 2:
            win_c = tuple(out[f"task_{cb[0]}_temporal"].tolist())
        else:
            win_c = None
        _add_task(
            5, labels_c, anchor_c,
            _encode_query_compound(int(ids_arr[anchor_c]), spec, win_c, 4), 4,
            window=win_c,
            component_types=spec,
        )

    if n_tasks == 0:                          # guarantee >=1 task
        labels = np.zeros(n, dtype=np.float32)
        labels[cidx_to_row[seed_node]] = 1.0
        _add_task(2, labels, cidx_to_row[seed_node],
                  _encode_query_temporal((0.0, 1.0), 4), 4,
                  window=(0.0, 1.0))
    out["n_tasks"] = np.array(n_tasks, dtype=np.int64)
    return out


# ---------------------------------------------------------------------------
# Scoring + baseline comparison (mirrors src/training/train_v3.py:_eval)
# ---------------------------------------------------------------------------

def _repo_on_path() -> Path:
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def _load_checkpoint(ckpt_dir: str):
    """Build (encoder, query_encoder, euclidean, c_val) from a trained run.

    Auto-detects geometry from summary.json config.model and uses the same
    construction as src/training/train_v3.py:_build_encoder so hyperbolic and
    Euclidean checkpoints are both loadable. Returns None if dir is missing.
    """
    import json
    import torch
    ck = Path(ckpt_dir)
    if not (ck / "summary.json").is_file() or not (ck / "encoder.pt").is_file():
        return None
    cfg = json.loads((ck / "summary.json").read_text())["config"]
    euclidean = cfg["model"] != "hyperbolic"
    from src.data.corpus_dataset import CorpusDataset
    from src.modelsv3.hyperbolic_gnnV3 import KettleGraphReasonerV3
    from src.modelsv3.euclidean_v3 import EuclideanReasonerV3
    from src.modelsv3.query_encoder import QueryToBall
    ref = CorpusDataset(corpus_dir=cfg.get("corpus", "src/data/corpus/tier1"),
                        split="val", split_seed=0, include_tasks={2})
    # E5: honor the attn_type_table key (absent => legacy True) so both
    # pre- and post-2026-07-10 checkpoints load strict.
    net = (ref.num_edge_types_max
           if cfg.get("attn_type_table", True) else None)
    if euclidean:
        enc = EuclideanReasonerV3(
            node_feat_dim=ref.node_feat_dim,
            edge_feat_dim=ref.edge_feat_dim_schema,
            hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
            type_dim=cfg["type_dim"], num_edge_types_max=net,
            node_feat_dim_schema=ref.node_feat_dim_schema,
        )
    else:
        enc = KettleGraphReasonerV3(
            node_feat_dim=ref.node_feat_dim,
            edge_feat_dim=ref.edge_feat_dim_schema,
            hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
            type_dim=cfg["type_dim"], c=cfg["curvature"],
            num_edge_types_max=net,
            node_feat_dim_schema=ref.node_feat_dim_schema,
            tangent_scale_init=cfg.get("tangent_scale", 0.1),
        )
    enc.load_state_dict(torch.load(ck / "encoder.pt", map_location="cpu"))
    enc.eval()
    qe = QueryToBall(query_dim=ref.query_dim, hidden_dim=cfg["hidden_dim"],
                     c=cfg["curvature"], euclidean=euclidean)
    qe.load_state_dict(torch.load(ck / "query_encoder.pt", map_location="cpu"))
    qe.eval()
    c_val = getattr(enc, "c", torch.tensor(float(cfg["curvature"])))
    return enc, qe, euclidean, c_val


def _score_model(files, enc, qe, euclidean, c_val) -> dict:
    import torch
    from src.data.corpus_dataset import _build_graph_tensors, _build_task_tensors
    from src.modelsv3.distance_scoring import score_from_embeddings
    from src.training.metrics import MetricAccumulator
    acc = MetricAccumulator()
    with torch.no_grad():
        for f in files:
            with np.load(f) as npz:
                g = _build_graph_tensors(npz)
                emb = enc(g["x"], g["edge_index"], g["edge_type"],
                          g["edge_descriptor"],
                          node_descriptor=g["node_descriptor"]).node_embeddings
                for j in range(int(npz["n_tasks"])):
                    t = _build_task_tensors(npz, j)
                    qp = qe(t["query"])
                    sc = score_from_embeddings(node_embeddings=emb,
                                               query_point=qp, c=c_val,
                                               euclidean=euclidean)
                    acc.add(sc.detach().cpu(), t["labels"], t["task_type"])
    return acc.summary()                 # {"overall": {...}, "by_task_type": {...}}


def _score_heuristic(files, kind: str) -> dict:
    """Non-model baselines, fed to the SAME MetricAccumulator.

      anchor  : rank by -graph_distance from the task anchor (the structural
                "heuristic subgraph retrieval" the learned model replaces).
                Task-agnostic; same code for every task type.
      random  : seeded gaussian per node (noise floor).
      oracle  : reads the task labels directly via the rule that DEFINED
                them -- a near-ceiling, NOT a fair competitor. Dispatches
                on task_type because each task has its own label rule.
    """
    import torch
    from src.data.corpus_dataset import _build_task_tensors
    from src.training.metrics import MetricAccumulator
    acc = MetricAccumulator()
    for fi, f in enumerate(files):
        with np.load(f) as npz:
            x = npz["x"].astype(np.float32)
            ei = npz["edge_index"].astype(np.int64)
            ea = npz["edge_attr"].astype(np.float32)
            n = x.shape[0]
            adj: list[list[int]] = [[] for _ in range(n)]
            for a, b in zip(ei[0], ei[1]):
                adj[int(a)].append(int(b))
                adj[int(b)].append(int(a))
            for j in range(int(npz["n_tasks"])):
                t = _build_task_tensors(npz, j)
                tt = int(npz[f"task_{j}_type"])
                if kind == "random":
                    sc = np.random.default_rng(1234 + 31 * fi + j) \
                        .standard_normal(n).astype(np.float32)
                elif kind == "anchor":
                    anchor = int(npz[f"task_{j}_anchor_row"])
                    dist = np.full(n, -1, dtype=np.int64)
                    dist[anchor] = 0
                    fr = [anchor]
                    while fr:
                        nxt = []
                        for u in fr:
                            for v in adj[u]:
                                if dist[v] < 0:
                                    dist[v] = dist[u] + 1
                                    nxt.append(v)
                        fr = nxt
                    far = int(dist.max()) + 1 if (dist >= 0).any() else 1
                    sc = (-np.where(dist < 0, far, dist)).astype(np.float32)
                elif kind == "oracle":
                    sc = _oracle_score_for_task(
                        tt, x, ea, ei, adj, npz, j, n,
                    )
                else:
                    raise ValueError(kind)
                acc.add(torch.from_numpy(sc),
                        torch.from_numpy(npz[f"task_{j}_labels"]
                                         .astype(np.float32)),
                        tt)
    return acc.summary()                 # {"overall": {...}, "by_task_type": {...}}


def _oracle_score_for_task(task_type: int, x: np.ndarray, edge_attr: np.ndarray,
                           edge_index: np.ndarray, adj: list[list[int]],
                           npz, j: int, n: int) -> np.ndarray:
    """Per-task oracle: recompute the label rule from graph structure.

    For tasks where the label rule IS deterministic from structure (0, 3,
    4) the oracle is essentially the label itself -- it serves as a
    "perfect heuristic" ceiling. Task-2 (temporal) reads the t_start /
    t_end columns directly. Task-5 (compound) is the min of two
    component oracles.
    """
    if task_type == 2:
        q = npz[f"task_{j}_query"].astype(np.float32)
        ws, we = float(q[6]), float(q[7])
        ts, te = x[:, 21], x[:, 22]
        return np.maximum(
            0.0, np.minimum(te, we) - np.maximum(ts, ws)
        ).astype(np.float32)

    if task_type == 0:
        # Backward-BFS along PROVENANCE-category directed edges from anchor.
        anchor = int(npz[f"task_{j}_anchor_row"])
        node_layer = x[:, NODE_TYPE_DIM_ACTUAL:NODE_TYPE_DIM_ACTUAL + 4].argmax(axis=1)
        prov_back: list[list[int]] = [[] for _ in range(n)]
        E = edge_index.shape[1]
        for i in range(E):
            if edge_attr[i, EDGE_TYPE_DIM + CAT_PROVENANCE] >= 0.5:
                a = int(edge_index[0, i]); b = int(edge_index[1, i])
                prov_back[a].append(b)
                prov_back[b].append(a)            # undirected (mixed-direction chain)
        scores = np.zeros(n, dtype=np.float32)
        max_hops = int(npz[f"task_{j}_max_hops"])
        visited: dict[int, int] = {anchor: 0}
        queue: list[tuple[int, int]] = [(anchor, 0)]
        while queue:
            u, d = queue.pop(0)
            if d >= max_hops:
                continue
            for v in prov_back[u]:
                if v not in visited:
                    visited[v] = d + 1
                    queue.append((v, d + 1))
        for nid, d in visited.items():
            if node_layer[nid] == LAYER_SOURCE:
                scores[nid] = 1.0
            elif d > 0:
                scores[nid] = 1.0 / (d + 1)
        scores[anchor] = 1.0
        return scores

    if task_type == 3:
        # Recompute the multihop label rule (alpha=0.85) from adjacency.
        anchor = int(npz[f"task_{j}_anchor_row"])
        max_hops = int(npz[f"task_{j}_max_hops"])
        alpha = 0.85
        cutoff = 0.15
        # Edge-type freq from edge_attr argmax over type cols [0:EDGE_TYPE_DIM].
        E = edge_index.shape[1]
        ets = edge_attr[:, :EDGE_TYPE_DIM].argmax(axis=1)
        type_freq: dict[int, int] = {}
        adj_typed: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for i in range(E):
            et = int(ets[i])
            type_freq[et] = type_freq.get(et, 0) + 1
            a = int(edge_index[0, i]); b = int(edge_index[1, i])
            adj_typed[a].append((b, et))
            adj_typed[b].append((a, et))
        deg_total = np.array([len(x_) for x_ in adj_typed], dtype=np.int64)
        max_freq = max(type_freq.values()) if type_freq else 1
        scores = np.zeros(n, dtype=np.float32)
        visited: dict[int, tuple[int, list[int]]] = {}
        queue = [(anchor, 0, [])]
        while queue:
            u, d, path = queue.pop(0)
            if u in visited:
                continue
            visited[u] = (d, path)
            if d >= max_hops:
                continue
            for v, et in adj_typed[u]:
                if v not in visited:
                    queue.append((v, d + 1, path + [et]))
        for nid, (d, path) in visited.items():
            ds = alpha ** d
            if path:
                rarity = float(np.mean([
                    1.0 - (type_freq.get(et, 1) / max_freq) for et in path
                ]))
            else:
                rarity = 0.0
            bp = 1.0 / max(float(np.log1p(int(deg_total[nid]))), 1.0)
            scores[nid] = ds * (1.0 + rarity) * bp
        mx = float(scores.max())
        if mx > 0:
            scores /= mx
        scores[scores < cutoff] = 0.0
        return scores.astype(np.float32)

    if task_type == 4:
        # Composite: temporal overlap AND non-temporal-edge BFS reach.
        anchor = int(npz[f"task_{j}_anchor_row"])
        max_hops = int(npz[f"task_{j}_max_hops"])
        q = npz[f"task_{j}_query"].astype(np.float32)
        ws, we = float(q[6]), float(q[7])
        ts, te = x[:, 21], x[:, 22]
        overlap = np.maximum(
            0.0, np.minimum(te, we) - np.maximum(ts, ws)
        ).astype(np.float32)
        # Non-structural-cat edges only.
        E = edge_index.shape[1]
        is_struct = edge_attr[:, EDGE_TYPE_DIM + CAT_STRUCTURAL] >= 0.5
        adj_nt: list[list[int]] = [[] for _ in range(n)]
        for i in range(E):
            if is_struct[i]:
                continue
            a = int(edge_index[0, i]); b = int(edge_index[1, i])
            adj_nt[a].append(b); adj_nt[b].append(a)
        dist = np.full(n, -1, dtype=np.int64)
        dist[anchor] = 0
        fr = [anchor]
        while fr:
            nxt = []
            for u in fr:
                if dist[u] >= max_hops:
                    continue
                for v in adj_nt[u]:
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            fr = nxt
        in_reach = (dist >= 0).astype(np.float32)
        return (overlap * in_reach).astype(np.float32)

    if task_type == 5:
        # Compound: oracle = element-wise min of the two component oracles.
        # We don't know which task indices the components came from, but
        # the query encodes which TYPES were composed (slots q[0..5]).
        q = npz[f"task_{j}_query"].astype(np.float32)
        comp_types = [t for t in range(6) if t != 5 and q[t] >= 0.5]
        if len(comp_types) < 2:
            # Fall back to anchor-overlap heuristic.
            anchor = int(npz[f"task_{j}_anchor_row"])
            sc = np.zeros(n, dtype=np.float32); sc[anchor] = 1.0
            return sc
        a_score = _oracle_score_for_task(
            comp_types[0], x, edge_attr, edge_index, adj, npz, j, n,
        )
        b_score = _oracle_score_for_task(
            comp_types[1], x, edge_attr, edge_index, adj, npz, j, n,
        )
        return np.minimum(a_score, b_score).astype(np.float32)

    # Unknown task type: noise floor.
    return np.zeros(n, dtype=np.float32)


def _agg(rows: list[dict], keys: list[str]) -> dict:
    """Mean +/- std across seed runs for the given metric keys."""
    out = {}
    for k in keys:
        vals = np.array([float(r[k]) for r in rows], dtype=np.float64)
        out[k] = (float(vals.mean()), float(vals.std()))
    return out


def cmd_score(args) -> int:
    _repo_on_path()
    import json
    files = sorted(Path(args.corpus).glob("graph_*.npz"))
    if not files:
        raise SystemExit(f"no graph_*.npz under {args.corpus}")
    loaded = _load_checkpoint(args.checkpoint)
    if loaded is None:
        raise SystemExit(f"no usable checkpoint at {args.checkpoint}")
    enc, qe, euc, c_val = loaded
    print(json.dumps(_score_model(files, enc, qe, euc, c_val),
                      indent=2, default=float))
    return 0


# The project's own controlled hyp-vs-euc pair (compare_transfer.json):
# architecture-matched (h64/l3/type8, train_graphs_frac=0.5, task 2), 3 seeds.
_HYP_SEEDS = [f"runs/v3_transfer_hyp_seed{s}" for s in (0, 1, 2)]
_EUC_SEEDS = [f"runs/v3_transfer_euc_seed{s}" for s in (0, 1, 2)]
_SHIPPED = "runs/sweep_arch_hyp/h128_l4_seed1"
_KEYS = ["p@5", "r@5", "ndcg@5", "p@10", "r@10", "ndcg@10",
         "p@20", "r@20", "ndcg@20"]


_TASK_NAMES = {
    0: "PROVENANCE", 1: "ER", 2: "TEMPORAL",
    3: "MULTIHOP", 4: "SUBGRAPH", 5: "COMPOUND",
}


def cmd_compare(args) -> int:
    _repo_on_path()
    files = sorted(Path(args.corpus).glob("graph_*.npz"))
    if not files:
        raise SystemExit(f"no graph_*.npz under {args.corpus}")
    # Per-task-type instance counts (so the header reflects the actual
    # multi-task corpus, not legacy "task-2 instances").
    type_counts: dict[int, int] = {}
    n_tasks_total = 0
    for f in files:
        with np.load(f) as npz:
            n_tasks_total += int(npz["n_tasks"])
            for j in range(int(npz["n_tasks"])):
                t = int(npz[f"task_{j}_type"])
                type_counts[t] = type_counts.get(t, 0) + 1
    cnt_str = ", ".join(f"{_TASK_NAMES.get(t, str(t))}={c}"
                        for t, c in sorted(type_counts.items()))
    print(f"corpus: {len(files)} graphs / {n_tasks_total} task instances "
          f"[{cnt_str}]\n  ({args.corpus})\n")

    # Each table entry: (label, full_summary). full_summary has shape
    # {"overall": {ndcg@10: (mean, std), ...},
    #  "by_task_type": {t: {ndcg@10: (mean, std), ...}, ...}}
    table: list[tuple[str, dict]] = []

    def _wrap(summary: dict) -> dict:
        """Convert single-run summary -> mean±0 dict (mirrors _agg shape)."""
        out: dict = {"overall": {k: (v, 0.0) for k, v in summary["overall"].items()}}
        out["by_task_type"] = {
            t: {k: (v, 0.0) for k, v in row.items()}
            for t, row in summary["by_task_type"].items()
        }
        return out

    # heuristics
    for label, kind in (("random (floor)", "random"),
                        ("anchor-BFS (heuristic retrieval)", "anchor"),
                        ("oracle (per-task label rule)", "oracle")):
        table.append((label, _wrap(_score_heuristic(files, kind))))

    # learned models -- aggregate across seeds via _agg on overall AND per task type.
    def group(label, dirs):
        runs: list[dict] = []
        for d in dirs:
            ld = _load_checkpoint(d)
            if ld is None:
                print(f"  (skip missing checkpoint {d})")
                continue
            runs.append(_score_model(files, *ld))
        if not runs:
            return
        # Aggregate "overall" across runs.
        agg_overall = _agg([r["overall"] for r in runs], _KEYS)
        # Aggregate per task type (only types present in ALL runs).
        types_in_common = set(runs[0]["by_task_type"].keys())
        for r in runs[1:]:
            types_in_common &= set(r["by_task_type"].keys())
        agg_by_type = {
            t: _agg([r["by_task_type"][t] for r in runs], _KEYS)
            for t in sorted(types_in_common)
        }
        table.append((label, {"overall": agg_overall, "by_task_type": agg_by_type}))

    group(f"euclidean baseline (h64/l3, n={len(_EUC_SEEDS)})", _EUC_SEEDS)
    group(f"hyperbolic matched (h64/l3, n={len(_HYP_SEEDS)})", _HYP_SEEDS)
    group("hyperbolic shipped (h128/l4)", [_SHIPPED])

    show = ["ndcg@10", "p@10", "r@10", "ndcg@5", "ndcg@20"]
    w = max(len(lbl) for lbl, _ in table)

    def _print_block(title: str, get_row):
        print(title)
        hdr = f"{'method':<{w}}  " + "  ".join(f"{k:>14}" for k in show)
        print(hdr)
        print("-" * len(hdr))
        for lbl, m in table:
            row = get_row(m)
            if row is None:
                continue
            cells = []
            for k in show:
                mu, sd = row[k]
                cells.append(f"{mu:.4f}±{sd:.3f}" if sd else f"{mu:.4f}      ")
            print(f"{lbl:<{w}}  " + "  ".join(f"{c:>14}" for c in cells))
        print()

    _print_block("=== Overall (macro over all tasks) ===",
                 lambda m: m["overall"])
    for t in sorted(type_counts):
        name = _TASK_NAMES.get(t, str(t))
        n_t = type_counts[t]
        _print_block(f"=== Task {t} ({name}) -- {n_t} instances ===",
                     lambda m, t=t: m["by_task_type"].get(t))

    # the hypothesis test: matched hyp vs matched euc, vs the real heuristic
    d = {lbl: m for lbl, m in table}
    euc = next((m for l, m in d.items() if l.startswith("euclidean")), None)
    hyp = next((m for l, m in d.items() if l.startswith("hyperbolic matched")),
               None)
    heur = d.get("anchor-BFS (heuristic retrieval)")
    if hyp and euc and heur:
        print("\nhypothesis test (ndcg@10, overall macro):")
        print(f"  hyperbolic matched : {hyp['overall']['ndcg@10'][0]:.4f}")
        print(f"  euclidean matched  : {euc['overall']['ndcg@10'][0]:.4f}  "
              f"(hyp-euc delta {hyp['overall']['ndcg@10'][0]-euc['overall']['ndcg@10'][0]:+.4f})")
        print(f"  anchor-BFS heuristic: {heur['overall']['ndcg@10'][0]:.4f}  "
              f"(hyp-heuristic delta {hyp['overall']['ndcg@10'][0]-heur['overall']['ndcg@10'][0]:+.4f})")
    return 0


def _stage_b_best_epoch(cfg, model, qe, full_train_ds, device,
                        val_fraction: float = 0.1,
                        unfreeze_encoder: bool = False):
    """Slim re-implementation of train_v3._stage_b with **best-epoch
    checkpoint selection**. The query head's state at the epoch with the
    highest held-out val ndcg@10 is restored at the end; the noisy final
    epoch is discarded.

    The vanilla _stage_b trains for N epochs and leaves whatever weights
    happen to land in qe at the end. Observed training traces show
    rank_accuracy oscillating wildly between epochs (1.0 -> 0.06 -> 1.0)
    and final-epoch evals therefore underestimate per-task ceiling. This
    helper splits the corpus's train index into 90% train / 10% val,
    runs the same inner loop as _stage_b, evaluates on val after each
    epoch via train_v3._eval, and restores the best-epoch state.
    """
    import copy
    import torch
    from src.modelsv3.contrastive import poincare_infonce  # noqa: F401  (parity import)
    from src.modelsv3.ranking import (
        pairwise_ranking_loss, listwise_ranking_loss,
        sampled_infonce_ranking_loss,
    )
    from src.modelsv3.stage_b_bilinear import (
        bilinear_pairwise_loss, bilinear_listwise_loss,
    )
    from src.training.train_v3 import _encode, _sample_to_device, _eval

    # unfreeze_encoder=True does JOINT training (encoder + query head)
    # on the real-graph train split. This deliberately violates the
    # "real graphs are eval-only" CLAUDE.md commitment to test the
    # alternative hypothesis "the encoder geometry, not the head, is
    # the cap". If joint train still loses to anchor-BFS, the
    # architecture itself is bounded above the heuristic.
    if unfreeze_encoder:
        for p in model.parameters():
            p.requires_grad = True
        model.train()
        params = list(model.parameters()) + list(qe.parameters())
    else:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        params = list(qe.parameters())

    lr_q = cfg.lr if cfg.lr_query is None else cfg.lr_query
    opt = torch.optim.Adam(params, lr=lr_q)
    euclidean = cfg.model == "euclidean"
    c_val = getattr(model, "c", torch.tensor(cfg.curvature))
    sb_temp = (cfg.temperature if cfg.stage_b_temperature is None
               else cfg.stage_b_temperature)

    # Train/val split over the dataset's flat index.
    rng = np.random.default_rng(cfg.seed + 202)
    n = len(full_train_ds)
    if n < 4:
        # Too small to split; fall back to final-epoch behavior.
        from src.training.train_v3 import _stage_b as _stage_b_vanilla
        _stage_b_vanilla(cfg, model, qe, full_train_ds, device)
        return {"epochs": cfg.query_epochs, "best_epoch": cfg.query_epochs - 1,
                "best_val_ndcg10": float("nan"), "history": []}
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    val_idx = perm[:n_val].astype(np.int64)
    train_idx = perm[n_val:].astype(np.int64)

    mode = "JOINT encoder+head" if unfreeze_encoder else "head-only (encoder frozen)"
    print(f"[stage-B*] {mode}; best-epoch tracking: "
          f"train={len(train_idx)}  val={len(val_idx)}  "
          f"epochs={cfg.query_epochs}")

    # Build a one-graph-at-a-time val "dataset" by indexing the parent.
    class _IdxView:
        def __init__(self, parent, idx):
            self._p = parent
            self._idx = list(idx)
        def __len__(self):
            return len(self._idx)
        def __getitem__(self, i):
            return self._p[int(self._idx[i])]
    val_view = _IdxView(full_train_ds, val_idx)

    # Track best-epoch state for BOTH encoder and head when joint-training,
    # so the restored checkpoint is consistent.
    best_qe_state = copy.deepcopy(qe.state_dict())
    best_enc_state = (copy.deepcopy(model.state_dict())
                      if unfreeze_encoder else None)
    best_val = float("-inf")
    best_epoch = -1
    history: list[dict] = []

    for epoch in range(cfg.query_epochs):
        qe.train()
        if unfreeze_encoder:
            model.train()
        order = train_idx[rng.permutation(len(train_idx))]
        rank_accs_epoch: list[float] = []
        for batch_idx in order:
            sample = _sample_to_device(full_train_ds[int(batch_idx)], device)
            if unfreeze_encoder:
                node_emb = _encode(model, sample)
            else:
                with torch.no_grad():
                    node_emb = _encode(model, sample)
            if cfg.stage_b_head == "bilinear":
                scores = qe(sample.query, node_emb)
                if cfg.stage_b_loss == "pairwise":
                    loss, diag = bilinear_pairwise_loss(
                        scores, sample.labels, margin=cfg.margin,
                        n_pairs=cfg.stage_b_n_pairs,
                        pos_threshold=cfg.stage_b_pos_threshold,
                        neg_threshold=cfg.stage_b_neg_threshold,
                    )
                elif cfg.stage_b_loss == "listwise":
                    loss, diag = bilinear_listwise_loss(
                        scores, sample.labels, temperature=sb_temp,
                    )
                else:
                    raise ValueError(
                        f"stage_b_head=bilinear + stage_b_loss="
                        f"{cfg.stage_b_loss!r} not supported")
            else:
                q_point = qe(sample.query)
                if cfg.stage_b_loss == "pairwise":
                    loss, diag = pairwise_ranking_loss(
                        query_point=q_point, node_embeddings=node_emb,
                        labels=sample.labels, c=c_val, margin=cfg.margin,
                        n_pairs=cfg.stage_b_n_pairs,
                        pos_threshold=cfg.stage_b_pos_threshold,
                        neg_threshold=cfg.stage_b_neg_threshold,
                        euclidean=euclidean,
                    )
                elif cfg.stage_b_loss == "listwise":
                    loss, diag = listwise_ranking_loss(
                        query_point=q_point, node_embeddings=node_emb,
                        labels=sample.labels, c=c_val, temperature=sb_temp,
                        euclidean=euclidean,
                    )
                elif cfg.stage_b_loss == "infonce":
                    loss, diag = sampled_infonce_ranking_loss(
                        query_point=q_point, node_embeddings=node_emb,
                        labels=sample.labels, c=c_val,
                        n_negatives=cfg.stage_b_negatives,
                        temperature=sb_temp,
                        n_positives=cfg.stage_b_n_positives,
                        pos_threshold=cfg.stage_b_pos_threshold,
                        neg_threshold=cfg.stage_b_neg_threshold,
                        euclidean=euclidean,
                    )
                else:
                    raise ValueError(f"unknown stage_b_loss: {cfg.stage_b_loss!r}")

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[stage-B*] non-finite loss at epoch {epoch}")
            rank_accs_epoch.append(float(diag.get("rank_accuracy", float("nan"))))

        # End of epoch: eval on val split (encoder + head in eval mode).
        qe.eval()
        if unfreeze_encoder:
            model.eval()
        val_summary = _eval(cfg, model, qe, val_view, device)
        val_n10 = float(val_summary["overall"]["ndcg@10"])
        train_ra = (float(np.nanmean(rank_accs_epoch))
                    if rank_accs_epoch else float("nan"))
        print(f"[stage-B*] epoch {epoch}  train_rank_acc_mean={train_ra:.3f}  "
              f"val_ndcg@10={val_n10:.4f}"
              + ("   <-- best" if val_n10 > best_val else ""))
        history.append({"epoch": epoch, "train_rank_acc_mean": train_ra,
                        "val_ndcg@10": val_n10})
        if val_n10 > best_val:
            best_val = val_n10
            best_epoch = epoch
            best_qe_state = copy.deepcopy(qe.state_dict())
            if unfreeze_encoder:
                best_enc_state = copy.deepcopy(model.state_dict())

    # Restore best-epoch weights (both encoder and head if joint-trained).
    qe.load_state_dict(best_qe_state)
    if unfreeze_encoder and best_enc_state is not None:
        model.load_state_dict(best_enc_state)
    print(f"[stage-B*] restored epoch {best_epoch}  "
          f"(best val ndcg@10 = {best_val:.4f}, mode={mode})")
    for p in model.parameters():
        p.requires_grad = True
    return {"epochs": cfg.query_epochs, "best_epoch": best_epoch,
            "best_val_ndcg10": best_val, "history": history}


def _finetune_one(corpus: str, ckpt_dir: str, query_epochs: int | None,
                  test_files: list, task_type: int = 2,
                  use_best_epoch: bool = True,
                  unfreeze_encoder: bool = False):
    """Fresh QueryToBall trained on real TRAIN split with the checkpoint's
    encoder FROZEN; evaluated on the disjoint real TEST split.

    ``task_type`` filters the (CorpusDataset) index to one task type so a
    fresh, task-specific head is trained per task. Without this filter the
    Stage-B trainer mixes query types and the small head can't learn five
    distinct query-to-anchor mappings simultaneously.

    Returns (frozen_head_overall, adapted_head_overall, cfg.model). The
    "frozen" row is the checkpoint's own synthetic-trained query head on
    the same held-out test files (before/after for this encoder). The
    encoder (Stage-A competence) is never updated -- only the small task
    adapter -- which respects "real graphs are eval-only" for the
    reasoning model while probing whether per-task signal is recoverable
    from the frozen real-graph embeddings.
    """
    import dataclasses
    import json
    import tempfile
    import torch
    from src.data.corpus_dataset import CorpusDataset
    from src.modelsv3.query_encoder import QueryToBall
    from src.training.train_v3 import (
        Config, _build_encoder, _stage_b, _eval,
    )

    sc = json.loads((Path(ckpt_dir) / "summary.json").read_text())["config"]
    valid = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in sc.items() if k in valid})
    cfg.corpus = corpus
    cfg.task = task_type
    cfg.skip_stage_a = True
    cfg.load_encoder = str(Path(ckpt_dir) / "encoder.pt")
    if query_epochs is not None:
        cfg.query_epochs = query_epochs
    cfg.out = tempfile.mkdtemp(prefix="ftune_")
    device = torch.device("cpu")

    train_ds = CorpusDataset(corpus_dir=corpus, split="train", split_seed=0,
                             include_tasks={task_type})
    test_ds = CorpusDataset(corpus_dir=corpus, split="test", split_seed=0,
                            include_tasks={task_type})

    enc = _build_encoder(cfg, train_ds).to(device)
    enc.load_state_dict(torch.load(cfg.load_encoder, map_location=device))
    enc.eval()
    euclidean = cfg.model == "euclidean"

    # before: the checkpoint's own synthetic-trained head on held-out test
    qe_frozen = QueryToBall(
        query_dim=train_ds.query_dim, hidden_dim=cfg.hidden_dim,
        c=cfg.curvature, euclidean=euclidean,
        arch=cfg.query_head_arch, norm=cfg.query_head_norm,
    ).to(device)
    qe_frozen.load_state_dict(
        torch.load(Path(ckpt_dir) / "query_encoder.pt", map_location=device))
    qe_frozen.eval()
    frozen = _eval(cfg, enc, qe_frozen, test_ds, device)["overall"]

    # after: a FRESH head trained on the real TRAIN split (encoder frozen).
    # use_best_epoch=True splits 90/10 train/val on the train slice, runs
    # _eval on the val split each epoch, and restores the query-head state
    # at the epoch with best val ndcg@10 -- avoids the final-epoch
    # pessimism observed in the task-2 trace (peak epoch ~4, then drift).
    qe = QueryToBall(
        query_dim=train_ds.query_dim, hidden_dim=cfg.hidden_dim,
        c=cfg.curvature, euclidean=euclidean,
        arch=cfg.query_head_arch, norm=cfg.query_head_norm,
    ).to(device)
    if use_best_epoch:
        _stage_b_best_epoch(cfg, enc, qe, train_ds, device,
                            unfreeze_encoder=unfreeze_encoder)
    else:
        if unfreeze_encoder:
            raise ValueError("unfreeze_encoder requires use_best_epoch=True "
                             "(vanilla _stage_b ignores the flag)")
        _stage_b(cfg, enc, qe, train_ds, device)
    adapted = _eval(cfg, enc, qe, test_ds, device)["overall"]
    return frozen, adapted, cfg.model


def cmd_finetune(args) -> int:
    _repo_on_path()
    from src.data.corpus_dataset import CorpusDataset

    # Per-task counts on the test split (drives which tasks we adapt for).
    full_test = CorpusDataset(corpus_dir=args.corpus, split="test",
                              split_seed=0)
    test_files = list(full_test.files)
    type_counts: dict[int, int] = {}
    for f in test_files:
        with np.load(f) as npz:
            for j in range(int(npz["n_tasks"])):
                t = int(npz[f"task_{j}_type"])
                type_counts[t] = type_counts.get(t, 0) + 1
    print(f"held-out TEST split: {len(test_files)} graphs / "
          f"{sum(type_counts.values())} task instances "
          f"[{', '.join(f'{_TASK_NAMES.get(t,t)}={c}' for t,c in sorted(type_counts.items()))}]\n")

    # Which task types to adapt for. ER (1) has no real-graph generator.
    selected_types = sorted(t for t in type_counts if t in {0, 2, 3, 4, 5}
                            and type_counts[t] >= 4)
    if args.task_types:
        selected_types = [int(t) for t in args.task_types
                          if int(t) in selected_types]
    print(f"adapting heads for tasks: {selected_types}\n")

    # Heuristics (task-aware via _score_heuristic).
    heur_summary = _score_heuristic(test_files, "anchor")
    rand_summary = _score_heuristic(test_files, "random")
    oracle_summary = _score_heuristic(test_files, "oracle")

    # Per-task adapted-head numbers per arm.
    # arm_results[arm_label][task_type] = {"frozen": agg, "adapted": agg}
    arm_results: dict[str, dict[int, dict[str, dict]]] = {}

    def _arm(label: str, dirs: list[str]):
        arm_results[label] = {}
        for t in selected_types:
            print(f"  [{label}] task {t} ({_TASK_NAMES.get(t, t)}) ...")
            froz, adap = [], []
            for d in dirs:
                if not (Path(d) / "encoder.pt").is_file():
                    continue
                try:
                    fr, ad, _ = _finetune_one(
                        args.corpus, d, args.query_epochs, test_files,
                        task_type=t,
                        unfreeze_encoder=args.unfreeze_encoder,
                    )
                except Exception as e:
                    print(f"    (skip {d}: {e!r})")
                    continue
                froz.append(fr); adap.append(ad)
            if froz:
                arm_results[label][t] = {
                    "frozen": _agg(froz, _KEYS),
                    "adapted": _agg(adap, _KEYS),
                }

    selected_arms = args.arms or {"euclidean", "hyperbolic", "shipped"}
    if "euclidean" in selected_arms:
        _arm(f"euclidean (h64/l3, n={len(_EUC_SEEDS)})", _EUC_SEEDS)
    if "hyperbolic" in selected_arms:
        _arm(f"hyperbolic (h64/l3, n={len(_HYP_SEEDS)})", _HYP_SEEDS)
    if "shipped" in selected_arms:
        _arm("hyperbolic shipped (h128/l4)", [_SHIPPED])

    # Print one block per task: random / anchor / oracle / frozen / adapted.
    show = ["ndcg@10", "p@10", "r@10", "ndcg@5", "ndcg@20"]

    for t in selected_types:
        name = _TASK_NAMES.get(t, str(t))
        rows: list[tuple[str, dict]] = []
        rows.append(("random (floor)",
                     {k: (v, 0.0) for k, v in
                      rand_summary["by_task_type"].get(t, {}).items()}))
        rows.append(("anchor-BFS (heuristic)",
                     {k: (v, 0.0) for k, v in
                      heur_summary["by_task_type"].get(t, {}).items()}))
        rows.append(("oracle (per-task label rule)",
                     {k: (v, 0.0) for k, v in
                      oracle_summary["by_task_type"].get(t, {}).items()}))
        for arm_label in arm_results:
            cell = arm_results[arm_label].get(t)
            if not cell:
                continue
            rows.append((f"{arm_label} — frozen head (synthetic)",
                         cell["frozen"]))
            rows.append((f"{arm_label} — adapted head (real)",
                         cell["adapted"]))

        if not rows:
            print(f"=== Task {t} ({name}) — no data ===\n")
            continue
        w = max(len(lbl) for lbl, _ in rows)
        print(f"=== Task {t} ({name}) — {type_counts.get(t, 0)} instances ===")
        hdr = f"{'method':<{w}}  " + "  ".join(f"{k:>14}" for k in show)
        print(hdr)
        print("-" * len(hdr))
        for lbl, m in rows:
            cells = []
            for k in show:
                if k not in m:
                    cells.append("--")
                    continue
                mu, sd = m[k]
                cells.append(f"{mu:.4f}±{sd:.3f}" if sd else f"{mu:.4f}      ")
            print(f"{lbl:<{w}}  " + "  ".join(f"{c:>14}" for c in cells))
        print()

    # Headline: did adaptation close the gap to anchor-BFS on any task?
    # Pick the best-named arm that actually ran.
    hyp_label = next((l for l in arm_results if "hyperbolic (h64" in l), None)
    if hyp_label is None:
        hyp_label = next((l for l in arm_results if "shipped" in l), None)
    if hyp_label is None:
        hyp_label = next(iter(arm_results), None)
    if hyp_label:
        print(f"=== Adaptation summary (ndcg@10, {hyp_label}) ===")
        print(f"{'task':<14} {'anchor':>8} {'oracle':>8} {'frozen':>8} "
              f"{'adapted':>9} {'lift':>7} {'gap_to_anchor':>13}")
        for t in selected_types:
            cell = arm_results[hyp_label].get(t)
            if not cell:
                continue
            anc = heur_summary["by_task_type"].get(t, {}).get("ndcg@10", 0.0)
            orc = oracle_summary["by_task_type"].get(t, {}).get("ndcg@10", 0.0)
            froz = cell["frozen"]["ndcg@10"][0]
            adap = cell["adapted"]["ndcg@10"][0]
            lift = adap - froz
            gap = adap - anc
            print(f"{_TASK_NAMES.get(t,t):<14} {anc:>8.4f} {orc:>8.4f} "
                  f"{froz:>8.4f} {adap:>9.4f} {lift:>+7.4f} {gap:>+13.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Neo4j real-graph eval exporter")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export", help="export a real domain_only eval sample")
    pe.add_argument("--config", default="kettle_config.yaml")
    pe.add_argument("--out", required=True)
    pe.add_argument("--num-graphs", type=int, default=16)
    pe.add_argument("--max-nodes", type=int, default=400)
    pe.add_argument("--tasks-per-graph", type=int, default=3)
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--sampler", choices=("delocalized", "anchor_ball"),
                    default="delocalized",
                    help="delocalized = temporally-stratified multi-seed "
                         "(removes the locality shortcut); anchor_ball = "
                         "legacy single-seed BFS (locality-confounded)")
    pe.add_argument("--n-seeds", type=int, default=4,
                    help="temporal strata / seed balls per graph "
                         "(delocalized sampler only)")
    pe.set_defaults(func=cmd_export)

    ps = sub.add_parser("score", help="score an eval corpus with a checkpoint")
    ps.add_argument("--corpus", required=True)
    ps.add_argument("--checkpoint", default=_SHIPPED)
    ps.set_defaults(func=cmd_score)

    pc = sub.add_parser("compare", help="hyp vs euc vs heuristic on a corpus")
    pc.add_argument("--corpus", required=True)
    pc.set_defaults(func=cmd_compare)

    pf = sub.add_parser("finetune",
                        help="Stage-B head fine-tune on real train split, "
                             "eval on held-out test split (encoder frozen)")
    pf.add_argument("--corpus", required=True)
    pf.add_argument("--query-epochs", type=int, default=None,
                    help="override Stage-B epochs (default: checkpoint's)")
    pf.add_argument("--task-types", nargs="+", default=None,
                    help="restrict adapted-head training to these task type "
                         "ids (default: all present in {0,2,3,4,5})")
    pf.add_argument("--arms", nargs="+", default=None,
                    choices=["euclidean", "hyperbolic", "shipped"],
                    help="which checkpoint arms to fine-tune (default: all "
                         "3 -- 35 finetune runs at 5 tasks; pick 'shipped' "
                         "for the cheap 5-run pilot)")
    pf.add_argument("--unfreeze-encoder", action="store_true",
                    help="JOINT-train encoder + query head on real-graph "
                         "train split (violates eval-only commitment; tests "
                         "whether the encoder geometry itself is the cap)")
    pf.set_defaults(func=cmd_finetune)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

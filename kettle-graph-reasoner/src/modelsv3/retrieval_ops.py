r"""v3.1 Phase 5 — operational retrieval ops over a manifold index.

A small, training-free library that turns the exported
``manifold_index.npz`` (see ``export_manifold_index.py``) into the
geometric layer the future graph kernel calls:

    load_query_encoder(run_dir)       selectable QueryToBall head ->
                                      query -> query_point (ball point)
    nearest(query, k)                 query point -> k closest nodes
    nearest_node(node, k)             node -> k closest nodes (same graph)
    bridge(a, b, k)                   geodesic midpoint -> k closest nodes
    graph_far_geometry_near(...)      geometry/graph disagreement pairs
    expand_provenance(seeds, hops)    provenance-only neighbourhood
    retrieve_then_rerank(query, ...)  v3.1 retrieve -> pluggable reranker
    hybrid_retrieve_expand_rerank(..) nearest -> expand_prov -> rerank

The query head is SELECTABLE: ``load_query_encoder(run_dir)`` loads the
trained ``QueryToBall`` from any standard run dir (the locked v3.1
baseline synthetic head, or ``runs/v3.1-real-head-hyp-h128-l4-seed0``
the real-graph head) and returns ``query -> query_point`` to feed
``nearest``/``retrieve_then_rerank``. No architecture change; the graph
encoder is untouched (the index already holds node embeddings). The
head only maps a query into that encoder's ball, so the index MUST be
exported from the same (frozen, SHA-checkable) encoder the head was
trained against.

Reuses ``score_from_embeddings``, ``P.dist/expmap0/logmap0``,
``_midpoint`` and ``_bfs_hop_matrix`` — no new geometry code. The
reranker is a ``Callable[[np.ndarray], np.ndarray]`` (candidate node
rows -> scores); ``oracle_reranker`` / ``identity_reranker`` are
provided, and ``v2_reranker.py`` supplies the trained-v2 one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from ..modelsv2.layers import poincare_ops as P
from .distance_ops import pairwise_distance_matrix
from .distance_scoring import score_from_embeddings
from .eval_retrieval_midpoint import _midpoint
from .eval_retrieval_nn import _bfs_hop_matrix

EDGE_CAT_PROVENANCE = 0
Reranker = Callable[[np.ndarray], np.ndarray]


@dataclass
class ManifoldIndex:
    """Loaded ``manifold_index.npz`` + meta. Rows are aligned across all
    fields; ``embedding`` is (n_nodes, dim)."""

    embedding: np.ndarray
    graph_idx: np.ndarray
    node_idx: np.ndarray
    neo4j_node_id: np.ndarray
    radius: np.ndarray
    out_degree: np.ndarray
    in_degree: np.ndarray
    collapse_flag: np.ndarray
    node_type: np.ndarray
    layer: np.ndarray
    depth: np.ndarray
    meta: dict

    @property
    def euclidean(self) -> bool:
        return self.meta.get("model") == "euclidean"

    @property
    def c(self) -> float:
        return float(self.meta.get("curvature", 1.0))

    def graph_mask(self, gi: int) -> np.ndarray:
        return self.graph_idx == gi

    def emb_t(self, mask: Optional[np.ndarray] = None) -> torch.Tensor:
        e = self.embedding if mask is None else self.embedding[mask]
        return torch.from_numpy(np.ascontiguousarray(e)).float()


def load_index(npz_path: str | Path) -> ManifoldIndex:
    npz_path = Path(npz_path)
    z = np.load(npz_path)
    meta_path = npz_path.with_name(npz_path.stem + "_meta.json")
    meta = {}
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())
    return ManifoldIndex(
        embedding=z["embedding"], graph_idx=z["graph_idx"],
        node_idx=z["node_idx"],
        neo4j_node_id=(
            z["neo4j_node_id"] if "neo4j_node_id" in z.files
            else np.full_like(z["node_idx"], -1, dtype=np.int64)
        ),
        radius=z["radius"],
        out_degree=z["out_degree"], in_degree=z["in_degree"],
        collapse_flag=z["collapse_flag"], node_type=z["node_type"],
        layer=z["layer"], depth=z["depth"], meta=meta,
    )


# ---------------------------------------------------------------------------
# selectable query head
# ---------------------------------------------------------------------------

def load_query_encoder(run_dir: str | Path, query_dim: int = 18):
    """Load a trained ``QueryToBall`` head from ``run_dir`` as a
    SELECTABLE retrieval head. Returns a callable ``query ->
    query_point`` (a ball point) ready for ``nearest`` /
    ``retrieve_then_rerank``.

    ``run_dir`` is any standard run directory holding ``summary.json``
    (the training config) and ``query_encoder.pt`` (the trained head):
    the locked v3.1 baseline (synthetic head), or
    ``runs/v3.1-real-head-hyp-h128-l4-seed0`` (the real-graph head),
    etc. The graph encoder is deliberately NOT loaded here — the
    manifold index already holds node embeddings; the head only maps a
    query into that encoder's ball.

    INVARIANT: the index passed to ``nearest`` must have been exported
    from the SAME (frozen) encoder this head was trained against. That
    is SHA-checkable (``lock_baseline.sha256_file`` on the run dirs'
    ``encoder.pt``); the v3.1 baseline and the real-head share a
    byte-identical encoder, so the baseline ``manifold_index.npz`` is
    valid for either head. Read-only: the head is ``eval()``-d and
    frozen, and the v3 architecture is unchanged — this is wiring, not a
    model edit.

    ``query_dim`` defaults to the corpus constant (18); it is the only
    thing ``build_query_encoder`` needs from a dataset, so a tiny shim
    is passed instead of coupling this training-free lib to a corpus.
    """
    import json
    from types import SimpleNamespace

    from .eval_candidate_recall import build_query_encoder

    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "summary.json").read_text())["config"]
    qenc = build_query_encoder(cfg, SimpleNamespace(query_dim=query_dim))
    qenc.load_state_dict(
        torch.load(run_dir / "query_encoder.pt", map_location="cpu"))
    qenc.eval()
    for p in qenc.parameters():
        p.requires_grad = False

    def _to_point(query) -> torch.Tensor:
        q = (query if isinstance(query, torch.Tensor)
             else torch.as_tensor(np.asarray(query), dtype=torch.float32))
        with torch.no_grad():
            return qenc(q.float())

    return _to_point


# ---------------------------------------------------------------------------
# distance helpers
# ---------------------------------------------------------------------------

def _dist_to_point(emb: torch.Tensor, point: torch.Tensor,
                   c: float, euclidean: bool) -> np.ndarray:
    """Distance from every row of ``emb`` to a single ``point`` (D,)."""
    scores = score_from_embeddings(emb, point, c=c, euclidean=euclidean)
    return (-scores).detach().cpu().numpy()  # score = -dist


def _topk(values: np.ndarray, k: int, largest: bool,
          exclude: Optional[set[int]] = None) -> list[int]:
    order = np.argsort(values)
    if largest:
        order = order[::-1]
    out: list[int] = []
    for idx in order:
        i = int(idx)
        if exclude and i in exclude:
            continue
        out.append(i)
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------

def nearest(index: ManifoldIndex, query_point: torch.Tensor, k: int = 10,
            graph_idx: Optional[int] = None,
            drop_collapsed: bool = True) -> list[int]:
    """Global row indices of the k nodes closest to ``query_point``
    (a ball point from ``QueryToBall``). Restrict to one graph with
    ``graph_idx``. With ``drop_collapsed`` (default), the redundant
    members of near-duplicate clusters are skipped — one representative
    per cluster survives (plan §7), so top-k is not wasted on copies."""
    if graph_idx is None:
        rows = np.arange(len(index.graph_idx))
    else:
        rows = np.where(index.graph_mask(graph_idx))[0]
    emb = torch.from_numpy(
        np.ascontiguousarray(index.embedding[rows])).float()
    d = _dist_to_point(emb, query_point, index.c, index.euclidean)
    excl = {int(j) for j in np.where(index.collapse_flag[rows])[0]} \
        if drop_collapsed else None
    local = _topk(d, k, largest=False, exclude=excl)
    return [int(rows[i]) for i in local]


def nearest_node(index: ManifoldIndex, graph_idx: int, node_idx: int,
                 k: int = 10) -> list[int]:
    """k nearest nodes to a given node, within the same graph."""
    rows = np.where(index.graph_mask(graph_idx))[0]
    local_seed = int(np.where(index.node_idx[rows] == node_idx)[0][0])
    emb = torch.from_numpy(np.ascontiguousarray(index.embedding[rows])).float()
    seed = emb[local_seed]
    d = _dist_to_point(emb, seed, index.c, index.euclidean)
    local = _topk(d, k, largest=False, exclude={local_seed})
    return [int(rows[i]) for i in local]


def bridge(index: ManifoldIndex, graph_idx: int, a: int, b: int,
           k: int = 5) -> list[int]:
    """k nodes nearest the geodesic midpoint of nodes ``a`` and ``b``
    (the 'what lies between A and B?' op). Endpoints excluded."""
    rows = np.where(index.graph_mask(graph_idx))[0]
    emb = torch.from_numpy(np.ascontiguousarray(index.embedding[rows])).float()
    la = int(np.where(index.node_idx[rows] == a)[0][0])
    lb = int(np.where(index.node_idx[rows] == b)[0][0])
    m = _midpoint(emb[la], emb[lb], index.c, index.euclidean)
    d = _dist_to_point(emb, m, index.c, index.euclidean)
    local = _topk(d, k, largest=False, exclude={la, lb})
    return [int(rows[i]) for i in local]


def graph_far_geometry_near(index: ManifoldIndex, edge_index: np.ndarray,
                            graph_idx: int, q: float = 5.0,
                            max_pairs: int = 50) -> list[tuple[int, int]]:
    """Node pairs that are geometrically close but graph-distant — the
    'geometry says similar, graph says far' diagnostic pairs an
    operational system would surface (missing edges / bad resolution).
    Returns up to ``max_pairs`` (node_idx, node_idx) tuples."""
    rows = np.where(index.graph_mask(graph_idx))[0]
    emb = torch.from_numpy(np.ascontiguousarray(index.embedding[rows])).float()
    N = emb.size(0)
    # Cap-guarded full matrix: bit-identical to the old inline path at
    # N <= cap so the returned (i,j) pairs are unchanged; loud error
    # > cap (this op returns specific pairs — sampling would change them).
    D = pairwise_distance_matrix(emb, index.c, index.euclidean).numpy()
    hop = _bfs_hop_matrix(edge_index, N)
    iu, ju = np.triu_indices(N, k=1)
    h = hop[iu, ju]
    reach = h >= 0
    g = D[iu, ju][reach]
    h = h[reach].astype(float)
    ii, jj = iu[reach], ju[reach]
    if g.size == 0:
        return []
    g_lo = np.percentile(g, q)
    h_hi = np.percentile(h, 100 - q)
    sel = np.where((g <= g_lo) & (h >= h_hi))[0]
    sel = sel[np.argsort(g[sel])][:max_pairs]
    return [(int(index.node_idx[rows][ii[s]]),
             int(index.node_idx[rows][jj[s]])) for s in sel]


def expand_provenance(edge_index: np.ndarray, edge_type: np.ndarray,
                       edge_descriptor: np.ndarray, seeds: list[int],
                       hops: int = 1) -> set[int]:
    """``hops``-hop neighbourhood of ``seeds`` along provenance edges
    only (edge category 0 == EDGE_CAT_PROVENANCE), undirected. The
    exact-graph expansion that surrounds geometric candidates with
    their source/derivation chain before reranking."""
    cat = edge_descriptor[:, 0:4].argmax(axis=1)
    edge_cat = cat[edge_type]
    mask = edge_cat == EDGE_CAT_PROVENANCE
    src, dst = edge_index[0][mask], edge_index[1][mask]
    adj: dict[int, list[int]] = {}
    for s, d in zip(src, dst):
        adj.setdefault(int(s), []).append(int(d))
        adj.setdefault(int(d), []).append(int(s))
    frontier = set(int(s) for s in seeds)
    seen = set(frontier)
    for _ in range(hops):
        nxt: set[int] = set()
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in seen:
                    nxt.add(v)
        seen |= nxt
        frontier = nxt
        if not frontier:
            break
    return seen


def retrieve_then_rerank(index: ManifoldIndex, query_point: torch.Tensor,
                         reranker: Reranker, graph_idx: int,
                         C: int = 50, k: int = 10) -> list[int]:
    """v3.1 first-stage retrieve (top-C by hyperbolic distance) then
    reorder by ``reranker``. ``reranker`` maps the candidate row indices
    (global) to scores; higher = more relevant. Returns top-k rows."""
    cand = nearest(index, query_point, k=C, graph_idx=graph_idx)
    if not cand:
        return []
    scores = np.asarray(reranker(np.asarray(cand, dtype=np.int64)),
                        dtype=np.float64)
    order = np.argsort(scores)[::-1][:k]
    return [int(cand[o]) for o in order]


def hybrid_retrieve_expand_rerank(
    index: ManifoldIndex, query_point: torch.Tensor, reranker: Reranker,
    graph_idx: int, edge_index: np.ndarray, edge_type: np.ndarray,
    edge_descriptor: np.ndarray, C: int = 50, k: int = 10,
    expand_hops: int = 1,
) -> list[int]:
    """nearest -> provenance expansion -> rerank. The product path: a
    geometric candidate set, widened with its provenance chain, then
    finished by the (v2) reranker."""
    rows = np.where(index.graph_mask(graph_idx))[0]
    cand_rows = nearest(index, query_point, k=C, graph_idx=graph_idx)
    cand_nodes = [int(index.node_idx[r]) for r in cand_rows]
    expanded_nodes = expand_provenance(
        edge_index, edge_type, edge_descriptor, cand_nodes, hops=expand_hops)
    node_to_row = {int(index.node_idx[r]): int(r) for r in rows}
    pool = [node_to_row[n] for n in expanded_nodes if n in node_to_row]
    if not pool:
        return []
    scores = np.asarray(reranker(np.asarray(pool, dtype=np.int64)),
                        dtype=np.float64)
    order = np.argsort(scores)[::-1][:k]
    return [int(pool[o]) for o in order]


# ---------------------------------------------------------------------------
# reranker factories
# ---------------------------------------------------------------------------

def identity_reranker(index: ManifoldIndex,
                      query_point: torch.Tensor,
                      graph_idx: int) -> Reranker:
    """No-op reranker: preserves the v3.1 distance order (score =
    -dist). Lets the hybrid harness measure 'v3.1 alone'."""
    rows = np.where(index.graph_mask(graph_idx))[0]
    emb = torch.from_numpy(np.ascontiguousarray(index.embedding[rows])).float()
    row_to_local = {int(r): i for i, r in enumerate(rows)}
    d = _dist_to_point(emb, query_point, index.c, index.euclidean)

    def _rr(cand_rows: np.ndarray) -> np.ndarray:
        return np.array([-d[row_to_local[int(r)]] for r in cand_rows])

    return _rr


def oracle_reranker(labels_by_row: dict[int, float]) -> Reranker:
    """Upper-bound reranker: true relevance labels. Quantifies the gap a
    real reranker could close on the v3.1 candidate set."""

    def _rr(cand_rows: np.ndarray) -> np.ndarray:
        return np.array([labels_by_row.get(int(r), 0.0) for r in cand_rows])

    return _rr


def order_ball(node_emb, anchor_row: int, ball_rows, c=1.0):
    """Order a retrieval ball by hyperbolic distance to the anchor node.

    The deployable form of the 2026-07-07 capability finding
    (runs/probe_capability_ballrank): ordering the BFS/Cypher ball around
    the query anchor by emb-distance is near-oracle on the real archival
    graph (ndcg@10 ALL 0.885 vs hop-order 0.690; provenance 0.999,
    subgraph 0.986), zero training, no heads. Validated on a fresh
    live-Neo4j export (0.840 vs 0.657).

    Parameters: node_emb (N, D) frozen-encoder ball-point embeddings;
    anchor_row index into node_emb; ball_rows iterable of candidate row
    indices (the Cypher/BFS neighborhood). Returns ball_rows sorted most-
    relevant-first. Use hop-order with this as tie-break if a structural
    prior is wanted (hop_tb_emb scored 0.840-1.000 per family).
    """
    import torch as _t
    rows = _t.as_tensor(list(ball_rows), dtype=_t.long)
    sc = score_from_embeddings(node_emb[rows], node_emb[anchor_row], c=c)
    return [int(r) for r in rows[_t.argsort(sc, descending=True)]]


def multi_anchor_order(node_emb, anchor_rows, ball_rows, c=1.0):
    """Order a retrieval ball by min hyperbolic distance to ANY anchor
    (max-score union of the anchors' balls).

    The deployable form of the verified multi-anchor composition: with an
    informative 2nd anchor the union arm scored +0.471 ndcg@10 over
    single-anchor ordering (stress_multi_anchor 2026-07-10; replicated
    bit-for-bit in runs/geometry_probes/p4_multi_anchor, where the Karcher
    mean / geodesic midpoint challengers reached only +0.12/+0.07 and were
    rejected). Caveats carried from the probes: anchors must be
    INFORMATIVE — a random 2nd anchor is neutral-to-harmful (refute
    slices) — so callers pass user-supplied or task-derived anchors only.

    Parameters: node_emb (N, D) frozen ball-point embeddings; anchor_rows
    iterable of anchor row indices; ball_rows iterable of candidate rows.
    Returns ball_rows sorted most-relevant-first. With one anchor this is
    exactly ``order_ball``.
    """
    import torch as _t
    rows = _t.as_tensor(list(ball_rows), dtype=_t.long)
    sc = _t.stack([
        score_from_embeddings(node_emb[rows], node_emb[int(a)], c=c)
        for a in anchor_rows
    ]).max(dim=0).values
    return [int(r) for r in rows[_t.argsort(sc, descending=True)]]

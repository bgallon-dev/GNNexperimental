"""
graphcache.py -- one-pull, numpy-backed graph cache for the topology check.

Why this exists
---------------
The old topology check streamed the whole graph (and every named subgraph)
into NetworkX, repeatedly, and evaluated temporal ``[*1..3]`` variable-length
paths *per edge* in Cypher WHERE clauses. On a 327k-node graph that ran
overnight without finishing.

This module pulls the lifecycle-clean graph **once** into compact numpy
arrays (a CSR adjacency + per-node label table + an edge-type vector) and
derives every per-scope metric -- degree stats, exact connected components,
per-subgraph node/edge counts, and a sampled Gromov delta -- from boolean
**masks over that single cache**. No subgraph touches Neo4j again; the
temporal split is answered from a Year-reachability table computed once by
bounded label propagation outward from the (few) Year nodes, which is
correct for both a 1-hop and an up-to-K-hop linkage convention.

Everything here is pure numpy (no scipy / igraph / graph-tool). BFS is
vectorized via the standard ragged-gather expansion so a single traversal
is O(V + E) array ops rather than a Python per-node loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

NodeFilter = str  # a Cypher predicate string (from core.lifecycle_predicate)


# ---------------------------------------------------------------------------
# Vectorized graph primitives (operate on an undirected CSR)
# ---------------------------------------------------------------------------

def _ragged_gather(indptr: np.ndarray, indices: np.ndarray,
                   frontier: np.ndarray) -> np.ndarray:
    """Return all neighbor node-ids of every node in ``frontier``.

    Classic vectorized ragged expansion: build the flat list of CSR slot
    positions for the frontier without a Python loop, then index ``indices``.
    """
    if frontier.size == 0:
        return frontier
    starts = indptr[frontier]
    counts = indptr[frontier + 1] - starts
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=indices.dtype)
    # offset[i] = position within the i-th frontier node's neighbor block
    seg = np.repeat(counts.cumsum() - counts, counts)
    offset = np.arange(total) - seg
    slot = np.repeat(starts, counts) + offset
    return indices[slot]


def bfs(indptr: np.ndarray, indices: np.ndarray, source: int,
        n: int, allowed: np.ndarray | None = None) -> np.ndarray:
    """Single-source BFS over a CSR. Returns int32 distances (-1 = unreached).

    If ``allowed`` (bool, length n) is given, traversal only enters nodes
    where allowed[node] is True -- used for induced-subgraph distances.
    """
    dist = np.full(n, -1, dtype=np.int32)
    if allowed is not None and not allowed[source]:
        return dist
    dist[source] = 0
    frontier = np.array([source], dtype=np.int64)
    d = 0
    while frontier.size:
        d += 1
        nbrs = _ragged_gather(indptr, indices, frontier)
        if nbrs.size == 0:
            break
        nbrs = np.unique(nbrs)
        if allowed is not None:
            nbrs = nbrs[allowed[nbrs]]
        nbrs = nbrs[dist[nbrs] < 0]
        if nbrs.size == 0:
            break
        dist[nbrs] = d
        frontier = nbrs
    return dist


def connected_components(indptr: np.ndarray, indices: np.ndarray, n: int,
                         allowed: np.ndarray | None = None) -> np.ndarray:
    """Exact weakly-connected component labels via repeated BFS.

    O(V + E) total: each node is enqueued once. ``allowed`` restricts to an
    induced subgraph (component ids are only meaningful for allowed nodes).
    """
    comp = np.full(n, -1, dtype=np.int64)
    order = np.arange(n, dtype=np.int64)
    if allowed is not None:
        order = order[allowed]
    cid = 0
    for s in order:
        if comp[s] != -1:
            continue
        # iterative flood from s
        comp[s] = cid
        frontier = np.array([s], dtype=np.int64)
        while frontier.size:
            nbrs = _ragged_gather(indptr, indices, frontier)
            if nbrs.size:
                nbrs = np.unique(nbrs)
                if allowed is not None:
                    nbrs = nbrs[allowed[nbrs]]
                nbrs = nbrs[comp[nbrs] == -1]
                comp[nbrs] = cid
                frontier = nbrs
            else:
                break
        cid += 1
    return comp


def snowball_sample(indptr: np.ndarray, indices: np.ndarray, n: int,
                    allowed: np.ndarray, seed_node: int,
                    cap: int) -> np.ndarray:
    """Connected BFS sample of <= cap nodes around seed_node within allowed."""
    picked = np.zeros(n, dtype=bool)
    picked[seed_node] = True
    count = 1
    frontier = np.array([seed_node], dtype=np.int64)
    while frontier.size and count < cap:
        nbrs = _ragged_gather(indptr, indices, frontier)
        if nbrs.size == 0:
            break
        nbrs = np.unique(nbrs)
        nbrs = nbrs[allowed[nbrs] & ~picked[nbrs]]
        if nbrs.size == 0:
            break
        room = cap - count
        if nbrs.size > room:
            nbrs = nbrs[:room]
        picked[nbrs] = True
        count += nbrs.size
        frontier = nbrs
    return np.flatnonzero(picked)


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------

@dataclass
class GraphCache:
    """Compact numpy view of the lifecycle-clean graph, pulled once.

    Built by :func:`build_cache`. All per-scope topology metrics are derived
    from boolean masks over these arrays -- no further Neo4j round-trips.
    """

    n: int
    id2idx: dict[int, int]     # neo4j node id -> contiguous cache index
    # per-node label table, CSR-style: label ids of node i are
    # lab_ids[lab_indptr[i]:lab_indptr[i+1]]
    label_names: list[str]
    lab_indptr: np.ndarray
    lab_ids: np.ndarray
    # undirected edges (deduplicated via id(a) < id(b)), idx-space endpoints
    src: np.ndarray            # int32
    dst: np.ndarray            # int32
    etype: np.ndarray          # int32 edge-type id
    etype_names: list[str]
    # undirected CSR over all cached edges
    indptr: np.ndarray
    indices: np.ndarray
    # filled lazily by ensure_year_reachability(); keyed by (label, prop, hops)
    _year_cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = None  # type: ignore

    # -- label masks ----------------------------------------------------------

    def label_mask(self, include_labels: list[str]) -> np.ndarray:
        """Bool[n]: node has at least one label in include_labels.

        Empty include_labels => all True (the spec convention for "all").
        """
        if not include_labels:
            return np.ones(self.n, dtype=bool)
        name_to_id = {nm: i for i, nm in enumerate(self.label_names)}
        target = np.array(
            [name_to_id[l] for l in include_labels if l in name_to_id],
            dtype=self.lab_ids.dtype,
        )
        mask = np.zeros(self.n, dtype=bool)
        if target.size == 0:
            return mask
        hit = np.isin(self.lab_ids, target)
        if hit.any():
            counts = self.lab_indptr[1:] - self.lab_indptr[:-1]
            node_of_slot = np.repeat(np.arange(self.n), counts)
            np.logical_or.at(mask, node_of_slot[hit], True)
        return mask

    # -- edge masks -----------------------------------------------------------

    def edge_mask(self, node_mask: np.ndarray,
                  include_rel_types: list[str] | None,
                  exclude_rel_types: list[str] | None) -> np.ndarray:
        """Bool[E]: both endpoints in node_mask and rel-type passes filters."""
        m = node_mask[self.src] & node_mask[self.dst]
        name_to_id = {nm: i for i, nm in enumerate(self.etype_names)}
        if include_rel_types:
            inc = np.array([name_to_id[t] for t in include_rel_types
                            if t in name_to_id], dtype=self.etype.dtype)
            m &= np.isin(self.etype, inc) if inc.size else np.zeros_like(m)
        if exclude_rel_types:
            exc = np.array([name_to_id[t] for t in exclude_rel_types
                            if t in name_to_id], dtype=self.etype.dtype)
            if exc.size:
                m &= ~np.isin(self.etype, exc)
        return m

    def induced_csr(self, edge_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Undirected CSR restricted to the edges in edge_mask."""
        s = self.src[edge_mask]
        d = self.dst[edge_mask]
        u = np.concatenate([s, d])
        v = np.concatenate([d, s])
        order = np.argsort(u, kind="stable")
        u, v = u[order], v[order]
        indptr = np.zeros(self.n + 1, dtype=np.int64)
        np.add.at(indptr, u + 1, 1)
        np.cumsum(indptr, out=indptr)
        return indptr, v.astype(np.int64)

    # -- Year reachability (temporal split) -----------------------------------

    def ensure_year_reachability(
        self, ylabel: str, prop_values: dict[int, float], max_hops: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute, per node, (min_year, max_year, has_reachable_year).

        Bounded label propagation outward from the year-labelled nodes:
        ``max_hops`` sparse passes over the edge list. ``prop_values`` maps
        node idx -> numeric year for year-labelled, lifecycle-clean nodes.
        Returns (min_y[n] float, max_y[n] float, has[n] bool); min/max are
        +inf / -inf where no year node is within max_hops.
        """
        if self._year_cache is None:
            self._year_cache = {}
        key = (ylabel, max_hops, len(prop_values))
        if key in self._year_cache:
            return self._year_cache[key]

        min_y = np.full(self.n, np.inf, dtype=np.float64)
        max_y = np.full(self.n, -np.inf, dtype=np.float64)
        if prop_values:
            idx = np.fromiter(prop_values.keys(), dtype=np.int64,
                              count=len(prop_values))
            val = np.fromiter(prop_values.values(), dtype=np.float64,
                              count=len(prop_values))
            min_y[idx] = val
            max_y[idx] = val

        for _ in range(max(0, max_hops)):
            # relax along both directions of every undirected edge
            cand_min = np.minimum(min_y[self.src], min_y[self.dst])
            cand_max = np.maximum(max_y[self.src], max_y[self.dst])
            new_min = min_y.copy()
            new_max = max_y.copy()
            np.minimum.at(new_min, self.src, cand_min)
            np.minimum.at(new_min, self.dst, cand_min)
            np.maximum.at(new_max, self.src, cand_max)
            np.maximum.at(new_max, self.dst, cand_max)
            if np.array_equal(new_min, min_y) and np.array_equal(new_max, max_y):
                break
            min_y, max_y = new_min, new_max

        has = np.isfinite(min_y)
        self._year_cache[key] = (min_y, max_y, has)
        return min_y, max_y, has

    def temporal_mask(self, base_mask: np.ndarray, is_ylabel: np.ndarray,
                      min_y: np.ndarray, max_y: np.ndarray, has: np.ndarray,
                      comparison: str, cutoff: float) -> np.ndarray:
        """Apply the original temporal_filter semantics, vectorized.

        A node passes iff it IS a year node, OR has no reachable year, OR
        some reachable year satisfies ``year <comp> cutoff``. For < <= > >=
        this is exact via min/max; for = / != it is a documented range
        approximation (the Kettle config only uses < and >=).
        """
        if comparison == "<":
            sat = min_y < cutoff
        elif comparison == "<=":
            sat = min_y <= cutoff
        elif comparison == ">":
            sat = max_y > cutoff
        elif comparison == ">=":
            sat = max_y >= cutoff
        elif comparison == "=":
            sat = (min_y <= cutoff) & (max_y >= cutoff)        # approx
        elif comparison == "!=":
            sat = ~((min_y == cutoff) & (max_y == cutoff))     # approx
        else:
            raise ValueError(f"Invalid temporal comparison: {comparison!r}")
        return base_mask & (is_ylabel | ~has | sat)


# ---------------------------------------------------------------------------
# Builder -- the single graph pull
# ---------------------------------------------------------------------------

def build_cache(session, lifecycle_pred: str,
                progress: Callable[[str], None] | None = None) -> GraphCache:
    """Stream the lifecycle-clean graph once and build a GraphCache.

    Two queries only:
      1. nodes:  id + labels   (lifecycle predicate on n)
      2. edges:  id(a),id(b),type  (id(a)<id(b), lifecycle on both endpoints)

    Everything downstream (full-graph stats, components, per-subgraph stats,
    Gromov delta) is computed in numpy from the returned cache.
    """
    log = progress or (lambda _m: None)
    pred = lifecycle_pred or "true"
    pred_a = pred.replace("n.`", "a.`")
    pred_b = pred.replace("n.`", "b.`")

    # --- nodes ---
    log("pulling nodes...")
    raw_ids: list[int] = []
    raw_labels: list[list[str]] = []
    for row in session.run(  # pyright: ignore[reportArgumentType]
        f"MATCH (n) WHERE {pred} RETURN id(n) AS id, labels(n) AS lbls"
    ):
        raw_ids.append(row["id"])
        raw_labels.append(row["lbls"] or [])
    n = len(raw_ids)
    id2idx = {nid: i for i, nid in enumerate(raw_ids)}

    # label table (CSR of small int ids)
    label_names: list[str] = []
    label_id: dict[str, int] = {}
    lab_counts = np.empty(n, dtype=np.int64)
    flat_lab: list[int] = []
    for i, lbls in enumerate(raw_labels):
        lab_counts[i] = len(lbls)
        for nm in lbls:
            j = label_id.get(nm)
            if j is None:
                j = len(label_names)
                label_id[nm] = j
                label_names.append(nm)
            flat_lab.append(j)
    lab_indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lab_counts, out=lab_indptr[1:])
    lab_ids = np.array(flat_lab, dtype=np.int32) if flat_lab else np.empty(0, np.int32)
    log(f"  {n:,} nodes, {len(label_names)} distinct labels")

    # --- edges (undirected, dedup via id(a) < id(b)) ---
    log("pulling edges...")
    es: list[int] = []
    ed: list[int] = []
    et: list[int] = []
    etype_names: list[str] = []
    etype_id: dict[str, int] = {}
    skipped = 0
    for row in session.run(  # pyright: ignore[reportArgumentType]
        f"MATCH (a)-[r]-(b) WHERE id(a) < id(b) AND {pred_a} AND {pred_b} "
        f"RETURN id(a) AS a, id(b) AS b, type(r) AS t"
    ):
        ia = id2idx.get(row["a"])
        ib = id2idx.get(row["b"])
        if ia is None or ib is None:
            skipped += 1
            continue
        t = row["t"]
        tj = etype_id.get(t)
        if tj is None:
            tj = len(etype_names)
            etype_id[t] = tj
            etype_names.append(t)
        es.append(ia)
        ed.append(ib)
        et.append(tj)
    src = np.array(es, dtype=np.int32) if es else np.empty(0, np.int32)
    dst = np.array(ed, dtype=np.int32) if ed else np.empty(0, np.int32)
    etype = np.array(et, dtype=np.int32) if et else np.empty(0, np.int32)
    log(f"  {src.size:,} undirected edges, {len(etype_names)} rel types"
        + (f" ({skipped:,} edge rows skipped: endpoint not in node set)"
           if skipped else ""))

    # --- undirected CSR over all cached edges ---
    u = np.concatenate([src, dst]).astype(np.int64)
    v = np.concatenate([dst, src]).astype(np.int64)
    order = np.argsort(u, kind="stable")
    u, v = u[order], v[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    if u.size:
        np.add.at(indptr, u + 1, 1)
    np.cumsum(indptr, out=indptr)

    return GraphCache(
        n=n, id2idx=id2idx,
        label_names=label_names, lab_indptr=lab_indptr, lab_ids=lab_ids,
        src=src, dst=dst, etype=etype, etype_names=etype_names,
        indptr=indptr, indices=v,
    )


def pull_year_values(session, ylabel: str, prop: str,
                     id2idx: dict[int, int]) -> dict[int, float]:
    """Fetch numeric year values for year-labelled nodes (one small query)."""
    out: dict[int, float] = {}
    for row in session.run(  # pyright: ignore[reportArgumentType]
        f"MATCH (y:`{ylabel}`) RETURN id(y) AS id, y.`{prop}` AS v"
    ):
        idx = id2idx.get(row["id"])
        v = row["v"]
        if idx is None or v is None:
            continue
        try:
            out[idx] = float(v)
        except (TypeError, ValueError):
            continue
    return out

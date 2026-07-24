r"""Live Neo4j bounded-subgraph source for the KGR serving layer.

Reuses (does NOT fork) ``scripts/neo4j_eval_export``'s driver/session
helpers -- they connect to the server's DEFAULT ``neo4j`` database and
deliberately ignore the stale ``.env NEO4J_DATABASE`` (the only path
proven to pull the 327k-node archival graph). The lifecycle-clean graph
is pulled ONCE via ``graphcache.build_cache`` and memoized on the source,
so the expensive 327k pull is amortized across every query in a process;
each per-query subgraph stays bounded by ``max_nodes`` (the 200-400-node
regime the encoder was trained for -- never feed the whole graph to the
encoder; that is an architectural non-negotiable).

The induced topology (degree/clustering/labels/temporal) comes from the
numpy cache; the per-query DIRECTED rel-types are recovered with the same
bounded ``MATCH (a)-[r]->(b) WHERE id(a) IN $ids AND id(b) IN $ids`` query
``_encode_graph`` uses (the cache stores edges undirected/dedup'd; the
encoder contract needs direction). Node ordering is the reference's
``sorted(cache_idx)`` so a re-pull is parity-comparable to an exported
NPZ.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
for _p in (str(_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse, don't fork: default-`neo4j`-DB driver/session (neo4j_eval_export:
# 94/105/114) and the one-pull numpy cache (graphcache).
from neo4j_eval_export import _driver, _session  # type: ignore  # noqa: E402
from graph_diagnostics.core import (  # noqa: E402
    DiagnosticConfig,
    lifecycle_predicate,
)
from graph_diagnostics.graphcache import (  # noqa: E402
    build_cache,
    pull_year_values,
)

from .schema_map import SchemaMap

_DEFAULT_CONFIG = _SCRIPTS / "kettle_config.yaml"


@dataclass
class SubgraphPull:
    """One bounded neighborhood, ready for ``tensor_contract.encode_subgraph``.

    ``node_ids`` is sorted by cache index (the reference's
    ``nodes = np.array(sorted(picked))`` order) so a re-pull of the same id
    set is bit-comparable to an exported NPZ. ``edges`` are induced DIRECTED
    (src_id, dst_id, rel_type). ``t_start/t_end`` are the GLOBALLY-normalized
    Year-reachability bounds (mirrors neo4j_eval_export:217-230), aligned to
    ``node_ids``.
    """

    node_ids: list[int]
    node_labels: list[list[str]]
    edges: list[tuple[int, int, str]]
    t_start: np.ndarray
    t_end: np.ndarray
    seed_id: int
    # cache label-ID space (the reference's deterministic, hash-stable
    # frequency-ordering domain -- see tensor_contract). label_names is the
    # GLOBAL cache label table; node_label_ids[r] are the lids of node r.
    label_names: list[str]
    node_label_ids: list[list[int]]
    subgraph_spec: str = "domain_only"

    @property
    def n(self) -> int:
        return len(self.node_ids)


class Neo4jSource:
    """Live source over the archival graph. Build once, query many.

    Parameters
    ----------
    config_path : kettle_config.yaml (lifecycle predicate + subgraph specs)
    schema_map  : the SchemaMap (only its ``temporal`` spec is used here:
                  Year label/property/max_hops for the temporal features)
    subgraph    : which kettle_config subgraph spec to scope to
                  (default ``domain_only`` -- the L3 domain layer the
                  encoder was characterized on)
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        schema_map: SchemaMap | None = None,
        subgraph: str = "domain_only",
    ) -> None:
        self.config_path = Path(config_path) if config_path else _DEFAULT_CONFIG
        self.schema_map = schema_map or SchemaMap.from_yaml()
        self.subgraph = subgraph
        self.cfg = DiagnosticConfig.from_yaml(self.config_path)
        spec = (self.cfg.subgraphs or {}).get(subgraph)
        if spec is None:
            raise SystemExit(
                f"{self.config_path} has no '{subgraph}' subgraph spec")
        self.spec = spec

        # lazily-built, process-memoized cache (the 327k one-time pull)
        self._drv = None
        self._cache = None
        self._indptr = None
        self._indices = None
        self._member = None          # bool[n] node_mask for the subgraph
        self._idx2id = None          # cache idx -> neo4j id
        self._t_start = None         # global [0,1] per cache idx
        self._t_end = None

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        if self._drv is not None:
            self._drv.close()
            self._drv = None

    def __enter__(self) -> "Neo4jSource":
        self._ensure_cache()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- the one-time pull ---------------------------------------------------

    def _ensure_cache(self) -> None:
        if self._cache is not None:
            return
        self._drv = _driver()
        self._drv.verify_connectivity()
        s = _session(self._drv)
        try:
            pred = lifecycle_predicate(self.cfg, var="n")
            cache = build_cache(s, pred, progress=lambda m: print("  " + m))
            node_mask = cache.label_mask(self.spec.get("include_labels") or [])
            edge_mask = cache.edge_mask(
                node_mask,
                self.spec.get("include_rel_types") or [],
                self.spec.get("exclude_rel_types") or [],
            )
            indptr, indices = cache.induced_csr(edge_mask)

            # Global Year-reachability -> per-node [0,1] temporal bounds.
            # Byte-exact mirror of neo4j_eval_export:217-230 (global lo/hi
            # over all Year values; clip to [0,1]).
            tspec = self.schema_map.temporal
            yvals = pull_year_values(
                s, tspec.year_label, tspec.year_property, cache.id2idx)
            min_y, max_y, has_y = cache.ensure_year_reachability(
                tspec.year_label, yvals, tspec.max_hops)
            if yvals:
                lo, hi = min(yvals.values()), max(yvals.values())
            else:
                lo, hi = 0.0, 1.0
            span = (hi - lo) or 1.0
            t_start = np.where(has_y, (min_y - lo) / span, 0.0).astype(np.float64)
            t_end = np.where(has_y, (max_y - lo) / span, 0.0).astype(np.float64)
            self._t_start = np.clip(t_start, 0.0, 1.0)
            self._t_end = np.clip(t_end, 0.0, 1.0)

            idx2id = np.empty(cache.n, dtype=np.int64)
            for nid, ix in cache.id2idx.items():
                idx2id[ix] = nid

            self._cache = cache
            self._indptr = indptr
            self._indices = indices
            self._member = node_mask
            self._idx2id = idx2id
        finally:
            s.close()

    # -- bounded subgraph extraction ----------------------------------------

    def _khop_ball(self, seed_idx: int, k_hops: int, cap: int) -> list[int]:
        """Connected BFS ball: <= ``cap`` member nodes within ``k_hops`` of
        ``seed_idx`` over the induced CSR (depth-bounded variant of
        ``neo4j_eval_export._bfs_ball``)."""
        indptr, indices, member = self._indptr, self._indices, self._member
        if not member[seed_idx]:
            return []
        picked = {seed_idx}
        frontier = [seed_idx]
        depth = 0
        while frontier and len(picked) < cap and depth < k_hops:
            depth += 1
            nxt: list[int] = []
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
        return sorted(picked)            # canonical order = sorted cache idx

    def _resolve_seed_indices(self, seed_ids: list[int]) -> list[int]:
        out: list[int] = []
        missing: list[int] = []
        for nid in seed_ids:
            ix = self._cache.id2idx.get(int(nid))
            if ix is None or not self._member[ix]:
                missing.append(int(nid))
            else:
                out.append(int(ix))
        if not out:
            raise ValueError(
                f"none of seed_ids {seed_ids} are in the lifecycle-clean "
                f"'{self.subgraph}' subgraph (missing/filtered: {missing})")
        if missing:
            print(f"  [neo4j_source] note: {len(missing)} seed id(s) not in "
                  f"subgraph, ignored: {missing[:8]}")
        return out

    def pull_subgraph(
        self,
        *,
        seed_ids: list[int] | None = None,
        cypher: str | None = None,
        k_hops: int = 2,
        max_nodes: int = 400,
    ) -> SubgraphPull:
        """Pull one bounded neighborhood.

        Seed-id mode (product path): union of <=``max_nodes`` k-hop balls
        around the resolved seeds. Cypher mode: the Cypher must
        ``RETURN id(n) AS id``; the returned ids are intersected with the
        lifecycle-clean subgraph and capped at ``max_nodes``.
        """
        self._ensure_cache()
        cache = self._cache

        if (seed_ids is None) == (cypher is None):
            raise ValueError("pass exactly one of seed_ids or cypher")

        if seed_ids is not None:
            seed_idxs = self._resolve_seed_indices(seed_ids)
            picked: set[int] = set()
            per = max(8, max_nodes // max(1, len(seed_idxs)))
            for sidx in seed_idxs:
                for c in self._khop_ball(sidx, k_hops, per):
                    picked.add(c)
                if len(picked) >= max_nodes:
                    break
            node_idx = sorted(picked)[:max_nodes]
            seed_cache_idx = seed_idxs[0]
        else:
            drv = _driver()
            try:
                drv.verify_connectivity()
                s = _session(drv)
                try:
                    raw = [int(r["id"]) for r in s.run(cypher)]  # type: ignore
                finally:
                    s.close()
            finally:
                drv.close()
            idxs = [
                cache.id2idx[i] for i in raw
                if i in cache.id2idx and self._member[cache.id2idx[i]]
            ]
            if not idxs:
                raise ValueError(
                    "cypher returned no nodes inside the lifecycle-clean "
                    f"'{self.subgraph}' subgraph")
            node_idx = sorted(set(idxs))[:max_nodes]
            seed_cache_idx = node_idx[0]

        if len(node_idx) < 2:
            raise ValueError(
                f"pulled subgraph too small (n={len(node_idx)}); widen "
                f"k_hops/seeds")

        node_ids = [int(self._idx2id[c]) for c in node_idx]

        # labels per node from the cache label table (==_encode_graph:364-371).
        # Keep lids (the reference's deterministic ordering domain) AND names.
        names = list(cache.label_names)
        node_labels: list[list[str]] = []
        node_label_ids: list[list[int]] = []
        for c in node_idx:
            lo, hi = cache.lab_indptr[c], cache.lab_indptr[c + 1]
            lids = [int(x) for x in cache.lab_ids[lo:hi]]
            node_label_ids.append(lids)
            node_labels.append([names[i] for i in lids])

        # induced DIRECTED edges (==_encode_graph:395-407): the cache is
        # undirected, so re-query for direction + rel-type.
        edges: list[tuple[int, int, str]] = []
        drv = _driver()
        try:
            drv.verify_connectivity()
            s = _session(drv)
            try:
                for row in s.run(  # type: ignore
                    "MATCH (a)-[r]->(b) WHERE id(a) IN $ids AND id(b) IN $ids "
                    "RETURN id(a) AS a, id(b) AS b, type(r) AS t",
                    ids=node_ids,
                ):
                    edges.append((int(row["a"]), int(row["b"]), str(row["t"])))
            finally:
                s.close()
        finally:
            drv.close()

        return SubgraphPull(
            node_ids=node_ids,
            node_labels=node_labels,
            edges=edges,
            t_start=self._t_start[node_idx].astype(np.float64),
            t_end=self._t_end[node_idx].astype(np.float64),
            seed_id=int(self._idx2id[seed_cache_idx]),
            label_names=names,
            node_label_ids=node_label_ids,
            subgraph_spec=self.subgraph,
        )

    def pull_by_ids(self, node_ids: list[int],
                    seed_id: int | None = None) -> SubgraphPull:
        """Pull the EXACT given neo4j-id set (no BFS growth) -- used by the
        parity gate to re-encode a reference NPZ's node set live. Ordering
        is sorted cache index (the reference's order). ``seed_id`` must be
        the reference's sampling seed so the depth-from-seed feature
        matches bit-for-bit (defaults to the first node otherwise)."""
        self._ensure_cache()
        cache = self._cache
        pairs = [(cache.id2idx[i], int(i)) for i in node_ids
                 if i in cache.id2idx]
        if not pairs:
            raise ValueError("no requested ids present in the cache")
        pairs.sort(key=lambda p: p[0])
        node_idx = [p[0] for p in pairs]
        ids = [p[1] for p in pairs]

        names = list(cache.label_names)
        node_labels = []
        node_label_ids: list[list[int]] = []
        for c in node_idx:
            lo, hi = cache.lab_indptr[c], cache.lab_indptr[c + 1]
            lids = [int(x) for x in cache.lab_ids[lo:hi]]
            node_label_ids.append(lids)
            node_labels.append([names[i] for i in lids])

        edges: list[tuple[int, int, str]] = []
        drv = _driver()
        try:
            drv.verify_connectivity()
            s = _session(drv)
            try:
                for row in s.run(  # type: ignore
                    "MATCH (a)-[r]->(b) WHERE id(a) IN $ids AND id(b) IN $ids "
                    "RETURN id(a) AS a, id(b) AS b, type(r) AS t",
                    ids=ids,
                ):
                    edges.append((int(row["a"]), int(row["b"]), str(row["t"])))
            finally:
                s.close()
        finally:
            drv.close()

        sid = ids[0]
        if seed_id is not None and int(seed_id) in set(ids):
            sid = int(seed_id)
        return SubgraphPull(
            node_ids=ids,
            node_labels=node_labels,
            edges=edges,
            t_start=self._t_start[node_idx].astype(np.float64),
            t_end=self._t_end[node_idx].astype(np.float64),
            seed_id=sid,
            label_names=names,
            node_label_ids=node_label_ids,
            subgraph_spec=self.subgraph,
        )

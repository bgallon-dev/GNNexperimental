r"""Subgraph -> exact ``corpus_dataset._build_graph_tensors`` input.

This is a faithful, behaviour-preserving refactor of
``scripts/neo4j_eval_export.py:_encode_graph`` (lines 357-556): same
type-slot frequency ordering, same depth-from-seed BFS, same deterministic
8-dim identity ``rng(neo4j_id & 0xFFFFFFFF)``, same schema-descriptor
assembly, same ``(N,32)`` x layout and ``(E,30)`` edge_attr layout. The
ONLY changes vs the reference are:

  * the two hardcoded calls ``_node_layer`` / ``_edge_category`` are
    replaced by :class:`~src.service.schema_map.SchemaMap` (the
    de-hardcoding -- see schema_map.yaml);
  * input is a pre-pulled :class:`SubgraphPull` (no Neo4j session here --
    the session work lives in ``neo4j_source.py``), so this module is pure
    and unit-/parity-testable in isolation.

The emitted dict is the **NPZ-equivalent** of one tier1 graph. It is fed
straight into ``corpus_dataset._build_graph_tensors`` (a plain dict
supports ``npz["key"]`` access, which is all that function needs), so the
encoder's input contract is enforced by the SAME code training/export use
-- zero re-derivation drift. That equivalence is the spine the parity gate
(verify.py P1 / tests/test_service_parity.py) checks bit-for-bit.

``encode_query`` is a byte-exact mirror of
``src/data/feature_encoder.py:encode_query`` (re-implemented here to avoid
the ``graph_builder``/``schema_sampler`` import chain that module pulls in
at import time -- the exact same reason ``neo4j_eval_export`` mirrors
``_encode_query_temporal``).

Pure numpy + the SchemaMap. No torch / neo4j import here.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .schema_map import SchemaMap

# --- KGR contract constants (mirrored from feature_encoder.py /
#     corpus_dataset.py -- the authoritative dims). Kept local so this
#     module has no torch/corpus import.
NODE_FEAT_DIM = 32
EDGE_FEAT_DIM = 30
QUERY_FEAT_DIM = 18
NODE_TYPE_DIM_ACTUAL = 12     # feature_encoder.py:41
EDGE_TYPE_DIM = 25            # feature_encoder.py:43
MAX_NODE_TYPES = 16
MAX_EDGE_TYPES = 30
NUM_LAYERS = 4
LAYER_ENTITY = 2
QUERY_TASK_DIM = 6


class SubgraphPull(Protocol):
    """Duck-typed contract of what ``neo4j_source`` hands ``encode_subgraph``.

    ``node_ids`` is the canonical node ordering (neo4j ids) -- ``neo4j_source``
    returns it sorted by cache index, mirroring the reference's
    ``nodes = np.array(sorted(picked))`` so parity holds against an
    exported NPZ. ``edges`` are induced DIRECTED edges among those nodes.
    """

    node_ids: Sequence[int]                 # (N,) canonical-ordered neo4j ids
    label_names: Sequence[str]              # GLOBAL cache label table
    node_label_ids: Sequence[Sequence[int]]  # (N,) cache lids per node
    edges: Sequence[tuple[int, int, str]]   # (E,) (src_id, dst_id, rel_type)
    t_start: Sequence[float]                # (N,) normalized [0,1]
    t_end: Sequence[float]                  # (N,) normalized [0,1]
    seed_id: int                            # neo4j id of the BFS-depth seed


# ---------------------------------------------------------------------------
# query encoding -- byte-exact mirror of feature_encoder.encode_query
# ---------------------------------------------------------------------------

def encode_query(
    task_type: int,
    temporal_window: tuple[float, float] | None = None,
    max_hops: int = 4,
    anchor_features: np.ndarray | None = None,
    component_tasks: tuple[int, ...] = (),
) -> np.ndarray:
    """Mirror of ``src/data/feature_encoder.py:encode_query`` (lines
    179-236). Layout: [0:6] task flags (multi-hot for compound),
    [6:8] temporal window, [8] max_hops/10, [9] pad,
    [10:18] anchor identity (ER / ER-compound only; zero otherwise)."""
    q = np.zeros(QUERY_FEAT_DIM, dtype=np.float32)
    if 0 <= task_type < QUERY_TASK_DIM:
        q[task_type] = 1.0
    if task_type == 5:
        for c in component_tasks:
            if 0 <= c < 5:
                q[c] = 1.0
    if temporal_window is not None:
        q[6] = float(temporal_window[0])
        q[7] = float(temporal_window[1])
    q[8] = max_hops / 10.0
    if anchor_features is not None:
        q[10:18] = anchor_features[24:32]
    return q


# ---------------------------------------------------------------------------
# graph encoding -- faithful refactor of neo4j_eval_export._encode_graph
# ---------------------------------------------------------------------------

def encode_subgraph(pull: "SubgraphPull", schema_map: SchemaMap) -> dict:
    """Encode one pulled neighborhood into the exact tier1 NPZ-equivalent
    dict (the input ``corpus_dataset._build_graph_tensors`` consumes).

    Faithful line-for-line port of ``_encode_graph`` with ``_node_layer``
    -> ``schema_map.node_layer`` and ``_edge_category`` ->
    ``schema_map.edge_category``; every other computation (frequency
    ordering, primary-label tie-break, clustering, depth BFS, identity
    rng, schema descriptors) is preserved verbatim.
    """
    node_ids = [int(i) for i in pull.node_ids]
    n = len(node_ids)
    if n == 0:
        raise ValueError("encode_subgraph: empty subgraph")
    id_to_row = {nid: r for r, nid in enumerate(node_ids)}

    # --- primary label + node-type / layer ids (==_encode_graph:363-392) ---
    # Byte-exact port: ordering is done in the cache LABEL-ID space (the
    # reference's domain). Integer-set iteration is hash-stable in CPython
    # (hash(int)==int, never randomized), so `sorted({lid}, key=-freq)`
    # reproduces _encode_graph deterministically across processes -- unlike
    # a label-NAME set, whose tie-break is PYTHONHASHSEED-dependent (this
    # was the P1 x_type[0:12] divergence; decision-tree branch 2).
    lab_names = list(pull.label_names)
    per_node_labels: list[list[int]] = [list(ls) for ls in pull.node_label_ids]
    freq = np.zeros(len(lab_names), dtype=np.int64)
    for ls in per_node_labels:
        for lid in ls:
            freq[lid] += 1
    present = sorted({lid for ls in per_node_labels for lid in ls},
                     key=lambda l: -freq[l])
    type_id = {lid: i for i, lid in enumerate(present)}

    def primary(lids: list[int]) -> int:
        # prefer a non-generic label; tie-break by rarity across the sample
        cand = [l for l in lids if lab_names[l] != "Entity"] or lids
        return min(cand, key=lambda l: freq[l])

    node_type = np.zeros(n, dtype=np.int64)
    node_layer = np.zeros(n, dtype=np.int64)
    prim_label_of_type: dict[int, str] = {}
    for r, lids in enumerate(per_node_labels):
        pl = primary(lids)
        tid = type_id[pl]
        node_type[r] = tid
        node_layer[r] = schema_map.node_layer(lab_names[pl])
        prim_label_of_type[tid] = lab_names[pl]

    # --- directed edges among sampled nodes (==_encode_graph:394-430) ---
    es: list[int] = []
    ed: list[int] = []
    et_names: list[str] = []
    for src_id, dst_id, rel in pull.edges:
        ra = id_to_row.get(int(src_id))
        rb = id_to_row.get(int(dst_id))
        if ra is None or rb is None:
            continue
        es.append(ra)
        ed.append(rb)
        et_names.append(str(rel))
    E = len(es)
    edge_index = np.zeros((2, E), dtype=np.int64)
    edge_attr = np.zeros((E, EDGE_FEAT_DIM), dtype=np.float32)

    uniq_rt = sorted(set(et_names), key=lambda t: -et_names.count(t))
    rt_id = {t: i for i, t in enumerate(uniq_rt)}

    in_deg = np.zeros(n)
    out_deg = np.zeros(n)
    nbr: list[set] = [set() for _ in range(n)]
    for i in range(E):
        a, b = es[i], ed[i]
        edge_index[0, i] = a
        edge_index[1, i] = b
        tt = rt_id[et_names[i]]
        if tt < EDGE_TYPE_DIM:
            edge_attr[i, tt] = 1.0
        edge_attr[i, EDGE_TYPE_DIM + schema_map.edge_category(et_names[i])] = 1.0
        edge_attr[i, EDGE_TYPE_DIM + 4] = 1.0          # all Neo4j rels directed
        out_deg[a] += 1
        in_deg[b] += 1
        nbr[a].add(b)
        nbr[b].add(a)

    # --- node features (N,32) (==_encode_graph:432-481) ---
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

    seed_id = int(pull.seed_id)
    seed_row = id_to_row.get(seed_id, 0)
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

    t_start = np.asarray(pull.t_start, dtype=np.float64)
    t_end = np.asarray(pull.t_end, dtype=np.float64)
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
        ts = float(t_start[r])
        te = float(t_end[r])
        x[r, to + 0] = ts
        x[r, to + 1] = te
        x[r, to + 2] = te - ts
        # deterministic 8-dim identity (==_encode_graph:480): rng seeded by
        # the neo4j id. task 2 query never reads it; present for layout
        # fidelity so x is byte-identical to the exported NPZ.
        idr = np.random.default_rng(node_ids[r] & 0xFFFFFFFF)
        x[r, to + 3: to + 11] = idr.standard_normal(8).astype(np.float32)

    # --- schema descriptor (==_encode_graph:483-512) ---
    n_node_types = len(present)
    nla = np.zeros(MAX_NODE_TYPES, dtype=np.int64)
    for tid in range(min(n_node_types, MAX_NODE_TYPES)):
        nla[tid] = schema_map.node_layer(prim_label_of_type.get(tid, "Entity"))

    n_edge_types = len(uniq_rt)
    eca = np.zeros(MAX_EDGE_TYPES, dtype=np.int64)
    edir = np.zeros(MAX_EDGE_TYPES, dtype=np.float32)
    esrc = np.zeros((MAX_EDGE_TYPES, 4), dtype=np.float32)
    etgt = np.zeros((MAX_EDGE_TYPES, 4), dtype=np.float32)
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
        eca[ti] = schema_map.edge_category(t)
        edir[ti] = 1.0
        for L in seen_src.get(ti, {LAYER_ENTITY}):
            esrc[ti, L] = 1.0
        for L in seen_tgt.get(ti, {LAYER_ENTITY}):
            etgt[ti, L] = 1.0

    return {
        "x": x,
        "neo4j_node_id": np.asarray(node_ids, dtype=np.int64),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "schema_n_node_types": np.array(n_node_types, dtype=np.int64),
        "schema_n_edge_types": np.array(n_edge_types, dtype=np.int64),
        "schema_node_layer_assignment": nla,
        "schema_edge_category": eca,
        "schema_edge_directed": edir,
        "schema_edge_source_layers": esrc,
        "schema_edge_target_layers": etgt,
    }


def build_graph_tensors(npz_like: dict) -> dict:
    """Run the emitted dict through the AUTHORITATIVE contract builder
    (``corpus_dataset._build_graph_tensors``) -- imported lazily so this
    module stays torch-free until an encode is actually requested. The
    returned dict has the encoder-ready tensors:
    ``x, edge_index, edge_type, edge_descriptor, node_descriptor``."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.data.corpus_dataset import _build_graph_tensors  # noqa: E402

    return _build_graph_tensors(npz_like)

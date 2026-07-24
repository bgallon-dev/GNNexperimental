r"""jsonl code graph -> tier1-schema NPZ.

Maps a `kgr_codegraph_odin` export onto the *exact* feature layout the
frozen KGR encoder was trained on (see
``src/data/feature_encoder.py`` docstrings and
``src/data/corpus_dataset._build_graph_tensors``):

  node x:   [0:12] type one-hot | [12:16] layer one-hot |
            [16:21] log-deg, log-in, log-out, clustering, depth/5 |
            [21:24] temporal (zeros — code has no time) |
            [24:32] deterministic per-node identity vector
  edge_attr:[0:25] type one-hot | [25:29] category one-hot | [29] directed

The 4 abstract layers (source/claim/entity/auxiliary) and 4 edge
categories (provenance/reference/structural/co-occurrence) are generic
slots — the encoder reads them through the schema descriptor and has no
hard-coded domain types, so this mapping is schema-portable by design.

Node-kind -> layer is a 4-level containment hierarchy (repo/module ->
class/fn/method -> in-body stmts -> external refs), which is what the
hyperbolic encoder's learned radial depth expects.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAX_NODE_TYPES = 16
MAX_EDGE_TYPES = 30
NUM_LAYERS = 4
NODE_FEAT_DIM = 32
EDGE_FEAT_DIM = 30
NODE_TYPE_DIM_ACTUAL = 12
EDGE_TYPE_ONEHOT_DIMS = 25

# Layer ids: 0=source, 1=claim, 2=entity, 3=auxiliary.
_KIND_LAYER = {
    "Repository": 0, "Module": 0,
    "Class": 1, "Function": 1, "Method": 1,
    "CallSite": 2, "Assignment": 2, "Return": 2,
    "Import": 3, "ExternalSymbol": 3, "ExternalPackage": 3,
}
# Category ids: 0=provenance, 1=reference, 2=structural, 3=co-occurrence.
_REL_CATEGORY = {
    "CONTAINS": 2, "DEFINES": 2, "INHERITS_FROM": 2,
    "CALLS": 1, "RESOLVES_TO": 1, "IMPORTS": 1,
    "IMPORTS_RAW": 1, "DEPENDS_ON_PACKAGE": 1,
    "ASSIGNS": 0, "RETURNS": 0,
}
# Containment edges used for hierarchy-depth BFS.
_DEPTH_RELS = {"CONTAINS", "DEFINES"}


def _stable_seed(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


@dataclass
class CodeGraph:
    node_ids: list[str]                       # row order
    id_to_row: dict[str, int]
    kind_of: dict[str, str]                   # node id -> kind
    file_of: dict[str, str]                   # node id -> file_path ("" if none)
    npz_path: Path
    removed_edge_ids: set[str] = field(default_factory=set)
    n_nodes: int = 0
    n_edges_kept: int = 0
    kind_to_type_id: dict[str, int] = field(default_factory=dict)
    rel_to_type_id: dict[str, int] = field(default_factory=dict)


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_npz(
    data_dir: Path,
    out_npz: Path,
    required_edge_ids: set[str] | None = None,
    with_dummy_task: bool = False,
    nodes_iter=None,
    edges_iter=None,
) -> CodeGraph:
    """Read nodes/edges jsonl (or pre-loaded record iterators), drop
    ``required_edge_ids`` (the held-out answer edges), and write a
    single tier1-schema graph NPZ.

    When ``nodes_iter`` / ``edges_iter`` are provided (e.g. from
    ``pack_loader.PackContext.iter_nodes/iter_edges``), they are
    consumed in place of reading ``<data_dir>/nodes.jsonl`` and
    ``<data_dir>/edges.jsonl``. The records must have the same field
    set as the jsonl rows."""
    data_dir = Path(data_dir)
    required_edge_ids = required_edge_ids or set()

    if nodes_iter is None:
        nodes = list(_read_jsonl(data_dir / "nodes.jsonl"))
    else:
        nodes = list(nodes_iter)
    node_ids = sorted(n["id"] for n in nodes)
    id_to_row = {nid: i for i, nid in enumerate(node_ids)}
    by_id = {n["id"]: n for n in nodes}
    kind_of = {n["id"]: n["kind"] for n in nodes}
    file_of = {n["id"]: (n.get("file_path") or "") for n in nodes}
    n = len(node_ids)

    kinds = sorted({n["kind"] for n in nodes})
    kind_to_type_id = {k: i for i, k in enumerate(kinds)}

    if edges_iter is None:
        edges_all = list(_read_jsonl(data_dir / "edges.jsonl"))
    else:
        edges_all = list(edges_iter)
    rels = sorted({e["native_relation"] for e in edges_all})
    rel_to_type_id = {r: i for i, r in enumerate(rels)}

    edges = [
        e for e in edges_all
        if e["id"] not in required_edge_ids
        and e["source_id"] in id_to_row
        and e["target_id"] in id_to_row
    ]
    E = len(edges)

    # --- structural stats (mirror feature_encoder.encode_nodes) ---
    in_deg = np.zeros(n, np.float32)
    out_deg = np.zeros(n, np.float32)
    nbr: list[set[int]] = [set() for _ in range(n)]
    for e in edges:
        s, t = id_to_row[e["source_id"]], id_to_row[e["target_id"]]
        out_deg[s] += 1
        in_deg[t] += 1
        nbr[s].add(t)
        nbr[t].add(s)
    total_deg = in_deg + out_deg

    clustering = np.zeros(n, np.float32)
    for i in range(n):
        ns = list(nbr[i])
        k = len(ns)
        if k < 2:
            continue
        links = sum(
            1
            for a in range(k)
            for b in range(a + 1, k)
            if ns[b] in nbr[ns[a]]
        )
        clustering[i] = 2.0 * links / (k * (k - 1))

    # --- containment depth via BFS from Repository over CONTAINS/DEFINES ---
    depth = np.zeros(n, np.float32)
    children: list[list[int]] = [[] for _ in range(n)]
    for e in edges:
        if e["native_relation"] in _DEPTH_RELS:
            children[id_to_row[e["source_id"]]].append(id_to_row[e["target_id"]])
    roots = [
        id_to_row[nid] for nid in node_ids if kind_of[nid] == "Repository"
    ]
    seen = set(roots)
    dq = deque((r, 0) for r in roots)
    while dq:
        node, d = dq.popleft()
        depth[node] = d
        for ch in children[node]:
            if ch not in seen:
                seen.add(ch)
                dq.append((ch, d + 1))

    # --- node features (32) ---
    x = np.zeros((n, NODE_FEAT_DIM), np.float32)
    for nid in node_ids:
        r = id_to_row[nid]
        kind = kind_of[nid]
        tid = kind_to_type_id[kind]
        if tid < NODE_TYPE_DIM_ACTUAL:
            x[r, tid] = 1.0
        x[r, NODE_TYPE_DIM_ACTUAL + _KIND_LAYER.get(kind, 3)] = 1.0
        x[r, 16] = np.log1p(total_deg[r])
        x[r, 17] = np.log1p(in_deg[r])
        x[r, 18] = np.log1p(out_deg[r])
        x[r, 19] = clustering[r]
        x[r, 20] = depth[r] / 5.0
        # [21:24] temporal: zeros (code has no temporal axis)
        rng = np.random.default_rng(_stable_seed(nid))
        x[r, 24:32] = rng.standard_normal(8).astype(np.float32)

    # --- edges (index + 30-d attr) ---
    edge_index = np.zeros((2, E), np.int64)
    edge_attr = np.zeros((E, EDGE_FEAT_DIM), np.float32)
    src_layers_seen: list[set[int]] = [set() for _ in range(MAX_EDGE_TYPES)]
    tgt_layers_seen: list[set[int]] = [set() for _ in range(MAX_EDGE_TYPES)]
    for i, e in enumerate(edges):
        s, t = id_to_row[e["source_id"]], id_to_row[e["target_id"]]
        edge_index[0, i] = s
        edge_index[1, i] = t
        rid = rel_to_type_id[e["native_relation"]]
        if rid < EDGE_TYPE_ONEHOT_DIMS:
            edge_attr[i, rid] = 1.0
        cat = _REL_CATEGORY.get(e["native_relation"], 2)
        edge_attr[i, EDGE_TYPE_ONEHOT_DIMS + cat] = 1.0
        edge_attr[i, EDGE_TYPE_ONEHOT_DIMS + 4] = 1.0  # all code edges directed
        if rid < MAX_EDGE_TYPES:
            src_layers_seen[rid].add(_KIND_LAYER.get(kind_of[e["source_id"]], 3))
            tgt_layers_seen[rid].add(_KIND_LAYER.get(kind_of[e["target_id"]], 3))

    # --- schema descriptor arrays (tier1 contract) ---
    schema_node_layer = np.full(MAX_NODE_TYPES, -1, np.int64)
    for kind, tid in kind_to_type_id.items():
        if tid < MAX_NODE_TYPES:
            schema_node_layer[tid] = _KIND_LAYER.get(kind, 3)

    schema_edge_cat = np.full(MAX_EDGE_TYPES, -1, np.int64)
    schema_edge_dir = np.zeros(MAX_EDGE_TYPES, np.float32)
    schema_edge_src = np.zeros((MAX_EDGE_TYPES, NUM_LAYERS), np.float32)
    schema_edge_tgt = np.zeros((MAX_EDGE_TYPES, NUM_LAYERS), np.float32)
    for rel, rid in rel_to_type_id.items():
        if rid >= MAX_EDGE_TYPES:
            continue
        schema_edge_cat[rid] = _REL_CATEGORY.get(rel, 2)
        schema_edge_dir[rid] = 1.0
        for L in src_layers_seen[rid]:
            schema_edge_src[rid, L] = 1.0
        for L in tgt_layers_seen[rid]:
            schema_edge_tgt[rid, L] = 1.0

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict = dict(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        schema_node_layer_assignment=schema_node_layer,
        schema_edge_category=schema_edge_cat,
        schema_edge_directed=schema_edge_dir,
        schema_edge_source_layers=schema_edge_src,
        schema_edge_target_layers=schema_edge_tgt,
        n_tasks=np.int64(1 if with_dummy_task else 0),
    )
    if with_dummy_task:
        # Stage A ignores sample.query / sample.labels (per train_v3
        # module docstring); we only need one task slot so CorpusDataset
        # builds a non-empty (graph, task) index.
        arrays["task_0_query"] = np.zeros(18, np.float32)
        arrays["task_0_labels"] = np.zeros(n, np.float32)
        arrays["task_0_type"] = np.int64(0)
    np.savez(out_npz, **arrays)

    return CodeGraph(
        node_ids=node_ids,
        id_to_row=id_to_row,
        kind_of=kind_of,
        file_of=file_of,
        npz_path=out_npz,
        removed_edge_ids=set(required_edge_ids),
        n_nodes=n,
        n_edges_kept=E,
        kind_to_type_id=kind_to_type_id,
        rel_to_type_id=rel_to_type_id,
    )

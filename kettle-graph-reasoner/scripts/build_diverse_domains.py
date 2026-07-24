r"""Build a DIVERSE-TOPOLOGY eval corpus for the schema-portability test.

The frozen encoder was trained on exactly one topology family: the
GraphBuilder's tree-like, layered source->claim->entity provenance DAG
(delta ~ 0, power-law, mean degree ~4.6). Real archival + code graphs
measured close to that shape. This script builds graphs whose *topology*
departs from tree-like in controlled, named ways, while remaining valid
encoder inputs (same 4-layer schema abstraction, same feature/edge
encoders, same TaskGenerator). The question is where the zero-training
ball-rank capability (emb-order the BFS ball) holds vs breaks as the graph
stops looking like a tree.

Six domain families (name -> real-world analog -> expected geometry stress):
  deep_tree        taxonomy / org chart      tree-like (POSITIVE CONTROL)
  scale_free       citation / web            hub power-law, cycles
  dense_community  social clusters           high clustering, modular
  bipartite        recommendation / gene-dz  two populations, cross edges
  grid2d           transit / spatial mesh    planar lattice, many cycles
  ring_mesh        molecule / circuit        concentric rings + chords

Each graph reuses the project's encode_nodes / encode_edges /
schema.to_tensor_dict / TaskGenerator so the npz is byte-compatible with
probe_capability_ballrank and _build_graph_tensors. Only the TOPOLOGY is
new. Each npz is tagged with `domain_family` and per-graph structural
stats (mean degree, clustering, cycle excess, a sampled delta-hyperbolicity
proxy) so the probe can correlate capability with structure.

Run from kettle-graph-reasoner/:
    py -m scripts.build_diverse_domains \
        --out src/data/corpus/diverse_domains --per-family 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import networkx as nx

# The data modules cross-import each other with bare names; put src/data on
# the path so `import graph_builder` etc. resolve (matches corpus_builder).
_DATA = Path(__file__).resolve().parents[1] / "src" / "data"
if str(_DATA) not in sys.path:
    sys.path.insert(0, str(_DATA))

from graph_builder import SyntheticGraph, NodeData, EdgeData  # noqa: E402
from schema_sampler import (  # noqa: E402
    SchemaDescriptor, EdgeTypeSpec,
    LAYER_SOURCE, LAYER_CLAIM, LAYER_ENTITY, LAYER_AUXILIARY,
    EDGE_CAT_PROVENANCE, EDGE_CAT_REFERENCE,
    EDGE_CAT_STRUCTURAL, EDGE_CAT_COOCCURRENCE,
)
from feature_encoder import encode_nodes, encode_edges, encode_query  # noqa: E402
from task_generator import TaskGenerator  # noqa: E402


# ---------------------------------------------------------------------------
# A single fixed schema, shared across all domain families. Node types spread
# across the 4 abstract layers; edge types across the 4 categories. Keeping
# the *schema* fixed while the *topology* varies isolates topology as the
# independent variable (the schema descriptor the encoder reads is constant).
# ---------------------------------------------------------------------------
# node type -> layer
NODE_LAYERS = [
    LAYER_SOURCE, LAYER_SOURCE,            # 0,1
    LAYER_CLAIM, LAYER_CLAIM, LAYER_CLAIM,  # 2,3,4
    LAYER_ENTITY, LAYER_ENTITY, LAYER_ENTITY,  # 5,6,7
    LAYER_AUXILIARY, LAYER_AUXILIARY,       # 8,9
]
TYPES_BY_LAYER = {
    LAYER_SOURCE: [0, 1],
    LAYER_CLAIM: [2, 3, 4],
    LAYER_ENTITY: [5, 6, 7],
    LAYER_AUXILIARY: [8, 9],
}


def _build_schema() -> SchemaDescriptor:
    all_layers = [LAYER_SOURCE, LAYER_CLAIM, LAYER_ENTITY, LAYER_AUXILIARY]
    specs = [
        # provenance (directed): claim/source -> source/claim
        EdgeTypeSpec(0, EDGE_CAT_PROVENANCE, [LAYER_CLAIM], [LAYER_SOURCE],
                     [2, 3, 4], [0, 1], True),
        EdgeTypeSpec(1, EDGE_CAT_PROVENANCE, [LAYER_CLAIM], [LAYER_CLAIM],
                     [2, 3, 4], [2, 3, 4], True),
        # reference (directed): claim -> entity
        EdgeTypeSpec(2, EDGE_CAT_REFERENCE, [LAYER_CLAIM], [LAYER_ENTITY],
                     [2, 3, 4], [5, 6, 7], True),
        EdgeTypeSpec(3, EDGE_CAT_REFERENCE, [LAYER_SOURCE], [LAYER_ENTITY],
                     [0, 1], [5, 6, 7], True),
        # structural (directed): any -> auxiliary, and same-layer links
        EdgeTypeSpec(4, EDGE_CAT_STRUCTURAL, all_layers, [LAYER_AUXILIARY],
                     list(range(10)), [8, 9], True),
        EdgeTypeSpec(5, EDGE_CAT_STRUCTURAL, all_layers, all_layers,
                     list(range(10)), list(range(10)), True),
        # co-occurrence (undirected): entity <-> entity
        EdgeTypeSpec(6, EDGE_CAT_COOCCURRENCE, [LAYER_ENTITY], [LAYER_ENTITY],
                     [5, 6, 7], [5, 6, 7], False),
        EdgeTypeSpec(7, EDGE_CAT_COOCCURRENCE, all_layers, all_layers,
                     list(range(10)), list(range(10)), False),
    ]
    return SchemaDescriptor(
        n_source_types=2, n_claim_types=3, n_entity_types=3,
        n_auxiliary_types=2, n_node_types=10, n_edge_types=len(specs),
        node_layer_assignment=list(NODE_LAYERS), edge_specs=specs,
        n_provenance_edges=2, n_reference_edges=2, n_structural_edges=2,
        n_cooccurrence_edges=2, seed=0,
    )


_SCHEMA = _build_schema()
# edge_type -> category / directed, for wiring
_ECAT = {e.index: e.category for e in _SCHEMA.edge_specs}
_EDIR = {e.index: e.directed for e in _SCHEMA.edge_specs}
# category -> representative edge types
_PROV_SRC = 0      # claim->source
_PROV_CLAIM = 1    # claim->claim
_REF = 2           # claim->entity
_STRUCT_AUX = 4    # any->auxiliary
_STRUCT_SAME = 5   # same-layer
_COOC = 6          # entity<->entity


def _type_for_layer(layer: int, rng) -> int:
    opts = TYPES_BY_LAYER[layer]
    return int(opts[rng.integers(0, len(opts))])


# ---------------------------------------------------------------------------
# Topology generators. Each returns (roles, edges):
#   roles: list[int] length N, the LAYER of each node (0..3)
#   edges: list[(u, v, edge_type)]
# Node ids are 0..N-1. Layers are assigned to reflect each topology's
# structural roles (hubs/roots -> source, interior -> claim, leaves -> entity,
# scaffolding -> auxiliary), then node *types* are sampled within-layer.
# ---------------------------------------------------------------------------
def gen_deep_tree(n, rng):
    """Balanced k-ary tree. Root=source, interior=claim, leaves=entity.

    POSITIVE CONTROL: this is the tree shape the encoder was trained on;
    the ball-rank capability should be at or near its archival strength.
    """
    k = int(rng.integers(2, 4))
    parent = {0: None}
    order = [0]
    frontier = [0]
    while len(order) < n and frontier:
        p = frontier.pop(0)
        for _ in range(k):
            if len(order) >= n:
                break
            c = len(order)
            parent[c] = p
            order.append(c)
            frontier.append(c)
    # depth of each node
    depth = {0: 0}
    for c in order[1:]:
        depth[c] = depth[parent[c]] + 1
    maxd = max(depth.values())
    roles = [LAYER_CLAIM] * n
    edges = []
    for c in order:
        if c == 0:
            roles[c] = LAYER_SOURCE
        elif depth[c] == maxd or all(parent.get(x) != c for x in order):
            roles[c] = LAYER_ENTITY  # leaf
        else:
            roles[c] = LAYER_CLAIM
    # leaves: nodes that are never a parent
    is_parent = set(parent[c] for c in order if parent[c] is not None)
    for c in order:
        if c != 0 and c not in is_parent:
            roles[c] = LAYER_ENTITY
    for c in order:
        p = parent[c]
        if p is None:
            continue
        if roles[c] == LAYER_ENTITY:
            et = _REF          # claim/source -> entity reference
        elif roles[p] == LAYER_SOURCE:
            et = _PROV_SRC
        else:
            et = _PROV_CLAIM
        # provenance/reference point child -> parent-ish; keep child as src
        edges.append((c, p, et))
    return roles, edges


def gen_scale_free(n, rng):
    """Barabasi-Albert preferential attachment (m>=2 -> hubs + cycles).

    Hubs -> source, high-degree interior -> claim, leaves -> entity.
    """
    m = int(rng.integers(2, 4))
    G = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(0, 2**31)))
    deg = dict(G.degree())
    ranked = sorted(deg, key=lambda x: -deg[x])
    n_src = max(1, n // 25)
    n_ent = int(n * 0.5)
    src = set(ranked[:n_src])
    ent = set(ranked[-n_ent:])
    roles = []
    for v in range(n):
        if v in src:
            roles.append(LAYER_SOURCE)
        elif v in ent:
            roles.append(LAYER_ENTITY)
        else:
            roles.append(LAYER_CLAIM)
    edges = []
    for u, v in G.edges():
        edges.append(_wire(u, v, roles, rng))
    return roles, edges


def gen_dense_community(n, rng):
    """k dense clusters (near-clique) with sparse bridges. High clustering."""
    k = int(rng.integers(3, 6))
    sizes = [n // k] * k
    for i in range(n - sum(sizes)):
        sizes[i] += 1
    p_in, p_out = 0.45, 0.01
    G = nx.stochastic_block_model(
        sizes, [[p_in if i == j else p_out for j in range(k)]
                for i in range(k)],
        seed=int(rng.integers(0, 2**31)))
    deg = dict(G.degree())
    ranked = sorted(deg, key=lambda x: -deg[x])
    src = set(ranked[:max(1, n // 25)])
    ent = set(ranked[-int(n * 0.5):])
    roles = [LAYER_CLAIM] * n
    for v in range(n):
        if v in src:
            roles[v] = LAYER_SOURCE
        elif v in ent:
            roles[v] = LAYER_ENTITY
    edges = [_wire(u, v, roles, rng) for u, v in G.edges()]
    return roles, edges


def gen_bipartite(n, rng):
    """Two populations, edges only cross between them (claims <-> entities)."""
    n_a = n // 2
    A = list(range(n_a))          # claims
    B = list(range(n_a, n))       # entities
    roles = [LAYER_CLAIM] * n
    for v in B:
        roles[v] = LAYER_ENTITY
    # a few claims promoted to source (roots of the bipartite fan)
    for v in A[:max(1, n_a // 12)]:
        roles[v] = LAYER_SOURCE
    edges = []
    deg = int(rng.integers(2, 4))
    for a in A:
        for _ in range(deg):
            b = B[int(rng.integers(0, len(B)))]
            edges.append(_wire(a, b, roles, rng))
    return roles, edges


def gen_grid2d(n, rng):
    """2D lattice, 4-neighbor. Planar, uniform degree, many short cycles."""
    side = max(2, int(round(np.sqrt(n))))
    G = nx.grid_2d_graph(side, side)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    N = G.number_of_nodes()
    deg = dict(G.degree())
    # corners/low-degree -> source; interior high-degree -> entity/claim
    corners = [v for v in range(N) if deg[v] == 2]
    roles = [LAYER_CLAIM] * N
    for v in corners[:max(1, N // 25)]:
        roles[v] = LAYER_SOURCE
    interior = sorted(range(N), key=lambda v: -deg[v])[:int(N * 0.5)]
    for v in interior:
        if roles[v] != LAYER_SOURCE:
            roles[v] = LAYER_ENTITY
    edges = [_wire(u, v, roles, rng) for u, v in G.edges()]
    return roles, edges


def gen_ring_mesh(n, rng):
    """Concentric rings (cycles) + radial spokes + random chords.

    Molecule/circuit flavor: locally cyclic, high delta-hyperbolicity.
    """
    n_rings = max(2, int(rng.integers(3, 6)))
    per = max(3, n // n_rings)
    ring_of = []
    idx = 0
    rings = []
    while idx < n:
        r = list(range(idx, min(idx + per, n)))
        if len(r) < 3 and rings:
            rings[-1].extend(r)
        else:
            rings.append(r)
        idx += per
    n = sum(len(r) for r in rings)
    roles = [LAYER_CLAIM] * n
    edges = []
    for ri, ring in enumerate(rings):
        L = len(ring)
        for i in range(L):
            a, b = ring[i], ring[(i + 1) % L]
            edges.append(_wire(a, b, roles, rng))
        # spokes to next ring
        if ri + 1 < len(rings):
            nxt = rings[ri + 1]
            for i, a in enumerate(ring):
                b = nxt[i % len(nxt)]
                edges.append(_wire(a, b, roles, rng))
    # innermost ring -> source; outermost -> entity
    for v in rings[0]:
        roles[v] = LAYER_SOURCE
    for v in rings[-1]:
        roles[v] = LAYER_ENTITY
    # random chords
    for _ in range(n // 8):
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b:
            edges.append(_wire(a, b, roles, rng))
    # rewire edge types now that roles are final
    edges = [_wire(u, v, roles, rng) for (u, v, _) in edges]
    return roles, edges


def _wire(u, v, roles, rng):
    """Pick a schema edge type consistent with the endpoints' layers."""
    lu, lv = roles[u], roles[v]
    layers = {lu, lv}
    if LAYER_ENTITY in layers and layers == {LAYER_ENTITY}:
        et = _COOC
    elif LAYER_ENTITY in layers:
        et = _REF
    elif layers == {LAYER_SOURCE} or (LAYER_SOURCE in layers and
                                      LAYER_CLAIM in layers):
        et = _PROV_SRC
    elif layers == {LAYER_CLAIM}:
        et = _PROV_CLAIM
    else:
        et = _STRUCT_SAME
    return (u, v, et)


GENERATORS = {
    "deep_tree": gen_deep_tree,
    "scale_free": gen_scale_free,
    "dense_community": gen_dense_community,
    "bipartite": gen_bipartite,
    "grid2d": gen_grid2d,
    "ring_mesh": gen_ring_mesh,
}


# ---------------------------------------------------------------------------
# Assemble a SyntheticGraph from (roles, edges): assign node types + temporal
# windows + identity vectors, add auxiliary time-period scaffolding so the
# temporal task has structure, and plant entity duplicates for the ER task.
# ---------------------------------------------------------------------------
def _assemble(roles, edges, rng) -> SyntheticGraph:
    n = len(roles)
    # BFS depth from the first source (or node 0) for the depth feature
    adj = [[] for _ in range(n)]
    for u, v, _ in edges:
        adj[u].append(v)
        adj[v].append(u)
    root = next((i for i, l in enumerate(roles) if l == LAYER_SOURCE), 0)
    depth = [0] * n
    seen = {root}
    dq = [root]
    while dq:
        x = dq.pop(0)
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                depth[y] = depth[x] + 1
                dq.append(y)
    maxd = max(depth) or 1

    nodes = {}
    for v in range(n):
        layer = roles[v]
        # temporal window correlated with depth so the temporal task is not
        # pure noise (mirrors the builder's parent-constrained windows)
        base = depth[v] / (maxd + 1)
        start = float(np.clip(base + rng.uniform(-0.1, 0.1), 0.0, 0.95))
        end = float(np.clip(start + rng.uniform(0.05, 0.3), start + 0.02, 1.0))
        nodes[v] = NodeData(
            node_id=v, node_type=_type_for_layer(layer, rng), layer=layer,
            depth=min(depth[v], 5),
            temporal_start=start, temporal_end=end,
            identity_vector=rng.standard_normal(8).astype(np.float32),
        )
    edge_objs = [EdgeData(u, v, et, _ECAT[et], _EDIR[et])
                 for (u, v, et) in edges]

    # auxiliary time-period scaffolding (like builder Phase 4)
    n_periods = max(6, min(20, n // 20))
    next_id = n
    period_nodes = []
    for i in range(n_periods):
        t_lo, t_hi = i / n_periods, (i + 1) / n_periods
        pid = next_id
        next_id += 1
        nodes[pid] = NodeData(
            node_id=pid, node_type=_type_for_layer(LAYER_AUXILIARY, rng),
            layer=LAYER_AUXILIARY, depth=0,
            temporal_start=t_lo, temporal_end=t_hi,
            identity_vector=rng.standard_normal(8).astype(np.float32),
        )
        period_nodes.append((pid, t_lo, t_hi))
    for v in range(n):
        nd = nodes[v]
        if nd.layer in (LAYER_SOURCE, LAYER_CLAIM):
            for pid, t_lo, t_hi in period_nodes:
                if nd.temporal_start < t_hi and nd.temporal_end > t_lo:
                    edge_objs.append(EdgeData(v, pid, _STRUCT_AUX,
                                              EDGE_CAT_STRUCTURAL, True))
                    break

    # plant entity duplicates for ER (Tier 1 exact / Tier 2 near)
    dup_pairs = []
    entity_ids = [v for v in range(n) if roles[v] == LAYER_ENTITY]
    rng.shuffle(entity_ids)
    claim_ids = [v for v in range(n) if roles[v] == LAYER_CLAIM]
    for eid in entity_ids[:max(1, len(entity_ids) // 12)]:
        orig = nodes[eid]
        tier = 1 if rng.random() < 0.5 else 2
        vec = (orig.identity_vector.copy() if tier == 1 else
               orig.identity_vector + rng.normal(0, 0.15, 8).astype(np.float32))
        did = next_id
        next_id += 1
        nodes[did] = NodeData(
            node_id=did, node_type=orig.node_type, layer=LAYER_ENTITY,
            depth=orig.depth, temporal_start=orig.temporal_start,
            temporal_end=orig.temporal_end, identity_vector=vec,
            is_duplicate_of=eid, duplicate_tier=tier)
        if claim_ids:
            c = int(claim_ids[rng.integers(0, len(claim_ids))])
            edge_objs.append(EdgeData(c, did, _REF, EDGE_CAT_REFERENCE, True))
        dup_pairs.append((eid, did, tier))

    return SyntheticGraph(schema=_SCHEMA, nodes=nodes, edges=edge_objs,
                          duplicate_pairs=dup_pairs, seed=0)


# ---------------------------------------------------------------------------
# Structural stats for the topology<->capability correlation.
# ---------------------------------------------------------------------------
def _stats(graph: SyntheticGraph) -> dict:
    G = nx.Graph()
    G.add_nodes_from(graph.nodes.keys())
    for e in graph.edges:
        G.add_edge(e.source, e.target)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    deg = [d for _, d in G.degree()]
    clustering = nx.average_clustering(G) if n > 2 else 0.0
    excess = (m - (n - 1)) / max(n, 1)  # 0 for a tree
    # sampled 4-point delta-hyperbolicity proxy on the largest component
    delta = _delta_proxy(G)
    return {
        "n_nodes": n, "n_edges": m,
        "mean_degree": float(np.mean(deg)) if deg else 0.0,
        "max_degree": int(max(deg)) if deg else 0,
        "clustering": float(clustering),
        "cycle_excess": float(excess),
        "delta_proxy": float(delta),
    }


def _delta_proxy(G, n_quads=400, seed=0):
    """Sampled Gromov 4-point delta on shortest-path metric (normalized by
    diameter). 0 = tree-like; larger = less hyperbolic."""
    if G.number_of_nodes() < 4:
        return 0.0
    comps = list(nx.connected_components(G))
    G = G.subgraph(max(comps, key=len)).copy()
    nodes = list(G.nodes())
    if len(nodes) < 4:
        return 0.0
    rng = np.random.default_rng(seed)
    # precompute a bounded number of source SSSPs
    srcs = nodes if len(nodes) <= 60 else [
        nodes[int(i)] for i in rng.choice(len(nodes), 60, replace=False)]
    dist = {s: nx.single_source_shortest_path_length(G, s) for s in srcs}
    diam = max((max(d.values()) for d in dist.values()), default=1) or 1
    worst = 0.0
    ss = list(dist.keys())
    for _ in range(n_quads):
        a, b, c, d = (ss[int(rng.integers(0, len(ss)))] for _ in range(4))
        try:
            d_ab, d_cd = dist[a][b], dist[c][d]
            d_ac, d_bd = dist[a][c], dist[b][d]
            d_ad, d_bc = dist[a][d], dist[b][c]
        except KeyError:
            continue
        s1, s2, s3 = d_ab + d_cd, d_ac + d_bd, d_ad + d_bc
        two = sorted([s1, s2, s3])[-2:]
        worst = max(worst, (two[1] - two[0]) / 2.0)
    return worst / diam


def _save(graph, family, stats, out_dir, idx):
    node_features, id_to_row = encode_nodes(graph)
    edge_index, edge_attr = encode_edges(graph, id_to_row)
    schema_tensor = _SCHEMA.to_tensor_dict()
    tasks = TaskGenerator(seed=idx + 1).generate_all_tasks(graph, id_to_row)
    save = {
        "x": node_features, "edge_index": edge_index, "edge_attr": edge_attr,
        "duplicate_pairs": np.zeros((0, 3), dtype=np.int64),
        "seed": np.array(idx), "schema_seed": np.array(0),
        "domain_family": np.array(family),
    }
    for k, v in schema_tensor.items():
        save[f"schema_{k}"] = v
    for k, v in stats.items():
        save[f"stat_{k}"] = np.array(v, dtype=np.float32)
    save["n_tasks"] = np.array(len(tasks))
    for j, t in enumerate(tasks):
        arow = id_to_row.get(t.anchor_node, 0)
        af = node_features[arow] if 0 <= arow < node_features.shape[0] else None
        save[f"task_{j}_type"] = np.array(t.task_type)
        save[f"task_{j}_anchor_row"] = np.array(arow)
        save[f"task_{j}_labels"] = t.labels
        save[f"task_{j}_max_hops"] = np.array(t.max_hops)
        save[f"task_{j}_query"] = encode_query(
            task_type=t.task_type, anchor_row=arow,
            temporal_window=t.temporal_window, max_hops=t.max_hops,
            anchor_features=af, component_tasks=t.component_tasks)
        tw = t.temporal_window if t.temporal_window else (0.0, 0.0)
        save[f"task_{j}_temporal"] = np.array(tw, dtype=np.float32)
    np.savez_compressed(out_dir / f"graph_{idx:06d}.npz", **save)
    return len(tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="src/data/corpus/diverse_domains")
    ap.add_argument("--per-family", type=int, default=15)
    ap.add_argument("--node-lo", type=int, default=150)
    ap.add_argument("--node-hi", type=int, default=350)
    ap.add_argument("--seed", type=int, default=20260713)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    idx = 0
    manifest = []
    for family, gen in GENERATORS.items():
        made = 0
        for _ in range(args.per_family):
            n = int(rng.integers(args.node_lo, args.node_hi))
            g_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
            roles, edges = gen(n, g_rng)
            graph = _assemble(roles, edges, g_rng)
            # need entity anchors + a few graded tasks to be useful
            if not any(l == LAYER_ENTITY for l in
                       (nd.layer for nd in graph.nodes.values())):
                continue
            stats = _stats(graph)
            nt = _save(graph, family, stats, out_dir, idx)
            manifest.append({"idx": idx, "family": family, "n_tasks": nt,
                             **stats})
            idx += 1
            made += 1
        print(f"{family:16s} built {made} graphs")

    import json
    (out_dir / "diversity_manifest.json").write_text(json.dumps(manifest,
                                                                 indent=2))
    # per-family structural summary
    print("\n=== structural stats by family (means) ===")
    print(f"{'family':16s} {'n':>4} {'deg':>6} {'clust':>7} "
          f"{'cyc_exc':>8} {'delta':>7}")
    for fam in GENERATORS:
        sub = [m for m in manifest if m["family"] == fam]
        if not sub:
            continue
        def mean(k):
            return sum(s[k] for s in sub) / len(sub)
        print(f"{fam:16s} {len(sub):>4} {mean('mean_degree'):>6.2f} "
              f"{mean('clustering'):>7.3f} {mean('cycle_excess'):>8.3f} "
              f"{mean('delta_proxy'):>7.3f}")
    print(f"\n{idx} graphs -> {out_dir}")


if __name__ == "__main__":
    main()

"""
neo4j_subgraph_task_export.py
=============================

Real-graph -> tier1-schema NPZ exporter for **task 4 (subgraph)**.

This is the G1 follow-on for task 4. It is the subgraph-task analogue of
``neo4j_eval_export.py`` (which only emits TASK_TEMPORAL): same live
sampling + the exact ``_encode_graph`` graph machinery (so P1 bit-exact
parity holds -- the graph tensors ARE the reference's), but the per-graph
task block is a byte-faithful mirror of
``src/data/task_generator.py:generate_subgraph_tasks`` (lines 474-546)
computed on the encoded real graph, exactly as ``_temporal_task`` mirrors
``generate_temporal_tasks``.

Why task 4 is well-defined on ``domain_only`` (and 0/5 are NOT): the
subgraph task is a temporal+depth BFS ball from an entity-layer anchor --
entity-layer nodes, per-node Year-derived temporal bounds, and adjacency
all exist in a domain_only neighborhood. Provenance (0) needs the
source/claim layers domain_only excludes; compound (5) composes it.

Usage
-----
    py scripts/neo4j_subgraph_task_export.py \
        --config scripts/kettle_config.yaml \
        --out src/data/corpus/real_subgraph_eval \
        --num-graphs 150 --max-nodes 300 --tasks-per-graph 3 --seed 0
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the verified machinery (do not fork): cache, sampling, and the
# exact contract encoder _encode_graph (P1 proved this is bit-parity).
import neo4j_eval_export as nee  # type: ignore
from graph_diagnostics.core import lifecycle_predicate
from graph_diagnostics.graphcache import build_cache, connected_components

from src.service.tensor_contract import encode_query  # verified mirror

LAYER_ENTITY = 2  # schema_sampler.LAYER_ENTITY (== schema_map default)


def _subgraph_tasks(npz: dict, rng: np.random.Generator,
                    n_tasks: int) -> int:
    """Byte-faithful mirror of generate_subgraph_tasks (task_generator.py
    :474-546) on the encoded real graph. Mutates ``npz`` in place: strips
    the temporal task block _encode_graph baked in and writes task_j_*
    (type 4). Returns n_tasks written."""
    for k in [k for k in npz if k.startswith("task_") or k == "n_tasks"]:
        del npz[k]

    x = npz["x"]
    N = x.shape[0]
    ei = npz["edge_index"]
    layer = np.where(x[:, 12:16].sum(1) > 0, x[:, 12:16].argmax(1), -1)
    ts = x[:, 21].astype(np.float64)
    te = x[:, 22].astype(np.float64)

    entity_rows = [r for r in range(N) if layer[r] == LAYER_ENTITY]
    adj: dict[int, list[int]] = defaultdict(list)
    for k in range(ei.shape[1]):
        a, b = int(ei[0, k]), int(ei[1, k])
        adj[a].append(b)
        adj[b].append(a)

    written = 0
    if entity_rows:
        n_actual = min(n_tasks, len(entity_rows))
        anchors = rng.choice(entity_rows, size=n_actual, replace=False)
        for anchor in anchors:
            anchor = int(anchor)
            max_depth = int(rng.integers(2, 5))
            wc = (ts[anchor] + te[anchor]) / 2.0
            wh = float(rng.uniform(0.1, 0.3))
            ws = max(0.0, wc - wh)
            we = min(1.0, wc + wh)

            labels = np.zeros(N, dtype=np.float32)
            visited: set[int] = set()
            queue: list[tuple[int, int]] = [(anchor, 0)]
            while queue:
                nid, dist = queue.pop(0)
                if nid in visited or dist > max_depth:
                    continue
                if te[nid] < ws or ts[nid] > we:   # temporal filter
                    continue
                visited.add(nid)
                labels[nid] = 1.0
                for nb in adj.get(nid, ()):
                    if nb not in visited:
                        queue.append((nb, dist + 1))

            if labels.sum() > 1:   # need >=2 nodes (==task_generator:537)
                j = written
                npz[f"task_{j}_type"] = np.array(4, dtype=np.int64)
                npz[f"task_{j}_anchor_row"] = np.array(anchor,
                                                       dtype=np.int64)
                npz[f"task_{j}_labels"] = labels
                npz[f"task_{j}_query"] = encode_query(
                    task_type=4, temporal_window=(ws, we),
                    max_hops=max_depth).astype(np.float32)
                npz[f"task_{j}_max_hops"] = np.array(max_depth,
                                                     dtype=np.int64)
                npz[f"task_{j}_temporal"] = np.array((ws, we),
                                                     dtype=np.float32)
                written += 1

    if written == 0:   # guarantee >=1 task (mirror nee fallback)
        labels = np.zeros(N, dtype=np.float32)
        seed_row = 0
        labels[seed_row] = 1.0
        npz["task_0_type"] = np.array(4, dtype=np.int64)
        npz["task_0_anchor_row"] = np.array(seed_row, dtype=np.int64)
        npz["task_0_labels"] = labels
        npz["task_0_query"] = encode_query(
            task_type=4, temporal_window=(0.0, 1.0),
            max_hops=4).astype(np.float32)
        npz["task_0_max_hops"] = np.array(4, dtype=np.int64)
        npz["task_0_temporal"] = np.array((0.0, 1.0), dtype=np.float32)
        written = 1
    npz["n_tasks"] = np.array(written, dtype=np.int64)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_SCRIPTS / "kettle_config.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-graphs", type=int, default=150)
    ap.add_argument("--max-nodes", type=int, default=300)
    ap.add_argument("--tasks-per-graph", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=4)
    a = ap.parse_args()

    cfg, spec = nee._spec_from_config(a.config)
    rng = np.random.default_rng(a.seed)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    drv = nee._driver()
    try:
        drv.verify_connectivity()
        with nee._session(drv) as s:
            print("pulling domain_only via graphcache...")
            cache = build_cache(s, lifecycle_predicate(cfg, var="n"),
                                progress=lambda m: print("  " + m))
            node_mask = cache.label_mask(spec.get("include_labels") or [])
            edge_mask = cache.edge_mask(
                node_mask, spec.get("include_rel_types") or [],
                spec.get("exclude_rel_types") or [])
            indptr, indices = cache.induced_csr(edge_mask)

            yvals = nee._pull_year_values(s, cache.id2idx)
            min_y, max_y, has_y = cache.ensure_year_reachability(
                "Year", yvals, cfg.temporal_max_hops)
            lo, hi = (min(yvals.values()), max(yvals.values())) if yvals \
                else (0.0, 1.0)
            span = (hi - lo) or 1.0
            t_start = np.clip(np.where(has_y, (min_y - lo) / span, 0.0),
                              0.0, 1.0).astype(np.float64)
            t_end = np.clip(np.where(has_y, (max_y - lo) / span, 0.0),
                            0.0, 1.0).astype(np.float64)

            comp = connected_components(indptr, indices, cache.n,
                                        allowed=node_mask)
            lab = comp[node_mask]
            giant = int(np.bincount(lab[lab >= 0]).argmax())
            giant_member = node_mask & (comp == giant)
            anchor_pool = np.flatnonzero(giant_member & has_y)
            if anchor_pool.size == 0:
                raise SystemExit("no Year-linked giant-component nodes")

            idx2id = np.empty(cache.n, dtype=np.int64)
            for nid, ix in cache.id2idx.items():
                idx2id[ix] = nid
            mid = np.where(has_y, (t_start + t_end) / 2.0, np.nan)

            print(f"sampling {a.num_graphs} delocalized neighborhoods "
                  f"(<= {a.max_nodes} nodes)...")
            written = 0
            for gi in range(a.num_graphs):
                nodes, seed_node = nee._delocalized_sample(
                    indptr, indices, giant_member, anchor_pool, mid,
                    rng, a.n_seeds, a.max_nodes)
                if nodes.size < 8:
                    continue
                npz = nee._encode_graph(
                    s, cache, nodes, seed_node, idx2id,
                    t_start, t_end, rng, a.tasks_per_graph, gi)
                nt = _subgraph_tasks(npz, rng, a.tasks_per_graph)
                fp = out_dir / f"graph_{written:06d}.npz"
                np.savez_compressed(fp, **npz)
                written += 1
                print(f"  graph_{written-1:06d}: N={npz['x'].shape[0]} "
                      f"E={npz['edge_index'].shape[1]} task4={nt}")
            print(f"\nwrote {written} task-4 eval graphs to {out_dir}")
    finally:
        drv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

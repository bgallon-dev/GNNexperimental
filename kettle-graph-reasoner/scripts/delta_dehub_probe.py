"""
delta_dehub_probe.py -- is domain_only's low Gromov delta genuine shallow
hierarchy, or a hub-induced small-world artifact?

The shipped topology screen (graph_diagnostics/checks/topology.py) reports
domain_only as "moderately hyperbolic" (2*delta_max/D = 0.286, diam 7). That
verdict is untrustworthy here for two reasons established in the prior turn:

  1. delta is quantized to {0, 0.5, 1.0, ...} at diam 7 -- coarser than the
     0.25 / 0.5 verdict bands -- and the headline ratio is driven by
     delta_max (worst of N) while the mean is ~0.08 (degenerately tree-like).
  2. domain_only is hub-dominated (max degree 7361 on 66k nodes). Hubs
     collapse pairwise distances, mechanically depressing delta independent
     of any latent curvature. A genuine shallow hierarchy and a hub
     small-world are indistinguishable to the unweighted-BFS delta.

This probe breaks both degeneracies on the SAME giant component:

  (A) baseline       -- unweighted BFS, full giant comp (parity vs shipped).
  (B) de-hubbed       -- drop top-k% nodes by degree, recompute the giant
                          component, re-measure. If diameter rises and
                          delta/diam stays low  -> genuine hierarchy.
                          If it shatters / delta rises -> hub artifact.
  (C) inverse-log-deg -- full giant comp, continuous edge weight
      weighted         w(u,v) = log(1+deg(u)) + log(1+deg(v)) so routing
                          through a hub is expensive. Breaks the integer
                          quantization; the 0.25/0.5 bands regain meaning.
  (D) (B)+(C)         -- de-hubbed AND weighted, the strongest screen.

Every condition reports 2*delta/D under the MEAN, P95, and MAX
normalizations -- not delta_max alone -- so the spread itself is visible.

Reuses the shipped one-pull cache (build_cache / induced_csr /
connected_components); adds only hub removal + a scipy weighted-Dijkstra
landmark metric. The shipped screen is not modified. Run from scripts/:

    py delta_dehub_probe.py --config kettle_config.yaml --subgraph domain_only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as scipy_cc
from scipy.sparse.csgraph import shortest_path

from graph_diagnostics.core import (
    DiagnosticConfig, _driver, lifecycle_predicate,
)
from graph_diagnostics.graphcache import (
    build_cache, connected_components as np_cc,
)


# ---------------------------------------------------------------------------
# Build the domain_only giant component as a scipy CSR (relabeled 0..G-1)
# ---------------------------------------------------------------------------

def giant_component_edges(cache, spec):
    """Return (src_idx, dst_idx, n_member, etype_of_kept_edges) in cache-index
    space for the giant weakly-connected component of the named subgraph."""
    node_mask = cache.label_mask(spec.get("include_labels") or [])
    if spec.get("temporal_filter"):
        raise SystemExit("This probe targets non-temporal subgraphs "
                         "(domain_only); temporal_filter not handled.")
    edge_mask = cache.edge_mask(
        node_mask,
        spec.get("include_rel_types") or [],
        spec.get("exclude_rel_types") or [],
    )
    indptr, indices = cache.induced_csr(edge_mask)
    comp = np_cc(indptr, indices, cache.n, allowed=node_mask)
    labels = comp[node_mask]
    valid = labels[labels >= 0]
    if valid.size == 0:
        raise SystemExit("Empty subgraph after masking.")
    giant_label = int(np.bincount(valid).argmax())
    member = node_mask & (comp == giant_label)
    keep = edge_mask & member[cache.src] & member[cache.dst]
    return cache.src[keep], cache.dst[keep], int(member.sum()), cache.etype[keep]


def build_simple_graph(s: np.ndarray, d: np.ndarray):
    """Relabel to 0..G-1, drop self-loops & parallel edges, return
    (relabel_nodes, edge_u, edge_v) for a simple undirected graph."""
    nodes = np.unique(np.concatenate([s, d]))
    remap = np.full(int(nodes.max()) + 1, -1, dtype=np.int64)
    remap[nodes] = np.arange(nodes.size)
    u = remap[s]
    v = remap[d]
    keep = u != v                                   # drop self-loops
    u, v = u[keep], v[keep]
    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    pair = np.stack([lo, hi], axis=1)
    pair = np.unique(pair, axis=0)                  # drop parallel edges
    return nodes, pair[:, 0], pair[:, 1]


def csr_from_edges(n: int, u: np.ndarray, v: np.ndarray,
                   weight: np.ndarray | None = None) -> csr_matrix:
    """Symmetric scipy CSR over n nodes. weight=None -> unit (BFS) weights."""
    ru = np.concatenate([u, v])
    rv = np.concatenate([v, u])
    if weight is None:
        w = np.ones(ru.size, dtype=np.float64)
    else:
        w = np.concatenate([weight, weight]).astype(np.float64)
    return csr_matrix((w, (ru, rv)), shape=(n, n))


# ---------------------------------------------------------------------------
# Gromov delta from a landmark distance matrix
# ---------------------------------------------------------------------------

def gromov_from_landmarks(graph: csr_matrix, n: int, n_landmarks: int,
                          n_samples: int, seed: int, unweighted: bool,
                          label: str) -> dict | None:
    """Mirror the shipped estimator: shortest paths from L random landmarks
    to all nodes (diameter = max finite), delta from the (L,L) submatrix."""
    rng = np.random.default_rng(seed)
    L = min(n_landmarks, n)
    landmarks = rng.choice(n, size=L, replace=False)
    t0 = time.monotonic()
    D = shortest_path(graph, method="D", directed=False,
                      indices=landmarks, unweighted=unweighted)
    finite = D[np.isfinite(D)]
    if finite.size == 0:
        return None
    diam = float(finite.max())
    Dmm = D[:, landmarks]                            # (L, L), exact
    deltas: list[float] = []
    for _ in range(n_samples):
        i, j, k, l = rng.choice(L, size=4, replace=False)
        ab, cd = Dmm[i, j], Dmm[k, l]
        ac, bd = Dmm[i, k], Dmm[j, l]
        ad, bc = Dmm[i, l], Dmm[j, k]
        if not np.isfinite([ab, cd, ac, bd, ad, bc]).all():
            continue
        s = sorted([ab + cd, ac + bd, ad + bc], reverse=True)
        deltas.append((s[0] - s[1]) / 2.0)
    if not deltas:
        return None
    da = np.array(deltas)
    return {
        "label": label, "nodes": n, "landmarks": L,
        "samples": da.size, "diam": diam,
        "d_mean": float(da.mean()),
        "d_p95": float(np.percentile(da, 95)),
        "d_max": float(da.max()),
        "r_mean": 2 * float(da.mean()) / diam if diam else float("nan"),
        "r_p95": 2 * float(np.percentile(da, 95)) / diam if diam else float("nan"),
        "r_max": 2 * float(da.max()) / diam if diam else float("nan"),
        "secs": time.monotonic() - t0,
    }


def verdict(r: float) -> str:
    if not np.isfinite(r):
        return "indeterminate"
    if r < 0.25:
        return "strongly hyperbolic"
    if r < 0.5:
        return "moderately hyperbolic"
    return "NOT strongly hyperbolic"


def print_row(res: dict | None, note: str = "") -> None:
    if res is None:
        print(f"  {note:<34s}  (no finite samples)")
        return
    print(
        f"  {res['label']:<34s} n={res['nodes']:>7,} "
        f"diam={res['diam']:>7.2f}  "
        f"d(mean/p95/max)={res['d_mean']:.3f}/{res['d_p95']:.3f}/"
        f"{res['d_max']:.3f}  "
        f"2d/D(mean/p95/max)={res['r_mean']:.3f}/{res['r_p95']:.3f}/"
        f"{res['r_max']:.3f}  [{verdict(res['r_max'])} by max; "
        f"{verdict(res['r_mean'])} by mean]  ({res['secs']:.1f}s)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--subgraph", default="domain_only")
    ap.add_argument("--landmarks", type=int, default=300,
                    help="BFS/Dijkstra source nodes (shipped default 200)")
    ap.add_argument("--samples", type=int, default=20000,
                    help="4-tuples sampled (framework asks ~10k+)")
    ap.add_argument("--hub-pcts", type=float, nargs="+",
                    default=[0.01, 0.1, 1.0],
                    help="top-%% of nodes by degree to remove for (B)/(D)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = DiagnosticConfig.from_yaml(args.config)
    if not cfg.subgraphs or args.subgraph not in cfg.subgraphs:
        print(f"Subgraph {args.subgraph!r} not in config "
              f"({sorted(cfg.subgraphs or [])})", file=sys.stderr)
        return 2
    spec = cfg.subgraphs[args.subgraph]

    driver = _driver()
    try:
        driver.verify_connectivity()
        # No db= : hit the default `neo4j` DB (NEO4J_DATABASE in .env is
        # stale per project notes); mirrors the shipped `check` path.
        with driver.session() as session:
            print("Pulling lifecycle-clean graph (one query pair)...")
            cache = build_cache(session, lifecycle_predicate(cfg, "n"),
                                progress=lambda m: print(" ", m))
    finally:
        driver.close()

    s, d, n_member, et = giant_component_edges(cache, spec)
    if s.size == 0:
        print(f"\n{args.subgraph}: no edges after masking — check "
              f"include_rel_types names against the live graph.",
              file=sys.stderr)
        return 2
    nodes, eu, ev = build_simple_graph(s, d)
    G = nodes.size
    print(f"\n{args.subgraph} giant component: {G:,} nodes "
          f"(mask said {n_member:,}), {eu.size:,} simple undirected edges")
    # rel-type composition of the kept edges -- grounds the spec in the
    # actual graph (confirms the include_rel_types names matched real data)
    bc = np.bincount(et, minlength=len(cache.etype_names))
    comp = sorted(((cache.etype_names[i], int(c)) for i, c in enumerate(bc)
                   if c), key=lambda kv: -kv[1])
    print("rel-type composition (pre-dedup): "
          + ", ".join(f"{nm}={c:,}" for nm, c in comp))

    deg = np.zeros(G, dtype=np.int64)
    np.add.at(deg, eu, 1)
    np.add.at(deg, ev, 1)
    print(f"degree: mean={deg.mean():.2f} max={int(deg.max())} "
          f"p99={int(np.percentile(deg, 99))} "
          f"top-5={sorted(deg.tolist())[-5:]}")
    print("\nlegend: r=2*delta/D. shipped screen reports r_max only; the "
          "mean/p95 columns expose the quantization tail.\n")

    base = csr_from_edges(G, eu, ev, weight=None)

    # (A) baseline -- parity vs shipped (expect diam ~7, r_max ~0.28-0.33)
    print("(A) baseline  -- unweighted BFS, full giant component")
    print_row(gromov_from_landmarks(base, G, args.landmarks, args.samples,
                                    args.seed, unweighted=True, label="unweighted/full"))

    # (C) inverse-log-degree weighted, full giant component
    w = np.log1p(deg[eu].astype(np.float64)) + np.log1p(deg[ev].astype(np.float64))
    wgraph = csr_from_edges(G, eu, ev, weight=w)
    print("\n(C) weighted  -- w(u,v)=log(1+deg u)+log(1+deg v), full giant comp")
    print_row(gromov_from_landmarks(wgraph, G, args.landmarks, args.samples,
                                    args.seed, unweighted=False,
                                    label="logdeg-weighted/full"))

    # (B) / (D): drop top-k% hubs, recompute giant comp, re-measure
    order = np.argsort(-deg)                          # highest degree first
    for pct in args.hub_pcts:
        k = max(1, int(round(G * pct / 100.0)))
        drop = np.zeros(G, dtype=bool)
        drop[order[:k]] = True
        ekeep = ~(drop[eu] | drop[ev])
        u2, v2 = eu[ekeep], ev[ekeep]
        if u2.size == 0:
            print(f"\n(B) de-hubbed top-{pct}% ({k} nodes): no edges remain.")
            continue
        # recompute the giant component AFTER hub removal
        sub = csr_from_edges(G, u2, v2, weight=None)
        ncomp, lab = scipy_cc(sub, directed=False)
        kept_nodes = np.flatnonzero(~drop)
        comp_sizes = np.bincount(lab[kept_nodes])
        gid = int(comp_sizes.argmax())
        gmask = (lab == gid) & (~drop)
        gnodes = np.flatnonzero(gmask)
        gshare = gnodes.size / kept_nodes.size
        # relabel the surviving giant component
        rl = np.full(G, -1, dtype=np.int64)
        rl[gnodes] = np.arange(gnodes.size)
        e2 = gmask[u2] & gmask[v2]
        gu, gv = rl[u2[e2]], rl[v2[e2]]
        gd = np.zeros(gnodes.size, dtype=np.int64)
        np.add.at(gd, gu, 1)
        np.add.at(gd, gv, 1)

        print(f"\n(B) de-hubbed top-{pct}% by degree "
              f"(removed {k:,}; {ncomp:,} comps after; "
              f"giant {gnodes.size:,} = {gshare:.1%} of survivors; "
              f"new max deg {int(gd.max()) if gd.size else 0})")
        gB = csr_from_edges(gnodes.size, gu, gv, weight=None)
        print_row(gromov_from_landmarks(gB, gnodes.size, args.landmarks,
                                        args.samples, args.seed,
                                        unweighted=True,
                                        label=f"unweighted/de-hub{pct}%"))
        # (D) de-hubbed AND log-degree weighted
        wB = (np.log1p(gd[gu].astype(np.float64))
              + np.log1p(gd[gv].astype(np.float64)))
        gD = csr_from_edges(gnodes.size, gu, gv, weight=wB)
        print_row(gromov_from_landmarks(gD, gnodes.size, args.landmarks,
                                        args.samples, args.seed,
                                        unweighted=False,
                                        label=f"weighted/de-hub{pct}%"))

    print("\nread: (A) should match the shipped 0.286-ish r_max as a parity "
          "check. Then ask of (B)/(C)/(D): does diam RISE and r stay LOW "
          "(genuine hierarchy survives hub removal / continuous metric), or "
          "does the graph SHATTER and r RISE (the low delta was a hub "
          "artifact -> the +0.094 hyp-euc is regularization, not curvature)?")
    return 0


if __name__ == "__main__":
    sys.exit(main())

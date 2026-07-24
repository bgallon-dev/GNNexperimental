"""
Graph topology for GNN training, multi-subgraph edition.

Reports (for the full graph AND for each named subgraph in config.subgraphs):
    - node/edge count, density
    - degree distribution stats
    - weakly-connected component structure (giant component share) -- EXACT
    - sampled Gromov delta-hyperbolicity (4-point condition) with
      delta / diameter ratio as the hyperbolic-fit score

This is the methodological heart of the KGR training-target selection: by
running delta per subgraph (Mention-Entity bipartite, domain-only, end-to-end,
temporal holdouts), you see which framing is most hyperbolic before you train
a single model. That's the right order of operations -- geometry first, then
architecture.

Implementation note (2026-04 rewrite)
-------------------------------------
The whole check now runs off a SINGLE graph pull. ``graphcache.build_cache``
streams the lifecycle-clean graph once into numpy arrays; every per-scope
metric below is computed from boolean masks over that cache. No subgraph
re-queries Neo4j, and the temporal split is answered from a Year-reachability
table computed once by bounded label propagation -- replacing the per-edge
``[*1..3]`` variable-length path expansion that made this run overnight.

Gromov delta: for each random 4-tuple (a,b,c,d), compute the three pairwise-
sum distances, sort descending as S1 >= S2 >= S3. Delta = (S1 - S2) / 2.
We report max/mean/p95 over a sample drawn from the all-pairs distances among
a set of BFS landmark nodes. Lower delta/diameter -> more tree-like ->
hyperbolic embedding justified. Thresholds (Borassi et al.):
    ratio < 0.25: strongly hyperbolic
    0.25-0.5   : moderately hyperbolic
    > 0.5      : not strongly hyperbolic
"""
from __future__ import annotations

import math
import random
import statistics
import time
from typing import Any

from graph_diagnostics.core import (
    CheckResult, Finding, Severity, DiagnosticConfig, lifecycle_predicate,
)


def run(session, config: DiagnosticConfig) -> CheckResult:
    result = CheckResult(check="topology")

    try:
        import numpy as np  # noqa: F401
        from graph_diagnostics.graphcache import build_cache, pull_year_values
    except ImportError:
        result.findings.append(Finding(
            check="topology", code="topology_skipped_no_numpy",
            severity=Severity.INFO,
            message="numpy not installed; topology check skipped. "
                    "Add numpy to scripts/requirements.txt.",
        ))
        return result

    pred = lifecycle_predicate(config, var="n")
    try:
        cache = build_cache(session, pred)
    except Exception as exc:  # pragma: no cover - connection/query failure
        result.findings.append(Finding(
            check="topology", code="graph_pull_failed",
            severity=Severity.HIGH,
            message=f"Could not pull graph into cache: "
                    f"{type(exc).__name__}: {exc}",
        ))
        return result

    import numpy as np

    # Full graph: every node, every cached edge.
    full_nodes = np.ones(cache.n, dtype=bool)
    full_edges = np.ones(cache.src.size, dtype=bool)
    _scope_topology(cache, full_nodes, full_edges, "full", config, result)

    if config.subgraphs:
        _report_subgraphs(session, cache, config, result, pull_year_values)

    return result


# ---------------------------------------------------------------------------
# Per-scope: stats + components + Gromov, all from numpy masks over the cache
# ---------------------------------------------------------------------------

def _scope_topology(cache, node_mask, edge_mask, scope: str,
                    config: DiagnosticConfig, result: CheckResult) -> None:
    import numpy as np

    n = int(node_mask.sum())
    m = int(edge_mask.sum())

    if n == 0:
        result.findings.append(Finding(
            check="topology", code=f"empty_graph:{scope}",
            severity=Severity.CRITICAL,
            message=f"{scope} graph is empty after lifecycle filtering.",
        ))
        return

    # --- degree distribution (induced) ---
    indptr, indices = cache.induced_csr(edge_mask)
    deg_all = (indptr[1:] - indptr[:-1])
    degs = np.sort(deg_all[node_mask])
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0.0

    def pct(p: float) -> int:
        idx = max(0, min(len(degs) - 1, int(round(p * (len(degs) - 1)))))
        return int(degs[idx])

    mean_deg = float(degs.mean()) if degs.size else 0.0
    p50, p95, p99 = pct(0.50), pct(0.95), pct(0.99)
    max_deg = int(degs[-1]) if degs.size else 0
    top1 = max(1, len(degs) // 100)
    top1_edges = int(degs[-top1:].sum())
    hub_share = top1_edges / (2 * m) if m > 0 else 0.0

    result.findings.append(Finding(
        check="topology", code=f"graph_stats:{scope}",
        severity=Severity.INFO,
        message=(
            f"[{scope}] n={n:,} m={m:,} density={density:.6f} "
            f"mean_deg={mean_deg:.2f} p50={p50} p95={p95} p99={p99} "
            f"max={max_deg} hub_share={hub_share:.2%}"
        ),
        count=n,
        details={
            "scope": scope, "nodes": n, "edges": m, "density": density,
            "mean_degree": mean_deg, "p50": p50, "p95": p95, "p99": p99,
            "max_degree": max_deg, "hub_share_top_1_pct": hub_share,
        },
    ))
    if mean_deg < 2:
        result.findings.append(Finding(
            check="topology", code=f"very_sparse:{scope}",
            severity=Severity.HIGH,
            message=f"[{scope}] mean degree {mean_deg:.2f} is very low.",
            count=n,
        ))
    if hub_share > 0.5:
        result.findings.append(Finding(
            check="topology", code=f"extreme_hub_concentration:{scope}",
            severity=Severity.HIGH,
            message=(
                f"[{scope}] top 1% of nodes carry {hub_share:.1%} of edges. "
                f"Use sampling caps or log-scale normalization."
            ),
            count=top1,
        ))

    # --- exact connected components (induced) ---
    from graph_diagnostics.graphcache import connected_components
    allowed = node_mask if scope != "full" else None
    comp = connected_components(indptr, indices, cache.n, allowed=allowed)
    labels = comp[node_mask]
    sizes = np.sort(np.bincount(labels[labels >= 0]))[::-1]
    sizes = sizes[sizes > 0]
    if sizes.size:
        total = int(sizes.sum())
        giant = int(sizes[0])
        giant_share = giant / total
        severity = Severity.INFO
        if giant_share < 0.9:
            severity = Severity.HIGH
        elif giant_share < 0.98:
            severity = Severity.MEDIUM
        result.findings.append(Finding(
            check="topology", code=f"components:{scope}",
            severity=severity,
            message=(
                f"[{scope}] {sizes.size} components. Giant: {giant:,} "
                f"({giant_share:.2%}). "
                f"2nd: {int(sizes[1]) if sizes.size > 1 else 0:,}."
            ),
            count=int(sizes.size),
            details={
                "scope": scope, "giant_size": giant,
                "giant_share": giant_share,
                "top_10_sizes": [int(x) for x in sizes[:10]],
                "isolated_count": int((sizes == 1).sum()),
            },
        ))
        giant_label = int(np.bincount(labels[labels >= 0]).argmax())
        giant_member = node_mask & (comp == giant_label)
    else:
        giant_member = node_mask

    # --- sampled Gromov delta on the giant component ---
    _gromov(cache, indptr, indices, giant_member, scope, config, result)


# ---------------------------------------------------------------------------
# Gromov delta: BFS landmarks + all-pairs-among-landmarks 4-tuple sampling
# ---------------------------------------------------------------------------

def _gromov(cache, indptr, indices, member, scope: str,
            config: DiagnosticConfig, result: CheckResult) -> None:
    import numpy as np
    from graph_diagnostics.graphcache import bfs, snowball_sample

    gc_nodes = np.flatnonzero(member)
    if gc_nodes.size < 4:
        return

    rng = np.random.default_rng(config.split_seed)
    allowed = member
    # Cap the region delta is measured on; delta is an advisory 3-bucket
    # verdict, so a connected snowball sample is an accepted approximation.
    if gc_nodes.size > config.gromov_max_nodes:
        seed = int(gc_nodes[rng.integers(gc_nodes.size)])
        sample_nodes = snowball_sample(
            indptr, indices, cache.n, member, seed,
            config.gromov_sample_max_nodes,
        )
        allowed = np.zeros(cache.n, dtype=bool)
        allowed[sample_nodes] = True
    else:
        sample_nodes = gc_nodes

    m_land = min(config.gromov_landmarks, sample_nodes.size)
    landmarks = sample_nodes[
        rng.choice(sample_nodes.size, size=m_land, replace=False)
    ]

    # BFS from each landmark over the (induced, possibly snowballed) graph.
    # Keep only each row's landmark columns (+ a running diameter) so peak
    # memory is O(L^2), not O(L * n) -- at 327k nodes the latter is ~260 MB.
    deadline = time.monotonic() + config.gromov_timeout_sec
    rows: list[np.ndarray] = []
    diam = 0
    timed_out = False
    for li in landmarks:
        if time.monotonic() > deadline:
            timed_out = True
            break
        drow = bfs(indptr, indices, int(li), cache.n, allowed=allowed)
        rows.append(drow[landmarks].astype(np.int32))   # length m_land
        fin = drow[drow >= 0]
        if fin.size:
            diam = max(diam, int(fin.max()))
        del drow
    if len(rows) < 4:
        return
    L = len(rows)
    Dmm = np.vstack(rows)[:, :L]                   # (L, L) pairwise, exact

    deltas: list[float] = []
    for _ in range(config.gromov_sample_size):
        i, j, k, l = rng.choice(L, size=4, replace=False)
        d_ab, d_cd = Dmm[i, j], Dmm[k, l]
        d_ac, d_bd = Dmm[i, k], Dmm[j, l]
        d_ad, d_bc = Dmm[i, l], Dmm[j, k]
        if min(d_ab, d_cd, d_ac, d_bd, d_ad, d_bc) < 0:
            continue
        s = sorted([d_ab + d_cd, d_ac + d_bd, d_ad + d_bc], reverse=True)
        deltas.append((s[0] - s[1]) / 2.0)

    if timed_out:
        result.findings.append(Finding(
            check="topology", code=f"gromov_timeout:{scope}",
            severity=Severity.INFO,
            message=(
                f"[{scope}] Gromov BFS timed out after "
                f"{config.gromov_timeout_sec:.0f}s "
                f"({L} of {config.gromov_landmarks} landmarks done). "
                f"Partial results reported. Raise gromov_timeout_sec for more."
            ),
            count=L,
        ))
    if not deltas:
        return

    max_delta = max(deltas)
    mean_delta = statistics.mean(deltas)
    p95_delta = sorted(deltas)[int(0.95 * (len(deltas) - 1))]
    ratio = (2 * max_delta / diam) if diam else float("nan")

    if math.isnan(ratio):
        severity = Severity.INFO
        verdict = "indeterminate"
    elif ratio < 0.25:
        severity = Severity.INFO
        verdict = "strongly hyperbolic -- Poincare ball well-justified"
    elif ratio < 0.5:
        severity = Severity.INFO
        verdict = "moderately hyperbolic -- hyperbolic GNN likely helpful"
    else:
        severity = Severity.MEDIUM
        verdict = (
            "not strongly hyperbolic -- Euclidean baseline may match or beat "
            "hyperbolic GNN on this subgraph"
        )

    result.findings.append(Finding(
        check="topology", code=f"gromov_delta:{scope}",
        severity=severity,
        message=(
            f"[{scope}] delta_max={max_delta:.2f} mean={mean_delta:.2f} "
            f"p95={p95_delta:.2f} diam={diam} 2d/D={ratio:.3f} -- {verdict}"
        ),
        count=len(deltas),
        details={
            "scope": scope,
            "delta_max": max_delta, "delta_mean": mean_delta,
            "delta_p95": p95_delta, "diameter_est": diam,
            "hyperbolic_ratio": ratio, "verdict": verdict,
            "giant_component_nodes": int(gc_nodes.size),
            "giant_component_edges": int(
                (member[cache.src] & member[cache.dst]).sum()
            ),
            "samples_used": len(deltas),
            "samples_attempted": config.gromov_sample_size,
            "landmarks_used": L,
            "sampled_region_nodes": int(sample_nodes.size),
        },
    ))


# ---------------------------------------------------------------------------
# Per-subgraph: derive masks from the cache (no Neo4j round-trips)
# ---------------------------------------------------------------------------

def _report_subgraphs(session, cache, config, result, pull_year_values) -> None:
    import numpy as np

    for name, spec in config.subgraphs.items():
        try:
            node_mask = cache.label_mask(spec.get("include_labels") or [])

            temporal = spec.get("temporal_filter")
            if temporal:
                ylabel = temporal.get("label", "Year")
                prop = temporal["property"]
                hops = config.temporal_max_hops
                yvals = pull_year_values(session, ylabel, prop, cache.id2idx)
                min_y, max_y, has = cache.ensure_year_reachability(
                    ylabel, yvals, hops,
                )
                is_ylabel = cache.label_mask([ylabel])
                node_mask = cache.temporal_mask(
                    node_mask, is_ylabel, min_y, max_y, has,
                    temporal["comparison"], float(temporal["cutoff"]),
                )

            edge_mask = cache.edge_mask(
                node_mask,
                spec.get("include_rel_types") or [],
                spec.get("exclude_rel_types") or [],
            )

            n_sub = int(node_mask.sum())
            m_sub = int(edge_mask.sum())

            # label / rel-type distributions over the masked subgraph
            lab_counts = _label_distribution(cache, node_mask)
            rel_counts = _rel_distribution(cache, edge_mask)

            result.findings.append(Finding(
                check="topology", code=f"subgraph_stats:{name}",
                severity=Severity.INFO,
                message=(
                    f"[subgraph:{name}] nodes={n_sub:,} edges={m_sub:,} "
                    f"({spec.get('description', '')})"
                ),
                count=n_sub,
                details={
                    "subgraph": name,
                    "description": spec.get("description"),
                    "nodes": n_sub, "edges": m_sub,
                    "label_distribution": lab_counts,
                    "rel_type_distribution": rel_counts,
                },
            ))

            if n_sub == 0:
                result.findings.append(Finding(
                    check="topology", code=f"subgraph_empty:{name}",
                    severity=Severity.MEDIUM,
                    message=(
                        f"Subgraph {name!r} is empty. Check filters -- "
                        f"lifecycle exclusion + label/rel/temporal filters "
                        f"may be over-restrictive."
                    ),
                ))
                continue

            _scope_topology(cache, node_mask, edge_mask,
                            f"subgraph:{name}", config, result)
        except Exception as exc:  # pragma: no cover
            result.findings.append(Finding(
                check="topology", code=f"subgraph_failed:{name}",
                severity=Severity.LOW,
                message=f"Subgraph {name!r} topology failed: "
                        f"{type(exc).__name__}: {exc}",
            ))


def _label_distribution(cache, node_mask) -> dict[str, int]:
    import numpy as np
    counts = cache.lab_indptr[1:] - cache.lab_indptr[:-1]
    node_of_slot = np.repeat(np.arange(cache.n), counts)
    keep = node_mask[node_of_slot]
    ids = cache.lab_ids[keep]
    if ids.size == 0:
        return {}
    bc = np.bincount(ids, minlength=len(cache.label_names))
    out = {cache.label_names[i]: int(bc[i]) for i in range(len(bc)) if bc[i]}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _rel_distribution(cache, edge_mask) -> dict[str, int]:
    import numpy as np
    ids = cache.etype[edge_mask]
    if ids.size == 0:
        return {}
    bc = np.bincount(ids, minlength=len(cache.etype_names))
    out = {cache.etype_names[i]: int(bc[i]) for i in range(len(bc)) if bc[i]}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))

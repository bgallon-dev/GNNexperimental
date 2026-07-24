r"""P1 — radius-as-hierarchy diagnostic (Docs/GEOMETRY_READOUT_PROBES_PLAN.md).

Does the Poincare radius ||h|| encode structural depth beyond what degree
explains? Per graph: Spearman rho(radius, depth), rho(radius, log1p(deg)),
and the depth partial (rank-residualized on log-degree). Zero training.

Inputs: manifold_index.npz files (frozen h128 tier1, h32 tier1, h32 real
200-graph archival) + the tutorstructure code graph embedded with the h32
suggester checkpoint (depth = undirected BFS from in-degree-0 roots).

    py -m scripts.probe_radius_hierarchy --out runs/geometry_probes/p1_radius/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MIN_NODES = 30

INDEXES = {
    "h128_tier1val": "frozen/kgr-v1.0-2026-07-07/encoder_baseline/manifold_index.npz",
    "h32_tier1val": "runs/geometry_probes/p1_radius/index_h32_tier1val.npz",
    "h32_real200": "runs/geometry_probes/p1_radius/index_h32_real.npz",
}


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    # average ties
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rankdata(x), _rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """rho(x, y | z) on ranks: residualize rank(x), rank(y) on rank(z)."""
    rx, ry, rz = (_rankdata(v) for v in (x, y, z))
    rz = rz - rz.mean()
    zz = (rz * rz).sum()
    if zz <= 0:
        return _spearman(x, y)
    ex = rx - rx.mean() - (rx - rx.mean()) @ rz / zz * rz
    ey = ry - ry.mean() - (ry - ry.mean()) @ rz / zz * rz
    den = np.sqrt((ex * ex).sum() * (ey * ey).sum())
    return float((ex * ey).sum() / den) if den > 0 else float("nan")


def _per_graph_stats(radius, depth, logdeg, graph_idx) -> list[dict]:
    rows = []
    for gi in np.unique(graph_idx):
        m = graph_idx == gi
        if m.sum() < MIN_NODES:
            continue
        r, d, g = radius[m], depth[m], logdeg[m]
        if np.ptp(d) == 0 or np.ptp(r) == 0:
            continue
        rows.append({
            "graph": int(gi), "n": int(m.sum()),
            "rho_depth": _spearman(r, d),
            "rho_logdeg": _spearman(r, g),
            "rho_depth_given_deg": _partial_spearman(r, d, g),
        })
    return rows


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n_graphs": 0}
    rd = np.array([x["rho_depth"] for x in rows])
    rg = np.array([x["rho_logdeg"] for x in rows])
    rp = np.array([x["rho_depth_given_deg"] for x in rows])
    dom_sign = 1.0 if np.median(rd) >= 0 else -1.0
    return {
        "n_graphs": len(rows),
        "rho_depth_median": float(np.median(rd)),
        "rho_depth_abs_median": float(np.median(np.abs(rd))),
        "frac_dominant_sign": float((np.sign(rd) == dom_sign).mean()),
        "rho_logdeg_median": float(np.median(rg)),
        "rho_logdeg_abs_median": float(np.median(np.abs(rg))),
        "partial_rho_depth_median": float(np.median(rp)),
        "partial_retention": float(
            np.median(np.abs(rp)) / max(np.median(np.abs(rd)), 1e-12)),
    }


def _from_index(path: Path) -> dict:
    z = np.load(path)
    radius = z["radius"].astype(np.float64)
    depth = z["depth"].astype(np.float64)
    logdeg = np.log1p(z["in_degree"].astype(np.float64)
                      + z["out_degree"].astype(np.float64))
    rows = _per_graph_stats(radius, depth, logdeg, z["graph_idx"])
    return {"per_graph": rows, "summary": _summarize(rows)}


def _codegraph(npz_path: Path, ckpt: Path) -> dict:
    from collections import deque

    from src.codegraph.harness import _build_encoder, _embed
    from src.data.corpus_dataset import _build_graph_tensors

    with np.load(npz_path) as z:
        g = _build_graph_tensors(z)
    enc, cfg = _build_encoder(ckpt, g, "cpu")
    emb = _embed(enc, g, "cpu")
    from src.modelsv2.layers import poincare_ops as P
    radius = P.logmap0(emb, getattr(enc, "c", 1.0)).norm(dim=-1).numpy() \
        .astype(np.float64)

    ei = g["edge_index"].numpy()
    n = emb.shape[0]
    indeg = np.bincount(ei[1], minlength=n)
    outdeg = np.bincount(ei[0], minlength=n)
    logdeg = np.log1p((indeg + outdeg).astype(np.float64))
    adj: list[list[int]] = [[] for _ in range(n)]
    for s, t in zip(ei[0], ei[1]):
        adj[int(s)].append(int(t))
        adj[int(t)].append(int(s))
    roots = np.where(indeg == 0)[0]
    depth = np.full(n, np.inf)
    dq = deque()
    for r in roots:
        depth[r] = 0.0
        dq.append(int(r))
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if depth[v] == np.inf:
                depth[v] = depth[u] + 1.0
                dq.append(v)
    keep = np.isfinite(depth)
    rows = _per_graph_stats(radius[keep], depth[keep], logdeg[keep],
                            np.zeros(int(keep.sum()), dtype=np.int64))
    return {"per_graph": rows, "summary": _summarize(rows),
            "n_unreachable_dropped": int((~keep).sum())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str,
                   default="runs/geometry_probes/p1_radius/results.json")
    p.add_argument("--codegraph-npz", type=str,
                   default="runs/blend_h32_suggester/graph_tutorstructure_patch.npz")
    p.add_argument("--codegraph-ckpt", type=str,
                   default="runs/width-h32-hyp-l4-s0")
    args = p.parse_args()

    torch.manual_seed(0)
    results: dict = {"min_nodes": MIN_NODES, "arms": {}}
    for name, rel in INDEXES.items():
        path = _ROOT / rel
        if not path.exists():
            results["arms"][name] = {"error": f"missing {rel}"}
            continue
        results["arms"][name] = _from_index(path)
    results["arms"]["h32_codegraph"] = _codegraph(
        _ROOT / args.codegraph_npz, _ROOT / args.codegraph_ckpt)

    out = _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    # per_graph lists are large for the 200-graph arm; keep them, they're small dicts
    out.write_text(json.dumps(results, indent=2))

    print("=" * 78)
    print("P1 radius-hierarchy probe  (bar: |rho_depth| median >= 0.30, "
          "same sign >= 70%, partial retains >= 50%)")
    for name, arm in results["arms"].items():
        s = arm.get("summary", {})
        if not s or s.get("n_graphs", 0) == 0:
            print(f"  {name:<16} SKIP ({arm.get('error', 'no usable graphs')})")
            continue
        print(f"  {name:<16} n_graphs={s['n_graphs']:<4} "
              f"rho_depth med={s['rho_depth_median']:+.3f} "
              f"|med|={s['rho_depth_abs_median']:.3f} "
              f"sign%={s['frac_dominant_sign']:.2f}  "
              f"rho_deg |med|={s['rho_logdeg_abs_median']:.3f}  "
              f"partial med={s['partial_rho_depth_median']:+.3f} "
              f"retention={s['partial_retention']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

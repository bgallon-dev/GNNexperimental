"""
distortion_probe.py -- the decisive curvature-vs-regularization instrument.

Background. The Gromov-delta screen is exhausted on this corpus: domain_only
is hub-and-spoke (delta degenerate via distance-collapse), the
containment/provenance edges are a forest of stars (delta trivially 0 because
a tree is 0-hyperbolic), end_to_end is hub-dominated diam-6. delta cannot
tell whether the measured +0.094 hyp-euc retrieval advantage is the geometry
fitting native negative curvature, or the bounded Poincare ball acting as an
implicit norm regularizer.

This asks the question delta cannot: **does the Poincare ball embed the
structure the model actually conditions on with materially lower distortion
than Euclidean at MATCHED dimension and matched optimizer budget, and does a
learnable curvature settle away from 0?** That is the Sarkar/Sala
tree-embedding-fidelity test, and it bears directly on the Gemynd backbone
decision in a way no delta variant can.

Controlled experiment. The ONLY differences between the two arms are the
metric and the learnable curvature; identical point parameters (same init,
same seed), identical Adam lr / steps / pair-minibatch schedule, identical
learnable global scale, identical matched dimension. Euclidean is the control
condition (cf. CLAUDE.md). Reuses the shipped one-pull cache + numpy BFS, and
the project's own numerically-stable Poincare ops (expmap0/dist accept a
tensor c -- the module was built for exactly this).

Two orthogonal reads (fork "c"):
  (balls)    k-hop balls around sampled domain entities on the FULL graph --
             retrieval-time locality, the per-query unit the model sees.
             Reports the DISTRIBUTION across balls + a hyp-wins rate.
  (backbone) the giant component with super-hubs degree-capped (kills the
             hub-collapse confound that broke the delta screen), connected
             snowball sample. One number per dim.

Run from scripts/ (Neo4j must be up; hits default `neo4j` DB per the stale
NEO4J_DATABASE note):

    py distortion_probe.py --config kettle_config.yaml --mode both
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from graph_diagnostics.core import (  # noqa: E402
    DiagnosticConfig, _driver, lifecycle_predicate,
)
from graph_diagnostics.graphcache import (  # noqa: E402
    build_cache, bfs, connected_components, snowball_sample,
)
from src.models.layers.poincare_ops import expmap0, dist  # noqa: E402

# Domain-entity labels that anchor real queries (from kettle_config domain_only)
_ANCHOR_LABELS = ["Entity", "Person", "Place", "Organization", "Refuge",
                  "Species", "Event", "Activity", "Observation"]


# ---------------------------------------------------------------------------
# Sample extraction -> a small connected graph as (n, edge_u, edge_v)
# ---------------------------------------------------------------------------

def _relabel(n_cache: int, member: np.ndarray, src: np.ndarray,
             dst: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Restrict edges to `member` nodes and relabel them to 0..k-1."""
    rl = np.full(n_cache, -1, dtype=np.int64)
    nodes = np.flatnonzero(member)
    rl[nodes] = np.arange(nodes.size)
    keep = member[src] & member[dst]
    u, v = rl[src[keep]], rl[dst[keep]]
    ok = u != v
    return nodes.size, u[ok], v[ok]


def khop_balls(cache, k: int, n_balls: int, min_n: int, max_n: int,
               seed: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """k-hop balls on the FULL undirected graph around sampled anchors."""
    rng = np.random.default_rng(seed)
    anchor = cache.label_mask(_ANCHOR_LABELS)
    cand = np.flatnonzero(anchor)
    rng.shuffle(cand)
    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    for s in cand:
        if len(out) >= n_balls:
            break
        d = bfs(cache.indptr, cache.indices, int(s), cache.n)
        ball = (d >= 0) & (d <= k)
        cnt = int(ball.sum())
        if cnt < min_n:
            continue
        if cnt > max_n:                       # keep the closest max_n nodes
            order = np.argsort(np.where(d >= 0, d, 1 << 30))
            ball = np.zeros(cache.n, dtype=bool)
            ball[order[:max_n]] = True
        n, u, v = _relabel(cache.n, ball, cache.src, cache.dst)
        if u.size >= n:                       # need enough edges to be graphy
            out.append((n, u, v))
    return out


def degree_capped_backbone(cache, cap: int, sample_n: int, seed: int
                           ) -> tuple[int, np.ndarray, np.ndarray]:
    """Giant component with degree>cap nodes removed (kills hub-collapse),
    then a connected snowball sample of <= sample_n nodes."""
    deg = cache.indptr[1:] - cache.indptr[:-1]
    keep = deg <= cap
    n, u, v = _relabel(cache.n, keep, cache.src, cache.dst)
    # CSR over the capped graph for component + snowball (numpy primitives)
    ru = np.concatenate([u, v]); rv = np.concatenate([v, u])
    order = np.argsort(ru, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, ru[order] + 1, 1)
    np.cumsum(indptr, out=indptr)
    indices = rv[order].astype(np.int64)
    comp = connected_components(indptr, indices, n)
    sizes = np.bincount(comp[comp >= 0])
    giant = int(sizes.argmax())
    gmask = comp == giant
    gnodes = np.flatnonzero(gmask)
    rng = np.random.default_rng(seed)
    if gnodes.size > sample_n:
        s0 = int(gnodes[rng.integers(gnodes.size)])
        samp = snowball_sample(indptr, indices, n, gmask, s0, sample_n)
        smask = np.zeros(n, dtype=bool); smask[samp] = True
    else:
        smask = gmask
    return _relabel(n, smask, u, v)


def all_pairs_dist(n: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Unweighted shortest-path matrix via BFS from every node (numpy)."""
    ru = np.concatenate([u, v]); rv = np.concatenate([v, u])
    order = np.argsort(ru, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, ru[order] + 1, 1)
    np.cumsum(indptr, out=indptr)
    indices = rv[order].astype(np.int64)
    D = np.empty((n, n), dtype=np.int32)
    for i in range(n):
        D[i] = bfs(indptr, indices, i, n)
    return D


# ---------------------------------------------------------------------------
# Embedding fit -- identical params for both arms; metric is the only change
# ---------------------------------------------------------------------------

def fit(D: np.ndarray, u: np.ndarray, v: np.ndarray, dim: int, arm: str,
        steps: int, lr: float, seed: int) -> dict:
    torch.manual_seed(seed)
    n = D.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    dg = D[iu, ju]
    fin = dg > 0
    iu, ju, dg = iu[fin], ju[fin], dg[fin]
    I = torch.from_numpy(iu).long()
    J = torch.from_numpy(ju).long()
    Dg = torch.from_numpy(dg.astype(np.float32))
    P = torch.nn.Parameter(torch.randn(n, dim) * 1e-3)        # shared init
    log_s = torch.nn.Parameter(torch.zeros(()))               # global scale
    params = [P, log_s]
    if arm == "hyp":
        rho = torch.nn.Parameter(torch.zeros(()))             # c = softplus -> ~0.69
        params.append(rho)
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(seed)
    npair = dg.size
    bs = min(16384, npair)
    t0 = time.monotonic()
    for _ in range(steps):
        idx = torch.from_numpy(rng.integers(0, npair, size=bs)).long()
        i, j, dt = I[idx], J[idx], Dg[idx]
        s = torch.nn.functional.softplus(log_s) + 1e-4
        if arm == "hyp":
            c = torch.nn.functional.softplus(rho) + 1e-4
            x = expmap0(P, c)
            de = dist(x[i], x[j], c)
        else:
            de = (P[i] - P[j]).norm(dim=-1)
        loss = ((s * de / dt - 1.0) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # ---- eval on ALL finite pairs ----
    with torch.no_grad():
        s = (torch.nn.functional.softplus(log_s) + 1e-4).item()
        if arm == "hyp":
            c = (torch.nn.functional.softplus(rho) + 1e-4).item()
            xb = expmap0(P, torch.tensor(c))
            Dm = _pairwise(xb, c)
        else:
            c = float("nan")
            Dm = torch.cdist(P, P)
        Dm = (Dm * s).numpy()
    mre = float(np.mean(np.abs(Dm[iu, ju] - dg) / dg))
    rday = Dm[iu, ju] / dg
    mult = float(rday.max() * (1.0 / rday.min()))             # Sarkar wc
    mp = _map(Dm, u, v, n)
    return {"arm": arm, "dim": dim, "n": n, "mre": mre,
            "mult_distortion": mult, "map": mp, "c": c,
            "secs": time.monotonic() - t0}


def _pairwise(x: torch.Tensor, c: float) -> torch.Tensor:
    n = x.shape[0]
    out = torch.empty(n, n)
    for i in range(n):                                        # row-wise: O(n) mem
        out[i] = dist(x[i:i + 1].expand_as(x), x, torch.tensor(c))
    return out


def _map(Dm: np.ndarray, u: np.ndarray, v: np.ndarray, n: int) -> float:
    """Mean average precision of 1-hop neighbor retrieval (Nickel-Kiela)."""
    adj: list[set] = [set() for _ in range(n)]
    for a, b in zip(u.tolist(), v.tolist()):
        adj[a].add(b); adj[b].add(a)
    order = np.argsort(Dm, axis=1)                            # nearest first
    aps: list[float] = []
    for node in range(n):
        nb = adj[node]
        if not nb:
            continue
        hit = 0
        prec = 0.0
        rank = 0
        for cand in order[node]:
            if cand == node:
                continue
            rank += 1
            if cand in nb:
                hit += 1
                prec += hit / rank
                if hit == len(nb):
                    break
        aps.append(prec / len(nb))
    return float(np.mean(aps)) if aps else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _verdict(rows: list[dict]) -> str:
    by = {(r["arm"], r["dim"]): r for r in rows}
    msgs = []
    for dim in sorted({r["dim"] for r in rows}):
        h, e = by.get(("hyp", dim)), by.get(("euc", dim))
        if not h or not e:
            continue
        dmre = e["mre"] - h["mre"]                            # >0 = hyp better
        dmap = h["map"] - e["map"]                            # >0 = hyp better
        tag = "HYP better" if (dmre > 0.02 and dmap > 0.01) else (
              "EUC >= HYP (regularization, not curvature)"
              if dmre <= 0.005 else "marginal / mixed")
        msgs.append(f"dim={dim}: dMRE={dmre:+.4f} dmAP={dmap:+.4f} "
                    f"c={h['c']:.3g} -> {tag}")
    return "  " + "\n  ".join(msgs) if msgs else "  (insufficient arms)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--mode", choices=["balls", "backbone", "both"],
                    default="both")
    ap.add_argument("--dims", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--khop", type=int, default=3)
    ap.add_argument("--n-balls", type=int, default=20)
    ap.add_argument("--ball-min", type=int, default=80)
    ap.add_argument("--ball-max", type=int, default=600)
    ap.add_argument("--hub-cap", type=int, default=50)
    ap.add_argument("--backbone-n", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = DiagnosticConfig.from_yaml(args.config)
    driver = _driver()
    try:
        driver.verify_connectivity()
        with driver.session() as session:        # no db= -> default `neo4j`
            print("Pulling lifecycle-clean graph (one query pair)...")
            cache = build_cache(session, lifecycle_predicate(cfg, "n"),
                                progress=lambda m: print(" ", m))
    finally:
        driver.close()

    if args.mode in ("balls", "both"):
        print(f"\n==== (balls) {args.khop}-hop balls around domain anchors, "
              f"full graph; per-query locality ====")
        balls = khop_balls(cache, args.khop, args.n_balls, args.ball_min,
                           args.ball_max, args.seed)
        print(f"kept {len(balls)} balls "
              f"(sizes {[b[0] for b in balls]})")
        for dim in args.dims:
            dmres, dmaps, cs, hw = [], [], [], 0
            for bi, (n, u, v) in enumerate(balls):
                D = all_pairs_dist(n, u, v)
                h = fit(D, u, v, dim, "hyp", args.steps, args.lr,
                        args.seed + bi)
                e = fit(D, u, v, dim, "euc", args.steps, args.lr,
                        args.seed + bi)
                dmres.append(e["mre"] - h["mre"])
                dmaps.append(h["map"] - e["map"])
                cs.append(h["c"])
                hw += int(h["mre"] < e["mre"])
            print(f"  dim={dim}: over {len(balls)} balls  "
                  f"median dMRE={statistics.median(dmres):+.4f} "
                  f"(mean {statistics.mean(dmres):+.4f})  "
                  f"median dmAP={statistics.median(dmaps):+.4f}  "
                  f"hyp-wins-MRE={hw}/{len(balls)}  "
                  f"c~{statistics.median(cs):.3g}  "
                  f"[>0 favors hyp; ~0/neg = regularization story]")

    if args.mode in ("backbone", "both"):
        print(f"\n==== (backbone) giant comp, degree<= {args.hub_cap}, "
              f"snowball<= {args.backbone_n}; hub-collapse removed ====")
        n, u, v = degree_capped_backbone(cache, args.hub_cap,
                                         args.backbone_n, args.seed)
        print(f"backbone sample: n={n:,}, edges={u.size:,}, "
              f"mean_deg={2 * u.size / max(1, n):.2f}")
        D = all_pairs_dist(n, u, v)
        diam = int(D[D >= 0].max())
        print(f"sample diameter={diam}")
        rows = []
        for dim in args.dims:
            for arm in ("hyp", "euc"):
                r = fit(D, u, v, dim, arm, args.steps, args.lr, args.seed)
                rows.append(r)
                print(f"  {arm} dim={dim:>2}: MRE={r['mre']:.4f} "
                      f"mAP={r['map']:.4f} "
                      f"mult_distortion={r['mult_distortion']:.2f} "
                      f"c={r['c']:.4g} ({r['secs']:.0f}s)")
        print("verdict (backbone):")
        print(_verdict(rows))

    print("\nDecision rule: hyperbolic is curvature-JUSTIFIED only if it "
          "shows materially lower MRE AND higher mAP than Euclidean at "
          "matched dim, with learnable c NOT collapsing toward ~1e-4. "
          "If Euclidean matches it at equal capacity, the +0.094 retrieval "
          "edge is regularization -- a pure-hyperbolic Gemynd backbone is "
          "unjustified and a cheaper Euclidean+norm-constraint should be "
          "tested as the real comparator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

r"""V2 MVP-3 baseline: chain recovery on the live long-chain testbed.

For each exported chain (src/data/corpus/long_chains_v1.jsonl), starting
from the head node only, how much of the chain does each zero-training
strategy recover within a fixed visited-node budget? This sets the bar
the learned propagation engine (MVP-3) must beat.

Strategies (graph-only; the emb-pruned walk joins once neighborhoods are
exported + encoded):
  bfs_ball   plain BFS over chain rels, radius 4 (the retrieval ball) --
             blind beyond radius BY CONSTRUCTION for len>4 chains
  typed_beam beam walk (width B, depth 9): frontier pruned by same-type
             coherence bonus + low out-degree preference (hub avoidance)
  random_beam same budget, random pruning (control)

Metrics: chain-node recall within budget; TAIL recall (chain nodes at
hop>4 from head -- the regime ball retrieval cannot see).

    py -m scripts.chain_recovery_baseline --n-chains 100 --beam 5
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

from scripts.neo4j_reader import get_driver

CHAIN_RELS = ["HAS_CLAIM", "SOURCED_FROM", "EVIDENCED_BY", "PROCESSED_BY"]


def fetch_neighborhood(session, head: str, depth: int, cap: int):
    """Directed frontier expansion over chain rels. Returns adjacency
    {u: [(v, rel_type)]} bounded by ``cap`` nodes."""
    adj: dict[str, list] = {}
    seen = {head}
    frontier = [head]
    for _ in range(depth):
        if not frontier or len(seen) >= cap:
            break
        recs = session.run(
            "MATCH (u)-[r]->(v) WHERE elementId(u) IN $f "
            "AND type(r) IN $rels "
            "RETURN elementId(u) AS u, type(r) AS t, elementId(v) AS v",
            f=frontier, rels=CHAIN_RELS)
        nxt = []
        for rec in recs:
            adj.setdefault(rec["u"], []).append((rec["v"], rec["t"]))
            if rec["v"] not in seen and len(seen) < cap:
                seen.add(rec["v"])
                nxt.append(rec["v"])
        frontier = nxt
    return adj


def bfs_ball(adj, head, radius):
    d = {head: 0}
    q = deque([head])
    while q:
        u = q.popleft()
        if d[u] >= radius:
            continue
        for v, _t in adj.get(u, []):
            if v not in d:
                d[v] = d[u] + 1
                q.append(v)
    return d


def beam_walk(adj, head, beam, depth, budget, rng=None):
    """Typed-coherence beam walk (rng=None) or random beam (rng set)."""
    visited = [head]
    vset = {head}
    frontier = [(head, None)]
    outdeg = {u: len(vs) for u, vs in adj.items()}
    for _ in range(depth):
        cand = []
        for u, ut in frontier:
            for v, vt in adj.get(u, []):
                if v in vset:
                    continue
                if rng is not None:
                    score = rng.random()
                else:
                    coh = 0.0 if vt == ut else 1.0     # same-type bonus
                    score = coh + 0.01 * outdeg.get(v, 0)  # hub avoidance
                cand.append((score, v, vt))
        cand.sort(key=lambda x: (x[0], x[1]))
        keep = []
        for sc, v, vt in cand:
            if v in vset:
                continue
            vset.add(v)
            visited.append(v)
            keep.append((v, vt))
            if len(keep) >= beam or len(visited) >= budget:
                break
        frontier = keep
        if not frontier or len(visited) >= budget:
            break
    return vset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", default="src/data/corpus/long_chains_v1.jsonl")
    ap.add_argument("--n-chains", type=int, default=100)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--out", default="runs/chain_recovery_baseline")
    args = ap.parse_args()

    rows_in = [json.loads(l) for l in open(args.chains, encoding="utf-8")]
    # one chain per head, longest first, up to n
    by_head: dict[str, dict] = {}
    for r in sorted(rows_in, key=lambda r: -r["len"]):
        by_head.setdefault(r["head"], r)
    chains = list(by_head.values())[: args.n_chains]
    print(f"{len(chains)} chains (one per head)")

    drv = get_driver()
    rng = random.Random(0)
    res = []
    with drv.session() as s:
        for ch in chains:
            head, targets = ch["head"], set(ch["nodes"][1:])
            adj = fetch_neighborhood(s, head, args.depth, args.cap)
            d = bfs_ball(adj, head, 4)
            tail = {t for t in targets if t not in d or d[t] > 4}
            ball_hits = sum(1 for t in targets if t in d and d[t] <= 4)
            tb = beam_walk(adj, head, args.beam, args.depth, args.budget)
            rb = beam_walk(adj, head, args.beam, args.depth, args.budget,
                           rng=rng)
            row = {
                "len": ch["len"], "n_tail": len(tail),
                "ball_recall": ball_hits / len(targets),
                "typed_recall": len(tb & targets) / len(targets),
                "random_recall": len(rb & targets) / len(targets),
                "typed_tail": (len(tb & tail) / len(tail)) if tail else None,
                "random_tail": (len(rb & tail) / len(tail)) if tail else None,
            }
            res.append(row)

    def _avg(key):
        v = [r[key] for r in res if r[key] is not None]
        return sum(v) / len(v) if v else float("nan")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {k: _avg(k) for k in
               ("ball_recall", "typed_recall", "random_recall",
                "typed_tail", "random_tail")}
    summary["n_chains"] = len(res)
    summary["config"] = vars(args)
    (out / "recovery_results.json").write_text(
        json.dumps({"summary": summary, "rows": res}, indent=2))
    print(f"\n=== chain recovery (budget {args.budget} nodes, "
          f"beam {args.beam}) ===")
    print(f"  bfs_ball  (r<=4) recall: {summary['ball_recall']:.3f}")
    print(f"  typed_beam       recall: {summary['typed_recall']:.3f}   "
          f"TAIL(hop>4): {summary['typed_tail']:.3f}")
    print(f"  random_beam      recall: {summary['random_recall']:.3f}   "
          f"TAIL(hop>4): {summary['random_tail']:.3f}")
    print(f"\nreport: {out / 'recovery_results.json'}")


if __name__ == "__main__":
    main()

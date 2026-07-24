r"""Exercise the KGR Context Service against GROUND TRUTH -- what does it
actually do? Uses the all6 task labels as the relevant set and measures
whether the service's ordered context is genuinely relevant.

    py -m scripts.kgr_context_exercise
"""

from __future__ import annotations

import glob
from collections import deque, defaultdict

import numpy as np

from src.service.context_service import KGRContextService

CORPUS = "src/data/corpus/real_domain_eval_all6"
FAMILY = {0: "provenance", 1: "entity", 2: "temporal", 3: "multihop",
          4: "subgraph", 5: "compound"}
K = 10


def _bfs(adj, src, n):
    d = np.full(n, -1, np.int64); d[src] = 0; q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if d[v] < 0:
                d[v] = d[u] + 1; q.append(v)
    return d


def main() -> None:
    svc = KGRContextService()
    files = sorted(glob.glob(f"{CORPUS}/*.npz"))[:60]

    # ---- 1. does the ordered context actually surface relevant nodes? ----
    prec = defaultdict(list); rec = defaultdict(list)
    ma_single, ma_multi = [], []       # multi-anchor on compound
    wrong_prec, right_prec = [], []    # anchor-fragility
    wrong_disc, right_disc = [], []
    rng = np.random.default_rng(0)

    for f in files:
        z = dict(np.load(f, allow_pickle=True))
        h = svc.load_graph(z)
        n = h.n
        for i in range(int(z["n_tasks"])):
            fam = FAMILY.get(int(z[f"task_{i}_type"]), "?")
            anchor = int(z[f"task_{i}_anchor_row"])
            labels = z[f"task_{i}_labels"]
            mh = int(z[f"task_{i}_max_hops"])
            rel = set(np.flatnonzero(labels >= 0.5).tolist()) - {anchor}
            if not rel:
                continue
            res = svc.order_context(h, anchor, top_k=K, ball_hops=mh)
            if not res.items:
                continue
            got = [it.row for it in res.items]
            hit = sum(1 for r in got if r in rel)
            prec[fam].append(hit / len(got))
            rec[fam].append(hit / len(rel))
            right_prec.append(hit / len(got))
            right_disc.append(res.discrimination)

            # anchor-fragility: same task, WRONG anchor (random node)
            wa = int(rng.integers(0, n))
            if wa != anchor:
                rw = svc.order_context(h, wa, top_k=K, ball_hops=mh)
                if rw.items:
                    gw = [it.row for it in rw.items]
                    wrong_prec.append(sum(1 for r in gw if r in rel) / len(gw))
                    wrong_disc.append(rw.discrimination)

            # multi-anchor on compound: add a real 2nd relevant node
            if fam == "compound" and len(rel) >= 2:
                adj = h._adj
                d = _bfs(adj, anchor, n)
                far = max(rel, key=lambda r: d[r] if d[r] >= 0 else 999)
                s = svc.order_context(h, anchor, top_k=K, ball_hops=mh)
                m = svc.order_context(h, [anchor, far], top_k=K, ball_hops=mh)
                relm = rel - {far}
                if s.items and m.items and relm:
                    ma_single.append(
                        sum(1 for it in s.items if it.row in relm) / len(s.items))
                    ma_multi.append(
                        sum(1 for it in m.items if it.row in relm) / len(m.items))

    print("=== 1. ordered-context precision/recall @10 vs GROUND TRUTH ===")
    print(f"  {'family':<12}{'n':>5}{'precision':>11}{'recall':>9}")
    for fam in sorted(prec):
        print(f"  {fam:<12}{len(prec[fam]):>5}"
              f"{np.mean(prec[fam]):>11.3f}{np.mean(rec[fam]):>9.3f}")
    allp = [x for v in prec.values() for x in v]
    allr = [x for v in rec.values() for x in v]
    print(f"  {'ALL':<12}{len(allp):>5}{np.mean(allp):>11.3f}{np.mean(allr):>9.3f}")

    print("\n=== 2. multi-anchor vs single on COMPOUND (real 2nd anchor) ===")
    if ma_single:
        print(f"  single-anchor precision@10: {np.mean(ma_single):.3f}")
        print(f"  multi-anchor  precision@10: {np.mean(ma_multi):.3f}   "
              f"(delta {np.mean(ma_multi)-np.mean(ma_single):+.3f}, n={len(ma_single)})")
    else:
        print("  (no evaluable compound cases)")

    print("\n=== 3. anchor-fragility: right vs WRONG (random) anchor ===")
    print(f"  right anchor: precision@10 {np.mean(right_prec):.3f}  "
          f"discrimination {np.mean(right_disc):.3f}")
    print(f"  wrong anchor: precision@10 {np.mean(wrong_prec):.3f}  "
          f"discrimination {np.mean(wrong_disc):.3f}   (n={len(wrong_prec)})")
    print(f"  -> a wrong anchor drops precision by "
          f"{np.mean(right_prec)-np.mean(wrong_prec):+.3f}; discrimination is "
          f"the caller-visible warning signal.")

    # ---- 4. missing-link mode on the real code graph ----
    print("\n=== 4. missing-link mode (tutorstructure code graph) ===")
    try:
        from src.codegraph.ingest import build_npz
        from src.codegraph import cases as C
        from src.codegraph.harness import TASKS
        from pathlib import Path
        import json
        repo = Path("../tutorstructure_patch")
        cg = build_npz(repo, Path("runs/_exercise_tutor.npz"),
                       C.collect_required_edges(repo, TASKS))
        h = svc.load_graph(cg.npz_path)
        names = {}
        for line in open(repo / "nodes.jsonl", encoding="utf-8"):
            d = json.loads(line); names[d.get("id")] = (d.get("name") or "")[:40]
        row_to_id = {r: i for i, r in cg.id_to_row.items()}
        res = svc.suggest_missing_links(h, 100, top_k=6, min_hop=2)
        anm = names.get(row_to_id.get(100), "row100")
        print(f"  anchor: {anm}  (top missing-link candidates, hop>=2):")
        for it in res.items:
            print(f"    {names.get(row_to_id.get(it.row),'?'):<40} "
                  f"hop={it.hop} score={it.score:.3f}")
    except Exception as e:
        print(f"  (skipped: {type(e).__name__}: {e})")

    # ---- 5. live Neo4j round-trip (if up) ----
    print("\n=== 5. live Neo4j round-trip ===")
    try:
        import subprocess, sys
        out = "runs/_exercise_live"
        subprocess.run(
            [sys.executable, "neo4j_eval_export.py", "export", "--config",
             "kettle_config.yaml", "--out", f"../{out}", "--num-graphs", "1",
             "--max-nodes", "300", "--tasks-per-graph", "1", "--seed", "321",
             "--sampler", "delocalized", "--n-seeds", "4"],
            cwd="scripts", check=True, capture_output=True, timeout=120)
        f = sorted(glob.glob(f"{out}/graph_*.npz"))[0]
        z = dict(np.load(f, allow_pickle=True))
        h = svc.load_graph(z)
        anchor = int(z["task_0_anchor_row"])
        res = svc.order_context(h, anchor, top_k=5, ball_hops=4)
        print(f"  live graph {h.n} nodes, anchor neo4j id {h.node_ids[anchor]}")
        for it in res.items:
            print(f"    rank {it.rank}: neo4j id {it.node_id}  score {it.score:.3f}  {it.rationale}")
    except Exception as e:
        print(f"  (Neo4j unavailable: {type(e).__name__})")


if __name__ == "__main__":
    main()

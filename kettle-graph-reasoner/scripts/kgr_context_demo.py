r"""KGR Context Service — end-to-end demo on a real archival graph.

Shows the deployable path: load a query neighborhood, order context for
one or two anchors, and print the annotated top-k an LLM would receive.

    py -m scripts.kgr_context_demo [--graph N] [--topk 12]
"""

from __future__ import annotations

import argparse
import glob

from src.service.context_service import KGRContextService

CORPUS = "src/data/corpus/real_domain_eval_all6"


def _print(title, res):
    print(f"\n{title}")
    print(f"  candidates={res.n_candidates}  discrimination="
          f"{res.discrimination:.3f}  anchors={res.anchors}")
    print(f"  {'rank':>4} {'node_id':>10} {'score':>8} {'hop':>4}  rationale")
    for it in res.items:
        print(f"  {it.rank:>4} {str(it.node_id):>10} {it.score:>8.3f} "
              f"{str(it.hop):>4}  {it.rationale}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=int, default=0)
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()

    svc = KGRContextService()
    f = sorted(glob.glob(f"{CORPUS}/*.npz"))[args.graph]
    print(f"loading {f}")
    h = svc.load_graph(f)
    print(f"embedded {h.n} nodes (frozen encoder)")

    # single-anchor ball ordering (the near-oracle capability)
    _print("=== single-anchor context (ball hop<=4) ===",
           svc.order_context(h, 0, top_k=args.topk, ball_hops=4))

    # multi-anchor min-distance (compound/union queries)
    a2 = h.n // 3
    _print(f"=== multi-anchor context (anchors 0 + {a2}) ===",
           svc.order_context(h, [0, a2], top_k=args.topk, ball_hops=4))

    # whole-neighborhood (no ball restriction) for broad context
    _print("=== whole-neighborhood context (no ball limit) ===",
           svc.order_context(h, 0, top_k=args.topk))


if __name__ == "__main__":
    main()

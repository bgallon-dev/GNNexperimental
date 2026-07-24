r"""V2 MVP-3 testbed: export LONG provenance chains from the live graph.

Chains of length >= --min-len over the provenance rel types
(HAS_CLAIM / SOURCED_FROM / EVIDENCED_BY / PROCESSED_BY, per
kettle_config.yaml `provenance_chain`), i.e. longer than the retrieval
ball radius (max_hops 4) — the regime where ball-ordering is blind by
construction and a chain engine (typed walk / learned propagation) must
earn its keep. Output: JSONL of {head, nodes, rel_types, labels, len}.

    py -m scripts.export_long_chains --min-len 6 --max-len 9 --limit 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.neo4j_reader import get_driver  # reuses .env + driver setup

CHAIN_RELS = "HAS_CLAIM|SOURCED_FROM|EVIDENCED_BY|PROCESSED_BY"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-len", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=9)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--require-terminal", action="store_true",
                    help="only maximal chains (endpoint has no outgoing "
                         "chain rel); default off - endpoints of long "
                         "chains in this graph continue onward")
    ap.add_argument("--out", default="src/data/corpus/long_chains_v1.jsonl")
    args = ap.parse_args()

    term = ("WHERE NOT (b)-[:" + CHAIN_RELS + "]->() "
            if args.require_terminal else "")
    q = (
        f"MATCH p=(a)-[:{CHAIN_RELS}*{args.min_len}..{args.max_len}]->(b) "
        + term +
        "WITH p LIMIT $lim "
        "RETURN [n IN nodes(p) | elementId(n)] AS ids, "
        "       [n IN nodes(p) | labels(n)[0]] AS labs, "
        "       [r IN relationships(p) | type(r)] AS rels"
    )
    drv = get_driver()
    rows = []
    with drv.session() as s:  # default db per project convention
        for rec in s.run(q, lim=args.limit):
            rows.append({
                "head": rec["ids"][0],
                "nodes": rec["ids"],
                "labels": rec["labs"],
                "rel_types": rec["rels"],
                "len": len(rec["ids"]) - 1,
            })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    lens = [r["len"] for r in rows]
    print(f"exported {len(rows)} chains "
          f"(len {min(lens) if lens else 0}-{max(lens) if lens else 0}) "
          f"-> {out}")


if __name__ == "__main__":
    main()

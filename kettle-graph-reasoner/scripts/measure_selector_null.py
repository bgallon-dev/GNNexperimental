r"""Measure whether the evidence selector returns anything on-topic.

Reproduces (and re-runs) the blocking table in
``Docs/COMPRESSION_SCREEN_PREREGISTRATION.md`` §0: compile every persisted
candidate bundle offline under a ranking strategy and ask, of the resulting
``core_evidence`` items, how many contain any of their own question's anchor
terms, and how many are empty ``(mention pointer; see provenance)`` stubs.

This is the gate instrument for the prerequisite in §3: the BFS anchor-term
rate must exceed ~9% by a wide margin before any arm comparison means
anything. Run it before and after any selector change.

Anchor terms come from the graded fixture whose question text matches the
bundle's, across both fixture corpora (dev/pilot and the historical stress
questions). ``--terms names`` instead uses each question's own resolved anchor
display names, which is the weaker, more generous reading; the default
``fixture`` is the stricter one the pre-registration used.

Usage (from kettle-graph-reasoner/):

    py -m scripts.measure_selector_null
    py -m scripts.measure_selector_null --strategy lexical
    py -m scripts.measure_selector_null --json runs/selector/bfs_before.json

Reads persisted artifacts read-only. Touches no database.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import sys

from src.evidence.offline import compile_bundle, load_bundle, load_question

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PACKETS = _ROOT / "research_workspace" / "packets"
_FIXTURE_GLOBS = (
    str(_ROOT / "tests" / "fixtures" / "evidence_corpus" / "**" / "*.json"),
    str(_ROOT.parent / "historical-research-packets" / "stress-questions"
        / "fixtures" / "*.json"),
)
# The compiler's placeholder for a node it seated in an evidence slot without
# any renderable text. 33% of BFS core evidence was this at measurement time.
_STUB = "(mention pointer; see provenance)"


def _fixture_terms() -> dict[str, tuple[str, list[str]]]:
    """question text -> (fixture id, anchor terms), across both corpora."""
    out: dict[str, tuple[str, list[str]]] = {}
    for pattern in _FIXTURE_GLOBS:
        for f in glob.glob(pattern, recursive=True):
            try:
                d = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(d, dict) and d.get("anchor_terms") and d.get("text"):
                out[d["text"].strip()] = (d.get("fixture_id", pathlib.Path(f).stem),
                                          list(d["anchor_terms"]))
    return out


def _terms_for(question, fixtures, mode):
    """Anchor terms for one question, plus where they came from."""
    if mode == "names":
        return ([c.display_name for c in question.candidates], "anchor-names")
    hit = fixtures.get(question.text.strip())
    if hit:
        return (hit[1], hit[0])
    # No graded fixture: fall back to resolved anchor names so the bundle is
    # still scored rather than silently dropped from the denominator.
    return ([c.display_name for c in question.candidates], "fallback-names")


def _on_topic(display: str, terms) -> bool:
    low = display.lower()
    return any(t.lower() in low for t in terms if t)


def _revisions():
    for q in sorted(glob.glob(str(_PACKETS / "*" / "revisions" / "*" /
                                  "question.json"))):
        d = pathlib.Path(q).parent
        if (d / "candidates.json").exists():
            yield d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strategy", default="bfs", help="bfs | lexical")
    ap.add_argument("--terms", default="fixture", choices=("fixture", "names"))
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--verbose", action="store_true",
                    help="list each bundle's rate")
    args = ap.parse_args(argv)

    fixtures = _fixture_terms()
    rows, errors = [], []
    n_items = n_topic = n_stub = 0

    for d in _revisions():
        try:
            question = load_question(d / "question.json")
            bundle = load_bundle(d / "candidates.json")
            packet = compile_bundle(question, bundle, strategy=args.strategy)
        except Exception as exc:                       # report, never mask
            errors.append({"bundle": str(d.relative_to(_ROOT)),
                           "error": f"{type(exc).__name__}: {exc}"})
            continue

        terms, origin = _terms_for(question, fixtures, args.terms)
        items = packet.get("core_evidence", [])
        displays = [i.get("display", "") for i in items]
        topic = sum(1 for s in displays if _on_topic(s, terms))
        stubs = sum(1 for s in displays if _STUB in s)
        n_items += len(items)
        n_topic += topic
        n_stub += stubs
        rows.append({
            "bundle": d.parent.parent.name, "revision": d.name,
            "question_id": question.question_id, "terms_from": origin,
            "n_domain_nodes": len(bundle.nodes),
            "n_provenance_nodes": len(bundle.provenance_nodes),
            "n_core": len(items), "n_on_topic": topic, "n_stub": stubs,
        })

    if errors:
        print(f"!! {len(errors)} bundle(s) failed to compile:")
        for e in errors[:5]:
            print(f"   {e['bundle']}: {e['error']}")

    if not n_items:
        print("no core evidence compiled; nothing to measure")
        return 1

    zero = sum(1 for r in rows if r["n_on_topic"] == 0)
    allstub = sum(1 for r in rows if r["n_core"] and r["n_stub"] == r["n_core"])
    nodes = sorted({r["n_domain_nodes"] for r in rows})

    if args.verbose:
        print(f"{'bundle':<22} {'rev':<5} {'nodes':>5} {'core':>5} "
              f"{'topic':>6} {'stub':>5}  terms_from")
        for r in sorted(rows, key=lambda r: -r["n_on_topic"]):
            print(f"  {r['bundle']:<20} {r['revision']:<5} "
                  f"{r['n_domain_nodes']:>5} {r['n_core']:>5} "
                  f"{r['n_on_topic']:>6} {r['n_stub']:>5}  {r['terms_from']}")

    print(f"\n=== selector null screen — strategy={args.strategy} "
          f"terms={args.terms} ===")
    print(f"  bundles compiled                {len(rows)}")
    print(f"  domain nodes per bundle         {nodes}   "
          f"<- the list every arm ranks")
    print(f"  core_evidence items             {n_items}")
    print(f"  ...containing own anchor terms  {n_topic}/{n_items}   "
          f"{100 * n_topic / n_items:.1f}%")
    print(f"  ...empty '{_STUB}'  {n_stub}/{n_items}   "
          f"{100 * n_stub / n_items:.1f}%")
    print(f"  bundles with ZERO on-topic      {zero}/{len(rows)}   "
          f"{100 * zero / len(rows):.0f}%")
    print(f"  bundles 100% stubs              {allstub}/{len(rows)}")
    print(f"\n  GATE: anchor-term rate must far exceed 9.0% for an arm "
          f"comparison to mean anything.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "strategy": args.strategy, "terms_mode": args.terms,
            "n_bundles": len(rows), "n_items": n_items,
            "n_on_topic": n_topic, "n_stub": n_stub,
            "on_topic_rate": n_topic / n_items, "stub_rate": n_stub / n_items,
            "bundles_zero_on_topic": zero, "bundles_all_stub": allstub,
            "domain_node_counts": nodes, "rows": rows, "errors": errors,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

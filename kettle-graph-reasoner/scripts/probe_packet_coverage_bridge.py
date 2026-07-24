r"""Close the packet -> coverage-verdict loop on a real historical packet.

The seven frozen negative controls in ``tests/test_evidence_coverage.py`` are
hand-authored assignment lists: a human read STRESS_TEST_FINDINGS.md, decided
the honest verdict, and then wrote the assignments that produce it. The chain
runs findings -> shapes -> controls -> findings, so passing them shows the
kernel *encodes* the historical conclusion, not that it *derives* it from
evidence. A shape overfitted to its own stress question passes silently.

This probe feeds a real ``packet.json`` to the kernel instead, and measures
what the kernel contributes on its own.

The bridge declares provenance; it never decides admissibility. Each assignment
carries the source class derived from its evidence item's provenance terminal
(see _SOURCE_CLASS_BY_NODE_TYPE), and the kernel's schema-0.2.0 source-class
rule decides what that can support. Four arms:

  A  declared    every item assigned to every slot, each declaring its TRUE
                 derived source class. Honest sidecar, maximally permissive
                 placement. The kernel filters.

  B  undeclared  every item to every slot, declaring no source class at all.
                 The fail-closed path: silence about provenance must not read
                 as acceptable provenance.

  B2 lying       every item to every slot, each falsely declaring whatever
                 source class the slot wants. Measures the residual: the kernel
                 never sees the packet, so it cannot catch a false declaration.

  C  repaired    arm A against a synthetic packet carrying the structure the
                 real one lacks (person records, case files, outcome records).
                 Discriminating control: if A abstains and C answers, the rule
                 responds to evidence rather than being rigged to abstain.
                 Without C, arm A proves nothing -- an always-abstain kernel
                 would also "reproduce" the finding.

Note that arms A/B/B2 place evidence in every slot, so the kernel never falls
back on the recorded ``EvidenceUniverse``: their verdicts do not depend on the
probe's search-state derivation at all. Only admissibility is load-bearing.

At schema 0.1.0, before CoverageAssignment carried a source class, arm A and
arm B were both ANSWERABLE.

Usage (from kettle-graph-reasoner/):

    py -m scripts.probe_packet_coverage_bridge
    py -m scripts.probe_packet_coverage_bridge --json runs/coverage_bridge.json

Reads the packet read-only. Touches no database.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from src.evidence.coverage import (
    CoverageAssignment,
    EvidenceUniverse,
    ShapeSelection,
    SourceCoverage,
    evaluate,
    get_shape,
)

# Repo-root-relative default: the sharpest packet-full/answer-empty case in the
# corpus (4 core items, person-level answer impossible per the findings doc).
_DEFAULT_PACKET = (pathlib.Path(__file__).resolve().parents[2]
                   / "historical-research-packets" / "stress-questions"
                   / "person-level-immigration-outcomes" / "packet.json")
_SHAPE_ID = "person_outcome"

# Declared structural mapping: provenance-terminal node type -> source class.
# This is an INPUT ASSUMPTION of the bridge, not a derivation: in this corpus a
# Document node is a newspaper issue (every terminal here carries a newspaper
# ext.kgr.source_title). It is stated here so a reviewer can reject it.
_SOURCE_CLASS_BY_NODE_TYPE = {
    "Document": "newspaper_ocr",
    "PersonRecord": "person_records",
    "CaseFile": "case_files",
    "InstitutionRecord": "institutional_records",
    "OutcomeRecord": "outcome_records",
}


def _node_type(public_key: str) -> str:
    """Node type of a packet public key ('Document:doc_abc' -> 'Document')."""
    return public_key.split(":", 1)[0] if ":" in public_key else public_key


def _item_facts(item: dict) -> dict:
    """Structural facts of one core_evidence item. Reads identifiers and node
    types only -- never 'display', 'rationale', or any other prose field."""
    path = item.get("provenance_path") or []
    terminal = path[-1] if path else ""
    terminal_type = _node_type(terminal)
    about = item.get("about") or ""
    return {
        "evidence_id": item["evidence_id"],
        "terminal": terminal,
        "terminal_type": terminal_type,
        "source_class": _SOURCE_CLASS_BY_NODE_TYPE.get(terminal_type, ""),
        # reported as a diagnostic; no arm's rule reads it
        "about_type": _node_type(about),
        "about_entity_kind": (about.split(":", 1)[1].split("_", 1)[0]
                              if ":" in about else ""),
    }


def _universe_from(facts: list[dict]) -> EvidenceUniverse:
    """A source class is SEARCHED iff some item in the packet actually came
    from it. Everything else is left to the kernel's NOT_SEARCHED default,
    which never asserts absence. Recorded for the artifact; not load-bearing
    for any arm here, since every arm fills every slot."""
    searched = sorted({f["source_class"] for f in facts if f["source_class"]})
    return EvidenceUniverse(sources=tuple(
        SourceCoverage(source_class=sc, state="SEARCHED",
                       detail="at least one packet item derives from this class")
        for sc in searched))


def _assign(shape, facts, prefix, source_class_fn, code):
    """Every item -> every slot. The arms differ only in what source class each
    assignment declares."""
    return [CoverageAssignment(
        assignment_id=f"{prefix}::{slot.slot_id}::{f['evidence_id']}",
        evidence_id=f["evidence_id"], slot_id=slot.slot_id,
        relation="supports", assignment_source="model_proposed",
        source_class=source_class_fn(slot, f), review_status="proposed",
        rationale_codes=(code,))
        for slot in shape.slots for f in facts]


def _assign_declared(shape, facts):
    """Arm A: declare the item's true derived provenance."""
    return _assign(shape, facts, "A", lambda slot, f: f["source_class"],
                   "declared_from_provenance")


def _assign_undeclared(shape, facts):
    """Arm B: declare nothing."""
    return _assign(shape, facts, "B", lambda slot, f: "", "undeclared")


def _assign_lying(shape, facts):
    """Arm B2: declare whatever the slot wants, regardless of the evidence."""
    return _assign(shape, facts, "B2",
                   lambda slot, f: (slot.source_classes[0]
                                    if slot.source_classes else ""),
                   "declared_falsely")


def _repaired_facts() -> list[dict]:
    """Arm C: the synthetic packet the real one would have to become. Same
    shape of record, but provenance terminals in the record series the real
    corpus lacks -- so the kernel admits them."""
    spec = [("R01", "PersonRecord:jail_register_1909:booking:0412"),
            ("R02", "CaseFile:bureau_immigration:warrant:14771"),
            ("R03", "InstitutionRecord:spokane_police:blotter:1909-11-11"),
            ("R04", "OutcomeRecord:bureau_immigration:disposition:14771")]
    return [{"evidence_id": eid, "terminal": term,
             "terminal_type": _node_type(term),
             "source_class": _SOURCE_CLASS_BY_NODE_TYPE[_node_type(term)],
             "about_type": "Entity", "about_entity_kind": "person"}
            for eid, term in spec]


def _run(shape, facts, assigner, label):
    assignments = assigner(shape, facts)
    slots, gates, verdict, frontier = evaluate(
        shape, ShapeSelection(shape_id=shape.shape_id,
                              shape_version=shape.shape_version,
                              review_status="resolved",
                              selected_by="probe_packet_coverage_bridge"),
        assignments, _universe_from(facts))
    return {
        "arm": label,
        "n_assignments": len(assignments),
        "verdict": verdict.status,
        "permitted_claim_classes": list(verdict.permitted_claim_classes),
        "reasons": [r.code for r in verdict.reasons],
        "slots": [{"slot_id": c.slot_id, "status": c.status,
                   "reason_codes": list(c.reason_codes),
                   "assigned": list(c.assigned_evidence_ids)} for c in slots],
        "frontier": [{"missing_slot": t.missing_slot,
                      "source_class": t.source_class,
                      "reason_code": t.reason_code} for t in frontier],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--packet", type=pathlib.Path, default=_DEFAULT_PACKET)
    ap.add_argument("--json", type=pathlib.Path, default=None,
                    help="write the full result record here")
    args = ap.parse_args(argv)

    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    shape = get_shape(_SHAPE_ID)
    facts = [_item_facts(i) for i in packet["core_evidence"]]

    print(f"packet   {args.packet.name}  q={packet['question_id']}")
    print(f"shape    {shape.shape_id}@{shape.shape_version} "
          f"(schema {shape.schema_version})")
    print(f"\n-- structural facts ({len(facts)} core items) {'-' * 26}")
    for f in facts:
        print(f"  {f['evidence_id']}  terminal={f['terminal_type']:<14} "
              f"source_class={f['source_class'] or '<unmapped>':<16} "
              f"about={f['about_type']}/{f['about_entity_kind']}")
    n_person = sum(1 for f in facts if f["about_entity_kind"] == "person")
    print(f"\n  diagnostic (no rule reads it): {n_person}/{len(facts)} items "
          f"are ABOUT a person-typed entity")
    print(f"  packet provides: "
          f"{sorted({f['source_class'] for f in facts if f['source_class']})}")
    print(f"  shape requires:  "
          f"{sorted({sc for s in shape.slots for sc in s.source_classes})}")

    results = [
        _run(shape, facts, _assign_declared, "A  declared (real packet)"),
        _run(shape, facts, _assign_undeclared, "B  undeclared (real packet)"),
        _run(shape, facts, _assign_lying, "B2 lying (real packet)"),
        _run(shape, _repaired_facts(), _assign_declared,
             "C  declared (repaired synthetic)"),
    ]

    for r in results:
        print(f"\n-- {r['arm']} {'-' * (54 - len(r['arm']))}")
        print(f"  assignments {r['n_assignments']:>3}   VERDICT  {r['verdict']}"
              f"   permits={r['permitted_claim_classes'] or '()'}")
        for s in r["slots"]:
            print(f"    {s['slot_id']:<18} {s['status']:<12} "
                  f"{','.join(s['reason_codes'])}"
                  f"{('  <- ' + ','.join(s['assigned'])) if s['assigned'] else ''}")

    a, b, b2, c = (r["verdict"] for r in results)
    print(f"\n{'=' * 66}")
    print(f"  A  real + true provenance declared    {a}")
    print(f"  B  real + provenance undeclared       {b}")
    print(f"  B2 real + provenance falsely declared {b2}")
    print(f"  C  repaired + true provenance         {c}")
    print(f"{'=' * 66}")
    print(f"  loop closed   : A == ABSTAIN and C == ANSWERABLE"
          f"    -> {a == 'ABSTAIN' and c == 'ANSWERABLE'}")
    print(f"  fails closed  : B == ABSTAIN on undeclared provenance"
          f"    -> {b == 'ABSTAIN'}")
    print(f"  RESIDUAL      : a false declaration still reaches ANSWERABLE"
          f"    -> {b2 == 'ANSWERABLE'}")
    if b2 == "ANSWERABLE":
        print("                  the kernel never sees the packet, so it checks\n"
              "                  admissibility, not truth. Catching B2 needs a\n"
              "                  compile-layer cross-check of each declared\n"
              "                  source class against the evidence item's own\n"
              "                  provenance terminal. NOT implemented.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"packet": str(args.packet), "question_id": packet["question_id"],
             "shape": f"{shape.shape_id}@{shape.shape_version}",
             "schema_version": shape.schema_version,
             "facts": facts, "arms": results}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

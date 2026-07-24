r"""Labeling workbench (T2 support): automate the WORK of grading, never
the JUDGMENT.

``generate`` assembles one candidate worksheet per fixture question --
anchor-term entity hits, claims matching each term in scope, and their
source documents -- presented in SEEDED-RANDOM order (seed = question
id) so no ranking strategy under T4 evaluation can anchor the historian.
``ingest`` reads the graded worksheet back, keeps grades 1-3, and
mechanically completes everything the rubric assigns to the engineer:
hop locality, low-degree status, and Page/Document provenance
obligations. The historian only ever reads text and types a digit.

Epistemic boundary (see the grading discussion, 2026-07-12): worksheet
assembly and field completion are mechanical; the grade itself is human.
Pilot questions get worksheets too, but their graded output must be kept
out of engineering view per RUBRIC.md.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from . import canonical

WORKSHEET_VERSION = "0.1.0"
_CLAIMS_PER_TERM = 40
_LOCAL_HOPS = 1          # rubric: local = within 1 hop of a confirmed anchor

WORKSHEET_COLUMNS = ("grade", "why", "public_key", "kind", "term",
                     "text", "source_doc", "date")


def _claims_for_term(session_factory, term: str,
                     scope: dict[str, Any]) -> list[dict[str, Any]]:
    """Claims whose normalized sentence contains ``term``, with their
    first Page/Document source. Deterministic ORDER BY claim key."""
    cy = (
        "MATCH (c:Claim) WHERE toLower(c.normalized_sentence) CONTAINS $t "
        "AND coalesce(c.quarantine_status,'') <> 'quarantined' "
        "OPTIONAL MATCH (c)-[:SOURCED_FROM|EVIDENCED_BY]-(p:Paragraph)"
        "<-[:HAS_PARAGRAPH]-(:Section)<-[:HAS_SECTION]-(pg:Page) "
        "OPTIONAL MATCH (d:Document)-[:HAS_PAGE]->(pg) "
        "RETURN c.claim_id AS cid, c.normalized_sentence AS text, "
        "c.claim_date AS date, pg.page_id AS page, d.doc_id AS doc "
        "ORDER BY cid LIMIT $lim")
    out = []
    with session_factory() as s:
        for rec in s.run(cy, t=term.lower(), lim=_CLAIMS_PER_TERM):
            out.append({
                "public_key": f"Claim:{rec['cid']}",
                "kind": "claim", "term": term,
                "text": str(rec["text"] or "")[:240],
                "source_doc": (f"Document:{rec['doc']}" if rec["doc"]
                               else (f"Page:{rec['page']}" if rec["page"]
                                     else "")),
                "date": str(rec["date"] or ""),
            })
    return out


def generate_worksheet(fixture_path: str | Path, resolver,
                       session_factory, out_path: str | Path) -> dict:
    """Build the worksheet CSV for one fixture question. Returns a small
    summary dict. Rows are seeded-random shuffled (seed = question id)."""
    fixture = json.loads(Path(fixture_path).read_text("utf-8"))
    scope = fixture.get("scope", {})
    qid = canonical.question_id(fixture["text"], scope)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for term in fixture.get("anchor_terms", ()):
        for cand in resolver.resolve(term):
            if cand.public_key in seen:
                continue
            seen.add(cand.public_key)
            rows.append({
                "public_key": cand.public_key, "kind": "entity",
                "term": term, "text": cand.display_name,
                "source_doc": "", "date": "",
            })
        for claim in _claims_for_term(session_factory, term, scope):
            if claim["public_key"] in seen:
                continue
            seen.add(claim["public_key"])
            rows.append(claim)

    # seeded shuffle: reproducible, and NOT any strategy's ordering
    random.Random(qid).shuffle(rows)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=WORKSHEET_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({"grade": "", "why": "", **r})
    return {"question_id": qid, "fixture": str(fixture_path),
            "worksheet": str(out_path), "rows": len(rows),
            "entities": sum(1 for r in rows if r["kind"] == "entity"),
            "claims": sum(1 for r in rows if r["kind"] == "claim")}


# -- ingest --------------------------------------------------------------------

def _grade_rows(worksheet_path: str | Path) -> list[dict[str, Any]]:
    with open(worksheet_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    graded = []
    for i, r in enumerate(rows, start=2):       # 1-based + header line
        g = (r.get("grade") or "").strip()
        if not g:
            continue
        if g not in ("0", "1", "2", "3"):
            raise ValueError(f"{worksheet_path}:{i}: grade must be blank "
                             f"or 0-3, got {g!r}")
        if g != "0":
            graded.append({**r, "grade": int(g)})
    return graded


def _mechanical_fields(row: dict[str, Any], anchor_keys: list[str],
                       session_factory) -> dict[str, Any]:
    """Engineer-verified fields per rubric: hop locality (within 1 hop of
    any confirmed-anchor entity over the claim/mention layer), low-degree
    flag (<= snapshot 25th pct is computed at eval time from the bundle;
    here we record raw degree), and the Page/Document obligation."""
    def _resolve_id(s, key: str):
        label, _, value = key.partition(":")
        if value.startswith("neo4j#"):         # explicit fallback key
            rec = s.run("MATCH (n) WHERE id(n) = $nid RETURN id(n) AS nid",
                        nid=int(value[len("neo4j#"):])).single()
        else:
            prop = _KEY_PROPS.get(label)
            if prop is None:
                return None
            rec = s.run(
                f"MATCH (n:`{label}`) WHERE toString(n.`{prop}`) "
                f"= toString($v) RETURN id(n) AS nid", v=value).single()
        return None if rec is None or rec["nid"] is None else int(rec["nid"])

    with session_factory() as s:
        nid = _resolve_id(s, row["public_key"])
        if nid is None:
            raise ValueError(f"worksheet key does not resolve: "
                             f"{row['public_key']}")
        degree = int(s.run(
            "MATCH (n) WHERE id(n) = $nid OPTIONAL MATCH (n)-[r]-() "
            "RETURN count(r) AS deg", nid=nid).single()["deg"])
        anchor_ids = [i for i in (_resolve_id(s, ak) for ak in anchor_keys)
                      if i is not None]
        # a graded node that IS an anchor is local by definition (and
        # Neo4j's shortestPath refuses identical start/end nodes)
        local = nid in anchor_ids
        other = [a for a in anchor_ids if a != nid]
        if not local and other:
            hop = s.run(
                "MATCH (n) WHERE id(n) = $nid "
                "MATCH (a) WHERE id(a) IN $aids "
                "MATCH p = shortestPath((n)-[*..4]-(a)) "
                "RETURN min(length(p)) AS h", nid=nid,
                aids=other).single()
            local = hop and hop["h"] is not None and hop["h"] <= _LOCAL_HOPS
    return {"degree": degree,
            "hop_locality": "local" if local else "nonlocal"}


_KEY_PROPS: dict[str, str] = {}     # populated from resolver config at ingest


def ingest_worksheet(fixture_path: str | Path, worksheet_path: str | Path,
                     session_factory, key_properties: dict[str, str],
                     *, anchor_keys: list[str], labeled_by: str,
                     labeled_at: str) -> dict:
    """Fold grades back into the fixture JSON, auto-completing the
    engineer fields. Overwrites essential_nodes/provenance_obligations;
    grading content itself is copied verbatim from the human."""
    global _KEY_PROPS
    _KEY_PROPS = dict(key_properties)
    fixture = json.loads(Path(fixture_path).read_text("utf-8"))
    graded = _grade_rows(worksheet_path)
    essential, obligations = [], []
    for row in graded:
        mech = _mechanical_fields(row, anchor_keys, session_factory)
        essential.append({
            "public_key": row["public_key"],
            "grade": row["grade"],
            "hop_locality": mech["hop_locality"],
            "degree": mech["degree"],
            "low_degree": None,     # resolved against the eval snapshot
            "why": (row.get("why") or "").strip(),
        })
        if row["grade"] == 3:
            obligations.append({
                "public_key": row["public_key"],
                "expected_source": row.get("source_doc") or None,
                "known_missing": not row.get("source_doc"),
            })
    fixture["essential_nodes"] = essential
    fixture["provenance_obligations"] = obligations
    fixture["labeled_by"] = labeled_by
    fixture["labeled_at"] = labeled_at
    Path(fixture_path).write_text(
        json.dumps(fixture, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    return {"fixture": str(fixture_path), "graded": len(graded),
            "essential_nodes": len(essential),
            "obligations": len(obligations),
            "grade_histogram": dict(sorted(
                __import__("collections").Counter(
                    e["grade"] for e in essential).items()))}

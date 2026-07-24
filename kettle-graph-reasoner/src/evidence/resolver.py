r"""Anchor resolver (plan T1): the explorer's deterministic entity/date
search, generalized behind an injectable read-only session factory.

Sources of truth:
- Query shapes come from ``src/service/explorer_app.py`` ``/api/search``
  (numeric-id exact lookup; lowercase name-contains over domain labels)
  plus a year/date lane; vector search may CONTRIBUTE candidates when
  configured but is never required.
- Sessions must be server-enforced read-only (``default_access_mode=READ``),
  the same discipline as ``explorer_app.read_session``.

Determinism: results are ordered by (match-method priority, display name,
public key) -- never by database return order -- and the full candidate
list hashes to ``resolution_hash``, which is what an ``AnchorConfirmation``
must cite. Confirmation against any other list fails contract validation.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from . import canonical
from .contracts import AnchorConfirmation, AnchorResolution, ResearchQuestion

# match-method priority: lower sorts first
_METHOD_ORDER = {"exact_id": 0, "date": 1, "name_exact": 2,
                 "name_contains": 3, "vector": 4}

# label -> property holding the stable public identifier, verified against
# the live archival schema 2026-07-12 (500-node samples: coverage 100%
# everywhere except Event 499/500 and Concept 0/45). Labels absent here
# fall back to the snapshot-local numeric id WITH an explicit fallback
# flag (plan T3 gate: "public key or explicit fallback warning").
DEFAULT_KEY_PROPERTIES: dict[str, str] = {
    "Entity": "entity_id",
    "Person": "entity_id",
    "Place": "entity_id",
    "Organization": "entity_id",
    "Activity": "entity_id",
    "Parcel": "entity_id",
    "Period": "period_id",
    "Event": "event_id",
    "Observation": "observation_id",
    "Measurement": "measurement_id",
    "Claim": "claim_id",
    "Document": "doc_id",
    "Page": "page_id",
    "Paragraph": "paragraph_id",
    "Mention": "mention_id",
    "Year": "year",
}

_NAME_COALESCE = ("coalesce(n.name, n.title, n.event_type, "
                  "n.observation_type, n.activity_type, n.value, "
                  "n.normalized_sentence, "
                  "toString(n.year), toString(id(n)))")

# session factory: () -> context manager yielding an object with
# .run(query, **params) returning iterable records (neo4j-driver shaped)
SessionFactory = Callable[[], Any]


class AnchorResolver:
    def __init__(self, session_factory: SessionFactory, *,
                 snapshot_epoch: str,
                 domain_labels: Iterable[str],
                 key_properties: Mapping[str, str] | None = None,
                 max_results: int = 25):
        self._session = session_factory
        self._epoch = snapshot_epoch
        self._domain = sorted(domain_labels)
        self._keys = dict(key_properties or DEFAULT_KEY_PROPERTIES)
        self._max = max_results

    # -- public surface ---------------------------------------------------

    def resolve(self, query: str,
                year: int | None = None) -> tuple[AnchorResolution, ...]:
        """Deterministic candidate list for one search term.

        Digit-only queries are treated as snapshot-local ids (exact lane);
        a ``year`` (or a 4-digit query in a plausible year range) adds the
        date lane; everything else is a lowercase name-contains search.
        """
        q = " ".join(query.split())
        rows: list[dict] = []
        if q.isdigit() and not _looks_like_year(q):
            rows += self._by_id(int(q))
        if year is not None:
            rows += self._by_year(year)
        elif _looks_like_year(q):
            rows += self._by_year(int(q))
        if not q.isdigit() and len(q) >= 2:
            rows += self._by_name(q)
        return self._to_resolutions(rows)

    def resolve_for_question(
            self, question: ResearchQuestion,
            terms: Iterable[str]) -> tuple[AnchorResolution, ...]:
        out: list[AnchorResolution] = []
        seen: set[str] = set()
        years: tuple[int | None, ...] = (question.scope.year_start,)
        for term in terms:
            for cand in self.resolve(term, year=None if term else years[0]):
                if cand.public_key not in seen:
                    seen.add(cand.public_key)
                    out.append(cand)
        return _rerank(out)

    @staticmethod
    def resolution_hash(candidates: Iterable[AnchorResolution]) -> str:
        return canonical.content_hash([c.to_dict() for c in candidates])

    def make_confirmation(self, candidates: Iterable[AnchorResolution],
                          confirmed_keys: Iterable[str],
                          rejected_keys: Iterable[str] = (), *,
                          confirmed_by: str,
                          confirmed_at: str) -> AnchorConfirmation:
        """Bind a human confirmation to exactly this candidate list and
        snapshot. Validation happens on the question, where the list is."""
        return AnchorConfirmation(
            confirmed_keys=tuple(confirmed_keys),
            rejected_keys=tuple(rejected_keys),
            resolution_hash=self.resolution_hash(candidates),
            snapshot_epoch=self._epoch,
            confirmed_by=confirmed_by,
            confirmed_at=confirmed_at,
        )

    # -- query lanes -------------------------------------------------------

    def _by_id(self, nid: int) -> list[dict]:
        cy = (f"MATCH (n) WHERE id(n) = $nid "
              f"RETURN id(n) AS id, labels(n) AS labels, "
              f"{_NAME_COALESCE} AS name, properties(n) AS props")
        return self._run(cy, {"nid": nid}, "exact_id",
                         f"id(n) = {nid}")

    def _by_year(self, year: int) -> list[dict]:
        cy = ("MATCH (n:Year) WHERE n.year = $y OR toString(n.year) = $ys "
              "RETURN id(n) AS id, labels(n) AS labels, "
              f"{_NAME_COALESCE} AS name, properties(n) AS props")
        return self._run(cy, {"y": year, "ys": str(year)}, "date",
                         f"Year.year = {year}")

    def _by_name(self, q: str) -> list[dict]:
        # coalesce mirrors explorer_app /api/search plus the claim-layer
        # text properties (Claim content lives in normalized_sentence)
        cy = ("MATCH (n) WHERE any(l IN labels(n) WHERE l IN $dom) AND "
              "toLower(coalesce(n.name,n.title,n.event_type,"
              "n.observation_type,n.activity_type,n.value,n.text,'') + ' ' + "
              "coalesce(n.normalized_sentence,'')) "
              "CONTAINS $q "
              "RETURN id(n) AS id, labels(n) AS labels, "
              f"{_NAME_COALESCE} AS name, properties(n) AS props "
              "LIMIT $lim")
        rows = self._run(cy, {"q": q.lower(), "dom": self._domain,
                              "lim": self._max}, "name_contains",
                         f"name contains {q.lower()!r}")
        ql = q.lower()
        for r in rows:  # exact name match outranks substring match
            if str(r["name"]).lower() == ql:
                r["method"] = "name_exact"
                r["evidence"] = f"name == {ql!r}"
        return rows

    def _run(self, cypher: str, params: dict, method: str,
             evidence: str) -> list[dict]:
        out = []
        with self._session() as s:
            for rec in s.run(cypher, **params):
                out.append({"id": int(rec["id"]),
                            "labels": list(rec["labels"]),
                            "name": str(rec["name"])[:80],
                            "props": dict(rec["props"]),
                            "method": method, "evidence": evidence})
        return out

    # -- shaping -----------------------------------------------------------

    def _public_key(self, labels: list[str], props: Mapping[str, Any],
                    nid: int) -> tuple[str, str, bool]:
        """(public_key, label, is_fallback). Uses the first label with a
        configured key property whose value is present."""
        for label in sorted(labels):
            prop = self._keys.get(label)
            if prop and props.get(prop) is not None:
                return f"{label}:{props[prop]}", label, False
        label = sorted(labels)[0] if labels else "Node"
        return f"{label}:neo4j#{nid}", label, True

    def _to_resolutions(self, rows: list[dict]) -> tuple[AnchorResolution, ...]:
        by_key: dict[str, dict] = {}
        for r in rows:
            key, label, fallback = self._public_key(r["labels"], r["props"],
                                                    r["id"])
            r.update(key=key, label=label, fallback=fallback)
            prev = by_key.get(key)
            if prev is None or (_METHOD_ORDER[r["method"]]
                                < _METHOD_ORDER[prev["method"]]):
                by_key[key] = r
        ordered = sorted(by_key.values(),
                         key=lambda r: (_METHOD_ORDER[r["method"]],
                                        r["name"].lower(), r["key"]))
        return tuple(
            AnchorResolution(public_key=r["key"], label=r["label"],
                             display_name=r["name"], match_method=r["method"],
                             match_evidence=r["evidence"], rank=i + 1,
                             snapshot_epoch=self._epoch, neo4j_id=r["id"],
                             key_is_fallback=r["fallback"])
            for i, r in enumerate(ordered))


def _rerank(cands: list[AnchorResolution]) -> tuple[AnchorResolution, ...]:
    ordered = sorted(cands, key=lambda c: (_METHOD_ORDER[c.match_method],
                                           c.display_name.lower(),
                                           c.public_key))
    return tuple(AnchorResolution(**{**c.__dict__, "rank": i + 1})
                 for i, c in enumerate(ordered))


def _looks_like_year(q: str) -> bool:
    return q.isdigit() and len(q) == 4 and 1500 <= int(q) <= 2100

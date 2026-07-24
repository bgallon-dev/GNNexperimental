r"""Live Neo4j ``ProjectionBackend`` (plan T3): the EvidenceGraphSource.

Sits BESIDE the tensor-parity source (``src.service.neo4j_source``) --
reuses its one-time lifecycle-clean graphcache pull (327k nodes memoized
per process) without modifying ``SubgraphPull`` or any serving code. All
live reads run through sessions opened with ``default_access_mode=READ``
(server-enforced; same discipline as ``explorer_app.read_session``).

Scope boundaries carried from the plan:
- domain candidate space = the ``domain_only`` spec (the validated KGR
  regime); quarantined/lifecycle-excluded nodes are absent from the cache
  member mask, so they can never be admitted (T3 gate).
- ``provenance_closure`` pulls the doc/extraction layer around chosen
  domain nodes via bounded variable-length traversal over the provenance
  + containment rel types; this layer is persisted for replay but never
  encoded (never an ``end_to_end`` encoder input).
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from src.service.neo4j_source import Neo4jSource

from .resolver import DEFAULT_KEY_PROPERTIES

# node properties worth persisting in a packet (embeddings and large
# blobs stay out of research artifacts). Claim-layer text properties are
# included: normalized_sentence/source_sentence ARE the evidence content.
NODE_PROP_ALLOWLIST = (
    "name", "title", "value", "year", "event_type", "observation_type",
    "activity_type", "entity_type", "claim_id", "doc_id", "page_id",
    "paragraph_id", "date", "canonical_name", "alias", "normalized_form",
    "normalized_sentence", "source_sentence", "claim_type",
    "epistemic_status", "extraction_confidence", "review_status", "run_id",
    "claim_date", "date_start", "date_end", "report_year",
)
REL_PROP_ALLOWLIST = ("confidence", "extraction_run", "review_status",
                      "reviewed_by", "source")
# provenance + containment + claim/mention semantic links (verified against
# the live schema 2026-07-12: domain entities attach to the claim layer via
# SUBJECT_OF_CLAIM/TOPIC_OF_CLAIM and to the doc layer via REFERS_TO
# mentions -- without these the closure misses the actual evidence)
PROVENANCE_REL_TYPES = ("HAS_CLAIM", "EVIDENCED_BY", "SOURCED_FROM",
                        "PROCESSED_BY", "HAS_PAGE", "HAS_SECTION",
                        "HAS_PARAGRAPH", "CONTAINS_MENTION",
                        "SUBJECT_OF_CLAIM", "TOPIC_OF_CLAIM", "REFERS_TO")
# depth 4 reaches Entity -> Mention -> Paragraph -> Section -> Page
_CLOSURE_DEPTH = 4
_CLOSURE_ROW_CAP = 20_000
_BATCH = 200


def _read_session(drv):
    try:
        from neo4j import READ_ACCESS
    except ImportError:  # very old driver
        READ_ACCESS = "READ"
    return drv.session(default_access_mode=READ_ACCESS)


class EvidenceGraphSource:
    """Build once per process; every query stays bounded by the profile."""

    def __init__(self, subgraph: str = "domain_only",
                 key_properties: dict[str, str] | None = None) -> None:
        self._src = Neo4jSource(subgraph=subgraph)
        self._src._ensure_cache()          # reuse, don't fork (see module doc)
        cache = self._src._cache
        self._indptr = self._src._indptr
        self._indices = self._src._indices
        self._member = self._src._member
        self._idx2id = self._src._idx2id
        self._t_start = self._src._t_start
        self._id2idx = cache.id2idx
        self._keys = dict(key_properties or DEFAULT_KEY_PROPERTIES)
        self._drv = self._src._drv

        deg = self._indptr[1:] - self._indptr[:-1]
        self._degree = deg
        member_deg = deg[self._member]
        self._low_thresh = int(np.percentile(member_deg, 25)) if \
            member_deg.size else 0
        self._nb_cache: dict[int, tuple[int, ...]] = {}
        self._payload_cache: dict[int, dict[str, Any]] = {}

        # snapshot epoch: label histogram + edge count over the clean cache
        lab_counts = np.bincount(cache.lab_ids,
                                 minlength=len(cache.label_names))
        hist = {str(cache.label_names[i]): int(lab_counts[i])
                for i in range(len(cache.label_names))}
        blob = repr((sorted(hist.items()), int(cache.n),
                     int(len(self._indices)))).encode()
        self.snapshot_epoch = "counts-" + hashlib.sha256(blob).hexdigest()[:16]

    def close(self) -> None:
        self._src.close()

    def __enter__(self) -> "EvidenceGraphSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- ProjectionBackend surface -----------------------------------------

    def anchor_index(self, public_key: str) -> int | None:
        label, _, value = public_key.partition(":")
        if value.startswith("neo4j#"):                 # explicit fallback key
            idx = self._id2idx.get(int(value[len("neo4j#"):]))
            return idx if idx is not None and self._member[idx] else None
        prop = self._keys.get(label)
        if prop is None:
            return None
        cy = (f"MATCH (n:`{label}`) WHERE n.`{prop}` = $v "
              f"OR toString(n.`{prop}`) = toString($v) "
              f"RETURN id(n) AS id LIMIT 2")
        with _read_session(self._drv) as s:
            ids = [int(r["id"]) for r in s.run(cy, v=value)]
        if len(ids) != 1:                              # missing or ambiguous
            return None
        idx = self._id2idx.get(ids[0])
        return idx if idx is not None and self._member[idx] else None

    def neighbors(self, idx: int) -> Sequence[int]:
        nb = self._nb_cache.get(idx)
        if nb is None:
            lo, hi = int(self._indptr[idx]), int(self._indptr[idx + 1])
            nb = tuple(sorted(int(v) for v in self._indices[lo:hi]
                              if self._member[v]))
            self._nb_cache[idx] = nb
        return nb

    def degree(self, idx: int) -> int:
        return int(self._degree[idx])

    def low_degree_threshold(self) -> int:
        return self._low_thresh

    def node_payloads(self, idxs: Sequence[int]) -> list[dict[str, Any]]:
        missing = [i for i in idxs if i not in self._payload_cache]
        for chunk_start in range(0, len(missing), _BATCH):
            chunk = missing[chunk_start:chunk_start + _BATCH]
            ids = [int(self._idx2id[i]) for i in chunk]
            id2idx_local = {int(self._idx2id[i]): i for i in chunk}
            cy = ("MATCH (n) WHERE id(n) IN $ids RETURN id(n) AS id, "
                  "labels(n) AS labels, properties(n) AS props")
            with _read_session(self._drv) as s:
                for rec in s.run(cy, ids=ids):
                    nid = int(rec["id"])
                    self._payload_cache[id2idx_local[nid]] = self._shape(
                        nid, list(rec["labels"]), dict(rec["props"]),
                        id2idx_local[nid])
        return [self._payload_cache[i] for i in idxs]

    def relationships(self, idxs: Sequence[int]) -> list[dict[str, Any]]:
        ids = [int(self._idx2id[i]) for i in idxs]
        id_to_idx = {int(self._idx2id[i]): i for i in idxs}
        cy = ("MATCH (a)-[r]->(b) WHERE id(a) IN $ids AND id(b) IN $ids "
              "RETURN id(r) AS rid, type(r) AS t, id(a) AS a, id(b) AS b, "
              "properties(r) AS props ORDER BY rid")
        out = []
        with _read_session(self._drv) as s:
            for rec in s.run(cy, ids=ids):
                out.append({
                    "rel_id": int(rec["rid"]), "rel_type": str(rec["t"]),
                    "start_idx": id_to_idx[int(rec["a"])],
                    "end_idx": id_to_idx[int(rec["b"])],
                    "properties": _allow(dict(rec["props"]),
                                         REL_PROP_ALLOWLIST)})
        return out

    def provenance_closure(
            self, idxs: Sequence[int]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        domain_ids = [int(self._idx2id[i]) for i in idxs]
        rel_pat = "|".join(PROVENANCE_REL_TYPES)
        warnings: list[str] = []
        rel_rows: dict[int, dict[str, Any]] = {}
        node_ids: set[int] = set()
        cy = (f"MATCH (n) WHERE id(n) IN $ids "
              f"MATCH p=(n)-[:{rel_pat}*1..{_CLOSURE_DEPTH}]-(m) "
              f"UNWIND relationships(p) AS r "
              f"RETURN DISTINCT id(r) AS rid, type(r) AS t, "
              f"id(startNode(r)) AS a, id(endNode(r)) AS b, "
              f"properties(r) AS props LIMIT $cap")
        for start in range(0, len(domain_ids), _BATCH):
            chunk = domain_ids[start:start + _BATCH]
            with _read_session(self._drv) as s:
                rows = list(s.run(cy, ids=chunk, cap=_CLOSURE_ROW_CAP))
            if len(rows) >= _CLOSURE_ROW_CAP:
                warnings.append(
                    f"provenance closure hit the {_CLOSURE_ROW_CAP}-row cap "
                    f"for one batch; closure may be incomplete")
            for rec in rows:
                rid = int(rec["rid"])
                a, b = int(rec["a"]), int(rec["b"])
                rel_rows[rid] = {"rel_id": rid, "rel_type": str(rec["t"]),
                                 "a": a, "b": b,
                                 "properties": _allow(dict(rec["props"]),
                                                      REL_PROP_ALLOWLIST)}
                node_ids.update((a, b))

        # payloads for every closure endpoint (domain + doc-layer);
        # quarantined nodes are EXCLUDED here -- the domain layer is
        # already lifecycle-clean, but the closure runs raw traversal
        # (T3 gate: no quarantined node enters a bundle)
        payload_by_id: dict[int, dict[str, Any]] = {}
        quarantined: set[int] = set()
        all_ids = sorted(node_ids)
        for start in range(0, len(all_ids), _BATCH):
            chunk = all_ids[start:start + _BATCH]
            cy2 = ("MATCH (n) WHERE id(n) IN $ids RETURN id(n) AS id, "
                   "labels(n) AS labels, properties(n) AS props")
            with _read_session(self._drv) as s:
                for rec in s.run(cy2, ids=chunk):
                    nid = int(rec["id"])
                    props = dict(rec["props"])
                    if props.get("quarantine_status") == "quarantined":
                        quarantined.add(nid)
                        continue
                    payload_by_id[nid] = self._shape(
                        nid, list(rec["labels"]), props, None)
        if quarantined:
            warnings.append(f"{len(quarantined)} quarantined node(s) "
                            f"excluded from the provenance closure")

        prels = [{"rel_id": r["rel_id"], "rel_type": r["rel_type"],
                  "start_key": payload_by_id[r["a"]]["public_key"],
                  "end_key": payload_by_id[r["b"]]["public_key"],
                  "properties": r["properties"]}
                 for r in sorted(rel_rows.values(),
                                 key=lambda r: r["rel_id"])
                 if r["a"] not in quarantined and r["b"] not in quarantined]
        return list(payload_by_id.values()), prels, warnings

    def term_claim_recall(
            self, terms: Sequence[str], cap_per_term: int = 40
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        """Bounded lexical claim recall (plan amendment A2, 2026-07-13).

        For each anchor term: non-quarantined Claims whose normalized
        sentence contains the term (ORDER BY claim_id, LIMIT cap), plus
        one-hop entity attachments (SUBJECT_OF_CLAIM/TOPIC_OF_CLAIM) and
        each claim's provenance chain to Page/Document. Every recalled
        node payload is origin-tagged ``term_recall:<term>``. Same
        return shape as ``provenance_closure``."""
        warnings: list[str] = []
        claim_ids: dict[int, str] = {}          # nid -> first matching term
        cy = ("MATCH (c:Claim) WHERE toLower(c.normalized_sentence) "
              "CONTAINS $t AND coalesce(c.quarantine_status,'') <> "
              "'quarantined' RETURN id(c) AS nid ORDER BY c.claim_id "
              "LIMIT $lim")
        for term in terms:
            n_before = len(claim_ids)
            with _read_session(self._drv) as s:
                rows = list(s.run(cy, t=term.lower(), lim=cap_per_term))
            for rec in rows:
                claim_ids.setdefault(int(rec["nid"]), term)
            if len(rows) >= cap_per_term:
                warnings.append(f"term recall for {term!r} hit the "
                                f"{cap_per_term}-claim cap")
            if len(claim_ids) == n_before and not rows:
                warnings.append(f"term recall for {term!r}: 0 claims")

        rel_rows: dict[int, dict[str, Any]] = {}
        node_ids: set[int] = set(claim_ids)
        rel_pat = ("SUBJECT_OF_CLAIM|TOPIC_OF_CLAIM|SOURCED_FROM|"
                   "EVIDENCED_BY|HAS_CLAIM|HAS_PARAGRAPH|HAS_SECTION|"
                   "HAS_PAGE")
        cy2 = (f"MATCH (c) WHERE id(c) IN $ids "
               f"MATCH p=(c)-[:{rel_pat}*1..{_CLOSURE_DEPTH}]-(m) "
               f"UNWIND relationships(p) AS r "
               f"RETURN DISTINCT id(r) AS rid, type(r) AS t, "
               f"id(startNode(r)) AS a, id(endNode(r)) AS b, "
               f"properties(r) AS props LIMIT $cap")
        ids = sorted(claim_ids)
        for start in range(0, len(ids), _BATCH):
            with _read_session(self._drv) as s:
                rows = list(s.run(cy2, ids=ids[start:start + _BATCH],
                                  cap=_CLOSURE_ROW_CAP))
            if len(rows) >= _CLOSURE_ROW_CAP:
                warnings.append(f"term-recall closure hit the "
                                f"{_CLOSURE_ROW_CAP}-row cap for one batch")
            for rec in rows:
                rid = int(rec["rid"])
                a, b = int(rec["a"]), int(rec["b"])
                rel_rows[rid] = {"rel_id": rid, "rel_type": str(rec["t"]),
                                 "a": a, "b": b,
                                 "properties": _allow(dict(rec["props"]),
                                                      REL_PROP_ALLOWLIST)}
                node_ids.update((a, b))

        payload_by_id: dict[int, dict[str, Any]] = {}
        quarantined: set[int] = set()
        all_ids = sorted(node_ids)
        for start in range(0, len(all_ids), _BATCH):
            cy3 = ("MATCH (n) WHERE id(n) IN $ids RETURN id(n) AS id, "
                   "labels(n) AS labels, properties(n) AS props")
            with _read_session(self._drv) as s:
                for rec in s.run(cy3, ids=all_ids[start:start + _BATCH]):
                    nid = int(rec["id"])
                    props = dict(rec["props"])
                    if props.get("quarantine_status") == "quarantined":
                        quarantined.add(nid)
                        continue
                    shaped = self._shape(nid, list(rec["labels"]), props,
                                         None)
                    term = claim_ids.get(nid)
                    shaped["origin"] = (f"term_recall:{term}" if term
                                        else "term_recall:chain")
                    payload_by_id[nid] = shaped
        if quarantined:
            warnings.append(f"{len(quarantined)} quarantined node(s) "
                            f"excluded from term recall")

        prels = [{"rel_id": r["rel_id"], "rel_type": r["rel_type"],
                  "start_key": payload_by_id[r["a"]]["public_key"],
                  "end_key": payload_by_id[r["b"]]["public_key"],
                  "properties": r["properties"]}
                 for r in sorted(rel_rows.values(),
                                 key=lambda r: r["rel_id"])
                 if r["a"] not in quarantined and r["b"] not in quarantined]
        return list(payload_by_id.values()), prels, warnings

    # -- shaping -------------------------------------------------------------

    def _shape(self, nid: int, labels: list[str], props: dict[str, Any],
               idx: int | None) -> dict[str, Any]:
        key, label, fallback = self._public_key(labels, props, nid)
        raw_date = None
        # preference: explicit dates, then claim/document dates, then years
        for date_prop in ("date", "claim_date", "date_start", "year",
                          "report_year"):
            if props.get(date_prop) is not None:
                raw_date = str(props[date_prop])
                break
        normalized = None
        if idx is not None and raw_date is not None:
            normalized = float(self._t_start[idx])
        return {
            "public_key": key, "key_is_fallback": fallback, "label": label,
            "neo4j_id": nid, "properties": _allow(props, NODE_PROP_ALLOWLIST),
            "raw_date": raw_date, "normalized_date": normalized,
        }

    def _public_key(self, labels: list[str], props: dict[str, Any],
                    nid: int) -> tuple[str, str, bool]:
        for label in sorted(labels):
            prop = self._keys.get(label)
            if prop and props.get(prop) is not None:
                return f"{label}:{props[prop]}", label, False
        label = sorted(labels)[0] if labels else "Node"
        return f"{label}:neo4j#{nid}", label, True


def _allow(props: dict[str, Any], allowlist: tuple[str, ...]) -> dict[str, Any]:
    return {k: props[k] for k in allowlist
            if k in props and isinstance(props[k], (str, int, float, bool))}

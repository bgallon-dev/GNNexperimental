r"""Deterministic evidence-packet compiler (plan T5, v0.1 MVP).

Consumes a confirmed ``ResearchQuestion``, a projected ``CandidateBundle``
(with provenance closure attached) and a ranking; emits the packet dict
(canonically serializable) plus a deterministic Markdown rendering.

v0.1 selection rules (plan T5):
- Core evidence requires a complete provenance path to a Page or Document
  inside the bundle, plus explicit asset status.
- Eligible evidence is selected round-robin across confirmed anchors
  before filling by rank.
- Missing provenance becomes uncertainty/frontier content, never core.
- Anchors + mandatory path intermediaries must fit the unique-node budget
  or compilation FAILS (obligations are never truncated); everything else
  is budget-trimmed with the trim REPORTED in ``overflow``.
- ``corroboration_groups``/``contradiction_sets`` are always present;
  without reviewed sidecars they are empty with
  ``coverage_status: not_annotated`` (empty sidecars are a valid
  acceptance case; T6 fills them from curated records only).

Path ordering is the exact typed-path ordering (the plan's expected v1
ship): (relation-priority sum, length, provenance-completeness, canonical
path id). ``default_v1`` assigns every relation priority 0, so today this
reduces to (length, path id) -- the hook exists for family profiles.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Mapping, Sequence

from . import canonical
from .contracts import (
    AnnotationRecord,
    CandidateBundle,
    ContractError,
    RankedCandidate,
    ResearchQuestion,
)

# 0.1.1: attachment-centric core evidence
# 0.1.2: claims-before-mentions attachment priority
# 0.2.0: amendments A1/A2 (2026-07-13) -- term-recall lane in core
#        selection; attachment_ranking metric for T4
# 0.2.1: per-item `date` (EPP 0.1.1) -- evidence node's own date, else
#        the first dated node on its provenance chain (source document)
# 0.2.2: Page terminals extend to their parent Document (issue-level
#        citation; also where the dates live)
# 0.2.3: datable core candidates outside the question's year scope are
#        excluded (counted in overflow); undated stays eligible
# 0.2.4: keyed exclusion records with reasons (EPP 0.1.2 / S11) -- counts
#        alone cannot tell downstream change analysis WHICH items left
# 0.2.5: two defects found by the 39-bundle audit (2026-07-16), which
#        measured 9.0% of BFS core evidence containing ANY of its own
#        question's anchor terms:
#        (a) core selection never consulted the strategy -- it took
#            gather_attachment_items order (via-priority, public key) while
#            attachment_ranking, the per-item scorer, was wired ONLY to the
#            T4 metric. The metric judged an order the compiler never
#            emitted. Now both call order_attachment_items.
#        (b) an out-of-scope item consumed its lane's round-robin turn, so
#            a lane headed by an anachronistic ingest run forfeited its
#            share of core. Scope is a filter, not a competitor.
#        bfs is byte-identical across this bump; lexical/kgr change.
# 0.2.6: pointer nodes are no longer seated as evidence. The same audit's
#        other finding: ~30% of BFS core evidence rendered as the empty
#        "(mention pointer; see provenance)", and EVERY stub was a Mention
#        (bfs core was {Claim: 328, Mention: 140}). Mention carries only
#        run_id and normalized_form -- a single token ("spokane"), not a
#        statement -- so the stub text was honest and the defect was seating
#        the node at all. Eligibility now requires a renderable statement
#        (_TEXT_PROPS), keyed on content rather than label.
#        Measured over the 39 bundles: stubs 29.9% -> 0.0% (bfs) and
#        12.0% -> 0.0% (lexical); claims seated 328 -> 444; anchor-term rate
#        bfs 25.0% -> 37.6%, lexical 35.0% -> 40.1% (instrument:
#        scripts/measure_selector_null.py, generous substring rule -- read
#        the deltas, not the absolutes).
#        Two bundles attach nothing but pointers and now seat NOTHING rather
#        than 12 stubs. To keep that from becoming a silent empty packet,
#        core selection now emits thinness warnings + overflow counters
#        (core_unfilled_slots, claims_never_seated) naming the unseated
#        claims. Those two bundles are the standing evidence that anchor
#        attachment + term recall do not reach every claim in a bundle.
#        ALL strategies change bytes across this bump.
COMPILER_VERSION = "0.2.6"
PACKET_SCHEMA_VERSION = "0.1.2"

_ISO_DATE_RE = None


def _iso_date(raw) -> str | None:
    """Normalize a raw date string to partial ISO-8601 (YYYY[-MM[-DD]]),
    or None when nothing date-like is present."""
    global _ISO_DATE_RE
    if _ISO_DATE_RE is None:
        import re
        _ISO_DATE_RE = re.compile(
            r"\b(1[5-9]\d\d|20\d\d)(?:-(\d{2})(?:-(\d{2}))?)?\b")
    if raw is None:
        return None
    m = _ISO_DATE_RE.search(str(raw))
    if not m:
        return None
    out = m.group(1)
    if m.group(2):
        out += f"-{m.group(2)}"
        if m.group(3):
            out += f"-{m.group(3)}"
    return out
_PROVENANCE_TERMINALS = ("Page", "Document")
_PROV_SEARCH_DEPTH = 6
# The properties a node can carry a renderable statement in. A node with none
# of them cannot be evidence: it is a pointer to where something was said, not
# the saying of it. `Mention` is the case that matters on this corpus -- its
# only properties are `run_id` and `normalized_form`, and normalized_form is a
# single normalized token ("spokane", or the OCR variant "snokane"), not a
# statement. Seating one spends a core_evidence slot on the honest but empty
# "(mention pointer; see provenance)". Keyed on content, not on the label, so a
# new pointer type needs no change here.
_TEXT_PROPS = ("name", "title", "normalized_sentence")
# attachment-kind priority: statements about a node beat pointers to it
_VIA_PRIORITY = {"SUBJECT_OF_CLAIM": 0, "TOPIC_OF_CLAIM": 1,
                 "HAS_CLAIM": 1, "EVIDENCED_BY": 2, "SOURCED_FROM": 2,
                 "REFERS_TO": 4, "CONTAINS_MENTION": 4}


def _node_text(n: ProjectedNode) -> str | None:
    """The node's renderable statement, or None if it carries none.

    Single source of truth for both evidence eligibility and display: if these
    ever disagree, contentless nodes get seated in evidence slots again."""
    for prop in _TEXT_PROPS:
        value = n.properties.get(prop)
        if value:
            return str(value)
    return None


class PacketObligationError(ContractError):
    """Forced packet obligations exceed the hard unique-node budget."""


# -- provenance path search ---------------------------------------------------

def _edge_graph(rels, keys):
    """(adjacency, rel-type-by-edge) over one relationship set. Neighbor
    sets are sorted at use for determinism."""
    adj: dict[str, set[str]] = {k: set() for k in keys}
    rel_type: dict[tuple[str, str], str] = {}
    for r in rels:
        adj[r.start_key].add(r.end_key)
        adj[r.end_key].add(r.start_key)
        for e in ((r.start_key, r.end_key), (r.end_key, r.start_key)):
            rel_type.setdefault(e, r.rel_type)
    return adj, rel_type


def _labels(bundle: CandidateBundle) -> dict[str, str]:
    return {n.public_key: n.label for n in (*bundle.nodes,
                                            *bundle.provenance_nodes)}


def _provenance_path(start: str, adj, label, rel_type
                     ) -> tuple[list[str], list[str]] | None:
    """Shortest path from ``start`` to any Page/Document node (BFS over
    the full bundle graph, sorted neighbors). Returns (keys, rel_types)
    or None when the snapshot has no complete chain."""
    if start not in adj:
        return None
    prev: dict[str, str] = {start: ""}
    q = deque([(start, 0)])
    while q:
        u, d = q.popleft()
        if label.get(u) in _PROVENANCE_TERMINALS and u != start:
            keys = [u]
            while prev[keys[-1]]:
                keys.append(prev[keys[-1]])
            keys.reverse()
            # a Page terminal extends to its parent Document when the
            # bundle has it: the issue-level citation (and its date)
            # completes the chain (compiler 0.2.2)
            if label.get(keys[-1]) == "Page":
                for nb in sorted(adj[keys[-1]]):
                    if label.get(nb) == "Document" and nb not in keys:
                        keys.append(nb)
                        break
            rels = [rel_type[(keys[i], keys[i + 1])]
                    for i in range(len(keys) - 1)]
            return keys, rels
        if d >= _PROV_SEARCH_DEPTH:
            continue
        for v in sorted(adj[u]):
            if v not in prev:
                prev[v] = u
                q.append((v, d + 1))
    return None


# -- exact typed-path ordering -------------------------------------------------

def order_explanatory_paths(bundle: CandidateBundle,
                            relation_priority: Mapping[str, int] | None = None
                            ) -> list[dict[str, Any]]:
    _, rel_type = _edge_graph(bundle.relationships, _labels(bundle))
    prio = dict(relation_priority or {})
    rows = []
    for p in bundle.paths:
        rels = [rel_type.get((p[i], p[i + 1]), "?")
                for i in range(len(p) - 1)]
        path_id = canonical.content_hash(list(p))[:12]
        rows.append({
            "path_id": path_id,
            "keys": list(p),
            "rel_types": rels,
            "length": len(p) - 1,
            "_sort": (sum(prio.get(t, 0) for t in rels), len(p) - 1,
                      path_id),
        })
    rows.sort(key=lambda r: r["_sort"])
    for r in rows:
        r.pop("_sort")
    return rows


# -- attachment items (shared by the compiler and the T4 metric) ----------------

def gather_attachment_items(bundle: CandidateBundle,
                            ranking: Sequence[RankedCandidate]
                            ) -> tuple[list[dict[str, Any]],
                                       list[RankedCandidate]]:
    """All provenance-complete evidence items, in deterministic
    discovery order: ranked domain nodes' attachments first (claims
    before mentions), then the term-recall claim pool (amendment A2).
    Each item: evidence_key, about, prov_keys/rels, via, rc, lane.
    Second return: ranked non-anchor domain nodes with no attachment."""
    label = _labels(bundle)
    prov_adj, prov_rel_type = _edge_graph(bundle.provenance_relationships,
                                          label)
    by_key = {n.public_key: n for n in bundle.nodes}
    node_by_key = {n.public_key: n
                   for n in (*bundle.nodes, *bundle.provenance_nodes)}

    def _attachments(domain_key: str) -> list[dict[str, Any]]:
        out = []
        for nb in sorted(prov_adj.get(domain_key, ())):
            if nb in by_key:               # domain-domain edge, not evidence
                continue
            node = node_by_key.get(nb)
            if node is not None and _node_text(node) is None:
                # A pointer, not evidence. Before this the anchor lanes seated
                # these and ~30% of bfs core_evidence rendered as an empty
                # stub; the slot now falls through to a statement instead.
                continue
            if label.get(nb) in _PROVENANCE_TERMINALS:
                keys, rels = [nb], []
            else:
                found = _provenance_path(nb, prov_adj, label, prov_rel_type)
                if found is None:
                    continue
                keys, rels = found
            out.append({"evidence_key": nb, "about": domain_key,
                        "prov_keys": keys, "prov_rels": rels,
                        "via": prov_rel_type[(domain_key, nb)]})
        # evidence-bearing statements outrank pointers
        out.sort(key=lambda a: (_VIA_PRIORITY.get(a["via"], 5),
                                a["evidence_key"]))
        return out

    eligible: list[dict[str, Any]] = []
    missing_prov: list[RankedCandidate] = []
    claimed: set[str] = set()
    for rc in ranking:
        atts = [a for a in _attachments(rc.public_key)
                if a["evidence_key"] not in claimed]
        if not atts:
            if rc.public_key not in bundle.anchors:
                missing_prov.append(rc)
            continue
        for a in atts:
            claimed.add(a["evidence_key"])
            a["rc"] = rc
            a["lane"] = (rc.contributing_anchors[0]
                         if rc.contributing_anchors else bundle.anchors[0])
            eligible.append(a)

    # term-recall claims (origin-tagged; amendment A2): about = an
    # attached entity when one is in the bundle, else the claim itself
    known = set(label)
    synthetic_rc = RankedCandidate(
        public_key="__term_recall__", strategy=(ranking[0].strategy
                                                if ranking else "bfs"),
        rank=len(ranking) + 1, micro_score=0, hop=99,
        rationale="term-recall pool (amendment A2)")
    for n in sorted(bundle.provenance_nodes, key=lambda n: n.public_key):
        if not n.origin.startswith("term_recall:") or \
                n.origin == "term_recall:chain" or n.label != "Claim":
            continue
        if n.public_key in claimed:
            continue
        found = _provenance_path(n.public_key, prov_adj, label,
                                 prov_rel_type)
        if found is None:
            continue
        about = n.public_key
        for nb in sorted(prov_adj.get(n.public_key, ())):
            if prov_rel_type[(n.public_key, nb)] in ("SUBJECT_OF_CLAIM",
                                                     "TOPIC_OF_CLAIM") \
                    and nb in known:
                about = nb
                break
        keys, rels = found
        claimed.add(n.public_key)
        eligible.append({
            "evidence_key": n.public_key, "about": about,
            "prov_keys": keys, "prov_rels": rels,
            "via": f"TERM_RECALL[{n.origin.split(':', 1)[1]}]",
            "rc": synthetic_rc, "lane": "term_recall"})
    return eligible, missing_prov


def attachment_ranking(bundle: CandidateBundle,
                       ranking: Sequence[RankedCandidate],
                       question: ResearchQuestion | None = None
                       ) -> list[dict[str, Any]]:
    """T4 metric amendment A1 (adopted 2026-07-13): the ordered evidence
    list a strategy produces, judged instead of domain-node rankings
    (which are anchor-only on this corpus; all deltas were exactly 0.0).

    Per-strategy ordering lever:
    - ``bfs``     -- unbudgeted round-robin selection order (the
                     compiler's own deterministic order)
    - ``lexical`` -- question-token overlap of the evidence node's text,
                     desc; then via-priority, key
    - ``kgr``     -- the frozen encoder's micro-score of the item's
                     ABOUT node, desc (the encoder cannot embed claims
                     inside its validated regime -- its reach is via the
                     entities evidence attaches to; reported honestly);
                     then via-priority, key
    """
    items, _ = gather_attachment_items(bundle, ranking)
    return order_attachment_items(items, bundle, ranking, question)


def order_attachment_items(items: list[dict[str, Any]],
                           bundle: CandidateBundle,
                           ranking: Sequence[RankedCandidate],
                           question: ResearchQuestion | None = None
                           ) -> list[dict[str, Any]]:
    """Per-strategy ordering over already-gathered attachment items.

    Shared by ``attachment_ranking`` (the T4 metric) and ``compile_packet``
    (core selection) so the metric judges the order the compiler actually
    emits. Before 0.2.5 these were two different functions: the metric
    scored per item, the compiler took ``gather_attachment_items`` order
    (via-priority, then public key) and never consulted the strategy at
    all -- so a strategy could win T4 while changing nothing it shipped.
    """
    strategy = ranking[0].strategy if ranking else "bfs"

    def _tiebreak(it):
        return (_VIA_PRIORITY.get(it["via"].split("[")[0], 5),
                it["evidence_key"])

    if strategy == "lexical" and question is not None:
        from .ranking import _tokens

        q_tokens = _tokens(question.text)
        text_of = {n.public_key: " ".join(
            str(v) for v in n.properties.values())
            for n in (*bundle.nodes, *bundle.provenance_nodes)}

        def _lex(it):
            overlap = len(q_tokens & _tokens(
                text_of.get(it["evidence_key"], "")))
            return canonical.to_micro_score(
                overlap / max(1, len(q_tokens)))

        items.sort(key=lambda it: (-_lex(it), *_tiebreak(it)))
    elif strategy == "kgr":
        micro = {rc.public_key: rc.micro_score for rc in ranking}
        floor = min(micro.values()) - 1 if micro else 0
        items.sort(key=lambda it: (-micro.get(it["about"], floor),
                                   *_tiebreak(it)))
    else:  # bfs: unbudgeted round-robin, identical to core selection
        lanes: dict[str, deque] = {a: deque() for a in bundle.anchors}
        lanes["term_recall"] = deque()
        for it in items:
            lanes.setdefault(it["lane"], deque()).append(it)
        order, out = list(lanes), []
        while any(lanes.values()):
            for a in order:
                if lanes[a]:
                    out.append(lanes[a].popleft())
        items = out
    return items


def attachment_lane_rankings(bundle: CandidateBundle,
                             items: Sequence[Mapping[str, Any]]
                             ) -> dict[str, list[str]]:
    """The two observable lanes over an attachment ranking: full order,
    and the anchor-nonadjacent suborder (about not a confirmed anchor)."""
    anchors = set(bundle.anchors)
    return {
        "core": [it["evidence_key"] for it in items],
        "nonlocal_discovery": [it["evidence_key"] for it in items
                               if it["about"] not in anchors],
    }


# -- the compiler --------------------------------------------------------------

def compile_packet(question: ResearchQuestion,
                   bundle: CandidateBundle,
                   ranking: Sequence[RankedCandidate],
                   annotations: Iterable[AnnotationRecord] = (),
                   compiler_version: str = COMPILER_VERSION
                   ) -> dict[str, Any]:
    question.validate(require_confirmation=True)
    bundle.validate()
    if bundle.question_id != question.question_id:
        raise ContractError("bundle was projected for a different question")
    budget = question.budget
    label = _labels(bundle)
    by_key = {n.public_key: n for n in bundle.nodes}
    prov_by_key = {n.public_key: n for n in bundle.provenance_nodes}

    # forced obligations: anchors + anchor-pair path intermediaries
    forced: list[str] = list(bundle.anchors)
    for p in bundle.paths:
        for k in p[1:-1]:
            if k not in forced:
                forced.append(k)
    if len(forced) > budget.max_unique_nodes:
        raise PacketObligationError(
            f"{len(forced)} forced nodes (anchors + path intermediaries) "
            f"exceed max_unique_nodes={budget.max_unique_nodes}; "
            f"compilation fails rather than truncating obligations")

    used: set[str] = set(forced)

    def fits(extra: Iterable[str]) -> bool:
        return len(used | set(extra)) <= budget.max_unique_nodes

    # -- core evidence: provenance-layer ATTACHMENTS of ranked domain
    #    nodes plus the term-recall claim pool (amendment A1/A2,
    #    2026-07-13). The archival schema's evidence lives in the claim
    #    layer; a domain node is the SUBJECT of evidence, the attachment
    #    IS the evidence item. Anchors participate.
    eligible, missing_prov = gather_attachment_items(bundle, ranking)
    # 0.2.5: order by the strategy BEFORE laning. The round-robin below
    # preserves relative order within a lane, so this is what lets a
    # strategy express a preference over evidence at all; the lanes still
    # enforce anchor diversity. For `bfs` this is its own round-robin
    # order, so bfs packets are unchanged (byte-identical).
    eligible = order_attachment_items(eligible, bundle, ranking, question)

    raw_date_of = {n.public_key: n.raw_date
                   for n in (*bundle.nodes, *bundle.provenance_nodes)}

    def _item_date(evidence_key: str, prov_keys) -> str | None:
        """The item's own date, else the first dated node along its
        provenance chain (typically the source document/issue)."""
        for key in (evidence_key, *prov_keys):
            d = _iso_date(raw_date_of.get(key))
            if d:
                return d
        return None

    def _out_of_scope(e) -> bool:
        """Datable evidence outside the question's year scope never
        enters core (compiler 0.2.3; the 1987-ordinance finding).
        Undated evidence is NOT excluded -- absence of a date is
        uncertainty, not proof of anachronism."""
        sc = question.scope
        if sc.year_start is None and sc.year_end is None:
            return False
        d = _item_date(e["evidence_key"], e["prov_keys"])
        if not d:
            return False
        year = int(d[:4])
        return ((sc.year_start is not None and year < sc.year_start)
                or (sc.year_end is not None and year > sc.year_end))

    lanes: dict[str, deque] = {a: deque() for a in bundle.anchors}
    lanes["term_recall"] = deque()
    for e in eligible:                       # ranking order preserved
        lanes.setdefault(e["lane"], deque()).append(e)
    core: list[dict[str, Any]] = []
    order = list(lanes)
    exclusion_records: list[dict[str, Any]] = []
    while len(core) < budget.core_evidence and any(lanes.values()):
        progressed = False
        for a in order:
            if len(core) >= budget.core_evidence:
                break
            # 0.2.5: scope is a FILTER, not a competitor. An anachronistic
            # item never had a claim on the slot, so it must not consume the
            # lane's turn -- keep drawing until the lane yields a datable-in-
            # scope (or undated) item. Before this, a lane whose head was a
            # run of out-of-scope claims silently forfeited its whole share
            # of core: the term-recall lane is ordered by claim public key,
            # which begins with the ingest run id, so one anachronistic run
            # sorting first could starve every in-scope claim behind it.
            # `budget_displaced` stays turn-consuming -- that item DID
            # compete and lost to the unique-node budget.
            while lanes[a]:
                e = lanes[a].popleft()
                progressed = True
                if _out_of_scope(e):
                    d = _item_date(e["evidence_key"], e["prov_keys"])
                    exclusion_records.append({
                        "public_key": e["evidence_key"],
                        "reason": "out_of_scope",
                        "detail": f"dated {d}, outside question scope "
                                  f"{question.scope.year_start}-"
                                  f"{question.scope.year_end}",
                        **({"date": d} if d else {})})
                    continue
                new_nodes = [e["about"], e["evidence_key"], *e["prov_keys"]]
                if not fits(new_nodes):
                    exclusion_records.append({
                        "public_key": e["evidence_key"],
                        "reason": "budget_displaced",
                        "detail": "eligible but did not fit the unique-node "
                                  "budget"})
                    break
                used.update(new_nodes)
                core.append(e)
                break
        if not progressed:
            break
    exclusion_records.sort(key=lambda x: (x["reason"], x["public_key"]))
    out_of_scope = sum(1 for x in exclusion_records
                       if x["reason"] == "out_of_scope")
    core_trimmed = sum(1 for x in exclusion_records
                       if x["reason"] == "budget_displaced")

    # A thin packet must say why it is thin. Dropping pointer nodes from
    # evidence eligibility (see _TEXT_PROPS) is right, but it must not turn a
    # visibly-empty packet into a silently-empty one: two bundles on this
    # corpus attach nothing but Mentions, and would otherwise report no
    # evidence and no reason. Absence is never silently converted.
    seated = {e["evidence_key"] for e in core}
    unseated_claims = sorted(
        n.public_key for n in bundle.provenance_nodes
        if n.label == "Claim" and n.public_key not in seated)
    unfilled = max(0, budget.core_evidence - len(core))
    thinness_warnings: list[str] = []
    if unfilled and unseated_claims:
        thinness_warnings.append(
            f"{unfilled} of {budget.core_evidence} core slots went unfilled "
            f"while {len(unseated_claims)} claim(s) in the bundle were never "
            f"seated: they neither attach to a ranked anchor by a "
            f"statement-bearing edge nor appear in the term-recall pool. The "
            f"anchors' remaining attachments were pointers with no renderable "
            f"statement and are not evidence.")
    if not core:
        thinness_warnings.append(
            "no core evidence was seated; this packet asserts nothing and "
            "must not be read as an absence of evidence in the source")

    def _display(key: str) -> str:
        n = by_key.get(key) or prov_by_key.get(key)
        if n is None:
            return key
        text = _node_text(n)
        # Retained as a defensive floor: evidence eligibility now rejects
        # textless nodes (_node_text is None), so a Mention should never reach
        # a core slot. If one ever does, say so honestly rather than render a
        # bare surface form as if it were a statement.
        if text is None and n.label == "Mention":
            return "(mention pointer; see provenance)"
        return str(text or key)[:160]

    core_out = [{
        "evidence_id": f"E{i+1:02d}",
        "public_key": e["evidence_key"],
        "label": label[e["evidence_key"]],
        "display": _display(e["evidence_key"]),
        **({"date": d} if (d := _item_date(e["evidence_key"],
                                           e["prov_keys"])) else {}),
        "about": e["about"],
        "about_display": _display(e["about"]),
        "attached_via": e["via"],
        "strategy": e["rc"].strategy,
        "rank": e["rc"].rank,
        "display_score": canonical.format_micro_score(e["rc"].micro_score),
        "hop": e["rc"].hop,
        "rationale": f"{e['rc'].rationale}; attached to "
                     f"{e['about']} via {e['via']}",
        "provenance_path": e["prov_keys"],
        "provenance_rel_types": e["prov_rels"],
        "source": e["prov_keys"][-1],
        "asset_status": "linked",
    } for i, e in enumerate(core)]

    # -- explanatory paths --
    paths_out = order_explanatory_paths(bundle)[:budget.explanatory_paths]

    # -- context: remaining ranked nodes; reserved nonlocal slots --
    core_about = {c["about"] for c in core_out}
    remaining = [rc for rc in ranking
                 if rc.public_key not in core_about
                 and rc.public_key not in bundle.anchors]
    nonlocal_pool = deque(rc for rc in remaining if rc.hop >= 2)
    local_pool = deque(rc for rc in remaining if rc.hop < 2)
    context: list[RankedCandidate] = []
    reserved = min(budget.context_reserved_nonlocal, len(nonlocal_pool))
    context_trimmed = 0
    while len(context) < budget.context_items and (nonlocal_pool or
                                                   local_pool):
        want_nonlocal = (len(context) < reserved)
        pool = (nonlocal_pool if (want_nonlocal and nonlocal_pool)
                else (local_pool or nonlocal_pool))
        rc = pool.popleft()
        if not fits([rc.public_key]):
            context_trimmed += 1
            continue
        used.add(rc.public_key)
        context.append(rc)
    context_out = [{
        "public_key": rc.public_key, "label": label[rc.public_key],
        "display": _display(rc.public_key), "rank": rc.rank,
        "hop": rc.hop,
        **({"date": d} if (d := _iso_date(
            raw_date_of.get(rc.public_key))) else {}),
        "lane": "nonlocal_discovery" if rc.hop >= 2 else "core",
    } for rc in context]

    # -- uncertainties + research frontier --
    uncertainties = [{
        "public_key": rc.public_key,
        "label": label[rc.public_key],
        "display": _display(rc.public_key),
        "kind": "missing_provenance",
        "detail": "no provenance-complete evidence attachment in this "
                  "snapshot",
    } for rc in missing_prov[:budget.frontier_entries]]
    frontier = [{
        "public_key": rc.public_key,
        "display": _display(rc.public_key),
        "kind": "missing_provenance" if any(
            u["public_key"] == rc.public_key for u in uncertainties)
        else "unexplored_nonlocal",
        "rank": rc.rank,
    } for rc in (*missing_prov, *nonlocal_pool)][:budget.frontier_entries]

    packet = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "compiler_version": compiler_version,
        "question_id": question.question_id,
        "question": question.text,
        "family": question.family,
        "snapshot_epoch": bundle.snapshot_epoch,
        "candidate_hash": bundle.dependency_hash,
        "anchors": list(bundle.anchors),
        "strategy": ranking[0].strategy if ranking else "none",
        "core_evidence": core_out,
        "explanatory_paths": paths_out,
        "context": context_out,
        "uncertainties": uncertainties,
        "research_frontier": frontier,
        "corroboration_groups": [],
        "contradiction_sets": [],
        "coverage_status": ("not_annotated" if not list(annotations)
                            else "annotated"),
        "budgets": budget.to_dict(),
        "unique_nodes_used": len(used),
        "exclusions": exclusion_records,
        "overflow": {**bundle.overflow,
                     "core_trimmed_for_node_budget": core_trimmed,
                     "core_excluded_out_of_scope": out_of_scope,
                     "context_trimmed_for_node_budget": context_trimmed,
                     "core_unfilled_slots": unfilled,
                     "claims_never_seated": len(unseated_claims)},
        "warnings": sorted({*bundle.warnings, *thinness_warnings}),
    }
    validate_packet(packet, bundle)
    return packet


# -- validation ----------------------------------------------------------------

def validate_packet(packet: Mapping[str, Any],
                    bundle: CandidateBundle) -> None:
    known = {n.public_key for n in (*bundle.nodes, *bundle.provenance_nodes)}
    budget = packet["budgets"]

    def _req(cond: bool, msg: str) -> None:
        if not cond:
            raise ContractError(f"packet invalid: {msg}")

    ids = [c["evidence_id"] for c in packet["core_evidence"]]
    _req(len(set(ids)) == len(ids), "duplicate evidence ids")
    _req(len(ids) <= budget["core_evidence"], "core budget exceeded")
    _req(len(packet["explanatory_paths"]) <= budget["explanatory_paths"],
         "path budget exceeded")
    _req(len(packet["context"]) <= budget["context_items"],
         "context budget exceeded")
    _req(len(packet["research_frontier"]) <= budget["frontier_entries"],
         "frontier budget exceeded")
    _req(packet["unique_nodes_used"] <= budget["max_unique_nodes"],
         "unique-node budget exceeded")
    for c in packet["core_evidence"]:
        _req(c["public_key"] in known, f"unknown key {c['public_key']}")
        _req(c["about"] in known, f"unknown about-key {c['about']}")
        _req(c["asset_status"] == "linked", "core item without asset status")
        _req(bool(c["provenance_path"]), "core item without provenance")
        for k in c["provenance_path"]:
            _req(k in known, f"provenance key {k} not in bundle")
    for section in ("context", "uncertainties", "research_frontier"):
        for item in packet[section]:
            _req(item["public_key"] in known,
                 f"{section} key {item['public_key']} not in bundle")
    for p in packet["explanatory_paths"]:
        for k in p["keys"]:
            _req(k in known, f"path key {k} not in bundle")
    core_pk = {c["public_key"] for c in packet["core_evidence"]}
    for x in packet.get("exclusions", []):
        _req(x["public_key"] in known,
             f"exclusion key {x['public_key']} not in bundle")
        _req(x["public_key"] not in core_pk,
             f"exclusion key {x['public_key']} is also core (S11)")
    _req(packet["coverage_status"] in ("not_annotated", "annotated"),
         "bad coverage_status")


# -- rendering -----------------------------------------------------------------

def render_markdown(packet: Mapping[str, Any]) -> str:
    L: list[str] = []
    L.append(f"# Evidence packet — {packet['question_id']}")
    L.append("")
    L.append(f"**Question ({packet['family']}):** {packet['question']}")
    L.append("")
    L.append(f"Strategy `{packet['strategy']}` · snapshot "
             f"`{packet['snapshot_epoch']}` · compiler "
             f"`{packet['compiler_version']}` · candidates "
             f"`{packet['candidate_hash'][:16]}…`")
    L.append("")
    L.append(f"**Anchors:** " + ", ".join(f"`{a}`" for a in packet["anchors"]))
    L.append("")
    L.append("## Core evidence")
    L.append("")
    for c in packet["core_evidence"]:
        L.append(f"### {c['evidence_id']} — {c['display']}")
        date = f", dated {c['date']}" if c.get("date") else ""
        L.append(f"- key `{c['public_key']}` ({c['label']}), about "
                 f"`{c['about']}` ({c['about_display']}), "
                 f"rank {c['rank']}, score {c['display_score']}, "
                 f"hop {c['hop']}{date}")
        L.append(f"- inclusion: {c['rationale']}")
        chain = " → ".join(f"`{k}`" for k in c["provenance_path"])
        L.append(f"- provenance [{c['asset_status']}]: {chain}")
        L.append("")
    L.append("## Explanatory paths")
    L.append("")
    for p in packet["explanatory_paths"]:
        chain = " → ".join(f"`{k}`" for k in p["keys"])
        L.append(f"- `{p['path_id']}` (len {p['length']}): {chain}")
    L.append("")
    L.append("## Context")
    L.append("")
    for c in packet["context"]:
        L.append(f"- `{c['public_key']}` — {c['display']} "
                 f"(rank {c['rank']}, hop {c['hop']}, lane {c['lane']})")
    L.append("")
    L.append("## Uncertainties")
    L.append("")
    for u in packet["uncertainties"]:
        L.append(f"- [{u['kind']}] `{u['public_key']}` — {u['display']}: "
                 f"{u['detail']}")
    if not packet["uncertainties"]:
        L.append("- none recorded")
    L.append("")
    L.append("## Research frontier")
    L.append("")
    for f in packet["research_frontier"]:
        L.append(f"- [{f['kind']}] `{f['public_key']}` — {f['display']}")
    if not packet["research_frontier"]:
        L.append("- none recorded")
    L.append("")
    L.append("## Source criticism")
    L.append("")
    L.append(f"Coverage: `{packet['coverage_status']}` — corroboration "
             f"groups: {len(packet['corroboration_groups'])}, contradiction "
             f"sets: {len(packet['contradiction_sets'])}. Unknown stance or "
             f"lineage never counts as support, contradiction, or "
             f"independence.")
    L.append("")
    L.append(f"_Unique nodes used: {packet['unique_nodes_used']} / "
             f"{packet['budgets']['max_unique_nodes']}; overflow: "
             f"{canonical.canonical_dumps(packet['overflow'])}_")
    L.append("")
    return "\n".join(L)

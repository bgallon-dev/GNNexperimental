r"""Candidate projection engine (plan T3): bounded, deterministic,
obligation-first.

Pure logic over a ``ProjectionBackend`` protocol so the engine is
hermetically testable; the live Neo4j implementation lives in
``neo4j_backend.py``. Neo4j (through the backend) owns exact topology --
this module never invents edges and never touches the encoder.

`default_v1` admission order (plan T3, in priority order):

1. confirmed anchors                              (mandatory)
2. anchor-to-anchor typed-path intermediaries     (mandatory)
3. remaining slots: deterministic round-robin BFS by anchor,
   with every 5th remaining slot reserved for low-degree candidates
   (falls back to the regular queue when none is available)

If the mandatory set alone exceeds ``max_domain_nodes`` the projection
FAILS (``ObligationOverflowError``) -- obligations are never truncated
silently. Everything else that was discovered but not admitted is counted
in ``CandidateBundle.overflow``.

Determinism: neighbor iteration is sorted, BFS queues are FIFO over
sorted expansions, path enumeration is lexicographic DFS over sorted
neighbors, and every admission tie-break is (hop, node index).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Protocol, Sequence

from .contracts import (
    CandidateBundle,
    ContractError,
    ProjectedNode,
    ProjectedRelationship,
    ResearchQuestion,
    TraversalProfile,
)


class ObligationOverflowError(ContractError):
    """Mandatory anchors + path intermediaries exceed the node budget."""


class ProjectionBackend(Protocol):
    """Minimal graph surface the engine needs. Indices are backend-local
    (cache indices for the live backend); only public keys leave here.

    ``node_payloads`` dicts must carry: public_key, key_is_fallback,
    label, neo4j_id, properties (allowlisted), raw_date, normalized_date.
    ``relationships`` dicts: rel_id, rel_type, start_idx, end_idx,
    properties. ``provenance_closure`` returns (node payloads keyed the
    same way but with 'neo4j_id'-based idx fields absent, relationship
    dicts with start_key/end_key already resolved) for the doc/extraction
    layer around the given domain nodes."""

    snapshot_epoch: str

    def anchor_index(self, public_key: str) -> int | None: ...
    def neighbors(self, idx: int) -> Sequence[int]: ...   # member-only, sorted
    def degree(self, idx: int) -> int: ...
    def low_degree_threshold(self) -> int: ...             # snapshot 25th pct
    def node_payloads(self, idxs: Sequence[int]) -> list[dict[str, Any]]: ...
    def relationships(self, idxs: Sequence[int]) -> list[dict[str, Any]]: ...
    def provenance_closure(
        self, idxs: Sequence[int]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]: ...
    def term_claim_recall(
        self, terms: Sequence[str], cap_per_term: int = 40
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]: ...


# -- exact path enumeration ---------------------------------------------------

_DFS_EXPANSION_BUDGET = 200_000


def _paths_between(backend: ProjectionBackend, a: int, b: int,
                   max_len: int, cap: int) -> tuple[list[tuple[int, ...]], bool]:
    """Simple paths a->b, <= max_len edges, lexicographic DFS order over
    sorted neighbors. Stops at ``cap`` paths or at a fixed expansion
    budget (hub-degree real graphs make full enumeration intractable);
    returns (paths, truncated). Truncation is REPORTED, never silent."""
    found: list[tuple[int, ...]] = []
    truncated = False
    budget = _DFS_EXPANSION_BUDGET
    stack: list[tuple[int, tuple[int, ...]]] = [(a, (a,))]
    while stack:
        if len(found) >= cap or budget <= 0:
            truncated = True
            break
        budget -= 1
        node, path = stack.pop()
        if node == b and len(path) > 1:
            found.append(path)
            continue
        if len(path) - 1 >= max_len:
            continue
        # push reversed so the smallest neighbor is explored first (DFS)
        for nb in reversed(backend.neighbors(node)):
            if nb not in path:                      # simple paths only
                stack.append((nb, path + (nb,)))
    return found, truncated


# -- BFS lanes ----------------------------------------------------------------

def _bfs_order(backend: ProjectionBackend, anchor: int, depth: int,
               discovery_cap: int) -> list[tuple[int, int]]:
    """[(node, hop)] in BFS discovery order from one anchor, hop <= depth,
    excluding the anchor itself. Deterministic: frontier expansion visits
    sorted neighbors. ``discovery_cap`` bounds the lane on hub-heavy real
    graphs -- the admission loop can only consume max_domain_nodes anyway."""
    seen = {anchor}
    out: list[tuple[int, int]] = []
    q: deque[tuple[int, int]] = deque([(anchor, 0)])
    while q and len(out) < discovery_cap:
        node, hop = q.popleft()
        if hop >= depth:
            continue
        for nb in backend.neighbors(node):
            if nb not in seen:
                seen.add(nb)
                out.append((nb, hop + 1))
                q.append((nb, hop + 1))
                if len(out) >= discovery_cap:
                    break
    return out


# -- the projection -----------------------------------------------------------

def project_candidates(backend: ProjectionBackend,
                       question: ResearchQuestion) -> CandidateBundle:
    question.validate(require_confirmation=True)
    profile: TraversalProfile = question.profile
    anchors = list(question.confirmation.confirmed_keys)

    # 1. resolve anchors; unknown keys fail explicitly (no substitution)
    anchor_idx: dict[str, int] = {}
    for key in anchors:
        idx = backend.anchor_index(key)
        if idx is None:
            raise ContractError(
                f"confirmed anchor {key!r} is not in the snapshot's "
                f"lifecycle-clean domain projection (no fallback substitution)")
        anchor_idx[key] = idx

    # 2. anchor-pair exact paths + mandatory intermediaries
    paths_by_pair: list[tuple[int, ...]] = []
    truncated_pairs = 0
    idxs = [anchor_idx[k] for k in anchors]
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            paths, truncated = _paths_between(
                backend, idxs[i], idxs[j],
                profile.max_path_length, profile.max_paths_per_anchor_pair)
            paths_by_pair.extend(paths)
            truncated_pairs += int(truncated)

    hop: dict[int, int] = {i: 0 for i in idxs}
    mandatory: list[int] = list(dict.fromkeys(idxs))
    for p in paths_by_pair:
        for step, node in enumerate(p[1:-1], start=1):
            if node not in hop:
                mandatory.append(node)
                hop[node] = min(step, len(p) - 1 - step)
    if len(mandatory) > profile.max_domain_nodes:
        raise ObligationOverflowError(
            f"{len(mandatory)} mandatory nodes (anchors + path "
            f"intermediaries) exceed max_domain_nodes="
            f"{profile.max_domain_nodes}; narrow anchors or scope instead "
            f"of truncating obligations")

    # 3. remaining slots: round-robin BFS by anchor; every 5th slot from
    #    the low-degree lane
    lane_cap = 50 * profile.max_domain_nodes
    lanes = [deque(_bfs_order(backend, anchor_idx[k],
                              profile.expansion_depth, lane_cap))
             for k in anchors]
    low_thresh = backend.low_degree_threshold()
    low_lane: deque[tuple[int, int]] = deque()
    admitted = set(mandatory)
    picked: list[int] = list(mandatory)
    discovered_not_admitted = 0
    slot = 0
    while len(picked) < profile.max_domain_nodes and (
            any(lanes) or low_lane):
        slot += 1
        take_low = (slot % profile.low_degree_slot_every == 0)
        node_hop: tuple[int, int] | None = None
        if take_low:
            while low_lane and low_lane[0][0] in admitted:
                low_lane.popleft()
            if low_lane:
                node_hop = low_lane.popleft()
        if node_hop is None:
            for lane in lanes:                     # round-robin: rotate below
                while lane and lane[0][0] in admitted:
                    lane.popleft()
                if lane:
                    cand = lane.popleft()
                    if backend.degree(cand[0]) <= low_thresh and not take_low:
                        low_lane.append(cand)      # defer to the reserved lane
                        continue
                    node_hop = cand
                    break
            lanes.append(lanes.pop(0))
        if node_hop is None:
            if not any(lanes) and not low_lane:
                break
            continue
        node, h = node_hop
        admitted.add(node)
        picked.append(node)
        hop[node] = min(hop.get(node, h), h)
    discovered = {n for lane in lanes for n, _ in lane} | \
        {n for n, _ in low_lane}
    discovered_not_admitted = len(discovered - admitted)

    # 4. materialize
    payloads = backend.node_payloads(picked)
    warnings: list[str] = []
    nodes: list[ProjectedNode] = []
    key_of: dict[int, str] = {}
    for idx, pay in zip(picked, payloads):
        key, fallback = pay["public_key"], bool(pay["key_is_fallback"])
        key_of[idx] = key
        if fallback:
            warnings.append(f"{key} uses a snapshot-local fallback key "
                            f"(no configured key property)")
        nodes.append(ProjectedNode(
            public_key=key, label=pay["label"], neo4j_id=pay["neo4j_id"],
            properties=pay.get("properties", {}),
            raw_date=pay.get("raw_date"),
            normalized_date=pay.get("normalized_date"),
            degree=backend.degree(idx), hop=hop.get(idx, 0),
            key_is_fallback=fallback))

    rels = [ProjectedRelationship(
        rel_id=r["rel_id"], rel_type=r["rel_type"],
        start_key=key_of[r["start_idx"]], end_key=key_of[r["end_idx"]],
        properties=r.get("properties", {}))
        for r in backend.relationships(picked)]

    overflow = {"path_enumeration_truncated_pairs": truncated_pairs,
                "discovered_not_admitted": discovered_not_admitted}
    return CandidateBundle(
        question_id=question.question_id,
        snapshot_epoch=backend.snapshot_epoch,
        anchors=tuple(anchors),
        nodes=tuple(nodes),
        relationships=tuple(rels),
        paths=tuple(tuple(key_of[n] for n in p) for p in paths_by_pair),
        profile=profile,
        overflow=overflow,
        warnings=tuple(sorted(set(warnings))),
    )


def attach_provenance(bundle: CandidateBundle,
                      backend: ProjectionBackend,
                      domain_idxs: Sequence[int]) -> CandidateBundle:
    """Attach the doc/extraction provenance closure AFTER domain ranking
    (plan T3). The closure is persisted on the bundle for offline replay
    but is never presented to the encoder and sits outside the domain cap.
    ``domain_idxs`` must be the same backend-local indices the bundle was
    projected from (ordering irrelevant)."""
    pnodes_raw, prels_raw, closure_warnings = \
        backend.provenance_closure(domain_idxs)
    domain_keys = {n.public_key for n in bundle.nodes}
    pnodes = tuple(
        _prov_node(p) for p in sorted(pnodes_raw,
                                      key=lambda p: p["public_key"])
        if p["public_key"] not in domain_keys)
    prels = tuple(
        ProjectedRelationship(
            rel_id=r["rel_id"], rel_type=r["rel_type"],
            start_key=r["start_key"], end_key=r["end_key"],
            properties=r.get("properties", {}))
        for r in sorted(prels_raw, key=lambda r: r["rel_id"]))
    fallback_warnings = tuple(
        f"{n.public_key} uses a snapshot-local fallback key "
        f"(no configured key property)"
        for n in pnodes if n.key_is_fallback)
    return CandidateBundle.from_dict({
        **bundle.to_dict(),
        "provenance_nodes": [n.to_dict() for n in pnodes],
        "provenance_relationships": [r.to_dict() for r in prels],
        "warnings": sorted({*bundle.warnings, *closure_warnings,
                            *fallback_warnings}),
    })


def _prov_node(p: dict) -> ProjectedNode:
    return ProjectedNode(
        public_key=p["public_key"], label=p["label"],
        neo4j_id=p["neo4j_id"], properties=p.get("properties", {}),
        raw_date=p.get("raw_date"),
        normalized_date=p.get("normalized_date"),
        degree=p.get("degree", 0), hop=p.get("hop", 0),
        key_is_fallback=bool(p["key_is_fallback"]),
        origin=p.get("origin", ""))


def attach_term_recall(bundle: CandidateBundle,
                       backend: ProjectionBackend,
                       terms: Sequence[str]) -> CandidateBundle:
    """Merge the bounded lexical claim-recall source into the bundle
    (plan amendment A2, adopted 2026-07-13). Recalled nodes carry
    ``origin=term_recall:<term>``; nodes already present (domain or
    closure) keep their existing entry; relationships dedup by rel_id.
    Without this source ~71% of graded evidence was unreachable
    (FINDINGS 2026-07-13)."""
    pnodes_raw, prels_raw, recall_warnings = backend.term_claim_recall(terms)
    have = {n.public_key for n in (*bundle.nodes, *bundle.provenance_nodes)}
    new_nodes = tuple(
        _prov_node(p) for p in sorted(pnodes_raw,
                                      key=lambda p: p["public_key"])
        if p["public_key"] not in have)
    have_rels = {r.rel_id for r in (*bundle.relationships,
                                    *bundle.provenance_relationships)}
    new_rels = tuple(
        ProjectedRelationship(
            rel_id=r["rel_id"], rel_type=r["rel_type"],
            start_key=r["start_key"], end_key=r["end_key"],
            properties=r.get("properties", {}))
        for r in sorted(prels_raw, key=lambda r: r["rel_id"])
        if r["rel_id"] not in have_rels)
    fallback_warnings = tuple(
        f"{n.public_key} uses a snapshot-local fallback key "
        f"(no configured key property)"
        for n in new_nodes if n.key_is_fallback)
    merged = CandidateBundle.from_dict({
        **bundle.to_dict(),
        "provenance_nodes": [n.to_dict() for n in
                             (*bundle.provenance_nodes, *new_nodes)],
        "provenance_relationships": [r.to_dict() for r in
                                     (*bundle.provenance_relationships,
                                      *new_rels)],
        "overflow": {**bundle.overflow,
                     "term_recall_nodes_added": len(new_nodes)},
        "warnings": sorted({*bundle.warnings, *recall_warnings,
                            *fallback_warnings}),
    })
    merged.validate()
    return merged

r"""Evidence-workspace core types (plan: Public contracts).

Stdlib dataclasses only. Every type round-trips ``to_dict``/``from_dict``
losslessly (reconstruction exactness condition 2) and validates eagerly:
a contract violation raises ``ContractError`` at construction/validation
time, never during compilation.

Identity rules carried from the plan:
- Public identifiers are configured typed keys (``Claim:<claim_id>``);
  numeric Neo4j IDs are snapshot-local metadata only.
- ``question_id`` derives from canonical question text + scope.
- Anchor confirmation is mandatory and is tied to both the resolver
  output hash and the graph snapshot -- a confirmation minted against a
  different resolution or snapshot is stale and must fail explicitly.
- Stance ``unknown`` never counts as support, contradiction, or
  independent corroboration (enforced where stances are consumed; the
  contract here only guarantees the vocabulary).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping

from . import canonical

# 0.2.0 (2026-07-13, pre-data amendment): ProjectedNode.origin source tag;
# bundles may carry term-recall claims beside the anchor closure
SCHEMA_VERSION = "0.2.0"

QUESTION_FAMILIES = (
    "policy_change",
    "provenance",
    "institutional_bridge",
    "contested_claim",
)

STANCES = ("supports", "contradicts", "neutral", "unknown")

RANKING_STRATEGIES = ("kgr", "bfs", "lexical")


class ContractError(ValueError):
    """A contract violation. Message names the field and the rule."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractError(msg)


@dataclass(frozen=True)
class TraversalProfile:
    """Bounds on candidate projection (plan T3, `default_v1`)."""

    name: str = "default_v1"
    max_anchors: int = 8
    expansion_depth: int = 3
    max_domain_nodes: int = 400
    max_path_length: int = 4
    max_paths_per_anchor_pair: int = 50
    low_degree_slot_every: int = 5

    def validate(self) -> None:
        _require(bool(self.name), "TraversalProfile.name is required")
        for f in ("max_anchors", "expansion_depth", "max_domain_nodes",
                  "max_path_length", "max_paths_per_anchor_pair",
                  "low_degree_slot_every"):
            _require(getattr(self, f) > 0, f"TraversalProfile.{f} must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TraversalProfile":
        return cls(**d)


DEFAULT_V1 = TraversalProfile()


@dataclass(frozen=True)
class PacketBudget:
    """Hard packet budgets (plan T5). Forced obligations that exceed the
    budget fail compilation; they are never silently truncated."""

    core_evidence: int = 12
    explanatory_paths: int = 5
    context_items: int = 8
    context_reserved_nonlocal: int = 2
    frontier_entries: int = 8
    max_unique_nodes: int = 80

    def validate(self) -> None:
        for f in ("core_evidence", "explanatory_paths", "context_items",
                  "frontier_entries", "max_unique_nodes"):
            _require(getattr(self, f) > 0, f"PacketBudget.{f} must be > 0")
        _require(0 <= self.context_reserved_nonlocal <= self.context_items,
                 "PacketBudget.context_reserved_nonlocal must fit in context_items")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PacketBudget":
        return cls(**d)


@dataclass(frozen=True)
class QuestionScope:
    """Calendar/place/collection scope. Calendar years are inclusive and
    kept in calendar form here; normalized model-time values live beside
    them in the candidate bundle (both forms are always preserved)."""

    year_start: int | None = None
    year_end: int | None = None
    places: tuple[str, ...] = ()
    collections: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.year_start is not None and self.year_end is not None:
            _require(self.year_start <= self.year_end,
                     "QuestionScope.year_start must be <= year_end")

    def to_dict(self) -> dict[str, Any]:
        return {"year_start": self.year_start, "year_end": self.year_end,
                "places": list(self.places),
                "collections": list(self.collections)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "QuestionScope":
        return cls(year_start=d.get("year_start"), year_end=d.get("year_end"),
                   places=tuple(d.get("places", ())),
                   collections=tuple(d.get("collections", ())))


@dataclass(frozen=True)
class AnchorResolution:
    """One resolver candidate. ``public_key`` is the typed key
    (``Label:value``); ``neo4j_id`` is snapshot-local metadata only and
    never appears in public references."""

    public_key: str
    label: str
    display_name: str
    match_method: str        # exact_id | name_contains | date | vector
    match_evidence: str
    rank: int
    snapshot_epoch: str
    neo4j_id: int | None = None
    key_is_fallback: bool = False   # no configured key property existed

    def validate(self) -> None:
        _require(bool(self.public_key), "AnchorResolution.public_key is required")
        _require(":" in self.public_key,
                 "AnchorResolution.public_key must be a typed key 'Label:value'")
        _require(bool(self.label), "AnchorResolution.label is required")
        _require(self.rank >= 1, "AnchorResolution.rank must be >= 1")
        _require(bool(self.snapshot_epoch),
                 "AnchorResolution.snapshot_epoch is required")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AnchorResolution":
        return cls(**d)


@dataclass(frozen=True)
class AnchorConfirmation:
    """Human confirmation of anchors, bound to one resolver output and one
    graph snapshot. ``resolution_hash`` is the content hash of the full
    resolver candidate list the human actually saw."""

    confirmed_keys: tuple[str, ...]
    rejected_keys: tuple[str, ...]
    resolution_hash: str
    snapshot_epoch: str
    confirmed_by: str
    confirmed_at: str            # supplied by the caller; not machine time

    def validate(self) -> None:
        _require(len(self.confirmed_keys) >= 1,
                 "AnchorConfirmation requires at least one confirmed key")
        _require(len(set(self.confirmed_keys)) == len(self.confirmed_keys),
                 "AnchorConfirmation.confirmed_keys contains duplicates")
        overlap = set(self.confirmed_keys) & set(self.rejected_keys)
        _require(not overlap,
                 f"keys both confirmed and rejected: {sorted(overlap)}")
        for f in ("resolution_hash", "snapshot_epoch", "confirmed_by",
                  "confirmed_at"):
            _require(bool(getattr(self, f)), f"AnchorConfirmation.{f} is required")

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["confirmed_keys"] = list(self.confirmed_keys)
        d["rejected_keys"] = list(self.rejected_keys)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AnchorConfirmation":
        return cls(confirmed_keys=tuple(d["confirmed_keys"]),
                   rejected_keys=tuple(d.get("rejected_keys", ())),
                   resolution_hash=d["resolution_hash"],
                   snapshot_epoch=d["snapshot_epoch"],
                   confirmed_by=d["confirmed_by"],
                   confirmed_at=d["confirmed_at"])


@dataclass(frozen=True)
class ResearchQuestion:
    text: str
    family: str
    scope: QuestionScope
    profile: TraversalProfile = DEFAULT_V1
    budget: PacketBudget = PacketBudget()
    candidates: tuple[AnchorResolution, ...] = ()
    confirmation: AnchorConfirmation | None = None
    schema_version: str = SCHEMA_VERSION

    @property
    def question_id(self) -> str:
        return canonical.question_id(self.text, self.scope.to_dict())

    def validate(self, *, require_confirmation: bool = False) -> None:
        _require(len(self.text.strip()) >= 10,
                 "ResearchQuestion.text must be a real question (>= 10 chars)")
        _require(self.family in QUESTION_FAMILIES,
                 f"ResearchQuestion.family {self.family!r} not in "
                 f"{QUESTION_FAMILIES}")
        self.scope.validate()
        self.profile.validate()
        self.budget.validate()
        for c in self.candidates:
            c.validate()
        if require_confirmation:
            _require(self.confirmation is not None,
                     "compilation requires an AnchorConfirmation; none present")
        if self.confirmation is not None:
            self.confirmation.validate()
            _require(len(self.confirmation.confirmed_keys)
                     <= self.profile.max_anchors,
                     f"{len(self.confirmation.confirmed_keys)} confirmed anchors "
                     f"exceed profile max_anchors={self.profile.max_anchors}")
            known = {c.public_key for c in self.candidates}
            unknown = set(self.confirmation.confirmed_keys) - known
            _require(not unknown,
                     f"confirmed keys not among resolver candidates "
                     f"(stale confirmation?): {sorted(unknown)}")
            res_hash = canonical.content_hash(
                [c.to_dict() for c in self.candidates])
            _require(self.confirmation.resolution_hash == res_hash,
                     "AnchorConfirmation.resolution_hash does not match the "
                     "candidate list on this question (stale confirmation)")
            epochs = {c.snapshot_epoch for c in self.candidates
                      if c.public_key in self.confirmation.confirmed_keys}
            _require(epochs == {self.confirmation.snapshot_epoch},
                     "AnchorConfirmation.snapshot_epoch does not match the "
                     "confirmed candidates' snapshot (stale snapshot)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "question_id": self.question_id,
            "text": self.text,
            "family": self.family,
            "scope": self.scope.to_dict(),
            "profile": self.profile.to_dict(),
            "budget": self.budget.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "confirmation": (self.confirmation.to_dict()
                             if self.confirmation else None),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ResearchQuestion":
        return cls(
            text=d["text"], family=d["family"],
            scope=QuestionScope.from_dict(d["scope"]),
            profile=TraversalProfile.from_dict(d["profile"]),
            budget=PacketBudget.from_dict(d["budget"]),
            candidates=tuple(AnchorResolution.from_dict(c)
                             for c in d.get("candidates", ())),
            confirmation=(AnchorConfirmation.from_dict(d["confirmation"])
                          if d.get("confirmation") else None),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class ProjectedNode:
    public_key: str
    label: str
    neo4j_id: int
    properties: dict[str, Any] = field(default_factory=dict)
    raw_date: str | None = None
    normalized_date: float | None = None
    degree: int = 0
    hop: int = 0
    key_is_fallback: bool = False
    # candidate source: "" = anchor closure (the default projection);
    # "term_recall:<term>" = the amendment's bounded lexical claim recall
    origin: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ProjectedNode":
        return cls(**d)


@dataclass(frozen=True)
class ProjectedRelationship:
    rel_id: int
    rel_type: str
    start_key: str
    end_key: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ProjectedRelationship":
        return cls(**d)


def _rel_sort_key(r: "ProjectedRelationship") -> tuple[str, str, str, int]:
    """Canonical, total edge order over PORTABLE keys (rel_id is a unique
    last-resort tiebreak, never the primary sort field). Makes the compiled
    packet and candidate_hash invariant to the order a backend emits edges in
    -- a restore/compaction that reassigns Neo4j ids, or a non-Neo4j exporter,
    yields the same edge multiset in a different list order, and must not
    change the packet."""
    return (r.start_key, r.end_key, r.rel_type, r.rel_id)


@dataclass(frozen=True)
class CandidateBundle:
    """Immutable projected snapshot; stored with every revision so the
    packet replays without a live database.

    ``nodes`` is the domain projection presented to ranking (capped by
    ``profile.max_domain_nodes``). ``provenance_nodes``/``_relationships``
    are the post-ranking provenance/containment closure (Pages, Documents,
    extraction runs); they are NEVER fed to the encoder and sit outside
    the domain cap, but must be persisted for offline replay."""

    question_id: str
    snapshot_epoch: str
    anchors: tuple[str, ...]
    nodes: tuple[ProjectedNode, ...]
    relationships: tuple[ProjectedRelationship, ...]
    paths: tuple[tuple[str, ...], ...]      # ordered public-key sequences
    profile: TraversalProfile
    provenance_nodes: tuple[ProjectedNode, ...] = ()
    provenance_relationships: tuple[ProjectedRelationship, ...] = ()
    overflow: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        # Canonicalize the backend-emission-sensitive collections so
        # candidate_hash and the compiled packet are a pure function of bundle
        # CONTENT, not of the order a backend happened to emit edges/closure
        # nodes in (edges: Cypher ORDER BY rid; provenance_nodes: two-phase
        # closure+recall concat). Domain `nodes` (engine admission order over
        # sorted neighbors), `anchors` (question-derived; drives round-robin
        # lanes) and `paths` are already reproducible and are left as-is.
        # Idempotent: an already-canonical bundle re-canonicalizes to itself.
        object.__setattr__(self, "relationships",
                           tuple(sorted(self.relationships, key=_rel_sort_key)))
        object.__setattr__(self, "provenance_relationships",
                           tuple(sorted(self.provenance_relationships,
                                        key=_rel_sort_key)))
        object.__setattr__(self, "provenance_nodes",
                           tuple(sorted(self.provenance_nodes,
                                        key=lambda n: n.public_key)))

    def validate(self) -> None:
        _require(bool(self.question_id), "CandidateBundle.question_id required")
        _require(bool(self.snapshot_epoch), "CandidateBundle.snapshot_epoch required")
        self.profile.validate()
        keys = [n.public_key for n in self.nodes]
        _require(len(set(keys)) == len(keys),
                 "CandidateBundle.nodes contains duplicate public keys")
        pkeys = [n.public_key for n in self.provenance_nodes]
        _require(len(set(pkeys)) == len(pkeys),
                 "CandidateBundle.provenance_nodes contains duplicate keys")
        _require(not set(pkeys) & set(keys),
                 "provenance_nodes duplicate domain node keys")
        keyset = set(keys)
        missing_anchor = set(self.anchors) - keyset
        _require(not missing_anchor,
                 f"anchors missing from projected nodes: {sorted(missing_anchor)}")
        _require(len(self.nodes) <= self.profile.max_domain_nodes,
                 f"{len(self.nodes)} nodes exceed profile "
                 f"max_domain_nodes={self.profile.max_domain_nodes}")
        for r in self.relationships:
            _require(r.start_key in keyset and r.end_key in keyset,
                     f"relationship {r.rel_id} references keys outside the bundle")
        allkeys = keyset | set(pkeys)
        for r in self.provenance_relationships:
            _require(r.start_key in allkeys and r.end_key in allkeys,
                     f"provenance relationship {r.rel_id} references keys "
                     f"outside the bundle")
        for p in self.paths:
            _require(len(p) - 1 <= self.profile.max_path_length,
                     f"path longer than profile max_path_length: {p}")
            dangling = set(p) - keyset
            _require(not dangling,
                     f"path references keys outside the bundle: {sorted(dangling)}")

    @property
    def dependency_hash(self) -> str:
        return canonical.content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "question_id": self.question_id,
            "snapshot_epoch": self.snapshot_epoch,
            "anchors": list(self.anchors),
            "nodes": [n.to_dict() for n in self.nodes],
            "relationships": [r.to_dict() for r in self.relationships],
            "paths": [list(p) for p in self.paths],
            "profile": self.profile.to_dict(),
            "provenance_nodes": [n.to_dict() for n in self.provenance_nodes],
            "provenance_relationships": [r.to_dict() for r in
                                         self.provenance_relationships],
            "overflow": dict(self.overflow),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CandidateBundle":
        return cls(
            question_id=d["question_id"],
            snapshot_epoch=d["snapshot_epoch"],
            anchors=tuple(d["anchors"]),
            nodes=tuple(ProjectedNode.from_dict(n) for n in d["nodes"]),
            relationships=tuple(ProjectedRelationship.from_dict(r)
                                for r in d["relationships"]),
            paths=tuple(tuple(p) for p in d["paths"]),
            profile=TraversalProfile.from_dict(d["profile"]),
            provenance_nodes=tuple(ProjectedNode.from_dict(n)
                                   for n in d.get("provenance_nodes", ())),
            provenance_relationships=tuple(
                ProjectedRelationship.from_dict(r)
                for r in d.get("provenance_relationships", ())),
            overflow=dict(d.get("overflow", {})),
            warnings=tuple(d.get("warnings", ())),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class RankedCandidate:
    public_key: str
    strategy: str                 # kgr | bfs | lexical
    rank: int
    micro_score: int              # quantized; raw floats are not persisted
    hop: int
    contributing_anchors: tuple[str, ...] = ()
    bfs_rank: int | None = None
    lexical_rank: int | None = None
    rationale: str = ""

    def validate(self) -> None:
        _require(self.strategy in RANKING_STRATEGIES,
                 f"RankedCandidate.strategy {self.strategy!r} not in "
                 f"{RANKING_STRATEGIES}")
        _require(self.rank >= 1, "RankedCandidate.rank must be >= 1")
        _require(isinstance(self.micro_score, int),
                 "RankedCandidate.micro_score must be an int (quantized)")

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["contributing_anchors"] = list(self.contributing_anchors)
        d["display_score"] = canonical.format_micro_score(self.micro_score)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RankedCandidate":
        d = {k: v for k, v in d.items() if k != "display_score"}
        d["contributing_anchors"] = tuple(d.get("contributing_anchors", ()))
        return cls(**d)


@dataclass(frozen=True)
class AnnotationRecord:
    """Versioned sidecar judgment. ``unknown`` stance never counts as
    support, contradiction, or independent corroboration."""

    record_id: str
    claim_cluster: str
    source_work: str
    manifestation: str
    lineage_root: str            # "" == unknown lineage (explicitly unknown)
    stance: str
    directness: str              # direct | secondhand | derived | unknown
    asset_target: str
    reviewer: str
    reviewed_at: str
    version: int = 1
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        _require(bool(self.record_id), "AnnotationRecord.record_id required")
        _require(self.stance in STANCES,
                 f"AnnotationRecord.stance {self.stance!r} not in {STANCES}")
        _require(self.version >= 1, "AnnotationRecord.version must be >= 1")
        _require(bool(self.reviewer), "AnnotationRecord.reviewer required")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AnnotationRecord":
        return cls(**d)


@dataclass(frozen=True)
class BuildManifest:
    """Everything a revision's bytes depend on. Two builds under one
    manifest must be byte-identical; a changed field here is what mints
    the next revision."""

    question_id: str
    revision: int
    candidate_hash: str
    snapshot_epoch: str
    model_sha: str
    schema_map_hash: str
    routing_hash: str
    compiler_version: str
    sidecar_hashes: dict[str, str]
    runtime_settings: dict[str, Any]
    output_hashes: dict[str, str] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        for f in ("question_id", "candidate_hash", "snapshot_epoch",
                  "compiler_version"):
            _require(bool(getattr(self, f)), f"BuildManifest.{f} is required")
        _require(self.revision >= 1, "BuildManifest.revision must be >= 1")

    @property
    def dependency_hash(self) -> str:
        """Hash over the input-dependency fields only (not output hashes):
        equality here is the revision-reuse test."""
        d = self.to_dict()
        d.pop("output_hashes")
        d.pop("revision")
        return canonical.content_hash(d)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BuildManifest":
        return cls(**d)

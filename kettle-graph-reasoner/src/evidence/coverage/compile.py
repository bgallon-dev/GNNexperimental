r"""Canonical, database-free coverage.json compilation (plan T7.0 / §11).

Consumes only portable artifacts and explicit sidecar inputs:

* ``question`` -- a ``ResearchQuestion`` (the pipeline's ``question.json``);
* ``bundle``   -- a ``CandidateBundle`` (``candidates.json``);
* ``packet``   -- a compiled packet dict (``packet.json``);
* ``shape`` + ``selection`` -- the explicit, reviewed answer-shape binding;
* ``assignments`` -- reviewed or model-proposed evidence-to-slot records; and
* ``universe`` -- the recorded search receipts (evidence universe).

It runs the deterministic evaluator and emits an immutable ``CoverageArtifact``.
There is NO database access and NO semantic extraction here: assignments and
search receipts are supplied, not inferred from evidence text. That honesty is
the point of the smallest T7.1 baseline -- the deterministic *kernel* is real;
nomination is left to T7.1's later rule/lexical lane and T7.4's learned scorer.

Every assignment's ``evidence_id`` must resolve to a real evidence handle in
the packet or the candidate snapshot; a dangling id fails closed here rather
than producing a plausible-but-ungrounded artifact.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Any, Iterable, Mapping, Sequence

from .. import canonical
from ..contracts import CandidateBundle, ResearchQuestion
from coverage_reasoner import evaluator
from coverage_reasoner.contracts import (
    AnswerShape,
    CoverageArtifact,
    CoverageAssignment,
    CoverageContractError,
    CoverageManifest,
    EvidenceUniverse,
    ShapeSelection,
    SourceCoverage,
    COVERAGE_SCHEMA_VERSION,
)

COVERAGE_COMPILER_VERSION = "0.1.0"


# Packet sections beyond core_evidence whose items are named by public_key.
_PACKET_KEY_SECTIONS = ("context", "uncertainties", "research_frontier",
                        "exclusions")


def _packet_section_ids(packet: Mapping[str, Any], section: str,
                        id_fields: tuple[str, ...]) -> set[str]:
    """Evidence ids from one packet section. Robust: an absent section
    contributes nothing. Fail-closed: a present section must be a list of
    objects, and each object must carry at least one non-empty string id in
    ``id_fields`` -- a malformed item raises rather than being silently skipped,
    which could otherwise mask a genuinely dangling reference downstream."""
    raw = packet.get(section)
    if raw is None:
        return set()
    if not isinstance(raw, (list, tuple)):
        raise CoverageContractError(
            f"packet.{section} must be a list when present, "
            f"got {type(raw).__name__}")
    ids: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, MappingABC):
            raise CoverageContractError(
                f"packet.{section}[{i}] must be an object")
        found = False
        for field in id_fields:
            if field in item and item[field] is not None:
                value = item[field]
                if not (isinstance(value, str) and value):
                    raise CoverageContractError(
                        f"packet.{section}[{i}].{field} must be a non-empty "
                        f"string")
                ids.add(value)
                found = True
        if not found:
            raise CoverageContractError(
                f"packet.{section}[{i}] carries none of the id fields "
                f"{id_fields}")
    return ids


def _bundle_node_keys(nodes: Iterable[Any], label: str) -> set[str]:
    """Public keys from a candidate-bundle node collection. Accepts either
    ``ProjectedNode`` dataclasses or their mappings; fail-closed on any node
    lacking a non-empty string ``public_key``."""
    ids: set[str] = set()
    for i, n in enumerate(nodes):
        key = (n.get("public_key") if isinstance(n, MappingABC)
               else getattr(n, "public_key", None))
        if not (isinstance(key, str) and key):
            raise CoverageContractError(
                f"candidate_bundle.{label}[{i}] has no valid public_key")
        ids.add(key)
    return ids


def valid_evidence_ids(packet: Mapping[str, Any],
                       bundle: CandidateBundle) -> set[str]:
    """The set of evidence handles an assignment may reference: the packet's
    minted ``evidence_id`` strings and public keys (across core evidence,
    context, uncertainties, research frontier, and exclusions), plus every
    domain and provenance node public key in the candidate snapshot. This is the
    portable analogue of the plan invariant "every assignment resolves to an
    evidence ID in the packet or candidate snapshot".

    Extraction is robust to absent sections but fails closed on malformed ones:
    a present-but-mistyped section, a non-object item, or an item missing every
    expected id field is a contract error, not a silent skip."""
    if not isinstance(packet, MappingABC):
        raise CoverageContractError(
            f"packet must be a mapping, got {type(packet).__name__}")
    ids = _packet_section_ids(packet, "core_evidence",
                              ("evidence_id", "public_key"))
    for section in _PACKET_KEY_SECTIONS:
        ids |= _packet_section_ids(packet, section, ("public_key",))

    nodes = getattr(bundle, "nodes", None)
    prov = getattr(bundle, "provenance_nodes", None)
    if nodes is None and isinstance(bundle, MappingABC):
        nodes = bundle.get("nodes")
        prov = bundle.get("provenance_nodes")
    if nodes is None:
        raise CoverageContractError(
            "candidate_bundle exposes no 'nodes' projection")
    ids |= _bundle_node_keys(nodes, "nodes")
    ids |= _bundle_node_keys(prov or (), "provenance_nodes")
    return ids


def compile_coverage(question: ResearchQuestion,
                     bundle: CandidateBundle,
                     packet: Mapping[str, Any],
                     shape: AnswerShape,
                     selection: ShapeSelection,
                     assignments: Iterable[CoverageAssignment] = (),
                     universe: EvidenceUniverse = EvidenceUniverse(),
                     *,
                     packet_id: str = "",
                     packet_revision: int = 0,
                     runtime_settings: Mapping[str, Any] | None = None
                     ) -> CoverageArtifact:
    """Portable artifacts + explicit sidecars -> immutable ``CoverageArtifact``."""
    assignments = list(assignments)

    # 1. validate every input eagerly.
    if not isinstance(packet, MappingABC):
        raise CoverageContractError(
            f"packet must be a mapping, got {type(packet).__name__}")
    question.validate(require_confirmation=True)
    bundle.validate()
    shape.validate()
    selection.validate()
    universe.validate()
    for a in assignments:
        a.validate()

    # Materialize the NOT_SEARCHED default (plan: an unlisted source class is
    # NOT_SEARCHED) for every source class the shape references but the caller
    # did not list. This keeps a source-class verdict reason -- emitted when a
    # required slot is unfilled -- grounded in the stored universe, so
    # compiling an honest abstention with an under-specified universe no longer
    # fails CoverageArtifact.validate(). Caller-declared states are preserved.
    universe = _complete_universe(universe, shape)

    # 2. cross-artifact identity must line up (no silent cross-question mixing).
    if bundle.question_id != question.question_id:
        raise CoverageContractError(
            "bundle was projected for a different question than the one given")
    if packet.get("question_id") not in (None, question.question_id):
        raise CoverageContractError(
            "packet was compiled for a different question than the one given")
    if packet.get("candidate_hash") not in (None, bundle.dependency_hash):
        raise CoverageContractError(
            "packet was compiled from a different candidate bundle")
    if packet.get("snapshot_epoch") not in (None, bundle.snapshot_epoch):
        raise CoverageContractError(
            "packet was compiled from a different snapshot epoch")
    if selection.shape_id != shape.shape_id or \
            selection.shape_version != shape.shape_version:
        raise CoverageContractError(
            "shape selection does not name the supplied shape "
            f"({selection.shape_id}@{selection.shape_version} vs "
            f"{shape.shape_id}@{shape.shape_version})")

    # 3. every assignment must ground to a slot and to a real evidence handle.
    slot_ids = shape.slot_ids()
    handles = valid_evidence_ids(packet, bundle)
    aids = [a.assignment_id for a in assignments]
    if len(set(aids)) != len(aids):
        raise CoverageContractError("duplicate assignment_id in inputs")
    for a in assignments:
        if a.slot_id not in slot_ids:
            raise CoverageContractError(
                f"assignment {a.assignment_id} references slot {a.slot_id!r} "
                f"absent from shape {shape.shape_id}")
        if a.evidence_id not in handles:
            raise CoverageContractError(
                f"assignment {a.assignment_id} references evidence "
                f"{a.evidence_id!r} that resolves to no packet/candidate "
                f"evidence handle (dangling evidence id)")

    # 4. run the deterministic kernel.
    slot_cov, gate_results, verdict, frontier = evaluator.evaluate(
        shape, selection, assignments, universe)
    verdict = _with_gate_results(verdict, gate_results)

    # 5. dependency manifest (byte-identity is a function of these).
    packet_hash = canonical.content_hash(dict(packet))
    candidate_hash = bundle.dependency_hash
    manifest = CoverageManifest(
        question_id=question.question_id,
        packet_hash=packet_hash,
        candidate_hash=candidate_hash,
        snapshot_epoch=bundle.snapshot_epoch,
        shape_id=shape.shape_id,
        shape_version=shape.shape_version,
        shape_selection_hash=canonical.content_hash(selection.to_dict()),
        assignments_hash=canonical.content_hash(
            [a.to_dict() for a in _sorted_assignments(assignments)]),
        evidence_universe_hash=canonical.content_hash(universe.to_dict()),
        coverage_compiler_version=COVERAGE_COMPILER_VERSION,
        packet_id=packet_id,
        packet_revision=packet_revision,
        runtime_settings=dict(runtime_settings or {}),
    )

    artifact = CoverageArtifact(
        question_id=question.question_id,
        packet_hash=packet_hash,
        candidate_hash=candidate_hash,
        snapshot_epoch=bundle.snapshot_epoch,
        shape=shape,
        bindings=dict(selection.bindings),
        evidence_universe=universe,
        assignments=tuple(_sorted_assignments(assignments)),
        slot_coverage=slot_cov,
        verdict=verdict,
        research_frontier=frontier,
        manifest=manifest,
        packet_id=packet_id,
        packet_revision=packet_revision,
        schema_version=COVERAGE_SCHEMA_VERSION,
    )
    artifact.validate()

    # record the output hash on the manifest inside the artifact bytes so a
    # validate step can detect tampering (mirrors BuildManifest.output_hashes).
    body = artifact.to_dict()
    body["build_manifest"]["output_hashes"] = {
        "coverage.json": canonical.content_hash(_manifest_free(body))}
    compiled = CoverageArtifact.from_dict(body)
    compiled.validate()
    return compiled


def coverage_bytes(artifact: CoverageArtifact) -> bytes:
    """Canonical UTF-8 bytes of a coverage artifact (newline-terminated)."""
    return canonical.canonical_bytes(artifact.to_dict())


# -- helpers -------------------------------------------------------------------

def _complete_universe(universe: EvidenceUniverse,
                       shape: AnswerShape) -> EvidenceUniverse:
    """Add a ``NOT_SEARCHED`` receipt for every shape source class the caller
    did not list, preserving declared states verbatim. Deterministic
    (source-class-sorted) so it never perturbs byte-identity."""
    declared = {s.source_class for s in universe.sources}
    extra: list[SourceCoverage] = []
    for slot in shape.slots:
        for sc in slot.source_classes:
            if sc not in declared:
                declared.add(sc)
                extra.append(SourceCoverage(
                    sc, "NOT_SEARCHED",
                    detail="defaulted: not listed in the provided evidence "
                           "universe"))
    if not extra:
        return universe
    merged = tuple(sorted(universe.sources + tuple(extra),
                          key=lambda s: s.source_class))
    completed = EvidenceUniverse(sources=merged)
    completed.validate()
    return completed


def _sorted_assignments(assignments: Sequence[CoverageAssignment]
                        ) -> list[CoverageAssignment]:
    # canonical, emission-order-invariant assignment order.
    return sorted(assignments,
                  key=lambda a: (a.slot_id, a.evidence_id, a.assignment_id))


def _with_gate_results(verdict, gate_results):
    # the evaluator already attaches gate results to the verdict; keep them in
    # declared shape-gate order for stable bytes.
    from coverage_reasoner.contracts import AnswerabilityVerdict

    return AnswerabilityVerdict(
        status=verdict.status, reasons=verdict.reasons,
        permitted_claim_classes=verdict.permitted_claim_classes,
        prohibited_inferences=verdict.prohibited_inferences,
        licensed_subclaims=verdict.licensed_subclaims,
        hard_gate_results=tuple(gate_results),
        confidence_micro=None)


def _manifest_free(body: Mapping[str, Any]) -> dict[str, Any]:
    # hash the artifact body with the mutable output_hashes field removed, so
    # the recorded output hash is a stable function of everything else.
    out = dict(body)
    bm = dict(out["build_manifest"])
    bm.pop("output_hashes", None)
    out["build_manifest"] = bm
    return out

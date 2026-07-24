r"""Offline, database-free packet compilation from persisted artifacts.

The online pipeline (``pipeline.EvidencePipeline.compile``) resolves anchors,
projects, and closes provenance against a live Neo4j source, then writes
``question.json`` + ``candidates.json`` beside the packet. Everything AFTER
projection is host-neutral:

* ``compiler.compile_packet`` consumes only the in-memory ``CandidateBundle``
  (no session, no Cypher).
* the ``bfs`` and ``lexical`` ranking lanes read only the bundle (+ question
  text) -- see ``ranking.bfs_ranking`` / ``ranking.lexical_ranking``.

This module wires those already-pure pieces into a standalone compile path so
an archive can export the two artifacts once and a historian can compile,
re-compile, validate, and interpret packets with NO database.

Ranking-*computation* is where host coupling remains. The ``kgr`` lane needs
the frozen encoder's ``pull_by_ids`` and host-local manifold row ids
(``ranking.kgr_ranking`` / ``ranking.KGREmbedder``), so it cannot be computed
offline and is gated here. A ``kgr`` ranking computed ONLINE can still be
*compiled* offline by serializing the ``RankedCandidate`` list and supplying it
via ``compile_bundle(ranking=...)`` -- proof that compilation, not ranking, is
the portable boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .contracts import (
    AnnotationRecord,
    CandidateBundle,
    ContractError,
    RankedCandidate,
    ResearchQuestion,
)

# Ranking lanes that are fully computable without a live host.
OFFLINE_STRATEGIES = ("bfs", "lexical")


class OfflineRankingError(ContractError):
    """A ranking strategy was requested that cannot run without a live host."""


def rank_bundle(bundle: CandidateBundle, question: ResearchQuestion,
                strategy: str = "bfs") -> list[RankedCandidate]:
    """Compute a ranking over a projected bundle with NO database access.

    ``bfs`` reads only the bundle's own edges; ``lexical`` reads the bundle
    plus the question text. ``kgr`` is refused: it needs the frozen encoder
    and host-local manifold row ids that only exist beside the live source.
    Compile a ``kgr`` packet offline by supplying a ranking precomputed
    online (see ``compile_bundle(ranking=...)``)."""
    from .ranking import bfs_ranking, lexical_ranking

    if strategy == "bfs":
        return bfs_ranking(bundle)
    if strategy == "lexical":
        return lexical_ranking(bundle, question)
    if strategy == "kgr":
        raise OfflineRankingError(
            "kgr ranking cannot be computed offline: it requires the frozen "
            "encoder and host-local manifold row ids (pull_by_ids). Compute "
            "it online and pass the serialized ranking via --ranking, or use "
            "--strategy bfs|lexical.")
    raise OfflineRankingError(
        f"unknown ranking strategy {strategy!r}; offline lanes are "
        f"{OFFLINE_STRATEGIES}")


def validate_ranking_for_bundle(ranking: Sequence[RankedCandidate],
                                bundle: CandidateBundle, *,
                                allow_partial: bool = False) -> None:
    """A SUPPLIED ranking must be faithful to THIS bundle.

    The ``--ranking`` escape hatch (offline compilation of an online-computed
    ``kgr`` ranking) trusts caller-provided data, so it is validated here
    rather than silently trusted:

    * every entry is a well-formed ``RankedCandidate`` (``rc.validate()``);
    * no key is listed twice;
    * every key names a domain node of this bundle -- an out-of-bundle key
      would otherwise reach ``label[rc.public_key]`` and crash the compiler
      with ``KeyError`` instead of a clean contract error;
    * unless ``allow_partial``, every domain node is covered -- a ranking whose
      node set no longer matches the (re-projected) bundle is a stale/wrong
      ranking and is rejected rather than silently producing a wrong-but-valid
      packet. ``allow_partial`` exists for a ``kgr`` ranking that legitimately
      dropped nodes without embeddings; it names the risk instead of hiding
      it."""
    domain = {n.public_key for n in bundle.nodes}
    seen: set[str] = set()
    for rc in ranking:
        rc.validate()
        if rc.public_key in seen:
            raise OfflineRankingError(
                f"supplied ranking lists {rc.public_key!r} more than once")
        seen.add(rc.public_key)
    unknown = seen - domain
    if unknown:
        raise OfflineRankingError(
            f"supplied ranking references {len(unknown)} key(s) not in the "
            f"bundle's domain nodes (stale ranking / wrong bundle?): "
            f"{sorted(unknown)[:5]}")
    if not allow_partial:
        missing = domain - seen
        if missing:
            raise OfflineRankingError(
                f"supplied ranking covers {len(seen)} of {len(domain)} domain "
                f"nodes; {len(missing)} uncovered (stale ranking / wrong "
                f"bundle?): {sorted(missing)[:5]} -- pass allow_partial to "
                f"compile a deliberately partial ranking anyway")


def compile_bundle(question: ResearchQuestion, bundle: CandidateBundle, *,
                   strategy: str = "bfs",
                   ranking: Sequence[RankedCandidate] | None = None,
                   annotations: Iterable[AnnotationRecord] = (),
                   allow_partial_ranking: bool = False
                   ) -> dict[str, Any]:
    """Database-free ``question`` + ``bundle`` -> packet dict.

    If ``ranking`` is supplied it is validated against the bundle
    (``validate_ranking_for_bundle``) and then used verbatim -- this is how an
    online-computed ``kgr`` ranking is compiled offline; otherwise the ranking
    is computed offline via ``strategy``. All obligation, scope, and
    determinism guarantees come from ``compiler.compile_packet`` unchanged."""
    from .compiler import compile_packet

    if ranking is None:
        ranking = rank_bundle(bundle, question, strategy)
    else:
        validate_ranking_for_bundle(ranking, bundle,
                                    allow_partial=allow_partial_ranking)
    return compile_packet(question, bundle, ranking, annotations=annotations)


# -- artifact loaders (exactly the persisted online-pipeline outputs) ---------

def load_question(path: str | Path) -> ResearchQuestion:
    """Load a serialized ``ResearchQuestion`` (the pipeline's question.json).

    The persisted question carries its own anchor candidates + confirmation,
    so ``compile_packet``'s ``require_confirmation`` gate is satisfiable
    offline without re-running the resolver."""
    return ResearchQuestion.from_dict(_read_json(path))


def load_bundle(path: str | Path) -> CandidateBundle:
    """Load a serialized ``CandidateBundle`` (the pipeline's candidates.json)."""
    return CandidateBundle.from_dict(_read_json(path))


def load_ranking(path: str | Path) -> list[RankedCandidate]:
    """Load a precomputed ``RankedCandidate`` list (e.g. an online kgr run)."""
    return [RankedCandidate.from_dict(r) for r in _read_json(path)]


def load_annotations(path: str | Path) -> list[AnnotationRecord]:
    """Load a sidecar ``AnnotationRecord`` list."""
    return [AnnotationRecord.from_dict(a) for a in _read_json(path)]


def _read_json(path: str | Path) -> Any:
    # utf-8-sig tolerates a leading BOM: externally-authored ranking /
    # annotation files (esp. Windows editors, PowerShell 5.1 `Set-Content
    # -Encoding utf8`) commonly carry one; the engine's own canonical bytes
    # never do, so this only ever helps.
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

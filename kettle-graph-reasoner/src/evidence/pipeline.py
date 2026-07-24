r"""Shared question->packet pipeline used by both the CLI and the HTTP
adapter (single implementation; the API is a thin transport).

Holds ONE ``EvidenceGraphSource`` per process -- the 327k-node cache pull
is amortized across requests, which is the point of ``serve``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from . import canonical
from .contracts import (
    BuildManifest,
    PacketBudget,
    QuestionScope,
    ResearchQuestion,
    TraversalProfile,
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = _ROOT / "research_workspace"
DEFAULT_DOMAIN_LABELS = ["Entity", "Person", "Place", "Organization",
                         "Event", "Activity", "Concept", "Period", "Year"]


def load_fixture(path_or_obj: str | Mapping[str, Any]) -> dict[str, Any]:
    """Accept a fixture JSON path or an inline question object with the
    same shape (text/family/scope/anchor_terms)."""
    if isinstance(path_or_obj, Mapping):
        d = dict(path_or_obj)
    else:
        d = json.loads(Path(path_or_obj).read_text("utf-8"))
    scope = d.get("scope", {})
    return {
        "text": d["text"],
        "family": d["family"],
        "scope": QuestionScope(
            year_start=scope.get("year_start"),
            year_end=scope.get("year_end"),
            places=tuple(scope.get("places", ())),
            collections=tuple(scope.get("collections", ()))),
        "anchor_terms": list(d.get("anchor_terms", ())),
        "profile": TraversalProfile(),
        "budget": PacketBudget(),
    }


class EvidencePipeline:
    """Build once, serve many. All graph access is read-only."""

    def __init__(self, workspace_root: str | Path = DEFAULT_WORKSPACE):
        from .neo4j_backend import EvidenceGraphSource, _read_session
        from .resolver import AnchorResolver
        from .store import Workspace

        self.src = EvidenceGraphSource()
        self.resolver = AnchorResolver(
            lambda: _read_session(self.src._drv),
            snapshot_epoch=self.src.snapshot_epoch,
            domain_labels=DEFAULT_DOMAIN_LABELS)
        self.workspace = Workspace(workspace_root)
        self.runtime = canonical.deterministic_runtime()

    def close(self) -> None:
        self.src.close()

    # -- steps ---------------------------------------------------------------

    def resolve(self, fixture) -> tuple:
        from .resolver import _rerank

        fx = load_fixture(fixture)
        seen, out = set(), []
        for term in fx["anchor_terms"]:
            for cand in self.resolver.resolve(term):
                if cand.public_key not in seen:
                    seen.add(cand.public_key)
                    out.append(cand)
        return fx, _rerank(out)

    def compile(self, fixture, confirm_keys, *, confirmed_by: str,
                confirmed_at: str, write: bool = True) -> dict[str, Any]:
        from .compiler import (
            COMPILER_VERSION,
            compile_packet,
            render_markdown,
        )
        from .projection import (
            attach_provenance,
            attach_term_recall,
            project_candidates,
        )
        from .ranking import bfs_ranking

        fx, cands = self.resolve(fixture)
        conf = self.resolver.make_confirmation(
            cands, list(confirm_keys),
            confirmed_by=confirmed_by, confirmed_at=confirmed_at)
        question = ResearchQuestion(
            text=fx["text"], family=fx["family"], scope=fx["scope"],
            profile=fx["profile"], budget=fx["budget"],
            candidates=cands, confirmation=conf)
        question.validate(require_confirmation=True)

        bundle = project_candidates(self.src, question)
        idxs = [self.src._id2idx[n.neo4j_id] for n in bundle.nodes]
        bundle = attach_provenance(bundle, self.src, idxs)
        # amendment A2 (2026-07-13): bounded lexical claim recall beside
        # the anchor closure -- without it ~71% of graded evidence was
        # unreachable (FINDINGS)
        bundle = attach_term_recall(bundle, self.src, fx["anchor_terms"])
        ranking = bfs_ranking(bundle)
        packet = compile_packet(question, bundle, ranking)
        markdown = render_markdown(packet)

        summary = {
            "question_id": question.question_id,
            "anchors": list(bundle.anchors),
            "domain_nodes": len(bundle.nodes),
            "closure_nodes": len(bundle.provenance_nodes),
            "core_evidence": len(packet["core_evidence"]),
            "paths": len(packet["explanatory_paths"]),
            "context": len(packet["context"]),
            "uncertainties": len(packet["uncertainties"]),
            "frontier": len(packet["research_frontier"]),
            "unique_nodes_used": packet["unique_nodes_used"],
            "overflow": packet["overflow"],
        }
        if not write:
            return {"written": False, **summary}

        files = {
            "question.json": canonical.canonical_bytes(question.to_dict()),
            "candidates.json": canonical.canonical_bytes(bundle.to_dict()),
            "packet.json": canonical.canonical_bytes(packet),
            "packet.md": markdown.encode("utf-8"),
        }
        manifest = BuildManifest(
            question_id=question.question_id, revision=1,
            candidate_hash=bundle.dependency_hash,
            snapshot_epoch=bundle.snapshot_epoch,
            model_sha="strategy:bfs_v1;encoder:none",
            schema_map_hash=canonical.hash_bytes(
                (_ROOT / "src" / "service" / "schema_map.yaml").read_bytes()),
            routing_hash=canonical.hash_bytes(
                (_ROOT / "src" / "service" / "routing.yaml").read_bytes()),
            compiler_version=COMPILER_VERSION, sidecar_hashes={},
            runtime_settings=self.runtime)
        with self.workspace.lock():
            ref = self.workspace.write_revision(manifest, files)
        return {"written": True, "packet": ref.name, "reused": ref.reused,
                "revision": ref.revision, "path": str(ref.path), **summary}

r"""T3 live smoke: EvidenceGraphSource against the real archival graph.

Opt-in (needs live Neo4j): ``py scripts/evidence_live_smoke.py``.

Deterministically picks two connected mid-degree domain nodes as anchors
(fallback typed keys), runs anchor resolution -> confirmation ->
projection -> provenance closure, and checks the T3 gate conditions on
live data: bounds respected, no non-member node admitted, fallback keys
warned, reconstruction byte-exact, repeat projection byte-identical.
Prints a JSON verdict; exit 0 only if every check passes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evidence import canonical  # noqa: E402
from src.evidence.contracts import (  # noqa: E402
    AnchorConfirmation,
    AnchorResolution,
    CandidateBundle,
    QuestionScope,
    ResearchQuestion,
)
from src.evidence.neo4j_backend import EvidenceGraphSource  # noqa: E402
from src.evidence.projection import (  # noqa: E402
    attach_provenance,
    project_candidates,
)


def pick_anchor_indices(src: EvidenceGraphSource) -> tuple[int, int]:
    """Two deterministic, adjacent, mid-degree member nodes: the
    lowest-index member with degree in [5, 40] and its first such
    neighbor (2-hop fallback)."""
    member = src._member
    for idx in range(len(member)):
        if not member[idx] or not 5 <= src.degree(idx) <= 40:
            continue
        for nb in src.neighbors(idx):
            if 5 <= src.degree(nb) <= 40:
                return idx, nb
        for nb in src.neighbors(idx):
            for nb2 in src.neighbors(nb):
                if nb2 != idx and 5 <= src.degree(nb2) <= 40:
                    return idx, nb2
    raise SystemExit("no suitable anchor pair found (unexpected)")


def main() -> int:
    t0 = time.time()
    checks: dict[str, bool] = {}
    print("[smoke] building EvidenceGraphSource (one-time 327k pull) ...")
    with EvidenceGraphSource() as src:
        print(f"[smoke] epoch={src.snapshot_epoch} "
              f"low_degree_threshold={src.low_degree_threshold()} "
              f"({time.time()-t0:.0f}s)")

        a_idx, b_idx = pick_anchor_indices(src)
        payloads = src.node_payloads([a_idx, b_idx])
        cands = tuple(
            AnchorResolution(
                public_key=p["public_key"], label=p["label"],
                display_name=str(p["properties"].get("name", p["public_key"])),
                match_method="exact_id", match_evidence="smoke pick",
                rank=i + 1, snapshot_epoch=src.snapshot_epoch,
                neo4j_id=p["neo4j_id"],
                key_is_fallback=p["key_is_fallback"])
            for i, p in enumerate(payloads))
        conf = AnchorConfirmation(
            confirmed_keys=tuple(c.public_key for c in cands),
            rejected_keys=(),
            resolution_hash=canonical.content_hash(
                [c.to_dict() for c in cands]),
            snapshot_epoch=src.snapshot_epoch,
            confirmed_by="live-smoke", confirmed_at="2026-07-11")
        question = ResearchQuestion(
            text="Live smoke: what connects the two picked anchors?",
            family="institutional_bridge", scope=QuestionScope(),
            candidates=cands, confirmation=conf)
        print(f"[smoke] anchors: "
              f"{[c.public_key for c in cands]} "
              f"(deg {src.degree(a_idx)}/{src.degree(b_idx)})")

        t1 = time.time()
        bundle = project_candidates(src, question)
        t_proj = time.time() - t1
        bundle.validate()
        checks["bundle_validates"] = True
        checks["bounded"] = len(bundle.nodes) <= \
            question.profile.max_domain_nodes
        checks["anchors_present"] = set(bundle.anchors) <= \
            {n.public_key for n in bundle.nodes}
        member_ok = all(
            (lambda ix: ix is not None and bool(src._member[ix]))(
                src._id2idx.get(n.neo4j_id))
            for n in bundle.nodes)
        checks["only_member_nodes_admitted"] = member_ok
        fallback_warned = all(
            any(n.public_key in w for w in bundle.warnings)
            for n in bundle.nodes if n.key_is_fallback)
        checks["fallback_keys_warned"] = fallback_warned

        # provenance closure over the picked domain nodes
        idxs = [src._id2idx[n.neo4j_id] for n in bundle.nodes]
        t2 = time.time()
        full = attach_provenance(bundle, src, idxs)
        t_prov = time.time() - t2
        full.validate()
        checks["provenance_validates"] = True
        doc_layer = {n.label for n in full.provenance_nodes}
        print(f"[smoke] projection: {len(bundle.nodes)} nodes / "
              f"{len(bundle.relationships)} rels / {len(bundle.paths)} paths "
              f"({t_proj:.1f}s); closure: {len(full.provenance_nodes)} nodes "
              f"/ {len(full.provenance_relationships)} rels ({t_prov:.1f}s); "
              f"layers={sorted(doc_layer)}")

        # reconstruction + repeatability
        blob = canonical.canonical_bytes(full.to_dict())
        again = CandidateBundle.from_dict(full.to_dict())
        checks["reconstruction_byte_exact"] = \
            canonical.canonical_bytes(again.to_dict()) == blob
        bundle2 = project_candidates(src, question)
        checks["repeat_projection_identical"] = \
            bundle2.to_dict() == bundle.to_dict()

    ok = all(checks.values())
    print(json.dumps({"checks": checks, "overflow": dict(full.overflow),
                      "warnings": len(full.warnings),
                      "elapsed_s": round(time.time() - t0, 1),
                      "verdict": "PASS" if ok else "FAIL"},
                     indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

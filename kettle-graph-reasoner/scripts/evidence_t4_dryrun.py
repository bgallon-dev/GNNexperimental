r"""T4 dry-run on EXPLORATORY LLM grades (live Neo4j).

Runs the pre-registered evaluation harness end-to-end using the
LLM-graded dev worksheets under runs/evidence_workspace_eval/llm_grades/.

EPISTEMIC STATUS: exploratory plumbing validation ONLY. LLM grades can
never feed a release verdict (measuring rankers against an LLM's reading
of claim text is circular); the sealed pilots are untouched; canonical
fixtures in tests/fixtures are NOT modified. Expected pre-registered
outcome at n=6: every cell insufficient_evidence -> BFS. The informative
outputs are (a) does the whole stack run, (b) the coverage question --
what fraction of graded evidence appears in each ranking's universe at
all -- which decides whether the T4 metric needs a pre-data amendment.

Output: runs/evidence_workspace_eval/t4_dryrun_llm_<date>.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DATE = "2026-07-13"
GRADES_DIR = ROOT / "runs" / "evidence_workspace_eval" / "llm_grades"
LOCAL_HOPS = 1


def main() -> int:
    offline = "--offline" in sys.argv

    from src.evidence.contracts import CandidateBundle, ResearchQuestion
    from src.evidence.evaluation import evaluate_strategy_table, lane_score
    from src.evidence.ranking import (
        bfs_ranking,
        lane_rankings,
        lexical_ranking,
    )
    from src.evidence.store import Workspace

    corpus = ROOT / "tests" / "fixtures" / "evidence_corpus" / "dev"
    if offline:
        # bundles are replay-complete: bfs/lexical lanes + coverage need
        # no live graph; the kgr lane (frozen-encoder re-encode) does
        pipe = None
        workspace = Workspace(ROOT / "research_workspace")
        embedder = None
        print("[dryrun] OFFLINE mode: kgr lane skipped (needs live Neo4j)")
    else:
        from src.evidence.pipeline import EvidencePipeline
        from src.evidence.ranking import KGREmbedder, kgr_ranking

        pipe = EvidencePipeline()
        workspace = pipe.workspace
        embedder = KGREmbedder(pipe.src)

    # map question text -> packet id via stored question.json files
    text_to_pid = {}
    for d in sorted(workspace.packets.iterdir()):
        latest = workspace.latest(d.name)
        if not latest:
            continue
        q = json.loads(workspace.read_revision_file(
            d.name, latest["revision"], "question.json"))
        text_to_pid[q["text"]] = (d.name, latest["revision"])

    def hop_locality(key: str, bundle) -> str:
        """Offline 1-hop rule over the persisted closure: a graded node
        adjacent to any anchor in the bundle's provenance/domain edges is
        local. Exact for in-closure nodes (the closure IS the replayed
        neighborhood); graded keys absent from the bundle are nonlocal by
        definition (nothing local was left out of a bundle that contains
        the full anchor closure)."""
        anchors = set(bundle.anchors)
        if key in anchors:
            return "local"
        for r in (*bundle.provenance_relationships, *bundle.relationships):
            if (r.start_key == key and r.end_key in anchors) or \
                    (r.end_key == key and r.start_key in anchors):
                return "local"
        return "nonlocal"

    cases, coverage, per_question = [], [], {}
    try:
        for fx_path in sorted(corpus.glob("q*.json")):
            fixture = json.loads(fx_path.read_text("utf-8"))
            pid, rev = text_to_pid[fixture["text"]]
            bundle = CandidateBundle.from_dict(json.loads(
                workspace.read_revision_file(pid, rev, "candidates.json")))
            question = ResearchQuestion.from_dict(json.loads(
                workspace.read_revision_file(pid, rev, "question.json")))

            if fixture.get("essential_nodes"):
                # ingested fixture labels (live-computed locality);
                # provenance in fixture['labeled_by']
                graded_nodes = [
                    {"public_key": n["public_key"],
                     "grade": int(n["grade"]),
                     "hop_locality": n.get("hop_locality") or "nonlocal",
                     "why": n.get("why", "")}
                    for n in fixture["essential_nodes"]]
            else:
                grades_path = GRADES_DIR / f"dev_{fx_path.stem}.grades.json"
                if not grades_path.exists():
                    print(f"[dryrun] no labels for {fx_path.stem}, skip")
                    continue
                grades = json.loads(grades_path.read_text("utf-8"))
                graded_nodes = [
                    {"public_key": key, "grade": int(g["grade"]),
                     "hop_locality": hop_locality(key, bundle),
                     "why": g.get("why", "")}
                    for key, g in sorted(grades.items())]

            domain = {n.public_key for n in bundle.nodes}
            closure = {n.public_key for n in bundle.provenance_nodes}
            cov = {
                "question": fx_path.stem,
                "graded": len(graded_nodes),
                "graded_in_domain_ranking": sum(
                    1 for n in graded_nodes if n["public_key"] in domain),
                "graded_in_closure": sum(
                    1 for n in graded_nodes if n["public_key"] in closure),
                "graded_absent_from_bundle": sum(
                    1 for n in graded_nodes
                    if n["public_key"] not in domain | closure),
            }
            coverage.append(cov)
            print(json.dumps(cov))

            t0 = time.time()
            # amendment A1: judge ATTACHMENT rankings (domain rankings
            # are anchor-only here; deltas were exactly 0.0)
            from src.evidence.compiler import (
                attachment_lane_rankings,
                attachment_ranking,
            )

            def att_lanes(domain_ranking):
                items = attachment_ranking(bundle, domain_ranking,
                                           question=question)
                return attachment_lane_rankings(bundle, items), items

            rankings, universes = {}, {}
            rankings["bfs"], items_bfs = att_lanes(bfs_ranking(bundle))
            rankings["lexical"], _ = att_lanes(
                lexical_ranking(bundle, question))
            universes["attachment"] = {it["evidence_key"]
                                       for it in items_bfs}
            if not offline:
                from src.evidence.ranking import kgr_ranking
                rankings["kgr"], _ = att_lanes(kgr_ranking(bundle,
                                                           embedder))
            cov["graded_in_attachment_ranking"] = sum(
                1 for n in graded_nodes
                if n["public_key"] in universes["attachment"])
            per_question[fx_path.stem] = {
                lane: {s: lane_score(r[lane], graded_nodes, lane)
                       for s, r in rankings.items()}
                for lane in ("core", "nonlocal_discovery")}
            cases.append({
                "question_id": question.question_id,
                "family": fixture["family"],
                "graded_nodes": graded_nodes,
                "rankings": rankings})
            print(json.dumps({fx_path.stem: per_question[fx_path.stem],
                              "rank_s": round(time.time() - t0, 1)}))

        candidates = ["lexical"] if offline else ["kgr", "lexical"]
        tables = {
            cand: evaluate_strategy_table(cases, candidate=cand)
            for cand in candidates}
    finally:
        if pipe is not None:
            pipe.close()

    report = {
        "date": DATE,
        "epistemic_status": "EXPLORATORY-LLM-GRADES; not release evidence",
        "offline": offline,
        "n_questions": len(cases),
        "coverage": coverage,
        "per_question_lane_scores": per_question,
        "strategy_tables": tables,
    }
    suffix = "_offline" if offline else ""
    out = GRADES_DIR.parent / f"t4_dryrun_llm_{DATE}{suffix}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(f"[dryrun] wrote {out}")
    table0 = tables[candidates[0]]
    summary = {name: c["status"] for name, c in
               table0["cells"].items() if c["n_cases"]}
    print(json.dumps({f"{candidates[0]}_cells": summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

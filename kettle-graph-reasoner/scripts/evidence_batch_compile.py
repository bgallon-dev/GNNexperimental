r"""Batch-compile every Tranche A question (engineering smoke; live Neo4j).

For each fixture: resolve anchors, auto-confirm the top entity/event
candidates AS AN ENGINEERING SMOKE (confirmed_by records this; research
use requires human confirmation), compile a packet revision, and record
the outcome. Questions with no confirmable anchors are recorded as
findings, not skipped silently. Also live-smokes the lexical and KGR
lanes on the first successful bundle.

Output: runs/evidence_workspace_eval/batch_compile_<date>.json
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
MAX_SMOKE_ANCHORS = 3


def main() -> int:
    from src.evidence.pipeline import EvidencePipeline
    from src.evidence.ranking import (
        KGREmbedder,
        kgr_ranking,
        lane_rankings,
        lexical_ranking,
    )

    corpus = ROOT / "tests" / "fixtures" / "evidence_corpus"
    out_dir = ROOT / "runs" / "evidence_workspace_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = EvidencePipeline()
    results = []
    lane_smoke = None
    try:
        for split in ("dev", "pilot"):
            for fx_path in sorted((corpus / split).glob("q*.json")):
                name = f"{split}/{fx_path.stem}"
                t0 = time.time()
                fx, cands = pipe.resolve(fx_path)
                confirmable = [c for c in cands
                               if c.match_method in ("name_exact",
                                                     "name_contains")
                               and c.label in ("Entity", "Person", "Event",
                                               "Organization", "Place")]
                if not confirmable:
                    results.append({
                        "question": name, "family": fx["family"],
                        "status": "NO_CONFIRMABLE_ANCHORS",
                        "candidates_seen": len(cands),
                        "detail": "resolver returned no entity/event "
                                  "candidates; question needs claim-"
                                  "anchoring or new anchor terms"})
                    print(json.dumps(results[-1]))
                    continue
                keys = [c.public_key for c in
                        confirmable[:MAX_SMOKE_ANCHORS]]
                try:
                    r = pipe.compile(
                        fx_path, keys,
                        confirmed_by="engineering-batch-smoke",
                        confirmed_at=DATE)
                    results.append({
                        "question": name, "family": fx["family"],
                        "status": "COMPILED",
                        "anchors": keys,
                        "packet": r["packet"], "reused": r["reused"],
                        "core_evidence": r["core_evidence"],
                        "domain_nodes": r["domain_nodes"],
                        "closure_nodes": r["closure_nodes"],
                        "unique_nodes_used": r["unique_nodes_used"],
                        "uncertainties": r["uncertainties"],
                        "elapsed_s": round(time.time() - t0, 1)})
                except Exception as ex:  # noqa: BLE001 -- record, continue
                    results.append({
                        "question": name, "family": fx["family"],
                        "status": f"FAILED:{type(ex).__name__}",
                        "anchors": keys, "detail": str(ex)[:300]})
                print(json.dumps(results[-1]))

        # lane smoke on dev/q01's bundle (lexical + frozen-encoder kgr)
        from src.evidence.contracts import CandidateBundle
        first = next(r for r in results if r["status"] == "COMPILED")
        pid, rev = first["packet"].split("@")
        bundle = CandidateBundle.from_dict(json.loads(
            pipe.workspace.read_revision_file(pid, int(rev),
                                              "candidates.json")))
        from src.evidence.contracts import ResearchQuestion
        question = ResearchQuestion.from_dict(json.loads(
            pipe.workspace.read_revision_file(pid, int(rev),
                                              "question.json")))
        t0 = time.time()
        lex = lexical_ranking(bundle, question)
        t_lex = time.time() - t0
        t0 = time.time()
        kgr = kgr_ranking(bundle, KGREmbedder(pipe.src))
        t_kgr = time.time() - t0
        lane_smoke = {
            "bundle": first["packet"],
            "lexical": {"n": len(lex), "top3": [rc.public_key
                                                for rc in lex[:3]],
                        "elapsed_s": round(t_lex, 2)},
            "kgr": {"n": len(kgr), "top3": [rc.public_key
                                            for rc in kgr[:3]],
                    "elapsed_s": round(t_kgr, 2)},
            "lanes": {k: len(v) for k, v in
                      lane_rankings(kgr).items()},
        }
        print(json.dumps(lane_smoke))
    finally:
        pipe.close()

    by_status = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    report = {"date": DATE, "by_status": by_status,
              "results": results, "lane_smoke": lane_smoke}
    out = out_dir / f"batch_compile_{DATE}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(f"[batch] wrote {out}")
    print(json.dumps(by_status))
    return 0 if by_status.get("COMPILED", 0) else 1


if __name__ == "__main__":
    sys.exit(main())

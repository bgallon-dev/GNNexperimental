r"""Test KGR as a CODEBASE tool: ball-ordering on a real code graph.

The archival capability (emb-order the BFS ball, ndcg 0.885 vs hop 0.690,
zero training) has never been measured on code. This probe runs the same
regime on tutorstructure: for each ranking case, candidates = the hop<=4
ball around the anchor symbol (what an IDE/PR tool would pull), arms =
hop-order vs emb-order vs random vs oracle, plus the de-localized
local/nonlocal split. Also prints a qualitative "related symbols" demo
(top emb-neighbors at hop>=3, real names) for the recommender use-case.

    py -m scripts.probe_code_tool
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.codegraph import cases as C
from src.codegraph.harness import TASKS
from src.codegraph.ingest import build_npz
from src.modelsv3.distance_scoring import score_from_embeddings
from src.training.metrics import ndcg_at_k

from scripts.blend_pool_experiment import _Ctx

REPO = Path("../tutorstructure_patch")
CKPT = Path("frozen/kgr-v1.0-2026-07-07/encoder_baseline")
import sys
KEEP = "--keep-edges" in sys.argv
OUT = Path("runs/probe_code_tool_keepedges" if KEEP else "runs/probe_code_tool")
BALL_HOPS = 4.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = _Ctx(REPO, CKPT, OUT, torch.device("cpu"), keep_answer_edges=KEEP)
    # row -> human name, via the same npz build (cached) + nodes.jsonl
    cg = build_npz(REPO, OUT / f"graph_{REPO.name}.npz",
                   set() if KEEP else C.collect_required_edges(REPO, TASKS))
    row_to_id = {r: i for i, r in cg.id_to_row.items()}
    names = {}
    with open(REPO / "nodes.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            names[d.get("id")] = (d.get("name")
                                  or d.get("qualified_name")
                                  or str(d.get("id")))[:60]

    def nm(row: int) -> str:
        return names.get(row_to_id.get(row), f"row{row}")

    rows = []
    for cs in ctx.cases:
        d = ctx.hops(cs.query_row)
        posset = {r for r in cs.pos_rows if r != C.ABSTAIN_ROW}
        ball = [r for r in range(ctx.n_nodes)
                if r != cs.query_row and 1 <= d[r] <= BALL_HOPS]
        pos_in = [r for r in ball if r in posset]
        if len(ball) < 5 or not pos_in or len(pos_in) == len(ball):
            continue
        rows_t = torch.tensor(ball, dtype=torch.long)
        lab = torch.tensor([1.0 if r in posset else 0.0 for r in ball])
        d_e = ctx.d_emb(cs.query_row, rows_t)
        hp = torch.from_numpy(d[ball]).float()
        seed = int.from_bytes(cs.case_id.encode()[-8:], "little")
        arms = {
            "hop_order": -hp,
            "emb_order": -d_e,
            "random": torch.from_numpy(
                np.random.default_rng(seed)
                .standard_normal(len(ball)).astype(np.float32)),
            "oracle": lab.clone(),
        }
        min_hop = min(float(d[r]) for r in pos_in)
        row = {"task": cs.task,
               "locality": "local" if min_hop <= 1 else "nonlocal"}
        for a, sc in arms.items():
            row[f"{a}"] = ndcg_at_k(sc, lab, 10)
        rows.append(row)

    arms = ("hop_order", "emb_order", "random", "oracle")

    def _tbl(title, sub):
        if not sub:
            return None
        cell = {a: sum(r[a] for r in sub) / len(sub) for a in arms}
        print(f"{title:<38} {len(sub):>5} "
              + " ".join(f"{cell[a]:>9.3f}" for a in arms))
        return cell

    report = {}
    print(f"\n=== CODE ball-rank (hop<={BALL_HOPS:.0f}) ndcg@10 ===")
    print(f"{'cell':<38} {'n':>5} " + " ".join(f"{a:>9}" for a in arms))
    for task in sorted({r["task"] for r in rows}):
        report[task] = _tbl(task, [r for r in rows if r["task"] == task])
    for loc in ("local", "nonlocal"):
        report[loc] = _tbl(f"ALL|{loc}",
                           [r for r in rows if r["locality"] == loc])
    report["ALL"] = _tbl("ALL", rows)
    (OUT / "code_tool_results.json").write_text(json.dumps(report, indent=2))

    # qualitative: related symbols (emb-near, graph-far)
    print("\n=== related-symbols demo (top emb-neighbors at hop>=3) ===")
    anchors = [cs.query_row for cs in ctx.cases[::971]][:3]
    for a in anchors:
        d = ctx.hops(a)
        far = [r for r in range(ctx.n_nodes) if d[r] >= 3]
        if not far:
            continue
        rt = torch.tensor(far, dtype=torch.long)
        sc = -ctx.d_emb(a, rt)
        top = [far[i] for i in torch.argsort(sc, descending=True)[:6]]
        print(f"  {nm(a)}:")
        for r in top:
            print(f"      {nm(r)}  (hop {d[r]:.0f})")
    print(f"\nreport: {OUT / 'code_tool_results.json'}")


if __name__ == "__main__":
    main()

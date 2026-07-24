r"""D1 — QueryToBall placement diagnostic (Docs/QTB_PLACEMENT_PLAN.md).

Where does q land relative to the positives, per corpus? Reports, per
(checkpoint, corpus, task):

    r_q          tangent radius of the query point  ||logmap0(q)||
    r_pos        tangent radius of positive nodes (label >= 0.5)
    d_q_pos      hyperbolic dist q -> positives (mean over positives)
    d_q_all      hyperbolic dist q -> all nodes (placement context)
    cos_align    cosine between logmap0(q) and the mean positive tangent
                 direction (directional placement, radius-independent)

Radial misplacement shows as r_q drifting off r_pos on real only;
directional misplacement shows as cos_align dropping on real only.

    py -m scripts.probe_qtb_radius --out runs/geometry_probes/qtb_placement/d1_radius.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv2.layers import poincare_ops as P  # noqa: E402
from src.modelsv3.eval_candidate_recall import _load  # noqa: E402

CELLS = [
    # (name, checkpoint dir, task, corpus, split)
    ("task2_tier1", "runs/v2trunk-h32-locked", 2, "src/data/corpus/tier1", "val"),
    ("task2_real", "runs/v2trunk-h32-locked", 2, "src/data/corpus/real_domain_eval_all6", "all"),
    ("task4_tier1", "runs/sweep_taskdiversity_h32/task4_seed1", 4, "src/data/corpus/tier1", "val"),
    ("task4_real", "runs/sweep_taskdiversity_h32/task4_seed1", 4, "src/data/corpus/real_domain_eval_all6", "all"),
]


def _stats(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)),
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)), "n": len(a)}


def run_cell(ckpt_dir: str, task: int, corpus: str, split: str) -> dict:
    ds = CorpusDataset(corpus_dir=corpus, split=split, split_seed=0,
                       include_tasks={task})
    ckpt = Path(ckpt_dir) / "encoder.pt"
    encoder, qenc, cfg = _load(ckpt, ckpt.parent / "summary.json", ds)
    c = getattr(encoder, "c", torch.tensor(1.0))
    cache: dict[int, torch.Tensor] = {}
    rq, rpos, dqp, dqa, cal = [], [], [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            gi, _ = ds.index[i]
            s = ds[i]
            if gi not in cache:
                out = encoder(s.x, s.edge_index, s.edge_type,
                              s.edge_descriptor,
                              node_descriptor=s.node_descriptor)
                cache[gi] = out.node_embeddings.detach()
            emb = cache[gi]
            q = qenc(s.query)
            if q.dim() == 2:
                q = q.squeeze(0)
            pos = (s.labels >= 0.5).nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue
            q_tan = P.logmap0(q.unsqueeze(0), c).squeeze(0)
            pos_tan = P.logmap0(emb[pos], c)
            rq.append(float(q_tan.norm()))
            rpos.append(float(pos_tan.norm(dim=-1).mean()))
            d_all = P.dist(emb, q.unsqueeze(0).expand_as(emb), c,
                           keepdim=False)
            dqp.append(float(d_all[pos].mean()))
            dqa.append(float(d_all.mean()))
            mean_dir = pos_tan.mean(dim=0)
            denom = q_tan.norm() * mean_dir.norm()
            cal.append(float((q_tan @ mean_dir) / denom.clamp_min(1e-12)))
    return {"n_samples": len(rq), "r_q": _stats(rq), "r_pos": _stats(rpos),
            "d_q_pos": _stats(dqp), "d_q_all": _stats(dqa),
            "cos_align": _stats(cal)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str,
                    default="runs/geometry_probes/qtb_placement/d1_radius.json")
    args = ap.parse_args()
    results = {}
    for name, ckpt, task, corpus, split in CELLS:
        results[name] = run_cell(ckpt, task, corpus, split)
        r = results[name]
        print(f"{name:<13} n={r['n_samples']:<5} "
              f"r_q={r['r_q']['mean']:.3f}±{r['r_q']['std']:.3f}  "
              f"r_pos={r['r_pos']['mean']:.3f}±{r['r_pos']['std']:.3f}  "
              f"d(q,pos)={r['d_q_pos']['mean']:.3f}  "
              f"d(q,all)={r['d_q_all']['mean']:.3f}  "
              f"cos_align={r['cos_align']['mean']:+.3f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

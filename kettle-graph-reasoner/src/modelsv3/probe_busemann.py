r"""P2 — Busemann (horosphere) scoring-family probe.

Docs/GEOMETRY_READOUT_PROBES_PLAN.md. The c' diagnosis showed
``-dist``-to-a-point cannot express extended answer sets in-distribution,
and the bilinear fix was refuted in transfer (d^2 params, generalization
liability). The Busemann function toward an ideal point xi is the
d-parameter middle rung: horosphere level sets ('everything downstream in
direction xi'), no trained weights. This probe is fit-or-zero-training by
construction — fitting happens ONLY on the train split, per task, and is
a grid pick over one scalar (plus an optional task-level direction), in
the mixture_offsets fit-don't-train pattern.

Scoring families evaluated side by side on identical cases:

    point       -dist(q, x)                       (deployed reference)
    bus_qdir    -B_xi(x),  xi = q/||q||           (zero training)
    mix         -(dist + beta * B_xi),  beta      (1 scalar / task, fit
                grid-picked on the TRAIN split)    on train, frozen, then
                                                   applied to val/test)
    bus_taskxi  -B_xi(x),  xi = normalized mean    (task-level direction,
                tangent of train positives)        fit on train)

Usage
-----
    py -m src.modelsv3.probe_busemann \
        --checkpoint runs/sweep_taskdiversity_h32/task4_seed0/encoder.pt \
        --task 4 --out runs/geometry_probes/p2_busemann/task4_seed0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv3.distance_scoring import score_from_embeddings  # noqa: E402
from src.modelsv3.eval_candidate_recall import _load  # noqa: E402
from src.modelsv3.geometry_readout import (  # noqa: E402
    busemann,
    ideal_point_from_query,
)
from src.modelsv2.layers import poincare_ops as P  # noqa: E402
from src.training.metrics import ndcg_at_k  # noqa: E402

BETA_GRID = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
K_EVAL = 10


# ---------------------------------------------------------------------------
# per-sample family scores
# ---------------------------------------------------------------------------

def _family_scores(emb: torch.Tensor, q_point: torch.Tensor,
                   c, euclidean: bool,
                   task_xi: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """All family score vectors for one case. ``mix:<beta>`` variants are
    included for every grid beta; the caller picks per split."""
    point = score_from_embeddings(emb, q_point, c=c, euclidean=euclidean)
    out = {"point": point}
    if euclidean:
        return out  # Busemann is the hyperbolic family; no euclidean arm
    qp = q_point.squeeze(0) if q_point.dim() == 2 else q_point
    xi = ideal_point_from_query(qp)
    b = busemann(xi, emb, c)
    out["bus_qdir"] = -b
    for beta in BETA_GRID:
        out[f"mix:{beta}"] = point - beta * b
    if task_xi is not None:
        out["bus_taskxi"] = -busemann(task_xi, emb, c)
    return out


def _fit_task_xi(dataset: CorpusDataset, encoder, emb_cache: dict,
                 c) -> torch.Tensor | None:
    """Task-level ideal direction: normalized mean tangent-at-origin
    vector of train positives (labels >= 0.5). Fit-only, no gradients."""
    acc = None
    n = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            gi, _ = dataset.index[i]
            s = dataset[i]
            emb = _embed(encoder, s, gi, emb_cache)
            pos = (s.labels >= 0.5).nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue
            t = P.logmap0(emb[pos], c).mean(dim=0)
            acc = t if acc is None else acc + t
            n += 1
    if acc is None or float(acc.norm()) < 1e-9:
        return None
    return acc / acc.norm()


def _embed(encoder, s, gi: int, cache: dict) -> torch.Tensor:
    if gi not in cache:
        out = encoder(s.x, s.edge_index, s.edge_type, s.edge_descriptor,
                      node_descriptor=s.node_descriptor)
        cache[gi] = out.node_embeddings.detach()
    return cache[gi]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _eval_split(dataset: CorpusDataset, encoder, qenc, c, euclidean: bool,
                task_xi: torch.Tensor | None,
                emb_cache: dict) -> dict[str, list[tuple[int, float]]]:
    """Per-family list of (task_type, ndcg@10) over the split."""
    per_family: dict[str, list[tuple[int, float]]] = {}
    with torch.no_grad():
        for i in range(len(dataset)):
            gi, _ = dataset.index[i]
            s = dataset[i]
            emb = _embed(encoder, s, gi, emb_cache)
            fam = _family_scores(emb, qenc(s.query), c, euclidean, task_xi)
            for name, scores in fam.items():
                per_family.setdefault(name, []).append(
                    (int(s.task_type), ndcg_at_k(scores, s.labels, K_EVAL)))
    return per_family


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def run_probe(checkpoint: Path, summary: Path, corpus_dir: str,
              task: int | None, split_seed: int, out_path: Path) -> dict:
    include = {task} if task is not None else None
    train_ds = CorpusDataset(corpus_dir=corpus_dir, split="train",
                             split_seed=split_seed, include_tasks=include)
    val_ds = CorpusDataset(corpus_dir=corpus_dir, split="val",
                           split_seed=split_seed, include_tasks=include)
    encoder, qenc, cfg = _load(checkpoint, summary, val_ds)
    euclidean = cfg["model"] == "euclidean"
    c_val = getattr(encoder, "c",
                    torch.tensor(float(cfg.get("curvature", 1.0))))
    # graph_idx is split-relative -> one cache per dataset, never shared
    cache_train: dict[int, torch.Tensor] = {}
    cache_val: dict[int, torch.Tensor] = {}

    task_xi = None if euclidean else _fit_task_xi(
        train_ds, encoder, cache_train, c_val)

    train_fam = _eval_split(train_ds, encoder, qenc, c_val, euclidean,
                            task_xi, cache_train)
    val_fam = _eval_split(val_ds, encoder, qenc, c_val, euclidean,
                          task_xi, cache_val)

    # Fit beta on TRAIN only: argmax train ndcg@10 over the grid.
    fitted_beta = None
    if not euclidean:
        beta_means = {beta: _mean([v for _, v in train_fam[f"mix:{beta}"]])
                      for beta in BETA_GRID}
        fitted_beta = max(beta_means, key=beta_means.get)

    def summarize(fam: dict[str, list[tuple[int, float]]]) -> dict:
        out: dict[str, dict] = {}
        names = ["point", "bus_qdir", "bus_taskxi"]
        if fitted_beta is not None:
            names.append(f"mix:{fitted_beta}")
        for name in names:
            if name not in fam:
                continue
            rows = fam[name]
            by_task: dict[str, float] = {}
            for t in sorted({t for t, _ in rows}):
                by_task[str(t)] = _mean([v for tt, v in rows if tt == t])
            key = "mix_fit" if name.startswith("mix:") else name
            out[key] = {"overall": _mean([v for _, v in rows]),
                        "by_task": by_task, "n": len(rows)}
        return out

    results = {
        "checkpoint": str(checkpoint),
        "model_kind": cfg["model"],
        "query_head_arch": cfg.get("query_head_arch", "qh0"),
        "corpus": corpus_dir,
        "task": task,
        "k_eval": K_EVAL,
        "beta_grid": list(BETA_GRID),
        "fitted_beta": fitted_beta,
        "task_xi_fit": task_xi is not None,
        "train": summarize(train_fam),
        "val": summarize(val_fam),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    _print(results)
    return results


def _print(r: dict) -> None:
    print("=" * 72)
    print(f"P2 Busemann probe  task={r['task']}  ckpt={r['checkpoint']}")
    print(f"fitted beta (train) = {r['fitted_beta']}")
    for split in ("train", "val"):
        print(f"-- {split} (ndcg@{r['k_eval']}) --")
        for fam, d in r[split].items():
            print(f"  {fam:<10} overall={d['overall']:.4f}  "
                  + "  ".join(f"t{t}={v:.4f}" for t, v in d["by_task"].items()))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--summary", type=str, default=None)
    p.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    p.add_argument("--task", type=int, default=4, help="-1 for all tasks")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()
    ckpt = Path(args.checkpoint)
    summary = Path(args.summary) if args.summary else ckpt.parent / "summary.json"
    run_probe(ckpt, summary, args.corpus,
              None if args.task < 0 else int(args.task),
              args.split_seed, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

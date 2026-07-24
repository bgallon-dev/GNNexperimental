"""
Option-2 test — does a general BILINEAR scoring head remove the c' wall?

Established: the deployed family score = -dist(query_point, node_emb) (dist
to ONE ball point) structurally cannot express interval-overlap relevance
(P2: best oracle-overfit q* = 0.656, below anchor-BFS 0.714; oracle 0.988;
a temporal-aware head on the SAME frozen emb = 0.760).

Bilinear score = query^T M node_emb is STRICTLY more expressive than
-dist-to-a-point (the latter is a constrained special case), so it should
remove c' *if the family is the wall*. The ONLY changed variable vs P2 is
the score function: same frozen locked encoder, same hardened_250 task-2
split, same pairwise-hinge objective semantics (replicated faithfully from
ranking.pairwise_ranking_loss — pos>=0.75 / neg<=0.25, top/bottom-10%
fallback, n_pairs sampled w/ replacement, relu(margin - (s_pos-s_neg))),
same ndcg@10. Reference scorers reuse P2's exact helpers.

Two regimes (separates c' from c, like P2):
  - CEILING : per-task overfit M to that task's oracle labels
              -> the NEW family's ceiling. vs best-q* 0.656 = is c' removed?
  - LEARNED : one M trained on TRAIN-graph task-2 signal, eval held-out
              -> achievable. vs trained-head ~0.47 / option-1 0.760 = does
              the objective (c) FIND the solution?
Plus a cheap orthogonal diagnostic: CEILING with the raw 9-D npz query
(contains the window q[6:8] explicitly) vs the deployed 18-D query --
attributes any failure to the family vs the query representation.

Pre-registered decision tree:
  ceil_18d > best_q*+0.05            -> c' REMOVED by the bilinear family
  ceil_18d ~ best_q* & ceil_raw >>   -> not the family; 18-D query transform
                                        drops the window (representation fix)
  ceil_18d ~ best_q* & ceil_raw ~    -> deeper than scoring family (revisit)
  learned ~ ceiling                  -> objective (c) fine; head-class swap
                                        alone suffices
  learned << ceiling                 -> c' removed but objective doesn't find
                                        it; need head-class AND objective work
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))       # scripts/

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv3.eval_retrieval_nn import _load_encoder  # noqa: E402
from src.training.metrics import ndcg_at_k  # noqa: E402
from probe_scoring_ceiling import (  # noqa: E402  (sibling module)
    _bfs_dist, _best_qstar, _fit_temporal_mlp, TS, TE)


def _pair_hinge(scores, labels, margin=0.5, n_pairs=64,
                pos_th=0.75, neg_th=0.25, rng=None):
    """Faithful replica of ranking.pairwise_ranking_loss semantics with a
    generic score (higher = more relevant): d:=-score, so
    relu(margin + d_pos - d_neg) == relu(margin - (s_pos - s_neg))."""
    labels = labels.clamp(0.0, 1.0)
    N = labels.numel()
    pos = torch.nonzero(labels >= pos_th, as_tuple=False).flatten()
    neg = torch.nonzero(labels <= neg_th, as_tuple=False).flatten()
    if pos.numel() == 0 or neg.numel() == 0:
        sv, si = torch.sort(labels, descending=True)
        k = max(1, N // 10)
        if pos.numel() == 0:
            pos = si[:k]
        if neg.numel() == 0:
            neg = si[-k:]
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.new_zeros(())
    ps = pos[torch.randint(0, pos.numel(), (n_pairs,), generator=rng)]
    ns = neg[torch.randint(0, neg.numel(), (n_pairs,), generator=rng)]
    return torch.relu(
        margin - (scores.index_select(0, ps) - scores.index_select(0, ns))
    ).mean()


class Bilinear(torch.nn.Module):
    """score(q, n) = q^T M n  (the general bilinear form; -dist-to-a-point
    is a strict special case). qdim ~ 18, H ~ 128 -> M tiny."""

    def __init__(self, qdim, hdim):
        super().__init__()
        self.M = torch.nn.Parameter(torch.zeros(qdim, hdim))
        torch.nn.init.xavier_uniform_(self.M, gain=0.1)

    def forward(self, q, node_emb):           # q:(Q,) node_emb:(N,H)
        return node_emb @ (self.M.t() @ q)    # (N,)


def _fit_ceiling(q, emb, lab, qdim, hdim, steps=300, seed=0):
    """Per-task overfit M to this task's oracle labels (upper bound)."""
    torch.manual_seed(seed)
    head = Bilinear(qdim, hdim)
    opt = torch.optim.Adam(head.parameters(), lr=5e-2, weight_decay=0.0)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        opt.zero_grad()
        s = head(q, emb)
        loss = _pair_hinge(s, lab, rng=g)
        if not torch.isfinite(loss) or loss.item() == 0.0:
            break
        loss.backward()
        opt.step()
    with torch.no_grad():
        return head(q, emb).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus",
                    default="src/data/corpus/real_domain_eval_hardened_250")
    ap.add_argument("--checkpoint",
                    default="runs/v3.1-baseline-hyp-h128-l4-seed1/encoder.pt")
    ap.add_argument("--task", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    corpus = str((root / a.corpus).resolve())
    ckpt = (root / a.checkpoint).resolve()

    ds = CorpusDataset(corpus_dir=corpus, split="all", split_seed=0,
                       include_tasks={a.task})
    model, cfg = _load_encoder(ckpt, ckpt.parent / "summary.json", ds)
    c = getattr(model, "c", torch.tensor(float(cfg["curvature"])))
    print(f"[opt2] encoder {cfg['model']} h{cfg['hidden_dim']}/"
          f"l{cfg['num_layers']} c={float(c):.3f} | {Path(corpus).name} "
          f"| {len(ds)} task-{a.task} samples")

    rng = np.random.default_rng(a.seed)
    gids = sorted({ds.index[i][0] for i in range(len(ds))})
    rng.shuffle(gids)
    n_te = max(2, len(gids) // 5)
    test_g = set(gids[:n_te])
    emb_cache: dict[int, np.ndarray] = {}
    tasks = []
    with torch.no_grad():
        for i in range(len(ds)):
            gi, j = ds.index[i]
            s = ds[i]
            if gi not in emb_cache:
                out = model(s.x, s.edge_index, s.edge_type,
                            s.edge_descriptor,
                            node_descriptor=s.node_descriptor)
                emb_cache[gi] = out.node_embeddings.numpy().astype(np.float32)
            npz = np.load(ds.files[gi])
            q9 = npz[f"task_{j}_query"].astype(np.float32)
            x = npz["x"].astype(np.float32)
            tasks.append(dict(
                gi=gi, emb=emb_cache[gi],
                lab=npz[f"task_{j}_labels"].astype(np.float32),
                ts=x[:, TS], te=x[:, TE], ws=float(q9[6]), we=float(q9[7]),
                anc=int(npz[f"task_{j}_anchor_row"]),
                ei=npz["edge_index"].astype(np.int64), n=x.shape[0],
                q9=q9, q18=s.query.numpy().astype(np.float32)))
            npz.close()

    qdim18 = tasks[0]["q18"].shape[0]
    qdim9 = tasks[0]["q9"].shape[0]
    H = tasks[0]["emb"].shape[1]

    # LEARNED bilinear: one M trained on TRAIN-graph task-2 signal (18-D q)
    torch.manual_seed(a.seed)
    learned = Bilinear(qdim18, H)
    optL = torch.optim.Adam(learned.parameters(), lr=1e-2, weight_decay=1e-5)
    gL = torch.Generator().manual_seed(a.seed)
    tr_tasks = [t for t in tasks if t["gi"] not in test_g]
    for ep in range(40):
        order = np.random.default_rng(a.seed + ep).permutation(len(tr_tasks))
        for k in order:
            t = tr_tasks[k]
            optL.zero_grad()
            s = learned(torch.tensor(t["q18"]), torch.tensor(t["emb"]))
            loss = _pair_hinge(s, torch.tensor(t["lab"]), rng=gL)
            if torch.isfinite(loss) and loss.item() > 0:
                loss.backward()
                optL.step()

    # probe #4 (option-1 reference): MLP node_emb->(ts,te) on train graphs
    Xtr = np.concatenate([t["emb"] for t in tr_tasks])
    Ytr = np.concatenate([np.stack([t["ts"], t["te"]], 1)
                          for t in tr_tasks])
    net, mu, sd = _fit_temporal_mlp(Xtr, Ytr, seed=a.seed)

    R = {k: [] for k in ("oracle", "anchor", "best_q*", "opt1_probed",
                          "bilin_ceil_18d", "bilin_ceil_raw",
                          "bilin_learned")}
    used = 0
    for t in tasks:
        if t["gi"] not in test_g:
            continue
        L = torch.tensor(t["lab"])
        if (L >= 0.75).sum() == 0 or (L <= 0.25).sum() == 0:
            continue
        used += 1
        emb = t["emb"]; et = torch.tensor(emb)
        ov = np.maximum(0.0, np.minimum(t["te"], t["we"])
                        - np.maximum(t["ts"], t["ws"]))
        R["oracle"].append(ndcg_at_k(torch.tensor(ov), L, 10))
        R["anchor"].append(ndcg_at_k(
            torch.tensor(_bfs_dist(t["n"], t["ei"], t["anc"])), L, 10))
        R["best_q*"].append(ndcg_at_k(
            torch.tensor(_best_qstar(et, L, c)), L, 10))
        with torch.no_grad():
            pr = net(torch.tensor((emb - mu) / sd,
                                  dtype=torch.float32)).numpy()
        ovp = np.maximum(0.0, np.minimum(pr[:, 1], t["we"])
                         - np.maximum(pr[:, 0], t["ws"]))
        R["opt1_probed"].append(ndcg_at_k(torch.tensor(ovp), L, 10))
        R["bilin_ceil_18d"].append(ndcg_at_k(torch.tensor(_fit_ceiling(
            torch.tensor(t["q18"]), et, L, qdim18, H, seed=a.seed)), L, 10))
        R["bilin_ceil_raw"].append(ndcg_at_k(torch.tensor(_fit_ceiling(
            torch.tensor(t["q9"]), et, L, qdim9, H, seed=a.seed)), L, 10))
        with torch.no_grad():
            sl = learned(torch.tensor(t["q18"]), et).numpy()
        R["bilin_learned"].append(ndcg_at_k(torch.tensor(sl), L, 10))

    m = {k: float(np.mean(v)) for k, v in R.items()}
    print(f"\n[opt2] mean ndcg@10 over {used} held-out task-{a.task} "
          f"graphs (frozen locked encoder; only the SCORE fn changes):")
    print(f"  oracle (true overlap)            : {m['oracle']:.4f}  ceiling")
    print(f"  anchor-BFS                       : {m['anchor']:.4f}  heuristic")
    print(f"  best q*  (OLD -dist family ceil) : {m['best_q*']:.4f}")
    print(f"  option-1 overlap-on-probed-emb   : {m['opt1_probed']:.4f}  "
          f"[target]")
    print(f"  BILINEAR ceiling (18-D query)    : {m['bilin_ceil_18d']:.4f}  "
          f"<< the c' test")
    print(f"  BILINEAR ceiling (raw 9-D query) : {m['bilin_ceil_raw']:.4f}  "
          f"[query-repr control]")
    print(f"  BILINEAR learned (train->test)   : {m['bilin_learned']:.4f}  "
          f"vs trained-head ~0.47")

    bq = m["best_q*"]; c18 = m["bilin_ceil_18d"]
    craw = m["bilin_ceil_raw"]; lrn = m["bilin_learned"]
    if c18 > bq + 0.05:
        head = (f"c' REMOVED by the bilinear family: ceiling {c18:.3f} > "
                f"old-family ceiling {bq:.3f} (+{c18-bq:.3f}).")
    elif craw > bq + 0.05:
        head = (f"family OK but 18-D query transform DROPS the window: "
                f"18-D ceil {c18:.3f} ~ old {bq:.3f}, raw-9D ceil "
                f"{craw:.3f}. Fix = query representation, not just head.")
    else:
        head = (f"bilinear ceiling {c18:.3f} ~ old {bq:.3f}: bottleneck is "
                f"DEEPER than the scoring family (revisit).")
    if c18 > bq + 0.05:
        if lrn >= c18 - 0.05:
            obj = (f" Objective (c) FINE: learned {lrn:.3f} ~ ceiling "
                   f"{c18:.3f} -> head-class swap alone suffices.")
        elif lrn > bq + 0.05:
            obj = (f" Objective (c) PARTIAL: learned {lrn:.3f} beats old "
                   f"{bq:.3f} but trails ceiling {c18:.3f} -> head-class "
                   f"helps, objective work recovers the rest.")
        else:
            obj = (f" Objective (c) is the live wall: learned {lrn:.3f} ~ "
                   f"old family despite the expressive head -> need "
                   f"head-class AND objective work (honest-prior risk).")
    else:
        obj = ""
    print(f"\n[opt2] VERDICT: {head}{obj}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

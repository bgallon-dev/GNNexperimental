"""
Step 2 — does the Stage-B OBJECTIVE recover the 0.69 -> 0.76 gap?

Option-2 result: the bilinear head's ceiling is 0.759 but trained under the
faithful thresholded pairwise-hinge it reaches only 0.688. That 0.07 gap was
ATTRIBUTED to the objective: the hinge binarizes labels (pos>=0.75 /
neg<=0.25), throwing away the graded relevance and all "middle" nodes, which
is exactly the fine-ranking signal ndcg@10 rewards for continuous
interval-overlap relevance.

This tests that attribution directly. ONLY the training objective changes;
head class (bilinear q^T M n), frozen locked encoder, hardened_250 task-2
split, seed, optimizer schedule (40 epochs, lr 1e-2) and eval are IDENTICAL
to the option-2 learned run -- so the HINGE arm must reproduce ~0.688 as the
control (proves the only changed variable is the loss).

Objectives (faithful replicas of ranking.py semantics on the generic
bilinear score s = q^T M n, higher = more relevant):
  - hinge    : relu(margin - (s_pos - s_neg)) over sampled thresholded
               pairs  == pairwise_ranking_loss.  [CONTROL ~0.688]
  - listwise : p = labels/sum(labels) over ALL nodes (graded, no
               thresholding); loss = -(p * log_softmax(s/T)).sum()
               == listwise_ranking_loss.  [the lever under test]

Pre-registered decision:
  listwise >= ceiling-0.02 (~0.74+)  -> OBJECTIVE is the lever; gap
        recovered; fix for the 0.69->0.76 segment = listwise loss.
  hinge+0.03 < listwise < ceiling-0.02 -> partial lever; quantify residual.
  listwise ~ hinge (~0.688)          -> objective form NOT the lever;
        the +0.07 attribution was wrong (honest negative) -> gap is
        optimization/capacity, revisit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))        # scripts/

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv3.eval_retrieval_nn import _load_encoder  # noqa: E402
from src.training.metrics import ndcg_at_k  # noqa: E402
from probe_scoring_ceiling import (  # noqa: E402
    _bfs_dist, _best_qstar, _fit_temporal_mlp, TS, TE)
from probe_bilinear_head import Bilinear, _pair_hinge, _fit_ceiling  # noqa: E402


def _listwise(scores, labels, temperature=1.0, eps=1e-9):
    """Faithful replica of ranking.listwise_ranking_loss with s := -dist
    (s is the bilinear score, higher = more relevant -> logits = s/T)."""
    labels = labels.clamp(0.0, 1.0)
    total = labels.sum()
    if total < eps:
        return scores.new_zeros(())
    p = labels / total
    logits = scores / float(temperature)
    log_q = logits - torch.logsumexp(logits, dim=0)
    return -(p * log_q).sum()


def _train_bilinear(tr_tasks, qdim, H, objective, seed, epochs=40,
                    lr=1e-2, temperature=1.0):
    """IDENTICAL schedule to the option-2 learned run; only `objective`
    (and, for listwise, its known-critical `temperature`) differs. The
    hinge keeps its project-default margin=0.5; fairness = the alternative
    gets its analogous critical knob tuned, not crippled at an arbitrary 1.0
    (v3 CLAUDE.md explicitly prescribes sweeping listwise temperature)."""
    torch.manual_seed(seed)
    head = Bilinear(qdim, H)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-5)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        order = np.random.default_rng(seed + ep).permutation(len(tr_tasks))
        for k in order:
            t = tr_tasks[k]
            opt.zero_grad()
            s = head(torch.tensor(t["q18"]), torch.tensor(t["emb"]))
            lab = torch.tensor(t["lab"])
            if objective == "hinge":
                loss = _pair_hinge(s, lab, rng=g)
            elif objective == "listwise":
                loss = _listwise(s, lab, temperature=temperature)
            else:
                raise ValueError(objective)
            if torch.isfinite(loss) and loss.item() > 0:
                loss.backward()
                opt.step()
    return head


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
    print(f"[step2] encoder {cfg['model']} h{cfg['hidden_dim']}/"
          f"l{cfg['num_layers']} | {Path(corpus).name} | {len(ds)} "
          f"task-{a.task} samples")

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
                q18=s.query.numpy().astype(np.float32)))
            npz.close()

    qdim = tasks[0]["q18"].shape[0]
    H = tasks[0]["emb"].shape[1]
    tr_tasks = [t for t in tasks if t["gi"] not in test_g]

    h_hinge = _train_bilinear(tr_tasks, qdim, H, "hinge", a.seed)
    # listwise: sweep its known-critical temperature (v3 CLAUDE.md), keep
    # the BEST -- a fair test of the objective, not of an untuned knob.
    temps = [0.25, 0.5, 1.0, 2.0, 5.0]
    list_heads = {T: _train_bilinear(tr_tasks, qdim, H, "listwise",
                                     a.seed, temperature=T) for T in temps}

    Xtr = np.concatenate([t["emb"] for t in tr_tasks])
    Ytr = np.concatenate([np.stack([t["ts"], t["te"]], 1)
                          for t in tr_tasks])
    net, mu, sd = _fit_temporal_mlp(Xtr, Ytr, seed=a.seed)

    base = ("oracle", "anchor", "best_q*", "opt1", "bilin_ceiling",
            "learn_hinge")
    R = {k: [] for k in base}
    RL = {T: [] for T in temps}          # per-temperature listwise
    used = 0
    for t in tasks:
        if t["gi"] not in test_g:
            continue
        L = torch.tensor(t["lab"])
        if (L >= 0.75).sum() == 0 or (L <= 0.25).sum() == 0:
            continue
        used += 1
        emb = t["emb"]; et = torch.tensor(emb); q = torch.tensor(t["q18"])
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
        R["opt1"].append(ndcg_at_k(torch.tensor(ovp), L, 10))
        R["bilin_ceiling"].append(ndcg_at_k(torch.tensor(_fit_ceiling(
            q, et, L, qdim, H, seed=a.seed)), L, 10))
        with torch.no_grad():
            R["learn_hinge"].append(ndcg_at_k(h_hinge(q, et), L, 10))
            for T in temps:
                RL[T].append(ndcg_at_k(list_heads[T](q, et), L, 10))

    m = {k: float(np.mean(v)) for k, v in R.items()}
    mL = {T: float(np.mean(v)) for T, v in RL.items()}
    bestT = max(mL, key=mL.get)
    m["learn_listwise"] = mL[bestT]
    print(f"\n[step2] mean ndcg@10 over {used} held-out task-{a.task} "
          f"graphs (frozen encoder; bilinear head; ONLY objective varies):")
    print(f"  oracle                         : {m['oracle']:.4f}")
    print(f"  anchor-BFS                     : {m['anchor']:.4f}")
    print(f"  old -dist family ceiling       : {m['best_q*']:.4f}")
    print(f"  option-1 (hand formula)        : {m['opt1']:.4f}")
    print(f"  bilinear CEILING (overfit)     : {m['bilin_ceiling']:.4f}  "
          f"[target]")
    print(f"  bilinear learned, HINGE        : {m['learn_hinge']:.4f}  "
          f"[control ~0.688]")
    print(f"  listwise by temperature        : "
          + "  ".join(f"T{T}={mL[T]:.3f}" for T in temps))
    print(f"  bilinear learned, LISTWISE*    : {m['learn_listwise']:.4f}  "
          f"<< the test (best T={bestT})")

    hi, li, ce = m["learn_hinge"], m["learn_listwise"], m["bilin_ceiling"]
    print(f"\n[step2] listwise-hinge = {li-hi:+.3f}   "
          f"ceiling-listwise = {ce-li:+.3f}")
    if li >= ce - 0.02:
        v = (f"OBJECTIVE IS THE LEVER: listwise {li:.3f} reaches the head "
             f"ceiling {ce:.3f}. The 0.69->0.76 segment recovers by "
             f"swapping the thresholded hinge for the graded listwise loss. "
             f"Confirms the attribution.")
    elif li > hi + 0.03:
        v = (f"PARTIAL: listwise {li:.3f} beats hinge {hi:.3f} "
             f"(+{li-hi:.3f}) but trails ceiling {ce:.3f} "
             f"(residual {ce-li:.3f}). Objective is A lever, not the whole "
             f"segment; some gap is optimization/capacity.")
    else:
        v = (f"HONEST NEGATIVE: listwise {li:.3f} ~ hinge {hi:.3f}. The "
             f"objective FORM is not the 0.69->0.76 lever; the +0.07 "
             f"attribution was wrong -- that gap is optimization/capacity, "
             f"not loss shape. Revisit before investing in a loss change.")
    print(f"[step2] VERDICT: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

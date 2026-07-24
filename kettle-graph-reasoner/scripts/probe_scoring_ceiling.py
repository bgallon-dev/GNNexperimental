"""
P2 — scoring-family ceiling (Stage-B bottleneck investigation).

P1 established: node temporal scope SURVIVES the frozen encoder (MLP R2~0.90).
So the head HAS the signal. Remaining fork:
  (c)  the ranking objective doesn't pressure the head to use it, OR
  (c') the deployed scoring family -- score = -dist(query_point, node_emb)
       to a SINGLE ball point -- structurally cannot express interval
       overlap, regardless of objective or head capacity.

Decisive test, all ndcg@10 on the SAME held-out task-2 graphs
(hardened_250, the corpus the 0.475/0.93 numbers came from), frozen locked
encoder:

  1. oracle             true overlap(min(te,we)-max(ts,ws))  -> ~0.93 sanity
  2. anchor-BFS         -graph_dist from task anchor          -> ~0.65 sanity
  3. best single q*     per-task OVERFIT a ball point to the oracle labels
                        via the project's OWN pairwise_ranking_loss, score
                        -dist(q*, node_emb). Upper bound on ANY learnable
                        QueryToBall in the deployed family. (tests c')
  4. overlap-on-probed  MLP probe node_emb->(ts,te) [train graphs], then
                        the oracle FORMULA on predictions. "Right head class,
                        SAME frozen encoder" achievable number. (tests c)

Reading:
  3 << oracle  (~anchor or below)  => c' CONFIRMED: dist-to-a-point cannot
       express the task even with a perfect oracle-fit point + temporally
       rich embeddings. Fix = scoring HEAD CLASS, not objective tuning.
  4 ~ oracle  &  3 << 4            => constructive proof the fix is an
       interval/temporal-aware head on the SAME encoder; quantifies headroom.
  3 ~ oracle                       => family fine; bottleneck is purely the
       learned objective/query-map (c).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv3.eval_retrieval_nn import _load_encoder  # noqa: E402
from src.modelsv3.ranking import pairwise_ranking_loss  # noqa: E402
from src.training.metrics import ndcg_at_k  # noqa: E402
from src.modelsv2.layers import poincare_ops as P  # noqa: E402

TS, TE = 21, 22


def _bfs_dist(n, ei, anchor):
    adj = [[] for _ in range(n)]
    for a, b in zip(ei[0], ei[1]):
        adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    d = np.full(n, -1, np.int64); d[anchor] = 0; fr = [anchor]
    while fr:
        nx = []
        for u in fr:
            for v in adj[u]:
                if d[v] < 0:
                    d[v] = d[u] + 1; nx.append(v)
        fr = nx
    far = int(d.max()) + 1 if (d >= 0).any() else 1
    return -np.where(d < 0, far, d).astype(np.float32)


def _best_qstar(node_emb_t, labels_t, c, steps=250):
    """Per-task UPPER BOUND for the -dist(q, .) family: overfit a ball
    point to this task's oracle labels with the project's own Stage-B
    pairwise loss."""
    H = node_emb_t.shape[1]
    v = torch.zeros(H, requires_grad=True)            # 1-D: loss expands it
    opt = torch.optim.Adam([v], lr=5e-2)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        opt.zero_grad()
        q = P.expmap0(v, c)
        loss, _ = pairwise_ranking_loss(
            q, node_emb_t, labels_t, c=c, margin=0.5, n_pairs=64,
            pos_threshold=0.75, neg_threshold=0.25, rng=g)
        if not torch.isfinite(loss) or loss.item() == 0.0:
            break
        loss.backward(); opt.step()
    with torch.no_grad():
        q = P.expmap0(v, c)
        qd = P.dist(q.unsqueeze(0).expand(node_emb_t.shape[0], -1),
                    node_emb_t, c)
        return (-qd).numpy()


def _fit_temporal_mlp(Xtr, Ytr, seed=0, steps=800):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 256),
                              torch.nn.GELU(), torch.nn.Linear(256, 2))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(Ytr, dtype=torch.float32)
    for _ in range(steps):
        opt.zero_grad()
        ((net(xt) - yt) ** 2).mean().backward(); opt.step()
    return net, mu, sd


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
    print(f"[P2] encoder {cfg['model']} h{cfg['hidden_dim']}/"
          f"l{cfg['num_layers']} c={float(c):.3f} | "
          f"{Path(corpus).name} | {len(ds)} task-{a.task} samples")

    # collect per-task tensors + frozen embeddings (encoder unconditional:
    # cache emb per graph)
    rng = np.random.default_rng(a.seed)
    gids = sorted({ds.index[i][0] for i in range(len(ds))})
    rng.shuffle(gids)
    n_te = max(2, len(gids) // 5)
    test_g = set(gids[:n_te])
    emb_cache: dict[int, np.ndarray] = {}
    tasks = []  # (gi, emb, labels, ts, te, ws, we, anchor, ei, n)
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
            q = npz[f"task_{j}_query"].astype(np.float32)
            x = npz["x"].astype(np.float32)
            tasks.append((
                gi, emb_cache[gi],
                npz[f"task_{j}_labels"].astype(np.float32),
                x[:, TS], x[:, TE], float(q[6]), float(q[7]),
                int(npz[f"task_{j}_anchor_row"]),
                npz["edge_index"].astype(np.int64), x.shape[0]))
            npz.close()

    # probe #4: MLP node_emb->(ts,te) trained on TRAIN graphs only
    Xtr = np.concatenate([t[1] for t in tasks if t[0] not in test_g])
    Ytr = np.concatenate([np.stack([t[3], t[4]], 1)
                          for t in tasks if t[0] not in test_g])
    net, mu, sd = _fit_temporal_mlp(Xtr, Ytr, seed=a.seed)

    sc = {k: [] for k in ("oracle", "anchor", "best_qstar",
                           "overlap_probed")}
    n_used = 0
    for (gi, emb, lab, ts, te, ws, we, anc, ei, n) in tasks:
        if gi not in test_g:
            continue
        L = torch.tensor(lab)
        if (L >= 0.75).sum() == 0 or (L <= 0.25).sum() == 0:
            continue  # degenerate for pairwise ceiling; skip uniformly
        n_used += 1
        ov = np.maximum(0.0, np.minimum(te, we) - np.maximum(ts, ws))
        sc["oracle"].append(ndcg_at_k(torch.tensor(ov), L, 10))
        sc["anchor"].append(ndcg_at_k(torch.tensor(_bfs_dist(n, ei, anc)),
                                      L, 10))
        et = torch.tensor(emb)
        sc["best_qstar"].append(
            ndcg_at_k(torch.tensor(_best_qstar(et, L, c)), L, 10))
        with torch.no_grad():
            pr = net(torch.tensor((emb - mu) / sd,
                                  dtype=torch.float32)).numpy()
        ov_p = np.maximum(0.0, np.minimum(pr[:, 1], we)
                          - np.maximum(pr[:, 0], ws))
        sc["overlap_probed"].append(ndcg_at_k(torch.tensor(ov_p), L, 10))

    print(f"\n[P2] mean ndcg@10 over {n_used} held-out task-{a.task} "
          f"graphs (frozen locked encoder):")
    o = np.mean(sc["oracle"]); an = np.mean(sc["anchor"])
    bq = np.mean(sc["best_qstar"]); op = np.mean(sc["overlap_probed"])
    print(f"  1. oracle (true overlap)        : {o:.4f}   [ceiling]")
    print(f"  2. anchor-BFS                   : {an:.4f}   [heuristic]")
    print(f"  3. best single q*  (-dist fam)  : {bq:.4f}   "
          f"[deployed-family CEILING, oracle-fit]")
    print(f"  4. overlap on probed temporal   : {op:.4f}   "
          f"[right head-class, SAME encoder]")
    print(f"\n[P2] reference: trained QueryToBall ~0.47 (memory, adapted "
          f"head). gaps: oracle-bestq*={o-bq:+.3f}  4-vs-3={op-bq:+.3f}")
    if bq < an + 0.03 and op > bq + 0.10:
        v = ("c' CONFIRMED: the -dist-to-a-single-point family cannot "
             "express interval overlap even with a perfect oracle-fit "
             "point; a temporal/interval-aware head on the SAME frozen "
             "encoder recovers it (#4). FIX = scoring head class, NOT "
             "objective/scale/geometry.")
    elif bq >= o - 0.05:
        v = ("family is fine; bottleneck is purely the learned "
             "objective/query-map (c) -- a good q* EXISTS, training "
             "doesn't find it from the 18-D query under pairwise hinge.")
    else:
        v = (f"intermediate: best-q* {bq:.3f} sits between anchor {an:.3f} "
             f"and oracle {o:.3f}; family is partially expressive. "
             f"Report; both head-class and objective levers in play.")
    print(f"[P2] VERDICT: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

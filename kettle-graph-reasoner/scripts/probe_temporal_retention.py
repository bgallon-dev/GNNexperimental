"""
P1 — encoder retention probe (Stage-B bottleneck investigation).

Question: task-2 relevance = interval overlap of the query window [q6,q7]
with each node's temporal scope [x21,x22] (the oracle, ndcg 0.93). The
Stage-B head scores nodes by dist(query_point, node_emb). For that to work
the node's temporal scope must be RECOVERABLE from node_emb. But Stage-A is
unconditional contrastive — it is never asked to retain x[:,21:22]. This
probe asks: does it survive anyway?

Cheap, decisive, no GNN training: run the FROZEN locked encoder over the
hardened real-eval corpus (the corpus the 0.475/0.93 numbers came from),
fit a ridge probe node_emb -> (ts,te), report held-out-graph R² against
two controls (orthogonal-signal discipline):
  - shuffle floor: targets row-permuted within test  -> chance R² (~0)
  - raw-context : probe from raw x WITHOUT cols 21,22 -> is temporal
                  inferable from other input features regardless of encoder?

Pre-registered decision (both ts and te, held-out R²):
  R² >= 0.70  -> info survives; encoder NOT the bottleneck -> run P2
  R² <= 0.20  -> encoder destroyed it (a'); fix = Stage-A retention aux
  else        -> partial; report, lean by magnitude vs raw-context
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

TS_COL, TE_COL = 21, 22


def _mlp_r2(Xtr, Ytr, Xte, Yte, seed=0, steps=800):
    """Small 2-layer MLP probe (nonlinear recoverability). Held-out R2."""
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    xe = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(Ytr, dtype=torch.float32)
    ye = torch.tensor(Yte, dtype=torch.float32)
    net = torch.nn.Sequential(
        torch.nn.Linear(xt.shape[1], 256), torch.nn.GELU(),
        torch.nn.Linear(256, 2))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((net(xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(xe)
    ss_res = ((ye - pred) ** 2).sum(0)
    ss_tot = ((ye - ye.mean(0)) ** 2).sum(0) + 1e-12
    return (1.0 - ss_res / ss_tot).numpy()


def _ridge_r2(Xtr, Ytr, Xte, Yte, lam=1.0):
    """Closed-form ridge; return per-column R2 on the test split."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1), np.float32)], 1)
    Xte = np.concatenate([Xte, np.ones((len(Xte), 1), np.float32)], 1)
    d = Xtr.shape[1]
    W = np.linalg.solve(
        Xtr.T @ Xtr + lam * np.eye(d, dtype=np.float32), Xtr.T @ Ytr
    )
    pred = Xte @ W
    ss_res = ((Yte - pred) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus",
                    default="src/data/corpus/real_domain_eval_hardened_250")
    ap.add_argument("--checkpoint",
                    default="runs/v3.1-baseline-hyp-h128-l4-seed1/encoder.pt")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--task", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    corpus = str((root / a.corpus).resolve())
    ckpt = (root / a.checkpoint).resolve()
    summ = Path(a.summary) if a.summary else ckpt.parent / "summary.json"

    ds = CorpusDataset(corpus_dir=corpus, split="val", split_seed=0,
                       include_tasks={a.task})
    if len(ds) == 0:  # eval corpora often have no split -> use all
        ds = CorpusDataset(corpus_dir=corpus, split="all", split_seed=0,
                           include_tasks={a.task})
    model, cfg = _load_encoder(ckpt, summ, ds)
    print(f"[P1] encoder {cfg['model']} h{cfg['hidden_dim']}/"
          f"l{cfg['num_layers']} | corpus={Path(corpus).name} "
          f"| {len(ds)} task-{a.task} samples")

    embs, temp, rawctx, gids = [], [], [], []
    seen = {}
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            gid = ds.index[i][0]
            if gid in seen:           # one graph once (encoder is unconditional)
                continue
            seen[gid] = 1
            out = model(s.x, s.edge_index, s.edge_type, s.edge_descriptor,
                        node_descriptor=s.node_descriptor)
            e = out.node_embeddings.detach().cpu().numpy().astype(np.float32)
            x = s.x.detach().cpu().numpy().astype(np.float32)
            embs.append(e)
            temp.append(x[:, [TS_COL, TE_COL]])
            rc = np.delete(x, [TS_COL, TE_COL], axis=1)
            rawctx.append(rc)
            gids.append(np.full(len(e), gid, dtype=np.int64))

    E = np.concatenate(embs); Y = np.concatenate(temp)
    RC = np.concatenate(rawctx); G = np.concatenate(gids)
    ug = np.unique(G)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(ug)
    n_te = max(1, len(ug) // 5)
    te_g = set(ug[:n_te].tolist())
    te = np.array([g in te_g for g in G])
    tr = ~te
    print(f"[P1] nodes: {tr.sum()} train / {te.sum()} test "
          f"({len(ug)-n_te}/{n_te} graphs); emb dim={E.shape[1]}")

    if len(ug) < 5:
        print(f"[P1] WARN: only {len(ug)} unique graphs -> thin split, "
              f"treat R2 as indicative not robust")
    r2_emb = _ridge_r2(E[tr], Y[tr], E[te], Y[te])
    r2_raw = _ridge_r2(RC[tr], Y[tr], RC[te], Y[te])
    Yte_shuf = Y[te][rng.permutation(te.sum())]
    r2_shuf = _ridge_r2(E[tr], Y[tr], E[te], Yte_shuf)
    r2_emb_mlp = _mlp_r2(E[tr], Y[tr], E[te], Y[te], seed=a.seed)

    def fmt(v):
        return f"ts={v[0]:+.3f} te={v[1]:+.3f} mean={float(v.mean()):+.3f}"

    print(f"\n[P1] LINEAR R2(node_emb -> temporal):   {fmt(r2_emb)}")
    print(f"[P1] MLP    R2(node_emb -> temporal):   {fmt(r2_emb_mlp)}  "
          f"(nonlinear: is it FULLY there?)")
    print(f"[P1] control R2(raw x minus 21,22):     {fmt(r2_raw)}  "
          f"(inferable from other input feats at all?)")
    print(f"[P1] shuffle floor (chance):            {fmt(r2_shuf)}")

    m = float(np.mean(r2_emb))
    mm = float(np.mean(r2_emb_mlp))
    best = max(m, mm)
    if best >= 0.70:
        verdict = ("SURVIVES -- encoder is NOT the bottleneck. Temporal "
                   "scope is recoverable from node_emb "
                   f"(best R2={best:+.3f}). -> proceed to P2 "
                   "(scoring-family / objective).")
    elif best <= 0.20:
        verdict = ("DESTROYED -- encoder (a') IS a bottleneck. Stage-A "
                   "contrastive does not retain temporal scope; the head "
                   "cannot score on what is gone. Fix = Stage-A retention "
                   "auxiliary (reconstruct x[:,21:22] from node_emb).")
    else:
        lean = ("below raw-context: encoder LOSES recoverability the "
                "input had" if best < float(np.mean(r2_raw)) - 0.05
                else "~ raw-context: limited by input/GNN-smear, not "
                "destroyed")
        verdict = (f"PARTIAL (best mean R2={best:+.3f}); {lean}. The signal "
                   f"is present but smeared; P2 needed to test the scoring "
                   f"family / objective.")
    print(f"\n[P1] VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

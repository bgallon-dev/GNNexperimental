r"""V2 MVP-0: global projection head on the FROZEN v1.0 trunk.

Decisive question (Docs/KGR_V2_PLAN.md): does the frozen trunk already
carry region-identity information that a projection can expose as a
GLOBAL space — without touching the local space? Train a small MLP
projection with cross-graph InfoNCE on tier1 (positives = 2-hop
same-graph pairs; negatives = nodes of OTHER graphs in batch), then
re-run the oracle_loo protocol on tutorstructure pool cases in the
projected space. BAR: nonlocal oracle_loo > 0.126 (frozen space: 0.115).

    py -m scripts.mvp0_global_projection
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.codegraph import cases as C
from src.codegraph.harness import _build_encoder
from src.data.corpus_dataset import _build_graph_tensors
from src.training.metrics import ndcg_at_k

from scripts.blend_pool_experiment import _Ctx

CKPT = Path("frozen/kgr-v1.0-2026-07-07/encoder_baseline")
OUT = Path("runs/mvp0_global_projection")
N_GRAPHS, EPOCHS, BATCH_G, PROJ_D, TAU = 120, 3, 8, 32, 0.1


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device("cpu")

    # 1) embed tier1 graphs with the frozen trunk
    files = sorted(Path("src/data/corpus/tier1").glob("graph_*.npz"))[:N_GRAPHS]
    enc = None
    embs, nbrs = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        g = _build_graph_tensors(z)
        if enc is None:
            enc, _ = _build_encoder(CKPT, g, device)
        with torch.no_grad():
            e = enc(g["x"], g["edge_index"], g["edge_type"],
                    g["edge_descriptor"],
                    node_descriptor=g["node_descriptor"]).node_embeddings
        embs.append(e)
        n = e.shape[0]
        adj = [[] for _ in range(n)]
        for s, t in zip(*g["edge_index"].numpy()):
            adj[int(s)].append(int(t))
            adj[int(t)].append(int(s))
        # 2-hop neighbor list per node (positives pool)
        two = []
        for u in range(n):
            s2 = set(adj[u])
            for v in adj[u]:
                s2.update(adj[v])
            s2.discard(u)
            two.append(sorted(s2) or [u])
        nbrs.append(two)
    print(f"embedded {len(embs)} tier1 graphs")

    proj = nn.Sequential(nn.Linear(embs[0].shape[1], 128), nn.ReLU(),
                         nn.Linear(128, PROJ_D))
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        order = rng.permutation(len(embs))
        tot, nb = 0.0, 0
        for b0 in range(0, len(order), BATCH_G):
            gids = order[b0:b0 + BATCH_G]
            if len(gids) < 2:
                continue
            za, zp, gid_of = [], [], []
            for gi in gids:
                e = embs[gi]
                n = e.shape[0]
                sel = rng.choice(n, min(48, n), replace=False)
                pos = [nbrs[gi][u][rng.integers(len(nbrs[gi][u]))]
                       for u in sel]
                za.append(e[torch.from_numpy(sel)])
                zp.append(e[torch.tensor(pos)])
                gid_of += [int(gi)] * len(sel)
            a = nn.functional.normalize(proj(torch.cat(za)), dim=-1)
            p = nn.functional.normalize(proj(torch.cat(zp)), dim=-1)
            gid_t = torch.tensor(gid_of)
            logits = (a @ torch.cat([p, a]).T) / TAU
            # mask: same-graph non-positive anchors excluded from negatives
            A = a.shape[0]
            same = gid_t.unsqueeze(1) == torch.cat([gid_t, gid_t]).unsqueeze(0)
            mask = same.clone()
            mask[torch.arange(A), torch.arange(A)] = False   # keep positive
            logits = logits.masked_fill(mask, float("-inf"))
            loss = nn.functional.cross_entropy(
                logits, torch.arange(A))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        print(f"[ep {ep}] loss={tot/max(nb,1):.4f}")
    proj.eval()
    torch.save(proj.state_dict(), OUT / "projection.pt")

    # 3) eval: oracle_loo protocol on tutorstructure in projected space
    ctx = _Ctx(Path("../tutorstructure_patch"), CKPT, OUT, device)
    with torch.no_grad():
        pe = nn.functional.normalize(proj(ctx.emb), dim=-1)
    rows = []
    for cs in ctx.cases:
        posset = {r for r in cs.pos_rows if r != C.ABSTAIN_ROW}
        if len(posset) < 2:
            continue
        pool = ctx.pools.get(cs.task, np.empty(0, np.int64)).tolist()
        cand = sorted(set(pool) | posset)
        if len(cand) <= len(posset):
            continue
        ct = torch.tensor(cand)
        lab = torch.tensor([1.0 if r in posset else 0.0 for r in cand])
        d_q = ctx.hops(cs.query_row)
        loc = "local" if min(float(d_q[r]) for r in posset) <= 1 else "nonlocal"
        pos_list = sorted(posset)
        dmat = torch.stack(
            [1.0 - pe[ct] @ pe[p] for p in pos_list], dim=1)
        idx = {r: i for i, r in enumerate(cand)}
        big = torch.finfo(dmat.dtype).max
        for j, p in enumerate(pos_list):
            if p in idx:
                dmat[idx[p], j] = big
        rows.append({"loc": loc,
                     "oracle": ndcg_at_k(-dmat.min(dim=1).values, lab, 10)})
    res = {}
    for loc in ("nonlocal", "local"):
        sub = [r["oracle"] for r in rows if r["loc"] == loc]
        res[loc] = {"oracle_loo_proj": sum(sub) / max(len(sub), 1),
                    "n": len(sub)}
        print(f"{loc}: oracle_loo(projected) = {res[loc]['oracle_loo_proj']:.3f} "
              f"(n={len(sub)})")
    res["bar"] = 0.126
    res["frozen_ref_nonlocal"] = 0.115
    res["verdict_clears_bar"] = bool(
        res["nonlocal"]["oracle_loo_proj"] > 0.126)
    print(f"BAR (>0.126): "
          f"{'CLEARED' if res['verdict_clears_bar'] else 'NOT cleared'}")
    (OUT / "mvp0_results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

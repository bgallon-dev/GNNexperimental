r"""V2 MVP-1 bar 2: projection on the LANDMARK trunk; oracle_loo > 0.126?

Same protocol as mvp0_global_projection, but: trunk = mvp1 landmark
encoder, projection trained on tier1_lm8, tutorstructure eval graph gets
the same +8 landmark dims (case/query building keeps the original 32-d x).

    py -m scripts.mvp1_global_eval
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.codegraph import cases as C
from src.codegraph.harness import _build_encoder, TASKS
from src.codegraph.ingest import build_npz
from src.data.corpus_dataset import _build_graph_tensors
from src.training.metrics import ndcg_at_k

CKPT = Path("runs/mvp1-lm8-h128-l4-s0")
OUT = Path("runs/mvp1_global_eval")
EPOCHS, BATCH_G, PROJ_D, TAU, K = 3, 8, 32, 0.1, 8


def _bfs(adj, src, n):
    d = np.full(n, np.inf, np.float32)
    d[src] = 0.0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if d[v] == np.inf:
                d[v] = d[u] + 1.0
                q.append(v)
    return d


def _adj_from(ei, n):
    adj = [[] for _ in range(n)]
    for s, t in zip(*ei):
        adj[int(s)].append(int(t))
        adj[int(t)].append(int(s))
    return adj


def _landmarks(adj, n, seed):
    rng = np.random.default_rng(seed)
    lms = rng.choice(n, min(K, n), replace=False)
    f = np.zeros((n, K), np.float32)
    for j, lm in enumerate(lms):
        f[:, j] = 1.0 / (1.0 + _bfs(adj, int(lm), n))
    return f


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device("cpu")

    # 1) train projection on tier1_lm8 with the mvp1 trunk
    files = sorted(Path("src/data/corpus/tier1_lm8").glob("graph_*.npz"))
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
        adj = _adj_from(z["edge_index"], n)
        two = []
        for u in range(n):
            s2 = set(adj[u])
            for v in adj[u]:
                s2.update(adj[v])
            s2.discard(u)
            two.append(sorted(s2) or [u])
        nbrs.append(two)
    print(f"embedded {len(embs)} tier1_lm8 graphs (trunk={CKPT.name})")

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
            A = a.shape[0]
            same = gid_t.unsqueeze(1) == torch.cat([gid_t, gid_t]).unsqueeze(0)
            mask = same.clone()
            mask[torch.arange(A), torch.arange(A)] = False
            logits = logits.masked_fill(mask, float("-inf"))
            loss = nn.functional.cross_entropy(logits, torch.arange(A))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        print(f"[ep {ep}] loss={tot/max(nb,1):.4f}")
    proj.eval()
    torch.save(proj.state_dict(), OUT / "projection.pt")

    # 2) eval on tutorstructure: same npz build, +landmark dims for the
    #    trunk; ORIGINAL 32-d x for case/query building
    repo = Path("../tutorstructure_patch")
    cg = build_npz(repo, OUT / "graph_tutor.npz",
                   C.collect_required_edges(repo, TASKS))
    with np.load(cg.npz_path) as z:
        d = {k: z[k] for k in z.files}
    x32 = d["x"].astype(np.float32)
    n = x32.shape[0]
    adj = _adj_from(d["edge_index"], n)
    d["x"] = np.concatenate([x32, _landmarks(adj, n, 4242)], axis=1)
    g = _build_graph_tensors(d)
    with torch.no_grad():
        emb = enc(g["x"], g["edge_index"], g["edge_type"],
                  g["edge_descriptor"],
                  node_descriptor=g["node_descriptor"]).node_embeddings
        pe = nn.functional.normalize(proj(emb), dim=-1)
    cases, pools, _ = C.load_repo_cases(repo, cg, x32, TASKS, repo.name)
    dcache = {}

    def hops(src):
        if src not in dcache:
            dcache[src] = _bfs(adj, src, n)
        return dcache[src]

    rows = []
    for cs in cases:
        if cs.task_family not in ("ranking", "abstain_ranking") \
                or cs.query_row2 >= 0:
            continue
        posset = {r for r in cs.pos_rows if r != C.ABSTAIN_ROW}
        if len(posset) < 2:
            continue
        pool = pools.get(cs.task, np.empty(0, np.int64)).tolist()
        cand = sorted(set(pool) | posset)
        if len(cand) <= len(posset):
            continue
        ct = torch.tensor(cand)
        lab = torch.tensor([1.0 if r in posset else 0.0 for r in cand])
        d_q = hops(cs.query_row)
        loc = ("local" if min(float(d_q[r]) for r in posset) <= 1
               else "nonlocal")
        pos_list = sorted(posset)
        dmat = torch.stack([1.0 - pe[ct] @ pe[p] for p in pos_list], dim=1)
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
        print(f"{loc}: oracle_loo(projected, mvp1 trunk) = "
              f"{res[loc]['oracle_loo_proj']:.3f} (n={len(sub)})")
    res["bar"] = 0.126
    res["mvp0_frozen_trunk_ref"] = 0.118
    res["verdict_clears_bar"] = bool(
        res["nonlocal"]["oracle_loo_proj"] > 0.126)
    print(f"BAR (>0.126): "
          f"{'CLEARED' if res['verdict_clears_bar'] else 'NOT cleared'}")
    (OUT / "mvp1_bar2_results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

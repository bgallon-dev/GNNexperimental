r"""Typed-walk chaining: graph provides transit, embedding prunes.
Beam walk: expand 1-hop frontier stepwise (graph edges = transit); at each
step keep top-B nodes by emb-dist to their PARENT (local ordering) with a
same-edge-type coherence bonus. Score = earlier-step-first, emb tie-break.
Compare vs ball_only emb-order on the SAME ball. py -m scripts.probe_typed_walk"""
from __future__ import annotations
import json
from collections import deque
from pathlib import Path
import numpy as np
import torch
from src.codegraph.harness import _build_encoder
from src.data.corpus_dataset import _build_graph_tensors
from src.modelsv3.distance_scoring import score_from_embeddings
from src.training.metrics import ndcg_at_k

FAMILY = {0:"provenance",1:"entity_res",2:"temporal",3:"multihop",4:"subgraph",5:"compound"}
CKPT = Path("frozen/kgr-v1.0-2026-07-07/encoder_baseline")
OUT = Path("runs/probe_typed_walk"); BEAM = 5

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    enc = None; rows = []
    for f in sorted(Path("src/data/corpus/real_domain_eval_all6").glob("graph_*.npz")):
        z = np.load(f, allow_pickle=True); g = _build_graph_tensors(z)
        if enc is None: enc,_ = _build_encoder(CKPT, g, torch.device("cpu"))
        with torch.no_grad():
            emb = enc(g["x"],g["edge_index"],g["edge_type"],g["edge_descriptor"],
                      node_descriptor=g["node_descriptor"]).node_embeddings
        n = emb.shape[0]; et = g["edge_type"].numpy()
        adj = [[] for _ in range(n)]
        for k,(s,t) in enumerate(zip(*g["edge_index"].numpy())):
            adj[int(s)].append((int(t),int(et[k]))); adj[int(t)].append((int(s),int(et[k])))
        D = np.full((emb.shape[0],), 0.0)
        for i in range(int(z["n_tasks"])):
            fam = FAMILY.get(int(z[f"task_{i}_type"]),"?")
            anchor = int(z[f"task_{i}_anchor_row"]); mh = int(z[f"task_{i}_max_hops"])
            labels = torch.from_numpy(z[f"task_{i}_labels"]).float()
            dq = deque([anchor]); dist = {anchor:0}
            while dq:
                u = dq.popleft()
                if dist[u] >= mh: continue
                for v,_t in adj[u]:
                    if v not in dist: dist[v]=dist[u]+1; dq.append(v)
            ball = [r for r in dist if r != anchor]
            if len(ball) < 5: continue
            lab = labels[torch.tensor(ball)]
            if float(lab.max())<=0 or float(lab.min())==float(lab.max()): continue
            de = -score_from_embeddings(emb[torch.tensor(ball)], emb[anchor], c=enc.c)
            # beam walk
            reached = {}; frontier = [(anchor,-1)]
            for step in range(mh):
                cand = []
                for u,ut in frontier:
                    du = -score_from_embeddings(
                        emb[torch.tensor([v for v,_ in adj[u]])], emb[u], c=enc.c)
                    for j,(v,vt) in enumerate(adj[u]):
                        if v==anchor or v in reached: continue
                        bonus = 0.5 if vt==ut else 0.0
                        cand.append((float(du[j])-bonus, v, vt))
                cand.sort()
                keep = []
                seen = set()
                for sc,v,vt in cand:
                    if v in seen: continue
                    seen.add(v); keep.append((sc,v,vt))
                    if len(keep)>=BEAM: break
                for r,(sc,v,vt) in enumerate(keep):
                    reached[v]=(step,r)
                frontier = [(v,vt) for _,v,vt in keep]
                if not frontier: break
            walk_sc = torch.tensor([
                -(reached[b][0]*100.0+reached[b][1]) if b in reached
                else torch.finfo(torch.float32).min for b in ball])
            row = {"family":fam,
                   "ball_emb": ndcg_at_k(-de, lab, 10),
                   "typed_walk": ndcg_at_k(walk_sc, lab, 10)}
            rows.append(row)
    print(f"{'family':<12}{'n':>6}{'ball_emb':>10}{'typed_walk':>12}")
    rep = {}
    for fam in sorted({r["family"] for r in rows})+["ALL"]:
        sub = rows if fam=="ALL" else [r for r in rows if r["family"]==fam]
        c = {a: sum(r[a] for r in sub)/len(sub) for a in ("ball_emb","typed_walk")}
        rep[fam] = c | {"n": len(sub)}
        print(f"{fam:<12}{len(sub):>6}{c['ball_emb']:>10.3f}{c['typed_walk']:>12.3f}")
    (OUT/"typed_walk_results.json").write_text(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()

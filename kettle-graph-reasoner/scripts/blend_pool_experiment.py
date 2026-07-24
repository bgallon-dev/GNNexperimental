r"""Blend experiment for the pool-retrieval bottleneck (pre-registered).

Follow-up to ``scripts/probe_pool_bottleneck.py`` (see
runs/probe_pool_bottleneck/PROBE_FINDINGS.md), which decomposed the pool
failure into (1) a query-side information deficit (the head only sees a
random identity code; querying at emb[anchor] directly = 3x the trained
head) and (2) a LOW point-scorer ceiling (oracle query point at another
same-case positive: ndcg@10 0.117 — answers do not cluster around any
point).

Arms (per-task heads, trained fresh per seed, file-level 70/15/15 split):

  cond_point  QueryToBall on logmap0(emb[anchor]) -> query point ->
              -dist scoring. The "condition the head on the anchor
              embedding" fix. PREDICTION: lands ~0.07-0.12, capped by
              the point-scorer family ceiling.
  blend       per-task MLP over per-candidate features
              [d_emb, exp(-d_emb), d_hop, 1/(1+d_hop), unreachable,
               cand radius, anchor radius, x[cand][:21], x[anchor][:21],
               same-type flag] -> scalar score. NOT a point scorer.
              PREDICTION: can exceed the 0.117 ceiling.

Baselines recomputed on the same eval cases: anchor_emb, anchor_bfs,
random, oracle_loo (reference ceiling).

Pre-registered bars (test|nonlocal pool ndcg@10, 3 seeds):
  SUCCESS bar: mean > anchor_bfs on the same cells (probe: 0.086)
  ESCAPE  bar: mean > 0.117 (the point-scorer oracle ceiling)

E4 MVP-A (Docs/ARCH_EFFICIENCY_PLAN.md, opt-in via --mixture-feats K):
adds (a) a standalone ``mixture`` arm — the K-offset prototype-fit head
(src/modelsv3/mixture_offsets.py) fit on the train split per seed — and
(b) a ``blend_mix`` arm: the same blend MLP with 2K extra per-candidate
features (raw distance to each mixture query point + exp(-d)). The plain
``blend`` arm is still trained at the legacy feature width with the same
seed stream, so parents are measured in-run on identical cells.
MVP-A bars: blend_mix > max(blend, mixture) + 2*std(blend); headline
ESCAPE 0.117 unchanged. Default --mixture-feats 0 = bit-identical legacy.

Run from kettle-graph-reasoner/:
    py -m scripts.blend_pool_experiment --repo ../tutorstructure_patch \
        --ckpt frozen/kgr-v1.0-2026-07-07/encoder_baseline \
        --out runs/blend_pool_scorer
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.codegraph import cases as C
from src.codegraph.harness import _build_encoder, _embed, TASKS
from src.codegraph.ingest import build_npz
from src.codegraph.metrics_ext import mrr
from src.data.corpus_dataset import _build_graph_tensors
from src.modelsv3.distance_scoring import score_from_embeddings
from src.modelsv3.mixture_offsets import (
    fit_mixture_offsets,
    mixture_query_points,
)
from src.modelsv3.query_encoder import QueryToBall
from src.modelsv3.ranking import pairwise_ranking_loss
from src.modelsv2.layers import poincare_ops as P
from src.training.metrics import ndcg_at_k, recall_at_k

HOP_CAP = 20.0
HOP_UNREACH = 25.0
X_FEATS = 21          # x[:, :21] = type one-hots + kind + degrees + depth
NEG_SAMPLE = 32
CAND_CAP = 256


def _bfs(adj, src, n):
    d = np.full(n, np.inf, np.float32)
    d[src] = 0.0
    dq = deque([src])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if d[v] == np.inf:
                d[v] = d[u] + 1.0
                dq.append(v)
    return d


class _Ctx:
    """Repo context shared across seeds/arms: embeddings, features, BFS."""

    def __init__(self, repo_dir: Path, ckpt: Path, out_dir: Path, device,
                 keep_answer_edges: bool = False):
        name = repo_dir.name
        required = (set() if keep_answer_edges
                    else C.collect_required_edges(repo_dir, TASKS))
        cg = build_npz(repo_dir, out_dir / f"graph_{name}.npz", required)
        with np.load(cg.npz_path) as z:
            g = _build_graph_tensors(z)
            self.x = z["x"].astype(np.float32)
        self.enc, self.cfg = _build_encoder(ckpt, g, device)
        # E2: geometry attrs so euclidean-control checkpoints score with
        # L2. Hyperbolic path unchanged (euclidean=False is the
        # score_from_embeddings default it always used).
        self.euclidean = self.cfg.get("model", "hyperbolic") == "euclidean"
        self.c = getattr(self.enc, "c", 1.0)
        self.emb = _embed(self.enc, g, device)
        self.n_nodes = cg.n_nodes
        self.device = device
        self.radius = self.emb.norm(dim=-1).cpu().numpy()
        self.top_type = self.x[:, :16].argmax(axis=1)
        cs_all, self.pools, _ = C.load_repo_cases(
            repo_dir, cg, self.x, TASKS, name)
        # single-anchor ranking-family cases with >=1 real positive
        self.cases = [
            c for c in cs_all
            if c.task_family in ("ranking", "abstain_ranking")
            and c.query_row2 < 0
            and any(r != C.ABSTAIN_ROW for r in c.pos_rows)
        ]
        adj: list[list[int]] = [[] for _ in range(self.n_nodes)]
        ei = g["edge_index"].numpy()
        for s, t in zip(ei[0], ei[1]):
            adj[int(s)].append(int(t))
            adj[int(t)].append(int(s))
        self._adj = adj
        self._dcache: dict[int, np.ndarray] = {}

    def hops(self, src: int) -> np.ndarray:
        if src not in self._dcache:
            self._dcache[src] = _bfs(self._adj, src, self.n_nodes)
        return self._dcache[src]

    def d_emb(self, anchor: int, rows: torch.Tensor) -> torch.Tensor:
        """Embedding distance from emb[anchor] to emb[rows] (positive)."""
        return -score_from_embeddings(
            self.emb[rows], self.emb[anchor], c=self.c,
            euclidean=self.euclidean)

    def features(self, cs, cand: list[int], V_task: torch.Tensor | None = None,
                 k_pad: int = 0) -> torch.Tensor:
        """(n_cand, F) blend features. All torch on self.device.

        MVP-A: with ``V_task`` (K,d tangent offsets) appends 2K columns —
        distance to each mixture query point + exp(-d). ``k_pad`` keeps
        the width stable (zero block) for tasks with no fitted offsets.
        Default call (V_task=None, k_pad=0) is bit-identical legacy."""
        a = cs.query_row
        rows = torch.tensor(cand, dtype=torch.long, device=self.device)
        de = self.d_emb(a, rows)                       # (n,)
        hp = self.hops(a)[cand]
        hp = np.where(np.isfinite(hp), np.minimum(hp, HOP_CAP), HOP_UNREACH)
        hp_t = torch.from_numpy(hp.astype(np.float32)).to(self.device)
        unreach = (hp_t >= HOP_UNREACH).float()
        cand_np = np.asarray(cand)
        feats = [
            de.unsqueeze(1),
            torch.exp(-de).unsqueeze(1),
            hp_t.unsqueeze(1) / HOP_CAP,
            (1.0 / (1.0 + hp_t)).unsqueeze(1),
            unreach.unsqueeze(1),
            torch.from_numpy(self.radius[cand_np]).to(
                self.device).unsqueeze(1),
            torch.full((len(cand), 1), float(self.radius[a]),
                       device=self.device),
            torch.from_numpy(self.x[cand_np, :X_FEATS]).to(self.device),
            torch.from_numpy(
                np.tile(self.x[a, :X_FEATS], (len(cand), 1))
            ).to(self.device),
            torch.from_numpy(
                (self.top_type[cand_np] == self.top_type[a])
                .astype(np.float32)
            ).to(self.device).unsqueeze(1),
        ]
        if V_task is not None:
            qps = mixture_query_points(self.emb, a, V_task, self.c)
            d_mix = torch.stack(
                [-score_from_embeddings(self.emb[rows], q, c=self.c)
                 for q in qps], dim=1)                 # (n, K') distances
            if k_pad > d_mix.shape[1]:  # kmeans clamped K' < K: keep width
                d_mix = torch.cat(
                    [d_mix,
                     d_mix[:, -1:].expand(-1, k_pad - d_mix.shape[1])],
                    dim=1)
            feats += [d_mix, torch.exp(-d_mix)]
        elif k_pad > 0:
            feats.append(torch.zeros(len(cand), 2 * k_pad,
                                     device=self.device))
        return torch.cat(feats, dim=1)

    @property
    def n_feats(self) -> int:
        return 7 + 2 * X_FEATS + 1


def _pairwise_hinge(scores, labels, margin):
    """Score-space pairwise hinge + rank accuracy (all pos x neg pairs)."""
    pos = scores[labels >= 0.5]
    neg = scores[labels < 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return None, 0.0
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    loss = torch.relu(margin - diff).mean()
    acc = float((diff > 0).float().mean().item())
    return loss, acc


def _train_cand(ctx, cs, rng, neg_sample=NEG_SAMPLE, cand_cap=CAND_CAP):
    """Training candidate list: positives + hardnegs + sampled pool negs,
    same recipe/cap as the harness."""
    pool = ctx.pools.get(cs.task, np.empty(0, np.int64))
    posset = {r for r in cs.pos_rows if r != C.ABSTAIN_ROW}
    negs = [r for r in cs.hardneg_rows if r != C.ABSTAIN_ROW]
    if pool.size:
        negs += [int(r) for r in pool[rng.integers(0, pool.size, neg_sample)]]
    cand = list(dict.fromkeys(
        [r for r in cs.pos_rows if r != C.ABSTAIN_ROW] + negs))
    CAND_CAP = cand_cap
    if len(cand) > CAND_CAP:
        keep_pos = [r for r in cand if r in posset]
        keep_rest = [r for r in cand if r not in posset]
        n_rest = max(CAND_CAP - len(keep_pos), 0)
        if len(keep_rest) > n_rest:
            sel = rng.choice(len(keep_rest), n_rest, replace=False)
            keep_rest = [keep_rest[int(i)] for i in sel]
        cand = keep_pos + keep_rest
    return cand, posset


def _train_cond_point(ctx, train_cases, args, seed):
    torch.manual_seed(seed)
    head = QueryToBall(
        query_dim=ctx.cfg["hidden_dim"], hidden_dim=ctx.cfg["hidden_dim"],
        c=ctx.cfg["curvature"], euclidean=False, arch="qh0",
    ).to(ctx.device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    rng = np.random.default_rng(seed)
    for ep in range(args.epochs):
        for idx in rng.permutation(len(train_cases)):
            cs = train_cases[idx]
            cand, posset = _train_cand(ctx, cs, rng,
                neg_sample=args.neg_sample,
                cand_cap=max(CAND_CAP, args.neg_sample + 32))
            if not cand:
                continue
            rows = torch.tensor(cand, dtype=torch.long, device=ctx.device)
            lab = torch.tensor(
                [1.0 if r in posset else 0.0 for r in cand],
                device=ctx.device)
            qin = P.logmap0(ctx.emb[cs.query_row], ctx.enc.c).detach()
            qp = head(qin)
            loss, _ = pairwise_ranking_loss(
                qp, ctx.emb[rows], lab, c=ctx.enc.c,
                margin=ctx.cfg["margin"])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


def _fit_mixture(ctx, train_cases, k: int, seed: int, key_fn=None,
                 min_cases: int = 0):
    """K-offset prototype fit (zero gradients) on the train split.
    Reseeds torch so the kmeans init is deterministic per seed.
    key_fn=None -> keyed by task (MVP-A). MVP-B/C pass conditioned keys
    (e.g. (task, anchor_type)); cells with < min_cases train cases are
    dropped — scoring falls back to the task-level offsets, so the
    conditioned arms are do-no-harm vs MVP-A by construction."""
    torch.manual_seed(seed)
    by_key: dict = {}
    for cs in train_cases:
        pos = [r for r in cs.pos_rows
               if r != C.ABSTAIN_ROW and r != cs.query_row]
        if pos:
            key = key_fn(cs) if key_fn is not None else cs.task
            by_key.setdefault(key, []).append((cs.query_row, pos))
    if min_cases:
        by_key = {k_: v for k_, v in by_key.items() if len(v) >= min_cases}
    return fit_mixture_offsets(ctx.emb, by_key, ctx.enc.c, k=k)


def _deg_bins(ctx, train_cases) -> np.ndarray:
    """MVP-C: decile bounds of train-case anchor degree (free at query
    time; one global binning, keys are (task, bin))."""
    degs = [len(ctx._adj[cs.query_row]) for cs in train_cases]
    return np.quantile(np.asarray(degs, dtype=np.float64),
                       [i / 10.0 for i in range(1, 10)])


def _train_blend(ctx, train_cases, args, seed, V_by_task=None):
    """V_by_task=None -> legacy blend (bit-identical to the landed runs);
    otherwise the MVP-A blend_mix variant with 2K mixture features."""
    torch.manual_seed(seed)
    k_mix = args.mixture_feats if V_by_task is not None else 0
    net = nn.Sequential(
        nn.Linear(ctx.n_feats + 2 * k_mix, 64), nn.ReLU(), nn.Linear(64, 1),
    ).to(ctx.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(seed)
    for ep in range(args.epochs):
        tot, acc, nb = 0.0, 0.0, 0
        for idx in rng.permutation(len(train_cases)):
            cs = train_cases[idx]
            cand, posset = _train_cand(ctx, cs, rng,
                neg_sample=args.neg_sample,
                cand_cap=max(CAND_CAP, args.neg_sample + 32))
            if not cand:
                continue
            lab = torch.tensor(
                [1.0 if r in posset else 0.0 for r in cand],
                device=ctx.device)
            v_t = V_by_task.get(cs.task) if V_by_task is not None else None
            sc = net(ctx.features(cs, cand, v_t, k_pad=k_mix)).squeeze(-1)
            loss, pair_acc = _pairwise_hinge(sc, lab, ctx.cfg["margin"])
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            acc += pair_acc
            nb += 1
        if ep in (0, args.epochs - 1):
            print(f"      blend [ep {ep:>2}] loss={tot/max(nb,1):.4f} "
                  f"pair_acc={acc/max(nb,1):.3f}")
    net.eval()
    return net


@torch.no_grad()
def _eval_arms(ctx, eval_cases, heads_by_task, nets_by_task,
               nets_mix_by_task=None, V_by_task=None, k_mix=0,
               V_typed=None, V_deg=None, deg_bins=None):
    """Score every eval case under every arm; returns row dicts."""
    rows = []
    for cs in eval_cases:
        posset = {r for r in cs.pos_rows if r != C.ABSTAIN_ROW}
        pool_rows = ctx.pools.get(cs.task, np.empty(0, np.int64)).tolist()
        cand = sorted(set(pool_rows) | posset)
        if len(cand) <= len(posset):
            continue
        rows_t = torch.tensor(cand, dtype=torch.long, device=ctx.device)
        lab = torch.tensor([1.0 if r in posset else 0.0 for r in cand])
        d_q = ctx.hops(cs.query_row)
        finite_pos = [float(d_q[r]) for r in posset]
        min_hop = min(finite_pos) if finite_pos else np.inf
        locality = "local" if min_hop <= 1.0 else "nonlocal"

        arms: dict[str, torch.Tensor] = {}
        # learned arms
        head = heads_by_task.get(cs.task)
        if head is not None:
            qin = P.logmap0(ctx.emb[cs.query_row], ctx.enc.c)
            arms["cond_point"] = score_from_embeddings(
                ctx.emb[rows_t], head(qin), c=ctx.enc.c).cpu()
        net = nets_by_task.get(cs.task)
        if net is not None:
            arms["blend"] = net(ctx.features(cs, cand)).squeeze(-1).cpu()
        # MVP-A arms (only when --mixture-feats > 0)
        v_t = V_by_task.get(cs.task) if V_by_task is not None else None

        def _mix_score(V):
            qps = mixture_query_points(ctx.emb, cs.query_row, V, ctx.enc.c)
            return torch.stack(
                [score_from_embeddings(ctx.emb[rows_t], q, c=ctx.enc.c)
                 for q in qps]).max(0).values.cpu()

        if v_t is not None:
            arms["mixture"] = _mix_score(v_t)
            # MVP-B: per-(task, anchor-type) offsets, task-level fallback.
            if V_typed is not None:
                vt2 = V_typed.get(
                    (cs.task, int(ctx.top_type[cs.query_row])))
                arms["mixture_typed"] = _mix_score(
                    vt2 if vt2 is not None else v_t)
            # MVP-C: per-(task, anchor-degree-decile), task-level fallback.
            if V_deg is not None and deg_bins is not None:
                b = int(np.searchsorted(
                    deg_bins, float(len(ctx._adj[cs.query_row]))))
                vd = V_deg.get((cs.task, b))
                arms["mixture_deg"] = _mix_score(
                    vd if vd is not None else v_t)
        net_mix = (nets_mix_by_task or {}).get(cs.task)
        if net_mix is not None:
            arms["blend_mix"] = net_mix(
                ctx.features(cs, cand, v_t, k_pad=k_mix)).squeeze(-1).cpu()
        # baselines
        arms["anchor_emb"] = -ctx.d_emb(cs.query_row, rows_t).cpu()
        hp = d_q[cand]
        hp = np.where(np.isfinite(hp), hp, ctx.n_nodes + 1.0)
        arms["anchor_bfs"] = torch.from_numpy((-hp).astype(np.float32))
        seed = int.from_bytes(cs.case_id.encode()[-8:], "little")
        arms["random"] = torch.from_numpy(
            np.random.default_rng(seed)
            .standard_normal(len(cand)).astype(np.float32))
        if len(posset) >= 2:
            pos_list = sorted(posset)
            dmat = torch.stack(
                [ctx.d_emb(p, rows_t) for p in pos_list], dim=1).cpu()
            idx_map = {r: i for i, r in enumerate(cand)}
            big = torch.finfo(dmat.dtype).max
            for j, p in enumerate(pos_list):
                i = idx_map.get(p)
                if i is not None:
                    dmat[i, j] = big
            arms["oracle_loo"] = -dmat.min(dim=1).values

        row = {"task": cs.task, "split": cs.split, "locality": locality}
        for arm, sc in arms.items():
            row[f"{arm}_ndcg10"] = ndcg_at_k(sc, lab, 10)
            row[f"{arm}_mrr"] = mrr(sc, lab)
            row[f"{arm}_r10"] = recall_at_k(sc, lab, 10)
        rows.append(row)
    return rows


ARMS = ("cond_point", "blend", "mixture", "mixture_typed", "mixture_deg",
        "blend_mix", "anchor_emb", "anchor_bfs", "random", "oracle_loo")


def _route(rows, arms_order):
    """MVP-A': per-task validation-gated arm choice (the reranker_router
    pattern). Choose per task by mean ndcg on val|nonlocal (>=5 rows,
    else the first arm = do-no-harm default); score test|nonlocal with
    each row's chosen arm, falling back to the default arm when the
    chosen value is missing for a row."""
    default = arms_order[0]
    choice = {}
    for t in sorted({r["task"] for r in rows}):
        sub = [r for r in rows
               if r["task"] == t and r["split"] == "val"
               and r["locality"] == "nonlocal"
               and all(f"{a}_ndcg10" in r for a in arms_order)]
        if len(sub) >= 5:
            means = {a: sum(r[f"{a}_ndcg10"] for r in sub) / len(sub)
                     for a in arms_order}
            choice[t] = max(means, key=means.get)
        else:
            choice[t] = default
    test = [r for r in rows
            if r["split"] == "test" and r["locality"] == "nonlocal"]
    vals = [r.get(f"{choice[r['task']]}_ndcg10",
                  r.get(f"{default}_ndcg10")) for r in test]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals) if vals else float("nan")), choice


def _agg(rows, arm):
    vals = [r[f"{arm}_ndcg10"] for r in rows if f"{arm}_ndcg10" in r]
    if not vals:
        return None
    return {
        "ndcg@10": sum(vals) / len(vals),
        "mrr": sum(r[f"{arm}_mrr"] for r in rows
                   if f"{arm}_mrr" in r) / len(vals),
        "r@10": sum(r[f"{arm}_r10"] for r in rows
                    if f"{arm}_r10" in r) / len(vals),
        "n": len(vals),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="../tutorstructure_patch")
    ap.add_argument("--ckpt",
                    default="frozen/kgr-v1.0-2026-07-07/encoder_baseline")
    ap.add_argument("--out", default="runs/blend_pool_scorer")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--neg-sample", type=int, default=NEG_SAMPLE)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--mixture-feats", type=int, default=0,
                    help="E4 MVP-A: K mixture offsets fit per task on the "
                         "train split; adds a standalone 'mixture' arm and "
                         "a 'blend_mix' arm with 2K extra features. "
                         "0 = off (bit-identical legacy run).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"[1/3] load + embed (ckpt={args.ckpt})")
    ctx = _Ctx(Path(args.repo), Path(args.ckpt), out_dir, device)
    if args.mixture_feats and ctx.euclidean:
        raise SystemExit("--mixture-feats requires a hyperbolic checkpoint "
                         "(gyro-frame offsets are Mobius ops)")
    C.assign_file_split(ctx.cases, args.split_seed, (0.70, 0.15, 0.15))
    tasks = sorted({c.task for c in ctx.cases})
    train_cases = [c for c in ctx.cases if c.split == "train"]
    eval_cases = [c for c in ctx.cases if c.split in ("val", "test")]
    print(f"    {len(ctx.cases)} cases ({len(train_cases)} train / "
          f"{len(eval_cases)} eval), tasks: {tasks}")

    all_rows: dict[int, list[dict]] = {}
    for seed in seeds:
        print(f"[2/3] seed {seed}: train per-task heads "
              f"({args.epochs} epochs)")
        V_by_task = (_fit_mixture(ctx, train_cases, args.mixture_feats, seed)
                     if args.mixture_feats else None)
        V_typed = V_deg = deg_bins = None
        if V_by_task is not None:
            print(f"    mixture offsets fit: {len(V_by_task)} tasks, "
                  f"K={args.mixture_feats}")
            # MVP-B: (task, anchor node-type) cells, >=20 train cases each.
            V_typed = _fit_mixture(
                ctx, train_cases, args.mixture_feats, seed,
                key_fn=lambda cs: (cs.task, int(ctx.top_type[cs.query_row])),
                min_cases=20)
            # MVP-C: (task, anchor-degree decile) cells, same sparsity guard.
            deg_bins = _deg_bins(ctx, train_cases)
            V_deg = _fit_mixture(
                ctx, train_cases, args.mixture_feats, seed,
                key_fn=lambda cs: (cs.task, int(np.searchsorted(
                    deg_bins, float(len(ctx._adj[cs.query_row]))))),
                min_cases=20)
            print(f"    conditioned fits: typed {len(V_typed)} cells, "
                  f"deg {len(V_deg)} cells (>=20 cases each; "
                  f"task-level fallback)")
        heads, nets, nets_mix = {}, {}, {}
        for task in tasks:
            tr = [c for c in train_cases if c.task == task]
            if not tr:
                continue
            print(f"    {task}: n={len(tr)}")
            heads[task] = _train_cond_point(ctx, tr, args, seed)
            nets[task] = _train_blend(ctx, tr, args, seed)
            if V_by_task is not None:
                nets_mix[task] = _train_blend(ctx, tr, args, seed,
                                              V_by_task=V_by_task)
        all_rows[seed] = _eval_arms(ctx, eval_cases, heads, nets,
                                    nets_mix_by_task=nets_mix,
                                    V_by_task=V_by_task,
                                    k_mix=args.mixture_feats,
                                    V_typed=V_typed, V_deg=V_deg,
                                    deg_bins=deg_bins)
        # persist per-case rows: routing / re-analysis without retraining
        (out_dir / f"rows_seed{seed}.json").write_text(
            json.dumps(all_rows[seed]))

    print("[3/3] aggregate")
    report = {"config": vars(args), "n_feats": ctx.n_feats, "cells": {}}

    def _cells(rows):
        out = {"overall": {a: _agg(rows, a) for a in ARMS}}
        for split in ("val", "test"):
            for loc in ("local", "nonlocal"):
                sub = [r for r in rows
                       if r["split"] == split and r["locality"] == loc]
                out[f"{split}|{loc}"] = {a: _agg(sub, a) for a in ARMS}
        for task in sorted({r["task"] for r in rows}):
            sub = [r for r in rows if r["task"] == task
                   and r["split"] == "test"]
            out[f"task:{task}|test"] = {a: _agg(sub, a) for a in ARMS}
        return out

    for seed, rows in all_rows.items():
        report["cells"][str(seed)] = _cells(rows)

    # cross-seed mean+-std for the learned arms on the headline cell
    headline = {}
    for arm in ARMS:
        vals = []
        for seed in seeds:
            cell = report["cells"][str(seed)]["test|nonlocal"].get(arm)
            if cell:
                vals.append(cell["ndcg@10"])
        if vals:
            m = sum(vals) / len(vals)
            sd = (sum((v - m) ** 2 for v in vals)
                  / max(len(vals) - 1, 1)) ** 0.5
            headline[arm] = {"mean": m, "std": sd, "per_seed": vals}
    report["headline_test_nonlocal_ndcg10"] = headline
    bfs_bar = headline.get("anchor_bfs", {}).get("mean", float("nan"))
    blend_m = headline.get("blend", {}).get("mean", float("nan"))
    report["bars"] = {
        "success_bar_anchor_bfs": bfs_bar,
        "escape_bar_point_ceiling": 0.117,
        "blend_clears_success": bool(blend_m > bfs_bar),
        "blend_clears_escape": bool(blend_m > 0.117),
    }
    if args.mixture_feats:
        bm = headline.get("blend_mix", {}).get("mean", float("nan"))
        parents = {
            "blend": headline.get("blend", {}).get("mean", float("nan")),
            "mixture": headline.get("mixture", {}).get("mean", float("nan")),
        }
        sd_pl = headline.get("blend", {}).get("std", float("nan"))
        bar = max(parents.values()) + 2 * sd_pl
        report["bars_mvpA"] = {
            "parents_mean": parents,
            "blend_std_in_run": sd_pl,
            "bar_beat_parents_by_2sd": bar,
            "blend_mix_mean": bm,
            "clears_parents_2sd": bool(bm > bar),
            "clears_escape_0117": bool(bm > 0.117),
        }
        # MVP-A': per-task val-gated routing (primary pair blend<->mixture;
        # extended set exploratory). Bar: routed_pair > blend mean.
        routed = {"pair": [], "ext": [], "choices": {}}
        for seed in seeds:
            v, ch = _route(all_rows[seed], ("blend", "mixture"))
            routed["pair"].append(v)
            v2, ch2 = _route(all_rows[seed], ("blend", "mixture",
                                              "mixture_typed", "mixture_deg"))
            routed["ext"].append(v2)
            routed["choices"][str(seed)] = {"pair": ch, "ext": ch2}
        def _ms(vals):
            m = sum(vals) / len(vals)
            sd = (sum((x - m) ** 2 for x in vals) / max(len(vals) - 1, 1)) ** 0.5
            return m, sd
        rp_m, rp_s = _ms(routed["pair"])
        re_m, re_s = _ms(routed["ext"])
        report["bars_mvpA_prime"] = {
            "routed_pair_mean": rp_m, "routed_pair_std": rp_s,
            "routed_ext_mean": re_m, "routed_ext_std": re_s,
            "per_seed_pair": routed["pair"], "per_seed_ext": routed["ext"],
            "choices": routed["choices"],
            "bar_blend_mean": parents["blend"],
            "routed_pair_clears_blend": bool(rp_m > parents["blend"]),
            "routed_pair_clears_escape_0117": bool(rp_m > 0.117),
        }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))

    print("\n=== HEADLINE: test|nonlocal pool ndcg@10 "
          "(mean+-std over seeds) ===")
    for arm in ARMS:
        if arm in headline:
            h = headline[arm]
            print(f"  {arm:<12} {h['mean']:.3f} +- {h['std']:.3f}   "
                  f"per-seed {['%.3f' % v for v in h['per_seed']]}")
    print(f"\n  SUCCESS bar (> anchor_bfs {bfs_bar:.3f}): "
          f"{'CLEARED' if report['bars']['blend_clears_success'] else 'NOT cleared'}")
    print(f"  ESCAPE  bar (> 0.117 point ceiling):  "
          f"{'CLEARED' if report['bars']['blend_clears_escape'] else 'NOT cleared'}")
    if args.mixture_feats:
        ba = report["bars_mvpA"]
        print(f"\n  MVP-A blend_mix {ba['blend_mix_mean']:.3f} vs "
              f"parents blend {ba['parents_mean']['blend']:.3f} / "
              f"mixture {ba['parents_mean']['mixture']:.3f}; "
              f"bar {ba['bar_beat_parents_by_2sd']:.3f}: "
              f"{'CLEARED' if ba['clears_parents_2sd'] else 'NOT cleared'}; "
              f"escape 0.117: "
              f"{'CLEARED' if ba['clears_escape_0117'] else 'NOT cleared'}")
        bp = report["bars_mvpA_prime"]
        print(f"  MVP-A' routed(blend<->mixture) "
              f"{bp['routed_pair_mean']:.3f} +- {bp['routed_pair_std']:.3f} "
              f"vs blend {bp['bar_blend_mean']:.3f}: "
              f"{'CLEARED' if bp['routed_pair_clears_blend'] else 'NOT cleared'}; "
              f"escape 0.117: "
              f"{'CLEARED' if bp['routed_pair_clears_escape_0117'] else 'NOT cleared'}")
        print(f"  MVP-A' routed(ext incl. typed/deg) "
              f"{bp['routed_ext_mean']:.3f} +- {bp['routed_ext_std']:.3f} "
              f"(exploratory)")
    print(f"\nreport: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

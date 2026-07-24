r"""Multi-repo code-graph training/eval harness for KGR.

Frozen shipped encoder + a fresh per-task ``QueryToBall`` head, trained
across a multi-repo code corpus and evaluated **leave-one-repo-out**:
each repo is the held-out eval exactly once, heads retrained per fold on
the other repos. Reports nDCG@10, MRR, Recall@10, Recall@50, and the
positive-vs-hard-negative margin, broken down by task type and split,
under both within-candidate and corpus-wide scoring, against random and
anchor-BFS baselines. The CV headline is the macro-average over folds
(equal-weighted repos).

With a single repo it falls back to a file-level 70/15/15 split.

Run from ``kettle-graph-reasoner/``:

    python -m src.codegraph.harness \
        --corpus-root ../corpus_validation \
        --extra-repo  ../tutorstructure_patch \
        --ckpt runs/sweep_arch_hyp/h128_l4_seed1 \
        --out  runs/codegraph_cv
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Optional: registers "privateuseone" for torch-directml when available.
try:  # pragma: no cover
    import torch_directml  # noqa: F401
except ImportError:
    pass

from ..data.corpus_dataset import _build_graph_tensors
from ..modelsv3.distance_scoring import score_from_embeddings
from ..modelsv3.hyperbolic_gnnV3 import KettleGraphReasonerV3
from ..modelsv2.layers import poincare_ops as P
from ..modelsv3.query_encoder import QueryToBall
from ..modelsv3.ranking import pairwise_ranking_loss
from torch import nn
from . import cases as C
from .classifier_head import TaskClassifierHead
from .ingest import build_npz
from .metrics_ext import CodeMetricAccumulator, ClassificationMetricAccumulator

TASKS = set(C.TASK_IDS)


@dataclass
class Repo:
    name: str
    emb: torch.Tensor                 # (N, hidden) frozen node embeddings
    pools: dict[str, np.ndarray]      # task -> corpus-wide candidate rows
    cases: list                       # list[cases.Case]
    n_nodes: int
    n_edges_kept: int
    answer_edges_ablated: int
    _adj: list[list[int]]
    _dcache: dict[int, np.ndarray]

    def dist(self, src: int) -> np.ndarray:
        if src not in self._dcache:
            d = np.full(self.n_nodes, np.inf, np.float32)
            d[src] = 0.0
            dq = deque([src])
            while dq:
                u = dq.popleft()
                for v in self._adj[u]:
                    if d[v] == np.inf:
                        d[v] = d[u] + 1.0
                        dq.append(v)
            d[~np.isfinite(d)] = self.n_nodes + 1.0
            self._dcache[src] = d
        return self._dcache[src]


def _discover_repos(corpus_root: Path, extra: list[str]) -> list[Path]:
    repos: list[Path] = []
    if corpus_root.is_dir():
        repos += sorted(
            p for p in corpus_root.iterdir()
            if (p / "nodes.jsonl").is_file()
        )
    for e in extra:
        p = Path(e)
        if (p / "nodes.jsonl").is_file():
            repos.append(p)
    return repos


def _build_encoder(ckpt: Path, g: dict, device):
    cfg = json.load(open(ckpt / "summary.json"))["config"]
    # E5: post-2026-07-10 checkpoints have no attention type_emb table;
    # absent key => legacy True (frozen v1.0 etc. keep loading strict).
    net = (g["edge_descriptor"].shape[0]
           if cfg.get("attn_type_table", True) else None)
    if cfg.get("model", "hyperbolic") == "euclidean":
        # E2: euclidean-control checkpoints load through the same entry.
        # No curvature / tangent scale; downstream scoring must pass
        # euclidean=True (callers check cfg["model"]).
        from src.modelsv3.euclidean_v3 import EuclideanReasonerV3
        enc = EuclideanReasonerV3(
            node_feat_dim=g["x"].shape[1],
            edge_feat_dim=g["edge_descriptor"].shape[1],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            type_dim=cfg["type_dim"],
            num_edge_types_max=net,
            node_feat_dim_schema=g["node_descriptor"].shape[1],
        )
        enc.load_state_dict(torch.load(ckpt / "encoder.pt",
                                       map_location="cpu"))
        enc.to(device).eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        return enc, cfg
    enc = KettleGraphReasonerV3(
        node_feat_dim=g["x"].shape[1],
        edge_feat_dim=g["edge_descriptor"].shape[1],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        type_dim=cfg["type_dim"],
        c=cfg["curvature"],
        num_edge_types_max=net,
        node_feat_dim_schema=g["node_descriptor"].shape[1],
        tangent_scale_init=cfg["tangent_scale"],
    )
    enc.load_state_dict(torch.load(ckpt / "encoder.pt", map_location="cpu"))
    enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, cfg


@torch.no_grad()
def _embed(enc, g: dict, device) -> torch.Tensor:
    out = enc(
        g["x"].to(device),
        g["edge_index"].to(device),
        g["edge_type"].to(device),
        g["edge_descriptor"].to(device),
        node_descriptor=g["node_descriptor"].to(device),
    )
    return out.node_embeddings


def _query_input(cs) -> np.ndarray:
    """Return the per-case query input tensor for the head.
    Single-anchor cases → 32-d (just ``cs.query_vec``); pair-anchored
    cases → 40-d (``cs.query_vec`` concatenated with the 8-d 2nd-anchor
    identity stored in ``cs.query_vec_extra``)."""
    if cs.query_vec_extra is None:
        return cs.query_vec
    return np.concatenate([cs.query_vec, cs.query_vec_extra])


def _head_query_dim(first_case) -> int:
    return C.QUERY_DIM + (8 if first_case.query_vec_extra is not None else 0)


def _make_ranking_head(query_dim, cfg, device, with_abstain: bool):
    head = QueryToBall(
        query_dim=query_dim,
        hidden_dim=cfg["hidden_dim"],
        c=cfg["curvature"],
        euclidean=False,
        arch="qh0",
    ).to(device)
    if with_abstain:
        # Learnable abstain embedding lives on the head module so it
        # ships with state_dict and is restored on resume. Initialized
        # small to land near the origin (same recipe as encoder
        # node_in / QueryToBall tangent_scale=0.1).
        head.abstain_emb = nn.Parameter(
            torch.randn(cfg["hidden_dim"], device=device) * 0.01
        )
    return head


def _abstain_point(head, enc) -> torch.Tensor:
    """Map the head's learnable abstain tangent vector onto the ball."""
    return P.expmap0(head.abstain_emb, enc.c)


def _train_head(enc, train_cases, repos, cfg, args, device):
    """Dispatch on the family of the first training case. Returns a
    head module whose forward signature depends on family:

    * ``ranking`` / ``abstain_ranking`` → ``QueryToBall`` (with optional
      ``abstain_emb`` attribute for the abstain task).
    * ``classification`` → ``TaskClassifierHead``.

    The caller (``_run_fold`` / ``_run_fixed_split``) reads
    ``train_cases[0].task_family`` to dispatch eval correctly."""
    if not train_cases:
        # No training data — return a default ranking head; eval will be
        # all-model-near-random for this fold/task. Single-anchored.
        return _make_ranking_head(C.QUERY_DIM, cfg, device, with_abstain=False)
    family = train_cases[0].task_family
    if family == "classification":
        return _train_classifier_head(train_cases, cfg, args, device)
    return _train_ranking_head(enc, train_cases, repos, cfg, args, device,
                               family=family)


def _train_ranking_head(enc, train_cases, repos, cfg, args, device, family):
    query_dim = _head_query_dim(train_cases[0])
    with_abstain = (family == "abstain_ranking")
    head = _make_ranking_head(query_dim, cfg, device, with_abstain)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    epochs = 2 if args.quick else args.epochs
    for ep in range(epochs):
        order = rng.permutation(len(train_cases))
        tot, acc, nb = 0.0, 0.0, 0
        for idx in order:
            cs = train_cases[idx]
            rp = repos[cs.repo]
            pool = rp.pools.get(cs.task, np.empty(0, np.int64))
            # Keep ABSTAIN_ROW in hardnegs when present — for commit-
            # positive ABSTAIN_TARGET_RANKING cases the abstain sentinel
            # is the load-bearing hardneg ("when a real target exists,
            # do NOT abstain"). The candidate-build path below handles
            # the sentinel via head.abstain_emb (mapped onto the ball).
            negs = list(cs.hardneg_rows)
            if args.neg_sample > 0 and pool.size:
                negs += [
                    int(r)
                    for r in pool[rng.integers(0, pool.size, args.neg_sample)]
                ]
            # Order: positives first (including possible ABSTAIN_ROW
            # sentinel), then hard negs + sampled negs. Dedup preserves
            # first occurrence.
            cand = list(dict.fromkeys(cs.pos_rows + negs))
            posset = set(cs.pos_rows)
            # Defense-in-depth: cap candidate-set size per case. If the
            # generator ever emits an unbounded positives list (or a
            # future task does), an uncapped cand of 1000+ entries
            # turns each training step into a 1000+-row tensor build +
            # pairwise loss with quadratic per-pair work. Keeps positives
            # (which carry the ground-truth signal) over sampled negs.
            CAND_CAP = 256
            if len(cand) > CAND_CAP:
                keep_pos = [r for r in cand if r in posset or r == C.ABSTAIN_ROW]
                keep_rest = [r for r in cand if r not in posset and r != C.ABSTAIN_ROW]
                if len(keep_pos) >= CAND_CAP:
                    sel = rng.choice(len(keep_pos), CAND_CAP, replace=False)
                    cand = [keep_pos[int(i)] for i in sel]
                else:
                    n_rest = CAND_CAP - len(keep_pos)
                    if len(keep_rest) > n_rest:
                        sel = rng.choice(len(keep_rest), n_rest, replace=False)
                        keep_rest = [keep_rest[int(i)] for i in sel]
                    cand = keep_pos + keep_rest
            # Build candidate embedding tensor; ABSTAIN_ROW pulls from
            # the head's learnable abstain_emb, mapped onto the ball.
            real_rows = [r for r in cand if r != C.ABSTAIN_ROW]
            real_idx_map = {r: i for i, r in enumerate(real_rows)}
            real_t = torch.tensor(real_rows, dtype=torch.long, device=device)
            real_emb = rp.emb[real_t] if real_t.numel() else rp.emb[:0]
            if with_abstain and C.ABSTAIN_ROW in cand:
                abs_pt = _abstain_point(head, enc).unsqueeze(0)
                # Reconstruct in original cand order.
                pieces = []
                for r in cand:
                    if r == C.ABSTAIN_ROW:
                        pieces.append(abs_pt)
                    else:
                        pieces.append(real_emb[real_idx_map[r]].unsqueeze(0))
                cand_emb = torch.cat(pieces, dim=0)
            else:
                cand_emb = real_emb
                cand = real_rows  # drop sentinel from candidate list
            if cand_emb.shape[0] == 0:
                continue
            lab = torch.tensor(
                [1.0 if r in posset else 0.0 for r in cand], device=device
            )
            qp = head(torch.from_numpy(_query_input(cs)).to(device))
            loss, diag = pairwise_ranking_loss(
                qp, cand_emb, lab, c=enc.c, margin=cfg["margin"]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            acc += float(diag.get("rank_accuracy", 0.0))
            nb += 1
        if ep == 0 or ep == epochs - 1:
            print(
                f"      [ep {ep:>2}] loss={tot/max(nb,1):.4f} "
                f"rank_acc={acc/max(nb,1):.3f}  (n={len(train_cases)})"
            )
    head.eval()
    return head


def _train_classifier_head(train_cases, cfg, args, device):
    label_set = train_cases[0].label_set
    n_labels = C.LABEL_SET_SIZES[label_set]
    query_dim = _head_query_dim(train_cases[0])
    head = TaskClassifierHead(
        query_dim=query_dim,
        hidden_dim=cfg["hidden_dim"],
        n_labels=n_labels,
        c=cfg["curvature"],
    ).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    epochs = 2 if args.quick else args.epochs
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 64
    for ep in range(epochs):
        order = rng.permutation(len(train_cases))
        tot, hits, nb = 0.0, 0, 0
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            batch = [train_cases[int(i)] for i in batch_idx]
            qvecs = torch.from_numpy(
                np.stack([_query_input(cs) for cs in batch])
            ).to(device)
            labels = torch.tensor([cs.label for cs in batch],
                                  dtype=torch.long, device=device)
            logits = head(qvecs)
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * labels.shape[0]
            hits += int((logits.argmax(dim=-1) == labels).sum().item())
            nb += labels.shape[0]
        if ep == 0 or ep == epochs - 1:
            print(
                f"      [ep {ep:>2}] cls_loss={tot/max(nb,1):.4f} "
                f"acc={hits/max(nb,1):.3f}  (n={len(train_cases)}, "
                f"labels={n_labels})"
            )
    head.eval()
    return head


@torch.no_grad()
def _eval_case(head, enc, rp: Repo, cs, mode, kind, device):
    """Ranking + abstain_ranking eval. Returns (scores, labels, hardneg_mask)."""
    posset = set(cs.pos_rows)
    with_abstain = (cs.task_family == "abstain_ranking")
    # Build candidate list; the abstain sentinel is always added to the
    # within-mode candidate list (it's the case's "extra option") and
    # for pool-mode it's added so the model has the chance to pick it.
    if mode == "within":
        cand = list(dict.fromkeys(cs.pos_rows + cs.hardneg_rows))
    else:
        pool_rows = rp.pools.get(cs.task, np.empty(0, np.int64)).tolist()
        cand = sorted(set(pool_rows) | (posset - {C.ABSTAIN_ROW}))
        if with_abstain:
            cand = [C.ABSTAIN_ROW] + cand    # prepend so it's always present
    real_rows = [r for r in cand if r != C.ABSTAIN_ROW]

    lab = torch.tensor([1.0 if r in posset else 0.0 for r in cand])
    hardset = set(cs.hardneg_rows)
    hmask = torch.tensor([r in hardset for r in cand], dtype=torch.bool)

    if kind == "model":
        qp = head(torch.from_numpy(_query_input(cs)).to(device))
        real_t = torch.tensor(real_rows, dtype=torch.long, device=device)
        real_emb = rp.emb[real_t] if real_t.numel() else rp.emb[:0]
        if with_abstain and getattr(head, "abstain_emb", None) is not None and C.ABSTAIN_ROW in cand:
            abs_pt = _abstain_point(head, enc).unsqueeze(0)
            # Reconstruct cand order.
            real_idx_map = {r: i for i, r in enumerate(real_rows)}
            pieces = []
            for r in cand:
                if r == C.ABSTAIN_ROW:
                    pieces.append(abs_pt)
                else:
                    pieces.append(real_emb[real_idx_map[r]].unsqueeze(0))
            cand_emb = torch.cat(pieces, dim=0)
        else:
            cand_emb = real_emb
            if C.ABSTAIN_ROW in cand:
                # Head doesn't have an abstain_emb (e.g., no training
                # cases for this task in this fold). Score abstain as
                # neutral (zero distance proxy).
                # Reorder cand to drop the sentinel for scoring; we'll
                # also drop the corresponding lab/hmask entries.
                keep = [i for i, r in enumerate(cand) if r != C.ABSTAIN_ROW]
                lab = lab[keep]
                hmask = hmask[keep]
                cand = real_rows
        if cand_emb.shape[0] == 0:
            return (torch.zeros(0), torch.zeros(0),
                    torch.zeros(0, dtype=torch.bool))
        sc = score_from_embeddings(cand_emb, qp, c=enc.c).cpu()
    elif kind == "random":
        seed = int.from_bytes(cs.case_id.encode()[-8:], "little")
        sc = torch.from_numpy(
            np.random.default_rng(seed).standard_normal(len(cand)).astype(
                np.float32
            )
        )
    else:  # anchor: -shortest-path hops from the query node
        # ABSTAIN_ROW gets the worst possible distance (graph diameter +1)
        # so the BFS baseline never picks abstain — a fair structural
        # control: anchor heuristic CAN'T abstain, so the model has to
        # beat it on its own terms.
        d_q = rp.dist(cs.query_row)
        far = float(rp.n_nodes + 1)
        sc_vals = np.array(
            [-d_q[r] if r != C.ABSTAIN_ROW else -far for r in cand],
            dtype=np.float32,
        )
        sc = torch.from_numpy(sc_vals)
    return sc, lab, hmask


@torch.no_grad()
def _eval_classification_case(head, cs, kind, device):
    """Classification eval. Returns (pred, label, n_labels)."""
    n_labels = C.LABEL_SET_SIZES[cs.label_set]
    if kind == "model":
        if isinstance(head, TaskClassifierHead):
            qvec = torch.from_numpy(_query_input(cs)).to(device).unsqueeze(0)
            logits = head(qvec).squeeze(0).cpu()
            pred = int(logits.argmax().item())
        else:
            # No classifier was trained for this task (e.g., empty train);
            # fall back to "no signal" = always predict majority class.
            pred = 0
    elif kind == "random":
        seed = int.from_bytes(cs.case_id.encode()[-8:], "little")
        pred = int(
            np.random.default_rng(seed).integers(0, n_labels)
        )
    else:  # anchor: majority class (index 0 by LABEL_SETS construction)
        pred = 0
    return pred, cs.label, n_labels


def _load_repo(repo_dir: Path, out_dir: Path, idx, args, enc, cfg, device):
    name = repo_dir.name
    required = (
        set()
        if args.keep_answer_edges
        else C.collect_required_edges(repo_dir, TASKS)
    )
    cg = build_npz(
        repo_dir, out_dir / f"graph_{idx:02d}_{name}.npz", required
    )
    with np.load(cg.npz_path) as z:
        g = _build_graph_tensors(z)
        x = z["x"]
    if enc is None:
        enc, cfg = _build_encoder(Path(args.ckpt), g, device)
    emb = _embed(enc, g, device)
    cs, pools, stats = C.load_repo_cases(repo_dir, cg, x, TASKS, name)
    adj: list[list[int]] = [[] for _ in range(cg.n_nodes)]
    ei = g["edge_index"].numpy()
    for s, t in zip(ei[0], ei[1]):
        adj[int(s)].append(int(t))
        adj[int(t)].append(int(s))
    rp = Repo(
        name=name, emb=emb, pools=pools, cases=cs,
        n_nodes=cg.n_nodes, n_edges_kept=cg.n_edges_kept,
        answer_edges_ablated=len(required), _adj=adj, _dcache={},
    )
    return rp, enc, cfg, stats


def _load_pack_repo(pack_ctx, repo_summary, out_dir: Path, idx, args,
                    enc, cfg, device):
    """Pack-mode equivalent of ``_load_repo`` — consumes pre-decoded
    record iterators from the kgr_pack binary instead of reading jsonl.
    Same return shape (rp, enc, cfg, stats)."""
    from .pack_loader import PackContext  # noqa: F401 (type)
    name = repo_summary.name
    # Materialize the cases once; we need them both for required-edge
    # collection and for load_repo_cases. The generator can't be
    # rewound. For django at 660k cases this is ~700 MB of dicts —
    # acceptable per-repo, freed at the end of _load_pack_repo.
    cases_list = list(pack_ctx.iter_cases(repo_summary.idx))
    required = (
        set()
        if args.keep_answer_edges
        else C.collect_required_edges(Path("."), TASKS, cases_iter=cases_list)
    )
    cg = build_npz(
        Path(name),                                     # used only for name
        out_dir / f"graph_{idx:02d}_{name}.npz",
        required,
        nodes_iter=pack_ctx.iter_nodes(repo_summary.idx),
        edges_iter=pack_ctx.iter_edges(repo_summary.idx),
    )
    with np.load(cg.npz_path) as z:
        g = _build_graph_tensors(z)
        x = z["x"]
    if enc is None:
        enc, cfg = _build_encoder(Path(args.ckpt), g, device)
    emb = _embed(enc, g, device)
    cs, pools, stats = C.load_repo_cases(
        Path("."), cg, x, TASKS, name, cases_iter=cases_list,
    )
    adj: list[list[int]] = [[] for _ in range(cg.n_nodes)]
    ei = g["edge_index"].numpy()
    for s, t in zip(ei[0], ei[1]):
        adj[int(s)].append(int(t))
        adj[int(t)].append(int(s))
    rp = Repo(
        name=name, emb=emb, pools=pools, cases=cs,
        n_nodes=cg.n_nodes, n_edges_kept=cg.n_edges_kept,
        answer_edges_ablated=len(required), _adj=adj, _dcache={},
    )
    return rp, enc, cfg, stats


def main() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"[harness] TF32 on (matmul + cudnn), cudnn.benchmark on")
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", default="../corpus_validation")
    ap.add_argument("--extra-repo", action="append", default=[])
    ap.add_argument("--ckpt", default="runs/sweep_arch_hyp/h128_l4_seed1")
    ap.add_argument("--out", default="runs/codegraph_cv")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--neg-sample", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.5,
                    help="fraction of held-out repo files used as val "
                    "(rest = test) in leave-one-repo-out folds")
    ap.add_argument("--folds", type=int, default=0,
                    help="cap number of LORO folds (0 = all repos)")
    ap.add_argument("--repo-split", default="",
                    help="path to JSON with {'train': [...], 'test': [...]}. "
                    "Switches eval from LORO-CV to a fixed train/test split: "
                    "heads train ONCE on union of train_split cases, then "
                    "each test repo evals as its own 'fold'. Massive speedup "
                    "at 60-repo scale and the right design for a stratified "
                    "category holdout.")
    ap.add_argument("--keep-answer-edges", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="2 epochs — pipeline smoke test")
    ap.add_argument("--corpus-format", default="jsonl",
                    choices=["jsonl", "pack"],
                    help="'jsonl' = read per-repo training_cases.jsonl + "
                    "training_cases_v2.jsonl + nodes/edges.jsonl from "
                    "--corpus-root subdirs. 'pack' = read the kgr_pack "
                    "v0.2 binary corpus from --corpus-root (which must "
                    "be a pack directory containing header.bin, "
                    "cases.bin, etc.; train-debug mode required since "
                    "the harness needs original node ids for the "
                    "identity-vector seed). Pack mode is 45-50x smaller "
                    "on disk + 10-50x faster ingest at full corpus scale.")
    ap.add_argument("--task-families", nargs="+",
                    default=["ranking", "abstain_ranking", "classification"],
                    choices=["ranking", "abstain_ranking", "classification"],
                    help="restrict to listed task families; default=all 3")
    ap.add_argument("--max-cases-per-task", type=int, default=0,
                    help="Cap the number of *training* cases per task per "
                    "fold (deterministic subsample, seeded). 0 = no cap. "
                    "Required for the v0.2 corpus at 21+ repo scale, where "
                    "PARENT_SCOPE_RANKING alone exceeds 1M cases per fold "
                    "and per-case sequential training blows out wall time.")
    ap.add_argument("--max-eval-cases-per-task-per-repo", type=int, default=0,
                    help="Cap the number of *eval* cases per (held-out "
                    "repo, task) (deterministic subsample, seeded by "
                    "(seed, task, repo)). 0 = no cap. Required at 21+ "
                    "repo scale because per-case pool-mode eval allocates "
                    "the full (pool_size, hidden) embedding subset; with "
                    "scipy at 276k nodes / 99k CALL_TARGET cases that's "
                    "~80 min per task per repo before the cap.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # Filter TASKS by family selection at module-global TASKS so the
    # downstream pool computation / case loading is consistent.
    global TASKS
    fam_filter = set(args.task_families)
    TASKS = {t for t in C.TASK_IDS
             if C.DEFAULT_TASK_FAMILY.get(t, "ranking") in fam_filter}
    print(f"[task families] {sorted(fam_filter)}  "
          f"-> {len(TASKS)} task types active")

    pack_ctx = None
    if args.corpus_format == "pack":
        # Open the pack once; mmaps stay alive across all per-repo
        # loads. Closed at end of run.
        from .pack_loader import PackContext
        pack_ctx = PackContext(Path(args.corpus_root))
        repo_summaries = list(pack_ctx.iter_repos())
        if args.extra_repo:
            print(f"[1/4] pack mode: --extra-repo ignored ({args.extra_repo})")
        print(f"[1/4] pack {pack_ctx.dir} v0.2 "
              f"({len(repo_summaries)} repos, {pack_ctx.case_count:,} cases)")
        repo_names = [r.name for r in repo_summaries]
    else:
        repo_dirs = _discover_repos(Path(args.corpus_root), args.extra_repo)
        if not repo_dirs:
            raise SystemExit(
                f"no repos (dirs with nodes.jsonl) under {args.corpus_root} "
                f"or --extra-repo {args.extra_repo}"
            )
        repo_summaries = None
        repo_names = [p.name for p in repo_dirs]
        print(f"[1/4] {len(repo_dirs)} repos: {repo_names}")

    print("[2/4] ingest + embed each repo once (frozen encoder)")
    repos: dict[str, Repo] = {}
    enc, cfg = None, None
    if pack_ctx is not None:
        for i, rsum in enumerate(repo_summaries):
            rp, enc, cfg, stats = _load_pack_repo(
                pack_ctx, rsum, out_dir, i, args, enc, cfg, device
            )
            repos[rp.name] = rp
            print(f"      {rp.name}: {rp.n_nodes}n/{rp.n_edges_kept}e "
                  f"(-{rp.answer_edges_ablated} ans) "
                  f"{stats['n_cases']} cases {stats['task_counts']}")
    else:
        for i, rd in enumerate(repo_dirs):
            rp, enc, cfg, stats = _load_repo(
                rd, out_dir, i, args, enc, cfg, device
            )
            repos[rp.name] = rp
            print(f"      {rp.name}: {rp.n_nodes}n/{rp.n_edges_kept}e "
                  f"(-{rp.answer_edges_ablated} ans) "
                  f"{stats['n_cases']} cases {stats['task_counts']}")

    names = list(repos)
    acc = {k: CodeMetricAccumulator() for k in ("model", "random", "anchor")}
    cls_acc = {k: ClassificationMetricAccumulator()
               for k in ("model", "random", "anchor")}

    if args.repo_split:
        split = json.loads(Path(args.repo_split).read_text())
        train_names = [n for n in split["train"] if n in repos]
        test_names = [n for n in split["test"] if n in repos]
        missing = [n for n in split["train"] + split["test"] if n not in repos]
        print(f"[3/4] fixed split: train={len(train_names)} repos, "
              f"test={len(test_names)} repos"
              + (f"  (missing from corpus: {missing})" if missing else ""))
        # Mark cases so the eval loop can filter cleanly.
        for n in train_names:
            for c in repos[n].cases:
                c.split = "train"
        for n in test_names:
            for c in repos[n].cases:
                c.split = "test"
        train_cases = [c for n in train_names for c in repos[n].cases]
        _run_fixed_split(test_names, train_cases, repos, enc, cfg,
                         args, device, acc, cls_acc)
    elif len(names) >= 2:
        folds = names if args.folds <= 0 else names[: args.folds]
        print(f"[3/4] leave-one-repo-out CV over {len(folds)} folds")
        for held in folds:
            print(f"  -- fold: hold out '{held}' --")
            C.assign_file_split(
                repos[held].cases, args.split_seed,
                (0.0, args.val_frac, 1.0 - args.val_frac),
            )
            train_cases = [
                c for n in names if n != held for c in repos[n].cases
            ]
            _run_fold(held, train_cases, repos, enc, cfg, args, device,
                      acc, cls_acc)
    else:
        held = names[0]
        print("[3/4] single repo -> file-level 70/15/15 split")
        C.assign_file_split(
            repos[held].cases, args.split_seed, (0.70, 0.15, 0.15)
        )
        train_cases = [c for c in repos[held].cases if c.split == "train"]
        _run_fold(held, train_cases, repos, enc, cfg, args, device,
                  acc, cls_acc)

    print("[4/4] write report")
    report = {
        "config": vars(args),
        "encoder_config": cfg,
        "repos": {
            n: {
                "n_nodes": r.n_nodes,
                "n_edges_kept": r.n_edges_kept,
                "answer_edges_ablated": r.answer_edges_ablated,
                "n_cases": len(r.cases),
                "pool_sizes": {t: int(p.size) for t, p in r.pools.items()},
            }
            for n, r in repos.items()
        },
        "model": acc["model"].summary(),
        "baseline_random": acc["random"].summary(),
        "baseline_anchor": acc["anchor"].summary(),
        "model_classification": cls_acc["model"].summary(),
        "baseline_random_classification": cls_acc["random"].summary(),
        "baseline_anchor_classification": cls_acc["anchor"].summary(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    if report["model"]:
        print("\n=== model — CV (macro over folds) by split|mode ===")
        _tbl(report["model"].get("cv_by_split_mode", {}))
        print("\n=== model — CV by task|split|mode ===")
        _tbl(report["model"].get("cv_by_task_split_mode", {}))
        print("\n=== anchor-BFS baseline — CV by split|mode ===")
        _tbl(report["baseline_anchor"].get("cv_by_split_mode", {}))
        print("\n=== DE-LOCALIZED - ndcg@10 by split|mode|locality ===")
        print("    local = nearest positive at hop<=1 (anchor-BFS near-oracle "
              "by construction);")
        print("    nonlocal = hop>=2 (a genuine 'beats heuristic' test). "
              "Read the nonlocal|pool rows.")
        _tbl_delocalized(report)
    if report["model_classification"]:
        print("\n=== classification — CV by task|split ===")
        _tbl_cls(report["model_classification"].get("cv_by_task_split", {}))
        print("\n=== classification — random baseline by task|split ===")
        _tbl_cls(report["baseline_random_classification"].get(
            "cv_by_task_split", {}))
        print("\n=== classification — majority-class baseline by task|split ===")
        _tbl_cls(report["baseline_anchor_classification"].get(
            "cv_by_task_split", {}))
    print(f"\nreport: {out_dir / 'report.json'}")
    if pack_ctx is not None:
        pack_ctx.close()


def _task_family_of(task: str) -> str:
    return C.DEFAULT_TASK_FAMILY.get(task, "ranking")


def _eval_one(head, enc, rp, cs, device, acc, cls_acc, fold_name, split):
    """Per-case eval dispatch — calls the right helper based on family
    and pushes results into the right accumulator."""
    if cs.task_family == "classification":
        for kind in ("model", "random", "anchor"):
            pred, label, n_labels = _eval_classification_case(
                head, cs, kind, device
            )
            cls_acc[kind].add(fold_name, cs.task, split, pred, label, n_labels)
    else:
        locality = _case_locality(rp, cs)
        for mode in ("within", "pool"):
            for kind in ("model", "random", "anchor"):
                sc, lab, hm = _eval_case(
                    head, enc, rp, cs, mode, kind, device
                )
                acc[kind].add(fold_name, cs.task, split, mode, sc, lab, hm,
                              locality=locality)


def _case_locality(rp, cs) -> str:
    """Bucket a ranking/abstain case by the anchor-BFS hop distance from
    the query node to its nearest positive answer. "local" (hop<=1) means
    the answer is adjacent to the anchor, so the -shortest-path anchor
    baseline is near-oracle by construction; "nonlocal" (hop>=2) is where
    a "beats heuristic" claim is a genuine structural-reasoning test.
    "na" when the case has no scorable positive."""
    pos = [r for r in cs.pos_rows if r != C.ABSTAIN_ROW]
    if not pos:
        return "na"
    d_q = rp.dist(cs.query_row)
    min_hop = min(float(d_q[r]) for r in pos)
    return "local" if min_hop <= 1.0 else "nonlocal"


def _run_fold(held, train_cases, repos, enc, cfg, args, device, acc, cls_acc):
    for task in sorted(TASKS):
        tr = [c for c in train_cases if c.task == task]
        tr = _maybe_cap_training_cases(tr, task, args)
        print(f"    {task} ({_task_family_of(task)}): train n={len(tr)}")
        head = _train_head(enc, tr, repos, cfg, args, device)
        ev = [
            c for c in repos[held].cases
            if c.task == task and c.split in ("val", "test")
        ]
        ev = _maybe_cap_eval_cases(ev, task, held, args)
        if len(ev) and ev[0].split == ev[-1].split:
            pass  # all same split; common case
        for cs in ev:
            _eval_one(head, enc, repos[held], cs, device,
                      acc, cls_acc, held, cs.split)


def _stable_seed(*parts) -> int:
    """Deterministic 32-bit seed derived from arbitrary parts. Replaces
    Python's ``hash()`` which is process-randomized via PYTHONHASHSEED
    — using it for sampling means the same ``--seed`` produces a
    different training/eval subsample on every invocation, which broke
    cross-process reproducibility on the v0.1 regression. blake2b is
    cryptographic but cheap; we only need the digest as an int."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(key, digest_size=4).digest(), "little"
    )


def _maybe_cap_training_cases(tr: list, task: str, args) -> list:
    """Deterministic subsample of training cases when --max-cases-per-task
    is set. Seeded by (args.seed, task) so each task's subsample is
    reproducible across re-runs but independent across tasks (no
    correlated cap behaviour)."""
    cap = getattr(args, "max_cases_per_task", 0)
    if cap <= 0 or len(tr) <= cap:
        return tr
    rng = np.random.default_rng(_stable_seed(args.seed, task))
    idx = rng.choice(len(tr), size=cap, replace=False)
    return [tr[int(i)] for i in idx]


def _maybe_cap_eval_cases(ev: list, task: str, fold_name: str, args) -> list:
    """Deterministic subsample of eval cases per (held-out repo, task)
    when --max-eval-cases-per-task-per-repo is set. Pool-mode eval is
    O(pool_size) per case; for held-out scipy at ~276k nodes this is
    the load-bearing wall-time fix."""
    cap = getattr(args, "max_eval_cases_per_task_per_repo", 0)
    if cap <= 0 or len(ev) <= cap:
        return ev
    rng = np.random.default_rng(_stable_seed(args.seed, task, fold_name))
    idx = rng.choice(len(ev), size=cap, replace=False)
    return [ev[int(i)] for i in idx]


def _run_fixed_split(test_names, train_cases, repos, enc, cfg,
                     args, device, acc, cls_acc):
    """Train one head per task on the union train set, then loop test
    repos for eval. Saves ~N_test× head trainings vs running LORO with
    the same effective split."""
    for task in sorted(TASKS):
        tr = [c for c in train_cases if c.task == task]
        tr = _maybe_cap_training_cases(tr, task, args)
        print(f"    {task} ({_task_family_of(task)}): train n={len(tr)}")
        head = _train_head(enc, tr, repos, cfg, args, device)
        for held in test_names:
            ev = [
                c for c in repos[held].cases
                if c.task == task and c.split == "test"
            ]
            ev = _maybe_cap_eval_cases(ev, task, held, args)
            for cs in ev:
                _eval_one(head, enc, repos[held], cs, device,
                          acc, cls_acc, held, "test")


def _tbl(d: dict) -> None:
    cols = ["ndcg@10", "mrr", "r@10", "r@50", "pos_hardneg_margin",
            "folds", "n"]
    print(f"{'cell':<40} " + " ".join(f"{c:>9}" for c in cols))
    for k, v in d.items():
        print(
            f"{k:<40} "
            + " ".join(
                f"{v[c]:>9.3f}" if c not in ("folds", "n")
                else f"{int(v.get(c, 0)):>9}"
                for c in cols
            )
        )


def _tbl_delocalized(report: dict) -> None:
    """Stack model / anchor-BFS / random ndcg@10 side by side, split by
    anchor-adjacency, so the locality confound in the aggregate is explicit.
    The nonlocal|pool cell is the fair 'beats heuristic' headline."""
    m = report["model"].get("cv_by_locality_mode", {})
    a = report["baseline_anchor"].get("cv_by_locality_mode", {})
    r = report["baseline_random"].get("cv_by_locality_mode", {})
    cols = ["model", "anchor", "random", "folds", "n"]
    print(f"{'split|mode|locality':<32} " + " ".join(f"{c:>9}" for c in cols))
    for k in sorted(set(m) | set(a) | set(r)):
        cell = m.get(k) or a.get(k) or r.get(k) or {}
        vals = [
            m.get(k, {}).get("ndcg@10", float("nan")),
            a.get(k, {}).get("ndcg@10", float("nan")),
            r.get(k, {}).get("ndcg@10", float("nan")),
        ]
        print(
            f"{k:<32} "
            + " ".join(f"{v:>9.3f}" for v in vals)
            + f" {int(cell.get('folds', 0)):>9} {int(cell.get('n', 0)):>9}"
        )


def _tbl_cls(d: dict) -> None:
    cols = ["accuracy", "macro_f1", "folds", "n"]
    print(f"{'cell':<48} " + " ".join(f"{c:>10}" for c in cols))
    for k, v in d.items():
        print(
            f"{k:<48} "
            + " ".join(
                f"{v[c]:>10.3f}" if c not in ("folds", "n")
                else f"{int(v.get(c, 0)):>10}"
                for c in cols
            )
        )


if __name__ == "__main__":
    main()

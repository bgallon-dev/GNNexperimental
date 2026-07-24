r"""v3.2 reranker (MVP combo) — fixes the 3 WS3 root causes.

WS3 showed a v2 reranker closes only ~10-19% of the oracle gap because
it was (1) fed an off-task retriever's candidates, (2) trained with a
pointwise BCE/MSE loss, (3) trained on full graphs not the retriever's
hard top-C. This module fixes all three while REUSING the v2
architecture and writing the SAME hybrid-JSON schema as
``v2_reranker.py`` (so the existing comparison/report path is unchanged):

  retriever (FROZEN)  = v3.1 encoder + the WS2 per-task qh1 head
                        (or the v3.1 baseline for temporal). Defines the
                        candidate set: top-C by -hyp_dist(q, node).
  scorer (TRAINABLE)  = a fresh v2 KettleGraphReasoner (reused arch).
  reranking score     = retriever_score + scale * (v2_score - 0.5)
                        scale is a learned scalar INIT 0.0 -> at init the
                        order is EXACTLY the retriever's, so the reranker
                        is regression-proof by construction (the WS3
                        temporal regression cannot recur) and can only
                        ADD signal where it helps.
  loss                = candidate-restricted ListNet listwise CE over
                        the retriever's top-C (hard negatives by
                        construction), reusing ranking.py's listwise
                        pattern + pos/neg fallback, on scalar scores.

v3.3 recipe ablation (``--combine-mode blend``): the v3.2 residual above
(``v32``, the default, kept byte-identical) pins the retriever at unit
weight and squashes v2 through a bounded, sigmoid-saturated (v2-0.5)
term, so v2 can only nudge -- WS3's plain raw-v2 deployment beat it on
the geometry-sensitive tasks. ``blend`` instead combines
``a*z(retriever | cand) + b*z(logit(v2) | cand)``: v2 enters via its
*pre-sigmoid* logit (un-saturated discriminative range) and both signals
are z-standardized over the candidate set; ``a`` init 1.0 / ``b`` init
0.0 keeps the at-init order exactly the retriever's (regression-proof
preserved) while the learned b/a ratio CAN reach the v2-dominant regime.
This isolates whether v3.2's wall was the recipe, not the architecture
(same v2; no architectural commitment touched).

Architectural commitments: the v3 encoder stays frozen + query-agnostic
(retriever loaded read-only); the reranker is the downstream consumer
(not bound by tiny-by-design); never MSE; query lives only in v2.

Usage
-----
    py -m src.modelsv3.reranker_v32 \
        --retriever-run runs/sweep_taskdiversity/task0_seed0 \
        --task 0 --seed 0 --epochs 25 --topc 50 \
        --out runs/reranker_v32/task0_seed0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.corpus_dataset import CorpusDataset  # noqa: E402
from src.modelsv2.hyperbolic_gnnV2 import KettleGraphReasoner as V2  # noqa: E402
from src.modelsv3.distance_scoring import score_from_embeddings  # noqa: E402
from src.modelsv3.eval_candidate_recall import (  # noqa: E402
    _build_encoder,
    build_query_encoder,
)
from src.training.metrics import ndcg_at_k  # noqa: E402

K = 10
RELEVANCE = 0.5


def _load_retriever(retriever_run: Path, dataset: CorpusDataset):
    """Frozen v3.1 encoder + its query head (qh1 from a WS2 cell, or the
    v3.1 baseline). Read-only — the encoder commitment is preserved."""
    cfg = json.loads((retriever_run / "summary.json").read_text())["config"]
    enc = _build_encoder(cfg, dataset)
    enc.load_state_dict(torch.load(retriever_run / "encoder.pt",
                                   map_location="cpu"))
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    qe = build_query_encoder(cfg, dataset)
    qe.load_state_dict(torch.load(retriever_run / "query_encoder.pt",
                                  map_location="cpu"))
    qe.eval()
    for p in qe.parameters():
        p.requires_grad = False
    c = getattr(enc, "c", torch.tensor(float(cfg.get("curvature", 1.0))))
    return enc, qe, c, cfg["model"] == "euclidean", \
        cfg.get("query_head_arch", "qh0")


def _build_v2(dataset: CorpusDataset, hidden_dim: int, num_layers: int,
              type_dim: int) -> V2:
    """Fresh v2 scorer (reused architecture; same construction as
    v2_reranker._load_v2 but trainable / fresh-init)."""
    return V2(
        node_feat_dim=dataset.node_feat_dim,
        edge_feat_dim=dataset.edge_feat_dim_schema,
        query_dim=dataset.query_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        type_dim=type_dim,
        num_edge_types_max=dataset.num_edge_types_max,
        node_feat_dim_schema=dataset.node_feat_dim_schema,
        depth_attn=True,
    )


def _listnet_loss(combined_C: torch.Tensor, labels_C: torch.Tensor,
                  temperature: float) -> torch.Tensor:
    """ListNet listwise CE over the candidate set, reusing the
    ranking.py listwise pattern but on scalar scores (higher = more
    relevant). Graded labels -> graded target distribution; binary ->
    uniform over positives. Degenerate (no positive mass) -> 0."""
    lab = labels_C.clamp(0.0, 1.0)
    total = lab.sum()
    if total < 1e-9:
        return combined_C.new_zeros(())
    p = lab / total
    logits = combined_C / float(temperature)
    log_q = logits - torch.logsumexp(logits, dim=0)
    return -(p * log_q).sum()


def _combine(rs: torch.Tensor, cand: torch.Tensor, v2s: torch.Tensor,
             params, mode: str) -> torch.Tensor:
    """Combine the frozen retriever score with the trainable v2 score
    over the CANDIDATE set. Two recipes, selected by ``mode``:

    ``v32`` (legacy, byte-identical): z(retriever) + scale*(v2 - 0.5).
      ``params`` is the single scalar ``scale`` (init 0). The retriever
      coefficient is frozen at 1 and v2 enters through a bounded,
      sigmoid-saturated (v2-0.5) term -> v2 can only nudge.

    ``blend`` (v3.3): a*z(retriever | cand) + b*z(logit(v2) | cand).
      ``params`` is ``(a, b)``. Both signals are z-standardized over the
      candidate set (commensurate scales) and v2 enters through its
      *pre-sigmoid* logit (exact inverse of the sigmoid head), which
      restores v2's discriminative range exactly among the confident top
      candidates where the sigmoid saturates and flattens it. With
      a init 1.0 / b init 0.0 the combined score is exactly z(retriever)
      -> the candidate ORDER is exactly the retriever's, so the
      regression-proof-at-init property is preserved exactly; unlike
      ``v32`` the learned b/a ratio can reach the v2-dominant regime
      (the way WS3's raw-v2 deployment lets it)."""
    rc = rs[cand]
    rz = (rs - rc.mean()) / (rc.std() + 1e-6)
    if mode == "v32":
        scale = params
        return rz + scale * (v2s - 0.5)
    a, b = params
    v2l = torch.logit(v2s.clamp(1e-6, 1.0 - 1e-6))
    v2c = v2l[cand]
    v2z = (v2l - v2c.mean()) / (v2c.std() + 1e-6)
    return a * rz + b * v2z


def _rerank_vec(base: torch.Tensor, cand: torch.Tensor,
                vals: torch.Tensor) -> torch.Tensor:
    """Score vector that keeps non-candidates below all candidates and
    orders the candidates by ``vals`` — identical convention to
    v2_reranker._rerank_scores so ndcg is directly comparable to WS3."""
    out = torch.full_like(base, float("-inf"))
    out[cand] = vals
    return out


def _mrr(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    order = torch.argsort(scores, descending=True)[:k]
    for rank, idx in enumerate(order, start=1):
        if labels[idx] >= RELEVANCE:
            return 1.0 / rank
    return 0.0


def run(retriever_run: Path, task: int, seed: int, epochs: int, topc: int,
        lr: float, temperature: float, hidden_dim: int, num_layers: int,
        type_dim: int, corpus: str, out_dir: Path,
        combine_mode: str = "v32") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CorpusDataset(corpus_dir=corpus, split="train",
                             split_seed=0, include_tasks={task})
    val_ds = CorpusDataset(corpus_dir=corpus, split="val",
                           split_seed=0, include_tasks={task})
    enc, qe, c_val, euclidean, retr_arch = _load_retriever(
        retriever_run, train_ds)
    v2 = _build_v2(train_ds, hidden_dim, num_layers, type_dim)
    if combine_mode == "v32":
        scale = nn.Parameter(torch.tensor(0.0))  # init 0 -> retriever order
        blend_params = [scale]
        combine_args = scale
    else:  # blend: a*z(retr) + b*z(logit v2); init (1,0) -> retriever order
        a_p = nn.Parameter(torch.tensor(1.0))
        b_p = nn.Parameter(torch.tensor(0.0))
        blend_params = [a_p, b_p]
        combine_args = (a_p, b_p)
    opt = torch.optim.Adam(list(v2.parameters()) + blend_params, lr=lr)

    def _blend_snapshot() -> dict:
        if combine_mode == "v32":
            return {"scale": float(scale.detach())}
        return {"a": float(a_p.detach()), "b": float(b_p.detach())}

    def _retr_scores(s):
        with torch.no_grad():
            emb = enc(s.x, s.edge_index, s.edge_type, s.edge_descriptor,
                      node_descriptor=s.node_descriptor).node_embeddings
            rs = score_from_embeddings(emb, qe(s.query), c=c_val,
                                       euclidean=euclidean)
        return rs

    n_params = sum(p.numel() for p in v2.parameters()) + 1
    hist = []
    for ep in range(epochs):
        v2.train()
        order = np.random.permutation(len(train_ds))
        tot = 0.0
        nb = 0
        for i in order:
            s = train_ds[int(i)]
            rs = _retr_scores(s)
            C = min(topc, rs.numel())
            cand = torch.topk(rs, k=C, largest=True).indices
            v2s = v2(s.x, s.edge_index, s.edge_type, s.edge_descriptor,
                     s.query, node_descriptor=s.node_descriptor,
                     task_type=s.task_type).node_scores
            combined = _combine(rs.detach(), cand, v2s, combine_args,
                                combine_mode)
            loss = _listnet_loss(combined[cand], s.labels[cand],
                                 temperature)
            if not torch.isfinite(loss) or float(loss) == 0.0:
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(v2.parameters()) + blend_params, 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        hist.append({"epoch": ep, "mean_loss": tot / max(nb, 1),
                     **_blend_snapshot()})

    # ---- eval: retriever-alone vs +residual-v2 vs oracle, over the
    # SAME top-C, ndcg/mrr exactly as v2_reranker (WS3-comparable) ----
    v2.eval()
    rows = []
    with torch.no_grad():
        for j in range(len(val_ds)):
            s = val_ds[j]
            rs = _retr_scores(s)
            C = min(topc, rs.numel())
            cand = torch.topk(rs, k=C, largest=True).indices
            v2s = v2(s.x, s.edge_index, s.edge_type, s.edge_descriptor,
                     s.query, node_descriptor=s.node_descriptor,
                     task_type=s.task_type).node_scores
            combined = _combine(rs, cand, v2s, combine_args, combine_mode)
            retr = _rerank_vec(rs, cand, rs[cand])
            hyb = _rerank_vec(rs, cand, combined[cand])
            orc = _rerank_vec(rs, cand, s.labels[cand])
            rows.append({
                "retr_ndcg@10": ndcg_at_k(retr, s.labels, K),
                "retr_mrr@10": _mrr(retr, s.labels, K),
                "hybrid_ndcg@10": ndcg_at_k(hyb, s.labels, K),
                "hybrid_mrr@10": _mrr(hyb, s.labels, K),
                "oracle_ndcg@10": ndcg_at_k(orc, s.labels, K),
            })

    def _m(key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    v31 = _m("retr_ndcg@10")
    v31_mrr = _m("retr_mrr@10")
    trained_hyb = _m("hybrid_ndcg@10")
    trained_hyb_mrr = _m("hybrid_mrr@10")
    orc = _m("oracle_ndcg@10")
    gap = orc - v31

    # Validation-gated deployment ("do no harm"): the scale-init-0
    # residual is only regression-proof AT INIT; after training the
    # learned residual can overfit and DEGRADE an already-near-optimal
    # retriever order on val (observed on temporal). So ship the residual
    # ONLY if it strictly beats the retriever on the held-out val set;
    # otherwise fall back to scale=0 (pure retriever). This makes
    # no-regression a true DEPLOYMENT guarantee while still reporting the
    # ungated trained result for honest science.
    helps = trained_hyb > v31 + 1e-6
    dep_hyb = trained_hyb if helps else v31
    dep_hyb_mrr = trained_hyb_mrr if helps else v31_mrr
    if combine_mode == "v32":
        learned_blend = {"learned_scale": float(scale.detach())}
        deployed_blend = {"deployed_scale":
                          (float(scale.detach()) if helps else 0.0)}
        trained_blend = {"learned_scale": float(scale.detach())}
    else:
        la, lb = float(a_p.detach()), float(b_p.detach())
        learned_blend = {"learned_a": la, "learned_b": lb}
        # not deployed -> fall back to the init (a=1, b=0), which by
        # construction reproduces the retriever order exactly.
        deployed_blend = {"deployed_a": (la if helps else 1.0),
                          "deployed_b": (lb if helps else 0.0)}
        trained_blend = {"learned_a": la, "learned_b": lb}

    summary = {
        # v31_* mirror v2_reranker (WS3 1:1); "v3.1-alone" == per-task
        # qh1 retriever order. hybrid_* = the DEPLOYED (val-gated) model.
        "v31_ndcg@10": v31, "v31_mrr@10": v31_mrr,
        "hybrid_ndcg@10": dep_hyb, "hybrid_mrr@10": dep_hyb_mrr,
        "oracle_ndcg@10": orc,
    }
    res = {
        "task": task, "seed": seed,
        "retriever_run": str(retriever_run),
        "retriever_query_head": retr_arch,
        "n_samples": len(rows),
        "topc": topc, "epochs": epochs,
        "combine_mode": combine_mode,
        **learned_blend,
        **deployed_blend,
        "residual_deployed": bool(helps),
        "validation_gated": True,
        "n_params_v2": n_params,
        "summary": summary,
        # DEPLOYED (what ships): regression-free by construction.
        "hybrid_beats_retriever": bool(dep_hyb > v31),
        "gap_closed_frac": (float((dep_hyb - v31) / gap)
                            if gap > 1e-9 else None),
        "regression_vs_retriever": bool(dep_hyb < v31 - 1e-6),
        # TRAINED (ungated, honest science: did the residual learn?).
        "trained": {
            "hybrid_ndcg@10": trained_hyb,
            "hybrid_mrr@10": trained_hyb_mrr,
            "gap_closed_frac": (float((trained_hyb - v31) / gap)
                                if gap > 1e-9 else None),
            "regression_vs_retriever": bool(trained_hyb < v31 - 1e-6),
            **trained_blend,
        },
        "ceiling_oracle_ndcg@10": orc,
    }
    (out_dir / "hybrid.json").write_text(json.dumps(res, indent=2))
    (out_dir / "train_history.json").write_text(json.dumps(hist, indent=2))
    torch.save({"model_state": v2.state_dict(),
                **_blend_snapshot(),
                "cfg": {"hidden_dim": hidden_dim, "num_layers": num_layers,
                        "type_dim": type_dim, "task": task, "seed": seed,
                        "combine_mode": combine_mode}},
               out_dir / "reranker.pt")
    print(f"[rr32] task{task} seed{seed} mode={combine_mode}: retr={v31:.4f} "
          f"trained={trained_hyb:.4f} deployed={dep_hyb:.4f} "
          f"oracle={orc:.4f} deployed_gap={res['gap_closed_frac']} "
          f"residual_deployed={helps} "
          f"trained_regress={res['trained']['regression_vs_retriever']} "
          f"deployed_regress={res['regression_vs_retriever']}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retriever-run", type=str, required=True,
                    help="WS2 per-task qh1 cell (or v3.1 baseline) dir")
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--topc", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--type-dim", type=int, default=8)
    ap.add_argument("--corpus", type=str, default="src/data/corpus/tier1")
    ap.add_argument("--combine-mode", type=str, default="v32",
                    choices=["v32", "blend"],
                    help="v32 = legacy z(retr)+scale*(v2-0.5) residual; "
                         "blend = v3.3 a*z(retr|cand)+b*z(logit v2|cand)")
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()
    run(Path(a.retriever_run), a.task, a.seed, a.epochs, a.topc, a.lr,
        a.temperature, a.hidden_dim, a.num_layers, a.type_dim, a.corpus,
        Path(a.out), combine_mode=a.combine_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())

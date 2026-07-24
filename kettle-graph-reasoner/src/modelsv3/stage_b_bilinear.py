"""Opt-in bilinear Stage-B scoring head + faithful score-based losses.

The deployed Stage-B path scores nodes by ``-dist(QueryToBall(query),
node_emb)`` (distance to a single ball point). A multi-step diagnosis showed
that family structurally cannot express interval-overlap relevance (c'); a
general bilinear ``score = q^T M node_emb`` removes the wall (probe-validated).
This module is that head as REAL, opt-in Stage-B code.

The head + the two losses are **verbatim ports** of the functions that
produced the validated probe numbers:
  - ``BilinearStageBHead``      <- scripts/probe_bilinear_head.Bilinear:82-92
  - ``bilinear_pairwise_loss``  <- scripts/probe_bilinear_head._pair_hinge:57-79
  - ``bilinear_listwise_loss``  <- scripts/probe_stageb_objective._listwise:53-63
Fidelity is by identity: the math is copied, not re-derived, so a difference
vs the probe is a wiring bug, not a hypothesis result. ``ranking.py`` is NOT
modified (guarded core; semantics are replicated here, exactly as the probes
deliberately did).

Losses take a precomputed score vector (the head's output), mirroring the
probe structure ``s = head(q, emb); loss = _pair_hinge(s, lab)``. They return
``(loss, diag)`` with diag keys compatible with ``ranking.py`` so
``stage_b_history.json`` / sweep readers stay shape-stable.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class BilinearStageBHead(nn.Module):
    """``score(q, node_emb) = q^T M node_emb``  (general bilinear form;
    ``-dist``-to-a-point is a strict special case). ``query_dim`` ~ 18,
    ``hidden_dim`` ~ 128 -> ``M`` is ~2.3K params (far inside the
    tiny-by-design budget). Verbatim port of probe_bilinear_head.Bilinear:
    full ``M`` (no low-rank — that is an untested function class that would
    reopen the confound this build closes). Geometry-agnostic by design:
    no expmap0/curvature/tangent_scale (exactly what the probe validated).
    """

    def __init__(self, query_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.M = nn.Parameter(torch.zeros(query_dim, hidden_dim))
        nn.init.xavier_uniform_(self.M, gain=0.1)

    def forward(self, query: Tensor, node_emb: Tensor) -> Tensor:
        """``query``: ``(query_dim,)``  ``node_emb``: ``(N, hidden_dim)``
        -> ``(N,)`` scores (higher = more relevant)."""
        q = query.squeeze(0) if query.dim() == 2 else query
        return node_emb @ (self.M.t() @ q)


def bilinear_pairwise_loss(
    scores: Tensor,
    labels: Tensor,
    margin: float = 0.5,
    n_pairs: int = 16,
    pos_threshold: float = 0.75,
    neg_threshold: float = 0.25,
    rng: Optional[torch.Generator] = None,
) -> tuple[Tensor, dict]:
    """Verbatim ``probe_bilinear_head._pair_hinge`` semantics (itself a
    faithful replica of ``ranking.pairwise_ranking_loss``): with score
    ``s`` and ``d := -s``, ``relu(margin + d_pos - d_neg)`` ==
    ``relu(margin - (s_pos - s_neg))``. Same pos/neg thresholding +
    top/bottom-10% fallback + sampled pairs."""
    labels = labels.clamp(0.0, 1.0)
    N = labels.numel()
    pos = torch.nonzero(labels >= pos_threshold, as_tuple=False).flatten()
    neg = torch.nonzero(labels <= neg_threshold, as_tuple=False).flatten()
    if pos.numel() == 0 or neg.numel() == 0:
        _, si = torch.sort(labels, descending=True)
        k = max(1, N // 10)
        if pos.numel() == 0:
            pos = si[:k]
        if neg.numel() == 0:
            neg = si[-k:]
    if pos.numel() == 0 or neg.numel() == 0:
        z = scores.new_zeros(())
        return z, {"loss": 0.0, "n_valid_pairs": 0, "rank_accuracy": 0.0}
    ps = pos[torch.randint(0, pos.numel(), (n_pairs,), generator=rng)]
    ns = neg[torch.randint(0, neg.numel(), (n_pairs,), generator=rng)]
    s_pos = scores.index_select(0, ps)
    s_neg = scores.index_select(0, ns)
    loss = torch.relu(margin - (s_pos - s_neg)).mean()
    with torch.no_grad():
        diag = {
            "loss": float(loss.detach().item()),
            "n_valid_pairs": int(n_pairs),
            "mean_pos_score": float(s_pos.mean().item()),
            "mean_neg_score": float(s_neg.mean().item()),
            "rank_accuracy": float((s_pos > s_neg).float().mean().item()),
        }
    return loss, diag


def bilinear_listwise_loss(
    scores: Tensor,
    labels: Tensor,
    temperature: float = 1.0,
    eps: float = 1e-9,
) -> tuple[Tensor, dict]:
    """Verbatim ``probe_stageb_objective._listwise`` semantics (faithful
    replica of ``ranking.listwise_ranking_loss``): ``p = labels/sum``,
    ``logits = s/T``, ``loss = -(p * log_softmax(logits)).sum()``."""
    labels = labels.clamp(0.0, 1.0)
    total = labels.sum()
    if total < eps:
        z = scores.new_zeros(())
        return z, {"loss": 0.0, "entropy_p": 0.0, "entropy_q": 0.0}
    p = labels / total
    logits = scores / float(temperature)
    log_q = logits - torch.logsumexp(logits, dim=0)
    loss = -(p * log_q).sum()
    with torch.no_grad():
        pnz = p[p > 0]
        entropy_p = float(-(pnz * pnz.log()).sum().item())
        q = log_q.exp()
        qnz = q[q > 0]
        entropy_q = float(-(qnz * qnz.log()).sum().item())
        diag = {"loss": float(loss.detach().item()),
                "entropy_p": entropy_p, "entropy_q": entropy_q}
    return loss, diag


# --- downstream-safety guards (Layer B; Layer A = absence of query_encoder.pt)

def _stage_b_head(cfg) -> str:
    """Read the head tag from a Config object OR a summary.json['config']
    dict; default 'qtb' (backward-compatible with pre-existing runs)."""
    if isinstance(cfg, dict):
        return cfg.get("stage_b_head", "qtb")
    return getattr(cfg, "stage_b_head", "qtb")


def assert_qtb_run(cfg) -> None:
    """Raise if a run is NOT a standard QueryToBall run. Use before any
    code path that assumes query_encoder.pt -> ball point ->
    score_from_embeddings(-dist)."""
    h = _stage_b_head(cfg)
    if h != "qtb":
        raise ValueError(
            f"run has stage_b_head={h!r}, not 'qtb' — its scoring head is "
            f"NOT a QueryToBall ball-point/-dist head. Use the bilinear "
            f"eval path (scripts/eval_bilinear_hardened.py); do not load it "
            f"as a QueryToBall."
        )


def assert_bilinear_run(cfg) -> None:
    """Raise if a run is NOT a bilinear Stage-B run."""
    h = _stage_b_head(cfg)
    if h != "bilinear":
        raise ValueError(f"run has stage_b_head={h!r}, expected 'bilinear'")

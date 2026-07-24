r"""Stage-B ranking losses for KGR v3.

Stage A is InfoNCE (see contrastive.py). Stage B trains the
``QueryToBall`` head with the graph encoder frozen. The loss choice
here is **deliberately not MSE**: MSE on a scalar distance against
scalar labels reintroduces the pointwise / mean-collapse regime that
stage A was designed to escape — the query head would learn a mean-ish
query point that works okay on average for a task.

Instead:
  - ``pairwise_ranking_loss``: default. For each task example, sample a
    high-label ("positive") node and a low-label ("negative") node and
    enforce ``dist(q, n) > dist(q, p) + margin``.
  - ``listwise_ranking_loss``: optional fallback. Softmax over all
    nodes, using labels as soft targets — a ranking equivalent of
    cross-entropy that keeps the whole pipeline in the "exploit
    distance structure" regime.

Both operate on raw distances (``dist_p`` for hyperbolic,
``||u - v||_2`` for Euclidean). The ``euclidean`` flag propagates to
the same distance function used in ``distance_scoring.py``.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor

from ..modelsv2.layers import poincare_ops as P

Curvature = Union[float, Tensor]


def _pairwise_dist(
    query_point: Tensor,
    node_embeddings: Tensor,
    c: Curvature,
    euclidean: bool,
) -> Tensor:
    """Returns a 1-D ``(N,)`` distance vector from ``query_point`` to each node."""
    N = node_embeddings.size(0)
    q_exp = query_point.unsqueeze(0).expand(N, -1)
    if euclidean:
        return (node_embeddings - q_exp).norm(dim=-1, p=2)
    return P.dist(node_embeddings, q_exp, c, keepdim=False)


def pairwise_ranking_loss(
    query_point: Tensor,
    node_embeddings: Tensor,
    labels: Tensor,
    c: Curvature = 1.0,
    margin: float = 0.5,
    n_pairs: int = 16,
    pos_threshold: float = 0.75,
    neg_threshold: float = 0.25,
    euclidean: bool = False,
    rng: Optional[torch.Generator] = None,
) -> tuple[Tensor, dict]:
    r"""Margin-based pairwise ranking loss over (positive, negative) node pairs.

    For each sampled pair, loss += ``max(0, margin + dist(q, p) - dist(q, n))``.

    Parameters
    ----------
    query_point : (D,)
        Output of ``QueryToBall``.
    node_embeddings : (N, D)
        Frozen outputs of the v3 graph encoder.
    labels : (N,)
        Per-node relevance labels in [0, 1].
    margin : float
        Desired distance gap between positives and negatives.
    n_pairs : int
        Number of (p, n) pairs to sample per example.
    pos_threshold, neg_threshold : float
        Nodes with ``labels >= pos_threshold`` are positive candidates;
        ``labels <= neg_threshold`` are negative candidates. When either
        pool is empty, we fall back to "top-10%% vs bottom-10%%" on
        sorted labels so the loss is still well-defined.
    euclidean : bool
        Use L2 distance when True; hyperbolic ``dist_p`` otherwise.
    rng : torch.Generator, optional
        For deterministic sampling.

    Returns
    -------
    loss : scalar Tensor
    diag : dict
        ``n_valid_pairs``, ``mean_pos_dist``, ``mean_neg_dist``,
        ``rank_accuracy`` — fraction of sampled pairs with
        ``dist(q,p) < dist(q,n)``.
    """
    N = node_embeddings.size(0)
    labels = labels.clamp(0.0, 1.0)

    pos_idx = torch.nonzero(labels >= pos_threshold, as_tuple=False).flatten()
    neg_idx = torch.nonzero(labels <= neg_threshold, as_tuple=False).flatten()

    # Fallback: if either pool is empty, use top-10% / bottom-10% of the
    # label distribution. ndcg-style: at least *some* ranking signal exists
    # unless all labels are identical.
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        sorted_vals, sorted_idx = torch.sort(labels, descending=True)
        k = max(1, N // 10)
        if pos_idx.numel() == 0:
            pos_idx = sorted_idx[:k]
        if neg_idx.numel() == 0:
            neg_idx = sorted_idx[-k:]

    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        # Degenerate — all labels identical. Loss is well-defined as 0.
        zero = node_embeddings.new_zeros(())
        return zero, {
            "loss": 0.0,
            "n_valid_pairs": 0,
            "mean_pos_dist": 0.0,
            "mean_neg_dist": 0.0,
            "rank_accuracy": 0.0,
        }

    n_pairs = int(n_pairs)
    device = node_embeddings.device
    # sampling with replacement — cheap and fine for small pools.
    if rng is None:
        p_sample = pos_idx[torch.randint(0, pos_idx.numel(), (n_pairs,), device=device)]
        n_sample = neg_idx[torch.randint(0, neg_idx.numel(), (n_pairs,), device=device)]
    else:
        p_sample = pos_idx[
            torch.randint(0, pos_idx.numel(), (n_pairs,), generator=rng, device=device)
        ]
        n_sample = neg_idx[
            torch.randint(0, neg_idx.numel(), (n_pairs,), generator=rng, device=device)
        ]

    dist_all = _pairwise_dist(query_point, node_embeddings, c, euclidean)  # (N,)
    d_pos = dist_all.index_select(0, p_sample)  # (n_pairs,)
    d_neg = dist_all.index_select(0, n_sample)  # (n_pairs,)

    loss = torch.relu(margin + d_pos - d_neg).mean()

    with torch.no_grad():
        diag = {
            "loss": float(loss.detach().item()),
            "n_valid_pairs": int(n_pairs),
            "mean_pos_dist": float(d_pos.mean().item()),
            "mean_neg_dist": float(d_neg.mean().item()),
            "rank_accuracy": float((d_pos < d_neg).float().mean().item()),
        }
    return loss, diag


def listwise_ranking_loss(
    query_point: Tensor,
    node_embeddings: Tensor,
    labels: Tensor,
    c: Curvature = 1.0,
    temperature: float = 1.0,
    euclidean: bool = False,
    eps: float = 1e-9,
) -> tuple[Tensor, dict]:
    r"""Softmax-over-nodes listwise loss using labels as soft targets.

    Interprets labels as an unnormalized relevance distribution; the
    model's scores (``-dist``) define an induced distribution over nodes
    via softmax at ``temperature``. Loss is the KL-equivalent
    cross-entropy: ``-Σ p_i log q_i`` with ``p_i = labels / sum(labels)``.
    When all labels are zero the loss is defined as 0 (no signal).

    Useful as a fallback if ``pairwise_ranking_loss`` plateaus early —
    listwise gets signal from every node at once, which can help when
    the positive/negative pools are both small.
    """
    N = node_embeddings.size(0)
    labels = labels.clamp(0.0, 1.0)
    total = labels.sum()
    if total < eps:
        zero = node_embeddings.new_zeros(())
        return zero, {
            "loss": 0.0,
            "entropy_p": 0.0,
            "entropy_q": 0.0,
        }
    p = labels / total  # (N,)

    dist = _pairwise_dist(query_point, node_embeddings, c, euclidean)  # (N,)
    logits = -dist / float(temperature)
    log_q = logits - torch.logsumexp(logits, dim=0)  # (N,)
    loss = -(p * log_q).sum()

    with torch.no_grad():
        p_nonzero = p[p > 0]
        entropy_p = -(p_nonzero * p_nonzero.log()).sum()
        q = log_q.exp()
        q_nonzero = q[q > 0]
        entropy_q = -(q_nonzero * q_nonzero.log()).sum()
        diag = {
            "loss": float(loss.detach().item()),
            "entropy_p": float(entropy_p.item()),
            "entropy_q": float(entropy_q.item()),
        }
    return loss, diag


def _split_pos_neg(
    labels: Tensor, pos_threshold: float, neg_threshold: float
) -> tuple[Tensor, Tensor]:
    """Positive / negative candidate indices with the same top-10% /
    bottom-10% fallback as ``pairwise_ranking_loss`` (lines 99-111) so
    the InfoNCE loss is well-defined whenever any ranking signal exists."""
    N = labels.numel()
    pos_idx = torch.nonzero(labels >= pos_threshold, as_tuple=False).flatten()
    neg_idx = torch.nonzero(labels <= neg_threshold, as_tuple=False).flatten()
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        _, sorted_idx = torch.sort(labels, descending=True)
        k = max(1, N // 10)
        if pos_idx.numel() == 0:
            pos_idx = sorted_idx[:k]
        if neg_idx.numel() == 0:
            neg_idx = sorted_idx[-k:]
    return pos_idx, neg_idx


def sampled_infonce_ranking_loss(
    query_point: Tensor,
    node_embeddings: Tensor,
    labels: Tensor,
    c: Curvature = 1.0,
    n_negatives: int = 128,
    temperature: float = 1.0,
    n_positives: int = 8,
    pos_threshold: float = 0.75,
    neg_threshold: float = 0.25,
    euclidean: bool = False,
    rng: Optional[torch.Generator] = None,
) -> tuple[Tensor, dict]:
    r"""Multi-positive InfoNCE retrieval loss with SAMPLED negatives.

    v3.1 Phase 3 stage-B option. Unlike ``listwise_ranking_loss`` (full
    softmax over *all* nodes, single temperature, label-weighted
    targets), this draws a fixed pool of ``n_negatives`` hard-ish
    negatives (``labels <= neg_threshold``) and, for each of
    ``n_positives`` sampled positives, treats retrieval as a
    classification of "which point is the relevant one":

        L = mean_p  -log( e^{-d(q,p)/T}
                           / (e^{-d(q,p)/T} + Σ_n e^{-d(q,n)/T}) )

    This matches top-k retrieval far better than the pairwise hinge:
    every positive competes against a broad shared negative pool, so the
    query point is pulled to rank the *whole* relevant set above the
    irrelevant set (less per-batch rank-accuracy volatility). Never MSE.

    Returns ``(loss, diag)`` with ``rank_accuracy`` defined exactly as in
    ``pairwise_ranking_loss`` (fraction of (pos, neg) combinations with
    ``d(q,p) < d(q,n)``) so Phase-3 can compare its volatility against
    the pairwise baseline.
    """
    labels = labels.clamp(0.0, 1.0)
    pos_idx, neg_idx = _split_pos_neg(labels, pos_threshold, neg_threshold)
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        zero = node_embeddings.new_zeros(())
        return zero, {
            "loss": 0.0, "n_pos_used": 0, "n_neg_used": 0,
            "mean_pos_dist": 0.0, "mean_neg_dist": 0.0, "rank_accuracy": 0.0,
        }

    device = node_embeddings.device
    n_pos = int(min(n_positives, max(1, pos_idx.numel())))
    n_neg = int(n_negatives)

    def _draw(pool: Tensor, k: int) -> Tensor:
        if rng is None:
            sel = torch.randint(0, pool.numel(), (k,), device=device)
        else:
            sel = torch.randint(0, pool.numel(), (k,), generator=rng,
                                 device=device)
        return pool[sel]

    p_sample = _draw(pos_idx, n_pos)   # (n_pos,)
    n_sample = _draw(neg_idx, n_neg)   # (n_neg,)

    dist_all = _pairwise_dist(query_point, node_embeddings, c, euclidean)  # (N,)
    d_pos = dist_all.index_select(0, p_sample)  # (n_pos,)
    d_neg = dist_all.index_select(0, n_sample)  # (n_neg,)

    inv_t = 1.0 / float(temperature)
    # logits = -dist / T. For each positive, denominator = that positive
    # plus the shared negative pool.
    pos_logit = (-d_pos * inv_t).unsqueeze(1)            # (n_pos, 1)
    neg_logit = (-d_neg * inv_t).unsqueeze(0).expand(n_pos, -1)  # (n_pos, n_neg)
    logits = torch.cat([pos_logit, neg_logit], dim=1)    # (n_pos, 1+n_neg)
    log_denom = torch.logsumexp(logits, dim=1)           # (n_pos,)
    loss = -(pos_logit.squeeze(1) - log_denom).mean()

    with torch.no_grad():
        rank_acc = (d_pos.unsqueeze(1) < d_neg.unsqueeze(0)).float().mean()
        diag = {
            "loss": float(loss.detach().item()),
            "n_pos_used": int(n_pos),
            "n_neg_used": int(n_neg),
            "mean_pos_dist": float(d_pos.mean().item()),
            "mean_neg_dist": float(d_neg.mean().item()),
            "rank_accuracy": float(rank_acc.item()),
        }
    return loss, diag

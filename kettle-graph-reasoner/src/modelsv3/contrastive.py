r"""Poincaré-space InfoNCE + positive-pair sampler for KGR v3.

Stage-A training signal: contrastive on Poincaré-ball node embeddings.

Positive pairs (mixed per anchor, default 50/50):
  - Edge positive: random out-neighbor. Orthogonal to x because edges
    are not encoded in the node feature vector.
  - Same-label-different-features: same node-type one-hot AND low
    feature cosine (< low_cos_threshold). Forces the encoder to
    abstract node-type from graph context rather than feature cosine.
    Fallback chain: low-cos → any-same-label → edge → random.

Negatives: intra-graph only, all other nodes minus k-hop neighbors of
the anchor (default k=1). Inter-graph negatives would be too easy —
they'd teach the encoder to distinguish "my graph" from "other
graphs" rather than meaningful within-graph similarity.

Why InfoNCE, not triplet margin:
  ``dist_p`` diverges as ``||h|| → 1/√c``. A margin-based loss lets one
  boundary-hugging negative dominate the gradient and drag the anchor
  outward — the exact boundary-saturation failure mode documented in
  ``CLAUDE.md``. InfoNCE's log-sum-exp softly clamps already-far
  negatives' contribution, pairing cleanly with the radial-reg decay
  from the existing training recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from ..modelsv2.layers import poincare_ops as P


# Node-type one-hot occupies the first 12 dims of the 32-D node feature
# vector (see feature_encoder.encode_nodes and corpus_dataset.Sample).
NODE_TYPE_SLICE = slice(0, 12)


def _khop_mask(edge_index: np.ndarray, N: int, k: int) -> np.ndarray:
    """``(N, N)`` bool mask. ``out[i, j] = True`` iff ``j`` is within ``k``
    hops of ``i`` (treating edges as undirected for exclusion purposes;
    self-loops included). ``k <= 0`` returns the identity mask.

    For the tier-1 corpus (N ≲ 300, k ≤ 2) this is a one-shot cost.
    Kept for back-compat / tests; production sampling now uses
    ``_khop_neighbor_rows`` which never materializes the dense matrix.
    """
    if k <= 0:
        return np.eye(N, dtype=bool)
    A = np.zeros((N, N), dtype=bool)
    idx = np.arange(N)
    A[idx, idx] = True
    src, dst = edge_index[0], edge_index[1]
    A[src, dst] = True
    A[dst, src] = True  # undirected for exclusion
    reach = A.copy()
    A_int = A.astype(np.int32)
    for _ in range(k - 1):
        reach = reach | (reach.astype(np.int32) @ A_int > 0)
    return reach


def _khop_neighbor_rows(
    edge_index: np.ndarray, N: int, k: int
) -> list[np.ndarray]:
    """Per-node sorted int64 array of indices within ``k`` hops (incl.
    self, undirected). Memory ~ O(N + E·k) instead of O(N²) — required
    for large code graphs (pydantic at N=62k makes the dense mask
    14 GiB)."""
    if k <= 0:
        return [np.array([i], dtype=np.int64) for i in range(N)]

    from scipy.sparse import csr_matrix

    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    rows = np.concatenate([src, dst, np.arange(N, dtype=np.int64)])
    cols = np.concatenate([dst, src, np.arange(N, dtype=np.int64)])
    data = np.ones(rows.size, dtype=np.int8)
    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    A.data[:] = 1  # collapse duplicates to 1 (we treat as bool below)
    reach = A
    cur = A
    for _ in range(k - 1):
        cur = (cur @ A).astype(bool).astype(np.int8)
        reach = (reach + cur).astype(bool).astype(np.int8)
    reach = reach.astype(bool).tocsr()
    out: list[np.ndarray] = []
    indptr = reach.indptr
    indices = reach.indices.astype(np.int64)
    for i in range(N):
        out.append(np.sort(indices[indptr[i]:indptr[i + 1]]))
    return out


@dataclass
class SamplerBatch:
    """Output of ``PositiveSampler.sample``.

    ``valid_mask[a, n]`` is True iff node ``n`` is a valid candidate in
    anchor ``a``'s softmax — positive is included (True); self and
    k-hop neighbors are excluded (False).
    """

    anchor_idx: np.ndarray    # (A,) int64
    positive_idx: np.ndarray  # (A,) int64
    valid_mask: np.ndarray    # (A, N) bool


class PositiveSampler:
    """Mixed-signal positive-pair sampler for a single synthetic graph.

    Usage: build once per graph (cheap — pre-computes out-neighbor lists,
    node-type labels, normalized features, and the k-hop mask), then call
    ``sample(n_anchors)`` per training step.
    """

    def __init__(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        neighbor_exclude_k: int = 1,
        edge_fraction: float = 0.5,
        low_cos_threshold: float = 0.4,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if x.ndim != 2:
            raise ValueError(f"x must be (N, F); got shape {x.shape}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be (2, E); got shape {edge_index.shape}")
        self.N = int(x.shape[0])
        self.x = x.astype(np.float32, copy=False)
        self.edge_index = edge_index.astype(np.int64, copy=False)
        self.edge_fraction = float(edge_fraction)
        self.low_cos_threshold = float(low_cos_threshold)
        self.rng = rng if rng is not None else np.random.default_rng()

        # Per-node out-neighbor lists (bucket by src).
        src = self.edge_index[0]
        dst = self.edge_index[1]
        order = np.argsort(src, kind="stable")
        src_sorted = src[order]
        dst_sorted = dst[order]
        breaks = np.searchsorted(src_sorted, np.arange(self.N + 1))
        self._out_neighbors = [dst_sorted[breaks[i]:breaks[i + 1]] for i in range(self.N)]

        # Node-type label recovered from the one-hot block at x[:, 0:12].
        # -1 where the block is all-zero (shouldn't happen under the current
        # feature encoder but is a cheap guard for malformed rows).
        type_block = self.x[:, NODE_TYPE_SLICE]
        sums = type_block.sum(axis=1)
        self._node_type = np.where(sums > 0, type_block.argmax(axis=1), -1)

        # L2-normalized x for cosine similarity in the property-positive rule.
        norms = np.linalg.norm(self.x, axis=1, keepdims=True)
        self._x_unit = self.x / np.clip(norms, 1e-9, None)

        # k-hop exclusion neighbors as a per-node list of row indices.
        # We never materialize the dense (N, N) mask: at N~60k that costs
        # ~14 GiB. ``sample()`` builds an (A, N) row-slice on the fly.
        self._khop_rows = _khop_neighbor_rows(
            self.edge_index, self.N, int(neighbor_exclude_k)
        )

    def _edge_positive(self, i: int) -> Optional[int]:
        nbrs = self._out_neighbors[i]
        if nbrs.size == 0:
            return None
        return int(self.rng.choice(nbrs))

    def _property_positive(self, i: int) -> Optional[int]:
        label = self._node_type[i]
        if label < 0:
            return None
        same = np.nonzero((self._node_type == label) & (np.arange(self.N) != i))[0]
        if same.size == 0:
            return None
        cos = self._x_unit[same] @ self._x_unit[i]
        low_cos = same[cos < self.low_cos_threshold]
        if low_cos.size > 0:
            return int(self.rng.choice(low_cos))
        # Fallback within the property-positive tier: same-label regardless of cos.
        return int(self.rng.choice(same))

    def _sample_positive(self, i: int) -> int:
        prefer_edge = self.rng.random() < self.edge_fraction
        primary = self._edge_positive if prefer_edge else self._property_positive
        secondary = self._property_positive if prefer_edge else self._edge_positive
        p = primary(i)
        if p is None:
            p = secondary(i)
        if p is None:
            # Last-resort: random other node.
            choices = np.arange(self.N)
            choices = choices[choices != i]
            p = int(self.rng.choice(choices))
        return p

    def sample(self, n_anchors: int) -> SamplerBatch:
        n = min(int(n_anchors), self.N)
        anchors = self.rng.choice(self.N, size=n, replace=False).astype(np.int64)
        positives = np.empty(n, dtype=np.int64)
        for idx, a in enumerate(anchors):
            positives[idx] = self._sample_positive(int(a))

        # valid_mask[a] = everything NOT within k hops of anchor a.
        # Then re-enable the positive (in case it happens to be a k-hop neighbor,
        # which is common for edge positives since an out-neighbor is 1-hop).
        valid = np.ones((n, self.N), dtype=bool)
        for ai, a in enumerate(anchors):
            valid[ai, self._khop_rows[int(a)]] = False
        valid[np.arange(n), positives] = True
        return SamplerBatch(
            anchor_idx=anchors,
            positive_idx=positives,
            valid_mask=valid,
        )


def poincare_infonce(
    node_emb: Tensor,
    anchor_idx: Tensor,
    positive_idx: Tensor,
    valid_mask: Tensor,
    c,
    temperature: float = 1.0,
    use_tangent_approx: bool = False,
    n_neg_sample: int = 0,
    rng: torch.Generator | None = None,
    extra_neg_emb: Tensor | None = None,
) -> tuple[Tensor, dict]:
    r"""InfoNCE loss on hyperbolic node embeddings.

    Parameters
    ----------
    node_emb : (N, D)
        Poincaré-ball embeddings for one graph.
    anchor_idx : (A,) long
        Anchor node indices into ``node_emb``.
    positive_idx : (A,) long
        Positive node indices (one per anchor) into ``node_emb``.
    valid_mask : (A, N) bool
        True where node ``n`` is a valid candidate in anchor ``a``'s softmax.
        Positives must be True; self and k-hop neighbors should be False.
    c : float or Tensor
        Curvature.
    temperature : float
        Softmax temperature. Hyperbolic distances span roughly [0, 10];
        start at 1.0, sweep {1.0, 3.0, 10.0}. ``τ < 1`` typically saturates
        softmax and indicates origin collapse (distances bunched near 0).
    use_tangent_approx : bool
        If True, use ``-||logmap0(u) - logmap0(v)||`` as similarity instead
        of ``-dist_p``. Stability escape hatch; if it wins, raise
        ``--radial-reg-weight-end`` to prevent origin-clustering drift.
    n_neg_sample : int
        If > 0, sample this many random negatives per anchor from the
        valid pool instead of softmax-over-all-N. The positive is always
        kept. Memory and step time scale with K not N, which is the only
        thing that lets a 60-repo corpus with pydantic (62k nodes) train
        at h1024 in normal GPU budget. Statistically equivalent to full-N
        InfoNCE when N >> K (standard SimCLR / CLIP practice). When 0
        (default), full-N softmax — backwards-compatible with synthetic
        tier1 corpora where N ≲ 300.
    rng : torch.Generator | None
        Optional generator for deterministic negative sampling.

    Returns
    -------
    loss : scalar Tensor
    diag : dict[str, float]
        mean_pos_sim, mean_neg_sim, eff_negs_per_anchor, mean_h_norm,
        max_h_norm. Watch ``eff_negs_per_anchor`` — it silently collapses
        when the k-hop mask eats most of the negative pool in dense
        regions.
    """
    if anchor_idx.dim() != 1 or positive_idx.dim() != 1:
        raise ValueError("anchor_idx and positive_idx must be 1-D")
    if valid_mask.dim() != 2:
        raise ValueError("valid_mask must be 2-D (A, N)")
    A = anchor_idx.size(0)
    N, D = node_emb.shape
    if valid_mask.shape != (A, N):
        raise ValueError(f"valid_mask must be ({A}, {N}); got {tuple(valid_mask.shape)}")

    anchor_emb = node_emb.index_select(0, anchor_idx)  # (A, D)
    pos_emb = node_emb.index_select(0, positive_idx)   # (A, D)

    # --- subsampled negatives path ---------------------------------------
    # Cuts the (A, N, D) distance broadcast down to (A, K, D). Required
    # to fit the 60-repo corpus at h1024 in a single-GPU budget; full-N
    # softmax over pydantic (N=62k) needs ~16 GB just for one intermediate.
    if n_neg_sample > 0:
        K = int(n_neg_sample)
        device = node_emb.device
        # Per-anchor uniform sample from the valid pool (excluding the
        # positive — we add it explicitly so it's always in the softmax).
        valid_no_pos = valid_mask.clone()
        valid_no_pos.scatter_(1, positive_idx.unsqueeze(1), False)
        valid_counts = valid_no_pos.sum(dim=1).clamp_min(1)
        # Indices: sample with replacement from each anchor's row. This
        # is the standard SimCLR/CLIP approximation; collision rate is
        # K/N ~ 3% for K=2048 N=62k — negligible.
        rand = torch.rand(A, K, device=device, generator=rng)
        # Map [0, 1) -> [0, valid_count_a) per row, then map to actual
        # node ids via a cumulative-rank trick on the valid mask.
        ranks = (rand * valid_counts.unsqueeze(1).float()).long()  # (A, K)
        # cumsum gives, for each (a, n), the rank of n in a's valid pool.
        valid_cumsum = valid_no_pos.cumsum(dim=1)  # (A, N) long
        # For each (a, k), find n such that valid_cumsum[a, n] == ranks[a,k]+1
        # and valid_no_pos[a, n] is True. Use searchsorted along each row.
        neg_idx = torch.searchsorted(
            valid_cumsum, (ranks + 1).clamp_max(valid_counts.unsqueeze(1) - 1 + 1)
        )  # (A, K) long
        neg_idx = neg_idx.clamp_max(N - 1)

        neg_emb = node_emb[neg_idx]                                # (A, K, D)
        if use_tangent_approx:
            anc_tan = P.logmap0(anchor_emb, c)                     # (A, D)
            pos_tan = P.logmap0(pos_emb, c)                        # (A, D)
            neg_tan = P.logmap0(neg_emb, c)                        # (A, K, D)
            pos_dist = (anc_tan - pos_tan).norm(dim=-1, p=2)       # (A,)
            neg_dist = (anc_tan.unsqueeze(1) - neg_tan).norm(dim=-1, p=2)  # (A, K)
        else:
            pos_dist = P.dist(anchor_emb, pos_emb, c, keepdim=False)       # (A,)
            neg_dist = P.dist(
                anchor_emb.unsqueeze(1), neg_emb, c, keepdim=False
            )                                                              # (A, K)

        pos_sim = -pos_dist
        neg_sim = -neg_dist
        # Cross-graph negatives (detached queue rows from OTHER graphs):
        # appended to every anchor's softmax. Targets the pool-retrieval
        # ceiling — same-graph softmax never exceeds ~N=300 on tier1,
        # while eval pools are 3k-8k. Opt-in; None = bit-exact legacy.
        if extra_neg_emb is not None and extra_neg_emb.numel():
            if use_tangent_approx:
                x_tan = P.logmap0(extra_neg_emb, c)
                x_dist = (anc_tan.unsqueeze(1) - x_tan.unsqueeze(0)).norm(dim=-1, p=2)
            else:
                x_dist = P.dist(
                    anchor_emb.unsqueeze(1), extra_neg_emb.unsqueeze(0),
                    c, keepdim=False)
            neg_sim = torch.cat([neg_sim, -x_dist], dim=1)
        # Concat positive as column 0; softmax over (1 + K [+ M]).
        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / temperature
        logsumexp = torch.logsumexp(all_sim, dim=1)
        loss = -(pos_sim / temperature - logsumexp).mean()

        # Diagnostics (under no_grad to avoid keeping the (A, K, D)
        # graph any longer than needed).
        with torch.no_grad():
            h_norm = node_emb.norm(dim=-1, p=2)
            diag = {
                "loss": float(loss.detach().item()),
                "mean_pos_sim": float(pos_sim.mean().item()),
                "mean_neg_sim": float(neg_sim.mean().item()),
                "eff_negs_per_anchor": float(K),
                "mean_h_norm": float(h_norm.mean().item()),
                "max_h_norm": float(h_norm.max().item()),
            }
        return loss, diag
    # --- end subsampled path; full-N below (back-compat) -----------------

    if use_tangent_approx:
        tan = P.logmap0(node_emb, c)                          # (N, D)
        anc_tan = tan.index_select(0, anchor_idx)             # (A, D)
        diff = anc_tan.unsqueeze(1) - tan.unsqueeze(0)        # (A, N, D)
        neg_dist = diff.norm(dim=-1, p=2)                      # (A, N)
    else:
        u = anchor_emb.unsqueeze(1)                           # (A, 1, D)
        v = node_emb.unsqueeze(0)                             # (1, N, D)
        neg_dist = P.dist(u, v, c, keepdim=False)             # (A, N)

    sim = -neg_dist                                            # (A, N)

    # Mask invalid candidates with a large negative so softmax zeroes them.
    neg_inf = torch.finfo(sim.dtype).min
    masked_sim = torch.where(valid_mask, sim, torch.full_like(sim, neg_inf))

    pos_sim = sim.gather(1, positive_idx.unsqueeze(1)).squeeze(1)  # (A,)
    logsumexp = torch.logsumexp(masked_sim / temperature, dim=1)    # (A,)
    loss = -(pos_sim / temperature - logsumexp).mean()

    with torch.no_grad():
        pos_mask = torch.zeros_like(valid_mask)
        pos_mask.scatter_(1, positive_idx.unsqueeze(1), True)
        neg_only = valid_mask & ~pos_mask
        neg_counts = neg_only.sum(dim=1).float()
        safe = neg_counts.clamp_min(1.0)
        mean_neg_per_anchor = (
            torch.where(neg_only, sim, torch.zeros_like(sim)).sum(dim=1) / safe
        )
        h_norm = node_emb.norm(dim=-1, p=2)
        diag = {
            "loss": float(loss.detach().item()),
            "mean_pos_sim": float(pos_sim.mean().item()),
            "mean_neg_sim": float(mean_neg_per_anchor.mean().item()),
            "eff_negs_per_anchor": float(neg_counts.mean().item()),
            "mean_h_norm": float(h_norm.mean().item()),
            "max_h_norm": float(h_norm.max().item()),
        }

    return loss, diag

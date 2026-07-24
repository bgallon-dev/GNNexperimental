r"""Euclidean-v3 — same-architecture baseline for the geometric claim.

Mandatory baseline per the v3 plan. v3 (hyperbolic + embedding-first +
contrastive) vs v1/v2 (hyperbolic + score-first + pointwise) doesn't
isolate geometry — the regime differs. Euclidean-v3 (Euclidean +
embedding-first + contrastive) isolates geometry within the new
regime.

Architecture:
  - Input projection: Euclidean ``Linear`` (no ``expmap0``, no
    ``tangent_scale``).
  - ``SchemaEncoder``: identical to v3.
  - ``EdgeTypedAttention`` in Euclidean mode (``euclidean=True`` skips
    the ``logmap0`` wrapper).
  - ``EuclideanMessagePassing`` (reused from v1's
    ``euclidean_plus_baseline.py``): weighted neighbor sum → linear →
    ReLU. Consumes the same per-edge attention weights as the
    hyperbolic MP stack.
  - ``DepthAttentionEuclidean``: same structure as v3's
    ``DepthAttention`` but operates directly on Euclidean embeddings
    (no ``logmap0``/``expmap0`` round-trip needed — the snapshots
    already live in a flat space).

What's intentionally NOT here:
  - No curvature, no ``_c``, no ``_sync_c``.
  - No ``tangent_scale`` (Euclidean scale is unregularized).
  - No radial-reg (no "ball boundary" to fear).
  - No intra-stack depth re-mixing — keeping that in would require
    an encoder round-trip through tangent/ball that Euclidean
    doesn't have; the simpler version still implements the final
    depth-attention aggregation, matching v3's common case.

Scoring adapter: ``distance_scoring.score_from_embeddings(...,
euclidean=True)``. Same ranking path as v3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor, nn

from ..models.euclidean_plus_baseline import EuclideanMessagePassing
from ..modelsv2.layers.edge_attention import EdgeTypedAttention
from ..modelsv2.layers.schema_encoder import SchemaEncoder
from .hyperbolic_gnnV3 import _make_rmsnorm


class DepthAttentionEuclidean(nn.Module):
    """Depth-attention with Euclidean snapshots. Structure mirrors
    ``hyperbolic_gnnV3.DepthAttention`` so parameter counts line up at
    matched ``hidden_dim`` / ``num_layers``."""

    def __init__(self, num_layers: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.depth_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_layers)
        )
        self.key_norm = _make_rmsnorm(hidden_dim)

    def forward(self, snapshots: List[Tensor], query_idx: int) -> Tensor:
        stack = torch.stack(snapshots, dim=0)           # (L', N, D)
        keys = self.key_norm(stack)                      # (L', N, D)
        q = self.depth_queries[query_idx]                # (D,)
        logits = torch.einsum("d,lnd->ln", q, keys)      # (L', N)
        alpha = logits.softmax(dim=0)                    # (L', N)
        return (alpha.unsqueeze(-1) * stack).sum(dim=0)  # (N, D)


@dataclass
class EuclideanEmbeddingOutput:
    node_embeddings: Tensor                  # (N, hidden_dim) Euclidean
    edge_type_embeddings: Tensor             # (T, type_dim)
    per_round_embeddings: Optional[List[Tensor]] = None


class EuclideanReasonerV3(nn.Module):
    """Euclidean-geometry counterpart to ``KettleGraphReasonerV3``.

    Same ``forward`` signature (minus curvature). Produces per-node
    Euclidean embeddings; downstream scoring uses L2 distance via
    ``distance_scoring.score_from_embeddings(..., euclidean=True)``.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        type_dim: int = 8,
        num_edge_types_max: Optional[int] = None,
        node_feat_dim_schema: Optional[int] = None,
        depth_attn: bool = True,
        log_depth: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.type_dim = type_dim
        self.depth_attn_enabled = bool(depth_attn)
        self.log_depth = bool(log_depth)

        # Match hyperbolic-v3's small-gain Xavier init so initial
        # embedding magnitudes are comparable across arms.
        self.node_in = nn.Linear(node_feat_dim, hidden_dim)
        nn.init.xavier_uniform_(self.node_in.weight, gain=0.05)
        nn.init.zeros_(self.node_in.bias)

        self.schema_encoder = SchemaEncoder(
            edge_feat_dim=edge_feat_dim,
            type_dim=type_dim,
            node_feat_dim=node_feat_dim_schema,
        )

        self.attn_layers = nn.ModuleList(
            EdgeTypedAttention(
                node_dim=hidden_dim,
                num_edge_types=num_edge_types_max,
                type_dim=type_dim,
                euclidean=True,
            )
            for _ in range(num_layers)
        )
        self.mp_layers = nn.ModuleList(
            EuclideanMessagePassing(in_dim=hidden_dim, out_dim=hidden_dim)
            for _ in range(num_layers)
        )
        self.depth_attention: Optional[DepthAttentionEuclidean] = (
            DepthAttentionEuclidean(num_layers=num_layers, hidden_dim=hidden_dim)
            if self.depth_attn_enabled
            else None
        )

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_descriptor: Tensor,
        node_descriptor: Optional[Tensor] = None,
    ) -> EuclideanEmbeddingOutput:
        h = self.node_in(node_features)
        edge_type_emb, _ = self.schema_encoder(edge_descriptor, node_descriptor)

        per_round: Optional[List[Tensor]] = [] if self.log_depth else None
        snapshots: List[Tensor] = []
        for attn, mp in zip(self.attn_layers, self.mp_layers):
            alpha = attn(h, edge_index, edge_type, type_emb_override=edge_type_emb)
            h = mp(h, edge_index, edge_weight=alpha)
            snapshots.append(h)
            if per_round is not None:
                per_round.append(h)

        if self.depth_attention is not None:
            h = self.depth_attention(snapshots, query_idx=self.num_layers - 1)

        return EuclideanEmbeddingOutput(
            node_embeddings=h,
            edge_type_embeddings=edge_type_emb,
            per_round_embeddings=per_round,
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

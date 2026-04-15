r"""Euclidean+ baseline — KGR minus the Poincaré ball.

Purpose: isolate the geometry variable. Shares every non-geometric component
with ``KettleGraphReasoner`` (SchemaEncoder, EdgeTypedAttention, depth head
supervision, radial / L2 norm regularizer) so the only difference against the
hyperbolic model is whether message passing happens on the Poincaré ball or
in flat Euclidean space.

Three-way read after this lands:

    vanilla Euclidean GAT (euclidean)   — tests "did the scaffolding help?"
    Euclidean+                           — tests geometry alone
    Hyperbolic (kettle)                  — full system

Attention runs in Euclidean mode (``EdgeTypedAttention(euclidean=True)`` skips
the ``logmap0`` wrapper). Message passing is a straight weighted neighbor
sum + linear + activation — the natural Euclidean analogue of
``HyperbolicMessagePassing``. Small-gain init is preserved on the message-pass
linears so the init-time embedding scale matches the hyperbolic model.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor, nn

from .hyperbolic_gnn import KGROutput
from .layers.edge_attention import EdgeTypedAttention
from .layers.hyp_message_pass import _scatter_add
from .layers.schema_encoder import SchemaEncoder


class EuclideanMessagePassing(nn.Module):
    """Weighted neighbor sum → linear → activation. Consumes per-edge weights
    from ``EdgeTypedAttention`` the same way ``HyperbolicMessagePassing`` does,
    but without any ball ops."""

    def __init__(self, in_dim: int, out_dim: int, use_bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.weight, gain=0.05)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must be shape (2, E)")
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        msg = x.index_select(0, src)
        if edge_weight is None:
            deg = _scatter_add(
                torch.ones(src.size(0), 1, dtype=x.dtype, device=x.device),
                dst,
                dim_size=N,
            ).clamp_min(1.0)
            w = 1.0 / deg.index_select(0, dst)
        else:
            w = edge_weight.unsqueeze(-1) if edge_weight.dim() == 1 else edge_weight
        msg = msg * w

        agg = _scatter_add(msg, dst, dim_size=N)
        h = agg @ self.weight.t()
        if self.bias is not None:
            h = h + self.bias
        return torch.relu(h)


class EuclideanPlusBaseline(nn.Module):
    """Geometry-only ablation of ``KettleGraphReasoner``."""

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        query_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        type_dim: int = 8,
        num_edge_types_max: Optional[int] = None,
        node_feat_dim_schema: Optional[int] = None,
        log_depth: bool = False,
        **_ignored,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.type_dim = type_dim
        self.log_depth = bool(log_depth)

        # Small-gain init on the input projection matches the hyperbolic model,
        # so initial embedding scales are comparable across both.
        self.node_in = nn.Linear(node_feat_dim, hidden_dim)
        nn.init.xavier_uniform_(self.node_in.weight, gain=0.05)
        nn.init.zeros_(self.node_in.bias)

        self.query_in = nn.Linear(query_dim, hidden_dim)

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

        self.node_score = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_score = nn.Sequential(
            nn.Linear(hidden_dim * 2 + type_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_descriptor: Tensor,
        query: Tensor,
        node_descriptor: Optional[Tensor] = None,
    ) -> KGROutput:
        h = self.node_in(node_features)
        q = self.query_in(query.view(-1)) if query.dim() == 1 else self.query_in(query).view(-1)

        edge_type_emb, _ = self.schema_encoder(edge_descriptor, node_descriptor)

        per_round: Optional[List[Tensor]] = [] if self.log_depth else None
        for attn, mp in zip(self.attn_layers, self.mp_layers):
            alpha = attn(h, edge_index, edge_type, type_emb_override=edge_type_emb)
            h = mp(h, edge_index, edge_weight=alpha)
            if per_round is not None:
                per_round.append(h)

        N = h.size(0)
        q_exp = q.unsqueeze(0).expand(N, -1)
        node_logits = self.node_score(torch.cat([h, q_exp], dim=-1)).squeeze(-1)
        node_scores = torch.sigmoid(node_logits)

        src, dst = edge_index[0], edge_index[1]
        h_s = h.index_select(0, src)
        h_d = h.index_select(0, dst)
        t_r = edge_type_emb.index_select(0, edge_type)
        E = edge_index.size(1)
        q_e = q.unsqueeze(0).expand(E, -1)
        edge_logits = self.edge_score(torch.cat([h_s, h_d, t_r, q_e], dim=-1)).squeeze(-1)
        edge_scores = torch.sigmoid(edge_logits)

        return KGROutput(
            node_scores=node_scores,
            edge_scores=edge_scores,
            node_embeddings=h,
            edge_type_embeddings=edge_type_emb,
            per_round_embeddings=per_round,
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

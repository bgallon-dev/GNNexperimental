r"""Euclidean homogeneous GAT baseline.

The *control condition* for the KGR experiment. Mirrors the API of
``KettleGraphReasoner`` exactly — same ``forward`` signature, same
``KGROutput`` return type — so the training and evaluation pipelines are
drop-in comparable. But the representation is Euclidean (no Poincaré
ball, no Möbius ops) and attention is homogeneous (no edge typing).

This isolates the combined effect of the two experimental ingredients
(hyperbolic geometry + edge-typed attention) against the standard GNN
stack they replace. Per the research hypothesis in CLAUDE.md, the
experimental model should beat this baseline at equivalent parameter
count on structural reasoning tasks. If it doesn't, the geometry and
attention story are wrong.

The ``edge_type``, ``edge_descriptor``, and ``node_descriptor`` inputs are
accepted and ignored; they exist only so the two models are
signature-compatible. That is intentional: a fair A/B needs the same
data to flow into both.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .hyperbolic_gnn import KGROutput
from .layers.hyp_message_pass import _scatter_add


class EuclideanGATLayer(nn.Module):
    """Single GAT-style layer, Euclidean, homogeneous (no edge types)."""

    def __init__(self, in_dim: int, out_dim: int, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(out_dim))
        self.a_dst = nn.Parameter(torch.empty(out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.negative_slope = negative_slope
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.view(1, -1))
        nn.init.xavier_uniform_(self.a_dst.view(1, -1))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        N = x.size(0)
        h = self.W(x)  # (N, out_dim)
        src, dst = edge_index[0], edge_index[1]

        # GAT score: a_src · h_j + a_dst · h_i (separable, standard GATv1 form).
        e_src = (h * self.a_src).sum(dim=-1)  # (N,)
        e_dst = (h * self.a_dst).sum(dim=-1)  # (N,)
        raw = torch.nn.functional.leaky_relu(
            e_src.index_select(0, src) + e_dst.index_select(0, dst),
            negative_slope=self.negative_slope,
        )  # (E,)

        # Scatter-softmax over incoming edges per receiver.
        per_recv_max = torch.full(
            (N,), float("-inf"), dtype=raw.dtype, device=raw.device
        ).scatter_reduce(0, dst, raw, reduce="amax", include_self=False)
        per_recv_max = torch.where(
            torch.isfinite(per_recv_max), per_recv_max, torch.zeros_like(per_recv_max)
        )
        exp = (raw - per_recv_max.index_select(0, dst)).exp()
        denom = _scatter_add(exp.unsqueeze(-1), dst, dim_size=N).squeeze(-1).clamp_min(1e-15)
        alpha = exp / denom.index_select(0, dst)

        # Aggregate.
        msg = h.index_select(0, src) * alpha.unsqueeze(-1)
        agg = _scatter_add(msg, dst, dim_size=N)
        return torch.nn.functional.elu(agg + self.bias)


class EuclideanBaseline(nn.Module):
    """Euclidean homogeneous-GAT counterpart to ``KettleGraphReasoner``."""

    _zero_type: Tensor

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,         # accepted for API parity; unused
        query_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        type_dim: int = 8,          # accepted for API parity; unused
        **_ignored,                 # soak up KGR-only kwargs (c, learnable_c, ...)
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Kept so edge-score-head width matches the experimental model exactly
        # (identical parameter budget at matching hidden_dim).
        self.type_dim = type_dim

        self.node_in = nn.Linear(node_feat_dim, hidden_dim)
        self.query_in = nn.Linear(query_dim, hidden_dim)

        self.gat_layers = nn.ModuleList(
            EuclideanGATLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        )

        # Dummy edge-type embedding (unused semantically, zeros) so the
        # edge-score head has an identical input width to the experimental
        # model. This keeps parameter-count A/B comparisons clean.
        self.register_buffer("_zero_type", torch.zeros(type_dim))

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
        edge_type: Tensor,                     # unused
        edge_descriptor: Tensor,               # unused
        query: Tensor,
        node_descriptor: Optional[Tensor] = None,  # unused
    ) -> KGROutput:
        del edge_type, edge_descriptor, node_descriptor  # noqa: F841  (API parity)

        h = self.node_in(node_features)
        q = self.query_in(query.view(-1)) if query.dim() == 1 else self.query_in(query).view(-1)

        for layer in self.gat_layers:
            h = layer(h, edge_index)

        N = h.size(0)
        q_exp = q.unsqueeze(0).expand(N, -1)
        node_logits = self.node_score(torch.cat([h, q_exp], dim=-1)).squeeze(-1)
        node_scores = torch.sigmoid(node_logits)

        src, dst = edge_index[0], edge_index[1]
        E = edge_index.size(1)
        h_s = h.index_select(0, src)
        h_d = h.index_select(0, dst)
        t_r = self._zero_type.unsqueeze(0).expand(E, -1)
        q_e = q.unsqueeze(0).expand(E, -1)
        edge_logits = self.edge_score(torch.cat([h_s, h_d, t_r, q_e], dim=-1)).squeeze(-1)
        edge_scores = torch.sigmoid(edge_logits)

        return KGROutput(
            node_scores=node_scores,
            edge_scores=edge_scores,
            node_embeddings=h,
            edge_type_embeddings=torch.zeros(
                0, self.type_dim, dtype=h.dtype, device=h.device
            ),
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

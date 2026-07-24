r"""KettleGraphReasonerV3 — embedding-first, query-agnostic.

Forked from ``src.modelsv2.hyperbolic_gnnV2``. Differences from v2:

- **No scoring heads.** ``node_score`` / ``edge_score`` and their
  hierarchy-subspace variants are removed; the model outputs ball-space
  embeddings only. Scoring is a downstream operation: distance from a
  query point, a reference node, or a cluster centroid.
- **No query input.** ``query_in`` is removed. Stage-A (contrastive
  pretraining) sees only graph structure + node features. Queries enter
  in stage B through a separate ``QueryToBall`` module with this encoder
  frozen.
- **No hierarchy subspace.** The ``hierarchy_subspace_dim`` knob was a
  scoring-head partition; without scoring heads it has no meaning.
  ``DepthAttention`` is constructed with ``k=0``.
- **No hyp/init distance features.** Those were scoring-head conditioning
  signals; irrelevant for raw embeddings.

Kept verbatim from v2:
- Euclidean→tangent→ball input projection, small-gain Xavier (``gain=0.05``)
  on ``node_in``, learnable ``tangent_scale`` (init 0.1).
- ``SchemaEncoder`` + ``EdgeTypedAttention`` + ``HyperbolicMessagePassing``
  stack with optional intra-stack depth-attention re-mixing.
- ``DepthAttention`` final aggregation with RMSNorm'd keys and
  zero-initialized depth queries.
- ``_sync_c`` for learnable curvature propagation.

Output contract: ``node_embeddings`` are Poincaré-ball points (same space
as v2's ``node_embeddings``). Downstream distance ops are in
``distance_scoring.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, cast

import torch
from torch import Tensor, nn

from ..modelsv2.layers import poincare_ops as P
from ..modelsv2.layers.edge_attention import EdgeTypedAttention
from ..modelsv2.layers.hyp_message_pass import HyperbolicMessagePassing
from ..modelsv2.layers.schema_encoder import SchemaEncoder


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def _make_rmsnorm(dim: int) -> nn.Module:
    rms = getattr(nn, "RMSNorm", None)
    return rms(dim) if rms is not None else _RMSNorm(dim)


class DepthAttention(nn.Module):
    """Per-layer pseudo-query softmax attention over tangent-space snapshots.

    Simplified from v2: no hierarchy-subspace slicing. All keys pass through
    RMSNorm. Zero-initialized depth queries give uniform averaging at init.
    """

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
class KGREmbeddingOutput:
    node_embeddings: Tensor                # (N, hidden_dim) on the Poincaré ball
    edge_type_embeddings: Tensor           # (T, type_dim) schema-encoded
    per_round_embeddings: Optional[List[Tensor]] = None  # only when log_depth=True


class KettleGraphReasonerV3(nn.Module):
    _c: Tensor

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        type_dim: int = 8,
        c: float = 1.0,
        learnable_c: bool = False,
        num_edge_types_max: Optional[int] = None,
        node_feat_dim_schema: Optional[int] = None,
        activation: str = "relu",
        depth_attn: bool = True,
        depth_attn_intra_stack: bool = False,
        log_depth: bool = False,
        tangent_scale_init: float = 0.1,
        per_layer_agg_scale: Optional[float] = None,
        per_layer_tan_clamp: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.type_dim = type_dim
        self.log_depth = bool(log_depth)
        self.depth_attn_enabled = bool(depth_attn)
        self.depth_attn_intra_stack = bool(depth_attn_intra_stack) and bool(depth_attn)

        if learnable_c:
            self._c = nn.Parameter(torch.tensor(float(c)))
        else:
            self.register_buffer("_c", torch.tensor(float(c)))

        # Euclidean → tangent-at-origin → ball. Small-gain Xavier + learnable
        # tangent_scale is the known-stable recipe against boundary saturation
        # (see kettle-graph-reasoner/CLAUDE.md "Known Issues").
        self.node_in = nn.Linear(node_feat_dim, hidden_dim)
        nn.init.xavier_uniform_(self.node_in.weight, gain=0.05)
        nn.init.zeros_(self.node_in.bias)
        self.tangent_scale = nn.Parameter(torch.tensor(float(tangent_scale_init)))

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
                c=c,
                learnable_c=False,
            )
            for _ in range(num_layers)
        )
        self.mp_layers = nn.ModuleList(
            HyperbolicMessagePassing(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                c=c,
                learnable_c=False,
                activation=activation,
                agg_scale_init=per_layer_agg_scale,
                out_tan_clamp=per_layer_tan_clamp,
            )
            for _ in range(num_layers)
        )

        self.depth_attention: Optional[DepthAttention] = (
            DepthAttention(num_layers=num_layers, hidden_dim=hidden_dim)
            if self.depth_attn_enabled
            else None
        )

    @property
    def c(self) -> Tensor:
        return self._c.clamp_min(P.MIN_NORM)

    def _sync_c(self) -> None:
        if not isinstance(self._c, nn.Parameter):
            return
        src = self._c.detach()
        with torch.no_grad():
            for m in list(self.attn_layers) + list(self.mp_layers):
                buf = cast(Tensor, m._c)
                if isinstance(buf, nn.Parameter):  # pragma: no cover
                    continue
                buf.copy_(src)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_descriptor: Tensor,
        node_descriptor: Optional[Tensor] = None,
    ) -> KGREmbeddingOutput:
        """Graph encoder forward. No query, no task_type — the v3 encoder is
        unconditional. All query conditioning happens in a separate head
        (``query_encoder.QueryToBall``) trained in stage B.
        """
        self._sync_c()
        c = self.c

        h_tan = self.node_in(node_features) * self.tangent_scale
        h = P.expmap0(h_tan, c)

        edge_type_emb, _ = self.schema_encoder(edge_descriptor, node_descriptor)

        per_round: Optional[List[Tensor]] = [] if self.log_depth else None
        tangent_snapshots: List[Tensor] = []
        for l_idx, (attn, mp) in enumerate(zip(self.attn_layers, self.mp_layers)):
            if (
                self.depth_attn_intra_stack
                and l_idx > 0
                and self.depth_attention is not None
            ):
                mixed_tan = self.depth_attention(tangent_snapshots, query_idx=l_idx)
                h_input = P.expmap0(mixed_tan, c)
            else:
                h_input = h
            alpha = attn(
                h_input, edge_index, edge_type, type_emb_override=edge_type_emb
            )
            h = mp(h_input, edge_index, edge_weight=alpha)
            tangent_snapshots.append(P.logmap0(h, c))
            if per_round is not None:
                per_round.append(h)

        if self.depth_attention is not None:
            h_flat = self.depth_attention(
                tangent_snapshots, query_idx=self.num_layers - 1
            )
            # Keep node_embeddings consistent with the tangent the depth
            # attention produces, so radial-reg targets the representation
            # downstream distance ops will actually see.
            h = P.expmap0(h_flat, c)

        return KGREmbeddingOutput(
            node_embeddings=h,
            edge_type_embeddings=edge_type_emb,
            per_round_embeddings=per_round,
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

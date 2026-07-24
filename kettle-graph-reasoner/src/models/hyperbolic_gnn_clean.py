r"""KettleGraphReasonerClean — distance-based scoring.

The scoring function is hyperbolic distance: score_i = -dist(h_i, h_q, c).
No MLP scoring head. No concatenation with tangent coordinates. No aux
depth head. No radial regularizer. No concat_depth. No subspace routing.
The model's task is to arrange nodes on the Poincaré ball so that
distance-to-query-embedding matches label relevance. If hyperbolic
geometry helps ranking on your tasks, this is the minimal architecture
that shows it.

The message-passing stack is preserved: edge-typed attention produces
per-edge weights, hyperbolic message passing aggregates in tangent-at-
receiver space, Möbius matvec + HNN nonlinearity per layer. The
architectural commitments from CLAUDE.md (hyperbolic geometry, edge-typed
attention, schema-portable types) are all honored. What's been removed
is the Euclidean MLP scoring head that was making the loss rotationally
invariant on the ball.

Design note on edge scoring. Edges have no separate learned head; their
score is the endpoint average of node scores. If node_score_i is a
monotone function of -dist(h_i, h_q), then edge_score is a monotone
function of the average of those two distances. This is what the original
loss.py already does for edge *labels*, so it's consistent with how the
data was constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, cast

import torch
from torch import Tensor, nn

from .layers import poincare_ops as P
from .layers.edge_attention import EdgeTypedAttention
from .layers.hyp_message_pass import HyperbolicMessagePassing
from .layers.schema_encoder import SchemaEncoder


@dataclass
class KGROutputClean:
    node_scores: Tensor  # (N,) — sigmoid of -dist(h_i, h_q), so in [0, 1]
    edge_scores: Tensor  # (E,) — endpoint average of node scores
    node_embeddings: Tensor  # (N, hidden_dim) — final hyperbolic states
    edge_type_embeddings: Tensor  # (T, type_dim)
    # Diagnostic: the scalar negative-distance logits before sigmoid. Useful
    # for seeing how separated nodes become and for debugging score dynamics
    # independent of the sigmoid saturation.
    node_logits: Tensor  # (N,) — -dist(h_i, h_q, c)
    query_point: Tensor  # (hidden_dim,) — h_q on the ball


class KettleGraphReasonerClean(nn.Module):
    r"""Same encoder + MP stack as KettleGraphReasoner; scoring replaced
    with -dist(h_i, h_q, c).

    Forward flow:
      1. node_features → tangent-at-origin → Poincaré ball via expmap0
      2. query → tangent-at-origin → Poincaré ball via expmap0
      3. L rounds of (EdgeTypedAttention → HyperbolicMessagePassing)
      4. node_logits = -dist(h_i, h_q, c)
      5. node_scores = sigmoid(node_logits)
      6. edge_scores = 0.5 * (node_scores[src] + node_scores[dst])
    """

    _c: Tensor

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        query_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        type_dim: int = 8,
        c: float = 1.0,
        learnable_c: bool = False,
        num_edge_types_max: Optional[int] = None,
        node_feat_dim_schema: Optional[int] = None,
        activation: str = "relu",
        tangent_scale_init: float = 0.10,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.type_dim = type_dim

        if learnable_c:
            self._c = nn.Parameter(torch.tensor(float(c)))
        else:
            self.register_buffer("_c", torch.tensor(float(c)))

        # Euclidean → tangent-at-origin → ball. Same small-gain init and
        # learnable tangent_scale as the original, since those control
        # where init lands on the ball and that matters regardless of the
        # scoring function.
        self.node_in = nn.Linear(node_feat_dim, hidden_dim)
        nn.init.xavier_uniform_(self.node_in.weight, gain=0.05)
        nn.init.zeros_(self.node_in.bias)
        self.tangent_scale = nn.Parameter(torch.tensor(float(tangent_scale_init)))

        # Query projection produces a tangent-space vector; same scaling
        # applied before expmap0 so h_q lives at the same radial scale as
        # the initial node embeddings. They're comparable from step 0.
        self.query_in = nn.Linear(query_dim, hidden_dim)
        nn.init.xavier_uniform_(self.query_in.weight, gain=0.05)
        nn.init.zeros_(self.query_in.bias)

        print(f"[DEBUG init] tangent_scale        = {self.tangent_scale.item():.4f}")
        print(f"[DEBUG init] c (curvature)        = {float(self._c):.4f}  learnable={learnable_c}")
        print(f"[DEBUG init] hidden_dim           = {hidden_dim}")
        print(f"[DEBUG init] num_layers           = {num_layers}")
        print(f"[DEBUG init] type_dim             = {type_dim}")
        with torch.no_grad():
            v = self.node_in.weight @ torch.randn(node_feat_dim, 1024, device=self.node_in.weight.device)
            v = v * self.tangent_scale
            vnorm = v.norm(dim=0)
            print(
                f"[DEBUG init] sim tangent ||v||    "
                f"mean={vnorm.mean().item():.4f} "
                f"max={vnorm.max().item():.4f}  "
                f"(target pre-expmap0: 0.05–0.3)"
            )

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
            )
            for _ in range(num_layers)
        )

    @property
    def c(self) -> Tensor:
        return self._c.clamp_min(P.MIN_NORM)

    def _sync_c(self) -> None:
        """Propagate the model's c into each child module when c is learnable."""
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
        query: Tensor,
        node_descriptor: Optional[Tensor] = None,
        task_type: Optional[int] = None,  # unused here; kept for interface parity
    ) -> KGROutputClean:
        self._sync_c()
        c = self.c

        # Nodes to the ball.
        h_tan = self.node_in(node_features) * self.tangent_scale
        h = P.expmap0(h_tan, c)

        # Query to the ball. Same tangent_scale so query lives at the same
        # radial scale as initial node embeddings.
        q_flat = query.view(-1) if query.dim() == 1 else query.view(-1)
        q_tan = self.query_in(q_flat) * self.tangent_scale
        h_q = P.expmap0(q_tan, c)  # (hidden_dim,) on the ball

        # Schema → type embeddings.
        edge_type_emb, _ = self.schema_encoder(edge_descriptor, node_descriptor)

        # Message-passing stack.
        for attn, mp in zip(self.attn_layers, self.mp_layers):
            alpha = attn(h, edge_index, edge_type, type_emb_override=edge_type_emb)
            h = mp(h, edge_index, edge_weight=alpha)

        # Distance-based scoring. h_q is a single point; broadcast to all
        # nodes via expansion before dist.
        N = h.size(0)
        h_q_exp = h_q.unsqueeze(0).expand(N, -1)
        d = P.dist(h, h_q_exp, c)  # (N,)
        node_logits = -d  # closer → higher score
        node_scores = torch.sigmoid(node_logits)

        # Edge scores: endpoint average. No learned params.
        src, dst = edge_index[0], edge_index[1]
        edge_scores = 0.5 * (node_scores.index_select(0, src) + node_scores.index_select(0, dst))

        return KGROutputClean(
            node_scores=node_scores,
            edge_scores=edge_scores,
            node_embeddings=h,
            edge_type_embeddings=edge_type_emb,
            node_logits=node_logits,
            query_point=h_q,
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

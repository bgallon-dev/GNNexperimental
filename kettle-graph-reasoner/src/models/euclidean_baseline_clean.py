r"""EuclideanBaselineClean — KGR without the Poincaré ball.

The matched control for ``KettleGraphReasonerClean``. Exactly one variable
differs: geometry. Every other architectural choice is identical.

What this model has in common with ``KettleGraphReasonerClean``:
  - No learned MLP scoring heads. Node score is ``sigmoid(-dist(h_i, h_q))``.
  - Edge score is the endpoint average of node scores (no learned edge head).
  - Same EdgeTypedAttention (in ``euclidean=True`` mode to skip the logmap0 wrapper).
  - Same SchemaEncoder for edge-type embeddings.
  - Same small-gain init (``xavier_uniform_(gain=0.05)``) on the input projection.
  - Same learnable tangent_scale, applied to the input projection output.
  - Same hidden_dim=64 default, same num_layers=3 default.
  - Same forward signature (``node_features, edge_index, edge_type,
    edge_descriptor, query, node_descriptor``) and compatible output dataclass.

What differs from ``KettleGraphReasonerClean``:
  - No ``expmap0``. Node embeddings and query embedding live in ``R^d``.
  - ``dist(h_i, h_q)`` is ``||h_i - h_q||_2`` instead of Poincaré distance.
  - Message passing is ``EuclideanMessagePassing`` (weighted neighbor sum →
    linear → activation). No Möbius matvec, no HNN nonlinearity wrapper.

What this is NOT:
  - It is not a replacement for ``EuclideanBaseline`` or ``EuclideanPlusBaseline``.
    Those exist for different ablations (GAT-only, geometry-only-with-MLP-head).
    This model is specifically the matched control for the clean scoring-head
    ablation. Three baselines, three distinct variables isolated.

Run comparison::

    py -m src.training.train_clean --model clean --task 2 --out runs/hyp_task2
    py -m src.training.train_clean --model clean_euc --task 2 --out runs/euc_task2

If hyperbolic beats Euclidean on task 2 (where hyperbolic reached 0.74 val
nDCG@10), with both trained identically, geometry is doing real work. If
they come out within noise, geometry is not the variable that matters on
this data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .euclidean_plus_baseline import EuclideanMessagePassing
from .layers.edge_attention import EdgeTypedAttention
from .layers.schema_encoder import SchemaEncoder


@dataclass
class EuclideanOutputClean:
    """Parallels ``KGROutputClean`` field-for-field so the training loop
    and metrics can consume either without a type check."""

    node_scores: Tensor           # (N,) — sigmoid(-||h - h_q||)
    edge_scores: Tensor           # (E,) — endpoint average of node scores
    node_embeddings: Tensor       # (N, hidden_dim) — final Euclidean states
    edge_type_embeddings: Tensor  # (T, type_dim)
    node_logits: Tensor           # (N,) — -||h_i - h_q|| (diagnostic)
    query_point: Tensor           # (hidden_dim,) — h_q in R^d


class EuclideanBaselineClean(nn.Module):
    r"""Matched control for ``KettleGraphReasonerClean``.

    Forward flow:
      1. node_features → Linear → h ∈ R^d (scaled by tangent_scale for parity)
      2. query → Linear → h_q ∈ R^d (same tangent_scale)
      3. L rounds of (EdgeTypedAttention(euclidean=True) → EuclideanMessagePassing)
      4. node_logits = -||h_i - h_q||_2
      5. node_scores = sigmoid(node_logits)
      6. edge_scores = 0.5 * (node_scores[src] + node_scores[dst])

    Note on ``tangent_scale`` in a Euclidean model: it has no geometric
    meaning here (there's no tangent space), but it preserves the init-time
    embedding magnitude matching with the hyperbolic model. Initial node
    embedding norms are ``tangent_scale * ||W_in @ x||`` in both models,
    so at step 0 the two models produce distance values on the same scale.
    That makes the sigmoid operate in the same regime at init. Keep it
    learnable so the model can adjust away from the matched init if it
    helps; that's a fair-fight condition since the hyperbolic model can too.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        query_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        type_dim: int = 8,
        num_edge_types_max: Optional[int] = None,
        node_feat_dim_schema: Optional[int] = None,
        tangent_scale_init: float = 0.10,
        # Accepted for API parity with KettleGraphReasonerClean; unused here
        # because there's no curvature in Euclidean space. Kept so the same
        # config dict can construct either model.
        c: float = 1.0,
        learnable_c: bool = False,
        activation: str = "relu",
        **_ignored,
    ) -> None:
        super().__init__()
        del c, learnable_c  # unused; kept in signature for config parity

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.type_dim = type_dim

        # Same input projection shape and init as hyperbolic_gnn_clean.
        self.node_in = nn.Linear(node_feat_dim, hidden_dim)
        nn.init.xavier_uniform_(self.node_in.weight, gain=0.05)
        nn.init.zeros_(self.node_in.bias)
        self.tangent_scale = nn.Parameter(torch.tensor(float(tangent_scale_init)))

        # Same query projection shape and init as hyperbolic_gnn_clean.
        self.query_in = nn.Linear(query_dim, hidden_dim)
        nn.init.xavier_uniform_(self.query_in.weight, gain=0.05)
        nn.init.zeros_(self.query_in.bias)

        print(f"[DEBUG init] tangent_scale        = {self.tangent_scale.item():.4f}")
        print(f"[DEBUG init] hidden_dim           = {hidden_dim}")
        print(f"[DEBUG init] num_layers           = {num_layers}")
        print(f"[DEBUG init] type_dim             = {type_dim}")
        print(f"[DEBUG init] geometry             = Euclidean (no curvature)")
        with torch.no_grad():
            v = self.node_in.weight @ torch.randn(
                node_feat_dim, 1024, device=self.node_in.weight.device
            )
            v = v * self.tangent_scale
            vnorm = v.norm(dim=0)
            print(
                f"[DEBUG init] sim embedding ||v||  "
                f"mean={vnorm.mean().item():.4f} "
                f"max={vnorm.max().item():.4f}  "
                f"(target: 0.05–0.3, matching hyperbolic init)"
            )

        # Same SchemaEncoder as hyperbolic_gnn_clean.
        self.schema_encoder = SchemaEncoder(
            edge_feat_dim=edge_feat_dim,
            type_dim=type_dim,
            node_feat_dim=node_feat_dim_schema,
        )

        # Same EdgeTypedAttention, in euclidean mode (skips logmap0 wrapper).
        # This matches EuclideanPlusBaseline's attention setup exactly.
        self.attn_layers = nn.ModuleList(
            EdgeTypedAttention(
                node_dim=hidden_dim,
                num_edge_types=num_edge_types_max,
                type_dim=type_dim,
                euclidean=True,
            )
            for _ in range(num_layers)
        )

        # Euclidean message passing — matches EuclideanPlusBaseline. The
        # xavier_uniform init with gain=0.05 inside EuclideanMessagePassing
        # mirrors the hyperbolic model's small-gain init on the Möbius
        # matvec weights, so per-layer scale drift is comparable.
        self.mp_layers = nn.ModuleList(
            EuclideanMessagePassing(in_dim=hidden_dim, out_dim=hidden_dim)
            for _ in range(num_layers)
        )

        # NOTE: We accept the `activation` kwarg for parity but don't use it;
        # EuclideanMessagePassing hardcodes torch.relu, matching the
        # hyperbolic default. If you change the hyperbolic activation via
        # the `activation=` kwarg, also change EuclideanMessagePassing's
        # internal activation to keep parity. Flagged here so this isn't
        # a silent drift point.
        if activation != "relu":
            raise NotImplementedError(
                f"activation={activation!r} not supported by EuclideanMessagePassing "
                "(it hardcodes torch.relu). To test a different activation with "
                "matched Euclidean geometry, update EuclideanMessagePassing first."
            )

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_descriptor: Tensor,
        query: Tensor,
        node_descriptor: Optional[Tensor] = None,
        task_type: Optional[int] = None,  # unused; interface parity
    ) -> EuclideanOutputClean:
        del task_type  # unused; interface parity

        # Input projection, scaled — NO expmap0, stays in R^d.
        h = self.node_in(node_features) * self.tangent_scale

        q_flat = query.view(-1) if query.dim() == 1 else query.view(-1)
        h_q = self.query_in(q_flat) * self.tangent_scale  # (hidden_dim,) in R^d

        # Schema → type embeddings.
        edge_type_emb, _ = self.schema_encoder(edge_descriptor, node_descriptor)

        # Message-passing stack.
        for attn, mp in zip(self.attn_layers, self.mp_layers):
            alpha = attn(h, edge_index, edge_type, type_emb_override=edge_type_emb)
            h = mp(h, edge_index, edge_weight=alpha)

        # Euclidean distance scoring. Broadcast h_q to all nodes.
        N = h.size(0)
        h_q_exp = h_q.unsqueeze(0).expand(N, -1)
        d = (h - h_q_exp).norm(dim=-1)  # (N,) — Euclidean L2 distance
        node_logits = -d
        node_scores = torch.sigmoid(node_logits)

        # Edge scores: endpoint average. No learned params. Mirrors the
        # hyperbolic model's edge-score construction exactly.
        src, dst = edge_index[0], edge_index[1]
        edge_scores = 0.5 * (
            node_scores.index_select(0, src) + node_scores.index_select(0, dst)
        )

        return EuclideanOutputClean(
            node_scores=node_scores,
            edge_scores=edge_scores,
            node_embeddings=h,
            edge_type_embeddings=edge_type_emb,
            node_logits=node_logits,
            query_point=h_q,
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

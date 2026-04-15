r"""KettleGraphReasoner — full experimental model.

Stacks the locked components into a query-conditioned subgraph relevance
scorer:

    node features  ──▶ Euclidean→ball (expmap0)  ──┐
    schema descs   ──▶ SchemaEncoder ──────────────┤
    edge indices,                                   │  L × (EdgeTypedAttention →
    edge types     ─────────────────────────────────┼──  HyperbolicMessagePassing)
    query vector   ──▶ query encoder ───────────────┘
                                                    │
                            ┌───────────────────────┴───────┐
                            ▼                               ▼
                   per-node score head              per-edge score head
                   (log0(h_i) || q)                 (log0(h_s) || log0(h_d)
                                                     || t_r || q)

Outputs are structural (scores), never linguistic (Commitment #4). Query
enters only through the scoring heads — a minimal conditioning mechanism
that keeps the message-passing stack query-agnostic and therefore
reusable across queries on the same graph. A more expressive gating-in-
attention variant is a deliberate v2 choice, not forced now.

Parameter budget (at hidden=32, L=3, type_dim=8): ~20K parameters,
well under the 500K–2M envelope. Hidden dim is the knob to scale up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, cast

import torch
from torch import Tensor, nn

from .layers import poincare_ops as P
from .layers.edge_attention import EdgeTypedAttention
from .layers.hyp_message_pass import HyperbolicMessagePassing
from .layers.schema_encoder import SchemaEncoder


class _RMSNorm(nn.Module):
    """RMSNorm with a learned per-dim scale. Used when torch.nn.RMSNorm
    is unavailable (PyTorch < 2.4)."""

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
    """Per-layer pseudo-query softmax attention over tangent-space
    snapshots from each MP round.

    Keys are RMSNorm'd (Technique 3) to prevent magnitude-dominant
    layers. When ``hierarchy_subspace_dim > 0`` the first k dims encode
    depth-as-magnitude for Task 0 and are passed through un-normalized;
    RMSNorm only touches the proximity slice ``[:, k:]``. Queries are
    zero-initialized (Technique 4) so the model starts at uniform
    averaging. Softmax over depth introduces competition (Technique 5).
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        hierarchy_subspace_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.k = int(hierarchy_subspace_dim)
        self.depth_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_layers)
        )
        prox_dim = hidden_dim - self.k if self.k > 0 else hidden_dim
        self.key_norm_prox = _make_rmsnorm(prox_dim)

    def _norm_keys(self, tangents: Tensor) -> Tensor:
        if self.k > 0:
            hier = tangents[..., : self.k]
            prox = self.key_norm_prox(tangents[..., self.k :])
            return torch.cat([hier, prox], dim=-1)
        return self.key_norm_prox(tangents)

    def forward(self, snapshots: List[Tensor], query_idx: int) -> Tensor:
        """snapshots: list of (N, D) tangent tensors, length L' <= num_layers.

        Returns (N, D): depth-attended tangent mixture using
        ``depth_queries[query_idx]`` as the pseudo-query.
        """
        stack = torch.stack(snapshots, dim=0)  # (L', N, D)
        keys = self._norm_keys(stack)          # (L', N, D)
        q = self.depth_queries[query_idx]       # (D,)
        logits = torch.einsum("d,lnd->ln", q, keys)  # (L', N)
        alpha = logits.softmax(dim=0)                 # (L', N)
        return (alpha.unsqueeze(-1) * stack).sum(dim=0)  # (N, D)


@dataclass
class KGROutput:
    node_scores: Tensor  # (N,) — per-node relevance in [0, 1]
    edge_scores: Tensor  # (E,) — per-edge relevance in [0, 1]
    node_embeddings: Tensor  # (N, hidden_dim) — final hyperbolic states
    edge_type_embeddings: Tensor  # (T, type_dim) — schema-encoded types
    # AttnRes Phase-1 diagnostic. None unless model was constructed with
    # log_depth=True. Each entry is the node-embedding tensor *after* that
    # message-passing round, in ball coordinates — same space as
    # node_embeddings (which equals the last element when populated).
    per_round_embeddings: Optional[List[Tensor]] = None


class KettleGraphReasoner(nn.Module):
    _c: Tensor

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        query_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        type_dim: int = 8,
        c: float = 1.0,
        learnable_c: bool = False,
        num_edge_types_max: Optional[int] = None,
        node_feat_dim_schema: Optional[int] = None,
        activation: str = "relu",
        hierarchy_subspace_dim: int = 0,
        log_depth: bool = False,
        depth_attn: bool = True,
        depth_attn_intra_stack: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.type_dim = type_dim
        self.log_depth = bool(log_depth)
        self.depth_attn_enabled = bool(depth_attn)
        self.depth_attn_intra_stack = bool(depth_attn_intra_stack) and bool(depth_attn)
        # Subspace partitioning: when > 0, the first k tangent-at-origin
        # coordinates are reserved for Task 0 (graded hierarchy / 1/d depth),
        # the remaining hidden_dim-k coordinates feed the shared proximity
        # head used by Tasks 1-4. Logmap0 on the Poincaré ball is linear at
        # origin, so slicing tangent coordinates is a valid subspace
        # projection (equivalent to a Poincaré-ball distance on the
        # k-dimensional sub-ball). Zero = disabled, single-head behavior.
        self.hierarchy_subspace_dim = int(hierarchy_subspace_dim)
        if not 0 <= self.hierarchy_subspace_dim <= hidden_dim:
            raise ValueError(
                f"hierarchy_subspace_dim={hierarchy_subspace_dim} must be in [0, {hidden_dim}]"
            )

        if learnable_c:
            self._c = nn.Parameter(torch.tensor(float(c)))
        else:
            self.register_buffer("_c", torch.tensor(float(c)))

        # Euclidean → tangent-at-origin → ball.
        # CRITICAL: tangent vectors entering expmap0 must have norm in [0.1, 1.0].
        # Xavier default gives ||v|| ~ sqrt(hidden_dim), which saturates tanh() and
        # pins every point to the Poincaré boundary at initialization — the ball's
        # interior is then unused and the hyperbolic geometry reduces to Euclidean.
        # Small-gain init puts initial ||v|| around 0.2–0.4; the learnable
        # tangent_scale lets the model grow (or shrink) the ball usage during
        # training.
        self.node_in = nn.Linear(node_feat_dim, hidden_dim)
        nn.init.xavier_uniform_(self.node_in.weight, gain=0.05)
        nn.init.zeros_(self.node_in.bias)
        self.tangent_scale = nn.Parameter(torch.tensor(0.1))
        self.query_in = nn.Linear(query_dim, hidden_dim)

        print(f"[DEBUG init] tangent_scale        = {self.tangent_scale.item():.4f}")
        print(f"[DEBUG init] c (curvature)        = {float(self._c):.4f}  learnable={learnable_c}")
        print(f"[DEBUG init] hidden_dim           = {hidden_dim}")
        print(f"[DEBUG init] num_layers           = {num_layers}")
        print(f"[DEBUG init] type_dim             = {type_dim}")
        print(f"[DEBUG init] hierarchy_subspace   = {self.hierarchy_subspace_dim}")
        print(
            f"[DEBUG init] node_in.weight       "
            f"mean={self.node_in.weight.mean().item():+.4e} "
            f"std={self.node_in.weight.std().item():.4e} "
            f"max|w|={self.node_in.weight.abs().max().item():.4e}"
        )
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
        print(
            f"[DEBUG init] query_in.weight      "
            f"mean={self.query_in.weight.mean().item():+.4e} "
            f"std={self.query_in.weight.std().item():.4e}"
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
                learnable_c=False,  # share c across stack via this module
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

        self.depth_attention: Optional[DepthAttention] = (
            DepthAttention(
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                hierarchy_subspace_dim=self.hierarchy_subspace_dim,
            )
            if self.depth_attn_enabled
            else None
        )

        # Scoring heads operate in Euclidean tangent space at the origin so
        # the concatenation with the query vector is a proper inner-product
        # operation (not a coordinate hack on the ball).
        k = self.hierarchy_subspace_dim
        prox_dim = hidden_dim - k if k > 0 else hidden_dim
        self.node_score = nn.Sequential(
            nn.Linear(prox_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_score = nn.Sequential(
            nn.Linear(prox_dim * 2 + type_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        if k > 0:
            self.node_score_hier = nn.Sequential(
                nn.Linear(k + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.edge_score_hier = nn.Sequential(
                nn.Linear(k * 2 + type_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    @property
    def c(self) -> Tensor:
        return self._c.clamp_min(P.MIN_NORM)

    def _sync_c(self) -> None:
        """Propagate the model's ``c`` into each child module so learnable-c
        (if enabled at the top level) stays consistent across layers."""
        if not isinstance(self._c, nn.Parameter):
            return  # buffer — set once in __init__, children share the value
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
        task_type: Optional[int] = None,
    ) -> KGROutput:
        """
        Parameters
        ----------
        node_features : (N, node_feat_dim)
        edge_index    : (2, E)
        edge_type     : (E,) long
        edge_descriptor : (T_edge, edge_feat_dim)  — schema, re-encoded per call
        query         : (query_dim,) or (1, query_dim) — a single query vector
        node_descriptor : (T_node, node_feat_dim_schema), optional
        """
        self._sync_c()
        c = self.c

        # Nodes: Euclidean → tangent-at-origin → ball.
        h_tan = self.node_in(node_features) * self.tangent_scale
        h = P.expmap0(h_tan, c)

        # Query → hidden; flatten to (hidden_dim,).
        q = (
            self.query_in(query.view(-1))
            if query.dim() == 1
            else self.query_in(query).view(-1)
        )

        # Schema → type embeddings.
        edge_type_emb, _ = self.schema_encoder(edge_descriptor, node_descriptor)

        # Message-passing stack. Always stash tangent-at-origin snapshots
        # (cheap, and depth attention reads them). Ball-space per-round
        # embeddings are additionally stashed for diagnostics when log_depth.
        per_round: Optional[List[Tensor]] = [] if self.log_depth else None
        tangent_snapshots: List[Tensor] = []
        for l_idx, (attn, mp) in enumerate(zip(self.attn_layers, self.mp_layers)):
            if self.depth_attn_intra_stack and l_idx > 0 and self.depth_attention is not None:
                mixed_tan = self.depth_attention(tangent_snapshots, query_idx=l_idx)
                h_input = P.expmap0(mixed_tan, c)
            else:
                h_input = h
            alpha = attn(h_input, edge_index, edge_type, type_emb_override=edge_type_emb)
            h = mp(h_input, edge_index, edge_weight=alpha)
            tangent_snapshots.append(P.logmap0(h, c))
            if per_round is not None:
                per_round.append(h)

        # Depth-wise attention over all rounds → attended tangent for the
        # scoring head. When disabled, fall back to the raw last-round view.
        if self.depth_attention is not None:
            h_flat = self.depth_attention(
                tangent_snapshots, query_idx=self.num_layers - 1
            )
            # Keep node_embeddings consistent with what the scoring head sees
            # so downstream radial reg targets the attended representation.
            h = P.expmap0(h_flat, c)
        else:
            h_flat = P.logmap0(h, c)  # (N, hidden_dim)
        N = h_flat.size(0)
        q_exp = q.unsqueeze(0).expand(N, -1)

        k = self.hierarchy_subspace_dim
        use_hier = k > 0 and task_type == 0
        if k > 0:
            h_slice = h_flat[:, :k] if use_hier else h_flat[:, k:]
            node_head = self.node_score_hier if use_hier else self.node_score
            edge_head = self.edge_score_hier if use_hier else self.edge_score
        else:
            h_slice = h_flat
            node_head = self.node_score
            edge_head = self.edge_score

        node_logits = node_head(torch.cat([h_slice, q_exp], dim=-1)).squeeze(-1)
        node_scores = torch.sigmoid(node_logits)

        src, dst = edge_index[0], edge_index[1]
        h_s = h_slice.index_select(0, src)
        h_d = h_slice.index_select(0, dst)
        t_r = edge_type_emb.index_select(0, edge_type)
        E = edge_index.size(1)
        q_e = q.unsqueeze(0).expand(E, -1)
        edge_logits = edge_head(torch.cat([h_s, h_d, t_r, q_e], dim=-1)).squeeze(-1)
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

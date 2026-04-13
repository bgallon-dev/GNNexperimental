r"""Edge-typed heterogeneous attention.

Produces per-edge scalar weights ``α_{ji}`` consumed by
``HyperbolicMessagePassing`` via its ``edge_weight`` argument. Each edge
type gets its own learned type embedding; receiver and sender are
projected through shared ``W_q`` / ``W_k`` matrices and combined with the
type embedding in a GAT-style additive score (Velickovic 2018). Scores
are softmax-normalized over the in-neighbors of each receiver.

Why score in tangent-at-origin space rather than on the ball:
attention is a similarity calculation that wants a linear inner-product
geometry. ``logmap0`` gives us a faithful Euclidean view at the ball's
center; keeping attention there avoids nesting Möbius ops inside a
softmax (extra cost, worse gradients) without sacrificing representational
power — the *aggregation* still happens hyperbolically, one level up in
``HyperbolicMessagePassing``. This is the separation-of-concerns the
plan locked in: this module decides *how much each edge matters*; the
message-pass module decides *how neighbors combine geometrically*.

Schema-portability (CLAUDE.md Commitment #3): this module is
parametrized by ``num_edge_types`` and learns a ``(num_edge_types,
type_dim)`` embedding table. It does *not* hardcode semantics for any
specific edge label. The ``schema_encoder`` (next sub-step) will replace
the lookup with descriptor-conditioned embeddings; the forward API here
is designed so that swap is additive.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from . import poincare_ops as P
from .hyp_message_pass import _scatter_add


class EdgeTypedAttention(nn.Module):
    r"""GAT-style additive attention, typed per edge.

    Forward:
        score_{ji} = LeakyReLU(a^T [W_q · log0(x_i) || W_k · log0(x_j) ||
                                     W_t · e_{type(j,i)}])
        α_{ji}    = softmax_{j ∈ N(i)}(score_{ji})

    Output is a ``(E,)`` tensor aligned with ``edge_index`` — drops straight
    into ``HyperbolicMessagePassing.forward(edge_weight=...)``.
    """

    _c: Tensor

    def __init__(
        self,
        node_dim: int,
        num_edge_types: int | None = None,
        type_dim: int = 8,
        head_dim: int | None = None,
        c: float = 1.0,
        learnable_c: bool = False,
        negative_slope: float = 0.2,
        euclidean: bool = False,
    ) -> None:
        """``num_edge_types=None`` means this module has no internal type
        embedding table — callers must always supply ``type_emb_override``
        in forward. Use this when a SchemaEncoder owns type embeddings
        upstream, to avoid a dead-parameter nn.Embedding."""
        super().__init__()
        self.node_dim = node_dim
        self.num_edge_types = num_edge_types
        self.type_dim = type_dim
        self.head_dim = head_dim if head_dim is not None else node_dim
        self.negative_slope = negative_slope
        self.euclidean = euclidean

        if learnable_c:
            self._c = nn.Parameter(torch.tensor(float(c)))
        else:
            self.register_buffer("_c", torch.tensor(float(c)))

        if num_edge_types is not None:
            self.type_emb: nn.Embedding | None = nn.Embedding(num_edge_types, type_dim)
            nn.init.normal_(self.type_emb.weight, std=0.1)
        else:
            self.type_emb = None
        self.W_q = nn.Linear(node_dim, self.head_dim, bias=False)
        self.W_k = nn.Linear(node_dim, self.head_dim, bias=False)
        self.W_t = nn.Linear(type_dim, self.head_dim, bias=False)
        self.a = nn.Parameter(torch.empty(3 * self.head_dim))

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_t.weight)
        nn.init.xavier_uniform_(self.a.view(1, -1))

    @property
    def c(self) -> Tensor:
        return self._c.clamp_min(P.MIN_NORM)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        type_emb_override: Tensor | None = None,
    ) -> Tensor:
        """``type_emb_override`` (shape ``(T, type_dim)``) lets a SchemaEncoder
        supply descriptor-derived embeddings in place of the internal
        ``nn.Embedding`` lookup. Same edge_type indices are used; only the
        embedding table differs."""
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must be shape (2, E)")
        if edge_type.shape != (edge_index.size(1),):
            raise ValueError("edge_type must be shape (E,)")
        N = x.size(0)
        c = self.c

        # Euclidean view of node points via log at origin. In euclidean mode
        # nodes are already Euclidean — skip the no-op logmap.
        x_tan = x if self.euclidean else P.logmap0(x, c)  # (N, node_dim)
        q = self.W_q(x_tan)  # (N, head_dim)
        k = self.W_k(x_tan)  # (N, head_dim)

        src, dst = edge_index[0], edge_index[1]
        q_dst = q.index_select(0, dst)  # receiver
        k_src = k.index_select(0, src)  # sender
        if type_emb_override is not None:
            if type_emb_override.size(-1) != self.type_dim:
                raise ValueError(
                    f"type_emb_override last dim must be {self.type_dim}; "
                    f"got {type_emb_override.size(-1)}"
                )
            t_emb = type_emb_override.index_select(0, edge_type)
        elif self.type_emb is not None:
            t_emb = self.type_emb(edge_type)
        else:
            raise ValueError(
                "EdgeTypedAttention was built with num_edge_types=None; "
                "type_emb_override is required"
            )
        t = self.W_t(t_emb)  # (E, head_dim)

        feat = torch.cat([q_dst, k_src, t], dim=-1)  # (E, 3*head_dim)
        raw = torch.nn.functional.leaky_relu(
            feat @ self.a, negative_slope=self.negative_slope
        )  # (E,)

        # Numerically stable scatter-softmax over incoming edges per receiver.
        per_recv_max = torch.full(
            (N,), float("-inf"), dtype=raw.dtype, device=raw.device
        ).scatter_reduce(0, dst, raw, reduce="amax", include_self=False)
        per_recv_max = torch.where(
            torch.isfinite(per_recv_max), per_recv_max, torch.zeros_like(per_recv_max)
        )
        shifted = raw - per_recv_max.index_select(0, dst)
        exp = shifted.exp()
        denom = _scatter_add(exp.unsqueeze(-1), dst, dim_size=N).squeeze(-1)
        denom = denom.clamp_min(P.MIN_NORM)
        alpha = exp / denom.index_select(0, dst)
        return alpha

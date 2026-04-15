r"""Hyperbolic message passing layer.

Single round of neighbor aggregation on the Poincaré ball, composed from the
pure-function primitives in ``poincare_ops``. Replaces Euclidean
``GCNConv``/``GATConv`` for the experimental model. Edge-typed attention
weights are *consumed*, not computed here (see ``edge_attention.py``).

Aggregation is **tangent-at-receiver**: for each receiver ``i`` we linearize
the neighbors in ``T_{x_i} B^d`` via ``logmap(x_j, x_i, c)``, sum (weighted)
in that tangent space, then ``expmap`` back. Rationale in the plan file:
origin-anchored aggregation distorts points near the boundary, which is
where hierarchy-heavy provenance chains live.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor, nn

from . import poincare_ops as P


def _make_scatter_add() -> Callable[[Tensor, Tensor, int], Tensor]:
    """Resolve a scatter-add implementation once at import time. Prefers
    PyG / torch_scatter; falls back to ``index_add`` so the layer works
    without optional deps installed. Resolving per-call via try/except paid
    for traceback allocation on every forward — this hoists it out."""
    try:  # pragma: no cover - import probe
        from torch_geometric.utils import scatter as _pyg_scatter  # type: ignore[import-not-found]

        return lambda src, index, dim_size: _pyg_scatter(
            src, index, dim=0, dim_size=dim_size, reduce="sum"
        )
    except ImportError:
        pass
    try:  # pragma: no cover - import probe
        from torch_scatter import scatter as _ts_scatter  # type: ignore[import-not-found]

        return lambda src, index, dim_size: _ts_scatter(
            src, index, dim=0, dim_size=dim_size, reduce="sum"
        )
    except ImportError:
        pass

    def _fallback(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
        out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        return out.index_add(0, index, src)

    return _fallback


_scatter_add: Callable[[Tensor, Tensor, int], Tensor] = _make_scatter_add()


_ACTIVATIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "relu": torch.relu,
    "leaky_relu": torch.nn.functional.leaky_relu,
    "gelu": torch.nn.functional.gelu,
    "identity": lambda x: x,
}


class HyperbolicMessagePassing(nn.Module):
    r"""One round of Möbius-space neighbor aggregation.

    Forward composition (receiver ``i``, neighbors ``j ∈ N(i)``):

    1. ``m_{j→i} = logmap(x_j, x_i, c)``
    2. ``m_{j→i} ← α_{ji} · m_{j→i}``  (``α_{ji}`` from ``edge_weight`` or
       uniform ``1/|N(i)|`` when ``edge_weight is None``)
    3. ``agg_i = Σ_j m_{j→i}`` via ``scatter_add``
    4. ``u_i = expmap(agg_i, x_i, c)``
    5. ``h_i = mobius_matvec(W, u_i, c)``
    6. ``h_i ← mobius_add(h_i, expmap0(b, c), c)``  (if ``use_bias``)
    7. ``h_i = expmap0(σ(logmap0(h_i, c)), c)``  (Ganea 2018 HNN nonlinearity)
    8. ``project(h_i, c)`` belt-and-suspenders

    Notes on step 7: ``σ = ReLU`` by default. Applying ReLU *in the tangent
    space at the origin* is the standard HNN convention, but it zeros
    negative components of a tangent vector, which has no canonical
    geometric meaning in hyperbolic space. If training stalls, swap to
    ``leaky_relu`` or move the nonlinearity before the final ``expmap`` on
    the tangent representation — the component interface does not change.
    """

    _c: Tensor

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        c: float = 1.0,
        learnable_c: bool = False,
        activation: str = "relu",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias

        if learnable_c:
            self._c = nn.Parameter(torch.tensor(float(c)))
        else:
            self.register_buffer("_c", torch.tensor(float(c)))

        # Small-gain init to keep mobius_matvec outputs inside the ball at
        # initialization. Xavier default gives ||W v|| ~ O(sqrt(d)) which
        # saturates tanh in mobius_matvec and pins points to the boundary.
        # See CLAUDE.md "Known Issues: Poincaré ball saturation".
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.weight, gain=0.05)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation: {activation}")
        self._activation = _ACTIVATIONS[activation]

    @property
    def c(self) -> Tensor:
        return self._c.clamp_min(P.MIN_NORM)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must be shape (2, E)")
        N = x.size(0)
        c = self.c

        src, dst = edge_index[0], edge_index[1]
        # 1. linearize neighbors in T_{x_dst} B^d
        x_src = x.index_select(0, src)
        x_dst = x.index_select(0, dst)
        msg = P.logmap(x_src, x_dst, c)  # (E, in_dim)

        # 2. weighting
        if edge_weight is None:
            deg = _scatter_add(
                torch.ones(src.size(0), 1, dtype=x.dtype, device=x.device),
                dst,
                dim_size=N,
            ).clamp_min(1.0)  # isolated nodes: denom=1, numerator=0 anyway
            w = (1.0 / deg.index_select(0, dst))  # (E, 1)
        else:
            if edge_weight.dim() == 1:
                w = edge_weight.unsqueeze(-1)
            else:
                w = edge_weight
        msg = msg * w

        # 3. sum in tangent space at each receiver
        agg = _scatter_add(msg, dst, dim_size=N)  # (N, in_dim)

        # 4. back to the ball at each node's basepoint (zero agg → exp at x = x)
        u = P.expmap(agg, x, c)

        # 5. hyperbolic linear
        h = P.mobius_matvec(self.weight, u, c)

        # 6. Möbius bias (tangent-at-origin parameter, lifted before add)
        if self.bias is not None:
            b_ball = P.expmap0(self.bias.expand_as(h), c)
            h = P.mobius_add(h, b_ball, c)

        # 7. HNN nonlinearity (tangent-at-origin). See class docstring.
        # No rescaling here — scale is controlled upstream by the
        # small-gain `self.weight` init (and at the model input by
        # `KettleGraphReasoner.tangent_scale`). Rescaling inside the
        # nonlinearity compounded with those, giving three damping
        # factors per layer and erasing radial variance across nodes.
        h_tan = P.logmap0(h, c)
        h_tan = self._activation(h_tan)
        h = P.expmap0(h_tan, c)

        # 8. final project
        return P.project(h, c)

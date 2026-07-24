r"""QueryToBall — stage-B query encoder for KGR v3 / v3.1.

Maps a per-task query vector (``query_dim``-D Euclidean) to a single
point on the Poincaré ball that shares curvature ``c`` with the v3
graph encoder. At scoring time, per-node relevance = ``-dist_p(q_point,
node_emb, c)``.

Trained in stage B with the graph encoder frozen. Uses the same
small-gain Xavier (gain 0.05) + learnable ``tangent_scale`` (init 0.1)
recipe as the graph encoder's input projection on *every* linear that
feeds a tangent vector toward ``expmap0`` — this is the boundary-
saturation mitigation from kettle-graph-reasoner/CLAUDE.md and it is
non-negotiable for all architectures here.

v3.1 query-head sweep
---------------------
The original head (now ``arch="qh0"``) is a rigid 2-layer MLP. Its old
docstring claimed "adding capacity here has no obvious benefit"; v3.1's
Phase 2 directly tests that claim by making the architecture selectable:

  qh0  18 -> hidden -> hidden                 (baseline; UNCHANGED)
  qh1  18 -> 256 -> hidden
  qh2  18 -> 256 -> 256 -> hidden  (residual + Norm + GELU; default target)
  qh3  18 -> 512 -> 256 -> hidden
  qh4  query-prototype adapter -> hidden

``qh0`` keeps the exact attribute names (``fc1``, ``fc2``,
``tangent_scale``) and forward of the pre-v3.1 module so the locked
baseline ``query_encoder.pt`` loads with ``strict=True``. Every variant
builds under ``self.body`` so its parameter names never collide with
``qh0``; a variant checkpoint only loads into a same-arch module. Eval
loaders read ``query_head_arch`` from ``summary.json["config"]``
(default ``"qh0"`` for pre-v3.1 checkpoints).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..modelsv2.layers import poincare_ops as P

QH_ARCHS = ("qh0", "qh1", "qh2", "qh3", "qh4")


# ---------------------------------------------------------------------------
# norm helper (RMSNorm fallback for torch < 2.4, mirrors modelsv2)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def _make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "rmsnorm":
        rms = getattr(nn, "RMSNorm", None)
        return rms(dim) if rms is not None else _RMSNorm(dim)
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"unknown norm {kind!r}; use layernorm or rmsnorm")


def _small_gain_(linear: nn.Linear, gain: float = 0.05) -> nn.Linear:
    """The boundary-saturation init recipe: small-gain Xavier weight,
    zero bias. Applied to every linear in the head so the tangent norm
    feeding expmap0 stays in the ball interior at init regardless of
    depth/width."""
    nn.init.xavier_uniform_(linear.weight, gain=gain)
    nn.init.zeros_(linear.bias)
    return linear


# ---------------------------------------------------------------------------
# residual block used by qh2
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    """``x + GELU(Norm(W2 GELU(Norm(W1 x))))`` — pre-norm residual,
    width-preserving. Small-gain init keeps it near-identity at start."""

    def __init__(self, dim: int, norm: str) -> None:
        super().__init__()
        self.n1 = _make_norm(norm, dim)
        self.fc1 = _small_gain_(nn.Linear(dim, dim))
        self.n2 = _make_norm(norm, dim)
        self.fc2 = _small_gain_(nn.Linear(dim, dim))
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc1(self.n1(x))
        h = self.act(h)
        h = self.fc2(self.n2(h))
        return x + self.act(h)


# ---------------------------------------------------------------------------
# query-prototype adapter used by qh4
# ---------------------------------------------------------------------------

class _PrototypeAdapter(nn.Module):
    """Project the query to a soft mixture over ``n_proto`` learned
    task/query prototypes, then read out a hidden-dim tangent vector.
    ``proj_in`` and ``proto_values`` carry the small-gain init; the
    final readout feeds expmap0."""

    def __init__(self, query_dim: int, hidden_dim: int,
                 n_proto: int = 16, key_dim: int = 256) -> None:
        super().__init__()
        self.proj_in = _small_gain_(nn.Linear(query_dim, key_dim))
        self.act = nn.GELU()
        self.proto_keys = nn.Parameter(torch.randn(n_proto, key_dim) * 0.02)
        self.proto_values = nn.Parameter(torch.randn(n_proto, hidden_dim) * 0.02)
        self.scale = key_dim ** -0.5
        self.residual = _small_gain_(nn.Linear(key_dim, hidden_dim))

    def forward(self, q: Tensor) -> Tensor:
        h = self.act(self.proj_in(q))                       # (B, key_dim)
        attn = torch.softmax(
            (h @ self.proto_keys.t()) * self.scale, dim=-1)  # (B, n_proto)
        mixed = attn @ self.proto_values                     # (B, hidden)
        return mixed + self.residual(h)


# ---------------------------------------------------------------------------
# QueryToBall
# ---------------------------------------------------------------------------

class QueryToBall(nn.Module):
    r"""Map a Euclidean query vector to a point on the Poincaré ball.

    Parameters
    ----------
    query_dim : int
        Dimension of the Euclidean query input (``dataset.query_dim``;
        18 in the tier-1 corpus).
    hidden_dim : int
        Ball embedding dimension. Must match the graph encoder.
    c : float
        Curvature; matches the graph encoder. ``.c`` exposes the
        clamped tensor.
    tangent_scale_init : float
        Initial learnable scalar multiplying the tangent vector before
        ``expmap0`` (default 0.1, mirrors the encoder).
    euclidean : bool
        If True, skip ``expmap0`` and return the Euclidean output.
    arch : str
        One of ``QH_ARCHS``. ``qh0`` is the unchanged 2-layer baseline.
    norm : str
        ``"layernorm"`` (default) or ``"rmsnorm"`` — only used by the
        ``qh2``/``qh3`` variants.
    """

    _c: Tensor

    def __init__(
        self,
        query_dim: int,
        hidden_dim: int,
        c: float = 1.0,
        tangent_scale_init: float = 0.1,
        euclidean: bool = False,
        arch: str = "qh0",
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        if arch not in QH_ARCHS:
            raise ValueError(f"unknown arch {arch!r}; choose from {QH_ARCHS}")
        self.query_dim = int(query_dim)
        self.hidden_dim = int(hidden_dim)
        self.euclidean = bool(euclidean)
        self.arch = arch
        self.norm_kind = norm
        self.register_buffer("_c", torch.tensor(float(c)))
        self.tangent_scale = nn.Parameter(torch.tensor(float(tangent_scale_init)))

        if arch == "qh0":
            # EXACT pre-v3.1 module: attribute names + forward must not
            # change so the locked baseline checkpoint loads strictly.
            self.fc1 = nn.Linear(self.query_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
            nn.init.xavier_uniform_(self.fc1.weight, gain=0.05)
            nn.init.xavier_uniform_(self.fc2.weight, gain=0.05)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)
        else:
            self.body = self._build_body(arch, norm)

    # -- variant bodies (all under self.body; query_dim -> hidden_dim) -----

    def _build_body(self, arch: str, norm: str) -> nn.Module:
        qd, hd = self.query_dim, self.hidden_dim
        act = nn.GELU()
        if arch == "qh1":
            return nn.Sequential(
                _small_gain_(nn.Linear(qd, 256)), act,
                _small_gain_(nn.Linear(256, hd)),
            )
        if arch == "qh2":
            return nn.Sequential(
                _small_gain_(nn.Linear(qd, 256)),
                _make_norm(norm, 256), nn.GELU(),
                _ResidualBlock(256, norm),
                _small_gain_(nn.Linear(256, hd)),
            )
        if arch == "qh3":
            return nn.Sequential(
                _small_gain_(nn.Linear(qd, 512)),
                _make_norm(norm, 512), nn.GELU(),
                _small_gain_(nn.Linear(512, 256)),
                _make_norm(norm, 256), nn.GELU(),
                _small_gain_(nn.Linear(256, hd)),
            )
        if arch == "qh4":
            return _PrototypeAdapter(qd, hd)
        raise ValueError(f"no body for arch {arch!r}")

    @property
    def c(self) -> Tensor:
        return self._c.clamp_min(P.MIN_NORM)

    def forward(self, query: Tensor) -> Tensor:
        r"""``query``: ``(query_dim,)`` or ``(B, query_dim)``. Returns a
        point per query: ``(hidden_dim,)`` or ``(B, hidden_dim)``."""
        q = query if query.dim() == 2 else query.unsqueeze(0)
        if self.arch == "qh0":
            h = self.fc1(q)
            h = torch.relu(h)
            h = self.fc2(h) * self.tangent_scale
        else:
            h = self.body(q) * self.tangent_scale
        if not self.euclidean:
            h = P.expmap0(h, self.c)
        return h.squeeze(0) if query.dim() == 1 else h

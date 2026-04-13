r"""Poincaré ball operations (pure-function math layer).

Atomic hyperbolic primitives used throughout the KGR experimental model.
All functions are pure (no ``nn.Module`` state) and operate on the Poincaré
ball of curvature ``c > 0``. Arrays are ``(..., d)`` shaped; the last
dimension is the embedding dimension.

Conventions (locked — do not drift from these):

1. **Argument order.**
   - ``expmap(v, x, c)`` and ``logmap(y, x, c)``: the non-basepoint argument
     comes first, then the basepoint, then curvature.
     (Note: geoopt's API is ``ball.expmap(x, v)`` / ``ball.logmap(x, y)`` —
     the opposite order. Never mix.)
   - ``mobius_add(x, y, c)``: left operand first (Möbius addition is
     non-commutative; order matters).
   - ``dist(x, y, c)``, ``parallel_transport(x, y, v, c)``: source then
     target then (for transport) the vector.
2. **Curvature sign.** This module uses ``c > 0`` (Ganea et al. 2018
   convention). geoopt uses signed ``k`` with ``k = -c`` for the Poincaré
   ball. Tests cross to geoopt through a single ``_to_geoopt_k`` helper.
3. **`c` dtype.** Accepted as a Python float or a ``torch.Tensor``.
   Internally promoted to a tensor matching ``x.dtype`` / ``x.device`` so
   ``c`` can later be made a learnable parameter without API changes.

Invariant: every op that could emit a point outside the ball
(``mobius_add`` / ``expmap`` / ``expmap0`` / ``mobius_matvec``) ends with a
call to ``project``.
"""

from __future__ import annotations

from typing import Union

import torch

Curvature = Union[float, torch.Tensor]

MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}
_TANH_CLAMP = 15.0
_ARTANH_CLAMP = 1.0 - 1e-7


def _as_c(c: Curvature, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(c, torch.Tensor):
        return c.to(dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(c, dtype=ref.dtype, device=ref.device)


def _tanh(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-_TANH_CLAMP, _TANH_CLAMP).tanh()


class _Artanh(torch.autograd.Function):
    """Inverse hyperbolic tangent with boundary-safe forward *and* backward.

    Matches geoopt's ``artanh`` numerically: forward uses the log form
    ``0.5 * (log1p(x) - log1p(-x))`` after clamping to ``±(1 - 1e-7)``;
    backward clamps the same input before computing ``1 / (1 - x²)``.
    Without the backward clamp, gradients through ``dist`` / ``logmap``
    near the ball boundary NaN silently — this class exists to prevent
    exactly that failure mode.
    """

    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-_ARTANH_CLAMP, _ARTANH_CLAMP)
        ctx.save_for_backward(x)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output / (1 - x.pow(2))


def _artanh(x: torch.Tensor) -> torch.Tensor:
    return _Artanh.apply(x)  # type: ignore[return-value]


def project(x: torch.Tensor, c: Curvature = 1.0, eps: float | None = None) -> torch.Tensor:
    c_t = _as_c(c, x)
    if eps is None:
        eps = BALL_EPS[x.dtype] if x.dtype in BALL_EPS else 1e-5
    maxnorm = (1 - eps) / c_t.clamp_min(MIN_NORM).sqrt()
    norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def lambda_x(x: torch.Tensor, c: Curvature = 1.0, keepdim: bool = True) -> torch.Tensor:
    c_t = _as_c(c, x)
    return 2.0 / (1.0 - c_t * x.pow(2).sum(dim=-1, keepdim=keepdim)).clamp_min(MIN_NORM)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: Curvature = 1.0) -> torch.Tensor:
    c_t = _as_c(c, x)
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c_t * xy + c_t * y2) * x + (1 - c_t * x2) * y
    denom = 1 + 2 * c_t * xy + c_t.pow(2) * x2 * y2
    return project(num / denom.clamp_min(MIN_NORM), c_t)


def mobius_scalar_mul(r, x: torch.Tensor, c: Curvature = 1.0) -> torch.Tensor:
    c_t = _as_c(c, x)
    r_t = torch.as_tensor(r, dtype=x.dtype, device=x.device) if not isinstance(r, torch.Tensor) else r.to(x.dtype)
    sqrt_c = c_t.clamp_min(MIN_NORM).sqrt()
    x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    out = _tanh(r_t * _artanh(sqrt_c * x_norm)) * (x / (sqrt_c * x_norm))
    return project(out, c_t)


def mobius_matvec(M: torch.Tensor, x: torch.Tensor, c: Curvature = 1.0) -> torch.Tensor:
    c_t = _as_c(c, x)
    sqrt_c = c_t.clamp_min(MIN_NORM).sqrt()
    x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    mx = torch.matmul(x, M.transpose(-1, -2))
    mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    res_c = _tanh(mx_norm / x_norm * _artanh(sqrt_c * x_norm)) * (mx / (sqrt_c * mx_norm))
    cond = (mx == 0).all(dim=-1, keepdim=True)
    res_zero = torch.zeros((), dtype=res_c.dtype, device=res_c.device)
    return project(torch.where(cond, res_zero, res_c), c_t)


def expmap(v: torch.Tensor, x: torch.Tensor, c: Curvature = 1.0) -> torch.Tensor:
    c_t = _as_c(c, x)
    sqrt_c = c_t.clamp_min(MIN_NORM).sqrt()
    v_norm = v.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    lam = lambda_x(x, c_t, keepdim=True)
    second = _tanh(sqrt_c * lam * v_norm / 2.0) * (v / (sqrt_c * v_norm))
    return mobius_add(x, second, c_t)


def expmap0(v: torch.Tensor, c: Curvature = 1.0) -> torch.Tensor:
    c_t = _as_c(c, v)
    sqrt_c = c_t.clamp_min(MIN_NORM).sqrt()
    v_norm = v.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    out = _tanh(sqrt_c * v_norm) * (v / (sqrt_c * v_norm))
    return project(out, c_t)


def logmap(y: torch.Tensor, x: torch.Tensor, c: Curvature = 1.0) -> torch.Tensor:
    c_t = _as_c(c, x)
    sqrt_c = c_t.clamp_min(MIN_NORM).sqrt()
    sub = mobius_add(-x, y, c_t)
    sub_norm = sub.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    lam = lambda_x(x, c_t, keepdim=True)
    return (2.0 / (sqrt_c * lam)) * _artanh(sqrt_c * sub_norm) * (sub / sub_norm)


def logmap0(y: torch.Tensor, c: Curvature = 1.0) -> torch.Tensor:
    c_t = _as_c(c, y)
    sqrt_c = c_t.clamp_min(MIN_NORM).sqrt()
    y_norm = y.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    return (y / (sqrt_c * y_norm)) * _artanh(sqrt_c * y_norm)


def dist(x: torch.Tensor, y: torch.Tensor, c: Curvature = 1.0, keepdim: bool = False) -> torch.Tensor:
    c_t = _as_c(c, x)
    sqrt_c = c_t.clamp_min(MIN_NORM).sqrt()
    diff_norm = mobius_add(-x, y, c_t).norm(dim=-1, keepdim=keepdim, p=2)
    return (2.0 / sqrt_c) * _artanh(sqrt_c * diff_norm)


def _gyration(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # Ungar gyration identity. geoopt's form with k = -c:
    #   a = -k² uw v² - k vw + 2k² uv vw   →   a = -c² uw v² + c vw + 2c² uv vw
    #   b = -k² vw u² + k uw              →   b = -c² vw u² - c uw
    #   d = 1 - 2k uv + k² u² v²          →   d = 1 + 2c uv + c² u² v²
    u2 = u.pow(2).sum(dim=-1, keepdim=True)
    v2 = v.pow(2).sum(dim=-1, keepdim=True)
    uv = (u * v).sum(dim=-1, keepdim=True)
    uw = (u * w).sum(dim=-1, keepdim=True)
    vw = (v * w).sum(dim=-1, keepdim=True)
    c2 = c.pow(2)
    a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
    b = -c2 * vw * u2 - c * uw
    d = 1 + 2 * c * uv + c2 * u2 * v2
    return w + 2 * (a * u + b * v) / d.clamp_min(MIN_NORM)


def parallel_transport(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, c: Curvature = 1.0) -> torch.Tensor:
    c_t = _as_c(c, x)
    lam_x = lambda_x(x, c_t, keepdim=True)
    lam_y = lambda_x(y, c_t, keepdim=True)
    return _gyration(y, -x, v, c_t) * lam_x / lam_y


def parallel_transport0(y: torch.Tensor, v: torch.Tensor, c: Curvature = 1.0) -> torch.Tensor:
    c_t = _as_c(c, y)
    return v * (1.0 - c_t * y.pow(2).sum(dim=-1, keepdim=True)).clamp_min(MIN_NORM)

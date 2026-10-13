"""Temporal models of boiling: per-mode correlation and the exact update.

Boiling relaxes each Fourier mode of a layer toward fresh, statistically
identical turbulence. Two Gauss-Markov models are available, both stationary
(the mode's variance, and hence the spatial PSD, is preserved at every step)
and both sampled exactly at any time step:

- ``"ar1"``: first order (Ornstein-Uhlenbeck). Retention after a lag ``s``
  (in units of the mode's time constant ``tau``) is ``exp(-s)``. The process is
  continuous but not differentiable: its innovations are white, so the change
  over a short step grows as ``sqrt(dt/tau)``.
- ``"ar2"``: second order, critically damped (a continuous-time AR(2), i.e. a
  Matérn-3/2 correlation in time), retention ``(1 + 2s) exp(-2s)``. The mode
  carries a rate state as well as its value, so it is differentiable: the
  change over a short step grows as ``dt/tau``. Its integral time scale is
  ``tau``, the same as ``"ar1"``'s, so both decorrelate over the same long-lag
  horizon and differ at short lags.

The functions take NumPy (or CuPy, where an ``xp`` argument is given) arrays.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

BOIL_MODELS = ("ar1", "ar2")
# Scale dependence of the per-mode time constant: ``"kolmogorov"`` gives a mode
# at spatial frequency ``f`` the eddy-turnover ``tau_boil * (f/f_ref)^(-2/3)``,
# ``"uniform"`` gives every mode ``tau_boil``.
BOIL_SCALINGS = ("kolmogorov", "uniform")


def check_model(model: str) -> str:
    """Validate a ``boil_model`` name and return it."""
    if model not in BOIL_MODELS:
        raise ValueError(
            f"boil_model must be one of {BOIL_MODELS}; got {model!r}. 'ar1' "
            "(default) is first-order boiling with white innovations, 'ar2' a "
            "smooth second-order process with the same integral time scale."
        )
    return model


def check_scaling(scaling: str) -> str:
    """Validate a ``boil_scaling`` name and return it."""
    if scaling not in BOIL_SCALINGS:
        raise ValueError(
            f"boil_scaling must be one of {BOIL_SCALINGS}; got {scaling!r}. "
            "'kolmogorov' gives each Fourier mode the eddy-turnover time "
            "tau_boil * (f/f_ref)^(-2/3) (fine structure boils faster), "
            "'uniform' gives every spatial scale the one time constant tau_boil."
        )
    return scaling


def retention(s: Any, model: str, xp: Any = np) -> Any:
    """Correlation of a mode with itself ``s = |lag| / tau`` time constants later."""
    if model == "ar1":
        return xp.exp(-s)
    return (1.0 + 2.0 * s) * xp.exp(-2.0 * s)


def _exp_tail3(y: Any, xp: Any = np) -> Any:
    """``exp(y) - 1 - y - y**2/2`` for ``y >= 0``, without cancellation for small ``y``.

    A 20-term series below ``y = 1`` (truncation error below ``1e-19`` relative)
    and the direct expression above it, where nothing cancels.
    """
    small = xp.minimum(y, 1.0)
    term = small ** 3 / 6.0
    series = term
    for k in range(4, 24):
        term = term * small / k
        series = series + term
    direct = xp.expm1(xp.minimum(y, 700.0)) - y - 0.5 * y * y
    return xp.where(y < 1.0, series, direct)


def decorrelation(s: Any, model: str, xp: Any = np) -> Any:
    """``1 - retention(s)``, accurate for small ``s`` (no cancellation)."""
    if model == "ar1":
        return -xp.expm1(-s)
    # 1 - (1 + w) e^{-w} = e^{-w} (e^w - 1 - w) with w = 2 s.
    w = 2.0 * s
    tail = _exp_tail3(w, xp) + 0.5 * w * w
    return xp.where(w < 700.0, xp.exp(-w) * tail, 1.0)


def ar2_coefficients(u: Any, xp: Any = np) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """Exact one-step update of the critically damped ``"ar2"`` process.

    The state of a mode is its value ``x`` and its rate scaled to unit variance
    ``y = (dx/dt) / omega`` with ``omega = 2 / tau``; ``u = omega * dt``. The
    stationary state has unit, uncorrelated ``x`` and ``y``, and the step is

    ``x' = f11 x + f12 y + l11 z1``
    ``y' = -f12 x + f22 y + l21 z1 + l22 z2``

    with ``z1``, ``z2`` independent unit normals: the transition matrix
    ``exp(-u) [[1 + u, u], [-u, 1 - u]]`` and the Cholesky factor of the
    innovation covariance ``I - F F^T``. ``u = 0`` (an infinite ``tau``) leaves
    the state unchanged. Returns ``(f11, f12, f22, l11, l21, l22)``.
    """
    u = xp.asarray(u, dtype=np.float64)
    e = xp.exp(-xp.minimum(u, 350.0))
    e = xp.where(u < 350.0, e, 0.0)
    e2 = e * e
    y = 2.0 * u
    small = y < 1.0
    tail = _exp_tail3(y, xp)
    # Innovation covariance; the small-u forms avoid cancelling 1 - e^{-2u}(...).
    q11 = xp.where(small, e2 * tail, 1.0 - e2 * (1.0 + y + 0.5 * y * y))
    q22 = xp.where(small, e2 * (tail + y + y), 1.0 - e2 * (1.0 - y + 0.5 * y * y))
    q12 = 0.5 * y * y * e2
    l11 = xp.sqrt(xp.clip(q11, 0.0, None))
    safe = xp.where(l11 > 0.0, l11, 1.0)
    l21 = xp.where(l11 > 0.0, q12 / safe, 0.0)
    l22 = xp.sqrt(xp.clip(q22 - l21 * l21, 0.0, None))
    return e * (1.0 + u), e * u, e * (1.0 - u), l11, l21, l22

"""Analysis and validation utilities: Zernikes, temporal PSDs, decorrelation.

These turn a pile of phase screens into the diagnostics AO people actually
check turbulence against:

- :func:`zernike_basis` / :func:`zernike_decompose` — a Noll-ordered Zernike
  basis on a circular pupil and a least-squares projection onto it.
- :func:`noll_variance` / :func:`noll_residual_variance` — the Kolmogorov
  Zernike-mode variances and post-correction residuals of Noll (1976), the
  textbook thing to validate a decomposition against.
- :func:`temporal_psd` / :func:`fit_power_law` — a one-sided temporal power
  spectrum and a log-log slope fit (frozen-flow gives power-law regimes).
- :func:`differential_variance` — angular (anisoplanatism) decorrelation.

Everything works on NumPy or CuPy input (device arrays are brought to the host
for the reductions). Reference: Noll, R. J. (1976), JOSA 66, 207.
"""

from __future__ import annotations

from math import factorial
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from .backend import to_numpy

__all__ = [
    "noll_to_zernike",
    "zernike_basis",
    "zernike_decompose",
    "noll_variance",
    "noll_residual_variance",
    "temporal_psd",
    "fit_power_law",
    "differential_variance",
]

# Noll (1976) Table IV: residual wavefront variance after perfectly correcting
# the first J Zernike modes, in units of (D/r0)^{5/3} rad^2. Delta[1] is the
# residual after removing piston (i.e. the total minus piston).
_NOLL_RESIDUAL = {
    1: 1.0299, 2: 0.582, 3: 0.134, 4: 0.111, 5: 0.0880, 6: 0.0648,
    7: 0.0587, 8: 0.0525, 9: 0.0463, 10: 0.0401, 11: 0.0377, 12: 0.0352,
    13: 0.0328, 14: 0.0304, 15: 0.0279, 16: 0.0267, 17: 0.0255, 18: 0.0243,
    19: 0.0232, 20: 0.0220, 21: 0.0208,
}


def _noll_residual_coeff(j):
    """Delta_j in (D/r0)^{5/3} units, tabulated then asymptotic (Noll 1976)."""
    if j < 1:
        raise ValueError("Noll index j must be >= 1")
    if j in _NOLL_RESIDUAL:
        return _NOLL_RESIDUAL[j]
    # Large-J asymptote Delta_J ~ 0.2944 J^{-sqrt(3)/2}.
    return 0.2944 * j ** (-np.sqrt(3.0) / 2.0)


def noll_to_zernike(j: int) -> Tuple[int, int]:
    """Radial/azimuthal orders ``(n, m)`` for Noll single index ``j`` (>= 1)."""
    if j < 1:
        raise ValueError("Noll index j must be >= 1")
    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * ((j1 + ((n + 1) % 2)) // 2))
    return n, m


def _radial(n, m, rho):
    m = abs(m)
    out = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        c = ((-1) ** k * factorial(n - k)) / (
            factorial(k)
            * factorial((n + m) // 2 - k)
            * factorial((n - m) // 2 - k)
        )
        out += c * rho ** (n - 2 * k)
    return out


def zernike_basis(
    n_modes: int, n_pixels: int, diameter_pixels: Optional[float] = None
) -> np.ndarray:
    """Noll-ordered Zernike basis over a circular pupil.

    Parameters
    ----------
    n_modes : int
        Number of modes, starting at Noll ``j = 1`` (piston).
    n_pixels : int
        Grid size; the returned array is ``(n_modes, n_pixels, n_pixels)``.
    diameter_pixels : float, optional
        Pupil diameter in pixels (default ``n_pixels``). The pupil is the
        inscribed disc; values are 0 outside it.

    Returns
    -------
    basis : ndarray
        ``(n_modes, n_pixels, n_pixels)``. Each mode is orthonormalised so that
        its variance over the pupil is 1 (Noll normalisation), and the pupil
        mask is ``basis[0] != 0`` (piston).
    """
    if n_modes < 1 or n_pixels < 2:
        raise ValueError("n_modes >= 1 and n_pixels >= 2 required")
    radius = (n_pixels if diameter_pixels is None else diameter_pixels) / 2.0
    grid = (np.arange(n_pixels) - (n_pixels - 1) / 2.0) / radius
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    rho = np.hypot(xx, yy)
    theta = np.arctan2(yy, xx)
    mask = rho <= 1.0

    basis = np.zeros((n_modes, n_pixels, n_pixels))
    for idx in range(n_modes):
        j = idx + 1
        n, m = noll_to_zernike(j)
        norm = np.sqrt(n + 1.0) * (1.0 if m == 0 else np.sqrt(2.0))
        ang = np.cos(m * theta) if m >= 0 else np.sin(-m * theta)
        z = norm * _radial(n, m, rho) * ang
        basis[idx] = np.where(mask, z, 0.0)
    return basis


def zernike_decompose(
    phase: ArrayLike, n_modes: int, basis: Optional[np.ndarray] = None
) -> np.ndarray:
    """Least-squares Zernike coefficients of ``phase`` over the pupil.

    Parameters
    ----------
    phase : ndarray
        ``(n, n)`` screen or ``(count, n, n)`` stack (NumPy or CuPy).
    n_modes : int
        Number of Noll modes to fit.
    basis : ndarray, optional
        A precomputed :func:`zernike_basis` (reused across many calls to avoid
        rebuilding it). Must match ``n_modes`` and the screen size.

    Returns
    -------
    coeffs : ndarray
        ``(n_modes,)`` or ``(count, n_modes)`` coefficients in the same units
        as ``phase``.
    """
    phase = to_numpy(phase).astype(np.float64)
    single = phase.ndim == 2
    if single:
        phase = phase[None]
    n_pixels = phase.shape[-1]
    if basis is None:
        basis = zernike_basis(n_modes, n_pixels)
    mask = basis[0] != 0
    design = basis[:, mask].T  # (n_pupil, n_modes)
    data = phase[:, mask].T  # (n_pupil, count)
    coeffs, *_ = np.linalg.lstsq(design, data, rcond=None)
    coeffs = coeffs.T  # (count, n_modes)
    return coeffs[0] if single else coeffs


def noll_variance(j: int, diameter: float, r0: float) -> float:
    """Kolmogorov variance of Zernike mode ``j`` [rad^2] (Noll 1976).

    The per-mode variance is ``Delta_{j-1} - Delta_j`` in ``(D/r0)^{5/3}``
    units. ``j = 1`` (piston) has no finite variance, so ``j >= 2``.
    """
    if j < 2:
        raise ValueError("piston (j=1) has no finite variance; use j >= 2")
    coeff = _noll_residual_coeff(j - 1) - _noll_residual_coeff(j)
    return coeff * (diameter / r0) ** (5.0 / 3.0)


def noll_residual_variance(j: int, diameter: float, r0: float) -> float:
    """Residual wavefront variance [rad^2] after correcting the first ``j``
    Zernike modes (Noll 1976)."""
    return _noll_residual_coeff(j) * (diameter / r0) ** (5.0 / 3.0)


def temporal_psd(series: ArrayLike, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided temporal power spectral density of a time series.

    Parameters
    ----------
    series : array_like
        Samples along the **last** axis (e.g. a pupil pixel or a Zernike
        coefficient over frames); leading axes are treated as independent
        series and averaged.
    dt : float
        Sample spacing [s].

    Returns
    -------
    freq, psd : ndarray
        Positive frequencies [Hz] (excluding DC) and the averaged PSD, scaled
        so that ``sum(psd) * df`` approximates the series variance.
    """
    series = to_numpy(series).astype(np.float64)
    series = series - series.mean(axis=-1, keepdims=True)
    n = series.shape[-1]
    spectrum = np.fft.rfft(series, axis=-1)
    psd = (np.abs(spectrum) ** 2) * (2.0 * dt / n)
    psd = psd.reshape(-1, psd.shape[-1]).mean(axis=0)
    freq = np.fft.rfftfreq(n, d=dt)
    return freq[1:], psd[1:]


def fit_power_law(
    freq: ArrayLike,
    psd: ArrayLike,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> Tuple[float, float]:
    """Fit ``psd ~ freq**slope`` over ``[fmin, fmax]`` (log-log least squares).

    Returns
    -------
    slope, amplitude : float
        Such that ``psd ≈ amplitude * freq**slope`` in the band.
    """
    freq = np.asarray(freq, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    band = np.ones(freq.shape, dtype=bool)
    if fmin is not None:
        band &= freq >= fmin
    if fmax is not None:
        band &= freq <= fmax
    band &= psd > 0
    if band.sum() < 2:
        raise ValueError("need at least two positive points in the band")
    slope, intercept = np.polyfit(np.log(freq[band]), np.log(psd[band]), 1)
    return float(slope), float(np.exp(intercept))


def differential_variance(reference: ArrayLike, other: ArrayLike) -> float:
    """Variance of ``other - reference`` [same units squared].

    With ``reference`` the on-axis OPD/phase and ``other`` an off-axis one, this
    is the angular (anisoplanatism) error; it grows as ``(theta/theta0)^{5/3}``
    and reaches ~1 rad^2 at the isoplanatic angle.
    """
    diff = to_numpy(other).astype(np.float64) - to_numpy(reference).astype(np.float64)
    return float(np.var(diff))

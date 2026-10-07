"""Closed-form and numerically integrated reference curves for turbulence.

These are the curves pyturb validates itself against, made public so you can
check your own AO simulation against the same numbers, including at **finite
outer scale** ``L0``, where the Kolmogorov tables (Noll 1976) overstate the
low-order power. Everything is phase in rad^2 at the wavelength ``r0`` is
quoted at, unless a function says otherwise.

- :func:`structure_function` — von Kármán / Kolmogorov phase structure
  function ``D(r)``.
- :func:`zernike_variance` — variance of Noll mode ``j`` for a pupil of
  diameter ``D``, any ``L0``.
- :func:`image_motion_variance` — one-axis tip/tilt image motion [arcsec^2].
- :func:`seeing_fwhm` — long-exposure FWHM including the outer-scale
  reduction (Tokovinin 2002).
- :func:`pixel_temporal_psd` and :func:`zernike_temporal_psd` — frozen-flow
  temporal spectra of one pupil point and of a Zernike mode.
- :func:`differential_phase_variance` — point-wise anisoplanatic phase
  variance between two directions through a layered profile.

References
----------
- Noll, R. J. (1976), JOSA 66, 207.
- Conan, J.-M., Rousset, G. & Madec, P.-Y. (1995), JOSA A 12, 1559.
- Tokovinin, A. (2002), PASP 114, 1156.
- Sasiela, R. J. (1994), *Electromagnetic Wave Propagation in Turbulence*.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from aocore import RAD_TO_ARCSEC
from numpy.typing import ArrayLike
from scipy import integrate, special

from .analysis import noll_to_zernike
from .infinite import phase_covariance
from .profiles import Layer, _fractions, _trapezoid

__all__ = [
    "structure_function",
    "zernike_variance",
    "image_motion_variance",
    "seeing_fwhm",
    "pixel_temporal_psd",
    "zernike_temporal_psd",
    "differential_phase_variance",
]

# Exact Kolmogorov phase-PSD constant, (24/5 Gamma(6/5))^(5/6) Gamma(11/6)^2 /
# (2 pi^(11/3)) = 0.022896..., usually rounded to 0.023. The exact value
# reproduces Noll's table and pyturb.phase_covariance.
_PSD_CONSTANT = (
    (24.0 / 5.0 * special.gamma(6.0 / 5.0)) ** (5.0 / 6.0)
    * special.gamma(11.0 / 6.0) ** 2 / (2.0 * np.pi ** (11.0 / 3.0))
)


def _psd(f: np.ndarray, r0: float, L0: float) -> np.ndarray:
    """Von Kármán phase PSD [rad^2 m^2] at spatial frequency ``f`` [1/m]."""
    f0_sq = 0.0 if np.isinf(L0) else 1.0 / L0 ** 2
    return _PSD_CONSTANT * r0 ** (-5.0 / 3.0) * (f ** 2 + f0_sq) ** (-11.0 / 6.0)


def _radial_filter(f: np.ndarray, n: int, radius: float) -> np.ndarray:
    """``(n+1) [J_{n+1}(2 pi f R) / (pi f R)]^2``: azimuthally averaged |Q_j|^2."""
    x = np.pi * f * radius
    with np.errstate(invalid="ignore", divide="ignore"):
        q = special.jv(n + 1, 2.0 * x) / x
    q = np.where(x == 0, 1.0 if n == 0 else 0.0, q)
    return (n + 1) * q ** 2


def _angular_factor(j: int, m: int, phi: np.ndarray) -> np.ndarray:
    """Angular part of |Q_j|^2 in Noll's convention (mean 1 over phi)."""
    if m == 0:
        return np.ones_like(phi)
    if j % 2 == 0:
        return 2.0 * np.cos(m * phi) ** 2
    return 2.0 * np.sin(m * phi) ** 2


def structure_function(r: ArrayLike, r0: float, L0: float = np.inf) -> np.ndarray:
    """Phase structure function ``D(r)`` [rad^2] at separation ``r`` [m].

    ``6.88 (r/r0)^{5/3}`` for Kolmogorov (``L0=inf``); ``2 [C(0) - C(r)]`` from
    :func:`pyturb.phase_covariance` for finite ``L0``.
    """
    r = np.asarray(r, dtype=np.float64)
    if np.isinf(L0):
        return 6.88 * (r / r0) ** (5.0 / 3.0)
    return 2.0 * (phase_covariance(0.0, r0, L0) - phase_covariance(r, r0, L0))


def zernike_variance(j: int, diameter: float, r0: float, L0: float = np.inf) -> float:
    """Variance [rad^2] of Noll Zernike mode ``j`` (>= 2) over a circular pupil.

    ``integral PSD(f) |Q_j(f)|^2 d^2f`` with the von Kármán PSD, evaluated
    numerically. For ``L0=inf`` it reproduces Noll's Kolmogorov values
    (:func:`pyturb.analysis.noll_variance`); a finite outer scale lowers the
    low orders most (tip/tilt by ~40% at ``L0/D = 3``).
    """
    if j < 2:
        raise ValueError("piston (j=1) has no finite variance; use j >= 2")
    n, _m = noll_to_zernike(int(j))
    radius = diameter / 2.0
    scale = 1.0 / (np.pi * radius)  # f = x * scale with x = pi f R

    def integrand(x: float) -> float:
        f = x * scale
        return 2.0 * np.pi * f * _psd(f, r0, L0) * _radial_filter(f, n, radius) * scale

    # Bessel oscillations: integrate decade by decade in x.
    edges = [0.0, 1.0, 10.0, 100.0, 1e3, 1e4]
    total = sum(integrate.quad(integrand, lo, hi, limit=2000)[0]
                for lo, hi in zip(edges[:-1], edges[1:]))
    return float(total)


def image_motion_variance(diameter: float, r0: float, wavelength: float,
                          L0: float = np.inf) -> float:
    """One-axis image-motion (tilt) variance [arcsec^2] for a pupil of diameter ``D``.

    ``r0`` is at ``wavelength`` [m]. For Kolmogorov turbulence this is the
    textbook ``0.182 (D/r0)^{5/3} (lambda/D)^2`` (Zernike tilt); a finite
    ``L0`` reduces it substantially for large telescopes.
    """
    a2 = zernike_variance(2, diameter, r0, L0)  # rad^2 of tilt coefficient
    # Noll tilt Z2 = 2 rho cos(theta): angle = lambda/(2 pi) * 2 a / R.
    sigma_rad = wavelength / (2.0 * np.pi) * 2.0 / (diameter / 2.0)
    return float(a2 * sigma_rad ** 2 * RAD_TO_ARCSEC ** 2)


def seeing_fwhm(r0: float, wavelength: float = 500e-9, L0: float = np.inf) -> float:
    """Long-exposure seeing FWHM [arcsec], with the outer-scale correction.

    ``0.98 lambda/r0`` (as :func:`pyturb.seeing_from_r0`) times
    ``sqrt(1 - 2.183 (r0/L0)^{0.356})`` (Tokovinin 2002, accurate for
    ``L0/r0 > 20``). At ``L0 = 25 m`` and ``r0 = 0.1 m`` the delivered FWHM is
    ~17% below the Kolmogorov value; :attr:`pyturb.Atmosphere.seeing` is the
    Kolmogorov value.
    """
    fwhm = 0.98 * wavelength / r0 * RAD_TO_ARCSEC
    if np.isinf(L0):
        return float(fwhm)
    ratio = r0 / L0
    if L0 / r0 < 20:
        raise ValueError(
            "seeing_fwhm's outer-scale correction (Tokovinin 2002) is only valid "
            f"for L0/r0 > 20; got L0/r0 = {L0 / r0:.3g}"
        )
    return float(fwhm * np.sqrt(1.0 - 2.183 * ratio ** 0.356))


def pixel_temporal_psd(freq: ArrayLike, r0: float, wind_speed: float) -> np.ndarray:
    """One-sided temporal PSD [rad^2/Hz] of one pupil point under frozen flow.

    Kolmogorov, single layer: ``0.0774 r0^{-5/3} V^{5/3} f^{-8/3}``.
    """
    freq = np.asarray(freq, dtype=np.float64)
    return 0.0774 * r0 ** (-5.0 / 3.0) * wind_speed ** (5.0 / 3.0) * freq ** (-8.0 / 3.0)


def zernike_temporal_psd(
    freq: ArrayLike,
    j: int,
    diameter: float,
    r0: float,
    wind_speed: float,
    wind_direction: float = 0.0,
    L0: float = np.inf,
) -> np.ndarray:
    """One-sided temporal PSD [rad^2/Hz] of Noll mode ``j`` for one frozen layer.

    ``PSD_j(nu) = 2/V integral PSD(f) |Q_j(f)|^2 df_perp`` with
    ``f_parallel = nu / V`` (Conan, Rousset & Madec 1995), integrated
    numerically. ``wind_direction`` [deg] is in the same frame as
    :attr:`pyturb.Layer.wind_direction` and
    :func:`pyturb.analysis.zernike_basis` (from +x = axis 1 toward +y =
    axis 0), so ``wind_direction=0`` is wind along x, where Noll ``j = 2``
    (tip) is the along-wind tilt. The frame matters for non-symmetric modes:
    tilt along the wind has a different spectrum from tilt across it.
    Integrates (over ``nu``) to :func:`zernike_variance`. For a profile, sum the
    layers' spectra, each with its own ``r0_i`` and wind.
    """
    if j < 2:
        raise ValueError("piston (j=1) has no finite variance; use j >= 2")
    n, m = noll_to_zernike(int(j))
    radius = diameter / 2.0
    psi = np.deg2rad(wind_direction)
    freq = np.atleast_1d(np.asarray(freq, dtype=np.float64))
    # Perpendicular frequencies: symmetric log grid, dense near 0.
    u = np.geomspace(1e-4 / diameter, 200.0 / diameter, 1500)
    f_perp = np.concatenate((-u[::-1], [0.0], u))
    out = np.empty(freq.shape)
    for start in range(0, freq.size, 256):  # bounded (chunk, n_perp) work arrays
        f_par = freq[start:start + 256, None] / wind_speed
        fx = f_par * np.cos(psi) - f_perp[None, :] * np.sin(psi)
        fy = f_par * np.sin(psi) + f_perp[None, :] * np.cos(psi)
        f = np.hypot(fx, fy)
        integrand = (_psd(f, r0, L0) * _radial_filter(f, n, radius)
                     * _angular_factor(j, m, np.arctan2(fy, fx)))
        out[start:start + 256] = 2.0 / wind_speed * _trapezoid(integrand, f_perp, axis=-1)
    return out


def differential_phase_variance(
    theta: Union[float, ArrayLike],
    layers: Sequence[Layer],
    r0: float,
    L0: Optional[float] = None,
) -> np.ndarray:
    """Point-wise phase variance [rad^2] between directions ``theta`` [arcsec] apart.

    ``sum_i D_i(h_i * theta)``, each layer's structure function at its own
    footprint separation, with ``r0_i^{-5/3} = f_i r0^{-5/3}``. ``r0`` and the
    layer altitudes must be along the same line of sight. For Kolmogorov
    turbulence (``L0=numpy.inf``) this is ``(theta/theta0)^{5/3}``, ~1 rad^2 at
    the isoplanatic angle; a finite outer scale lowers it. ``L0=None``
    (default) uses each layer's own ``L0``. No aperture averaging: a
    pupil-averaged, piston-removed differential variance is smaller.
    """
    theta = np.asarray(theta, dtype=np.float64)
    fractions = _fractions(list(layers))
    total = np.zeros_like(theta)
    for layer, fraction in zip(layers, fractions):
        if fraction <= 0:
            continue
        r0_i = r0 * fraction ** (-3.0 / 5.0)
        sep = layer.altitude * np.tan(theta / RAD_TO_ARCSEC)
        layer_L0 = layer.L0 if L0 is None else L0
        total = total + structure_function(sep, r0_i, layer_L0)
    return total

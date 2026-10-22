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
- :func:`covariance` and :func:`space_time_structure_function` — the
  space-time covariance and structure function between pupil points at a time
  lag, for a layered profile under frozen flow, optionally with pyturb's
  boiling model (what :class:`pyturb.Atmosphere` simulates; see
  :meth:`pyturb.Atmosphere.covariance`).
- :func:`travel_decorrelation_time` — an on-sky boiling time constant per
  layer from its wind speed (Guesalaga et al. 2014), for ``tau_boil``.
- :func:`gems_boil_time` — the outer-scale ``tau_boil`` per layer for the
  default scale-dependent boiling, calibrated on the same GeMS measurements.
- :func:`scintillation_transfer` and :func:`scintillation_index` — the
  weak-perturbation intensity transfer function of a layer and the
  scintillation index of a profile (Roddier 1981), the references for
  :meth:`pyturb.Atmosphere.pupil_field`.

References
----------
- Noll, R. J. (1976), JOSA 66, 207.
- Conan, J.-M., Rousset, G. & Madec, P.-Y. (1995), JOSA A 12, 1559.
- Tokovinin, A. (2002), PASP 114, 1156.
- Sasiela, R. J. (1994), *Electromagnetic Wave Propagation in Turbulence*.
- Guesalaga, A., Neichel, B., Cortés, A., Béchet, C. & Guzmán, D. (2014), "Using the
  Cn2 and wind profiler method with wide-field laser-guide-stars adaptive optics to
  quantify the frozen-flow decay", MNRAS 440, 1925-1933, doi:10.1093/mnras/stu366.
- Roddier, F. (1981), "The effects of atmospheric turbulence in optical astronomy",
  Progress in Optics 19, 281-376.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from aocore import RAD_TO_ARCSEC
from numpy.typing import ArrayLike
from scipy import integrate, special

from . import _boiling
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
    "covariance",
    "space_time_structure_function",
    "travel_decorrelation_time",
    "gems_boil_time",
    "scintillation_transfer",
    "scintillation_index",
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


# ---------------------------------------------------------------------------
# Space-time covariance (frozen flow + boiling)
# ---------------------------------------------------------------------------
# Retention below 1e-17 beyond this many time constants (both boil models).
_S_MAX = 40.0
# J0 periods integrated explicitly in _boil_g; the rest of the oscillating
# tail is below ~1e-7 of the result.
_OSC_CYCLES = 500
# Separations above this many distinct values are interpolated from a table.
_DIRECT_MAX = 64
_GL_X, _GL_W = np.polynomial.legendre.leggauss(12)
_DECADE_PANELS = 10


def _gauss_legendre(edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of 12-point Gauss-Legendre on each panel between ``edges``."""
    lo, hi = edges[:-1, None], edges[1:, None]
    half = 0.5 * (hi - lo)
    return (0.5 * (lo + hi) + half * _GL_X).ravel(), (half * _GL_W).ravel()


def _log_edges(lo: float, hi: float) -> np.ndarray:
    """Geometric panel edges from ``lo`` to ``hi``, ``_DECADE_PANELS`` per decade."""
    count = max(1, int(np.ceil(_DECADE_PANELS * np.log10(hi / lo))))
    return np.geomspace(lo, hi, count + 1)


def _radial_weight(f: np.ndarray, a: float) -> np.ndarray:
    """``2 pi f PSD(f)`` for ``r0 = 1``, ``a = 1/L0^2``: the Hankel-transform weight."""
    return 2.0 * np.pi * _PSD_CONSTANT * f * (f * f + a) ** (-11.0 / 6.0)


def _one_minus_j0(x: np.ndarray) -> np.ndarray:
    """``1 - J0(x)``, accurate for small ``x``."""
    x2 = x * x
    return np.where(x < 1e-3, x2 / 4.0 - x2 * x2 / 64.0, 1.0 - special.j0(x))


class _KolmogorovBoil:
    """Boiled von Kármán covariance for ``r0 = 1`` at one lag (spectral-engine model).

    Each mode keeps ``retention(|lag| / tau(f))`` of its correlation, with
    ``tau(f) = tau_boil * max(f L0, 1)^(-2/3)``. Splits the half structure
    function ``C(0) - B_lag(r)`` into ``s0 + g(r)``:

    - ``s0 = 2 pi int f PSD (1 - rho) df``, the point-wise decorrelation
      (``r = 0``);
    - ``g(r) = 2 pi int f PSD rho (1 - J0(2 pi f r)) df``, the spatial part.

    Neither difference cancels numerically, so both stay accurate where they
    are small (short lags, close points).
    """

    def __init__(self, L0: float, tau: float, lag: float, model: str) -> None:
        self.a = 1.0 / L0 ** 2
        self.f_ref = 1.0 / L0
        self.s_ref = lag / tau  # lag in time constants at and below f_ref
        self.model = model
        # Beyond f_max every mode has decorrelated (retention < 1e-17).
        self.f_max = self.f_ref * max(1.0, (_S_MAX / self.s_ref) ** 1.5)
        a, f_ref = self.a, self.f_ref
        # Below f_ref the retention is constant: closed form.
        low = (_boiling.decorrelation(np.float64(self.s_ref), model)
               * 2.0 * np.pi * _PSD_CONSTANT * 0.6
               * (a ** (-5.0 / 6.0) - (2.0 * a) ** (-5.0 / 6.0)))
        mid = 0.0
        if self.f_max > f_ref:
            f, w = _gauss_legendre(_log_edges(f_ref, self.f_max))
            mid = np.sum(w * _radial_weight(f, a)
                         * _boiling.decorrelation(self._s(f), model))
        # Above f_max: fully decorrelated, int f (f^2 + a)^(-11/6) in closed form.
        tail = 2.0 * np.pi * _PSD_CONSTANT * 0.6 * (self.f_max ** 2 + a) ** (-5.0 / 6.0)
        self.s0 = float(low + mid + tail)

    def _s(self, f: np.ndarray) -> np.ndarray:
        return self.s_ref * np.maximum(f / self.f_ref, 1.0) ** (2.0 / 3.0)

    def _g_one(self, r: float) -> float:
        a, f_ref, f_max = self.a, self.f_ref, self.f_max
        half_period = 0.5 / r  # of J0(2 pi f r) in f
        # Oscillating part: panels at most half a J0 period wide.
        f_osc = min(f_max, _OSC_CYCLES / r)
        edges = [np.linspace(0.0, f_ref, max(1, int(np.ceil(f_ref / half_period))) + 1)]
        top = f_ref
        if f_osc > f_ref:
            # Geometric panels until they reach half a period, then uniform.
            switch = min(f_osc, half_period / (10.0 ** (1.0 / _DECADE_PANELS) - 1.0))
            if switch > f_ref:
                edges.append(_log_edges(f_ref, switch)[1:])
                top = switch
            if f_osc > top:
                count = int(np.ceil((f_osc - top) / half_period))
                edges.append(np.linspace(top, f_osc, count + 1)[1:])
                top = f_osc
        f, w = _gauss_legendre(np.concatenate(edges))
        rho = _boiling.retention(self._s(f), self.model)
        total = np.sum(w * _radial_weight(f, a) * rho
                       * _one_minus_j0(2.0 * np.pi * f * r))
        if f_max > top:
            # J0 has averaged out: 1 - J0 -> 1 (the remaining oscillating
            # integral is below ~1e-7 of the total).
            f, w = _gauss_legendre(_log_edges(top, f_max))
            total += np.sum(w * _radial_weight(f, a)
                            * _boiling.retention(self._s(f), self.model))
        return float(total)

    def g(self, r: np.ndarray) -> np.ndarray:
        """``g`` at separations ``r`` [m] (any shape), tabulated for many values."""
        flat = r.ravel()
        out = np.zeros_like(flat)
        positive = flat > 0
        values, inverse = np.unique(flat[positive], return_inverse=True)
        if values.size == 0:
            return out.reshape(r.shape)
        if values.size <= _DIRECT_MAX:
            g = np.array([self._g_one(v) for v in values])
        else:
            # g is a smooth, positive, power-law-like function of r: a cubic
            # spline in log-log on 40 nodes per decade is accurate to ~1e-7.
            lo, hi = values[0], values[-1]
            count = max(16, int(np.ceil(40 * np.log10(hi / lo))) + 1)
            grid = np.geomspace(lo, hi, count)
            table = np.array([self._g_one(v) for v in grid])
            from scipy.interpolate import CubicSpline

            spline = CubicSpline(np.log(grid), np.log(table))
            g = np.exp(spline(np.log(values)))
            g[0], g[-1] = table[0], table[-1]
        out[positive] = g[inverse]
        return out.reshape(r.shape)


def _points(points: ArrayLike, name: str) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"{name} must be one (x, y) point or an (N, 2) array of them [m]; got "
            f"shape {np.shape(points)}"
        )
    return pts


def _space_time_half(
    points_a: ArrayLike,
    points_b: ArrayLike,
    lag: Union[float, ArrayLike],
    layers: Sequence[Layer],
    r0: float,
    L0: Optional[float],
    tau_boil: Union[float, Sequence[float], None],
    boil_model: str,
    boil_scaling: str,
    need_variance: bool,
) -> Tuple[float, np.ndarray]:
    """``(C(0), C(0) - B)``: total variance and half the structure function, rad^2.

    ``B`` is the covariance between ``phi(a, t)`` and ``phi(b, t + lag)``,
    shape ``(Na, Nb)`` or ``(T, Na, Nb)`` for a 1-D ``lag``.
    """
    _boiling.check_model(boil_model)
    _boiling.check_scaling(boil_scaling)
    if not np.isfinite(r0) or r0 <= 0:
        raise ValueError("r0 must be positive and finite [m]")
    a = _points(points_a, "points_a")
    b = _points(points_b, "points_b")
    lags = np.asarray(lag, dtype=np.float64)
    if lags.ndim > 1 or not np.all(np.isfinite(lags)):
        raise ValueError("lag must be a finite scalar or 1-D array of lags [s]")
    layers = list(layers)
    fractions = _fractions(layers)
    if tau_boil is None:
        taus = np.full(len(layers), np.inf)
    else:
        taus = np.asarray(tau_boil, dtype=np.float64)
        if taus.ndim == 0:
            taus = np.full(len(layers), float(taus))
        elif taus.shape != (len(layers),):
            raise ValueError("tau_boil must be a scalar or one value per layer")
        if np.any(np.isnan(taus)) or np.any(taus <= 0):
            raise ValueError("tau_boil values must be positive (inf: frozen layer)")
    base = b[None, :, :] - a[:, None, :]  # (Na, Nb, 2)
    variance = 0.0
    half = np.zeros(lags.shape + base.shape[:2])
    # Boiling layers sharing (L0, tau_boil) share one numerical transform per lag.
    groups: Dict[Tuple[float, float, Tuple[int, ...]],
                 List[Tuple[float, np.ndarray]]] = {}
    for layer, fraction, tau in zip(layers, fractions, taus):
        if fraction <= 0:
            continue
        layer_L0 = float(layer.L0 if L0 is None else L0)
        if np.isnan(layer_L0) or layer_L0 <= 0:
            raise ValueError("L0 must be positive (numpy.inf for Kolmogorov)")
        finite_L0 = bool(np.isfinite(layer_L0))
        if need_variance and not finite_L0:
            raise ValueError(
                "covariance needs a finite outer scale L0: the Kolmogorov "
                "(L0=inf) phase variance is infinite. Use "
                "space_time_structure_function, which is finite for L0=inf "
                "without boiling."
            )
        boils = bool(np.isfinite(tau)) and bool(np.any(lags != 0.0))
        if boils and not finite_L0:
            raise ValueError(
                "boiling with a Kolmogorov layer (L0=inf) has no finite "
                "space-time statistics: modes at every scale decorrelate while "
                "the largest carry unbounded power. pyturb's spectral engine "
                "stays finite only through its finite grid (its boiling "
                "reference frequency is then the grid fundamental); give the "
                "layer a finite L0."
            )
        weight = fraction * r0 ** (-5.0 / 3.0)  # r0_i^(-5/3)
        c0 = phase_covariance(0.0, 1.0, layer_L0).item() if finite_L0 else np.inf
        variance += weight * c0
        wind = np.asarray(layer.wind_vector, dtype=np.float64)
        for index, value in np.ndenumerate(lags):
            sep = np.linalg.norm(base + wind * value, axis=-1)
            if not np.isfinite(tau) or value == 0.0:
                half[index] += weight * 0.5 * structure_function(sep, 1.0, layer_L0)
            elif boil_scaling == "uniform":
                rho = float(_boiling.retention(abs(value) / tau, boil_model))
                frozen = c0 - phase_covariance(sep, 1.0, layer_L0)
                half[index] += weight * ((1.0 - rho) * c0 + rho * frozen)
            else:
                key = (layer_L0, float(tau), index)
                groups.setdefault(key, []).append((weight, sep))
    for (layer_L0, tau, index), members in groups.items():
        model = _KolmogorovBoil(layer_L0, tau, abs(float(lags[index])), boil_model)
        g = model.g(np.stack([sep for _weight, sep in members]))
        for (weight, _sep), g_layer in zip(members, g):
            half[index] += weight * (model.s0 + g_layer)
    return variance, half


def _unit_scale(units: str, wavelength: float) -> float:
    if units == "phase":
        return 1.0
    if units == "opd":
        return (wavelength / (2.0 * np.pi)) ** 2
    raise ValueError(f"units must be 'phase' or 'opd'; got {units!r}")


def covariance(
    points_a: ArrayLike,
    points_b: ArrayLike,
    lag: Union[float, ArrayLike] = 0.0,
    *,
    layers: Sequence[Layer],
    r0: float,
    L0: Optional[float] = None,
    tau_boil: Union[float, Sequence[float], None] = None,
    boil_model: str = "ar1",
    boil_scaling: str = "kolmogorov",
    units: str = "phase",
    wavelength: float = 500e-9,
) -> np.ndarray:
    """Space-time covariance ``<phi(a, t) phi(b, t + lag)>`` of a layered atmosphere.

    The statistics :class:`pyturb.Atmosphere` simulates, for estimator design
    (MMSE, predictive, Kalman): von Kármán layers under frozen flow, each
    following pyturb's convention ``phi(r, t) = phi_0(r + wind_vector * t)``,
    optionally boiling. Layer ``i`` contributes ``f_i B_i(|b - a +
    wind_vector_i * lag|)`` (``f_i`` its normalised ``cn2_fraction``) with

    ``B_i(rho) = 2 pi int f PSD_i(f) R(|lag| / tau_i(f)) J0(2 pi f rho) df``,

    the von Kármán covariance (:func:`pyturb.phase_covariance`) when the layer
    does not boil or ``lag = 0``. ``R`` is the per-mode retention of
    ``boil_model``: ``exp(-s)`` for ``"ar1"``, ``(1 + 2s) exp(-2s)`` for
    ``"ar2"``. With ``boil_scaling="kolmogorov"`` (the spectral engine) a mode
    at spatial frequency ``f`` has ``tau(f) = tau_boil * (f L0)^(-2/3)`` for
    ``f >= 1/L0`` and ``tau_boil`` below, so ``tau_boil`` is the lifetime of the
    outer-scale structure and fine structure decorrelates faster; the Hankel
    transform is then evaluated numerically (relative accuracy ~1e-7). With
    ``"uniform"`` (the extruder, and the spectral engine's
    ``boil_scaling="uniform"``) every scale has ``tau_boil``, and the
    covariance is the frozen-flow one times ``R(|lag| / tau_boil)``.

    Parameters
    ----------
    points_a, points_b : array_like
        Pupil-plane points [m], one ``(x, y)`` or an ``(N, 2)`` array, with x
        along axis 1 (columns) and y along axis 0 (rows), as in
        :meth:`pyturb.Atmosphere.opd_at`.
    lag : float or 1-D array_like
        Time [s] from ``points_a`` to ``points_b``. A 1-D array adds a
        leading lag axis to the result.
    layers : sequence of Layer
        The profile: ``cn2_fraction`` (normalised to sum to 1), wind vector and
        ``L0`` of each layer. Altitudes are not used (one source at infinity).
    r0 : float
        Total Fried parameter [m] along the line of sight, at ``wavelength``.
    L0 : float, optional
        Outer scale [m] for every layer, overriding each layer's own. Must be
        finite (the Kolmogorov variance is infinite); see
        :func:`space_time_structure_function` for ``L0=inf``.
    tau_boil : float or sequence, optional
        Boiling time constant [s], scalar or one per layer (``numpy.inf`` for a
        frozen layer), as :class:`pyturb.Atmosphere`'s ``tau_boil``. ``None``
        (default) is pure frozen flow.
    boil_model : {"ar1", "ar2"}, optional
        Temporal boiling model, as :class:`pyturb.Atmosphere`'s ``boil_model``.
    boil_scaling : {"kolmogorov", "uniform"}, optional
        Scale dependence of boiling, as :class:`pyturb.Atmosphere`'s
        ``boil_scaling``: ``"kolmogorov"`` (default, the spectral engine's
        default) or ``"uniform"`` (the extruder, or the spectral engine with
        ``boil_scaling="uniform"``).
    units : {"phase", "opd"}, optional
        ``"phase"`` (default): rad^2 at ``wavelength``. ``"opd"``: m^2.
    wavelength : float, optional
        Wavelength [m] ``r0`` is given at. Default 500 nm.

    Returns
    -------
    ndarray
        ``(Na, Nb)``, or ``(T, Na, Nb)`` for ``T`` lags.

    Notes
    -----
    The result is a stationary, infinite-screen statistic. A simulated screen
    differs in what its finite grid cannot hold: power above the grid's
    Nyquist frequency, and on the periodic spectral engine the largest scales
    (see ``oversample``). Differences such as the structure function are far
    less sensitive to the large scales than the covariance itself.

    Examples
    --------
    >>> import numpy as np, pyturb
    >>> layers = [pyturb.Layer(0.0, 0.7, 10.0, 0.0), pyturb.Layer(8e3, 0.3, 25.0, 90.0)]
    >>> pts = np.array([[0.0, 0.0], [0.5, 0.0]])
    >>> c = pyturb.theory.covariance(pts, pts, lag=2e-3, layers=layers, r0=0.15,
    ...                              tau_boil=1.0, units="opd")
    >>> c.shape
    (2, 2)
    """
    variance, half = _space_time_half(
        points_a, points_b, lag, layers, r0, L0, tau_boil, boil_model,
        boil_scaling, need_variance=True,
    )
    return (variance - half) * _unit_scale(units, wavelength)


def space_time_structure_function(
    points_a: ArrayLike,
    points_b: ArrayLike,
    lag: Union[float, ArrayLike] = 0.0,
    *,
    layers: Sequence[Layer],
    r0: float,
    L0: Optional[float] = None,
    tau_boil: Union[float, Sequence[float], None] = None,
    boil_model: str = "ar1",
    boil_scaling: str = "kolmogorov",
    units: str = "phase",
    wavelength: float = 500e-9,
) -> np.ndarray:
    """Space-time structure function ``<[phi(a, t) - phi(b, t + lag)]^2>``.

    ``2 [C(0) - B]`` with ``B`` from :func:`covariance` (same arguments and
    model), computed without that subtraction, so it stays accurate for close
    points and short lags. Unlike :func:`covariance` it is finite for
    Kolmogorov layers (``L0=inf``), as long as they do not boil: with boiling,
    modes at every scale decorrelate while carrying unbounded power. At
    ``lag=0`` it is :func:`structure_function` summed over the layers; at
    ``a = b`` it is the temporal structure function of one pupil point (the
    mean-square change over ``lag``).
    """
    _variance, half = _space_time_half(
        points_a, points_b, lag, layers, r0, L0, tau_boil, boil_model,
        boil_scaling, need_variance=False,
    )
    return 2.0 * half * _unit_scale(units, wavelength)


def travel_decorrelation_time(
    speed: ArrayLike,
    per_metre: float = 0.157,
    per_second: float = 0.365,
) -> Union[float, np.ndarray]:
    """Boiling time constant [s] of a layer from the distance its wind carries it.

    ``tau = 1 / (per_metre * |speed| + per_second)``. Guesalaga et al. (2014)
    followed the turbulence of individual layers with the GeMS wide-field
    laser-guide-star AO system and found each layer's correlation decaying
    roughly linearly with lag, ``rho(lag) ~ 1 - r lag``, with a rate that
    grows with wind speed, ``r = 0.157 v + 0.365`` s^-1 (``v`` in m/s): a
    layer loses correlation mainly with the distance it has travelled, plus a
    slower part that does not depend on the wind. The defaults are that fit;
    ``1 / r`` is the time constant whose AR(1) retention ``exp(-lag/tau)`` has
    the same initial slope.

    The fit covers 74 GeMS layer sequences with winds of 0-24 m/s, as a
    whole-layer, roughly linear decay; it says nothing about how the decay
    depends on spatial scale. Use it with ``boil_scaling="uniform"``, which
    gives every scale this time constant. With the default
    ``boil_scaling="kolmogorov"``, ``tau_boil`` is the lifetime of the outer
    scale only and the scales an AO system senses would boil one to two
    orders of magnitude faster than measured. ``boil_model="ar2"`` keeps the
    same integral time scale but starts flat (quadratic) instead of linear at
    short lags. Treat values outside the fitted speed range as extrapolation.

    Parameters
    ----------
    speed : float or array_like
        Layer wind speed [m/s]; the sign is ignored. An array gives one time
        constant per layer, e.g. ``[layer.wind_speed for layer in layers]``.
    per_metre : float, optional
        Decorrelation per metre of travel [m^-1]. Default 0.157.
    per_second : float, optional
        Wind-independent decorrelation rate [s^-1]. Default 0.365.

    Returns
    -------
    float or ndarray
        ``tau`` [s], a float for a scalar ``speed``, else an array of its
        shape. ``inf`` (frozen flow) where the rate is zero.

    References
    ----------
    Guesalaga, A., Neichel, B., Cortés, A., Béchet, C. & Guzmán, D. (2014),
    "Using the Cn2 and wind profiler method with wide-field laser-guide-stars
    adaptive optics to quantify the frozen-flow decay", MNRAS 440, 1925-1933,
    doi:10.1093/mnras/stu366: per-layer frozen-flow decay measured on GeMS.

    Examples
    --------
    >>> import pyturb
    >>> layers = pyturb.get_profile("two-layer")
    >>> tau = pyturb.theory.travel_decorrelation_time(
    ...     [layer.wind_speed for layer in layers])
    >>> atm = pyturb.Atmosphere(layers, r0=0.15, n=64, tau_boil=tau,
    ...                         boil_scaling="uniform")
    """
    for name, value in (("per_metre", per_metre), ("per_second", per_second)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"{name} must be finite and >= 0 (a decorrelation rate); got "
                f"{value!r}"
            )
    v = np.abs(np.asarray(speed, dtype=np.float64))
    if not np.all(np.isfinite(v)):
        raise ValueError("speed must be finite [m/s]")
    rate = per_metre * v + per_second
    with np.errstate(divide="ignore"):
        tau = np.where(rate > 0, 1.0 / np.where(rate > 0, rate, 1.0), np.inf)
    return float(tau) if tau.ndim == 0 else tau


# Outer-scale boiling lifetimes [s] at these wind speeds [m/s] (gems_boil_time).
_GEMS_SPEEDS = (5.0, 10.0, 20.0)
_GEMS_TAU_B = (2.5, 1.4, 0.74)


def gems_boil_time(speed: ArrayLike) -> Union[float, np.ndarray]:
    """Outer-scale boiling lifetime [s] for ``boil_scaling="kolmogorov"``, from GeMS.

    The ``tau_boil`` per layer for the spectral engine's default,
    scale-dependent boiling (each mode at spatial frequency ``f`` has
    ``tau(f) = tau_boil * max(f L0, 1)^(-2/3)``), derived from the decorrelation
    Guesalaga et al. (2014) measured on GeMS. They followed each layer in the
    space-time cross-correlation of tip/tilt-removed laser-guide-star slopes (8 m
    pupil, 16x16 subapertures of 0.5 m), deconvolved by the zero-lag
    autocorrelation, and took the slope of the energy in a 3x3-subaperture window
    around the correlation peak beyond 0.1 s, finding a rate ``0.157 v + 0.365``
    s^-1 for wind speed ``v`` (see :func:`travel_decorrelation_time`). The
    lifetimes here, 2.5, 1.4 and 0.74 s at 5, 10 and 20 m/s (2.50, 1.44, 0.74 s
    before rounding), are those for which the analytic windowed energy of the
    von Kármán spectrum (``L0 = 25 m``, AR(1) modes, not deconvolved) has that
    rate as the slope of its logarithm over 0.1-0.5 s. In between, ``1 / tau`` is
    linear in speed; outside 5-20 m/s the end values are held.

    The measured rate is a linear slope of the deconvolved statistic, not a
    logarithmic one. Under that definition, AR(1) modes with these lifetimes
    decay at about 0.9, 0.6 and 0.5 times the published rate at 5, 10 and 20 m/s,
    and no AR(1) law reaches the published rate above about 6 m/s: over 0.1-0.5 s
    the linear slope of any mixture of exponential decays is at most about
    1.3 s^-1. The GeMS decay is linear in time and includes the dispersion of the
    wind speed within a layer, which AR(1) boiling of a single layer does not
    represent. Treat these lifetimes as an approximate, AR(1) description of the
    GeMS measurements, faithful in order of magnitude.

    Use it with the default ``boil_scaling="kolmogorov"`` and
    ``boil_model="ar1"``; with ``"uniform"`` scaling use
    :func:`travel_decorrelation_time` instead, which describes the same
    measurements as one rate for every scale. The calibration assumes
    ``L0 = 25 m``: the outer-scale lifetime that gives the same decay of the
    sensed scales grows with ``L0`` (recalibrated at 10 m/s, 0.63, 1.49 and 2.3
    times these values for ``L0`` = 10, 50 and 100 m), so treat other outer
    scales as an extrapolation. The calibration covers lags of 0.1-0.5 s; its
    extrapolation to millisecond lags is not constrained by the GeMS data.

    Parameters
    ----------
    speed : float or array_like
        Layer wind speed [m/s]; the sign is ignored. An array gives one value
        per layer, e.g. ``[layer.wind_speed for layer in layers]``.

    Returns
    -------
    float or ndarray
        ``tau_boil`` [s], a float for a scalar ``speed``, else an array of its
        shape.

    References
    ----------
    Guesalaga, A., Neichel, B., Cortés, A., Béchet, C. & Guzmán, D. (2014),
    MNRAS 440, 1925-1933, doi:10.1093/mnras/stu366.

    Examples
    --------
    >>> import pyturb
    >>> layers = pyturb.get_profile("two-layer")
    >>> tau = pyturb.theory.gems_boil_time([layer.wind_speed for layer in layers])
    >>> atm = pyturb.Atmosphere(layers, r0=0.15, n=64, tau_boil=tau)
    """
    v = np.abs(np.asarray(speed, dtype=np.float64))
    if not np.all(np.isfinite(v)):
        raise ValueError("speed must be finite [m/s]")
    rate = np.interp(v, _GEMS_SPEEDS, 1.0 / np.asarray(_GEMS_TAU_B))
    tau = 1.0 / rate
    return float(tau) if tau.ndim == 0 else tau


def scintillation_transfer(
    f: ArrayLike,
    altitude: ArrayLike,
    wavelength: ArrayLike,
) -> np.ndarray:
    """Weak-perturbation transfer [1/m] from a layer's OPD to relative intensity.

    For a thin layer at range ``h`` above the pupil with a small OPD ``w``, a
    plane wave arrives with relative intensity ``1 + delta`` whose spectrum is
    ``delta(f) = T(f) w(f)``, with

    ``T(f) = 2 k sin(pi lambda h f^2)``, ``k = 2 pi / lambda``,

    the Fresnel filter of first-order (Rytov/Born) perturbation theory, in the
    sign convention of :meth:`pyturb.Atmosphere.pupil_field` (phase
    ``exp(+i k w)``): a positive OPD bump (a converging lens) brightens the
    pupil. Below the Fresnel frequency ``1 / sqrt(lambda h)`` it tends to the
    achromatic transport-of-intensity limit ``4 pi^2 h f^2``, i.e.
    ``delta = -h laplacian(w)``. The layers of a profile add their
    contributions; for a band, average ``T`` over the wavelengths with the
    weights of the intensity. Multiplying ``|T|^2`` by the OPD power spectrum
    and integrating gives the scintillation index
    (:func:`scintillation_index`); multiplying ``T`` by other transfer
    functions gives cross-spectra, e.g. for the covariance between intensity
    and slopes.

    Parameters
    ----------
    f : array_like
        Spatial frequency [cycles/m] (its magnitude).
    altitude : float or array_like
        Range of the layer above the pupil [m] along the line of sight.
    wavelength : float or array_like
        Wavelength [m].

    Returns
    -------
    ndarray
        ``T`` [relative intensity per metre of OPD], broadcast over the inputs.
    """
    f = np.asarray(f, dtype=np.float64)
    h = np.asarray(altitude, dtype=np.float64)
    lam = np.asarray(wavelength, dtype=np.float64)
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        raise ValueError("altitude must be finite and >= 0 [m]")
    if np.any(lam <= 0) or not np.all(np.isfinite(lam)):
        raise ValueError("wavelength must be positive and finite [m]")
    return 2.0 * (2.0 * np.pi / lam) * np.sin(np.pi * lam * h * f * f)


def scintillation_index(
    layers: Sequence[Layer],
    r0: float,
    wavelength: ArrayLike,
    *,
    r0_wavelength: float = 500e-9,
    zenith_angle: float = 0.0,
) -> Union[float, np.ndarray]:
    """Scintillation index ``sigma_I^2`` of a star through a layered profile (Roddier).

    The variance of the relative intensity at a point in the pupil, in the
    weak-perturbation (Rytov) approximation for Kolmogorov turbulence:

    ``sigma_I^2 = 19.2 lambda^(-7/6) sec(z)^(11/6) integral Cn^2(h) h^(5/6) dh``

    (four times the log-amplitude variance), with each layer's ``Cn^2 dh =
    f_i r0^(-5/3) / (0.423 k_0^2)`` from the profile's normalised
    ``cn2_fraction`` ``f_i`` and the total ``r0`` at ``r0_wavelength``
    (``k_0 = 2 pi / r0_wavelength``). It is the integral of
    :func:`scintillation_transfer` squared times the Kolmogorov OPD spectrum,
    so it scales as ``lambda^(-7/6)`` and is dominated by the high layers.

    It describes a point source seen through a point aperture: a telescope
    larger than the Fresnel scale ``sqrt(lambda h)`` averages the fluctuations
    down. The weak-perturbation result holds while ``sigma_I^2`` is well
    below 1; stronger scintillation saturates. The outer scale hardly matters
    (the Fresnel filter suppresses scales larger than ``sqrt(lambda h)``).
    A simulation reproduces it only up to its grid: the power above the
    Nyquist frequency of the pupil sampling is missing (several percent unless
    the pixel is well below the Fresnel scale).

    Parameters
    ----------
    layers : sequence of Layer
        The profile: altitudes [m] at zenith and ``cn2_fraction`` weights.
    r0 : float
        Total Fried parameter [m] at zenith, at ``r0_wavelength``.
    wavelength : float or array_like
        Observing wavelength(s) [m].
    r0_wavelength : float, optional
        Wavelength [m] ``r0`` is quoted at. Default 500 nm.
    zenith_angle : float, optional
        Zenith angle [deg]: ranges and the path through each layer grow as
        ``sec(z)``. Default 0.

    Returns
    -------
    float or ndarray
        ``sigma_I^2`` (dimensionless), a float for a scalar ``wavelength``.

    References
    ----------
    Roddier, F. (1981), Progress in Optics 19, 281-376.

    Examples
    --------
    >>> import pyturb
    >>> layers = pyturb.get_profile("paranal-median")
    >>> s2 = pyturb.theory.scintillation_index(layers, r0=0.15, wavelength=0.5e-6)
    """
    if not np.isfinite(r0) or r0 <= 0:
        raise ValueError("r0 must be positive and finite [m]")
    if not np.isfinite(r0_wavelength) or r0_wavelength <= 0:
        raise ValueError("r0_wavelength must be positive and finite [m]")
    if not np.isfinite(zenith_angle) or not 0.0 <= zenith_angle < 90.0:
        raise ValueError("zenith_angle must be in [0, 90) degrees")
    lam = np.asarray(wavelength, dtype=np.float64)
    if np.any(lam <= 0) or not np.all(np.isfinite(lam)):
        raise ValueError("wavelength must be positive and finite [m]")
    layers = list(layers)
    fractions = _fractions(layers)
    h = np.array([layer.altitude for layer in layers], dtype=np.float64)
    if np.any(h < 0):
        raise ValueError("layer altitudes must be >= 0 [m]")
    k0 = 2.0 * np.pi / r0_wavelength
    cn2_dh = fractions * r0 ** (-5.0 / 3.0) / (0.423 * k0 * k0)  # [m^(1/3)]
    moment = float(np.sum(cn2_dh * h ** (5.0 / 6.0)))
    secant = 1.0 / np.cos(np.deg2rad(zenith_angle))
    index = 19.2 * lam ** (-7.0 / 6.0) * secant ** (11.0 / 6.0) * moment
    return float(index) if index.ndim == 0 else index

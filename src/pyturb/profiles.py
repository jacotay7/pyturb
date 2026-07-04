"""Turbulence-profile building blocks and standard atmospheres.

A profile is simply a list of :class:`Layer` objects: each carries a fraction
of the total turbulence (:math:`C_n^2\\,dh`), an altitude, a wind vector and an
outer scale. :func:`get_profile` returns ready-made, named profiles for common
sites and teaching cases; :func:`hufnagel_valley` and :func:`discretize_cn2`
build custom ones from a continuous :math:`C_n^2(h)` model.

Integrated quantities that AO users always need — isoplanatic angle
:math:`\\theta_0`, coherence time :math:`\\tau_0`, Greenwood frequency — are
provided as free functions that take a profile plus a total :math:`r_0`.

All altitudes and winds are given *at zenith*; :class:`pyturb.Atmosphere`
applies the airmass scaling for a chosen zenith angle.

References
----------
- Hardy, J. W. (1998), *Adaptive Optics for Astronomical Telescopes*.
- Roddier, F. (1999), *Adaptive Optics in Astronomy*.
- Hufnagel & Valley continuous :math:`C_n^2` model (HV 5/7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

__all__ = [
    "Layer",
    "get_profile",
    "list_profiles",
    "hufnagel_valley",
    "bufton_wind",
    "discretize_cn2",
    "isoplanatic_angle",
    "coherence_time",
    "greenwood_frequency",
    "mean_turbulence_height",
    "effective_wind_speed",
]


@dataclass
class Layer:
    """One turbulent layer.

    Parameters
    ----------
    altitude : float
        Height above the telescope [m], at zenith.
    cn2_fraction : float
        Fraction of the total integrated turbulence (:math:`C_n^2\\,dh`) in
        this layer. Fractions across a profile are normalised to sum to 1 by
        :class:`pyturb.Atmosphere`; relative values are what matter.
    wind_speed : float
        Wind speed [m/s].
    wind_direction : float
        Wind direction [deg], measured from axis 0 toward axis 1. Arbitrary
        (need not be axis-aligned).
    L0 : float
        Outer scale [m] for this layer. Default 25 m.
    """

    altitude: float
    cn2_fraction: float
    wind_speed: float = 10.0
    wind_direction: float = 0.0
    L0: float = 25.0

    @property
    def wind_vector(self):
        """``(vx, vy)`` wind components [m/s] along axes 0 and 1."""
        theta = np.deg2rad(self.wind_direction)
        return self.wind_speed * np.cos(theta), self.wind_speed * np.sin(theta)


# ---------------------------------------------------------------------------
# Named profiles
# ---------------------------------------------------------------------------
# Representative profiles for common sites. Altitudes in metres, wind in m/s,
# fractions are relative Cn2 weights (normalised on use). These are convenient,
# citable starting points — edit or build your own with discretize_cn2().

def _single_layer() -> List[Layer]:
    return [Layer(altitude=0.0, cn2_fraction=1.0, wind_speed=10.0, wind_direction=0.0)]


def _two_layer() -> List[Layer]:
    # Ground layer plus a fast high-altitude (jet-stream) layer.
    return [
        Layer(altitude=0.0, cn2_fraction=0.7, wind_speed=8.0, wind_direction=0.0),
        Layer(altitude=10000.0, cn2_fraction=0.3, wind_speed=30.0, wind_direction=90.0),
    ]


def _paranal_median() -> List[Layer]:
    # Condensed, Paranal-like 9-layer median profile (ESO reference, binned).
    altitudes = [30, 140, 281, 562, 1125, 2250, 4500, 9000, 18000]
    fractions = [0.242, 0.12, 0.098, 0.059, 0.043, 0.061, 0.13, 0.13, 0.117]
    winds = [5.5, 6.6, 6.7, 8.0, 9.9, 15.0, 25.0, 32.0, 14.0]
    directions = [0, 20, 40, 60, 90, 120, 150, 180, 210]
    return [
        Layer(a, f, w, d)
        for a, f, w, d in zip(altitudes, fractions, winds, directions)
    ]


def _mauna_kea() -> List[Layer]:
    # Representative Mauna Kea / TMT-style 7-layer profile.
    altitudes = [0, 500, 1000, 2000, 4000, 8000, 16000]
    fractions = [0.35, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10]
    winds = [6.0, 7.0, 8.0, 10.0, 15.0, 28.0, 12.0]
    directions = [0, 30, 60, 90, 120, 150, 180]
    return [
        Layer(a, f, w, d)
        for a, f, w, d in zip(altitudes, fractions, winds, directions)
    ]


def _hv57(n_layers: int = 10) -> List[Layer]:
    heights = np.geomspace(10.0, 25000.0, 4096)
    cn2 = hufnagel_valley(heights)
    return discretize_cn2(heights, cn2, n_layers=n_layers, wind="bufton")


_PROFILES = {
    "single-layer": _single_layer,
    "two-layer": _two_layer,
    "paranal-median": _paranal_median,
    "mauna-kea": _mauna_kea,
    "hv57": _hv57,
}


def list_profiles() -> List[str]:
    """Names accepted by :func:`get_profile` and ``Atmosphere.from_profile``."""
    return sorted(_PROFILES)


def get_profile(name: str) -> List[Layer]:
    """Return a fresh list of :class:`Layer` for a named profile.

    Names: ``"single-layer"``, ``"two-layer"``, ``"paranal-median"``,
    ``"mauna-kea"``, ``"hv57"``. See :func:`list_profiles`.
    """
    key = str(name).lower()
    if key not in _PROFILES:
        raise ValueError(
            f"Unknown profile {name!r}. Available: {list_profiles()}."
        )
    return _PROFILES[key]()


# ---------------------------------------------------------------------------
# Continuous models and discretisation
# ---------------------------------------------------------------------------
def hufnagel_valley(h, wind_rms=21.0, ground=1.7e-14):
    r"""Hufnagel-Valley :math:`C_n^2(h)` model [m^{-2/3}].

    Parameters
    ----------
    h : array_like
        Altitude(s) above the telescope [m].
    wind_rms : float
        High-altitude wind pseudo-parameter :math:`v` [m/s]. ``21`` with
        ``ground=1.7e-14`` gives the standard "HV 5/7" profile (r0 ~ 5 cm,
        isoplanatic angle ~ 7 urad at 500 nm).
    ground : float
        Ground-layer coefficient :math:`A` [m^{-2/3}].
    """
    h = np.asarray(h, dtype=np.float64)
    return (
        0.00594 * (wind_rms / 27.0) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000.0)
        + 2.7e-16 * np.exp(-h / 1500.0)
        + ground * np.exp(-h / 100.0)
    )


def bufton_wind(h):
    """Bufton wind-speed model [m/s] versus altitude [m] (peaks near 9.4 km)."""
    h = np.asarray(h, dtype=np.float64)
    return 5.0 + 30.0 * np.exp(-(((h - 9400.0) / 4800.0) ** 2))


def discretize_cn2(heights, cn2, n_layers=10, wind="bufton", L0=25.0):
    r"""Bin a continuous :math:`C_n^2(h)` profile into equivalent layers.

    The altitude range is split into ``n_layers`` contiguous log-spaced bins;
    each output layer carries that bin's integrated :math:`C_n^2\,dh` (so total
    turbulence is preserved) and sits at the bin's :math:`C_n^2`-weighted mean
    height (preserving the turbulence centroid that sets :math:`\theta_0`).

    Parameters
    ----------
    heights : array_like
        Sorted altitude grid [m].
    cn2 : array_like
        :math:`C_n^2` on that grid [m^{-2/3}].
    n_layers : int
        Number of output layers.
    wind : str, float, or array_like
        ``"bufton"`` for the Bufton model, a scalar for uniform wind, or a
        per-layer array of speeds [m/s].
    L0 : float
        Outer scale assigned to every layer [m].
    """
    heights = np.asarray(heights, dtype=np.float64)
    cn2 = np.asarray(cn2, dtype=np.float64)
    if heights.ndim != 1 or heights.shape != cn2.shape:
        raise ValueError("heights and cn2 must be 1-D arrays of equal length")
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")

    edges = np.geomspace(max(heights[0], 1.0), heights[-1], n_layers + 1)
    edges[0] = heights[0]
    edges[-1] = heights[-1] + 1.0  # include the top point
    layers: List[Layer] = []
    weights, centroids = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (heights >= lo) & (heights < hi)
        if not mask.any():
            continue
        h_bin, c_bin = heights[mask], cn2[mask]
        integral = np.trapezoid(c_bin, h_bin) if h_bin.size > 1 else c_bin[0]
        if integral <= 0:
            continue
        centroid = np.average(h_bin, weights=c_bin)
        weights.append(integral)
        centroids.append(centroid)

    weights = np.asarray(weights)
    weights /= weights.sum()
    centroids = np.asarray(centroids)

    if isinstance(wind, str):
        if wind.lower() != "bufton":
            raise ValueError("wind string must be 'bufton'")
        speeds = bufton_wind(centroids)
    else:
        speeds = np.broadcast_to(np.asarray(wind, dtype=np.float64), weights.shape)

    for h, f, v in zip(centroids, weights, speeds):
        layers.append(Layer(float(h), float(f), float(v), 0.0, L0=L0))
    return layers


# ---------------------------------------------------------------------------
# Integrated quantities
# ---------------------------------------------------------------------------
def _fractions(layers):
    frac = np.asarray([layer.cn2_fraction for layer in layers], dtype=np.float64)
    if np.any(frac < 0):
        raise ValueError("cn2_fraction values must be non-negative")
    total = frac.sum()
    if total <= 0:
        raise ValueError("cn2_fraction values must sum to a positive number")
    return frac / total


def mean_turbulence_height(layers):
    r"""Effective turbulence height :math:`\bar h = (\sum f_i h_i^{5/3})^{3/5}` [m]."""
    frac = _fractions(layers)
    h = np.asarray([layer.altitude for layer in layers], dtype=np.float64)
    return float((np.sum(frac * h ** (5.0 / 3.0))) ** (3.0 / 5.0))


def effective_wind_speed(layers):
    r"""Effective wind speed :math:`\bar v = (\sum f_i v_i^{5/3})^{3/5}` [m/s]."""
    frac = _fractions(layers)
    v = np.asarray([layer.wind_speed for layer in layers], dtype=np.float64)
    return float((np.sum(frac * v ** (5.0 / 3.0))) ** (3.0 / 5.0))


def isoplanatic_angle(layers, r0):
    r"""Isoplanatic angle :math:`\theta_0 = 0.314\, r_0 / \bar h` [rad].

    ``r0`` and the layer altitudes must be expressed along the same
    line of sight (both at zenith, or both scaled for the airmass).
    """
    h_bar = mean_turbulence_height(layers)
    if h_bar == 0:
        return np.inf
    return 0.314 * r0 / h_bar


def coherence_time(layers, r0):
    r"""Atmospheric coherence time :math:`\tau_0 = 0.314\, r_0 / \bar v` [s]."""
    v_bar = effective_wind_speed(layers)
    if v_bar == 0:
        return np.inf
    return 0.314 * r0 / v_bar


def greenwood_frequency(layers, r0):
    r"""Greenwood frequency :math:`f_G = 0.134 / \tau_0` [Hz]."""
    tau0 = coherence_time(layers, r0)
    return 0.134 / tau0

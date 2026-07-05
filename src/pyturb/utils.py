"""Small utilities: seeing/r0 conversions and statistical validation."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from .backend import to_numpy

__all__ = [
    "r0_from_seeing",
    "seeing_from_r0",
    "r0_at_wavelength",
    "opd_to_phase",
    "phase_to_opd",
    "air_refractivity",
    "water_vapour_refractivity",
    "structure_function",
]

_RAD_TO_ARCSEC = 180.0 / np.pi * 3600.0


def air_refractivity(wavelength: ArrayLike) -> Union[float, np.ndarray]:
    """Refractivity ``n - 1`` of standard dry air at ``wavelength`` [m].

    Edlén (1966) dispersion formula for dry air at 15 °C, 101.325 kPa, 0.03 %
    CO2::

        (n - 1) x 1e8 = 8342.13 + 2406030 / (130 - s^2) + 15997 / (38.9 - s^2)

    with ``s = 1 / lambda`` in micron^-1. Turbulence OPD scales with ``n - 1``,
    so the *ratio* of this quantity between two wavelengths is the (weak,
    ~1-2 % across the visible-to-NIR) chromatic correction to an otherwise
    achromatic path length. Temperature/pressure scale ``n - 1`` overall and so
    cancel in that ratio; only the dispersion shape matters here.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength in metres.

    Returns
    -------
    float or ndarray
        ``n - 1`` (dimensionless), same shape as ``wavelength``.
    """
    sigma2 = (1.0 / (np.asarray(wavelength, dtype=np.float64) * 1e6)) ** 2
    refractivity = (
        8342.13 + 2406030.0 / (130.0 - sigma2) + 15997.0 / (38.9 - sigma2)
    ) * 1e-8
    return refractivity if refractivity.ndim else float(refractivity)


def water_vapour_refractivity(wavelength: ArrayLike) -> Union[float, np.ndarray]:
    r"""Dispersion of the water-vapour refractivity at ``wavelength`` [m].

    The wavelength-dependent part of the pure-water-vapour refractivity from
    Ciddor (1996), *Refractive index of air: new equations for the visible and
    near infrared*, Applied Optics 35, 1566::

        (n_wv - 1) x 1e8 = 1.022 (295.235 + 2.6422 s^2
                                  - 0.032380 s^4 + 0.004028 s^6)

    with ``s = 1 / lambda`` in micron^-1. The overall scale depends on the
    water-vapour partial pressure/temperature (a density factor) and cancels in
    any *ratio* between wavelengths, so this returns the dispersion **shape**
    only — exactly what the wet/dry chromatic OPD split needs.

    Water vapour disperses differently from dry air (:func:`air_refractivity`):
    across the visible-to-NIR the two shapes track loosely, but into the
    mid-IR the dry term flattens while the wet term keeps more structure. That
    divergence is the "wet–dry" problem in interferometry, where the
    water-vapour contribution to the turbulent OPD dominates.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength in metres.

    Returns
    -------
    float or ndarray
        A quantity proportional to the water-vapour ``n - 1`` (dimensionless),
        same shape as ``wavelength``.
    """
    sigma2 = (1.0 / (np.asarray(wavelength, dtype=np.float64) * 1e6)) ** 2
    refractivity = (
        1.022e-8
        * (
            295.235
            + 2.6422 * sigma2
            - 0.032380 * sigma2**2
            + 0.004028 * sigma2**3
        )
    )
    return refractivity if refractivity.ndim else float(refractivity)


def opd_to_phase(opd: ArrayLike, wavelength: float) -> Union[float, np.ndarray]:
    """Convert optical path difference [m] to phase [rad] at ``wavelength`` [m].

    ``phase = 2 pi * opd / wavelength``. OPD is achromatic, so the same OPD
    gives different phase at different wavelengths.
    """
    return opd * (2.0 * np.pi / wavelength)


def phase_to_opd(phase: ArrayLike, wavelength: float) -> Union[float, np.ndarray]:
    """Convert phase [rad] at ``wavelength`` [m] to optical path difference [m].

    ``opd = phase * wavelength / (2 pi)``.
    """
    return phase * (wavelength / (2.0 * np.pi))


def r0_from_seeing(seeing: float, wavelength: float = 500e-9) -> float:
    """Fried parameter (m) from seeing FWHM (arcsec) at ``wavelength`` (m).

    Uses the Kolmogorov relation ``FWHM = 0.98 lambda / r0``.
    """
    return 0.98 * wavelength / (seeing / _RAD_TO_ARCSEC)


def seeing_from_r0(r0: float, wavelength: float = 500e-9) -> float:
    """Seeing FWHM (arcsec) from the Fried parameter (m) at ``wavelength`` (m)."""
    return 0.98 * wavelength / r0 * _RAD_TO_ARCSEC


def r0_at_wavelength(r0: float, wavelength_in: float, wavelength_out: float) -> float:
    """Rescale the Fried parameter between wavelengths (``r0 ~ lambda^(6/5)``).

    Example: convert an r0 quoted at 500 nm to the K band,
    ``r0_at_wavelength(0.15, 500e-9, 2.2e-6)``.
    """
    return r0 * (wavelength_out / wavelength_in) ** (6.0 / 5.0)


def structure_function(
    phase: ArrayLike,
    pixel_scale: float = 1.0,
    max_separation: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged (along both axes) phase structure function.

    ``D(r) = <[phase(x) - phase(x + r)]^2>``, estimated from pixel pairs
    separated along each image axis and averaged. For Kolmogorov turbulence
    the expectation is ``D(r) = 6.88 (r / r0)^(5/3)``.

    Parameters
    ----------
    phase : ndarray
        A single screen ``(n, n)`` or a stack ``(count, n, n)``; NumPy or
        CuPy (device arrays are copied to the host).
    pixel_scale : float, optional
        Pixel size in metres; separations are returned in the same unit.
    max_separation : int, optional
        Largest pixel separation to evaluate (default ``n // 4``).

    Returns
    -------
    r, D : ndarray
        Separations and the structure function estimate at each separation.
    """
    phase = to_numpy(phase).astype(np.float64)
    if phase.ndim == 2:
        phase = phase[None]
    n = min(phase.shape[-2], phase.shape[-1])
    if max_separation is None:
        max_separation = n // 4
    max_separation = int(max_separation)
    if not 1 <= max_separation < n:
        raise ValueError("max_separation must be in [1, n)")

    separations = np.arange(1, max_separation + 1)
    result = np.empty(max_separation)
    for i, s in enumerate(separations):
        along_rows = phase[..., s:, :] - phase[..., :-s, :]
        along_cols = phase[..., :, s:] - phase[..., :, :-s]
        result[i] = 0.5 * (np.mean(along_rows**2) + np.mean(along_cols**2))
    return separations * pixel_scale, result

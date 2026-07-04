"""Small utilities: seeing/r0 conversions and statistical validation."""

from __future__ import annotations

import numpy as np

from .backend import to_numpy

__all__ = [
    "r0_from_seeing",
    "seeing_from_r0",
    "r0_at_wavelength",
    "structure_function",
]

_RAD_TO_ARCSEC = 180.0 / np.pi * 3600.0


def r0_from_seeing(seeing, wavelength=500e-9):
    """Fried parameter (m) from seeing FWHM (arcsec) at ``wavelength`` (m).

    Uses the Kolmogorov relation ``FWHM = 0.98 lambda / r0``.
    """
    return 0.98 * wavelength / (seeing / _RAD_TO_ARCSEC)


def seeing_from_r0(r0, wavelength=500e-9):
    """Seeing FWHM (arcsec) from the Fried parameter (m) at ``wavelength`` (m)."""
    return 0.98 * wavelength / r0 * _RAD_TO_ARCSEC


def r0_at_wavelength(r0, wavelength_in, wavelength_out):
    """Rescale the Fried parameter between wavelengths (``r0 ~ lambda^(6/5)``).

    Example: convert an r0 quoted at 500 nm to the K band,
    ``r0_at_wavelength(0.15, 500e-9, 2.2e-6)``.
    """
    return r0 * (wavelength_out / wavelength_in) ** (6.0 / 5.0)


def structure_function(phase, pixel_scale=1.0, max_separation=None):
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

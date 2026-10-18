"""Fresnel propagation of starlight through the layers: pupil fields and scintillation.

pyturb's layers are thin phase screens at their line-of-sight ranges. A unit
plane wave from the star enters above the highest layer. At each layer the
field picks up the layer's phase, ``exp(+i 2 pi OPD / lambda)``, and then
propagates down to the next layer (or the telescope) with the paraxial
angular-spectrum transfer function ``H(f) = exp(-i pi lambda z f^2)`` (``z > 0``
downstream, ``f`` in cycles per metre): :class:`aocore.AngularSpectrumPropagator`
with ``paraxial=True``. The phase advances by ``k OPD`` with ``k = 2 pi /
lambda`` and the transfer function belongs to the same ``exp(+i k z)``
convention, so a positive OPD bump (a converging lens) focuses light
downstream, and the Fraunhofer image of an OPD ramp ``a x`` sits at ``+a``
(aocore ``CONVENTIONS.md`` rule 3.1). A layer at the telescope (altitude 0)
adds phase only, so it causes no scintillation.

In the weak-perturbation limit a layer at range ``h`` turns OPD into relative
intensity with the transfer function ``2 k sin(pi lambda h f^2)``
(:func:`pyturb.theory.scintillation_transfer`); integrated over a Kolmogorov
profile this is Roddier's (1981) scintillation index
(:func:`pyturb.theory.scintillation_index`). The propagation here is the
full split-step solution, valid beyond weak perturbation as long as the grid
resolves the field.

:meth:`pyturb.Atmosphere.pupil_field` applies this to an atmosphere's layers at
one time and direction; :func:`propagate_layers` applies it to any stack of
layer OPD maps.
"""

from __future__ import annotations

from typing import Any, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from aocore import AngularSpectrumPropagator
from aocore.backend import Backend, backend_of

__all__ = ["PupilField", "propagate_layers"]


class PupilField(NamedTuple):
    """A star's light at the telescope pupil (:meth:`pyturb.Atmosphere.pupil_field`).

    Attributes
    ----------
    field : ndarray
        Complex pupil field relative to the unperturbed plane wave from the
        star's direction (unit amplitude without turbulence), ``(n, n)`` for
        one wavelength or ``(W, n, n)`` for ``W`` wavelengths.
    intensity : ndarray
        ``(n, n)`` relative intensity ``sum_w weight_w |field_w|^2`` (weights
        normalised to sum to 1): its mean is close to 1 and its variance is
        the scintillation index.
    opd : ndarray
        ``(n, n)`` summed OPD of the layers [m], the same map as
        :meth:`pyturb.Atmosphere.opd` at that time and direction (to rounding).
    """

    field: Any
    intensity: Any
    opd: Any


def _wavelengths(wavelength: Union[float, Sequence[float]]) -> Tuple[np.ndarray, bool]:
    """Validated wavelengths [m] as a 1-D array, and whether a scalar was given."""
    waves = np.asarray(wavelength, dtype=np.float64)
    if waves.ndim > 1 or waves.size == 0:
        raise ValueError("wavelength must be a scalar or a non-empty 1-D sequence [m]")
    if not np.all(np.isfinite(waves)) or np.any(waves <= 0):
        raise ValueError(
            f"wavelengths must be positive and finite [m]; got {wavelength!r}")
    return np.atleast_1d(waves), waves.ndim == 0


def _weights(weights: Optional[Sequence[float]], count: int) -> np.ndarray:
    """Per-wavelength weights normalised to sum to 1 (equal weights by default)."""
    if weights is None:
        return np.full(count, 1.0 / count)
    w = np.atleast_1d(np.asarray(weights, dtype=np.float64))
    if w.shape != (count,):
        raise ValueError(
            f"weights must have one value per wavelength ({count}); got shape {w.shape}"
        )
    if not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError(
            "weights must be finite and non-negative with a positive sum (they are "
            "normalised to sum to 1, e.g. the photon spectrum over the band)"
        )
    return w / w.sum()


class _FresnelChain:
    """Layer-by-layer propagation for one geometry: grid, ranges and wavelengths.

    Holds one :class:`aocore.AngularSpectrumPropagator` per wavelength and
    non-zero gap between consecutive layers (highest first, then down to the
    pupil), so repeated frames reuse the transfer functions.
    """

    def __init__(self, shape: Tuple[int, int], pixel_scale: float,
                 altitudes: np.ndarray, wavelengths: np.ndarray,
                 backend: Backend) -> None:
        altitudes = np.asarray(altitudes, dtype=np.float64)
        # Highest layer first; layers at one range keep their given order.
        self.order = np.argsort(-altitudes, kind="stable")
        ranges = altitudes[self.order]
        self.gaps = ranges - np.append(ranges[1:], 0.0)
        self.backend = backend
        self.props: List[List[Optional[AngularSpectrumPropagator]]] = [
            [AngularSpectrumPropagator(shape, pixel_scale, float(lam), float(gap),
                                       paraxial=True, backend=backend)
             if gap > 0 else None for gap in self.gaps]
            for lam in wavelengths
        ]

    def run(self, screens: Any, phase_per_unit: Sequence[float]) -> Any:
        """Pupil fields ``(W, ny, nx)`` for ``screens`` ``(L, ny, nx)``.

        Wavelength ``w`` sees the phase ``phase_per_unit[w] * screens[l]`` [rad]
        at layer ``l``.
        """
        xp = self.backend.xp
        cdtype = self.backend.complex_dtype
        fields = []
        for props, scale in zip(self.props, phase_per_unit):
            field = None
            for layer, prop in zip(self.order, props):
                screen = xp.exp((1j * float(scale)) * screens[layer]).astype(
                    cdtype, copy=False)
                if field is None:
                    field = screen  # the unit plane wave times the top layer
                else:
                    field *= screen
                if prop is not None:
                    field = prop.forward(field)
            fields.append(field)
        return xp.stack(fields)


def propagate_layers(
    opd: Any,
    altitudes: Sequence[float],
    pixel_scale: float,
    wavelength: Union[float, Sequence[float]],
) -> Any:
    """Propagate a unit plane wave through layer OPD maps down to the pupil.

    The field enters above the highest layer, takes each layer's phase
    ``exp(+i 2 pi opd[l] / lambda)`` and propagates to the next layer below (and
    from the lowest layer to the pupil) with the paraxial angular-spectrum
    transfer function ``exp(-i pi lambda z f^2)``
    (:class:`aocore.AngularSpectrumPropagator`, ``paraxial=True``). The grid is
    periodic: each map is one period of a periodic screen, as the FFT screens
    of ``engine="spectral"`` are. Energy is conserved: the mean of
    ``|field|^2`` over the grid is 1 to rounding.

    Parameters
    ----------
    opd : ndarray
        ``(L, ny, nx)`` OPD [m] of each layer (NumPy or CuPy). float32 input
        propagates in single precision, anything else in double; the result
        stays on the input's device.
    altitudes : sequence of float
        Range of each layer from the pupil [m] along the line of sight
        (altitude times ``sec(zenith angle)``), ``>= 0``. Layers at one range
        are applied in the given order (their phases commute).
    pixel_scale : float
        Grid pitch [m].
    wavelength : float or sequence of float
        Wavelength(s) [m]. A scalar returns ``(ny, nx)``, a sequence
        ``(W, ny, nx)``.

    Returns
    -------
    ndarray
        Complex pupil field(s), relative to the unperturbed plane wave.

    Notes
    -----
    The CPU transforms run through aocore's backend, whose thread count is
    ``AOCORE_FFT_WORKERS`` (not :func:`pyturb.set_threads`).

    Examples
    --------
    >>> import numpy as np, pyturb
    >>> opd = 1e-7 * np.random.default_rng(0).standard_normal((2, 64, 64))
    >>> field = pyturb.scintillation.propagate_layers(opd, [8e3, 0.0], 0.02, 1.6e-6)
    >>> field.shape
    (64, 64)
    """
    if getattr(opd, "ndim", None) != 3:
        raise ValueError(
            "opd must be an (L, ny, nx) array of layer OPD maps [m]; got shape "
            f"{getattr(opd, 'shape', None)}"
        )
    if getattr(opd, "dtype", np.dtype(np.float64)).kind not in "fiu":
        raise ValueError("opd must be real (OPD in metres), not complex")
    altitudes = np.asarray(altitudes, dtype=np.float64)
    if altitudes.shape != (opd.shape[0],):
        raise ValueError(
            f"altitudes must give one range per layer ({opd.shape[0]}); got shape "
            f"{altitudes.shape}"
        )
    if not np.all(np.isfinite(altitudes)) or np.any(altitudes < 0):
        raise ValueError("altitudes must be finite and >= 0 [m] (0 is the pupil)")
    if not np.isfinite(pixel_scale) or pixel_scale <= 0:
        raise ValueError("pixel_scale must be positive and finite [m]")
    waves, scalar = _wavelengths(wavelength)
    backend = backend_of(opd)
    screens = backend.asarray(opd, dtype="real")
    chain = _FresnelChain(screens.shape[1:], float(pixel_scale), altitudes, waves,
                          backend)
    fields = chain.run(screens, 2.0 * np.pi / waves)
    return fields[0] if scalar else fields

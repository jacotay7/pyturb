"""Adapters that plug a :class:`pyturb.Atmosphere` into other optics packages.

:class:`HCIPyLayer` behaves like an HCIPy atmospheric layer (``layer.t``,
``layer(wavefront)``, ``evolve_until``, ``phase_for``, ``reset``), so an HCIPy
simulation — including pyRTC's HCIPy interface — can swap HCIPy's single
CPU ``InfiniteAtmosphericLayer`` for a multi-layer, optionally GPU, pyturb
atmosphere without restructuring its optics. HCIPy itself is not imported
here; the adapter works on any object with HCIPy's ``Wavefront`` interface.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .atmosphere import Atmosphere
from .backend import to_numpy

__all__ = ["HCIPyLayer"]


class HCIPyLayer:
    """A :class:`pyturb.Atmosphere` presented as an HCIPy atmospheric layer.

    Parameters
    ----------
    atmosphere : Atmosphere
        The turbulence to apply. Its pupil sampling must match the HCIPy
        grid: ``atmosphere.n`` points across with spacing
        ``atmosphere.pixel_scale`` (e.g. ``hcipy.make_pupil_grid(atm.n,
        atm.diameter)``).
    pupil_grid : hcipy.CartesianGrid, optional
        If given, checked against the atmosphere's sampling at construction.

    Notes
    -----
    Field ordering: an HCIPy pupil field reshaped to 2-D is indexed
    ``[y, x]``, and pyturb's OPD ``[axis 0, axis 1]`` is flattened onto it
    row-major, so pyturb axis 0 is HCIPy's **y** and axis 1 its **x**. pyturb's
    ``Layer.wind_vector`` points where the wind comes from (the pattern moves
    along ``-wind_vector``); HCIPy layer velocities are the pattern's motion.

    The OPD at the current time is evaluated once and reused for every
    wavelength (WFS and science), converted with the atmosphere's
    ``dispersion`` model. ``engine="extrude"`` atmospheres only move forward
    in time; call :meth:`reset` to start over.

    Examples
    --------
    >>> import hcipy, pyturb                                    # doctest: +SKIP
    >>> atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8,
    ...                                      n=128, diameter=8.0, engine="extrude")
    >>> grid = hcipy.make_pupil_grid(128, 8.0)
    >>> layer = pyturb.interop.HCIPyLayer(atm, grid)
    >>> layer.t += 1e-3
    >>> wf = layer(hcipy.Wavefront(hcipy.make_circular_aperture(8.0)(grid), 1.6e-6))
    """

    def __init__(self, atmosphere: Atmosphere, pupil_grid: Optional[Any] = None) -> None:
        self.atmosphere = atmosphere
        self._t = 0.0
        self._cache: Optional[Tuple[float, np.ndarray]] = None
        if pupil_grid is not None:
            self._check_grid(pupil_grid)

    def _check_grid(self, grid: Any) -> None:
        atm = self.atmosphere
        shape = tuple(int(s) for s in getattr(grid, "shape", ()))
        if shape != (atm.n, atm.n):
            raise ValueError(
                f"pupil grid has shape {shape}, but the atmosphere samples "
                f"{atm.n} x {atm.n} points; build the grid with "
                f"hcipy.make_pupil_grid({atm.n}, {atm.diameter})"
            )
        delta = np.asarray(getattr(grid, "delta", [atm.pixel_scale] * 2), dtype=float)
        if not np.allclose(delta, atm.pixel_scale, rtol=1e-6):
            raise ValueError(
                f"pupil grid spacing {tuple(delta)} m differs from the atmosphere's "
                f"pixel_scale {atm.pixel_scale} m (diameter / n)"
            )

    # -- HCIPy layer protocol -------------------------------------------
    @property
    def t(self) -> float:
        """Simulation time [s]; setting it moves the turbulence to that time."""
        return self._t

    @t.setter
    def t(self, value: float) -> None:
        self.evolve_until(value)

    def evolve_until(self, t: float) -> None:
        """Move to time ``t`` [s] (frozen flow; never backwards on the extruder)."""
        self._t = float(t)

    def reset(self, make_independent_realization: bool = False) -> None:
        """Return to ``t = 0`` with the same turbulence.

        ``make_independent_realization`` is accepted for HCIPy compatibility;
        for a new realisation build a new ``Atmosphere`` with another seed.
        """
        if make_independent_realization:
            raise NotImplementedError(
                "build a new pyturb.Atmosphere with a different seed for an "
                "independent realisation"
            )
        self.atmosphere.reset()
        self._t = 0.0
        self._cache = None

    def opd(self) -> np.ndarray:
        """OPD [m] at the current time, flattened in HCIPy field order."""
        if self._cache is None or self._cache[0] != self._t:
            opd = to_numpy(self.atmosphere.opd(self._t)).astype(np.float64)
            self._cache = (self._t, opd.ravel())
        return self._cache[1]

    def phase_for(self, wavelength: float) -> np.ndarray:
        """Phase [rad] at ``wavelength`` [m], flattened in HCIPy field order."""
        atm = self.atmosphere
        scale = atm._chromatic_scale(float(wavelength))
        return self.opd() * (scale * 2.0 * np.pi / float(wavelength))

    def forward(self, wavefront: Any) -> Any:
        """Return a copy of ``wavefront`` with the turbulent phase applied."""
        out = wavefront.copy()
        out.electric_field *= np.exp(1j * self.phase_for(wavefront.wavelength))
        return out

    def backward(self, wavefront: Any) -> Any:
        """Return a copy of ``wavefront`` with the turbulent phase removed."""
        out = wavefront.copy()
        out.electric_field *= np.exp(-1j * self.phase_for(wavefront.wavelength))
        return out

    __call__ = forward

"""Frozen-flow (Taylor) evolution of a single turbulent layer, spectral method.

A :class:`FourierFlowScreen` draws one fixed realisation of a von Kármán phase
screen's Fourier coefficients and then evaluates the screen at any continuous
translation ``(sx, sy)`` in metres by applying the shift theorem — multiplying
each spatial-frequency mode by ``exp(2 pi i (fx sx + fy sy))`` before the
inverse FFT. This gives **exact sub-pixel translation in an arbitrary
direction** at the cost of one FFT per frame, which is extremely fast on the
GPU and requires no row extrusion or interpolation.

The trade-off versus :class:`pyturb.InfinitePhaseScreen` is periodicity: the
FFT-grid part of the screen repeats with period ``n * pixel_scale``, so the
same turbulence eventually blows back into the pupil. For studies where that
wrap is acceptable (most Monte-Carlo and short closed-loop runs) this is the
fastest engine; for genuinely unbounded, non-repeating screens use the
extrusion engine.

The subharmonic low-frequency modes translate consistently with the main grid
(each is an explicit sinusoid, shifted by the same phasor), so the restored
tip/tilt/low-order power blows with the wind too.
"""

from __future__ import annotations

import numpy as np

from .fourier import PhaseScreen

__all__ = ["FourierFlowScreen"]


class FourierFlowScreen:
    """One frozen (Taylor) layer evaluated at arbitrary continuous offsets.

    Parameters
    ----------
    template : PhaseScreen
        A configured generator supplying the PSD amplitude filters, grid,
        backend and dtype. The layer inherits ``n``, ``pixel_scale``, ``r0``,
        ``L0`` and ``device`` from it.
    seed : int, optional
        Seed for the fixed coefficient realisation. Reuse the same seed to
        reproduce a layer exactly.

    Examples
    --------
    >>> import pyturb
    >>> from pyturb.flow import FourierFlowScreen
    >>> template = pyturb.PhaseScreen(n=256, pixel_scale=0.02, r0=0.15)
    >>> layer = FourierFlowScreen(template, seed=0)
    >>> a = layer.translate(0.0, 0.0)          # (256, 256) radians
    >>> b = layer.translate(1.3, -0.4)         # blown 1.3 m / -0.4 m
    """

    def __init__(self, template: PhaseScreen, seed=None):
        self.template = template
        self.xp = template.xp
        self.n = template.n
        self.pixel_scale = template.pixel_scale
        self.r0 = template.r0
        self.L0 = template.L0
        self.device = template.device
        self.dtype = template.dtype
        self._fft = template._fft
        self._cdtype = template.xp.dtype(template._cdtype)
        self._rng = self.xp.random.default_rng(seed)
        self.reseed()

    def reseed(self, seed=None):
        """Draw a fresh fixed realisation of the layer's Fourier coefficients.

        With no argument, advances the existing random stream; pass ``seed``
        to restart it. Use this to get a statistically independent layer while
        keeping the same PSD/grid configuration.
        """
        if seed is not None:
            self._rng = self.xp.random.default_rng(seed)
        n = self.n
        noise = self._rng.standard_normal((2, n, n), dtype=self.dtype)
        self._spectrum = (
            (noise[0] + 1j * noise[1]) * self.template._amplitude
        ).astype(self._cdtype)
        self._sh_coeffs = []
        for amp_p, _basis in self.template._sh_bases:
            noise = self._rng.standard_normal((2, 3, 3), dtype=self.dtype)
            self._sh_coeffs.append(
                ((noise[0] + 1j * noise[1]) * amp_p).astype(self._cdtype)
            )
        return self

    def translate(self, sx, sy):
        """Return the screen blown by ``(sx, sy)`` metres, shape ``(n, n)``.

        ``sx`` is displacement along axis 0 (rows), ``sy`` along axis 1
        (columns). Values are phase in radians at the layer's reference
        wavelength; the array lives on the layer's device.
        """
        xp, n = self.xp, self.n
        f = self._f
        # Shift theorem on the periodic FFT grid: multiply each mode by its
        # translation phasor, then inverse-FFT. Separable in x and y.
        phasor_x = xp.exp((2j * np.pi * sx) * f).astype(self._cdtype)
        phasor_y = xp.exp((2j * np.pi * sy) * f).astype(self._cdtype)
        spectrum = self._spectrum * phasor_x[:, None] * phasor_y[None, :]
        field = self._fft.ifft2(spectrum, axes=(-2, -1)) * (n * n)
        screen = field.real

        if self._sh_coeffs:
            low = None
            for (_amp, basis), coeff, fp in zip(
                self.template._sh_bases, self._sh_coeffs, self.template._sh_freqs
            ):
                px = xp.exp((2j * np.pi * sx) * fp).astype(self._cdtype)
                py = xp.exp((2j * np.pi * sy) * fp).astype(self._cdtype)
                shifted = coeff * px[:, None] * py[None, :]
                contribution = basis.T @ (shifted @ basis)
                low = contribution if low is None else low + contribution
            low = low.real
            low -= low.mean()
            screen = screen + low

        return xp.ascontiguousarray(screen.astype(self.dtype, copy=False))

    @property
    def _f(self):
        return self.template._f

    def __repr__(self):
        return (
            f"FourierFlowScreen(n={self.n}, pixel_scale={self.pixel_scale}, "
            f"r0={self.r0}, L0={self.L0}, device={self.device!r})"
        )

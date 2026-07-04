"""Infinite frozen-flow phase screens (Assémat, Wilson & Gendron 2006).

The screen is extruded one row at a time: each new row is drawn from the
Gaussian distribution of von Kármán phase conditioned on the rows already at
the leading edge of the screen,

    x_new = A @ z + B @ beta,

where ``z`` holds the phase in the stencil rows, ``beta`` is white noise,
and ``A``, ``B`` follow from the von Kármán phase covariance function. This
produces screens of unbounded length with the correct spatial statistics —
the standard way to simulate wind-blown (Taylor frozen-flow) turbulence in
adaptive-optics loops.

Reference: Assémat, Wilson & Gendron (2006), Optics Express 14, 988.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma, kv

from .backend import get_array_module
from .fourier import PhaseScreen

__all__ = ["InfinitePhaseScreen", "phase_covariance"]


def phase_covariance(r, r0, L0):
    """Von Kármán phase covariance :math:`C_\\phi(r)` in rad^2.

    Parameters
    ----------
    r : array_like
        Separation(s) in metres.
    r0 : float
        Fried parameter in metres.
    L0 : float
        Outer scale in metres (must be finite).

    Notes
    -----
    ``C(0) = 0.0863 (L0/r0)^(5/3)`` is the total von Kármán phase variance.
    """
    if not np.isfinite(L0):
        raise ValueError("phase_covariance requires a finite outer scale L0")
    r = np.asarray(r, dtype=np.float64)
    prefactor = (
        (L0 / r0) ** (5.0 / 3.0)
        * (2.0 ** (-5.0 / 6.0) * gamma(11.0 / 6.0) / np.pi ** (8.0 / 3.0))
        * ((24.0 / 5.0) * gamma(6.0 / 5.0)) ** (5.0 / 6.0)
    )
    x = 2.0 * np.pi * r / L0
    # x^(5/6) K_{5/6}(x) -> 2^(-1/6) Gamma(5/6) as x -> 0.
    core = np.full_like(x, 2.0 ** (-1.0 / 6.0) * gamma(5.0 / 6.0))
    nonzero = x > 0
    core[nonzero] = x[nonzero] ** (5.0 / 6.0) * kv(5.0 / 6.0, x[nonzero])
    return prefactor * core


class InfinitePhaseScreen:
    """Endless frozen-flow phase screen, extruded row by row.

    The current ``(n, n)`` screen is available as :attr:`screen`; each call
    to :meth:`step` shifts it by one row (one ``pixel_scale`` of wind
    travel along axis 0) and synthesises a new correlated row at the edge.
    Translate ``pixel_scale`` into wind speed via your loop rate:
    ``wind_speed = pixel_scale / dt`` per step.

    Parameters
    ----------
    n : int
        Screen size in pixels.
    pixel_scale : float
        Pixel size in metres.
    r0 : float
        Fried parameter in metres (at the wavelength of the returned phase).
    L0 : float, optional
        Outer scale in metres. Must be finite (the conditional-covariance
        construction is von Kármán); default 25 m.
    stencil_rows : int, optional
        Number of edge rows the new row is conditioned on. Default 2
        (per Assémat & Wilson; more rows cost setup time for marginal gain).
    seed, device, dtype
        As for :class:`pyturb.PhaseScreen`.

    Examples
    --------
    >>> import pyturb
    >>> layer = pyturb.InfinitePhaseScreen(n=128, pixel_scale=0.05,
    ...                                    r0=0.15, L0=25, seed=0)
    >>> for _ in range(100):
    ...     phase = layer.step()        # advance wind by one pixel
    """

    def __init__(
        self,
        n,
        pixel_scale,
        r0,
        L0=25.0,
        stencil_rows=2,
        seed=None,
        device="cpu",
        dtype="float32",
    ):
        if not np.isfinite(L0) or L0 <= 0:
            raise ValueError(
                "InfinitePhaseScreen requires a finite positive outer scale L0"
            )
        if not 1 <= stencil_rows < n:
            raise ValueError("stencil_rows must be in [1, n)")

        self.n = int(n)
        self.pixel_scale = float(pixel_scale)
        self.r0 = float(r0)
        self.L0 = float(L0)
        self.stencil_rows = int(stencil_rows)
        self.device = device

        self.xp = get_array_module(device)
        self.dtype = self.xp.dtype(dtype)

        self._build_extrusion_matrices()

        # Seed the initial screen with an FFT screen (subharmonics included
        # so large-scale power is present from the start).
        generator = PhaseScreen(
            n=self.n,
            pixel_scale=self.pixel_scale,
            r0=self.r0,
            L0=self.L0,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        self._rng = generator._rng  # share one stream for reproducibility
        self._screen = generator.generate()

    def _build_extrusion_matrices(self):
        """Compute A (mean) and B (noise-colouring) extrusion matrices.

        With Z the stencil rows and X the new row, the conditional
        distribution of X | Z has mean A @ Z and covariance B @ B.T where
        A = C_xz C_zz^-1 and B B^T = C_xx - A C_xz^T. Done once, in float64
        on the CPU (needs scipy.special), then moved to the target device.
        """
        n, m, dx = self.n, self.stencil_rows, self.pixel_scale

        # Coordinates: stencil rows at y = 0..m-1, new row at y = m,
        # x = 0..n-1 (units of pixels; scaled by dx below). Row-major
        # ordering matches screen[-m:].ravel().
        yz, xz = np.mgrid[0:m, 0:n]
        stencil = np.column_stack((xz.ravel(), yz.ravel())).astype(np.float64)
        new_row = np.column_stack(
            (np.arange(n, dtype=np.float64), np.full(n, float(m)))
        )

        def cov(a, b):
            separation = np.hypot(
                a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1]
            )
            return phase_covariance(separation * dx, self.r0, self.L0)

        c_zz = cov(stencil, stencil)
        c_xz = cov(new_row, stencil)
        c_xx = cov(new_row, new_row)

        # C_zz is symmetric positive-definite but ill-conditioned for fine
        # grids; solve via lstsq for robustness (one-time cost).
        a_matrix = np.linalg.lstsq(c_zz, c_xz.T, rcond=None)[0].T

        residual = c_xx - a_matrix @ c_xz.T
        residual = (residual + residual.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(residual)
        b_matrix = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))

        self._a = self.xp.asarray(a_matrix, dtype=self.dtype)
        self._b = self.xp.asarray(b_matrix, dtype=self.dtype)

    @property
    def screen(self):
        """Current ``(n, n)`` phase screen in radians (device array)."""
        return self._screen

    def step(self, steps=1):
        """Advance the wind by ``steps`` rows and return the screen.

        Each step shifts the screen one row along axis 0 and extrudes a new
        statistically consistent row at ``screen[-1]``.
        """
        if steps < 1:
            raise ValueError("steps must be >= 1")
        xp = self.xp
        screen = self._screen
        for _ in range(steps):
            z = screen[-self.stencil_rows :].ravel()
            beta = self._rng.standard_normal(self.n, dtype=self.dtype)
            new_row = self._a @ z + self._b @ beta
            screen = xp.concatenate((screen[1:], new_row[None, :]))
        self._screen = screen
        return screen

    def __repr__(self):
        return (
            f"InfinitePhaseScreen(n={self.n}, pixel_scale={self.pixel_scale}, "
            f"r0={self.r0}, L0={self.L0}, stencil_rows={self.stencil_rows}, "
            f"device={self.device!r}, dtype={self.dtype.name!r})"
        )

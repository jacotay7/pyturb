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

Unlike the periodic spectral engine (:class:`pyturb.FourierFlowScreen`), this
screen never repeats, so it is the right tool for long closed-loop runs. Two
implementation choices make it practical at loop rate:

- a **ring buffer**: new rows are extruded into pre-allocated storage and the
  window is advanced by an index, so a step costs one small mat-vec instead of
  copying the whole ``(n, n)`` screen (the old ``concatenate`` per step);
- **sub-pixel, continuous evolution**: :meth:`advance` moves the screen by any
  fractional number of pixels and the pupil is interpolated (Catmull-Rom cubic
  by default, or linear) at the exact offset, so wind travel of ``v*dt`` is not
  forced onto an integer grid. At an integer offset the interpolation is exact.

Reference: Assémat, Wilson & Gendron (2006), Optics Express 14, 988.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg
from scipy.special import gamma, kv

from .backend import get_array_module
from .fourier import PhaseScreen

__all__ = ["InfinitePhaseScreen", "phase_covariance"]


def _spd_solve(spd, rhs):
    """Solve ``spd @ x = rhs`` for a symmetric-positive-definite ``spd``.

    Uses a Cholesky factorisation (fast and stable), falling back to a
    least-squares solve if ``spd`` is too ill-conditioned to factor — the
    covariance matrices get poorly conditioned on very fine grids.
    """
    try:
        return linalg.cho_solve(linalg.cho_factor(spd), rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(spd, rhs, rcond=None)[0]


def phase_covariance(r: ArrayLike, r0: float, L0: float) -> np.ndarray:
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

    The current ``(n, n)`` screen is available as :attr:`screen`. Advance the
    wind either by whole pixels with :meth:`step` or by any continuous distance
    (in pixels) with :meth:`advance`; new turbulence is synthesised at the
    leading edge (``screen[-1]``) as needed and older rows are recycled, so
    memory stays bounded no matter how long the run.

    Translate pixels into wind speed via your loop rate: a step of
    ``wind_speed * dt / pixel_scale`` pixels advances the screen by ``v*dt``
    metres. :meth:`advance` accepts the fractional result directly.

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
    interp : {"cubic", "linear"}, optional
        Sub-pixel interpolation kernel used by :meth:`advance`. ``"cubic"``
        (Catmull-Rom, default) preserves high-frequency power better; both are
        exact at integer offsets. Unused by :meth:`step`.
    seed, device, dtype
        As for :class:`pyturb.PhaseScreen`.

    Examples
    --------
    >>> import pyturb
    >>> layer = pyturb.InfinitePhaseScreen(n=128, pixel_scale=0.05,
    ...                                    r0=0.15, L0=25, seed=0)
    >>> for _ in range(100):
    ...     phase = layer.step()        # advance wind by one whole pixel
    >>> phase = layer.advance(0.37)     # ...and by 0.37 of a pixel (sub-pixel)
    """

    def __init__(
        self,
        n: int,
        pixel_scale: float,
        r0: float,
        L0: float = 25.0,
        stencil_rows: int = 2,
        interp: str = "cubic",
        seed: Any = None,
        device: str = "cpu",
        dtype: Union[str, np.dtype] = "float32",
    ):
        if not np.isfinite(L0) or L0 <= 0:
            raise ValueError(
                "InfinitePhaseScreen requires a finite positive outer scale L0"
            )
        if not 1 <= stencil_rows < n:
            raise ValueError("stencil_rows must be in [1, n)")
        if interp not in ("cubic", "linear"):
            raise ValueError("interp must be 'cubic' or 'linear'")

        self.n = int(n)
        self.pixel_scale = float(pixel_scale)
        self.r0 = float(r0)
        self.L0 = float(L0)
        self.stencil_rows = int(stencil_rows)
        self.interp = interp
        self.device = device

        self.xp = get_array_module(device)
        self.dtype = self.xp.dtype(dtype)
        if self.dtype not in (self.xp.dtype("float32"), self.xp.dtype("float64")):
            raise ValueError("dtype must be float32 or float64")

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

        # Ring buffer: pre-allocated storage holding a moving window of rows.
        # ``_buf[i]`` is virtual row ``_base + i``; ``_fill`` rows are valid.
        # The pupil samples virtual rows ``[_travel, _travel + n)`` (cubic needs
        # one row below and two above), so a handful of spare rows suffice.
        self._margin = self.stencil_rows + 6
        self._capacity = self.n + self._margin
        self._buf = self.xp.empty((self._capacity, self.n), dtype=self.dtype)
        self._buf[: self.n] = generator.generate()
        self._base = 0  # virtual index of _buf[0]
        self._fill = self.n  # number of valid rows in _buf
        self._travel = 0.0  # continuous wind offset in pixels (monotonic)

        self._grid = self.xp.arange(self.n)  # output-row indices, reused
        self._advance_to(0.0)

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
        # ordering matches _buf[fill-m:fill].ravel().
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

        # A = C_xz C_zz^{-1}; solve C_zz X = C_xz^T via Cholesky (lstsq fallback).
        a_matrix = _spd_solve(c_zz, c_xz.T).T

        residual = c_xx - a_matrix @ c_xz.T
        residual = (residual + residual.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(residual)
        b_matrix = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))

        self._a = self.xp.asarray(a_matrix, dtype=self.dtype)
        self._b = self.xp.asarray(b_matrix, dtype=self.dtype)

    # ------------------------------------------------------------------
    # ring-buffer extrusion
    # ------------------------------------------------------------------
    def _extrude_one(self):
        """Synthesise one new leading-edge row into the ring buffer."""
        if self._fill == self._capacity:
            self._compact()
        z = self._buf[self._fill - self.stencil_rows : self._fill].ravel()
        beta = self._rng.standard_normal(self.n, dtype=self.dtype)
        self._buf[self._fill] = self._a @ z + self._b @ beta
        self._fill += 1

    def _compact(self):
        """Drop consumed rows below the window to make room, keeping one spare.

        The pupil never looks below ``floor(_travel) - 1``, so everything below
        that is free to recycle -- *once it has actually been extruded*.
        ``self._travel`` is the caller's eventual target, already committed by
        ``_advance_to`` before extrusion catches up (see there for why); for a
        jump larger than the buffer, that target can be far beyond anything
        extruded so far, so the naive bound must be capped at what extrusion
        still needs to keep going (the last ``stencil_rows`` rows) or this
        computes a negative-size copy. Amortised O(1) rows per step.
        """
        target_bound = int(np.floor(self._travel)) - 1
        stencil_bound = self._base + self._fill - self.stencil_rows
        keep_from = min(target_bound, stencil_bound) - self._base
        keep_from = max(1, keep_from)  # always free at least one row
        keep = self._fill - keep_from
        self._buf[:keep] = self._buf[keep_from : self._fill].copy()
        self._base += keep_from
        self._fill = keep

    def _ensure(self, top_virtual_index):
        """Extrude until virtual row ``top_virtual_index`` exists."""
        while self._base + self._fill - 1 < top_virtual_index:
            self._extrude_one()

    def _sample(self, travel):
        """Interpolate the ``(n, n)`` pupil at continuous offset ``travel``."""
        xp = self.xp
        # Output row i samples the screen at virtual position travel + i;
        # local index into the buffer is that minus _base.
        positions = (travel - self._base) + self._grid  # float, shape (n,)
        i0 = xp.floor(positions).astype(xp.int64)
        t = (positions - i0).astype(self.dtype)[:, None]  # (n, 1)

        def rows(offset):
            return self._buf[xp.clip(i0 + offset, 0, self._fill - 1)]

        p0 = rows(0)
        p1 = rows(1)
        if self.interp == "linear":
            screen = (1.0 - t) * p0 + t * p1
        else:  # Catmull-Rom cubic
            pm1 = rows(-1)
            p2 = rows(2)
            t2 = t * t
            t3 = t2 * t
            screen = 0.5 * (
                2.0 * p0
                + (-pm1 + p1) * t
                + (2.0 * pm1 - 5.0 * p0 + 4.0 * p1 - p2) * t2
                + (-pm1 + 3.0 * p0 - 3.0 * p1 + p2) * t3
            )
        return xp.ascontiguousarray(screen.astype(self.dtype, copy=False))

    def _advance_to(self, travel):
        if travel < self._travel:
            raise ValueError("wind travel is monotonic; travel cannot decrease")
        self._travel = float(travel)
        self._ensure(int(np.floor(self._travel)) + self.n + 2)
        self._current = self._sample(self._travel)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @property
    def screen(self) -> Any:
        """Current ``(n, n)`` phase screen in radians (device array)."""
        return self._current

    def step(self, steps: int = 1) -> Any:
        """Advance the wind by ``steps`` whole pixels and return the screen.

        Each step shifts the screen one row along axis 0 and extrudes a new
        statistically consistent row at ``screen[-1]``. For fractional (wind
        ``v*dt``) motion use :meth:`advance`.
        """
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self._advance_to(self._travel + steps)
        return self._current

    def advance(self, pixels: float) -> Any:
        """Advance the wind by ``pixels`` (any non-negative float) and return it.

        The pupil is interpolated at the exact sub-pixel offset (see ``interp``);
        integer offsets are reproduced exactly. Successive calls accumulate, so
        ``advance(0.5)`` twice lands on the same screen as ``step(1)``.
        """
        pixels = float(pixels)
        if pixels < 0:
            raise ValueError("pixels must be >= 0 (wind travel is monotonic)")
        self._advance_to(self._travel + pixels)
        return self._current

    @property
    def travel(self) -> float:
        """Total wind travel so far, in pixels (metres = ``travel * pixel_scale``)."""
        return self._travel

    def __repr__(self) -> str:
        return (
            f"InfinitePhaseScreen(n={self.n}, pixel_scale={self.pixel_scale}, "
            f"r0={self.r0}, L0={self.L0}, stencil_rows={self.stencil_rows}, "
            f"interp={self.interp!r}, device={self.device!r}, "
            f"dtype={self.dtype.name!r})"
        )

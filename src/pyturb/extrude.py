"""Arbitrary-direction, sub-pixel, non-periodic frozen-flow layers.

This is the extrusion engine behind :class:`pyturb.Atmosphere` when
``engine="extrude"``. Each layer extrudes von Kármán turbulence one row at a
time along its **own wind axis** (the 1-D Assémat–Wilson recurrence), keeping
the screen in a ring buffer, and the pupil is read out by sampling a **rotated,
sub-pixel interpolation grid**. That gives, exactly:

- any wind direction (the grid is rotated, the data is not),
- any continuous ``v*dt`` of travel (sub-pixel interpolation), and
- a screen that never repeats (unlike the periodic spectral engine).

The spectral engine (:class:`pyturb.FourierFlowScreen`) remains the faster,
fixed-period default; this engine is the one to reach for on long closed-loop
runs where a wrapping screen would bias the statistics.

Two facts keep the setup cheap across a multi-layer atmosphere:

* the mean matrix ``A = C_xz C_zz^-1`` is **independent of r0** (the ``r0^{-5/3}``
  prefactor cancels), so all layers that share ``L0`` share one ``A``;
* the noise matrix scales as ``B = B_unit * r0^{-5/6}``, one cheap per-layer
  rescale of a single ``B_unit``.
"""

from __future__ import annotations

import numpy as np

from .backend import get_array_module
from .fourier import PhaseScreen
from .infinite import phase_covariance

__all__ = ["ExtrudedAtmosphere"]


def _catmull_rom_weights(t):
    """Catmull-Rom cubic weights for the four taps at fraction ``t`` in [0, 1)."""
    t2 = t * t
    t3 = t2 * t
    return (
        0.5 * (-t + 2.0 * t2 - t3),
        0.5 * (2.0 - 5.0 * t2 + 3.0 * t3),
        0.5 * (t + 4.0 * t2 - 3.0 * t3),
        0.5 * (-t2 + t3),
    )


def build_extrusion(width, stencil_rows, pixel_scale, L0, xp, dtype):
    """Return ``(A, B_unit)`` for extruding one ``width``-pixel row.

    Built at ``r0 = 1``; ``A`` is r0-independent and ``B`` for a layer of Fried
    parameter ``r0`` is ``B_unit * r0**(-5/6)``. Done once in float64 (needs
    scipy.special via :func:`phase_covariance`), then moved to the device.
    """
    m, n = int(stencil_rows), int(width)
    yz, xz = np.mgrid[0:m, 0:n]
    stencil = np.column_stack((xz.ravel(), yz.ravel())).astype(np.float64)
    new_row = np.column_stack((np.arange(n, dtype=np.float64), np.full(n, float(m))))

    def cov(a, b):
        sep = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
        return phase_covariance(sep * pixel_scale, 1.0, L0)

    c_zz = cov(stencil, stencil)
    c_xz = cov(new_row, stencil)
    c_xx = cov(new_row, new_row)

    a_matrix = np.linalg.lstsq(c_zz, c_xz.T, rcond=None)[0].T
    residual = c_xx - a_matrix @ c_xz.T
    residual = (residual + residual.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(residual)
    b_unit = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    return xp.asarray(a_matrix, dtype=dtype), xp.asarray(b_unit, dtype=dtype)


class _ExtrudeLayer:
    """One turbulent layer: wind-aligned ring buffer + rotated pupil sampler.

    The screen extrudes along ``+row`` (its wind axis). The pupil's ``n x n``
    pixel grid is expressed in this wind-aligned frame once at construction as
    ``along`` (row) and ``perp`` (column) coordinates in pixels; a readout at
    wind travel ``s`` pixels samples the buffer at ``along + s`` (rows) and
    ``perp`` (columns), interpolated to sub-pixel accuracy.
    """

    def __init__(
        self,
        n,
        width,
        pixel_scale,
        r0,
        L0,
        wind_vx,
        wind_vy,
        altitude_los,
        stencil_rows,
        interp,
        xp,
        dtype,
        rng,
        a_matrix,
        b_unit,
    ):
        self.n = int(n)
        self.W = int(width)
        self.dx = float(pixel_scale)
        self.xp = xp
        self.dtype = xp.dtype(dtype)
        self.m = int(stencil_rows)
        self.interp = interp
        self._rng = rng

        speed = float(np.hypot(wind_vx, wind_vy))
        self.speed = speed
        c = wind_vx / speed if speed > 0 else 1.0
        s = wind_vy / speed if speed > 0 else 0.0
        self._cos, self._sin = c, s
        self.altitude_pix = float(altitude_los) / self.dx

        self._a = a_matrix
        self._b = (b_unit * (float(r0) ** (-5.0 / 6.0))).astype(self.dtype)

        # Pupil pixel coordinates (centred) projected into the wind frame.
        g = np.arange(self.n, dtype=np.float64) - (self.n - 1) / 2.0
        gx = g[:, None]
        gy = g[None, :]
        along = gx * c + gy * s
        perp = -gx * s + gy * c
        self._along = xp.asarray(along, dtype=np.float64)
        self._perp = xp.asarray(perp + self.W / 2.0, dtype=np.float64)
        self._along_min = float(along.min())
        self._along_max = float(along.max())

        # Ring buffer of extruded rows. Virtual row ``base + i`` lives at
        # ``_buf[i]``; the readout window spans a fixed ~1.41 n rows, so a
        # constant capacity plus a small lookahead suffices for any run length.
        span = int(np.ceil(self._along_max)) - int(np.floor(self._along_min)) + 3
        self._capacity = span + self.n + self.m + 32
        self._buf = xp.empty((self._capacity, self.W), dtype=self.dtype)
        self._base = int(np.floor(self._along_min)) - 2
        # Seed with a von Kármán screen (subharmonics included), sliced to the
        # rows the buffer starts with.
        gen = PhaseScreen(
            n=self.W,
            pixel_scale=self.dx,
            r0=float(r0),
            L0=float(L0),
            seed=rng,
            device="gpu" if xp is not np else "cpu",
            dtype=self.dtype.name,
        )
        init = gen.generate()  # (W, W)
        fill0 = span
        self._buf[:fill0] = init[:fill0]
        self._fill = fill0
        self._travel = 0.0

    # -- ring-buffer extrusion ----------------------------------------
    def _extrude_one(self):
        if self._fill == self._capacity:
            self._compact()
        z = self._buf[self._fill - self.m : self._fill].ravel()
        beta = self._rng.standard_normal(self.W, dtype=self.dtype)
        self._buf[self._fill] = self._a @ z + self._b @ beta
        self._fill += 1

    def _compact(self):
        lowest = int(np.floor(self._along_min + self._travel)) - 1
        keep_from = max(1, lowest - self._base)
        keep = self._fill - keep_from
        self._buf[:keep] = self._buf[keep_from : self._fill].copy()
        self._base += keep_from
        self._fill = keep

    def _ensure(self):
        top = int(np.ceil(self._along_max + self._travel)) + 2
        while self._base + self._fill - 1 < top:
            self._extrude_one()

    # -- readout ------------------------------------------------------
    def set_travel(self, travel):
        if travel < self._travel:
            raise ValueError("wind travel is monotonic; travel cannot decrease")
        self._travel = float(travel)
        self._ensure()

    def offsets_for_direction(self, thx, thy):
        """(along, perp) pixel shift of the footprint for an off-axis angle.

        ``thx``/``thy`` are direction tangents (``tan(theta)``); the footprint
        moves by ``altitude * tan(theta)`` in the world frame, projected onto
        the layer's wind axes.
        """
        dx_pix = self.altitude_pix * thx
        dy_pix = self.altitude_pix * thy
        along = dx_pix * self._cos + dy_pix * self._sin
        perp = -dx_pix * self._sin + dy_pix * self._cos
        return along, perp

    def sample(self, off_along=0.0, off_perp=0.0):
        """Interpolate the ``(n, n)`` pupil at the current travel + offset."""
        xp = self.xp
        buf, fill = self._buf, self._fill
        row = self._along + (self._travel + off_along) - self._base
        col = self._perp + off_perp
        r0 = xp.floor(row).astype(xp.int64)
        c0 = xp.floor(col).astype(xp.int64)
        fr = row - r0
        fc = col - c0

        if self.interp == "linear":
            taps_r = ((0, 1.0 - fr), (1, fr))
            taps_c = ((0, 1.0 - fc), (1, fc))
        else:
            wr = _catmull_rom_weights(fr)
            wc = _catmull_rom_weights(fc)
            taps_r = tuple(zip((-1, 0, 1, 2), wr))
            taps_c = tuple(zip((-1, 0, 1, 2), wc))

        out = None
        for dr, weight_r in taps_r:
            rr = xp.clip(r0 + dr, 0, fill - 1)
            row_term = None
            for dc, weight_c in taps_c:
                cc = xp.clip(c0 + dc, 0, self.W - 1)
                term = weight_c * buf[rr, cc]
                row_term = term if row_term is None else row_term + term
            contrib = weight_r * row_term
            out = contrib if out is None else out + contrib
        return out.astype(self.dtype, copy=False)


class ExtrudedAtmosphere:
    """Multi-layer extrusion engine used by :class:`pyturb.Atmosphere`.

    Holds one :class:`_ExtrudeLayer` per turbulent layer (sharing the ``A``
    matrix across layers of equal ``L0``) and sums their pupil samples into a
    single reference-wavelength phase screen. The public :class:`Atmosphere`
    wraps this and converts to OPD.
    """

    def __init__(
        self,
        n,
        pixel_scale,
        layer_r0,
        layer_L0,
        layer_wind,
        layer_altitude_los,
        field_of_view_pix=0.0,
        stencil_rows=2,
        interp="cubic",
        device="cpu",
        dtype="float32",
        seeds=None,
    ):
        self.n = int(n)
        self.dx = float(pixel_scale)
        self.xp = get_array_module(device)
        xp = self.xp
        self.device = device
        self.dtype = xp.dtype(dtype)

        # One uniform buffer width covers the rotated pupil (diagonal ~n*sqrt2)
        # plus the off-axis footprint travel, for every layer/direction.
        width = int(np.ceil(self.n * np.sqrt(2.0) + 2.0 * field_of_view_pix)) + 4
        self.width = width

        # Share A across layers with identical L0; B rescales per layer.
        self._ab_cache = {}
        self.layers = []
        seeds = list(seeds) if seeds is not None else [None] * len(layer_r0)
        for r0, L0, (vx, vy), alt, seed in zip(
            layer_r0, layer_L0, layer_wind, layer_altitude_los, seeds
        ):
            key = round(float(L0), 9)
            if key not in self._ab_cache:
                self._ab_cache[key] = build_extrusion(
                    width, stencil_rows, self.dx, L0, xp, self.dtype
                )
            a_matrix, b_unit = self._ab_cache[key]
            rng = xp.random.default_rng(seed)
            self.layers.append(
                _ExtrudeLayer(
                    n=self.n,
                    width=width,
                    pixel_scale=self.dx,
                    r0=r0,
                    L0=L0,
                    wind_vx=vx,
                    wind_vy=vy,
                    altitude_los=alt,
                    stencil_rows=stencil_rows,
                    interp=interp,
                    xp=xp,
                    dtype=self.dtype,
                    rng=rng,
                    a_matrix=a_matrix,
                    b_unit=b_unit,
                )
            )

    def set_time(self, t):
        """Advance every layer to simulation time ``t`` seconds."""
        for layer in self.layers:
            layer.set_travel(layer.speed * t / self.dx)

    def integrate(self, thx=0.0, thy=0.0):
        """Summed reference-wavelength phase ``(n, n)`` toward one direction."""
        total = None
        for layer in self.layers:
            oa, op = layer.offsets_for_direction(thx, thy)
            contrib = layer.sample(oa, op)
            total = contrib if total is None else total + contrib
        return self.xp.ascontiguousarray(total)

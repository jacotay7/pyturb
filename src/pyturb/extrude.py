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

Caveat: the sub-pixel readout is a position-dependent low-pass filter,
exact at integer pixel travel and most aggressive at half-pixel travel (a
Catmull-Rom kernel has zero gain at the Nyquist frequency for a half-pixel
shift). This puts a 5-15% deviation into the finest-scale (1-2 px) structure
function that oscillates with the sub-pixel phase of the wind travel — a
spurious line at ``wind_speed / pixel_scale`` Hz and harmonics. It does not
affect statistics at separations of a few pixels and up. The spectral engine
has no such artifact (exact at any offset) but is periodic; prefer it when
fine-scale (near-Nyquist) fidelity matters more than non-periodicity.

Two facts keep the setup cheap across a multi-layer atmosphere:

* the mean matrix ``A = C_xz C_zz^-1`` is **independent of r0** (the ``r0^{-5/3}``
  prefactor cancels), so all layers that share ``L0`` share one ``A``;
* the noise matrix scales as ``B = B_unit * r0^{-5/6}``, one cheap per-layer
  rescale of a single ``B_unit``.

Every layer's ring buffer lives as a slab of one contiguous ``(L, cap, W)``
array, so the per-frame pupil readout — the hot path — is a **single fused
gather** over all layers at once instead of one bicubic gather per layer. On
the GPU this removes the per-layer kernel-launch latency that otherwise
dominates a multi-layer frame.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from .backend import get_array_module
from .fourier import PhaseScreen
from .infinite import _spd_solve, phase_covariance

__all__ = ["ExtrudedAtmosphere"]


def _catmull_rom_weights(t: Any) -> Tuple[Any, Any, Any, Any]:
    """Catmull-Rom cubic weights for the four taps at fraction ``t`` in [0, 1)."""
    t2 = t * t
    t3 = t2 * t
    return (
        0.5 * (-t + 2.0 * t2 - t3),
        0.5 * (2.0 - 5.0 * t2 + 3.0 * t3),
        0.5 * (t + 4.0 * t2 - 3.0 * t3),
        0.5 * (-t2 + t3),
    )


def build_extrusion(
    width: int,
    stencil_rows: int,
    pixel_scale: float,
    L0: float,
    xp: Any,
    dtype: Any,
) -> Tuple[Any, Any]:
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

    a_matrix = _spd_solve(c_zz, c_xz.T).T
    residual = c_xx - a_matrix @ c_xz.T
    residual = (residual + residual.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(residual)
    b_unit = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    return xp.asarray(a_matrix, dtype=dtype), xp.asarray(b_unit, dtype=dtype)


class _ExtrudeLayer:
    """One turbulent layer: wind-aligned ring buffer + rotated pupil geometry.

    The screen extrudes along ``+row`` (its wind axis) into a slab of a shared
    ``(L, capacity, W)`` buffer owned by :class:`ExtrudedAtmosphere`. The pupil's
    ``n x n`` pixel grid is expressed in this wind-aligned frame once at
    construction as ``along`` (row) and ``perp`` (column) coordinates in pixels;
    a readout at wind travel ``s`` pixels samples the buffer at ``along + s``
    (rows) and ``perp`` (columns), interpolated to sub-pixel accuracy. The
    interpolation itself is batched across layers by the parent, so this class
    only holds the geometry and drives the (amortised) row extrusion.
    """

    def __init__(
        self,
        n: int,
        width: int,
        pixel_scale: float,
        r0: float,
        L0: float,
        wind_vx: float,
        wind_vy: float,
        altitude_los: float,
        stencil_rows: int,
        interp: str,
        xp: Any,
        dtype: Any,
        rng: Any,
        a_matrix: Any,
        b_unit: Any,
        buf: Any,
        magnification: float = 1.0,
        fov_margin_pix: float = 0.0,
    ) -> None:
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

        # Pupil pixel coordinates (centred) projected into the wind frame. The
        # LGS cone effect shrinks the footprint by ``magnification =
        # 1 - h/H_LGS`` for a guide star at finite altitude.
        mag = float(magnification)
        g = (np.arange(self.n, dtype=np.float64) - (self.n - 1) / 2.0) * mag
        gx = g[:, None]
        gy = g[None, :]
        along = gx * c + gy * s
        perp = -gx * s + gy * c
        self._along = xp.asarray(along, dtype=np.float64)
        self._perp = xp.asarray(perp + self.W / 2.0, dtype=np.float64)
        self._along_min = float(along.min())
        self._along_max = float(along.max())
        # Off-axis directions shift the along-wind footprint by up to
        # +/- fov_margin_pix (see offsets_for_direction / field_of_view);
        # the ring buffer must be extruded and sized to cover that reach too,
        # not just the on-axis pupil, or sample() silently clamps to the last
        # extruded row for any off-axis request.
        self._fov_margin = float(fov_margin_pix)

        # Ring buffer slab (a view into the parent's contiguous (L, cap, W)
        # array). Virtual row ``base + i`` lives at ``_buf[i]``; the readout
        # window spans a fixed ~1.41 n rows plus the off-axis margin, so a
        # constant capacity plus a small lookahead suffices for any run length.
        self._buf = buf
        self._capacity = int(buf.shape[0])
        extent_min = self._along_min - self._fov_margin
        span = (
            int(np.ceil(self._along_max + self._fov_margin))
            - int(np.floor(extent_min))
            + 3
        )
        self._base = int(np.floor(extent_min)) - 2
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
    def _extrude_one(self) -> None:
        if self._fill == self._capacity:
            self._compact()
        z = self._buf[self._fill - self.m : self._fill].ravel()
        beta = self._rng.standard_normal(self.W, dtype=self.dtype)
        self._buf[self._fill] = self._a @ z + self._b @ beta
        self._fill += 1

    def _compact(self) -> None:
        # ``self._travel`` is the eventual target set by set_travel() before
        # extrusion catches up; for a jump bigger than the buffer it can be
        # far beyond what has actually been extruded, so the bound derived
        # from it must be capped at what extrusion still needs to keep going
        # (the last ``m`` stencil rows), or this computes a negative-size copy.
        target_bound = int(
            np.floor(self._along_min - self._fov_margin + self._travel)
        ) - 1
        stencil_bound = self._base + self._fill - self.m
        keep_from = min(target_bound, stencil_bound) - self._base
        keep_from = max(1, keep_from)
        keep = self._fill - keep_from
        self._buf[:keep] = self._buf[keep_from : self._fill].copy()
        self._base += keep_from
        self._fill = keep

    def _ensure(self) -> None:
        top = int(np.ceil(self._along_max + self._fov_margin + self._travel)) + 2
        while self._base + self._fill - 1 < top:
            self._extrude_one()

    # -- readout geometry ---------------------------------------------
    def set_travel(self, travel: float) -> None:
        if travel < self._travel:
            raise ValueError("wind travel is monotonic; travel cannot decrease")
        self._travel = float(travel)
        self._ensure()

    def offsets_for_direction(self, thx: float, thy: float) -> Tuple[float, float]:
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


class ExtrudedAtmosphere:
    """Multi-layer extrusion engine used by :class:`pyturb.Atmosphere`.

    Holds one :class:`_ExtrudeLayer` per turbulent layer (sharing the ``A``
    matrix across layers of equal ``L0``) whose ring buffers are slabs of a
    single contiguous ``(L, capacity, W)`` array. Extrusion is driven per layer
    (amortised, one small mat-vec per new row) but the per-frame pupil readout
    is a single fused bicubic gather over all layers, summed into one
    reference-wavelength phase screen. The public :class:`Atmosphere` wraps this
    and converts to OPD.
    """

    def __init__(
        self,
        n: int,
        pixel_scale: float,
        layer_r0: Sequence[float],
        layer_L0: Sequence[float],
        layer_wind: Sequence[Tuple[float, float]],
        layer_altitude_los: Sequence[float],
        field_of_view_pix: float = 0.0,
        stencil_rows: int = 2,
        interp: str = "cubic",
        lgs_altitude_los: Optional[float] = None,
        device: str = "cpu",
        dtype: Union[str, np.dtype] = "float32",
        seeds: Optional[Sequence[Any]] = None,
    ) -> None:
        self.n = int(n)
        self.dx = float(pixel_scale)
        self.xp = get_array_module(device)
        xp = self.xp
        self.device = device
        self.dtype = xp.dtype(dtype)
        self.interp = interp
        self.m = int(stencil_rows)

        # One uniform buffer width covers the rotated pupil (diagonal ~n*sqrt2)
        # plus the off-axis footprint travel, for every layer/direction.
        width = int(np.ceil(self.n * np.sqrt(2.0) + 2.0 * field_of_view_pix)) + 4
        self.width = width

        # One uniform ring-buffer capacity, sized for the worst case (a 45-deg
        # wind, no LGS cone shrink) so every layer's slab is big enough. This
        # makes the buffer a single contiguous (L, cap, W) array whose slabs
        # can be gathered in one kernel at readout.
        capacity = self._uniform_capacity(self.n, field_of_view_pix, self.m)
        self.capacity = capacity

        # Share A across layers with identical L0; B rescales per layer.
        self._ab_cache: dict = {}
        self.layers: List[_ExtrudeLayer] = []
        n_layers = len(layer_r0)
        self._buf = xp.empty((n_layers, capacity, width), dtype=self.dtype)
        seeds = list(seeds) if seeds is not None else [None] * n_layers
        for i, (r0, L0, (vx, vy), alt, seed) in enumerate(
            zip(layer_r0, layer_L0, layer_wind, layer_altitude_los, seeds)
        ):
            key = round(float(L0), 9)
            if key not in self._ab_cache:
                self._ab_cache[key] = build_extrusion(
                    width, stencil_rows, self.dx, L0, xp, self.dtype
                )
            a_matrix, b_unit = self._ab_cache[key]
            rng = xp.random.default_rng(seed)
            # LGS cone: a layer at altitude h seen from a guide star at range
            # H_LGS has its footprint magnified by (1 - h/H_LGS).
            if lgs_altitude_los is not None:
                magnification = max(0.0, 1.0 - float(alt) / float(lgs_altitude_los))
            else:
                magnification = 1.0
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
                    buf=self._buf[i],
                    magnification=magnification,
                    fov_margin_pix=field_of_view_pix,
                )
            )

        # Stack the (never-changing) rotated pupil grids once, so the readout
        # touches one (L, n, n) array instead of a Python list. Drop the
        # per-layer copies to keep only one on the device.
        self._along = xp.stack([layer._along for layer in self.layers])
        self._perp = xp.stack([layer._perp for layer in self.layers])
        for layer in self.layers:
            layer._along = None
            layer._perp = None
        self._layer_index = xp.arange(n_layers)[:, None, None]

    @staticmethod
    def _uniform_capacity(n: int, fov_margin_pix: float, stencil_rows: int) -> int:
        """Ring-buffer row capacity that fits every wind direction at this ``n``.

        The along-wind extent of the rotated pupil is largest at a 45-deg wind
        (``(n-1)/2 * sqrt2``); add the off-axis reach and the readout lookahead.
        Independent of per-layer direction/LGS magnification (both only shrink
        the extent), so all slabs share one capacity.
        """
        worst_along = (n - 1) / 2.0 * np.sqrt(2.0)
        span = (
            int(np.ceil(worst_along + fov_margin_pix))
            - int(np.floor(-worst_along - fov_margin_pix))
            + 3
        )
        return span + n + int(stencil_rows) + 32

    def set_time(self, t: float) -> None:
        """Advance every layer to simulation time ``t`` seconds."""
        for layer in self.layers:
            layer.set_travel(layer.speed * t / self.dx)

    def integrate(self, thx: float = 0.0, thy: float = 0.0) -> Any:
        """Summed reference-wavelength phase ``(n, n)`` toward one direction.

        On the GPU this is a single fused bicubic (or bilinear) gather over all
        layers' ring buffers at once — one kernel launch instead of one per
        layer — then summed over the layer axis. On the CPU the per-layer loop
        keeps each gather a cache-friendly ``(n, n)`` and wins instead, so the
        backend picks the readout: identical results, best throughput on each.
        The gather indices are the rotated pupil grids offset by each layer's
        wind travel and its (per-layer) off-axis footprint shift.
        """
        if self.xp is np:
            return self._integrate_looped(thx, thy)
        return self._integrate_batched(thx, thy)

    def _readout_shifts(self, thx: float, thy: float):
        """Per-layer (along-shift, perp-shift, fill) readout scalars, host-side.

        The along-wind shift keeps travel and base combined in float64 before
        touching the grid, so precision holds over an unbounded run.
        """
        layers = self.layers
        shift_along = np.empty(len(layers), dtype=np.float64)
        shift_perp = np.empty(len(layers), dtype=np.float64)
        fill = np.empty(len(layers), dtype=np.int64)
        for i, layer in enumerate(layers):
            off_along, off_perp = layer.offsets_for_direction(thx, thy)
            shift_along[i] = (layer._travel - layer._base) + off_along
            shift_perp[i] = off_perp
            fill[i] = layer._fill
        return shift_along, shift_perp, fill

    def _integrate_batched(self, thx: float, thy: float) -> Any:
        """GPU readout: one fused gather over the stacked ``(L, cap, W)`` buffer."""
        xp = self.xp
        shift_along, shift_perp, fill = self._readout_shifts(thx, thy)
        sa = xp.asarray(shift_along)[:, None, None]
        sp = xp.asarray(shift_perp)[:, None, None]
        row = self._along + sa  # (L, n, n) float64
        col = self._perp + sp
        r0 = xp.floor(row).astype(xp.int64)
        c0 = xp.floor(col).astype(xp.int64)
        fr = row - r0
        fc = col - c0

        if self.interp == "linear":
            taps_r = ((0, 1.0 - fr), (1, fr))
            taps_c = ((0, 1.0 - fc), (1, fc))
        else:
            taps_r = tuple(zip((-1, 0, 1, 2), _catmull_rom_weights(fr)))
            taps_c = tuple(zip((-1, 0, 1, 2), _catmull_rom_weights(fc)))

        buf = self._buf
        li = self._layer_index
        fill_max = xp.asarray(fill - 1)[:, None, None]
        wmax = self.width - 1
        out = None
        for dr, weight_r in taps_r:
            rr = xp.clip(r0 + dr, 0, fill_max)
            row_term = None
            for dc, weight_c in taps_c:
                cc = xp.clip(c0 + dc, 0, wmax)
                term = weight_c * buf[li, rr, cc]  # (L, n, n) fused gather
                row_term = term if row_term is None else row_term + term
            contrib = weight_r * row_term
            out = contrib if out is None else out + contrib
        total = out.sum(axis=0)  # sum over layers -> (n, n)
        return xp.ascontiguousarray(total.astype(self.dtype, copy=False))

    def _integrate_looped(self, thx: float, thy: float) -> Any:
        """CPU readout: per-layer ``(n, n)`` gather, summed as we go.

        A big fused ``(L, n, n)`` fancy-index gather is memory-bound and cache-
        hostile on NumPy; looping keeps each gather 2-D and contiguous, which is
        markedly faster on the CPU than the batched path the GPU prefers.
        """
        xp = self.xp
        shift_along, shift_perp, fill = self._readout_shifts(thx, thy)
        wmax = self.width - 1
        total = None
        for i in range(len(self.layers)):
            buf = self._buf[i]
            row = self._along[i] + shift_along[i]
            col = self._perp[i] + shift_perp[i]
            r0 = xp.floor(row).astype(xp.int64)
            c0 = xp.floor(col).astype(xp.int64)
            fr = row - r0
            fc = col - c0
            if self.interp == "linear":
                taps_r = ((0, 1.0 - fr), (1, fr))
                taps_c = ((0, 1.0 - fc), (1, fc))
            else:
                taps_r = tuple(zip((-1, 0, 1, 2), _catmull_rom_weights(fr)))
                taps_c = tuple(zip((-1, 0, 1, 2), _catmull_rom_weights(fc)))
            fmax = int(fill[i]) - 1
            out = None
            for dr, weight_r in taps_r:
                rr = xp.clip(r0 + dr, 0, fmax)
                row_term = None
                for dc, weight_c in taps_c:
                    cc = xp.clip(c0 + dc, 0, wmax)
                    term = weight_c * buf[rr, cc]
                    row_term = term if row_term is None else row_term + term
                contrib = weight_r * row_term
                out = contrib if out is None else out + contrib
            total = out if total is None else total + out
        return xp.ascontiguousarray(total.astype(self.dtype, copy=False))

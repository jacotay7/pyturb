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
affect statistics at separations of a few pixels and up. ``interp="lanczos"``
(6-tap Lanczos-3) has a flatter sub-Nyquist response and roughly halves the
finest-scale deficit and cuts the travel-phase flicker ~3x versus the default
cubic; the exact-Nyquist mode is still lost (no interpolator can shift a
critically sampled signal there). The spectral engine has no such artifact
(exact at any offset) but is periodic; prefer it when fine-scale (near-Nyquist)
fidelity matters more than non-periodicity.

Two facts keep the setup cheap across a multi-layer atmosphere:

* the mean matrix ``A = C_xz C_zz^-1`` is **independent of r0** (the ``r0^{-5/3}``
  prefactor cancels), so all layers that share ``L0`` share one ``A``;
* the noise matrix scales as ``B = B_unit * r0^{-5/6}``, one cheap per-layer
  rescale of a single ``B_unit``.

Every layer's ring buffer lives as a slab of one contiguous ``(L, cap, W)``
array, so the per-frame pupil readout — the hot path — touches all layers at
once. For the ``"cubic"`` and ``"lanczos"`` kernels it is a **single custom
CUDA kernel** on the GPU (and a fused ``prange`` Numba pass on CPU, when the
optional ``pyturb[accel]`` extra is installed) that does the whole rotated,
sub-pixel, per-layer-wind-shifted gather and the layer sum in one pass, with
the interpolation weights held in registers; ``"linear"`` uses a batched
tap-broadcast gather. Either way there is no per-layer kernel-launch latency.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import _accel
from .backend import get_array_module
from .fourier import PhaseScreen
from .infinite import _lanczos_weights, _spd_solve, phase_covariance

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


# Fused GPU readout: one kernel does the whole multi-layer bicubic (Catmull-Rom)
# gather -- rotated, sub-pixel, per-layer wind-shifted -- summed over layers into
# the (n, n) pupil, in a single pass with the tap weights held in registers. This
# replaces materialising 16 (L, n, n) index/gather temporaries per frame; it is
# bit-identical to that path (both accumulate the four-tap products in double).
_READOUT_SRC = r"""
extern "C" __global__ void extrude_readout_{suf}(
    const {T}* __restrict__ buf, const double* __restrict__ along,
    const double* __restrict__ perp, const double* __restrict__ sa,
    const double* __restrict__ sp, const long long* __restrict__ fillm1,
    {T}* __restrict__ out, int L, int n, long long cap, int W)
{{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long npix = (long long)n * n;
    if (idx >= npix) return;
    double acc = 0.0;
    for (int l = 0; l < L; ++l) {{
        long long p2 = (long long)l * npix + idx;
        double row = along[p2] + sa[l];
        double col = perp[p2] + sp[l];
        long long r0 = (long long)floor(row);
        long long c0 = (long long)floor(col);
        double fr = row - r0, fc = col - c0;
        double fr2 = fr * fr, fr3 = fr2 * fr, fc2 = fc * fc, fc3 = fc2 * fc;
        double wr[4], wc[4];
        wr[0] = 0.5 * (-fr + 2.0 * fr2 - fr3);
        wr[1] = 0.5 * (2.0 - 5.0 * fr2 + 3.0 * fr3);
        wr[2] = 0.5 * (fr + 4.0 * fr2 - 3.0 * fr3);
        wr[3] = 0.5 * (-fr2 + fr3);
        wc[0] = 0.5 * (-fc + 2.0 * fc2 - fc3);
        wc[1] = 0.5 * (2.0 - 5.0 * fc2 + 3.0 * fc3);
        wc[2] = 0.5 * (fc + 4.0 * fc2 - 3.0 * fc3);
        wc[3] = 0.5 * (-fc2 + fc3);
        long long fmax = fillm1[l];
        const {T}* bl = buf + (long long)l * cap * W;
        double val = 0.0;
        for (int a = 0; a < 4; ++a) {{
            long long rr = r0 + (a - 1);
            rr = rr < 0 ? 0 : (rr > fmax ? fmax : rr);
            const {T}* brow = bl + rr * W;
            double rs = 0.0;
            for (int b = 0; b < 4; ++b) {{
                long long cc = c0 + (b - 1);
                cc = cc < 0 ? 0 : (cc > W - 1 ? W - 1 : cc);
                rs += wc[b] * (double)brow[cc];
            }}
            val += wr[a] * rs;
        }}
        acc += val;
    }}
    out[idx] = ({T})acc;
}}
"""

# Lanczos-3 variant: the flatter sub-Nyquist kernel used by the highest-fidelity
# readout. 6 taps per axis (36 per layer) with windowed-sinc weights computed in
# registers; otherwise identical structure to the cubic kernel above.
_READOUT_LANCZOS_SRC = r"""
__device__ __forceinline__ double _sincpi(double x) {{
    if (x == 0.0) return 1.0;
    double px = 3.14159265358979323846 * x;
    return sin(px) / px;
}}
__device__ __forceinline__ void _lanczos6(double t, double* w) {{
    double total = 0.0;
    for (int k = 0; k < 6; ++k) {{
        double x = t - (double)(k - 2);
        double s = _sincpi(x) * _sincpi(x * (1.0 / 3.0));
        w[k] = s;
        total += s;
    }}
    double inv = 1.0 / total;
    for (int k = 0; k < 6; ++k) w[k] *= inv;
}}
extern "C" __global__ void extrude_readout_lz_{suf}(
    const {T}* __restrict__ buf, const double* __restrict__ along,
    const double* __restrict__ perp, const double* __restrict__ sa,
    const double* __restrict__ sp, const long long* __restrict__ fillm1,
    {T}* __restrict__ out, int L, int n, long long cap, int W)
{{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long npix = (long long)n * n;
    if (idx >= npix) return;
    double acc = 0.0;
    for (int l = 0; l < L; ++l) {{
        long long p2 = (long long)l * npix + idx;
        double row = along[p2] + sa[l];
        double col = perp[p2] + sp[l];
        long long r0 = (long long)floor(row);
        long long c0 = (long long)floor(col);
        double wr[6], wc[6];
        _lanczos6(row - r0, wr);
        _lanczos6(col - c0, wc);
        long long fmax = fillm1[l];
        const {T}* bl = buf + (long long)l * cap * W;
        double val = 0.0;
        for (int a = 0; a < 6; ++a) {{
            long long rr = r0 + (a - 2);
            rr = rr < 0 ? 0 : (rr > fmax ? fmax : rr);
            const {T}* brow = bl + rr * W;
            double rs = 0.0;
            for (int b = 0; b < 6; ++b) {{
                long long cc = c0 + (b - 2);
                cc = cc < 0 ? 0 : (cc > W - 1 ? W - 1 : cc);
                rs += wc[b] * (double)brow[cc];
            }}
            val += wr[a] * rs;
        }}
        acc += val;
    }}
    out[idx] = ({T})acc;
}}
"""

_readout_kernels: dict = {}


def _get_readout_kernel(dtype: Any, interp: str = "cubic"):
    """Compile (once) and return the fused readout kernel for a dtype/interp."""
    name = np.dtype(dtype).name
    key = (name, interp)
    ker = _readout_kernels.get(key)
    if ker is None:
        import cupy

        ctype, suf = ("float", "f") if name == "float32" else ("double", "d")
        src, fn = (
            (_READOUT_LANCZOS_SRC, f"extrude_readout_lz_{suf}")
            if interp == "lanczos"
            else (_READOUT_SRC, f"extrude_readout_{suf}")
        )
        ker = cupy.RawKernel(src.format(T=ctype, suf=suf), fn)
        _readout_kernels[key] = ker
    return ker


def _along_extent_and_span(
    n: int, wind_vx: float, wind_vy: float, magnification: float, fov_margin_pix: float
) -> Tuple[float, int]:
    """This layer's own along-wind half-extent and the buffer span it needs.

    The along-wind extent of a rotated ``n x n`` pupil grid is
    ``(n-1)/2 * (|cos theta| + |sin theta|)``, ranging from ``(n-1)/2`` for
    axis-aligned wind up to ``(n-1)/2 * sqrt(2)`` at a 45-degree wind — so a
    layer whose wind is closer to axis-aligned, or whose LGS-cone
    ``magnification`` shrinks its footprint, needs less along-wind buffer than
    the worst-case 45-degree, unmagnified layer. Mirrors the geometry
    :class:`_ExtrudeLayer` computes for itself (``along_min``/``along_max``),
    so a caller sizing a shared buffer from this gets exactly what that layer
    will actually use.
    """
    speed = float(np.hypot(wind_vx, wind_vy))
    c = wind_vx / speed if speed > 0 else 1.0
    s = wind_vy / speed if speed > 0 else 0.0
    half = (n - 1) / 2.0 * float(magnification)
    along_extent = half * (abs(c) + abs(s))
    span = (
        int(np.ceil(along_extent + fov_margin_pix))
        - int(np.floor(-along_extent - fov_margin_pix))
        + 3
    )
    return along_extent, span


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
    is a single fused gather over all layers (a custom CUDA / Numba kernel for
    the cubic and Lanczos interpolators), summed into one reference-wavelength
    phase screen. The public :class:`Atmosphere` wraps this and converts to OPD.

    ``capacity``/``width`` are shared by every layer's slab (needed for the
    fused gather), but are sized to the largest requirement **actually
    present** among the layers — each layer's own wind direction (axis-aligned
    wind needs ~29% less along-wind buffer than a 45-degree wind), LGS
    ``magnification``, and (per-layer) ``field_of_view_pix`` all shrink what
    that layer needs, so a run where no layer sits at both the worst altitude
    and the worst direction uses less buffer than the old blanket
    every-layer-at-45-degrees assumption.
    """

    def __init__(
        self,
        n: int,
        pixel_scale: float,
        layer_r0: Sequence[float],
        layer_L0: Sequence[float],
        layer_wind: Sequence[Tuple[float, float]],
        layer_altitude_los: Sequence[float],
        field_of_view_pix: Union[float, Sequence[float]] = 0.0,
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

        layer_wind = list(layer_wind)
        layer_altitude_los = list(layer_altitude_los)
        n_layers = len(layer_r0)
        # A scalar broadcasts to every layer (the pre-per-layer behaviour,
        # still used directly by tests); a sequence gives each layer its own
        # off-axis reach (what Atmosphere passes, scaled by that layer's own
        # altitude).
        if np.ndim(field_of_view_pix) == 0:
            fov_list = [float(field_of_view_pix)] * n_layers
        else:
            fov_list = [float(f) for f in field_of_view_pix]

        # LGS cone: a layer at altitude h seen from a guide star at range
        # H_LGS has its footprint magnified by (1 - h/H_LGS). Computed once,
        # up front, since it feeds both the array-sizing pass below and each
        # layer's construction.
        if lgs_altitude_los is not None:
            magnifications = [
                max(0.0, 1.0 - float(alt) / float(lgs_altitude_los))
                for alt in layer_altitude_los
            ]
        else:
            magnifications = [1.0] * n_layers

        # Size the shared (L, cap, W) buffer to the largest per-layer need
        # actually present (see class docstring), not a blanket worst case.
        along_extents, spans = [], []
        for (vx, vy), mag, fov_i in zip(layer_wind, magnifications, fov_list):
            extent, span = _along_extent_and_span(self.n, vx, vy, mag, fov_i)
            along_extents.append(extent)
            spans.append(span)
        width = int(np.ceil(2.0 * max(along_extents) + 2.0 * max(fov_list))) + 4
        self.width = width
        capacity = max(spans) + self.n + self.m + 32
        self.capacity = capacity

        # Share A across layers with identical L0; B rescales per layer.
        self._ab_cache: dict = {}
        self.layers: List[_ExtrudeLayer] = []
        self._buf = xp.empty((n_layers, capacity, width), dtype=self.dtype)
        seeds = list(seeds) if seeds is not None else [None] * n_layers
        for i, (r0, L0, (vx, vy), alt, seed, mag, fov_i) in enumerate(
            zip(layer_r0, layer_L0, layer_wind, layer_altitude_los, seeds,
                magnifications, fov_list)
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
                    buf=self._buf[i],
                    magnification=mag,
                    fov_margin_pix=fov_i,
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

    def set_time(self, t: float) -> None:
        """Advance every layer to simulation time ``t`` seconds."""
        for layer in self.layers:
            layer.set_travel(layer.speed * t / self.dx)

    def integrate(self, thx: float = 0.0, thy: float = 0.0) -> Any:
        """Summed reference-wavelength phase ``(n, n)`` toward one direction.

        The rotated pupil grids, offset by each layer's wind travel and its
        (per-layer) off-axis footprint shift, index every layer's ring buffer.
        For ``interp="cubic"``/``"lanczos"`` the whole gather and layer sum run
        in one pass: a custom CUDA kernel on the GPU, a fused ``prange`` Numba
        kernel on the CPU (when ``pyturb[accel]`` is installed), or a batched
        tap-broadcast gather as the NumPy fallback. ``interp="linear"`` uses the
        tap-broadcast gather (looped per layer on CPU, batched on GPU). All paths
        give identical results; the backend picks the fastest.
        """
        if self.xp is np:
            if self.interp in ("cubic", "lanczos") and _accel.HAVE_NUMBA:
                return self._integrate_cpu_fused(thx, thy)
            return self._integrate_looped(thx, thy)
        return self._integrate_batched(thx, thy)

    def _integrate_cpu_fused(self, thx: float, thy: float) -> Any:
        """Fused Numba readout on CPU (one parallel pass, no temps).

        Catmull-Rom or Lanczos-3 per :attr:`interp`; both mirror the GPU kernels.
        """
        shift_along, shift_perp, fill = self._readout_shifts(thx, thy)
        out = np.empty((self.n, self.n), dtype=self.dtype)
        kernel = (
            _accel.extrude_lanczos_readout
            if self.interp == "lanczos"
            else _accel.extrude_cubic_readout
        )
        kernel(
            self._buf, self._along, self._perp,
            shift_along, shift_perp, (fill - 1).astype(np.int64), out,
        )
        return out

    def _taps(self, fr: Any, fc: Any, xp: Any):
        """(row taps, col taps) as ``(offset, weight)`` pairs for the interp kernel.

        ``"linear"`` 2-tap, ``"cubic"`` 4-tap Catmull-Rom, or ``"lanczos"`` 6-tap
        Lanczos-3 (flatter sub-Nyquist response, fewer finest-scale artifacts).
        """
        if self.interp == "linear":
            return ((0, 1.0 - fr), (1, fr)), ((0, 1.0 - fc), (1, fc))
        if self.interp == "lanczos":
            off_r, w_r = _lanczos_weights(fr, xp)
            off_c, w_c = _lanczos_weights(fc, xp)
            return tuple(zip(off_r, w_r)), tuple(zip(off_c, w_c))
        return (
            tuple(zip((-1, 0, 1, 2), _catmull_rom_weights(fr))),
            tuple(zip((-1, 0, 1, 2), _catmull_rom_weights(fc))),
        )

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
        """GPU readout: one fused gather over the stacked ``(L, cap, W)`` buffer.

        For the default Catmull-Rom ``interp="cubic"`` this is a single custom
        CUDA kernel doing the whole multi-layer bicubic gather and layer sum in
        one pass; ``"linear"``/``"lanczos"`` fall back to the tap-broadcast
        gather below.
        """
        xp = self.xp
        if self.interp in ("cubic", "lanczos"):
            return self._integrate_fused_kernel(thx, thy)
        shift_along, shift_perp, fill = self._readout_shifts(thx, thy)
        sa = xp.asarray(shift_along)[:, None, None]
        sp = xp.asarray(shift_perp)[:, None, None]
        row = self._along + sa  # (L, n, n) float64
        col = self._perp + sp
        r0 = xp.floor(row).astype(xp.int64)
        c0 = xp.floor(col).astype(xp.int64)
        fr = row - r0
        fc = col - c0

        taps_r, taps_c = self._taps(fr, fc, xp)

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

    def _integrate_fused_kernel(self, thx: float, thy: float) -> Any:
        """Fused single-kernel readout (Catmull-Rom or Lanczos-3, see kernels)."""
        xp = self.xp
        shift_along, shift_perp, fill = self._readout_shifts(thx, thy)
        L, n = len(self.layers), self.n
        out = xp.empty((n, n), dtype=self.dtype)
        ker = _get_readout_kernel(self.dtype, self.interp)
        sa = xp.asarray(shift_along)
        sp = xp.asarray(shift_perp)
        fillm1 = xp.asarray(fill - 1)
        threads = 256
        blocks = (n * n + threads - 1) // threads
        ker(
            (blocks,), (threads,),
            (self._buf, self._along, self._perp, sa, sp, fillm1, out,
             np.int32(L), np.int32(n), np.int64(self.capacity), np.int32(self.width)),
        )
        return out

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
            taps_r, taps_c = self._taps(fr, fc, xp)
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

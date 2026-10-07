"""GPU fast path for the spectral engine: fused kernels and CUDA-graph replay.

A spectral frame is a fixed chain of a few dozen small kernels (phasors, layer
sum, inverse FFT, subharmonics, crop). On the GPU the arithmetic is cheap, so
the frame rate is set by Python and kernel-launch overhead on the host (about
1 ms per frame on a modest CPU). This module removes most of it:

- a **fused kernel** sums the frozen-flow-shifted layer spectra in one pass,
  reading the ``(L, n, n)`` stack once instead of materialising two
  ``(L, n, n)`` complex temporaries;
- every per-frame quantity is computed **on the device from a float64
  displacement buffer** (reduced modulo the screen period, phases formed in
  float64), so the whole frame depends only on device memory;
- that makes the single-frame path capturable as a **CUDA graph**, replayed
  with one launch per frame after copying the new displacements in.

The LGS cone frame (``lgs_altitude``) cannot collapse the layers before the
transform, since each layer zooms differently. It runs as its own captured
chain: a kernel forms each layer's Hermitian half-spectrum, one batched real
inverse FFT gives every layer's screen, one kernel zooms and sums them, and the
subharmonic field is evaluated directly at the zoomed positions as a low-rank
product (the zoom is linear and separable). Boiling's per-mode AR(1) update is
one kernel that draws its noise inline from the same Philox stream.

Capture runs under a private memory pool kept alive with the graph, so the
temporaries the graph writes on replay are never handed to other arrays.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

_SRC = r"""
#include <cupy/complex.cuh>
extern "C" __global__ void spectral_layer_sum_{suf}(
    const complex<{T}>* __restrict__ spec, const complex<{T}>* __restrict__ ph0,
    const complex<{T}>* __restrict__ ph1, complex<{T}>* __restrict__ out,
    int L, int ns, long long total, {T} scale)
{{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long long plane = (long long)ns * ns;
    long long b = idx / plane;
    long long rem = idx - b * plane;
    int i = (int)(rem / ns);
    int j = (int)(rem - (long long)i * ns);
    complex<{T}> acc(0, 0);
    for (int l = 0; l < L; ++l) {{
        long long row = (b * L + l) * ns;
        acc += spec[l * plane + rem] * ph0[row + i] * ph1[row + j];
    }}
    out[idx] = acc * scale;
}}
extern "C" __global__ void subharmonic_field_{suf}(
    const complex<{T}>* __restrict__ shifted, const complex<{T}>* __restrict__ basis,
    {T}* __restrict__ out, int P, int ns, long long total)
{{
    // low[b,i,j] = Re sum_{{p,a,c}} shifted[b,p,a,c] basis[p,a,i] basis[p,c,j]
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long long plane = (long long)ns * ns;
    long long b = idx / plane;
    long long rem = idx - b * plane;
    int i = (int)(rem / ns);
    int j = (int)(rem - (long long)i * ns);
    complex<{T}> acc(0, 0);
    for (int p = 0; p < P; ++p) {{
        const complex<{T}>* bp = basis + (long long)p * 3 * ns;
        const complex<{T}>* sp = shifted + (b * P + p) * 9;
        for (int a = 0; a < 3; ++a) {{
            complex<{T}> row(0, 0);
            for (int c = 0; c < 3; ++c) row += sp[a * 3 + c] * bp[c * ns + j];
            acc += bp[a * ns + i] * row;
        }}
    }}
    out[idx] = acc.real();
}}
extern "C" __global__ void hermitian_layers_{suf}(
    const complex<{T}>* __restrict__ spec, const complex<{T}>* __restrict__ ph0,
    const complex<{T}>* __restrict__ ph1, complex<{T}>* __restrict__ out,
    int ns, int nh, long long total, {T} half_scale)
{{
    // Half-plane (j < nh = ns/2 + 1) Hermitian part of each layer's shifted
    // spectrum X = spec * ph0[i] * ph1[j]: out = (X[i,j] + conj(X[-i,-j])) / 2
    // times the transform scale. Its inverse real FFT is Re(ifft2(X)), the
    // layer's screen, at half the cost of the complex transform.
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long long hplane = (long long)ns * nh;
    long long l = idx / hplane;
    long long rem = idx - l * hplane;
    int i = (int)(rem / nh);
    int j = (int)(rem - (long long)i * nh);
    int i2 = i == 0 ? 0 : ns - i;
    int j2 = j == 0 ? 0 : ns - j;
    long long plane = (long long)ns * ns;
    const complex<{T}>* sl = spec + l * plane;
    const complex<{T}>* p0 = ph0 + l * ns;
    const complex<{T}>* p1 = ph1 + l * ns;
    complex<{T}> x = sl[(long long)i * ns + j] * p0[i] * p1[j];
    complex<{T}> y = sl[(long long)i2 * ns + j2] * p0[i2] * p1[j2];
    out[idx] = (x + conj(y)) * half_scale;
}}
extern "C" __global__ void subharmonic_taps_{suf}(
    const complex<{T}>* __restrict__ shifted, const complex<{T}>* __restrict__ zb,
    complex<{T}>* __restrict__ out, int n, long long total)
{{
    // out[l,p,a,j] = sum_c shifted[l,p,a,c] * zb[l,p,c,j]: the column half of
    // each layer's zoomed subharmonic field (see lgs_zoom).
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long long row = idx / n;            // (l, p, a)
    int j = (int)(idx - row * n);
    long long lp = row / 3;             // (l, p)
    const complex<{T}>* s = shifted + row * 3;
    const complex<{T}>* z = zb + lp * 3 * n + j;
    out[idx] = s[0] * z[0] + s[1] * z[n] + s[2] * z[2 * n];
}}
extern "C" __global__ void lgs_zoom_{suf}(
    const {T}* __restrict__ screens, const long long* __restrict__ idx,
    const {T}* __restrict__ w, {T}* __restrict__ out, int L, int T, int n, int ns)
{{
    // Separable cone zoom of every layer's screen, summed over layers:
    // out[i,j] = sum_l sum_a w[l,a,i] sum_b w[l,b,j] screen_l[idx[l,a,i], idx[l,b,j]].
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= (long long)n * n) return;
    int i = (int)(p / n);
    int j = (int)(p - (long long)i * n);
    long long plane = (long long)ns * ns;
    {T} total = 0;
    for (int l = 0; l < L; ++l) {{
        const {T}* f = screens + l * plane;
        const long long* ix = idx + (long long)l * T * n;
        const {T}* wl = w + (long long)l * T * n;
        {T} acc = 0;
        for (int a = 0; a < T; ++a) {{
            const {T}* row = f + ix[a * n + i] * ns;
            {T} rs = 0;
            for (int b = 0; b < T; ++b) rs += wl[b * n + j] * row[ix[b * n + j]];
            acc += wl[a * n + i] * rs;
        }}
        total += acc;
    }}
    out[p] = total;
}}
extern "C" __global__ void add_low_rank_{suf}(
    const complex<{T}>* __restrict__ zr, const complex<{T}>* __restrict__ zc,
    const {T}* __restrict__ low_mean, {T}* __restrict__ out, int R, int n)
{{
    // out[i,j] += Re sum_r zr[r,i] zc[r,j] - low_mean: the zoomed subharmonic
    // field of every layer (rank R = layers x levels x 3). A shared-memory
    // tiled product: 16 x 16 threads per 64 x 64 output tile, 4 x 4 per thread.
    __shared__ complex<{T}> sr[16][64];
    __shared__ complex<{T}> sc[16][64];
    int ty = threadIdx.y, tx = threadIdx.x;
    int i0 = blockIdx.y * 64, j0 = blockIdx.x * 64;
    {T} acc[4][4];
    for (int u = 0; u < 4; ++u)
        for (int v = 0; v < 4; ++v) acc[u][v] = 0;
    for (int r0 = 0; r0 < R; r0 += 16) {{
        // Thread (ty, tx) loads row r0 + ty, columns tx + 16 v of both tiles.
        int r = r0 + ty;
        for (int v = 0; v < 4; ++v) {{
            int c = tx + 16 * v;
            sr[ty][c] = (r < R && i0 + c < n) ? zr[(long long)r * n + i0 + c]
                                              : complex<{T}>(0, 0);
            sc[ty][c] = (r < R && j0 + c < n) ? zc[(long long)r * n + j0 + c]
                                              : complex<{T}>(0, 0);
        }}
        __syncthreads();
        for (int k = 0; k < 16; ++k) {{
            complex<{T}> a[4], b[4];
            for (int u = 0; u < 4; ++u) a[u] = sr[k][ty + 16 * u];
            for (int v = 0; v < 4; ++v) b[v] = sc[k][tx + 16 * v];
            for (int u = 0; u < 4; ++u)
                for (int v = 0; v < 4; ++v)
                    acc[u][v] += a[u].real() * b[v].real() - a[u].imag() * b[v].imag();
        }}
        __syncthreads();
    }}
    {T} mean = low_mean[0];
    for (int u = 0; u < 4; ++u) {{
        int i = i0 + ty + 16 * u;
        if (i >= n) continue;
        for (int v = 0; v < 4; ++v) {{
            int j = j0 + tx + 16 * v;
            if (j < n) out[(long long)i * n + j] += acc[u][v] - mean;
        }}
    }}
}}
"""

# Boiling's AR(1) update with its noise drawn inline (see boil_blend_rng).
_BOIL_SRC = r"""
#include <cupy/complex.cuh>
extern "C" __global__ void boil_blend_rng_{suf}(
    complex<{T}>* __restrict__ spec, const {T}* __restrict__ amp,
    const {T}* __restrict__ a, const {T}* __restrict__ b, long long total,
    unsigned int key0, unsigned int key1, unsigned long long offset)
{{
    // spec = spec*a + b*((noise[0] + 1j*noise[1])*amp) for a (2, *spec.shape)
    // standard-normal draw. Element e of that draw is z[e % {PB}] of Philox
    // block offset + e / {PB}, exactly what CudaNormalGenerator writes, so it is
    // generated here instead of round-tripping through memory. Every product
    // and sum is rounded separately (no fused multiply-add), as the chain of
    // elementwise array operations rounds it: the result is bit-identical.
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long first = q * {PB};
    if (first >= total) return;
    {T} zr[{PB}], zi[2 * {PB}];
    {NORMAL}(offset + (unsigned long long)q, key0, key1, zr);
    long long e = total + first;  // this thread's first imaginary-part element
    long long blk = e / {PB};
    int lane = (int)(e - blk * {PB});
    {NORMAL}(offset + (unsigned long long)blk, key0, key1, zi);
    if (lane) {NORMAL}(offset + (unsigned long long)blk + 1, key0, key1, zi + {PB});
    for (int m = 0; m < {PB}; ++m) {{
        long long k = first + m;
        if (k >= total) break;
        {T} ak = a[k], bk = b[k], am = amp[k];
        complex<{T}> s = spec[k];
        {T} re = {ADD}({MUL}(s.real(), ak), {MUL}(bk, {MUL}(zr[m], am)));
        {T} im = {ADD}({MUL}(s.imag(), ak), {MUL}(bk, {MUL}(zi[lane + m], am)));
        spec[k] = complex<{T}>(re, im);
    }}
}}
"""

_kernels: dict = {}


def _kernel(name: str, real_dtype: Any):
    """Compile (once per dtype) and return one of the kernels in ``_SRC``."""
    import cupy

    dtype_name = np.dtype(real_dtype).name
    key = (name, dtype_name)
    kernel = _kernels.get(key)
    if kernel is None:
        ctype, suf = ("float", "f") if dtype_name == "float32" else ("double", "d")
        if name == "boil_blend_rng":
            from ._rng import _DEVICE_SRC

            per_block, normal = (4, "philox_normal4_f") if suf == "f" else (
                2, "philox_normal2_d")
            src = _DEVICE_SRC + _BOIL_SRC.format(
                T=ctype, suf=suf, PB=per_block, NORMAL=normal,
                MUL="__fmul_rn" if suf == "f" else "__dmul_rn",
                ADD="__fadd_rn" if suf == "f" else "__dadd_rn")
        else:
            src = _SRC.format(T=ctype, suf=suf)
        kernel = cupy.RawKernel(src, f"{name}_{suf}")
        _kernels[key] = kernel
    return kernel


def boil_blend_rng(coeffs: Any, amps: Any, a: Any, b: Any, rng: Any) -> None:
    """In-place AR(1) boil of complex ``coeffs`` with noise from ``rng`` drawn inline.

    Bit-identical to ``noise = rng.standard_normal((2, *coeffs.shape))``
    followed by ``coeffs = coeffs*a + b*((noise[0] + 1j*noise[1])*amps)``, and
    advances ``rng`` (a :class:`pyturb._rng.CudaNormalGenerator`) by the same
    amount, without writing and re-reading the noise. ``amps``, ``a`` and
    ``b`` are real with ``coeffs``' shape; all must be C-contiguous.
    """
    total = coeffs.size
    rdtype = coeffs.real.dtype
    key0, key1, offset, per_block = rng._reserve(2 * total, rdtype)
    threads = 256
    n_threads = -(-total // per_block)
    _kernel("boil_blend_rng", rdtype)(
        ((n_threads + threads - 1) // threads,), (threads,),
        (coeffs, amps, a, b, np.int64(total), np.uint32(key0), np.uint32(key1),
         np.uint64(offset)),
    )


class _Replay:
    """One captured frame: graph, stream, private pool, input and output buffers."""

    __slots__ = ("graph", "stream", "pool", "disp", "out")


class SpectralGPU:
    """Device-side spectral frames for one :class:`pyturb.Atmosphere`.

    Holds references to the atmosphere's stacked spectra and subharmonic
    arrays (which must be updated in place, as boiling does) plus the
    float64 constants needed to form phasors on the device. With an LGS
    (``lgs_altitude``) the frame is the per-layer cone path
    (:meth:`compute_lgs`) instead of the collapsed layer sum (:meth:`compute`).
    """

    def __init__(self, atm: Any, use_graph: bool) -> None:
        import cupy

        self.cp = cupy
        self.atm = atm
        ns = atm.n_screen
        self.ns = ns
        self.L = atm._spectra.shape[0]
        self.rdtype = atm._spectra.real.dtype
        self.cdtype = atm._cdtype
        self.period = float(atm._screen_period_m)
        self.f64 = cupy.asarray(np.fft.fftfreq(ns, d=atm.pixel_scale), dtype=np.float64)
        self.n_sh = atm._n_sh
        if self.n_sh:
            self.fp64 = cupy.asarray(atm._sh_freqs_host, dtype=np.float64)  # (P, 3)
            self.basis = cupy.ascontiguousarray(atm._sh_basis)  # (P, 3, ns)
        self.crop = atm._crop
        self.out_dtype = atm.dtype_out
        self.use_graph = use_graph
        # The constructor's LGS cone taps (None without lgs_altitude).
        self.lgs_taps = (None if atm._lgs_mag is None
                         else atm._zoom_taps(atm._lgs_mag))
        self._replays: dict = {}
        self._zoom_basis: dict = {}
        self._graph_failed = False

    @property
    def _graph(self) -> Optional[Any]:
        """The captured single-frame graph, if any (``None`` before capture)."""
        rep = self._replays.get("lgs" if self.lgs_taps is not None else "frame")
        return None if rep is None else rep.graph

    # ------------------------------------------------------------------
    def _phasors(self, disp: Any) -> Any:
        """Main-grid phasors ``exp(2 pi i f s)``: (B, L) float64 shifts to (B, L, ns)."""
        cp = self.cp
        shift = cp.mod(disp, self.period)
        return cp.exp((2j * np.pi) * shift[..., None] * self.f64).astype(self.cdtype)

    def _sh_shifted(self, disp0: Any, disp1: Any) -> Any:
        """Shifted subharmonic coefficients ``(B, P, L, 3, 3)`` for ``(B, L)`` shifts."""
        cp = self.cp
        # Subharmonic level p repeats only every 3**p periods: phase in
        # float64 cycles, reduced modulo one, then unit phasors.
        cyc0 = disp0[:, None, :, None] * self.fp64[None, :, None, :]  # (B,P,L,3)
        cyc1 = disp1[:, None, :, None] * self.fp64[None, :, None, :]
        sh0 = cp.exp((2j * np.pi) * cp.mod(cyc0, 1.0)).astype(self.cdtype)
        sh1 = cp.exp((2j * np.pi) * cp.mod(cyc1, 1.0)).astype(self.cdtype)
        return self.atm._sh_coeffs[None] * sh0[..., None] * sh1[..., None, :]

    def compute(self, disp0: Any, disp1: Any) -> Any:
        """Pupil phase ``(B, n, n)`` for float64 device displacements ``(B, L)``."""
        cp = self.cp
        atm = self.atm
        B = disp0.shape[0]
        ns = self.ns
        ph0 = self._phasors(disp0)
        ph1 = self._phasors(disp1)
        spectrum = cp.empty((B, ns, ns), dtype=self.cdtype)
        total = B * ns * ns
        threads = 256
        _kernel("spectral_layer_sum", self.rdtype)(
            ((total + threads - 1) // threads,), (threads,),
            (atm._spectra, ph0, ph1, spectrum, np.int32(self.L), np.int32(ns),
             np.int64(total), self.rdtype.type(ns * ns)),
        )
        field = atm._fft.ifft2(spectrum, axes=(-2, -1)).real
        if self.n_sh:
            shifted = cp.ascontiguousarray(
                self._sh_shifted(disp0, disp1).sum(axis=2))  # (B, P, 3, 3)
            # The basis products are tiny (9 terms per level per pixel) and
            # cuBLAS cannot run inside a CUDA-graph capture, so one kernel
            # evaluates them per pixel.
            low = cp.empty((B, ns, ns), dtype=self.rdtype)
            _kernel("subharmonic_field", self.rdtype)(
                ((total + threads - 1) // threads,), (threads,),
                (shifted, self.basis, low, np.int32(self.n_sh), np.int32(ns),
                 np.int64(total)),
            )
            low -= low.mean(axis=(-2, -1), keepdims=True)
            field = field + low
        out = field[:, self.crop, self.crop]
        return cp.ascontiguousarray(out.astype(self.out_dtype, copy=False))

    def _zoomed_basis(self, taps: Any) -> Any:
        """Subharmonic basis zoomed by ``taps`` and its screen mean (cached per taps).

        ``zb[l, p, a, i] = sum_t w[l, t, i] * basis[p, a, idx[l, t, i]]``
        (complex, ``(L, P, 3, n)``) and ``mb[p, a]``, the mean of
        ``basis[p, a]`` over the screen, both formed once in float64.
        """
        key = id(taps[0])
        cached = self._zoom_basis.get(key)
        if cached is not None and cached[0] is taps[0]:
            return cached[1]
        cp = self.cp
        idx = cp.asnumpy(taps[0])  # (L, T, n)
        w = cp.asnumpy(taps[1]).astype(np.float64)
        basis = cp.asnumpy(self.basis).astype(np.complex128)  # (P, 3, ns)
        zb = np.einsum("ltn,paltn->lpan", w, basis[:, :, idx])
        mb = basis.mean(axis=-1)  # (P, 3)
        out = (cp.asarray(zb, dtype=self.cdtype), cp.asarray(mb, dtype=self.cdtype))
        self._zoom_basis[key] = (taps[0], out)
        return out

    def compute_lgs(self, disp0: Any, disp1: Any, taps: Any) -> Any:
        """LGS-cone pupil phase ``(n, n)`` for float64 device displacements ``(1, L)``.

        Every layer is shifted and inverse-FFT'd on its own: only the real
        part of a layer's screen is used, so one kernel forms the Hermitian
        half of each shifted spectrum and a batched real inverse FFT (half
        the work of a complex one) returns the ``(L, ns, ns)`` screens. One
        kernel then cone-zooms every layer with ``taps`` (``(L, T, n)``
        indices and weights, see ``Atmosphere._zoom_taps``), sums the layers
        and adds the subharmonic field evaluated directly at the zoomed
        positions. No cuBLAS and no ``(L, n, ns)`` intermediates, so the
        frame is graph-capturable.
        """
        cp = self.cp
        atm = self.atm
        ns, L, n = self.ns, self.L, atm.n
        threads = 256
        ph0 = self._phasors(disp0[0])  # (L, ns)
        ph1 = self._phasors(disp1[0])
        nh = ns // 2 + 1
        total = L * ns * nh
        half = cp.empty((L, ns, nh), dtype=self.cdtype)
        _kernel("hermitian_layers", self.rdtype)(
            ((total + threads - 1) // threads,), (threads,),
            (atm._spectra, ph0, ph1, half, np.int32(ns), np.int32(nh),
             np.int64(total), self.rdtype.type(0.5 * ns * ns)),
        )
        screens = atm._fft.irfft2(half, s=(ns, ns), axes=(-2, -1))  # (L, ns, ns)
        idx, w = taps
        if self.n_sh:
            zb, mb = self._zoomed_basis(taps)
            shifted = cp.ascontiguousarray(
                self._sh_shifted(disp0, disp1)[0].transpose(1, 0, 2, 3))  # (L,P,3,3)
            zcols = cp.empty_like(zb)
            total = zb.size
            _kernel("subharmonic_taps", self.rdtype)(
                ((total + threads - 1) // threads,), (threads,),
                (shifted, zb, zcols, np.int32(n), np.int64(total)),
            )
            # Every layer's full-screen subharmonic mean, summed.
            low_mean = (shifted * mb[None, :, :, None] * mb[None, :, None, :]
                        ).real.sum(keepdims=True).ravel()
        out = cp.empty((n, n), dtype=self.rdtype)
        _kernel("lgs_zoom", self.rdtype)(
            ((n * n + threads - 1) // threads,), (threads,),
            (screens, idx, w, out, np.int32(L), np.int32(idx.shape[1]), np.int32(n),
             np.int32(ns)),
        )
        if self.n_sh:
            tiles = (n + 63) // 64
            _kernel("add_low_rank", self.rdtype)(
                (tiles, tiles), (16, 16),
                (zb, zcols, low_mean, out, np.int32(zb.size // n), np.int32(n)),
            )
        return out.astype(self.out_dtype, copy=False)

    def batch(self, disp0: np.ndarray, disp1: np.ndarray) -> Any:
        """Eager frames for host float64 displacements ``(B, L)``."""
        cp = self.cp
        return self.compute(cp.asarray(disp0, dtype=np.float64),
                            cp.asarray(disp1, dtype=np.float64))

    def lgs(self, disp0: np.ndarray, disp1: np.ndarray, taps: Any) -> Any:
        """Eager LGS-cone frame for host float64 displacements ``(L,)``."""
        cp = self.cp
        return self.compute_lgs(cp.asarray(disp0, dtype=np.float64)[None],
                                cp.asarray(disp1, dtype=np.float64)[None], taps)

    # ------------------------------------------------------------------
    def single(self, disp0: np.ndarray, disp1: np.ndarray) -> Any:
        """One frame ``(n, n)``, replayed from a CUDA graph when possible.

        With an LGS this is the constructor's cone (:meth:`compute_lgs`).
        """
        if self.lgs_taps is not None:
            key = "lgs"
            taps = self.lgs_taps

            def fn(d0: Any, d1: Any) -> Any:
                return self.compute_lgs(d0, d1, taps)

            def eager() -> Any:
                return self.lgs(disp0, disp1, taps)
        else:
            key = "frame"

            def fn(d0: Any, d1: Any) -> Any:
                return self.compute(d0, d1)[0]

            def eager() -> Any:
                return self.batch(disp0[None], disp1[None])[0]
        if not self.use_graph or self._graph_failed:
            return eager()
        rep = self._replays.get(key)
        if rep is None:
            rep = self._capture(fn)
            if rep is None:
                return eager()
            self._replays[key] = rep
        cp = self.cp
        current = cp.cuda.get_current_stream()
        rep.stream.wait_event(current.record())
        host = np.stack((disp0, disp1)).astype(np.float64)[:, None, :]
        with rep.stream:
            rep.disp.set(host)
            rep.graph.launch()
        current.wait_event(rep.stream.record())
        return rep.out.copy()

    def _capture(self, fn: Any) -> Optional[_Replay]:
        """Capture ``fn(disp0, disp1) -> (n, n)`` as a graph reading ``rep.disp``."""
        cp = self.cp
        try:
            rep = _Replay()
            rep.stream = cp.cuda.Stream(non_blocking=True)
            rep.pool = cp.cuda.MemoryPool()
            rep.disp = cp.zeros((2, 1, self.L), dtype=np.float64)
            n = self.atm.n
            rep.out = cp.empty((n, n), dtype=self.out_dtype)
            with cp.cuda.using_allocator(rep.pool.malloc), rep.stream:
                # Warm the FFT plan on this stream, then capture one frame.
                rep.out[...] = fn(rep.disp[0], rep.disp[1])
                rep.stream.synchronize()
                rep.stream.begin_capture()
                try:
                    rep.out[...] = fn(rep.disp[0], rep.disp[1])
                finally:
                    rep.graph = rep.stream.end_capture()
            return rep
        except Exception:  # driver/runtime without usable graph capture
            self._graph_failed = True
            return None

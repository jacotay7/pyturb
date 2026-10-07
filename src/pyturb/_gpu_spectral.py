"""GPU fast path for the spectral engine: fused layer sum and CUDA-graph replay.

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
        kernel = cupy.RawKernel(_SRC.format(T=ctype, suf=suf), f"{name}_{suf}")
        _kernels[key] = kernel
    return kernel


class SpectralGPU:
    """Device-side spectral frames for one :class:`pyturb.Atmosphere`.

    Holds references to the atmosphere's stacked spectra and subharmonic
    arrays (which must be updated in place, as boiling does) plus the
    float64 constants needed to form phasors on the device.
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
        self._graph: Optional[Any] = None
        self._graph_failed = False

    # ------------------------------------------------------------------
    def _phasors(self, disp: Any) -> Any:
        """Main-grid phasors ``exp(2 pi i f s)``: (B, L) float64 shifts to (B, L, ns)."""
        cp = self.cp
        shift = cp.mod(disp, self.period)
        return cp.exp((2j * np.pi) * shift[..., None] * self.f64).astype(self.cdtype)

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
            # Subharmonic level p repeats only every 3**p periods: phase in
            # float64 cycles, reduced modulo one, then unit phasors.
            cyc0 = disp0[:, None, :, None] * self.fp64[None, :, None, :]  # (B,P,L,3)
            cyc1 = disp1[:, None, :, None] * self.fp64[None, :, None, :]
            sh0 = cp.exp((2j * np.pi) * cp.mod(cyc0, 1.0)).astype(self.cdtype)
            sh1 = cp.exp((2j * np.pi) * cp.mod(cyc1, 1.0)).astype(self.cdtype)
            shifted = cp.ascontiguousarray((
                atm._sh_coeffs[None] * sh0[..., None] * sh1[..., None, :]
            ).sum(axis=2))  # (B, P, 3, 3)
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

    def batch(self, disp0: np.ndarray, disp1: np.ndarray) -> Any:
        """Eager frames for host float64 displacements ``(B, L)``."""
        cp = self.cp
        return self.compute(cp.asarray(disp0, dtype=np.float64),
                            cp.asarray(disp1, dtype=np.float64))

    # ------------------------------------------------------------------
    def single(self, disp0: np.ndarray, disp1: np.ndarray) -> Any:
        """One frame ``(n, n)``, replayed from a CUDA graph when possible."""
        if not self.use_graph or self._graph_failed:
            return self.batch(disp0[None], disp1[None])[0]
        cp = self.cp
        if self._graph is None:
            self._capture()
            if self._graph is None:
                return self.batch(disp0[None], disp1[None])[0]
        current = cp.cuda.get_current_stream()
        self._stream.wait_event(current.record())
        host = np.stack((disp0, disp1)).astype(np.float64)[:, None, :]
        with self._stream:
            self._disp.set(host)
            self._graph.launch()
        current.wait_event(self._stream.record())
        return self._out.copy()

    def _capture(self) -> None:
        cp = self.cp
        try:
            self._stream = cp.cuda.Stream(non_blocking=True)
            self._pool = cp.cuda.MemoryPool()
            self._disp = cp.zeros((2, 1, self.L), dtype=np.float64)
            n = self.atm.n
            self._out = cp.empty((n, n), dtype=self.out_dtype)
            with cp.cuda.using_allocator(self._pool.malloc), self._stream:
                # Warm the FFT plan on this stream, then capture one frame.
                self._out[...] = self.compute(self._disp[0], self._disp[1])[0]
                self._stream.synchronize()
                self._stream.begin_capture()
                try:
                    self._out[...] = self.compute(self._disp[0], self._disp[1])[0]
                finally:
                    graph = self._stream.end_capture()
            self._graph = graph
        except Exception:  # driver/runtime without usable graph capture
            self._graph = None
            self._graph_failed = True

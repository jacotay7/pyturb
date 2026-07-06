"""Optional Numba-accelerated CPU kernels (transparent, with NumPy fallback).

The CPU hot paths — the spectral engine's per-frame layer sum and the
extruder's per-frame bicubic readout — are memory-bandwidth bound in plain
NumPy because each fuses several broadcast multiplies into large ``(L, n, n)``
temporaries. A single fused, ``prange``-parallel pass over the data does the
same arithmetic reading the inputs once and writing the output once, across all
cores. These mirror the GPU's fused kernels so both backends do the same work
per frame.

Numba is an *optional* dependency: if it is not importable, :data:`HAVE_NUMBA`
is ``False`` and callers fall back to the NumPy expressions. Nothing here
changes results beyond float round-off (the fused reductions accumulate in the
input dtype for the spectral sum and in double for the readout, matching each
NumPy path).
"""

from __future__ import annotations

from importlib.util import find_spec

HAVE_NUMBA = find_spec("numba") is not None


if HAVE_NUMBA:
    import numpy as np
    from numba import njit, prange

    @njit(fastmath=True, cache=True)
    def spectral_layer_sum(spectra, px, py, out):  # pragma: no cover - jit
        """``out[i,j] = sum_l spectra[l,i,j] * px[l,i] * py[l,j]``.

        The separable frozen-flow shift and layer sum in one fused pass: the
        ``(L, n, n)`` spectrum stack is read once and the ``(n, n)`` shifted sum
        written once, instead of materialising two ``(L, n, n)`` complex
        products and reducing them.

        Kept single-threaded on purpose: reducing a power-of-two-sized complex
        stack over the leading (layer) axis strides the inner loop by a whole
        2-D plane, so a ``prange`` over pixels hits cache-set aliasing exactly
        at the power-of-two grids AO uses (256/512/1024). The serial fused pass
        is a robust, monotonic win over NumPy at every size; the far larger
        parallel win goes to the extruder readout, whose gather is
        data-dependent and does not alias.
        """
        n_layers, n_rows, n_cols = spectra.shape
        for i in range(n_rows):
            for j in range(n_cols):
                acc = spectra[0, i, j] * px[0, i] * py[0, j]
                for lyr in range(1, n_layers):
                    acc += spectra[lyr, i, j] * px[lyr, i] * py[lyr, j]
                out[i, j] = acc

    @njit(parallel=True, fastmath=True, cache=True)
    def extrude_cubic_readout(buf, along, perp, sa, sp, fillm1, out):  # pragma: no cover
        """Fused multi-layer Catmull-Rom pupil readout, summed over layers.

        ``buf`` is the ``(L, cap, W)`` ring-buffer stack; ``along``/``perp`` are
        the ``(L, n, n)`` rotated pupil grids; ``sa``/``sp`` the per-layer
        along/perp readout shifts; ``fillm1`` the per-layer last valid row.
        Accumulates in double, matching the tap-broadcast gather it replaces.
        """
        n_layers = buf.shape[0]
        width = buf.shape[2]
        n = out.shape[0]
        for i in prange(n):
            for j in range(n):
                acc = 0.0
                for lyr in range(n_layers):
                    row = along[lyr, i, j] + sa[lyr]
                    col = perp[lyr, i, j] + sp[lyr]
                    r0 = int(np.floor(row))
                    c0 = int(np.floor(col))
                    fr = row - r0
                    fc = col - c0
                    fr2 = fr * fr
                    fr3 = fr2 * fr
                    fc2 = fc * fc
                    fc3 = fc2 * fc
                    wr = (
                        0.5 * (-fr + 2.0 * fr2 - fr3),
                        0.5 * (2.0 - 5.0 * fr2 + 3.0 * fr3),
                        0.5 * (fr + 4.0 * fr2 - 3.0 * fr3),
                        0.5 * (-fr2 + fr3),
                    )
                    wc = (
                        0.5 * (-fc + 2.0 * fc2 - fc3),
                        0.5 * (2.0 - 5.0 * fc2 + 3.0 * fc3),
                        0.5 * (fc + 4.0 * fc2 - 3.0 * fc3),
                        0.5 * (-fc2 + fc3),
                    )
                    fmax = fillm1[lyr]
                    val = 0.0
                    for a in range(4):
                        rr = r0 + (a - 1)
                        if rr < 0:
                            rr = 0
                        elif rr > fmax:
                            rr = fmax
                        rs = 0.0
                        for b in range(4):
                            cc = c0 + (b - 1)
                            if cc < 0:
                                cc = 0
                            elif cc > width - 1:
                                cc = width - 1
                            rs += wc[b] * buf[lyr, rr, cc]
                        val += wr[a] * rs
                    acc += val
                out[i, j] = acc

    @njit(fastmath=True, inline="always")
    def _sincpi(x):  # pragma: no cover - jit helper
        if x == 0.0:
            return 1.0
        px = 3.141592653589793 * x
        return np.sin(px) / px

    @njit(parallel=True, fastmath=True, cache=True)
    def extrude_lanczos_readout(buf, along, perp, sa, sp, fillm1, out):  # noqa: E501  pragma: no cover
        """Fused multi-layer Lanczos-3 (6-tap) pupil readout, summed over layers.

        The higher-fidelity counterpart of :func:`extrude_cubic_readout`: a
        flatter sub-Nyquist windowed-sinc kernel (6 taps per axis) that halves
        the extruder's finest-scale structure-function deficit. Accumulates in
        double, matching the tap-broadcast Lanczos gather it replaces.
        """
        n_layers = buf.shape[0]
        width = buf.shape[2]
        n = out.shape[0]
        for i in prange(n):
            for j in range(n):
                acc = 0.0
                for lyr in range(n_layers):
                    row = along[lyr, i, j] + sa[lyr]
                    col = perp[lyr, i, j] + sp[lyr]
                    r0 = int(np.floor(row))
                    c0 = int(np.floor(col))
                    tr = row - r0
                    tc = col - c0
                    wr0 = _sincpi(tr + 2.0) * _sincpi((tr + 2.0) / 3.0)
                    wr1 = _sincpi(tr + 1.0) * _sincpi((tr + 1.0) / 3.0)
                    wr2 = _sincpi(tr) * _sincpi(tr / 3.0)
                    wr3 = _sincpi(tr - 1.0) * _sincpi((tr - 1.0) / 3.0)
                    wr4 = _sincpi(tr - 2.0) * _sincpi((tr - 2.0) / 3.0)
                    wr5 = _sincpi(tr - 3.0) * _sincpi((tr - 3.0) / 3.0)
                    sr = wr0 + wr1 + wr2 + wr3 + wr4 + wr5
                    wr = (wr0 / sr, wr1 / sr, wr2 / sr, wr3 / sr, wr4 / sr, wr5 / sr)
                    wc0 = _sincpi(tc + 2.0) * _sincpi((tc + 2.0) / 3.0)
                    wc1 = _sincpi(tc + 1.0) * _sincpi((tc + 1.0) / 3.0)
                    wc2 = _sincpi(tc) * _sincpi(tc / 3.0)
                    wc3 = _sincpi(tc - 1.0) * _sincpi((tc - 1.0) / 3.0)
                    wc4 = _sincpi(tc - 2.0) * _sincpi((tc - 2.0) / 3.0)
                    wc5 = _sincpi(tc - 3.0) * _sincpi((tc - 3.0) / 3.0)
                    sc = wc0 + wc1 + wc2 + wc3 + wc4 + wc5
                    wc = (wc0 / sc, wc1 / sc, wc2 / sc, wc3 / sc, wc4 / sc, wc5 / sc)
                    fmax = fillm1[lyr]
                    val = 0.0
                    for a in range(6):
                        rr = r0 + (a - 2)
                        if rr < 0:
                            rr = 0
                        elif rr > fmax:
                            rr = fmax
                        rs = 0.0
                        for b in range(6):
                            cc = c0 + (b - 2)
                            if cc < 0:
                                cc = 0
                            elif cc > width - 1:
                                cc = width - 1
                            rs += wc[b] * buf[lyr, rr, cc]
                        val += wr[a] * rs
                    acc += val
                out[i, j] = acc

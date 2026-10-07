"""Optional Numba-accelerated CPU kernels (transparent, with NumPy fallback).

The CPU hot paths — the spectral engine's per-frame layer sum, its boiling
update and LGS cone zoom, and the extruder's per-frame bicubic readout — are
memory-bandwidth bound in plain
NumPy because each fuses several broadcast multiplies into large ``(L, n, n)``
temporaries. A single fused, ``prange``-parallel pass over the data does the
same arithmetic reading the inputs once and writing the output once, across all
cores. These mirror the GPU's fused kernels so both backends do the same work
per frame.

Numba is an *optional* dependency: if it is not importable, :data:`HAVE_NUMBA`
is ``False`` and callers fall back to the NumPy expressions. Nothing here
changes results beyond float round-off (the fused reductions accumulate in the
input dtype for the spectral sum and in double for the readout, matching each
NumPy path); the boiling update and the cone zoom round every operation exactly
as their NumPy expressions do, so those two are bit-identical to them.
"""

from __future__ import annotations

from importlib.util import find_spec

HAVE_NUMBA = find_spec("numba") is not None


if HAVE_NUMBA:
    import numpy as np
    from numba import njit, prange

    @njit(fastmath=True, cache=True)
    def _spectral_rows(spectra, ph0, ph1, out, i0, i1):  # pragma: no cover - jit
        """Rows ``i0:i1`` of :func:`spectral_layer_sum` (the per-pixel kernel)."""
        n_layers, n_rows, n_cols = spectra.shape
        for i in range(i0, i1):
            for j in range(n_cols):
                acc = spectra[0, i, j] * ph0[0, i] * ph1[0, j]
                for lyr in range(1, n_layers):
                    acc += spectra[lyr, i, j] * ph0[lyr, i] * ph1[lyr, j]
                out[i, j] = acc

    @njit(parallel=True, cache=True)
    def _spectral_rows_parallel(spectra, ph0, ph1, out):  # pragma: no cover - jit
        for i in prange(spectra.shape[1]):
            _spectral_rows(spectra, ph0, ph1, out, i, i + 1)

    # Below this many layer-pixels the serial pass is used: it takes ~2 ms or
    # less, and a parallel region that small gains little and is easily
    # stalled by one preempted thread on a busy host (measured on a 12-core
    # Neoverse-N1, where 9 x 512² runs ~5x faster in parallel).
    _PARALLEL_MIN_ELEMS = 1 << 21

    def spectral_layer_sum(spectra, ph0, ph1, out):  # pragma: no cover - needs Numba
        """``out[i,j] = sum_l spectra[l,i,j] * ph0[l,i] * ph1[l,j]``.

        The separable frozen-flow shift and layer sum in one fused pass: the
        ``(L, n, n)`` spectrum stack is read once and the ``(n, n)`` shifted sum
        written once, instead of materialising two ``(L, n, n)`` complex
        products and reducing them.

        Every pixel runs the same compiled per-pixel loop
        (:func:`_spectral_rows`), and large stacks split the rows across
        threads, so the result is bit-identical whatever the thread count. The
        innermost loop runs over layers, a whole plane apart; at the
        power-of-two grids AO uses (256/512/1024) those reads alias in the
        cache, which limits one core to ~10 ms at 9 x 512² -- the threads
        recover most of it without changing that per-pixel arithmetic.
        """
        if spectra.size >= _PARALLEL_MIN_ELEMS:
            _spectral_rows_parallel(spectra, ph0, ph1, out)
        else:
            _spectral_rows(spectra, ph0, ph1, out, 0, spectra.shape[1])

    @njit(parallel=True, cache=True)
    def boil_blend(spec, noise, amp, a, b):  # pragma: no cover - jit
        """In-place AR(1) boil: ``spec = spec*a + b*((noise[0] + 1j*noise[1])*amp)``.

        ``spec`` is complex and every other operand real with ``spec``'s shape
        (``noise`` has a leading axis of 2: real and imaginary draws). Each
        product and sum is rounded exactly as the NumPy expression rounds it
        (no ``fastmath``, so no fused multiply-add), so the result is
        bit-identical to it, without its five full-size temporaries.
        """
        s = spec.ravel()
        n0 = noise[0].ravel()
        n1 = noise[1].ravel()
        am = amp.ravel()
        av = a.ravel()
        bv = b.ravel()
        for k in prange(s.size):
            ak = av[k]
            bk = bv[k]
            re = s[k].real * ak + bk * (n0[k] * am[k])
            im = s[k].imag * ak + bk * (n1[k] * am[k])
            s[k] = complex(re, im)

    @njit(inline="always")
    def _zoom_layer(field, idx, w, lyr, i, j):  # pragma: no cover - jit helper
        """One layer's zoomed value at output pixel ``(i, j)`` (see lgs_zoom)."""
        n_taps = idx.shape[1]
        out = w[lyr, 0, i] * _zoom_row(field, idx, w, lyr, idx[lyr, 0, i], j)
        for a in range(1, n_taps):
            out = out + w[lyr, a, i] * _zoom_row(field, idx, w, lyr, idx[lyr, a, i], j)
        return out

    @njit(inline="always")
    def _zoom_row(field, idx, w, lyr, r, j):  # pragma: no cover - jit helper
        """Column-tap sum ``sum_b w_b(j) * field[r, idx_b(j)]`` (in tap order)."""
        n_taps = idx.shape[1]
        acc = w[lyr, 0, j] * field[r, idx[lyr, 0, j]]
        for b in range(1, n_taps):
            acc = acc + w[lyr, b, j] * field[r, idx[lyr, b, j]]
        return acc

    @njit(parallel=True, cache=True)
    def lgs_zoom(screens, idx, w, out):  # pragma: no cover - jit
        """Separable cone zoom of ``(L, ns, ns)`` layer screens, summed into ``out``.

        ``idx``/``w`` are the ``(L, T, n)`` tap indices and weights (shared by
        rows and columns). Per output pixel and layer the arithmetic is the
        NumPy per-layer gather's, in the same order: the taps along columns
        summed first (``w_b(j) * f``), each row sum weighted by ``w_a(i)``, the
        rows summed, then the layers. No ``fastmath``, so every operation is
        rounded as there and the result is bit-identical, without the ``T**2``
        full-size gathers and temporaries per layer.
        """
        n_layers = idx.shape[0]
        n = idx.shape[2]
        for i in prange(n):
            for j in range(n):
                total = _zoom_layer(screens[0], idx, w, 0, i, j)
                for lyr in range(1, n_layers):
                    total = total + _zoom_layer(screens[lyr], idx, w, lyr, i, j)
                out[i, j] = total

    @njit(parallel=True, cache=True)
    def infinite_cubic(buf, i0, t, t2, t3, fmax, consts, out):  # pragma: no cover - jit
        """Catmull-Rom row readout of :class:`pyturb.InfinitePhaseScreen`.

        ``out[i] = 0.5*(2 p0 + (-pm1 + p1) t + (2 pm1 - 5 p0 + 4 p1 - p2) t2
        + (-pm1 + 3 p0 - 3 p1 + p2) t3)`` with ``p_k = buf[clip(i0[i] + k)]``,
        evaluated operation by operation in the order NumPy evaluates that
        expression (no ``fastmath``; ``consts`` = ``[0.5, 2, 3, 4, 5]`` in the
        buffer's dtype keeps every operation in that dtype), so the result is
        bit-identical to it without its ~20 full-size temporaries.
        """
        n = out.shape[0]
        half, two, three = consts[0], consts[1], consts[2]
        four, five = consts[3], consts[4]
        for i in prange(n):
            r = i0[i]
            rm1 = min(max(r - 1, 0), fmax)
            rp0 = min(max(r, 0), fmax)
            rp1 = min(max(r + 1, 0), fmax)
            rp2 = min(max(r + 2, 0), fmax)
            ti = t[i, 0]
            t2i = t2[i, 0]
            t3i = t3[i, 0]
            for j in range(n):
                p0 = buf[rp0, j]
                p1 = buf[rp1, j]
                pm1 = buf[rm1, j]
                p2 = buf[rp2, j]
                acc = two * p0 + (-pm1 + p1) * ti
                acc = acc + (two * pm1 - five * p0 + four * p1 - p2) * t2i
                acc = acc + (-pm1 + three * p0 - three * p1 + p2) * t3i
                out[i, j] = half * acc

    @njit(parallel=True, cache=True)
    def infinite_taps(buf, i0, w, offsets, fmax, out):  # pragma: no cover - jit
        """``out[i] = sum_k w[k][i] * buf[clip(i0[i] + offsets[k])]``, in ``k`` order.

        The linear/Lanczos row readout of :class:`pyturb.InfinitePhaseScreen`,
        rounded as the NumPy tap loop is (no ``fastmath``): bit-identical.
        """
        n = out.shape[0]
        n_taps = offsets.shape[0]
        for i in prange(n):
            r0 = min(max(i0[i] + offsets[0], 0), fmax)
            w0 = w[0, i]
            for j in range(n):
                out[i, j] = w0 * buf[r0, j]
            for k in range(1, n_taps):
                rk = min(max(i0[i] + offsets[k], 0), fmax)
                wk = w[k, i]
                for j in range(n):
                    out[i, j] = out[i, j] + wk * buf[rk, j]

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

"""The optional Numba CPU kernels must match the NumPy fallback exactly.

pyturb dispatches the CPU frozen-flow hot paths to fused Numba kernels when
Numba is importable and to NumPy expressions otherwise. Both must agree (to
float round-off), or a user's results would depend on whether Numba happens to
be installed. These tests force each path and compare.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyturb
from pyturb import _accel

pytestmark = pytest.mark.skipif(
    not _accel.HAVE_NUMBA, reason="accel comparison needs Numba installed"
)


def _frame(atm):
    return pyturb.to_numpy(next(iter(atm.frames(dt=1e-3, steps=1)))[1])


def _both_paths(monkeypatch, make):
    """Return (numba_result, numpy_result) for a freshly built atmosphere."""
    accel_out = _frame(make())
    monkeypatch.setattr(_accel, "HAVE_NUMBA", False)
    numpy_out = _frame(make())
    return accel_out, numpy_out


def test_spectral_layer_sum_matches_numpy(monkeypatch):
    def make():
        return pyturb.Atmosphere.from_profile(
            "paranal-median", seeing=0.8, n=128, device="cpu", seed=7
        )

    numba_out, numpy_out = _both_paths(monkeypatch, make)
    rms = np.sqrt(np.mean(numpy_out ** 2))
    assert np.abs(numba_out - numpy_out).max() < 1e-5 * rms


@pytest.mark.parametrize("interp", ["cubic", "lanczos"])
def test_extrude_readout_matches_numpy(monkeypatch, interp):
    def make():
        return pyturb.Atmosphere.from_profile(
            "paranal-median", seeing=0.8, n=128, device="cpu",
            engine="extrude", interp=interp, seed=7,
        )

    numba_out, numpy_out = _both_paths(monkeypatch, make)
    # The extrude readout accumulates in double on both paths -> bit-identical.
    assert np.array_equal(numba_out, numpy_out)


def test_spectral_layer_sum_kernel_direct():
    """The bare kernel reproduces the ``(spectra*px*py).sum(0)`` expression."""
    rng = np.random.default_rng(0)
    L, n = 9, 96
    spectra = (rng.standard_normal((L, n, n)) + 1j * rng.standard_normal((L, n, n))
               ).astype(np.complex64)
    px = (rng.standard_normal((L, n)) + 1j * rng.standard_normal((L, n))).astype(
        np.complex64)
    py = (rng.standard_normal((L, n)) + 1j * rng.standard_normal((L, n))).astype(
        np.complex64)
    ref = (spectra * px[:, :, None] * py[:, None, :]).sum(0)
    out = np.empty((n, n), dtype=np.complex64)
    _accel.spectral_layer_sum(spectra, px, py, out)
    assert np.abs(ref - out).max() < 1e-4 * np.abs(ref).mean()


def test_spectral_layer_sum_is_thread_independent():
    """Large stacks split rows across threads; every pixel must still come
    from the same compiled per-pixel loop, so the result is the serial one bit
    for bit (whatever the thread count)."""
    rng = np.random.default_rng(1)
    L, n = 9, 512  # above the parallel threshold
    spectra = (rng.standard_normal((L, n, n)) + 1j * rng.standard_normal((L, n, n))
               ).astype(np.complex64)
    px = np.exp(2j * np.pi * rng.random((L, n))).astype(np.complex64)
    py = np.exp(2j * np.pi * rng.random((L, n))).astype(np.complex64)
    serial = np.empty((n, n), dtype=np.complex64)
    _accel._spectral_rows(spectra, px, py, serial, 0, n)
    parallel = np.empty_like(serial)
    _accel.spectral_layer_sum(spectra, px, py, parallel)
    np.testing.assert_array_equal(parallel.view(np.uint32), serial.view(np.uint32))


@pytest.mark.parametrize("interp", ["cubic", "linear", "lanczos"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_lgs_zoom_matches_numpy_exactly(monkeypatch, interp, dtype):
    # The fused cone zoom rounds every operation as the NumPy gather does.
    def make():
        return pyturb.Atmosphere.from_profile(
            "paranal-median", seeing=0.8, n=96, device="cpu", seed=7,
            lgs_altitude=90e3, interp=interp, dtype=dtype,
        )

    numba_out, numpy_out = _both_paths(monkeypatch, make)
    np.testing.assert_array_equal(numba_out, numpy_out)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_boil_blend_matches_numpy_exactly(monkeypatch, dtype):
    # Boiling's fused AR(1) update must give the NumPy expression's bits, so
    # a boiled run does not depend on whether Numba is installed. The (2, L,
    # n, n) noise draw is the same either way; compare several boiled frames.
    def run():
        atm = pyturb.Atmosphere.from_profile(
            "paranal-median", seeing=0.8, n=64, device="cpu", seed=7,
            tau_boil=[0.05, np.inf, 0.1, 0.02, np.inf, 0.3, 0.05, 0.05, 1.0],
            dtype=dtype,
        )
        frames = [f for _, f in atm.frames(dt=1e-3, steps=4)]
        return np.stack(frames), atm._spectra.copy(), atm._sh_coeffs.copy()

    accel = run()
    monkeypatch.setattr(_accel, "HAVE_NUMBA", False)
    plain = run()
    for got, ref in zip(accel[1:], plain[1:]):
        np.testing.assert_array_equal(got, ref)
    # Frames also go through the (Numba vs NumPy) layer sum: round-off only.
    rms = np.sqrt(np.mean(plain[0] ** 2))
    assert np.abs(accel[0] - plain[0]).max() < 1e-5 * rms


@pytest.mark.parametrize("interp", ["cubic", "linear", "lanczos"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_infinite_readout_matches_numpy_exactly(monkeypatch, interp, dtype):
    # InfinitePhaseScreen's fused row readout rounds every operation as the
    # NumPy expression does: whole and sub-pixel steps give the same bits.
    def run():
        layer = pyturb.InfinitePhaseScreen(64, 0.05, 0.15, 25.0, interp=interp,
                                           seed=3, dtype=dtype)
        return np.stack([layer.step(), layer.advance(0.37), layer.advance(2.81)])

    accel = run()
    monkeypatch.setattr(_accel, "HAVE_NUMBA", False)
    np.testing.assert_array_equal(accel, run())

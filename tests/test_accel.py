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

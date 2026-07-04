import importlib.util

import numpy as np
import pytest

import pyturb
from pyturb.backend import get_array_module

CUPY_INSTALLED = importlib.util.find_spec("cupy") is not None


def test_cpu_module_is_numpy():
    assert get_array_module("cpu") is np
    assert get_array_module("numpy") is np


def test_unknown_device_raises():
    with pytest.raises(ValueError, match="Unknown device"):
        get_array_module("tpu")


@pytest.mark.skipif(CUPY_INSTALLED, reason="CuPy is installed")
def test_gpu_without_cupy_raises_helpful_error():
    with pytest.raises(ImportError, match="CuPy"):
        pyturb.PhaseScreen(n=32, pixel_scale=0.01, r0=0.1, device="gpu")


@pytest.mark.skipif(not CUPY_INSTALLED, reason="CuPy not installed")
def test_gpu_generation_matches_cpu_statistics():
    import cupy

    gen = pyturb.PhaseScreen(n=64, pixel_scale=0.02, r0=0.1, seed=0, device="gpu")
    screen = gen.generate()
    assert isinstance(screen, cupy.ndarray)
    host = pyturb.to_numpy(screen)
    assert host.shape == (64, 64)
    assert np.isfinite(host).all()


def test_to_numpy_passthrough():
    array = np.ones((3, 3))
    assert pyturb.to_numpy(array) is not None
    np.testing.assert_array_equal(pyturb.to_numpy(array), array)


def test_seeing_conversions_round_trip():
    r0 = pyturb.r0_from_seeing(1.0)
    assert pyturb.seeing_from_r0(r0) == pytest.approx(1.0)
    # 1 arcsec seeing at 500 nm is roughly a 10 cm Fried parameter.
    assert r0 == pytest.approx(0.1, rel=0.05)


def test_r0_wavelength_scaling():
    r0_k_band = pyturb.r0_at_wavelength(0.15, 500e-9, 2.2e-6)
    assert r0_k_band == pytest.approx(0.15 * (2.2e-6 / 500e-9) ** 1.2)
    assert r0_k_band > 0.15


def test_structure_function_input_validation():
    screen = np.zeros((16, 16))
    with pytest.raises(ValueError):
        pyturb.structure_function(screen, max_separation=16)
    r, d = pyturb.structure_function(screen)
    assert len(r) == 4  # default n // 4
    np.testing.assert_array_equal(d, 0.0)


def test_fft_workers_setting_is_result_invariant():
    import pyturb
    prev = pyturb.set_fft_workers(None)
    try:
        g = pyturb.PhaseScreen(n=128, pixel_scale=0.02, r0=0.15, seed=0,
                               dtype="float64")
        single = g.generate(3)
        pyturb.set_fft_workers(-1)
        assert pyturb.get_fft_workers() == -1
        g2 = pyturb.PhaseScreen(n=128, pixel_scale=0.02, r0=0.15, seed=0,
                                dtype="float64")
        threaded = g2.generate(3)
        np.testing.assert_allclose(single, threaded, rtol=1e-10)
        with pytest.raises(ValueError):
            pyturb.set_fft_workers(0)
    finally:
        pyturb.set_fft_workers(prev)

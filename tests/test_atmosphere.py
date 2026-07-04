import numpy as np
import pytest

import pyturb
from pyturb.flow import FourierFlowScreen


def test_construction_requires_exactly_one_of_r0_seeing():
    layers = pyturb.get_profile("single-layer")
    with pytest.raises(ValueError):
        pyturb.Atmosphere(layers)
    with pytest.raises(ValueError):
        pyturb.Atmosphere(layers, r0=0.1, seeing=0.8)


def test_opd_shape_and_units():
    atm = pyturb.Atmosphere.from_profile("two-layer", r0=0.15, diameter=8.0, n=128,
                                         seed=1)
    opd = atm.opd()
    assert opd.shape == (128, 128)
    # OPD is in metres: sub-micron-to-micron RMS for a typical atmosphere.
    assert 1e-8 < float(pyturb.to_numpy(opd).std()) < 1e-5
    # Phase output scales as 2 pi / wavelength.
    phase = atm.opd(wavelength=500e-9)
    ratio = float(pyturb.to_numpy(phase).std()) / float(pyturb.to_numpy(opd).std())
    assert abs(ratio - 2 * np.pi / 500e-9) / (2 * np.pi / 500e-9) < 1e-4


def test_seeing_round_trips_at_zenith():
    atm = pyturb.Atmosphere.from_profile("single-layer", seeing=0.8, n=64,
                                         zenith_angle=0.0)
    assert abs(atm.seeing - 0.8) < 1e-6


def test_zenith_angle_worsens_seeing():
    atm0 = pyturb.Atmosphere.from_profile("single-layer", seeing=0.8, n=64,
                                          zenith_angle=0.0)
    atm60 = pyturb.Atmosphere.from_profile("single-layer", seeing=0.8, n=64,
                                           zenith_angle=60.0)
    # sec(60) = 2 airmasses -> seeing worsens by 2^{3/5}.
    assert abs(atm60.seeing / atm0.seeing - 2.0 ** (3.0 / 5.0)) < 1e-6
    assert atm60.r0 < atm0.r0


def test_frozen_flow_is_deterministic_and_continuous():
    atm = pyturb.Atmosphere.from_profile("two-layer", r0=0.15, diameter=8.0, n=128,
                                         seed=1)
    a = pyturb.to_numpy(atm.opd(0.0))
    b = pyturb.to_numpy(atm.opd(0.0))
    assert np.array_equal(a, b)  # random access is deterministic
    # frames() continues the clock across calls
    ts = [t for t, _ in atm.frames(dt=1e-3, steps=3)]
    assert ts == [0.0, 1e-3, 2e-3]
    assert abs(atm.time - 3e-3) < 1e-12
    atm.reset()
    assert atm.time == 0.0


def test_frozen_flow_shifts_screen_by_wind():
    # One-layer atmosphere, pure wind along axis 0: opd at time t equals the
    # t=0 screen rolled by the integer number of pixels blown.
    layer = pyturb.Layer(0.0, 1.0, wind_speed=10.0, wind_direction=0.0, L0=25.0)
    atm = pyturb.Atmosphere([layer], r0=0.15, diameter=8.0, n=128, subharmonics=0,
                            dtype="float64", seed=2)
    base = pyturb.to_numpy(atm.opd(0.0))
    dx = atm.pixel_scale
    t = 5 * dx / 10.0  # blow exactly 5 pixels
    shifted = pyturb.to_numpy(atm.opd(t))
    assert np.allclose(shifted, np.roll(base, -5, axis=0), atol=1e-6)


def test_directions_on_axis_matches_and_offset_decorrelates():
    atm = pyturb.Atmosphere.from_profile("paranal-median", r0=0.15, diameter=8.0,
                                         n=128, dtype="float64", seed=3)
    out = atm.opd(0.0, directions=[(0.0, 0.0), (30.0, 0.0)])
    assert out.shape == (2, 128, 128)
    on_axis = pyturb.to_numpy(atm.opd(0.0))
    assert np.allclose(pyturb.to_numpy(out[0]), on_axis, atol=1e-9)
    # A large off-axis angle must decorrelate relative to on-axis.
    a = pyturb.to_numpy(out[0]).ravel()
    b = pyturb.to_numpy(out[1]).ravel()
    corr = np.corrcoef(a, b)[0, 1]
    assert corr < 0.95


def test_sample_matches_total_structure_function():
    # Kolmogorov (L0=inf) so the ensemble matches 6.88 (r/r0)^{5/3}.
    atm = pyturb.Atmosphere.from_profile("paranal-median", r0=0.15, diameter=8.0,
                                         n=256, L0=np.inf, dtype="float64", seed=4)
    phase = atm.sample(12, wavelength=500e-9)
    assert phase.shape == (12, 256, 256)
    r, D = pyturb.structure_function(phase, atm.pixel_scale, max_separation=24)
    theory = 6.88 * (r / atm.r0) ** (5.0 / 3.0)
    ratio = D[3:] / theory[3:]
    assert np.all(ratio > 0.8)
    assert np.all(ratio < 1.2)


def _cupy_available():
    try:
        pyturb.get_array_module("gpu")
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _cupy_available(), reason="CuPy not installed")
def test_gpu_matches_cpu_statistics():
    kw = dict(seeing=0.8, diameter=8.0, n=256, L0=np.inf, dtype="float64", seed=9)
    atm_c = pyturb.Atmosphere.from_profile("paranal-median", device="cpu", **kw)
    atm_g = pyturb.Atmosphere.from_profile("paranal-median", device="gpu", **kw)
    r0 = atm_c.r0
    for atm in (atm_c, atm_g):
        phase = atm.sample(10, wavelength=500e-9)
        r, D = pyturb.structure_function(phase, atm.pixel_scale, max_separation=24)
        ratio = D[3:] / (6.88 * (r[3:] / r0) ** (5.0 / 3.0))
        assert np.all(ratio > 0.8) and np.all(ratio < 1.2)
    # A GPU frame comes back as a device array.
    _, opd = next(atm_g.frames(dt=1e-3, steps=1))
    assert type(opd).__module__.startswith("cupy")


def test_fourier_flow_translation_exact_integer_pixels():
    ps = pyturb.PhaseScreen(n=128, pixel_scale=0.02, r0=0.15, L0=25,
                            subharmonics=0, dtype="float64", seed=3)
    fl = FourierFlowScreen(ps, seed=5)
    base = fl.translate(0.0, 0.0)
    shifted = fl.translate(7 * 0.02, 0.0)
    assert np.allclose(shifted, np.roll(base, -7, axis=0), atol=1e-9)

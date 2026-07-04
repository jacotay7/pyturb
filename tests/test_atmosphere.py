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


_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


def test_field_of_view_oversizes_screen():
    atm0 = pyturb.Atmosphere.from_profile("paranal-median", r0=0.15, n=128,
                                          diameter=8.0, seed=1)
    atmf = pyturb.Atmosphere.from_profile("paranal-median", r0=0.15, n=128,
                                          diameter=8.0, field_of_view=30.0, seed=1)
    assert atm0.n_screen == 128 and atm0.margin_pix == 0
    assert atmf.n_screen > 128 and atmf.margin_pix > 0
    # Output is still the pupil size regardless of oversizing.
    assert atmf.opd().shape == (128, 128)
    assert atmf.sample(2).shape == (2, 128, 128)


def test_off_axis_is_a_pure_shift_without_wrap():
    # Single layer, no wind: an off-axis direction chosen to land on an exact
    # integer-pixel footprint shift must equal the on-axis screen rolled, with
    # no wrap contamination because field_of_view oversized the screen.
    h = 8000.0
    atm = pyturb.Atmosphere([pyturb.Layer(h, 1.0, 0.0, 0.0, L0=np.inf)],
                            r0=0.15, n=160, diameter=10.0, field_of_view=25.0,
                            subharmonics=0, dtype="float64", seed=7)
    k = 6
    theta_as = (k * atm.pixel_scale / (h * atm.airmass)) / _ARCSEC_TO_RAD
    # Work in phase (radians) so it matches flow.translate()'s units.
    lam = atm.wavelength
    off = pyturb.to_numpy(atm.opd(0.0, directions=[(theta_as, 0.0)], wavelength=lam))[0]
    full = pyturb.to_numpy(atm._layers[0].flow.translate(0.0, 0.0))
    rolled = np.roll(full, -k, axis=0)[atm._crop, atm._crop]
    # Relative error is O(1e-6) of the screen RMS (arcsec-rounding only).
    assert np.abs(off - rolled).max() < 1e-3 * rolled.std()


def test_boiling_decorrelates_as_expected_and_is_stationary():
    # Ensemble-averaged temporal autocorrelation of a boiling layer must follow
    # exp(-t/tau); a single realisation is dominated by a few low-frequency
    # modes and is far too noisy to test.
    tau, dt = 0.05, 0.005
    lags = [1, 2, 5]
    num = {k: 0.0 for k in lags}
    den0 = 0.0
    denk = {k: 0.0 for k in lags}
    rms_first, rms_last = [], []
    for m in range(60):
        atm = pyturb.Atmosphere([pyturb.Layer(0.0, 1.0, 0.0, 0.0, L0=25.0)],
                                r0=0.15, n=64, diameter=8.0, tau_boil=tau,
                                dtype="float64", seed=1000 + m)
        frames = [o for _, o in atm.frames(dt=dt, steps=max(lags) + 1)]
        f0 = (frames[0] - frames[0].mean()).ravel()
        den0 += np.dot(f0, f0)
        rms_first.append(frames[0].std())
        rms_last.append(frames[-1].std())
        for k in lags:
            fk = (frames[k] - frames[k].mean()).ravel()
            num[k] += np.dot(f0, fk)
            denk[k] += np.dot(fk, fk)
    for k in lags:
        corr = num[k] / np.sqrt(den0 * denk[k])
        assert abs(corr - np.exp(-k * dt / tau)) < 0.05
    # Boiling preserves the spatial variance (stationary RMS) in the ensemble.
    assert abs(np.mean(rms_last) / np.mean(rms_first) - 1.0) < 0.1


def test_frozen_flow_default_is_not_boiling():
    # Without tau_boil, frames() must be pure frozen flow: the spectra never
    # mutate, so opd(t) is deterministic before and after stepping.
    atm = pyturb.Atmosphere.from_profile("two-layer", r0=0.15, n=64, seed=1)
    before = pyturb.to_numpy(atm.opd(0.05))
    list(atm.frames(dt=1e-3, steps=5))
    after = pyturb.to_numpy(atm.opd(0.05))
    assert np.array_equal(before, after)


def test_invalid_boiling_and_fov():
    layers = pyturb.get_profile("single-layer")
    with pytest.raises(ValueError):
        pyturb.Atmosphere(layers, r0=0.15, tau_boil=-1.0)
    with pytest.raises(ValueError):
        pyturb.Atmosphere(layers, r0=0.15, field_of_view=-5.0)


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


# ---------------------------------------------------------------------------
# extrusion engine (engine="extrude"): non-periodic, arbitrary-direction
# ---------------------------------------------------------------------------
from pyturb.infinite import phase_covariance  # noqa: E402


def test_extrude_frames_shape_units_and_determinism():
    kw = dict(seeing=0.8, diameter=8.0, n=64, engine="extrude", L0=25.0)
    a = pyturb.Atmosphere.from_profile("two-layer", seed=7, **kw)
    b = pyturb.Atmosphere.from_profile("two-layer", seed=7, **kw)
    fa = [np.array(o) for _, o in a.frames(dt=1e-3, steps=4)]
    fb = [np.array(o) for _, o in b.frames(dt=1e-3, steps=4)]
    assert fa[0].shape == (64, 64)
    assert 0 < fa[0].std() < 1e-5              # OPD in metres, micron-scale
    for x, y in zip(fa, fb):                    # same seed -> identical
        np.testing.assert_array_equal(x, y)


def test_extrude_structure_function_matches_total_r0():
    """Summed frozen-flow frames follow the total-r0 von Karman law."""
    kw = dict(seeing=0.8, diameter=8.0, n=64, engine="extrude", L0=25.0,
              dtype="float64")
    acc = None
    for seed in range(40):
        atm = pyturb.Atmosphere.from_profile("paranal-median", seed=seed, **kw)
        phase = atm.opd(0.0, wavelength=500e-9)   # reference-wavelength phase
        r, d = pyturb.structure_function(phase, atm.pixel_scale)
        acc = d if acc is None else acc + d
        r0 = atm.r0
    d = acc / 40
    theory = 2.0 * (phase_covariance(0.0, r0, 25.0) - phase_covariance(r, r0, 25.0))
    mid = (r >= 4 * atm.pixel_scale) & (r <= 8.0 / 4)
    ratio = d[mid] / theory[mid]
    assert np.all(ratio > 0.8) and np.all(ratio < 1.2)


def test_extrude_is_non_periodic_unlike_spectral():
    """The spectral screen repeats after one period; the extruder does not."""
    # subharmonics=0 so the spectral screen is cleanly periodic on the FFT grid
    # (subharmonic modes have longer periods and would spoil the contrast).
    kw = dict(seeing=0.8, diameter=4.0, n=48, seed=1, L0=25.0, subharmonics=0)
    # One layer so the period is well-defined: n*pixel_scale of travel.
    layers = [pyturb.Layer(altitude=0.0, cn2_fraction=1.0, wind_speed=10.0,
                           wind_direction=0.0, L0=25.0)]
    period_t = (4.0 / 48) * 48 / 10.0            # n*dx / wind_speed

    spec = pyturb.Atmosphere(layers, engine="spectral", **kw)
    s0 = pyturb.to_numpy(spec.opd(0.0))
    sT = pyturb.to_numpy(spec.opd(period_t))
    assert np.corrcoef(s0.ravel(), sT.ravel())[0, 1] > 0.99   # periodic

    ext = pyturb.Atmosphere(layers, engine="extrude", **kw)
    e0 = pyturb.to_numpy(ext.opd(0.0))
    eT = pyturb.to_numpy(ext.opd(period_t))
    assert abs(np.corrcoef(e0.ravel(), eT.ravel())[0, 1]) < 0.9  # not periodic


def test_extrude_off_axis_variance_grows_with_angle():
    atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, n=64,
                                         field_of_view=30, engine="extrude", seed=2)
    opds = pyturb.to_numpy(atm.opd(0.0, directions=[(0, 0), (10, 0), (25, 0)]))
    v10 = np.var(opds[1] - opds[0])
    v25 = np.var(opds[2] - opds[0])
    assert v25 > v10 > 0.0


def test_extrude_rejects_boiling():
    with pytest.raises(ValueError):
        pyturb.Atmosphere.from_profile("two-layer", seeing=0.8, n=32,
                                       engine="extrude", tau_boil=0.1)


@pytest.mark.parametrize("engine", ["spectral", "extrude"])
def test_gpu_extrude_and_spectral_match_theory(engine):
    """Both engines, on both devices, follow the total-r0 von Karman law.

    (CuPy and NumPy have independent RNG streams, so the *realisations* differ;
    only the ensemble statistics are comparable — each is checked against
    theory, the repo's standard.)
    """
    try:
        pyturb.get_array_module("gpu")
    except ImportError:
        pytest.skip("CuPy not available")
    kw = dict(seeing=0.8, diameter=8.0, n=64, L0=25.0, engine=engine,
              dtype="float64")
    for dev in ("cpu", "gpu"):
        acc = None
        for seed in range(16):
            atm = pyturb.Atmosphere.from_profile("paranal-median", device=dev,
                                                 seed=seed, **kw)
            r, d = pyturb.structure_function(
                pyturb.to_numpy(atm.opd(0.0, wavelength=500e-9)), atm.pixel_scale)
            acc = d if acc is None else acc + d
            r0 = atm.r0
        d = acc / 16
        theory = 2.0 * (phase_covariance(0.0, r0, 25.0)
                        - phase_covariance(r, r0, 25.0))
        mid = (r >= 4 * atm.pixel_scale) & (r <= 8.0 / 4)
        ratio = d[mid] / theory[mid]
        assert np.all(ratio > 0.8) and np.all(ratio < 1.2), (engine, dev)


def test_lgs_cone_effect_grows_as_beacon_lowers():
    """Focal anisoplanatism (NGS-vs-LGS difference) increases at lower H_LGS."""
    from pyturb.analysis import differential_variance
    kw = dict(seeing=0.8, n=64, engine="extrude", seed=1)
    ngs = pyturb.to_numpy(
        pyturb.Atmosphere.from_profile("paranal-median", **kw).opd(0.0, wavelength=500e-9))
    var = []
    for H in (200e3, 90e3, 30e3):
        lgs = pyturb.Atmosphere.from_profile("paranal-median", lgs_altitude=H, **kw)
        var.append(differential_variance(
            ngs, pyturb.to_numpy(lgs.opd(0.0, wavelength=500e-9))))
    assert 0 < var[0] < var[1] < var[2]


def test_lgs_requires_extruder_and_positive_altitude():
    with pytest.raises(ValueError):
        pyturb.Atmosphere.from_profile("two-layer", seeing=0.8, n=32,
                                       lgs_altitude=90e3)  # spectral engine
    with pytest.raises(ValueError):
        pyturb.Atmosphere.from_profile("two-layer", seeing=0.8, n=32,
                                       engine="extrude", lgs_altitude=-1.0)

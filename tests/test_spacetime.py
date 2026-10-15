"""Space-time covariance theory and the boiling models it describes.

The theory side is checked against closed forms and an independent quadrature;
the simulation side against ensembles of seeded ``Atmosphere.frames`` runs,
with bounds from the seed-to-seed scatter.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import integrate, special

import pyturb
from pyturb import _boiling, theory
from pyturb.theory import _PSD_CONSTANT

L0 = 25.0
R0 = 0.15


def _layers():
    return [
        pyturb.Layer(0.0, 0.6, 8.0, 30.0, L0),
        pyturb.Layer(6e3, 0.4, 20.0, 200.0, 40.0),
    ]


def _grid(n=5, spacing=0.3):
    x = (np.arange(n) - (n - 1) / 2) * spacing
    X, Y = np.meshgrid(x, x)
    return np.stack([X.ravel(), Y.ravel()], axis=-1)


# ---------------------------------------------------------------- theory


def test_zero_lag_is_closed_form_von_karman():
    pts = _grid()
    for tau in (None, 0.3):
        cov = theory.covariance(pts, pts, 0.0, layers=_layers(), r0=R0, tau_boil=tau)
        sep = np.linalg.norm(pts[None, :, :] - pts[:, None, :], axis=-1)
        # Layer i has r0_i = r0 f_i^(-3/5).
        expect = (pyturb.phase_covariance(sep, R0 * 0.6 ** -0.6, L0)
                  + pyturb.phase_covariance(sep, R0 * 0.4 ** -0.6, 40.0))
        np.testing.assert_allclose(cov, expect, rtol=1e-12)


def test_short_lag_boiling_tends_to_frozen_flow():
    pts = _grid()
    frozen = theory.covariance(pts, pts, 1e-9, layers=_layers(), r0=R0)
    for model in _boiling.BOIL_MODELS:
        boiled = theory.covariance(pts, pts, 1e-9, layers=_layers(), r0=R0,
                                   tau_boil=1.0, boil_model=model)
        assert np.abs(boiled - frozen).max() < 1e-6 * frozen.max()


def _quad_covariance(r, lag, tau, model, L0=L0):
    """``2 pi int f PSD rho J0 df`` for r0 = 1 by adaptive quadrature."""
    a = 1.0 / L0 ** 2

    def integrand(f):
        s = lag / tau * max(f * L0, 1.0) ** (2.0 / 3.0)
        rho = float(_boiling.retention(np.float64(s), model))
        return (2 * np.pi * _PSD_CONSTANT * f * (f * f + a) ** (-11 / 6) * rho
                * special.j0(2 * np.pi * f * r))

    edges = [0.0, 1.0 / L0, *np.geomspace(1.0 / L0, 1e3, 120)[1:]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        return sum(integrate.quad(integrand, lo, hi, limit=200)[0]
                   for lo, hi in zip(edges[:-1], edges[1:]))


@pytest.mark.parametrize("model", ["ar1", "ar2"])
def test_boiled_covariance_matches_direct_hankel_quadrature(model):
    layer = [pyturb.Layer(0.0, 1.0, 0.0, 0.0, L0)]
    c0 = pyturb.phase_covariance(0.0, 1.0, L0)
    for lag, tau, r in ((0.02, 1.0, 0.3), (0.1, 0.5, 3.0), (0.4, 0.2, 0.05)):
        got = theory.covariance([0.0, 0.0], [r, 0.0], lag, layers=layer, r0=1.0,
                                tau_boil=tau, boil_model=model)[0, 0]
        assert abs(got - _quad_covariance(r, lag, tau, model)) < 1e-9 * c0


def test_frozen_flow_follows_pyturb_convention():
    # phi(r, t) = phi0(r + v t): phi(b, t + lag) is phi(a, t) for b = a - v lag.
    layer = pyturb.Layer(0.0, 1.0, 10.0, 30.0, L0)
    v = np.array(layer.wind_vector)
    a = np.array([0.2, -0.1])
    lag = 0.03
    sf = theory.space_time_structure_function
    assert sf(a, a - v * lag, lag, layers=[layer], r0=R0)[0, 0] == 0.0
    upwind = sf(a, a + v * lag, lag, layers=[layer], r0=R0)[0, 0]
    expect = theory.structure_function(2 * np.linalg.norm(v) * lag, R0, L0)
    assert upwind == pytest.approx(expect, rel=1e-12)


def test_uniform_scaling_is_frozen_flow_times_retention():
    layer = [pyturb.Layer(0.0, 1.0, 12.0, 70.0, L0)]
    pts = _grid(4)
    frozen = theory.covariance(pts, pts, 0.05, layers=layer, r0=R0)
    for model in _boiling.BOIL_MODELS:
        boiled = theory.covariance(pts, pts, 0.05, layers=layer, r0=R0, tau_boil=0.2,
                                   boil_model=model, boil_scaling="uniform")
        rho = _boiling.retention(0.05 / 0.2, model)
        np.testing.assert_allclose(boiled, rho * frozen, rtol=1e-12)


def test_structure_function_is_twice_covariance_difference():
    pts = _grid(4, 0.5)
    kw = dict(layers=_layers(), r0=R0, tau_boil=[0.4, np.inf])
    lags = np.array([0.0, 0.004, 0.1])
    cov = theory.covariance(pts, pts, lags, **kw)
    sf = theory.space_time_structure_function(pts, pts, lags, **kw)
    assert cov.shape == sf.shape == (3, 16, 16)
    variance = theory.covariance([0.0, 0.0], [0.0, 0.0], **kw)[0, 0]
    np.testing.assert_allclose(sf, 2 * (variance - cov), rtol=1e-9,
                               atol=1e-12 * variance)


def test_boiling_decorrelates_one_point_and_ar2_is_smooth():
    layer = [pyturb.Layer(0.0, 1.0, 0.0, 0.0, L0)]
    sf = theory.space_time_structure_function
    lags = np.array([1e-4, 2e-4, 0.5, 50.0])
    d = {m: sf([0, 0], [0, 0], lags, layers=layer, r0=R0, tau_boil=0.05,
               boil_model=m)[:, 0, 0] for m in _boiling.BOIL_MODELS}
    total = 2 * pyturb.phase_covariance(0.0, R0, L0)
    for model in _boiling.BOIL_MODELS:
        assert d[model][-1] == pytest.approx(total, rel=1e-9)  # fully decorrelated
    # Short lags: rough AR(1) changes faster than smooth AR(2) ...
    assert d["ar2"][0] < 0.5 * d["ar1"][0]
    # ... and its mean-square change grows faster with the lag (smoothness).
    assert d["ar2"][1] / d["ar2"][0] > d["ar1"][1] / d["ar1"][0] * 1.15


def test_many_separations_use_an_accurate_table():
    layer = [pyturb.Layer(0.0, 1.0, 9.0, 10.0, L0)]
    rng = np.random.default_rng(3)
    pts = rng.uniform(-4, 4, size=(40, 2))
    kw = dict(layers=layer, r0=R0, tau_boil=0.3)
    full = theory.covariance(pts, pts, 0.01, **kw)  # 1600 separations: table
    for i, j in ((0, 1), (5, 30), (12, 12), (39, 2)):
        direct = theory.covariance(pts[i], pts[j], 0.01, **kw)[0, 0]
        assert full[i, j] == pytest.approx(direct, rel=1e-6)


def test_units_and_kolmogorov_limits():
    layer = [pyturb.Layer(0.0, 1.0, 5.0, 0.0, np.inf)]
    pts = [[0.0, 0.0], [0.4, 0.0]]
    rad2 = theory.space_time_structure_function(pts, pts, 0.0, layers=layer, r0=R0)
    np.testing.assert_allclose(rad2[0, 1], 6.88 * (0.4 / R0) ** (5 / 3), rtol=1e-12)
    opd = theory.space_time_structure_function(pts, pts, 0.0, layers=layer, r0=R0,
                                               units="opd", wavelength=1e-6)
    np.testing.assert_allclose(opd, rad2 * (1e-6 / (2 * np.pi)) ** 2, rtol=1e-12)
    with pytest.raises(ValueError, match="finite outer scale"):
        theory.covariance(pts, pts, layers=layer, r0=R0)
    with pytest.raises(ValueError, match="L0=inf"):
        theory.space_time_structure_function(pts, pts, 0.1, layers=layer, r0=R0,
                                             tau_boil=1.0)
    with pytest.raises(ValueError, match="boil_model"):
        theory.covariance(pts, pts, layers=_layers(), r0=R0, boil_model="ar3")
    with pytest.raises(ValueError, match="units"):
        theory.covariance(pts, pts, layers=_layers(), r0=R0, units="nm")
    with pytest.raises(ValueError, match="one value per layer"):
        theory.covariance(pts, pts, layers=_layers(), r0=R0, tau_boil=[1.0])
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        theory.covariance([1.0, 2.0, 3.0], pts, layers=_layers(), r0=R0)
    for kw, match in ((dict(boil_scaling="fast"), "boil_scaling"),
                      (dict(r0=-1.0), "r0 must"),
                      (dict(lag=[[0.0]]), "lag must"),
                      (dict(tau_boil=-1.0), "positive"),
                      (dict(L0=0.0), "L0 must")):
        args = dict(layers=_layers(), r0=R0)
        args.update(kw)
        with pytest.raises(ValueError, match=match):
            theory.covariance(pts, pts, **args)


def test_matches_aobasis_von_karman_covariance():
    aobasis = pytest.importorskip("aobasis")
    if not hasattr(aobasis, "von_karman_covariance"):
        pytest.skip("aobasis without von_karman_covariance (added in aobasis 2.3)")
    pts = _grid(4, 0.7)
    layer = [pyturb.Layer(0.0, 1.0, 10.0, 0.0, L0)]
    ours = theory.covariance(pts, pts, layers=layer, r0=R0, units="opd")
    theirs = aobasis.von_karman_covariance(pts, pts, r0=R0, L0=L0, units="opd")
    # aobasis uses the rounded 6.88 constant (pyturb the exact 6.8839...).
    np.testing.assert_allclose(ours, theirs, rtol=1e-3)


# ------------------------------------------------------------ Atmosphere


def test_atmosphere_covariance_uses_its_own_model():
    atm = pyturb.Atmosphere(_layers(), r0=0.12, zenith_angle=30.0, n=16,
                            tau_boil=[0.5, 2.0], boil_model="ar2", seed=1,
                            dispersion="edlen")
    pts = _grid(3)
    got = atm.covariance(pts, pts, [0.0, 0.01])
    expect = theory.covariance(pts, pts, [0.0, 0.01], layers=atm.layers,
                               r0=atm.r0_los, tau_boil=[0.5, 2.0], boil_model="ar2",
                               units="opd", wavelength=atm.wavelength)
    np.testing.assert_allclose(got, expect, rtol=1e-12)
    phase = atm.structure_function(pts, pts, 0.01, wavelength=1.6e-6)
    scale = atm._chromatic_scale(1.6e-6) * 2 * np.pi / 1.6e-6
    np.testing.assert_allclose(phase, atm.structure_function(pts, pts, 0.01) * scale ** 2,
                               rtol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyturb.ExtrudeBoilingPerformanceWarning)
        ext = pyturb.Atmosphere(_layers(), r0=0.12, n=16, tau_boil=0.5,
                                engine="extrude", seed=1)
    uniform = theory.covariance(pts, pts, 0.01, layers=ext.layers, r0=ext.r0_los,
                                tau_boil=0.5, boil_scaling="uniform", units="opd")
    np.testing.assert_allclose(ext.covariance(pts, pts, 0.01), uniform, rtol=1e-12)
    for kw, match in ((dict(power_law=3.5), "von Karman"),
                      (dict(lgs_altitude=90e3), "lgs_altitude")):
        other = pyturb.Atmosphere(_layers(), r0=0.12, n=16, **kw)
        with pytest.raises(ValueError, match=match):
            other.covariance(pts, pts)


# ------------------------------------------------------------ simulation


def _double_difference(frames, shift, lag_frames):
    """``<[phi(a,t) - phi(b,t) - phi(a,t+lag) + phi(b,t+lag)]^2>``, b = a + shift px in x.

    Piston-free: the spectral engine's screen has no mean over the FFT
    screen, so piston-sensitive statistics differ from the infinite-screen
    theory, but this one does not.
    """
    n = frames.shape[-1]
    x = frames[:, :, : n - shift] - frames[:, :, shift:]
    return np.mean((x[:-lag_frames] - x[lag_frames:]) ** 2)


@pytest.mark.parametrize("model", ["ar1", "ar2"])
def test_simulated_space_time_statistics_match_theory(model):
    """Frozen flow + boiling frames against the covariance API, within ensemble error.

    1 m separation on a 6.25 cm grid. The power the grid cannot hold (above
    Nyquist) lowers the simulated statistic by ~1%, except for ``"ar2"`` at
    one frame (~5%): its short-lag change comes mostly from the finest scales,
    which boil fastest.
    """
    n, diameter, dt, tau, shift = 32, 2.0, 0.01, 0.2, 16
    layers = [pyturb.Layer(0.0, 1.0, 5.0, 30.0, 8.0)]
    lag_frames = (1, 5, 20)
    estimates = {k: [] for k in lag_frames}
    for seed in range(12):
        atm = pyturb.Atmosphere(layers, r0=0.1, diameter=diameter, n=n, oversample=4,
                                tau_boil=tau, boil_model=model, seed=seed)
        frames = np.array([pyturb.to_numpy(f) for _, f in atm.frames(dt, 60)],
                          dtype=np.float64)
        for k in lag_frames:
            estimates[k].append(_double_difference(frames, shift, k))
    a = np.array([0.0, 0.0])
    b = np.array([shift * diameter / n, 0.0])
    w = np.array([1.0, -1.0])
    for k in lag_frames:
        # Theory from the covariance matrix of (phi(a), phi(b)) at lags 0 and k dt.
        cov = atm.covariance([a, b], [a, b], [0.0, k * dt])
        expect = 2 * w @ cov[0] @ w - 2 * w @ cov[1] @ w
        sf = atm.structure_function
        via_sf = (2 * sf(a, b, 0)[0, 0] + 2 * sf(a, a, k * dt)[0, 0]
                  - sf(a, b, k * dt)[0, 0] - sf(b, a, k * dt)[0, 0])
        assert via_sf == pytest.approx(expect, rel=1e-8)
        e = np.array(estimates[k])
        error = e.std(ddof=1) / np.sqrt(e.size)
        assert abs(e.mean() - expect) < 4.0 * error + 0.07 * expect, (k, e.mean(), expect)
    # The two models differ by far more than that tolerance at one frame.
    other = "ar2" if model == "ar1" else "ar1"
    alt = pyturb.Atmosphere(layers, r0=0.1, diameter=diameter, n=n, tau_boil=tau,
                            boil_model=other, seed=0)
    alt_cov = alt.covariance([a, b], [a, b], [0.0, dt])
    alt_dd = 2 * w @ alt_cov[0] @ w - 2 * w @ alt_cov[1] @ w
    cov = atm.covariance([a, b], [a, b], [0.0, dt])
    dd = 2 * w @ cov[0] @ w - 2 * w @ cov[1] @ w
    assert abs(alt_dd - dd) > 0.3 * dd


def test_simulated_frozen_flow_matches_convention():
    # One pixel of travel per frame along +x (wind_direction 0).
    n, diameter = 32, 2.0
    dx = diameter / n
    layers = [pyturb.Layer(0.0, 1.0, 5.0, 0.0, 8.0)]
    atm = pyturb.Atmosphere(layers, r0=0.1, diameter=diameter, n=n, oversample=2,
                            dtype="float64", seed=4)
    dt = dx / 5.0
    frames = np.array([pyturb.to_numpy(f) for _, f in atm.frames(dt, 4)])
    m = 3  # frames later, phi(a - v lag) equals phi(a) now
    same = frames[m][:, : n - m] - frames[0][:, m:]
    # Up to piston: each frame's low-frequency part has its screen mean removed.
    assert np.ptp(same) < 1e-9 * np.abs(frames).max()
    a = [0.0, 0.0]
    assert atm.structure_function(a, [-m * dx, 0.0], m * dt)[0, 0] < 1e-20
    assert atm.structure_function(a, [m * dx, 0.0], m * dt)[0, 0] > 1e-16


@pytest.mark.parametrize("model", ["ar1", "ar2"])
def test_boiling_keeps_mode_power_stationary(model, device):
    """Uneven steps keep every mode at its stationary power (spatial PSD preserved)."""
    layers = [pyturb.Layer(0.0, 1.0, 0.0, 0.0, 8.0)]
    atm = pyturb.Atmosphere(layers, r0=0.1, n=64, tau_boil=0.05, boil_model=model,
                            dtype="float64", device=device, seed=2)
    amps = pyturb.to_numpy(atm._amplitudes[0])
    keep = amps > 0
    rng = np.random.default_rng(0)
    power = []
    for dt in rng.uniform(1e-4, 0.03, 60):
        atm.evolve(float(dt))
        spec = pyturb.to_numpy(atm._spectra[0])
        power.append(np.mean(np.abs(spec[keep]) ** 2 / (2 * amps[keep] ** 2)))
    # ~4000 modes per sample: each mean is within a few percent of 1.
    assert abs(np.mean(power) - 1.0) < 0.02
    assert max(abs(p - 1.0) for p in power) < 0.1


def test_ar2_starts_from_the_ar1_screen_and_replays():
    layers = _layers()
    kw = dict(r0=0.12, n=32, tau_boil=0.2, seed=6)
    ar1 = pyturb.Atmosphere(layers, **kw)
    ar2 = pyturb.Atmosphere(layers, boil_model="ar2", **kw)
    f1 = [pyturb.to_numpy(f) for _, f in ar1.frames(1e-3, 3)]
    f2 = [pyturb.to_numpy(f) for _, f in ar2.frames(1e-3, 3)]
    np.testing.assert_array_equal(f1[0], f2[0])
    assert not np.array_equal(f1[2], f2[2])
    ar2.reset()
    np.testing.assert_array_equal(
        f2[2], [pyturb.to_numpy(f) for _, f in ar2.frames(1e-3, 3)][2])
    ar2.reset()
    ar2.evolve(1e-3)
    np.testing.assert_array_equal(f2[2], pyturb.to_numpy(ar2.evolve(1e-3)))
    again = pyturb.Atmosphere.from_config(ar2.to_config())
    assert again.boil_model == "ar2"
    assert ar2.metadata["boil_model"] == "ar2"
    np.testing.assert_array_equal(
        f2[2], [pyturb.to_numpy(f) for _, f in again.frames(1e-3, 3)][2])


def test_ar2_per_frame_change_is_smaller_and_matches_theory():
    """At 1 kHz the smooth model changes the screen far less per frame."""
    layers = [pyturb.Layer(0.0, 1.0, 0.0, 0.0, 8.0)]
    dt, shift = 1e-3, 16
    measured = {}
    for model in _boiling.BOIL_MODELS:
        atm = pyturb.Atmosphere(layers, r0=0.1, diameter=2.0, n=32, oversample=4,
                                tau_boil=0.05, boil_model=model, seed=3)
        frames = np.array([pyturb.to_numpy(f) for _, f in atm.frames(dt, 40)],
                          dtype=np.float64)
        measured[model] = _double_difference(frames, shift, 1)
        a, b = [0.0, 0.0], [shift * 2.0 / 32, 0.0]
        sf = atm.structure_function
        expect = (2 * sf(a, b, 0)[0, 0] + 2 * sf(a, a, dt)[0, 0]
                  - sf(a, b, dt)[0, 0] - sf(b, a, dt)[0, 0])
        assert measured[model] == pytest.approx(expect, rel=0.25)
    assert measured["ar2"] < 0.3 * measured["ar1"]


def test_boil_model_validation():
    with pytest.raises(ValueError, match="boil_model"):
        pyturb.Atmosphere(_layers(), r0=0.1, n=16, boil_model="gauss")
    with pytest.raises(ValueError, match="engine='spectral'"):
        pyturb.Atmosphere(_layers(), r0=0.1, n=16, boil_model="ar2", engine="extrude")


# ------------------------------------------------------------ uniform scaling


@pytest.mark.parametrize("model", ["ar1", "ar2"])
def test_uniform_scaling_gives_every_mode_one_time_constant(model, device):
    layers = [pyturb.Layer(0.0, 0.6, 8.0, 30.0, L0),
              pyturb.Layer(6e3, 0.4, 0.0, 0.0, 40.0)]
    kw = dict(r0=0.12, n=32, tau_boil=[0.3, 1.5], boil_model=model, seed=2,
              device=device)
    uni = pyturb.Atmosphere(layers, boil_scaling="uniform", **kw)
    kol = pyturb.Atmosphere(layers, **kw)
    assert uni.boil_scaling == "uniform" and kol.boil_scaling == "kolmogorov"
    for tau_map in (uni._boil_main_tau, uni._boil_sh_tau):
        tau_map = pyturb.to_numpy(tau_map)
        layer_axis = 0 if tau_map.ndim == 3 else 1
        expect = np.expand_dims(np.array([0.3, 1.5]),
                                tuple(a for a in range(tau_map.ndim) if a != layer_axis))
        np.testing.assert_allclose(tau_map, np.broadcast_to(expect, tau_map.shape),
                                   rtol=1e-6)
    # The Kolmogorov map spans a wide range of time constants on the same grid.
    kol_main = pyturb.to_numpy(kol._boil_main_tau)[0]
    assert kol_main.max() / kol_main.min() > 10.0
    # The per-step update coefficients are then the same for every mode.
    uni.evolve(0.05)
    main = uni._boil_cache[1][0] if model == "ar1" else uni._boil_cache[1][0][0]
    main = pyturb.to_numpy(main)
    for layer in range(2):
        assert np.ptp(main[layer]) <= 1e-6 * abs(main[layer]).max()
    assert abs(main[0].flat[0] - main[1].flat[0]) > 0.01  # per-layer tau_boil


def test_uniform_boiling_decorrelates_every_scale_at_tau_boil():
    """Counterpart of test_boiling_is_scale_dependent_not_uniform: with
    boil_scaling="uniform" the fitted time constant is tau_boil in every band."""
    n, diam, tau_boil, dt, steps = 64, 8.0, 0.10, 5e-3, 250
    layer = pyturb.Layer(0.0, 1.0, wind_speed=0.0)  # no wind: isolate boiling
    atm = pyturb.Atmosphere([layer], r0=0.15, diameter=diam, n=n, seed=0,
                            tau_boil=tau_boil, dtype="float64", subharmonics=0,
                            boil_scaling="uniform")
    frames = np.stack([pyturb.to_numpy(o) for _, o in atm.frames(dt=dt, steps=steps)])
    F = np.fft.fft2(frames, axes=(1, 2))
    f = np.fft.fftfreq(n, diam / n)
    fr = np.hypot(f[:, None], f[None, :])
    lags = np.arange(1, 8)

    def band_tau(lo, hi):
        x = F[:, (fr >= lo) & (fr < hi)]
        ac = [np.real(np.vdot(x[:-k].ravel(), x[k:].ravel()))
              / np.real(np.vdot(x.ravel(), x.ravel())) for k in lags]
        return -1.0 / np.polyfit(lags * dt, np.log(np.clip(ac, 1e-3, 1)), 1)[0]

    # 0.3-4 cycles/m: structure from ~3 m down to the 12.5 cm pixel scale, a
    # range over which the Kolmogorov scaling would change tau ~6x.
    taus = np.array([band_tau(lo, hi) for lo, hi in ((0.3, 0.6), (0.6, 1.2),
                                                       (1.2, 2.4), (2.4, 4.0))])
    assert np.all(np.abs(taus / tau_boil - 1.0) < 0.25), taus
    assert taus.max() / taus.min() < 1.4, taus


@pytest.mark.parametrize("model", ["ar1", "ar2"])
def test_simulated_uniform_boiling_matches_uniform_theory(model):
    """Uniform-boiling spectral frames against ``boil_scaling="uniform"`` theory.

    Same piston-free double difference as the Kolmogorov test above, on a
    static layer so that boiling is all that changes the screen (with wind,
    frozen-flow travel dominates the statistic at multi-pixel lags and both
    scalings predict nearly the same). The two scalings' predictions then
    differ by far more than the tolerance, so this also pins which theory the
    simulation follows; :meth:`Atmosphere.covariance` composing frozen flow
    with uniform boiling is checked against the theory function directly.
    """
    n, diameter, dt, tau, shift = 32, 2.0, 0.01, 0.2, 16
    layers = [pyturb.Layer(0.0, 1.0, 0.0, 0.0, 8.0)]
    lag_frames = (1, 10)
    estimates = {k: [] for k in lag_frames}
    for seed in range(10):
        atm = pyturb.Atmosphere(layers, r0=0.1, diameter=diameter, n=n, oversample=4,
                                tau_boil=tau, boil_model=model, seed=seed,
                                boil_scaling="uniform")
        frames = np.array([pyturb.to_numpy(f) for _, f in atm.frames(dt, 50)],
                          dtype=np.float64)
        for k in lag_frames:
            estimates[k].append(_double_difference(frames, shift, k))
    a = np.array([0.0, 0.0])
    b = np.array([shift * diameter / n, 0.0])
    w = np.array([1.0, -1.0])

    def double_difference(cov):
        return 2 * w @ cov[0] @ w - 2 * w @ cov[1] @ w

    for k in lag_frames:
        cov = atm.covariance([a, b], [a, b], [0.0, k * dt])
        expect = double_difference(cov)
        e = np.array(estimates[k])
        error = e.std(ddof=1) / np.sqrt(e.size)
        assert abs(e.mean() - expect) < 4.0 * error + 0.07 * expect, (k, e.mean(), expect)
        kol = theory.covariance([a, b], [a, b], [0.0, k * dt], layers=atm.layers,
                                r0=atm.r0_los, tau_boil=tau, boil_model=model,
                                units="opd", wavelength=atm.wavelength)
        assert abs(double_difference(kol) - expect) > 0.5 * expect
    windy = pyturb.Atmosphere([pyturb.Layer(0.0, 1.0, 5.0, 30.0, 8.0)], r0=0.1, n=16,
                              tau_boil=tau, boil_model=model, boil_scaling="uniform")
    lags = [0.0, 0.03]
    np.testing.assert_allclose(
        windy.covariance([a, b], [a, b], lags),
        theory.covariance([a, b], [a, b], lags, layers=windy.layers, r0=windy.r0_los,
                          tau_boil=tau, boil_model=model, boil_scaling="uniform",
                          units="opd", wavelength=windy.wavelength),
        rtol=1e-12)


def test_default_boil_scaling_is_the_engines_own():
    kw = dict(r0=0.12, n=16, tau_boil=0.2, seed=3)
    default = pyturb.Atmosphere(_layers(), **kw)
    explicit = pyturb.Atmosphere(_layers(), boil_scaling="kolmogorov", **kw)
    assert default.boil_scaling == "kolmogorov"
    for (_t, x), (_u, y) in zip(default.frames(1e-3, 4), explicit.frames(1e-3, 4)):
        np.testing.assert_array_equal(pyturb.to_numpy(x), pyturb.to_numpy(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyturb.ExtrudeBoilingPerformanceWarning)
        ext = pyturb.Atmosphere(_layers(), engine="extrude", **kw)
        ext_uniform = pyturb.Atmosphere(_layers(), engine="extrude",
                                        boil_scaling="uniform", **kw)
    assert ext.boil_scaling == ext_uniform.boil_scaling == "uniform"
    np.testing.assert_array_equal(pyturb.to_numpy(ext.evolve(1e-3)),
                                  pyturb.to_numpy(ext_uniform.evolve(1e-3)))


def test_boil_scaling_round_trips_and_validates():
    uni = pyturb.Atmosphere(_layers(), r0=0.12, n=16, tau_boil=0.2, seed=3,
                            boil_scaling="uniform")
    config = uni.to_config()
    assert config["boil_scaling"] == uni.metadata["boil_scaling"] == "uniform"
    again = pyturb.Atmosphere.from_config(config)
    assert again.boil_scaling == "uniform"
    np.testing.assert_array_equal(
        [pyturb.to_numpy(f) for _, f in uni.frames(1e-3, 3)],
        [pyturb.to_numpy(f) for _, f in again.frames(1e-3, 3)])
    # A config written before boil_scaling existed takes the engine's default.
    del config["boil_scaling"]
    assert pyturb.Atmosphere.from_config(config).boil_scaling == "kolmogorov"
    with pytest.raises(ValueError, match="boil_scaling must be one of"):
        pyturb.Atmosphere(_layers(), r0=0.1, n=16, boil_scaling="fast")
    with pytest.raises(ValueError, match="per-mode handle"):
        pyturb.Atmosphere(_layers(), r0=0.1, n=16, engine="extrude",
                          boil_scaling="kolmogorov")


def test_travel_decorrelation_time():
    f = theory.travel_decorrelation_time
    # Guesalaga et al. (2014): r = 0.157 v + 0.365 s^-1.
    assert f(0.0) == pytest.approx(1.0 / 0.365, rel=1e-12)
    assert f(10.0) == pytest.approx(1.0 / 1.935, rel=1e-12)
    assert isinstance(f(10.0), float)
    speeds = np.array([[0.0, 5.0, -24.0], [3.0, 12.0, 24.0]])
    tau = f(speeds)
    assert tau.shape == (2, 3)
    np.testing.assert_allclose(tau, 1.0 / (0.157 * np.abs(speeds) + 0.365), rtol=1e-12)
    assert np.all(np.diff(tau[1]) < 0)  # faster wind, faster decorrelation
    assert f(4.0, per_metre=0.5, per_second=1.0) == pytest.approx(1.0 / 3.0)
    assert f(0.0, per_second=0.0) == np.inf  # no decorrelation: frozen flow
    layers = _layers()
    taus = f([layer.wind_speed for layer in layers])
    atm = pyturb.Atmosphere(layers, r0=0.12, n=16, tau_boil=taus,
                            boil_scaling="uniform")
    np.testing.assert_allclose(atm.tau_boil, taus)
    for kw, match in ((dict(per_metre=-0.1), "per_metre"),
                      (dict(per_second=np.nan), "per_second"),
                      (dict(speed=np.inf), "speed")):
        args = dict(speed=1.0)
        args.update(kw)
        with pytest.raises(ValueError, match=match):
            f(**args)

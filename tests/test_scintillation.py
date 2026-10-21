"""Fresnel propagation through the layers, scintillation theory and sparse-time boiling.

The propagation is checked against first-order (weak-perturbation) theory per
Fourier mode, against an independent angular-spectrum loop, against the exact
tilted-plane-wave solution for an off-axis star, and in ensemble against the
grid's own weak-perturbation variance; the theory against Roddier's closed
form and a hand calculation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from aocore import ARCSEC_TO_RAD, AngularSpectrumPropagator
from scipy import special

import pyturb
from pyturb import _boiling, theory
from pyturb.scintillation import propagate_layers
from pyturb.theory import _PSD_CONSTANT


def _independent_chain(opd, altitudes, pitch, lam, tilt=None):
    """Plain NumPy angular-spectrum loop: phase, then exp(-i pi lam dz f^2), top down."""
    n0, n1 = opd.shape[1:]
    f0 = np.fft.fftfreq(n0, pitch)[:, None]
    f1 = np.fft.fftfreq(n1, pitch)[None, :]
    f2 = f0 ** 2 + f1 ** 2
    order = np.argsort(-np.asarray(altitudes), kind="stable")
    ranges = np.append(np.asarray(altitudes, dtype=np.float64)[order], 0.0)
    u = np.ones((n0, n1), complex) if tilt is None else tilt.astype(complex)
    for i, layer in enumerate(order):
        u = u * np.exp(2j * np.pi / lam * opd[layer])
        u = np.fft.ifft2(np.fft.fft2(u) * np.exp(-1j * np.pi * lam
                                               * (ranges[i] - ranges[i + 1]) * f2))
    return u


def _radial_frequency(n, pitch):
    f = np.fft.fftfreq(n, pitch)
    return np.hypot(f[:, None], f[None, :])


# ------------------------------------------------------------- propagation


@pytest.mark.parametrize("altitude, lam", [(5e3, 0.5e-6), (20e3, 0.5e-6), (8e3, 1.6e-6)])
def test_weak_layer_intensity_follows_the_fresnel_transfer(altitude, lam):
    """First order: delta I(f) = 2 k sin(pi lam h f^2) OPD(f), mode by mode."""
    n, pitch = 128, 0.01
    rng = np.random.default_rng(0)
    phase_rms = 0.01  # rad: second-order terms ~1% of the first-order signal
    opd = phase_rms * lam / (2 * np.pi) * rng.standard_normal((1, n, n))
    field = propagate_layers(opd, [altitude], pitch, lam)
    signal = np.fft.fft2(np.abs(field) ** 2 - 1.0)
    first = theory.scintillation_transfer(_radial_frequency(n, pitch), altitude,
                                          lam) * np.fft.fft2(opd[0])
    assert np.linalg.norm(signal - first) < 0.03 * np.linalg.norm(first)
    # The opposite sign convention would give the opposite intensity pattern.
    assert np.linalg.norm(signal + first) > 1.9 * np.linalg.norm(first)


def test_transfer_function_limits():
    T = theory.scintillation_transfer
    lam, h = 0.8e-6, 1e4
    f_fresnel = 1.0 / np.sqrt(lam * h)
    low = np.array([1e-3, 1e-2]) * f_fresnel
    # Transport of intensity, delta I = -h laplacian(OPD): achromatic 4 pi^2 h f^2.
    for wavelength in (lam, 2.2e-6):
        np.testing.assert_allclose(T(low, h, wavelength), 4 * np.pi ** 2 * h * low ** 2,
                                   rtol=1e-4)
    assert np.all(T(np.linspace(0, 50, 11), 0.0, lam) == 0.0)  # no range, no intensity
    assert abs(T(f_fresnel, h, lam)) < 1e-6 * 2 * (2 * np.pi / lam)  # first zero
    assert T(f_fresnel / np.sqrt(2), h, lam) == pytest.approx(2 * 2 * np.pi / lam)
    assert T([[1.0], [2.0]], [0.0, 1e3, 2e3], lam).shape == (2, 3)
    with pytest.raises(ValueError, match="altitude"):
        T(1.0, -1.0, lam)
    with pytest.raises(ValueError, match="wavelength"):
        T(1.0, 1e3, 0.0)


def test_energy_is_conserved_at_any_strength():
    n, pitch = 96, 0.02
    rng = np.random.default_rng(1)
    opd = 2e-6 * rng.standard_normal((3, n, n))  # many waves of phase per pixel
    fields = propagate_layers(opd, [12e3, 3e3, 0.0], pitch, [0.5e-6, 1.0e-6])
    np.testing.assert_allclose(np.mean(np.abs(fields) ** 2, axis=(-2, -1)), 1.0,
                               rtol=0, atol=1e-12)
    single = propagate_layers(opd.astype(np.float32), [12e3, 3e3, 0.0], pitch, 0.5e-6)
    assert single.dtype == np.complex64
    assert abs(np.mean(np.abs(single.astype(np.complex128)) ** 2) - 1.0) < 1e-5


def test_matches_an_independent_angular_spectrum_chain():
    n, pitch, lam = 64, 0.03, 0.7e-6
    rng = np.random.default_rng(2)
    opd = 3e-7 * rng.standard_normal((4, n, n))
    altitudes = [2e3, 9e3, 0.0, 2e3]  # unsorted, with a repeated range
    field = propagate_layers(opd, altitudes, pitch, lam)
    np.testing.assert_allclose(field, _independent_chain(opd, altitudes, pitch, lam),
                               rtol=0, atol=1e-10)
    # One layer: exactly aocore's paraxial angular-spectrum propagator.
    one = propagate_layers(opd[:1], [5e3], pitch, lam)
    prop = AngularSpectrumPropagator((n, n), pitch, lam, 5e3, paraxial=True)
    np.testing.assert_array_equal(one, prop.forward(np.exp(2j * np.pi / lam * opd[0])))


def test_ground_layers_add_no_scintillation():
    n, pitch, lam = 64, 0.02, 0.6e-6
    rng = np.random.default_rng(3)
    high, ground = 2e-7 * rng.standard_normal((2, 1, n, n))
    alone = propagate_layers(high, [10e3], pitch, lam)
    both = propagate_layers(np.concatenate([high, ground]), [10e3, 0.0], pitch, lam)
    np.testing.assert_allclose(np.abs(both), np.abs(alone), rtol=0, atol=1e-12)
    np.testing.assert_allclose(both, alone * np.exp(2j * np.pi / lam * ground[0]),
                               rtol=0, atol=1e-12)
    layers = [pyturb.Layer(0.0, 1.0, 10.0, 30.0)]
    atm = pyturb.Atmosphere(layers, r0=0.05, n=64, diameter=2.0, zenith_angle=40.0,
                            seed=1)
    intensity = pyturb.to_numpy(atm.pupil_field(t=0.1).intensity)
    assert np.max(np.abs(intensity - 1.0)) < 1e-5


def test_geometric_limit_is_the_phase_of_the_opd():
    """Layers at the pupil: the field is exp(i 2 pi OPD / lambda) of Atmosphere.opd."""
    layers = [pyturb.Layer(0.0, 0.6, 7.0, 20.0), pyturb.Layer(0.0, 0.4, 15.0, 160.0)]
    atm = pyturb.Atmosphere(layers, r0=0.1, n=48, diameter=1.5, oversample=2,
                            field_of_view=2.0, dtype="float64", seed=4)
    t, direction, lam = 0.13, (1.2, -0.8), 1.2e-6
    pf = atm.pupil_field(t=t, direction=direction, wavelength=lam)
    opd = atm.opd(t, directions=[direction])[0]
    np.testing.assert_allclose(pf.opd, opd, rtol=0, atol=1e-12 * np.abs(opd).max())
    np.testing.assert_allclose(pf.field, np.exp(2j * np.pi / lam * opd), rtol=0,
                               atol=1e-9)
    np.testing.assert_allclose(pf.intensity, 1.0, rtol=0, atol=1e-12)


def test_pupil_field_follows_the_clock_and_the_opd():
    layers = [pyturb.Layer(0.0, 0.5, 8.0, 0.0), pyturb.Layer(9e3, 0.5, 20.0, 90.0)]
    atm = pyturb.Atmosphere(layers, r0=0.12, n=32, diameter=1.0, field_of_view=3.0,
                            tau_boil=0.5, seed=5)
    frames = atm.frames(2e-3, 3)
    for t, frame in frames:
        pf = atm.pupil_field()  # default t is the current clock
        np.testing.assert_allclose(pf.opd, frame, rtol=0,
                                   atol=1e-5 * np.abs(frame).max())
        np.testing.assert_array_equal(pf.field, atm.pupil_field(t=t).field)
    later = atm.evolve(5e-3)
    np.testing.assert_allclose(atm.pupil_field().opd, later, rtol=0,
                               atol=1e-5 * np.abs(later).max())
    off = atm.pupil_field(direction=(2.0, 1.0), wavelength=[0.6e-6], weights=[1.0])
    expect = atm.opd(atm.time, directions=[(2.0, 1.0)])[0]
    np.testing.assert_allclose(off.opd, expect, rtol=0, atol=1e-5 * np.abs(expect).max())
    assert off.field.shape == (1, 32, 32) and off.field.dtype == np.complex64
    assert off.intensity.dtype == np.float32 and off.opd.dtype == np.float32
    np.testing.assert_allclose(off.intensity, np.abs(off.field[0]) ** 2, rtol=2e-6)
    assert atm.pupil_field().field.shape == (32, 32)


def test_weights_average_the_intensity():
    atm = pyturb.Atmosphere.from_profile("two-layer", r0=0.1, n=32, diameter=1.0,
                                         seed=6)
    waves = [0.5e-6, 0.9e-6, 1.6e-6]
    pf = atm.pupil_field(wavelength=waves, weights=[2.0, 1.0, 1.0])
    power = np.abs(pf.field) ** 2
    np.testing.assert_allclose(pf.intensity, 0.5 * power[0] + 0.25 * power[1]
                               + 0.25 * power[2], rtol=1e-5)
    equal = atm.pupil_field(wavelength=waves)
    np.testing.assert_allclose(equal.intensity, power.mean(axis=0), rtol=1e-5)
    np.testing.assert_array_equal(equal.field, pf.field)


def test_off_axis_field_is_the_tilted_plane_wave():
    """A tilted plane wave through the unshifted layers has the off-axis intensity.

    The tilt is a whole number of cycles over the periodic screen, and the
    layer spectra are band-limited so the direct calculation, whose spectrum is
    shifted by the tilt, does not alias at the grid's Nyquist frequency.
    """
    lam = 1.0e-6
    layers = [pyturb.Layer(8e3, 0.5, 10.0, 30.0), pyturb.Layer(3e3, 0.5, 5.0, 100.0)]
    atm = pyturb.Atmosphere(layers, r0=0.15, n=64, diameter=0.64, subharmonics=0,
                            dtype="float64", field_of_view=2.0, seed=3)
    ns, pitch = atm.n_screen, atm.pixel_scale
    fpix = np.abs(np.fft.fftfreq(ns))
    atm._spectra *= (fpix[:, None] < 0.2) & (fpix[None, :] < 0.2)
    t = 0.01
    main, _low = atm._layer_screen_parts(t, 0.0, 0.0)
    opd = np.asarray(main) * atm.wavelength / (2 * np.pi)
    x = (np.arange(ns) - (ns - 1) / 2) * pitch
    crop = (atm._crop, atm._crop)
    theta = 2 * lam / (ns * pitch)  # two cycles across the screen [rad]
    for axis in (1, 0):
        direction = (theta / ARCSEC_TO_RAD, 0.0) if axis == 1 else (
            0.0, theta / ARCSEC_TO_RAD)
        coord = x[None, :] if axis == 1 else x[:, None]
        intensity = atm.pupil_field(t=t, direction=direction, wavelength=lam).intensity
        assert np.var(intensity) > 0.01
        # pyturb's footprints phi(r + theta h) see the wave exp(-i k theta . r).
        for sign, close in ((-1.0, True), (1.0, False)):
            tilt = np.broadcast_to(np.exp(sign * 2j * np.pi / lam * theta * coord),
                                   (ns, ns))
            direct = np.abs(_independent_chain(opd, atm._alt, pitch, lam, tilt)) ** 2
            error = np.max(np.abs(direct[crop] - intensity))
            if close:
                assert error < 2e-3 * np.std(intensity)
            else:
                assert error > 0.5 * np.std(intensity)


def _grid_weak_variance(atm, waves, weights):
    """Weak-perturbation intensity variance of the simulated grid, mode by mode."""
    fr = _radial_frequency(atm.n_screen, atm.pixel_scale)
    amps = pyturb.to_numpy(atm._amplitudes).astype(np.float64)  # phase, per mode
    w = np.asarray(weights) / np.sum(weights)
    total = 0.0
    for layer, altitude in enumerate(atm._alt):
        transfer = sum(wi * theory.scintillation_transfer(fr, altitude, lam)
                       for wi, lam in zip(w, waves))
        total += np.sum(transfer ** 2 * amps[layer] ** 2)
    return total * (atm.wavelength / (2 * np.pi)) ** 2


def test_scintillation_variance_matches_the_grid_theory(device):
    """Ensemble intensity variance against the grid's per-mode first-order theory.

    Two weighted wavelengths add coherently, mode by mode. Weak turbulence
    (sigma_I^2 ~ 0.01) keeps higher orders below a percent; 16 seeds give a
    standard error of ~1.5%.
    """
    layers = [pyturb.Layer(0.0, 0.5, 5.0, 0.0), pyturb.Layer(4e3, 0.3, 10.0, 40.0),
              pyturb.Layer(12e3, 0.2, 25.0, 200.0)]
    waves, weights = [0.7e-6, 1.0e-6], [1.0, 3.0]
    variances, expect = [], None
    for seed in range(16):
        atm = pyturb.Atmosphere(layers, r0=0.4, n=128, diameter=1.28, zenith_angle=20.0,
                                seed=seed, device=device)
        pf = atm.pupil_field(wavelength=waves, weights=weights)
        intensity = pyturb.to_numpy(pf.intensity).astype(np.float64)
        assert abs(intensity.mean() - 1.0) < 1e-5  # the pupil is the whole screen
        variances.append(intensity.var())
        if expect is None:
            expect = _grid_weak_variance(atm, waves, weights)
    assert 0.005 < expect < 0.02
    assert np.mean(variances) == pytest.approx(expect, rel=0.06)


def test_pupil_field_is_reproducible():
    kw = dict(r0=0.1, n=32, diameter=1.0, tau_boil=1.0, boil_model="ar2", seed=7)
    atm = pyturb.Atmosphere.from_profile("two-layer", **kw)
    atm.advance(0.03)
    first = atm.pupil_field(wavelength=[0.5e-6, 1.0e-6])
    np.testing.assert_array_equal(first.field,
                                  atm.pupil_field(wavelength=[0.5e-6, 1.0e-6]).field)
    atm.reset().advance(0.03)
    np.testing.assert_array_equal(first.field,
                                  atm.pupil_field(wavelength=[0.5e-6, 1.0e-6]).field)
    again = pyturb.Atmosphere.from_config(atm.to_config())
    again.advance(0.03)
    np.testing.assert_array_equal(
        first.intensity, again.pupil_field(wavelength=[0.5e-6, 1.0e-6]).intensity)


def test_pupil_field_validation():
    layers = [pyturb.Layer(0.0, 0.5), pyturb.Layer(5e3, 0.5)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        extrude = pyturb.Atmosphere(layers, r0=0.1, n=16, engine="extrude")
    with pytest.raises(ValueError, match="engine='spectral'"):
        extrude.pupil_field()
    lgs = pyturb.Atmosphere(layers, r0=0.1, n=16, lgs_altitude=90e3)
    with pytest.raises(ValueError, match="spherical wave"):
        lgs.pupil_field()
    with pytest.raises(ValueError, match="spherical wave"):
        lgs.scintillation_index()
    atm = pyturb.Atmosphere(layers, r0=0.1, n=16, field_of_view=1.0)
    for kw, match in ((dict(direction=(0.5, 0.0, 90e3)), "source at infinity"),
                      (dict(direction=(2.0, 0.0)), "field_of_view"),
                      (dict(wavelength=-1e-6), "positive"),
                      (dict(wavelength=[[1e-6]]), "1-D"),
                      (dict(wavelength=[1e-6, 2e-6], weights=[1.0]), "one value per"),
                      (dict(weights=[-1.0]), "non-negative"),
                      (dict(weights=[np.nan]), "non-negative")):
        with pytest.raises(ValueError, match=match):
            atm.pupil_field(**kw)
    opd = np.zeros((2, 8, 8))
    for args, match in (((opd[0], [0.0], 0.1, 1e-6), r"\(L, ny, nx\)"),
                        ((opd + 0j, [0.0, 1.0], 0.1, 1e-6), "real"),
                        ((opd, [0.0], 0.1, 1e-6), "one range per layer"),
                        ((opd, [0.0, -1.0], 0.1, 1e-6), ">= 0"),
                        ((opd, [0.0, 1.0], 0.0, 1e-6), "pixel_scale"),
                        ((opd, [0.0, 1.0], 0.1, np.inf), "positive and finite")):
        with pytest.raises(ValueError, match=match):
            propagate_layers(*args)


@pytest.mark.gpu
def test_gpu_propagation_matches_the_cpu():
    import cupy as cp

    rng = np.random.default_rng(8)
    opd = (3e-7 * rng.standard_normal((3, 64, 64))).astype(np.float32)
    cpu = propagate_layers(opd, [10e3, 4e3, 0.0], 0.02, [0.6e-6, 1.2e-6])
    gpu = propagate_layers(cp.asarray(opd), [10e3, 4e3, 0.0], 0.02, [0.6e-6, 1.2e-6])
    assert isinstance(gpu, cp.ndarray) and gpu.dtype == np.complex64
    np.testing.assert_allclose(cp.asnumpy(gpu), cpu, rtol=0, atol=2e-5)
    # Whole pipeline: the GPU atmosphere given the CPU one's realisation.
    layers = [pyturb.Layer(0.0, 0.5, 8.0, 10.0), pyturb.Layer(9e3, 0.5, 20.0, 100.0)]
    kw = dict(r0=0.1, n=48, diameter=1.5, field_of_view=2.0, seed=9)
    host = pyturb.Atmosphere(layers, **kw)
    device = pyturb.Atmosphere(layers, device="gpu", **kw)
    device._spectra[...] = cp.asarray(host._spectra)
    device._sh_coeffs[...] = cp.asarray(host._sh_coeffs)
    a = host.pupil_field(t=0.02, direction=(1.0, 0.5), wavelength=[0.7e-6, 1.6e-6])
    b = device.pupil_field(t=0.02, direction=(1.0, 0.5), wavelength=[0.7e-6, 1.6e-6])
    np.testing.assert_allclose(cp.asnumpy(b.field), a.field, rtol=0, atol=5e-5)
    np.testing.assert_allclose(cp.asnumpy(b.opd), a.opd, rtol=0,
                               atol=1e-5 * np.abs(a.opd).max())


# ------------------------------------------------------------------ theory


def _roddier_closed_form(r0, lam, altitude):
    """Kolmogorov sigma_I^2 of one layer: int d^2f T(f)^2 PSD_opd(f), exactly.

    ``4 pi C r0^(-5/3) (pi lam h)^(5/6) J`` with ``J = int u^(-11/6) sin^2 u du
    = 2^(5/6) (-Gamma(-5/6) cos(5 pi / 12)) / 2`` and ``r0`` at ``lam``.
    """
    j = 0.5 * 2 ** (5 / 6) * (-special.gamma(-5 / 6) * np.cos(5 * np.pi / 12))
    return (4 * np.pi * _PSD_CONSTANT * r0 ** (-5 / 3)
            * (np.pi * lam * altitude) ** (5 / 6) * j)


def test_roddier_index_is_the_integral_of_the_transfer_function():
    r0, lam, h = 0.12, 0.7e-6, 9e3
    r0_lam = r0 * (lam / 0.5e-6) ** 1.2
    expect = _roddier_closed_form(r0_lam, lam, h)
    index = theory.scintillation_index([pyturb.Layer(h, 1.0)], r0, lam)
    assert index == pytest.approx(expect, rel=2e-3)  # 19.2 and 0.423 are rounded
    # The closed form is the transfer function's integral (checked numerically).
    f = np.geomspace(1e-3, 1e4, 400001)
    psd = _PSD_CONSTANT * r0_lam ** (-5 / 3) * f ** (-11 / 3) * (lam / (2 * np.pi)) ** 2
    integrand = 2 * np.pi * f * theory.scintillation_transfer(f, h, lam) ** 2 * psd
    from pyturb.profiles import _trapezoid

    assert _trapezoid(integrand, f) == pytest.approx(expect, rel=2e-3)


def test_scintillation_index_hand_calculation():
    layers = [pyturb.Layer(0.0, 3.0), pyturb.Layer(4e3, 1.0), pyturb.Layer(16e3, 1.0)]
    r0, lam = 0.1, 0.55e-6
    # By hand: fractions 0.6, 0.2, 0.2; total Cn2 dh = r0^(-5/3) / (0.423 k^2).
    k = 2 * np.pi / 0.5e-6
    cn2 = 0.1 ** (-5 / 3) / (0.423 * k ** 2)  # 6.95e-13 m^(1/3)
    moment = cn2 * (0.2 * 4e3 ** (5 / 6) + 0.2 * 16e3 ** (5 / 6))
    expect = 19.2 * lam ** (-7 / 6) * moment
    assert expect == pytest.approx(0.2246, rel=1e-3)
    assert theory.scintillation_index(layers, r0, lam) == pytest.approx(expect, rel=1e-12)
    # Zenith angle: sec^(11/6); wavelength: lambda^(-7/6); r0 at another wavelength.
    assert theory.scintillation_index(layers, r0, lam, zenith_angle=60.0) == (
        pytest.approx(expect * 2 ** (11 / 6), rel=1e-12))
    many = theory.scintillation_index(layers, r0, [lam, 2 * lam])
    np.testing.assert_allclose(many, expect * np.array([1.0, 2 ** (-7 / 6)]), rtol=1e-12)
    r0_ir = r0 * (1.0e-6 / 0.5e-6) ** 1.2
    assert theory.scintillation_index(layers, r0_ir, lam, r0_wavelength=1e-6) == (
        pytest.approx(expect, rel=1e-12))
    # Atmosphere: zenith r0 at its reference wavelength, its own zenith angle.
    atm = pyturb.Atmosphere(layers, r0=r0, n=16, zenith_angle=30.0)
    assert atm.scintillation_index(lam) == pytest.approx(
        theory.scintillation_index(layers, r0, lam, zenith_angle=30.0), rel=1e-12)
    assert atm.scintillation_index() == pytest.approx(
        theory.scintillation_index(layers, r0, 0.5e-6, zenith_angle=30.0), rel=1e-12)
    dry = pyturb.Atmosphere(layers, r0=r0, n=16, dispersion="edlen")
    ratio = pyturb.air_refractivity(2.2e-6) / pyturb.air_refractivity(0.5e-6)
    assert dry.scintillation_index(2.2e-6) == pytest.approx(
        theory.scintillation_index(layers, r0, 2.2e-6) * ratio ** 2, rel=1e-12)
    with pytest.raises(ValueError, match="Kolmogorov"):
        pyturb.Atmosphere(layers, r0=r0, n=16, power_law=3.5).scintillation_index()
    for args, kw, match in (((layers, -1.0, lam), {}, "r0"),
                            ((layers, r0, 0.0), {}, "wavelength"),
                            ((layers, r0, lam), dict(zenith_angle=90.0), "zenith"),
                            ((layers, r0, lam), dict(r0_wavelength=0.0), "r0_wavelength"),
                            (([pyturb.Layer(-5.0, 1.0)], r0, lam), {}, "altitudes")):
        with pytest.raises(ValueError, match=match):
            theory.scintillation_index(*args, **kw)


def test_fine_grids_approach_the_roddier_index():
    """The grid's first-order variance falls short of Roddier by its missing
    above-Nyquist power: under 1% at a pixel of 1/8 of the Fresnel scale, about
    13% at 1/2 (deterministic: the generator's mode amplitudes, no draws)."""
    lam, h = 0.7e-6, 9e3  # Fresnel scale sqrt(lam h) = 7.9 cm
    layers = [pyturb.Layer(h, 1.0, 0.0)]
    shortfall = {}
    for n in (64, 256):
        atm = pyturb.Atmosphere(layers, r0=0.15, n=n, diameter=2.56, seed=0)
        grid = _grid_weak_variance(atm, [lam], [1.0])
        shortfall[n] = 1.0 - grid / atm.scintillation_index(lam)
    assert 0.0 < shortfall[256] < 0.02
    assert 0.08 < shortfall[64] < 0.2


def test_short_lag_boil_time():
    g = theory.short_lag_boil_time
    assert g(10.0) == pytest.approx(20.0) and g(4.9999) == pytest.approx(20.0, rel=1e-3)
    assert g(2.0) == pytest.approx(3.9) and g(0.0) == g(3.0) == pytest.approx(3.9)
    assert isinstance(g(10.0), float)
    # Linear in 1/tau between 3 and 5 m/s, held outside.
    assert 1 / g(4.0) == pytest.approx(0.5 * (1 / 3.9 + 1 / 20.0))
    assert g(30.0) == pytest.approx(20.0)
    assert np.all(np.diff(g(np.linspace(0, 30, 61))) >= 0)
    tau = g(np.array([[-10.0, 1.0], [4.0, 40.0]]))
    assert tau.shape == (2, 2) and tau[0, 0] == pytest.approx(20.0)
    with pytest.raises(ValueError, match="speed"):
        g([1.0, np.inf])
    # Above 4 m/s the short-lag lifetime is about ten times the GeMS one.
    for v in (5.0, 10.0, 20.0):
        assert 7 < g(v) / theory.gems_boil_time(v) < 30
    # It slows the simulated decorrelation accordingly: over 5 ms the per-frame
    # change of a boiling, static (wind-free) layer falls by about sqrt(b).
    layer = [pyturb.Layer(altitude=0.0, cn2_fraction=1.0, wind_speed=0.0)]
    ch = {}
    for name, tb in (("gems", theory.gems_boil_time(10.0)), ("short", g(10.0))):
        atm = pyturb.Atmosphere(layer, r0=0.1, diameter=8.0, n=64, tau_boil=tb, seed=3)
        a = np.asarray(atm.evolve(0.005), float)
        ch[name] = np.std(np.asarray(atm.evolve(0.005), float) - a)
    assert 0.2 < ch["short"] / ch["gems"] < 0.45   # sqrt(1.44 / 20) = 0.27


def test_gems_boil_time():
    g = theory.gems_boil_time
    assert g(5.0) == pytest.approx(2.5) and g(10.0) == pytest.approx(1.4)
    assert g(20.0) == pytest.approx(0.74)
    assert isinstance(g(10.0), float)
    # Linear in 1/tau between the calibration speeds, held outside them.
    assert 1 / g(7.5) == pytest.approx(0.5 * (1 / 2.5 + 1 / 1.4))
    assert 1 / g(15.0) == pytest.approx(0.5 * (1 / 1.4 + 1 / 0.74))
    assert g(0.0) == g(2.0) == pytest.approx(2.5)
    assert g(35.0) == pytest.approx(0.74)
    speeds = np.array([[-10.0, 6.0], [12.0, 40.0]])
    tau = g(speeds)
    assert tau.shape == (2, 2) and tau[0, 0] == pytest.approx(1.4)
    assert np.all(np.diff(g(np.linspace(0, 30, 61))) <= 0)  # faster wind, shorter
    with pytest.raises(ValueError, match="speed"):
        g([1.0, np.nan])
    layers = pyturb.get_profile("two-layer")
    taus = g([layer.wind_speed for layer in layers])
    atm = pyturb.Atmosphere(layers, r0=0.12, n=16, tau_boil=taus)
    np.testing.assert_allclose(atm.tau_boil, taus)
    assert atm.boil_scaling == "kolmogorov"


# ----------------------------------------------------------- sparse time


@pytest.mark.parametrize("engine, model", [("spectral", "ar1"), ("spectral", "ar2"),
                                           ("extrude", "ar1")])
def test_advance_steps_exactly_as_frames_and_evolve(engine, model):
    layers = [pyturb.Layer(0.0, 0.6, 6.0, 10.0), pyturb.Layer(7e3, 0.4, 18.0, 250.0)]
    kw = dict(r0=0.12, n=32, diameter=2.0, tau_boil=[0.4, 0.9], engine=engine,
              boil_model=model, seed=12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyturb.ExtrudeBoilingPerformanceWarning)
        a = pyturb.Atmosphere(layers, **kw)
        b = pyturb.Atmosphere(layers, **kw)
        c = pyturb.Atmosphere(layers, **kw)
    dt = 2e-3
    reference = [pyturb.to_numpy(f) for _, f in a.frames(dt, 4)]
    stepped = [pyturb.to_numpy(b.opd(t=b.time))]
    evolved = [pyturb.to_numpy(c.opd(t=c.time))]
    for _ in range(3):
        assert b.advance(dt) is b
        stepped.append(pyturb.to_numpy(b.opd(t=b.time)))
        evolved.append(pyturb.to_numpy(c.evolve(dt)))
    np.testing.assert_array_equal(stepped, reference)
    np.testing.assert_array_equal(evolved, reference)
    assert b.time == pytest.approx(3 * dt)
    for bad in (0.0, -1e-3, np.nan, np.inf):
        with pytest.raises(ValueError, match="positive"):
            b.advance(bad)


def test_one_jump_is_the_s_step_process():
    """An AR(1) sampled every s steps is an AR(1) with rho^s and 1 - rho^(2s)."""
    layers = [pyturb.Layer(0.0, 0.5, 0.0, 0.0), pyturb.Layer(5e3, 0.5, 0.0, 0.0, 40.0)]
    atm = pyturb.Atmosphere(layers, r0=0.1, n=32, tau_boil=[0.05, 0.3], dtype="float64",
                            seed=1)
    T, s = 1e-3, 30
    a1, _b1, (a1_sh, _b1_sh) = atm._boil_coeffs(T)
    a_s, b_s, (a_s_sh, b_s_sh) = atm._boil_coeffs(s * T)
    np.testing.assert_allclose(a_s, a1 ** s, rtol=1e-12)
    np.testing.assert_allclose(b_s ** 2, 1.0 - a1 ** (2 * s), rtol=1e-12)
    np.testing.assert_allclose(a_s_sh, a1_sh ** s, rtol=1e-12)
    np.testing.assert_allclose(b_s_sh ** 2, 1.0 - a1_sh ** (2 * s), rtol=1e-12)
    # AR(2): the transition over s T is the s-th power of the one over T, and
    # its innovation covariance is the stationary remainder I - F F^T.
    u = 2 * T / np.array([0.05, 0.3, 2.0])
    f11, f12, f22, l11, l21, l22 = _boiling.ar2_coefficients(u)
    f11s, f12s, f22s, l11s, l21s, l22s = _boiling.ar2_coefficients(s * u)
    for i in range(u.size):
        one = np.array([[f11[i], f12[i]], [-f12[i], f22[i]]])
        many = np.array([[f11s[i], f12s[i]], [-f12s[i], f22s[i]]])
        np.testing.assert_allclose(many, np.linalg.matrix_power(one, s), rtol=1e-10,
                                   atol=1e-14)
        chol = np.array([[l11s[i], 0.0], [l21s[i], l22s[i]]])
        np.testing.assert_allclose(chol @ chol.T, np.eye(2) - many @ many.T,
                                   rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("model", ["ar1", "ar2"])
def test_frames_s_steps_apart_are_correlated_by_the_retention(model, device):
    """Ensemble: one jump of s T and s steps of T both leave correlation R(s T / tau).

    Measured on the stored Fourier modes, each normalised by its amplitude:
    with uniform scaling every mode has the same retention, and the ~1000
    modes per screen are independent samples of it.
    """
    layer = [pyturb.Layer(0.0, 1.0, 0.0, 0.0)]  # static: boiling alone
    tau, T, s = 0.1, 1e-3, 50
    expect = float(_boiling.retention(s * T / tau, model))
    sums = {"jump": np.zeros(3), "steps": np.zeros(3)}
    for seed in range(6):
        for how in sums:
            atm = pyturb.Atmosphere(layer, r0=0.1, n=32, tau_boil=tau,
                                    boil_scaling="uniform", boil_model=model,
                                    dtype="float64", seed=seed, device=device)
            amps = pyturb.to_numpy(atm._amplitudes[0])
            keep = amps > 0
            first = pyturb.to_numpy(atm._spectra[0])[keep] / amps[keep]
            for _ in range(1 if how == "jump" else s):
                atm.advance(s * T if how == "jump" else T)
            later = pyturb.to_numpy(atm._spectra[0])[keep] / amps[keep]
            sums[how] += [np.real(np.vdot(first, later)), np.vdot(first, first).real,
                          np.vdot(later, later).real]
    for how, (cross, var0, var1) in sums.items():
        assert cross / np.sqrt(var0 * var1) == pytest.approx(expect, abs=0.03), how
        assert var1 / var0 == pytest.approx(1.0, abs=0.06), how  # stationary

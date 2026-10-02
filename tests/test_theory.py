import numpy as np
import pytest

import pyturb
from pyturb import analysis as A
from pyturb import theory as T

D, R0 = 8.0, 0.15


@pytest.mark.parametrize("j", [2, 3])
def test_kolmogorov_tilt_variance_matches_noll(j):
    assert T.zernike_variance(j, D, R0) == pytest.approx(A.noll_variance(j, D, R0),
                                                         rel=0.01)


def test_kolmogorov_higher_modes_match_noll_within_table_rounding():
    # Noll's Delta_j table has three significant figures, so per-mode
    # differences of small Deltas carry a few percent of rounding.
    for j in range(4, 12):
        assert T.zernike_variance(j, D, R0) == pytest.approx(
            A.noll_variance(j, D, R0), rel=0.03)


def test_image_motion_is_the_textbook_coefficient():
    # sigma^2 = 0.182 (D/r0)^{5/3} (lambda/D)^2 per axis for Kolmogorov.
    lam = 500e-9
    unit = (lam / D * 180 / np.pi * 3600) ** 2 * (D / R0) ** (5 / 3)
    assert T.image_motion_variance(D, R0, lam) / unit == pytest.approx(0.182, rel=0.01)


def test_finite_outer_scale_reduces_low_orders_most():
    tilt = T.zernike_variance(2, D, R0, 25.0) / T.zernike_variance(2, D, R0)
    high = T.zernike_variance(11, D, R0, 25.0) / T.zernike_variance(11, D, R0)
    assert tilt < 0.2 < 0.85 < high < 1.0


def test_finite_outer_scale_tilt_matches_simulation():
    n, L0 = 64, 25.0
    atm = pyturb.Atmosphere([pyturb.Layer(0.0, 1.0, 10.0, 0.0, L0=L0)], r0=R0,
                            diameter=D, n=n, oversample=4, seed=3, dtype="float64")
    coeffs = A.zernike_decompose(atm.sample(400, wavelength=atm.wavelength), 3,
                                 A.zernike_basis(3, n))
    for j in (2, 3):
        ratio = coeffs[:, j - 1].var() / T.zernike_variance(j, D, R0, L0)
        assert 0.88 < ratio < 1.12


@pytest.mark.parametrize("j, direction, L0", [(2, 0.0, 25.0), (3, 0.0, 25.0),
                                              (4, 30.0, np.inf), (7, 90.0, 25.0)])
def test_temporal_psd_integrates_to_the_mode_variance(j, direction, L0):
    freq = np.geomspace(1e-3, 5e3, 2000)
    psd = T.zernike_temporal_psd(freq, j, D, R0, 10.0, direction, L0)
    integral = np.sum(0.5 * (psd[1:] + psd[:-1]) * np.diff(freq))
    assert integral == pytest.approx(T.zernike_variance(j, D, R0, L0), rel=0.03)


def test_along_wind_tilt_spectrum_matches_simulation():
    # Frozen flow along axis 0, so Noll j=2 (cos, axis-0 tilt) is along the
    # wind. 96 px across the pupil keeps discretisation leakage into the
    # fitted tilt small over 2-20 Hz.
    n, v, dt, L0 = 96, 10.0, 2e-3, 25.0
    basis = A.zernike_basis(2, n)
    psds = []
    for seed in range(2):
        atm = pyturb.Atmosphere([pyturb.Layer(0.0, 1.0, v, 0.0, L0=L0)], r0=R0,
                                diameter=D, n=n, oversample=4, seed=seed,
                                dtype="float64")
        frames = pyturb.to_numpy(atm.opd(t=np.arange(1000) * dt,
                                         wavelength=atm.wavelength))
        freq, p = A.temporal_psd(A.zernike_decompose(frames, 2, basis)[:, 1], dt)
        psds.append(p)
    measured = np.mean(psds, axis=0)
    theory = T.zernike_temporal_psd(freq, 2, D, R0, v, 0.0, L0)
    band = (freq >= 2) & (freq <= 20)
    assert 0.7 < np.median(measured[band] / theory[band]) < 1.4


def test_seeing_fwhm_outer_scale_correction():
    kolmogorov = T.seeing_fwhm(0.1)
    assert kolmogorov == pytest.approx(pyturb.seeing_from_r0(0.1))
    assert T.seeing_fwhm(0.1, L0=25.0) / kolmogorov == pytest.approx(0.833, abs=0.002)
    with pytest.raises(ValueError, match="L0/r0 > 20"):
        T.seeing_fwhm(0.5, L0=5.0)


def test_differential_phase_variance_is_theta_over_theta0():
    layers = pyturb.get_profile("paranal-median")
    theta0 = pyturb.isoplanatic_angle(layers, R0) * 180 / np.pi * 3600  # arcsec
    angles = np.array([0.5, 1.0, 2.0]) * theta0
    var = T.differential_phase_variance(angles, layers, R0, L0=np.inf)
    np.testing.assert_allclose(var, (angles / theta0) ** (5 / 3), rtol=0.005)
    finite = T.differential_phase_variance(angles, layers, R0)  # per-layer L0=25 m
    assert np.all(finite < var)


def test_structure_function_limits():
    r = np.array([0.01, 0.1, 1.0])
    np.testing.assert_allclose(T.structure_function(r, R0), 6.88 * (r / R0) ** (5 / 3))
    small = T.structure_function(1e-3, R0, 1e4)
    assert small == pytest.approx(6.88 * (1e-3 / R0) ** (5 / 3), rel=0.01)

"""Tests for the analysis utilities (Zernikes, Noll, temporal PSD, angular)."""

import numpy as np
import pytest

import pyturb
from pyturb import analysis as A


def test_noll_to_zernike_mapping():
    expected = {1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0), 5: (2, -2),
                6: (2, 2), 7: (3, -1), 8: (3, 1), 11: (4, 0)}
    for j, nm in expected.items():
        assert A.noll_to_zernike(j) == nm


def test_basis_orthonormal_over_pupil():
    basis = A.zernike_basis(15, 256)
    mask = basis[0] != 0
    vecs = basis[:, mask]
    gram = (vecs @ vecs.T) / mask.sum()
    assert np.allclose(np.diag(gram), 1.0, atol=0.02)
    off = gram - np.diag(np.diag(gram))
    assert np.abs(off).max() < 0.03


def test_basis_matches_closed_form_low_orders():
    # Independent closed forms (Noll 1976) on an off-centre-sized pupil:
    # x = axis 1, y = axis 0, unit radius at diameter_pixels / 2.
    n, diameter = 40, 33.0
    basis = A.zernike_basis(11, n, diameter_pixels=diameter)
    c = (np.arange(n) - (n - 1) / 2.0) / (diameter / 2.0)
    y, x = np.meshgrid(c, c, indexing="ij")
    r2 = x**2 + y**2
    mask = r2 <= 1.0
    expected = {
        1: np.ones_like(x),
        2: 2.0 * x,
        3: 2.0 * y,
        4: np.sqrt(3.0) * (2.0 * r2 - 1.0),
        5: 2.0 * np.sqrt(6.0) * x * y,
        6: np.sqrt(6.0) * (x**2 - y**2),
        11: np.sqrt(5.0) * (6.0 * r2**2 - 6.0 * r2 + 1.0),
    }
    np.testing.assert_array_equal(basis[0] != 0, mask)
    for j, z in expected.items():
        np.testing.assert_allclose(basis[j - 1][mask], z[mask], rtol=0, atol=1e-12)
        assert np.all(basis[j - 1][~mask] == 0.0)


def test_basis_rejects_degenerate_requests():
    with pytest.raises(ValueError, match="diameter_pixels must be positive"):
        A.zernike_basis(3, 16, diameter_pixels=0.0)
    # A 2-pixel-wide pupil holds 4 pixel centres: 5 modes cannot be independent.
    with pytest.raises(ValueError, match="exceeds the 4 pixels inside the pupil"):
        A.zernike_basis(5, 16, diameter_pixels=2.0)


def test_decompose_recovers_known_coefficients():
    basis = A.zernike_basis(10, 128)
    truth = np.zeros(10)
    truth[[3, 5, 8]] = [1.7, -0.9, 0.4]
    phase = np.tensordot(truth, basis, axes=1)
    rec = A.zernike_decompose(phase, 10, basis=basis)
    np.testing.assert_allclose(rec, truth, atol=1e-9)


def test_decompose_stack_shape():
    gen = pyturb.PhaseScreen(n=64, pixel_scale=0.05, r0=0.15, seed=0)
    coeffs = A.zernike_decompose(gen.generate(5), 8)
    assert coeffs.shape == (5, 8)


def test_zernike_variances_match_noll():
    """Kolmogorov Zernike-mode variances follow Noll (1976).

    Per-mode variances scatter (finite screen under-samples tip/tilt; the
    square grid splits the two astigmatism modes), so the aggregate over a
    band of modes is the robust check.
    """
    D, n, r0 = 4.0, 128, 0.4  # D/r0 = 10
    gen = pyturb.PhaseScreen(n=n, pixel_scale=D / n, r0=r0, L0=np.inf,
                             seed=7, dtype="float64")
    basis = A.zernike_basis(15, n)
    coeffs = A.zernike_decompose(gen.generate(800), 15, basis=basis)
    measured = coeffs.var(axis=0)

    mid_meas = measured[3:15].sum()                     # modes j = 4..15
    mid_noll = sum(A.noll_variance(j, D, r0) for j in range(4, 16))
    assert 0.85 < mid_meas / mid_noll < 1.2
    # Astigmatism pair (j=5,6) averages to Noll despite the per-mode split.
    astig = 0.5 * (measured[4] + measured[5]) / A.noll_variance(5, D, r0)
    assert 0.8 < astig < 1.25


def test_noll_residual_decreases_and_matches_table():
    r = [A.noll_residual_variance(j, 8.0, 0.2) for j in range(1, 20)]
    assert all(np.diff(r) < 0)                          # correcting more helps
    # Delta_1 = 1.0299 (D/r0)^{5/3}: total minus piston.
    assert A.noll_residual_variance(1, 8.0, 0.2) == pytest.approx(
        1.0299 * (8.0 / 0.2) ** (5 / 3), rel=1e-6)
    with pytest.raises(ValueError):
        A.noll_variance(1, 8.0, 0.2)                    # piston undefined


def test_fit_power_law_recovers_synthetic_slope():
    freq = np.logspace(0, 3, 400)
    psd = 3.0 * freq ** (-2.6)
    slope, amp = A.fit_power_law(freq, psd, fmin=2, fmax=500)
    assert slope == pytest.approx(-2.6, abs=1e-6)
    assert amp == pytest.approx(3.0, rel=1e-6)


@pytest.mark.filterwarnings("ignore::pyturb.PeriodicWrapWarning")
def test_temporal_psd_frozen_flow_slope():
    """Pupil points under frozen flow show the ~ -8/3 power law.

    2048 ms of 10 m/s wind on a 4 m screen wraps the periodic spectral engine
    ~5 times; harmless here since the fit only uses 5-60 Hz, well away from
    the wrap-induced line at v/D ~ 2.5 Hz, but real enough to trigger
    PeriodicWrapWarning -- suppressed here as intentional.
    """
    layers = [pyturb.Layer(0.0, 1.0, wind_speed=10.0, wind_direction=0.0, L0=100.0)]
    atm = pyturb.Atmosphere(layers, r0=0.15, n=64, diameter=4.0, seed=3,
                            subharmonics=6)
    # Wind along x: seven pixels spread across the wind (down one column) are
    # averaged, since a single pixel's slope scatters by +-0.5 between seeds.
    series = np.array([np.array(o)[8:57:8, 32]
                       for _, o in atm.frames(dt=1e-3, steps=2048)]).T
    freq, psd = A.temporal_psd(series, 1e-3)
    slope, _ = A.fit_power_law(freq, psd, fmin=5, fmax=60)
    assert -3.3 < slope < -2.2                          # brackets -8/3


def _power_law_record(n, slope, seed):
    """A stationary, non-periodic record with PSD ~ f**slope.

    Synthesised on a 16x longer periodic grid and cropped, so the record's ends
    do not match -- the realistic case in which an untapered periodogram leaks.
    """
    rng = np.random.default_rng(seed)
    m = 16 * n
    f = np.fft.rfftfreq(m, d=1.0)
    amp = np.zeros_like(f)
    amp[1:] = f[1:] ** (slope / 2.0)
    spec = amp * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
    return np.fft.irfft(spec, n=m)[:n]


def test_temporal_psd_window_recovers_steep_power_law():
    # A steep f^{-8/3} record: the default Hann taper recovers the slope, the
    # untapered periodogram is flattened by leakage (the bias #6 describes).
    records = np.stack([_power_law_record(1024, -8.0 / 3.0, seed) for seed in range(16)])
    freq, hann = A.temporal_psd(records, 1e-3)
    _, rect = A.temporal_psd(records, 1e-3, window=None)
    slope_hann, _ = A.fit_power_law(freq, hann, fmin=20, fmax=300)
    slope_rect, _ = A.fit_power_law(freq, rect, fmin=20, fmax=300)
    assert abs(slope_hann - (-8.0 / 3.0)) < 0.15
    assert slope_rect > slope_hann + 0.3


@pytest.mark.parametrize("window", ["hann", None, "blackman"])
def test_temporal_psd_integrates_to_the_variance(window):
    rng = np.random.default_rng(0)
    series = rng.standard_normal((64, 2048)) * 3.0           # white, variance 9
    freq, psd = A.temporal_psd(series, 1e-3, window=window)
    df = freq[1] - freq[0]
    assert np.sum(psd) * df == pytest.approx(9.0, rel=0.03)


def test_temporal_psd_rejects_wrong_length_window():
    with pytest.raises(ValueError, match="length 100"):
        A.temporal_psd(np.zeros(100), 1e-3, window=np.ones(50))


def test_differential_variance_grows_with_angle():
    atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, n=64,
                                         field_of_view=30, seed=1)
    opds = pyturb.to_numpy(atm.opd(0.0, directions=[(0, 0), (10, 0), (25, 0)],
                                   wavelength=500e-9))
    v10 = A.differential_variance(opds[0], opds[1])
    v25 = A.differential_variance(opds[0], opds[2])
    assert 0.0 < v10 < v25

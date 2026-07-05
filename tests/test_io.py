"""Tests for screen I/O, chromatic OPD, and moment-conserving compression."""

import numpy as np
import pytest

import pyturb
from pyturb import profiles as P

# np.trapezoid requires NumPy >= 2.0; fall back to np.trapz on the
# numpy>=1.22 floor declared in pyproject.toml.
_trapezoid = getattr(np, "trapezoid", np.trapz)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ext", ["npz", "fits"])
def test_save_load_roundtrip(tmp_path, ext):
    if ext == "fits":
        pytest.importorskip("astropy")
    atm = pyturb.Atmosphere.from_profile("two-layer", seeing=0.8, n=48, seed=1)
    opd = pyturb.to_numpy(atm.opd(0.0))
    path = tmp_path / f"screen.{ext}"
    pyturb.save(path, opd, **atm.metadata)
    data, meta = pyturb.load(path)

    np.testing.assert_allclose(data, opd, rtol=1e-6)
    assert data.dtype.byteorder in ("=", "|")            # native byte order
    assert meta["units"] == "metres"
    assert meta["r0"] == pytest.approx(atm.r0, rel=1e-4)
    assert meta["pixel_scale"] == pytest.approx(atm.pixel_scale, rel=1e-6)
    assert meta["pyturb_version"] == pyturb.__version__


def test_save_load_stack_and_none_dropped(tmp_path):
    gen = pyturb.PhaseScreen(n=32, pixel_scale=0.02, r0=0.15, seed=0)
    screens = gen.generate(4)
    path = tmp_path / "cube.npz"
    pyturb.save(path, screens, pixel_scale=0.02, r0=0.15, seed=None, units="radians")
    data, meta = pyturb.load(path)
    np.testing.assert_array_equal(data, screens)
    assert data.shape == (4, 32, 32)
    assert "seed" not in meta                              # None values dropped


# ---------------------------------------------------------------------------
# chromatic OPD (dispersion)
# ---------------------------------------------------------------------------
def test_air_refractivity_decreases_with_wavelength():
    n_v = pyturb.air_refractivity(500e-9)
    n_k = pyturb.air_refractivity(2200e-9)
    assert 2.7e-4 < n_v < 2.9e-4                           # standard dry air
    assert n_k < n_v                                       # normal dispersion


def test_dispersion_scales_phase_but_not_opd():
    kw = dict(seeing=0.8, n=32, seed=1)
    achrom = pyturb.Atmosphere.from_profile("two-layer", **kw)
    edlen = pyturb.Atmosphere.from_profile("two-layer", dispersion="edlen", **kw)

    # Metre-valued OPD is identical (dispersion only affects phase output).
    np.testing.assert_array_equal(
        pyturb.to_numpy(achrom.opd(0.0)), pyturb.to_numpy(edlen.opd(0.0)))

    lam = 2200e-9
    pa = pyturb.to_numpy(achrom.opd(0.0, wavelength=lam))
    pe = pyturb.to_numpy(edlen.opd(0.0, wavelength=lam))
    ratio = (pyturb.air_refractivity(lam) / pyturb.air_refractivity(achrom.wavelength))
    np.testing.assert_allclose(pe, pa * ratio, rtol=1e-5)
    assert abs(ratio - 1) > 0.01                           # a real ~2% effect

    # At the reference wavelength the correction is exactly unity.
    np.testing.assert_allclose(
        pyturb.to_numpy(edlen.opd(0.0, wavelength=achrom.wavelength)),
        pyturb.to_numpy(achrom.opd(0.0, wavelength=achrom.wavelength)), rtol=1e-6)


def test_dispersion_validation():
    with pytest.raises(ValueError):
        pyturb.Atmosphere.from_profile("two-layer", seeing=0.8, n=16,
                                       dispersion="bogus")


# ---------------------------------------------------------------------------
# moment-conserving profile compression
# ---------------------------------------------------------------------------
def test_equivalent_layers_conserve_theta0_and_tau0():
    h = np.linspace(0.0, 20000.0, 4000)
    cn2 = pyturb.hufnagel_valley(h)
    wind = pyturb.bufton_wind(h)

    def moment(values):
        return (_trapezoid(cn2 * values ** (5 / 3), h)
                / _trapezoid(cn2, h)) ** (3 / 5)

    hbar_true, vbar_true = moment(h), moment(wind)

    for n_layers in (4, 6, 10):
        layers = pyturb.discretize_cn2(h, cn2, n_layers=n_layers, wind=wind,
                                       method="equivalent")
        assert sum(ly.cn2_fraction for ly in layers) == pytest.approx(1.0)
        assert P.mean_turbulence_height(layers) == pytest.approx(hbar_true, rel=1e-3)
        assert P.effective_wind_speed(layers) == pytest.approx(vbar_true, rel=1e-3)


def test_centroid_method_does_not_conserve_5_3_moment():
    h = np.linspace(0.0, 20000.0, 4000)
    cn2 = pyturb.hufnagel_valley(h)
    hbar_true = (_trapezoid(cn2 * h ** (5 / 3), h)
                 / _trapezoid(cn2, h)) ** (3 / 5)
    layers = pyturb.discretize_cn2(h, cn2, n_layers=4, method="centroid")
    # Centroid heights are lower than the 5/3-moment heights, so h_bar is off.
    assert abs(P.mean_turbulence_height(layers) / hbar_true - 1) > 0.05


def test_discretize_rejects_bad_method():
    with pytest.raises(ValueError):
        pyturb.discretize_cn2([0.0, 1.0], [1.0, 1.0], method="nope")

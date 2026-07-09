import numpy as np
import pytest

import pyturb
from pyturb import profiles

# np.trapezoid (NumPy >= 2.0) replaces np.trapz, later removed entirely.
# hasattr, not getattr's default, so np.trapz is never accessed when absent.
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def test_named_profiles_load_and_are_nonempty():
    for name in pyturb.list_profiles():
        layers = pyturb.get_profile(name)
        assert len(layers) >= 1
        assert all(layer.cn2_fraction >= 0 for layer in layers)


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        pyturb.get_profile("does-not-exist")
    with pytest.raises(ValueError):
        pyturb.profile_info("does-not-exist")


def test_every_profile_has_provenance():
    """Each named profile must carry a serialisable provenance record, and the
    traceable/representative distinction must match reality."""
    for name in pyturb.list_profiles():
        info = pyturb.profile_info(name)
        assert info.name == name
        assert isinstance(info.source, str) and info.source
        assert isinstance(info.caveat, str) and info.caveat
        assert info.wind_direction_measured is False  # no source tabulates it
    # Cited-table profiles are traceable; representative/teaching ones are not.
    assert pyturb.profile_info("mauna-kea").traceable
    assert pyturb.profile_info("keck").traceable
    assert pyturb.profile_info("las-campanas").traceable
    assert not pyturb.profile_info("paranal-median").traceable
    assert not pyturb.profile_info("cerro-pachon").traceable
    assert not pyturb.profile_info("two-layer").traceable


def test_metadata_records_profile_provenance():
    """from_profile records the profile name + provenance in metadata; a direct
    build (no named profile) reports None for those fields."""
    atm = pyturb.Atmosphere.from_profile("mauna-kea", seeing=0.8, n=32, seed=1)
    md = atm.metadata
    assert md["profile"] == "mauna-kea"
    assert md["profile_traceable"] is True
    assert "Guyon" in md["profile_source"]
    assert md["profile_site"] == "Mauna Kea"

    direct = pyturb.Atmosphere(pyturb.get_profile("two-layer"), seeing=0.8, n=32)
    assert direct.metadata["profile"] is None
    assert direct.metadata["profile_source"] is None


@pytest.mark.parametrize("name", ["cerro-pachon", "armazones"])
def test_site_profiles_are_physically_sane(name):
    """New site profiles: ground-layer-dominated, normalisable, and giving
    integrated quantities in the range expected for a good 8-m-class site."""
    assert name in pyturb.list_profiles()
    layers = pyturb.get_profile(name)
    assert len(layers) >= 6
    fracs = np.array([ly.cn2_fraction for ly in layers])
    assert np.all(fracs >= 0) and fracs.sum() > 0
    # Ground layer carries the most turbulence (both sites are ground-dominated).
    assert layers[0].altitude == 0.0
    assert np.argmax(fracs) == 0 and fracs[0] > 0.25
    # Build an atmosphere and check theta0/tau0 land in a sensible band at
    # 500 nm for r0 ~ 15 cm (a few arcsec seeing regime).
    atm = pyturb.Atmosphere.from_profile(name, r0=0.15, n=32)
    assert 0.5 < atm.theta0 < 8.0        # isoplanatic angle [arcsec]
    assert 1e-3 < atm.tau0 < 20e-3       # coherence time [s]


def test_layer_wind_vector():
    layer = pyturb.Layer(altitude=0.0, cn2_fraction=1.0, wind_speed=10.0,
                         wind_direction=90.0)
    vx, vy = layer.wind_vector
    assert abs(vx) < 1e-9
    assert abs(vy - 10.0) < 1e-9


def test_hufnagel_valley_57_gives_reasonable_r0():
    # HV 5/7 famously gives r0 ~ 5 cm at 500 nm.
    h = np.geomspace(1.0, 25000.0, 20000)
    cn2 = pyturb.hufnagel_valley(h)
    k = 2 * np.pi / 500e-9
    integral = _trapezoid(cn2, h)
    r0 = (0.423 * k**2 * integral) ** (-3.0 / 5.0)
    assert 0.03 < r0 < 0.08


def test_discretize_conserves_total_turbulence():
    h = np.geomspace(1.0, 25000.0, 8000)
    cn2 = pyturb.hufnagel_valley(h)
    layers = pyturb.discretize_cn2(h, cn2, n_layers=8)
    assert len(layers) == 8
    fracs = np.array([layer.cn2_fraction for layer in layers])
    assert abs(fracs.sum() - 1.0) < 1e-9
    # Discretised centroid of h^{5/3} should track the continuous profile.
    h_bar_disc = profiles.mean_turbulence_height(layers)
    weight = cn2 / _trapezoid(cn2, h)
    h_bar_cont = _trapezoid(weight * h ** (5.0 / 3.0), h) ** (3.0 / 5.0)
    assert abs(h_bar_disc - h_bar_cont) / h_bar_cont < 0.25


def test_integrated_quantities_two_layer_by_hand():
    layers = [
        pyturb.Layer(0.0, 0.5, wind_speed=10.0),
        pyturb.Layer(10000.0, 0.5, wind_speed=20.0),
    ]
    r0 = 0.15
    h_bar = (0.5 * 0.0 + 0.5 * 10000.0 ** (5 / 3)) ** (3 / 5)
    v_bar = (0.5 * 10.0 ** (5 / 3) + 0.5 * 20.0 ** (5 / 3)) ** (3 / 5)
    assert abs(profiles.mean_turbulence_height(layers) - h_bar) < 1e-3
    assert abs(profiles.effective_wind_speed(layers) - v_bar) < 1e-6
    assert abs(profiles.isoplanatic_angle(layers, r0) - 0.314 * r0 / h_bar) < 1e-9
    assert abs(profiles.coherence_time(layers, r0) - 0.314 * r0 / v_bar) < 1e-9

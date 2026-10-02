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


def test_discretize_assigns_per_output_layer_wind_array():
    # A wind array whose length is the number of output layers (not the input
    # grid) is assigned to the layers in order, without moment averaging.
    h = np.geomspace(10.0, 20000.0, 500)
    cn2 = pyturb.hufnagel_valley(h)
    speeds = [5.0, 12.0, 30.0, 18.0]
    layers = pyturb.discretize_cn2(h, cn2, n_layers=4, wind=speeds)
    assert [layer.wind_speed for layer in layers] == speeds
    altitudes = [layer.altitude for layer in layers]
    assert altitudes == sorted(altitudes)


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


def test_from_cn2_hv57_reproduces_r0_and_theta0():
    # HV 5/7 is defined by r0 ~ 5 cm and theta0 ~ 7 urad at 500 nm; the
    # moment-conserving discretisation must keep the profile's theta0.
    h = np.geomspace(1.0, 30e3, 20000)
    cn2 = pyturb.hufnagel_valley(h)
    atm = pyturb.Atmosphere.from_cn2(h, cn2, n_layers=12, n=32, source="HV 5/7")
    assert atm.r0 == pytest.approx(0.05, rel=0.1)
    k = 2 * np.pi / 500e-9
    theta0_cont = (2.914 * k**2 * _trapezoid(cn2 * h ** (5 / 3), h)) ** (-3 / 5)
    assert atm.theta0 / 206264.8 == pytest.approx(theta0_cont, rel=0.02)
    assert theta0_cont == pytest.approx(7e-6, rel=0.1)
    assert atm.metadata["profile_source"] == "HV 5/7"
    fixed = pyturb.Atmosphere.from_cn2(h, cn2, n_layers=12, n=32, r0=0.2)
    assert fixed.r0 == pytest.approx(0.2)


def test_discretize_cn2_wind_direction():
    h = np.linspace(0.0, 20e3, 4001)
    cn2 = pyturb.hufnagel_valley(h)
    # Directions straddling north: the circular mean is ~0 deg, not ~180.
    straddle = np.where(np.arange(h.size) % 2 == 0, 350.0, 10.0)
    layers = pyturb.discretize_cn2(h, cn2, n_layers=4, wind_direction=straddle)
    for layer in layers:
        assert min(layer.wind_direction, 360 - layer.wind_direction) < 1.0
    per_layer = pyturb.discretize_cn2(h, cn2, n_layers=3, wind_direction=[0, 90, 200])
    assert [layer.wind_direction for layer in per_layer] == [0, 90, 200]
    scalar = pyturb.discretize_cn2(h, cn2, n_layers=3, wind_direction=45.0)
    assert all(layer.wind_direction == 45.0 for layer in scalar)
    with pytest.raises(ValueError, match="wind_direction"):
        pyturb.discretize_cn2(h, cn2, n_layers=3, wind_direction=[1.0, 2.0])


GR2015_IDS = [f"paranal-p{i:02d}" for i in range(1, 15)]


@pytest.mark.parametrize("name", GR2015_IDS)
def test_paranal_gr2015_profiles_reproduce_their_published_table(name):
    # Garcia-Rissmann et al. (2015), MNRAS 448, 2594: Table 3 gives the layer
    # percentages, Table 2 the mean turbulence height and tau0 those layers
    # imply (with layer speeds beta * v_ref and the tabulated r0). Recomputing
    # both pins the transcription to the printed rounding.
    info = pyturb.profile_info(name)
    layers = pyturb.get_profile(name)
    assert info.traceable and "10.1093/mnras/stv169" in info.source
    assert info.outer_scale == 25.0 and all(layer.L0 == 25.0 for layer in layers)
    fractions = np.array([layer.cn2_fraction for layer in layers])
    assert fractions.sum() == pytest.approx(1.0)                 # whole percents
    assert profiles.mean_turbulence_height(layers) == pytest.approx(
        info.conditions["mean_height"], abs=6.0)                 # 0.01 km print
    tau0 = profiles.coherence_time(layers, info.conditions["r0"])
    assert tau0 == pytest.approx(info.conditions["tau0"], abs=0.06e-3)  # 0.1 ms
    atm = pyturb.Atmosphere.from_profile(name, r0=info.conditions["r0"], n=16)
    assert atm.metadata["profile_source"] == info.source


def test_paranal_gr2015_classes_and_probabilities():
    infos = [pyturb.profile_info(name) for name in GR2015_IDS]
    # The fourteen profiles cover the Paranal statistics they were drawn from.
    assert sum(i.conditions["probability"] for i in infos) == pytest.approx(1.0)
    assert {i.conditions["quality"] for i in infos} == {"good", "median", "bad"}
    seeing = [i.conditions["seeing_class"] for i in infos]
    r0 = [i.conditions["r0"] for i in infos]
    assert np.all(np.diff(seeing) >= 0) and np.all(np.diff(r0) <= 0)

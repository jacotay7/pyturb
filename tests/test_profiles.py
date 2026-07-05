import numpy as np
import pytest

import pyturb
from pyturb import profiles

# np.trapezoid requires NumPy >= 2.0; fall back to np.trapz on the
# numpy>=1.22 floor declared in pyproject.toml.
_trapezoid = getattr(np, "trapezoid", np.trapz)


def test_named_profiles_load_and_are_nonempty():
    for name in pyturb.list_profiles():
        layers = pyturb.get_profile(name)
        assert len(layers) >= 1
        assert all(layer.cn2_fraction >= 0 for layer in layers)


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        pyturb.get_profile("does-not-exist")


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

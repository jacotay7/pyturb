"""Tests for the arbitrary-direction, non-periodic extrusion engine."""

import numpy as np

import pyturb
from pyturb.extrude import ExtrudedAtmosphere
from pyturb.infinite import phase_covariance

N = 64
DX = 8.0 / N
R0 = 0.15
L0 = 25.0


def _engine(theta_deg=0.0, speed=10.0, seed=0, altitude=0.0, interp="cubic"):
    th = np.deg2rad(theta_deg)
    return ExtrudedAtmosphere(
        n=N, pixel_scale=DX, layer_r0=[R0], layer_L0=[L0],
        layer_wind=[(speed * np.cos(th), speed * np.sin(th))],
        layer_altitude_los=[altitude], seeds=[seed], interp=interp,
    )


def test_theta0_is_exact_frozen_flow_translation():
    """Wind along axis 0 by a whole pixel is an exact roll of the screen."""
    eng = _engine(0.0)
    eng.set_time(0.0)
    s0 = np.array(eng.integrate())
    eng.set_time(3.0 * DX / 10.0)          # travel = 3 pixels exactly
    s3 = np.array(eng.integrate())
    # New turbulence enters at one edge; the rest is the old screen shifted.
    np.testing.assert_allclose(s3[:-3], s0[3:], rtol=0, atol=1e-4 * s0.std())


def _sf_ensemble(theta_deg, draws=40):
    acc = None
    for k in range(draws):
        eng = _engine(theta_deg, seed=k)
        eng.set_time(0.0)
        r, d = pyturb.structure_function(np.array(eng.integrate()), DX)
        acc = d if acc is None else acc + d
    return r, acc / draws


def test_direction_invariance():
    """Turbulence statistics must not depend on the wind direction."""
    r, d0 = _sf_ensemble(0.0)
    _, d37 = _sf_ensemble(37.0)
    ratio = (d37 / d0)[2:16]
    assert np.all(ratio > 0.9) and np.all(ratio < 1.1)


def test_structure_function_matches_von_karman():
    """The extruded screen follows the von Karman structure function.

    D(r) = 2 [C(0) - C(r)] for the covariance the extrusion is built from.
    """
    r, d = _sf_ensemble(0.0, draws=60)
    theory = 2.0 * (phase_covariance(0.0, R0, L0) - phase_covariance(r, R0, L0))
    mid = (r >= 4 * DX) & (r <= 8.0 / 4)
    frac = np.sqrt(np.mean(((d[mid] - theory[mid]) / theory[mid]) ** 2))
    assert frac < 0.08


def test_non_periodic():
    """Unlike the spectral engine, the screen does not repeat after n pixels."""
    eng = _engine(0.0, seed=1)
    eng.set_time(0.0)
    a = np.array(eng.integrate())
    eng.set_time(N * DX / 10.0)             # travel a full screen width
    b = np.array(eng.integrate())
    assert abs(np.corrcoef(a.ravel(), b.ravel())[0, 1]) < 0.9


def test_subpixel_advance_is_continuous():
    """A tiny sub-pixel step changes the screen only slightly (no jumps)."""
    eng = _engine(0.0, seed=2)
    eng.set_time(0.0)
    a = np.array(eng.integrate())
    eng.set_time(0.05 * DX / 10.0)          # 0.05 pixel
    b = np.array(eng.integrate())
    rel = np.std(b - a) / np.std(a)
    assert 0.0 < rel < 0.2                   # moved, but only a little


def test_offaxis_decorrelates_with_angle():
    """Off-axis footprints from a high layer differ more at larger angles."""
    eng = ExtrudedAtmosphere(
        n=N, pixel_scale=DX, layer_r0=[R0], layer_L0=[L0],
        layer_wind=[(10.0, 0.0)], layer_altitude_los=[10000.0], seeds=[3],
        field_of_view_pix=10000.0 * np.tan(np.deg2rad(30 / 3600.0)) / DX,
    )
    eng.set_time(0.0)
    on = np.array(eng.integrate(0.0, 0.0))
    small = np.array(eng.integrate(np.tan(np.deg2rad(5 / 3600.0)), 0.0))
    large = np.array(eng.integrate(np.tan(np.deg2rad(20 / 3600.0)), 0.0))
    var_small = np.var(small - on)
    var_large = np.var(large - on)
    assert var_large > var_small > 0.0


def test_memory_bounded_over_long_run():
    """The per-layer ring buffer must not grow with the number of frames."""
    eng = _engine(37.0, seed=4)
    cap = eng.layers[0]._buf.shape[0]
    for k in range(1, 4001):
        eng.set_time(k * DX / 10.0)
        eng.integrate()
    assert eng.layers[0]._buf.shape[0] == cap
    assert np.isfinite(np.array(eng.integrate())).all()

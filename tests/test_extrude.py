"""Tests for the arbitrary-direction, non-periodic extrusion engine."""

import numpy as np
import pytest

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


def test_finescale_readout_flicker_is_bounded():
    """Characterizes (does not eliminate) the sub-pixel readout's known
    position-dependent behaviour: the finest-scale (1 px) structure function
    measurably differs between integer and half-pixel wind travel (the
    combination of the stencil recurrence's own discretisation bias and the
    Catmull-Rom kernel's position-dependent frequency response). Bounded to a
    known range so a much larger swing (a new bug) or ~0 swing (a change to
    the recurrence or interpolation that would invalidate this
    characterization) gets caught."""
    def d1px(screen):
        return 0.5 * (np.mean(np.diff(screen, axis=0) ** 2)
                      + np.mean(np.diff(screen, axis=1) ** 2))

    eng = _engine(0.0, seed=3)
    eng.set_time(0.0)
    at_int = d1px(np.array(eng.integrate()))
    eng.set_time(0.5 * DX / 10.0)
    at_half = d1px(np.array(eng.integrate()))
    rise = at_half / at_int - 1.0
    assert 0.03 < rise < 0.20


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

    # Growing variance alone doesn't rule out fabricated data: a screen
    # clamped to duplicate its last extruded row also "decorrelates" from
    # on-axis. Rule that out directly by checking neither off-axis readout
    # contains adjacent bitwise-identical rows (the clamping signature).
    for screen in (small, large):
        dup = np.sum(np.all(np.diff(screen, axis=0) == 0.0, axis=1))
        assert dup == 0


def test_memory_bounded_over_long_run():
    """The per-layer ring buffer must not grow with the number of frames."""
    eng = _engine(37.0, seed=4)
    cap = eng.layers[0]._buf.shape[0]
    for k in range(1, 4001):
        eng.set_time(k * DX / 10.0)
        eng.integrate()
    assert eng.layers[0]._buf.shape[0] == cap
    assert np.isfinite(np.array(eng.integrate())).all()


@pytest.mark.gpu
def test_gpu_batched_readout_matches_looped():
    """The fused-gather GPU readout equals the per-layer loop it replaces.

    Both paths are exercised on the same GPU buffers (on- and off-axis) so a
    regression in the batched gather can't hide behind independent RNG streams;
    they must agree to float64 round-off (only the layer-sum order differs)."""
    atm = pyturb.Atmosphere.from_profile(
        "paranal-median", seeing=0.8, n=48, engine="extrude", device="gpu",
        dtype="float64", seed=0, field_of_view=20)
    ext = atm._ext
    ext.set_time(0.05)
    for thx, thy in [(0.0, 0.0), (np.tan(np.deg2rad(12 / 3600.0)), 0.0)]:
        batched = pyturb.to_numpy(ext._integrate_batched(thx, thy))
        looped = pyturb.to_numpy(ext._integrate_looped(thx, thy))
        np.testing.assert_allclose(batched, looped, rtol=1e-10, atol=1e-9)


def test_large_single_time_jump_matches_many_small_steps():
    """set_time() by a huge single jump (more travel than the ring buffer
    holds) must not crash, and must exactly match reaching the same time via
    small steps -- extrusion is a deterministic recurrence; only the
    compaction bookkeeping should differ."""
    speed = 32.0  # m/s, e.g. a jet-stream layer
    big = _engine(0.0, speed=speed, seed=5)
    big.set_time(1.0)  # 1 s * 32 m/s / DX px/m -- far more than one screen width
    jumped = np.array(big.integrate())

    small = _engine(0.0, speed=speed, seed=5)
    for k in range(1, 101):
        small.set_time(k * 1.0 / 100.0)
    stepped = np.array(small.integrate())

    np.testing.assert_array_equal(jumped, stepped)
    assert np.isfinite(jumped).all()

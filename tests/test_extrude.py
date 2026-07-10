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


def test_extruded_engine_validates_its_grid_before_backend_dispatch():
    """The direct extrusion API shares the periodic engine's grid contract."""
    common = dict(
        layer_r0=[R0],
        layer_L0=[L0],
        layer_wind=[(1.0, 0.0)],
        layer_altitude_los=[0.0],
    )
    with pytest.raises(ValueError, match="integer"):
        ExtrudedAtmosphere(n=32.0, pixel_scale=DX, **common)
    with pytest.raises(ValueError, match="pixel_scale"):
        ExtrudedAtmosphere(n=N, pixel_scale=0.0, **common)
    with pytest.raises(ValueError, match="dtype"):
        ExtrudedAtmosphere(n=N, pixel_scale=DX, dtype="float16", **common)
    with pytest.raises(ValueError, match="field_of_view_pix"):
        ExtrudedAtmosphere(n=N, pixel_scale=DX, field_of_view_pix=[0.0, 1.0], **common)
    with pytest.raises(ValueError, match="seeds"):
        ExtrudedAtmosphere(n=N, pixel_scale=DX, seeds=[1, 2], **common)
    with pytest.raises(ValueError, match="tau_boil"):
        ExtrudedAtmosphere(n=N, pixel_scale=DX, tau_boil=[0.1, 0.2], **common)


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


def test_extruded_screen_is_isotropic_per_axis():
    """The along-wind and cross-wind structure functions must agree (isotropy).

    The azimuthally-averaged structure function used elsewhere sums both axes,
    which can *hide* an anisotropy where one axis runs high and the other low
    (the failure mode of extruding infinite-outer-scale/Kolmogorov turbulence:
    the recurrence over-builds large scales along the wind while the joint row
    covariance under-builds them across it). Finite-L0 von Karman is bounded and
    stays isotropic; this checks each axis separately against theory and their
    ratio, so such an anisotropy cannot slip through."""
    def sf_axis(s, axis, maxsep):
        return np.array([
            np.mean((s[k:, :] - s[:-k, :]) ** 2) if axis == 0
            else np.mean((s[:, k:] - s[:, :-k]) ** 2)
            for k in range(1, maxsep + 1)])

    maxsep = N // 4
    acc_a = acc_c = None
    for seed in range(60):
        eng = _engine(37.0, seed=seed)          # rotated grid exercises both axes
        eng.set_time(0.0)
        s = np.array(eng.integrate())
        a, c = sf_axis(s, 0, maxsep), sf_axis(s, 1, maxsep)
        acc_a = a if acc_a is None else acc_a + a
        acc_c = c if acc_c is None else acc_c + c
    da, dc = acc_a / 60, acc_c / 60
    r = np.arange(1, maxsep + 1) * DX
    theory = 2.0 * (phase_covariance(0.0, R0, L0) - phase_covariance(r, R0, L0))
    mid = np.arange(1, maxsep + 1) >= 3
    # Each axis tracks theory, and the two axes track each other (isotropy):
    # a Kolmogorov-style large-scale anisotropy would push this ratio well past
    # ~1.8, far outside this band.
    assert np.all((da / theory)[mid] > 0.82) and np.all((da / theory)[mid] < 1.15)
    assert np.all((dc / theory)[mid] > 0.82) and np.all((dc / theory)[mid] < 1.15)
    ratio = (da / dc)[mid]
    assert np.all(ratio > 0.85) and np.all(ratio < 1.20)


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


def test_lanczos_reduces_finescale_flicker_vs_cubic():
    """interp='lanczos' has a flatter sub-Nyquist response than the default
    cubic, so the finest-scale travel-phase flicker (integer vs half-pixel
    travel) is markedly smaller — the disclosed artifact, reduced."""
    def d1px(screen):
        return 0.5 * (np.mean(np.diff(screen, axis=0) ** 2)
                      + np.mean(np.diff(screen, axis=1) ** 2))

    def flicker(interp):
        rises = []
        for seed in range(6):
            eng = _engine(0.0, seed=seed, interp=interp)
            eng.set_time(0.0)
            at_int = d1px(np.array(eng.integrate()))
            eng.set_time(0.5 * DX / 10.0)
            at_half = d1px(np.array(eng.integrate()))
            rises.append(at_half / at_int - 1.0)
        return np.mean(rises)

    cubic = flicker("cubic")
    lanczos = flicker("lanczos")
    assert lanczos < 0.6 * cubic          # a clear reduction, not marginal
    assert abs(lanczos) < 0.06            # small in absolute terms


def test_lanczos_matches_von_karman_structure_function():
    """The Lanczos readout must not distort the ensemble statistics: it still
    follows the von Karman structure function at separations of a few pixels."""
    acc = None
    for k in range(60):
        eng = _engine(0.0, seed=k, interp="lanczos")
        eng.set_time(0.31 * DX / 10.0)     # a generic sub-pixel travel phase
        r, d = pyturb.structure_function(np.array(eng.integrate()), DX)
        acc = d if acc is None else acc + d
    d = acc / 60
    theory = 2.0 * (phase_covariance(0.0, R0, L0) - phase_covariance(r, R0, L0))
    mid = (r >= 3 * DX) & (r <= 8.0 / 4)
    ratio = d[mid] / theory[mid]
    assert np.all(ratio > 0.9) and np.all(ratio < 1.1)


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


def test_axis_aligned_wind_uses_smaller_buffer_than_diagonal():
    """A layer's own along-wind ring-buffer capacity and perpendicular width
    should reflect *its own* wind direction, not a blanket worst-case (45-deg)
    assumption shared by every layer: axis-aligned wind needs measurably less
    buffer than a 45-degree wind at the same screen size."""
    axis = _engine(theta_deg=0.0, seed=0)
    diag = _engine(theta_deg=45.0, seed=0)
    assert axis.capacity < diag.capacity
    assert axis.width < diag.width


def test_low_altitude_atmosphere_uses_smaller_buffer_with_field_of_view():
    """Only the highest layer needs the full field_of_view reach; an
    atmosphere of low-altitude layers should get a smaller shared buffer than
    one that includes a high-altitude layer, at the same field_of_view."""
    fov = 30.0  # arcsec
    ground_only = pyturb.Atmosphere.from_profile(
        "single-layer", seeing=0.8, n=64, engine="extrude",
        field_of_view=fov, seed=0)
    with_high_layer = pyturb.Atmosphere.from_profile(
        "paranal-median", seeing=0.8, n=64, engine="extrude",
        field_of_view=fov, seed=0)
    assert ground_only._ext.capacity < with_high_layer._ext.capacity
    assert ground_only._ext.width < with_high_layer._ext.width


def test_heterogeneous_fov_margin_list_gives_correct_statistics():
    """ExtrudedAtmosphere accepts a per-layer field_of_view_pix list (what
    Atmosphere now passes, scaled by each layer's own altitude): a low-margin
    ground layer (no anisoplanatism -- its footprint never shifts off-axis)
    and a high-margin, high-altitude layer must still combine into finite
    on-axis output, with off-axis decorrelation driven correctly by the
    altitude that actually has one."""
    eng = ExtrudedAtmosphere(
        n=N, pixel_scale=DX,
        layer_r0=[R0, R0], layer_L0=[L0, L0],
        layer_wind=[(10.0, 0.0), (10.0, 0.0)],
        layer_altitude_los=[0.0, 9000.0],
        field_of_view_pix=[0.0, 9000.0 * np.tan(np.deg2rad(20 / 3600.0)) / DX],
        seeds=[1, 2],
    )
    eng.set_time(0.0)
    on = np.array(eng.integrate(0.0, 0.0))
    off = np.array(eng.integrate(np.tan(np.deg2rad(15 / 3600.0)), 0.0))
    assert np.isfinite(on).all() and np.isfinite(off).all()
    assert np.var(off - on) > 0.0


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
@pytest.mark.parametrize("interp", ["cubic", "linear", "lanczos"])
def test_gpu_batched_readout_matches_looped(interp):
    """The fused-gather GPU readout equals the per-layer loop it replaces, for
    every interpolation kernel.

    Both paths are exercised on the same GPU buffers (on- and off-axis) so a
    regression in the batched gather can't hide behind independent RNG streams;
    they must agree to float64 round-off (only the layer-sum order differs)."""
    atm = pyturb.Atmosphere.from_profile(
        "paranal-median", seeing=0.8, n=48, engine="extrude", device="gpu",
        dtype="float64", seed=0, field_of_view=20, interp=interp)
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

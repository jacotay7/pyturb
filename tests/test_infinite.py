import numpy as np
import pytest

import pyturb
from pyturb.infinite import phase_covariance


def test_phase_covariance_variance():
    """C(0) must equal the analytic von Karman variance 0.0863 (L0/r0)^(5/3)."""
    r0, L0 = 0.15, 25.0
    c0 = phase_covariance(0.0, r0, L0)
    assert c0 == pytest.approx(0.0863 * (L0 / r0) ** (5.0 / 3.0), rel=1e-3)
    # Covariance must decay monotonically over moderate separations.
    r = np.linspace(0.0, L0, 50)
    c = phase_covariance(r, r0, L0)
    assert np.all(np.diff(c) < 0)


def test_requires_finite_outer_scale():
    with pytest.raises(ValueError):
        pyturb.InfinitePhaseScreen(n=32, pixel_scale=0.05, r0=0.1, L0=np.inf)
    with pytest.raises(ValueError):
        phase_covariance(1.0, 0.1, np.inf)


def test_step_shifts_screen():
    layer = pyturb.InfinitePhaseScreen(n=32, pixel_scale=0.05, r0=0.1, seed=0)
    before = np.array(layer.screen)
    after = layer.step()
    assert after.shape == (32, 32)
    # Frozen flow: previous rows move down by one.
    np.testing.assert_array_equal(np.array(after[:-1]), before[1:])
    assert not np.array_equal(np.array(after[-1]), before[-1])


def test_seed_reproducibility():
    a = pyturb.InfinitePhaseScreen(n=24, pixel_scale=0.05, r0=0.1, seed=9)
    b = pyturb.InfinitePhaseScreen(n=24, pixel_scale=0.05, r0=0.1, seed=9)
    np.testing.assert_array_equal(a.screen, b.screen)
    np.testing.assert_array_equal(a.step(5), b.step(5))


def test_advance_integer_matches_step():
    """advance() to a whole pixel must reproduce step() exactly (interp exact)."""
    a = pyturb.InfinitePhaseScreen(n=32, pixel_scale=0.05, r0=0.1, seed=3)
    b = pyturb.InfinitePhaseScreen(n=32, pixel_scale=0.05, r0=0.1, seed=3)
    # Two half-pixel advances land on the same screen as one whole step.
    a.advance(0.5)
    a.advance(0.5)
    np.testing.assert_array_equal(np.array(a.screen), np.array(b.step(1)))
    assert a.travel == pytest.approx(1.0)


def test_lanczos_interp_exact_at_integer_and_preserves_variance():
    """The 6-tap Lanczos readout is (to round-off) exact at integer offsets and
    keeps the screen finite with near-unchanged variance at a sub-pixel step."""
    a = pyturb.InfinitePhaseScreen(n=32, pixel_scale=0.05, r0=0.1, seed=3,
                                   interp="lanczos", dtype="float64")
    b = pyturb.InfinitePhaseScreen(n=32, pixel_scale=0.05, r0=0.1, seed=3,
                                   interp="lanczos", dtype="float64")
    a.advance(0.5)
    a.advance(0.5)
    np.testing.assert_allclose(np.array(a.screen), np.array(b.step(1)),
                               rtol=0, atol=1e-9)

    layer = pyturb.InfinitePhaseScreen(n=48, pixel_scale=0.05, r0=0.15, L0=25,
                                       seed=1, interp="lanczos", dtype="float64")
    v0 = np.var(np.array(layer.screen))
    layer.advance(0.37)
    assert np.isfinite(np.array(layer.screen)).all()
    assert 0.8 < np.var(np.array(layer.screen)) / v0 < 1.2


def test_invalid_interp_rejected():
    with pytest.raises(ValueError):
        pyturb.InfinitePhaseScreen(n=16, pixel_scale=0.05, r0=0.1, interp="nope")


def test_subpixel_frame_tracks_the_shift():
    """A half-pixel (linear) advance is exactly the midpoint of its neighbours.

    With frozen flow the screen at travel 0.5 is the travel-0 screen shifted up
    by half a row; with linear interpolation each interior row must equal the
    average of the two integer-offset neighbours to machine precision.
    """
    layer = pyturb.InfinitePhaseScreen(
        n=16, pixel_scale=0.05, r0=0.1, L0=20.0, seed=5, dtype="float64",
        interp="linear",
    )
    before = np.array(layer.screen)          # travel = 0  -> buffer rows [0, n)
    layer.advance(0.5)
    half = np.array(layer.screen)            # travel = 0.5
    layer.advance(0.5)
    after = np.array(layer.screen)           # travel = 1  -> buffer rows [1, n+1)
    assert not np.array_equal(half, before)
    assert not np.array_equal(half, after)
    # before[i] = buffer[i], after[i] = buffer[i+1], half[i] = mean of the two.
    mid = slice(1, -1)
    np.testing.assert_allclose(half[mid], 0.5 * (before[mid] + after[mid]),
                               rtol=0, atol=1e-6)


def test_memory_stays_bounded():
    """The ring buffer must not grow with the number of steps."""
    layer = pyturb.InfinitePhaseScreen(n=48, pixel_scale=0.05, r0=0.1, seed=1)
    cap = layer._buf.shape[0]
    for _ in range(3000):
        layer.step()
    assert layer._buf.shape[0] == cap          # never reallocated
    assert cap <= layer.n + 2 * layer.stencil_rows + 16
    assert np.isfinite(np.array(layer.screen)).all()


def test_large_single_advance_matches_many_small_steps():
    """A jump far bigger than the ring buffer must not crash, and must give
    exactly the same screen as reaching the same travel via small steps
    (extrusion is a deterministic recurrence; only the compaction bookkeeping
    should differ)."""
    n = 128
    big = pyturb.InfinitePhaseScreen(n=n, pixel_scale=0.05, r0=0.15, L0=25, seed=0)
    jumped = np.array(big.advance(300))  # far more than one screen width

    small = pyturb.InfinitePhaseScreen(n=n, pixel_scale=0.05, r0=0.15, L0=25, seed=0)
    for _ in range(300):
        stepped = np.array(small.advance(1.0))

    np.testing.assert_array_equal(jumped, stepped)
    assert np.isfinite(jumped).all()


def test_large_advance_triggers_multiple_compactions_without_crash():
    """A jump many multiples of the buffer capacity must still work (forces
    _compact() to run repeatedly mid-jump, not just once)."""
    layer = pyturb.InfinitePhaseScreen(n=32, pixel_scale=0.05, r0=0.1, seed=2)
    cap = layer._buf.shape[0]
    screen = np.array(layer.advance(50 * cap))
    assert np.isfinite(screen).all()
    assert layer._buf.shape[0] == cap  # buffer never reallocated


def test_subpixel_preserves_variance():
    """Sub-pixel stepping keeps the screen variance, self-consistently.

    The instantaneous variance of one small screen fluctuates with the
    large-scale modes as the screen blows, so we compare the *long-run mean*
    variance of the sub-pixel path against the exact integer-step baseline
    (same seed, same extruded sequence). L0 is kept near the screen size so the
    outer scale is well sampled and the mean is a stable estimator. Cubic
    interpolation attenuates high-frequency power only slightly.
    """
    kw = dict(n=48, pixel_scale=0.04, r0=0.12, L0=2.0, dtype="float64")

    base = pyturb.InfinitePhaseScreen(seed=7, **kw)
    base_var = np.mean([
        float(np.var(np.array(base.step()))) for _ in range(2000)
    ][200:])

    sub = pyturb.InfinitePhaseScreen(seed=7, **kw)
    variances = []
    for _ in range(2000):
        sub.advance(0.37)                      # irrational-ish sub-pixel stride
        variances.append(float(np.var(np.array(sub.screen))))
    v = np.array(variances[200:])

    assert np.isfinite(v).all() and (v > 0).all()   # no ring-buffer corruption
    assert 0.85 < v.mean() / base_var < 1.05         # mild cubic attenuation


def test_extruded_statistics_match_theory():
    """The structure function along the wind axis must follow von Karman.

    D(r) = 2 [C(0) - C(r)] for the covariance used to build the extrusion.
    """
    r0, L0, pixel_scale = 0.12, 4.0, 0.04
    layer = pyturb.InfinitePhaseScreen(
        n=32, pixel_scale=pixel_scale, r0=r0, L0=L0, seed=11, dtype="float64"
    )
    rows = []
    for _ in range(4000):
        layer.step()
        rows.append(np.array(layer.screen[-1]))
    history = np.stack(rows)  # (steps, n): axis 0 is wind travel

    max_separation = 12
    separations = np.arange(1, max_separation + 1)
    measured = np.array(
        [np.mean((history[s:] - history[:-s]) ** 2) for s in separations]
    )
    theory = 2.0 * (
        phase_covariance(0.0, r0, L0)
        - phase_covariance(separations * pixel_scale, r0, L0)
    )
    ratio = measured / theory
    assert np.all(ratio > 0.7)
    assert np.all(ratio < 1.3)
    assert abs(np.mean(ratio) - 1.0) < 0.15


def test_extruded_screen_is_isotropic_after_burn_in():
    """The extruded screen must be isotropic (no systematic along- vs cross-wind
    anisotropy) after the FFT-seeded rows have scrolled off.

    A single realisation is noisy at large separations -- a 2-row-stencil screen
    can show 10-20% apparent anisotropy by chance at ~1 m on one draw. Averaged
    over a modest ensemble and several decorrelated snapshots it is isotropic to
    within a few percent: this pins that the row recurrence introduces no
    *systematic* directional bias (measurements are taken at integer travel, so
    the sub-pixel interpolation low-pass -- a separate, finest-scale artifact --
    is not in play). Guards against a regression that would make the extruder
    genuinely anisotropic.
    """
    n, dx, r0, L0 = 48, 0.05, 0.15, 25.0
    seps = [8, 12, 16]  # 0.4-0.8 m; along axis 0 (wind), across axis 1
    snaps = [120 + 20 * k for k in range(8)]  # integer travel -> exact interp
    acc_along = np.zeros(len(seps))
    acc_cross = np.zeros(len(seps))
    for seed in range(50):
        scr = pyturb.InfinitePhaseScreen(n=n, pixel_scale=dx, r0=r0, L0=L0,
                                         seed=seed, dtype="float64")
        prev = 0
        for travel in snaps:
            scr.advance(travel - prev)
            prev = travel
            s = np.asarray(scr.screen)
            for i, d in enumerate(seps):
                acc_along[i] += np.mean((s[d:, :] - s[:-d, :]) ** 2)
                acc_cross[i] += np.mean((s[:, d:] - s[:, :-d]) ** 2)
    n_samp = 50 * len(snaps)
    d_along = acc_along / n_samp
    d_cross = acc_cross / n_samp
    theory = 2.0 * (phase_covariance(0.0, r0, L0)
                    - phase_covariance(np.array(seps) * dx, r0, L0))
    # Both axes match von Karman, and each other, to within ~10% on this
    # ensemble -- no systematic anisotropy (the claimed ~10-15% along-wind
    # deficit does not survive proper ensemble averaging).
    assert np.all(0.85 < d_along / theory) and np.all(d_along / theory < 1.15)
    assert np.all(0.85 < d_cross / theory) and np.all(d_cross / theory < 1.15)
    ratio = d_along.mean() / d_cross.mean()
    assert 0.85 < ratio < 1.15, f"along/cross anisotropy {ratio:.3f} out of band"

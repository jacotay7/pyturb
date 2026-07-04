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

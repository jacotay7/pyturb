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

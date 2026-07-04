import numpy as np
import pytest

import pyturb


def test_shapes_and_dtype():
    gen = pyturb.PhaseScreen(n=64, pixel_scale=0.02, r0=0.1, seed=0)
    single = gen.generate()
    assert single.shape == (64, 64)
    assert single.dtype == np.float32

    batch = gen.generate(5)
    assert batch.shape == (5, 64, 64)

    gen64 = pyturb.PhaseScreen(n=64, pixel_scale=0.02, r0=0.1, seed=0, dtype="float64")
    assert gen64.generate().dtype == np.float64


def test_seed_reproducibility():
    a = pyturb.PhaseScreen(n=32, pixel_scale=0.05, r0=0.1, seed=42).generate(3)
    b = pyturb.PhaseScreen(n=32, pixel_scale=0.05, r0=0.1, seed=42).generate(3)
    np.testing.assert_array_equal(a, b)

    c = pyturb.PhaseScreen(n=32, pixel_scale=0.05, r0=0.1, seed=43).generate(3)
    assert not np.array_equal(a, c)


def test_batch_screens_are_independent():
    # Consecutive screens come from the real/imag parts of one FFT. A single
    # pair can show large sample correlation by chance (screens are dominated
    # by a few low-order modes), but over many pairs the mean correlation of
    # independent screens must vanish.
    batch = pyturb.PhaseScreen(n=32, pixel_scale=0.02, r0=0.1, seed=1).generate(400)
    flat = batch.reshape(400, -1).astype(np.float64)
    flat -= flat.mean(axis=1, keepdims=True)
    a, b = flat[0::2], flat[1::2]
    pair_correlations = (a * b).sum(axis=1) / np.sqrt(
        (a**2).sum(axis=1) * (b**2).sum(axis=1)
    )
    assert abs(pair_correlations.mean()) < 0.1


def test_kolmogorov_structure_function():
    """Ensemble structure function must match D(r) = 6.88 (r/r0)^(5/3)."""
    r0 = 0.1
    pixel_scale = 0.02
    gen = pyturb.PhaseScreen(
        n=256,
        pixel_scale=pixel_scale,
        r0=r0,
        L0=np.inf,
        seed=7,
        dtype="float64",
    )
    screens = gen.generate(20)
    r, measured = pyturb.structure_function(screens, pixel_scale, max_separation=24)
    theory = 6.88 * (r / r0) ** (5.0 / 3.0)
    # Skip the smallest separations where pixel discretisation bites.
    ratio = measured[3:] / theory[3:]
    assert np.all(ratio > 0.85)
    assert np.all(ratio < 1.15)
    assert abs(np.mean(ratio) - 1.0) < 0.08


def test_von_karman_saturates_below_kolmogorov():
    """A finite outer scale must suppress large-separation power."""
    r0, pixel_scale = 0.1, 0.02
    common = dict(n=256, pixel_scale=pixel_scale, r0=r0, seed=3, dtype="float64")
    screens = pyturb.PhaseScreen(L0=1.0, **common).generate(10)
    r, measured = pyturb.structure_function(screens, pixel_scale, max_separation=60)
    theory_kolmogorov = 6.88 * (r / r0) ** (5.0 / 3.0)
    # Well beyond L0 the structure function saturates near twice the
    # variance, far below the unbounded Kolmogorov prediction.
    assert measured[-1] < 0.5 * theory_kolmogorov[-1]
    variance = 0.0863 * (1.0 / r0) ** (5.0 / 3.0)
    assert measured[-1] == pytest.approx(2.0 * variance, rel=0.35)


def test_zero_mean_low_frequency_correction():
    """Subharmonics must add power at large separations vs. none."""
    common = dict(n=128, pixel_scale=0.05, r0=0.1, L0=np.inf, seed=5, dtype="float64")
    with_sh = pyturb.PhaseScreen(subharmonics=3, **common).generate(10)
    without_sh = pyturb.PhaseScreen(subharmonics=0, **common).generate(10)
    _, d_with = pyturb.structure_function(with_sh, 0.05, max_separation=32)
    _, d_without = pyturb.structure_function(without_sh, 0.05, max_separation=32)
    assert d_with[-1] > 1.5 * d_without[-1]


def test_invalid_arguments():
    with pytest.raises(ValueError):
        pyturb.PhaseScreen(n=1, pixel_scale=0.01, r0=0.1)
    with pytest.raises(ValueError):
        pyturb.PhaseScreen(n=32, pixel_scale=-1, r0=0.1)
    with pytest.raises(ValueError):
        pyturb.PhaseScreen(n=32, pixel_scale=0.01, r0=0.1, L0=-5)
    with pytest.raises(ValueError):
        pyturb.PhaseScreen(n=32, pixel_scale=0.01, r0=0.1, dtype="int32")
    with pytest.raises(ValueError):
        pyturb.PhaseScreen(n=32, pixel_scale=0.01, r0=0.1).generate(0)

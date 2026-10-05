"""Seeded random streams: NumPy on the CPU, device-independent Philox on the GPU.

The GPU stream must be a pure function of the seed: the same seed has to give
the same turbulence on every CUDA device, whatever its SM count. The CPU stream
is NumPy's ``PCG64`` and must stay exactly what ``numpy.random.default_rng``
gives, so CPU results do not change between pyturb releases.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyturb
from pyturb._rng import _key_from_seed, default_rng

_M32 = np.uint64(0xFFFFFFFF)


def _philox4x32_10(ctr, key):
    """Reference Philox4x32-10 on uint64-held words (Salmon et al. 2011)."""
    c = [np.asarray(x, dtype=np.uint64) for x in ctr]
    k0, k1 = np.uint64(key[0]), np.uint64(key[1])
    for r in range(10):
        if r:
            k0 = (k0 + np.uint64(0x9E3779B9)) & _M32
            k1 = (k1 + np.uint64(0xBB67AE85)) & _M32
        p0 = np.uint64(0xD2511F53) * c[0]
        p1 = np.uint64(0xCD9E8D57) * c[2]
        c = [(p1 >> np.uint64(32)) ^ c[1] ^ k0, p1 & _M32,
             (p0 >> np.uint64(32)) ^ c[3] ^ k1, p0 & _M32]
    return c


def _reference_normals(seed, start_block, size, dtype):
    """Host reference of ``CudaNormalGenerator``: Philox blocks -> Box-Muller."""
    key = _key_from_seed(seed)
    per_block = 4 if np.dtype(dtype) == np.float32 else 2
    n_blocks = -(-size // per_block)
    ctr = np.arange(start_block, start_block + n_blocks, dtype=np.uint64)
    zero = np.zeros_like(ctr)
    c = _philox4x32_10([ctr & _M32, ctr >> np.uint64(32), zero, zero], key)
    if per_block == 4:
        out = []
        for p in range(2):
            u = c[2 * p].astype(np.float64) * 2.0**-32 + 2.0**-33
            t = c[2 * p + 1].astype(np.float64) * 2.0**-31
            r = np.sqrt(-2.0 * np.log(u))
            out += [r * np.cos(np.pi * t), r * np.sin(np.pi * t)]
    else:
        a = (c[0] >> np.uint64(5)) * np.uint64(1 << 26) + (c[1] >> np.uint64(6))
        e = (c[2] >> np.uint64(5)) * np.uint64(1 << 26) + (c[3] >> np.uint64(6))
        u = a.astype(np.float64) * 2.0**-53 + 2.0**-54
        t = e.astype(np.float64) * 2.0**-52
        r = np.sqrt(-2.0 * np.log(u))
        out = [r * np.cos(np.pi * t), r * np.sin(np.pi * t)]
    return np.stack(out, axis=1).ravel()[:size]


def test_reference_philox_matches_random123_known_answers():
    # Known-answer vectors from Random123's kat_vectors (philox4x32_10).
    cases = [
        ([0, 0, 0, 0], [0, 0],
         [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]),
        ([0xFFFFFFFF] * 4, [0xFFFFFFFF] * 2,
         [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD]),
        ([0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344], [0xA4093822, 0x299F31D0],
         [0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1]),
    ]
    for ctr, key, expected in cases:
        assert [int(w) for w in _philox4x32_10(ctr, key)] == expected


def test_cpu_stream_is_numpy_pcg64():
    # The CPU contract: pyturb's CPU streams are exactly numpy's default_rng.
    rng = default_rng(np, 123)
    assert isinstance(rng.bit_generator, np.random.PCG64)
    np.testing.assert_array_equal(
        rng.standard_normal(1000), np.random.default_rng(123).standard_normal(1000)
    )
    gen = pyturb.PhaseScreen(n=16, pixel_scale=0.05, r0=0.15, seed=4)
    np.testing.assert_array_equal(
        gen._rng.standard_normal(8), np.random.default_rng(4).standard_normal(8)
    )


def test_cpu_realisation_is_pinned():
    # A fixed seed reproduces the same CPU screen across pyturb releases.
    screen = pyturb.PhaseScreen(n=32, pixel_scale=0.05, r0=0.15, seed=0).generate()
    np.testing.assert_allclose(screen[0, :4], [16.879599, 15.353807, 14.753413,
                                               14.563261], rtol=1e-5)


# --------------------------------------------------------------------- GPU
@pytest.mark.gpu
@pytest.mark.parametrize("dtype, tol", [("float32", 5e-6), ("float64", 1e-12)])
def test_gpu_stream_matches_host_reference(dtype, tol):
    # The host reference has no notion of a device, so agreeing with it means
    # the stream depends only on (seed, position). Odd sizes, multi-dimensional
    # shapes and successive draws check the block counter bookkeeping.
    rng = default_rng(pyturb.get_array_module("gpu"), 2024)
    per_block = 4 if dtype == "float32" else 2
    block = 0
    for shape in [(7,), (3, 5), (2, 64, 64), (1,), (12_345,)]:
        got = pyturb.to_numpy(rng.standard_normal(shape, dtype=dtype))
        size = int(np.prod(shape))
        ref = _reference_normals(2024, block, size, dtype).reshape(shape)
        assert got.dtype == np.dtype(dtype) and got.shape == shape
        np.testing.assert_allclose(got, ref, rtol=0, atol=tol)
        block += -(-size // per_block)


@pytest.mark.gpu
def test_gpu_stream_does_not_depend_on_launch_configuration(monkeypatch):
    # Simulates a GPU with a different SM count: fewer, smaller thread blocks
    # (each thread then grid-strides over many Philox blocks) must give the
    # same bits.
    import pyturb._rng as rng_mod

    cupy = pyturb.get_array_module("gpu")
    for dtype in ("float32", "float64"):
        ref = rng_mod.CudaNormalGenerator(5).standard_normal((3, 70_001), dtype=dtype)
        monkeypatch.setattr(rng_mod, "_MAX_BLOCKS", 3)
        monkeypatch.setattr(rng_mod, "_THREADS", 32)
        got = rng_mod.CudaNormalGenerator(5).standard_normal((3, 70_001), dtype=dtype)
        monkeypatch.undo()
        assert cupy.array_equal(got, ref)


@pytest.mark.gpu
def test_gpu_stream_statistics():
    # Box-Muller over Philox: N(0, 1) moments, no lag correlation, and
    # independent streams for different seeds.
    cupy = pyturb.get_array_module("gpu")
    for dtype in ("float32", "float64"):
        z = default_rng(cupy, 11).standard_normal(4_000_000, dtype=dtype)
        z = pyturb.to_numpy(z).astype(np.float64)
        n = z.size
        assert abs(z.mean()) < 5 / np.sqrt(n)
        assert abs(z.var() - 1.0) < 5 * np.sqrt(2.0 / n)
        assert abs(np.mean(z**4) - 3.0) < 5 * np.sqrt(96.0 / n)
        for lag in (1, 2, 3, 4):
            assert abs(np.mean(z[:-lag] * z[lag:])) < 5 / np.sqrt(n)
        assert abs(np.mean(np.abs(z) > 3.0) - 2.6998e-3) < 5 * np.sqrt(2.7e-3 / n)
        other = pyturb.to_numpy(default_rng(cupy, 12).standard_normal(n, dtype=dtype))
        assert abs(np.mean(z * other)) < 5 / np.sqrt(n)


@pytest.mark.gpu
def test_gpu_rng_passes_generators_through():
    cupy = pyturb.get_array_module("gpu")
    rng = default_rng(cupy, 3)
    assert default_rng(cupy, rng) is rng
    own = cupy.random.default_rng(3)
    assert default_rng(cupy, own) is own
    with pytest.raises(TypeError, match="float32 and float64"):
        rng.standard_normal(4, dtype="int32")
    assert rng.standard_normal((0, 3), dtype="float32").shape == (0, 3)


# PhaseScreen(n=32, pixel_scale=0.05, r0=0.15, seed=0, dtype="float64") on the GPU.
GPU_SCREEN_SEED0 = [-10.804032081826902, -11.782270917301734, -12.245892157165642,
                    -12.248114092521758]


@pytest.mark.gpu
def test_gpu_realisation_is_pinned():
    # Golden values for the GPU stream: a host with any CUDA GPU must
    # reproduce them, which is the single-GPU check that results do not
    # depend on which device (or how many SMs) ran them.
    cupy = pyturb.get_array_module("gpu")
    z = pyturb.to_numpy(default_rng(cupy, 123).standard_normal(6, dtype="float64"))
    np.testing.assert_allclose(z, _reference_normals(123, 0, 6, "float64"), atol=1e-13)
    gen = pyturb.PhaseScreen(n=32, pixel_scale=0.05, r0=0.15, seed=0, device="gpu",
                             dtype="float64")
    np.testing.assert_allclose(pyturb.to_numpy(gen.generate())[0, :4],
                               GPU_SCREEN_SEED0, rtol=1e-9)


def _atmosphere_run(device_kw, engine, tau_boil):
    atm = pyturb.Atmosphere.from_profile(
        "keck", seeing=0.6, diameter=11.25, n=64, seed=123, engine=engine,
        tau_boil=tau_boil, **device_kw,
    )
    spectra = pyturb.to_numpy(atm._spectra) if engine == "spectral" else None
    first = pyturb.to_numpy(atm.opd(0.0))
    rest = [pyturb.to_numpy(o) for _, o in atm.frames(2e-3, 30)]
    return spectra, np.stack([first] + rest)


@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore::pyturb.ExtrudeBoilingPerformanceWarning")
@pytest.mark.parametrize("engine", ["spectral", "extrude"])
@pytest.mark.parametrize("tau_boil", [None, 0.05])
def test_same_seed_same_atmosphere_on_every_gpu(engine, tau_boil):
    import cupy

    count = cupy.cuda.runtime.getDeviceCount()
    if count < 2:
        pytest.skip("needs at least two CUDA devices (test_gpu_realisation_is_pinned "
                    "and the launch-configuration test cover a single GPU)")
    ref = None
    for index in range(count):
        # Both ways of choosing the device: an explicit index, and the current
        # CuPy device with a plain "gpu". On one device they must agree exactly.
        spectra, frames = _atmosphere_run({"device": f"gpu:{index}"}, engine, tau_boil)
        with cupy.cuda.Device(index):
            again = _atmosphere_run({"device": "gpu"}, engine, tau_boil)
        np.testing.assert_array_equal(frames, again[1])
        if ref is None:
            ref = (spectra, frames)
            continue
        if engine == "spectral":
            # The layer coefficients are elementwise functions of the noise,
            # so they are bit-identical on every device.
            np.testing.assert_array_equal(spectra, ref[0])
        # Frames go through cuFFT (spectral) or a cuBLAS recurrence (extrude),
        # whose kernel choice -- and so float32 summation order -- may differ
        # between GPU models: equal to float32 rounding, not necessarily bit
        # for bit. A different realisation would differ at order 1.
        np.testing.assert_allclose(frames, ref[1], rtol=0,
                                   atol=2e-5 * np.abs(ref[1]).max())

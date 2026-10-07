"""Seeded random streams: NumPy ``PCG64`` on the CPU, a device-independent
counter-based generator on the GPU.

On the CPU every pyturb stream is ``numpy.random.default_rng(seed)``.

On the GPU, ``cupy.random.default_rng`` is not used: its ``XORWOW``
generator keeps ``2048 x multiProcessorCount`` cuRAND states, and element
``i`` of a draw comes from state ``i % state_count``, so any draw larger than
that pool (every turbulence noise block) depends on how many SMs the GPU has.
:class:`CudaNormalGenerator` instead computes element ``i`` of the stream as a
pure function of ``(key, counter + i)``: Philox4x32-10 (Salmon et al. 2011,
"Parallel random numbers: as easy as 1, 2, 3") followed by a Box-Muller
transform. The stream therefore does not depend on the device, the launch
configuration, or the CuPy/cuRAND version. The integer stream is exact; the
Box-Muller step uses CUDA's ``log``/``sincospi`` (never the fast-math
intrinsics), which give the same bits on every device for a given CUDA
toolkit.

The two backends use different generators, so the same seed draws a different
realisation on the CPU than on the GPU.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np

__all__ = ["CudaNormalGenerator", "default_rng"]

# Device functions: element ``4 * b + j`` (float32) / ``2 * b + j`` (float64) of
# the stream is ``z[j]`` of block ``b``. Shared with kernels that draw the
# stream inline (``pyturb._gpu_spectral``).
_DEVICE_SRC = r"""
__device__ __forceinline__ void philox4x32_10(unsigned int c[4], unsigned int k0,
                                              unsigned int k1)
{
    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        if (r > 0) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
        unsigned int hi0 = __umulhi(0xD2511F53u, c[0]);
        unsigned int lo0 = 0xD2511F53u * c[0];
        unsigned int hi1 = __umulhi(0xCD9E8D57u, c[2]);
        unsigned int lo1 = 0xCD9E8D57u * c[2];
        unsigned int n0 = hi1 ^ c[1] ^ k0;
        unsigned int n2 = hi0 ^ c[3] ^ k1;
        c[0] = n0; c[1] = lo1; c[2] = n2; c[3] = lo0;
    }
}

// float32: one Philox block -> four normals (two Box-Muller pairs).
__device__ __forceinline__ void philox_normal4_f(unsigned long long ctr, unsigned int k0,
                                                 unsigned int k1, float z[4])
{
    unsigned int c[4] = {(unsigned int)ctr, (unsigned int)(ctr >> 32), 0u, 0u};
    philox4x32_10(c, k0, k1);
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        // u in (0, 1]: never 0, so log(u) is finite (|z| <= 6.8).
        float u = (float)c[2 * p] * 2.3283064365386963e-10f
                  + 1.1641532182693481e-10f;
        float t = (float)c[2 * p + 1] * 4.6566128730773926e-10f;  // 2 * [0, 1)
        float r = sqrtf(-2.0f * logf(u));
        float s, co;
        sincospif(t, &s, &co);
        z[2 * p] = r * co;
        z[2 * p + 1] = r * s;
    }
}

// float64: one Philox block -> two normals (one Box-Muller pair, 53-bit uniforms).
__device__ __forceinline__ void philox_normal2_d(unsigned long long ctr, unsigned int k0,
                                                 unsigned int k1, double z[2])
{
    unsigned int c[4] = {(unsigned int)ctr, (unsigned int)(ctr >> 32), 0u, 0u};
    philox4x32_10(c, k0, k1);
    unsigned long long a = ((unsigned long long)(c[0] >> 5) << 26) | (c[1] >> 6);
    unsigned long long e = ((unsigned long long)(c[2] >> 5) << 26) | (c[3] >> 6);
    // u in (0, 1): 53-bit uniform plus half an ulp.
    double u = (double)a * 1.1102230246251565e-16 + 5.551115123125783e-17;
    double t = (double)e * 2.220446049250313e-16;  // 2 * [0, 1)
    double r = sqrt(-2.0 * log(u));
    double s, co;
    sincospi(t, &s, &co);
    z[0] = r * co;
    z[1] = r * s;
}
"""

_SRC = _DEVICE_SRC + r"""
extern "C" __global__ void philox_normal_f(
    float* __restrict__ out, long long size, unsigned int k0, unsigned int k1,
    unsigned long long offset)
{
    long long nblk = (size + 3) / 4;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x; b < nblk;
         b += stride) {
        float z[4];
        philox_normal4_f(offset + (unsigned long long)b, k0, k1, z);
        long long i = b * 4;
        if (i + 4 <= size) {  // ``out`` is a fresh, 256-byte-aligned allocation
            reinterpret_cast<float4*>(out)[b] = make_float4(z[0], z[1], z[2], z[3]);
        } else {
            for (int j = 0; i + j < size; ++j) out[i + j] = z[j];
        }
    }
}

extern "C" __global__ void philox_normal_d(
    double* __restrict__ out, long long size, unsigned int k0, unsigned int k1,
    unsigned long long offset)
{
    long long nblk = (size + 1) / 2;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x; b < nblk;
         b += stride) {
        double z[2];
        philox_normal2_d(offset + (unsigned long long)b, k0, k1, z);
        long long i = b * 2;
        if (i + 2 <= size) {
            reinterpret_cast<double2*>(out)[b] = make_double2(z[0], z[1]);
        } else {
            out[i] = z[0];
        }
    }
}
"""

_kernels: dict = {}
_THREADS = 256
_MAX_BLOCKS = 1 << 16  # grid-stride beyond this; the output does not depend on it


def _kernel(suffix: str) -> Any:
    import cupy

    kernel = _kernels.get(suffix)
    if kernel is None:
        kernel = cupy.RawKernel(_SRC, f"philox_normal_{suffix}")
        _kernels[suffix] = kernel
    return kernel


def _key_from_seed(seed: Any) -> Tuple[int, int]:
    """Two 32-bit Philox key words from an int / ``None`` / ``SeedSequence`` seed."""
    if isinstance(seed, np.random.SeedSequence):
        seq = seed
    else:
        seq = np.random.SeedSequence(seed)
    k0, k1 = seq.generate_state(2, np.uint32)
    return int(k0), int(k1)


class CudaNormalGenerator:
    """Device-independent standard-normal stream on the GPU (CuPy).

    Philox4x32-10 keyed by ``seed`` (through :class:`numpy.random.SeedSequence`)
    with a 64-bit block counter, then Box-Muller. Element ``i`` of the stream
    depends only on the key and its position, so a seed gives the same numbers
    on every CUDA device, for any launch configuration. Each call consumes
    whole Philox blocks (4 float32 or 2 float64 values per block), advancing a
    host-side counter: drawing ``(a, b)`` then ``c`` values is reproducible,
    but not the same as one draw of ``a * b + c``.

    Parameters
    ----------
    seed : int, sequence of int, numpy.random.SeedSequence or None
        Anything :class:`numpy.random.SeedSequence` accepts. ``None`` draws
        fresh OS entropy.
    """

    def __init__(self, seed: Any = None) -> None:
        self._key = _key_from_seed(seed)
        self._counter = 0

    def __repr__(self) -> str:
        return f"CudaNormalGenerator(key={self._key}, counter={self._counter})"

    def _reserve(self, size: int, dtype: Any) -> Tuple[int, int, int, int]:
        """Consume ``size`` values without drawing them (for kernels drawing inline).

        Returns ``(key0, key1, offset, per_block)``: element ``i`` of the
        reserved stretch is element ``i % per_block`` of Philox block
        ``offset + i // per_block``, exactly as :meth:`standard_normal` of the
        same ``size`` and ``dtype`` would produce it, and the counter advances
        by the same amount.
        """
        per_block = 4 if np.dtype(dtype) == np.float32 else 2
        offset = self._counter
        self._counter = (self._counter + -(-int(size) // per_block)) % (1 << 64)
        return self._key[0], self._key[1], offset, per_block

    def standard_normal(
        self,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        dtype: Any = np.float64,
    ) -> Any:
        """Draw standard normals as a CuPy array (``numpy.random.Generator`` API)."""
        import cupy

        dt = np.dtype(dtype)
        if dt == np.float32:
            suffix, per_block = "f", 4
        elif dt == np.float64:
            suffix, per_block = "d", 2
        else:
            raise TypeError(
                f"standard_normal supports float32 and float64, not {dt.name} "
                "(the same set numpy.random.Generator.standard_normal accepts)"
            )
        shape: Tuple[int, ...]
        if size is None:
            shape = ()
        elif isinstance(size, (int, np.integer)):
            shape = (int(size),)
        else:
            shape = tuple(int(s) for s in size)
        out = cupy.empty(shape, dtype=dt)
        total = out.size
        if total == 0:
            return out
        n_blocks = -(-total // per_block)
        grid = min(-(-n_blocks // _THREADS), _MAX_BLOCKS)
        k0, k1 = self._key
        _kernel(suffix)(
            (grid,), (_THREADS,),
            (out, np.int64(total), np.uint32(k0), np.uint32(k1),
             np.uint64(self._counter)),
        )
        self._counter = (self._counter + n_blocks) % (1 << 64)
        return out


def default_rng(xp: Any, seed: Any = None) -> Any:
    """The pyturb random stream for backend ``xp`` (``numpy`` or ``cupy``).

    NumPy: ``numpy.random.default_rng(seed)``. CuPy: a
    :class:`CudaNormalGenerator` (device independent), unless ``seed`` is
    already a generator, which is used as is (a ``CudaNormalGenerator`` is
    shared; a ``cupy.random.Generator``/``BitGenerator`` the caller supplied
    keeps CuPy's own, device-dependent, stream).
    """
    if xp is np:
        return np.random.default_rng(seed)
    if isinstance(seed, CudaNormalGenerator):
        return seed
    if isinstance(seed, (xp.random.Generator, xp.random.BitGenerator)):
        return xp.random.default_rng(seed)
    return CudaNormalGenerator(seed)

"""Array-backend dispatch: NumPy on CPU, CuPy on GPU.

Every public class in pyturb takes a ``device`` argument. All heavy math is
written against the common NumPy/CuPy API, so the same code runs on either
backend. The only CPU-pinned work is the one-time covariance/matrix setup in
:class:`pyturb.InfinitePhaseScreen`, which needs ``scipy.special``.
"""

from __future__ import annotations

import numpy as np

_CPU_NAMES = frozenset({"cpu", "numpy"})
_GPU_NAMES = frozenset({"gpu", "cuda", "cupy"})


def get_array_module(device: str):
    """Return the array module (``numpy`` or ``cupy``) for a device name.

    Parameters
    ----------
    device : str
        ``"cpu"`` (alias ``"numpy"``) or ``"gpu"`` (aliases ``"cuda"``,
        ``"cupy"``).
    """
    name = str(device).lower()
    if name in _CPU_NAMES:
        return np
    if name in _GPU_NAMES:
        try:
            import cupy
        except ImportError as exc:
            raise ImportError(
                f"device={device!r} requires CuPy, which is not installed. "
                "Install the build matching your CUDA toolkit, e.g. "
                "'pip install pyturb[cuda12]' or 'pip install cupy-cuda12x'."
            ) from exc
        return cupy
    raise ValueError(
        f"Unknown device {device!r}. Expected one of "
        f"{sorted(_CPU_NAMES | _GPU_NAMES)}."
    )


def get_fft_module(xp):
    """Return an FFT module that preserves single precision.

    ``numpy.fft`` always computes in double precision, so on CPU we use
    ``scipy.fft`` (which keeps complex64 as complex64). On GPU, ``cupy.fft``
    already preserves precision.
    """
    if xp is np:
        import scipy.fft

        return scipy.fft
    return xp.fft


def to_numpy(array) -> np.ndarray:
    """Copy an array to host memory as a ``numpy.ndarray``.

    A no-op (beyond ``asarray``) for arrays that are already on the CPU.
    """
    if hasattr(array, "get"):  # CuPy device array
        return array.get()
    return np.asarray(array)

"""Array-backend dispatch: NumPy on CPU, CuPy on GPU.

Every public class in pyturb takes a ``device`` argument. All heavy math is
written against the common NumPy/CuPy API, so the same code runs on either
backend. The only CPU-pinned work is the one-time covariance/matrix setup in
:class:`pyturb.InfinitePhaseScreen`, which needs ``scipy.special``.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Optional

import numpy as np

_CPU_NAMES = frozenset({"cpu", "numpy"})
_GPU_NAMES = frozenset({"gpu", "cuda", "cupy"})


def get_array_module(device: str) -> ModuleType:
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


# CPU FFT thread count: None -> scipy default (1 thread); -1 -> all cores.
_fft_workers = None


def set_fft_workers(workers: Optional[int]) -> Optional[int]:
    """Set the thread count for CPU (SciPy) FFTs; affects all pyturb objects.

    ``None`` (default) is single-threaded; ``-1`` uses every core; a positive
    integer pins that many. No effect on the GPU path (CuPy). Returns the
    previous value.

    >>> import pyturb                       # doctest: +SKIP
    >>> pyturb.set_fft_workers(-1)          # use all cores for CPU FFTs
    """
    global _fft_workers
    if workers is not None and workers == 0:
        raise ValueError("workers must be None, -1, or a non-zero integer")
    previous = _fft_workers
    _fft_workers = None if workers is None else int(workers)
    return previous


def get_fft_workers() -> Optional[int]:
    """Return the current CPU FFT thread setting (see :func:`set_fft_workers`)."""
    return _fft_workers


class _ThreadedScipyFFT:
    """``scipy.fft`` wrapper that injects the pyturb ``workers`` setting.

    The 2-D transforms pyturb uses (``ifft2``/``fft2``) are threaded across
    cores when :func:`set_fft_workers` asks for it; everything else forwards to
    ``scipy.fft`` unchanged. The worker count is read at call time, so changing
    it affects screens that already exist.
    """

    def __init__(self, scipy_fft: ModuleType) -> None:
        self._m = scipy_fft

    def __getattr__(self, name: str) -> Any:
        return getattr(self._m, name)

    def ifft2(self, a: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("workers", _fft_workers)
        return self._m.ifft2(a, **kwargs)

    def fft2(self, a: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("workers", _fft_workers)
        return self._m.fft2(a, **kwargs)


def get_fft_module(xp: ModuleType) -> ModuleType:
    """Return an FFT module that preserves single precision.

    ``numpy.fft`` always computes in double precision, so on CPU we use
    ``scipy.fft`` (which keeps complex64 as complex64) wrapped so the
    :func:`set_fft_workers` thread count applies. On GPU, ``cupy.fft`` already
    preserves precision.
    """
    if xp is np:
        import scipy.fft

        return _ThreadedScipyFFT(scipy.fft)
    return xp.fft


def to_numpy(array: Any) -> np.ndarray:
    """Copy an array to host memory as a ``numpy.ndarray``.

    A no-op (beyond ``asarray``) for arrays that are already on the CPU.
    """
    if hasattr(array, "get"):  # CuPy device array
        return array.get()
    return np.asarray(array)

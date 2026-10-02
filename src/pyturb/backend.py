"""Array-backend dispatch: NumPy on CPU, CuPy on GPU.

Every public class in pyturb takes a ``device`` argument. All heavy math is
written against the common NumPy/CuPy API, so the same code runs on either
backend. The only CPU-pinned work is the one-time covariance/matrix setup in
:class:`pyturb.InfinitePhaseScreen`, which needs ``scipy.special``.
"""

from __future__ import annotations

import functools
import inspect
from contextlib import AbstractContextManager, nullcontext
from types import ModuleType
from typing import Any, Callable, Optional, Tuple, TypeVar

import numpy as np

_CPU_NAMES = frozenset({"cpu", "numpy"})
_GPU_NAMES = frozenset({"gpu", "cuda", "cupy"})


def _parse_device(device: str) -> Tuple[str, Optional[int]]:
    """Split ``"gpu:1"`` into ``("gpu", 1)``; ``"gpu"`` gives ``("gpu", None)``."""
    name = str(device).lower()
    base, sep, index = name.partition(":")
    if not sep:
        return name, None
    if base not in _GPU_NAMES or not index.isdigit():
        raise ValueError(
            f"Unknown device {device!r}. Expected one of "
            f"{sorted(_CPU_NAMES | _GPU_NAMES)}, or 'gpu:N' for GPU number N."
        )
    return base, int(index)


def get_array_module(device: str) -> ModuleType:
    """Return the array module (``numpy`` or ``cupy``) for a device name.

    Parameters
    ----------
    device : str
        ``"cpu"`` (alias ``"numpy"``) or ``"gpu"`` (aliases ``"cuda"``,
        ``"cupy"``), optionally with a GPU number: ``"gpu:1"``. Without a
        number the current CuPy device is used (normally GPU 0).
    """
    name, _index = _parse_device(device)
    if name in _CPU_NAMES:
        return np
    if name in _GPU_NAMES:
        try:
            import cupy
        except ImportError as exc:
            raise ImportError(
                f"device={device!r} requires CuPy, which is not installed. "
                "Install the build matching your CUDA toolkit, e.g. "
                "'pip install pyturb[cuda12]' or 'pip install \"cupy-cuda12x[ctk]\"' "
                "(use cuda13 / cupy-cuda13x for CUDA 13 drivers)."
            ) from exc
        return cupy
    raise ValueError(
        f"Unknown device {device!r}. Expected one of "
        f"{sorted(_CPU_NAMES | _GPU_NAMES)}."
    )


def device_context(device: str) -> AbstractContextManager:
    """Context making ``device``'s GPU current (no-op for CPU or a plain ``"gpu"``)."""
    _name, index = _parse_device(device)
    if index is None:
        return nullcontext()
    import cupy

    count = cupy.cuda.runtime.getDeviceCount()
    if index >= count:
        raise ValueError(
            f"device {device!r} asks for GPU {index}, but CuPy sees {count} "
            f"GPU(s) (numbered 0-{count - 1}; CUDA_VISIBLE_DEVICES and "
            "CUDA_DEVICE_ORDER control the numbering)."
        )
    return cupy.cuda.Device(index)


_F = TypeVar("_F", bound=Callable[..., Any])


def on_device(method: _F) -> _F:
    """Run a method with its object's ``device`` GPU current (:func:`device_context`)."""

    signature = inspect.signature(method)

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        device = getattr(self, "device", None)
        if device is None:  # a constructor, before self.device is set
            bound = signature.bind_partial(self, *args, **kwargs).arguments
            template = bound.get("template")
            device = bound.get("device", getattr(template, "device", "cpu"))
        with device_context(device):
            return method(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


# CPU FFT thread count: None -> scipy default (1 thread); -1 -> all cores.
_fft_workers = None


def set_fft_workers(workers: Optional[int]) -> Optional[int]:
    """Set the thread count for CPU (SciPy) FFTs; affects all pyturb objects.

    ``None`` (default) is single-threaded; ``-1`` uses every core; any positive
    integer pins that many. Other negative values follow ``scipy.fft``
    wraparound semantics (``os.cpu_count() + 1 + workers``, e.g. ``-2`` = all
    cores but one). ``0`` is rejected. No effect on the GPU path (CuPy). Returns
    the previous value.

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


_blas_controller: Any = None


def blas_single_thread() -> AbstractContextManager:
    """Context that runs CPU BLAS calls on one thread (restored on exit).

    The extruders issue many small, memory-bound matrix products per frame.
    A threaded BLAS (OpenBLAS sizes its pool to every core) gains nothing on
    them and leaves its threads spinning afterwards, which starves the Numba
    readout running on the same cores: on a 16-core host this halves the CPU
    extrude frame rate. Uses ``threadpoolctl`` (a pyturb dependency); a no-op
    if it is unavailable. The controller is built once, so entering the
    context costs a few microseconds.
    """
    global _blas_controller
    if _blas_controller is None:
        try:
            from threadpoolctl import ThreadpoolController

            _blas_controller = ThreadpoolController()
        except ImportError:  # pragma: no cover - threadpoolctl is a dependency
            _blas_controller = False
    if _blas_controller is False:
        return nullcontext()
    return _blas_controller.limit(limits=1, user_api="blas")

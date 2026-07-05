"""``pyturb.benchmark()`` — print your machine's throughput in a few seconds."""

from __future__ import annotations

import time
from typing import Dict

from .atmosphere import Atmosphere
from .backend import get_array_module


def _sync(device: str) -> None:
    if device != "cpu":
        import cupy

        cupy.cuda.runtime.deviceSynchronize()


def benchmark(
    n: int = 512,
    profile: str = "paranal-median",
    device: str = "cpu",
    seconds: float = 1.0,
    seeing: float = 0.8,
    engine: str = "spectral",
) -> Dict[str, float]:
    """Measure and print closed-loop and Monte-Carlo throughput.

    Parameters
    ----------
    n : int
        Pupil sampling.
    profile : str
        Named turbulence profile.
    device : str
        ``"cpu"`` or ``"gpu"``.
    seconds : float
        Wall-clock budget per measurement.
    seeing, engine
        Passed to :class:`Atmosphere`.

    Returns
    -------
    dict
        ``{"frames_per_s": ..., "screens_per_s": ...}``.
    """
    get_array_module(device)  # validates the device early
    atm = Atmosphere.from_profile(profile, seeing=seeing, n=n, device=device,
                                  engine=engine)

    gen = iter(atm.frames(dt=1e-3, steps=10 ** 9))
    for _ in range(3):
        next(gen)
    _sync(device)
    t0 = time.perf_counter()
    frames = 0
    while time.perf_counter() - t0 < seconds:
        next(gen)
        frames += 1
    _sync(device)
    fps = frames / (time.perf_counter() - t0)

    batch = max(1, min(64, (2 ** 22) // (n * n)))
    atm.sample(batch)
    _sync(device)
    t0 = time.perf_counter()
    drawn = 0
    while time.perf_counter() - t0 < seconds:
        atm.sample(batch)
        drawn += batch
    _sync(device)
    sps = drawn / (time.perf_counter() - t0)

    print(f"pyturb benchmark  n={n}  {profile}  device={device}  engine={engine}")
    print(f"  closed-loop frames : {fps:8.0f} /s")
    print(f"  Monte-Carlo screens: {sps:8.0f} /s")
    return {"frames_per_s": fps, "screens_per_s": sps}

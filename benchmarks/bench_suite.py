"""Comprehensive pyturb throughput suite — every standard use case, CPU + GPU.

Unlike ``bench_compare.py`` (head-to-head vs other libraries) and
``bench_frames.py`` (frame/screen rate only), this sweeps the operations a
pyturb user actually calls and reports a speed metric for each, so a change can
be judged against the whole surface at once:

* Monte-Carlo   — ``Atmosphere.sample`` and single-layer ``PhaseScreen.generate``
* Closed loop   — ``frames`` on the spectral, extrude and spectral+boiling engines
* Tomography    — ``opd(directions=...)`` for a guide-star constellation
* LGS cone      — ``frames`` with a finite-range sodium beacon
* Row extrusion — ``InfinitePhaseScreen.step`` / ``advance`` (single layer)
* Diagnostics   — ``zernike_decompose``, ``temporal_psd``, ``structure_function``

Run::

    python benchmarks/bench_suite.py                       # full sweep, markdown
    python benchmarks/bench_suite.py --n 512 --device gpu  # one size / device
    python benchmarks/bench_suite.py --quick               # short budget
    python benchmarks/bench_suite.py --json out.json       # self-describing raw artifact

GPU rows are skipped automatically when CuPy is unavailable. Numba (optional)
accelerates the CPU frozen-flow paths; its first call JIT-compiles, which the
warmup absorbs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Callable, Dict

import numpy as np

import pyturb

PROFILE = "paranal-median"
SEEING = 0.8
DIAMETER = 8.0
DIRECTIONS = [(0.0, 0.0), (15.0, 0.0), (0.0, 15.0), (10.0, 10.0), (-12.0, 6.0)]


def _package_version(name: str):
    """Return an installed package version without making it a requirement."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _revision():
    """Return the CI revision or local Git commit, when available."""
    revision = os.environ.get("GITHUB_SHA")
    if revision:
        return revision
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
        )
    except OSError:
        return None
    return result.stdout.strip() or None


def _source_dirty():
    """Report whether a local benchmark included uncommitted source changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    return bool(result.stdout.strip())


def _provenance():
    """Machine-readable context needed to interpret a throughput result."""
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "revision": _revision(),
        "source_dirty": _source_dirty(),
        "python": sys.version,
        "platform": platform.platform(),
        "pyturb": pyturb.__version__,
        "dependencies": {
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "numba": _package_version("numba"),
            "cupy": _package_version("cupy"),
        },
    }


def _sync(device: str) -> None:
    if device == "gpu":
        import cupy

        cupy.cuda.runtime.deviceSynchronize()


def _time(fn: Callable[[], object], seconds: float, device: str,
          warmup: int = 5) -> float:
    """Mean seconds per call: warm up (absorbs JIT/plan build), then run."""
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    iters = 0
    while time.perf_counter() - t0 < seconds:
        fn()
        iters += 1
    _sync(device)
    return (time.perf_counter() - t0) / iters


def _batch_for(n: int) -> int:
    """Monte-Carlo batch that keeps the working set ~16 M complex elements."""
    return max(1, min(64, (2 ** 24) // (n * n)))


def bench_device(n: int, device: str, seconds: float) -> Dict[str, float]:
    """Every use case at one (n, device); values are the natural rate (per s)."""
    out: Dict[str, float] = {}
    dx = DIAMETER / n

    def atm(**kw):
        return pyturb.Atmosphere.from_profile(
            PROFILE, seeing=SEEING, diameter=DIAMETER, n=n, device=device, **kw
        )

    # --- Monte-Carlo -----------------------------------------------------
    a = atm()
    batch = _batch_for(n)
    out["sample screens/s"] = batch / _time(lambda: a.sample(batch), seconds, device)
    gen = pyturb.PhaseScreen(n, dx, 0.15, 25.0, seed=0, device=device)
    out["generate screens/s"] = batch / _time(
        lambda: gen.generate(batch), seconds, device
    )

    # --- Closed-loop frozen flow ----------------------------------------
    for label, kw in (
        ("frames spectral fps", {}),
        ("frames extrude fps", {"engine": "extrude"}),
        ("frames boiling fps", {"tau_boil": 0.2}),
    ):
        it = iter(atm(**kw).frames(dt=1e-3, steps=10 ** 9))
        out[label] = 1.0 / _time(lambda it=it: next(it), seconds, device)

    # --- Off-axis / tomography ------------------------------------------
    a_fov = atm(field_of_view=20.0)
    dirs = DIRECTIONS
    out["tomography dirs/s"] = len(dirs) / _time(
        lambda: a_fov.opd(0.0, directions=dirs), seconds, device
    )

    # --- LGS cone --------------------------------------------------------
    it_lgs = iter(atm(lgs_altitude=90e3).frames(dt=1e-3, steps=10 ** 9))
    out["frames LGS fps"] = 1.0 / _time(lambda: next(it_lgs), seconds, device)

    # --- Single-layer row extrusion (InfinitePhaseScreen) ---------------
    layer = pyturb.InfinitePhaseScreen(n, dx, 0.15, 25.0, seed=0, device=device)
    out["infinite step/s"] = 1.0 / _time(lambda: layer.step(), seconds, device)
    out["infinite advance/s"] = 1.0 / _time(lambda: layer.advance(0.37), seconds, device)

    return out


def bench_analysis(n: int, seconds: float) -> Dict[str, float]:
    """Host-side diagnostics (device-independent; timed on a CPU screen stack)."""
    dx = DIAMETER / n
    a = pyturb.Atmosphere.from_profile(PROFILE, seeing=SEEING, n=n, device="cpu")
    screens = np.stack([a.sample() for _ in range(32)])
    basis = pyturb.zernike_basis(20, n)
    out = {
        "zernike_decompose stacks/s": 1.0 / _time(
            lambda: pyturb.zernike_decompose(screens[0], 20, basis=basis),
            seconds, "cpu"),
        "structure_function calls/s": 1.0 / _time(
            lambda: pyturb.structure_function(screens[0], dx), seconds, "cpu"),
        "temporal_psd calls/s": 1.0 / _time(
            lambda: pyturb.analysis.temporal_psd(screens.reshape(32, -1).T, 1e-3),
            seconds, "cpu"),
    }
    return out


def _print_table(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    # results[device][n][metric] = value
    for device, by_n in results.items():
        ns = sorted(by_n)
        metrics = list(next(iter(by_n.values())))
        print(f"\n### {device.upper()} — {PROFILE}, {len(pyturb.get_profile(PROFILE))} "
              f"layers, seeing {SEEING}\"\n")
        head = "| metric | " + " | ".join(f"n={n}" for n in ns) + " |"
        print(head)
        print("|" + "---|" * (len(ns) + 1))
        for m in metrics:
            cells = " | ".join(f"{by_n[n][m]:,.0f}" for n in ns)
            print(f"| {m} | {cells} |")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--device", nargs="+", default=["cpu", "gpu"],
                   choices=["cpu", "gpu"])
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--quick", action="store_true", help="0.3 s budget, n=512 only")
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()

    if args.quick:
        args.seconds = 0.3
        args.n = [512]

    devices = []
    for d in args.device:
        if d == "gpu":
            try:
                pyturb.get_array_module("gpu")
            except ImportError:
                print("(CuPy not installed — GPU skipped)")
                continue
        devices.append(d)

    print(f"# pyturb benchmark suite (pyturb {pyturb.__version__})")
    print(f"profile={PROFILE!r}  diameter={DIAMETER} m  seeing={SEEING}\"  "
          f"budget={args.seconds}s/cell")

    results: Dict[str, Dict[int, Dict[str, float]]] = {}
    for device in devices:
        results[device] = {}
        for n in args.n:
            results[device][n] = bench_device(n, device, args.seconds)

    _print_table(results)

    print("\n### Diagnostics (host-side, device-independent)\n")
    analysis = {n: bench_analysis(n, args.seconds) for n in args.n}
    metrics = list(next(iter(analysis.values())))
    print("| metric | " + " | ".join(f"n={n}" for n in args.n) + " |")
    print("|" + "---|" * (len(args.n) + 1))
    for m in metrics:
        print(f"| {m} | " + " | ".join(f"{analysis[n][m]:,.0f}" for n in args.n) + " |")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "provenance": _provenance(),
                    "configuration": {
                        "profile": PROFILE,
                        "seeing_arcsec": SEEING,
                        "diameter_m": DIAMETER,
                        "n": args.n,
                        "devices": devices,
                        "seconds_per_cell": args.seconds,
                        "monte_carlo_batch": {str(n): _batch_for(n) for n in args.n},
                    },
                    "results": results,
                    "analysis": analysis,
                },
                fh,
                indent=2,
            )
            fh.write("\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

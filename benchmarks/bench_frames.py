"""Benchmark pyturb OPD-frame throughput (CPU vs GPU) across screen sizes.

Run:  python benchmarks/bench_frames.py
      python benchmarks/bench_frames.py --profile paranal-median --n 256 512 1024

Prints frames/s for closed-loop evolution and screens/s for Monte-Carlo
sampling. GPU rows are skipped automatically if CuPy is unavailable.
"""

from __future__ import annotations

import argparse
import time

import pyturb


def _sync(device):
    if device == "gpu":
        import cupy

        cupy.cuda.runtime.deviceSynchronize()


def bench_frames(profile, device, n, steps, warmup=5):
    atm = pyturb.Atmosphere.from_profile(profile, seeing=0.8, diameter=8.0, n=n,
                                         device=device)
    for _ in atm.frames(dt=1e-3, steps=warmup):
        pass
    _sync(device)
    t0 = time.perf_counter()
    for _ in atm.frames(dt=1e-3, steps=steps):
        pass
    _sync(device)
    return steps / (time.perf_counter() - t0)


def bench_sample(profile, device, n, count, batch=8):
    atm = pyturb.Atmosphere.from_profile(profile, seeing=0.8, diameter=8.0, n=n,
                                         device=device)
    atm.sample(batch)  # warmup
    _sync(device)
    t0 = time.perf_counter()
    done = 0
    while done < count:
        atm.sample(batch)
        done += batch
    _sync(device)
    return done / (time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="paranal-median")
    p.add_argument("--n", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()

    devices = ["cpu"]
    try:
        pyturb.get_array_module("gpu")
        devices.append("gpu")
    except ImportError:
        print("(CuPy not installed — GPU rows skipped)")

    print(f"profile={args.profile!r}  layers="
          f"{len(pyturb.get_profile(args.profile))}\n")
    header = f"{'n':>6} {'device':>7} {'frames/s':>10} {'screens/s':>10}"
    print(header)
    print("-" * len(header))
    for n in args.n:
        for device in devices:
            steps = args.steps if device == "gpu" else max(20, args.steps // 10)
            fps = bench_frames(args.profile, device, n, steps)
            sps = bench_sample(args.profile, device, n, count=max(16, steps // 4))
            print(f"{n:>6} {device:>7} {fps:>10.0f} {sps:>10.1f}")


if __name__ == "__main__":
    main()

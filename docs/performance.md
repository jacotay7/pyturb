# Performance

Throughput depends on the GPU **and** the host CPU: a spectral frame is a
fixed chain of small kernels, so on the GPU the per-frame cost is mostly
launch overhead unless it is captured as a CUDA graph (the default). Measure
your own machine with `pyturb.benchmark(n=512, device="gpu")`, or the full
per-use-case sweep with `python benchmarks/bench_suite.py`.

## Typical numbers

9-layer `paranal-median`, frames per second (Monte-Carlo: screens per second),
pyturb 1.1, CuPy 14.2, on an Arm Neoverse-N1 host:

| | RTX 4060, 256² / 512² / 1024² | RTX A400, 256² / 512² / 1024² |
|---|---|---|
| `frames`, spectral (CUDA graph) | 6,451 / 4,591 / 1,304 | 4,832 / 1,290 / 327 |
| `opd(t=times)`, batched | 27,510 / 6,653 / 1,166 | 5,621 / 1,462 / 360 |
| `frames`, extrude | 1,174 / 482 / 120 | 410 / 103 / 25 |
| `frames`, spectral + boiling | 1,397 / 584 / 123 | 656 / 173 / 44 |
| `opd(directions=5)`, directions/s | 4,737 / 2,491 / 540 | 2,720 / 682 / 160 |
| `sample()`, screens/s | 33,292 / 6,993 / 1,715 | 9,309 / 2,466 / 615 |

pyturb 2.3 raised the RTX 4060 boiling row to 3,761 / 2,184 / 543 and added
an LGS-cone row of 6,073 / 1,896 / 362 frames/s (2.2: 205 / 167 / 49); see
`RESULTS.md` §7. More hardware, the CPU rows and the methodology are in
[`benchmarks/RESULTS.md`](https://github.com/jacotay7/pyturb/blob/main/benchmarks/RESULTS.md).

## GPU

- **Stay on the device.** Frames are CuPy arrays; copy to the host
  (`pyturb.to_numpy`) only when you must. Hand them to PyTorch/JAX without a
  copy via DLPack (see [Interop](interop.md)).
- **Offline series: ask for many times at once.** `atm.opd(t=times)` runs one
  batched transform over all of them: ~4x the frame-by-frame rate at 256² on
  an RTX 4060 (less on smaller GPUs, ~1.2x on an RTX A400, which is already
  compute-bound there).
- **CUDA graphs** (`cuda_graph=True`, default) replay each spectral frame,
  including the LGS cone, as one launch; this is what makes the frame rate
  independent of a slow host. Boiling adds one kernel per step (the noise is
  drawn inside it); the extruder still issues per-row kernels.
- **Several GPUs:** `device="gpu:1"`. CUDA numbers the fastest GPU first;
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` matches `nvidia-smi`.
- **float32** (default) is the fast path; `dtype="float64"` is slower,
  dramatically so on GPUs with little FP64 throughput (most consumer and
  workstation cards).
- **Bigger screens cost quadratically:** `oversample=2` makes the FFT 4x
  larger (RTX 4060, 512²: ~5,300 → ~1,360 spectral fps); at 256² it is
  nearly free.

## CPU

- `pip install "pyturb[accel]"` adds Numba kernels for the spectral layer
  sum, the boiling update, the LGS cone zoom and the extruder readout
  (several-fold on each). They give the same numbers as the NumPy fallback
  (bit for bit, except the layer sum's round-off).
- `pyturb.set_fft_workers(-1)` threads the SciPy FFTs (spectral frames,
  `sample()`).
- The extruder's row recurrence runs its small matrix products
  single-threaded (via `threadpoolctl`) and batched across layers: a threaded
  BLAS gains nothing on them and its spinning threads would starve the Numba
  readout. You don't need to set `OPENBLAS_NUM_THREADS` yourself.
- On shared machines, pin to idle cores (`taskset`) and compare runs at equal
  load; timings move by tens of percent with other users' work.

## Threads and parallel sweeps

By default the Numba kernels use every core (`NUMBA_NUM_THREADS`, the core
count), the SciPy FFTs one thread, and BLAS its own pool. That is right for
one simulation, but in a parallel sweep every worker process starts its own
pool per core: 8 workers on 16 cores run 128 spinning threads, and the sweep
can run an order of magnitude slower than one worker would.

`pyturb.set_threads(n)` caps pyturb's Numba kernels at `n` threads (in every
Python thread that runs them) and caps threaded FFTs, if you enabled them
with `set_fft_workers`, at `n` workers. Results do not depend on the setting:
each output element is computed the same way whatever the thread count. A
sweep then gives each worker one thread:

```python
import multiprocessing as mp
import numpy as np
import pyturb


def run(seed):
    atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, n=64,
                                         tau_boil=2.0, seed=seed)
    return float(np.std([pyturb.to_numpy(o) for _, o in atm.frames(1e-3, 5)]))


if __name__ == "__main__":
    with mp.get_context("spawn").Pool(2, initializer=pyturb.set_threads,
                                      initargs=(1,)) as pool:
        print(pool.map(run, range(4)))
```

Environment variables, read when a process starts:

- `NUMBA_NUM_THREADS`: the size of Numba's pool (set it before Numba is
  imported; it is the most `set_threads` can use).
- `OMP_NUM_THREADS`: honoured by pyturb's Numba kernels when
  `NUMBA_NUM_THREADS` is not set, and by OpenMP/OpenBLAS.
- `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS`: the BLAS pool. `set_threads`
  leaves BLAS alone because OpenBLAS's rounding depends on its thread count:
  limiting it (here or with `threadpoolctl`) is safe for speed but can change
  seeded screens in the last bit.

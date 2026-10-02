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

More hardware, the CPU rows and the methodology are in
[`benchmarks/RESULTS.md`](https://github.com/jacotay7/pyturb/blob/main/benchmarks/RESULTS.md).

## GPU

- **Stay on the device.** Frames are CuPy arrays; copy to the host
  (`pyturb.to_numpy`) only when you must. Hand them to PyTorch/JAX without a
  copy via DLPack (see [Interop](interop.md)).
- **Offline series: ask for many times at once.** `atm.opd(t=times)` runs one
  batched transform over all of them: ~4x the frame-by-frame rate at 256² on
  an RTX 4060 (less on smaller GPUs, ~1.2x on an RTX A400, which is already
  compute-bound there).
- **CUDA graphs** (`cuda_graph=True`, default) replay each spectral frame as
  one launch; this is what makes the frame rate independent of a slow host.
  Boiling, the LGS cone and the extruder still issue per-frame kernels.
- **Several GPUs:** `device="gpu:1"`. CUDA numbers the fastest GPU first;
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` matches `nvidia-smi`.
- **float32** (default) is the fast path; `dtype="float64"` is slower,
  dramatically so on GPUs with little FP64 throughput (most consumer and
  workstation cards).
- **Bigger screens cost quadratically:** `oversample=2` makes the FFT 4x
  larger (RTX 4060, 512²: ~5,300 → ~1,360 spectral fps); at 256² it is
  nearly free.

## CPU

- `pip install "pyturb[accel]"` adds Numba kernels for the spectral layer sum
  and the extruder readout (several-fold on the extruder).
- `pyturb.set_fft_workers(-1)` threads the SciPy FFTs (spectral frames,
  `sample()`).
- The extruder's row recurrence runs its small matrix products
  single-threaded (via `threadpoolctl`) and batched across layers: a threaded
  BLAS gains nothing on them and its spinning threads would starve the Numba
  readout. You don't need to set `OPENBLAS_NUM_THREADS` yourself.
- On shared machines, pin to idle cores (`taskset`) and compare runs at equal
  load; timings move by tens of percent with other users' work.

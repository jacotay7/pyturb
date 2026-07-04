# pyturb

**Fast, GPU-optional atmospheric phase screens for adaptive optics.**

`pyturb` does one thing well: it generates high-quality simulated atmospheric
turbulence phase screens — the standard input to adaptive-optics (AO) system
simulations, wavefront-sensing studies, and end-to-end instrument models. It
is small, has a two-class API, runs on NumPy by default, and switches to CUDA
(via CuPy) with a single argument.

```python
import pyturb

# Independent Kolmogorov / von Kármán screens (FFT method + subharmonics)
gen = pyturb.PhaseScreen(n=512, pixel_scale=0.01, r0=0.15, L0=25, seed=1)
phase = gen.generate()        # (512, 512) phase in radians
batch = gen.generate(100)     # (100, 512, 512), generated in one FFT batch

# Endless frozen-flow screen for closed-loop simulation
layer = pyturb.InfinitePhaseScreen(n=256, pixel_scale=0.01, r0=0.15, L0=25, seed=1)
for _ in range(1000):
    phase = layer.step()      # advance the wind by one pixel per step
```

GPU acceleration is one keyword away — everything else is identical:

```python
gen = pyturb.PhaseScreen(n=4096, pixel_scale=0.005, r0=0.15, device="gpu")
phase = gen.generate(64)              # cupy.ndarray on the GPU
host = pyturb.to_numpy(phase)         # copy back when you need it
```

## Installation

```bash
pip install pyturb            # CPU (NumPy + SciPy)
pip install pyturb[cuda12]    # + CuPy for CUDA 12.x
pip install pyturb[cuda11]    # + CuPy for CUDA 11.x
```

## What it simulates

| Class | Statistics | Use case |
|---|---|---|
| `PhaseScreen` | Kolmogorov (`L0=inf`) or von Kármán, FFT method with subharmonic low-frequency correction | Monte-Carlo ensembles: PSF statistics, error budgets, training data |
| `InfinitePhaseScreen` | Von Kármán, conditional-Gaussian row extrusion (Assémat & Wilson 2006) | Temporal / closed-loop AO simulation with Taylor frozen flow, unbounded duration |

Both return phase in **radians** at the wavelength at which `r0` is defined.
Helpers are included for the usual bookkeeping:

```python
r0 = pyturb.r0_from_seeing(0.8)                       # arcsec @ 500 nm -> m
r0_k = pyturb.r0_at_wavelength(r0, 500e-9, 2.2e-6)    # r0 ~ lambda^(6/5)
r, D = pyturb.structure_function(phase, pixel_scale=0.01)
```

## Fidelity

Plain FFT screens famously under-represent low spatial frequencies
(tip/tilt), because a periodic grid has no power below `1/(n * pixel_scale)`.
`pyturb` adds subharmonic levels (Lane et al. 1992; Johansson & Gavel 1994) —
8 by default — and integrates the steeply convex turbulence PSD over each
low-frequency cell rather than sampling it at cell centres. The resulting
ensemble structure function matches the Kolmogorov prediction
`D(r) = 6.88 (r/r0)^(5/3)` to about 1%. This is enforced by the test suite,
as are the analytic von Kármán phase variance (`0.0863 (L0/r0)^(5/3)`) and
the temporal statistics of extruded screens.

## Performance notes

- `generate(count)` batches all screens through a single FFT call, and each
  complex FFT yields two independent screens — prefer one big batch over a
  Python loop.
- Screens default to `float32`, the sweet spot for GPU throughput and ample
  precision for AO work; pass `dtype="float64"` if you want more.
- `InfinitePhaseScreen.step()` costs two small matrix-vector products, so
  it is fast even on CPU; the expensive covariance setup happens once in
  the constructor.

## API summary

```text
PhaseScreen(n, pixel_scale, r0, L0=25.0, subharmonics=8,
            seed=None, device="cpu", dtype="float32")
    .generate(count=None) -> (n, n) or (count, n, n) radians

InfinitePhaseScreen(n, pixel_scale, r0, L0=25.0, stencil_rows=2,
                    seed=None, device="cpu", dtype="float32")
    .screen               -> current (n, n) screen
    .step(steps=1)        -> screen after advancing the wind

phase_covariance(r, r0, L0)         # von Kármán covariance, rad^2
structure_function(phase, pixel_scale=1.0, max_separation=None)
r0_from_seeing(seeing, wavelength=500e-9)
seeing_from_r0(r0, wavelength=500e-9)
r0_at_wavelength(r0, wavelength_in, wavelength_out)
to_numpy(array)                     # device -> host copy
```

## References

- McGlamery, B. L. (1976), *Computer simulation studies of compensation of
  turbulence degraded images*, Proc. SPIE 74.
- Lane, R. G., Glindemann, A. & Dainty, J. C. (1992), *Simulation of a
  Kolmogorov phase screen*, Waves in Random Media 2, 209.
- Johansson, E. M. & Gavel, D. T. (1994), *Simulation of stellar speckle
  imaging*, Proc. SPIE 2200.
- Assémat, F., Wilson, R. W. & Gendron, E. (2006), *Method for simulating
  infinitely long and non stationary phase screens with optimized memory
  storage*, Optics Express 14, 988.
- Schmidt, J. D. (2010), *Numerical Simulation of Optical Wave Propagation
  with Examples in MATLAB*, SPIE Press.

## License

MIT

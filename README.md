# pyturb

[![CI](https://github.com/jacotay7/pyturb/actions/workflows/ci.yml/badge.svg)](https://github.com/jacotay7/pyturb/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyturb.svg)](https://pypi.org/project/pyturb/)
[![Python](https://img.shields.io/pypi/pyversions/pyturb.svg)](https://pypi.org/project/pyturb/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Fast, GPU-optional atmospheric turbulence for adaptive optics.**

`pyturb` generates the optical path differences (OPD) that an adaptive-optics
system sees through the atmosphere: full **layered turbulence** for a
representative sky, with per-layer wind, **frozen-flow time evolution**,
off-axis directions, and standard site profiles. It runs on NumPy by default
and switches to CUDA (via CuPy) with a single argument.

```python
import pyturb

# A representative atmosphere in one line: a named site profile scaled to a
# chosen seeing, at 30 deg zenith, for an 8 m telescope sampled at 512 px.
atm = pyturb.Atmosphere.from_profile(
    "paranal-median", seeing=0.8, zenith_angle=30, diameter=8.0, n=512, seed=1
)
print(atm.r0, atm.theta0, atm.tau0)     # Fried param, isoplanatic angle, tau0

# Closed-loop: OPD [m] frames under frozen-flow wind at a 1 kHz loop rate.
for t, opd in atm.frames(dt=1e-3, steps=2000):
    ...                                  # (512, 512) OPD in metres

# Off-axis / tomography: OPD toward several directions from the same volume.
# Set field_of_view so off-axis footprints sample real (non-wrapped) turbulence.
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8,
                                     field_of_view=30, n=512, seed=1)
opds = atm.opd(t=0.0, directions=[(0, 0), (10, 0), (0, 10)])  # arcsec offsets

# Boiling: add temporal decorrelation on top of frozen flow (per-layer tau).
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8,
                                     tau_boil=0.2, seed=1)     # seconds

# Monte-Carlo: a batch of independent integrated-atmosphere OPDs.
ensemble = atm.sample(256)               # (256, 512, 512)
```

GPU acceleration is one keyword away — everything else is identical:

```python
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, device="gpu")
for t, opd in atm.frames(dt=1e-3, steps=2000):
    ...                                  # cupy arrays, ~800 fps at 512^2
host = pyturb.to_numpy(opd)              # copy back when you need it
```

OPD is returned in **metres** and is achromatic; pass `wavelength=` to any
output method to get phase in radians at that wavelength instead.

### Two frozen-flow engines

`frames()` / `opd()` can be driven by either of two engines (`sample()` is
unaffected):

```python
# Default: spectral shift-theorem — exact sub-pixel, all layers in one FFT,
# fastest (~800 fps at 512^2 on GPU), but the screen is periodic.
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8)

# Extruder — Assémat-Wilson row extrusion in a wind-aligned frame with rotated
# sub-pixel sampling: unbounded and NON-periodic, the right choice for long
# closed-loop runs. Any wind direction, any v*dt, ~120 fps at 512^2 on GPU.
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, engine="extrude")
```

Both produce identical statistics (von Kármán to ~1–2%); pick the extruder when
a repeating screen would bias a long run, the spectral engine when you want raw
speed and the period is longer than your simulation.

### Lower-level building blocks

The single-layer generators the atmosphere is built from are public too:

```python
# Independent Kolmogorov / von Kármán screens (FFT method + subharmonics)
gen = pyturb.PhaseScreen(n=512, pixel_scale=0.01, r0=0.15, L0=25, seed=1)
phase = gen.generate(100)     # (100, 512, 512) radians, one FFT batch

# Endless frozen-flow screen (row extrusion, unbounded, non-periodic)
layer = pyturb.InfinitePhaseScreen(n=256, pixel_scale=0.01, r0=0.15, L0=25, seed=1)
for _ in range(1000):
    phase = layer.step()      # advance the wind by one pixel per step
```

## Installation

```bash
pip install pyturb            # CPU (NumPy + SciPy)
pip install pyturb[cuda12]    # + CuPy for CUDA 12.x
pip install pyturb[cuda11]    # + CuPy for CUDA 11.x
```

## What it simulates

| Class | What it produces | Use case |
|---|---|---|
| `Atmosphere` | Many layers summed into pupil OPD, per-layer wind, frozen-flow evolution, off-axis directions, site profiles | The complete AO atmosphere: closed-loop OPD, tomography inputs, Monte-Carlo ensembles |
| `PhaseScreen` | One layer: Kolmogorov (`L0=inf`) or von Kármán FFT screen with subharmonic low-frequency correction | Building block; independent single-layer ensembles |
| `InfinitePhaseScreen` | One layer: von Kármán row extrusion (Assémat & Wilson 2006) | Unbounded, non-periodic frozen-flow layer |

### Profiles

`Atmosphere.from_profile(name, ...)` accepts representative, citable profiles —
`"paranal-median"`, `"mauna-kea"`, `"keck"`, `"las-campanas"`, `"hv57"`
(Hufnagel–Valley 5/7), plus `"single-layer"` / `"two-layer"` for teaching and
quick tests. Build your own from a continuous `Cn2(h)` model:

```python
import numpy as np
h = np.geomspace(1, 25000, 4096)
# method="equivalent" (default) conserves theta0 AND tau0 exactly (Fusco 1999)
layers = pyturb.discretize_cn2(h, pyturb.hufnagel_valley(h), n_layers=10)
atm = pyturb.Atmosphere(layers, seeing=0.8, diameter=8.0, n=512)
```

The atmosphere reports the derived quantities every AO error budget needs:
integrated `r0`, `seeing`, isoplanatic angle `theta0`, coherence time `tau0`,
and `greenwood_frequency`. Bookkeeping helpers:

```python
r0 = pyturb.r0_from_seeing(0.8)                       # arcsec @ 500 nm -> m
r0_k = pyturb.r0_at_wavelength(r0, 500e-9, 2.2e-6)    # r0 ~ lambda^(6/5)
phase = pyturb.opd_to_phase(opd, wavelength=2.2e-6)   # OPD [m] -> phase [rad]
r, D = pyturb.structure_function(phase, pixel_scale=0.01)

pyturb.save("opd.fits", atm.opd(), **atm.metadata)    # or .npz; load() returns
data, meta = pyturb.load("opd.fits")                  #   (array, metadata)
```

OPD is achromatic by default. For the small (~2% visible→K) chromatic term from
air dispersion, pass `dispersion="edlen"`; then `opd(..., wavelength=λ)` scales
the path by the dry-air refractivity ratio (`pyturb.air_refractivity`).

## Diagnostics and advanced options

`pyturb.analysis` turns screens into the usual AO diagnostics: `zernike_basis` /
`zernike_decompose` (Noll-ordered), `noll_variance` / `noll_residual_variance`
(Kolmogorov theory to validate against), `temporal_psd` + `fit_power_law`, and
`differential_variance` for angular decorrelation.

More knobs, all off by default:

- **LGS cone effect** — `Atmosphere(engine="extrude", lgs_altitude=90e3)`
  magnifies each layer by `(1 − h/H_LGS)` for a finite-range beacon.
- **Non-Kolmogorov turbulence** — `PhaseScreen(power_law=…, inner_scale=…)` for
  a general PSD slope (`D(r) ~ r^{power_law−2}`) and a modified-von-Kármán inner
  scale.
- **Threaded CPU FFTs** — `pyturb.set_fft_workers(-1)` uses all cores (GPU
  unaffected).

Dropping pyturb output into HCIPy, poppy, or a DM-fitting loop:
see [`docs/interop.md`](docs/interop.md).

## Performance

Frozen-flow evolution batches all layers through a single FFT per frame. On an
RTX 5090, a 9-layer Paranal atmosphere (`benchmarks/bench_frames.py`):

| screen | GPU frames/s | GPU screens/s | CPU frames/s |
|---|---|---|---|
| 256² | ~840 | ~1350 | ~120 |
| 512² | ~820 | ~1100 | ~26 |
| 1024² | ~500 | ~350 | ~4 |

## How pyturb compares

pyturb sits alongside three excellent, well-established tools. They were built
for different jobs — aotools is an AO maths toolbox, soapy a full AO *system*
simulator, HCIPy a diffraction-propagation framework — so this is about picking
the right tool, not a winner. A detailed, honest, method-by-method write-up
(and what we learned from their source) is in
[`docs/comparison.md`](docs/comparison.md); raw benchmark tables are in
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

| | pyturb | aotools | soapy | HCIPy |
|---|:---:|:---:|:---:|:---:|
| GPU (CuPy) backend | ✅ | — | — | — |
| Batched Monte-Carlo screens | ✅ | — | — | — |
| Sub-pixel, any-direction flow | ✅ | — | ✅ | ✅ |
| Boiling (temporal decorrelation) | ✅ | — | — | — |
| OPD in metres (achromatic) | ✅ | — | — | — |
| Off-axis / tomography directions | ✅ | — | ◐ | ✅ |
| Named site profiles | ✅ | — | — | ✅ |
| Unbounded (non-periodic) screens | ✅ | ✅ | ✅ | ✅ |
| Scintillation (Fresnel) | non-goal | — | — | ✅ |

Measured head-to-head on an RTX 5090 (8 m pupil, 512²):

- **Monte-Carlo generation** — pyturb produces **14,000 independent 512²
  screens/s** on the GPU (55,000 at 256²) by drawing two screens per FFT and
  batching the stack; that is **~1000× the pure-Python FFT loops** in
  aotools/soapy, and ~13× even on one CPU core.
- **Frozen flow** — a full **9-layer 512² atmosphere at ~800 fps** on GPU via
  the spectral shift-theorem engine (exact sub-pixel, any direction, all layers
  in one FFT).
- **Accuracy** — pyturb's structure function tracks von Kármán theory to
  **~1%** systematic bias (measured on large ensembles; comparable to HCIPy
  and soapy, and lower than aotools). At the ensemble sizes a quick benchmark
  can afford, this metric is noisy enough that the exact ranking between
  tools moves run to run — see
  [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md#3-structure-function-accuracy--fractional-rms-error-vs-von-kármán-lower-is-better)
  for the full numbers with uncertainties and methodology.

**Where the others are stronger, honestly:** single-layer integer-pixel CPU
stepping in aotools/soapy is still faster *per step* than pyturb's per-layer
extruder sampling at small `n`. aotools adds tomographic reconstructors, soapy a
complete AO system, HCIPy Fresnel propagation and scintillation — all outside
pyturb's scope. The one capability all three had that pyturb lacked — *truly
unbounded, non-periodic* screens via the Assémat–Wilson extruder — is now
implemented as `engine="extrude"` (any wind direction, sub-pixel, GPU; ~120 fps
for a non-periodic 9-layer 512² atmosphere), as are FITS/npz I/O and
moment-conserving profile compression. See
[`docs/comparison.md`](docs/comparison.md#what-we-learned--and-what-were-adopting).

Run the comparison yourself: `python benchmarks/bench_compare.py --json out.json`.

## Fidelity

pyturb's turbulence is checked against analytic theory, not just asserted —
`python validation/validate.py` regenerates this gallery (structure function,
Zernike spectrum vs Noll, temporal PSD, angular decorrelation, extruder
stationarity), each with a PASS/FAIL tolerance:

![validation gallery](docs/images/validation.png)

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
- `InfinitePhaseScreen` extrudes into a **ring buffer**, so a step costs two
  small matrix-vector products (not a whole-screen copy) and memory stays
  bounded no matter how long the run; the expensive covariance setup happens
  once in the constructor. Use `.step()` for whole-pixel wind travel or
  `.advance(pixels)` for exact **sub-pixel** motion (`v*dt/pixel_scale`),
  interpolated with `interp="cubic"` (default) or `"linear"`.

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

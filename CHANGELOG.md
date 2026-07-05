# Changelog

All notable changes to pyturb are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

## [0.2.0] — unreleased

The "atmosphere" release: pyturb goes from a phase-screen library to a complete,
benchmarked, GPU-native AO atmosphere.

### Added

- **`Atmosphere`** — layered atmosphere summed to pupil OPD, with per-layer
  wind, airmass/zenith scaling, off-axis `directions=`, field-of-view
  oversizing, and integrated `r0` / `seeing` / `theta0` / `tau0` /
  `greenwood_frequency`. Built from named profiles via `from_profile`.
- **Two frozen-flow engines.** `engine="spectral"` (default): exact sub-pixel
  shift-theorem translation, all layers in one FFT, but periodic. Boiling
  (`tau_boil`) via a spectral AR(1). `engine="extrude"`: Assémat–Wilson row
  extrusion in a wind-aligned ring buffer with rotated sub-pixel sampling —
  unbounded, non-periodic, any wind direction.
- **`InfinitePhaseScreen`** rewritten with a ring buffer and sub-pixel
  `advance()` (Catmull-Rom / linear), memory bounded over arbitrarily long runs.
- **Named profiles**: `paranal-median`, `mauna-kea`, `keck`, `las-campanas`,
  `cerro-pachon`, `armazones`, `hv57`, `single-layer`, `two-layer`;
  `discretize_cn2(method=...)` with moment-conserving `"equivalent"`
  (conserves `theta0` and `tau0`), `"centroid"`, and `"optimal_grouping"` —
  the last chooses bin edges by dynamic programming to minimise the
  Cn²-weighted within-group spread of `h^{5/3}`, for MCAO/tomography layer
  compression (Saxenhuber et al. 2017).
- **`Atmosphere.evolve(dt)`** — single-step, in-seconds frozen-flow stepper
  (mirrors HCIPy's `evolve_until`); repeated calls reproduce `frames(dt)`.
- **`interp="lanczos"`** (6-tap Lanczos-3) sub-pixel readout on the extruder and
  `InfinitePhaseScreen` — a flatter sub-Nyquist kernel that cuts the extruder's
  finest-scale travel-phase flicker (~10% → ~3.5%) and structure-function
  deficit versus the default cubic, with no change to the extrusion statistics.
- **`pyturb.analysis`**: Zernike basis/decomposition, Noll (1976) mode
  variances, temporal PSD + power-law fit, angular decorrelation.
- **I/O**: `pyturb.save` / `pyturb.load` for `.npz` and FITS (optional astropy)
  with metadata; `Atmosphere.metadata`.
- **Chromatic OPD**: `dispersion="edlen"` (dry air) and `dispersion="ciddor"`
  with a `wet_fraction` water-vapour term for the mid-IR/interferometric
  "wet–dry" problem; `pyturb.air_refractivity` and
  `pyturb.water_vapour_refractivity`.
- **LGS cone effect**: `Atmosphere(engine="extrude", lgs_altitude=...)`.
- **Non-Kolmogorov spectra**: `PhaseScreen(power_law=..., inner_scale=...)`.
- **Threaded CPU FFT**: `pyturb.set_fft_workers()`.
- **GPU test path**: GPU tests marked `@pytest.mark.gpu`, run with
  `pytest --run-gpu` (a `device` fixture parameterises statistics tests over
  CPU/GPU); `.github/workflows/gpu.yml` runs them on a self-hosted GPU runner.
- **`pyturb.benchmark()`** convenience; `benchmarks/bench_compare.py`
  head-to-head vs aotools/soapy/HCIPy; `validation/validate.py` gallery.
- Docs: `docs/comparison.md`, `docs/interop.md`, `docs/validation.md`; examples
  gallery (`examples/01`–`05`).
- `py.typed` marker; version single-sourced from package metadata.

### Changed

- `Atmosphere` output is **OPD in metres** (achromatic); pass `wavelength=` for
  phase. `PhaseScreen` / `InfinitePhaseScreen` still return radians.
- `discretize_cn2` default method is now `"equivalent"` (moment-conserving).

### Performance

- **Spectral engine: collapse the layer axis before the transform.** The
  inverse FFT and subharmonic outer product are linear and shared across
  layers, so `Atmosphere._integrate` now sums the shifted spectra to one
  `(n, n)` array and inverse-FFTs *once* instead of once per layer (and sums
  each subharmonic level's `3x3` coefficients before the shared basis product).
  Identical output; measured on an RTX 5090, 9-layer paranal-median: **CPU
  25 → 87 fps at 512² (3.4×)**, **GPU 865 → 1232 fps (1.4×)**.
- **Fused GPU extruder readout.** Every layer's ring buffer is now a slab of
  one contiguous `(L, cap, W)` array and the GPU pupil readout is a single
  batched bicubic gather over all layers (the CPU keeps the per-layer loop it
  prefers, chosen by backend). Measured on an RTX 5090, 9-layer
  paranal-median: **121 → 870 fps at 256² (7.1×)** and **118 → 290 fps at
  512² (2.5×)**; CPU throughput unchanged.

## [0.1.0]

- Initial release: `PhaseScreen` (FFT + subharmonics) and `InfinitePhaseScreen`
  (Assémat–Wilson extrusion), NumPy/CuPy backends, structure-function tests.

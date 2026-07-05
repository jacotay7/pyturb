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
  `hv57`, `single-layer`, `two-layer`; `discretize_cn2(method="equivalent")`
  moment-conserving compression (conserves `theta0` and `tau0`).
- **`pyturb.analysis`**: Zernike basis/decomposition, Noll (1976) mode
  variances, temporal PSD + power-law fit, angular decorrelation.
- **I/O**: `pyturb.save` / `pyturb.load` for `.npz` and FITS (optional astropy)
  with metadata; `Atmosphere.metadata`.
- **Chromatic OPD**: `dispersion="edlen"` and `pyturb.air_refractivity`.
- **LGS cone effect**: `Atmosphere(engine="extrude", lgs_altitude=...)`.
- **Non-Kolmogorov spectra**: `PhaseScreen(power_law=..., inner_scale=...)`.
- **Threaded CPU FFT**: `pyturb.set_fft_workers()`.
- **`pyturb.benchmark()`** convenience; `benchmarks/bench_compare.py`
  head-to-head vs aotools/soapy/HCIPy; `validation/validate.py` gallery.
- Docs: `docs/comparison.md`, `docs/interop.md`, `docs/validation.md`; examples
  gallery (`examples/01`–`05`).
- `py.typed` marker; version single-sourced from package metadata.

### Changed

- `Atmosphere` output is **OPD in metres** (achromatic); pass `wavelength=` for
  phase. `PhaseScreen` / `InfinitePhaseScreen` still return radians.
- `discretize_cn2` default method is now `"equivalent"` (moment-conserving).

## [0.1.0]

- Initial release: `PhaseScreen` (FFT + subharmonics) and `InfinitePhaseScreen`
  (Assémat–Wilson extrusion), NumPy/CuPy backends, structure-function tests.

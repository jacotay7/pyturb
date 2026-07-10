# Changelog

All notable changes to pyturb are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Boiling on the non-periodic engine.** `tau_boil` now works with
  `engine="extrude"`, not just `"spectral"`. Each step blends the ring buffer
  toward a fresh, independently extruded screen (`buf = a·buf + √(1−a²)·fresh`,
  `a = exp(−dt/tau)`), so temporal decorrelation composes with frozen flow while
  the spatial covariance — and hence `r0` — is preserved and the screen stays
  non-periodic. Unlike the spectral engine's per-mode boiling, the extruder
  decorrelates every spatial scale at the single `tau_boil` rate (real space has
  no per-mode handle); staying non-periodic costs a modest deficit in the
  largest-scale power of the boiled screen, and re-extruding the fresh window
  makes a boiling frame markedly costlier than a frozen one. `lgs_altitude` now
  composes with boiling on the extruder too (the cone acts on readout geometry,
  boiling on the stored turbulence). On the GPU, boiling's fresh-screen
  extrusion is batched across layers (one pair of matmuls per row instead of a
  Python loop per layer per row), a several-fold speedup over the naive
  per-layer path. Constructing `Atmosphere(engine="extrude", tau_boil=...)` now
  raises `ExtrudeBoilingPerformanceWarning`, noting that this combination is
  still markedly slower than `engine="spectral"` boiling.

### Changed

- **`Atmosphere.sample()` is ~L× faster for shared-outer-scale profiles.** It
  now draws one aggregate phase screen per distinct `L0` rather than one per
  layer: independent von Kármán screens with the same PSD shape add exactly
  (`r0_agg^{-5/3} = Σ r0_i^{-5/3}`), so summing them is distributionally
  identical to summing per-layer draws. A 9-layer `paranal-median` (uniform
  `L0`) `sample()` is ~9× faster on both CPU and GPU (measured 9.0×/9.1× at
  512², ~30k screens/s on GPU). Layers that share `L0` are pooled; a profile
  with mixed `L0` uses one screen per distinct value. Reproducibility note: for
  a profile with **multiple layers sharing an `L0`**, `sample()` now consumes a
  different RNG stream, so a fixed `seed` yields a different (but
  statistically identical) realisation than before; single-layer-per-`L0`
  profiles are unchanged (the layer's own generator is reused).
- **Batched multi-direction tomography on the GPU.** `opd(directions=[...])`
  on `engine="spectral"` (without the LGS cone) now integrates all directions
  through one batched inverse FFT and one subharmonic matmul chain instead of a
  Python loop, ~1.8× faster at 512² by removing per-direction kernel-launch
  latency (bit-identical output). The CPU path keeps its per-direction fused
  loop, which is faster there than a batched transform.
- **Faster spectral LGS cone frames on the GPU.** The per-layer cone-zoom
  readout in `_integrate_lgs` (previously ~77% of the frame, a per-layer,
  per-tap Python loop) is now a handful of `take_along_axis` gathers batched
  over all layers and taps: ~3× at 256²/512² (275 → ~830 fps at 512²),
  bit-identical output. Gated to the GPU below a working-set threshold — on the
  CPU, and for large `(L, n, n_screen)` working sets on the GPU (≳1024²), the
  cache-friendlier per-layer loop is kept.

## [0.2.0] 

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
- **LGS cone effect**: `Atmosphere(lgs_altitude=...)` on **both** engines — the
  extruder samples its ring buffer on a magnified grid, the spectral engine
  zoom-resamples each layer's screen about the pupil centre by the same factor.
  On the spectral engine the cone now **composes with `tau_boil` boiling**,
  closing the previous cone/boiling mutual exclusivity.
- **Non-Kolmogorov spectra**: `PhaseScreen(power_law=..., inner_scale=...)`.
- **Threaded CPU FFT**: `pyturb.set_fft_workers()`.
- **GPU test path**: GPU tests marked `@pytest.mark.gpu`, run with
  `pytest --run-gpu` (a `device` fixture parameterises statistics tests over
  CPU/GPU); `.github/workflows/gpu.yml` runs them on a self-hosted GPU runner.
- **`pyturb.benchmark()`** convenience; `benchmarks/bench_suite.py`
  (per-use-case throughput sweep across CPU/GPU) and
  `benchmarks/bench_compare.py` head-to-head vs aotools/soapy/HCIPy;
  `validation/validate.py` gallery.
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
- **Batch all subharmonic levels into one matmul.** The low-frequency
  subharmonic correction shares one `(3, n)` sinusoid basis across levels and
  layers, so `Atmosphere._integrate`, `PhaseScreen.generate`,
  `FourierFlowScreen.translate` and the boiling update now evaluate every level
  in a couple of batched matmuls instead of a Python loop over levels (which was
  launch-latency bound on the GPU — ~78% of a frame). Identical output.
  Measured on an RTX 5090, 9-layer paranal-median frozen flow: **1,232 → 3,004
  fps at 512² GPU**, **3,234 fps at 256²**; single-layer Monte-Carlo generation
  **14,000 → 31,000 screens/s at 512²** (55,000 → 108,000 at 256²); CPU
  spectral **~87 → ~130 fps at 512²** before the accel extra below.
- **Fused GPU/CPU extruder readout kernel.** Every layer's ring buffer is a slab
  of one contiguous `(L, cap, W)` array, and the per-frame rotated, sub-pixel,
  per-layer-wind-shifted pupil gather runs in a single pass for the `"cubic"`
  and `"lanczos"` interpolators: a hand-written CUDA kernel on the GPU and a
  fused `prange` Numba kernel on the CPU (see the accel extra below), bit-exact
  with the previous tap-broadcast gather. Measured on an RTX 5090, 9-layer
  paranal-median: **121 → 4,484 fps at 256² GPU (37×)**, **118 → 1,730 fps at
  512² (15×)**, **50 → 602 fps at 1024²**; the `"lanczos"` readout is now a
  fused kernel too (~334 fps at 512² GPU, from ~120 fps).
- **Optional Numba CPU acceleration (`pip install pyturb[accel]`).** The CPU
  frozen-flow hot paths — the spectral engine's fused layer sum and the
  extruder's fused bicubic/Lanczos readout — run through Numba when it is
  importable, with a NumPy fallback otherwise (identical results to float
  round-off). Measured on a 32-core CPU, 9-layer paranal-median: spectral
  **~130 → 270 fps at 512²**, extruder **6 → 164 fps at 512² (27×)** and
  **28 → 966 fps at 256² (34×)**.
- **Geometry-derived extruder buffer sizing.** The shared ring buffer is now
  sized to the largest along-wind/off-axis requirement actually present among
  the layers (each layer's own wind direction and altitude), not a blanket
  every-layer-at-45-degrees-and-max-altitude assumption. Measured (n=512): a
  ground-layer-only atmosphere with `field_of_view=30"` uses ~68% less buffer
  memory; an axis-aligned atmosphere uses ~41% less even at
  `field_of_view=0`. Never worse than before.

## [0.1.0]

- Initial release: `PhaseScreen` (FFT + subharmonics) and `InfinitePhaseScreen`
  (Assémat–Wilson extrusion), NumPy/CuPy backends, structure-function tests.

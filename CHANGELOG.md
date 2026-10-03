# Changelog

All notable changes to pyturb are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Fourteen traceable Paranal profiles**, `paranal-p01` ... `paranal-p14`,
  from Garcia-Rissmann et al. (2015), MNRAS 448, 2594 (Tables 2-3): seeing
  classes 0.4"-1.4", each good/median/bad, with their published r0, tau0, mean
  height and probability of occurrence in the new
  `ProfileInfo.conditions`. Tests recompute the paper's mean height and tau0
  from the stored layers (#18).
- **31 more traceable site profiles** from refereed papers, each transcribed
  verbatim and checked against the rendered tables, with tests that recompute
  every published integrated quantity: TMT site testing at Maunakea 13N,
  Armazones, Tolar, Tolonchar and San Pedro Martir (Els et al. 2009,
  good/typical/bad), Cerro Pachon (Tokovinin & Travouillon 2006), Siding Spring
  (Goodwin et al. 2013, nine GL x FA combinations with modelled winds),
  Sutherland (Catala et al. 2013) and Mt Graham (Masciadri et al. 2010).
- `Atmosphere.from_profile` uses a traceable profile's published r0 when no
  `r0`/`seeing` is given, accepts `wind=`/`wind_direction=`, and warns for
  profiles whose source publishes no winds (their layers are static).
  `pyturb.with_wind(layers, speed, direction)` assigns winds to any profile.

### Fixed

- A layer with `cn2_fraction=0` no longer crashes `Atmosphere` (its Fried
  parameter would be infinite); zero-weight layers are left out of the
  simulation, with a per-layer `tau_boil` kept aligned.

## [1.1.0] - 2026-10-02

Highlights: an opt-in fix for pupil-sized periodic screens (`oversample`),
CUDA-graph GPU frames (5-8x on modest hosts), batched time series, per-source
LGS/NGS directions, `opd_at`, a `pyturb.theory` module, `from_cn2`,
replayable configurations, GPU selection, an HCIPy layer adapter, and tested
guides. Behaviour changes to be aware of: Python 3.10 is the minimum,
`analysis.temporal_psd` now tapers with a Hann window by default, invalid
per-layer inputs now raise, and `threadpoolctl` is a new dependency.

### Added

- **`device="gpu:N"`** picks a GPU on multi-GPU machines; every public
  method of `Atmosphere`, `PhaseScreen`, `FourierFlowScreen` and
  `InfinitePhaseScreen` runs with that GPU current and leaves the caller's
  current device unchanged (#20).
- **Replayable configuration**: `Atmosphere.to_config()` /
  `Atmosphere.from_config()`, and `metadata["config"]` (written by
  `pyturb.save`) holds every constructor input and the layer table, so a saved
  OPD made with an integer seed can be regenerated exactly (#20).
- **`pyturb.interop.HCIPyLayer`**: an Atmosphere presented as an HCIPy
  atmospheric layer (`t`, `layer(wavefront)`, `evolve_until`, `phase_for`,
  `reset`), e.g. for pyRTC's HCIPy simulator; plus a DLPack (PyTorch/JAX)
  recipe in the interop docs (#20).
- **Per-direction sources**: `opd(directions=[(thx, thy, altitude), ...])`
  gives each direction its own source range — a laser guide star at
  `altitude` or `None` for a star at infinity — so an LGS asterism, NGS and
  science directions read one turbulence realisation in one call, on both
  engines (#19).
- **`Atmosphere.opd_at(x, y, t, direction, altitude)`** samples the
  atmosphere at arbitrary pupil-plane coordinates (DM actuators,
  sub-apertures, sparse or multi-aperture layouts); on the pupil grid it
  reproduces `opd()` exactly. Reach beyond the pupil with `oversample`
  (spectral) or `field_of_view` (extrude) (#19).
- **`Atmosphere.from_cn2(heights, cn2, ...)`** builds an atmosphere straight
  from a measured or model Cn²(h) profile (compressed with `discretize_cn2`),
  taking the profile's own `r0` unless `r0`/`seeing` is given and recording a
  `source` in `metadata` (#18).
- **`discretize_cn2(wind_direction=...)`**: a scalar, a per-input-grid array
  (Cn²-weighted circular mean per bin) or one value per output layer, instead
  of every layer at 0 degrees (#18).
- **`pyturb.theory`**: reference curves for checking any simulation, at
  finite outer scale as well as Kolmogorov — `structure_function`,
  `zernike_variance` (any Noll mode, any `L0`; reproduces Noll's table for
  `L0=inf`), `image_motion_variance` (one-axis tilt, arcsec²),
  `seeing_fwhm` (Tokovinin 2002 outer-scale correction), `pixel_temporal_psd`
  and `zernike_temporal_psd` (Conan, Rousset & Madec 1995; along- vs
  across-wind aware), and `differential_phase_variance` through a profile.
  Each is tested against Noll's table or an exact identity and against
  pyturb's own simulations; `validation/validate.py` uses them (#17).
- **`opd(t=array)`** returns a stack of frames for many times (and
  directions) at once; on the GPU the spectral engine evaluates them in one
  batched transform (~26,000 frames/s at 256² and ~6,700 at 512² on an RTX 4060
  for the 9-layer `paranal-median`). Boiling is not applied to random-access
  times (#15).
- **`Atmosphere(cuda_graph=True)`**: on the GPU, each spectral frame
  (`frames`, `evolve`, single-direction `opd`) replays a captured CUDA graph
  built from fused layer-sum and subharmonic kernels with device-side float64
  phasors. Frames are bit-identical to ordinary execution (`cuda_graph=False`),
  boiling included; capture falls back to ordinary execution if unsupported.
  On an RTX 4060 with an Arm host, 9-layer spectral frames go from ~830 to
  ~6,300 fps at 256², ~820 to ~4,300 at 512² and ~460 to ~1,300 at 1024²
  (RTX A400: ~830/500/130 to ~4,800/1,300/330). Multi-direction spectral
  frames use the same fused kernels (#15).
- **`Atmosphere(oversample=...)`**: FFT screens for `sample()` and the spectral
  engine can be made larger than the pupil (default 1, unchanged). On a
  pupil-sized screen the FFT periodicity puts opposite pupil edges next to each
  other, so separations beyond ~D/4 come out low (structure function about
  −13% at D/2, ~0.65x near 0.9 D; tilt ~10-15% low; astigmatisms split
  ~0.6x/1.6x Noll). `oversample=4` matches von Kármán to ~2% rms out to 0.9 D.
  It also multiplies `time_to_wrap`, at the cost of an FFT `oversample**2`
  larger (RTX 4060, 512²: ~5,300 to ~1,360 spectral fps at 2x). Validation gains a large-scale structure
  function check out to 0.9 D (#3).

### Changed

- **The CPU extruder batches row extrusion across layers.** Layers sharing an
  outer scale share the extrusion matrices, so each row step is two matrix
  products for all of them instead of two mat-vecs per layer, and the CPU
  extrusion BLAS calls run single-threaded (`threadpoolctl`, a new
  dependency): a threaded BLAS gains nothing on these small memory-bound
  products and its spinning threads starve the Numba readout. 9-layer
  `paranal-median` at 512² on 16 Arm cores: ~42 to ~98 fps (no change at 256²,
  where the readout dominates). Each layer keeps its own random stream; frames
  agree with per-layer extrusion to ~1e-14. `InfinitePhaseScreen` and CPU
  extruder boiling use the same single-threaded BLAS (#16).
- Spectral boiling updates the stored spectra in place.
- **`analysis.temporal_psd` tapers with a Hann window by default**
  (`window="hann"`; `window=None` gives the previous untapered periodogram, and
  any `scipy.signal.get_window` name or an explicit array also works). The
  untapered periodogram leaks power upward on steep frozen-flow spectra: on a
  single pupil pixel it read slope −2.47 and 1.7x the theoretical level
  (−2.18 and 5.9x on the extruder) against −8/3 and 1x; with the taper,
  −2.63/1.2x and −2.81/0.8x. PSD values from this function change (#6).
- **Python 3.10 is the minimum**; CI tests 3.10, 3.12, 3.13 and 3.14, and the
  package declares per-version classifiers plus Documentation/Source/Changelog
  URLs (#13).
- **Release tags are bare versions** (`1.1.0`), and the release workflow
  checks the tag matches `pyproject.toml` before publishing (#1).
- The self-hosted GPU workflow is removed: no runner exists, so it queued on
  every pull request until GitHub cancelled it. GPU paths are checked with a
  local `pytest --run-gpu`, as `CONTRIBUTING.md` now describes (#11).
- New `interop` extra (HCIPy, poppy) used to exercise the interop recipes.

### Fixed

- `profile_info("hv57").outer_scale` is 25 m (its layers' value) rather than
  `None`; `bench_suite.py` records the installed CuPy version (it looked up a
  non-existent "cupy" distribution).
- **Per-layer inputs are validated.** A NaN or infinite wind speed produced
  all-NaN OPD frames and a negative altitude a NaN `theta0`; non-finite layer
  values, negative altitudes or wind speeds, and non-positive per-layer `L0`
  now raise a `ValueError` naming the layer. Non-integer `subharmonics`
  (`PhaseScreen`) and `steps` (`frames`, `InfinitePhaseScreen.step`) are
  rejected instead of truncated, and asking an `engine="extrude"` atmosphere
  for an earlier time than it has reached explains the streaming clock and
  `reset()` (#7).
- **Long spectral runs keep float32 accuracy.** Shift-theorem displacements
  are reduced modulo the screen period in float64 before the float32 phasors
  are formed (and subharmonic phasors built in float64), on every spectral
  path and in `FourierFlowScreen.translate`. A frame 9,000 periods later now
  reproduces to round-off; before, the error grew to ~0.7% after 40 minutes of
  wind and ~8% after 7 hours (#8).
- **GPU extras work in a clean environment.** `pyturb[cuda12]` now installs
  `cupy-cuda12x[ctk]`: CuPy 14 compiles every kernel at runtime and fails with
  "Failed to find CUDA headers" unless the CUDA headers are present, which a
  fresh environment without a system CUDA toolkit does not have. Added a
  `pyturb[cuda13]` extra (`cupy-cuda13x[ctk]`) for CUDA 13 drivers, and a
  troubleshooting note to the install docs (#9).
- `pyturb.benchmark()` no longer emits `PeriodicWrapWarning`: its timing loop
  runs past the spectral screen period by design, which says nothing about
  the throughput being measured.
- `examples/02_closed_loop.py` runs on `engine="extrude"` (its 0.5 s run is
  longer than the spectral engine's `time_to_wrap`), and its "last frame"
  panel shows the last frame (it matched frame times by float equality, so it
  showed `t = 0` twice). The README/docs quickstart loops stay within
  `time_to_wrap` and explain why (#4).
- `examples/05_gpu_benchmark.py` skips the GPU rows when CuPy is installed but
  no CUDA device is usable.
- `docs/interop.md`: the poppy recipe passes an astropy-unit pixel scale
  (poppy 1.2 rejects a bare float), and the DM-fitting residual is restricted
  to the pupil. Every README/docs code block and `examples/01`–`05` now run in
  CI (`tests/test_docs.py`) (#10).

### Documentation

- New guides: **Choosing an engine** (decision table and the
  periodicity/oversample trade-offs), **Turbulence profiles** (named profiles
  with provenance, custom layers, `from_cn2`, discretisation methods),
  **Boiling, LGS and dispersion**, **Performance** and **Troubleshooting**;
  the changelog, roadmap and contributing pages are in the site (#14).
- **Tutorial notebook** `tutorials/01_atmosphere_to_psf.ipynb`: a layered
  atmosphere drives a minimal modal AO loop, checked against Noll's
  fitting-error floor and the Maréchal Strehl; committed executed and re-run
  in CI (#14).
- `benchmarks/RESULTS.md` adds pyturb 1.1 on an Arm workstation (RTX 4060,
  RTX A400, 16 Neoverse-N1 cores) with versioned artifacts; the README shows
  the 1.1 numbers next to the 1.0.0 RTX 5090 reference (#21).
- Validation: the structure-function band, the Zernike check (now per mode,
  tip/tilt included) and the temporal-PSD check (now both engines, tolerance
  0.6-1.6x instead of 1-3x) are documented with what they actually measure;
  the astigmatism split and the shallow PSD are explained as periodicity and
  spectral leakage rather than "sampling" and "finite screen" effects (#12).
- The wind convention is stated: `wind_direction`/`wind_vector` point where
  the wind blows *from* and the pattern moves along `-wind_vector`, on every
  engine; a test pins it. A new Conventions section in Concepts covers axes,
  wind, directions, line-of-sight quantities and units (#5).
- Quickstart sections for boiling, the LGS cone, dispersion and the
  `PhaseScreen`/`InfinitePhaseScreen` building blocks; the API reference lists
  the warning classes, `phase_covariance`, `profile_info`/`ProfileInfo`,
  `to_numpy`, `get_array_module` and `get_fft_workers` (#13).
- Corrected stale or inaccurate claims in the README, docs index,
  `RESULTS.md`, `ROADMAP.md`, `CONTRIBUTING.md` and the `structure_function`
  docstring (#12).

## [1.0.0] - 2026-07-09

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

- **Construction is now model/state separated.** Immutable validated
  configurations define phase-screen, atmosphere, and extrusion inputs before
  backend dispatch; mutable runtime objects own only derived geometry, random
  streams, spectra, and ring buffers. The two extrusion implementations share
  their stencil-preserving ring compaction primitive.
- **Validation evidence expanded.** The CI validation gallery now asserts and
  plots finite-screen structure-function ratios at 1–8 pixels and the scalar
  zenith-projection laws for `r0`, `theta0`, and layer range.
- **Benchmark evidence is versioned.** `bench_suite.py --json` records the
  invocation, Git revision, platform, dependency versions, device selection,
  timing budget, and batch size with every result. The 1.0.0 reference artifact
  is checked in alongside the published tables.
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

## [0.2.0] - 2026-07-05

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

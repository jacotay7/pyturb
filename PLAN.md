# pyturb Roadmap: From Phase-Screen Library to the Go-To AO Atmosphere

**Goal.** Make `pyturb` the complete, fastest, and easiest way to get atmospheric
OPDs into an adaptive-optics workflow: a user should be able to stand up a
representative multi-layer atmosphere in three lines, ask for OPD frames at a
loop rate, in any direction, on CPU or GPU, and trust the statistics.

## Status

**Milestone M1 (core) — implemented.** The layered atmosphere is in place and
tested on CPU and the RTX 5090:

- `pyturb.Atmosphere` / `Atmosphere.from_profile(...)` — many layers summed into
  pupil OPD, per-layer wind, airmass/zenith scaling, integrated `r0` / `seeing`
  / `theta0` / `tau0` / `greenwood_frequency`. (`src/pyturb/atmosphere.py`)
- `pyturb.profiles` — `Layer`, named profiles (`paranal-median`, `mauna-kea`,
  `hv57`, `single-/two-layer`), Hufnagel–Valley model, Bufton wind, and
  `discretize_cn2`. (`src/pyturb/profiles.py`)
- Spectral frozen-flow engine `FourierFlowScreen` — exact sub-pixel,
  arbitrary-direction translation via the shift theorem (verified equal to
  `np.roll` at integer pixels to 1e-15). (`src/pyturb/flow.py`)
- `frames(dt, steps)` closed-loop OPD, `opd(t, directions=...)` off-axis
  anisoplanatism/tomography, `sample(count)` Monte-Carlo — all returning OPD in
  metres, achromatic, with `opd_to_phase` / `phase_to_opd` helpers.
- Batched-over-layers hot path: one `(L, n, n)` FFT per frame → **~820 fps at
  512² (9 layers) on GPU** (4× over the naive loop). `benchmarks/bench_frames.py`.
- 16 new tests (`test_atmosphere.py`, `test_profiles.py`), CI gains Python 3.12,
  `InfinitePhaseScreen` dtype validation unified, `examples/atmosphere.py`.

**Milestone M2 (product) — implemented.** The spectral engine gained the two
features that make it a rigorous product, validated against theory:

- **Boiling** (Phase 2.3): per-layer temporal decorrelation via a spectral
  AR(1) (`tau_boil=`), active while stepping `frames()`. Ensemble temporal
  autocorrelation matches `exp(-dt/tau)` to a few percent and the spatial PSD
  (hence r0) is preserved (~25% GPU overhead when on).
- **Field-of-view oversizing** (Phase 3.3): `field_of_view=` grows the
  generated screens so off-axis footprints sample genuinely different,
  non-wrapped turbulence — making the M1 `directions=` anisoplanatism feature
  physically correct rather than wrap-limited. Verified as an exact lateral
  shift (off-axis == on-axis rolled, to 1e-6 relative); off-axis differential
  variance grows with angle as expected. Nearly free on GPU; default
  `field_of_view=0` keeps the fastest path (~800 fps at 512²) unchanged.

M2's other Phase 3 items (OPD in metres, `directions=`) were already delivered
in M1. LGS cone effect (finite-altitude guide star) is deferred — it needs
per-layer magnification/resampling that doesn't fit the batched-FFT fast path
and deserves its own careful pass (tracked under M5/stretch).

**Milestone M3 (proof) — benchmark matrix implemented.** Phase 4's head-to-head
comparison is done: `benchmarks/bench_compare.py` measures `pyturb` against
`aotools`, `soapy` and `HCIPy` on one 8 m pupil across three axes — generation
throughput (independent screens/s), frozen-flow frame rate (frames/s), and
structure-function accuracy vs von Kármán theory — plus a feature matrix.
Numbers captured on the RTX 5090 in `benchmarks/RESULTS.md`; a detailed,
honest method-by-method write-up (with lessons learned from reading the other
codebases) in `docs/comparison.md`; and a summary table in the README.
Headlines: pyturb generates **14k independent 512² screens/s** on GPU (~1000×
the aotools/soapy Python FFT loops), runs a **9-layer 512² atmosphere at
~800 fps**, and has the **best structure-function accuracy (~2%)** of the four.
Honest caveat surfaced rather than hidden: aotools/soapy `add_row` is faster per
single-layer CPU step, but only moves one integer pixel on a fixed axis (no
sub-pixel, no arbitrary direction, no GPU); and all three offer unbounded
non-periodic screens that pyturb's spectral engine does not. Concrete adoptions
identified for the roadmap: the Assémat–Wilson extruder (non-periodic path),
moment-conserving profile compression, and FITS I/O. Phase 5 validation depth
(Zernike
spectra, temporal PSDs, angular decorrelation, long-run stationarity, GPU-parity
CI) is still open.

**Known limitation carried forward:** the spectral engine is periodic (screen
repeats after `n·pixel_scale`); the non-periodic extruder path (Phase 2.1),
LGS cone, and the remaining milestones below are still open.

Target end-state API (the whole plan works backwards from this):

```python
import pyturb

atm = pyturb.Atmosphere.from_profile(
    "paranal-median",            # named Cn2/wind profile
    seeing=0.8,                  # scale profile to a chosen seeing (arcsec @ 500 nm)
    zenith_angle=30.0,           # deg; scales r0 and layer ranges
    diameter=8.0,                # telescope pupil (m)
    n=512,                       # pupil sampling (pixels across the screen)
    device="gpu", seed=1,
)

# Closed-loop: OPD frames in metres, frozen flow with per-layer wind
for t, opd in atm.frames(dt=1e-3, steps=2000):
    ...                          # (512, 512) OPD [m], on the GPU

# Monte-Carlo: batch of independent integrated-atmosphere OPDs
opds = atm.sample(count=256)     # (256, 512, 512)

# Off-axis / tomography: OPD toward multiple directions from the same volume
opds = atm.opd(t=0.0, directions=[(0, 0), (10, 0), (0, 10)])  # arcsec offsets
```

---

## Critical review after the M3 comparison — the adoption backlog

Reading `aotools`, `soapy` and `HCIPy` line-by-line (see `docs/comparison.md`)
made pyturb's real gaps concrete. The discipline here is to adopt what makes
pyturb a *better atmosphere generator* and to **refuse scope creep** into
WFS/DM/reconstruction (soapy owns full-system sim; aotools owns the AO maths
toolbox; HCIPy owns diffraction propagation). pyturb stays "the fastest,
GPU-native, statistically-careful atmosphere," and gets deeper, not wider.

Prioritised backlog (each item names the library to learn from):

**P0 — the architectural gap.**

1. **Non-periodic extruder (Assémat–Wilson), done properly.** This is the one
   capability all three competitors have and pyturb only half-has: the spectral
   engine *wraps* after `n·pixel_scale`, which is wrong for long closed-loop
   runs. Combine the best of each source: aotools' **fractal 2ⁿ stencil** and
   **numerical robustness** (`cho_factor`/`cho_solve` with an lstsq fallback on
   `LinAlgError`), HCIPy's **interpolation** for sub-pixel arbitrary-direction
   stepping and a real `evolve_until(t)` clock, and pyturb's own edge —
   **batched, GPU-resident extrusion** (extrude `k` rows in one matmul, all
   layers stacked) that none of them have. Ship the trade-off explicitly:
   spectral engine = fixed-period, fastest, sub-pixel-free; extruder =
   unbounded, memory-light, the default for multi-second loops.

   **Stage 1 — DONE (single-layer `InfinitePhaseScreen`).** Rewrote the extruder
   with a **ring buffer** (pre-allocated storage, window advanced by index — a
   step is now one small mat-vec instead of copying the whole `(n, n)` screen
   via `concatenate`; memory bounded at `n + O(stencil)` rows, verified over
   3000+ steps) and **sub-pixel continuous evolution** (`advance(pixels)` +
   Catmull-Rom `interp="cubic"`/`"linear"`; exact at integer offsets, so
   `advance(0.5)` twice equals `step(1)` bit-for-bit, checked on CPU and GPU).
   lstsq + eigh(clip) robustness retained. GPU path works (device-resident,
   `n+8` buffer at n=1024, ~4200 steps/s) and is now *faster per single layer*
   than the spectral engine. 5 new tests; full suite green (44 passed).
   *Remaining:* **Stage 2** — arbitrary wind direction (extrude in wind-aligned
   coords, sample a rotated pupil grid → 2-D interpolation), an `evolve(dt)`
   clock keyed to `wind_speed`, and **Atmosphere integration** as
   `engine="extrude"` batched across heterogeneous layers on the GPU (the hard
   part: the spectral `_integrate` batches uniform layers into one FFT; the
   extruder must batch per-layer rotations/interpolations). `cho_factor` +
   fractal stencil are optional polish over the current lstsq + dense stencil.

**P1 — table stakes for AO users.**

2. **FITS + npz I/O for screens and atmospheres** (learn from `soapy`). AO users
   live in FITS. Save/load with metadata in the header: `r0`, `L0`,
   `pixel_scale`, `seed`, profile name, zenith, and pyturb version. `astropy` as
   an optional dependency. *(Roadmap 6.2.)*
3. **Moment-conserving profile compression** (learn from `aotools`
   `equivalent_layers` / `optimal_grouping` / GCTM, Saxenhuber 2017). pyturb's
   `discretize_cn2` conserves total `Cn²dh` and the θ₀ centroid but **not τ₀**
   (the `Cn²·v^{5/3}` moment). Add a moment-conserving mode that also condenses
   per-layer **wind**, so coherence time survives compression — and a
   tomography-optimal grouping option.
4. **Chromatic OPD option** (new — see the expanded Phase 3.1 below). Keep OPD
   native, but stop silently assuming it is perfectly achromatic; offer an
   optional air-dispersion model.

**P2 — depth and reach.**

5. **Analysis/validation utilities, exported** (learn from `aotools`
   `temporal_ps`, Zernikes). A small `pyturb.analysis` with: Zernike
   decomposition (Noll variances), temporal-PSD slope fitting (−11/3 along-wind,
   −14/3 transverse), and angular decorrelation. These double as the Phase 5
   validation gallery *and* as user-facing tools. Keep a tiny internal Zernike
   helper and export it.
6. **More named site profiles** (learn from `HCIPy`: Keck, Las Campanas, Mauna
   Kea tables). Cheap, high-value: add Keck, La Silla, Cerro Pachón, and an
   ELT/Armazones profile. Each is a few lines of cited numbers.
7. **LGS cone effect** (learn from `soapy` line-of-sight). Per-layer footprint
   magnification `(1 − h/H_LGS)` for finite-altitude guide stars. Already
   deferred to M5; the geometry drops into the Phase 3.2 `directions=` path.

**P3 — nice-to-have, keep hooks.**

8. **Non-Kolmogorov spectra**: general power-law exponent and inner scale
   (modified von Kármán / Hill) — a few lines in `_psd`, leave the hook.
9. **Threaded CPU FFT** (`scipy.fft(workers=)`) to close the single-thread CPU
   gap; keep the pure-NumPy path dependency-light.
10. **Interop recipes, not dependencies**: documented adapters for HCIPy
    wavefronts, poppy, and DM-fitting pipelines. `frames()` already returns a
    plain array, so these are docs, not code.

**Explicit non-goals reaffirmed** (from the comparison): WFS/DM/controller
simulation (soapy), tomographic reconstructors and slope covariance (aotools),
and Fresnel/scintillation propagation between layers (HCIPy). pyturb outputs
phase/OPD and hands amplitude effects and system modelling to those tools.

---

## Where the repo stands today

Solid, small, correct core — a good foundation:

- `PhaseScreen`: FFT (McGlamery) screens with a careful subharmonic
  low-frequency correction (integrated per-cell PSD, not centre-sampled);
  structure function verified to ~1% in tests. Batched generation, two
  screens per complex FFT.
- `InfinitePhaseScreen`: Assémat & Wilson (2006) row extrusion for unbounded
  frozen flow, seeded from an FFT screen.
- Clean NumPy/CuPy backend dispatch (`device="gpu"` just works — verified on
  this machine's RTX 5090), float32 default, seeded reproducibility.
- 20 tests including ensemble-statistics checks; CI on 3.9/3.11/3.13.

What is *missing* is everything between "a single phase screen" and "the OPD
an 8-m telescope sees through a real atmosphere": layers, wind vectors,
sub-pixel time evolution, geometry (zenith angle, off-axis directions),
OPD/wavelength handling, profiles, docs, benchmarks, and distribution.

---

## Phase 1 — Multi-layer atmosphere model (the core missing feature)

**New module `pyturb/atmosphere.py`.**

### 1.1 `Layer` specification

A lightweight dataclass describing one turbulent layer:

| Field | Meaning |
|---|---|
| `altitude` | height above the telescope [m] |
| `cn2_fraction` (or `r0`) | fraction of total Cn²·dh in this layer; per-layer `r0 = r0_total * frac^(-3/5)` |
| `wind_speed` | [m/s] |
| `wind_direction` | [deg], arbitrary — not axis-aligned |
| `L0` | outer scale [m], per layer (default 25) |

### 1.2 `Atmosphere` class

- Holds a list of `Layer`s + global parameters: total `r0` (or `seeing`),
  reference wavelength (500 nm), `zenith_angle`, pupil `diameter`, sampling
  `n`, `device`, `dtype`, `seed`.
- Owns one evolving screen per layer (see Phase 2) and sums them into the
  pupil. Per-layer screens must be **larger than the pupil** when off-axis
  directions or long time series are requested (see Phase 3 geometry).
- Zenith scaling done once, correctly: `r0_eff = r0 * cos(z)^(3/5)`, layer
  range `= altitude / cos(z)`, wind unchanged (it's horizontal).
- Independent per-layer RNG streams derived from one master seed
  (`numpy.random.SeedSequence.spawn`) so results are reproducible and
  layer-count-independent.

### 1.3 Standard profiles (`pyturb/profiles.py`)

Named, cited, ready-to-use profiles — this is what makes it "representative
atmosphere in one line":

- **ESO Paranal median / quartiles** (the 35-layer ESO reference profile used
  for ELT studies, plus a condensed ~9-layer version for speed).
- **Mauna Kea** (TMT site survey 7-layer).
- **Hufnagel–Valley 5/7** continuous model + a `discretize(hv57, n_layers)`
  helper (equivalent-layer binning preserving Cn² moments 0 and 5/3).
- **Simple presets**: `"single-layer"`, `"two-layer"` (ground + jet stream)
  for teaching and quick tests.
- All profiles rescalable: `from_profile(name, seeing=...)` renormalises
  Cn² fractions to the requested total seeing while keeping the shape.

### 1.4 Derived integrated quantities (free wins, users always need these)

On `Atmosphere` (and as free functions taking a profile):

- `r0` (integrated, at any wavelength), `seeing`
- `theta0` — isoplanatic angle (Cn²·h^(5/3) moment)
- `tau0` — coherence time (Cn²·v^(5/3) moment) and Greenwood frequency
- effective wind speed, effective turbulence height `h_bar`

**Tests:** integrated structure function of the summed screens matches the
single-`r0` prediction; `theta0`/`tau0` against hand-computed values for a
two-layer profile; zenith scaling laws.

---

## Phase 2 — Time evolution: wind vectors, sub-pixel motion, boiling

Today `InfinitePhaseScreen.step()` moves exactly one pixel along axis 0.
A real loop needs `v·dt` metres per frame in an arbitrary direction, where
`v·dt` is rarely an integer number of pixels.

### 2.1 Sub-pixel, arbitrary-direction frozen flow

Recommended design — **extrude coarse, shift fine**:

- Keep the Assémat–Wilson extruder as the source of new turbulence, extruding
  rows along the wind axis whenever accumulated travel crosses a pixel.
- Maintain each layer's screen with a margin of a few pixels; produce the
  output by **sub-pixel interpolation** of the stored screen at the exact
  fractional offset. Two interpolators, selectable:
  - `interp="spline"` (bi-cubic; local, cheap, slight high-f attenuation), default
  - `interp="fourier"` (exact for band-limited screens; natural on GPU)
- Arbitrary wind direction: extrude along the screen's own axis and rotate
  the *sampling geometry*, not the data — i.e. each layer's screen lives in
  wind-aligned coordinates, and the pupil is sampled from it with a rotated
  (and translated) interpolation grid. This keeps extrusion 1-D and exact.
- `step(dt)` per layer; `Atmosphere.frames(dt, steps)` drives all layers and
  sums. Time is a first-class input (`atm.opd(t=...)` for random access
  within the already-generated span).

### 2.2 Extrusion performance fixes (needed once steps get small)

- Replace the per-step `xp.concatenate` in `InfinitePhaseScreen.step` with a
  **ring buffer** (preallocated `(n + margin, n)` array + row index); today
  every step copies the whole screen.
- **Batch extrusion**: extrude `k` rows in one call (block-conditional
  formulation, or sequential rows with a single matmul against a stacked
  stencil) so GPU launch latency is amortised. On GPU, per-row matvecs at
  1 kHz are latency-bound.
- Keep the covariance setup in float64 on CPU (as now) but cache
  `(A, B)` matrices on disk keyed by `(n, dx, r0, L0, stencil_rows)` —
  setup is O((stencil·n)³) and dominates start-up for n ≥ 512.

### 2.3 Beyond frozen flow (optional but differentiating)

- **Boiling**: per-layer decorrelation with time constant `tau_boil` —
  implement as a spectral AR(1): each Fourier mode evolves as
  `a(t+dt) = α a(t) + sqrt(1-α²) ξ`, with `α = exp(-dt/τ(f))`. This composes
  cleanly with the FFT screen representation; offer a "fully spectral"
  screen mode (`FourierFlowScreen`) where frozen flow is a phase ramp
  `exp(2πi f·v dt)` and boiling is the AR(1) — the whole time evolution of a
  layer then becomes one elementwise complex multiply + IFFT per frame,
  which is extremely fast on GPU and gives sub-pixel flow for free.
  *(This spectral screen is likely the fastest GPU path overall and is worth
  implementing alongside the extruder; the extruder remains the
  memory-bounded/unbounded-duration option, the spectral screen the
  fixed-outer-period fast option. Document the trade-off.)*

**Tests:** temporal power spectrum of a single pupil pixel matches the
frozen-flow prediction (−8/3 slope regime); measured `tau0` from simulated
frames matches the profile's analytic `tau0`; a sub-pixel step of 0.5 px
twice equals one 1-px step to interpolation tolerance; wind direction 37°
gives the same statistics as 0°.

---

## Phase 3 — OPD, wavelength, and viewing geometry

### 3.1 OPD as the native product

Today screens are "radians at the wavelength where r0 is defined", which
pushes bookkeeping onto the user. Switch the internal/native unit to **OPD in
metres**:

- Generate with `r0` referenced at 500 nm (or user-set), convert amplitude
  once: `OPD = phase * λ_ref / 2π`.
- `atm.frames(...)` yields OPD [m]. Helpers: `phase = pyturb.opd_to_phase(opd, wavelength)`
  and a `wavelength=` convenience argument on output methods.
- Keep `PhaseScreen`/`InfinitePhaseScreen` returning radians for backward
  compatibility, but document OPD as the recommended workflow.

**Is OPD actually the right output? (a real physics question, not just API
taste.)** Short answer: **yes, OPD in metres is the correct native product** —
but the word "achromatic" needs a caveat, and there is a genuine system
dependence worth exposing as an option.

*Why OPD is the right currency.* Phase and OPD are related by
`φ(λ) = (2π/λ)·OPD`, where `OPD = ∫(n − n̄) dz` is a path length. A phase screen
is only meaningful with a wavelength attached — and `r0 ∝ λ^{6/5}`, so quoting
"radians" forces the user to track *which* wavelength. OPD removes that: it is
the wavelength-independent physical quantity, and a real AO system is inherently
multi-wavelength (LGS at 589 nm, NGS in the visible, science in the NIR). With
OPD native, each path is one division away — `φ = 2π·OPD/λ` — and pyturb never
has to know the sensing or science band. OPD is also strictly more general than
phase (phase is one lossless step away; the reverse needs a wavelength), and it
is numerically safe: OPD values ~µm stored in float32 keep ~7 significant
figures of *relative* precision (sub-picometre absolute), because magnitude,
not offset, is what float32 cares about. So the choice is not OPD-vs-phase — it
is "store the invariant, derive the rest."

*The caveat — turbulence OPD is only* **nearly** *achromatic.* The refractivity
of air `(n − 1)` is itself weakly wavelength-dependent (dispersion), varying
~1–2 % from the visible to the K band. So the OPD a 589 nm WFS measures differs
from the OPD at a 2.2 µm science wavelength by ~1 %. On a 1 µm-RMS wavefront
that is ~10 nm — negligible for routine AO, but a real error budget term for
high-contrast imaging, precision astrometry, and LGS systems (chromatic /
refractive anisoplanatism). Worse, the **dry-air and water-vapour** components
disperse differently, and wet (H₂O) turbulence is strongly chromatic in the IR
— the dominant term for the mid-IR and for interferometry (VLTI-class
"wet–dry" OPD). A single achromatic OPD silently sets all of this to zero.

*Recommendation.* Keep OPD native and default (it is right for ~all users), but
(a) **document the achromatic assumption explicitly** rather than burying it in
the word, and (b) offer an **optional chromatic model**: a `wavelength=` output
that, when a dispersion model is enabled, scales OPD by the dry-air refractivity
ratio `(n(λ) − 1)/(n(λ_ref) − 1)` (Ciddor/Edlén), with a later dry/wet split for
interferometry/mid-IR users who need it. This turns the current implicit
approximation into an explicit, opt-in physical model — a differentiator none of
the compared libraries offer, and it costs nothing on the fast path (default =
achromatic, one scalar per wavelength when enabled). Tracked as backlog item P1-4.

### 3.2 Directions and anisoplanatism

- `atm.opd(t, directions=[(θx, θy), ...])` — per layer, shift the sampling
  footprint by `altitude/cos(z) * tan(θ)`; batch all directions into one
  interpolation call → `(n_dir, n, n)`.
- This immediately serves MCAO/LTAO/GLAO tomography studies and PSF-R work,
  and it falls out of the Phase 2 interpolation machinery almost for free.
- **LGS cone effect** (stretch, small addition): scale each layer's footprint
  by `(1 − h/H_LGS)` when the source is at finite range `H_LGS`.

### 3.3 Pupil bookkeeping

- Screens sized from `diameter`, `n`, plus the automatically computed margin
  for requested max off-axis angle and max wind travel — the user never
  computes screen sizes by hand. Warn/grow when a request would leave the
  generated region.

**Tests:** two directions separated by `theta0` show the expected residual
variance (~1 rad² at λ_ref); direction (0,0) equals the on-axis path exactly;
OPD is wavelength-independent while phase scales as 1/λ.

---

## Phase 4 — Performance: prove "fastest", not just claim it

- **`benchmarks/` suite** (simple `pytest-benchmark` or standalone script):
  screens/s and frames/s vs `n` ∈ {256, 512, 1024, 2048, 4096}, CPU vs GPU,
  float32/float64, both evolution engines. Publish a results table in the
  docs (this machine: RTX 5090, CuPy 13.6) and the script so users can
  reproduce.
- **Head-to-head comparison** with `aotools`, `soapy`, and `HCIPy` screen
  generation/evolution — speed *and* structure-function accuracy in one
  table. This is the single most persuasive artifact for adoption.
- GPU engineering checklist (profile first, in this order):
  1. No host↔device syncs inside `frames()` loops (return device arrays;
     sync only on `to_numpy`).
  2. Batch across time and directions (Phases 2–3 designs already do).
  3. Preallocate all per-frame buffers; use `cupy.fuse`/`ElementwiseKernel`
     for the spectral-evolution multiply if profiling shows it matters.
  4. Optional CUDA streams to overlap layer computations.
- CPU: `scipy.fft` `workers=` for multithreaded FFTs; keep the pure-NumPy
  path dependency-light.
- Add `pyturb.benchmark()` convenience so a user can print their own
  machine's numbers in 10 s.

**Target (define "fastest" concretely):** ≥ 1000 fps of 512² 5-layer OPD
frames on a current consumer GPU; ≥ 10⁴ independent 256² screens/s batched.
Measure, then tune until the comparison table wins.

---

## Phase 5 — Validation depth (trust is a feature)

Add to the test suite (marked `slow` where needed, run in CI nightly):

- **Zernike decomposition**: measured Zernike variances vs Noll (1976) /
  von Kármán-corrected predictions for the first ~20 modes. (Requires a tiny
  internal Zernike helper — also useful to users; export it.)
- **Temporal PSDs** per Phase 2.
- **Angular decorrelation** per Phase 3.
- **Long-run stationarity**: extruded screen variance and structure function
  after 10⁵ steps show no drift (guards against conditional-covariance
  error accumulation).
- **GPU parity**: statistics tests parameterised over `device` and skipped
  when CuPy is absent; add a self-hosted GPU CI job (this server) or a
  scheduled manual workflow so the GPU path is continuously tested.
- Numerical guard: `InfinitePhaseScreen` currently accepts any dtype without
  the float32/float64 validation `PhaseScreen` has — unify.

Ship the validation as a *documented artifact*: a `validation.ipynb` /docs
page with every plot (structure function, PSDs, Zernike spectrum, angular
correlation) regenerated by CI, so users can see the evidence, not just
trust the README sentence.

---

## Phase 6 — Usability, docs, and distribution

### 6.1 Documentation site (biggest UX gap)

- **mkdocs-material** (or Sphinx+furo) on Read the Docs / GitHub Pages:
  - *Quickstart* (3 lines to OPD frames), *Concepts* (r0/L0/Cn²/tau0/theta0
    in one page for newcomers), *How-to* guides: closed-loop sim, Monte-Carlo
    ensembles, off-axis/tomography inputs, GPU tips, choosing screen engine.
  - Full API reference from docstrings (already NumPy-style — good).
  - The validation gallery (Phase 5) and benchmark tables (Phase 4).
- README: add an animated GIF of a wind-blown multi-layer screen (renders
  instantly on GitHub; disproportionate adoption value), badges (PyPI, CI,
  docs, coverage), and shrink details in favour of links into the docs.
- Examples: promote `examples/quickstart.py` to a small gallery —
  `01_screens.py`, `02_closed_loop.py`, `03_layered_atmosphere.py`,
  `04_off_axis.py`, `05_gpu_benchmark.py` — each < 60 lines and plot-final.

### 6.2 API polish

- Type hints on all public signatures + `py.typed` marker.
- `save`/`load` for screens and atmospheres: `.npz` with metadata, and FITS
  (astropy optional dependency) since AO users live in FITS. Include
  `r0`, `L0`, `pixel_scale`, seed, and pyturb version in headers.
- Interop adapters where they're one-screen-deep: `Atmosphere.frames()`
  output is already a plain array, so document recipes for HCIPy, poppy,
  and DM-fitting pipelines rather than hard dependencies.
- Single-source the version (`hatch` + `pyturb.__version__` from metadata)
  so `pyproject.toml` and `__init__.py` can't drift.

### 6.3 Distribution & project hygiene

- **PyPI release workflow** (GitHub Actions, trusted publishing, tag-driven)
  — nothing beats `pip install pyturb` actually existing. Then conda-forge.
- `CHANGELOG.md` (Keep a Changelog), `CONTRIBUTING.md`, issue templates.
- CI additions: `ruff check` + `ruff format --check`, coverage upload,
  Python 3.12 in the matrix (it's the CUDA env version here and is missing),
  a docs-build job, and the GPU job from Phase 5.
- Update `README`/`pyproject` classifiers when the atmosphere API lands
  (the "does one thing well / two-class API" framing will need refresh).

---

## Phase 7 — Deliberate non-goals / stretch (keep the scope crisp)

Documented as out of scope (with pointers), unless demand pulls them in:

- **Scintillation / Fresnel propagation between layers** — pyturb outputs
  phase/OPD; amplitude effects belong to propagation codes (HCIPy, PROPER).
  *Stretch*: a minimal angular-spectrum propagator over pyturb layers.
- Non-Kolmogorov power-law exponents and inner scale (Hill/modified von
  Kármán spectrum) — cheap to add to `_psd` if requested; leave hooks.
- Full AO system simulation (WFS, DM, controllers) — explicitly never; pyturb
  is the atmosphere that plugs into those tools.

---

## Suggested execution order

| Milestone | Contents | Outcome |
|---|---|---|
| **M1** (core) ✅ | Phase 1 + Phase 2 spectral engine | **Done** — `Atmosphere.from_profile(...).frames(dt)` works: frozen flow, sub-pixel, multi-layer, off-axis, GPU. (Ring-buffer extruder from 2.1–2.2 still open) |
| **M2** (product) ✅ | Phase 3 + spectral engine (2.3) | **Done** — OPD in metres, off-axis directions, field-of-view oversizing, boiling. (LGS cone deferred to M5) |
| **M3** (proof) 🟡 | Phase 4 ✅ + Phase 5 | **Phase 4 done** — published benchmarks + honest comparison vs aotools/soapy/HCIPy (`bench_compare.py`, `RESULTS.md`, `docs/comparison.md`). Phase 5 validation gallery still open. |
| **M3.5** (parity) 🟡 | **Backlog P0–P1**: non-periodic extruder (1) — **Stage 1 done** (ring buffer + sub-pixel), Stage 2 (arbitrary direction + `Atmosphere` `engine="extrude"`) open; FITS/npz I/O (2), moment-conserving compression (3), chromatic-OPD option (4) still open | close the capability gaps the comparison exposed — pyturb matches the others where it should and stays ahead where it already is |
| **M4** (adoption) | Phase 6 + backlog P2 (analysis utils 5, site profiles 6) | docs site, PyPI release `v0.2.0`, README with animation |
| **M5** (polish) | LGS cone (7), non-Kolmogorov hooks (8), threaded CPU FFT (9), interop recipes (10), conda-forge | differentiating extras |

Backlog item numbers refer to *Critical review after the M3 comparison* above.

Rules of thumb while executing: every new physics feature lands with an
ensemble-statistics test against theory (the repo's existing standard — keep
it); every performance claim lands with a benchmark script; every feature
lands with a docs page. **M3.5 is now the highest-value next step**: the
comparison showed the non-periodic extruder (P0-1) is the one architectural gap
between pyturb and the field — everything else is either already ahead or a
cheap add.

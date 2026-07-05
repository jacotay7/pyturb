# pyturb vs aotools, soapy & HCIPy

An honest, method-by-method comparison of `pyturb` with three widely used
open-source **Python** atmospheric phase-screen tools, what we learned from
reading their source, and where each one is genuinely stronger. This is not
a survey of the whole field: **COMPASS** (CUDA C++,
gitlab.obspm.fr/cosmic-rtc/compass) is a GPU-native, non-periodic,
end-to-end AO simulator that has been the ESO community's ELT-scale
workhorse for over a decade, and is a different category of tool (a
compiled simulator with a Python front-end) than the three importable
Python/NumPy/CuPy libraries compared here. On the specific axis of "fastest
non-periodic GPU atmosphere," COMPASS's hand-written CUDA extrusion kernels
substantially outperform pyturb's CuPy-based `engine="extrude"` for the same
job on the same GPU — see the note in "Where pyturb is already ahead" below.

The raw benchmark numbers (RTX 5090) live in
[`benchmarks/RESULTS.md`](https://github.com/jacotay7/pyturb/blob/main/benchmarks/RESULTS.md); reproduce them with
`python benchmarks/bench_compare.py`. This document is about *how* the libraries
work and *why* the numbers come out the way they do.

- **aotools** — [github.com/AOtools/aotools](https://github.com/AOtools/aotools)
- **soapy** — [github.com/AOtools/soapy](https://github.com/AOtools/soapy)
- **HCIPy** — [github.com/ehpor/hcipy](https://github.com/ehpor/hcipy)

## TL;DR

| | pyturb | aotools | soapy | HCIPy |
|---|---|---|---|---|
| Primary purpose | atmosphere for AO | AO toolbox | full AO system sim | high-contrast imaging |
| GPU | **CuPy** | — | — | — |
| Generation engine | FFT + integrated subharmonics | FFT + subharmonics | (reuses aotools) | extruder init |
| Frozen-flow engine | **spectral shift theorem** | Assémat–Wilson extruder | large-screen panning | extruder + interpolation |
| Sub-pixel / any direction | **yes / yes** | no / no | yes / yes | yes / yes |
| Unbounded (non-periodic) | **yes** (`engine="extrude"`) | **yes** | **yes** | **yes** |
| Boiling | **yes** | — | — | — |
| Scintillation (Fresnel) | non-goal | — | — | **yes** |
| Tomographic reconstructors | — | **yes** | **yes** | modal |
| FITS screen I/O | **yes** (`pyturb.save`) | — | **yes** | — |

Every library here is good; they were built for different jobs. Among these
three, pyturb's niche is being a *GPU-capable, statistically-careful
atmosphere generator* with a three-line API — not a full AO system simulator
(soapy), a reconstruction toolbox (aotools), or a diffraction propagation
framework (HCIPy). It is not the fastest GPU atmosphere generator in
absolute terms: COMPASS's CUDA-native extrusion beats pyturb's non-periodic
engine by roughly an order of magnitude on the same GPU (see above and
"Where pyturb is already ahead" below).

## The four methods, honestly

There are only a handful of ways to make and move a phase screen. Each library
picks a different point in the trade space.

### 1. FFT + subharmonics (screen *generation*)

Colour white noise with √PSD and inverse-FFT (McGlamery 1976); add subharmonic
low-frequency modes (Lane 1992; Johansson & Gavel 1994) to restore the tip/tilt
power a periodic grid cannot hold.

- **Who:** `pyturb.PhaseScreen`, `aotools.ft_sh_phase_screen`,
  `soapy` (calls aotools).
- **Strength:** i.i.d. screens for Monte-Carlo; simple; accurate if the
  subharmonics are done well.
- **Weakness:** periodic; a single screen is O(n² log n).
- **Where pyturb differs:** (a) it draws **two independent screens per complex
  FFT** and **batches the whole ensemble** into one call, and (b) it integrates
  the steeply-convex PSD **over each low-frequency cell** rather than sampling
  the cell centre. (a) is why pyturb generates ~1000× faster on GPU and ~13×
  faster on one CPU core; (b) is why its structure function tracks von Kármán to
  ~2% vs ~5% for the aotools/soapy `ft_sh` path (see RESULTS.md §3).

### 2. Assémat–Wilson stencil extrusion (unbounded frozen flow)

Keep only a thin strip of screen and *extrude* one new row at a time from a
learned autoregressive relation `X = A·Z + B·ξ`, where `Z` is a sparse
**stencil** of existing points, `A = Cov_xz·Cov_zz⁻¹`, and `B·Bᵀ =
Cov_xx − A·Cov_zx`. The stencil samples the strip on a **fractal 2ⁿ grid** so
the covariance matrices stay small no matter how long you extrude.

- **Who:** `aotools.PhaseScreenKolmogorov/VonKarman`, `soapy` (imports aotools),
  `HCIPy.InfiniteAtmosphericLayer`.
- **Strength:** **truly unbounded and non-periodic**, low memory, and each step
  is a cheap O(n·stencil) matvec — this is why aotools/soapy beat pyturb on
  single-layer CPU stepping at small n (RESULTS.md §2).
- **Weakness:** `add_row` advances exactly **one integer pixel along one axis**;
  sub-pixel and arbitrary direction need an interpolation layer on top (HCIPy
  adds bilinear interpolation; aotools/soapy do not). Setup is O((stencil·n)³)
  and per-step cost grows with n; no GPU path. aotools' implementation also
  copies the whole strip every step (`numpy.append`).
- This was the one capability all three had that pyturb lacked; pyturb now
  implements it too, as `Atmosphere(engine="extrude")` (method 4 below plus a
  wind-aligned ring buffer and rotated sub-pixel sampling). See "What we
  learned" below. Any sub-pixel interpolation layer — pyturb's Catmull-Rom
  included — is a position-dependent low-pass filter: exact at integer pixel
  travel, most aggressive at half-pixel travel (zero gain at the Nyquist
  frequency for a Catmull-Rom kernel). This shows up as a 5-15% fine-scale
  (1-2 px) structure-function deviation that oscillates with the sub-pixel
  phase of the wind travel — see the README's engine comparison for the
  measured numbers. The spectral engine (method 4) has no such artifact.

### 3. Large static screen + bilinear panning

Generate one big screen once, then slide an `n×n` window across it at `v·dt`
per frame, bilinearly interpolating for the sub-pixel remainder.

- **Who:** `soapy.atmosphere.atmos` (`moveScrns`), threaded with numba.
- **Strength:** trivially sub-pixel and any-direction; cheap per frame.
- **Weakness:** bounded by the big screen (wraps eventually), and the big
  screen's memory grows with run length × wind speed.

### 4. Spectral shift theorem + AR(1) boiling (pyturb)

Draw one fixed realisation of the screen's Fourier coefficients, then evaluate
any continuous translation `(sx, sy)` by multiplying each mode by its phasor
`exp(2πi(fₓsx + f_y sy))` before the inverse FFT — **exact sub-pixel
translation in an arbitrary direction, one FFT per frame**. Temporal
decorrelation ("boiling") is an AR(1) on each mode, composed into the same
multiply.

- **Who:** `pyturb.FourierFlowScreen` (and `Atmosphere.frames`).
- **Strength:** sub-pixel and any-direction *for free and exactly*; one
  elementwise multiply + IFFT is ideal for the GPU, and **all layers batch into
  a single `(L, n, n)` FFT** — the basis for ~800 fps of a 9-layer 512²
  atmosphere.
- **Weakness:** periodic (period `n·pixel_scale`), like any FFT screen, so it is
  not the right tool for genuinely unbounded runs — that is what pyturb's own
  extruder (method 2, `engine="extrude"`) is for. Both engines share the
  `Atmosphere` API; pick per run. Boiling costs real throughput: ~25-35% on
  GPU and ~45-50% on CPU for a 9-layer 512² atmosphere (see `RESULTS.md` §2),
  since each mode needs its own scale-dependent retention coefficient (finer
  spatial structure decorrelates faster, per Kolmogorov eddy-turnover
  scaling) rather than one scalar multiply per layer.

## Benchmark summary

Full tables in [`benchmarks/RESULTS.md`](https://github.com/jacotay7/pyturb/blob/main/benchmarks/RESULTS.md). Headlines
on an RTX 5090, 8 m pupil, 512²:

- **Generation:** pyturb **14,054 screens/s** (GPU, batched) vs 12–13/s for the
  aotools/soapy Python FFT loops — ~1000×. ~13× even on one CPU core.
- **Frozen flow:** pyturb runs a full **9-layer atmosphere at ~800 fps** on GPU.
  Single-layer integer-pixel CPU stepping in aotools/soapy is faster *per step*
  (5,200–5,400/s) but does strictly less general work and has no GPU path.
- **Accuracy:** pyturb ~**1%** systematic structure-function bias vs von
  Kármán on large ensembles — comparable to HCIPy/soapy and lower than
  aotools, but at benchmark-scale ensembles the metric's own Monte-Carlo
  noise is large enough that the ranking between tools is not reliable run to
  run; see `RESULTS.md` §3 for the numbers with uncertainties.

## What we learned — and what we're adopting

Reading these codebases sharpened pyturb's roadmap. Concrete take-aways:

**From aotools**

1. **The stencil extruder is the community standard** (soapy imports it
   verbatim). This shaped pyturb's non-periodic engine — now **implemented** as
   `Atmosphere(engine="extrude")`: the same Assémat–Wilson formulation, in a
   wind-aligned ring buffer with rotated sub-pixel sampling (any wind direction,
   any `v*dt`, GPU-resident). `cho_factor`/`cho_solve` with a least-squares
   fallback remains an optional numerical-robustness upgrade over the current
   lstsq solve. *(Roadmap 2.1–2.2, delivered.)*
2. **Moment-conserving profile compression** — **adopted.** Following
   `equivalent_layers` (Fusco 1999), `discretize_cn2(method="equivalent")` (now
   the default) sets each layer to the `Cn²·h^{5/3}` and `Cn²·v^{5/3}` moments,
   conserving θ₀ *and* τ₀ to <0.1 % (the old centroid method drifts up to
   ~12 %). `optimal_grouping`/GCTM tomography variants remain optional extras.
3. **Validation tooling** — **adopted.** `pyturb.analysis` now provides Zernike
   decomposition + Noll (1976) variances, a temporal-PSD slope fit, and angular
   decorrelation — the same family of checks as aotools' `temporal_ps`, plus the
   Zernike spectrum, usable both as validation and as user tools.

**From HCIPy**

4. **Interpolated extrusion for sub-pixel any-direction unbounded flow.**
   `InfiniteAtmosphericLayer` pairs the extruder with bilinear interpolation and
   a real `evolve_until(t)` clock. pyturb's non-periodic engine already samples
   sub-pixel in any direction; `Atmosphere.evolve(dt)` now gives the matching
   in-seconds clock so callers step in time, not integer pixels.
5. **More named site atmospheres** — **adopted.** Added `"keck"`,
   `"las-campanas"` and `"mauna-kea"` from HCIPy's cited tables (Guyon 2005 /
   Tokovinin et al. 2005 for Mauna Kea; see `pyturb.profiles` for the other
   citations), alongside representative `"paranal-median"`, `"cerro-pachon"`
   (Gemini South) and `"armazones"` (ELT) profiles, HV5/7 and the toy profiles.
6. **Clean Monte-Carlo reset.** `reset(make_independent_realization=True)` is a
   tidy way to draw independent realisations from a configured layer.
7. **Scintillation is explicitly *not* our job.** HCIPy's `MultiLayerAtmosphere`
   does Fresnel propagation between layers; pyturb outputs phase/OPD and defers
   amplitude effects to HCIPy/PROPER by design.

**From soapy**

8. **FITS I/O for screens** — **adopted.** `pyturb.save`/`pyturb.load` write
   `.fits` (optional astropy) or `.npz` by extension, round-tripping the data
   with `pixel_scale`, `r0`, `L0`, `wavelength`, `units`, `seed` and the pyturb
   version; `Atmosphere.metadata` supplies a ready-to-save provenance dict.
9. **Threaded CPU kernels.** soapy uses numba `prange` for interpolation and
   binning; pyturb can lean on `scipy.fft(workers=)` and, if needed, similar
   kernels to close the single-thread CPU gap.
10. **LGS cone geometry.** soapy's line-of-sight handles finite-altitude guide
    stars (cone effect); pyturb has deferred this (M5) and can reference soapy's
    per-layer magnification approach when it lands.

## Where pyturb is already ahead

The feature set that motivated building pyturb rather than extending an existing
tool (relative to aotools/soapy/HCIPy; see the COMPASS note above and below
for where a compiled, GPU-native end-to-end simulator still wins):

- **GPU-capable (CuPy).** None of the three has a GPU backend. `device="gpu"` is
  one keyword and everything else is identical. (COMPASS, a different category
  of tool, has had a GPU-native atmosphere for over a decade — see below.)
- **Batched Monte-Carlo.** Two screens per FFT and the whole ensemble in one
  call — the source of the ~1000× generation win over these three CPU-only
  libraries.
- **Spectral frozen flow.** Exact sub-pixel, arbitrary-direction translation at
  one FFT/frame, batched across all layers — the fastest multi-layer *periodic*
  GPU path among these three, and the reason a 9-layer 512² atmosphere runs at
  loop rate. Not the fastest *non-periodic* GPU path in absolute terms: see
  COMPASS below.
- **Boiling.** Per-layer temporal decorrelation (spectral AR(1)); none of the
  others model non-frozen turbulence.
- **Low systematic bias.** The integrated-per-cell subharmonic correction
  gives ~1% systematic structure-function bias on large ensembles — smaller
  than aotools' centre-sampled approach — though at practical (small)
  ensemble sizes the metric's own Monte-Carlo noise dominates that
  difference (see `RESULTS.md` §3).
- **OPD in metres, natively.** Achromatic OPD is the product; phase at any
  wavelength is a one-line helper. The others hand you radians at the r0
  wavelength and leave the bookkeeping to you.
- **Off-axis / tomography in one call.** `atm.opd(t, directions=[...])` batches
  all directions through one interpolation, with field-of-view oversizing so the
  footprints sample genuinely different turbulence.
- **Three-line API with cited profiles** and integrated `r0`, `seeing`, `θ₀`,
  `τ₀`, and Greenwood frequency out of the box.

## When to use which

- **pyturb** — you need fast, statistically-careful atmosphere OPDs, especially
  many of them or on a GPU: Monte-Carlo ensembles, closed-loop OPD frames,
  off-axis tomography inputs, PSF reconstruction.
- **aotools** — you need the broader AO maths toolbox: slope covariance,
  tomographic reconstructors, profile compression, Zernikes, conversions.
- **soapy** — you need a full end-to-end AO *system* simulation (WFS, DM,
  controllers), not just the atmosphere.
- **HCIPy** — you need diffraction propagation and scintillation for
  high-contrast imaging, with the atmosphere as one element in an optical train.

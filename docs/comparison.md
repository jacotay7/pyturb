# pyturb vs aotools, soapy & HCIPy

An honest, method-by-method comparison of `pyturb` with the three most widely
used open-source atmospheric phase-screen tools, what we learned from reading
their source, and where each one is genuinely stronger.

The raw benchmark numbers (RTX 5090) live in
[`../benchmarks/RESULTS.md`](../benchmarks/RESULTS.md); reproduce them with
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
| FITS screen I/O | roadmap | — | **yes** | — |

Every library here is good; they were built for different jobs. pyturb's niche
is being the *fastest, GPU-native, statistically-careful atmosphere generator*
with a three-line API — not a full AO system simulator (soapy), a
reconstruction toolbox (aotools), or a diffraction propagation framework
(HCIPy).

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
- **This is the one thing all three have that pyturb only partially has.** See
  "What we're adopting" below.

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
  atmosphere. Boiling is nearly free.
- **Weakness:** periodic (period `n·pixel_scale`), like any FFT screen, so it is
  not the right tool for genuinely unbounded runs — that is what the extruder
  (method 2) is for, and why pyturb is adding one.

## Benchmark summary

Full tables in [`../benchmarks/RESULTS.md`](../benchmarks/RESULTS.md). Headlines
on an RTX 5090, 8 m pupil, 512²:

- **Generation:** pyturb **14,054 screens/s** (GPU, batched) vs 12–13/s for the
  aotools/soapy Python FFT loops — ~1000×. ~13× even on one CPU core.
- **Frozen flow:** pyturb runs a full **9-layer atmosphere at ~800 fps** on GPU.
  Single-layer integer-pixel CPU stepping in aotools/soapy is faster *per step*
  (5,200–5,400/s) but does strictly less general work and has no GPU path.
- **Accuracy:** pyturb ~**2%** structure-function error vs von Kármán, the best
  of the four (HCIPy ~3.7%, aotools/soapy ~4.9%).

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
2. **Moment-conserving profile compression.** `equivalent_layers`,
   `optimal_grouping`, and GCTM (Saxenhuber 2017) compress a high-resolution
   Cn²(h) profile to N layers while conserving chosen turbulence moments — and
   `equivalent_layers` also condenses the **wind** per layer, conserving
   coherence time. pyturb's `discretize_cn2` currently conserves total Cn²dh and
   the θ₀ centroid but not τ₀; we should offer a moment-conserving mode.
3. **Validation tooling.** `temporal_ps` fits the −11/3 (along-wind) and −14/3
   (transverse) slope laws — exactly the kind of check the Phase 5 validation
   gallery needs.

**From HCIPy**

4. **Interpolated extrusion for sub-pixel any-direction unbounded flow.**
   `InfiniteAtmosphericLayer` pairs the extruder with bilinear interpolation and
   a real `evolve_until(t)` clock — the design pyturb's non-periodic engine
   should match so it is not limited to integer pixels.
5. **More named site atmospheres.** HCIPy ships Keck, Las Campanas, and Mauna
   Kea layer tables. pyturb has Paranal, Mauna Kea, HV5/7, and toy profiles —
   adding Keck/La Silla is a cheap, high-value win.
6. **Clean Monte-Carlo reset.** `reset(make_independent_realization=True)` is a
   tidy way to draw independent realisations from a configured layer.
7. **Scintillation is explicitly *not* our job.** HCIPy's `MultiLayerAtmosphere`
   does Fresnel propagation between layers; pyturb outputs phase/OPD and defers
   amplitude effects to HCIPy/PROPER by design.

**From soapy**

8. **FITS I/O for screens.** AO users live in FITS; soapy saves/loads screens
   with metadata. pyturb should add `.fits` (and `.npz`) save/load carrying
   `r0`, `L0`, `pixel_scale`, seed, and version. *(Roadmap 6.2.)*
9. **Threaded CPU kernels.** soapy uses numba `prange` for interpolation and
   binning; pyturb can lean on `scipy.fft(workers=)` and, if needed, similar
   kernels to close the single-thread CPU gap.
10. **LGS cone geometry.** soapy's line-of-sight handles finite-altitude guide
    stars (cone effect); pyturb has deferred this (M5) and can reference soapy's
    per-layer magnification approach when it lands.

## Where pyturb is already ahead

The feature set that motivated building pyturb rather than extending an existing
tool:

- **GPU-native (CuPy).** None of the three has a GPU backend. `device="gpu"` is
  one keyword and everything else is identical.
- **Batched Monte-Carlo.** Two screens per FFT and the whole ensemble in one
  call — the source of the ~1000× generation win.
- **Spectral frozen flow.** Exact sub-pixel, arbitrary-direction translation at
  one FFT/frame, batched across all layers — the fastest multi-layer GPU path
  and the reason a 9-layer 512² atmosphere runs at loop rate.
- **Boiling.** Per-layer temporal decorrelation (spectral AR(1)); none of the
  others model non-frozen turbulence.
- **Best statistical accuracy.** The integrated-per-cell subharmonic correction
  gives ~2% structure-function fidelity.
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

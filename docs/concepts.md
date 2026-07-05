# Concepts

A one-page primer on the turbulence quantities pyturb uses. All are properties
of an `Atmosphere` (`atm.r0`, `atm.theta0`, …) and free functions in
`pyturb`/`pyturb.profiles`.

## Cn²(h) — the turbulence profile

The refractive-index structure constant `Cn²` as a function of altitude `h`
says *how much* turbulence sits at each height. A **profile** is this integrated
into layers: a list of `Layer`s, each carrying a fraction of the total `Cn² dh`,
an altitude, a wind vector, and an outer scale. pyturb ships named profiles
(`paranal-median`, `keck`, …) and builds custom ones from a continuous model
with `discretize_cn2` (moment-conserving by default).

## r0 — the Fried parameter

`r0` is the aperture over which the wavefront stays roughly flat (~1 rad² of
phase variance). Smaller `r0` = worse seeing. It is **wavelength dependent**,
`r0 ∝ λ^{6/5}`, so it is always quoted at a reference wavelength (500 nm by
default). The layers combine as `r0^{-5/3} = Σ r0_i^{-5/3}`.

## seeing

The atmospheric PSF width, `seeing ≈ 0.98 λ / r0` [rad]. pyturb converts
between the two (`r0_from_seeing`, `seeing_from_r0`); `Atmosphere` takes either.

## L0 — the outer scale

The largest turbulent scale (tens of metres). Finite `L0` (von Kármán) caps the
low-frequency power that pure Kolmogorov (`L0=inf`) would let diverge, which
matters for tip/tilt and for the total wavefront variance
`0.0863 (L0/r0)^{5/3}`.

## θ0 — the isoplanatic angle

The angle over which the wavefront stays correlated: correcting on-axis still
helps a source within `θ0`. Set by the `Cn²·h^{5/3}` moment,
`θ0 = 0.314 r0 / h̄`. Two directions separated by `θ0` differ by ~1 rad² of
wavefront error — see [Validation](validation.md).

## τ0 — the coherence time

How long the wavefront stays correlated as the wind blows,
`τ0 = 0.314 r0 / v̄`, set by the `Cn²·v^{5/3}` moment. The **Greenwood
frequency** `f_G = 0.134 / τ0` is the AO loop bandwidth you need to keep up.

## OPD vs phase

pyturb's native output is **optical path difference in metres**, which is
achromatic (a path length). Phase at a wavelength is `φ = 2π·OPD/λ`. This
decouples the atmosphere from the sensing/science band; pass `wavelength=` to
get phase. (For the small ~1–2% chromatic term from air dispersion, see
`dispersion="edlen"`.)

## Frozen flow (Taylor hypothesis)

Turbulence is assumed to blow across the pupil frozen in shape at the layer's
wind velocity, so evolution in time is translation in space. pyturb offers two
engines for this — a fast periodic spectral one and an unbounded extruder — plus
optional **boiling** (`tau_boil`) for the residual non-frozen decorrelation.

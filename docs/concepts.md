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
`dispersion="edlen"` for dry air, or `dispersion="ciddor"` with a
`wet_fraction` to add the water-vapour term that matters in the mid-IR and for
interferometry — the "wet–dry" problem.)

## Frozen flow (Taylor hypothesis)

Turbulence is assumed to blow across the pupil frozen in shape at the layer's
wind velocity, so evolution in time is translation in space. pyturb offers two
engines for this — a fast periodic spectral one and an unbounded extruder — plus
optional **boiling** (`tau_boil`) for the residual non-frozen decorrelation.

## Conventions

- **Array axes.** Every output is an `(n, n)` array indexed `(y, x)`:
  **x runs along axis 1 (columns)** and **y along axis 0 (rows)**, both
  increasing with the index, with pixel centres at `(i - (n-1)/2) * pitch`
  and pitch `diameter / n`. This is the convention of aobasis, makewfs,
  solvephase, shmpipeline-ao and HCIPy, so a pyturb OPD flattened with
  `.ravel()` lands on an HCIPy pupil grid with x and y matching (see
  [Interop](interop.md)). Every x/y-labelled input below follows it.
- **Wind direction.** `Layer.wind_direction` is the direction the wind blows
  **from**, in degrees measured from +x (axis 1) toward +y (axis 0), and
  `Layer.wind_vector` returns `(vx, vy)` along (x, y) pointing the same way.
  The turbulence pattern moves along `-wind_vector`:
  `phi(r, t) = phi_0(r + wind_vector * t)` with `r = (x, y)`. With
  `wind_direction=0` the pattern travels toward decreasing **column** index
  (new turbulence enters at the right-hand, high-column edge); with `90` it
  travels toward decreasing row index. Both `Atmosphere` engines follow this;
  `InfinitePhaseScreen` always extrudes along y (axis 0), toward decreasing
  row index. If you compare against a tool that treats velocity as the
  pattern's motion, negate the vector; the axes need no swap. (HCIPy's
  `InfiniteAtmosphericLayer` velocity points the same way as `wind_vector`;
  see [Interop](interop.md#hcipy).)
- **Off-axis directions.** `directions=[(thx, thy)]` are angles in arcsec,
  `thx` along x (axis 1) and `thy` along y (axis 0). A layer at line-of-sight
  range `h` is sampled at `r + h * tan(theta)`, so `(thx, 0)` with `thx > 0`
  moves its footprint toward increasing column index.
- **Other x/y inputs.** `Atmosphere.opd_at(x, y)` takes offsets along x
  (columns) and y (rows); `FourierFlowScreen.translate(sx, sy)` shifts by
  `sx` along x and `sy` along y. `analysis.zernike_basis` uses the same frame
  with `theta = atan2(y, x)`, so Noll Z2 (tip, `cos theta`) varies along the
  columns and Z3 (tilt, `sin theta`) along the rows — identical to
  `aobasis.ZernikeBasisGenerator` on `aobasis.positions_from_mask` positions.
  `theory.zernike_temporal_psd(..., wind_direction)` uses the
  `Layer.wind_direction` frame, so `wind_direction=0` makes Z2 the along-wind
  tilt.
- **Line of sight vs zenith.** `seeing`/`r0` passed to the constructor are at
  zenith; `atm.r0`, `atm.seeing`, `atm.theta0` and `atm.tau0` are along the
  line of sight at `zenith_angle`. So `seeing=0.8` at 30° reports
  `atm.seeing ≈ 0.87`.
- **Units.** `Atmosphere` returns OPD in metres; pass `wavelength=` for phase in
  radians. `PhaseScreen`, `InfinitePhaseScreen` and `FourierFlowScreen` return
  phase in radians at the wavelength their `r0` is quoted at.

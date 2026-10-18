# Boiling, LGS cone and dispersion

Three effects on top of frozen-flow turbulence, each with a worked example.
The [theory module](api.md#theory) gives the reference curves to check them
against.

## Boiling

Real turbulence evolves as well as blowing past. `tau_boil` [s] (scalar or one
per layer) relaxes each layer toward fresh, statistically identical
turbulence while preserving its spatial statistics (so r0 is unchanged):

- `engine="spectral"`: each Fourier mode boils with its own time constant.
  With `boil_scaling="kolmogorov"` (the default) that is
  `tau(f) = tau_boil * (f / f_ref)^(-2/3)` for `f >= f_ref` (Kolmogorov eddy
  turnover; `f_ref = 1/L0`, or the grid fundamental for `L0=inf`), and
  `tau_boil` below `f_ref`, so fine structure decorrelates faster than the
  outer scale. With `boil_scaling="uniform"` every mode has `tau_boil` (see
  [Uniform boiling](#uniform-boiling-and-measured-decay-rates)).
- `engine="extrude"`: the readable ring buffer blends toward a fresh extruded
  screen at the single rate `tau_boil` for every scale (its only scaling,
  `"uniform"`); non-periodic, but much slower per frame (it warns).

!!! warning "`tau_boil` is the lifetime of the outer scale"
    On the spectral engine with the default `boil_scaling="kolmogorov"`,
    `tau_boil` is the time constant of the *largest*
    structure (`L0`). Smaller scales boil faster: with `L0 = 25 m`, 25 cm
    structure has `tau_boil / 21.5` and 5 cm structure `tau_boil / 63`. A
    `tau_boil` of tens of milliseconds therefore decorrelates the scales an AO
    system senses within about a millisecond. Frozen flow describes AO
    telemetry well over tens of milliseconds (Poyneer, van Dam & Véran 2009),
    which needs the sensed scales to outlive that; with the scaling above that
    means a `tau_boil` of the order of a second or more. Start from 1-10 s and
    use smaller values only on purpose.

How much does a frame change? The mean-square change of the screen at one
point over a step `dt` is the space-time structure function at zero
separation, `atm.structure_function(p, p, dt)`. For one static (no wind)
layer with `r0 = 0.15 m` at 500 nm and `L0 = 25 m`, at `dt = 1 ms` (a 1 kHz
loop), from the theory (an 8 m pupil, piston removed, agrees with simulated
frames to a few percent):

| `tau_boil` [s] | AR(1) point [nm rms] | AR(1) 8 m pupil, piston removed | AR(2) point | AR(2) 8 m pupil, piston removed |
|---|---|---|---|---|
| 0.01 | 871 | 646 | 496 | 420 |
| 0.05 | 407 | 307 | 123 | 109 |
| 0.2 | 206 | 156 | 33 | 30 |
| 1 | 92 | 70 | 7 | 6 |
| 5 | 41 | 31 | 1 | 1 |

### Temporal models: `boil_model`

- `boil_model="ar1"` (default): each mode is a first-order autoregression
  (an Ornstein-Uhlenbeck process), `c <- a c + sqrt(1 - a^2) w` with
  `a = exp(-dt/tau)` and a fresh draw `w` every step. Those innovations are
  white, so the process is continuous but nowhere smooth: the change over a
  step grows as `sqrt(dt/tau)`, a 1 ms step changes a mode by `sqrt(2 dt/tau)`
  of its rms, and no predictor can anticipate it from past frames.
- `boil_model="ar2"`: a critically damped second-order process (a
  continuous-time AR(2), Matérn-3/2 correlation `(1 + 2 lag/tau)
  exp(-2 lag/tau)`). Each mode carries a rate as well as its value, so it is
  differentiable in time and the change over a step grows as `dt/tau`; real
  boiling is smooth at short lags in the same way. Its integral time scale is
  `tau`, as for `"ar1"`, so the two decorrelate over the same horizon and
  differ at short lags (the table above). The update is the exact
  discretisation for any `dt`. Spectral engine only; it doubles the boiling
  state and noise per step.

Both models work with either `boil_scaling`. Both start from the same `t = 0`
screen for a given seed. Boiling acts while
stepping (`frames`/`evolve`/`advance`), not for random-access `opd(t)`. Each
step is exact for any `dt`, so `atm.advance(dt)` jumps straight to a frame
`dt` later in one step, with the statistics of many short steps (see
[Histories of frames](scintillation.md#histories-of-frames)). A quick check
that it decorrelates a static (no-wind) layer:

```python
import numpy as np
import pyturb

layer = [pyturb.Layer(0.0, 1.0, wind_speed=0.0, L0=25.0)]
atm = pyturb.Atmosphere(layer, r0=0.15, n=64, tau_boil=0.05, seed=1)
first = pyturb.to_numpy(atm.opd())
later = [pyturb.to_numpy(o) for _, o in atm.frames(dt=0.01, steps=11)][-1]
print(np.corrcoef(first.ravel(), later.ravel())[0, 1])   # well below 1 after 2 tau
```

### Uniform boiling and measured decay rates

The `f^(-2/3)` law is a dimensional argument about eddy lifetimes; what has
been measured on sky is how fast a whole layer decorrelates. Following
individual layers with the GeMS wide-field laser-guide-star system, Guesalaga
et al. (2014, MNRAS 440, 1925) found each layer's correlation falling roughly
linearly with lag, `rho(lag) ~ 1 - r lag`, at a rate that grows with the
layer's wind speed, `r = 0.157 v + 0.365` s⁻¹ (`v` in m/s): turbulence
decorrelates mainly with the distance it has travelled. That fit (74 layer
sequences, winds of 0-24 m/s) is a statement about the layer as a whole, not a
per-scale law.

`boil_scaling="uniform"` simulates that picture on the spectral engine: every
Fourier mode, main grid and subharmonics, has the one time constant
`tau_boil`, so the layer's correlation decays at the same rate at every scale
(as on the extruder, but periodic and fast).
`pyturb.theory.travel_decorrelation_time` turns wind speeds into the matching
time constants, `tau = 1 / (0.157 |v| + 0.365)`; with `boil_model="ar1"` its
retention `exp(-lag/tau)` has the measured initial slope.

```python
import pyturb

layers = pyturb.get_profile("two-layer")
tau = pyturb.theory.travel_decorrelation_time([l.wind_speed for l in layers])
atm = pyturb.Atmosphere(layers, r0=0.15, n=64, tau_boil=tau,
                        boil_scaling="uniform", seed=1)
print([round(t, 2) for t in tau], "s")           # faster wind, shorter tau
for t, opd in atm.frames(dt=1e-3, steps=5):
    pass
```

`Atmosphere.covariance` and `pyturb.theory.covariance(...,
boil_scaling="uniform")` give its space-time statistics: the frozen-flow
covariance times the retention `R(|lag| / tau_boil)`. Do not combine the
measured time constants with the default `"kolmogorov"` scaling: there
`tau_boil` is the outer-scale lifetime, and the scales an AO system senses
would decorrelate one to two orders of magnitude faster than measured. For
that scaling use the calibration below.

### Scale-dependent boiling calibrated on GeMS

To keep the eddy-turnover scaling and still match the measurements,
`pyturb.theory.gems_boil_time` gives the outer-scale lifetime per layer that
makes the sensed scales decay as GeMS saw. Guesalaga et al. measured the
decay of the energy in a 3x3-subaperture window around each layer's peak in
the space-time cross-correlation of tip/tilt-removed slopes (8 m pupil, 0.5 m
subapertures, lags 0.1-0.5 s). Computing the same windowed statistic from the
von Kármán spectrum (`L0 = 25 m`) with AR(1) modes boiling at `tau(f) =
tau_boil max(f L0, 1)^(-2/3)`, and solving for the `tau_boil` that decays at
the fitted rate `0.157 v + 0.365` s⁻¹, gives 2.5, 1.4 and 0.74 s at 5, 10
and 20 m/s. `gems_boil_time` is linear in `1 / tau` between those speeds and
holds the end values outside them.

```python
import pyturb

layers = pyturb.get_profile("two-layer")
tau = pyturb.theory.gems_boil_time([l.wind_speed for l in layers])
atm = pyturb.Atmosphere(layers, r0=0.15, n=64, tau_boil=tau, seed=1)
print([round(t, 2) for t in tau], "s")           # outer-scale lifetimes
```

With these lifetimes, 0.5 m structure (`f = 2` m⁻¹, a GeMS subaperture) in a
10 m/s layer lives `1.4 s / (2 * 25)^(2/3) = 0.10 s`, and 10 cm structure
35 ms. The calibration assumes `L0 = 25 m`: at
another outer scale, recalibrating gives lifetimes about 0.63, 1.49 and 2.3
times these for `L0` = 10, 50 and 100 m (at 10 m/s). Pair it with the default
`boil_model="ar1"`, which it was calibrated for.

## Space-time covariance

Estimators (MMSE, predictive, Kalman) need the statistics the simulator
draws from. `pyturb.theory.covariance` gives `<phi(a, t) phi(b, t + lag)>`
between any pupil points for a layered profile under frozen flow (pyturb's
convention `phi(r, t) = phi_0(r + wind_vector * t)`), with the boiling model
above computed as a Hankel transform of the von Kármán PSD times each mode's
retention; `pyturb.theory.space_time_structure_function` is its structure
function. `Atmosphere.covariance` and `Atmosphere.structure_function` fill in
the atmosphere's own layers, line-of-sight r0, L0, `tau_boil`, `boil_model`
and `boil_scaling` and return OPD (m²):

```python
import numpy as np
import pyturb

atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, n=64,
                                     diameter=8.0, tau_boil=2.0, seed=1)
x = np.linspace(-1.0, 1.0, 5)
actuators = np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)   # (x, y) [m]
lags = [0.0, 1e-3, 2e-3]
cov = atm.covariance(actuators, actuators, lags)                  # (3, 25, 25) m^2
print(cov.shape, np.sqrt(cov[0, 0, 0]) * 1e9, "nm rms per point")
```

At zero lag this is the closed-form von Kármán covariance
(`pyturb.phase_covariance`) summed over layers. It is the infinite-screen
statistic: a simulated screen lacks the power above its grid's Nyquist
frequency, and the spectral engine's screen has no mean over its FFT grid,
so piston-sensitive quantities (the covariance itself, one point's change
over time) come out lower in simulation. Piston-free statistics such as
differences between points match within ensemble error; that is what the test
suite checks. The theory needs a finite `L0` for the covariance (the
Kolmogorov variance is infinite) and for any boiling layer. `aobasis` (2.3+)
has a static `von_karman_covariance` too; it uses the rounded `6.88`
structure-function constant, so its values differ from pyturb's (exact
`6.8839...`) by about 0.06%.

## Laser guide star cone effect

A beacon at range `H` sees a layer at height `h` through a cone, so the
footprint shrinks by `1 - h/H` and high layers are sampled over a smaller
area than a star at infinity sees: focal anisoplanatism. Set `lgs_altitude`
for the whole atmosphere, or give individual directions their own source with
`(thx, thy, altitude)`:

```python
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, n=128,
                                     diameter=8.0, seed=1)
ngs, lgs = atm.opd(0.0, directions=[(0, 0, None), (0, 0, 90e3)])
cone = pyturb.to_numpy(lgs - ngs)
print(f"cone-effect error {cone.std() * 1e9:.0f} nm rms")
```

Both engines support it, as does `opd_at`. On the extruder, build with
`lgs_altitude=None` when you mix natural and laser sources (the buffer must
cover the widest footprint).

## Chromatic OPD

Turbulent OPD is achromatic to first order; outputs are OPD in metres unless
you pass `wavelength=`. For the small chromatic term from air dispersion:

- `dispersion="edlen"`: scales the OPD by the dry-air refractivity ratio
  `(n(λ) - 1) / (n(λ_ref) - 1)` (Edlén 1966), about 1–2% from the visible to
  the near-IR;
- `dispersion="ciddor"` with `wet_fraction`: blends in the water-vapour
  dispersion (Ciddor 1996), which matters in the mid-IR and for
  interferometry (the "wet–dry" problem). `wet_fraction` is the user-supplied
  share of the turbulent refractivity carried by water vapour; there are no
  weather inputs.

```python
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, n=64,
                                     dispersion="ciddor", wet_fraction=0.3, seed=1)
opd = pyturb.to_numpy(atm.opd())                  # metres, achromatic
for lam in (0.5e-6, 1.65e-6, 2.2e-6):
    phase = pyturb.to_numpy(atm.opd(wavelength=lam))
    scale = (phase * lam / (2 * np.pi)).std() / opd.std()
    print(f"{lam * 1e6:5.2f} um: effective OPD x{scale:.4f}")
```

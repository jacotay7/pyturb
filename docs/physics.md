# Boiling, LGS cone and dispersion

Three effects on top of frozen-flow turbulence, each with a worked example.
The [theory module](api.md#theory) gives the reference curves to check them
against.

## Boiling

Real turbulence evolves as well as blowing past. `tau_boil` [s] (scalar or one
per layer) relaxes each layer toward fresh, statistically identical
turbulence while preserving its spatial statistics (so r0 is unchanged):

- `engine="spectral"`: each Fourier mode is an AR(1) process with time
  constant `tau(f) = tau_boil * (f / f_ref)^(-2/3)` (Kolmogorov eddy
  turnover; `f_ref = 1/L0`), so fine structure decorrelates faster than the
  outer scale.
- `engine="extrude"`: the readable ring buffer blends toward a fresh extruded
  screen at the single rate `tau_boil` for every scale; non-periodic, but much
  slower per frame (it warns).

Boiling acts while stepping (`frames`/`evolve`), not for random-access
`opd(t)`. A quick check that it decorrelates a static (no-wind) layer:

```python
import numpy as np
import pyturb

layer = [pyturb.Layer(0.0, 1.0, wind_speed=0.0, L0=25.0)]
atm = pyturb.Atmosphere(layer, r0=0.15, n=64, tau_boil=0.05, seed=1)
first = pyturb.to_numpy(atm.opd())
later = [pyturb.to_numpy(o) for _, o in atm.frames(dt=0.01, steps=11)][-1]
print(np.corrcoef(first.ravel(), later.ravel())[0, 1])   # well below 1 after 2 tau
```

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

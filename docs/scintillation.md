# Scintillation

pyturb's layers are phase screens at different heights. Between them, light
diffracts: the phase of the high layers turns into intensity fluctuations by
the time it reaches the telescope. That is scintillation, the twinkling of
stars, and it is what a pupil-plane intensity measurement (a
Shack-Hartmann's subaperture fluxes, a pupil camera) sees.

`Atmosphere.pupil_field` propagates a star's light through the layers. A unit
plane wave enters above the highest layer, takes each layer's phase, and
propagates down to the next layer, and from the lowest layer to the telescope,
with the paraxial angular spectrum. It returns the complex pupil field at one
or several wavelengths, their weighted intensity and the summed OPD.

## From a profile to the pupil field

```python
import numpy as np
import pyturb

atm = pyturb.Atmosphere.from_profile("paranal-median", r0=0.15, zenith_angle=30.0,
                                     diameter=2.0, n=256, seed=1)
pf = atm.pupil_field(wavelength=0.7e-6)
print(pf.field.shape, pf.field.dtype)      # (256, 256) complex64
print(float(pf.intensity.mean()))           # 1: energy is conserved
opd = pf.opd                                # the same OPD as atm.opd(t=atm.time)
```

The scintillation index is the variance of the relative intensity at a point.
`Atmosphere.scintillation_index` gives the weak-perturbation theory for a
point source and a point aperture (Roddier 1981): `19.2 lambda^(-7/6)
sec(z)^(11/6) integral Cn^2(h) h^(5/6) dh`. A few independent realisations
reproduce it:

```python
import numpy as np
import pyturb

measured = []
for seed in range(4):
    atm = pyturb.Atmosphere.from_profile("paranal-median", r0=0.15, zenith_angle=30.0,
                                         diameter=2.0, n=256, seed=seed)
    intensity = pyturb.to_numpy(atm.pupil_field(wavelength=0.7e-6).intensity)
    measured.append(np.var(intensity))
print(np.mean(measured), atm.scintillation_index(0.7e-6))   # both about 0.12
```

Here the pixel (7.8 mm) is a tenth of the Fresnel scale `sqrt(lambda h)` of
the highest layers, and the grid holds all but about 1% of the scintillation
power. A pixel half the Fresnel scale misses more than 10% (the power above
the grid's Nyquist frequency). The ground layer adds phase but no intensity
fluctuation, and the high layers dominate: the index grows as `h^(5/6)`.

### Several wavelengths

Pass a sequence of wavelengths for a leading wavelength axis on `field`, and
`weights` (e.g. the photon spectrum over the band, normalised to sum to 1) for
the polychromatic `intensity`:

```python
import pyturb

atm = pyturb.Atmosphere.from_profile("paranal-median", r0=0.15, diameter=2.0,
                                     n=128, seed=1)
pf = atm.pupil_field(wavelength=[0.6e-6, 0.7e-6, 0.8e-6], weights=[1.0, 2.0, 1.0])
print(pf.field.shape, pf.intensity.shape)   # (3, 128, 128) (128, 128)
```

With `dispersion`, each wavelength's OPD is scaled as in `Atmosphere.opd`.

### Off-axis stars

`direction=(thx, thy)` [arcsec] propagates the light of a star in that
direction (within `field_of_view`). In the paraxial approximation the field of
a tilted plane wave is the tilt times the on-axis propagation through each
layer shifted by `theta * range`, the same footprints `Atmosphere.opd` uses
for `directions`; `pupil_field` returns that part, relative to the
unperturbed tilted wave. The off-axis OPD alone is `atm.opd(t,
directions=[...])`:

```python
import pyturb

atm = pyturb.Atmosphere.from_profile("paranal-median", r0=0.15, diameter=2.0,
                                     n=128, field_of_view=10.0, seed=1)
star = atm.pupil_field(direction=(10.0, 0.0), wavelength=1.65e-6)
target = atm.opd(t=atm.time, directions=[(5.0, 0.0), (0.0, 5.0)])  # (2, 128, 128)
```

## Histories of frames

Frozen flow is evaluated at any time directly; boiling is the only state that
steps. `Atmosphere.advance(dt)` moves the clock and boils every layer by one
exact step of `dt`, without computing an output: an AR(1) mode sampled every
`s` loop periods is itself an AR(1) with coefficient `rho^s` and innovation
variance `1 - rho^(2s)`, so one step of `s T` has the statistics of `s` steps
of `T` (`"ar2"` uses its exact transition too). A history of frames tens of
milliseconds apart then costs one boiling step per frame. `frames` and
`evolve` take one step per `dt`; inside a `frames` loop, or after `evolve` or
`advance`, `pupil_field()` (default `t = atm.time`) matches the latest frame.

```python
import pyturb

layers = pyturb.get_profile("paranal-median")
tau = pyturb.theory.gems_boil_time([layer.wind_speed for layer in layers])
atm = pyturb.Atmosphere(layers, r0=0.15, diameter=2.0, n=128, tau_boil=tau, seed=1)
history = []
for step in (0.0, 0.005, 0.01, 0.015):          # frames at 0, 5, 15 and 30 ms
    if step:
        atm.advance(step)
    history.append(atm.pupil_field(wavelength=0.7e-6))
```

The spectral screen is periodic: a history longer than `atm.time_to_wrap`
(here 62 ms for the 32 m/s layer on a 2 m screen) repeats turbulence; widen
the screen with `oversample` or `diameter` for longer ones.

`pyturb.theory.gems_boil_time` gives each layer the outer-scale lifetime
`tau_boil` for the default scale-dependent boiling (`tau(f) = tau_boil
max(f L0, 1)^(-2/3)`) that matches the decay Guesalaga et al. (2014) measured
per layer on GeMS: 2.5, 1.4 and 0.74 s at 5, 10 and 20 m/s, linear in
`1 / tau` between and held outside (see
[Boiling](physics.md#scale-dependent-boiling-calibrated-on-gems)).

## Conventions

- The field takes each layer's phase as `exp(+i 2 pi OPD / lambda)` and
  propagates over a range `z > 0` (downstream) with the paraxial transfer
  function `exp(-i pi lambda z f^2)`, `f` in cycles per metre. This is the
  layer-to-layer step `exp(-i pi lambda dh kappa^2)` usually written for
  multi-layer propagation, and exactly
  `aocore.AngularSpectrumPropagator(..., paraxial=True)`, which pyturb uses;
  no sign conversion is needed between them. Both belong to the
  `exp(+i k z)` convention of aocore, in which an OPD ramp `a x` images at
  `+a` (`CONVENTIONS.md` rule 3.1, checked in the test suite).
- In that convention a positive OPD bump (a converging lens) focuses light: to
  first order the relative intensity is `delta I(f) = 2 k sin(pi lambda h f^2)
  OPD(f)` for a layer at range `h` (`pyturb.theory.scintillation_transfer`),
  or `delta I = -h laplacian(OPD)` below the Fresnel frequency
  `1 / sqrt(lambda h)` (transport of intensity).
- Ranges are line-of-sight: `altitude * sec(zenith_angle)`.
- A star in direction `theta` arrives as the plane wave `exp(-i k theta . r)`,
  whose ray through pupil point `r` crossed the layer at range `h` at
  `r + theta h`: pyturb's off-axis footprints. The returned field omits that
  tilt and a constant phase.

## Theory

`pyturb.theory.scintillation_transfer(f, altitude, wavelength)` is the
first-order transfer function from a layer's OPD to relative intensity. The
layers add their contributions, and a band adds its wavelengths' transfer
functions with the intensity weights, mode by mode. Times another transfer
function and the OPD spectrum, it gives analytic cross-spectra, e.g. between
subaperture fluxes and slopes.

`pyturb.theory.scintillation_index(layers, r0, wavelength, zenith_angle=...)`
integrates its square against the Kolmogorov spectrum (Roddier's closed form).
Both are weak-perturbation results, valid while the index is well below 1.

## Limitations

- **Thin layers.** Each layer is one phase screen at one range; a thick layer
  is several thin ones (e.g. `discretize_cn2` with more layers).
- **Periodic screens.** The propagation runs on the spectral engine's
  periodic FFT screen (`n_screen` pixels): light leaving one edge enters at the
  other, consistent with the periodic turbulence. `engine="extrude"` is not
  supported (its screens are not periodic).
- **Paraxial.** The transfer function is the Fresnel approximation, exact for
  the angles any atmospheric grid holds (`lambda f << 1`).
- **Subharmonics.** Scales larger than the screen are applied as phase at the
  pupil: over atmospheric ranges they do not diffract (Fresnel number
  `L^2 / (lambda z)` in the thousands), and the shift they give the pattern
  below a layer (range times their slope) is left out: 2-6 mm rms for 8 m and
  2 m screens of `paranal-median` at `r0 = 0.15 m`, under half a pixel.
- **Stars only.** The source is a plane wave from infinity; a laser guide star
  (a spherical wave) is not supported.
- **Sampling.** The pixel must resolve the Fresnel scale `sqrt(lambda h)` and
  the intensity structure, and `r0` at the propagated wavelength should span
  several pixels; otherwise the field's spectrum reaches the grid's Nyquist
  frequency. Energy is conserved regardless, but power beyond Nyquist is
  lost or aliased.
- **No aperture averaging.** The intensity is per pixel; integrate it over
  subapertures for fluxes. `scintillation_index` is the point-aperture value.
- **Threads.** The CPU transforms run through aocore's backend, whose thread
  count is `AOCORE_FFT_WORKERS`, not `pyturb.set_threads`.

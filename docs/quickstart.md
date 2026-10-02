# Quickstart

## An atmosphere in one line

```python
import numpy as np
import pyturb

atm = pyturb.Atmosphere.from_profile(
    "paranal-median",   # named Cn2/wind profile
    seeing=0.8,         # arcsec @ 500 nm, at zenith
    zenith_angle=30,    # deg
    diameter=8.0,       # telescope pupil [m]
    n=512,              # pixels across the pupil
    seed=1,
)
```

## Closed-loop OPD frames

`frames()` yields `(time, opd)` where `opd` is `(n, n)` **in metres**:

```python
for t, opd in atm.frames(dt=1e-3, steps=200):
    ...                 # opd is a device array; pyturb.to_numpy(opd) to copy back
```

The default `engine="spectral"` is exact and fast but **periodic**: each layer's
screen repeats after `n * pixel_scale` metres of wind travel, so a run longer
than `atm.time_to_wrap` (0.25 s for this profile's 32 m/s layer on an 8 m
pupil) reuses turbulence and raises `PeriodicWrapWarning`.

For long runs where a repeating screen would bias the statistics, use the
non-periodic extruder engine:

```python
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8,
                                     engine="extrude")
```

For an offline time series (training data, PSD studies, pre-generated loops),
ask for many times at once: on the GPU the spectral engine evaluates them in
one batched transform, several times faster than stepping frame by frame.

```python
times = np.arange(512) * 1e-3                    # 0.5 s at 1 kHz
series = atm.opd(t=times)                        # (512, n, n); no boiling
```

## Monte-Carlo ensembles

```python
opds = atm.sample(256)              # (256, n, n) independent integrated OPDs
```

## Off-axis / tomography

```python
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8,
                                     field_of_view=30, n=512, seed=1)
opds = atm.opd(t=0.0, directions=[(0, 0), (10, 0), (0, 10)])   # arcsec offsets
```

Each direction can carry its own source: `(thx, thy, altitude)` with a laser
guide star's altitude [m], or `None` for a natural star or science target. An
LTAO/MCAO case (LGS asterism, NGS and science) then reads one turbulence
realisation in one call:

```python
lgs = [(30 * np.cos(a), 30 * np.sin(a), 90e3) for a in np.linspace(0, 2 * np.pi, 4)[:-1]]
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8,
                                     field_of_view=40, n=256, seed=1)
opds = atm.opd(t=0.0, directions=lgs + [(20, 0, None), (0, 0, None)])
```

To sample arbitrary points instead of the pupil grid — DM actuators,
sub-apertures, a sparse or multi-aperture layout — use `opd_at` with
coordinates in metres from the pupil centre (grow the screen with
`oversample` to reach beyond the pupil):

```python
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, n=256,
                                     oversample=4, seed=1)
x = np.array([-12.0, 0.0, 12.0])                  # three apertures on a 24 m baseline
values = atm.opd_at(x, np.zeros(3), t=0.0)        # OPD [m] at those points
```

## GPU

Everything above takes `device="gpu"` (requires CuPy); arrays come back as CuPy
and stay on the device until you call `pyturb.to_numpy(...)`.

```python
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, device="gpu")
```

## Boiling

Real turbulence is not perfectly frozen. `tau_boil` (seconds, scalar or one
per layer) adds temporal decorrelation on top of the wind while keeping the
spatial statistics; it acts while stepping with `frames()`/`evolve()`:

```python
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, n=256,
                                     tau_boil=0.05, seed=1)
for t, opd in atm.frames(dt=1e-3, steps=100):
    ...
```

The spectral engine boils each Fourier mode at its own rate (fine structure
faster); the extruder decorrelates all scales at `tau_boil` and is markedly
slower (it warns).

## Laser guide star cone effect

A beacon at finite altitude sees each layer through a cone, shrinking that
layer's footprint by `1 - h / lgs_altitude`:

```python
lgs = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, n=256,
                                     lgs_altitude=90e3, seed=1)
ngs = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, n=256,
                                     seed=1)
cone_error = pyturb.to_numpy(lgs.opd() - ngs.opd())   # same seed, same turbulence
```

## Single screens and layers

The building blocks are usable on their own; they return phase in radians at
the wavelength `r0` is quoted at:

```python
gen = pyturb.PhaseScreen(n=256, pixel_scale=0.02, r0=0.15, L0=25.0, seed=0)
batch = gen.generate(32)                       # (32, 256, 256) independent screens

layer = pyturb.InfinitePhaseScreen(n=128, pixel_scale=0.05, r0=0.15, seed=0)
phase = layer.advance(0.37)                    # blow 0.37 px along axis 0; never repeats
```

## Wavelengths and OPD

OPD is achromatic and returned in metres. Ask any output method for phase at a
wavelength, or convert with the helpers:

```python
phase = atm.opd(wavelength=1.65e-6)                    # radians at H band
phase = pyturb.opd_to_phase(atm.opd(), 1.65e-6)        # equivalently
```

OPD is achromatic by default. For the small chromatic term from air
dispersion, build with `dispersion="edlen"` (dry air) or `dispersion="ciddor"`
plus a `wet_fraction` (water vapour, for the mid-IR and interferometry); it
only changes outputs requested with `wavelength=`:

```python
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, n=256,
                                     dispersion="ciddor", wet_fraction=0.2)
phase_k = atm.opd(wavelength=2.2e-6)
```

Print your machine's throughput:

```python
pyturb.benchmark(n=512, device="gpu")
```

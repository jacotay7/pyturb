# Quickstart

## An atmosphere in one line

```python
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
for t, opd in atm.frames(dt=1e-3, steps=2000):
    ...                 # opd is a device array; pyturb.to_numpy(opd) to copy back
```

For long runs where a repeating screen would bias the statistics, use the
non-periodic extruder engine:

```python
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8,
                                     engine="extrude")
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

## GPU

Everything above takes `device="gpu"` (requires CuPy); arrays come back as CuPy
and stay on the device until you call `pyturb.to_numpy(...)`.

```python
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, device="gpu")
```

## Wavelengths and OPD

OPD is achromatic and returned in metres. Ask any output method for phase at a
wavelength, or convert with the helpers:

```python
phase = atm.opd(wavelength=1.65e-6)                    # radians at H band
phase = pyturb.opd_to_phase(atm.opd(), 1.65e-6)        # equivalently
```

Print your machine's throughput:

```python
pyturb.benchmark(n=512, device="gpu")
```

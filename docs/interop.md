# Using pyturb with other tools

`pyturb` outputs plain arrays (NumPy on CPU, CuPy on GPU) of **OPD in metres**,
so it drops into most optics pipelines without a hard dependency. `opd()`,
`frames()` and `sample()` all return `(n, n)` (or batched) arrays; convert to
phase at any wavelength with `pyturb.opd_to_phase(opd, wavelength)` or the
`wavelength=` argument. These recipes are deliberately dependency-light — copy
the few lines you need.

## HCIPy

Wrap a pyturb OPD as an HCIPy `Wavefront` phase, or feed frames into a
closed-loop sim:

```python
import numpy as np, hcipy, pyturb

pupil_grid = hcipy.make_pupil_grid(512, 8.0)
aperture = hcipy.make_circular_aperture(8.0)(pupil_grid)
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8,
                                     diameter=8.0, n=512, seed=1,
                                     engine="extrude")

wavelength = 1.65e-6
for t, opd in atm.frames(dt=1e-3, steps=1000):
    phase = pyturb.to_numpy(opd) * (2 * np.pi / wavelength)     # rad at 1.65 um
    wf = hcipy.Wavefront(aperture * np.exp(1j * phase.ravel()), wavelength)
    # ... propagate wf through your optical system
```

pyturb complements HCIPy: HCIPy owns diffraction propagation and scintillation;
pyturb gives you fast, non-periodic, GPU turbulence to drive it. (HCIPy's own
`InfiniteAtmosphericLayer` is fine too — use pyturb when you want the GPU/batched
speed or the extra profiles/analysis.)

## poppy

poppy consumes wavefront-error maps directly (OPD in metres), so a pyturb screen
is a drop-in `ArrayOpticalElement` / OPD:

```python
import numpy as np, poppy, pyturb

atm = pyturb.Atmosphere.from_profile("mauna-kea", seeing=0.7, diameter=8.0, n=512)
opd = pyturb.to_numpy(atm.opd())                    # metres

osys = poppy.OpticalSystem()
osys.add_pupil(poppy.CircularAperture(radius=4.0))
osys.add_pupil(poppy.ArrayOpticalElement(
    opd=opd, pixelscale=atm.pixel_scale, name="atmosphere"))
psf = osys.calc_psf(1.65e-6)
```

Or save the screen to FITS and load it wherever your pipeline reads OPD maps:

```python
pyturb.save("turbulence.fits", atm.opd(), **atm.metadata)   # OPD [m] + header
```

## DM fitting / reconstruction

Project an OPD onto Zernike modes with the built-in analysis helpers (e.g. to
estimate a DM command or a modal residual):

```python
import pyturb
from pyturb import analysis

atm = pyturb.Atmosphere.from_profile("keck", seeing=0.8, diameter=10.0, n=256)
opd = atm.opd()

basis = analysis.zernike_basis(n_modes=50, n_pixels=256)     # build once
coeffs = analysis.zernike_decompose(opd, 50, basis=basis)    # metres per mode
correction = (coeffs[:, None, None] * basis).sum(axis=0)     # fitted wavefront
residual = pyturb.to_numpy(opd) - correction                 # post-DM residual
```

For a real DM, replace `basis` with your influence-function matrix and use the
same least-squares projection (`numpy.linalg.lstsq`) that `zernike_decompose`
uses internally. `analysis.noll_residual_variance(n_modes, D, r0)` gives the
theoretical fitting-error floor to check against.

## General notes

- **Stay on the device.** On GPU, keep arrays as CuPy through your pipeline and
  only `pyturb.to_numpy(...)` at the end — device→host copies are the usual
  bottleneck in a closed loop.
- **Wavelength.** OPD is achromatic; if you need the small air-dispersion term,
  build the atmosphere with `dispersion="edlen"` and read out with
  `wavelength=`.
- **Reproducibility.** `Atmosphere.metadata` (and `pyturb.save`) record the
  parameters and pyturb version, so a saved screen carries its provenance.

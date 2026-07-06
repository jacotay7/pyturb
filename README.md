# pyturb

[![CI](https://github.com/jacotay7/pyturb/actions/workflows/ci.yml/badge.svg)](https://github.com/jacotay7/pyturb/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyturb.svg)](https://pypi.org/project/pyturb/)
[![Python](https://img.shields.io/pypi/pyversions/pyturb.svg)](https://pypi.org/project/pyturb/)
[![Docs](https://img.shields.io/badge/docs-jacotay7.github.io%2Fpyturb-teal.svg)](https://jacotay7.github.io/pyturb/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Documentation: [jacotay7.github.io/pyturb](https://jacotay7.github.io/pyturb/)**

**Fast, GPU-optional atmospheric turbulence for adaptive optics.**

`pyturb` generates the optical path differences (OPD) that an adaptive-optics
system sees through the atmosphere: full **layered turbulence** for a
representative sky, with per-layer wind, **frozen-flow time evolution**,
off-axis directions, and standard site profiles. It runs on NumPy by default
and switches to CUDA (via CuPy) with a single argument.

## Install

```bash
pip install pyturb            # CPU (NumPy + SciPy)
pip install pyturb[cuda12]    # + CuPy for CUDA 12.x
pip install pyturb[cuda11]    # + CuPy for CUDA 11.x
pip install pyturb[accel]     # + Numba, faster CPU frozen flow
```

## Quickstart

```python
import pyturb

atm = pyturb.Atmosphere.from_profile(
    "paranal-median", seeing=0.8, zenith_angle=30, diameter=8.0, n=512, seed=1
)
print(atm.r0, atm.theta0, atm.tau0)      # Fried param, isoplanatic angle, tau0

for t, opd in atm.frames(dt=1e-3, steps=2000):
    ...                                   # (512, 512) OPD [m], frozen flow

ensemble = atm.sample(256)                # (256, 512, 512) Monte-Carlo OPDs

atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, device="gpu")
for t, opd in atm.frames(dt=1e-3, steps=2000):
    ...                                   # cupy arrays, ~3,100 fps at 512^2
```

See **[Quickstart](https://jacotay7.github.io/pyturb/quickstart/)** for
off-axis/tomography, boiling, LGS, non-periodic frozen flow, and the
lower-level `PhaseScreen`/`InfinitePhaseScreen` building blocks; and
**[Concepts](https://jacotay7.github.io/pyturb/concepts/)** for r0, L0, Cn²,
θ0 and τ0 if you're new to AO.

## Benchmarks

Full 9-layer Paranal atmosphere, closed-loop OPD frames/s, on an RTX 5090
(GPU) and a 32-core CPU (`pyturb[accel]`) — see `benchmarks/bench_suite.py`:

| screen | GPU spectral | GPU extrude | GPU Monte-Carlo screens/s | CPU spectral | CPU extrude |
|---|---|---|---|---|---|
| 256² | ~3,200 | ~4,500 | ~11,600 | ~1,090 | ~970 |
| 512² | ~3,100 | ~1,700 | ~3,200 | ~270 | ~180 |
| 1024² | ~1,500 | ~600 | ~650 | ~62 | ~57 |

Single-layer Monte-Carlo (`PhaseScreen.generate`) draws ~31,000 independent
512² screens/s on the GPU (~108,000 at 256²). Run
`python -c "import pyturb; pyturb.benchmark()"` on your own machine, or
`python benchmarks/bench_suite.py` for the full per-use-case sweep. A
head-to-head against aotools, soapy and HCIPy lives in
**[Comparison](https://jacotay7.github.io/pyturb/comparison/)**.

## Features

- **Layered `Atmosphere`** — named site profiles (`paranal-median`,
  `mauna-kea`, `keck`, `las-campanas`, HV 5/7, ...) or a custom `Cn²(h)` model,
  summed to pupil OPD with per-layer wind and airmass/zenith scaling.
- **Two frozen-flow engines** — `engine="spectral"` (default): exact
  sub-pixel shift-theorem translation, all layers in one FFT, periodic.
  `engine="extrude"`: Assémat–Wilson row extrusion, unbounded and
  non-periodic. Both use fused CUDA/Numba kernels on the hot path.
- **Off-axis / tomography** — `atm.opd(t, directions=[...])` batches several
  guide-star directions through one call.
- **Boiling** — temporal decorrelation on top of frozen flow (`tau_boil`).
- **LGS cone effect** — finite-range sodium beacon focal anisoplanatism
  (`lgs_altitude`).
- **Chromatic OPD** — achromatic by default; `dispersion="edlen"`/`"ciddor"`
  for the dry-air/water-vapour term.
- **Diagnostics** (`pyturb.analysis`) — Zernike decomposition, Noll (1976)
  mode variances, temporal PSD + power-law fit, angular decorrelation.
- **I/O** — `pyturb.save`/`pyturb.load` for FITS (optional astropy) and
  `.npz`, with provenance metadata.
- **GPU-optional** — every class takes `device="gpu"` (CuPy); results are
  numerically identical to CPU (to float round-off).
- **Validated against theory** — structure function, Zernike spectrum,
  temporal PSD and angular decorrelation checked against analytic predictions
  in CI; see **[Validation](https://jacotay7.github.io/pyturb/validation/)**.

See the **[API reference](https://jacotay7.github.io/pyturb/api/)** for every
public function and class, and
**[Interop](https://jacotay7.github.io/pyturb/interop/)** for recipes with
HCIPy, poppy, and DM-fitting loops.

## License

MIT

# pyturb

[![CI](https://github.com/jacotay7/pyturb/actions/workflows/ci.yml/badge.svg)](https://github.com/jacotay7/pyturb/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyturb.svg)](https://pypi.org/project/pyturb/)
[![Python](https://img.shields.io/pypi/pyversions/pyturb.svg)](https://pypi.org/project/pyturb/)
[![Docs](https://img.shields.io/badge/docs-jacotay7.github.io%2Fpyturb-teal.svg)](https://jacotay7.github.io/pyturb/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Documentation: [jacotay7.github.io/pyturb](https://jacotay7.github.io/pyturb/)**

**Fast, GPU-optional atmospheric turbulence for adaptive optics.**

<p align="center">
  <img src="examples/keck_showcase.webp" width="503" alt="Animated Keck atmosphere showcase: frozen flow, boiling, and on/off-axis LGS turbulence.">
</p>

`pyturb` generates the optical path differences (OPD) that an adaptive-optics
system sees through the atmosphere: full **layered turbulence** for a
representative sky, with per-layer wind, **frozen-flow time evolution**,
off-axis directions, and standard site profiles. It runs on NumPy by default
and switches to CUDA (via CuPy) with a single argument.

## Install

```bash
pip install pyturb            # CPU (NumPy + SciPy)
pip install pyturb[cuda13]    # + CuPy for CUDA 13.x (with CUDA headers)
pip install pyturb[cuda12]    # + CuPy for CUDA 12.x (with CUDA headers)
pip install pyturb[cuda11]    # + CuPy 13.x for CUDA 11.x
pip install pyturb[accel]     # + Numba, faster CPU frozen flow
```

Pick the extra matching your driver's CUDA version (`nvidia-smi` shows it). If
you install CuPy yourself and the first GPU call fails with *"Failed to find
CUDA headers"*, install the headers too (`pip install "cupy-cuda12x[ctk]"`, or
`cupy-cuda13x[ctk]`) or point `CUDA_PATH` at a CUDA toolkit.

## Quickstart

```python
import pyturb

atm = pyturb.Atmosphere.from_profile(
    "paranal-median", seeing=0.8, zenith_angle=30, diameter=8.0, n=512, seed=1
)
print(atm.r0, atm.theta0, atm.tau0)      # Fried param, isoplanatic angle, tau0

for t, opd in atm.frames(dt=1e-3, steps=200):
    ...                                   # (512, 512) OPD [m], frozen flow
# The default spectral engine is periodic: keep runs under atm.time_to_wrap
# (0.25 s here) or pass engine="extrude" for unbounded, non-periodic flow.

ensemble = atm.sample(256)                # (256, 512, 512) Monte-Carlo OPDs

atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, device="gpu")
for t, opd in atm.frames(dt=1e-3, steps=200):
    ...                                   # cupy arrays; ~4,600 fps at 512^2 on an RTX 4060
```

See **[Quickstart](https://jacotay7.github.io/pyturb/quickstart/)** for
off-axis/tomography, boiling, LGS, non-periodic frozen flow, and the
lower-level `PhaseScreen`/`InfinitePhaseScreen` building blocks; and
**[Concepts](https://jacotay7.github.io/pyturb/concepts/)** for r0, L0, Cn²,
θ0 and τ0 if you're new to AO.

## Benchmarks

Full 9-layer Paranal atmosphere, frames/s (Monte-Carlo: screens/s).

**pyturb 1.1** on an RTX 4060 with an Arm (Neoverse-N1) host
([artifact](benchmarks/artifacts/v1.1.0-arm-rtx4060.json)):

| screen | GPU spectral | GPU `opd(t=times)` | GPU extrude | GPU Monte-Carlo |
|---|---|---|---|---|
| 256² | 6,451 | 27,510 | 1,174 | 33,292 |
| 512² | 4,591 | 6,653 | 482 | 6,993 |
| 1024² | 1,304 | 1,166 | 120 | 1,715 |

Spectral frames replay a captured CUDA graph, so they no longer depend on the
host CPU's speed; `opd(t=times)` evaluates an offline time series in one
batched transform.

**pyturb 1.0.0** on an RTX 5090 and a 32-core x86 CPU (`pyturb[accel]`;
before the 1.1 CUDA-graph and batching work,
[artifact](benchmarks/artifacts/v1.0.0-reference.json)):

| screen | GPU spectral | GPU extrude | GPU Monte-Carlo screens/s | CPU spectral | CPU extrude |
|---|---|---|---|---|---|
| 256² | 3,219 | 4,489 | 104,981 | 1,014 | 2,571 |
| 512² | 3,133 | 1,733 | 29,789 | 283 | 324 |
| 1024² | 1,492 | 602 | 5,970 | 70 | 62 |

The Monte-Carlo column is `Atmosphere.sample()`, the full 9-layer atmosphere.
Layers that share an outer scale are drawn as one aggregate screen (their PSDs
add exactly), so a uniform-`L0` profile costs one FFT, not nine. All
Monte-Carlo figures are batched, device-resident throughput (a batch of up to
64 kept on the GPU); a single call, or one that copies its result to the host,
is lower. Measure your own machine with
`python -c "import pyturb; pyturb.benchmark()"`, or `python
benchmarks/bench_suite.py` for the full per-use-case sweep. More hardware and
the head-to-head against aotools, soapy and HCIPy:
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) and
**[Comparison](https://jacotay7.github.io/pyturb/comparison/)**.

## Features

- **Layered `Atmosphere`** — named site profiles (`paranal-median`,
  `mauna-kea`, `keck`, `las-campanas`, HV 5/7, ...) or a custom `Cn²(h)` model,
  summed to pupil OPD with per-layer wind and airmass/zenith scaling.
- **Two frozen-flow engines** — `engine="spectral"` (default): exact
  sub-pixel shift-theorem translation, all layers in one FFT, periodic.
  `engine="extrude"`: Assémat–Wilson row extrusion, unbounded and
  non-periodic, read out by a fused CUDA kernel (GPU) or Numba kernel (CPU,
  with `[accel]`). FFT screens default to the pupil's size (fastest); pass
  `oversample=2`–`4` when statistics across the whole pupil matter (tilt,
  low-order modes), since a pupil-sized periodic screen underestimates them.
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
- **GPU-optional** — every class takes `device="gpu"` (CuPy); the GPU path is
  statistically identical to CPU (validated against the same theory to
  numerical precision). CuPy and NumPy have independent RNG streams, so a given
  `seed` draws a *different* realisation on each backend — the guarantee is
  matched statistics, not a bit-for-bit copy.
- **Validated against theory** — structure function, Zernike spectrum,
  temporal PSD and angular decorrelation checked against analytic predictions
  in CI; see **[Validation](https://jacotay7.github.io/pyturb/validation/)**.

See the **[API reference](https://jacotay7.github.io/pyturb/api/)** for every
public function and class, and
**[Interop](https://jacotay7.github.io/pyturb/interop/)** for recipes with
HCIPy, poppy, and DM-fitting loops.

## License

MIT

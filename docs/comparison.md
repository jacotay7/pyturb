# pyturb vs aotools, soapy & HCIPy

A feature and benchmark comparison of `pyturb` against three widely used
open-source Python atmospheric phase-screen tools. Raw numbers (RTX 5090) are
in
[`benchmarks/RESULTS.md`](https://github.com/jacotay7/pyturb/blob/main/benchmarks/RESULTS.md);
reproduce with `python benchmarks/bench_compare.py`.

- **aotools** — [github.com/AOtools/aotools](https://github.com/AOtools/aotools)
- **soapy** — [github.com/AOtools/soapy](https://github.com/AOtools/soapy)
- **HCIPy** — [github.com/ehpor/hcipy](https://github.com/ehpor/hcipy)

Not covered here: **COMPASS** (CUDA C++, gitlab.obspm.fr/cosmic-rtc/compass),
a compiled, GPU-native end-to-end AO simulator — a different category of tool
than the three importable Python/NumPy/CuPy libraries above, and still the
fastest non-periodic GPU atmosphere in absolute terms.

## Feature matrix

| | pyturb | aotools | soapy | HCIPy |
|---|---|---|---|---|
| Primary purpose | atmosphere for AO | AO toolbox | full AO system sim | high-contrast imaging |
| GPU backend | **yes** (CuPy) | — | — | — |
| Generation engine | FFT + integrated subharmonics | FFT + subharmonics | (reuses aotools) | extruder init |
| Frozen-flow engine | spectral shift theorem + extruder | Assémat–Wilson extruder | large-screen panning | extruder + interpolation |
| Sub-pixel translation | yes | — | yes | yes |
| Arbitrary wind direction | yes | — | — | yes |
| Unbounded (non-periodic) | yes (`engine="extrude"`) | yes | yes | yes |
| Batched Monte-Carlo | yes | — | — | — |
| Boiling (temporal decorrelation) | yes | — | — | — |
| LGS cone effect | yes | — | — | — |
| Off-axis / tomography directions | yes | — | partial | yes |
| Named site profiles | yes | — | — | yes |
| Scintillation (Fresnel) | non-goal | — | — | yes |
| Tomographic reconstructors | — | yes | yes | modal |
| FITS screen I/O | yes (`pyturb.save`) | — | yes | — |
| Integrated r0 / θ0 / τ0 | yes | yes | — | — |
| OPD in metres (achromatic) | yes | — | — | — |

## Benchmarks

RTX 5090 + 32-core CPU (`pyturb[accel]`), 8 m pupil, 512² pupil, von Kármán
`r0 = 0.15 m` @ 500 nm, `L0 = 25 m`:

| metric | pyturb (GPU / CPU) | aotools | soapy | HCIPy |
|---|---:|---:|---:|---:|
| Generation, batched (screens/s) | 30,629 / 286 | 12 | 12 | n/a |
| Frozen flow, 1-layer (fps) | 5,700 / 670 | 5,083 | 5,034 | 52 |
| Frozen flow, 9-layer atmosphere (fps) | 3,133 / 283 | — | — | — |
| Structure-function error vs von Kármán (256², ensemble) | 1.2% | 3.1% | 1.5% | 0.8% |

`n/a` = no batched i.i.d.-generation entry point; `—` = not offered by that
library. The accuracy row is device-independent. Full tables, methodology and
uncertainties:
[`benchmarks/RESULTS.md`](https://github.com/jacotay7/pyturb/blob/main/benchmarks/RESULTS.md).

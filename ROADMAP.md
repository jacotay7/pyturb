# pyturb Roadmap

The core is complete: `Atmosphere.from_profile(...).frames(dt)` delivers
multi-layer, sub-pixel, arbitrary-direction OPD on CPU or GPU, via a periodic
spectral engine or a non-periodic extruder, with boiling, off-axis/tomography,
LGS cone, chromatic OPD, named profiles, an analysis toolkit, FITS/npz I/O, a
benchmark suite, a validation gallery, docs, and CI. Version **0.2.0** is
build-ready.

This file tracks only what is **not yet done**. Ground rule from here on
(unchanged): every physics feature lands with an ensemble-statistics test
against theory; every performance claim lands with a benchmark; every feature
lands with a docs page.

## Release — cut v0.2.0

These need a maintainer with the right accounts; the workflows are already
written (`.github/workflows/release.yml`, `docs.yml`).

- [ ] Configure the PyPI **trusted publisher** for `pyturb`, then push tag
      `v0.2.0` to publish (the workflow builds + uploads).
- [ ] Enable **GitHub Pages** (Settings → Pages → Source: GitHub Actions) so the
      docs site deploys on push to main.
- [ ] Add an **animated GIF** of a wind-blown multi-layer screen to the README
      hero (disproportionate adoption value; needs a recorded asset).
- [ ] **conda-forge** package, after the PyPI release lands.

## Performance — extruder engine

The spectral engine already hits target loop rates; the extruder is correct but
not yet fused for the GPU.

- [ ] **Batch the per-layer gathers into one kernel.** Today
      `ExtrudedAtmosphere.integrate` loops layers with ~16 bicubic gathers each
      (~120 fps for a 9-layer 512² screen vs ~800 fps spectral). Stack the
      layers and fuse the sampling to close the gap.
- [ ] **`evolve(dt)` convenience** keyed to per-layer `wind_speed`, mirroring
      HCIPy's `evolve_until(t)`, so callers step in seconds not pixels.
- [ ] **Fractal 2ⁿ stencil** (aotools-style) as an option over the current dense
      stencil — smaller covariance setup and memory at large `n`.

## Physics & profiles

- [ ] **Tomography-optimal profile compression** — `optimal_grouping` and GCTM
      (Saxenhuber 2017) as additional `discretize_cn2(method=...)` options
      beyond the current `"equivalent"`/`"centroid"`, for MCAO/tomography work.
- [ ] **Dry/wet chromatic split** — extend `dispersion="edlen"` (dry air) with a
      water-vapour term so the chromatic OPD is correct in the mid-IR and for
      interferometry (the "wet–dry" problem), where wet turbulence dominates.
- [ ] **More site profiles** — Cerro Pachón (Gemini/GMT) and Armazones (ELT);
      each is a few lines of cited layer numbers, following `keck` /
      `las-campanas`.

## Testing & typing

- [ ] **Self-hosted GPU CI job** (or a scheduled manual workflow) so the CuPy
      path is continuously tested; the current matrix is CPU-only and skips GPU
      tests. Parameterise the statistics tests over `device`.
- [ ] **Complete type hints** on all public signatures (the `py.typed` marker
      already ships; annotations are currently partial).

## Non-goals — deliberately out of scope

Kept here so the scope stays crisp; pyturb is *the atmosphere*, not an AO system.

- **Full AO system simulation** (WFS, DM, controllers) — never; that is soapy's
  job. pyturb is the turbulence that plugs into it.
- **Tomographic reconstructors / slope covariance** — aotools' domain.
- **Scintillation / Fresnel propagation between layers** — HCIPy/PROPER own
  amplitude effects; pyturb outputs phase/OPD. *Possible stretch only if pulled
  by demand:* a minimal angular-spectrum propagator over pyturb's layers.

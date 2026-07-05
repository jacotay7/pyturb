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

## Performance — extruder engine

The spectral engine already hits target loop rates; the extruder is correct but
not yet fused for the GPU.

- **Fractal 2ⁿ stencil — investigated, declined (see below).** The premise
      ("smaller covariance setup and memory") does not hold for pyturb: pyturb
      already uses the minimal 2-row stencil, so an aotools-style fractal 2ⁿ
      stencil has the *same* covariance size (~2·width points) but needs a
      ~4·width-deep ring buffer (much *more* memory). Its only real draw would be
      enabling non-periodic **Kolmogorov (L0=inf)** extrusion via Fried's
      reference-point/structure-function recurrence — but a validated prototype
      showed that produces a **large-scale-anisotropic** screen (along-wind
      structure function ~+16 %, cross-wind ~−37 % at large separations;
      along/cross ratio ~1.8 vs von Karman's ~1.05). That anisotropy is
      intrinsic to extruding an unbounded outer scale (the finite-L0 restriction
      is fundamental, not incidental), so shipping it would mean a physically
      wrong model. Guarded by `test_extruded_screen_is_isotropic_per_axis`,
      which checks each axis separately (the azimuthal average had hidden this).
- [ ] **Tighten the per-layer along-wind buffer margin.** Every `_ExtrudeLayer`
      currently reserves the same `field_of_view` margin sized for the
      worst-case (highest/slowest) layer; lower layers could use a tighter,
      geometry-derived margin. Memory/perf-only, not a correctness issue.

## Physics & profiles

- [ ] **Time-varying atmosphere parameters** — gusting/wind-direction drift
      over a run, time-variable Cn² profiles, and per-layer (rather than
      global) `inner_scale`/`power_law`. All are currently static-per-run;
      each is a genuine physics extension, not a bug, and would need its own
      theory-referenced validation the way `tau_boil` did.

## Non-goals — deliberately out of scope

Kept here so the scope stays crisp; pyturb is *the atmosphere*, not an AO system.

- **Full AO system simulation** (WFS, DM, controllers) — never; that is soapy's
  job. pyturb is the turbulence that plugs into it.
- **Tomographic reconstructors / slope covariance** — aotools' domain.
- **Scintillation / Fresnel propagation between layers** — HCIPy/PROPER own
  amplitude effects; pyturb outputs phase/OPD. *Possible stretch only if pulled
  by demand:* a minimal angular-spectrum propagator over pyturb's layers.

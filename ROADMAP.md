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

- [x] **Batch the per-layer gathers into one kernel.** Every layer's ring
      buffer is now a slab of one contiguous `(L, cap, W)` array, and the GPU
      readout is a single fused bicubic gather over all layers
      (`_integrate_batched`); the CPU keeps the per-layer loop
      (`_integrate_looped`), which its cache prefers, chosen by backend in
      `ExtrudedAtmosphere.integrate`. Measured on an RTX 5090, 9-layer
      paranal-median: **121 → 870 fps at 256² (7.1×)** and **118 → 290 fps at
      512² (2.5×)**, closing most of the gap to the spectral engine; CPU
      throughput is unchanged. Locked in by `test_gpu_batched_readout_matches_looped`.
- [x] **`evolve(dt)` convenience** keyed to per-layer `wind_speed`, mirroring
      HCIPy's `evolve_until(t)`, so callers step in seconds not pixels
      (`Atmosphere.evolve`). Repeated calls reproduce `frames(dt)` (boiling
      included); see `test_evolve_steps_in_seconds_and_matches_frames`.
- [ ] **Fractal 2ⁿ stencil** (aotools-style) as an option over the current dense
      stencil — smaller covariance setup and memory at large `n`.
- [ ] **Tighten the per-layer along-wind buffer margin.** Every `_ExtrudeLayer`
      currently reserves the same `field_of_view` margin sized for the
      worst-case (highest/slowest) layer; lower layers could use a tighter,
      geometry-derived margin. Memory/perf-only, not a correctness issue.
- [ ] **Profile the CPU path for a real bottleneck.** `set_fft_workers(-1)`
      only buys ~1.25× on the spectral engine's CPU throughput (~25 → ~31 fps
      for a 9-layer 512² job, vs. aotools' ~420 fps integer-pixel loop on the
      same job) — the per-frame cost isn't FFT-bound, but nothing has profiled
      where it actually goes.

## Extrude-engine fidelity and feature-combination gaps

Two related follow-ups on `engine="extrude"` that surfaced together: it has a
disclosed sub-pixel artifact, and it can't combine with several other features
because they're implemented against different internal representations.

- [ ] **Reduce the sub-pixel interpolation artifact.** The extruder shows a
      ~5-15% structure-function deficit at 1-2 px separations and ~7.5%
      pk-pk flicker in finest-scale power as a function of sub-pixel wind
      travel phase (bounded and regression-tested via
      `test_finescale_readout_flicker_is_bounded`, but not reduced). A
      supersampled internal buffer decimated down before readout, or a
      longer-support interpolation kernel with flatter high-frequency
      response, are the two candidate fixes — either needs validation that it
      doesn't trade this artifact for degraded long-range covariance from a
      shorter effective stencil in physical units.
- [ ] **LGS cone × boiling × engine mutual exclusivity.** `tau_boil` requires
      `engine="spectral"` (per-mode retention needs discrete Fourier modes);
      `lgs_altitude` requires `engine="extrude"` (per-layer resampling doesn't
      batch across the FFT engine). Closing either direction — boiling inside
      the ring-buffer formalism, or a chirp/scaled resampling of the LGS cone
      inside the batched spectral engine — is new numerical machinery, not a
      bug fix, but would remove a real combinatorial gap in the feature matrix
      (see `Atmosphere`'s "Feature compatibility" docstring section for the
      current state).

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
- [ ] **Time-varying atmosphere parameters** — gusting/wind-direction drift
      over a run, time-variable Cn² profiles, and per-layer (rather than
      global) `inner_scale`/`power_law`. All are currently static-per-run;
      each is a genuine physics extension, not a bug, and would need its own
      theory-referenced validation the way `tau_boil` did.

## Testing & typing

- [x] **Self-hosted GPU CI job** (manual/scheduled workflow) so the CuPy path
      is tested off the CPU-only matrix. GPU tests carry `@pytest.mark.gpu` and
      are skipped unless `pytest --run-gpu` is passed (see `tests/conftest.py`);
      `.github/workflows/gpu.yml` runs them on a `[self-hosted, gpu]` runner on
      demand and weekly. Statistics tests are parameterised over `device` via a
      fixture, so the same body checks NumPy and CuPy. (Remaining: register a
      GPU runner; the workflow is a no-op until one is online.)
- [ ] **Complete type hints** on all public signatures (the `py.typed` marker
      already ships). Currently 76% of public-function parameters and 75% of
      public-function return types are annotated (up from 14%/30%); remaining
      gaps are private implementation classes with no `__all__` entry
      (`_ExtrudeLayer`, `_LayerState`, `_ThreadedScipyFFT`) and local closures.
- [ ] **Audit broad statistical test tolerances repo-wide.** Several existing
      tests use wide sanity-check bands (e.g. `ratio > 0.8 and ratio < 1.2`);
      at least one (`test_gpu_extrude_and_spectral_match_theory`) was checked
      and found to deliberately exclude the fine-scale range where a known
      artifact lives, rather than masking it — but the rest haven't been
      audited one by one. Tightening needs a dedicated per-test statistical
      study (what each tolerance actually needs to not flake), not a blanket
      pass.
- [ ] **Increase the accuracy benchmark's shared ensemble size** beyond the
      current `count=120` default to narrow the bootstrap-uncertainty bands in
      `benchmarks/RESULTS.md` §3 further.

## Non-goals — deliberately out of scope

Kept here so the scope stays crisp; pyturb is *the atmosphere*, not an AO system.

- **Full AO system simulation** (WFS, DM, controllers) — never; that is soapy's
  job. pyturb is the turbulence that plugs into it.
- **Tomographic reconstructors / slope covariance** — aotools' domain.
- **Scintillation / Fresnel propagation between layers** — HCIPy/PROPER own
  amplitude effects; pyturb outputs phase/OPD. *Possible stretch only if pulled
  by demand:* a minimal angular-spectrum propagator over pyturb's layers.

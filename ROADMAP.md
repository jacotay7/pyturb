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
- [x] **Profile the CPU path for a real bottleneck.** Profiling the 9-layer
      512² spectral frame showed the cost was an `(L, n, n)` inverse FFT (one
      transform *per layer*) plus the per-layer `.sum(axis=0)` reductions, not
      the per-frame elementwise work. Since the inverse FFT and the subharmonic
      outer product are linear and every layer shares one FFT grid / basis,
      `_integrate` now collapses the layer axis **before** the transform (sum
      the shifted spectra to one `(n, n)` array, inverse-FFT once; sum each
      level's `3x3` subharmonic coefficients before the shared basis product).
      Mathematically identical (`test_spectral_integrate_equals_sum_of_layer_translates`,
      matches a per-layer reference to 1e-13). Measured on an RTX 5090, 9-layer
      paranal-median: **CPU 25 → 87 fps at 512² (3.4×)**, 133 → 459 fps at 256²;
      **GPU 865 → 1232 fps (1.4×)**. Reproduce with `benchmarks/bench_frames.py`.

## Extrude-engine fidelity and feature-combination gaps

Two related follow-ups on `engine="extrude"` that surfaced together: it has a
disclosed sub-pixel artifact, and it can't combine with several other features
because they're implemented against different internal representations.

- [x] **Reduce the sub-pixel interpolation artifact.** Added `interp="lanczos"`
      (6-tap Lanczos-3), the longer-support-kernel fix: a much flatter
      sub-Nyquist response than the default Catmull-Rom cubic. On the extruder
      it cuts the half-pixel travel-phase flicker from ~10% to ~3.5% and roughly
      halves the finest-scale structure-function deficit, while still matching
      the von Kármán structure function at few-pixel separations
      (`test_lanczos_reduces_finescale_flicker_vs_cubic`,
      `test_lanczos_matches_von_karman_structure_function`). Because it is
      readout-only (the extrusion stencil is unchanged) there is no long-range
      covariance trade-off. Also available on `InfinitePhaseScreen`. The
      **exact-Nyquist (1 px) mode is still lost** — no interpolator can shift a
      critically sampled signal there; the supersampled-buffer approach remains
      open as the only way to recover it, at higher cost.
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

- [x] **Tomography-optimal profile compression** — `discretize_cn2(
      method="optimal_grouping")` now chooses bin edges by an exact
      dynamic-programming partition that minimises the Cn²-weighted within-group
      variance of `h^{5/3}` (Saxenhuber 2017 "optimal grouping"), landing one
      output layer on each turbulence spike where fixed log-spaced binning lumps
      them together (`test_optimal_grouping_resolves_bimodal_profile`). GCTM is
      still open as a further `method=` option.
- [x] **Dry/wet chromatic split** — `dispersion="ciddor"` with a `wet_fraction`
      weight blends the dry-air (Edlén) and water-vapour (Ciddor 1996)
      dispersion ratios, so the chromatic OPD is right in the mid-IR / for
      interferometry where wet turbulence dominates; `wet_fraction=0` reduces to
      `"edlen"`. New `pyturb.water_vapour_refractivity`. Remaining: a
      temperature/humidity-driven default `wet_fraction` from a site model.
- [x] **More site profiles** — added `cerro-pachon` (Gemini South) and
      `armazones` (ELT), representative/illustrative in the manner of
      `paranal-median` (labelled as such; not a specific cited table).
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
- [x] **Complete type hints** on all public signatures (the `py.typed` marker
      already ships). All 85 public callables now carry full parameter and
      return annotations (audited by an `inspect`-based sweep over every
      module's `__all__`); the previously-unannotated private classes
      (`_ExtrudeLayer`, `_LayerState`, `_ThreadedScipyFFT`, `ExtrudedAtmosphere`)
      are annotated too. Remaining gaps are only local closures.
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

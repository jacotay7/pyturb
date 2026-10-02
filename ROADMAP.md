# pyturb Roadmap

The core is complete: `Atmosphere.from_profile(...).frames(dt)` delivers
multi-layer, sub-pixel, arbitrary-direction OPD on CPU or GPU, via a periodic
spectral engine or a non-periodic extruder, with boiling, off-axis/tomography,
LGS cone, chromatic OPD, named profiles, an analysis toolkit, FITS/npz I/O, a
benchmark suite, a validation gallery, docs, and CI. Version 1.0.0 was
released on 2026-07-09.

This file tracks only what is **not yet done**. Ground rule from here on
(unchanged): every physics feature lands with an ensemble-statistics test
against theory; every performance claim lands with a benchmark; every feature
lands with a docs page.

## Physics & profiles

- [ ] **Time-varying atmosphere parameters** — gusting/wind-direction drift
      over a run, time-variable Cn² profiles, and per-layer (rather than
      global) `inner_scale`/`power_law`. All are currently static-per-run;
      each is a genuine physics extension, not a bug, and would need its own
      theory-referenced validation the way `tau_boil` did.
- [ ] **ESO 35-layer reference profiles** (median + Q1–Q4, Sarazin et al.
      2013 / ESO-258292, superseded for the VLT by ESO-399284) — not public;
      add with a provenance test if a verbatim copy becomes available. Paranal
      is covered by the published Garcia-Rissmann et al. (2015) set
      (`paranal-p01`...`p14`).

## Performance

- [ ] **Graph-capture the remaining host-bound GPU paths**: spectral boiling
      (its random draws), the LGS cone readout and the extruder still issue
      per-frame kernels and run at a few hundred frames/s on modest hosts.

## Non-goals — deliberately out of scope

Kept here so the scope stays crisp; pyturb is *the atmosphere*, not an AO system.

- **Full AO system simulation** (WFS, DM, controllers) — never; that is soapy's
  job. pyturb is the turbulence that plugs into it.
- **Tomographic reconstructors / slope covariance** — aotools' domain.
- **Scintillation / Fresnel propagation between layers** — HCIPy/PROPER own
  amplitude effects; pyturb outputs phase/OPD. *Possible stretch only if pulled
  by demand:* a minimal angular-spectrum propagator over pyturb's layers.

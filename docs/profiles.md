# Turbulence profiles

A profile is a list of `pyturb.Layer`s — altitude, fraction of the integrated
Cn², wind speed and direction, outer scale. Build an `Atmosphere` from a named
profile, from your own layers, or straight from a measured Cn²(h).

## Named profiles

r0 = 0.15 m at zenith for the θ0/τ0 columns (both scale with r0):

| name | layers | site | traceable to a published table | L0 [m] | θ0 | τ0 |
|---|---:|---|:---:|---:|---:|---:|
| `armazones` | 8 | Cerro Armazones (ELT, ~3060 m) | no | 25 | 1.73" | 3.1 ms |
| `cerro-pachon` | 8 | Cerro Pachon (~2715 m) | no | 25 | 1.70" | 3.0 ms |
| `hv57` | 10 | Hufnagel–Valley 5/7 model | yes (analytic) | 25 | 4.11" | 5.5 ms |
| `keck` | 7 | Mauna Kea (Keck, KAON 303) | yes | 20 | 1.87" | 2.6 ms |
| `las-campanas` | 7 | Las Campanas (Males et al. 2018) | yes | 25 | 1.77" | 2.5 ms |
| `mauna-kea` | 6 | Mauna Kea (Guyon 2005) | yes | 10 | 1.37" | 4.7 ms |
| `paranal-median` | 9 | Paranal (VLT) | no | 25 | 1.54" | 3.0 ms |
| `single-layer` | 1 | teaching case | — | 25 | ∞ | 4.7 ms |
| `two-layer` | 2 | teaching case | — | 25 | 2.00" | 2.8 ms |

### Paranal reference profiles (Garcia-Rissmann et al. 2015)

Fourteen typical Paranal profiles from [Garcia-Rissmann et al. 2015, MNRAS
448, 2594 (doi:10.1093/mnras/stv169)](https://doi.org/10.1093/mnras/stv169),
Tables 2–3, compiled by ESO from SLODAR, MASS-DIMM and SCIDAR measurements
(Sarazin et al. 2013). Each sits in a seeing class and is "good", "median" or
"bad" according to how much of its turbulence is near the ground (good = more
ground layer, easier to correct). Ten layers from 30 m to 14 km, whole-percent
fractions, layer speeds `beta * v_ref`, L0 = 25 m; wind directions are not in
the source and are all 0°. The published r0, τ0 (quoted for z = 30°),
mean height h̄ and probability of occurrence are in
`profile_info(name).conditions`; pyturb's tests recompute the paper's h̄ and τ0
from the stored layers.

| name | seeing class | quality | r0 [m] | τ0 [ms] | h̄ [km] | θ0 (at the published r0) | probability |
|---|---:|---|---:|---:|---:|---:|---:|
| `paranal-p01` | 0.4" | median | 0.186 | 4.6 | 3.73 | 3.23" | 7.0% |
| `paranal-p02` | 0.6" | good | 0.136 | 3.9 | 2.39 | 3.68" | 6.0% |
| `paranal-p03` | 0.6" | median | 0.136 | 3.8 | 3.73 | 2.36" | 12.0% |
| `paranal-p04` | 0.6" | bad | 0.136 | 3.9 | 4.88 | 1.80" | 6.0% |
| `paranal-p05` | 0.8" | good | 0.116 | 3.0 | 2.66 | 2.82" | 6.5% |
| `paranal-p06` | 0.8" | median | 0.116 | 3.0 | 3.82 | 1.97" | 13.0% |
| `paranal-p07` | 0.8" | bad | 0.116 | 3.1 | 4.45 | 1.69" | 6.5% |
| `paranal-p08` | 1.0" | good | 0.101 | 2.4 | 3.05 | 2.15" | 4.5% |
| `paranal-p09` | 1.0" | median | 0.101 | 2.5 | 3.81 | 1.72" | 9.0% |
| `paranal-p10` | 1.0" | bad | 0.101 | 2.4 | 4.47 | 1.46" | 4.5% |
| `paranal-p11` | 1.2" | good | 0.089 | 2.0 | 2.73 | 2.11" | 3.0% |
| `paranal-p12` | 1.2" | median | 0.089 | 2.1 | 3.53 | 1.63" | 6.0% |
| `paranal-p13` | 1.2" | bad | 0.089 | 2.0 | 3.69 | 1.56" | 3.0% |
| `paranal-p14` | 1.4" | median | 0.074 | 1.4 | 3.53 | 1.36" | 13.0% |

Reproduce a published case with its r0 (Table 2's τ0 equals `0.314 r0 / v̄`
with these winds and the tabulated r0, i.e. at the line of sight the paper
quotes); for a Monte-Carlo over Paranal conditions, draw profiles with the
listed probabilities (they sum to 100%):

```python
import numpy as np
import pyturb

info = pyturb.profile_info("paranal-p06")
atm = pyturb.Atmosphere.from_profile("paranal-p06", r0=info.conditions["r0"], n=128)
print(atm.tau0, info.conditions["tau0"])         # ~3.05 ms vs 3.0 ms published

names = [f"paranal-p{i:02d}" for i in range(1, 15)]
weights = [pyturb.profile_info(n).conditions["probability"] for n in names]
pick = np.random.default_rng(0).choice(names, p=weights)
```

### More traceable site profiles

Profiles from refereed papers. Each records how its numbers were obtained in
`profile_info(name).origin`:

- **table**: transcribed verbatim from a printed table and checked against
  the rendered page;
- **dataset**: computed by pyturb from the authors' published data file;
- **figure**: digitised from a plotted curve (vector paths read from the PDF
  where the figure is vector, otherwise traced from the raster), with an
  overlay check against the figure.

pyturb's tests recompute every published seeing, θ0 or τ0 the source gives.
r0 is the profile's own (from its tabulated Cn²·dh, or from the published
seeing where only percentages are given or where the profile is a per-altitude
median, see below) and is used when you pass neither `r0` nor `seeing`. L0 is
pyturb's 25 m where the source gives none.

| name | layers | r0 [m] | seeing | θ0 | winds | origin | source |
|---|---:|---:|---:|---:|---|---|---|
| `tmt-tolar-good` | 7 | 0.230 | 0.44" | 2.18" | none (0 m/s) | table | [Els+ 2009](https://doi.org/10.1086/599384) T4 |
| `tmt-tolar-typical` | 7 | 0.188 | 0.54" | 1.97" | none (0 m/s) | table | [Els+ 2009](https://doi.org/10.1086/599384) T4 |
| `tmt-tolar-bad` | 7 | 0.152 | 0.66" | 1.79" | none (0 m/s) | table | [Els+ 2009](https://doi.org/10.1086/599384) T4 |
| `tmt-armazones-good` | 7 | 0.229 | 0.44" | 2.36" | none (0 m/s) | table | [Els+ 2009](https://doi.org/10.1086/599384) T4 |
| `tmt-armazones-typical` | 7 | 0.185 | 0.55" | 2.09" | none (0 m/s) | table | [Els+ 2009](https://doi.org/10.1086/599384) T4 |
| `tmt-armazones-bad` | 7 | 0.144 | 0.70" | 1.88" | none (0 m/s) | table | [Els+ 2009](https://doi.org/10.1086/599384) T4 |
| `tmt-tolonchar-good` | 7 | 0.218 | 0.46" | 2.23" | none (0 m/s) | table | [Els+ 2009](https://doi.org/10.1086/599384) T4 |
| `tmt-tolonchar-typical` | 7 | 0.182 | 0.55" | 1.89" | none (0 m/s) | table | [Els+ 2009](https://doi.org/10.1086/599384) T4 |
| `tmt-tolonchar-bad` | 7 | 0.146 | 0.69" | 1.52" | none (0 m/s) | table | [Els+ 2009](https://doi.org/10.1086/599384) T4 |
| `tmt-san-pedro-martir-good` | 7 | 0.184 | 0.55" | 2.52" | none (0 m/s) | table | [Els+ 2009](https://doi.org/10.1086/599384) T4 |
| `tmt-san-pedro-martir-typical` | 7 | 0.144 | 0.70" | 2.17" | none (0 m/s) | table | [Els+ 2009](https://doi.org/10.1086/599384) T4 |
| `tmt-san-pedro-martir-bad` | 7 | 0.102 | 0.99" | 1.77" | none (0 m/s) | table | [Els+ 2009](https://doi.org/10.1086/599384) T4 |
| `tmt-maunakea-13n-good` | 7 | 0.198 | 0.51" | 3.26" | none (0 m/s) | table | [Els+ 2009](https://doi.org/10.1086/599384) T4 |
| `tmt-maunakea-13n-typical` | 7 | 0.153 | 0.66" | 2.98" | none (0 m/s) | table | [Els+ 2009](https://doi.org/10.1086/599384) T4 |
| `tmt-maunakea-13n-bad` | 7 | 0.112 | 0.90" | 2.66" | none (0 m/s) | table | [Els+ 2009](https://doi.org/10.1086/599384) T4 |
| `cerro-pachon-good` | 7 | 0.164 | 0.62" | 2.57" | none (0 m/s) | table | [Tokovinin & Travouillon 2006](https://doi.org/10.1111/j.1365-2966.2005.09813.x) T3 |
| `cerro-pachon-typical` | 7 | 0.135 | 0.75" | 2.22" | none (0 m/s) | table | [Tokovinin & Travouillon 2006](https://doi.org/10.1111/j.1365-2966.2005.09813.x) T3 |
| `cerro-pachon-bad` | 7 | 0.111 | 0.91" | 2.04" | none (0 m/s) | table | [Tokovinin & Travouillon 2006](https://doi.org/10.1111/j.1365-2966.2005.09813.x) T3 |
| `siding-spring-gl-good-fa-good` | 7 | 0.117 | 0.86" | 6.43" | modelled (Bufton) + directions | table | [Goodwin+ 2013](https://doi.org/10.1017/pasa.2012.009) T11–13 |
| `siding-spring-gl-good-fa-typical` | 7 | 0.107 | 0.94" | 3.72" | modelled (Bufton) + directions | table | [Goodwin+ 2013](https://doi.org/10.1017/pasa.2012.009) T11–13 |
| `siding-spring-gl-good-fa-bad` | 7 | 0.094 | 1.07" | 2.01" | modelled (Bufton) + directions | table | [Goodwin+ 2013](https://doi.org/10.1017/pasa.2012.009) T11–13 |
| `siding-spring-gl-typical-fa-good` | 7 | 0.086 | 1.18" | 6.37" | modelled (Bufton) + directions | table | [Goodwin+ 2013](https://doi.org/10.1017/pasa.2012.009) T11–13 |
| `siding-spring-gl-typical-fa-typical` | 7 | 0.082 | 1.24" | 3.71" | modelled (Bufton) + directions | table | [Goodwin+ 2013](https://doi.org/10.1017/pasa.2012.009) T11–13 |
| `siding-spring-gl-typical-fa-bad` | 7 | 0.075 | 1.35" | 2.01" | modelled (Bufton) + directions | table | [Goodwin+ 2013](https://doi.org/10.1017/pasa.2012.009) T11–13 |
| `siding-spring-gl-bad-fa-good` | 7 | 0.067 | 1.52" | 6.23" | modelled (Bufton) + directions | table | [Goodwin+ 2013](https://doi.org/10.1017/pasa.2012.009) T11–13 |
| `siding-spring-gl-bad-fa-typical` | 7 | 0.064 | 1.57" | 3.67" | modelled (Bufton) + directions | table | [Goodwin+ 2013](https://doi.org/10.1017/pasa.2012.009) T11–13 |
| `siding-spring-gl-bad-fa-bad` | 7 | 0.061 | 1.67" | 2.00" | modelled (Bufton) + directions | table | [Goodwin+ 2013](https://doi.org/10.1017/pasa.2012.009) T11–13 |
| `sutherland-median` | 7 | 0.072 | 1.40" | 1.96" | none (0 m/s) | table | [Catala+ 2013](https://doi.org/10.1093/mnras/stt1602) T3 |
| `mt-graham-good` | 20 | 0.217 | 0.47" | 3.03" | climatology (speeds only) | table; winds figure | [Masciadri+ 2010](https://doi.org/10.1111/j.1365-2966.2010.16313.x) T6+8; winds [Hagelin+ 2010](https://doi.org/10.1111/j.1365-2966.2010.17102.x) F3+7 |
| `mt-graham-typical` | 20 | 0.145 | 0.70" | 2.27" | climatology (speeds only) | table; winds figure | [Masciadri+ 2010](https://doi.org/10.1111/j.1365-2966.2010.16313.x) T6+8; winds [Hagelin+ 2010](https://doi.org/10.1111/j.1365-2966.2010.17102.x) F3+7 |
| `mt-graham-bad` | 20 | 0.096 | 1.05" | 1.57" | climatology (speeds only) | table; winds figure | [Masciadri+ 2010](https://doi.org/10.1111/j.1365-2966.2010.16313.x) T6+8; winds [Hagelin+ 2010](https://doi.org/10.1111/j.1365-2966.2010.17102.x) F3+7 |
| `maunakea-raven-mean` | 5 | 0.219 | 0.46" | 2.89" | none (0 m/s) | table | [Ono+ 2017](https://doi.org/10.1093/mnras/stw3083) T1 |
| `maunakea-cfht-mean` | 5 | 0.191 | 0.53" | 2.64" | none (0 m/s) | table | [Ono+ 2017](https://doi.org/10.1093/mnras/stw3083) T1 |
| `cerro-tololo-good` | 7 | 0.128 | 0.79" | 1.91" | none (0 m/s) | dataset | [Tokovinin+ 2003](https://doi.org/10.1046/j.1365-8711.2003.06231.x) data |
| `cerro-tololo-typical` | 7 | 0.106 | 0.95" | 1.75" | none (0 m/s) | dataset | [Tokovinin+ 2003](https://doi.org/10.1046/j.1365-8711.2003.06231.x) data |
| `cerro-tololo-bad` | 7 | 0.086 | 1.17" | 1.51" | none (0 m/s) | dataset | [Tokovinin+ 2003](https://doi.org/10.1046/j.1365-8711.2003.06231.x) data |
| `paranal-stereo-scidar-mean` | 24 | 0.158 | 0.64" | 1.81" | none (0 m/s) | figure | [Osborn+ 2018](https://doi.org/10.1093/mnras/sty1070) F2 |
| `la-palma-median` | 8 | 0.120 | 0.84" | 2.84" | none (0 m/s) | figure | [García-Lorenzo & Fuensalida 2011](https://doi.org/10.1111/j.1365-2966.2011.19186.x) F3b |
| `san-pedro-martir-median` | 37 | 0.128 | 0.79" | 1.91" | none (0 m/s) | figure | [Avila+ 2019](https://doi.org/10.1093/mnras/stz2672) F9 |

Notes on using them:

- **Winds.** The Siding Spring winds are a Bufton model with modelled
  directions, not measurements; the Mt Graham speeds are a climatology (see
  below) with all directions 0°. The others publish
  no winds: their layers are static (0 m/s) and `from_profile` warns. Supply
  winds for frozen flow, e.g. `wind="bufton"`, a scalar, or measured speeds
  (`pyturb.with_wind` does the same for a layer list).
- **TMT** (Els et al. 2009, Table 4): good/typical/bad are the median
  profiles around the 25/50/75 per cent DIMM seeing at five TMT candidate
  sites; the 0 km layer lumps the telescope to ~0.5 km. Site elevations are in
  `conditions["site_altitude_m"]`.
- **Cerro Pachón** (Tokovinin & Travouillon 2006): a model synthesised from
  MASS, balloon and SODAR data, ranked by free-atmosphere seeing; the authors
  suggest combining ground-layer and free-atmosphere conditions independently.
- **Siding Spring** (Goodwin et al. 2013): nine combinations of ground-layer
  (GL) and free-atmosphere (FA) conditions with their probabilities; the
  authors consider the FA strength conservative (~2x high).
- **Sutherland** (Catala et al. 2013): median MASS-DIMM profile; the 200 m
  layer holds everything below the 500 m MASS layer.
- **Mt Graham** (Masciadri et al. 2010): G-SCIDAR slab integrals placed at
  slab midpoints, dome seeing removed; the "good" dome-free row is not
  consistent with the paper's quoted dome seeing. The wind speeds come from a
  separate paper (Hagelin et al. 2010, digitised from Figs. 3 and 7): the
  SCIDAR mean below 1 km and ECMWF monthly medians, weighted to the campaign's
  months, above. They are not simultaneous with the Cn², and the jet they put
  at ~9 km makes τ0 about 25% shorter than the paper's median (3.6 vs 4.8 ms
  for `typical`). Same speeds for all three classes; override with `wind=`.
- **Maunakea, RAVEN and CFHT** (Ono et al. 2017, Table 1): mean profiles in
  five coarse bins (0–1.5, 1.5–3, 3–6, 6–12, >12 km) placed at the table's
  bin labels. `maunakea-raven-mean` carries the published per-bin median L0
  (13–34 m, biased toward ~2–3x the 8.2 m aperture).
- **Cerro Tololo** (Tokovinin et al. 2003): derived by pyturb from the
  authors' public file of 22,300 MASS-DIMM profiles, the same way Els et al.
  built the TMT rows: per-layer medians of the profiles within ±5% of the
  25/50/75% total-seeing percentiles. The total seeing reproduces the paper's
  Table 1 (0.79/0.95/1.17").

**Per-altitude medians.** The three figure profiles plot the median (or, for
Paranal, the mean) Cn² at each height. A per-altitude median is not the
profile of a median night: its integral falls well short of the median seeing
(by ~45% at San Pedro Mártir) and it under-weights the free atmosphere. pyturb
keeps the shape and sets the strength to the paper's median seeing
(`conditions["strength_from"] == "seeing"`). The resulting θ0 is close to
published for Paranal (1.81" vs 1.75") and San Pedro Mártir (1.91" vs 1.96",
from Avila et al. 2011), but 2.84" vs 2.22" at La Palma, where the ground
layer dominates the median shape. Digitisation itself is accurate to ~1% in
seeing.

```python
import pyturb

atm = pyturb.Atmosphere.from_profile("tmt-armazones-typical", wind="bufton", n=128)
print(atm.r0, atm.seeing, pyturb.profile_info("tmt-armazones-typical").source)
```

Sources that were checked and not added: Osborn et al. 2017 (Stereo-SCIDAR
winds; no per-layer table to pair with a profile), Catala et al. 2017
(Sutherland PML; integrated statistics only), the 2020/2021 Ali (Tibet)
studies (model output), the San Pedro Mártir quartile curves (per-altitude
quartiles would need a 9x rescale to match the quoted seeing), and ESO's
35-layer model (ESO-258292 / ESO-399284 are not public).

`paranal-median` (below) predates these and is a representative profile, not
a published table; prefer the `paranal-pNN` set for Paranal work.

"No" means a representative discretisation in the general shape of the site's
published statistics, not a reproduction of one table. **Wind directions are
illustrative in every profile** — none of the sources tabulate them.
`pyturb.profile_info(name)` returns this provenance, and `Atmosphere.metadata`
records it with every saved OPD.

```python
import pyturb

print(pyturb.profile_info("keck").source)
atm = pyturb.Atmosphere.from_profile("keck", seeing=0.6, zenith_angle=20, n=128)
```

## Your own layers

```python
layers = [
    pyturb.Layer(altitude=0, cn2_fraction=0.6, wind_speed=8, wind_direction=45),
    pyturb.Layer(altitude=12e3, cn2_fraction=0.4, wind_speed=30, wind_direction=270,
                 L0=30),
]
atm = pyturb.Atmosphere(layers, seeing=0.7, n=128)
```

Fractions are normalised for you. `wind_direction` is where the wind blows
*from* (see [Conventions](concepts.md#conventions)).

## From a measured or model Cn²(h)

`Atmosphere.from_cn2` compresses a profile on a fine height grid (MASS-DIMM,
SCIDAR, a model) into layers and builds the atmosphere; without `r0`/`seeing`
it uses the profile's own integrated r0:

```python
import numpy as np

h = np.geomspace(10, 25e3, 4000)                 # m above the telescope, zenith
cn2 = pyturb.hufnagel_valley(h)                  # or your measurement [m^-2/3]
atm = pyturb.Atmosphere.from_cn2(h, cn2, n_layers=12, n=128, source="HV 5/7")
print(atm.r0, atm.theta0)                        # ~0.05 m, ~1.4" (7 urad)
```

The compression (`pyturb.discretize_cn2`) always conserves the integrated Cn²;
`method` decides where the layers go:

- `"equivalent"` (default): log-spaced bins, each layer at the
  `h^{5/3}`-moment height and `v^{5/3}`-moment speed, so θ0 and τ0 are
  conserved.
- `"centroid"`: log-spaced bins at the Cn²-weighted mean height (simpler; θ0
  and τ0 not conserved exactly).
- `"optimal_grouping"`: bin edges chosen by dynamic programming to minimise
  the Cn²-weighted spread of `h^{5/3}` within each group (Saxenhuber et al.
  2017) — the choice when *where* the layers sit matters, as for MCAO or
  tomographic reconstruction layers.

Wind directions can be a scalar, one value per input height (each layer gets
the Cn²-weighted circular mean) or one per output layer:

```python
directions = np.where(h < 2e3, 30.0, 250.0)       # ground vs free atmosphere
layers = pyturb.discretize_cn2(h, cn2, n_layers=6, method="optimal_grouping",
                               wind_direction=directions)
```

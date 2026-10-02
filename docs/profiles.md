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

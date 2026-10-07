# Migrating to pyturb 2.0

pyturb 2.0 adopts the axis convention of the rest of its AO family —
[aobasis](https://github.com/jacotay7/aobasis), makewfs, solvephase,
shmpipeline-ao and [HCIPy](https://github.com/ehpor/hcipy): arrays are indexed
`(y, x)`, with **x along axis 1 (columns)** and **y along axis 0 (rows)**.
pyturb 1.x labelled axis 0 as x. Array shapes, pixel centres
(`(i - (n-1)/2) * pitch`), units and the turbulence statistics are unchanged;
what changed is which array axis every x/y-labelled input refers to.

## What changed

| Input | pyturb 1.x | pyturb 2.0 |
|---|---|---|
| `Layer.wind_direction` [deg, wind blows **from**] | from axis 0 toward axis 1 | from +x (axis 1) toward +y (axis 0) |
| `Layer.wind_vector` | `(v_axis0, v_axis1)` | `(vx, vy)` = `(v_axis1, v_axis0)` |
| `opd(directions=[(thx, thy)])`, `(thx, thy, altitude)` | `thx` along axis 0 | `thx` along x (axis 1), `thy` along y (axis 0) |
| `opd_at(x, y, direction=...)` | `x` along axis 0 | `x` along axis 1, `y` along axis 0 |
| `FourierFlowScreen.translate(sx, sy)` | `sx` along axis 0 | `sx` along x (axis 1), `sy` along y (axis 0) |
| `analysis.zernike_basis` / `zernike_decompose` | x = axis 0 (Z2 varies down the rows) | x = axis 1 (Z2 varies along the columns), as `aobasis` |
| `theory.zernike_temporal_psd(..., wind_direction)` | relative to the 1.x Zernike frame | relative to the 2.0 frame (numerically unchanged) |
| `to_config()` | `pyturb_config_version: 1` | `pyturb_config_version: 2` (v1 still loads, converted) |

The meaning of `wind_direction` is otherwise the same: it is the direction the
wind blows *from*, the pattern moves along `-wind_vector`, and with
`wind_direction=0` the pattern now travels toward decreasing **column** index.
`InfinitePhaseScreen` is unchanged: it extrudes along axis 0, which is now
called y.

## Reproducing 1.x results exactly

Each 1.x call maps to a 2.0 call that produces the same arrays (verified to
float64 round-off on both engines, with directions, LGS sources and `opd_at`):

| pyturb 1.x | pyturb 2.0, same arrays |
|---|---|
| `Layer(..., wind_direction=a)` | `Layer(..., wind_direction=90 - a)` |
| `with_wind(..., direction=a)`, `discretize_cn2(..., wind_direction=a)`, `from_profile(..., wind_direction=a)`, `from_cn2(..., wind_direction=a)` | the same with `90 - a` |
| `v0, v1 = layer.wind_vector` (components along axes 0, 1) | `v1, v0 = layer.wind_vector` |
| `opd(t, directions=[(a, b)])` | `opd(t, directions=[(b, a)])` |
| `opd(t, directions=[(a, b, h)])` | `opd(t, directions=[(b, a, h)])` |
| `opd_at(p, q, t, direction=(a, b))` | `opd_at(q, p, t, direction=(b, a))` |
| `flow.translate(a, b)` | `flow.translate(b, a)` |
| `zernike_basis(J, n)` | `zernike_basis(J, n).transpose(0, 2, 1)` |
| `zernike_decompose(phase, J)` | `zernike_decompose(phase.swapaxes(-1, -2), J)` |
| `Atmosphere.from_config(cfg_v1)` | unchanged: version-1 configs are converted on load |

Before:

```python
import pyturb  # 1.x

layers = [pyturb.Layer(0.0, 0.7, wind_speed=8.0, wind_direction=0.0),
          pyturb.Layer(1e4, 0.3, wind_speed=30.0, wind_direction=30.0)]
atm = pyturb.Atmosphere(layers, r0=0.15, n=128, field_of_view=20, seed=1)
opds = atm.opd(0.01, directions=[(10.0, 0.0), (0.0, 5.0)])
```

After, the same arrays:

```python
import pyturb  # 2.0

layers = [pyturb.Layer(0.0, 0.7, wind_speed=8.0, wind_direction=90.0),
          pyturb.Layer(1e4, 0.3, wind_speed=30.0, wind_direction=60.0)]
atm = pyturb.Atmosphere(layers, r0=0.15, n=128, field_of_view=20, seed=1)
opds = atm.opd(0.01, directions=[(0.0, 10.0), (5.0, 0.0)])
```

For new code, prefer stating the physics in the new frame directly (e.g.
`wind_direction=0` for wind along the columns) over mechanically applying the
mapping.

## Zernike modes

The 2.0 basis is the 1.x basis reflected about the array diagonal: as arrays,
`new[k] == old[k].T` for every mode, and the 2.0 coefficients of a phase screen
equal the 1.x coefficients of its transpose. Mode by mode the reflection
(`theta -> 90° - theta`) is a signed permutation of the 1.x modes with the same
radial order `n` and azimuthal order `|m|`:

- **`m = 0`** (piston, defocus, spherical, ...): unchanged.
- **odd `|m|`**: the cosine and sine modes swap, both with sign
  `(-1)^((|m|-1)/2)`.
- **even `|m| > 0`**: no swap; the cosine mode gets sign `(-1)^(|m|/2)` and the
  sine mode `(-1)^(|m|/2 + 1)`.

In Noll order (old 1.x mode on the left, as an array on the same grid):

| old | new | old | new | old | new |
|---|---|---|---|---|---|
| Z2 (tip) | +Z3 | Z9 | −Z10 | Z16 | +Z17 |
| Z3 (tilt) | +Z2 | Z10 | −Z9 | Z17 | +Z16 |
| Z4 | +Z4 | Z11 | +Z11 | Z18 | −Z19 |
| Z5 | +Z5 | Z12 | −Z12 | Z19 | −Z18 |
| Z6 | −Z6 | Z13 | +Z13 | Z20 | +Z21 |
| Z7 | +Z8 | Z14 | +Z14 | Z21 | +Z20 |
| Z8 | +Z7 | Z15 | −Z15 | | |

So a 1.x coefficient vector `c_old` becomes `c_new[k] = sign * c_old[j]` for
each `old Zj = sign * new Zk` row (the relation is exact; checked to 1e-14 over
the first 66 modes). In 2.0, Noll Z2 (`cos theta`) varies along the columns
and matches `aobasis.ZernikeBasisGenerator` on `aobasis.positions_from_mask`
positions to machine precision.

`theory.zernike_temporal_psd` returns the same numbers as before: it relates a
wind direction to the Zernike frame, and both moved together. Pass it the
same `wind_direction` you give the `Layer`; with `wind_direction=0`, Z2 is the
along-wind tilt.

## Site profiles

The named profiles keep their published `wind_direction` numbers; they are now
read in the 2.0 frame. Profiles carry no sky orientation (most directions are
illustrative placeholders), so this changes which array axis a layer blows
along, not any published quantity (`r0`, `theta0`, `tau0` and the wind speeds
are unchanged). To replay a 1.x run of a named profile exactly, convert its
directions:

```python
import pyturb

layers = pyturb.get_profile("keck")
legacy = pyturb.with_wind(layers, [layer.wind_speed for layer in layers],
                          [90.0 - layer.wind_direction for layer in layers])
atm = pyturb.Atmosphere(legacy, seeing=0.8, n=128, seed=1)  # == 1.x "keck"
```

## Saved configs and metadata

`Atmosphere.to_config()` (and the `"config"` entry of `atm.metadata` saved by
`pyturb.save`) now writes `pyturb_config_version: 2`. `from_config` still reads
version-1 configs written by pyturb 1.x, converting each layer's
`wind_direction` to `(90 - a) mod 360`, so an old saved run replays the frames
it recorded.

## HCIPy

pyturb arrays already flattened onto HCIPy grids with `.ravel()`; with 2.0, x
and y also agree, so direction tuples carry over without the axis swap 1.x
needed. See [Interop](interop.md#hcipy) for the velocity sign.

## Internal engine

`pyturb.extrude.ExtrudedAtmosphere` (used by `engine="extrude"`, not part of
the top-level API) still works in array-axis order: `layer_wind` entries are
`(v_axis0, v_axis1)`. Its direction arguments are renamed `slope0`/`slope1`
(tangents along axes 0 and 1) and `sample_points` takes `pix0`/`pix1`.

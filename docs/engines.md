# Choosing an engine

`Atmosphere` has two ways to evolve turbulence in time, and two knobs that
decide how faithful the FFT screens are. This page is the decision guide;
the [API reference](api.md) has every parameter.

## At a glance

| | `engine="spectral"` (default) | `engine="extrude"` |
|---|---|---|
| How it moves | Exact sub-pixel shift theorem on stored Fourier modes | Assémat–Wilson row extrusion into a ring buffer, read out on a rotated sub-pixel grid |
| Repeats? | **Yes**, after `n_screen * pixel_scale` metres of travel per layer (`atm.time_to_wrap`) | **No**, unbounded |
| Random access in time | Yes: `opd(t)` for any `t`, `opd(t=array)` batched | Streaming: `t` only moves forward (`reset()` restarts) |
| Fine-scale fidelity | Exact at any offset | Interpolation low-passes the 1–2 px scales by 5–15% (`interp="lanczos"` roughly halves that) |
| Large-scale fidelity | Pupil-sized screens underestimate separations beyond ~D/4 — use `oversample` | Good out to the outer scale |
| Boiling (`tau_boil`) | Per Fourier mode (fine structure boils faster) | One timescale for all scales; markedly slower |
| Non-Kolmogorov `power_law`, `inner_scale`, `L0=inf` | Yes | No (needs the von Kármán covariance) |
| LGS cone, off-axis directions, `opd_at` | Yes | Yes |
| GPU speed, 9 layers, 512² (RTX 4060) | ~4,300 fps (CUDA graph) | ~490 fps |

`sample()` (independent Monte-Carlo draws) does not depend on the engine.

## Rules of thumb

- **Short closed-loop runs, PSD studies, anything needing exact fine scales:**
  spectral, keeping the run under `atm.time_to_wrap` (a `PeriodicWrapWarning`
  tells you when a run crosses it).
- **Long runs** (seconds of wind or more, long-exposure PSFs, telemetry-length
  sequences): extrude. The repeat in the spectral engine biases every temporal
  statistic once it wraps.
- **Tip/tilt, low-order modes, anything across the whole pupil:** set
  `oversample=2`–`4` on the spectral engine and for `sample()`. A pupil-sized
  FFT screen is periodic across the pupil, so opposite edges are neighbours:
  the structure function is ~13% low at D/2 and ~0.65x theory near 0.9 D, tilt
  ~10-15% low, and the two astigmatisms split. `oversample=4` matches von
  Kármán to ~2% out to 0.9 D (see [Validation](validation.md)), and it
  multiplies `time_to_wrap` too.
- **Off-axis directions:** set `field_of_view` to the largest radius you will
  ask for; it oversizes the screens so footprints don't wrap or clamp.

```python
import pyturb

# Long, non-periodic closed loop:
loop = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, n=256,
                                      engine="extrude", seed=1)

# Statistically faithful low-order modes with the fast engine:
stats = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, n=256,
                                       oversample=4, seed=1)
print(stats.time_to_wrap, "s before the fastest layer repeats")
```

## Cost of the knobs

`oversample` multiplies the FFT side, so a spectral frame or a `sample()` draw
costs roughly `oversample**2` more (RTX 4060, 9 layers, 512²: ~5,300 →
~1,360 → ~300 spectral fps at 1x/2x/4x; at 256², 2x is nearly free). `field_of_view` adds `2 * h_max * tan(fov) / pixel_scale`
pixels to each side. On the extruder, `interp="lanczos"` costs 36 taps per
pixel per layer instead of 16; `"linear"` is not faster (it bypasses the fused
kernel).

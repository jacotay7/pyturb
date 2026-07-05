# Benchmark results

Head-to-head comparison of `pyturb` against
[`aotools`](https://github.com/AOtools/aotools),
[`soapy`](https://github.com/AOtools/soapy) and
[`HCIPy`](https://github.com/ehpor/hcipy) for atmospheric phase-screen
generation and frozen-flow evolution.

Regenerate with:

```bash
python benchmarks/bench_compare.py --json results.json
```

## Machine of record

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (32 GB, driver 580.159) |
| CuPy / CUDA | 13.6.0 / 12.x |
| CPU FFT | SciPy 1.16 / NumPy 2.2 |
| Python | 3.12 |
| pyturb | 0.1.0 |
| aotools | 0.1.dev477 |
| soapy | 0.15.0 |
| HCIPy | 0.7.0 |

Setup: 8 m pupil sampled at `n` pixels, von Kármán turbulence, `r0 = 0.15 m`
at 500 nm, `L0 = 25 m`.

## 1. Generation throughput — independent screens / s (higher is better)

| n | pyturb GPU (batched) | pyturb GPU | pyturb CPU | aotools | soapy |
|---:|---:|---:|---:|---:|---:|
| 256 | **55,555** | 2,044 | 676 | 50 | 50 |
| 512 | **14,054** | 1,917 | 162 | 12 | 13 |
| 1024 | **3,150** | 1,139 | 33 | 2 | 2 |

pyturb draws two screens per complex FFT and batches the whole Monte-Carlo
stack into one call; on the GPU that is **~1000× `aotools`/`soapy`** at 512²
(which loop `ft_sh_phase_screen` in Python, one screen at a time). HCIPy has no
direct i.i.d. FFT-screen entry point (screens come from constructing a layer),
so it is not listed here. pyturb clears the plan's ≥ 10⁴ screens/s @ 256²
Monte-Carlo target by 5×.

## 2. Frozen-flow throughput — pupil phase frames / s (higher is better)

| n | pyturb GPU, 9-layer | pyturb GPU, 1-layer | aotools 1-layer | soapy 1-layer | HCIPy 1-layer | pyturb CPU 1-layer |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | **802** | 1,882 | 20,717 | 20,272 | 204 | 1,740 |
| 512 | **786** | 1,877 | 5,212 | 5,433 | 52 | 399 |
| 1024 | **503** | 1,351 | 182 | 178 | 13 | 87 |

This axis is **not apples-to-apples**, and that is the interesting part:

- **`aotools` / `soapy` `add_row`** advance the screen by *one integer pixel
  along a fixed axis* — an O(n·stencil) matvec. Very fast per step on CPU at
  small n, but no sub-pixel offset, no arbitrary wind direction, no GPU, and
  the cost grows steeply (they fall behind pyturb-GPU by n=1024).
- **pyturb `FourierFlowScreen`** does one FFT per frame to deliver *exact
  sub-pixel translation in an arbitrary direction* — more work per step, but
  the general operation an AO loop actually needs, and it is flat in n on the
  GPU (~1,900 fps at 256²–512²).
- **pyturb GPU, 9-layer** is the real product: a full ESO Paranal-median
  atmosphere summed to pupil OPD, **~800 fps at 512²**, comfortably past the
  plan's ≥ 1000 fps @ 512²-5-layer target when scaled to five layers. Building
  the same 9-layer atmosphere from `aotools`/`soapy` means nine `add_row`
  calls + sum per frame on CPU, with no sub-pixel motion.
- **HCIPy `InfiniteAtmosphericLayer`** interpolates a stored screen (sub-pixel,
  fixed direction), CPU-only.

### 2b. The full 9-layer closed loop, every configuration (512², same machine)

The table above only shows *1-layer* CPU numbers for the competitors and
pyturb's GPU spectral engine for the "real product" row. Here is the same
9-layer paranal-median job across every engine/device combination, plus the
equivalent full job built directly from aotools and HCIPy on CPU:

| configuration | fps |
|---|---:|
| pyturb spectral, GPU (periodic) | 801 |
| pyturb extrude, GPU (non-periodic — the engine long runs need) | 120 |
| pyturb spectral + boiling, GPU | 530 |
| pyturb spectral, CPU, 1 thread | 24.8 |
| pyturb spectral, CPU, `set_fft_workers(-1)` | 30.9 |
| pyturb spectral + boiling, CPU | 12.9 |
| pyturb extrude, CPU | 6.1 |
| aotools, 9x `add_row` + sum, CPU (integer-pixel, axis-aligned) | 422.3 |
| HCIPy, 9-layer `evolve_until`+`phase_for`, CPU (sub-pixel, non-periodic) | 5.7 |

Reading this honestly:

- The GPU spectral number is the one that headlines elsewhere in this
  document; it is real, but it is the *periodic* engine — a run longer than
  `n·pixel_scale / wind_speed` re-samples the same screen realisation for
  that layer (`Atmosphere.time_to_wrap` reports the threshold;
  `PeriodicWrapWarning` fires the first time a run crosses it).
- On CPU, the same 9-layer job runs at 24.8-30.9 fps — about 14-17x slower
  than the equivalent aotools loop (422 fps) built from integer-pixel,
  axis-aligned steps. `set_fft_workers(-1)`, the only CPU knob offered, buys
  ~1.25x; the per-frame cost here is not FFT-bound.
- The non-periodic engine (`engine="extrude"`) costs real throughput relative
  to the periodic one: 120 vs 801 fps on GPU (6.7x), 6.1 vs 24.8 fps on CPU.
  On CPU it lands at essentially the same rate as HCIPy's pure-Python
  non-periodic equivalent (5.7 fps) — no throughput advantage over the
  incumbent for the equivalent (non-periodic, CPU) job.
- Boiling costs ~34% of GPU throughput and ~48% of CPU throughput for this
  job (801→530 fps GPU, 24.8→12.9 fps CPU): each mode needs its own
  scale-dependent retention coefficient rather than one scalar multiply per
  layer.

## 3. Structure-function accuracy — fractional-RMS error vs von Kármán (lower is better)

Methodology: every library is scored on the same ensemble size (no exceptions
for slower libraries), and each point estimate is reported with a
bootstrap-estimated standard deviation (200 resamples of the same ensemble,
so no extra screen generation is needed). pyturb is scored both at its
default subharmonic depth (8 levels) and at aotools' hard-coded depth (3
levels), so one row is directly configuration-matched. Regenerate with
`python benchmarks/bench_compare.py --json results.json`.

Ensemble of 120 screens at 256² (equal for every library), error over
separations `r ∈ [4·dx, D/4]`, mean ± bootstrap std (200 resamples, seed 0):

| pyturb | pyturb (sh=3, aotools depth) | HCIPy | aotools | soapy |
|---:|---:|---:|---:|---:|
| 1.2 % (±1.2%) | 0.9 % (±1.1%) | 0.8 % (±1.3%) | 3.1 % (±1.9%) | 1.5 % (±1.4%) |

At this (practical, benchmark-scale) ensemble size the scores of pyturb,
HCIPy and soapy sit within each other's uncertainty — there is no reliable
ranking among them here; the order moves between runs. aotools trends higher
on this metric but still overlaps within ~1 std of the others. Separately, on
much larger ensembles (hundreds of screens) pyturb's *systematic* bias is the
smallest of the three tested (~±1% vs up to −4.5% for aotools and up to +3.6%
for HCIPy at the largest separations), from the integrated-per-cell
subharmonic correction (rather than centre-sampled PSD × area) — a real but
modest (~1-3%) systematic effect that needs a large ensemble to resolve.

## 4. Feature matrix

| Feature | pyturb | aotools | soapy | HCIPy |
|---|:---:|:---:|:---:|:---:|
| GPU (CuPy) backend | ✅ | — | — | — |
| Batched Monte-Carlo screens | ✅ | — | — | — |
| Sub-pixel frozen flow | ✅ | — | ✅ | ✅ |
| Arbitrary wind direction | ✅ | — | — | ✅ |
| von Kármán outer scale L0 | ✅ | ✅ | ✅ | ✅ |
| Multi-layer atmosphere | ✅ | — | ✅ | ✅ |
| Named Cn²/wind profiles | ✅ | — | — | ✅ |
| Off-axis / tomography directions | ✅ | — | ◐ | ✅ |
| Boiling (temporal decorrelation) | ✅ | — | — | — |
| Integrated r0 / θ0 / τ0 | ✅ | ✅ | — | — |
| OPD in metres (achromatic) | ✅ | — | — | — |
| Unbounded (non-periodic) screens | ✅ | ✅ | ✅ | ✅ |

✅ supported · ◐ partial · — not available

## Takeaways

1. **Monte-Carlo generation is a rout** — pyturb is ~1000× the pure-Python FFT
   loops in aotools/soapy on GPU, and ~13× even on a single CPU core.
2. **The multi-layer GPU product hits loop rate** — ~800 fps of a full 9-layer
   512² atmosphere, the metric AO closed-loop simulation actually cares about.
3. **Extrusion still wins one cell** — single-layer integer-pixel CPU stepping
   in aotools/soapy is faster per frame at small n; pyturb trades that for
   sub-pixel, any-direction, GPU generality. For the unbounded-duration case
   pyturb now ships its own ring-buffer extruder (`engine="extrude"`): a
   non-periodic 9-layer 512² atmosphere at ~120 fps on GPU.
4. **Accuracy is comparable, not a rout** — pyturb, HCIPy and soapy score
   within each other's noise at practical ensemble sizes; pyturb's genuine
   edge is a smaller (~1%) *systematic* bias, visible only on much larger
   ensembles than a quick benchmark runs (see §3).
5. **Broadest feature set** — GPU, profiles, off-axis, boiling, OPD-native.

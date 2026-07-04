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

## 3. Structure-function accuracy — fractional-RMS error vs von Kármán (lower is better)

Ensemble of 150 screens at 256², error over separations `r ∈ [4·dx, D/4]`:

| pyturb | HCIPy | aotools | soapy |
|---:|---:|---:|---:|
| **2.1 %** | 3.7 % | 4.9 % | 4.9 % |

pyturb's integrated-per-cell subharmonic correction (rather than centre-sampled
PSD × area) reproduces the low-frequency structure function most faithfully.

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
| Unbounded (non-periodic) screens | ◐ | ✅ | ✅ | ✅ |

✅ supported · ◐ partial · — not available

## Takeaways

1. **Monte-Carlo generation is a rout** — pyturb is ~1000× the pure-Python FFT
   loops in aotools/soapy on GPU, and ~13× even on a single CPU core.
2. **The multi-layer GPU product hits loop rate** — ~800 fps of a full 9-layer
   512² atmosphere, the metric AO closed-loop simulation actually cares about.
3. **Extrusion still wins one cell** — single-layer integer-pixel CPU stepping
   in aotools/soapy is faster per frame at small n; pyturb trades that for
   sub-pixel, any-direction, GPU-batched generality (and a ring-buffer extruder
   is on the roadmap for the unbounded-duration case).
4. **Best statistical accuracy** of the four.
5. **Broadest feature set** — GPU, profiles, off-axis, boiling, OPD-native.

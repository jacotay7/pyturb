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
| CPU | 32-core; SciPy 1.16 / NumPy 2.2; Numba 0.63 (`pyturb[accel]`) |
| Python | 3.12 |
| pyturb | 0.2.0 |
| aotools | 0.1.dev477 |
| soapy | 0.15.0 |
| HCIPy | 0.7.0 |

Setup: 8 m pupil sampled at `n` pixels, von Kármán turbulence, `r0 = 0.15 m`
at 500 nm, `L0 = 25 m`. CPU rows use the optional `pyturb[accel]` (Numba)
extra; regenerate the per-use-case sweep with `python benchmarks/bench_suite.py`.

## 1. Generation throughput — independent screens / s (higher is better)

| n | pyturb GPU (batched) | pyturb GPU | pyturb CPU (batched) | aotools | soapy |
|---:|---:|---:|---:|---:|---:|
| 256 | **107,980** | 2,710 | 974 | 51 | 50 |
| 512 | **30,663** | 2,542 | 211 | 12 | 12 |

pyturb draws two screens per complex FFT, evaluates all subharmonic levels in
one batched matmul, and batches the whole Monte-Carlo stack into one call; on
the GPU that is **well over 1000× `aotools`/`soapy`** at 512² (they loop
`ft_sh_phase_screen` in Python, one screen at a time). HCIPy has no direct
i.i.d. FFT-screen entry point (screens come from constructing a layer), so it is
not listed here. pyturb clears the plan's ≥ 10⁴ screens/s @ 256² Monte-Carlo
target by 10×.

## 2. Frozen-flow throughput — pupil phase frames / s (higher is better)

| n | pyturb GPU, 9-layer | pyturb GPU, 1-layer | aotools 1-layer | soapy 1-layer | HCIPy 1-layer | pyturb CPU 1-layer |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | **3,094** | 5,663 | 18,433 | 17,387 | 201 | 2,927 |
| 512 | **3,161** | 5,741 | 5,083 | 5,034 | 52 | 666 |
| 1024 | **1,494** | 2,359 | 174 | 179 | 13 | 140 |

This axis is **not apples-to-apples**, and that is the interesting part:

- **`aotools` / `soapy` `add_row`** advance the screen by *one integer pixel
  along a fixed axis* — an O(n·stencil) matvec. Very fast per step on CPU at
  small n, but no sub-pixel offset, no arbitrary wind direction, no GPU, and
  the cost grows steeply — they fall behind pyturb-GPU by n=1024 (174/179 vs
  2,359 fps).
- **pyturb `FourierFlowScreen`** does one FFT per frame to deliver *exact
  sub-pixel translation in an arbitrary direction* — more work per step, but
  the general operation an AO loop actually needs, and it is flat in n on the
  GPU (~5,700 fps at 256²–512²).
- **pyturb GPU, 9-layer** is the real product: a full ESO Paranal-median
  atmosphere summed to pupil OPD, **~3,160 fps at 512²**. Building the same
  9-layer atmosphere from `aotools`/`soapy` means nine `add_row` calls + sum
  per frame on CPU, with no sub-pixel motion.
- **HCIPy `InfiniteAtmosphericLayer`** interpolates a stored screen (sub-pixel,
  fixed direction), CPU-only.

### 2b. The full 9-layer closed loop, every configuration (512², same machine)

The table above only shows *1-layer* CPU numbers for the competitors and
pyturb's GPU spectral engine for the "real product" row. Here is the same
9-layer paranal-median job across every engine/device combination, plus the
equivalent full job built directly from aotools and HCIPy on CPU:

CPU rows use the optional `pyturb[accel]` (Numba) extra.

| configuration | fps |
|---|---:|
| pyturb spectral, GPU (periodic) | 3,004 |
| pyturb extrude, GPU (non-periodic — the engine long runs need) | 1,729 |
| pyturb spectral + boiling, GPU | 2,100 |
| pyturb spectral, CPU (Numba accel) | 270 |
| pyturb spectral, CPU, `set_fft_workers(-1)` | 240 |
| pyturb spectral + boiling, CPU | 19 |
| pyturb extrude, CPU (Numba accel) | 164 |
| aotools, 9x `add_row` + sum, CPU (integer-pixel, axis-aligned) | 422 |
| HCIPy, 9-layer `evolve_until`+`phase_for`, CPU (sub-pixel, non-periodic) | 5.7 |

Reading this honestly:

- The GPU spectral number is the one that headlines elsewhere in this
  document; it is real, but it is the *periodic* engine — a run longer than
  `n·pixel_scale / wind_speed` re-samples the same screen realisation for
  that layer (`Atmosphere.time_to_wrap` reports the threshold;
  `PeriodicWrapWarning` fires the first time a run crosses it).
- On CPU, the same 9-layer job runs at ~270 fps — now within ~1.6x of the
  equivalent aotools loop (422 fps) built from integer-pixel, axis-aligned
  steps, and it does exact sub-pixel, arbitrary-direction motion the aotools
  loop cannot. `set_fft_workers(-1)` no longer helps here (it is ~11% slower):
  the per-frame cost is now the fused layer sum and the subharmonic matmul, not
  the single inverse FFT, so spreading that FFT across cores only adds
  dispatch overhead. Threaded FFTs still help at 1024² and for large
  Monte-Carlo batches, where the transform dominates.
- The non-periodic engine (`engine="extrude"`) costs ~1.7x throughput relative
  to the periodic one on GPU (1,729 vs 3,004 fps) — its fused CUDA readout
  kernel closed most of the old gap — and ~1.6x on CPU (164 vs 270 fps). On CPU
  it is ~29x HCIPy's pure-Python non-periodic equivalent (5.7 fps) for the same
  non-periodic job.
- Boiling costs ~30% of GPU throughput (3,004→2,100 fps) but far more on CPU
  (270→19 fps): the per-frame `(2, L, n, n)` fresh-noise draw for the AR(1)
  update is cheap on the GPU RNG and dominates the single-threaded CPU RNG.

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
   loops in aotools/soapy on GPU (comparing pyturb's batched, device-resident
   throughput against their single-call CPU latency), and ~13× even on a single
   CPU core.
2. **The multi-layer GPU product hits loop rate** — ~3,000 fps of a full
   9-layer 512² spectral atmosphere (§2b), the metric AO closed-loop simulation
   actually cares about.
3. **Extrusion still wins one cell** — single-layer integer-pixel CPU stepping
   in aotools/soapy is faster per frame at small n; pyturb trades that for
   sub-pixel, any-direction, GPU generality. For the unbounded-duration case
   pyturb now ships its own ring-buffer extruder (`engine="extrude"`): a
   non-periodic 9-layer 512² atmosphere at ~1,700 fps on GPU (§2b).
4. **Accuracy is comparable, not a rout** — pyturb, HCIPy and soapy score
   within each other's noise at practical ensemble sizes; pyturb's genuine
   edge is a smaller (~1%) *systematic* bias, visible only on much larger
   ensembles than a quick benchmark runs (see §3).
5. **Broadest feature set** — GPU, profiles, off-axis, boiling, OPD-native.

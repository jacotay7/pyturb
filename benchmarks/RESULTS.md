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

For pyturb's own throughput rows, capture the release evidence with
`python benchmarks/bench_suite.py --json benchmarks/artifacts/<name>.json`.
The JSON records the command, commit, runtime, dependencies, device selection,
batch size, and timing budget alongside every measurement; use it as the source
for future table updates rather than transcribing terminal output.

## Machine of record

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (32 GB, driver 580.159) |
| CuPy / CUDA | 13.6.0 / 12.x |
| CPU | 32-core; SciPy 1.16 / NumPy 2.2; Numba 0.63 (`pyturb[accel]`) |
| Python | 3.12 |
| pyturb | 1.0.0 (release artifact revision recorded below) |
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
- **pyturb GPU, 9-layer** is the real product: a full 9-layer
  `paranal-median` atmosphere (representative, not a published ESO table) summed to pupil OPD, **~3,160 fps at 512²**. Building the same
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
| pyturb spectral, GPU (periodic) | 3,133 |
| pyturb extrude, GPU (non-periodic — the engine long runs need) | 1,733 |
| pyturb spectral + boiling, GPU | 2,134 |
| pyturb extrude, CPU (Numba accel) | 324 |
| pyturb spectral, CPU (Numba accel) | 283 |
| pyturb spectral + boiling, CPU | 21 |
| aotools, 9x `add_row` + sum, CPU (integer-pixel, axis-aligned) | 422 † |
| HCIPy, 9-layer `evolve_until`+`phase_for`, CPU (sub-pixel, non-periodic) | 5.7 † |

pyturb rows are recorded in
[`artifacts/v1.0.0-reference.json`](artifacts/v1.0.0-reference.json), generated
by `bench_suite.py` on the machine of record (its provenance records
`source_dirty: true`: it was captured from a working tree with uncommitted
changes on top of the recorded revision, so treat it as indicative of 1.0.0
rather than bit-reproducible from that commit);
the aotools/HCIPy rows (†) are from the separate `bench_compare.py` harness and
are indicative cross-references, not part of the same run.

Reading this honestly:

- The GPU spectral number is the one that headlines elsewhere in this
  document; it is real, but it is the *periodic* engine — a run longer than
  `n·pixel_scale / wind_speed` re-samples the same screen realisation for
  that layer (`Atmosphere.time_to_wrap` reports the threshold;
  `PeriodicWrapWarning` fires the first time a run crosses it).
- On CPU, the same 9-layer job runs at ~283 fps on the spectral engine — within
  ~1.5x of the indicative aotools loop (422 fps) built from integer-pixel,
  axis-aligned steps, while doing exact sub-pixel, arbitrary-direction motion
  the aotools loop cannot.
- The non-periodic engine (`engine="extrude"`) costs ~1.8x throughput relative
  to the periodic one on GPU (1,733 vs 3,133 fps). On **CPU it is now the
  faster of the two** (324 vs 283 fps): its fused Numba readout — a per-frame
  rotated-gather over the ring buffer — beats a 512² inverse FFT on 32 cores,
  and it is ~75x HCIPy's pure-Python non-periodic equivalent (5.7 fps) for the
  same non-periodic, sub-pixel, any-direction job.
- Boiling costs ~32% of GPU throughput (3,133→2,134 fps) but far more on CPU
  (283→21 fps): the per-frame `(2, L, n, n)` fresh-noise draw for the AR(1)
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

## 5. pyturb 1.1 on an Arm workstation (RTX 4060, RTX A400, Neoverse-N1)

`bench_suite.py` on a second machine, after the 1.1 performance work (CUDA-graph
spectral frames, batched times, batched CPU extrusion). 9-layer
`paranal-median`, D = 8 m, 1 s per cell. Python 3.13, NumPy 2.5.3, SciPy 1.18.1,
CuPy 14.2.0 (the artifacts record it as `null`; the harness looked up the
wrong distribution name, fixed since), Numba 0.67, driver 580 / CUDA 13.0,
80-core Ampere Neoverse-N1 host with every run pinned to 16 idle cores.
Artifacts: [`v1.1.0-arm-rtx4060.json`](artifacts/v1.1.0-arm-rtx4060.json),
[`v1.1.0-arm-rtxa400.json`](artifacts/v1.1.0-arm-rtxa400.json),
[`v1.1.0-arm-neoverse-n1-16core.json`](artifacts/v1.1.0-arm-neoverse-n1-16core.json)
(revision `d38829e`, the pre-squash commit of #29; same tree as `main` at that
merge; `source_dirty: false`).

| metric | RTX 4060 256 / 512 / 1024 | RTX A400 256 / 512 / 1024 | 16× N1 CPU 256 / 512 / 1024 |
|---|---|---|---|
| `sample` screens/s | 33,292 / 6,993 / 1,715 | 9,309 / 2,466 / 615 | 704 / 173 / 43 |
| frames, spectral | 6,451 / 4,591 / 1,304 | 4,832 / 1,290 / 327 | 334 / 32 / 8 |
| `opd(times)` frames/s | 27,510 / 6,653 / 1,166 | 5,621 / 1,462 / 360 | 329 / 87 / 8 |
| frames, extrude | 1,174 / 482 / 120 | 410 / 103 / 25 | 398 / 82 / 21 |
| frames, spectral + boiling | 1,397 / 584 / 123 | 656 / 173 / 44 | 48 / 11 / 2 |
| tomography (5 dirs), dirs/s | 4,737 / 2,491 / 540 | 2,720 / 682 / 160 | 189 / 47 / 10 |
| frames, LGS cone | 208 / 168 / 49 | 139 / 36 / 11 | 25 / 6 / 0 |

Reading this:

- **Before/after 1.1 on the same RTX 4060:** spectral frames went from
  831 / 822 / 457 to 6,451 / 4,591 / 1,304 fps (CUDA-graph replay of fused
  kernels). Before, the frame rate was set by ~1.1 ms of host work per frame
  regardless of GPU or size. The 4060 now outruns the 1.0.0 RTX 5090 figures in
  §2b, which predate CUDA graphs (the 5090 was not re-measured).
- **CPU rows are indicative only.** On this shared host the CPU frequency
  governor ramps with load: the same 512² spectral loop measured anywhere from
  32 to 81 fps within one process. Compare CPU numbers only at equal load. The
  extruder's 512² CPU rate (82–98 fps in repeated runs) was ~40 before the
  batched extrusion.
- The LGS cone and boiling paths are not graph-captured, so they remain
  host-bound on the GPU.

## 6. pyturb 2.2 on the same Arm workstation

The §5 suite re-run on the same host after the 2.0 axis-convention change
(x = columns) and the move of Zernike bases to aobasis/aocore (2.1, 2.2).
Same configuration and protocol: 9-layer `paranal-median`, D = 8 m, 1 s per
cell, every run pinned to 16 idle cores (`taskset -c 16-31`). Python 3.13,
NumPy 2.5.3, SciPy 1.18.1, CuPy 14.2.0, Numba 0.67, driver 580.
Artifacts: [`v2.2.0-arm-rtx4060.json`](artifacts/v2.2.0-arm-rtx4060.json),
[`v2.2.0-arm-rtxa400.json`](artifacts/v2.2.0-arm-rtxa400.json),
[`v2.2.0-arm-neoverse-n1-16core.json`](artifacts/v2.2.0-arm-neoverse-n1-16core.json)
(revision `4922051`, `source_dirty: false`). The test suite, including the
GPU tests, and `validation/validate.py` pass on this machine.

| metric | RTX 4060 256 / 512 / 1024 | RTX A400 256 / 512 / 1024 | 16× N1 CPU 256 / 512 / 1024 |
|---|---|---|---|
| `sample` screens/s | 36,224 / 7,133 / 1,747 | 9,449 / 2,465 / 615 | 720 / 173 / 44 |
| frames, spectral | 6,649 / 4,508 / 1,304 | 4,815 / 1,285 / 327 | 222 / 51 / 8 |
| `opd(times)` frames/s | 27,652 / 6,673 / 1,166 | 5,614 / 1,461 / 359 | 312 / 100 / 8 |
| frames, extrude | 1,121 / 522 / 125 | 420 / 103 / 25 | 386 / 96 / 19 |
| frames, spectral + boiling | 1,415 / 608 / 124 | 666 / 173 / 44 | 45 / 12 / 2 |
| tomography (5 dirs), dirs/s | 5,081 / 2,485 / 540 | 2,715 / 681 / 159 | 191 / 47 / 10 |
| frames, LGS cone | 218 / 168 / 49 | 139 / 36 / 11 | 25 / 6 / 1 |

Reading this:

- **No regression from 2.0–2.2 on the GPU.** Every GPU cell is within
  0.95–1.09x of 1.1 (median 1.00x) on both cards. The axis-convention change
  and the aobasis/aocore migration do not touch the hot paths.
- **CPU rows move with the frequency governor, as in §5.** Individual cells
  range from 0.66x to 1.6x of 1.1 in both directions (the 1024² LGS cell
  rounds from 0.3 to 1 frame/s). Compare CPU numbers only at equal load.

## Takeaways

1. **Monte-Carlo generation is a rout** — pyturb is ~1000× the pure-Python FFT
   loops in aotools/soapy on GPU (comparing pyturb's batched, device-resident
   throughput against their single-call CPU latency), and ~13× even on a single
   CPU core.
2. **The multi-layer GPU product hits loop rate** — 3,133 fps of a full
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

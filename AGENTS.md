# AGENTS.md

Guidance for agents working in this repo. Keep it current if the CI workflow changes.

## Before considering any change done

Run these from the repo root (activate an env with the project installed
editable, e.g. `pip install -e ".[test,fits,docs]"`). These cover the local
equivalents of the CI checks.

```bash
ruff check .                                    # lint (must be clean, zero errors)
python -m pytest -q --cov=pyturb --cov-report=term-missing --cov-fail-under=85  # no-Numba path
python -m pytest -q tests/test_accel.py         # after installing .[accel]
python -m build --wheel && python -m pip install --force-reinstall dist/*.whl
python validation/validate.py --output /tmp/validation.png --metrics /tmp/validation.json
mkdocs build --strict                           # docs (only if you touched README/docs/mkdocs.yml)
python -c "import pyturb"                       # sanity import after any src/ change
python -m pytest -q --run-gpu                   # if you touched GPU code (needs CuPy + a GPU)
```

CI has no GPU runner, so `--run-gpu` locally is the only check of the CuPy
paths; record the GPU/CuPy versions in the PR when you run it.

CI additionally runs the test suite on Python 3.10, 3.12, 3.13, and 3.14. If
you only have one interpreter available, at minimum grep your diff for
anything that needs Python >=3.11 (`tomllib`, `ExceptionGroup`/`except*`,
`typing.Self`, etc.) — the project floor is `>=3.10`, and code should not
silently assume a newer numpy either (e.g. `np.trapezoid` requires NumPy
>= 2.0 and `np.trapz` was removed in a later release; the `numpy>=1.23` floor
needs a `np.trapezoid if hasattr(np, "trapezoid") else np.trapz` fallback,
already used in `profiles.py` — note `hasattr`, not `getattr`'s default,
since `getattr(np, "trapezoid", np.trapz)` still evaluates `np.trapz` eagerly
and breaks on NumPy releases that no longer have it). If in doubt, spin up a throwaway
`conda create -n py310check python=3.10` and run the suite there — this has
caught real bugs before.

## What "done" means here, beyond green tests

- **Exercise the actual behavior, not just the code path.** A test that
  calls a function and checks it doesn't throw is not a correctness test.
  Assert on values, statistics, or invariants that would actually catch the
  bug you just fixed or could plausibly introduce. See
  `tests/test_extrude.py::test_finescale_readout_flicker_is_bounded` or
  `tests/test_atmosphere.py::test_boiling_is_scale_dependent_not_uniform`
  for the pattern: characterize the real physical/statistical behavior with
  a bounded assertion, not just "it ran."
- **Check edge cases the existing tests don't reach**: a single large jump
  vs. many small steps (ring-buffer code in `extrude.py`/`infinite.py` has
  been bitten by this — compaction logic that only gets exercised by tiny
  steps hides bugs that surface on one big one), off-grid/boundary requests,
  values outside a declared range.
- **If you touch statistical/physical code**, verify against theory or a
  known reference where one exists (structure function vs. von Kármán
  theory, θ₀/τ₀ formulas, a cited profile table) rather than just checking
  the code runs. Don't trust a single-realization measurement — several
  seeds, or an ensemble average, distinguish a real effect from noise.
- **Match error message quality to the rest of the codebase**: when
  rejecting an invalid combination (see `Atmosphere.__init__`'s many
  `ValueError`s), say *why*, not just *that*. A bare "X requires Y" forces
  the next reader to spelunk the source to find out if it's a permanent
  architectural fact or a gap that might get lifted.
- **Update docstrings/README/RESULTS.md claims when behavior changes.**
  Several of the bugs found were docs stating something the code didn't
  actually do (or a claim that didn't survive scrutiny, e.g. a benchmark
  ranking within its own noise). A code fix that leaves a stale claim in
  place isn't finished.

## Style notes specific to this repo

- Comments and docstrings describe **current** behavior only — never
  "no more X" / "previously Y, now Z" / references to a past bug or a
  specific review. A future reader has no context for what "before" means;
  state what the code does now. (`CHANGELOG.md` is the one place that's
  supposed to narrate change over time.)
- Type annotations use `from __future__ import annotations` +
  `typing.Optional`/`Union` (not bare `X | Y`), matching the existing code.
- `ruff` line length is 90 (`pyproject.toml`); wrap before that, don't
  disable the rule.

## Axis convention (pyturb 2.0+)

- Arrays are indexed `(y, x)`: **x = axis 1 (columns), y = axis 0 (rows)**,
  shared with aobasis/makewfs/solvephase/shmpipeline-ao/HCIPy. Every public
  x/y-labelled input follows it (`Layer.wind_direction`/`wind_vector`,
  `directions=(thx, thy)`, `opd_at(x, y)`, `FourierFlowScreen.translate(sx,
  sy)`, `zernike_basis`, `theory.zernike_temporal_psd`).
- The engines (spectral flow, extruder, CUDA/Numba kernels) work in array-axis
  order. Convert only at the public boundary (`Atmosphere.__init__` wind,
  `_parse_directions`, `opd_at`, `translate`) and name internal quantities by
  axis (`v0`/`v1`, `slope0`/`slope1`, `disp0`/`disp1`, `pix0`/`pix1`) — never
  `x`/`y` for an axis-0/axis-1 value. `tests/test_axis_convention.py` pins the
  frame (pattern motion, off-axis footprint, translate, Zernike vs aobasis).
- 1.x used x = axis 0; `from_config` converts version-1 configs
  (`wind_direction -> 90 - a`). The mapping table lives in
  `docs/migration-2.md`.
- HCIPy's velocity sign is not uniform: measured on HCIPy 0.7,
  `InfiniteAtmosphericLayer` moves its pattern along `-velocity` (same as
  pyturb's `wind_vector`), `FiniteAtmosphericLayer` along `+velocity` with the
  components on (rows, cols). Re-measure before documenting a sign.

## Test-environment gotchas

- `tests/test_docs.py` executes every ```` ```python ```` block in `docs/`,
  skipping blocks whose optional imports are missing, so a block that passes in
  CI (no hcipy/poppy) can fail locally when only some of them are installed.
  Keep later blocks independent of state an optional block may or may not have
  mutated.

## Performance work: keeping seeded output bit-identical

Seeded screens are expected to stay bit-identical across optimizations (CPU on a
given platform and dependency set; the GPU's random draws everywhere, its OPD up
to cuFFT/cuBLAS kernel choice). Tolerance tests do not catch a change in the
last bit: before and after a hot-path change, save the outputs of many seeded
configurations (every engine, dtype, interp, boiling, LGS, directions, times,
`opd_at`, with and without Numba) and compare them bytewise.

- Numba kernels compiled with `fastmath=True` get their bits from LLVM's
  vectorizer: the spectral layer sum's layer reduction is split into
  interleaved partial sums with FMA contraction, chosen per ISA. Restructuring
  the loops (even adding `parallel=True` to the same body) changes the result.
  To parallelize such a kernel, keep the per-pixel function as it is and call
  it per row from a `prange` (see `_accel._spectral_rows`), then check bitwise.
- A fused kernel that replaces a chain of NumPy/CuPy elementwise operations
  must round each operation separately: Numba without `fastmath`, and
  `__fmul_rn`/`__fadd_rn` in CUDA (NVRTC contracts `a*b + c` into an FMA by
  default). Products with an exact zero (complex times a real promoted to
  complex) can be dropped; they only touch the sign of zero.
- `scipy.fft` (ducc) with `workers > 1` is not bit-identical to the
  single-threaded transform for a single 2-D array, so threading the FFT by
  default would change every seeded CPU screen.
- CuPy raises on cuBLAS calls during stream capture ("calling cuBLAS API
  during stream capture is currently unsupported"), so graph-captured frames
  use custom kernels instead of `matmul`. cuBLAS's summation order is not
  reproducible by a custom kernel, so replacing one changes the GPU OPD at
  float32 rounding.
- On the shared Arm bench host, Numba's OpenMP threads and OpenBLAS's threads
  both spin after use: a kernel timed alone can run several times faster than
  inside a frame. Profile frames in steady state (after warm-up), pin cores,
  and interleave baseline and candidate runs.

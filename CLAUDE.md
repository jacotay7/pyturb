# CLAUDE.md

Guidance for agents working in this repo. Keep it current if the CI workflow changes.

## Before considering any change done

Run these from the repo root (activate an env with the project installed
editable, e.g. `pip install -e ".[test,fits,docs]"`). All four mirror
`.github/workflows/ci.yml` exactly — if they pass locally, CI passes.

```bash
ruff check .                                    # lint (must be clean, zero errors)
python -m pytest -q --cov=pyturb --cov-report=term-missing   # full test suite + coverage
mkdocs build --strict                           # docs (only if you touched README/docs/mkdocs.yml)
python -c "import pyturb"                       # sanity import after any src/ change
```

CI additionally runs the test suite on Python 3.9, 3.11, 3.12, and 3.13. If
you only have one interpreter available, at minimum grep your diff for
anything that needs Python >=3.10 (`match` statements, `X | Y` type unions
used at runtime, etc.) — the project floor is `>=3.9`, and code should not
silently assume a newer numpy either (e.g. `np.trapezoid` requires NumPy
>= 2.0; the `numpy>=1.22` floor needs a `getattr(np, "trapezoid", np.trapz)`
fallback, already used in `profiles.py`). If in doubt, spin up a throwaway
`conda create -n py39check python=3.9` and run the suite there — this has
caught real bugs before.

## What "done" means here, beyond green tests

This codebase has been through an adversarial review that found real,
shipped bugs *behind* a 91%+ line-coverage test suite (silent data
corruption, crashes on documented inputs, a physically wrong model) — see
`trade_study_review/` for the full audit and how each was fixed. The lesson:
**passing tests and high coverage are necessary, not sufficient.** For any
nontrivial change:

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
  `typing.Optional`/`Union` (not bare `X | Y`), to stay valid on the
  `>=3.9` floor.
- `ruff` line length is 90 (`pyproject.toml`); wrap before that, don't
  disable the rule.

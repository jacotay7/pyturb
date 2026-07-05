# Contributing to pyturb

Thanks for your interest! pyturb aims to be the fastest, GPU-native, and
statistically-careful way to get atmospheric OPD into an AO workflow. A few
conventions keep it that way.

## Development setup

```bash
git clone https://github.com/jacotay7/pyturb
cd pyturb
pip install -e ".[test]"          # add ",fits" for the FITS I/O tests
pytest -q
```

For GPU work, install a CuPy build matching your CUDA toolkit
(`pip install cupy-cuda12x`); the suite skips GPU tests when CuPy is absent.

## The bar for a change

pyturb's credibility rests on three habits — please keep them:

1. **Every physics feature lands with an ensemble-statistics test against
   theory.** New turbulence behaviour must be shown to match a closed form
   (structure function, Noll variances, a PSD slope, …), not just "look right".
   See `tests/` for the pattern and `validation/validate.py` for the gallery.
2. **Every performance claim lands with a benchmark.** If you speed something
   up, add or update a script under `benchmarks/`.
3. **Every user-facing feature lands with docs.** A docstring at minimum; a
   `docs/` page or example if it's a new capability.

## Scope

pyturb is *the atmosphere*, not a full AO system. Please keep out of scope:
WFS/DM/controller simulation, tomographic reconstructors and slope
covariance, and Fresnel/scintillation propagation. We output
phase/OPD and hand those effects to the tools that own them — see
`docs/comparison.md`.

## Style

- `ruff check` must pass (`pip install ruff`) — CI enforces it. Match the
  surrounding style; the repo is hand-formatted, so `ruff format` is not imposed.
- NumPy-style docstrings; type hints on public signatures.
- Write backend-agnostic array code (works on NumPy and CuPy); avoid
  host↔device syncs inside hot loops.
- Prefer `float32` defaults for GPU throughput; keep a `float64` path for
  accuracy-sensitive setup.

## Pull requests

Small, focused PRs with tests are easiest to review. Note in the description
which of the three habits above your change satisfies. Update `CHANGELOG.md`
under *unreleased*.

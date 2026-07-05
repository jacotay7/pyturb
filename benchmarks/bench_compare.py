"""Head-to-head benchmark: pyturb vs aotools, soapy, and HCIPy.

Three axes, all on the same physical setup (an 8 m pupil sampled at ``n``
pixels, von Karman turbulence with ``r0 = 0.15 m`` and ``L0 = 25 m`` at
500 nm):

1. **Generation** -- independent von Karman screens per second (Monte-Carlo).
2. **Frozen flow** -- pupil phase frames per second while the layer blows.
3. **Accuracy**   -- fractional-RMS error of the ensemble structure function
   against the analytic von Karman prediction.

plus a static **feature matrix**.

Run::

    python benchmarks/bench_compare.py                 # full matrix, prints markdown
    python benchmarks/bench_compare.py --n 256 512     # choose sizes
    python benchmarks/bench_compare.py --json out.json  # also dump raw numbers

Comparison libraries are optional; any that are not importable are reported as
``n/a`` rather than failing the run. GPU rows require CuPy and appear only for
pyturb (aotools / soapy / HCIPy are CPU-only).

Fairness notes (printed with the results, because the engines do not all do the
same work per "frame"):

* pyturb's ``FourierFlowScreen`` applies the shift theorem -- one FFT per frame
  giving **exact sub-pixel translation in an arbitrary direction**.
* aotools / soapy ``PhaseScreenKolmogorov.add_row`` extrude **one integer pixel
  along a fixed axis** (an O(n * stencil) matvec, no sub-pixel, no direction).
  Fast per step on CPU, but not the same generality, and CPU-only.
* HCIPy ``InfiniteAtmosphericLayer`` interpolates a stored screen (sub-pixel,
  fixed direction), CPU-only.
* pyturb additionally batches independent screens and stacks layers into one
  FFT, which is where the GPU path pulls far ahead; the others have no batched
  or GPU path.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import pyturb

# 8 m pupil, standard von Karman layer at 500 nm.
DIAMETER = 8.0
R0 = 0.15
L0 = 25.0
WAVELENGTH = 500e-9


# ----------------------------------------------------------------------
# optional comparison libraries
# ----------------------------------------------------------------------
def _try(fn):
    try:
        return fn()
    except Exception:
        return None


_aotools_ps = _try(lambda: __import__("aotools.turbulence.phasescreen",
                                      fromlist=["ft_sh_phase_screen"]))
_aotools_inf = _try(lambda: __import__("aotools.turbulence.infinitephasescreen",
                                       fromlist=["PhaseScreenKolmogorov"]))
_soapy_atm = _try(lambda: __import__("soapy.atmosphere", fromlist=["phasescreen"]))
_hcipy = _try(lambda: __import__("hcipy"))
_vk = _try(lambda: __import__("aotools.turbulence",
                              fromlist=["structure_function_vk"]).structure_function_vk)

_have_gpu = _try(lambda: pyturb.get_array_module("gpu")) is not None


def _sync(device):
    if device == "gpu":
        import cupy

        cupy.cuda.runtime.deviceSynchronize()


def _rate(fn, seconds, device="cpu", warmup=3):
    """Iterations per second: warm up, then run ``fn`` for ~``seconds``."""
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    iters = 0
    while time.perf_counter() - t0 < seconds:
        fn()
        iters += 1
    _sync(device)
    return iters / (time.perf_counter() - t0)


# ----------------------------------------------------------------------
# 1. generation throughput (independent screens / s)
# ----------------------------------------------------------------------
def bench_generation(n, seconds):
    dx = DIAMETER / n
    out = {}

    for device in ("cpu", "gpu"):
        if device == "gpu" and not _have_gpu:
            continue
        g = pyturb.PhaseScreen(n, dx, R0, L0, seed=0, device=device)
        out[f"pyturb-{device}"] = _rate(lambda: g.generate(), seconds, device)
        # Batched Monte-Carlo -- one FFT for a whole stack (pyturb-only). Batch
        # size scales down with n to keep memory bounded; recorded per cell.
        batch = max(1, min(64, 2 ** int(np.log2(2 ** 24 / n / n))))
        gb = pyturb.PhaseScreen(n, dx, R0, L0, seed=0, device=device)
        out[f"pyturb-{device} (batched)"] = batch * _rate(
            lambda: gb.generate(batch), seconds, device)

    if _aotools_ps:
        s = [0]
        out["aotools"] = _rate(
            lambda: (s.__setitem__(0, s[0] + 1),
                     _aotools_ps.ft_sh_phase_screen(R0, n, dx, L0, 0.01, seed=s[0]))[1],
            seconds)
    if _soapy_atm:
        s = [0]
        out["soapy"] = _rate(
            lambda: (s.__setitem__(0, s[0] + 1),
                     _soapy_atm.phasescreen.ft_sh_phase_screen(R0, n, dx, L0, 0.01,
                                                               seed=s[0]))[1],
            seconds)
    # HCIPy has no direct iid FFT-screen entry point (screens come from a layer
    # whose construction includes covariance/interpolation setup) -> n/a here.
    return out


# ----------------------------------------------------------------------
# 2. frozen-flow frame throughput (pupil phase / s)
# ----------------------------------------------------------------------
def bench_flow(n, seconds):
    dx = DIAMETER / n
    out = {}

    for device in ("cpu", "gpu"):
        if device == "gpu" and not _have_gpu:
            continue
        tmpl = pyturb.PhaseScreen(n, dx, R0, L0, seed=0, device=device)
        ff = pyturb.FourierFlowScreen(tmpl, seed=0)
        k = [0]
        out[f"pyturb-{device}"] = _rate(
            lambda: (k.__setitem__(0, k[0] + 1), ff.translate(k[0] * dx, 0.0))[1],
            seconds, device)
        # The actual product: a full multi-layer atmosphere, batched over layers.
        atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8,
                                             diameter=DIAMETER, n=n, device=device)
        gen = iter(atm.frames(dt=1e-3, steps=10 ** 9))
        nl = len(pyturb.get_profile("paranal-median"))
        out[f"pyturb-{device} ({nl}-layer)"] = _rate(lambda: next(gen), seconds, device)

    if _aotools_inf:
        sc = _aotools_inf.PhaseScreenKolmogorov(n, dx, R0, L0)
        out["aotools"] = _rate(lambda: (sc.add_row(), sc.scrn)[1], seconds)
    if _soapy_atm:
        sc = _soapy_atm.infinitephasescreen.PhaseScreenKolmogorov(n, dx, R0, L0)
        out["soapy"] = _rate(lambda: (sc.add_row(), sc.scrn)[1], seconds)
    if _hcipy:
        grid = _hcipy.make_pupil_grid(n, DIAMETER)
        cn2 = _hcipy.Cn_squared_from_fried_parameter(R0, WAVELENGTH)
        lay = _hcipy.InfiniteAtmosphericLayer(grid, cn2, L0, velocity=10.0, height=0)
        t = [0.0]
        out["hcipy"] = _rate(
            lambda: (t.__setitem__(0, t[0] + 1e-3), lay.evolve_until(t[0]),
                     lay.phase_for(WAVELENGTH))[2],
            seconds)
    return out


# ----------------------------------------------------------------------
# 3. structure-function accuracy (fractional-RMS error vs von Karman)
# ----------------------------------------------------------------------
# This metric -- fractional RMS deviation of an ENSEMBLE-MEAN structure
# function from theory -- is dominated by Monte-Carlo noise at ensemble sizes
# a benchmark can afford (its spread across independent draws is comparable
# to the differences between libraries). Every library below therefore gets
# (a) exactly the same ensemble size and (b) a bootstrap-estimated standard
# deviation reported alongside the point estimate. pyturb is additionally
# scored at aotools' hard-coded subharmonic depth (3 levels, vs pyturb's own
# default of 8) so the comparison has one row at matched configuration.
def bench_accuracy(n, count, n_boot=200, boot_seed=0):
    dx = DIAMETER / n
    seps = np.arange(1, n // 4 + 1)
    r = seps * dx
    if _vk is None:
        return {}
    target = np.asarray(_vk(r, R0, L0))
    mid = (r >= 4 * dx) & (r <= DIAMETER / 4)

    def frac_err(dd_mean):
        rel = (dd_mean[mid] - target[mid]) / target[mid]
        return 100.0 * float(np.sqrt(np.mean(rel ** 2)))

    def score(get_screen, m):
        """(point estimate, bootstrap std) of frac_err over m screens.

        The point estimate uses the full pool exactly as before; the std
        comes from resampling that same pool with replacement, so it costs
        no extra (slow) screen generation -- only cheap array arithmetic.
        """
        pool = np.stack([
            pyturb.structure_function(
                np.asarray(get_screen(i)).reshape(n, n).astype(np.float64), dx
            )[1]
            for i in range(m)
        ])
        point = frac_err(pool.mean(axis=0))
        rng = np.random.default_rng(boot_seed)
        boot = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, m, size=m)
            boot[b] = frac_err(pool[idx].mean(axis=0))
        return point, float(boot.std())

    out = {}
    g = pyturb.PhaseScreen(n, dx, R0, L0, seed=1)
    out["pyturb"] = score(lambda i: g.generate(), count)
    g3 = pyturb.PhaseScreen(n, dx, R0, L0, subharmonics=3, seed=2)
    out["pyturb (sh=3, aotools depth)"] = score(lambda i: g3.generate(), count)
    if _aotools_ps:
        out["aotools"] = score(
            lambda i: _aotools_ps.ft_sh_phase_screen(R0, n, dx, L0, 0.01,
                                                     seed=1000 + i), count)
    if _soapy_atm:
        out["soapy"] = score(
            lambda i: _soapy_atm.phasescreen.ft_sh_phase_screen(R0, n, dx, L0, 0.01,
                                                                seed=2000 + i), count)
    if _hcipy:
        grid = _hcipy.make_pupil_grid(n, DIAMETER)
        cn2 = _hcipy.Cn_squared_from_fried_parameter(R0, WAVELENGTH)

        def hci(i):
            lay = _hcipy.InfiniteAtmosphericLayer(grid, cn2, L0, velocity=10.0,
                                                  height=0, seed=3000 + i)
            return np.asarray(lay.phase_for(WAVELENGTH))

        out["hcipy"] = score(hci, count)  # same ensemble size as every other library
    return out


# ----------------------------------------------------------------------
# feature matrix (static, curated from the libraries' documented APIs)
# ----------------------------------------------------------------------
FEATURES = [
    # feature,                       pyturb, aotools, soapy, hcipy
    ("GPU (CuPy) backend",              "yes", "no",  "no",  "no"),
    ("Batched Monte-Carlo screens",     "yes", "no",  "no",  "no"),
    ("Sub-pixel frozen flow",           "yes", "no",  "yes", "yes"),
    ("Arbitrary wind direction",        "yes", "no",  "no",  "yes"),
    ("von Karman outer scale L0",       "yes", "yes", "yes", "yes"),
    ("Multi-layer atmosphere",          "yes", "no",  "yes", "yes"),
    ("Named Cn2/wind profiles",         "yes", "no",  "no",  "yes"),
    ("Off-axis / tomography directions", "yes", "no", "part", "yes"),
    ("Boiling (temporal decorrelation)", "yes", "no", "no",  "no"),
    ("Integrated r0/theta0/tau0",       "yes", "yes", "no",  "no"),
    ("OPD in metres (achromatic)",      "yes", "no",  "no",  "no"),
    ("Unbounded (non-periodic) screens", "yes", "yes", "yes", "yes"),
]


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def _fmt(v, kind):
    if v is None:
        return "n/a"
    if kind == "rate":
        return f"{v:,.0f}" if v >= 1 else f"{v:.2g}"
    if kind == "pct_pm":
        mean, std = v
        return f"{mean:.1f}% (+/-{std:.1f}%)"
    return f"{v:.1f}%"


def _table(title, rows_by_n, order, kind):
    cols = []
    for res in rows_by_n.values():
        for k in res:
            if k not in cols:
                cols.append(k)
    cols.sort(key=lambda c: (order(c), c))
    print(f"\n### {title}\n")
    head = "| n | " + " | ".join(cols) + " |"
    print(head)
    print("|" + "---|" * (len(cols) + 1))
    for n, res in rows_by_n.items():
        cells = [_fmt(res.get(c), kind) for c in cols]
        print(f"| {n} | " + " | ".join(cells) + " |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--seconds", type=float, default=1.5,
                   help="wall-clock budget per timed cell")
    p.add_argument("--accuracy-count", type=int, default=120,
                   help="ensemble size for the structure-function accuracy "
                        "test -- same size for every library")
    p.add_argument("--accuracy-n", type=int, default=256,
                   help="single screen size for the accuracy test (n-robust, so "
                        "not swept -- a big ensemble at one size is what matters)")
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()

    libs = {"pyturb": pyturb.__version__,
            "aotools": getattr(_try(lambda: __import__("aotools")), "__version__", None),
            "soapy": getattr(_try(lambda: __import__("soapy")), "__version__", None),
            "hcipy": getattr(_hcipy, "__version__", None)}
    print("# pyturb comparison benchmark\n")
    print("Setup: 8 m pupil, von Karman r0=0.15 m @ 500 nm, L0=25 m.")
    print("Libraries: " + ", ".join(f"{k} {v}" for k, v in libs.items() if v))
    print(f"GPU: {'available' if _have_gpu else 'not available'}")

    gen = {n: bench_generation(n, args.seconds) for n in args.n}
    flow = {n: bench_flow(n, args.seconds) for n in args.n}
    # Accuracy is n-robust; run one large ensemble at a single size rather than
    # re-paying the slow CPU libraries' per-screen cost across the whole sweep.
    acc = {args.accuracy_n: bench_accuracy(args.accuracy_n, args.accuracy_count)}

    def gorder(c):
        return (0 if c.startswith("pyturb") else 1, "batch" in c)

    _table("Generation throughput (independent screens / s, higher is better)",
           gen, gorder, "rate")
    _table("Frozen-flow throughput (pupil phase frames / s, higher is better)",
           flow, gorder, "rate")
    _table("Structure-function error vs von Karman theory (lower is better; "
           "mean +/- bootstrap std over the ensemble, equal size for every "
           "library)", acc, lambda c: (0 if c.startswith("pyturb") else 1),
           "pct_pm")

    print("\n### Feature matrix\n")
    print("| Feature | pyturb | aotools | soapy | HCIPy |")
    print("|---|---|---|---|---|")
    for feat, *vals in FEATURES:
        print(f"| {feat} | " + " | ".join(vals) + " |")

    print("\n" + __doc__.split("Fairness notes")[1].replace("Fairness notes (",
          "Fairness notes: ("))

    if args.json:
        setup = {"diameter": DIAMETER, "r0": R0, "L0": L0, "wavelength": WAVELENGTH}
        acc_json = {
            n: {lib: {"mean_pct": mean, "std_pct": std} for lib, (mean, std) in res.items()}
            for n, res in acc.items()
        }
        with open(args.json, "w") as fh:
            json.dump({"libs": libs, "gpu": _have_gpu, "setup": setup,
                       "generation": gen, "flow": flow, "accuracy": acc_json,
                       "features": FEATURES}, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

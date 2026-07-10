"""Regenerate the pyturb validation gallery — evidence, not just claims.

Runs five ensemble checks against analytic turbulence theory and writes a
figure and machine-readable metrics. Each check also prints PASS/FAIL with its
tolerance, so this doubles as a (slow) integration test that can run in CI.

    python validation/validate.py
    python validation/validate.py --output /tmp/validation.png \
        --metrics /tmp/validation.json

Uses only pyturb + numpy + matplotlib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import pyturb  # noqa: E402
from pyturb import analysis  # noqa: E402

RESULTS = []


def check(name, ok, detail):
    RESULTS.append({"name": name, "passed": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def structure_function(ax):
    """Ensemble structure function vs Kolmogorov D(r) = 6.88 (r/r0)^{5/3}."""
    n, D, r0 = 256, 8.0, 0.15
    gen = pyturb.PhaseScreen(n=n, pixel_scale=D / n, r0=r0, L0=np.inf, seed=1,
                             dtype="float64")
    acc = None
    for _ in range(120):
        r, d = pyturb.structure_function(gen.generate(), D / n)
        acc = d if acc is None else acc + d
    measured = acc / 120
    theory = 6.88 * (r / r0) ** (5.0 / 3.0)
    band = (r >= 4 * D / n) & (r <= D / 4)
    err = np.sqrt(np.mean((measured[band] / theory[band] - 1) ** 2))
    ax.loglog(r, theory, "k--", label="Kolmogorov 6.88 (r/r0)$^{5/3}$")
    ax.loglog(r, measured, "o", ms=3, label="pyturb ensemble")
    ax.set(xlabel="separation r [m]", ylabel="D(r) [rad$^2$]",
           title=f"Structure function ({err*100:.1f}% rms)")
    ax.legend(fontsize=7)
    check("structure function vs Kolmogorov", err < 0.08, f"{err*100:.1f}% rms error")


def zernike_spectrum(ax):
    """Zernike-mode variances vs Noll (1976) for a Kolmogorov screen."""
    n, D, r0 = 128, 4.0, 0.4
    gen = pyturb.PhaseScreen(n=n, pixel_scale=D / n, r0=r0, L0=np.inf, seed=7,
                             dtype="float64")
    basis = analysis.zernike_basis(20, n)
    coeffs = analysis.zernike_decompose(gen.generate(800), 20, basis=basis)
    measured = coeffs.var(axis=0)
    j = np.arange(2, 21)
    noll = np.array([analysis.noll_variance(int(k), D, r0) for k in j])
    ax.semilogy(j, noll, "k--", marker="_", label="Noll (1976)")
    ax.semilogy(j, measured[1:20], "o", ms=3, label="pyturb")
    ax.set(xlabel="Noll index j", ylabel="mode variance [rad$^2$]",
           title="Zernike spectrum")
    ax.legend(fontsize=7)
    ratio = measured[3:20].sum() / noll[2:].sum()  # aggregate j=4..20
    check("Zernike variances vs Noll", 0.85 < ratio < 1.2,
          f"aggregate ratio {ratio:.2f}")


def temporal_psd(ax):
    """Single pupil point under frozen flow: temporal PSD slope **and** amplitude.

    The point-wise frozen-flow temporal PSD approaches ``f^{-8/3}`` at high
    frequency; measured over a finite screen and a realistic AO band the slope
    sits a little shallower (~ -2.3 to -2.5). This check runs a **non-wrapping**
    case (screen period ``n*pixel_scale = 8 m`` exceeds the 7 m of wind travel,
    so nothing repeats and no ``PeriodicWrapWarning`` fires), averages 8 seeds to
    stabilise the single-point periodogram, and reports:

    - the power-law **slope** with a bootstrap 95% CI over seeds, and
    - the **amplitude** relative to the analytic Kolmogorov frozen-flow
      single-point PSD ``W1(f) = 0.0774 r0^{-5/3} V^{5/3} f^{-8/3}`` (rad^2/Hz,
      one-sided). The ``r0^{-5/3}`` scaling is exact; the measured level sits
      within a factor of ~2 of theory for this band (the residual is the same
      finite-screen effect that shallows the slope).

    Phase (rad at 500 nm) is used, not OPD, so the amplitude is physical.
    """
    v, dt, steps, lam, r0 = 10.0, 1e-3, 700, 500e-9, 0.15
    layers = [pyturb.Layer(0.0, 1.0, wind_speed=v, wind_direction=0.0, L0=25.0)]
    freq, psds = None, []
    for seed in range(8):
        atm = pyturb.Atmosphere(layers, r0=r0, n=96, diameter=8.0, seed=seed,
                                subharmonics=8)
        assert atm.time_to_wrap > steps * dt  # non-wrapping by construction
        series = np.array([np.array(o)[48, 48]
                           for _, o in atm.frames(dt=dt, steps=steps, wavelength=lam)])
        f, p = analysis.temporal_psd(series, dt)
        freq = f
        psds.append(p)
    psds = np.array(psds)
    psd = psds.mean(axis=0)
    slope, amp = analysis.fit_power_law(freq, psd, fmin=5, fmax=40)
    # Bootstrap the ensemble slope over seeds (resample seeds -> mean PSD -> refit).
    rng = np.random.default_rng(0)
    boot = [analysis.fit_power_law(
                freq, psds[rng.integers(0, len(psds), len(psds))].mean(axis=0),
                fmin=5, fmax=40)[0]
            for _ in range(1000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Amplitude vs analytic frozen-flow single-point PSD (median over the band).
    analytic = 0.0774 * r0 ** (-5.0 / 3.0) * v ** (5.0 / 3.0) * freq ** (-8.0 / 3.0)
    band = (freq >= 5) & (freq <= 40)
    amp_ratio = float(np.median(psd[band] / analytic[band]))
    ax.loglog(freq, psd, lw=0.7, label="pyturb (8-seed mean)")
    ax.loglog(freq, amp * freq ** slope, "k--", label=f"fit f$^{{{slope:.2f}}}$")
    ax.loglog(freq, analytic, "r:", label="frozen-flow f$^{-8/3}$")
    ax.set(xlabel="frequency [Hz]", ylabel="PSD [rad$^2$/Hz]",
           title="Temporal PSD (1 pixel)")
    ax.legend(fontsize=7)
    ok = (-3.0 < slope < -2.0) and (1.0 < amp_ratio < 3.0)
    check("temporal PSD slope + amplitude (non-wrapping, 8-seed)", ok,
          f"slope {slope:.2f} (95% CI [{lo:.2f}, {hi:.2f}]), "
          f"amplitude {amp_ratio:.1f}x frozen-flow theory")


def angular_decorrelation(ax):
    """Anisoplanatism: differential variance vs (theta/theta0)^{5/3}."""
    atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, n=96,
                                         diameter=8.0, field_of_view=40,
                                         dtype="float64", seed=2)
    theta0 = atm.theta0
    # Stay near/below theta0, where the (theta/theta0)^{5/3} law holds; well
    # beyond it the differential variance saturates.
    angles = np.array([0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    dirs = [(0.0, 0.0)] + [(a, 0.0) for a in angles]
    opds = pyturb.to_numpy(atm.opd(0.0, directions=dirs, wavelength=500e-9))
    var = np.array([analysis.differential_variance(opds[0], opds[k + 1])
                    for k in range(len(angles))])
    theory = (angles / theta0) ** (5.0 / 3.0)
    ax.loglog(angles, theory, "k--", label="(θ/θ$_0$)$^{5/3}$")
    ax.loglog(angles, var, "o", ms=4, label="pyturb")
    ax.set(xlabel="off-axis angle [arcsec]", ylabel="residual var [rad$^2$]",
           title=f"Angular decorrelation (θ$_0$={theta0:.1f}\")")
    ax.legend(fontsize=7)
    # The power-law slope is the physics check (5/3); amplitude is within a
    # factor of a couple (finite screen + interpolation).
    slope = np.polyfit(np.log(angles), np.log(var), 1)[0]
    ratio = np.median(var / theory)
    check("angular decorrelation slope ~ 5/3", 1.3 < slope < 2.0,
          f"slope {slope:.2f}, amplitude x{ratio:.2f}")


def extruder_stationarity(ax):
    """Extruded screen variance shows no secular drift over a long run."""
    layer = pyturb.InfinitePhaseScreen(n=64, pixel_scale=0.05, r0=0.12, L0=2.0,
                                       seed=5, dtype="float64")
    var = []
    for _ in range(6000):
        layer.step()
        var.append(float(np.var(np.array(layer.screen))))
    var = np.array(var)
    window = np.convolve(var, np.ones(200) / 200, mode="valid")
    ax.plot(np.arange(len(window)), window, lw=0.8)
    ax.set(xlabel="step", ylabel="screen variance [rad$^2$]",
           title="Extruder stationarity (200-step mean)")
    first, second = var[:3000].mean(), var[3000:].mean()
    drift = abs(second - first) / first
    check("extruder long-run stationarity", drift < 0.1, f"{drift*100:.1f}% drift")


def _parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/images/validation.png"),
        help="path for the validation figure (default: docs/images/validation.png)",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        help="optional path for JSON validation metrics",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = _parse_args(argv)
    RESULTS.clear()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    structure_function(axes[0, 0])
    zernike_spectrum(axes[0, 1])
    temporal_psd(axes[0, 2])
    angular_decorrelation(axes[1, 0])
    extruder_stationarity(axes[1, 1])
    axes[1, 2].axis("off")
    summary = "\n".join(
        f"{'PASS' if result['passed'] else 'FAIL'}  {result['name']}"
        for result in RESULTS
    )
    axes[1, 2].text(0.02, 0.95, "pyturb validation\n\n" + summary, va="top",
                    family="monospace", fontsize=9, transform=axes[1, 2].transAxes)
    fig.suptitle(f"pyturb {pyturb.__version__} — turbulence validated against theory",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=110)
    plt.close(fig)
    print(f"\nwrote {args.output}")
    n_pass = sum(result["passed"] for result in RESULTS)
    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "pyturb_version": pyturb.__version__,
                    "checks": RESULTS,
                    "passed": n_pass,
                    "total": len(RESULTS),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.metrics}")
    print(f"{n_pass}/{len(RESULTS)} checks passed")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

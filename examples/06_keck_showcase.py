"""06 — Keck showcase: four turbulence cases, live throughput.

Renders an animated clip of the Keck (Mauna Kea) atmosphere evolving under
frozen-flow wind, in four panels that build on one another:

  1. NGS frozen flow  -- pure wind-blown frozen flow, ``engine="extrude"``
                          (Assemat-Wilson row extrusion; the screen never
                          repeats)
  2. NGS boiling       -- local temporal decorrelation on top of the wind
                          (``tau_boil``), ``engine="spectral"`` (its per-mode
                          AR(1) boiling is far cheaper than the extruder's;
                          the screen is periodic, but the run stays well
                          under one period, see ``DT``/``N_FRAMES`` below)
  3. LGS on-axis       -- finite-range sodium beacon (``lgs_altitude``); each
                          layer's footprint is magnified by the cone,
                          ``engine="extrude"``
  4. LGS off-axis       -- the cone beacon pointed off-axis (``directions``),
                          adding angular anisoplanatism, ``engine="extrude"``

Each panel is overlaid with the throughput pyturb sustained generating that
case (independent screens per second, measured on this machine with warm
kernels), so the clip doubles as a speed sheet across the feature set.
Boiling (panel 2) runs somewhat below the frozen panels -- an AR(1) update on
every stored mode, versus a translation -- but stays in the same thousands-of-
fps range on a GPU; ``engine="extrude"`` boiling would not (see
``ExtrudeBoilingPerformanceWarning``), which is why panel 2 uses ``"spectral"``.

Run:  ``python examples/06_keck_showcase.py``            (auto GPU if available)
      ``python examples/06_keck_showcase.py --device cpu --n 256``
      ``python examples/06_keck_showcase.py --out docs/assets/keck.webp``

Needs matplotlib + pillow (``pip install pyturb[docs]`` covers both); a CUDA
GPU (``pyturb[cuda12]``) turns the four thousand-fps panels into the headline.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, List, Optional

import numpy as np

import pyturb

# --- the shared atmosphere: one sky, sampled four ways -----------------------
SITE = "keck"       # KAON-303 Cn2 profile: Keck Observatory, Mauna Kea summit
SEEING = 0.8        # arcsec at 500 nm
DIAMETER = 10.0     # m (Keck aperture)
SEED = 1
FIELD_OF_VIEW = 12.0     # arcsec half-width the screens are oversized to cover
OFF_AXIS = 8.0           # arcsec off-axis pointing for panel 4 (~5x theta0,
                         # a clearly resolved anisoplanatic error -- see the
                         # on-panel "off-axis error" badge)
LGS_ALTITUDE = 90_000.0  # m — sodium layer
TAU_BOIL = 300e-3        # s — boiling decorrelation time (panel 2); on the
                         # order of the few-hundred-ms non-frozen-flow
                         # timescales typical of Mauna Kea-class turbulence

# --- the movie ---------------------------------------------------------------
DT = 2.2e-3         # s per frame (frozen-flow step); stays under panel 2's
N_FRAMES = 150      # spectral screen's wrap time (see time_to_wrap)
PLAYBACK_FPS = 25   # clip playback rate (independent of the simulation DT)
BENCH_WARMUP = 12   # warm the kernels before timing throughput
BENCH_RUNS = 120    # frames timed for the per-panel screens/s number


def _resolve_device(requested: str) -> str:
    """``"auto"`` -> ``"gpu"`` when CuPy imports, else ``"cpu"``."""
    if requested != "auto":
        return requested
    try:
        import cupy  # noqa: F401

        return "gpu"
    except Exception:
        return "cpu"


def _device_label(device: str) -> str:
    if device == "gpu":
        try:
            import cupy as cp

            name = cp.cuda.runtime.getDeviceProperties(0)["name"]
            name = name.decode() if isinstance(name, bytes) else str(name)
            return name.replace("NVIDIA GeForce ", "").strip()
        except Exception:
            return "GPU"
    import platform

    return f"CPU ({platform.machine()})"


def _make_sync(device: str) -> Callable[[], None]:
    """A no-op on CPU; a full device barrier on GPU (so timings are honest)."""
    if device != "gpu":
        return lambda: None
    import cupy as cp

    return cp.cuda.Stream.null.synchronize


class Panel:
    """One case: how to build its Atmosphere, step one frame, and (optionally)
    accumulate an extra on-panel metric while frames are collected for display.

    ``step(atm, i)`` returns the on-device OPD for frame ``i``, driven by a
    monotonically increasing simulation clock (required by ``engine="extrude"``;
    harmless for ``engine="spectral"``). A fresh Atmosphere is built for each
    pass over the panel (benchmarking, then frame collection).
    """

    def __init__(self, tag: str, title: str, build, step, metric=None):
        self.tag = tag
        self.title = title
        self._build = build
        self._step = step
        self._metric = metric
        self.fps: Optional[float] = None
        self.metric_values: List[float] = []
        self.badge_extra: Optional[str] = None

    def build(self):
        return self._build()

    def step(self, atm, i: int):
        return self._step(atm, i)

    def record_metric(self, atm, i: int) -> None:
        """Accumulate this panel's extra metric (if any) for frame ``i``."""
        if self._metric is not None:
            self.metric_values.append(self._metric(atm, i))


def build_panels(device: str) -> List[Panel]:
    """The four cases, each sharing the common Keck (Mauna Kea) atmosphere."""
    common = dict(
        seeing=SEEING, diameter=DIAMETER, n=None,  # n filled in below
        seed=SEED, field_of_view=FIELD_OF_VIEW, device=device,
    )

    def make(n, **extra):
        kw = dict(common)
        kw["n"] = n
        kw.update(extra)
        return pyturb.Atmosphere.from_profile(SITE, **kw)

    def frozen_step(atm, i):             # on-axis frozen flow
        return atm.opd(i * DT)

    def boil_step(atm, i):               # frozen flow + boiling (spectral)
        return atm.opd(0.0) if i == 0 else atm.evolve(DT)

    def offaxis_step(atm, i):            # off-axis cone beacon
        return atm.opd(i * DT, directions=[(OFF_AXIS, 0.0)])[0]

    def aniso_metric(atm, i):
        """Anisoplanatic error [nm rms] between on- and off-axis, this frame."""
        onax, offax = atm.opd(i * DT, directions=[(0.0, 0.0), (OFF_AXIS, 0.0)])
        onax, offax = pyturb.to_numpy(onax), pyturb.to_numpy(offax)
        return float(np.std(offax - onax) * 1e9)

    n = build_panels.n
    return [
        Panel(
            "ngs",
            "NGS Frozen Flow",
            lambda: make(n, engine="extrude"),
            frozen_step,
        ),
        Panel(
            "boil",
            "NGS Boiling",
            lambda: make(n, engine="spectral", tau_boil=TAU_BOIL),
            boil_step,
        ),
        Panel(
            "lgs_on",
            "LGS On-Axis",
            lambda: make(n, engine="extrude", lgs_altitude=LGS_ALTITUDE),
            frozen_step,
        ),
        Panel(
            "lgs_off",
            "LGS Off-Axis",
            lambda: make(n, engine="extrude", lgs_altitude=LGS_ALTITUDE),
            offaxis_step,
            metric=aniso_metric,
        ),
    ]


build_panels.n = 512  # set from CLI in main()


def benchmark_panel(panel: Panel, sync: Callable[[], None]) -> float:
    """Screens per second for this case, with warm kernels and a device barrier."""
    atm = panel.build()
    i = 0
    for _ in range(BENCH_WARMUP):
        panel.step(atm, i)
        i += 1
    sync()
    t0 = time.perf_counter()
    for _ in range(BENCH_RUNS):
        panel.step(atm, i)
        i += 1
    sync()
    return BENCH_RUNS / (time.perf_counter() - t0)


def collect_frames(panel: Panel) -> np.ndarray:
    """``(N_FRAMES, n, n)`` piston-removed OPD in microns, on the host.

    Also drives :meth:`Panel.record_metric` for panels that define one (a
    separate Atmosphere pass from :func:`benchmark_panel`, so the extra metric
    query never pollutes the throughput number).
    """
    atm = panel.build()
    out = np.empty((N_FRAMES, atm.n, atm.n), dtype=np.float32)
    for i in range(N_FRAMES):
        opd = pyturb.to_numpy(panel.step(atm, i)) * 1e6  # m -> micron
        out[i] = opd - opd.mean()                        # remove piston
        panel.record_metric(atm, i)
    if panel.metric_values:
        panel.badge_extra = f"off-axis error: {np.mean(panel.metric_values):.0f} nm rms"
    return out


def render_animation(panels, frames, atm0, device_label, out_path):
    """Assemble the four panels into one animated clip via matplotlib + pillow.

    Saved as animated WebP (true colour, no palette) rather than GIF: a
    quantized GIF palette shared across frames bands and flickers subtly on
    the smooth ``RdBu_r`` OPD gradient as the underlying data drifts across
    quantization boundaries frame to frame; WebP has no such step.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    ink, sub, bg = "#e6edf3", "#9aa7b4", "#0b0f14"
    accent = "#ffd166"
    highlight = "#ff8a5c"
    scale = np.percentile(np.abs(frames[0]), 99.5)  # shared symmetric OPD scale

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 8.8), dpi=68)
    fig.patch.set_facecolor(bg)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.80, bottom=0.115,
                        wspace=0.03, hspace=0.22)

    images = []
    for ax, panel, stack in zip(axes.flat, panels, frames):
        ax.set_facecolor(bg)
        im = ax.imshow(stack[0], cmap="RdBu_r", vmin=-scale, vmax=scale,
                       interpolation="bilinear", animated=True)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#233040")
        ax.set_title(panel.title, color=ink, fontsize=13, fontweight="bold", pad=16)
        rate = f"{panel.fps:,.0f} screens/s" if panel.fps else ""
        ax.text(0.045, 0.955, rate, transform=ax.transAxes, color=accent,
                fontsize=12, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.32", fc="#11161d", ec="#2c3846"))
        if panel.badge_extra:
            ax.text(0.045, 0.045, panel.badge_extra, transform=ax.transAxes,
                    color=highlight, fontsize=12, fontweight="bold",
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.32", fc="#11161d",
                              ec=highlight, lw=1.3))
        images.append(im)

    fig.text(0.5, 0.965, "pyturb — Keck atmosphere showcase",
             color=ink, fontsize=17, fontweight="bold", ha="center", va="top")
    fig.text(
        0.5, 0.925,
        f"Mauna Kea · r₀={atm0.r0:.2f} m  θ₀={atm0.theta0:.1f}″  "
        f"τ₀={atm0.tau0 * 1e3:.1f} ms  @ {atm0.wavelength * 1e9:.0f} nm · "
        f"{atm0.n}² · {device_label}",
        color=sub, fontsize=10.5, ha="center", va="top")

    tstamp = fig.text(0.985, 0.098, "", color=sub, fontsize=10,
                      ha="right", va="bottom", family="monospace")
    fig.text(
        0.015, 0.098,
        "wind-driven frozen flow · overlay = live throughput on this machine",
        color=sub, fontsize=9, ha="left", va="bottom")

    # shared colorbar
    cax = fig.add_axes([0.30, 0.048, 0.40, 0.016])
    cb = fig.colorbar(images[0], cax=cax, orientation="horizontal")
    cb.set_label("OPD [µm]", color=sub, fontsize=9)
    cb.outline.set_edgecolor("#233040")
    cax.tick_params(colors=sub, labelsize=8)

    def rgba_frame() -> Image.Image:
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        return Image.fromarray(buf).convert("RGB")

    pil_frames = []
    for i in range(N_FRAMES):
        for im, stack in zip(images, frames):
            im.set_data(stack[i])
        tstamp.set_text(f"t = {i * DT * 1e3:5.0f} ms")
        pil_frames.append(rgba_frame())
    plt.close(fig)

    pil_frames[0].save(
        out_path, format="WEBP", save_all=True, append_images=pil_frames[1:],
        duration=int(1000 / PLAYBACK_FPS), loop=0, quality=92, method=6)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--n", type=int, default=512, help="grid size (pixels/side)")
    p.add_argument("--frames", type=int, default=None,
                   help="number of frames (default 150)")
    p.add_argument("--out", default="keck_showcase.webp")
    args = p.parse_args()

    global N_FRAMES
    if args.frames is not None:
        N_FRAMES = args.frames
    build_panels.n = args.n

    device = _resolve_device(args.device)
    label = _device_label(device)
    sync = _make_sync(device)
    print(f"device: {device}  ({label})")

    panels = build_panels(device)
    atm0 = panels[0].build()
    print(f"{SITE}: {len(atm0.layers)} layers  r0={atm0.r0:.3f} m  "
          f"theta0={atm0.theta0:.2f}\"  tau0={atm0.tau0 * 1e3:.1f} ms  n={atm0.n}")

    print("benchmarking throughput per panel ...")
    for panel in panels:
        panel.fps = benchmark_panel(panel, sync)
        print(f"  {panel.tag:8s} {panel.fps:8,.0f} screens/s")

    print(f"rendering {N_FRAMES} frames x {len(panels)} panels ...")
    frames = [collect_frames(panel) for panel in panels]

    out = render_animation(panels, frames, atm0, label, args.out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

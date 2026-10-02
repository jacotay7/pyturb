"""02 — Closed-loop OPD frames under frozen-flow wind.

Step a multi-layer atmosphere at a 1 kHz loop rate and watch the wavefront
error evolve. Run: ``python examples/02_closed_loop.py``

Half a second of paranal-median wind is longer than the periodic spectral
engine's ``time_to_wrap`` (0.25 s for the 32 m/s layer on an 8 m screen), so
this uses the non-periodic ``engine="extrude"``.
"""

import numpy as np

import pyturb

STEPS = 500
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, diameter=8.0,
                                     n=256, seed=1, engine="extrude")
print(atm)

times, wfe_nm = [], []
frames = []
for k, (t, opd) in enumerate(atm.frames(dt=1e-3, steps=STEPS)):
    opd = pyturb.to_numpy(opd)
    times.append(t)
    wfe_nm.append(opd.std() * 1e9)              # OPD is in metres
    if k in (0, STEPS - 1):                      # first and last frame
        frames.append(opd)
print(f"mean wavefront error = {np.mean(wfe_nm):.0f} nm rms")

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(frames[0] * 1e9, cmap="RdBu_r")
    axes[0].set_title("OPD @ t=0 [nm]")
    axes[0].axis("off")
    axes[1].imshow(frames[-1] * 1e9, cmap="RdBu_r")
    axes[1].set_title(f"OPD @ t={times[-1]:.3f} s [nm]")
    axes[1].axis("off")
    axes[2].plot(np.array(times) * 1e3, wfe_nm)
    axes[2].set(xlabel="time [ms]", ylabel="WFE [nm rms]",
                title="Wavefront error vs time")
    fig.tight_layout()
    plt.show()
except ImportError:
    print("install matplotlib to see the plots")

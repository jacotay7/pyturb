"""02 — Closed-loop OPD frames under frozen-flow wind.

Step a multi-layer atmosphere at a 1 kHz loop rate and watch the wavefront
error evolve. Run: ``python examples/02_closed_loop.py``
"""

import numpy as np

import pyturb

atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, diameter=8.0,
                                     n=256, seed=1)
print(atm)

times, wfe_nm = [], []
frames = []
for t, opd in atm.frames(dt=1e-3, steps=500):
    opd = pyturb.to_numpy(opd)
    times.append(t)
    wfe_nm.append(opd.std() * 1e9)              # OPD is in metres
    if t in (0.0, 0.25, 0.499):
        frames.append(opd)
print(f"mean wavefront error = {np.mean(wfe_nm):.0f} nm rms")

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(frames[0] * 1e9, cmap="RdBu_r")
    axes[0].set_title("OPD @ t=0 [nm]")
    axes[0].axis("off")
    axes[1].imshow(frames[-1] * 1e9, cmap="RdBu_r")
    axes[1].set_title("OPD @ t=0.5 s [nm]")
    axes[1].axis("off")
    axes[2].plot(np.array(times) * 1e3, wfe_nm)
    axes[2].set(xlabel="time [ms]", ylabel="WFE [nm rms]",
                title="Wavefront error vs time")
    fig.tight_layout()
    plt.show()
except ImportError:
    print("install matplotlib to see the plots")

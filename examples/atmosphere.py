"""Full layered atmosphere: representative profile, wind evolution, OPD frames.

Run:  python examples/atmosphere.py
Requires matplotlib for the (optional) plots.
"""

import numpy as np

import pyturb

# Build a representative atmosphere in one line: a named site profile, scaled
# to a chosen seeing, observed at 30 deg zenith with an 8 m telescope.
atm = pyturb.Atmosphere.from_profile(
    "paranal-median",
    seeing=0.8,          # arcsec @ 500 nm, at zenith
    zenith_angle=30.0,   # deg
    diameter=8.0,        # m
    n=256,               # pixels across the pupil
    seed=1,
)
print(atm)
print(f"  line-of-sight r0 = {atm.r0 * 100:.1f} cm   seeing = {atm.seeing:.2f}\"")
print(f"  isoplanatic angle theta0 = {atm.theta0:.2f}\"")
print(f"  coherence time tau0 = {atm.tau0 * 1e3:.1f} ms   "
      f"Greenwood f_G = {atm.greenwood_frequency:.0f} Hz")

# Closed-loop: OPD frames in metres under frozen-flow wind at a 1 kHz loop rate.
frames = [pyturb.to_numpy(opd) for _, opd in atm.frames(dt=1e-3, steps=200)]
rms_nm = np.array([f.std() for f in frames]) * 1e9
print(f"\nGenerated {len(frames)} OPD frames, mean RMS = {rms_nm.mean():.0f} nm")

# Off-axis: OPD toward several directions from the same turbulent volume
# (anisoplanatism / tomography input).
directions = [(0, 0), (10, 0), (20, 0)]  # arcsec offsets
multi = pyturb.to_numpy(atm.opd(t=0.0, directions=directions))
for (thx, thy), opd in zip(directions, multi):
    resid = (opd - multi[0]).std() * 1e9
    print(f"  direction ({thx:2d},{thy:2d})\": residual vs on-axis = {resid:.0f} nm RMS")

# Monte-Carlo: independent integrated OPDs for PSF / error-budget ensembles.
ensemble = atm.sample(32)
print(f"\nMonte-Carlo ensemble: {ensemble.shape}, "
      f"RMS spread = {pyturb.to_numpy(ensemble).std(axis=(-2, -1)).std() * 1e9:.0f} nm")

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(frames[0] * 1e6, cmap="RdBu_r")
    axes[0].set_title("OPD [micron], t = 0")
    axes[1].imshow(frames[-1] * 1e6, cmap="RdBu_r")
    axes[1].set_title(f"OPD after {len(frames)} steps")
    axes[2].plot(np.arange(len(frames)) * 1e3, rms_nm)
    axes[2].set_xlabel("time [ms]")
    axes[2].set_ylabel("OPD RMS [nm]")
    axes[2].set_title("Wavefront error vs time")
    fig.tight_layout()
    plt.show()
except ImportError:
    print("\nInstall matplotlib to see the plots.")

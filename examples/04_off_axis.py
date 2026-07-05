"""04 — Off-axis / tomography: anisoplanatism vs angle.

Ask for OPD toward several directions from the same turbulent volume and watch
the wavefronts decorrelate with angle. Run: ``python examples/04_off_axis.py``
"""

import numpy as np

import pyturb
from pyturb.analysis import differential_variance

# field_of_view oversizes the screens so off-axis footprints sample genuinely
# different (non-wrapped) turbulence.
atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, diameter=8.0,
                                     n=256, field_of_view=40.0, dtype="float64",
                                     seed=1)
print(f"isoplanatic angle theta0 = {atm.theta0:.2f}\"")

angles = np.array([0.0, 2.0, 4.0, 8.0, 16.0])
dirs = [(a, 0.0) for a in angles]
opds = pyturb.to_numpy(atm.opd(0.0, directions=dirs, wavelength=500e-9))
var = [differential_variance(opds[0], opds[k]) for k in range(len(angles))]
for a, v in zip(angles, var):
    print(f"  {a:4.0f}\" off-axis: residual variance = {v:6.2f} rad^2")

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(opds[0], cmap="RdBu_r")
    axes[0].set_title("on-axis [rad]")
    axes[0].axis("off")
    axes[1].imshow(opds[-1], cmap="RdBu_r")
    axes[1].set_title(f"{angles[-1]:.0f}\" off-axis [rad]")
    axes[1].axis("off")
    axes[2].plot(angles, var, "o-")
    axes[2].set(xlabel="off-axis angle [arcsec]", ylabel="residual var [rad$^2$]",
                title="Anisoplanatism")
    fig.tight_layout()
    plt.show()
except ImportError:
    print("install matplotlib to see the plots")

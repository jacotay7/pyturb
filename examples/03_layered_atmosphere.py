"""03 — A layered atmosphere and its integrated quantities.

Build a representative atmosphere from a named profile and report the numbers
an AO error budget needs. Run: ``python examples/03_layered_atmosphere.py``
"""

import numpy as np

import pyturb

atm = pyturb.Atmosphere.from_profile(
    "paranal-median", seeing=0.8, zenith_angle=30.0, diameter=8.0, n=256, seed=1,
)
print(atm)
print(f"  line-of-sight r0 = {atm.r0 * 100:.1f} cm    seeing = {atm.seeing:.2f}\"")
print(f"  isoplanatic angle theta0 = {atm.theta0:.2f}\"")
print(f"  coherence time tau0 = {atm.tau0 * 1e3:.1f} ms   "
      f"Greenwood f_G = {atm.greenwood_frequency:.0f} Hz")

# r0 is wavelength dependent (r0 ~ lambda^{6/5}); OPD is not.
for lam_nm in (500, 1650, 2200):
    print(f"  r0 @ {lam_nm:>4} nm = {atm.r0_at(lam_nm * 1e-9) * 100:.1f} cm")

try:
    import matplotlib.pyplot as plt

    alt = [ly.altitude for ly in atm.layers]
    frac = [ly.cn2_fraction for ly in atm.layers]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.barh(np.arange(len(alt)), frac, tick_label=[f"{h/1e3:.1f} km" for h in alt])
    a.set(xlabel=r"relative $C_n^2\,dh$", title="Turbulence profile")
    b.imshow(pyturb.to_numpy(atm.opd(wavelength=500e-9)), cmap="RdBu_r")
    b.set_title("Integrated pupil phase [rad]")
    b.axis("off")
    fig.tight_layout()
    plt.show()
except ImportError:
    print("install matplotlib to see the plots")

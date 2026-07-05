"""01 — Independent phase screens and their statistics.

Generate a batch of Kolmogorov screens and confirm the structure function
matches theory. Run: ``python examples/01_screens.py``
"""

import numpy as np

import pyturb

R0, PIXEL_SCALE = 0.15, 0.02
gen = pyturb.PhaseScreen(n=256, pixel_scale=PIXEL_SCALE, r0=R0, L0=np.inf,
                         seed=1, dtype="float64")
screens = gen.generate(50)                      # (50, 256, 256), one FFT batch
print(f"generated {screens.shape}  rms = {screens.std():.2f} rad")

r, measured = pyturb.structure_function(screens, PIXEL_SCALE)
theory = 6.88 * (r / R0) ** (5.0 / 3.0)         # Kolmogorov
print(f"structure function / theory = {np.mean(measured / theory):.3f} (want ~1)")

try:
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4))
    a.imshow(pyturb.to_numpy(screens[0]), cmap="RdBu_r")
    a.set_title("Kolmogorov screen [rad]")
    a.axis("off")
    b.loglog(r, measured, "o", ms=3, label="measured")
    b.loglog(r, theory, "k--", label=r"$6.88\,(r/r_0)^{5/3}$")
    b.set(xlabel="separation [m]", ylabel=r"$D_\phi(r)$ [rad$^2$]",
          title="Structure function")
    b.legend()
    fig.tight_layout()
    plt.show()
except ImportError:
    print("install matplotlib to see the plots")

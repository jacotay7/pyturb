"""pyturb quickstart: generate screens, verify statistics, animate wind.

Run:  python examples/quickstart.py
Requires matplotlib for the plots (pip install matplotlib).
"""

import numpy as np

import pyturb

# --- Independent screens (Monte-Carlo style) -----------------------------
R0 = 0.15  # Fried parameter [m]
PIXEL_SCALE = 0.02  # [m/pixel]

gen = pyturb.PhaseScreen(
    n=256, pixel_scale=PIXEL_SCALE, r0=R0, L0=np.inf, seed=1, dtype="float64"
)
screens = gen.generate(20)
print(f"Generated {screens.shape} screens, rms = {screens.std():.2f} rad")

# Validate against Kolmogorov theory: D(r) = 6.88 (r/r0)^(5/3)
r, measured = pyturb.structure_function(screens, PIXEL_SCALE)
theory = 6.88 * (r / R0) ** (5.0 / 3.0)
print(f"Structure function vs theory: {np.mean(measured / theory):.3f} (want ~1)")

# --- Frozen-flow evolution (closed-loop style) ----------------------------
layer = pyturb.InfinitePhaseScreen(n=128, pixel_scale=PIXEL_SCALE, r0=R0, L0=25, seed=1)
frames = [np.array(layer.step()) for _ in range(64)]

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(pyturb.to_numpy(screens[0]), cmap="RdBu_r")
    axes[0].set_title("FFT screen (Kolmogorov)")
    axes[1].loglog(r, measured, label="measured")
    axes[1].loglog(r, theory, "--", label=r"$6.88\,(r/r_0)^{5/3}$")
    axes[1].set_xlabel("separation [m]")
    axes[1].set_ylabel(r"$D_\phi(r)$ [rad$^2$]")
    axes[1].legend()
    axes[1].set_title("Structure function")
    axes[2].imshow(frames[-1], cmap="RdBu_r")
    axes[2].set_title("Infinite screen after 64 steps")
    fig.tight_layout()
    plt.show()
except ImportError:
    print("Install matplotlib to see the plots.")

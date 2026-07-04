"""pyturb — fast, GPU-optional atmospheric phase screens for adaptive optics.

Two generators cover the common AO simulation needs:

- :class:`PhaseScreen` — statistically independent Kolmogorov / von Kármán
  screens via the FFT method with subharmonic low-frequency correction.
- :class:`InfinitePhaseScreen` — an endless frozen-flow screen extruded row
  by row (Assémat & Wilson 2006) for closed-loop temporal simulation.

Pass ``device="gpu"`` to either class to run on CUDA via CuPy. Phase is
always returned in radians at the wavelength implied by ``r0``.
"""

from .backend import get_array_module, to_numpy
from .fourier import PhaseScreen
from .infinite import InfinitePhaseScreen, phase_covariance
from .utils import (
    r0_at_wavelength,
    r0_from_seeing,
    seeing_from_r0,
    structure_function,
)

__version__ = "0.1.0"

__all__ = [
    "PhaseScreen",
    "InfinitePhaseScreen",
    "phase_covariance",
    "structure_function",
    "r0_from_seeing",
    "seeing_from_r0",
    "r0_at_wavelength",
    "to_numpy",
    "get_array_module",
    "__version__",
]

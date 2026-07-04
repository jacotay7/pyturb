"""pyturb — fast, GPU-optional atmospheric phase screens for adaptive optics.

Three levels of API cover the common AO simulation needs:

- :class:`Atmosphere` — a full layered atmosphere: many turbulent layers with
  per-layer wind summed into pupil OPD, with frozen-flow time evolution,
  off-axis directions, and standard site profiles. This is the high-level
  entry point most users want.
- :class:`PhaseScreen` — statistically independent Kolmogorov / von Kármán
  screens via the FFT method with subharmonic low-frequency correction.
- :class:`InfinitePhaseScreen` — an endless frozen-flow screen extruded row
  by row (Assémat & Wilson 2006) for closed-loop temporal simulation.

Pass ``device="gpu"`` to any of them to run on CUDA via CuPy. Phase screens
are returned in radians; :class:`Atmosphere` returns OPD in metres (achromatic)
unless a ``wavelength`` is given.
"""

from .atmosphere import Atmosphere
from .backend import get_array_module, to_numpy
from .flow import FourierFlowScreen
from .fourier import PhaseScreen
from .infinite import InfinitePhaseScreen, phase_covariance
from .io import load, save
from .profiles import (
    Layer,
    bufton_wind,
    coherence_time,
    discretize_cn2,
    effective_wind_speed,
    get_profile,
    greenwood_frequency,
    hufnagel_valley,
    isoplanatic_angle,
    list_profiles,
    mean_turbulence_height,
)
from .utils import (
    air_refractivity,
    opd_to_phase,
    phase_to_opd,
    r0_at_wavelength,
    r0_from_seeing,
    seeing_from_r0,
    structure_function,
)

__version__ = "0.1.0"

__all__ = [
    "Atmosphere",
    "Layer",
    "PhaseScreen",
    "InfinitePhaseScreen",
    "FourierFlowScreen",
    "phase_covariance",
    "structure_function",
    "get_profile",
    "list_profiles",
    "hufnagel_valley",
    "bufton_wind",
    "discretize_cn2",
    "isoplanatic_angle",
    "coherence_time",
    "greenwood_frequency",
    "mean_turbulence_height",
    "effective_wind_speed",
    "r0_from_seeing",
    "seeing_from_r0",
    "r0_at_wavelength",
    "opd_to_phase",
    "phase_to_opd",
    "air_refractivity",
    "save",
    "load",
    "to_numpy",
    "get_array_module",
    "__version__",
]

"""Validated immutable configuration shared by phase-screen engines.

This module deliberately contains no backend dispatch or mutable simulation
state. Constructors validate their physical screen inputs here before choosing
NumPy or CuPy, so the periodic and infinite engines share one contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from .profiles import Layer, _fractions
from .utils import r0_from_seeing

__all__ = ["AtmosphereConfig", "GridConfig", "ScreenConfig"]


@dataclass(frozen=True)
class GridConfig:
    """Validated numerical grid and backend-independent output precision."""

    n: int
    pixel_scale: float
    device: str
    dtype: np.dtype

    @classmethod
    def create(
        cls, n: int, pixel_scale: float, device: str, dtype: Any
    ) -> "GridConfig":
        """Validate values required before CPU/GPU backend dispatch."""
        if not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError("n must be an integer of at least 2")
        if not np.isfinite(pixel_scale) or pixel_scale <= 0:
            raise ValueError("pixel_scale must be positive and finite")
        try:
            float_dtype = np.dtype(dtype)
        except TypeError as exc:
            raise ValueError("dtype must be float32 or float64") from exc
        if float_dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("dtype must be float32 or float64")
        return cls(int(n), float(pixel_scale), device, float_dtype)


@dataclass(frozen=True)
class LayerConfig:
    """Immutable layer model stored inside :class:`AtmosphereConfig`."""

    altitude: float
    cn2_fraction: float
    wind_speed: float
    wind_direction: float
    L0: float

    def to_layer(self) -> Layer:
        """Return the public mutable layer value used by runtime calculations."""
        return Layer(
            altitude=self.altitude,
            cn2_fraction=self.cn2_fraction,
            wind_speed=self.wind_speed,
            wind_direction=self.wind_direction,
            L0=self.L0,
        )


@dataclass(frozen=True)
class ScreenConfig:
    """Physical and numerical inputs common to all phase-screen engines."""

    n: int
    pixel_scale: float
    r0: float
    L0: float
    device: str
    dtype: np.dtype

    @classmethod
    def create(
        cls,
        n: int,
        pixel_scale: float,
        r0: float,
        L0: Union[float, None],
        device: str,
        dtype: Any,
        *,
        finite_outer_scale: bool,
    ) -> "ScreenConfig":
        """Validate and normalise a phase-screen configuration.

        ``None`` selects Kolmogorov turbulence (``L0=inf``), which is allowed
        only for the periodic FFT engine. The infinite extruder requires a
        finite outer scale because its covariance recurrence has finite
        variance only in that case.
        """
        grid = GridConfig.create(n, pixel_scale, device, dtype)
        if not np.isfinite(r0) or r0 <= 0:
            raise ValueError("r0 must be positive and finite")
        outer_scale = np.inf if L0 is None else L0
        if np.isnan(outer_scale) or outer_scale <= 0:
            raise ValueError("L0 must be positive (use numpy.inf for Kolmogorov)")
        if finite_outer_scale and not np.isfinite(outer_scale):
            raise ValueError(
                "InfinitePhaseScreen requires a finite positive outer scale L0"
            )
        return cls(
            n=grid.n,
            pixel_scale=grid.pixel_scale,
            r0=float(r0),
            L0=float(outer_scale),
            device=grid.device,
            dtype=grid.dtype,
        )


@dataclass(frozen=True)
class AtmosphereConfig:
    """Validated immutable inputs for a layered atmospheric model.

    Runtime objects (FFT plans, random streams, spectra, and ring buffers) do
    not belong here. They are constructed by :class:`pyturb.Atmosphere` from
    this model configuration.
    """

    layers: Tuple[LayerConfig, ...]
    r0_zenith: float
    wavelength: float
    zenith_angle: float
    diameter: float
    grid: GridConfig
    L0_override: Optional[float]
    power_law: float
    inner_scale: float
    subharmonics: int
    field_of_view: float
    tau_boil: Tuple[float, ...]
    has_tau_boil: bool
    engine: str
    interp: str
    lgs_altitude: Optional[float]
    dispersion: Optional[str]
    wet_fraction: float
    seed: Optional[int]

    @classmethod
    def create(
        cls,
        layers: Sequence[Layer],
        r0: Optional[float],
        seeing: Optional[float],
        wavelength: float,
        zenith_angle: float,
        diameter: float,
        n: int,
        L0: Optional[float],
        power_law: float,
        inner_scale: float,
        subharmonics: int,
        field_of_view: float,
        tau_boil: Union[float, Sequence[float], None],
        engine: str,
        interp: str,
        lgs_altitude: Optional[float],
        dispersion: Optional[str],
        wet_fraction: float,
        device: str,
        dtype: Any,
        seed: Optional[int],
    ) -> "AtmosphereConfig":
        """Validate model inputs and normalise values shared by both engines."""
        model_layers = list(layers)
        if not model_layers:
            raise ValueError("at least one layer is required")
        if (r0 is None) == (seeing is None):
            raise ValueError("give exactly one of r0 or seeing")
        if not np.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("wavelength must be positive and finite [m]")
        if not np.isfinite(zenith_angle) or not 0.0 <= zenith_angle < 90.0:
            raise ValueError("zenith_angle must be in [0, 90) degrees")
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError("diameter must be positive and finite")
        if not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError("n must be an integer of at least 2")
        grid = GridConfig.create(n, float(diameter) / n, device, dtype)
        if not np.isfinite(field_of_view) or field_of_view < 0:
            raise ValueError("field_of_view must be finite and >= 0 arcsec")
        if engine not in ("spectral", "extrude"):
            raise ValueError("engine must be 'spectral' or 'extrude'")
        if interp not in ("cubic", "linear", "lanczos"):
            raise ValueError("interp must be 'cubic', 'linear', or 'lanczos'")
        if not np.isfinite(power_law) or power_law <= 2.0:
            raise ValueError("power_law must be > 2 and finite (Kolmogorov is 11/3)")
        if not np.isfinite(inner_scale) or inner_scale < 0:
            raise ValueError("inner_scale must be >= 0 and finite (0 disables it)")
        if not isinstance(subharmonics, (int, np.integer)) or subharmonics < 0:
            raise ValueError("subharmonics must be an integer >= 0")
        if L0 is not None and (np.isnan(L0) or L0 <= 0):
            raise ValueError(
                "L0 override must be positive (use numpy.inf for Kolmogorov)"
            )
        if engine == "extrude" and power_law != 11.0 / 3.0:
            raise ValueError(
                "non-Kolmogorov power_law requires engine='spectral': the "
                "extruder's row-to-row recurrence is a closed-form "
                "conditional distribution derived specifically for the von "
                "Karman covariance (power_law=11/3); generalizing it to other "
                "exponents needs a different closed form, not just a different "
                "PSD, so it is not offered here. power_law is available for "
                "sample() and engine='spectral' frames()/opd()."
            )
        if engine == "extrude" and inner_scale > 0:
            raise ValueError(
                "inner_scale requires engine='spectral': the extruder's "
                "recurrence has no inner-scale term in its closed-form "
                "covariance. inner_scale is available for sample() and "
                "engine='spectral' frames()/opd()."
            )
        if lgs_altitude is not None and (
            not np.isfinite(lgs_altitude) or lgs_altitude <= 0
        ):
            raise ValueError("lgs_altitude must be positive and finite [m]")
        if dispersion not in (None, "edlen", "ciddor"):
            raise ValueError("dispersion must be None, 'edlen', or 'ciddor'")
        if not np.isfinite(wet_fraction) or not 0.0 <= wet_fraction <= 1.0:
            raise ValueError("wet_fraction must be finite and in [0, 1]")
        if wet_fraction > 0.0 and dispersion != "ciddor":
            raise ValueError(
                "wet_fraction > 0 requires dispersion='ciddor' (the wet/dry "
                "chromatic split): dispersion='edlen' models dry air only and "
                "dispersion=None is achromatic, so neither has a water-vapour "
                "term to weight."
            )
        if seeing is not None:
            if not np.isfinite(seeing) or seeing <= 0:
                raise ValueError("seeing must be positive and finite [arcsec]")
            r0 = r0_from_seeing(seeing, wavelength)
        if not np.isfinite(r0) or r0 <= 0:
            raise ValueError(
                "r0 (or the r0 implied by seeing) must be positive and finite [m]"
            )
        fractions = _fractions(model_layers)
        configured_layers = tuple(
            LayerConfig(
                altitude=layer.altitude,
                cn2_fraction=float(fraction),
                wind_speed=layer.wind_speed,
                wind_direction=layer.wind_direction,
                L0=layer.L0 if L0 is None else float(L0),
            )
            for layer, fraction in zip(model_layers, fractions)
        )
        if engine == "extrude" and any(
            not np.isfinite(layer.L0) for layer in configured_layers
        ):
            raise ValueError(
                "engine='extrude' requires a finite outer scale L0 for every "
                "layer: the extruder's row recurrence conditions on the von "
                "Karman phase covariance, which is only well-defined (finite "
                "variance) for finite L0. Kolmogorov (L0=inf) is only "
                "available for engine='spectral' or sample()."
            )
        if tau_boil is None:
            tau = np.full(len(configured_layers), np.inf)
        else:
            tau = np.broadcast_to(
                np.asarray(tau_boil, dtype=np.float64), (len(configured_layers),)
            ).astype(np.float64)
        if np.any(~np.isfinite(tau) & ~np.isinf(tau)) or np.any(tau <= 0):
            raise ValueError("tau_boil must be positive (or None for frozen flow)")
        return cls(
            layers=configured_layers,
            r0_zenith=float(r0),
            wavelength=float(wavelength),
            zenith_angle=float(zenith_angle),
            diameter=float(diameter),
            grid=grid,
            L0_override=None if L0 is None else float(L0),
            power_law=float(power_law),
            inner_scale=float(inner_scale),
            subharmonics=int(subharmonics),
            field_of_view=float(field_of_view),
            tau_boil=tuple(float(value) for value in tau),
            has_tau_boil=tau_boil is not None,
            engine=engine,
            interp=interp,
            lgs_altitude=None if lgs_altitude is None else float(lgs_altitude),
            dispersion=dispersion,
            wet_fraction=float(wet_fraction),
            seed=seed,
        )


@dataclass(frozen=True)
class ExtrusionConfig:
    """Validated immutable inputs for the direct multi-layer extruder."""

    grid: GridConfig
    layer_r0: Tuple[float, ...]
    layer_L0: Tuple[float, ...]
    layer_wind: Tuple[Tuple[float, float], ...]
    layer_altitude_los: Tuple[float, ...]
    field_of_view_pix: Tuple[float, ...]
    stencil_rows: int
    interp: str
    lgs_altitude_los: Optional[float]
    seeds: Tuple[Any, ...]
    tau_boil: Tuple[float, ...]
    boil_seed: Optional[Any]

    @classmethod
    def create(
        cls,
        n: int,
        pixel_scale: float,
        layer_r0: Sequence[float],
        layer_L0: Sequence[float],
        layer_wind: Sequence[Tuple[float, float]],
        layer_altitude_los: Sequence[float],
        field_of_view_pix: Union[float, Sequence[float]],
        stencil_rows: int,
        interp: str,
        lgs_altitude_los: Optional[float],
        device: str,
        dtype: Any,
        seeds: Optional[Sequence[Any]],
        tau_boil: Optional[Sequence[float]],
        boil_seed: Optional[Any],
    ) -> "ExtrusionConfig":
        """Validate layer-aligned direct-extrusion inputs before dispatch."""
        grid = GridConfig.create(n, pixel_scale, device, dtype)
        r0_values = tuple(float(value) for value in layer_r0)
        L0_values = tuple(float(value) for value in layer_L0)
        wind_values = tuple((float(vx), float(vy)) for vx, vy in layer_wind)
        altitude_values = tuple(float(value) for value in layer_altitude_los)
        n_layers = len(r0_values)
        if n_layers == 0:
            raise ValueError("at least one extrusion layer is required")
        if not (
            len(L0_values) == len(wind_values) == len(altitude_values) == n_layers
        ):
            raise ValueError(
                "layer_r0, layer_L0, layer_wind and layer_altitude_los must "
                f"have equal length (got {n_layers}, {len(L0_values)}, "
                f"{len(wind_values)}, {len(altitude_values)}); each is one "
                "entry per layer, so a length mismatch would silently drop "
                "layers when zipped."
            )
        if any(not np.isfinite(value) or value <= 0 for value in r0_values):
            raise ValueError("every layer_r0 value must be positive and finite")
        if any(not np.isfinite(value) or value <= 0 for value in L0_values):
            raise ValueError("every layer_L0 value must be positive and finite")
        if any(
            not np.isfinite(value)
            for wind in wind_values for value in wind
        ):
            raise ValueError("every layer_wind component must be finite")
        if any(not np.isfinite(value) for value in altitude_values):
            raise ValueError("every layer_altitude_los value must be finite")
        if np.ndim(field_of_view_pix) == 0:
            fov_values = (float(field_of_view_pix),) * n_layers
        else:
            fov_values = tuple(float(value) for value in field_of_view_pix)
        if len(fov_values) != n_layers:
            raise ValueError("field_of_view_pix must be scalar or one value per layer")
        if any(not np.isfinite(value) or value < 0 for value in fov_values):
            raise ValueError("field_of_view_pix values must be finite and >= 0")
        if (
            not isinstance(stencil_rows, (int, np.integer))
            or not 1 <= stencil_rows < grid.n
        ):
            raise ValueError("stencil_rows must be in [1, n)")
        if interp not in ("cubic", "linear", "lanczos"):
            raise ValueError("interp must be 'cubic', 'linear', or 'lanczos'")
        if lgs_altitude_los is not None and (
            not np.isfinite(lgs_altitude_los) or lgs_altitude_los <= 0
        ):
            raise ValueError("lgs_altitude_los must be positive and finite")
        if lgs_altitude_los is not None and any(
            altitude >= lgs_altitude_los for altitude in altitude_values
        ):
            raise ValueError("lgs_altitude_los must exceed every layer altitude")
        if seeds is None:
            seed_values = (None,) * n_layers
        else:
            seed_values = tuple(seeds)
            if len(seed_values) != n_layers:
                raise ValueError("seeds must contain one value per layer")
        if tau_boil is None:
            tau_values = (float("inf"),) * n_layers
        elif np.ndim(tau_boil) == 0:
            tau_values = (float(tau_boil),) * n_layers
        else:
            tau_values = tuple(float(value) for value in tau_boil)
            if len(tau_values) != n_layers:
                raise ValueError("tau_boil must be scalar or one value per layer")
        if any(np.isnan(value) or value <= 0 for value in tau_values):
            raise ValueError("tau_boil values must be positive")
        return cls(
            grid=grid,
            layer_r0=r0_values,
            layer_L0=L0_values,
            layer_wind=wind_values,
            layer_altitude_los=altitude_values,
            field_of_view_pix=fov_values,
            stencil_rows=int(stencil_rows),
            interp=interp,
            lgs_altitude_los=(
                None if lgs_altitude_los is None else float(lgs_altitude_los)
            ),
            seeds=seed_values,
            tau_boil=tau_values,
            boil_seed=boil_seed,
        )

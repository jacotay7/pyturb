"""Layered atmosphere: many turbulent layers summed into a pupil OPD.

:class:`Atmosphere` is the high-level entry point. It takes a turbulence
profile (a list of :class:`pyturb.Layer`, or a named one), a total Fried
parameter or seeing, and a telescope geometry, and produces optical path
difference (OPD) frames — the standard input to an adaptive-optics simulation.

Two evolution modes share one physical model:

- :meth:`sample` draws statistically independent integrated OPDs (Monte-Carlo
  ensembles: PSF statistics, error budgets, training data).
- :meth:`frames` / :meth:`opd` evolve the layers under frozen-flow (Taylor)
  wind for closed-loop temporal simulation, with sub-pixel, arbitrary-direction
  motion via the spectral engine :class:`pyturb.flow.FourierFlowScreen`.

OPD is returned in **metres** and is achromatic (a path length); pass
``wavelength=`` to any output method to get phase in radians at that
wavelength instead.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import profiles as _profiles
from .backend import get_array_module
from .extrude import ExtrudedAtmosphere
from .flow import FourierFlowScreen
from .fourier import PhaseScreen
from .profiles import Layer
from .utils import (
    air_refractivity,
    r0_at_wavelength,
    r0_from_seeing,
    seeing_from_r0,
    water_vapour_refractivity,
)

__all__ = ["Atmosphere", "PeriodicWrapWarning"]

_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


class PeriodicWrapWarning(UserWarning):
    """The default ``engine="spectral"`` screen has wrapped (repeated).

    The spectral engine is periodic with period ``n_screen * pixel_scale``
    metres of wind travel per layer. Once a layer's cumulative travel passes
    that period, its "new" turbulence is the same realisation seen before,
    biasing any temporal statistic computed over the run (PSDs, long
    integrations, closed-loop error budgets). Silence with
    ``warnings.filterwarnings("ignore", category=pyturb.PeriodicWrapWarning)``
    if this is intentional (e.g. deliberately studying the periodic case), or
    switch to ``engine="extrude"`` for genuinely non-periodic frozen flow.
    """


class _LayerState:
    """Per-layer runtime: independent-draw generator + frozen-flow screen."""

    __slots__ = ("generator", "flow", "vx", "vy", "altitude_los")

    def __init__(
        self,
        generator: PhaseScreen,
        flow: Optional[FourierFlowScreen],
        vx: float,
        vy: float,
        altitude_los: float,
    ):
        self.generator = generator
        self.flow = flow
        self.vx = vx
        self.vy = vy
        self.altitude_los = altitude_los


class Atmosphere:
    """A multi-layer turbulent atmosphere producing pupil OPD.

    Parameters
    ----------
    layers : sequence of Layer
        The turbulence profile. ``cn2_fraction`` values are normalised to
        sum to 1 internally.
    r0 : float, optional
        Total Fried parameter [m] at ``wavelength`` and *at zenith*. Give
        either ``r0`` or ``seeing``.
    seeing : float, optional
        Total seeing FWHM [arcsec] at ``wavelength`` and at zenith. Converted
        to ``r0`` via the Kolmogorov relation.
    wavelength : float, optional
        Reference wavelength [m] at which ``r0``/``seeing`` are defined.
        Default 500 nm.
    zenith_angle : float, optional
        Zenith angle [deg]. Scales ``r0`` (``cos^{3/5}``) and layer ranges
        (``sec``) for the line of sight. Default 0.
    diameter : float, optional
        Pupil diameter [m]; with ``n`` this sets ``pixel_scale = diameter/n``.
        Default 8 m.
    n : int, optional
        Pupil sampling in pixels across the diameter. Default 512.
    L0 : float, optional
        Override the outer scale [m] for every layer. Default: use each
        layer's own ``L0``.
    power_law : float, optional
        Power-law index of the phase PSD, ``PSD ~ f^{-power_law}``. Default
        ``11/3`` (Kolmogorov/von Karman). Other values give non-Kolmogorov
        turbulence; ``r0`` then acts as an amplitude knob rather than the
        strict Fried parameter. Applies to :meth:`sample` and to
        ``engine="spectral"`` :meth:`frames`/:meth:`opd`. Requires
        ``engine="spectral"``: the extruder's row recurrence is a closed-form
        conditional distribution derived specifically for ``power_law=11/3``.
    inner_scale : float, optional
        Inner scale [m]. ``0`` (default) disables it; a positive value rolls
        off the PSD above the corresponding spatial frequency (see
        :class:`pyturb.PhaseScreen`). Same engine restriction as
        ``power_law``.
    subharmonics : int, optional
        Subharmonic levels for low-frequency correction. Default 8.
    field_of_view : float, optional
        Radius of the science/guide-star field [arcsec]. The generated screens
        are oversized so that off-axis footprints out to this radius sample
        genuinely different (non-wrapped) turbulence — set this whenever you
        use ``directions`` for anisoplanatism/tomography. Default 0 (screens
        exactly the pupil; fastest). Larger values cost a larger per-frame FFT.
    tau_boil : float or sequence, optional
        Boiling (temporal decorrelation) time constant [s] of each layer's
        *outer-scale* structure, scalar or one per layer. ``None`` (default)
        is pure frozen flow. When set, every mode relaxes via an AR(1) process
        toward fresh noise with retention ``exp(-dt/tau(f))`` while keeping
        its spatial statistics; finer spatial structure boils faster than
        ``tau_boil`` per Kolmogorov eddy-turnover scaling,
        ``tau(f) = tau_boil * (f/f_ref)^(-2/3)`` for ``f >= f_ref = 1/L0``
        (the grid fundamental frequency for Kolmogorov ``L0=inf``), clamped to
        ``tau_boil`` below ``f_ref``. Active only while stepping with
        :meth:`frames`. Requires ``engine="spectral"``.
    engine : {"spectral", "extrude"}, optional
        Frozen-flow engine for :meth:`frames`/:meth:`opd`. ``"spectral"``
        (default) is the fastest: an exact sub-pixel shift-theorem translation,
        all layers batched into one FFT — but the screen is **periodic** (it
        repeats after ``n*pixel_scale`` of travel). ``"extrude"`` is the
        Assémat–Wilson row extruder in a wind-aligned frame with rotated
        sub-pixel sampling: **unbounded and non-periodic**, the right choice for
        long closed-loop runs, at the cost of per-layer sampling instead of one
        batched FFT. Its sub-pixel interpolation is a position-dependent
        low-pass filter (exact at integer pixel travel, most attenuating at
        half-pixel travel), so its finest-scale (1-2 px) structure function
        deviates from theory by 5-15% and oscillates with the sub-pixel travel
        phase; ``"spectral"`` has no such artifact. Prefer ``"spectral"`` when
        fine-scale (near-Nyquist) fidelity matters more than non-periodicity.
        :meth:`sample` (Monte-Carlo) is unaffected by this choice.
    interp : {"cubic", "linear"}, optional
        Sub-pixel interpolation kernel for ``engine="extrude"``. Default cubic.
    lgs_altitude : float, optional
        Altitude [m] of a laser guide star (e.g. ``90e3`` for sodium). When
        set, each layer's footprint is magnified by ``(1 - h/lgs_altitude)`` —
        the **cone effect** (focal anisoplanatism) that a finite-range beacon
        senses. ``None`` (default) is a natural-guide-star / science source at
        infinity. Requires ``engine="extrude"`` (the cone needs per-layer
        resampling, which the batched spectral engine cannot do).
    dispersion : {None, "edlen", "ciddor"}, optional
        Chromatic model for the OPD. ``None`` (default) treats the turbulence
        OPD as perfectly achromatic (phase at ``wavelength`` is just
        ``2 pi * OPD / wavelength``). ``"edlen"`` additionally scales the OPD by
        the dry-air refractivity ratio ``(n(wavelength)-1)/(n(lambda_ref)-1)``
        before converting to phase — the small (~1-2 % visible-to-NIR)
        chromatic term that matters for high-contrast, astrometry and LGS error
        budgets. ``"ciddor"`` splits that scaling into a dry-air part and a
        water-vapour part weighted by ``wet_fraction`` (the "wet–dry" problem):
        because water vapour disperses differently from dry air, a dry-only
        correction is wrong in the mid-IR and for interferometry where the wet
        turbulence dominates. ``"ciddor"`` with ``wet_fraction=0`` is identical
        to ``"edlen"``. Only affects output methods called with an explicit
        ``wavelength``; the metre-valued OPD is unchanged.
    wet_fraction : float, optional
        Fraction (0–1) of the reference-wavelength turbulent refractivity
        carried by water vapour rather than dry air, used only by
        ``dispersion="ciddor"``. ``0`` (default) is pure dry air; typical
        near-surface visible/NIR values are small, but the wet term grows
        important in the thermal IR. Must be ``0`` unless
        ``dispersion="ciddor"``.
    device : str, optional
        ``"cpu"`` (default) or ``"gpu"``.
    dtype : str, optional
        ``"float32"`` (default) or ``"float64"``.
    seed : int, optional
        Master seed. Per-layer streams are spawned from it so results are
        reproducible and independent of layer count.

    Feature compatibility
    ----------------------
    ``engine="spectral"`` (default) and ``engine="extrude"`` are not two
    interchangeable ways to get the same features faster/slower -- each
    supports a different subset:

    - ``tau_boil`` (boiling), non-Kolmogorov ``power_law``/``inner_scale``,
      and Kolmogorov ``L0=inf`` all require ``engine="spectral"``: they need
      either discrete Fourier modes to apply a per-mode operation to, or a
      closed-form covariance that only exists for the standard von Karman
      case. None of these three can be combined with ``lgs_altitude`` or with
      genuinely non-periodic (``engine="extrude"``) evolution.
    - ``lgs_altitude`` (the LGS cone effect) requires ``engine="extrude"``:
      the cone shrinks each layer's footprint by a per-layer factor, which
      needs the extruder's per-layer resampling; the batched spectral engine
      cannot do it. It therefore cannot combine with ``tau_boil`` or
      non-Kolmogorov statistics, and runs at the extruder's throughput, not
      the spectral engine's (see ``benchmarks/RESULTS.md``).
    - ``directions`` (off-axis/tomography) works with both engines, but every
      requested direction must lie within the declared ``field_of_view`` (a
      ``ValueError`` is raised otherwise, in both engines) -- construct the
      ``Atmosphere`` with a ``field_of_view`` covering every direction you
      plan to request.

    Examples
    --------
    >>> import pyturb
    >>> atm = pyturb.Atmosphere.from_profile("two-layer", seeing=0.8,
    ...                                      diameter=8.0, n=256, seed=1)
    >>> opd = atm.opd()                    # (256, 256) OPD [m], t = 0
    >>> for t, frame in atm.frames(dt=1e-3, steps=10):
    ...     pass                           # closed-loop OPD [m]
    >>> ensemble = atm.sample(16)          # (16, 256, 256) independent OPDs
    """

    def __init__(
        self,
        layers: Sequence[Layer],
        r0: Optional[float] = None,
        seeing: Optional[float] = None,
        wavelength: float = 500e-9,
        zenith_angle: float = 0.0,
        diameter: float = 8.0,
        n: int = 512,
        L0: Optional[float] = None,
        power_law: float = 11.0 / 3.0,
        inner_scale: float = 0.0,
        subharmonics: int = 8,
        field_of_view: float = 0.0,
        tau_boil: Union[float, Sequence[float], None] = None,
        engine: str = "spectral",
        interp: str = "cubic",
        lgs_altitude: Optional[float] = None,
        dispersion: Optional[str] = None,
        wet_fraction: float = 0.0,
        device: str = "cpu",
        dtype: str = "float32",
        seed: Optional[int] = None,
    ):
        layers = list(layers)
        if not layers:
            raise ValueError("at least one layer is required")
        if (r0 is None) == (seeing is None):
            raise ValueError("give exactly one of r0 or seeing")
        if not 0.0 <= zenith_angle < 90.0:
            raise ValueError("zenith_angle must be in [0, 90) degrees")
        if diameter <= 0 or n < 2:
            raise ValueError("diameter must be positive and n >= 2")
        if field_of_view < 0:
            raise ValueError("field_of_view must be >= 0 arcsec")
        if engine not in ("spectral", "extrude"):
            raise ValueError("engine must be 'spectral' or 'extrude'")
        if power_law <= 2.0:
            raise ValueError("power_law must be > 2 (Kolmogorov is 11/3)")
        if inner_scale < 0:
            raise ValueError("inner_scale must be >= 0 (0 disables it)")
        if engine == "extrude":
            if power_law != 11.0 / 3.0:
                raise ValueError(
                    "non-Kolmogorov power_law requires engine='spectral': the "
                    "extruder's row-to-row recurrence is a closed-form "
                    "conditional distribution derived specifically for the "
                    "von Karman covariance (power_law=11/3); generalizing it "
                    "to other exponents needs a different closed form, not "
                    "just a different PSD, so it is not offered here. "
                    "power_law is available for sample() and "
                    "engine='spectral' frames()/opd()."
                )
            if inner_scale > 0:
                raise ValueError(
                    "inner_scale requires engine='spectral': the extruder's "
                    "recurrence has no inner-scale term in its closed-form "
                    "covariance. inner_scale is available for sample() and "
                    "engine='spectral' frames()/opd()."
                )
        if engine == "extrude" and tau_boil is not None:
            raise ValueError(
                "boiling (tau_boil) requires engine='spectral': the "
                "extruder has no discrete 'modes' to apply a per-mode "
                "retention coefficient to (see tau_boil's docstring for the "
                "scale-dependent model) -- only a ring buffer of rows."
            )
        if lgs_altitude is not None:
            if engine != "extrude":
                raise ValueError("lgs_altitude (cone effect) requires engine='extrude'")
            if lgs_altitude <= 0:
                raise ValueError("lgs_altitude must be positive [m]")
        if dispersion not in (None, "edlen", "ciddor"):
            raise ValueError("dispersion must be None, 'edlen', or 'ciddor'")
        if not 0.0 <= wet_fraction <= 1.0:
            raise ValueError("wet_fraction must be in [0, 1]")
        if wet_fraction > 0.0 and dispersion != "ciddor":
            raise ValueError(
                "wet_fraction > 0 requires dispersion='ciddor' (the wet/dry "
                "chromatic split): dispersion='edlen' models dry air only and "
                "dispersion=None is achromatic, so neither has a water-vapour "
                "term to weight."
            )
        self.engine = engine
        self.interp = interp
        self.lgs_altitude = lgs_altitude
        self.dispersion = dispersion
        self.wet_fraction = float(wet_fraction)

        self.wavelength = float(wavelength)
        if seeing is not None:
            r0 = r0_from_seeing(seeing, wavelength)
        self.r0_zenith = float(r0)
        self.zenith_angle = float(zenith_angle)
        self.diameter = float(diameter)
        self.n = int(n)
        self.subharmonics = int(subharmonics)
        self.power_law = float(power_law)
        self.inner_scale = float(inner_scale)
        self.field_of_view = float(field_of_view)
        self.device = device
        self.dtype = dtype
        self.pixel_scale = self.diameter / self.n

        cos_z = np.cos(np.deg2rad(self.zenith_angle))
        self.airmass = 1.0 / cos_z
        # Line-of-sight r0 shrinks with airmass; ranges stretch with sec(z).
        self.r0_los = self.r0_zenith * cos_z ** (3.0 / 5.0)

        # Normalise Cn2 fractions and store a copy of the (possibly L0-overridden)
        # profile for reporting and integrated-quantity calculations.
        frac = _profiles._fractions(layers)
        self.layers: List[Layer] = []
        for layer, f in zip(layers, frac):
            self.layers.append(
                Layer(
                    altitude=layer.altitude,
                    cn2_fraction=float(f),
                    wind_speed=layer.wind_speed,
                    wind_direction=layer.wind_direction,
                    L0=layer.L0 if L0 is None else float(L0),
                )
            )
        if engine == "extrude" and any(not np.isfinite(ly.L0) for ly in self.layers):
            raise ValueError(
                "engine='extrude' requires a finite outer scale L0 for every "
                "layer: the extruder's row recurrence conditions on the von "
                "Karman phase covariance, which is only well-defined (finite "
                "variance) for finite L0. Kolmogorov (L0=inf) is only "
                "available for engine='spectral' or sample()."
            )

        # Oversize the generated screens so off-axis footprints (up to
        # field_of_view radius) stay inside the screen instead of wrapping.
        # The highest layer needs the most margin; use one uniform size so the
        # per-frame FFT stays a single batched call.
        max_alt_los = max(layer.altitude for layer in self.layers) * self.airmass
        margin_m = max_alt_los * np.tan(self.field_of_view * _ARCSEC_TO_RAD)
        margin_pix = int(np.ceil(margin_m / self.pixel_scale))
        self.margin_pix = margin_pix
        self.n_screen = self.n + 2 * margin_pix
        self._crop = slice(margin_pix, margin_pix + self.n)
        # The spectral engine's screen is periodic with this many metres of
        # wind travel; used to warn once a run's cumulative travel wraps it.
        self._screen_period_m = self.n_screen * self.pixel_scale
        self._wrap_warned = False

        # Per-layer boiling time constants (s); inf/None means frozen flow.
        if tau_boil is None:
            tau = np.full(len(self.layers), np.inf)
        else:
            tau = np.broadcast_to(
                np.asarray(tau_boil, dtype=np.float64), (len(self.layers),)
            ).astype(np.float64)
        if np.any(tau <= 0):
            raise ValueError("tau_boil must be positive (or None for frozen flow)")
        self.tau_boil = tau

        self.xp = get_array_module(device)
        master = np.random.SeedSequence(seed)
        seeds = master.spawn(len(self.layers))
        self._boil_rng = self.xp.random.default_rng(
            int(master.spawn(1)[0].generate_state(1)[0])
        )
        self._layers: List[_LayerState] = []
        ext_r0, ext_L0, ext_wind, ext_alt, ext_seeds = [], [], [], [], []
        for layer, child in zip(self.layers, seeds):
            # Per-layer line-of-sight r0: r0_i^{-5/3} = f_i * r0_los^{-5/3}.
            r0_i = self.r0_los * layer.cn2_fraction ** (-3.0 / 5.0)
            gen_seed, flow_seed, ext_seed = child.spawn(3)
            generator = PhaseScreen(
                n=self.n_screen,
                pixel_scale=self.pixel_scale,
                r0=r0_i,
                L0=layer.L0,
                subharmonics=self.subharmonics,
                power_law=self.power_law,
                inner_scale=self.inner_scale,
                seed=int(gen_seed.generate_state(1)[0]),
                device=device,
                dtype=dtype,
            )
            # The spectral flow screen is only needed for engine="spectral".
            flow = (
                FourierFlowScreen(generator, seed=int(flow_seed.generate_state(1)[0]))
                if self.engine == "spectral"
                else None
            )
            vx, vy = layer.wind_vector
            altitude_los = layer.altitude * self.airmass
            self._layers.append(
                _LayerState(
                    generator=generator,
                    flow=flow,
                    vx=vx,
                    vy=vy,
                    altitude_los=altitude_los,
                )
            )
            ext_r0.append(r0_i)
            ext_L0.append(layer.L0)
            ext_wind.append((vx, vy))
            ext_alt.append(altitude_los)
            ext_seeds.append(int(ext_seed.generate_state(1)[0]))

        self._t = 0.0
        if self.engine == "spectral":
            self._build_batched()
        else:
            # Off-axis footprints out to field_of_view need the buffer wider by
            # that perpendicular travel; margin_pix already encodes it.
            self._ext_kwargs = dict(
                n=self.n,
                pixel_scale=self.pixel_scale,
                layer_r0=ext_r0,
                layer_L0=ext_L0,
                layer_wind=ext_wind,
                layer_altitude_los=ext_alt,
                field_of_view_pix=float(self.margin_pix),
                interp=self.interp,
                lgs_altitude_los=(None if lgs_altitude is None
                                  else float(lgs_altitude) * self.airmass),
                device=device,
                dtype=dtype,
                seeds=ext_seeds,
            )
            self._ext = ExtrudedAtmosphere(**self._ext_kwargs)

    def _build_batched(self):
        """Stack every layer's spectrum so a frame is one batched FFT.

        The hot loop (``_integrate``) is otherwise launch-latency bound on the
        GPU: one ``ifft2`` and a handful of small matmuls per layer. Stacking
        the layers into a leading axis turns each frame into a single
        ``(L, n, n)`` inverse FFT plus one matmul per subharmonic level.
        """
        xp = self.xp
        flows = [state.flow for state in self._layers]
        self._fft = self._layers[0].generator._fft
        self.dtype_out = self._layers[0].generator.dtype
        self._cdtype = xp.dtype(self._layers[0].generator._cdtype)
        self._spectra = xp.stack([flow._spectrum for flow in flows])  # (L,n,n)
        # PSD amplitude per layer, for boiling (AR(1) noise injection).
        self._amplitudes = xp.stack(
            [state.generator._amplitude for state in self._layers]
        )
        self._grid_f = self._layers[0].generator._f  # (n,) device
        self._vx = np.array([s.vx for s in self._layers], dtype=np.float64)
        self._vy = np.array([s.vy for s in self._layers], dtype=np.float64)
        self._alt = np.array([s.altitude_los for s in self._layers], dtype=np.float64)
        # Subharmonic modes share basis/frequencies across layers (same grid);
        # only the coefficients (and per-layer amplitude) differ, so stack those.
        template = self._layers[0].generator
        self._sh_batched = []
        for level, (_amp, basis) in enumerate(template._sh_bases):
            coeffs = xp.stack([flow._sh_coeffs[level] for flow in flows])  # (L,3,3)
            amps = xp.stack(
                [state.generator._sh_bases[level][0] for state in self._layers]
            )  # (L,3,3)
            self._sh_batched.append((coeffs, amps, basis, template._sh_freqs[level]))
        self._boil_main_tau, self._boil_sh_tau = self._build_boil_tau_maps()

    def _build_boil_tau_maps(self):
        """Per-mode boiling time constants (Kolmogorov eddy-turnover scaling).

        Eddy lifetime scales with eddy size, ``tau(l) ~ epsilon^{-1/3} l^{2/3}``
        (Kolmogorov 1941), i.e. ``tau(f) ~ f^{-2/3}``: fine structure
        decorrelates faster than large-scale structure. ``tau_boil`` is taken
        to set the decorrelation time of the outer-scale structure (frequency
        ``f_ref = 1/L0``, or the grid fundamental ``1/(n_screen*pixel_scale)``
        for Kolmogorov ``L0=inf``, which has no outer scale); every other mode
        gets ``tau(f) = tau_boil * (f/f_ref)^(-2/3)`` for ``f >= f_ref``,
        clamped to ``tau_boil`` below it (the inertial-range scaling has
        nothing to say about structure larger than the outer scale).

        References
        ----------
        Kolmogorov (1941); eddy turnover time scaling as used e.g. in
        Poyneer, van Dam & Veran (2009), JOSA A 26, 833 and Guesalaga et al.
        (2014), MNRAS 441, 1925 to characterise non-frozen-flow decorrelation.
        """
        xp = self.xp
        rdtype = self._spectra.real.dtype
        L0s = np.array([s.generator.L0 for s in self._layers], dtype=np.float64)
        df = 1.0 / (self.n_screen * self.pixel_scale)
        f_ref = np.where(np.isfinite(L0s), 1.0 / np.where(L0s > 0, L0s, np.inf), df)
        tau_boil_dev = xp.asarray(self.tau_boil, dtype=rdtype)  # (L,)
        f_ref_dev = xp.asarray(f_ref, dtype=rdtype)  # (L,)

        f = self._grid_f
        fr = xp.hypot(f[:, None], f[None, :]).astype(rdtype)  # (ns, ns)
        ratio = xp.clip(fr[None, :, :] / f_ref_dev[:, None, None], 1.0, None)
        main_tau = tau_boil_dev[:, None, None] * ratio ** (-2.0 / 3.0)

        sh_tau = []
        for _coeffs, _amps, _basis, fp in self._sh_batched:
            fr_sh = xp.hypot(fp[:, None], fp[None, :]).astype(rdtype)  # (3, 3)
            ratio_sh = xp.clip(fr_sh[None, :, :] / f_ref_dev[:, None, None], 1.0, None)
            sh_tau.append(tau_boil_dev[:, None, None] * ratio_sh ** (-2.0 / 3.0))
        return main_tau, sh_tau

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_profile(cls, name: str, **kwargs) -> "Atmosphere":
        """Build from a named profile (see :func:`pyturb.list_profiles`).

        Any :class:`Atmosphere` keyword may be passed, e.g.::

            Atmosphere.from_profile("paranal-median", seeing=0.8,
                                    zenith_angle=30, diameter=8, n=512)
        """
        return cls(_profiles.get_profile(name), **kwargs)

    # ------------------------------------------------------------------
    # integrated quantities
    # ------------------------------------------------------------------
    def r0_at(self, wavelength: float) -> float:
        """Line-of-sight total Fried parameter [m] at ``wavelength``."""
        return r0_at_wavelength(self.r0_los, self.wavelength, wavelength)

    @property
    def r0(self) -> float:
        """Line-of-sight total Fried parameter [m] at the reference wavelength."""
        return self.r0_los

    @property
    def seeing(self) -> float:
        """Line-of-sight seeing FWHM [arcsec] at the reference wavelength."""
        return seeing_from_r0(self.r0_los, self.wavelength)

    @property
    def theta0(self) -> float:
        """Isoplanatic angle [arcsec] at the reference wavelength."""
        theta0 = _profiles.isoplanatic_angle(self._los_layers(), self.r0_los)
        return theta0 / _ARCSEC_TO_RAD

    @property
    def tau0(self) -> float:
        """Coherence time [s] at the reference wavelength."""
        return _profiles.coherence_time(self.layers, self.r0_los)

    @property
    def greenwood_frequency(self) -> float:
        """Greenwood frequency [Hz] at the reference wavelength."""
        return _profiles.greenwood_frequency(self.layers, self.r0_los)

    @property
    def time_to_wrap(self) -> float:
        """Seconds of on-axis wind travel before the fastest layer wraps.

        Only meaningful for ``engine="spectral"``, which is periodic with
        period ``n_screen * pixel_scale`` metres of travel per layer; a run
        (via :meth:`frames` or :meth:`opd`) whose duration exceeds this
        re-samples the same screen realisation instead of independent
        turbulence for at least one layer. Returns ``inf`` for
        ``engine="extrude"`` (non-periodic) or if every layer has zero wind.
        """
        if self.engine != "spectral":
            return float("inf")
        speeds = np.array([layer.wind_speed for layer in self.layers])
        with np.errstate(divide="ignore"):
            periods = self._screen_period_m / speeds
        finite = periods[np.isfinite(periods)]
        return float(finite.min()) if finite.size else float("inf")

    def _check_wrap(self, disp_x: np.ndarray, disp_y: np.ndarray) -> None:
        """Warn once (per instance) the first time a spectral layer wraps."""
        disp = np.hypot(np.asarray(disp_x), np.asarray(disp_y))
        wrapped = disp > self._screen_period_m
        if not np.any(wrapped):
            return
        self._wrap_warned = True
        idx = int(np.argmax(wrapped))
        layer = self.layers[idx]
        warnings.warn(
            f"engine='spectral' layer {idx} (altitude={layer.altitude:.0f} m, "
            f"wind_speed={layer.wind_speed:.1f} m/s) has wrapped: its "
            f"cumulative travel ({disp[idx]:.2f} m) exceeds the screen's "
            f"period ({self._screen_period_m:.2f} m). Its 'new' turbulence "
            "is a repeat of previous turbulence, biasing temporal statistics "
            "over this run. Use engine='extrude' for non-periodic frozen "
            "flow, or keep run duration below `atm.time_to_wrap` seconds.",
            PeriodicWrapWarning,
            stacklevel=3,
        )

    def _los_layers(self) -> List[Layer]:
        # Layers with altitudes projected to the line of sight (for theta0).
        return [
            Layer(layer.altitude * self.airmass, layer.cn2_fraction,
                  layer.wind_speed, layer.wind_direction, layer.L0)
            for layer in self.layers
        ]

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def _to_opd(self, phase: Any, wavelength: Optional[float]) -> Any:
        """Convert reference-wavelength phase [rad] to the requested output.

        Returns achromatic OPD [m] when ``wavelength`` is ``None``; otherwise
        phase [rad] at ``wavelength``. With ``dispersion="edlen"`` the path
        length is scaled by the dry-air refractivity ratio before the phase
        conversion; with ``dispersion="ciddor"`` by a ``wet_fraction``-weighted
        blend of the dry-air and water-vapour dispersion ratios (see the
        ``dispersion`` constructor argument).
        """
        opd = phase * (self.wavelength / (2.0 * np.pi))
        if wavelength is None:
            return opd
        opd = opd * self._chromatic_scale(float(wavelength))
        return opd * (2.0 * np.pi / float(wavelength))

    def _chromatic_scale(self, wavelength: float) -> float:
        """OPD scale factor from the reference wavelength to ``wavelength``.

        ``1`` for ``dispersion=None`` (achromatic); the dry-air refractivity
        ratio for ``"edlen"``; and a ``wet_fraction``-weighted blend of the
        dry-air and water-vapour dispersion ratios for ``"ciddor"``. The wet
        term uses :func:`pyturb.water_vapour_refractivity`; both ratios are 1 at
        the reference wavelength, so the blend is too.
        """
        if self.dispersion is None:
            return 1.0
        dry = air_refractivity(wavelength) / air_refractivity(self.wavelength)
        if self.dispersion == "edlen":
            return dry
        wet = (
            water_vapour_refractivity(wavelength)
            / water_vapour_refractivity(self.wavelength)
        )
        w = self.wet_fraction
        return (1.0 - w) * dry + w * wet

    def opd(
        self,
        t: float = 0.0,
        directions: Optional[Sequence[Tuple[float, float]]] = None,
        wavelength: Optional[float] = None,
    ) -> Any:
        """OPD at time ``t`` [s] under frozen flow.

        Parameters
        ----------
        t : float
            Time since the start of the simulation [s]. Taylor frozen flow:
            each layer's phase evolves as ``phi(x, t) = phi_0(x + wind_vector
            * t)`` — a fixed pupil point sees the turbulence that was
            ``wind_vector * t`` metres further along the wind direction at
            ``t=0`` (the pattern is carried past the aperture by the wind,
            not translated bodily along ``+wind_vector`` in pupil
            coordinates).
        directions : sequence of (thx, thy), optional
            Off-axis directions [arcsec] from the on-axis line of sight. Each
            component of each direction (not just the radius) must lie within
            the ``field_of_view`` declared at construction, or the screens are
            not guaranteed to be oversized enough and the request raises
            ``ValueError`` instead of silently sampling stale/wrapped data.
            Each layer's footprint is shifted by ``altitude_los * tan(theta)``
            for anisoplanatism / tomography studies. If given, the result has
            a leading axis of length ``len(directions)``.
        wavelength : float, optional
            If given, return phase [rad] at this wavelength; otherwise OPD [m].
        """
        if directions is None:
            phase = self._phase(float(t), 0.0, 0.0)
            return self._to_opd(phase, wavelength)

        xp = self.xp
        out = []
        for thx, thy in directions:
            if abs(thx) > self.field_of_view or abs(thy) > self.field_of_view:
                raise ValueError(
                    f"direction ({thx}, {thy}) arcsec exceeds the declared "
                    f"field_of_view={self.field_of_view} arcsec; construct "
                    "the Atmosphere with a field_of_view covering every "
                    "direction you plan to request."
                )
            ox = np.tan(thx * _ARCSEC_TO_RAD)
            oy = np.tan(thy * _ARCSEC_TO_RAD)
            out.append(self._phase(float(t), ox, oy))
        stacked = xp.stack(out)
        return self._to_opd(stacked, wavelength)

    def _phase(self, t: float, ox: float, oy: float) -> Any:
        """Reference-wavelength pupil phase at time ``t`` toward slope (ox, oy).

        Dispatches to the periodic spectral engine or the non-periodic extruder;
        ``ox``/``oy`` are direction tangents (``tan(theta)``).
        """
        if self.engine == "spectral":
            return self._integrate(t, ox, oy)
        self._ext.set_time(t)
        return self._ext.integrate(ox, oy)

    def _integrate(self, t: float, ox: float, oy: float) -> Any:
        """Sum all layers at time ``t`` with per-layer angular offset slopes.

        Batched over layers: one ``(L, n, n)`` inverse FFT plus one matmul per
        subharmonic level, then a sum over the layer axis.
        """
        xp = self.xp
        cdtype = self._cdtype
        ns = self.n_screen
        # Per-layer displacement [m] along each axis (host-side, L is small).
        disp_x = self._vx * t + self._alt * ox
        disp_y = self._vy * t + self._alt * oy
        if not self._wrap_warned:
            self._check_wrap(disp_x, disp_y)
        sx = xp.asarray(disp_x, dtype=self._spectra.real.dtype)
        sy = xp.asarray(disp_y, dtype=self._spectra.real.dtype)
        f = self._grid_f
        # Separable shift-theorem phasors, shape (L, n_screen) each.
        phasor_x = xp.exp((2j * np.pi) * sx[:, None] * f[None, :]).astype(cdtype)
        phasor_y = xp.exp((2j * np.pi) * sy[:, None] * f[None, :]).astype(cdtype)
        spectra = self._spectra * phasor_x[:, :, None] * phasor_y[:, None, :]
        field = self._fft.ifft2(spectra, axes=(-2, -1)) * (ns * ns)
        total = field.real.sum(axis=0)

        if self._sh_batched:
            low = None
            for coeffs, _amps, basis, fp in self._sh_batched:
                px = xp.exp((2j * np.pi) * sx[:, None] * fp[None, :]).astype(cdtype)
                py = xp.exp((2j * np.pi) * sy[:, None] * fp[None, :]).astype(cdtype)
                shifted = coeffs * px[:, :, None] * py[:, None, :]  # (L,3,3)
                contribution = xp.matmul(basis.T, xp.matmul(shifted, basis))  # (L,ns,ns)
                summed = contribution.real.sum(axis=0)
                low = summed if low is None else low + summed
            low -= low.mean()
            total = total + low
        # Crop the central pupil region out of the (oversized) screen.
        total = total[self._crop, self._crop]
        return xp.ascontiguousarray(total.astype(self.dtype_out, copy=False))

    def _boil_step(self, dt: float) -> None:
        """Advance boiling by ``dt`` seconds: one AR(1) update per mode.

        Each Fourier mode relaxes toward a fresh draw of its stationary
        distribution with retention ``alpha = exp(-dt / tau(f))``, so the
        mode's temporal autocorrelation decays as ``exp(-dt / tau(f))`` while
        its spatial PSD (and hence r0) is preserved. ``tau(f)`` is scale
        dependent (see :meth:`_build_boil_tau_maps`): fine spatial structure
        boils faster than coarse structure, per Kolmogorov eddy-turnover
        scaling. Layers with infinite ``tau_boil`` are untouched (pure frozen
        flow).
        """
        xp = self.xp
        with np.errstate(divide="ignore"):
            alpha = np.where(
                np.isfinite(self.tau_boil), np.exp(-dt / self.tau_boil), 1.0
            )
        if np.all(alpha >= 1.0):  # every layer frozen — nothing to do
            return
        L = self._spectra.shape[0]
        a = xp.exp(-dt / self._boil_main_tau)
        b = xp.sqrt(xp.clip(1.0 - a * a, 0.0, None))
        noise = self._boil_rng.standard_normal(
            (2, L, self.n_screen, self.n_screen), dtype=self._spectra.real.dtype
        )
        fresh = (noise[0] + 1j * noise[1]) * self._amplitudes
        self._spectra = a * self._spectra + b * fresh.astype(self._cdtype)
        new_sh = []
        for (coeffs, amps, basis, fp), tau_sh in zip(self._sh_batched, self._boil_sh_tau):
            a_sh = xp.exp(-dt / tau_sh)
            b_sh = xp.sqrt(xp.clip(1.0 - a_sh * a_sh, 0.0, None))
            noise = self._boil_rng.standard_normal((2, L, 3, 3),
                                                   dtype=self._spectra.real.dtype)
            fresh = (noise[0] + 1j * noise[1]) * amps
            coeffs = a_sh * coeffs + b_sh * fresh.astype(self._cdtype)
            new_sh.append((coeffs, amps, basis, fp))
        self._sh_batched = new_sh

    def frames(
        self, dt: float, steps: int, wavelength: Optional[float] = None
    ) -> Iterator[Tuple[float, Any]]:
        """Yield ``(t, opd)`` for ``steps`` frames spaced by ``dt`` seconds.

        Advances the internal clock, so successive calls continue the wind.
        Use :meth:`reset` to return to ``t = 0``.

        Yields
        ------
        (float, ndarray)
            Simulation time [s] and the pupil OPD [m] (or phase [rad] if
            ``wavelength`` is given), shape ``(n, n)`` on the device.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        for _ in range(int(steps)):
            t = self._t
            phase = self._phase(t, 0.0, 0.0)
            yield t, self._to_opd(phase, wavelength)
            self._t += float(dt)
            if self.engine == "spectral":
                self._boil_step(float(dt))  # no-op unless tau_boil is finite

    def evolve(self, dt: float, wavelength: Optional[float] = None) -> Any:
        """Advance the wind by ``dt`` seconds and return the new pupil OPD.

        A single-step, in-seconds convenience over :meth:`frames` for callers
        driving their own loop (mirrors HCIPy's ``evolve_until``): each layer's
        frozen flow is stepped by its own ``wind_speed`` — you pass a time, not
        a pixel count. Advances the internal clock, so successive calls continue
        the wind; use :meth:`reset` to return to ``t = 0``.

        Repeated ``evolve(dt)`` calls reproduce the same sequence as
        ``frames(dt, ...)`` (from its second frame on): both apply one boiling
        step per ``dt`` when ``tau_boil`` is set.

        Parameters
        ----------
        dt : float
            Time step [s] (> 0).
        wavelength : float, optional
            If given, return phase [rad] at this wavelength; otherwise OPD [m].

        Returns
        -------
        ndarray
            The pupil OPD [m] (or phase [rad]) at the new time, shape
            ``(n, n)`` on the device.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        self._t += float(dt)
        if self.engine == "spectral":
            self._boil_step(float(dt))
        return self._to_opd(self._phase(self._t, 0.0, 0.0), wavelength)

    def sample(
        self, count: Optional[int] = None, wavelength: Optional[float] = None
    ) -> Any:
        """Draw statistically independent integrated OPDs.

        Each call produces fresh, uncorrelated realisations of the summed
        atmosphere (on-axis). Independent screens from every layer are added,
        so the ensemble structure function matches the total ``r0``.

        Parameters
        ----------
        count : int, optional
            Number of independent OPDs. If omitted a single ``(n, n)`` array
            is returned; otherwise the result is ``(count, n, n)``.
        wavelength : float, optional
            If given, return phase [rad] at this wavelength; otherwise OPD [m].
        """
        total = None
        for state in self._layers:
            screens = state.generator.generate(count)
            total = screens if total is None else total + screens
        # Crop the central pupil out of the (possibly oversized) screens.
        total = total[..., self._crop, self._crop]
        return self._to_opd(total, wavelength)

    def reset(self) -> "Atmosphere":
        """Reset the internal clock to ``t = 0``. Returns ``self``.

        For ``engine="extrude"`` the extruded layers are rebuilt from their
        seeds (wind travel is monotonic, so the run restarts identically).
        """
        self._t = 0.0
        self._wrap_warned = False
        if self.engine == "extrude":
            self._ext = ExtrudedAtmosphere(**self._ext_kwargs)
        return self

    @property
    def time(self) -> float:
        """Current internal clock [s] (advanced by :meth:`frames`)."""
        return self._t

    @property
    def metadata(self) -> Dict[str, Any]:
        """The parameters that define this atmosphere, for saving with output.

        A flat dict of scalars/strings suitable for :func:`pyturb.save`
        headers — geometry, line-of-sight ``r0``/``L0``, engine, and the
        integrated seeing/theta0/tau0 — so a saved OPD carries its provenance.
        """
        return {
            "units": "metres",
            "pixel_scale": self.pixel_scale,
            "diameter": self.diameter,
            "n": self.n,
            "r0": self.r0_los,
            "wavelength": self.wavelength,
            "seeing": self.seeing,
            "theta0": self.theta0,
            "tau0": self.tau0,
            "zenith_angle": self.zenith_angle,
            "n_layers": len(self.layers),
            "engine": self.engine,
            "dispersion": self.dispersion,
            "wet_fraction": self.wet_fraction,
        }

    def __repr__(self) -> str:
        return (
            f"Atmosphere(layers={len(self.layers)}, r0={self.r0_los:.3f} m, "
            f"seeing={self.seeing:.2f}\", theta0={self.theta0:.2f}\", "
            f"tau0={self.tau0 * 1e3:.1f} ms, n={self.n}, "
            f"diameter={self.diameter} m, device={self.device!r})"
        )

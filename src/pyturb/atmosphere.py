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

from . import _accel
from . import profiles as _profiles
from .backend import get_array_module
from .config import AtmosphereConfig
from .extrude import ExtrudedAtmosphere, _catmull_rom_weights
from .flow import FourierFlowScreen
from .fourier import PhaseScreen
from .infinite import _lanczos_weights
from .profiles import Layer
from .utils import (
    air_refractivity,
    r0_at_wavelength,
    seeing_from_r0,
    water_vapour_refractivity,
)

__all__ = ["Atmosphere", "PeriodicWrapWarning", "ExtrudeBoilingPerformanceWarning"]

_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)

# LGS cone zoom: above this batched working set (L * n * n_screen elements) the
# GPU's batched gather goes memory-bound and loses to the tight per-layer loop
# (measured crossover between 512² and 1024² at 9 layers); below it the batched
# path is ~3-8x faster by removing per-layer launch latency.
_LGS_BATCH_MAX_ELEMS = 4_000_000


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


class ExtrudeBoilingPerformanceWarning(UserWarning):
    """``tau_boil`` on ``engine="extrude"`` is markedly slower than ``"spectral"``.

    The spectral engine boils with one per-mode AR(1) update on its already-
    stored spectrum. The extruder has no such spectrum -- it works in real
    space -- so each boiling step must re-extrude an independent, fresh
    ``(rows, width)`` screen from scratch (seeding a few stencil rows, then
    running the same row recurrence forward) and blend it into the ring
    buffer. That extra extrusion, repeated every :meth:`Atmosphere.frames` /
    :meth:`Atmosphere.evolve` step, is the dominant cost: a boiling extrude
    frame is many times a frozen one, far more than the corresponding slowdown
    on the spectral engine. Prefer ``engine="spectral"`` for boiling-heavy runs
    unless genuine non-periodicity is required; silence with
    ``warnings.filterwarnings("ignore",
    category=pyturb.ExtrudeBoilingPerformanceWarning)`` if the cost is
    already accounted for.
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
    ) -> None:
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
        (``sec``) for the line of sight. Default 0. This is a **scalar
        airmass/range approximation**, not full slant-path geometry: the pupil
        grid and winds stay isotropic, so it does not apply the anisotropic
        pupil-to-tilted-layer coordinate transform a real slant path needs (a
        baseline in the zenith plane maps to a screen separation stretched by
        ``sec(z)`` versus the perpendicular one, making the structure function
        differ between axes by ``sec(z)^{5/3}`` — ~3.2x at 60 deg). Accurate as
        a seeing/range scaling near zenith; not a substitute for full slant-path
        geometry at large zenith angles.
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
        is pure frozen flow. When set, the layer relaxes via an AR(1) process
        toward fresh, statistically identical turbulence with retention
        ``exp(-dt/tau)`` while preserving its spatial statistics. Active only
        while stepping with :meth:`frames`/:meth:`evolve`. Works on **both**
        engines, but the two realise it differently:

        - ``engine="spectral"`` boils **per Fourier mode**, so finer spatial
          structure decorrelates faster than the outer scale per Kolmogorov
          eddy-turnover scaling, ``tau(f) = tau_boil * (f/f_ref)^(-2/3)`` for
          ``f >= f_ref = 1/L0`` (the grid fundamental for Kolmogorov
          ``L0=inf``), clamped to ``tau_boil`` below ``f_ref``.
        - ``engine="extrude"`` blends its ring buffer toward a fresh
          independently extruded screen -- a single-timescale AR(1) that
          decorrelates **all** spatial scales at ``tau_boil`` (real space has
          no per-mode handle). Staying non-periodic costs a modest deficit in
          the largest-scale power of the boiled screen, and re-extruding that
          fresh screen every step makes it markedly slower than spectral
          boiling (raises :class:`ExtrudeBoilingPerformanceWarning`); use
          ``"spectral"`` unless genuine non-periodicity is required.
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
        phase; ``"spectral"`` has no such artifact. (At larger separations the
        extruded field stays isotropic and on-theory to the method's usual
        few-percent accuracy — the row recurrence introduces no systematic
        large-scale along-vs-cross anisotropy.) Prefer ``"spectral"`` when
        fine-scale (near-Nyquist) fidelity matters more than non-periodicity.
        :meth:`sample` (Monte-Carlo) is unaffected by this choice.
    interp : {"cubic", "linear", "lanczos"}, optional
        Sub-pixel interpolation kernel for ``engine="extrude"``. ``"cubic"``
        (Catmull-Rom, default) is a good speed/quality balance; ``"lanczos"``
        (6-tap Lanczos-3) has a flatter sub-Nyquist response that reduces the
        extruder's finest-scale structure-function deficit and its travel-phase
        flicker (see the ``engine`` note), at the cost of more taps per readout;
        ``"linear"`` (2-tap) is the lowest fidelity. Note that ``"cubic"`` and
        ``"lanczos"`` run through a fused CUDA/Numba readout kernel while
        ``"linear"`` falls back to a generic tap-broadcast gather that
        materialises temporaries, so despite its lower tap count ``"linear"`` is
        typically **slower**, not faster, than the default ``"cubic"`` on both
        CPU and GPU; pick it for its low-pass character, not for speed.
    lgs_altitude : float, optional
        Altitude [m] of a laser guide star (e.g. ``90e3`` for sodium). When
        set, each layer's footprint is magnified by ``(1 - h/lgs_altitude)`` —
        the **cone effect** (focal anisoplanatism) that a finite-range beacon
        senses. ``None`` (default) is a natural-guide-star / science source at
        infinity. Works on **both** engines: the extruder samples its ring
        buffer on a magnified grid, and the spectral engine zoom-resamples each
        layer's screen about the pupil centre by the same factor (so on spectral
        the cone composes with ``tau_boil`` boiling). Must be greater than every
        layer altitude (a beacon at or below a layer is a degenerate cone).
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
        ``wavelength``; the metre-valued OPD is unchanged. Both models use the
        Edlén (1966) dry-air and Ciddor (1996) water-vapour *dispersion shapes*,
        validated against the independent Ciddor standard-air formula to
        < 0.05 % over 0.35-1.7 µm (the dry-air ratio stays within ~0.1 % to
        ~2.5 µm). They are scalar models with **no weather inputs**: the wet/dry
        density split is the user-supplied ``wet_fraction``, not a
        pressure/temperature/humidity computation, so treat the thermal-IR /
        interferometric wet-dominated regime as an order-of-magnitude estimate
        set by ``wet_fraction`` rather than a first-principles wet–dry model.
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

    - ``tau_boil`` (boiling) works on **both** engines, but with different
      spatial character: ``engine="spectral"`` boils per Fourier mode (finer
      structure faster, scale-resolved), while ``engine="extrude"`` blends its
      ring buffer toward fresh extruded turbulence at a single timescale across
      all scales, preserving non-periodicity (see ``tau_boil`` above).
    - non-Kolmogorov ``power_law``/``inner_scale`` and Kolmogorov ``L0=inf``
      still require ``engine="spectral"``: they need either discrete Fourier
      modes or a closed-form covariance that only exists for the standard von
      Karman case, so none can combine with non-periodic
      (``engine="extrude"``) evolution.
    - ``lgs_altitude`` (the LGS cone effect) works on **both** engines. On the
      extruder the cone shrinks each layer's footprint via the ring-buffer
      sampling grid; on the spectral engine each layer's screen is
      zoom-resampled about the pupil centre by the same factor. On either
      engine it composes with ``tau_boil`` boiling (the cone acts on readout
      geometry, boiling on the stored turbulence, so they are independent). The
      spectral cone falls back to a per-layer transform (each layer zooms
      differently), so it runs below the on-axis spectral throughput.
    - ``directions`` (off-axis/tomography) works with both engines, but every
      requested direction's radius ``sqrt(thx**2 + thy**2)`` must lie within the
      declared ``field_of_view`` (a ``ValueError`` is raised otherwise, in both
      engines) -- construct the ``Atmosphere`` with a ``field_of_view`` covering
      every direction you plan to request.

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
    ) -> None:
        config = AtmosphereConfig.create(
            layers, r0, seeing, wavelength, zenith_angle, diameter, n, L0,
            power_law, inner_scale, subharmonics, field_of_view, tau_boil,
            engine, interp, lgs_altitude, dispersion, wet_fraction, device,
            dtype, seed,
        )
        if config.engine == "extrude" and config.has_tau_boil:
            warnings.warn(
                "boiling (tau_boil) on engine='extrude' re-extrudes an "
                "independent fresh screen every step and is markedly slower "
                "than the spectral engine's per-mode AR(1) boiling -- see "
                "ExtrudeBoilingPerformanceWarning for why. Prefer "
                "engine='spectral' for boiling-heavy runs unless genuine "
                "non-periodicity is required.",
                ExtrudeBoilingPerformanceWarning,
                stacklevel=2,
            )
        self.config = config
        self.engine = config.engine
        self.interp = config.interp
        self.lgs_altitude = config.lgs_altitude
        self.dispersion = config.dispersion
        self.wet_fraction = config.wet_fraction
        self.wavelength = config.wavelength
        self.r0_zenith = config.r0_zenith
        self.zenith_angle = config.zenith_angle
        self.diameter = config.diameter
        self.n = config.grid.n
        self.subharmonics = config.subharmonics
        self.power_law = config.power_law
        self.inner_scale = config.inner_scale
        self.field_of_view = config.field_of_view
        self.device = config.grid.device
        self.dtype = config.grid.dtype.name
        self.pixel_scale = config.grid.pixel_scale

        cos_z = np.cos(np.deg2rad(self.zenith_angle))
        self.airmass = 1.0 / cos_z
        # Line-of-sight r0 shrinks with airmass; ranges stretch with sec(z).
        self.r0_los = self.r0_zenith * cos_z ** (3.0 / 5.0)

        self.layers = [layer.to_layer() for layer in config.layers]

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

        self.tau_boil = np.asarray(config.tau_boil, dtype=np.float64)

        self.xp = get_array_module(self.device)
        self.seed = config.seed
        # Set by from_profile() to the named profile; None for a direct build.
        self._profile_name: Optional[str] = None
        master = np.random.SeedSequence(self.seed)
        seeds = master.spawn(len(self.layers))
        self._boil_seed = int(master.spawn(1)[0].generate_state(1)[0])
        self._boil_rng = self.xp.random.default_rng(self._boil_seed)
        self._ext_boil_seed = int(master.spawn(1)[0].generate_state(1)[0])
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
                device=self.device,
                dtype=self.dtype,
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

        # sample() returns only the summed on-axis field, and independent
        # Gaussian screens sharing a PSD shape add: L layers with the same
        # L0/grid/power_law/inner_scale sum to one screen whose Fried parameter
        # satisfies r0_agg^{-5/3} = sum_i r0_i^{-5/3}. So sample() draws one
        # aggregate PhaseScreen per L0 group rather than one per layer --
        # distributionally identical, O(groups) FFTs per frame instead of
        # O(layers). Frozen flow / directions / LGS keep the per-layer
        # generators above. Singleton groups reuse the layer's own generator, so
        # a one-layer-per-L0 sample() stays bit-for-bit reproducible.
        groups: Dict[float, List[int]] = {}
        for i, ly in enumerate(self.layers):
            groups.setdefault(round(float(ly.L0), 9), []).append(i)
        self._sample_generators: List[PhaseScreen] = []
        for idxs, child in zip(groups.values(), master.spawn(len(groups))):
            if len(idxs) == 1:
                self._sample_generators.append(self._layers[idxs[0]].generator)
                continue
            r0_agg = sum(ext_r0[i] ** (-5.0 / 3.0) for i in idxs) ** (-3.0 / 5.0)
            self._sample_generators.append(
                PhaseScreen(
                    n=self.n_screen,
                    pixel_scale=self.pixel_scale,
                    r0=r0_agg,
                    L0=self.layers[idxs[0]].L0,
                    subharmonics=self.subharmonics,
                    power_law=self.power_law,
                    inner_scale=self.inner_scale,
                    seed=int(child.generate_state(1)[0]),
                    device=self.device,
                    dtype=self.dtype,
                )
            )

        # LGS cone effect in the spectral engine: each layer's screen is
        # zoom-sampled about the pupil centre by its magnification
        # ``1 - h/H_LGS`` at readout (the extruder handles its own cone
        # internally, so ``_lgs_mag`` stays None there). Sampling the *same*
        # realisation at magnified coordinates — not a differently-scaled draw
        # — is what makes it the focal-anisoplanatism (cone vs cylinder) error.
        # Validate the LGS cone on *both* engines: a beacon at or below a layer
        # gives a non-positive magnification, which collapses that layer's
        # footprint to a point (piston) instead of modelling focal
        # anisoplanatism. Only the spectral engine needs the precomputed
        # per-layer magnification array (``_lgs_mag``); the extruder derives its
        # own from ``lgs_altitude_los``, but must be rejected here just the same.
        self._lgs_mag: Optional[np.ndarray] = None
        if self.lgs_altitude is not None:
            lgs_los = self.lgs_altitude * self.airmass
            mag = 1.0 - np.array([s.altitude_los for s in self._layers]) / lgs_los
            if np.any(mag <= 0.0):
                raise ValueError(
                    f"lgs_altitude ({self.lgs_altitude:.0f} m) must exceed every "
                    "layer altitude (projected to the line of sight): a layer "
                    "at or above the beacon gives a degenerate (<= 0) cone "
                    "magnification, which would collapse that layer's footprint "
                    "to a point rather than model focal anisoplanatism. Rejected "
                    "on both engine='spectral' and engine='extrude'."
                )
            if self.engine == "spectral":
                self._lgs_mag = mag

        self._t = 0.0
        if self.engine == "spectral":
            self._build_batched()
        else:
            # Each layer's own off-axis reach scales with its own altitude
            # (only the highest layer needs the full field_of_view margin);
            # ExtrudedAtmosphere sizes its shared ring buffer to what layers
            # actually need rather than a blanket highest-altitude-for-everyone
            # margin (self.margin_pix, used above for the spectral crop).
            ext_fov_margin_pix = [
                alt * np.tan(self.field_of_view * _ARCSEC_TO_RAD) / self.pixel_scale
                for alt in ext_alt
            ]
            self._ext_kwargs = dict(
                n=self.n,
                pixel_scale=self.pixel_scale,
                layer_r0=ext_r0,
                layer_L0=ext_L0,
                layer_wind=ext_wind,
                layer_altitude_los=ext_alt,
                field_of_view_pix=ext_fov_margin_pix,
                interp=self.interp,
                lgs_altitude_los=(None if self.lgs_altitude is None
                                  else self.lgs_altitude * self.airmass),
                device=self.device,
                dtype=self.dtype,
                seeds=ext_seeds,
                tau_boil=(None if not config.has_tau_boil else list(self.tau_boil)),
                boil_seed=self._ext_boil_seed,
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
        # only the coefficients (and per-layer amplitude) differ. All levels
        # share the same 3x3 mode geometry, so every per-level array is stacked
        # along a leading level axis P: the whole low-frequency contribution
        # then evaluates as a handful of batched ops instead of a Python loop
        # over levels (the loop was launch-latency bound on the GPU).
        template = self._layers[0].generator
        self._n_sh = len(template._sh_bases)
        if self._n_sh:
            self._sh_coeffs = xp.stack(
                [xp.stack([flow._sh_coeffs[level] for flow in flows])
                 for level in range(self._n_sh)]
            )  # (P, L, 3, 3)
            self._sh_amps = xp.stack(
                [xp.stack([s.generator._sh_bases[level][0] for s in self._layers])
                 for level in range(self._n_sh)]
            )  # (P, L, 3, 3)
            self._sh_basis = xp.stack(
                [basis for _amp, basis in template._sh_bases]
            )  # (P, 3, n_screen)
            self._sh_freqs = xp.stack(
                [template._sh_freqs[level] for level in range(self._n_sh)]
            )  # (P, 3)
        self._boil_main_tau, self._boil_sh_tau = self._build_boil_tau_maps()
        if self._lgs_mag is not None:
            self._build_lgs_zoom()

    def _build_lgs_zoom(self):
        """Precompute the stacked per-layer LGS cone zoom taps (fixed magnification).

        Each layer's ``(n_screen, n_screen)`` screen is sampled at the central
        pupil grid scaled about the screen centre by that layer's cone
        magnification ``mag = 1 - h/H_LGS`` (isotropic, so rows and columns
        share one set of taps). The taps — clipped buffer indices and the
        interpolation weights (honouring ``interp``) — depend only on the fixed
        magnification, so they are built once and stacked into ``(L, T, n)``
        arrays (``T`` taps: 2 linear / 4 cubic / 6 lanczos). The per-frame
        readout is then a handful of batched ``take_along_axis`` gathers over
        all layers at once, not a per-layer/per-tap Python loop.
        """
        xp = self.xp
        centre = (self.n_screen - 1) / 2.0
        base = np.arange(self.n, dtype=np.float64) - (self.n - 1) / 2.0
        rdtype = self._spectra.real.dtype
        idx_layers, w_layers = [], []
        for mag in self._lgs_mag:
            pos = centre + base * float(mag)
            p0 = np.floor(pos).astype(np.int64)
            fr = pos - p0
            if self.interp == "linear":
                offsets: Tuple[int, ...] = (0, 1)
                weights: Tuple[Any, ...] = (1.0 - fr, fr)
            elif self.interp == "lanczos":
                offsets, weights = _lanczos_weights(fr, np)
            else:  # cubic
                offsets = (-1, 0, 1, 2)
                weights = _catmull_rom_weights(fr)
            idx_layers.append(
                np.stack([np.clip(p0 + off, 0, self.n_screen - 1) for off in offsets])
            )  # (T, n)
            w_layers.append(np.stack([np.asarray(w) for w in weights]))  # (T, n)
        self._zoom_idx = xp.asarray(np.stack(idx_layers))             # (L, T, n) int
        self._zoom_w = xp.asarray(np.stack(w_layers), dtype=rdtype)   # (L, T, n)

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

        if not self._n_sh:
            return main_tau, None
        fp = self._sh_freqs  # (P, 3)
        fr_sh = xp.hypot(fp[:, :, None], fp[:, None, :]).astype(rdtype)  # (P, 3, 3)
        ratio_sh = xp.clip(
            fr_sh[:, None, :, :] / f_ref_dev[None, :, None, None], 1.0, None
        )  # (P, L, 3, 3)
        sh_tau = tau_boil_dev[None, :, None, None] * ratio_sh ** (-2.0 / 3.0)
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

        The profile name and its :func:`pyturb.profile_info` provenance
        (traceable-vs-representative, source, site) are recorded in
        :attr:`metadata` so a saved OPD carries where its atmosphere came from.
        """
        atm = cls(_profiles.get_profile(name), **kwargs)
        atm._profile_name = str(name).lower()
        return atm

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
            coordinates). ``engine="spectral"`` (default) supports arbitrary
            random-access ``t``; ``engine="extrude"`` is **streaming**: ``t``
            must not decrease relative to a previous ``opd``/``frames``/
            ``evolve`` call on the same object (a decreasing ``t`` raises
            ``ValueError``), and :meth:`reset` restarts it at ``t=0``.
        directions : sequence of (thx, thy), optional
            Off-axis directions [arcsec] from the on-axis line of sight. Each
            direction's radius ``sqrt(thx**2 + thy**2)`` must lie within the
            ``field_of_view`` declared at construction, or the screens are
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
        oxs, oys = [], []
        for thx, thy in directions:
            radius = float(np.hypot(thx, thy))
            if radius > self.field_of_view:
                raise ValueError(
                    f"direction ({thx}, {thy}) arcsec has radius {radius:.3g} "
                    f"arcsec, exceeding the declared "
                    f"field_of_view={self.field_of_view} arcsec (a radius). The "
                    "screens are only oversized out to that radius, so a larger "
                    "request would sample wrapped (spectral) or clamped "
                    "(extrude) turbulence. Construct the Atmosphere with a "
                    "field_of_view covering every direction you plan to request."
                )
            oxs.append(np.tan(thx * _ARCSEC_TO_RAD))
            oys.append(np.tan(thy * _ARCSEC_TO_RAD))
        # On the GPU the spectral engine (without the per-layer LGS zoom)
        # batches all directions through one inverse FFT and one subharmonic
        # matmul chain -- ~1.8x at 512² by removing per-direction launch
        # latency. On the CPU the per-direction fused path (one threaded FFT +
        # Numba layer sum each) is faster than a batched scipy transform, so the
        # loop is kept there; the extruder (streaming readout) and the spectral
        # LGS cone (per-layer zoom) also stay a per-direction pass.
        if (self.engine == "spectral" and self._lgs_mag is None
                and self.xp is not np):
            stacked = self._integrate_dirs(float(t), oxs, oys)
        else:
            stacked = xp.stack(
                [self._phase(float(t), ox, oy) for ox, oy in zip(oxs, oys)]
            )
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

        The inverse FFT and the subharmonic outer product are both linear, and
        every layer shares one FFT grid / subharmonic basis, so the layer axis
        is collapsed **before** the transform: the shifted spectra are summed to
        a single ``(n, n)`` array and inverse-FFT'd once (not once per layer),
        and each subharmonic level's per-layer ``3x3`` coefficients are summed
        before the single shared basis outer product. Mathematically identical
        to summing the per-layer screens, but one FFT and no ``(L, n, n)``
        intermediate.
        """
        if self._lgs_mag is not None:
            return self._integrate_lgs(t, ox, oy)
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
        if xp is np and _accel.HAVE_NUMBA:
            # Fused single-pass layer sum (reads the (L, n, n) stack once).
            spectrum = np.empty((ns, ns), dtype=cdtype)
            _accel.spectral_layer_sum(self._spectra, phasor_x, phasor_y, spectrum)
        else:
            spectrum = (
                self._spectra * phasor_x[:, :, None] * phasor_y[:, None, :]
            ).sum(axis=0)
        field = self._fft.ifft2(spectrum, axes=(-2, -1)) * (ns * ns)
        total = field.real

        if self._n_sh:
            # All subharmonic levels at once: shift each level's per-layer 3x3
            # coefficients, sum over layers, then evaluate the shared sinusoid
            # bases as two batched matmuls collapsed to one (ns, 3P) @ (3P, ns).
            fp = self._sh_freqs  # (P, 3)
            px = xp.exp((2j * np.pi) * sx[None, :, None] * fp[:, None, :]).astype(cdtype)
            py = xp.exp((2j * np.pi) * sy[None, :, None] * fp[:, None, :]).astype(cdtype)
            shifted = (
                self._sh_coeffs * px[:, :, :, None] * py[:, :, None, :]
            ).sum(axis=1)  # (P, 3, 3)
            m = xp.matmul(shifted, self._sh_basis).reshape(self._n_sh * 3, ns)
            basis_flat = self._sh_basis.reshape(self._n_sh * 3, ns)
            low = (basis_flat.T @ m).real  # (ns, ns)
            low -= low.mean()
            total = total + low
        # Crop the central pupil region out of the (oversized) screen.
        total = total[self._crop, self._crop]
        return xp.ascontiguousarray(total.astype(self.dtype_out, copy=False))

    def _integrate_dirs(self, t: float, oxs: Sequence[float],
                        oys: Sequence[float]) -> Any:
        """Spectral integrate for several off-axis directions at once -> (D, n, n).

        Numerically identical to calling :meth:`_integrate` per direction, but
        the per-direction summed spectra are stacked and inverse-FFT'd in a
        single batched call, and the subharmonic contribution is evaluated for
        all directions in batched matmuls -- so D directions cost one FFT launch
        and one matmul chain, not D. The memory-heavy ``(L, n, n)`` layer sum
        stays a per-direction loop, so no ``(D, L, n, n)`` intermediate is ever
        formed.
        """
        xp = self.xp
        cdtype = self._cdtype
        ns = self.n_screen
        rdtype = self._spectra.real.dtype
        oxs = np.asarray(oxs, dtype=np.float64)
        oys = np.asarray(oys, dtype=np.float64)
        D = oxs.shape[0]
        disp_x = self._vx[None, :] * t + self._alt[None, :] * oxs[:, None]  # (D, L)
        disp_y = self._vy[None, :] * t + self._alt[None, :] * oys[:, None]
        if not self._wrap_warned:
            self._check_wrap(disp_x, disp_y)
        f = self._grid_f
        sx_all = xp.asarray(disp_x, dtype=rdtype)  # (D, L)
        sy_all = xp.asarray(disp_y, dtype=rdtype)
        specs = xp.empty((D, ns, ns), dtype=cdtype)
        for d in range(D):
            phx = xp.exp((2j * np.pi) * sx_all[d][:, None] * f[None, :]).astype(cdtype)
            phy = xp.exp((2j * np.pi) * sy_all[d][:, None] * f[None, :]).astype(cdtype)
            if xp is np and _accel.HAVE_NUMBA:
                spectrum = np.empty((ns, ns), dtype=cdtype)
                _accel.spectral_layer_sum(self._spectra, phx, phy, spectrum)
                specs[d] = spectrum
            else:
                specs[d] = (
                    self._spectra * phx[:, :, None] * phy[:, None, :]
                ).sum(axis=0)
        total = (self._fft.ifft2(specs, axes=(-2, -1)) * (ns * ns)).real  # (D, ns, ns)

        if self._n_sh:
            fp = self._sh_freqs  # (P, 3)
            px = xp.exp(
                (2j * np.pi) * sx_all[:, None, :, None] * fp[None, :, None, :]
            ).astype(cdtype)  # (D, P, L, 3)
            py = xp.exp(
                (2j * np.pi) * sy_all[:, None, :, None] * fp[None, :, None, :]
            ).astype(cdtype)
            # (D, P, 3, 3): shift each level's per-layer 3x3 coeffs, sum layers.
            shifted = (
                self._sh_coeffs[None] * px[:, :, :, :, None] * py[:, :, :, None, :]
            ).sum(axis=2)
            m = xp.matmul(shifted, self._sh_basis[None]).reshape(D, self._n_sh * 3, ns)
            basis_flat = self._sh_basis.reshape(self._n_sh * 3, ns)  # (3P, ns)
            low = xp.matmul(basis_flat.T[None], m).real  # (D, ns, ns)
            low = low - low.mean(axis=(-2, -1), keepdims=True)
            total = total + low
        total = total[:, self._crop, self._crop]
        return xp.ascontiguousarray(total.astype(self.dtype_out, copy=False))

    def _integrate_lgs(self, t: float, ox: float, oy: float) -> Any:
        """Spectral frame with the LGS cone: per-layer inverse FFT then zoom.

        Each layer's screen (frozen-flow shifted and, if boiling, boiled) is
        inverse-FFT'd, its subharmonic low-frequency part added, then sampled at
        the pupil grid scaled about the screen centre by the layer's cone
        magnification, and the layers summed. The layer axis cannot be collapsed
        before the transform (each layer zooms differently), so this is the
        slower per-layer path — used only when ``lgs_altitude`` is set.
        """
        xp = self.xp
        cdtype = self._cdtype
        ns = self.n_screen
        disp_x = self._vx * t + self._alt * ox
        disp_y = self._vy * t + self._alt * oy
        if not self._wrap_warned:
            self._check_wrap(disp_x, disp_y)
        sx = xp.asarray(disp_x, dtype=self._spectra.real.dtype)
        sy = xp.asarray(disp_y, dtype=self._spectra.real.dtype)
        f = self._grid_f
        phasor_x = xp.exp((2j * np.pi) * sx[:, None] * f[None, :]).astype(cdtype)
        phasor_y = xp.exp((2j * np.pi) * sy[:, None] * f[None, :]).astype(cdtype)
        spectra = self._spectra * phasor_x[:, :, None] * phasor_y[:, None, :]
        screens = (self._fft.ifft2(spectra, axes=(-2, -1)) * (ns * ns)).real  # (L,ns,ns)

        if self._n_sh:
            # Batch the phasor/shift across levels (small), but keep the layer
            # axis through the basis matmul: each layer's low-frequency screen
            # is zoomed differently below, so it cannot be collapsed here.
            fp = self._sh_freqs  # (P, 3)
            pxs = xp.exp((2j * np.pi) * sx[None, :, None] * fp[:, None, :]).astype(cdtype)
            pys = xp.exp((2j * np.pi) * sy[None, :, None] * fp[:, None, :]).astype(cdtype)
            # (P, L, 3, 3): per level, per layer, shifted 3x3 coefficients.
            shifted = self._sh_coeffs * pxs[:, :, :, None] * pys[:, :, None, :]
            low = None
            for p in range(self._n_sh):
                basis = self._sh_basis[p]  # (3, ns)
                contrib = xp.matmul(basis.T, xp.matmul(shifted[p], basis))  # (L,ns,ns)
                low = contrib.real if low is None else low + contrib.real
            low = low - low.mean(axis=(-2, -1), keepdims=True)
            screens = screens + low

        # Zoom-sample every layer's screen about the centre by its cone
        # magnification and sum the layers into the pupil.
        total = self._lgs_zoom(screens)
        return xp.ascontiguousarray(total.astype(self.dtype_out, copy=False))

    def _lgs_zoom(self, screens: Any) -> Any:
        """Cone-zoom the ``(L, n_screen, n_screen)`` layer screens into ``(n, n)``.

        Separable interpolation (rows then columns) with the precomputed
        per-layer taps (:meth:`_build_lgs_zoom`), summed over layers. On the GPU,
        below a working-set threshold, this is a handful of ``take_along_axis``
        gathers batched over all layers and taps (3-8x the old per-layer/per-tap
        Python loop at 256²-512²). On the CPU (the tight loop is
        cache-friendlier) or for a large batched working set on the GPU (the
        ``(L, n, n_screen)`` intermediates go memory-bound), the per-layer loop
        is used instead.
        """
        xp = self.xp
        idx, w = self._zoom_idx, self._zoom_w  # (L, T, n)
        n_layers, n_taps = idx.shape[0], idx.shape[1]
        n, ns = self.n, self.n_screen
        if xp is np or n_layers * n * ns > _LGS_BATCH_MAX_ELEMS:
            total = None
            for layer in range(n_layers):
                field = screens[layer]
                out = None
                for a in range(n_taps):
                    band = field[idx[layer, a]]  # (n, ns)
                    row_term = None
                    for b in range(n_taps):
                        term = w[layer, b][None, :] * band[:, idx[layer, b]]
                        row_term = term if row_term is None else row_term + term
                    contrib = w[layer, a][:, None] * row_term
                    out = contrib if out is None else out + contrib
                total = out if total is None else total + out
            return total
        rows = None  # row-interpolated screens, (L, n, ns)
        for a in range(n_taps):
            gather = xp.broadcast_to(idx[:, a, :, None], (n_layers, n, ns))
            band = xp.take_along_axis(screens, gather, axis=1)  # (L, n, ns)
            term = w[:, a, :, None] * band
            rows = term if rows is None else rows + term
        out = None  # then interpolate along columns, (L, n, n)
        for b in range(n_taps):
            gather = xp.broadcast_to(idx[:, b, None, :], (n_layers, n, n))
            col = xp.take_along_axis(rows, gather, axis=2)  # (L, n, n)
            term = w[:, b, None, :] * col
            out = term if out is None else out + term
        return out.sum(axis=0)  # sum layers -> (n, n)

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
        if self._n_sh:
            a_sh = xp.exp(-dt / self._boil_sh_tau)  # (P, L, 3, 3)
            b_sh = xp.sqrt(xp.clip(1.0 - a_sh * a_sh, 0.0, None))
            noise = self._boil_rng.standard_normal(
                (2, self._n_sh, L, 3, 3), dtype=self._spectra.real.dtype
            )
            fresh = (noise[0] + 1j * noise[1]) * self._sh_amps
            self._sh_coeffs = a_sh * self._sh_coeffs + b_sh * fresh.astype(self._cdtype)

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
            else:
                self._ext.boil_step(float(dt))  # no-op unless tau_boil is finite

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
        else:
            self._ext.boil_step(float(dt))
        return self._to_opd(self._phase(self._t, 0.0, 0.0), wavelength)

    def sample(
        self, count: Optional[int] = None, wavelength: Optional[float] = None
    ) -> Any:
        """Draw statistically independent integrated OPDs.

        Each call produces fresh, uncorrelated realisations of the summed
        atmosphere (on-axis), so the ensemble structure function matches the
        total ``r0``. Layers that share an outer scale are drawn as a single
        aggregate screen (independent von Kármán screens with the same PSD shape
        add exactly, ``r0_agg^{-5/3} = sum_i r0_i^{-5/3}``), so this costs one
        FFT per distinct ``L0`` rather than one per layer -- for a profile whose
        layers share ``L0`` it is ~L times faster than a per-layer sum, and
        distributionally identical.

        Parameters
        ----------
        count : int, optional
            Number of independent OPDs. If omitted a single ``(n, n)`` array
            is returned; otherwise the result is ``(count, n, n)``.
        wavelength : float, optional
            If given, return phase [rad] at this wavelength; otherwise OPD [m].
        """
        total = None
        for generator in self._sample_generators:
            screens = generator.generate(count)
            total = screens if total is None else total + screens
        # Crop the central pupil out of the (possibly oversized) screens.
        total = total[..., self._crop, self._crop]
        return self._to_opd(total, wavelength)

    def reset(self) -> "Atmosphere":
        """Reset to ``t = 0`` and restore the initial turbulence. Returns ``self``.

        Rewinds the internal clock and, when boiling (``tau_boil``) has run,
        restores the pre-boil turbulence and boiling RNG so a reused atmosphere
        replays identically. For ``engine="extrude"`` the extruded layers are
        rebuilt from their seeds (wind travel is monotonic, so the run restarts
        identically). For ``engine="spectral"`` frozen flow never mutates the
        stored spectra, but boiling does, so the stacked spectra, subharmonic
        coefficients and boiling RNG are rebuilt from the layers' fixed
        realisations.
        """
        self._t = 0.0
        self._wrap_warned = False
        if self.engine == "extrude":
            self._ext = ExtrudedAtmosphere(**self._ext_kwargs)
        elif np.any(np.isfinite(self.tau_boil)):
            # Boiling reassigns self._spectra / self._sh_coeffs in place; rebuild
            # them from the untouched per-layer flow realisations and rewind the
            # boil RNG so the boiled sequence repeats bit-for-bit.
            self._boil_rng = self.xp.random.default_rng(self._boil_seed)
            self._build_batched()
        return self

    @property
    def time(self) -> float:
        """Current internal clock [s] (advanced by :meth:`frames`)."""
        return self._t

    @property
    def metadata(self) -> Dict[str, Any]:
        """Provenance describing this atmosphere, for saving with output.

        A flat dict of scalars/strings suitable for :func:`pyturb.save`
        headers — geometry, line-of-sight ``r0``, the integrated
        seeing/theta0/tau0, and the main construction parameters — so a saved
        OPD records how it was made. For atmospheres built via
        :meth:`from_profile` it also records the profile name and its
        :func:`pyturb.profile_info` provenance (source, whether it is traceable
        to a published table, site, and the representativeness caveat). This is
        a **descriptive summary, not a full replayable checkpoint**: the
        per-layer arrays (altitudes, Cn2 fractions, winds) and any evolved/boiled
        stochastic state are not serialised here, so it cannot by itself
        reconstruct a specific evolved frame. ``L0``/``tau_boil`` are reported
        only when every layer shares one value (otherwise ``None``, which
        :func:`pyturb.save` drops).
        """
        l0_values = {round(float(layer.L0), 9) for layer in self.layers}
        uniform_l0 = float(self.layers[0].L0) if len(l0_values) == 1 else None
        tau = self.tau_boil
        if np.all(np.isfinite(tau)) and np.all(tau == tau[0]):
            uniform_tau = float(tau[0])  # every layer boiling at one rate
        else:
            uniform_tau = None           # frozen, or mixed/per-layer rates
        # Profile provenance (only for atmospheres built via from_profile).
        prof = _profiles.profile_info(self._profile_name) if self._profile_name else None
        return {
            "units": "metres",
            "pixel_scale": self.pixel_scale,
            "diameter": self.diameter,
            "n": self.n,
            "r0": self.r0_los,
            "L0": uniform_l0,
            "wavelength": self.wavelength,
            "seeing": self.seeing,
            "theta0": self.theta0,
            "tau0": self.tau0,
            "zenith_angle": self.zenith_angle,
            "n_layers": len(self.layers),
            "engine": self.engine,
            "interp": self.interp,
            "power_law": self.power_law,
            "inner_scale": self.inner_scale,
            "subharmonics": self.subharmonics,
            "field_of_view": self.field_of_view,
            "tau_boil": uniform_tau,
            "dispersion": self.dispersion,
            "wet_fraction": self.wet_fraction,
            "lgs_altitude": (0.0 if self.lgs_altitude is None
                             else float(self.lgs_altitude)),
            "device": self.device,
            "dtype": self.dtype,
            "seed": self.seed,
            "time": self._t,
            "profile": self._profile_name,
            "profile_source": None if prof is None else prof.source,
            "profile_traceable": None if prof is None else prof.traceable,
            "profile_site": None if prof is None else prof.site,
            "profile_caveat": None if prof is None else prof.caveat,
        }

    def __repr__(self) -> str:
        return (
            f"Atmosphere(layers={len(self.layers)}, r0={self.r0_los:.3f} m, "
            f"seeing={self.seeing:.2f}\", theta0={self.theta0:.2f}\", "
            f"tau0={self.tau0 * 1e3:.1f} ms, n={self.n}, "
            f"diameter={self.diameter} m, device={self.device!r})"
        )

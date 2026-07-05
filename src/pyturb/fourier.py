"""FFT-based atmospheric phase screens (Kolmogorov / von Kármán).

Screens are drawn by colouring complex white noise with the square root of
the atmospheric phase power spectral density and inverse-FFTing (McGlamery's
method). Because a periodic FFT grid cannot represent spatial frequencies
below 1/(n*pixel_scale), low-order aberrations (tip/tilt in particular) are
under-represented; optional subharmonics (Lane et al. 1992; Johansson &
Gavel 1994) restore that missing low-frequency power.

References
----------
- McGlamery (1976), "Computer simulation studies of compensation of
  turbulence degraded images".
- Lane, Glindemann & Dainty (1992), Waves in Random Media 2, 209.
- Johansson & Gavel (1994), Proc. SPIE 2200.
- Schmidt (2010), "Numerical Simulation of Optical Wave Propagation".
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from .backend import get_array_module, get_fft_module

__all__ = ["PhaseScreen"]


class PhaseScreen:
    """Generator of statistically independent atmospheric phase screens.

    Each call to :meth:`generate` returns fresh, uncorrelated realisations
    of von Kármán (finite ``L0``) or Kolmogorov (``L0=inf``) turbulence.
    For temporally correlated, frozen-flow screens see
    :class:`pyturb.InfinitePhaseScreen`.

    Parameters
    ----------
    n : int
        Screen size in pixels (screens are ``n x n``).
    pixel_scale : float
        Size of one pixel in metres.
    r0 : float
        Fried parameter in metres, defined at the wavelength at which the
        phase is expressed (the returned phase is in radians at that
        wavelength). Use :func:`pyturb.r0_at_wavelength` to rescale.
    L0 : float, optional
        Outer scale in metres. ``numpy.inf`` gives pure Kolmogorov
        statistics. Default 25 m.
    subharmonics : int, optional
        Number of subharmonic levels used to restore low spatial-frequency
        power that the periodic FFT grid cannot represent. ``0`` disables.
        Each level extends spectral coverage by a further factor of 3 in
        scale; the default of 8 reproduces the Kolmogorov structure
        function to within a few percent.
    power_law : float, optional
        Power-law index of the phase PSD, ``PSD ~ f^{-power_law}``. Default
        ``11/3`` (Kolmogorov, structure function ``~ r^{5/3}``). Other values
        give non-Kolmogorov turbulence (``D(r) ~ r^{power_law-2}``); ``r0`` then
        acts as an amplitude knob rather than the strict Fried parameter.
    inner_scale : float, optional
        Inner scale l0 [m]. ``0`` (default) disables it; a positive value rolls
        the PSD off above ``fm = 5.92/(2 pi l0)`` (modified von Kármán / Hill),
        flattening the structure function at small separations.
    seed : int, optional
        Seed for the random generator. Screens are reproducible for a fixed
        seed, backend and dtype.
    device : str, optional
        ``"cpu"`` (default) or ``"gpu"`` (requires CuPy).
    dtype : str or dtype, optional
        Floating dtype of the output, default ``"float32"`` (recommended
        for GPU work). Use ``"float64"`` for maximum accuracy.

    Examples
    --------
    >>> import pyturb
    >>> gen = pyturb.PhaseScreen(n=256, pixel_scale=0.02, r0=0.15, seed=0)
    >>> phase = gen.generate()          # (256, 256) radians
    >>> batch = gen.generate(32)        # (32, 256, 256), one FFT batch
    """

    def __init__(
        self,
        n: int,
        pixel_scale: float,
        r0: float,
        L0: float = 25.0,
        subharmonics: int = 8,
        power_law: float = 11.0 / 3.0,
        inner_scale: float = 0.0,
        seed: Any = None,
        device: str = "cpu",
        dtype: Union[str, np.dtype] = "float32",
    ):
        if n < 2:
            raise ValueError("n must be at least 2")
        if pixel_scale <= 0 or r0 <= 0:
            raise ValueError("pixel_scale and r0 must be positive")
        if L0 is None:
            L0 = np.inf
        if L0 <= 0:
            raise ValueError("L0 must be positive (use numpy.inf for Kolmogorov)")
        if subharmonics < 0:
            raise ValueError("subharmonics must be >= 0")
        if power_law <= 2.0:
            raise ValueError("power_law must be > 2 (Kolmogorov is 11/3)")
        if inner_scale < 0:
            raise ValueError("inner_scale must be >= 0 (0 disables it)")

        self.n = int(n)
        self.pixel_scale = float(pixel_scale)
        self.r0 = float(r0)
        self.L0 = float(L0)
        self.subharmonics = int(subharmonics)
        self.power_law = float(power_law)
        self.inner_scale = float(inner_scale)
        self.device = device

        self.xp = get_array_module(device)
        self._fft = get_fft_module(self.xp)
        self.dtype = self.xp.dtype(dtype)
        if self.dtype not in (self.xp.dtype("float32"), self.xp.dtype("float64")):
            raise ValueError("dtype must be float32 or float64")
        self._cdtype = "complex64" if self.dtype == "float32" else "complex128"
        self._rng = self.xp.random.default_rng(seed)

        self._build_filters()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _psd(self, f_squared):
        """Von Kármán phase PSD (rad^2 m^2) at squared frequency (m^-2).

        The general (modified) von Kármán form
        ``0.023 r0^{-5/3} exp(-f^2/fm^2) (f^2 + 1/L0^2)^{-power_law/2}``:
        ``power_law=11/3`` is Kolmogorov, the outer scale ``L0`` caps the low
        frequencies, and a non-zero ``inner_scale`` l0 rolls off the high ones
        with ``fm = 5.92/(2 pi l0)`` (Tatarski/Hill). For ``power_law != 11/3``
        the ``0.023 r0^{-5/3}`` amplitude is retained but ``r0`` is then just an
        amplitude knob, not strictly the Fried parameter.
        """
        f0_squared = 0.0 if np.isinf(self.L0) else 1.0 / self.L0**2
        psd = 0.023 * self.r0 ** (-5.0 / 3.0) * (f_squared + f0_squared) ** (
            -self.power_law / 2.0
        )
        if self.inner_scale > 0.0:
            fm = 5.92 / (2.0 * np.pi * self.inner_scale)
            psd = psd * np.exp(-f_squared / fm**2)
        return psd

    def _band_amplitude(self, fx, fy, width, samples=33):
        """Amplitude for a discrete mode representing a frequency cell.

        The Kolmogorov PSD is steeply convex (~ f^-11/3), so at low
        frequencies PSD(centre) * area misrepresents the power in a cell.
        A single mode at the cell centre with amplitude

            amp^2 = integral(PSD * f^2) / f_centre^2

        reproduces the cell's exact ensemble contribution to the structure
        function in the small-separation regime 1 - cos(2 pi f r) ~ f^2
        (which holds wherever the correction differs from PSD * area: only
        cells much wider than their centre frequency are strongly convex,
        and those have 2 pi f r << 1 across the screen). Evaluated by
        midpoint quadrature; reduces to sqrt(PSD * area) for flat PSD.
        """
        offsets = ((np.arange(samples) + 0.5) / samples - 0.5) * width
        f_squared = (fx + offsets[:, None]) ** 2 + (fy + offsets[None, :]) ** 2
        psd = self._psd(f_squared)
        band_power = psd.mean() * width**2
        f_squared_effective = (psd * f_squared).sum() / psd.sum()
        return np.sqrt(band_power * f_squared_effective / (fx**2 + fy**2))

    def _build_filters(self):
        n, dx = self.n, self.pixel_scale
        # Main FFT grid: amplitude filter sqrt(PSD(f)) * df, DC zeroed.
        df = 1.0 / (n * dx)
        f = np.fft.fftfreq(n, d=dx)
        f_squared = f[:, None] ** 2 + f[None, :] ** 2
        f_squared[0, 0] = 1.0  # placeholder, DC is zeroed below
        amplitude = np.sqrt(self._psd(f_squared)) * df
        amplitude[0, 0] = 0.0
        # Near DC the centre-value approximation is poor; integrate those
        # cells properly (a handful of cells, one-time cost).
        k_max = min(8, (n - 1) // 2)
        for i in range(-k_max, k_max + 1):
            for j in range(-k_max, k_max + 1):
                if i == 0 and j == 0:
                    continue
                amplitude[i, j] = self._band_amplitude(f[i], f[j], df)
        self._amplitude = self.xp.asarray(amplitude, dtype=self.dtype)
        # Frequencies of the main grid, kept for frozen-flow translation
        # (the shift theorem needs them); see pyturb.flow.FourierFlowScreen.
        self._f = self.xp.asarray(f, dtype=self.dtype)

        # Subharmonic modes: for each level p, a 3x3 grid of frequencies
        # spaced df/3^p. The 9 cells of level p exactly tile the DC hole
        # left by level p-1 (and by the main grid for p=1); each level's
        # own centre cell is excluded and covered by the next level.
        coords = (np.arange(n) - n / 2.0) * dx
        self._sh_bases = []  # list of (amplitude(3,3), basis(3,n))
        self._sh_freqs = []  # list of (3,) mode frequencies, for translation
        for p in range(1, self.subharmonics + 1):
            df_p = df / 3.0**p
            f_p = np.array([-1.0, 0.0, 1.0]) * df_p
            amp_p = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        continue
                    amp_p[i, j] = self._band_amplitude(f_p[i], f_p[j], df_p)
            basis = np.exp(2j * np.pi * f_p[:, None] * coords[None, :])
            self._sh_bases.append(
                (
                    self.xp.asarray(amp_p, dtype=self.dtype),
                    self.xp.asarray(basis, dtype=self._cdtype),
                )
            )
            self._sh_freqs.append(self.xp.asarray(f_p, dtype=self.dtype))

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    def generate(self, count: Optional[int] = None) -> Any:
        """Generate independent phase screens.

        Parameters
        ----------
        count : int, optional
            Number of screens. If omitted, a single ``(n, n)`` array is
            returned; otherwise the result has shape ``(count, n, n)``.

        Returns
        -------
        ndarray
            Phase in radians, on the selected device (``numpy`` array on
            CPU, ``cupy`` array on GPU).
        """
        squeeze = count is None
        count = 1 if squeeze else int(count)
        if count < 1:
            raise ValueError("count must be >= 1")

        xp, n = self.xp, self.n
        # The real and imaginary parts of one coloured-noise IFFT are two
        # independent screens, so one complex FFT yields a pair.
        n_fft = (count + 1) // 2
        noise = self._rng.standard_normal((2, n_fft, n, n), dtype=self.dtype)
        spectrum = (noise[0] + 1j * noise[1]) * self._amplitude
        field = self._fft.ifft2(spectrum, axes=(-2, -1)) * (n * n)
        screens = xp.concatenate((field.real, field.imag))[:count]

        if self._sh_bases:
            low = None
            for amp_p, basis in self._sh_bases:
                noise = self._rng.standard_normal((2, count, 3, 3), dtype=self.dtype)
                cn = (noise[0] + 1j * noise[1]) * amp_p
                # sum_ij cn_ij e^(2i pi f_i x) e^(2i pi f_j y), evaluated as
                # two small matmuls: (n,3) @ ((s,3,3) @ (3,n)) -> (s,n,n).
                contribution = basis.T @ (cn @ basis)
                low = contribution if low is None else low + contribution
            low = low.real
            low -= low.mean(axis=(-2, -1), keepdims=True)
            screens = screens + low

        screens = xp.ascontiguousarray(screens.astype(self.dtype, copy=False))
        return screens[0] if squeeze else screens

    def __repr__(self) -> str:
        extra = ""
        if self.power_law != 11.0 / 3.0:
            extra += f", power_law={self.power_law}"
        if self.inner_scale > 0.0:
            extra += f", inner_scale={self.inner_scale}"
        return (
            f"PhaseScreen(n={self.n}, pixel_scale={self.pixel_scale}, "
            f"r0={self.r0}, L0={self.L0}, subharmonics={self.subharmonics}{extra}, "
            f"device={self.device!r}, dtype={self.dtype.name!r})"
        )

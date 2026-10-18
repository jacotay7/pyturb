"""pyturb follows the AO stack conventions (aocore CONVENTIONS.md)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

import pyturb
from pyturb import analysis

conformance = pytest.importorskip("aocore.conformance")


def test_frozen_flow_moves_along_minus_wind_vector() -> None:
    n, pitch, speed, dt = 64, 0.1, 10.0, 0.03  # 3 pixels per step

    def frames(direction: float):
        layer = pyturb.Layer(altitude=0.0, cn2_fraction=1.0, wind_speed=speed,
                             wind_direction=direction)
        atm = pyturb.Atmosphere([layer], r0=0.2, diameter=n * pitch, n=n, seed=3)
        first = pyturb.to_numpy(atm.opd(0.0))
        second = pyturb.to_numpy(atm.opd(dt))
        return first, second, speed * dt / pitch

    conformance.check_wind_motion(frames)


def test_zernike_basis_is_noll_with_tip_along_x() -> None:
    cache: dict = {}

    def zernike(j: int, y: np.ndarray, x: Optional[np.ndarray] = None) -> np.ndarray:
        n = 128  # the grid aocore samples: pixel centres on [-1, 1] at pitch 2 / n
        if "basis" not in cache:
            cache["basis"] = analysis.zernike_basis(12, n)
        grid = (np.arange(n) - (n - 1) / 2.0) / (n / 2.0)
        yy, xx = np.meshgrid(grid, grid, indexing="ij")
        inside = np.hypot(xx, yy) <= 1.0
        return cache["basis"][j - 1][inside]

    conformance.check_zernike_basis(zernike, n=128)


def test_pupil_field_phase_images_an_opd_ramp_at_plus_its_slope() -> None:
    """Rule 3.1: the propagated field carries exp(+i 2 pi OPD / lambda)."""
    from aocore import FocalPlanePropagator

    from pyturb.scintillation import propagate_layers

    n, pitch, lam = 64, 0.05, 1e-6
    pixel_scale = lam / (2 * n * pitch)  # Nyquist-sampled image
    imager = FocalPlanePropagator((n, n), pitch, lam, pixel_scale, (48, 48))

    def image(opd: np.ndarray) -> np.ndarray:
        field = propagate_layers(np.asarray(opd)[None], [0.0], pitch, lam)
        return np.abs(imager.forward(field[None])[0]) ** 2

    conformance.check_tilt_direction(image, pupil_shape=(n, n), pitch=pitch,
                                     pixel_scale=pixel_scale)

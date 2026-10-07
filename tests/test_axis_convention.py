"""The (y, x) axis convention shared with aobasis, makewfs, solvephase and HCIPy.

Arrays are indexed ``(y, x)``: x runs along axis 1 (columns), y along axis 0
(rows). Every public x/y-labelled input follows it: ``Layer.wind_direction``
(measured from +x toward +y, the direction the wind blows from),
``directions=(thx, thy)``, ``FourierFlowScreen.translate(sx, sy)``,
``opd_at(x, y)`` and ``zernike_basis``.
"""

import numpy as np
import pytest

import pyturb
from pyturb import analysis as A
from pyturb.flow import FourierFlowScreen

_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


def _pattern_shift(a, b):
    """Integer (axis 0, axis 1) displacement d with b(r) ~ a(r - d).

    Phase correlation of the gradient fields: the raw screens are dominated by
    their largest scales, which would swamp the peak.
    """
    def grad(z):
        return np.diff(z, axis=0)[:, :-1] + np.diff(z, axis=1)[:-1, :]
    cross = np.fft.fft2(grad(b)) * np.conj(np.fft.fft2(grad(a)))
    corr = np.fft.ifft2(cross / (np.abs(cross) + 1e-30)).real
    i, j = np.unravel_index(np.argmax(corr), corr.shape)
    m = corr.shape[0]
    return ((i + m // 2) % m - m // 2, (j + m // 2) % m - m // 2)


def test_wind_vector_is_x_then_y():
    layer = pyturb.Layer(0.0, 1.0, wind_speed=10.0, wind_direction=30.0)
    vx, vy = layer.wind_vector
    assert vx == pytest.approx(10.0 * np.cos(np.deg2rad(30.0)))
    assert vy == pytest.approx(10.0 * np.sin(np.deg2rad(30.0)))


@pytest.mark.parametrize("engine", ["spectral", "extrude"])
def test_wind_direction_zero_moves_pattern_toward_lower_columns(device, engine):
    # (a) wind_direction=0 blows from +x: the pattern moves toward decreasing
    # column index, with no motion along the rows.
    layer = pyturb.Layer(0.0, 1.0, wind_speed=10.0, wind_direction=0.0, L0=25.0)
    atm = pyturb.Atmosphere([layer], r0=0.15, diameter=8.0, n=128, engine=engine,
                            device=device, dtype="float64", seed=5)
    frames = [pyturb.to_numpy(f) for _, f in
              atm.frames(dt=12 * atm.pixel_scale / 10.0, steps=2)]
    assert _pattern_shift(frames[0], frames[1]) == (0, -12)


@pytest.mark.parametrize("engine", ["spectral", "extrude"])
def test_oblique_wind_splits_into_x_columns_and_y_rows(engine):
    # A 30 degree wind travelling 20 px moves the pattern by -20 cos 30 = -17.3
    # px along x (columns) and -20 sin 30 = -10 px along y (rows).
    layer = pyturb.Layer(0.0, 1.0, wind_speed=10.0, wind_direction=30.0, L0=25.0)
    atm = pyturb.Atmosphere([layer], r0=0.15, diameter=8.0, n=128, engine=engine,
                            dtype="float64", seed=6)
    a = pyturb.to_numpy(atm.opd(0.0))
    b = pyturb.to_numpy(atm.opd(20 * atm.pixel_scale / 10.0))
    assert _pattern_shift(a, b) == (-10, -17)


@pytest.mark.parametrize("engine", ["spectral", "extrude"])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_off_axis_direction_shifts_footprint_along_its_own_axis(device, engine, axis):
    # (b) (thx > 0, 0) moves an elevated layer's footprint along +x (columns)
    # by h tan(thx); (0, thy > 0) along +y (rows). The OPD toward that
    # direction is then the on-axis screen read further along that axis, i.e.
    # the pattern appears displaced by -shift. 9.4 px is deliberately
    # sub-pixel; the integer correlation peak must land within a pixel.
    h, shift_px = 6000.0, 9.4
    atm = pyturb.Atmosphere([pyturb.Layer(h, 1.0, 0.0, 0.0, L0=25.0)], r0=0.15,
                            n=128, diameter=8.0, field_of_view=40.0, engine=engine,
                            device=device, dtype="float64", seed=9)
    theta = np.arctan(shift_px * atm.pixel_scale / h) / _ARCSEC_TO_RAD
    direction = (theta, 0.0) if axis == "x" else (0.0, theta)
    on, off = (pyturb.to_numpy(o) for o in
               atm.opd(0.0, directions=[(0.0, 0.0), direction]))
    d0, d1 = _pattern_shift(on, off)
    measured = -d1 if axis == "x" else -d0
    across = d0 if axis == "x" else d1
    assert abs(measured - shift_px) <= 1.0
    assert across == 0


def test_translate_sx_moves_along_columns(device):
    # (c) FourierFlowScreen.translate(sx, sy): sx along x (axis 1), sy along y.
    ps = pyturb.PhaseScreen(n=96, pixel_scale=0.05, r0=0.15, L0=25, subharmonics=3,
                            device=device, dtype="float64", seed=2)
    layer = FourierFlowScreen(ps, seed=4)
    base = pyturb.to_numpy(layer.translate(0.0, 0.0))
    along_x = pyturb.to_numpy(layer.translate(6 * 0.05, 0.0))
    along_y = pyturb.to_numpy(layer.translate(0.0, 6 * 0.05))
    # The FFT-grid part is an exact roll; subharmonics add a smooth residual.
    assert _pattern_shift(base, along_x) == (0, -6)
    assert _pattern_shift(base, along_y) == (-6, 0)


def test_opd_at_x_is_columns_and_y_is_rows():
    atm = pyturb.Atmosphere.from_profile("two-layer", r0=0.15, n=32, diameter=4.0,
                                         dtype="float64", seed=1)
    frame = pyturb.to_numpy(atm.opd(0.0))
    c = (atm.n - 1) / 2.0
    x = (np.array([3, 20, 28]) - c) * atm.pixel_scale    # column offsets
    y = (np.array([2, 7, 12]) - c) * atm.pixel_scale     # row offsets
    got = pyturb.to_numpy(atm.opd_at(x, y))
    np.testing.assert_allclose(got, frame[[2, 7, 12], [3, 20, 28]],
                               atol=1e-12 * np.abs(frame).max())


def test_zernike_tip_varies_along_columns():
    basis = A.zernike_basis(3, 64)
    centre = 32
    tip, tilt = basis[1], basis[2]
    # Z2 (tip, cos theta) grows with the column index and is constant down a
    # column; Z3 (tilt, sin theta) grows with the row index.
    assert np.all(np.diff(tip[centre, 8:56]) > 0)
    assert np.ptp(tip[8:56, centre]) < 1e-12
    assert np.all(np.diff(tilt[8:56, centre]) > 0)
    assert np.ptp(tilt[centre, 8:56]) < 1e-12


@pytest.mark.parametrize("n", [31, 64])
def test_zernike_basis_matches_aobasis(n):
    # (d) Same frame and normalisation as aobasis' Zernike generator evaluated
    # at positions_from_mask (x = columns, y = rows) on the same grid.
    aobasis = pytest.importorskip("aobasis")
    basis = A.zernike_basis(21, n)
    mask = basis[0] != 0
    positions = aobasis.positions_from_mask(mask, pitch=1.0)
    modes = aobasis.ZernikeBasisGenerator(positions, pupil_radius=n / 2.0).generate(21)
    np.testing.assert_allclose(modes.T, basis[:, mask], rtol=0, atol=1e-12)

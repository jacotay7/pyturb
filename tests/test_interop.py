import numpy as np
import pytest

import pyturb

hcipy = pytest.importorskip("hcipy")


def _layer(**kw):
    atm = pyturb.Atmosphere.from_profile("two-layer", r0=0.15, n=32, diameter=4.0,
                                         seed=2, dtype="float64", **kw)
    grid = hcipy.make_pupil_grid(32, 4.0)
    return atm, grid, pyturb.interop.HCIPyLayer(atm, grid)


def test_hcipy_layer_applies_the_pyturb_phase():
    atm, grid, layer = _layer()
    wavelength = 1.6e-6
    layer.t = 0.012
    wf = hcipy.Wavefront(hcipy.Field(np.ones(grid.size), grid), wavelength)
    out = layer(wf)
    expected = pyturb.to_numpy(atm.opd(0.012)).ravel() * 2 * np.pi / wavelength
    # The applied phase equals the pyturb phase modulo 2 pi.
    np.testing.assert_allclose(np.exp(1j * np.angle(out.electric_field)),
                               np.exp(1j * expected), atol=1e-9)
    back = layer.backward(out)
    np.testing.assert_allclose(back.electric_field, wf.electric_field, atol=1e-12)
    np.testing.assert_allclose(layer.phase_for(wavelength), expected)


def test_hcipy_layer_uses_the_dispersion_model_and_resets():
    atm, grid, layer = _layer(dispersion="edlen")
    layer.evolve_until(0.02)
    np.testing.assert_allclose(layer.phase_for(0.8e-6),
                               pyturb.to_numpy(atm.opd(0.02, wavelength=0.8e-6)).ravel())
    layer.reset()
    assert layer.t == 0.0
    np.testing.assert_allclose(layer.opd(), pyturb.to_numpy(atm.opd(0.0)).ravel())


def test_hcipy_layer_rejects_a_mismatched_grid():
    atm = pyturb.Atmosphere.from_profile("two-layer", r0=0.15, n=32, diameter=4.0)
    with pytest.raises(ValueError, match="make_pupil_grid"):
        pyturb.interop.HCIPyLayer(atm, hcipy.make_pupil_grid(64, 4.0))
    with pytest.raises(ValueError, match="spacing"):
        pyturb.interop.HCIPyLayer(atm, hcipy.make_pupil_grid(32, 8.0))

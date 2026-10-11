"""pyturb.set_threads: thread caps that never change results."""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys

import numpy as np
import pytest

import pyturb
from pyturb import _accel, backend


@pytest.fixture(autouse=True)
def _restore_threads():
    fft = pyturb.get_fft_workers()
    yield
    pyturb.set_threads(None)
    pyturb.set_fft_workers(fft)


def _worker_threads(_):
    import numba

    atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, n=32,
                                         tau_boil=0.1, seed=1)
    atm.evolve(1e-3)  # runs the parallel boiling kernel
    return pyturb.get_threads(), numba.get_num_threads()


def test_validation_and_previous_value():
    for bad in (0, -2, 1.5, True, "4"):
        with pytest.raises(ValueError, match="integer >= 1"):
            pyturb.set_threads(bad)
    assert pyturb.set_threads(3) is None
    assert pyturb.get_threads() == 3
    assert pyturb.set_threads(None) == 3
    assert pyturb.get_threads() is None


def test_omp_num_threads_parsing(monkeypatch):
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    for value, expect in (("3", 3), ("2,1", 2), ("0", None), ("x", None), ("", None)):
        monkeypatch.setenv("OMP_NUM_THREADS", value)
        assert backend._env_threads() == expect
    monkeypatch.setenv("NUMBA_NUM_THREADS", "4")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert backend._env_threads() is None  # Numba already sized its pool


def test_caps_threaded_fft_workers_only():
    pyturb.set_fft_workers(None)
    pyturb.set_threads(2)
    assert backend._fft_worker_count() is None  # never turns threading on
    pyturb.set_fft_workers(-1)
    assert backend._fft_worker_count() == min(2, os.cpu_count() or 1)
    pyturb.set_threads(None)
    assert backend._fft_worker_count() == -1


@pytest.mark.skipif(not _accel.HAVE_NUMBA, reason="needs Numba")
def test_worker_processes_use_the_cap():
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(2, initializer=pyturb.set_threads, initargs=(1,)) as pool:
        results = pool.map(_worker_threads, range(2))
    assert results == [(1, 1), (1, 1)]


@pytest.mark.skipif(not _accel.HAVE_NUMBA, reason="needs Numba")
def test_omp_num_threads_is_honoured_without_numba_num_threads():
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_NUM_THREADS"}
    env["OMP_NUM_THREADS"] = "1"
    code = ("import pyturb, numba\n"
            "atm = pyturb.Atmosphere.from_profile('paranal-median', seeing=0.8, "
            "n=32, tau_boil=0.1, seed=1)\n"
            "atm.evolve(1e-3)\n"
            "print(numba.get_num_threads())\n")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "1"


def _frames(**kw):
    atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, n=64, seed=5,
                                         **kw)
    return np.array([pyturb.to_numpy(f) for _, f in atm.frames(1e-3, 3)])


@pytest.mark.parametrize("kw", [dict(tau_boil=0.05),
                                dict(lgs_altitude=90e3, tau_boil=0.1),
                                dict(engine="extrude", field_of_view=5.0)])
def test_results_do_not_depend_on_the_thread_count(kw):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyturb.ExtrudeBoilingPerformanceWarning)
        default = _frames(**kw)
        pyturb.set_threads(1)
        single = _frames(**kw)
        pyturb.set_threads(3)
        three = _frames(**kw)
    np.testing.assert_array_equal(default, single)
    np.testing.assert_array_equal(default, three)

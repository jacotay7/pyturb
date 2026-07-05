"""Shared pytest fixtures and the GPU opt-in gate.

GPU (CuPy) tests are marked ``@pytest.mark.gpu`` and are **skipped by the
normal CPU-only CI**: they run only when the suite is invoked with
``--run-gpu`` on a machine that has a CUDA GPU and CuPy installed
(``pytest --run-gpu``). The ``device`` fixture parameterises a test over
``"cpu"`` (always) and ``"gpu"`` (only under ``--run-gpu``), so a single
statistics test body can be checked on both backends.
"""

from __future__ import annotations

import importlib.util

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run GPU (CuPy) tests, marked @pytest.mark.gpu (needs a CUDA GPU).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: test needs a CUDA GPU + CuPy; skipped unless --run-gpu is passed.",
    )


def _cupy_available() -> bool:
    return importlib.util.find_spec("cupy") is not None


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.gpu tests unless --run-gpu (and CuPy) are present."""
    if config.getoption("--run-gpu"):
        if _cupy_available():
            return
        skip = pytest.mark.skip(reason="--run-gpu given but CuPy is not importable")
    else:
        skip = pytest.mark.skip(reason="GPU test; pass --run-gpu to run it")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(params=["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def device(request) -> str:
    """Backend name: ``"cpu"`` always, ``"gpu"`` only when ``--run-gpu``.

    A test that takes this fixture runs once per available backend, so the
    same assertions guard the NumPy and CuPy paths.
    """
    return request.param

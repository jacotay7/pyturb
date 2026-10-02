"""The README/docs code blocks and the examples gallery must keep running.

Doc snippets are executed in order, one shared namespace per file (later blocks
build on earlier ones), at reduced size: pupil sizes of 256/512 become 64,
``steps=`` becomes 3, and ``Atmosphere``'s default ``n`` drops to 64. Without a
GPU, ``device="gpu"`` snippets run on the CPU instead. Blocks that need an
optional third-party library (HCIPy, poppy) are skipped when it is not
installed, as are later blocks that depended on a skipped one.
"""

from __future__ import annotations

import importlib.util
import re
import runpy
import warnings
from pathlib import Path

import pytest

import pyturb

ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = ["README.md"] + sorted(
    str(p.relative_to(ROOT)) for p in (ROOT / "docs").glob("*.md")
)
# 06_keck_showcase renders an animated WebP (minutes, needs Pillow); it is
# exercised by hand when the showcase is regenerated.
EXAMPLES = sorted(
    p.name for p in (ROOT / "examples").glob("0*.py") if not p.name.startswith("06")
)
OPTIONAL_MODULES = ("hcipy", "poppy", "matplotlib", "torch")


def _gpu_available() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _shrink(code: str, gpu: bool) -> str:
    code = re.sub(r"\b(?:512|256)\b", "64", code)
    if not gpu:
        code = code.replace('device="gpu"', 'device="cpu"')
    return re.sub(r"steps=\d+", "steps=3", code)


@pytest.fixture
def small_atmosphere(monkeypatch, tmp_path):
    """Run in a scratch directory with ``Atmosphere`` defaulting to n=64."""
    monkeypatch.chdir(tmp_path)
    original = pyturb.Atmosphere.__init__

    def init(self, layers, *args, **kwargs):
        kwargs.setdefault("n", 64)
        original(self, layers, *args, **kwargs)

    monkeypatch.setattr(pyturb.Atmosphere, "__init__", init)
    if importlib.util.find_spec("matplotlib") is not None:
        import matplotlib

        matplotlib.use("Agg")


@pytest.mark.parametrize("doc", DOC_FILES)
def test_doc_code_blocks_run(doc, small_atmosphere):
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / doc).read_text(), re.S)
    namespace: dict = {}
    skipped = []
    gpu = _gpu_available()
    for index, block in enumerate(blocks):
        missing = [
            m for m in OPTIONAL_MODULES
            if re.search(rf"\bimport [^\n]*\b{m}\b", block)
            and importlib.util.find_spec(m) is None
        ]
        if missing:
            skipped.append((index, missing))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("error", pyturb.PeriodicWrapWarning)
            try:
                exec(compile(_shrink(block, gpu), f"{doc}[{index}]", "exec"), namespace)
            except NameError:
                if not skipped:
                    raise
                skipped.append((index, ["depends on a skipped block"]))
    if skipped and len(skipped) == len(blocks):
        pytest.skip(f"every block in {doc} needs something unavailable: {skipped}")


@pytest.mark.parametrize("example", EXAMPLES)
def test_examples_run(example, small_atmosphere):
    # Examples run at their own sizes (every one passes n explicitly). None of
    # them may wrap the periodic spectral screen.
    with warnings.catch_warnings():
        warnings.simplefilter("error", pyturb.PeriodicWrapWarning)
        runpy.run_path(str(ROOT / "examples" / example), run_name="__main__")

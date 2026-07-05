import pytest

import pyturb


@pytest.mark.filterwarnings("ignore::pyturb.PeriodicWrapWarning")
def test_benchmark_returns_positive_rates():
    result = pyturb.benchmark(n=32, profile="two-layer", seconds=0.05)
    assert set(result) == {"frames_per_s", "screens_per_s"}
    assert result["frames_per_s"] > 0
    assert result["screens_per_s"] > 0


def test_benchmark_extrude_engine(capsys):
    result = pyturb.benchmark(n=32, profile="two-layer", seconds=0.05,
                              engine="extrude")
    assert result["frames_per_s"] > 0
    out = capsys.readouterr().out
    assert "engine=extrude" in out
    assert "closed-loop frames" in out
    assert "Monte-Carlo screens" in out


def test_benchmark_rejects_unknown_device():
    with pytest.raises(ValueError):
        pyturb.benchmark(n=32, seconds=0.05, device="not-a-real-device")

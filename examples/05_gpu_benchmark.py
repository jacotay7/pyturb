"""05 — Benchmark your machine (CPU, and GPU if CuPy is present).

Run: ``python examples/05_gpu_benchmark.py``
"""

import pyturb

devices = ["cpu"]
try:
    pyturb.get_array_module("gpu")
    devices.append("gpu")
except ImportError:
    print("(CuPy not installed — GPU row skipped)\n")

for device in devices:
    for n in (256, 512):
        pyturb.benchmark(n=n, device=device, seconds=1.0)
        print()

# The extruder engine (non-periodic) on the GPU, for long closed-loop runs:
if "gpu" in devices:
    pyturb.benchmark(n=512, device="gpu", engine="extrude", seconds=1.0)

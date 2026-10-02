"""05 — Benchmark your machine (CPU, and GPU if CuPy is present).

Run: ``python examples/05_gpu_benchmark.py``
"""

import pyturb

devices = ["cpu"]
try:
    cupy = pyturb.get_array_module("gpu")
    if cupy.cuda.runtime.getDeviceCount() > 0:
        devices.append("gpu")
    else:
        print("(no CUDA device visible — GPU row skipped)\n")
except ImportError:
    print("(CuPy not installed — GPU row skipped)\n")
except Exception as exc:  # CUDA driver/runtime missing or broken
    print(f"(CUDA unavailable: {exc} — GPU row skipped)\n")

for device in devices:
    for n in (256, 512):
        pyturb.benchmark(n=n, device=device, seconds=1.0)
        print()

# The extruder engine (non-periodic) on the GPU, for long closed-loop runs:
if "gpu" in devices:
    pyturb.benchmark(n=512, device="gpu", engine="extrude", seconds=1.0)

# Troubleshooting

**`RuntimeError: Failed to find CUDA headers`** on the first GPU call.
CuPy 14 compiles its kernels at runtime and needs the CUDA headers. Install
pyturb's GPU extra (`pip install "pyturb[cuda12]"` or `[cuda13]`), which pulls
`cupy-cudaXXx[ctk]` with the headers as wheels, or `pip install
"cupy-cuda12x[ctk]"` yourself, or point `CUDA_PATH` at a CUDA toolkit.

**`ImportError: device='gpu' requires CuPy`.** CuPy is not installed in this
environment; see above. Check with `python -c "import cupy;
print(cupy.cuda.runtime.getDeviceCount())"`.

**Which GPU am I on?** `device="gpu"` uses CuPy's current device (normally 0);
`device="gpu:1"` picks another. CUDA's default numbering puts the fastest GPU
first, which may differ from `nvidia-smi`; set `CUDA_DEVICE_ORDER=PCI_BUS_ID`
to match it.

**`PeriodicWrapWarning: ... has wrapped`.** The default spectral engine is
periodic: after `atm.time_to_wrap` seconds the fastest layer repeats
turbulence it has already shown. For long runs use `engine="extrude"`, or
enlarge the screen with `oversample` (it multiplies `time_to_wrap`). If the
repetition is intentional, silence it with
`warnings.filterwarnings("ignore", category=pyturb.PeriodicWrapWarning)`.
`pyturb.benchmark()` silences it for its own timing loop.

**`ValueError: engine='extrude' is streaming ...`.** The extruder only moves
forward in time. Calling `opd()` (default `t=0`) after `frames()` asks for the
past; use `atm.time`, `atm.reset()` to restart, or the spectral engine for
random access.

**`ExtrudeBoilingPerformanceWarning`.** Boiling on the extruder re-extrudes a
fresh screen every step. Prefer `engine="spectral"` for boiling-heavy runs.

**`ValueError: direction ... exceeding the declared field_of_view`.** Off-axis
directions must lie within the `field_of_view` radius the `Atmosphere` was
built with, which sizes the screens. Rebuild with a larger `field_of_view`.

**Low-order statistics look low (tilt, large separations).** A pupil-sized FFT
screen is periodic across the pupil. Use `oversample=2`–`4`; see
[Choosing an engine](engines.md).

**My wind blows the wrong way compared with another tool.** pyturb's
`wind_direction` is where the wind comes *from*; the pattern moves along
`-wind_vector`. Axis 0 of a pyturb array is HCIPy's y. See
[Conventions](concepts.md#conventions).

**CPU runs are slow.** Install the `accel` extra (Numba), enable threaded FFTs
with `pyturb.set_fft_workers(-1)`, and use `float32`. See
[Performance](performance.md).

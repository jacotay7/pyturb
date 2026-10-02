# Tutorial notebooks

Executed notebooks live in
[`tutorials/`](https://github.com/jacotay7/pyturb/tree/main/tutorials) and are
re-run by CI, so their outputs match the current release.

- [**01 — From atmosphere to a corrected PSF**](https://github.com/jacotay7/pyturb/blob/main/tutorials/01_atmosphere_to_psf.ipynb):
  a layered atmosphere drives a minimal modal AO loop (Zernike "DM",
  one-frame-delay integrator); the closed-loop residual is compared with
  Noll's fitting-error floor and the long-exposure Strehl with the Maréchal
  approximation.

Run them locally with `pip install "pyturb[plot]" jupyter`.

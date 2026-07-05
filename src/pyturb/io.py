"""Save and load phase/OPD screens with metadata (``.npz`` and FITS).

AO users live in FITS, so screens should round-trip with the parameters that
define them — ``pixel_scale``, ``r0``, ``L0``, ``wavelength``, ``units``,
``seed`` — plus the pyturb version that wrote them. :func:`save` picks the
format from the file extension (``.fits``/``.fit`` → FITS via astropy,
otherwise NumPy ``.npz``); :func:`load` returns ``(array, metadata)``.

FITS is an optional feature: it needs ``astropy`` (``pip install pyturb[fits]``).
The ``.npz`` path has no extra dependencies.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple, Union

import numpy as np

from .backend import to_numpy

__all__ = ["save", "load"]

_META_KEY = "__pyturb_meta__"


def _version() -> str:
    from . import __version__

    return __version__


def _clean_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Drop ``None`` values and coerce keys to strings."""
    return {str(k): v for k, v in meta.items() if v is not None}


def save(path: Union[str, os.PathLike], data: Any, **metadata: Any) -> None:
    """Write ``data`` (a phase/OPD array) to ``path`` with ``metadata``.

    Parameters
    ----------
    path : str
        Destination. ``.fits``/``.fit`` writes FITS (needs astropy); any other
        extension writes a NumPy ``.npz``.
    data : ndarray
        Screen or stack of screens (NumPy or CuPy; device arrays are copied to
        the host). Saved as float32/float64 as-is.
    **metadata
        Arbitrary scalars/strings to record alongside the data, e.g.
        ``pixel_scale=0.0156, r0=0.15, L0=25.0, wavelength=500e-9,
        units="metres", seed=1``. ``None`` values are dropped. The pyturb
        version is always added as ``pyturb_version``.

    Examples
    --------
    >>> import pyturb                                    # doctest: +SKIP
    >>> atm = pyturb.Atmosphere.from_profile("two-layer", seeing=0.8)
    >>> pyturb.save("opd.fits", atm.opd(), **atm.metadata)
    """
    array = np.ascontiguousarray(to_numpy(data))
    meta = _clean_meta(dict(metadata))
    meta.setdefault("pyturb_version", _version())

    path = str(path)
    if path.lower().endswith((".fits", ".fit")):
        _save_fits(path, array, meta)
    else:
        np.savez(path, data=array, **{_META_KEY: json.dumps(meta)})


def load(path: Union[str, os.PathLike]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load an array and its metadata written by :func:`save`.

    Returns
    -------
    (ndarray, dict)
        The data array (on the host) and its metadata dictionary.
    """
    path = str(path)
    if path.lower().endswith((".fits", ".fit")):
        return _load_fits(path)
    with np.load(path, allow_pickle=False) as handle:
        data = handle["data"]
        meta = json.loads(str(handle[_META_KEY])) if _META_KEY in handle else {}
    return data, meta


# ---------------------------------------------------------------------------
# FITS backend (optional astropy dependency)
# ---------------------------------------------------------------------------
def _require_astropy():
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - exercised only without astropy
        raise ImportError(
            "FITS support needs astropy — install it with `pip install pyturb[fits]` "
            "or use a .npz path instead."
        ) from exc
    return fits


def _fits_key(key: str) -> str:
    """FITS keyword: <=8 chars, upper case, else HIERARCH handles the rest."""
    return key.upper()


def _save_fits(path: str, array: np.ndarray, meta: Dict[str, Any]) -> None:
    import warnings

    fits = _require_astropy()
    from astropy.io.fits.verify import VerifyWarning

    hdu = fits.PrimaryHDU(data=array)
    header = hdu.header
    header["BUNIT"] = str(meta.get("units", "radians"))
    # Long keys (>8 chars) become HIERARCH cards, which round-trip fine; the
    # VerifyWarning about them is expected, so silence it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VerifyWarning)
        for key, value in meta.items():
            if isinstance(value, (int, float, bool, str)):
                header[_fits_key(key)] = value
            else:
                header[_fits_key(key)] = json.dumps(value)
        header.add_comment("Written by pyturb")
        hdu.writeto(path, overwrite=True)


def _load_fits(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    fits = _require_astropy()
    reserved = {"SIMPLE", "BITPIX", "EXTEND", "COMMENT", "HISTORY", "BUNIT"}
    naxis = {"NAXIS"} | {f"NAXIS{i}" for i in range(1, 10)}
    with fits.open(path) as hdul:
        hdu = hdul[0]
        # FITS stores big-endian; hand back a native-order array.
        data = np.asarray(hdu.data)
        data = data.astype(data.dtype.newbyteorder("="), copy=False)
        meta = {}
        for key, value in hdu.header.items():
            if not key or key in reserved or key in naxis:
                continue
            meta[key.lower()] = value
    return data, meta

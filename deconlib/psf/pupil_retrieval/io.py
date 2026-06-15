"""I/O for pupil function artifacts.

This module is temporarily separated from the main deconlib API while we
focus on stabilizing the deconvolution API.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import h5py
import numpy as np

from ..optics import Geometry, Optics, make_geometry
from ..pupil_retrieval.retrieval import PhaseRetrievalResult

__all__ = [
    "Pupil",
    "save_pupil",
    "load_pupil",
]

_PUPIL_FORMAT = "deconlib_pupil"
_FORMAT_VERSION = 1


@dataclass
class Pupil:
    """A complex 2D pupil bundled with the optics and sampling it lives on.

    ``pupil`` is in FFT corner-origin convention (matches ``make_pupil``).
    ``spacing`` is the *image-plane* pixel size (μm) the pupil is referenced
    against; pupil-plane sampling is determined by ``(shape, spacing)``.

    Attributes
    ----------
    pupil : ndarray, complex64, shape ``(ny, nx)``
    optics : Optics
    shape : (ny, nx)
    spacing : (dy, dx) μm — image-plane pixel size
    oversample, boundary_smoothing_sigma :
        Anti-aliased pupil-support parameters used by ``make_geometry``.
    source : "theoretical" | "phase_retrieval" | "zernike_fit"
    zernike_coefficients : ndarray, float64, shape ``(N,)`` or ``None``
    zernike_basis : "ansi" | "noll"
    zernike_normalization : "rms" | "p2v"
    retrieval_diagnostics : dict or ``None``
        When ``source == "phase_retrieval"``: keys
        ``mse_history``, ``support_error_history`` (lists),
        ``converged`` (bool), ``iterations`` (int).
    """

    pupil: np.ndarray
    optics: Optics
    shape: Tuple[int, int]
    spacing: Tuple[float, float]
    oversample: int = 8
    boundary_smoothing_sigma: float = 0.0
    source: str = "theoretical"
    zernike_coefficients: Optional[np.ndarray] = None
    zernike_basis: str = "ansi"
    zernike_normalization: str = "rms"
    retrieval_diagnostics: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.pupil.ndim != 2:
            raise ValueError(
                f"pupil must be 2D, got shape {self.pupil.shape}"
            )
        if tuple(self.shape) != self.pupil.shape:
            raise ValueError(
                f"shape {self.shape} does not match pupil array shape "
                f"{self.pupil.shape}"
            )

    @property
    def geometry(self) -> Geometry:
        """Rebuild the full :class:`Geometry` from stored parameters."""
        return make_geometry(
            shape=tuple(self.shape),
            spacing=tuple(self.spacing),
            optics=self.optics,
            oversample=self.oversample,
            boundary_smoothing_sigma=self.boundary_smoothing_sigma,
        )

    @classmethod
    def from_retrieval(
        cls,
        result: PhaseRetrievalResult,
        *,
        optics: Optics,
        shape: Tuple[int, int],
        spacing: Tuple[float, float],
        oversample: int = 8,
        boundary_smoothing_sigma: float = 0.0,
    ) -> "Pupil":
        return cls(
            pupil=result.pupil.astype(np.complex64, copy=False),
            optics=optics,
            shape=tuple(shape),
            spacing=tuple(spacing),
            oversample=oversample,
            boundary_smoothing_sigma=boundary_smoothing_sigma,
            source="phase_retrieval",
            retrieval_diagnostics={
                "mse_history": list(result.mse_history),
                "support_error_history": list(result.support_error_history),
                "converged": bool(result.converged),
                "iterations": int(result.iterations),
            },
        )


def _write_optics(group: h5py.Group, optics: Optics) -> None:
    group.attrs["wavelength"] = float(optics.wavelength)
    group.attrs["na"] = float(optics.na)
    group.attrs["ni"] = float(optics.ni)
    group.attrs["ns"] = float(optics.ns)


def _read_optics(group: h5py.Group) -> Optics:
    return Optics(
        wavelength=float(group.attrs["wavelength"]),
        na=float(group.attrs["na"]),
        ni=float(group.attrs["ni"]),
        ns=float(group.attrs["ns"]),
    )


def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _try_write_attr(g: h5py.Group, key: str, value: Any) -> None:
    """Best-effort scalar/array attr write for free-form metadata."""
    try:
        g.attrs[key] = value
    except (TypeError, ValueError):
        pass


def save_pupil(filepath: str | Path, pupil: Pupil, *, metadata: Optional[dict] = None) -> None:
    """Write a :class:`Pupil` to ``filepath`` as ``.pupil.h5``.

    ``metadata`` is an optional free-form dict written into the root
    group's attrs (used by pyvistra's I/O routing).
    """
    path = Path(filepath)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = _PUPIL_FORMAT
        f.attrs["version"] = _FORMAT_VERSION
        f.attrs["source"] = pupil.source
        f.attrs["created"] = _now_iso()

        if metadata:
            for k, v in metadata.items():
                _try_write_attr(f, k, v)

        f.create_dataset(
            "pupil",
            data=pupil.pupil.astype(np.complex64, copy=False),
            compression="gzip",
            compression_opts=3,
        )

        _write_optics(f.create_group("optics"), pupil.optics)

        g = f.create_group("geometry")
        g.attrs["shape"] = np.asarray(pupil.shape, dtype=np.int64)
        g.attrs["spacing"] = np.asarray(pupil.spacing, dtype=np.float64)
        g.attrs["oversample"] = int(pupil.oversample)
        g.attrs["boundary_smoothing_sigma"] = float(pupil.boundary_smoothing_sigma)

        if pupil.zernike_coefficients is not None:
            zg = f.create_group("zernike")
            zg.create_dataset(
                "coefficients",
                data=np.asarray(pupil.zernike_coefficients, dtype=np.float64),
            )
            zg.attrs["basis"] = pupil.zernike_basis
            zg.attrs["normalization"] = pupil.zernike_normalization

        if pupil.retrieval_diagnostics is not None:
            rg = f.create_group("retrieval")
            d = pupil.retrieval_diagnostics
            rg.attrs["converged"] = bool(d.get("converged", False))
            rg.attrs["iterations"] = int(d.get("iterations", 0))
            if "mse_history" in d:
                rg.create_dataset(
                    "mse_history",
                    data=np.asarray(d["mse_history"], dtype=np.float64),
                )
            if "support_error_history" in d:
                rg.create_dataset(
                    "support_error_history",
                    data=np.asarray(d["support_error_history"], dtype=np.float64),
                )


def load_pupil(filepath: str | Path) -> Pupil:
    """Read a ``.pupil.h5`` file and return a :class:`Pupil`."""
    path = Path(filepath)
    with h5py.File(path, "r") as f:
        fmt = f.attrs.get("format", "")
        if fmt != _PUPIL_FORMAT:
            raise ValueError(
                f"{path}: not a deconlib pupil file (format={fmt!r})"
            )

        source = str(f.attrs.get("source", "theoretical"))
        optics = _read_optics(f["optics"])

        g = f["geometry"]
        shape = tuple(int(v) for v in g.attrs["shape"])
        spacing = tuple(float(v) for v in g.attrs["spacing"])
        oversample = int(g.attrs["oversample"])
        bss = float(g.attrs["boundary_smoothing_sigma"])

        pupil_arr = f["pupil"][...].astype(np.complex64, copy=False)

        zernike_coefficients: Optional[np.ndarray] = None
        zernike_basis = "ansi"
        zernike_normalization = "rms"
        if "zernike" in f:
            zg = f["zernike"]
            zernike_coefficients = zg["coefficients"][...]
            zernike_basis = str(zg.attrs.get("basis", "ansi"))
            zernike_normalization = str(zg.attrs.get("normalization", "rms"))

        retrieval_diagnostics: Optional[dict] = None
        if "retrieval" in f:
            rg = f["retrieval"]
            retrieval_diagnostics = {
                "converged": bool(rg.attrs.get("converged", False)),
                "iterations": int(rg.attrs.get("iterations", 0)),
            }
            if "mse_history" in rg:
                retrieval_diagnostics["mse_history"] = rg["mse_history"][...].tolist()
            if "support_error_history" in rg:
                retrieval_diagnostics["support_error_history"] = (
                    rg["support_error_history"][...].tolist()
                )

    return Pupil(
        pupil=pupil_arr,
        optics=optics,
        shape=shape,
        spacing=spacing,
        oversample=oversample,
        boundary_smoothing_sigma=bss,
        source=source,
        zernike_coefficients=zernike_coefficients,
        zernike_basis=zernike_basis,
        zernike_normalization=zernike_normalization,
        retrieval_diagnostics=retrieval_diagnostics,
    )

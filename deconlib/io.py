"""HDF5 I/O for pupil functions and PSFs.

Single-file, h5py-backed format. Two artifacts:

* ``.pupil.h5`` — a 2D complex pupil + Optics + the sampling parameters
  needed to rebuild Geometry and evaluate the pupil as a PSF. Optionally
  carries a Zernike fit and/or phase-retrieval diagnostics.
* ``.psf.h5`` — a 2D/3D PSF image + Optics + voxel sampling.  Optionally
  carries distillation diagnostics.

Geometry's nine derived ndarrays are *not* serialized; the originating
``(shape, spacing, oversample, boundary_smoothing_sigma)`` params are
stored instead and ``make_geometry`` rebuilds the full Geometry on
demand via the :attr:`Pupil.geometry` property.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import h5py
import numpy as np

from .psf import (
    Geometry,
    Optics,
    PhaseRetrievalResult,
    PsfDistillationResult,
    make_geometry,
)

_PUPIL_FORMAT = "deconlib_pupil"
_PSF_FORMAT = "deconlib_psf"
_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


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


@dataclass
class Psf:
    """A 2D or 3D PSF image + the optics and voxel sampling it was computed at.

    ``psf`` is in FFT corner-origin convention. ``pixel_size`` is
    ``(dz, dy, dx)`` for 3D or ``(dy, dx)`` for 2D, in μm.

    Attributes
    ----------
    psf : ndarray, float32
    optics : Optics
    pixel_size : tuple of float
    z_planes : ndarray or None
        Explicit z grid in μm when sampling is non-uniform; otherwise
        z positions are implied by ``pixel_size[0]`` and the Z axis size.
    source : "theoretical" | "distilled" | "pupil_sampled"
    pupil_ref : str or None
        Reference filename of the pupil this PSF was sampled from.
    distillation_diagnostics : dict or None
        Populated when ``source == "distilled"``.
    """

    psf: np.ndarray
    optics: Optics
    pixel_size: Tuple[float, ...]
    z_planes: Optional[np.ndarray] = None
    source: str = "theoretical"
    pupil_ref: Optional[str] = None
    distillation_diagnostics: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.psf.ndim not in (2, 3):
            raise ValueError(
                f"psf must be 2D or 3D, got shape {self.psf.shape}"
            )
        if len(self.pixel_size) != self.psf.ndim:
            raise ValueError(
                f"pixel_size length {len(self.pixel_size)} does not match "
                f"psf ndim {self.psf.ndim}"
            )

    @classmethod
    def from_distillation(
        cls,
        result: PsfDistillationResult,
        *,
        optics: Optics,
        pixel_size: Tuple[float, ...],
    ) -> "Psf":
        return cls(
            psf=result.psf.astype(np.float32, copy=False),
            optics=optics,
            pixel_size=tuple(pixel_size),
            source="distilled",
            distillation_diagnostics={
                "positions": np.asarray(result.positions, dtype=np.int32),
                "amplitudes": np.asarray(result.amplitudes, dtype=np.float64),
                "chi2_history": list(result.chi2_history),
                "psf_change_history": list(result.psf_change_history),
                "amp_change_history": list(result.amp_change_history),
            },
        )


# ---------------------------------------------------------------------------
# Shared HDF5 helpers
# ---------------------------------------------------------------------------


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
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Pupil I/O
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PSF I/O
# ---------------------------------------------------------------------------


def save_psf(filepath: str | Path, psf: Psf, *, metadata: Optional[dict] = None) -> None:
    """Write a :class:`Psf` to ``filepath`` as ``.psf.h5``."""
    path = Path(filepath)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = _PSF_FORMAT
        f.attrs["version"] = _FORMAT_VERSION
        f.attrs["source"] = psf.source
        f.attrs["created"] = _now_iso()
        if psf.pupil_ref:
            f.attrs["pupil_ref"] = psf.pupil_ref

        if metadata:
            for k, v in metadata.items():
                _try_write_attr(f, k, v)

        f.create_dataset(
            "psf",
            data=psf.psf.astype(np.float32, copy=False),
            compression="gzip",
            compression_opts=3,
        )

        _write_optics(f.create_group("optics"), psf.optics)

        sg = f.create_group("sampling")
        sg.attrs["pixel_size"] = np.asarray(psf.pixel_size, dtype=np.float64)
        if psf.z_planes is not None:
            sg.create_dataset(
                "z_planes",
                data=np.asarray(psf.z_planes, dtype=np.float64),
            )

        if psf.distillation_diagnostics is not None:
            dg = f.create_group("distillation")
            d = psf.distillation_diagnostics
            if "positions" in d:
                dg.create_dataset(
                    "positions",
                    data=np.asarray(d["positions"], dtype=np.int32),
                )
            if "amplitudes" in d:
                dg.create_dataset(
                    "amplitudes",
                    data=np.asarray(d["amplitudes"], dtype=np.float64),
                )
            for key in ("chi2_history", "psf_change_history", "amp_change_history"):
                if key in d:
                    dg.create_dataset(
                        key, data=np.asarray(d[key], dtype=np.float64)
                    )


def load_psf(filepath: str | Path) -> Psf:
    """Read a ``.psf.h5`` file and return a :class:`Psf`."""
    path = Path(filepath)
    with h5py.File(path, "r") as f:
        fmt = f.attrs.get("format", "")
        if fmt != _PSF_FORMAT:
            raise ValueError(
                f"{path}: not a deconlib psf file (format={fmt!r})"
            )

        source = str(f.attrs.get("source", "theoretical"))
        pupil_ref = (
            str(f.attrs["pupil_ref"]) if "pupil_ref" in f.attrs else None
        )

        optics = _read_optics(f["optics"])
        sg = f["sampling"]
        pixel_size = tuple(float(v) for v in sg.attrs["pixel_size"])
        z_planes = sg["z_planes"][...] if "z_planes" in sg else None

        psf_arr = f["psf"][...].astype(np.float32, copy=False)

        distillation_diagnostics: Optional[dict] = None
        if "distillation" in f:
            dg = f["distillation"]
            distillation_diagnostics = {}
            for key in dg.keys():
                distillation_diagnostics[key] = dg[key][...]

    return Psf(
        psf=psf_arr,
        optics=optics,
        pixel_size=pixel_size,
        z_planes=z_planes,
        source=source,
        pupil_ref=pupil_ref,
        distillation_diagnostics=distillation_diagnostics,
    )


def _try_write_attr(g: h5py.Group, key: str, value: Any) -> None:
    """Best-effort scalar/array attr write for free-form metadata."""
    try:
        g.attrs[key] = value
    except (TypeError, ValueError):
        pass

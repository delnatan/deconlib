"""HDF5 I/O for PSF artifacts.

Single-file, h5py-backed format for PSF images.

* ``.psf.h5`` — a 2D/3D PSF image + Optics + voxel sampling. Optionally
  carries distillation diagnostics.

Geometry's nine derived ndarrays are *not* serialized; the originating
``(shape, spacing, z_planes)`` params are stored instead.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import h5py
import numpy as np

from .psf import (
    Optics,
    PsfDistillationResult,
)

_PSF_FORMAT = "deconlib_psf"
_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


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


def _try_write_attr(g: h5py.Group, key: str, value: Any) -> None:
    """Best-effort scalar/array attr write for free-form metadata."""
    try:
        g.attrs[key] = value
    except (TypeError, ValueError):
        pass


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

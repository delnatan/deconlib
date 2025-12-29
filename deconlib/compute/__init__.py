"""Computation modules for PSF and OTF generation."""

from .psf import (
    pupil_to_psf,
    pupil_to_psf_centered,
    compute_otf,
    pupil_to_vectorial_psf,
    pupil_to_vectorial_psf_centered,
)
from .confocal import (
    ConfocalOptics,
    compute_pinhole_function,
    compute_airy_radius,
    compute_confocal_psf,
    compute_confocal_psf_centered,
    compute_spinning_disk_psf,
    compute_spinning_disk_psf_centered,
)

__all__ = [
    # Widefield PSF (scalar)
    "pupil_to_psf",
    "pupil_to_psf_centered",
    "compute_otf",
    # Widefield PSF (vectorial)
    "pupil_to_vectorial_psf",
    "pupil_to_vectorial_psf_centered",
    # Confocal PSF
    "ConfocalOptics",
    "compute_pinhole_function",
    "compute_airy_radius",
    "compute_confocal_psf",
    "compute_confocal_psf_centered",
    "compute_spinning_disk_psf",
    "compute_spinning_disk_psf_centered",
]
